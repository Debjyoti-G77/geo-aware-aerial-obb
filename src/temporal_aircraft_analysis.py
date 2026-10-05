"""Multi-temporal spatial analysis for same-location aircraft detections.

This module compares *detected spatial aircraft presence* in two or more
georeferenced acquisitions.  A correspondence is not an aircraft identity:
it is a deterministic geometry-based association between detections in
consecutive timestamps.

All distances and polygon overlaps are calculated in a locally selected
metric CRS.  GeoJSON inputs and outputs retain RFC 7946 longitude/latitude
ordering (OGC:CRS84 semantics).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


FROZEN_VALIDATION_LOCATION_IDS = frozenset(
    {7, 10, 23, 25, 27, 31, 36, 42, 79, 84, 101, 108}
)
OUTPUT_CRS = "OGC:CRS84"
FINE_TUNED_CHECKPOINT_SHA256 = "4111cdab67bcc3c53623e4bd08aeb5943e4a49225c27c62172036b815991eced"
DEFAULT_MAX_DISTANCE_M = 50.0
DEFAULT_PERSISTENT_DISTANCE_M = 15.0
DEFAULT_GRID_SIZE_M = 100.0
EXPECTED_INFERENCE_POLICY = {
    "model_name": "rareplanes_finetuned_best.pt",
    "model_task": "obb",
    "inference_confidence": 0.25,
    "inference_iou": 0.45,
    "global_nms_iou": 0.50,
    "image_size": 1024,
    "tile_size": 1024,
    "tile_overlap": 0.20,
    "target_class_name": "plane",
    "device": "cpu",
}


class TemporalAnalysisError(RuntimeError):
    """Raised when temporal inputs or geometry violate the analysis protocol."""


@dataclass(frozen=True)
class TemporalScene:
    location_id: int
    scene_id: str
    acquisition_time: datetime
    predictions_path: Path
    raster_path: Path | None = None
    source_status: str = "reused"


@dataclass(frozen=True)
class TemporalDetection:
    detection_id: str
    location_id: int
    scene_id: str
    acquisition_time: datetime
    confidence: float
    geographic_polygon: Any
    projected_polygon: Any


@dataclass(frozen=True)
class CorrespondenceParameters:
    max_distance_m: float
    persistent_distance_m: float
    derivation: str
    median_aircraft_major_axis_m: float | None


@dataclass(frozen=True)
class TemporalCorrespondence:
    source_detection_id: str
    destination_detection_id: str
    source_scene_id: str
    destination_scene_id: str
    time_a: datetime
    time_b: datetime
    centroid_displacement_m: float
    polygon_iou: float
    confidence_a: float
    confidence_b: float
    category: str
    cost: float
    source_geographic_centroid: tuple[float, float]
    destination_geographic_centroid: tuple[float, float]


@dataclass(frozen=True)
class PairAnalysis:
    scene_a: str
    scene_b: str
    time_a: datetime
    time_b: datetime
    correspondences: tuple[TemporalCorrespondence, ...]
    appeared: tuple[TemporalDetection, ...]
    disappeared: tuple[TemporalDetection, ...]


@dataclass(frozen=True)
class TemporalAnalysisResult:
    location_id: int
    metric_crs: str
    parameters: CorrespondenceParameters
    scenes: tuple[TemporalScene, ...]
    detections_by_scene: Mapping[str, tuple[TemporalDetection, ...]]
    timestamp_statistics: tuple[dict[str, Any], ...]
    pairs: tuple[PairAnalysis, ...]
    activity_grid: Mapping[str, Any] | None


def parse_acquisition_time(value: str) -> datetime:
    """Parse an ISO-like RarePlanes timestamp into an aware UTC datetime."""

    text = str(value).strip()
    if not text:
        raise TemporalAnalysisError("Acquisition timestamp is missing")
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        parsed = None
        for pattern in ("%m/%d/%y %H:%M", "%m/%d/%Y %H:%M"):
            try:
                parsed = datetime.strptime(text, pattern)
                break
            except ValueError:
                continue
        if parsed is None:
            raise TemporalAnalysisError(f"Invalid acquisition timestamp: {value!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_temporal_scenes(scenes: Sequence[TemporalScene]) -> tuple[TemporalScene, ...]:
    if len(scenes) < 2:
        raise TemporalAnalysisError("Temporal analysis requires at least two acquisitions")
    locations = {scene.location_id for scene in scenes}
    if len(locations) != 1:
        raise TemporalAnalysisError("All temporal inputs must have the same location ID")
    ordered = tuple(sorted(scenes, key=lambda item: (item.acquisition_time, item.scene_id)))
    if len({scene.scene_id for scene in ordered}) != len(ordered):
        raise TemporalAnalysisError("Scene IDs must be unique")
    times = [scene.acquisition_time for scene in ordered]
    if len(set(times)) != len(times):
        raise TemporalAnalysisError("Acquisition timestamps must be unique")
    if tuple(scenes) != ordered:
        raise TemporalAnalysisError("Temporal inputs must be supplied in chronological order")
    return ordered


def _default_raster_validator(path: Path) -> bool:
    try:
        import rasterio

        with rasterio.open(path) as dataset:
            return bool(
                dataset.crs
                and dataset.width > 0
                and dataset.height > 0
                and dataset.transform
                and not dataset.transform.is_identity
            )
    except Exception:
        return False


def select_temporal_validation_location(
    metadata_path: Path,
    imagery_dir: Path,
    *,
    split_manifest_path: Path | None = None,
    raster_validator: Callable[[Path], bool] | None = None,
) -> tuple[int, tuple[dict[str, Any], ...], dict[str, Any]]:
    """Select repeated frozen-VAL TRAIN imagery by count, span, then loc_id.

    Selection uses metadata and georeferencing validity only.  Detector output,
    aircraft count, weather, and image appearance are never consulted.
    """

    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not imagery_dir.is_dir():
        raise FileNotFoundError(imagery_dir)
    validator = raster_validator or _default_raster_validator
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    allowed_scene_ids: set[str] | None = None
    if split_manifest_path is not None:
        if not split_manifest_path.is_file():
            raise FileNotFoundError(split_manifest_path)
        with split_manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            split_rows = list(csv.DictReader(handle))
        val_rows = [row for row in split_rows if str(row.get("split", "")).lower() == "val"]
        allowed_scene_ids = {str(row.get("scene_id", "")).strip() for row in val_rows}
        manifest_locations = {int(row["loc_id"]) for row in val_rows}
        if not allowed_scene_ids or not manifest_locations <= FROZEN_VALIDATION_LOCATION_IDS:
            raise TemporalAnalysisError("Split manifest does not describe the frozen internal validation set")

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            location_id = int(row.get("loc_id", ""))
        except ValueError:
            continue
        if location_id not in FROZEN_VALIDATION_LOCATION_IDS:
            continue
        if row.get("Train") != "1":
            continue
        scene_id = str(row.get("image_id", "")).strip()
        if allowed_scene_ids is not None and scene_id not in allowed_scene_ids:
            continue
        timestamp_text = str(row.get("collection_time_start", "")).strip()
        if not scene_id or not timestamp_text:
            continue
        raster_path = imagery_dir / f"{scene_id}.tif"
        if not raster_path.is_file() or not validator(raster_path):
            continue
        timestamp = parse_acquisition_time(timestamp_text)
        grouped[location_id].append(
            {
                "location_id": location_id,
                "scene_id": scene_id,
                "acquisition_time": timestamp,
                "raster_path": raster_path,
            }
        )

    eligible: list[tuple[int, float, int, tuple[dict[str, Any], ...]]] = []
    for location_id, items in grouped.items():
        ordered = tuple(sorted(items, key=lambda item: (item["acquisition_time"], item["scene_id"])))
        unique_times = {item["acquisition_time"] for item in ordered}
        if len(ordered) < 2 or len(unique_times) != len(ordered):
            continue
        span = (ordered[-1]["acquisition_time"] - ordered[0]["acquisition_time"]).total_seconds()
        eligible.append((len(ordered), span, location_id, ordered))
    if not eligible:
        raise TemporalAnalysisError(
            "No frozen validation location has at least two timestamped, georeferenced TRAIN scenes"
        )
    count, span, location_id, selected = min(
        eligible, key=lambda item: (-item[0], -item[1], item[2])
    )
    rule = {
        "rule": (
            "same frozen-VAL TRAIN location with >=2 timestamped valid rasters; "
            "maximize acquisition count, then temporal span, then lowest loc_id"
        ),
        "eligible_location_ids": sorted(item[2] for item in eligible),
        "selected_acquisition_count": count,
        "selected_temporal_span_seconds": span,
        "performance_data_used": False,
        "split_provenance": str(split_manifest_path) if split_manifest_path else "frozen location IDs",
    }
    return location_id, selected, rule


def validate_inference_summary(summary: Mapping[str, Any], scene_id: str) -> None:
    if summary.get("scene_id") != scene_id:
        raise TemporalAnalysisError(f"Inference summary scene mismatch for {scene_id}")
    for key, expected in EXPECTED_INFERENCE_POLICY.items():
        actual = summary.get(key)
        if isinstance(expected, float):
            compatible = isinstance(actual, (int, float)) and math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            compatible = actual == expected
        if not compatible:
            raise TemporalAnalysisError(
                f"Scene {scene_id} used incompatible {key}: {actual!r}; expected {expected!r}"
            )


def discover_compatible_output(predictions_root: Path, scene_id: str) -> Path | None:
    if not predictions_root.is_dir():
        return None
    matches: list[Path] = []
    for summary_path in sorted(predictions_root.rglob("inference_summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            validate_inference_summary(summary, scene_id)
        except (OSError, json.JSONDecodeError, TemporalAnalysisError):
            continue
        predictions_path = summary_path.with_name("predictions.geojson")
        completed_path = summary_path.parent.parent / "completed.json"
        if not predictions_path.is_file() or not completed_path.is_file():
            continue
        try:
            completed = json.loads(completed_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        settings = completed.get("settings") or {}
        if (
            str(completed.get("model_sha256", "")).lower() != FINE_TUNED_CHECKPOINT_SHA256
            or settings.get("boundary_policy") != "strict_all_obb_vertices_inside_valid_source_region"
        ):
            continue
        matches.append(predictions_path)
    if len(matches) > 1:
        raise TemporalAnalysisError(f"Multiple compatible inference outputs found for {scene_id}")
    return matches[0] if matches else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_or_generate_selected_outputs(
    selected_rows: Sequence[Mapping[str, Any]],
    *,
    location_id: int,
    predictions_root: Path,
    generated_root: Path,
    model_path: Path,
) -> tuple[TemporalScene, ...]:
    """Reuse fixed-policy output or infer only missing selected TRAIN scenes.

    The model is loaded once, and only if at least one selected scene lacks a
    compatible prior output.  This helper intentionally contains no tunable
    production inference values: it uses ``EXPECTED_INFERENCE_POLICY``.
    """

    resolved: list[TemporalScene | None] = []
    missing_indices: list[int] = []
    for index, row in enumerate(selected_rows):
        scene_id = str(row["scene_id"])
        existing = discover_compatible_output(predictions_root, scene_id)
        if existing is not None:
            resolved.append(
                TemporalScene(
                    location_id,
                    scene_id,
                    row["acquisition_time"],
                    existing,
                    Path(row["raster_path"]),
                    "reused",
                )
            )
        else:
            resolved.append(None)
            missing_indices.append(index)
    if missing_indices:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Compatible outputs are missing and the external checkpoint is unavailable: {model_path}"
            )
        actual_hash = _sha256(model_path)
        if actual_hash.lower() != FINE_TUNED_CHECKPOINT_SHA256:
            raise TemporalAnalysisError(
                f"Fine-tuned checkpoint SHA256 mismatch: {actual_hash}"
            )
        try:
            from .geospatial_inference import (
                SceneInferenceSettings,
                load_obb_model,
                process_raster_scene,
            )
        except ImportError:
            from geospatial_inference import (
                SceneInferenceSettings,
                load_obb_model,
                process_raster_scene,
            )

        load_start = time.perf_counter()
        loaded_model = load_obb_model(model_path, "cpu")
        load_seconds = time.perf_counter() - load_start
        for index in missing_indices:
            row = selected_rows[index]
            scene_id = str(row["scene_id"])
            scene_output = generated_root / "scene_inference" / scene_id
            settings = SceneInferenceSettings(
                model_path=model_path,
                raster_path=Path(row["raster_path"]),
                output_dir=scene_output,
                device="cpu",
                image_size=int(EXPECTED_INFERENCE_POLICY["image_size"]),
                confidence=float(EXPECTED_INFERENCE_POLICY["inference_confidence"]),
                model_iou=float(EXPECTED_INFERENCE_POLICY["inference_iou"]),
                global_nms_iou=float(EXPECTED_INFERENCE_POLICY["global_nms_iou"]),
                tile_size=int(EXPECTED_INFERENCE_POLICY["tile_size"]),
                tile_overlap=float(EXPECTED_INFERENCE_POLICY["tile_overlap"]),
                target_class=str(EXPECTED_INFERENCE_POLICY["target_class_name"]),
            )
            result = process_raster_scene(
                loaded_model,
                settings,
                model_load_seconds=load_seconds if index == missing_indices[0] else 0.0,
            )
            validate_inference_summary(result.summary.to_dict(), scene_id)
            resolved[index] = TemporalScene(
                location_id,
                scene_id,
                row["acquisition_time"],
                result.predictions_path,
                Path(row["raster_path"]),
                "generated",
            )
    scenes = tuple(item for item in resolved if item is not None)
    if len(scenes) != len(selected_rows):
        raise TemporalAnalysisError("Failed to resolve every selected temporal scene")
    return validate_temporal_scenes(scenes)


def _finite_polygon(geometry: Any) -> bool:
    if geometry.is_empty or not geometry.is_valid or geometry.geom_type != "Polygon" or geometry.area <= 0:
        return False
    return all(math.isfinite(value) for coordinate in geometry.exterior.coords for value in coordinate[:2])


def _coordinate_range_valid(geometry: Any) -> bool:
    return all(
        -180.0 <= float(longitude) <= 180.0 and -90.0 <= float(latitude) <= 90.0
        for longitude, latitude, *_ in geometry.exterior.coords
    )


def choose_local_metric_crs(geographic_polygons: Sequence[Any]) -> Any:
    """Choose local UTM, falling back to local azimuthal equidistant."""

    from pyproj import CRS

    if not geographic_polygons:
        raise TemporalAnalysisError("A metric CRS cannot be selected without geographic geometry")
    longitudes = [float(polygon.centroid.x) for polygon in geographic_polygons]
    latitudes = [float(polygon.centroid.y) for polygon in geographic_polygons]
    longitude = statistics.fmean(longitudes)
    latitude = statistics.fmean(latitudes)
    if not (-180.0 <= longitude <= 180.0 and -90.0 <= latitude <= 90.0):
        raise TemporalAnalysisError("Invalid geographic centroid for metric CRS selection")
    if -80.0 <= latitude <= 84.0:
        zone = min(60, max(1, int(math.floor((longitude + 180.0) / 6.0)) + 1))
        return CRS.from_epsg((32600 if latitude >= 0.0 else 32700) + zone)
    return CRS.from_proj4(
        f"+proj=aeqd +lat_0={latitude:.12f} +lon_0={longitude:.12f} +datum=WGS84 +units=m +no_defs"
    )


def _project_geometry(geometry: Any, transformer: Any) -> Any:
    from shapely.ops import transform

    return transform(transformer.transform, geometry)


def load_temporal_detections(
    scenes: Sequence[TemporalScene],
) -> tuple[str, dict[str, tuple[TemporalDetection, ...]]]:
    """Load and project compatible GeoJSON detections for ordered acquisitions."""

    from pyproj import CRS, Transformer
    from shapely.geometry import shape

    ordered = validate_temporal_scenes(scenes)
    geographic: dict[str, list[tuple[str, float, Any]]] = {}
    all_polygons: list[Any] = []
    for scene in ordered:
        try:
            collection = json.loads(scene.predictions_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TemporalAnalysisError(f"Cannot read predictions: {scene.predictions_path}") from exc
        if collection.get("type") != "FeatureCollection" or "crs" in collection:
            raise TemporalAnalysisError("Predictions must be an RFC 7946 FeatureCollection")
        records: list[tuple[str, float, Any]] = []
        for index, feature in enumerate(collection.get("features", [])):
            geometry = shape(feature.get("geometry"))
            if not _finite_polygon(geometry) or not _coordinate_range_valid(geometry):
                raise TemporalAnalysisError(f"Invalid geographic polygon in {scene.scene_id}")
            properties = feature.get("properties") or {}
            feature_scene = str(properties.get("scene_id", scene.scene_id))
            if feature_scene != scene.scene_id:
                raise TemporalAnalysisError(f"Prediction scene mismatch in {scene.scene_id}")
            confidence = float(properties.get("confidence", float("nan")))
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise TemporalAnalysisError(f"Invalid confidence in {scene.scene_id}")
            detection_id = str(properties.get("detection_id") or feature.get("id") or f"{scene.scene_id}:{index}")
            records.append((detection_id, confidence, geometry))
            all_polygons.append(geometry)
        geographic[scene.scene_id] = records

    # Empty timestamps are valid.  If every timestamp is empty, a raster
    # footprint is needed to establish a projection; report this explicitly.
    if not all_polygons:
        raise TemporalAnalysisError("All timestamps are empty; metric CRS cannot be inferred from detections")
    metric_crs = choose_local_metric_crs(all_polygons)
    transformer = Transformer.from_crs(CRS.from_user_input(OUTPUT_CRS), metric_crs, always_xy=True)
    detections: dict[str, tuple[TemporalDetection, ...]] = {}
    for scene in ordered:
        converted = []
        for detection_id, confidence, polygon in geographic[scene.scene_id]:
            projected = _project_geometry(polygon, transformer)
            if not _finite_polygon(projected):
                raise TemporalAnalysisError(f"Projection produced invalid geometry: {detection_id}")
            converted.append(
                TemporalDetection(
                    detection_id=detection_id,
                    location_id=scene.location_id,
                    scene_id=scene.scene_id,
                    acquisition_time=scene.acquisition_time,
                    confidence=confidence,
                    geographic_polygon=polygon,
                    projected_polygon=projected,
                )
            )
        detections[scene.scene_id] = tuple(sorted(converted, key=lambda item: item.detection_id))
    return metric_crs.to_string(), detections


def polygon_iou(first: Any, second: Any) -> float:
    if not _finite_polygon(first) or not _finite_polygon(second):
        return 0.0
    intersection = float(first.intersection(second).area)
    if intersection <= 0.0:
        return 0.0
    union = float(first.union(second).area)
    return intersection / union if union > 0.0 and math.isfinite(union) else 0.0


def _major_axis(polygon: Any) -> float:
    rectangle = polygon.minimum_rotated_rectangle
    coordinates = list(rectangle.exterior.coords)
    return max(math.dist(coordinates[index], coordinates[index + 1]) for index in range(4))


def derive_correspondence_parameters(
    detections: Sequence[TemporalDetection],
    *,
    max_distance_m: float | None = None,
    persistent_distance_m: float | None = None,
) -> CorrespondenceParameters:
    major_axes = [
        _major_axis(detection.projected_polygon)
        for detection in detections
        if _finite_polygon(detection.projected_polygon)
    ]
    median_axis = statistics.median(major_axes) if major_axes else None
    if max_distance_m is None:
        maximum = max(DEFAULT_MAX_DISTANCE_M, 2.0 * median_axis) if median_axis else DEFAULT_MAX_DISTANCE_M
    else:
        maximum = float(max_distance_m)
    if persistent_distance_m is None:
        persistent = max(DEFAULT_PERSISTENT_DISTANCE_M, 0.5 * median_axis) if median_axis else DEFAULT_PERSISTENT_DISTANCE_M
    else:
        persistent = float(persistent_distance_m)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise TemporalAnalysisError("Maximum correspondence distance must be positive and finite")
    if not math.isfinite(persistent) or persistent < 0.0 or persistent > maximum:
        raise TemporalAnalysisError("Persistent distance must be finite and within maximum distance")
    derivation = (
        "configurable override"
        if max_distance_m is not None or persistent_distance_m is not None
        else (
            "geometry-derived analysis parameters: max=max(50 m, 2x median OBB major axis); "
            "persistent=max(15 m, 0.5x median OBB major axis)"
        )
    )
    return CorrespondenceParameters(maximum, persistent, derivation, median_axis)


def match_consecutive_detections(
    source: Sequence[TemporalDetection],
    destination: Sequence[TemporalDetection],
    parameters: CorrespondenceParameters,
) -> tuple[tuple[TemporalCorrespondence, ...], tuple[TemporalDetection, ...], tuple[TemporalDetection, ...]]:
    """Deterministic one-to-one global greedy geometry matching.

    Candidate cost is ``0.70 * (1 - polygon IoU) + 0.30 * normalized distance``.
    Candidates beyond ``max_distance_m`` are ineligible.  Sorting all candidate
    edges globally before assignment makes the method deterministic and keeps
    both geometry overlap and displacement explicit; it is not identity tracking.
    """

    edges: list[tuple[float, float, float, str, str, TemporalDetection, TemporalDetection]] = []
    for first in source:
        for second in destination:
            distance = float(first.projected_polygon.centroid.distance(second.projected_polygon.centroid))
            if not math.isfinite(distance) or distance > parameters.max_distance_m:
                continue
            overlap = polygon_iou(first.projected_polygon, second.projected_polygon)
            cost = 0.70 * (1.0 - overlap) + 0.30 * (distance / parameters.max_distance_m)
            edges.append((cost, -overlap, distance, first.detection_id, second.detection_id, first, second))
    edges.sort(key=lambda item: item[:5])

    used_source: set[str] = set()
    used_destination: set[str] = set()
    matches: list[TemporalCorrespondence] = []
    for cost, negative_overlap, distance, _, _, first, second in edges:
        if first.detection_id in used_source or second.detection_id in used_destination:
            continue
        used_source.add(first.detection_id)
        used_destination.add(second.detection_id)
        category = "persistent" if distance <= parameters.persistent_distance_m else "displaced"
        matches.append(
            TemporalCorrespondence(
                source_detection_id=first.detection_id,
                destination_detection_id=second.detection_id,
                source_scene_id=first.scene_id,
                destination_scene_id=second.scene_id,
                time_a=first.acquisition_time,
                time_b=second.acquisition_time,
                centroid_displacement_m=distance,
                polygon_iou=-negative_overlap,
                confidence_a=first.confidence,
                confidence_b=second.confidence,
                category=category,
                cost=cost,
                source_geographic_centroid=(float(first.geographic_polygon.centroid.x), float(first.geographic_polygon.centroid.y)),
                destination_geographic_centroid=(float(second.geographic_polygon.centroid.x), float(second.geographic_polygon.centroid.y)),
            )
        )
    appeared = tuple(item for item in destination if item.detection_id not in used_destination)
    disappeared = tuple(item for item in source if item.detection_id not in used_source)
    return tuple(matches), appeared, disappeared


def _footprint_from_raster(path: Path) -> Any:
    import rasterio
    from shapely.geometry import Polygon

    try:
        from .geospatial_tiling import global_pixels_to_crs84
    except ImportError:
        from geospatial_tiling import global_pixels_to_crs84

    with rasterio.open(path) as dataset:
        if not dataset.crs or dataset.transform.is_identity:
            raise TemporalAnalysisError(f"Raster lacks valid georeferencing: {path}")
        points = ((0.0, 0.0), (float(dataset.width), 0.0), (float(dataset.width), float(dataset.height)), (0.0, float(dataset.height)))
        polygon = Polygon(global_pixels_to_crs84(points, dataset.transform, dataset.crs))
    if not _finite_polygon(polygon):
        raise TemporalAnalysisError(f"Invalid raster footprint: {path}")
    return polygon


def _timestamp_statistics(
    scenes: Sequence[TemporalScene],
    detections: Mapping[str, Sequence[TemporalDetection]],
    metric_crs: str,
) -> tuple[dict[str, Any], ...]:
    from pyproj import Transformer

    transformer = Transformer.from_crs(OUTPUT_CRS, metric_crs, always_xy=True)
    rows = []
    for scene in scenes:
        values = detections[scene.scene_id]
        confidences = [item.confidence for item in values]
        footprint_area = None
        if scene.raster_path is not None:
            footprint = _project_geometry(_footprint_from_raster(scene.raster_path), transformer)
            footprint_area = float(footprint.area)
        rows.append(
            {
                "location_id": scene.location_id,
                "scene_id": scene.scene_id,
                "acquisition_time": _iso(scene.acquisition_time),
                "detection_count": len(values),
                "footprint_area_km2": footprint_area / 1_000_000.0 if footprint_area else None,
                "aircraft_density_per_km2": len(values) / (footprint_area / 1_000_000.0) if footprint_area else None,
                "confidence_min": min(confidences) if confidences else None,
                "confidence_mean": statistics.fmean(confidences) if confidences else None,
                "confidence_median": statistics.median(confidences) if confidences else None,
                "confidence_max": max(confidences) if confidences else None,
                "inference_output_status": scene.source_status,
            }
        )
    return tuple(rows)


def build_activity_grid(
    detections: Sequence[TemporalDetection],
    metric_crs: str,
    *,
    timestamp_count: int,
    grid_size_m: float = DEFAULT_GRID_SIZE_M,
) -> dict[str, Any] | None:
    """Aggregate detection-centroid events in a simple metric grid."""

    if timestamp_count < 3 or not detections:
        return None
    if not math.isfinite(grid_size_m) or grid_size_m <= 0.0:
        raise TemporalAnalysisError("Activity-grid size must be positive and finite")
    from pyproj import Transformer
    from shapely.geometry import box, mapping
    from shapely.ops import transform

    reverse = Transformer.from_crs(metric_crs, OUTPUT_CRS, always_xy=True)
    cells: dict[tuple[int, int], list[TemporalDetection]] = defaultdict(list)
    for detection in detections:
        centroid = detection.projected_polygon.centroid
        key = (math.floor(centroid.x / grid_size_m), math.floor(centroid.y / grid_size_m))
        cells[key].append(detection)
    maximum = max(len(values) for values in cells.values())
    features = []
    for (column, row), values in sorted(cells.items()):
        metric_polygon = box(column * grid_size_m, row * grid_size_m, (column + 1) * grid_size_m, (row + 1) * grid_size_m)
        geographic_polygon = transform(reverse.transform, metric_polygon)
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(geographic_polygon),
                "properties": {
                    "grid_col": column,
                    "grid_row": row,
                    "grid_size_m": grid_size_m,
                    "detection_events": len(values),
                    "timestamp_count": len({item.acquisition_time for item in values}),
                    "highest_observed_activity": len(values) == maximum,
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def analyze_temporal_scenes(
    scenes: Sequence[TemporalScene],
    *,
    max_distance_m: float | None = None,
    persistent_distance_m: float | None = None,
    grid_size_m: float = DEFAULT_GRID_SIZE_M,
) -> TemporalAnalysisResult:
    ordered = validate_temporal_scenes(scenes)
    metric_crs, detections = load_temporal_detections(ordered)
    all_detections = tuple(item for scene in ordered for item in detections[scene.scene_id])
    parameters = derive_correspondence_parameters(
        all_detections,
        max_distance_m=max_distance_m,
        persistent_distance_m=persistent_distance_m,
    )
    pairs = []
    for first_scene, second_scene in zip(ordered, ordered[1:]):
        matches, appeared, disappeared = match_consecutive_detections(
            detections[first_scene.scene_id], detections[second_scene.scene_id], parameters
        )
        pairs.append(
            PairAnalysis(
                first_scene.scene_id,
                second_scene.scene_id,
                first_scene.acquisition_time,
                second_scene.acquisition_time,
                matches,
                appeared,
                disappeared,
            )
        )
    grid = build_activity_grid(all_detections, metric_crs, timestamp_count=len(ordered), grid_size_m=grid_size_m)
    return TemporalAnalysisResult(
        location_id=ordered[0].location_id,
        metric_crs=metric_crs,
        parameters=parameters,
        scenes=ordered,
        detections_by_scene=detections,
        timestamp_statistics=_timestamp_statistics(ordered, detections, metric_crs),
        pairs=tuple(pairs),
        activity_grid=grid,
    )


def _feature_collection(features: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": list(features)}


def serialize_temporal_detections(result: TemporalAnalysisResult) -> dict[str, Any]:
    from shapely.geometry import mapping

    features = []
    for scene in result.scenes:
        for detection in result.detections_by_scene[scene.scene_id]:
            features.append(
                {
                    "type": "Feature",
                    "id": detection.detection_id,
                    "geometry": mapping(detection.geographic_polygon),
                    "properties": {
                        "detection_id": detection.detection_id,
                        "location_id": detection.location_id,
                        "scene_id": detection.scene_id,
                        "acquisition_time": _iso(detection.acquisition_time),
                        "confidence": detection.confidence,
                        "output_crs": OUTPUT_CRS,
                    },
                }
            )
    return _feature_collection(features)


def serialize_correspondences(result: TemporalAnalysisResult) -> dict[str, Any]:
    features = []
    for pair in result.pairs:
        for match in pair.correspondences:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [list(match.source_geographic_centroid), list(match.destination_geographic_centroid)],
                    },
                    "properties": {
                        "source_detection_id": match.source_detection_id,
                        "destination_detection_id": match.destination_detection_id,
                        "source_scene_id": match.source_scene_id,
                        "destination_scene_id": match.destination_scene_id,
                        "time_a": _iso(match.time_a),
                        "time_b": _iso(match.time_b),
                        "centroid_displacement_m": match.centroid_displacement_m,
                        "polygon_iou": match.polygon_iou,
                        "confidence_a": match.confidence_a,
                        "confidence_b": match.confidence_b,
                        "category": match.category,
                        "assignment_cost": match.cost,
                        "identity_tracking": False,
                    },
                }
            )
    return _feature_collection(features)


def serialize_presence_changes(result: TemporalAnalysisResult) -> dict[str, Any]:
    from shapely.geometry import mapping

    features = []
    for pair in result.pairs:
        for category, detections, timestamp in (
            ("appeared", pair.appeared, pair.time_b),
            ("disappeared", pair.disappeared, pair.time_a),
        ):
            for detection in detections:
                features.append(
                    {
                        "type": "Feature",
                        "id": f"{category}:{pair.scene_a}:{pair.scene_b}:{detection.detection_id}",
                        "geometry": mapping(detection.geographic_polygon),
                        "properties": {
                            "category": category,
                            "detection_id": detection.detection_id,
                            "scene_id": detection.scene_id,
                            "location_id": detection.location_id,
                            "acquisition_time": _iso(timestamp),
                            "temporal_pair": f"{pair.scene_a}->{pair.scene_b}",
                            "identity_tracking": False,
                        },
                    }
                )
    return _feature_collection(features)


def build_activity_summary(result: TemporalAnalysisResult) -> dict[str, Any]:
    categories = Counter(match.category for pair in result.pairs for match in pair.correspondences)
    displacements = [match.centroid_displacement_m for pair in result.pairs for match in pair.correspondences]
    counts = [row["detection_count"] for row in result.timestamp_statistics]
    counts_by_scene = {
        str(row["scene_id"]): int(row["detection_count"])
        for row in result.timestamp_statistics
    }
    pair_rows = []
    for pair in result.pairs:
        pair_categories = Counter(match.category for match in pair.correspondences)
        pair_displacements = [match.centroid_displacement_m for match in pair.correspondences]
        pair_ious = [match.polygon_iou for match in pair.correspondences]
        matched = len(pair.correspondences)
        earlier_count = counts_by_scene[pair.scene_a]
        later_count = counts_by_scene[pair.scene_b]
        pair_rows.append(
            {
                "scene_a": pair.scene_a,
                "scene_b": pair.scene_b,
                "time_a": _iso(pair.time_a),
                "time_b": _iso(pair.time_b),
                "persistent": pair_categories["persistent"],
                "displaced": pair_categories["displaced"],
                "matched_total": matched,
                "appeared": len(pair.appeared),
                "disappeared": len(pair.disappeared),
                "conservation": {
                    "earlier_count": earlier_count,
                    "matched_plus_disappeared": matched + len(pair.disappeared),
                    "earlier_valid": earlier_count == matched + len(pair.disappeared),
                    "later_count": later_count,
                    "matched_plus_appeared": matched + len(pair.appeared),
                    "later_valid": later_count == matched + len(pair.appeared),
                },
                "displacement_m": {
                    "mean": statistics.fmean(pair_displacements) if pair_displacements else None,
                    "median": statistics.median(pair_displacements) if pair_displacements else None,
                    "max": max(pair_displacements) if pair_displacements else None,
                },
                "mean_polygon_iou": statistics.fmean(pair_ious) if pair_ious else None,
            }
        )
    highest_activity_cells = []
    if result.activity_grid is not None and result.activity_grid["features"]:
        maximum_events = max(
            int(feature["properties"]["detection_events"])
            for feature in result.activity_grid["features"]
        )
        highest_activity_cells = [
            dict(feature["properties"])
            for feature in result.activity_grid["features"]
            if int(feature["properties"]["detection_events"]) == maximum_events
        ]
    return {
        "analysis_semantics": (
            "Changes in detected spatial aircraft presence; no physical aircraft identity is established."
        ),
        "location_id": result.location_id,
        "output_crs": OUTPUT_CRS,
        "metric_crs": result.metric_crs,
        "correspondence_parameters": {
            "max_distance_m": result.parameters.max_distance_m,
            "persistent_distance_m": result.parameters.persistent_distance_m,
            "derivation": result.parameters.derivation,
            "median_aircraft_major_axis_m": result.parameters.median_aircraft_major_axis_m,
            "cost": "0.70 * (1 - polygon_iou) + 0.30 * (centroid_distance / max_distance)",
            "assignment": "deterministic global greedy one-to-one",
        },
        "timestamps": list(result.timestamp_statistics),
        "detection_count_trajectory": counts,
        "net_count_change": counts[-1] - counts[0],
        "pair_statistics": pair_rows,
        "persistent_correspondences": categories["persistent"],
        "displaced_correspondences": categories["displaced"],
        "appeared_presence": sum(len(pair.appeared) for pair in result.pairs),
        "disappeared_presence": sum(len(pair.disappeared) for pair in result.pairs),
        "displacement_m": {
            "count": len(displacements),
            "mean": statistics.fmean(displacements) if displacements else None,
            "median": statistics.median(displacements) if displacements else None,
            "max": max(displacements) if displacements else None,
        },
        "activity_grid_generated": result.activity_grid is not None,
        "activity_grid_cell_count": len(result.activity_grid["features"]) if result.activity_grid else 0,
        "highest_activity_cells": highest_activity_cells,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_temporal_outputs(result: TemporalAnalysisResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "temporal_detections": output_dir / "temporal_detections.geojson",
        "temporal_correspondences": output_dir / "temporal_correspondences.geojson",
        "appeared_disappeared": output_dir / "appeared_disappeared.geojson",
        "activity_summary_json": output_dir / "activity_summary.json",
        "activity_summary_csv": output_dir / "activity_summary.csv",
    }
    _write_json(outputs["temporal_detections"], serialize_temporal_detections(result))
    _write_json(outputs["temporal_correspondences"], serialize_correspondences(result))
    _write_json(outputs["appeared_disappeared"], serialize_presence_changes(result))
    summary = build_activity_summary(result)
    _write_json(outputs["activity_summary_json"], summary)
    with outputs["activity_summary_csv"].open("w", encoding="utf-8", newline="") as handle:
        fields = list(result.timestamp_statistics[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result.timestamp_statistics)
    if result.activity_grid is not None:
        outputs["activity_grid"] = output_dir / "activity_grid.geojson"
        _write_json(outputs["activity_grid"], result.activity_grid)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze same-location aircraft presence across time.")
    parser.add_argument("--manifest", help="CSV: location_id, scene_id, acquisition_time, predictions_path[, raster_path]")
    parser.add_argument("--metadata", default="datasets/RarePlanes/metadata/RarePlanes_Public_Metadata.csv")
    parser.add_argument("--split-manifest", default="datasets/RarePlanes_yolo_obb/manifests/split_scenes.csv")
    parser.add_argument("--imagery-dir", default="datasets/RarePlanes/train/imagery/PS-RGB_cog")
    parser.add_argument("--predictions-root", default="outputs")
    parser.add_argument("--model", default="rareplanes_finetuned_best.pt")
    parser.add_argument("--output", default="outputs/temporal_analysis")
    parser.add_argument("--max-distance-m", type=float)
    parser.add_argument("--persistent-distance-m", type=float)
    parser.add_argument("--grid-size-m", type=float, default=DEFAULT_GRID_SIZE_M)
    return parser


def load_analysis_manifest(path: Path) -> tuple[TemporalScene, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    base = path.parent
    scenes = []
    for row in rows:
        predictions = Path(row["predictions_path"])
        raster_text = str(row.get("raster_path", "")).strip()
        raster = Path(raster_text) if raster_text else None
        if not predictions.is_absolute():
            predictions = (base / predictions).resolve()
        if raster is not None and not raster.is_absolute():
            raster = (base / raster).resolve()
        scenes.append(
            TemporalScene(
                location_id=int(row["location_id"]),
                scene_id=row["scene_id"],
                acquisition_time=parse_acquisition_time(row["acquisition_time"]),
                predictions_path=predictions,
                raster_path=raster,
                source_status=row.get("source_status", "reused") or "reused",
            )
        )
    return validate_temporal_scenes(scenes)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.manifest:
            scenes = load_analysis_manifest(Path(args.manifest))
            output_dir = Path(args.output) / str(scenes[0].location_id)
        else:
            location_id, selected_rows, selection_rule = select_temporal_validation_location(
                Path(args.metadata),
                Path(args.imagery_dir),
                split_manifest_path=Path(args.split_manifest),
            )
            output_dir = Path(args.output) / str(location_id)
            scenes = resolve_or_generate_selected_outputs(
                selected_rows,
                location_id=location_id,
                predictions_root=Path(args.predictions_root),
                generated_root=output_dir,
                model_path=Path(args.model),
            )
        result = analyze_temporal_scenes(
            scenes,
            max_distance_m=args.max_distance_m,
            persistent_distance_m=args.persistent_distance_m,
            grid_size_m=args.grid_size_m,
        )
        outputs = write_temporal_outputs(result, output_dir)
    except (FileNotFoundError, KeyError, OSError, TemporalAnalysisError, ValueError) as exc:
        parser.error(str(exc))
        return 2
    payload = {"summary": build_activity_summary(result), "outputs": {key: str(value) for key, value in outputs.items()}}
    if not args.manifest:
        payload["selection"] = selection_rule
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
