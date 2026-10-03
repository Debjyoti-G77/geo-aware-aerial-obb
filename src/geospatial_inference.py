"""Streaming full-scene geospatial OBB inference for one raster scene.

The module deliberately keeps model integration, polygon duplicate resolution,
geographic conversion, and serialization separate from the tiling primitives
in :mod:`geospatial_tiling`. It never materializes source tiles on disk and
retains only lightweight detection metadata between tile predictions.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

try:
    from .geospatial_tiling import (
        GeometryError,
        Point,
        TileWindow,
        generate_tile_windows,
        global_pixels_to_crs84,
        local_obb_to_global_pixels,
        points_within_valid_tile_region,
        read_tile,
        stride_from_overlap,
        tiling_parameters_from_config,
        validate_tile_plan,
    )
    from .project_config import (
        ConfigError,
        config_value,
        ensure_directory,
        find_project_root,
        load_project_config,
        resolve_project_path,
    )
except ImportError:  # Support direct execution: python src/geospatial_inference.py
    from geospatial_tiling import (
        GeometryError,
        Point,
        TileWindow,
        generate_tile_windows,
        global_pixels_to_crs84,
        local_obb_to_global_pixels,
        points_within_valid_tile_region,
        read_tile,
        stride_from_overlap,
        tiling_parameters_from_config,
        validate_tile_plan,
    )
    from project_config import (
        ConfigError,
        config_value,
        ensure_directory,
        find_project_root,
        load_project_config,
        resolve_project_path,
    )


OUTPUT_CRS = "OGC:CRS84"
EXPECTED_PLANE_CLASS_NAME = "plane"


class GeospatialInferenceError(RuntimeError):
    """Raised when model, raster, geometry, or output validation fails."""


class DetectionRejected(GeospatialInferenceError):
    """A candidate cannot be mapped safely into valid source-image space."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class SceneInferenceSettings:
    model_path: Path
    raster_path: Path
    output_dir: Path
    device: str = "cpu"
    image_size: int = 1024
    confidence: float = 0.25
    model_iou: float = 0.45
    global_nms_iou: float = 0.50
    tile_size: int = 1024
    tile_overlap: float = 0.20
    target_class: str = EXPECTED_PLANE_CLASS_NAME

    def __post_init__(self) -> None:
        if self.device.lower() != "cpu":
            raise ValueError("Geospatial inference currently requires device=cpu")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 <= self.model_iou <= 1.0:
            raise ValueError("model_iou must be in [0, 1]")
        if not 0.0 <= self.global_nms_iou <= 1.0:
            raise ValueError("global_nms_iou must be in [0, 1]")
        stride_from_overlap(self.tile_size, self.tile_overlap)
        if not self.target_class:
            raise ValueError("target_class must be non-empty")


@dataclass(frozen=True)
class LoadedOBBModel:
    model: Any
    names: dict[int, str]
    task: str
    model_name: str


@dataclass(frozen=True)
class RawOBBDetection:
    detection_id: str
    detection_index: int
    class_id: int
    class_name: str
    confidence: float
    local_obb_vertices: tuple[Point, Point, Point, Point]


@dataclass(frozen=True)
class GlobalDetection:
    detection_id: str
    scene_id: str
    tile_id: str
    tile_row_off: int
    tile_col_off: int
    tile_pixel_bounds: tuple[float, float, float, float]
    tile_valid_region: tuple[float, float, float, float]
    tile_size: int
    source_width: int
    source_height: int
    class_id: int
    class_name: str
    confidence: float
    local_obb_vertices: tuple[Point, Point, Point, Point]
    global_pixel_obb_vertices: tuple[Point, Point, Point, Point]
    global_pixel_polygon: Any
    source_crs: str
    output_crs: str = OUTPUT_CRS
    geographic_polygon_vertices: tuple[Point, Point, Point, Point] | None = None
    geographic_polygon: Any | None = None


@dataclass(frozen=True)
class RejectedDetection:
    detection_id: str
    scene_id: str
    tile_id: str
    class_id: int
    class_name: str
    confidence: float
    rejection_reason: str
    local_obb_vertices: tuple[Point, Point, Point, Point]
    attempted_global_pixel_obb_vertices: tuple[Point, Point, Point, Point] | None
    attempted_global_polygon_valid: bool
    tile_valid_region: tuple[float, float, float, float]
    tile_source_image_bounds: tuple[float, float, float, float]
    row_off: int
    col_off: int
    diagnostic_status: str = "rejected_not_official_prediction"


@dataclass(frozen=True)
class SuppressionRecord:
    suppressed_detection_id: str
    kept_detection_id: str
    polygon_iou: float
    suppressed_tile_id: str
    kept_tile_id: str
    suppressed_confidence: float
    kept_confidence: float
    class_id: int
    class_name: str
    suppressed_local_obb_vertices: tuple[Point, Point, Point, Point]
    suppressed_global_pixel_obb_vertices: tuple[Point, Point, Point, Point]
    kept_global_pixel_obb_vertices: tuple[Point, Point, Point, Point]


@dataclass(frozen=True)
class PolygonNMSResult:
    kept: tuple[GlobalDetection, ...]
    suppressed: tuple[SuppressionRecord, ...]


@dataclass(frozen=True)
class SceneInferenceSummary:
    scene_id: str
    raster_path: str
    raster_width: int
    raster_height: int
    raster_bands: int
    raster_dtype: str
    source_crs: str
    output_crs: str
    tile_count: int
    tile_size: int
    tile_overlap: float
    integer_stride: int
    model_name: str
    model_task: str
    device: str
    image_size: int
    inference_confidence: float
    inference_iou: float
    global_nms_iou: float
    target_class_id: int
    target_class_name: str
    all_class_raw_detections: int
    raw_class_counts: dict[str, int]
    raw_target_candidates: int
    non_target_discarded: int
    invalid_or_padding_rejected: int
    rejection_counts: dict[str, int]
    pre_nms_target_count: int
    duplicates_suppressed: int
    final_target_count: int
    model_load_seconds: float
    scene_processing_seconds: float
    total_runtime_seconds: float
    spatial_validation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SceneInferenceResult:
    summary: SceneInferenceSummary
    detections: tuple[GlobalDetection, ...]
    suppressed: tuple[SuppressionRecord, ...]
    rejected: tuple[RejectedDetection, ...]
    predictions_path: Path
    summary_path: Path
    suppression_path: Path
    rejection_path: Path


def normalize_model_names(names: Any) -> dict[int, str]:
    """Normalize Ultralytics model names to an integer-keyed mapping."""

    if isinstance(names, Mapping):
        normalized = {int(index): str(name) for index, name in names.items()}
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        normalized = {index: str(name) for index, name in enumerate(names)}
    else:
        raise GeospatialInferenceError("Model class names are not a mapping or sequence")
    return dict(sorted(normalized.items()))


def validate_model_metadata(model: Any, model_path: Path) -> LoadedOBBModel:
    """Validate model capabilities required by geospatial OBB inference."""

    task = str(getattr(model, "task", ""))
    names = normalize_model_names(getattr(model, "names", None))
    if task != "obb":
        raise GeospatialInferenceError(
            f"Incompatible model task {task!r}; expected 'obb'"
        )
    if not names:
        raise GeospatialInferenceError("Incompatible model classes: no classes defined")
    return LoadedOBBModel(
        model=model,
        names=names,
        task=task,
        model_name=model_path.name,
    )


def load_obb_model(model_path: Path, device: str = "cpu") -> LoadedOBBModel:
    """Load and validate the OBB model exactly once for a process."""

    if device.lower() != "cpu":
        raise GeospatialInferenceError("Only CPU execution is allowed in this pipeline")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise GeospatialInferenceError("Ultralytics is unavailable") from exc
    model = YOLO(str(model_path), task="obb")
    return validate_model_metadata(model, model_path)


def resolve_target_class(target_class: str, names: Mapping[int, str]) -> tuple[int, str]:
    """Resolve a numeric ID or exact class name against model metadata."""

    try:
        class_id = int(target_class)
    except ValueError:
        matches = [index for index, name in names.items() if name == target_class]
        if len(matches) != 1:
            raise GeospatialInferenceError(
                f"Target class {target_class!r} is not present exactly once"
            )
        class_id = matches[0]
    if class_id not in names:
        raise GeospatialInferenceError(f"Target class ID {class_id} is not present")
    return class_id, names[class_id]


def _to_numpy(value: Any) -> Any:
    """Convert a Torch-like or NumPy-like object without importing Torch."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value


def extract_ultralytics_obb(
    result: Any,
    names: Mapping[int, str],
    tile: TileWindow,
) -> tuple[RawOBBDetection, ...]:
    """Extract pixel-space OBBs from ``result.obb.xyxyxyxy``.

    Ultralytics 8.4.165 was empirically verified to expose an ``N x 4 x 2``
    float tensor through this property. Normalized ``xyxyxyxyn`` coordinates
    and axis-aligned ``boxes`` are intentionally not used.
    """

    import numpy as np

    obb = getattr(result, "obb", None)
    if obb is None:
        return ()
    try:
        vertices = np.asarray(_to_numpy(obb.xyxyxyxy), dtype=np.float64)
        confidences = np.asarray(_to_numpy(obb.conf), dtype=np.float64)
        class_ids = np.asarray(_to_numpy(obb.cls), dtype=np.float64)
    except AttributeError as exc:
        raise GeospatialInferenceError(
            "Ultralytics result does not expose OBB xyxyxyxy/conf/cls properties"
        ) from exc

    if vertices.size == 0:
        return ()
    if vertices.ndim != 3 or vertices.shape[1:] != (4, 2):
        raise GeospatialInferenceError(
            f"Unexpected result.obb.xyxyxyxy shape: {vertices.shape}"
        )
    if confidences.shape != (vertices.shape[0],) or class_ids.shape != (
        vertices.shape[0],
    ):
        raise GeospatialInferenceError("OBB confidence/class arrays are misaligned")

    detections: list[RawOBBDetection] = []
    for index in range(vertices.shape[0]):
        class_id_float = float(class_ids[index])
        class_id = int(class_id_float)
        if class_id_float != class_id or class_id not in names:
            raise GeospatialInferenceError(
                f"Invalid OBB class ID at detection {index}: {class_id_float}"
            )
        points = tuple(
            (float(vertex[0]), float(vertex[1])) for vertex in vertices[index]
        )
        detections.append(
            RawOBBDetection(
                detection_id=f"{tile.tile_id}__d{index:04d}",
                detection_index=index,
                class_id=class_id,
                class_name=names[class_id],
                confidence=float(confidences[index]),
                local_obb_vertices=points,  # type: ignore[arg-type]
            )
        )
    return tuple(detections)


def partition_target_detections(
    detections: Sequence[RawOBBDetection], target_class_id: int
) -> tuple[tuple[RawOBBDetection, ...], tuple[RawOBBDetection, ...]]:
    targets = tuple(
        detection for detection in detections if detection.class_id == target_class_id
    )
    discarded = tuple(
        detection for detection in detections if detection.class_id != target_class_id
    )
    return targets, discarded


def raw_detection_to_global(
    detection: RawOBBDetection,
    tile: TileWindow,
    source_crs: Any,
) -> GlobalDetection:
    """Validate and convert one tile-local detection into full-scene pixels."""

    from shapely.geometry import Polygon

    if not math.isfinite(detection.confidence) or not 0.0 <= detection.confidence <= 1.0:
        raise DetectionRejected("confidence_outside_unit_interval")
    if not points_within_valid_tile_region(detection.local_obb_vertices, tile):
        raise DetectionRejected("vertex_outside_valid_tile_region")
    try:
        global_vertices = local_obb_to_global_pixels(
            detection.local_obb_vertices, tile, validate=True
        )
    except GeometryError as exc:
        raise DetectionRejected("global_pixel_coordinate_outside_source") from exc

    polygon = Polygon(global_vertices)
    if polygon.is_empty:
        raise DetectionRejected("empty_global_polygon")
    if not polygon.is_valid:
        raise DetectionRejected("invalid_global_polygon")
    if not math.isfinite(polygon.area) or polygon.area <= 0.0:
        raise DetectionRejected("zero_or_nonfinite_global_polygon_area")

    return GlobalDetection(
        detection_id=detection.detection_id,
        scene_id=tile.scene_id,
        tile_id=tile.tile_id,
        tile_row_off=tile.row_off,
        tile_col_off=tile.col_off,
        tile_pixel_bounds=tile.source_bounds.as_tuple(),
        tile_valid_region=tile.valid_region.as_tuple(),
        tile_size=tile.tile_size,
        source_width=tile.source_width,
        source_height=tile.source_height,
        class_id=detection.class_id,
        class_name=detection.class_name,
        confidence=detection.confidence,
        local_obb_vertices=detection.local_obb_vertices,
        global_pixel_obb_vertices=global_vertices,
        global_pixel_polygon=polygon,
        source_crs=str(source_crs),
    )


def build_rejected_detection(
    detection: RawOBBDetection,
    tile: TileWindow,
    rejection_reason: str,
) -> RejectedDetection:
    """Preserve diagnostic evidence without making an invalid candidate valid."""

    from shapely.geometry import Polygon

    attempted_vertices = None
    attempted_polygon_valid = False
    try:
        attempted_vertices = local_obb_to_global_pixels(
            detection.local_obb_vertices, tile, validate=False
        )
        attempted_polygon = Polygon(attempted_vertices)
        attempted_polygon_valid = bool(
            not attempted_polygon.is_empty
            and attempted_polygon.is_valid
            and math.isfinite(attempted_polygon.area)
            and attempted_polygon.area > 0.0
        )
    except (GeometryError, IndexError, TypeError, ValueError):
        attempted_vertices = None

    return RejectedDetection(
        detection_id=detection.detection_id,
        scene_id=tile.scene_id,
        tile_id=tile.tile_id,
        class_id=detection.class_id,
        class_name=detection.class_name,
        confidence=detection.confidence,
        rejection_reason=rejection_reason,
        local_obb_vertices=detection.local_obb_vertices,
        attempted_global_pixel_obb_vertices=attempted_vertices,
        attempted_global_polygon_valid=attempted_polygon_valid,
        tile_valid_region=tile.valid_region.as_tuple(),
        tile_source_image_bounds=tile.source_bounds.as_tuple(),
        row_off=tile.row_off,
        col_off=tile.col_off,
    )


def polygon_iou(first: Any, second: Any) -> float:
    """Return true polygon IoU, or zero for invalid/degenerate inputs."""

    try:
        if (
            first is None
            or second is None
            or first.is_empty
            or second.is_empty
            or not first.is_valid
            or not second.is_valid
            or first.area <= 0.0
            or second.area <= 0.0
        ):
            return 0.0
        intersection_area = float(first.intersection(second).area)
        if intersection_area <= 0.0:
            return 0.0
        union_area = float(first.union(second).area)
        if not math.isfinite(union_area) or union_area <= 0.0:
            return 0.0
        return intersection_area / union_area
    except Exception as exc:
        try:
            from shapely.errors import GEOSException
        except ImportError:
            GEOSException = RuntimeError  # type: ignore[assignment,misc]
        if isinstance(exc, (GEOSException, ValueError, TypeError)):
            return 0.0
        raise


def _bounds_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def greedy_polygon_nms(
    detections: Sequence[GlobalDetection], threshold: float = 0.50
) -> PolygonNMSResult:
    """Class-aware greedy NMS using oriented polygons in source-pixel space."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Polygon NMS threshold must be in [0, 1]")
    pending = sorted(
        detections, key=lambda item: (-item.confidence, item.detection_id)
    )
    kept: list[GlobalDetection] = []
    suppressed: list[SuppressionRecord] = []

    while pending:
        selected = pending.pop(0)
        kept.append(selected)
        survivors: list[GlobalDetection] = []
        for candidate in pending:
            if (
                candidate.scene_id != selected.scene_id
                or candidate.class_id != selected.class_id
                or not _bounds_overlap(
                    selected.global_pixel_polygon.bounds,
                    candidate.global_pixel_polygon.bounds,
                )
            ):
                survivors.append(candidate)
                continue
            overlap = polygon_iou(
                selected.global_pixel_polygon, candidate.global_pixel_polygon
            )
            if overlap > threshold:
                suppressed.append(
                    SuppressionRecord(
                        suppressed_detection_id=candidate.detection_id,
                        kept_detection_id=selected.detection_id,
                        polygon_iou=overlap,
                        suppressed_tile_id=candidate.tile_id,
                        kept_tile_id=selected.tile_id,
                        suppressed_confidence=candidate.confidence,
                        kept_confidence=selected.confidence,
                        class_id=candidate.class_id,
                        class_name=candidate.class_name,
                        suppressed_local_obb_vertices=candidate.local_obb_vertices,
                        suppressed_global_pixel_obb_vertices=(
                            candidate.global_pixel_obb_vertices
                        ),
                        kept_global_pixel_obb_vertices=(
                            selected.global_pixel_obb_vertices
                        ),
                    )
                )
            else:
                survivors.append(candidate)
        pending = survivors

    return PolygonNMSResult(tuple(kept), tuple(suppressed))


def add_geographic_geometry(
    detection: GlobalDetection, transform: Any, source_crs: Any
) -> GlobalDetection:
    """Attach a CRS84 polygon while preserving OBB vertex order."""

    from shapely.geometry import Polygon

    geographic_vertices = global_pixels_to_crs84(
        detection.global_pixel_obb_vertices, transform, source_crs
    )
    if len(geographic_vertices) != 4:
        raise GeospatialInferenceError("A geographic OBB must have four vertices")
    geographic_polygon = Polygon(geographic_vertices)
    if (
        geographic_polygon.is_empty
        or not geographic_polygon.is_valid
        or geographic_polygon.area <= 0.0
    ):
        raise GeospatialInferenceError(
            f"Invalid geographic polygon for {detection.detection_id}"
        )
    return replace(
        detection,
        source_crs=str(source_crs),
        geographic_polygon_vertices=geographic_vertices,  # type: ignore[arg-type]
        geographic_polygon=geographic_polygon,
    )


def detection_to_geojson_feature(
    detection: GlobalDetection,
    *,
    model_name: str,
    model_task: str,
    confidence: float,
    model_iou: float,
    tile_overlap: float,
    global_nms_iou: float,
) -> dict[str, Any]:
    """Serialize one final detection as an RFC 7946 Polygon feature."""

    vertices = detection.geographic_polygon_vertices
    if vertices is None or detection.geographic_polygon is None:
        raise GeospatialInferenceError(
            f"Detection has no geographic geometry: {detection.detection_id}"
        )
    ring = [[float(longitude), float(latitude)] for longitude, latitude in vertices]
    ring.append(ring[0].copy())
    return {
        "type": "Feature",
        "id": detection.detection_id,
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {
            "detection_id": detection.detection_id,
            "scene_id": detection.scene_id,
            "class_id": detection.class_id,
            "class_name": detection.class_name,
            "confidence": detection.confidence,
            "source_tile_id": detection.tile_id,
            "tile_row_off": detection.tile_row_off,
            "tile_col_off": detection.tile_col_off,
            "tile_pixel_bounds": list(detection.tile_pixel_bounds),
            "local_obb_vertices": [list(point) for point in detection.local_obb_vertices],
            "global_pixel_obb_vertices": [
                list(point) for point in detection.global_pixel_obb_vertices
            ],
            "source_crs": detection.source_crs,
            "output_crs": detection.output_crs,
            "model_name": model_name,
            "model_task": model_task,
            "inference_conf": confidence,
            "inference_iou": model_iou,
            "tile_size": detection.tile_size,
            "tile_overlap": tile_overlap,
            "global_nms_iou": global_nms_iou,
        },
    }


def build_feature_collection(
    detections: Sequence[GlobalDetection],
    *,
    model_name: str,
    model_task: str,
    settings: SceneInferenceSettings,
) -> dict[str, Any]:
    """Build an RFC 7946 FeatureCollection without a legacy CRS member."""

    return {
        "type": "FeatureCollection",
        "features": [
            detection_to_geojson_feature(
                detection,
                model_name=model_name,
                model_task=model_task,
                confidence=settings.confidence,
                model_iou=settings.model_iou,
                tile_overlap=settings.tile_overlap,
                global_nms_iou=settings.global_nms_iou,
            )
            for detection in detections
        ],
    }


def validate_feature_collection(
    feature_collection: Mapping[str, Any],
    expected_count: int,
    raster_footprint_crs84: Any,
) -> dict[str, Any]:
    """Validate output geometry, coordinate order/ranges, and raster overlap."""

    from shapely.geometry import shape

    if feature_collection.get("type") != "FeatureCollection":
        raise GeospatialInferenceError("Output is not a GeoJSON FeatureCollection")
    if "crs" in feature_collection:
        raise GeospatialInferenceError("RFC 7946 output must not contain a CRS member")
    features = feature_collection.get("features")
    if not isinstance(features, list) or len(features) != expected_count:
        raise GeospatialInferenceError("GeoJSON feature count does not match detections")

    invalid_geometry_count = 0
    nonfinite_coordinate_count = 0
    out_of_range_coordinate_count = 0
    outside_footprint_count = 0
    invalid_confidence_count = 0
    for feature in features:
        polygon = shape(feature["geometry"])
        if polygon.is_empty or not polygon.is_valid or polygon.area <= 0.0:
            invalid_geometry_count += 1
        coordinates = feature["geometry"]["coordinates"][0]
        for longitude, latitude in coordinates:
            if not math.isfinite(longitude) or not math.isfinite(latitude):
                nonfinite_coordinate_count += 1
            if not -180.0 <= longitude <= 180.0 or not -90.0 <= latitude <= 90.0:
                out_of_range_coordinate_count += 1
        if not polygon.intersects(raster_footprint_crs84):
            outside_footprint_count += 1
        confidence = feature["properties"]["confidence"]
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            invalid_confidence_count += 1

    valid = all(
        count == 0
        for count in (
            invalid_geometry_count,
            nonfinite_coordinate_count,
            out_of_range_coordinate_count,
            outside_footprint_count,
            invalid_confidence_count,
        )
    )
    return {
        "valid": valid,
        "feature_count": len(features),
        "expected_feature_count": expected_count,
        "invalid_geometry_count": invalid_geometry_count,
        "nonfinite_coordinate_count": nonfinite_coordinate_count,
        "out_of_range_coordinate_count": out_of_range_coordinate_count,
        "outside_raster_footprint_count": outside_footprint_count,
        "invalid_confidence_count": invalid_confidence_count,
    }


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, allow_nan=False)
        output_file.write("\n")


def _diagnostic_geojson_polygon(
    global_vertices: Sequence[Sequence[float]], transform: Any, source_crs: Any
) -> dict[str, Any]:
    geographic_vertices = global_pixels_to_crs84(
        global_vertices, transform, source_crs
    )
    ring = [[float(longitude), float(latitude)] for longitude, latitude in geographic_vertices]
    ring.append(ring[0].copy())
    return {"type": "Polygon", "coordinates": [ring]}


def _suppression_dict(
    record: SuppressionRecord, transform: Any, source_crs: Any
) -> dict[str, Any]:
    value = asdict(record)
    value["polygon_iou"] = float(value["polygon_iou"])
    value["diagnostic_status"] = "suppressed_not_official_prediction"
    value["suppressed_geographic_crs84_polygon"] = _diagnostic_geojson_polygon(
        record.suppressed_global_pixel_obb_vertices, transform, source_crs
    )
    value["kept_geographic_crs84_polygon"] = _diagnostic_geojson_polygon(
        record.kept_global_pixel_obb_vertices, transform, source_crs
    )
    return value


def process_raster_scene(
    loaded_model: LoadedOBBModel,
    settings: SceneInferenceSettings,
    *,
    model_load_seconds: float = 0.0,
) -> SceneInferenceResult:
    """Run streaming tile inference and export one scene's validated outputs."""

    import numpy as np
    import rasterio
    from shapely.geometry import Polygon

    if not settings.raster_path.is_file():
        raise FileNotFoundError(f"Raster file not found: {settings.raster_path}")
    target_class_id, target_class_name = resolve_target_class(
        settings.target_class, loaded_model.names
    )
    start = time.perf_counter()
    scene_id = settings.raster_path.stem
    all_class_count = 0
    raw_target_count = 0
    non_target_count = 0
    raw_class_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    candidates: list[GlobalDetection] = []
    rejected: list[RejectedDetection] = []

    with rasterio.open(settings.raster_path) as dataset:
        if dataset.count != 3:
            raise GeospatialInferenceError(
                f"Expected a 3-band raster, found {dataset.count} bands"
            )
        if dataset.crs is None:
            raise GeospatialInferenceError("Raster has no CRS")
        raster_dtype = str(dataset.dtypes[0])
        if any(str(dtype) != raster_dtype for dtype in dataset.dtypes):
            raise GeospatialInferenceError("Raster bands have inconsistent datatypes")
        if raster_dtype != "uint8":
            raise GeospatialInferenceError(
                f"Expected uint8 PS-RGB data, found {raster_dtype}"
            )

        source_crs = dataset.crs
        transform = dataset.transform
        raster_width = dataset.width
        raster_height = dataset.height
        raster_bands = dataset.count
        windows = list(
            generate_tile_windows(
                scene_id,
                raster_width,
                raster_height,
                settings.tile_size,
                settings.tile_overlap,
            )
        )
        plan = validate_tile_plan(windows, raster_width, raster_height)
        if not plan.valid:
            raise GeospatialInferenceError(f"Invalid tile plan: {plan}")

        for tile_number, tile in enumerate(windows, 1):
            tile_read = read_tile(dataset, tile, pad_to_tile_size=True)
            chw_rgb = tile_read.data
            if chw_rgb.shape != (3, settings.tile_size, settings.tile_size):
                raise GeospatialInferenceError(
                    f"Unexpected padded tile shape {chw_rgb.shape} for {tile.tile_id}"
                )
            if chw_rgb.dtype != np.uint8:
                raise GeospatialInferenceError(
                    f"Unexpected tile dtype {chw_rgb.dtype} for {tile.tile_id}"
                )
            hwc_rgb = np.moveaxis(chw_rgb, 0, -1)
            model_input = np.ascontiguousarray(hwc_rgb[..., ::-1])
            predictions = loaded_model.model.predict(
                source=model_input,
                imgsz=settings.image_size,
                conf=settings.confidence,
                iou=settings.model_iou,
                device="cpu",
                save=False,
                verbose=False,
            )
            if len(predictions) != 1:
                raise GeospatialInferenceError(
                    f"Expected one result for {tile.tile_id}, found {len(predictions)}"
                )
            result = predictions[0]
            raw_detections = extract_ultralytics_obb(result, loaded_model.names, tile)
            all_class_count += len(raw_detections)
            raw_class_counts.update(
                detection.class_name for detection in raw_detections
            )
            target_detections, discarded = partition_target_detections(
                raw_detections, target_class_id
            )
            raw_target_count += len(target_detections)
            non_target_count += len(discarded)

            for detection in target_detections:
                try:
                    candidate = raw_detection_to_global(
                        detection, tile, source_crs
                    )
                except DetectionRejected as exc:
                    rejection_counts[exc.reason] += 1
                    rejected.append(
                        build_rejected_detection(detection, tile, exc.reason)
                    )
                else:
                    candidates.append(candidate)

            print(
                f"[{tile_number}/{len(windows)}] {tile.tile_id}: "
                f"raw={len(raw_detections)}, {target_class_name}={len(target_detections)}",
                flush=True,
            )
            del predictions, result, raw_detections
            del model_input, hwc_rgb, chw_rgb, tile_read

        nms_result = greedy_polygon_nms(candidates, settings.global_nms_iou)
        final_detections = tuple(
            add_geographic_geometry(detection, transform, source_crs)
            for detection in nms_result.kept
        )
        footprint_pixel_vertices = (
            (0.0, 0.0),
            (float(raster_width), 0.0),
            (float(raster_width), float(raster_height)),
            (0.0, float(raster_height)),
        )
        footprint_vertices = global_pixels_to_crs84(
            footprint_pixel_vertices, transform, source_crs
        )
        raster_footprint_crs84 = Polygon(footprint_vertices)

    feature_collection = build_feature_collection(
        final_detections,
        model_name=loaded_model.model_name,
        model_task=loaded_model.task,
        settings=settings,
    )
    spatial_validation = validate_feature_collection(
        feature_collection, len(final_detections), raster_footprint_crs84
    )
    if not spatial_validation["valid"]:
        raise GeospatialInferenceError(
            f"Generated GeoJSON failed spatial validation: {spatial_validation}"
        )

    scene_processing_seconds = time.perf_counter() - start
    summary = SceneInferenceSummary(
        scene_id=scene_id,
        raster_path=str(settings.raster_path),
        raster_width=raster_width,
        raster_height=raster_height,
        raster_bands=raster_bands,
        raster_dtype=raster_dtype,
        source_crs=str(source_crs),
        output_crs=OUTPUT_CRS,
        tile_count=len(windows),
        tile_size=settings.tile_size,
        tile_overlap=settings.tile_overlap,
        integer_stride=stride_from_overlap(
            settings.tile_size, settings.tile_overlap
        ),
        model_name=loaded_model.model_name,
        model_task=loaded_model.task,
        device="cpu",
        image_size=settings.image_size,
        inference_confidence=settings.confidence,
        inference_iou=settings.model_iou,
        global_nms_iou=settings.global_nms_iou,
        target_class_id=target_class_id,
        target_class_name=target_class_name,
        all_class_raw_detections=all_class_count,
        raw_class_counts=dict(sorted(raw_class_counts.items())),
        raw_target_candidates=raw_target_count,
        non_target_discarded=non_target_count,
        invalid_or_padding_rejected=len(rejected),
        rejection_counts=dict(sorted(rejection_counts.items())),
        pre_nms_target_count=len(candidates),
        duplicates_suppressed=len(nms_result.suppressed),
        final_target_count=len(final_detections),
        model_load_seconds=model_load_seconds,
        scene_processing_seconds=scene_processing_seconds,
        total_runtime_seconds=model_load_seconds + scene_processing_seconds,
        spatial_validation=spatial_validation,
    )

    output_dir = ensure_directory(settings.output_dir)
    predictions_path = output_dir / "predictions.geojson"
    summary_path = output_dir / "inference_summary.json"
    suppression_path = output_dir / "duplicate_suppressions.json"
    rejection_path = output_dir / "rejected_candidates.json"
    _write_json(predictions_path, feature_collection)
    _write_json(summary_path, summary.to_dict())
    _write_json(
        suppression_path,
        {
            "global_polygon_nms_iou": settings.global_nms_iou,
            "suppressed_count": len(nms_result.suppressed),
            "suppressions": [
                _suppression_dict(record, transform, source_crs)
                for record in nms_result.suppressed
            ],
        },
    )
    _write_json(
        rejection_path,
        {
            "rejected_count": len(rejected),
            "rejections": [asdict(record) for record in rejected],
        },
    )

    # Validate the serialized GeoJSON, not only the in-memory representation.
    with predictions_path.open("r", encoding="utf-8") as predictions_file:
        serialized_collection = json.load(predictions_file)
    serialized_validation = validate_feature_collection(
        serialized_collection, len(final_detections), raster_footprint_crs84
    )
    if serialized_validation != spatial_validation:
        raise GeospatialInferenceError(
            "Serialized GeoJSON validation differs from in-memory validation"
        )

    return SceneInferenceResult(
        summary=summary,
        detections=final_detections,
        suppressed=nms_result.suppressed,
        rejected=tuple(rejected),
        predictions_path=predictions_path,
        summary_path=summary_path,
        suppression_path=suppression_path,
        rejection_path=rejection_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CPU OBB inference over one georeferenced raster scene."
    )
    parser.add_argument("--input", required=True, help="Input georeferenced raster")
    parser.add_argument("--output", help="Per-scene output directory")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--model", help="OBB model path")
    parser.add_argument("--confidence", "--conf", type=float)
    parser.add_argument("--iou", dest="model_iou", type=float)
    parser.add_argument("--global-nms-iou", type=float)
    parser.add_argument("--target-class", default=EXPECTED_PLANE_CLASS_NAME)
    parser.add_argument("--device", help="Must be cpu")
    parser.add_argument("--image-size", "--imgsz", dest="image_size", type=int)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=float)
    return parser


def settings_from_args(args: argparse.Namespace) -> SceneInferenceSettings:
    root = find_project_root()
    config = load_project_config(args.config, project_root=root)
    configured_tile_size, configured_overlap = tiling_parameters_from_config(config)
    raster_path = resolve_project_path(args.input, project_root=root)
    model_path = resolve_project_path(
        args.model or config_value(config, "paths", "model"), project_root=root
    )
    configured_output = resolve_project_path(
        config_value(config, "paths", "output_root"), project_root=root
    )
    output_dir = (
        resolve_project_path(args.output, project_root=root)
        if args.output
        else configured_output / "geospatial_inference" / raster_path.stem
    )
    return SceneInferenceSettings(
        model_path=model_path,
        raster_path=raster_path,
        output_dir=output_dir,
        device=args.device or str(config_value(config, "runtime", "device")),
        image_size=(
            args.image_size
            if args.image_size is not None
            else int(config_value(config, "runtime", "image_size"))
        ),
        confidence=(
            args.confidence
            if args.confidence is not None
            else float(config_value(config, "inference", "confidence"))
        ),
        model_iou=(
            args.model_iou
            if args.model_iou is not None
            else float(config_value(config, "inference", "iou"))
        ),
        global_nms_iou=(
            args.global_nms_iou
            if args.global_nms_iou is not None
            else float(config_value(config, "inference", "global_nms_iou"))
        ),
        tile_size=(
            args.tile_size if args.tile_size is not None else configured_tile_size
        ),
        tile_overlap=(
            args.overlap if args.overlap is not None else configured_overlap
        ),
        target_class=args.target_class,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = settings_from_args(args)
        model_start = time.perf_counter()
        loaded_model = load_obb_model(settings.model_path, settings.device)
        model_load_seconds = time.perf_counter() - model_start
        result = process_raster_scene(
            loaded_model, settings, model_load_seconds=model_load_seconds
        )
    except (
        ConfigError,
        DetectionRejected,
        FileNotFoundError,
        GeospatialInferenceError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
        return 2

    print(json.dumps(result.summary.to_dict(), indent=2), flush=True)
    print(f"GeoJSON: {result.predictions_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
