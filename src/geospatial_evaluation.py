"""Fixed-threshold evaluation of geospatial OBB predictions.

The evaluator consumes an RFC 7946 prediction GeoJSON, an official
RarePlanes CRS84 annotation GeoJSON, and the corresponding source raster.
All matching is performed in continuous source-pixel space. RarePlanes
four-point aircraft diamonds are preserved and also normalized to minimum-
area rotated rectangles for the primary OBB-to-OBB comparison.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


CRS84 = "OGC:CRS84"
DEFAULT_MATCH_IOU = 0.50


class EvaluationError(RuntimeError):
    """Raised when evaluation inputs or geometry are incompatible."""


@dataclass(frozen=True)
class Prediction:
    prediction_id: str
    confidence: float
    source_tile_id: str
    geographic_polygon: Any
    pixel_polygon: Any
    properties: dict[str, Any]


@dataclass(frozen=True)
class GroundTruth:
    gt_index: int
    diamond_geographic_polygon: Any
    diamond_pixel_polygon: Any
    obb_pixel_polygon: Any
    properties: dict[str, Any]


@dataclass(frozen=True)
class Match:
    prediction_id: str
    gt_index: int
    confidence: float
    normalized_obb_iou: float
    original_diamond_iou: float
    center_error_pixels: float
    center_error_metres: float
    source_tile_id: str


@dataclass(frozen=True)
class FalsePositive:
    prediction_id: str
    confidence: float
    nearest_gt_iou: float | None
    nearest_gt_index: int | None
    source_tile_id: str


@dataclass(frozen=True)
class FalseNegative:
    gt_index: int
    role: Any
    role_id: Any
    centroid_pixel: tuple[float, float]
    centroid_crs84: tuple[float, float]
    length: Any
    wingspan: Any
    properties: dict[str, Any]
    nearest_prediction_id: str | None = None
    nearest_prediction_iou: float | None = None
    nearest_prediction_confidence: float | None = None


@dataclass(frozen=True)
class MatchingResult:
    matches: tuple[tuple[int, int, float], ...]
    false_prediction_indices: tuple[int, ...]
    false_gt_indices: tuple[int, ...]


@dataclass(frozen=True)
class EvaluationResult:
    summary: dict[str, Any]
    matches: tuple[Match, ...]
    false_positives: tuple[FalsePositive, ...]
    false_negatives: tuple[FalseNegative, ...]
    nms_diagnostics: dict[str, Any]
    boundary_rejection_diagnostics: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise EvaluationError(f"Expected a JSON object: {path}")
    return value


def _validate_feature_collection(value: Mapping[str, Any], label: str) -> list[Any]:
    if value.get("type") != "FeatureCollection":
        raise EvaluationError(f"{label} is not a GeoJSON FeatureCollection")
    features = value.get("features")
    if not isinstance(features, list):
        raise EvaluationError(f"{label} has no feature list")
    return features


def polygon_iou(first: Any, second: Any) -> float:
    """Return true polygon IoU, with invalid/degenerate inputs scoring zero."""

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
        intersection = float(first.intersection(second).area)
        if intersection <= 0.0:
            return 0.0
        union = float(first.union(second).area)
        return intersection / union if math.isfinite(union) and union > 0.0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def minimum_rotated_rectangle(polygon: Any) -> Any:
    """Normalize a valid RarePlanes diamond to its minimum-area rectangle."""

    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0.0:
        raise EvaluationError("Cannot normalize an invalid ground-truth polygon")
    rectangle = polygon.minimum_rotated_rectangle
    if rectangle.is_empty or not rectangle.is_valid or rectangle.area <= 0.0:
        raise EvaluationError("Minimum rotated rectangle is invalid")
    return rectangle


def make_crs84_to_raster_transformer(raster_crs: Any) -> Any | None:
    """Return an always-XY transformer, or ``None`` for CRS84-equivalent data."""

    from pyproj import CRS, Transformer

    source = CRS.from_user_input(CRS84)
    target = CRS.from_user_input(raster_crs)
    if source.equals(target, ignore_axis_order=True):
        return None
    return Transformer.from_crs(source, target, always_xy=True)


def make_raster_to_crs84_transformer(raster_crs: Any) -> Any | None:
    """Return a raster-to-CRS84 always-XY transformer when needed."""

    from pyproj import CRS, Transformer

    source = CRS.from_user_input(raster_crs)
    target = CRS.from_user_input(CRS84)
    if source.equals(target, ignore_axis_order=True):
        return None
    return Transformer.from_crs(source, target, always_xy=True)


def geographic_polygon_to_pixel(
    polygon: Any,
    raster_transform: Any,
    crs84_to_raster: Any | None,
) -> Any:
    """Convert a CRS84 polygon to continuous full-scene pixel coordinates."""

    from shapely.geometry import Polygon

    coordinates = list(polygon.exterior.coords)
    pixel_coordinates: list[tuple[float, float]] = []
    inverse = ~raster_transform
    for longitude, latitude in coordinates:
        x, y = float(longitude), float(latitude)
        if crs84_to_raster is not None:
            x, y = crs84_to_raster.transform(x, y)
        column, row = inverse @ (x, y)
        pixel_coordinates.append((float(column), float(row)))
    converted = Polygon(pixel_coordinates)
    if converted.is_empty or not converted.is_valid or converted.area <= 0.0:
        raise EvaluationError("CRS84 polygon became invalid in source-pixel space")
    return converted


def pixel_point_to_crs84(
    column: float,
    row: float,
    raster_transform: Any,
    raster_to_crs84: Any | None,
) -> tuple[float, float]:
    """Convert one continuous pixel-space point to longitude/latitude."""

    x, y = raster_transform @ (float(column), float(row))
    if raster_to_crs84 is not None:
        x, y = raster_to_crs84.transform(x, y)
    return float(x), float(y)


def round_trip_error(
    geographic_polygon: Any,
    pixel_polygon: Any,
    raster_transform: Any,
    raster_to_crs84: Any | None,
) -> float:
    """Return maximum world-coordinate round-trip error for paired vertices."""

    world_vertices = list(geographic_polygon.exterior.coords)
    pixel_vertices = list(pixel_polygon.exterior.coords)
    if len(world_vertices) != len(pixel_vertices):
        raise EvaluationError("Round-trip vertex sequences are misaligned")
    maximum = 0.0
    for (expected_x, expected_y), (column, row) in zip(
        world_vertices, pixel_vertices, strict=True
    ):
        actual_x, actual_y = pixel_point_to_crs84(
            column, row, raster_transform, raster_to_crs84
        )
        maximum = max(
            maximum,
            abs(float(expected_x) - actual_x),
            abs(float(expected_y) - actual_y),
        )
    return maximum


def greedy_confidence_match(
    prediction_polygons: Sequence[Any],
    confidences: Sequence[float],
    gt_polygons: Sequence[Any],
    threshold: float = DEFAULT_MATCH_IOU,
) -> MatchingResult:
    """One-to-one confidence-ordered matching using true polygon IoU."""

    if len(prediction_polygons) != len(confidences):
        raise ValueError("Prediction polygons and confidences must align")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Match IoU threshold must be in [0, 1]")

    order = sorted(range(len(prediction_polygons)), key=lambda i: (-confidences[i], i))
    unmatched_gt = set(range(len(gt_polygons)))
    matches: list[tuple[int, int, float]] = []
    false_predictions: list[int] = []

    for prediction_index in order:
        candidates = [
            (polygon_iou(prediction_polygons[prediction_index], gt_polygons[gt_index]), gt_index)
            for gt_index in sorted(unmatched_gt)
        ]
        if not candidates:
            false_predictions.append(prediction_index)
            continue
        best_iou, best_gt_index = max(candidates, key=lambda item: (item[0], -item[1]))
        if best_iou >= threshold:
            matches.append((prediction_index, best_gt_index, best_iou))
            unmatched_gt.remove(best_gt_index)
        else:
            false_predictions.append(prediction_index)

    return MatchingResult(
        matches=tuple(matches),
        false_prediction_indices=tuple(false_predictions),
        false_gt_indices=tuple(sorted(unmatched_gt)),
    )


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def descriptive_statistics(values: Sequence[float]) -> dict[str, float | None]:
    """Return stable summary statistics, using nulls for an empty sample."""

    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    finite = [float(value) for value in values]
    return {
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "min": min(finite),
        "max": max(finite),
    }


def geodesic_distance_metres(
    first_crs84: tuple[float, float], second_crs84: tuple[float, float]
) -> float:
    """Calculate WGS84 ellipsoidal point distance with longitude/latitude order."""

    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    _, _, distance = geod.inv(
        first_crs84[0], first_crs84[1], second_crs84[0], second_crs84[1]
    )
    return abs(float(distance))


def _load_predictions(
    value: Mapping[str, Any],
    scene_id: str,
    raster_transform: Any,
    crs84_to_raster: Any | None,
) -> tuple[Prediction, ...]:
    from shapely.geometry import shape

    predictions: list[Prediction] = []
    for index, feature in enumerate(_validate_feature_collection(value, "Predictions")):
        properties = dict(feature.get("properties") or {})
        if properties.get("scene_id") != scene_id:
            raise EvaluationError(
                f"Prediction feature {index} belongs to {properties.get('scene_id')!r}, "
                f"not {scene_id!r}"
            )
        geographic = shape(feature.get("geometry"))
        if geographic.geom_type != "Polygon" or geographic.is_empty or not geographic.is_valid:
            raise EvaluationError(f"Prediction feature {index} is not a valid Polygon")
        confidence = float(properties.get("confidence"))
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise EvaluationError(f"Prediction feature {index} has invalid confidence")
        prediction_id = str(properties.get("detection_id") or feature.get("id") or f"prediction_{index}")
        predictions.append(
            Prediction(
                prediction_id=prediction_id,
                confidence=confidence,
                source_tile_id=str(properties.get("source_tile_id", "")),
                geographic_polygon=geographic,
                pixel_polygon=geographic_polygon_to_pixel(
                    geographic, raster_transform, crs84_to_raster
                ),
                properties=properties,
            )
        )
    return tuple(predictions)


def _load_ground_truth(
    value: Mapping[str, Any],
    scene_id: str,
    raster_transform: Any,
    crs84_to_raster: Any | None,
) -> tuple[GroundTruth, ...]:
    from shapely.geometry import shape

    crs_name = (((value.get("crs") or {}).get("properties") or {}).get("name"))
    if crs_name not in (None, "urn:ogc:def:crs:OGC:1.3:CRS84", CRS84):
        raise EvaluationError(f"Unexpected ground-truth CRS: {crs_name!r}")

    truths: list[GroundTruth] = []
    for index, feature in enumerate(_validate_feature_collection(value, "Ground truth")):
        properties = dict(feature.get("properties") or {})
        expected_id = f"{properties.get('loc_id')}_{properties.get('cat_id')}"
        if expected_id != scene_id:
            raise EvaluationError(
                f"Ground-truth feature {index} pairs to {expected_id!r}, not {scene_id!r}"
            )
        geographic = shape(feature.get("geometry"))
        if geographic.geom_type != "Polygon" or geographic.is_empty or not geographic.is_valid:
            raise EvaluationError(f"Ground-truth feature {index} is not a valid Polygon")
        pixel_diamond = geographic_polygon_to_pixel(
            geographic, raster_transform, crs84_to_raster
        )
        truths.append(
            GroundTruth(
                gt_index=index,
                diamond_geographic_polygon=geographic,
                diamond_pixel_polygon=pixel_diamond,
                obb_pixel_polygon=minimum_rotated_rectangle(pixel_diamond),
                properties=properties,
            )
        )
    return tuple(truths)


def _nearest_ground_truth(prediction: Prediction, truths: Sequence[GroundTruth]) -> tuple[float | None, int | None]:
    if not truths:
        return None, None
    candidates = [
        (polygon_iou(prediction.pixel_polygon, truth.obb_pixel_polygon), truth.gt_index)
        for truth in truths
    ]
    overlap, index = max(candidates, key=lambda item: (item[0], -item[1]))
    return overlap, index


def _nearest_prediction(
    truth: GroundTruth, predictions: Sequence[Prediction]
) -> tuple[str | None, float | None, float | None]:
    if not predictions:
        return None, None, None
    prediction = max(
        predictions,
        key=lambda item: (
            polygon_iou(item.pixel_polygon, truth.obb_pixel_polygon),
            item.confidence,
            item.prediction_id,
        ),
    )
    return (
        prediction.prediction_id,
        polygon_iou(prediction.pixel_polygon, truth.obb_pixel_polygon),
        prediction.confidence,
    )


def _diagnostic_pixel_polygon(
    value: Mapping[str, Any], vertex_key: str
) -> Any | None:
    """Reconstruct valid diagnostic geometry without clipping or repairing it."""

    from shapely.geometry import Polygon

    vertices = value.get(vertex_key)
    if not isinstance(vertices, Sequence) or isinstance(vertices, (str, bytes)):
        return None
    try:
        if len(vertices) != 4:
            return None
        polygon = Polygon(
            [(float(point[0]), float(point[1])) for point in vertices]
        )
    except (IndexError, TypeError, ValueError, OverflowError):
        return None
    if (
        polygon.is_empty
        or not polygon.is_valid
        or not math.isfinite(polygon.area)
        or polygon.area <= 0.0
    ):
        return None
    return polygon


def _diagnostic_best_gt(
    polygon: Any,
    truths: Sequence[GroundTruth],
    allowed_gt_indices: set[int] | None = None,
) -> tuple[int | None, float | None]:
    eligible_truths = [
        truth
        for truth in truths
        if allowed_gt_indices is None or truth.gt_index in allowed_gt_indices
    ]
    if polygon is None or not eligible_truths:
        return None, None
    overlap, gt_index = max(
        (
            polygon_iou(polygon, truth.obb_pixel_polygon),
            truth.gt_index,
        )
        for truth in eligible_truths
    )
    return gt_index, overlap


def _build_nms_diagnostics(
    suppression_value: Mapping[str, Any] | None,
    matches: Sequence[Match],
    false_positives: Sequence[FalsePositive],
    truths: Sequence[GroundTruth],
    unmatched_gt_indices: Sequence[int],
    match_iou: float,
) -> dict[str, Any]:
    if suppression_value is None:
        return {
            "available": False,
            "conclusion": "No duplicate-suppression artifact was supplied.",
        }
    tp_by_id = {match.prediction_id: match for match in matches}
    fp_by_id = {item.prediction_id: item for item in false_positives}
    unmatched = set(unmatched_gt_indices)
    records = []
    retained_geometry_count = 0
    matching_current_fn_count = 0
    for item_value in suppression_value.get("suppressions", []):
        item = dict(item_value)
        kept_id = str(item.get("kept_detection_id"))
        polygon = _diagnostic_pixel_polygon(
            item, "suppressed_global_pixel_obb_vertices"
        )
        reconstructable = polygon is not None
        retained_geometry_count += int(reconstructable)
        best_gt_index, best_gt_iou = _diagnostic_best_gt(polygon, truths)
        best_fn_index, best_fn_iou = _diagnostic_best_gt(
            polygon, truths, unmatched
        )
        meets_threshold = bool(
            best_gt_iou is not None and best_gt_iou >= match_iou
        )
        matches_current_fn = bool(
            best_fn_iou is not None and best_fn_iou >= match_iou
        )
        matching_current_fn_count += int(matches_current_fn)
        if kept_id in tp_by_id:
            accepted_status = "kept counterpart is an official TP"
        elif kept_id in fp_by_id:
            accepted_status = "kept counterpart is an official FP"
        else:
            accepted_status = "kept counterpart is absent from official predictions"
        records.append(
            {
                **item,
                "geometry_reconstructable": reconstructable,
                "best_gt_index": best_gt_index,
                "best_gt_normalized_obb_iou": best_gt_iou,
                "meets_match_iou_threshold": meets_threshold,
                "best_gt_is_current_fn": bool(best_gt_index in unmatched),
                "best_current_fn_gt_index": best_fn_index,
                "best_current_fn_normalized_obb_iou": best_fn_iou,
                "would_independently_match_current_fn": matches_current_fn,
                "accepted_counterpart_status": accepted_status,
            }
        )
    return {
        "available": True,
        "match_iou_threshold": match_iou,
        "suppressed_count": len(records),
        "suppressed_with_reconstructable_geometry": retained_geometry_count,
        "suppressed_independently_matching_current_fn": matching_current_fn_count,
        "records": records,
        "conclusion": (
            f"{matching_current_fn_count} suppressed candidates independently meet the "
            f"fixed IoU threshold for a currently unmatched GT object."
        ),
    }


def _build_boundary_diagnostics(
    rejection_value: Mapping[str, Any] | None,
    truths: Sequence[GroundTruth],
    unmatched_gt_indices: Sequence[int],
    match_iou: float,
) -> dict[str, Any]:
    if rejection_value is None:
        return {"available": False, "conclusion": "No rejection artifact was supplied."}
    unmatched = set(unmatched_gt_indices)
    records = []
    reconstructable_count = 0
    matching_current_fn_count = 0
    for item_value in rejection_value.get("rejections", []):
        item = dict(item_value)
        polygon = None
        if item.get("attempted_global_polygon_valid") is True:
            polygon = _diagnostic_pixel_polygon(
                item, "attempted_global_pixel_obb_vertices"
            )
        reconstructable = polygon is not None
        reconstructable_count += int(reconstructable)
        best_gt_index, best_gt_iou = _diagnostic_best_gt(polygon, truths)
        best_fn_index, best_fn_iou = _diagnostic_best_gt(
            polygon, truths, unmatched
        )
        meets_threshold = bool(
            best_gt_iou is not None and best_gt_iou >= match_iou
        )
        best_gt_already_matched = bool(
            best_gt_index is not None and best_gt_index not in unmatched
        )
        matches_current_fn = bool(
            best_fn_iou is not None and best_fn_iou >= match_iou
        )
        matching_current_fn_count += int(matches_current_fn)
        records.append(
            {
                **item,
                "geometry_reconstructable": reconstructable,
                "best_gt_index": best_gt_index,
                "best_gt_normalized_obb_iou": best_gt_iou,
                "meets_match_iou_threshold": meets_threshold,
                "best_gt_already_matched_by_accepted_prediction": (
                    best_gt_already_matched
                ),
                "best_gt_is_current_fn": bool(best_gt_index in unmatched),
                "best_current_fn_gt_index": best_fn_index,
                "best_current_fn_normalized_obb_iou": best_fn_iou,
                "would_independently_match_current_fn": matches_current_fn,
            }
        )
    return {
        "available": True,
        "match_iou_threshold": match_iou,
        "rejected_count": len(records),
        "rejected_with_reconstructable_geometry": reconstructable_count,
        "rejected_independently_matching_current_fn": matching_current_fn_count,
        "records": records,
        "conclusion": (
            f"{matching_current_fn_count} rejected candidates independently meet the fixed "
            "IoU threshold for a currently unmatched GT object."
        ),
    }


def evaluate_scene(
    prediction_geojson: Path,
    ground_truth_geojson: Path,
    raster_path: Path,
    *,
    match_iou: float = DEFAULT_MATCH_IOU,
    suppression_json: Path | None = None,
    rejection_json: Path | None = None,
) -> EvaluationResult:
    """Evaluate one exactly paired scene without executing model inference."""

    import rasterio

    if not 0.0 <= match_iou <= 1.0:
        raise ValueError("Match IoU threshold must be in [0, 1]")
    prediction_value = _read_json(prediction_geojson)
    ground_truth_value = _read_json(ground_truth_geojson)
    scene_id = raster_path.stem
    if ground_truth_geojson.stem != scene_id:
        raise EvaluationError("Ground-truth filename does not match raster scene ID")

    with rasterio.open(raster_path) as raster:
        if raster.crs is None:
            raise EvaluationError("Source raster has no CRS")
        raster_crs = raster.crs
        raster_transform = raster.transform
        raster_width, raster_height = raster.width, raster.height

    to_raster = make_crs84_to_raster_transformer(raster_crs)
    to_crs84 = make_raster_to_crs84_transformer(raster_crs)
    predictions = _load_predictions(
        prediction_value, scene_id, raster_transform, to_raster
    )
    truths = _load_ground_truth(
        ground_truth_value, scene_id, raster_transform, to_raster
    )

    round_trip_errors = [
        round_trip_error(
            prediction.geographic_polygon,
            prediction.pixel_polygon,
            raster_transform,
            to_crs84,
        )
        for prediction in predictions
    ] + [
        round_trip_error(
            truth.diamond_geographic_polygon,
            truth.diamond_pixel_polygon,
            raster_transform,
            to_crs84,
        )
        for truth in truths
    ]
    stored_pixel_errors: list[float] = []
    for prediction in predictions:
        stored_vertices = prediction.properties.get("global_pixel_obb_vertices")
        if stored_vertices is None:
            continue
        reconstructed_vertices = list(prediction.pixel_polygon.exterior.coords)[:-1]
        if len(stored_vertices) != len(reconstructed_vertices):
            raise EvaluationError(
                f"Stored pixel vertices are malformed for {prediction.prediction_id}"
            )
        for stored, reconstructed in zip(
            stored_vertices, reconstructed_vertices, strict=True
        ):
            stored_pixel_errors.append(
                math.hypot(
                    float(stored[0]) - float(reconstructed[0]),
                    float(stored[1]) - float(reconstructed[1]),
                )
            )
    maximum_stored_pixel_error = max(stored_pixel_errors, default=None)
    if maximum_stored_pixel_error is not None and maximum_stored_pixel_error > 1e-6:
        raise EvaluationError(
            "CRS84-to-pixel reconstruction disagrees with stored prediction vertices: "
            f"maximum error {maximum_stored_pixel_error} pixels"
        )

    matching = greedy_confidence_match(
        [item.pixel_polygon for item in predictions],
        [item.confidence for item in predictions],
        [item.obb_pixel_polygon for item in truths],
        match_iou,
    )
    matches: list[Match] = []
    for prediction_index, gt_index, obb_iou in matching.matches:
        prediction = predictions[prediction_index]
        truth = truths[gt_index]
        prediction_centroid = prediction.pixel_polygon.centroid
        truth_centroid = truth.diamond_pixel_polygon.centroid
        center_error_pixels = float(prediction_centroid.distance(truth_centroid))
        prediction_center_crs84 = pixel_point_to_crs84(
            prediction_centroid.x, prediction_centroid.y, raster_transform, to_crs84
        )
        truth_center_crs84 = pixel_point_to_crs84(
            truth_centroid.x, truth_centroid.y, raster_transform, to_crs84
        )
        matches.append(
            Match(
                prediction_id=prediction.prediction_id,
                gt_index=truth.gt_index,
                confidence=prediction.confidence,
                normalized_obb_iou=obb_iou,
                original_diamond_iou=polygon_iou(
                    prediction.pixel_polygon, truth.diamond_pixel_polygon
                ),
                center_error_pixels=center_error_pixels,
                center_error_metres=geodesic_distance_metres(
                    prediction_center_crs84, truth_center_crs84
                ),
                source_tile_id=prediction.source_tile_id,
            )
        )

    false_positives: list[FalsePositive] = []
    for index in matching.false_prediction_indices:
        prediction = predictions[index]
        nearest_iou, nearest_index = _nearest_ground_truth(prediction, truths)
        false_positives.append(
            FalsePositive(
                prediction_id=prediction.prediction_id,
                confidence=prediction.confidence,
                nearest_gt_iou=nearest_iou,
                nearest_gt_index=nearest_index,
                source_tile_id=prediction.source_tile_id,
            )
        )

    false_negatives: list[FalseNegative] = []
    for gt_index in matching.false_gt_indices:
        truth = truths[gt_index]
        centroid = truth.obb_pixel_polygon.centroid
        properties = truth.properties
        nearest_prediction_id, nearest_prediction_iou, nearest_confidence = (
            _nearest_prediction(truth, predictions)
        )
        false_negatives.append(
            FalseNegative(
                gt_index=truth.gt_index,
                role=properties.get("role"),
                role_id=properties.get("role_id"),
                centroid_pixel=(float(centroid.x), float(centroid.y)),
                centroid_crs84=pixel_point_to_crs84(
                    centroid.x, centroid.y, raster_transform, to_crs84
                ),
                length=properties.get("length"),
                wingspan=properties.get("wingspan"),
                properties=properties,
                nearest_prediction_id=nearest_prediction_id,
                nearest_prediction_iou=nearest_prediction_iou,
                nearest_prediction_confidence=nearest_confidence,
            )
        )

    tp, fp, fn = len(matches), len(false_positives), len(false_negatives)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    obb_statistics = descriptive_statistics([item.normalized_obb_iou for item in matches])
    diamond_statistics = descriptive_statistics([item.original_diamond_iou for item in matches])
    confidence_statistics = descriptive_statistics([item.confidence for item in matches])
    pixel_error_statistics = descriptive_statistics([item.center_error_pixels for item in matches])
    metre_error_statistics = descriptive_statistics([item.center_error_metres for item in matches])
    geometry_effect_statistics = descriptive_statistics(
        [
            item.normalized_obb_iou - item.original_diamond_iou
            for item in matches
        ]
    )
    mean_geometry_effect = (
        float(obb_statistics["mean"]) - float(diamond_statistics["mean"])
        if obb_statistics["mean"] is not None and diamond_statistics["mean"] is not None
        else None
    )

    suppression_value = _read_json(suppression_json) if suppression_json else None
    rejection_value = _read_json(rejection_json) if rejection_json else None
    nms_diagnostics = _build_nms_diagnostics(
        suppression_value,
        matches,
        false_positives,
        truths,
        matching.false_gt_indices,
        match_iou,
    )
    boundary_diagnostics = _build_boundary_diagnostics(
        rejection_value,
        truths,
        matching.false_gt_indices,
        match_iou,
    )
    summary = {
        "scene_id": scene_id,
        "prediction_geojson": str(prediction_geojson),
        "ground_truth_geojson": str(ground_truth_geojson),
        "raster_path": str(raster_path),
        "raster_crs": str(raster_crs),
        "raster_width": raster_width,
        "raster_height": raster_height,
        "prediction_coordinate_convention": "RFC 7946 CRS84 longitude, latitude",
        "ground_truth_coordinate_convention": "declared OGC CRS84 longitude, latitude",
        "evaluation_space": "continuous full-scene source pixels",
        "primary_match_geometry": "prediction rectangular OBB vs GT minimum-area rotated rectangle",
        "secondary_geometry": "prediction rectangular OBB vs original RarePlanes diamond",
        "match_iou_threshold": match_iou,
        "ground_truth_count": len(truths),
        "prediction_count": len(predictions),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_obb_iou": obb_statistics,
        "matched_prediction_confidence": confidence_statistics,
        "original_diamond_iou": diamond_statistics,
        "annotation_geometry_effect_mean_iou_points": mean_geometry_effect,
        "annotation_geometry_effect_obb_minus_diamond_iou": geometry_effect_statistics,
        "center_error_pixels": pixel_error_statistics,
        "center_error_metres": metre_error_statistics,
        "center_error_gt_reference": "centroid of unchanged official RarePlanes diamond",
        "maximum_crs84_round_trip_error_degrees": max(round_trip_errors, default=0.0),
        "maximum_prediction_stored_pixel_error": maximum_stored_pixel_error,
        "matching_rule": "confidence-descending greedy one-to-one polygon IoU",
        "metric_scope": "fixed-threshold single-scene measurement; not mAP",
    }
    return EvaluationResult(
        summary=summary,
        matches=tuple(matches),
        false_positives=tuple(false_positives),
        false_negatives=tuple(false_negatives),
        nms_diagnostics=nms_diagnostics,
        boundary_rejection_diagnostics=boundary_diagnostics,
    )


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation_outputs(result: EvaluationResult, output_dir: Path) -> tuple[Path, ...]:
    """Write deterministic evaluation artifacts without changing source data."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "evaluation_summary.json"
    matches_path = output_dir / "matches.csv"
    false_positives_path = output_dir / "false_positives.csv"
    false_negatives_path = output_dir / "false_negatives.csv"
    boundary_path = output_dir / "boundary_rejection_diagnostics.json"
    nms_path = output_dir / "nms_diagnostics.json"

    _write_json(summary_path, result.summary)
    _write_csv(
        matches_path,
        [asdict(item) for item in result.matches],
        tuple(Match.__dataclass_fields__),
    )
    _write_csv(
        false_positives_path,
        [asdict(item) for item in result.false_positives],
        tuple(FalsePositive.__dataclass_fields__),
    )
    fn_rows = []
    for item in result.false_negatives:
        row = asdict(item)
        row["properties"] = json.dumps(row["properties"], ensure_ascii=False, sort_keys=True)
        fn_rows.append(row)
    _write_csv(
        false_negatives_path,
        fn_rows,
        tuple(FalseNegative.__dataclass_fields__),
    )
    _write_json(boundary_path, result.boundary_rejection_diagnostics)
    _write_json(nms_path, result.nms_diagnostics)
    return (
        summary_path,
        matches_path,
        false_positives_path,
        false_negatives_path,
        boundary_path,
        nms_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one geospatial OBB prediction scene against RarePlanes ground truth."
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--raster", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--match-iou", type=float, default=DEFAULT_MATCH_IOU)
    parser.add_argument("--suppressions", type=Path)
    parser.add_argument("--rejections", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_scene(
        args.predictions,
        args.ground_truth,
        args.raster,
        match_iou=args.match_iou,
        suppression_json=args.suppressions,
        rejection_json=args.rejections,
    )
    paths = write_evaluation_outputs(result, args.output_dir)
    print(json.dumps(result.summary, indent=2, allow_nan=False))
    print("Output files:")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
