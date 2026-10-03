"""Offline forensic analysis of tile-boundary OBB rejection.

This module consumes completed, ignored TRAIN inference/evaluation artifacts.
It never executes YOLO and never changes the production acceptance rule.  The
alternative policies are controlled counterfactuals that reuse the production
polygon NMS and confidence-ordered polygon matching implementations.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence

try:
    from .batch_evaluation import MATCH_IOU, GLOBAL_NMS_IOU
    from .geospatial_evaluation import (
        _load_ground_truth,
        greedy_confidence_match,
        make_crs84_to_raster_transformer,
        polygon_iou,
        precision_recall_f1,
    )
    from .geospatial_inference import GlobalDetection, greedy_polygon_nms
    from .geospatial_tiling import generate_tile_windows
    from .project_config import find_project_root
except ImportError:  # Support direct execution.
    from batch_evaluation import MATCH_IOU, GLOBAL_NMS_IOU
    from geospatial_evaluation import (
        _load_ground_truth,
        greedy_confidence_match,
        make_crs84_to_raster_transformer,
        polygon_iou,
        precision_recall_f1,
    )
    from geospatial_inference import GlobalDetection, greedy_polygon_nms
    from geospatial_tiling import generate_tile_windows
    from project_config import find_project_root


POLICIES = ("current", "centroid", "area90", "area75")
TILE_ID_PATTERN = re.compile(r"__r(?P<row>\d+)_c(?P<col>\d+)$")


class BoundaryAnalysisError(RuntimeError):
    """Raised when retained artifacts are incomplete or inconsistent."""


@dataclass(frozen=True)
class RejectedGeometry:
    detection: GlobalDetection
    rejection_reason: str
    outside_vertex_count: int
    crossed_sides: tuple[str, ...]
    maximum_excursion_pixels: float
    polygon_area: float
    intersection_area: float
    inside_tile_fraction: float
    centroid_local: tuple[float, float]
    centroid_inside_tile: bool
    attempted_global_polygon_valid: bool
    global_polygon_inside_source: bool
    tile_edge_class: str
    has_padding: bool
    boundary_category: str


@dataclass(frozen=True)
class PolicyResult:
    policy: str
    admitted_rejected: int
    admitted_removed_by_nms: int
    predictions: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    kept: tuple[GlobalDetection, ...]
    matches: tuple[tuple[int, int, float], ...]
    false_gt_indices: tuple[int, ...]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _points(value: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    converted = tuple((float(point[0]), float(point[1])) for point in value)
    if len(converted) != 4 or any(not math.isfinite(v) for point in converted for v in point):
        raise BoundaryAnalysisError("Expected four finite OBB vertices")
    return converted


def _tile_offsets(tile_id: str) -> tuple[int, int]:
    match = TILE_ID_PATTERN.search(tile_id)
    if match is None:
        raise BoundaryAnalysisError(f"Cannot parse tile offsets: {tile_id}")
    return int(match.group("row")), int(match.group("col"))


def _tile_edge_class(
    bounds: Sequence[float], source_width: int, source_height: int
) -> str:
    _, _, right, bottom = (float(value) for value in bounds)
    right_edge = math.isclose(right, source_width, abs_tol=1e-9)
    bottom_edge = math.isclose(bottom, source_height, abs_tol=1e-9)
    if right_edge and bottom_edge:
        return "corner-edge anchored"
    if right_edge:
        return "right-edge anchored"
    if bottom_edge:
        return "bottom-edge anchored"
    return "interior sliding tile"


def classify_rejected_geometry(
    record: Mapping[str, Any], source_width: int, source_height: int, tile_size: int = 1024
) -> RejectedGeometry:
    """Measure a retained rejected candidate without changing its status."""

    from shapely.geometry import Point, Polygon, box

    local_vertices = _points(record["local_obb_vertices"])
    global_value = record.get("attempted_global_pixel_obb_vertices")
    if global_value is None:
        raise BoundaryAnalysisError(
            f"Rejected candidate has no reconstructable geometry: {record['detection_id']}"
        )
    global_vertices = _points(global_value)
    valid = tuple(float(value) for value in record["tile_valid_region"])
    source_tile_bounds = tuple(float(value) for value in record["tile_source_image_bounds"])
    left, top, right, bottom = valid
    outside_count = sum(
        not (left <= x <= right and top <= y <= bottom) for x, y in local_vertices
    )
    sides: set[str] = set()
    excursions: list[float] = []
    for x, y in local_vertices:
        if x < left:
            sides.add("left")
            excursions.append(left - x)
        if x > right:
            sides.add("right")
            excursions.append(x - right)
        if y < top:
            sides.add("top")
            excursions.append(top - y)
        if y > bottom:
            sides.add("bottom")
            excursions.append(y - bottom)

    local_polygon = Polygon(local_vertices)
    global_polygon = Polygon(global_vertices)
    valid_box = box(left, top, right, bottom)
    source_box = box(0.0, 0.0, float(source_width), float(source_height))
    polygon_valid = bool(
        record.get("attempted_global_polygon_valid")
        and not global_polygon.is_empty
        and global_polygon.is_valid
        and math.isfinite(global_polygon.area)
        and global_polygon.area > 0.0
    )
    area = float(local_polygon.area) if local_polygon.is_valid else 0.0
    intersection = float(local_polygon.intersection(valid_box).area) if area > 0.0 else 0.0
    inside_fraction = intersection / area if area > 0.0 else 0.0
    centroid = local_polygon.centroid
    centroid_inside = bool(valid_box.covers(Point(centroid.x, centroid.y)))
    source_inside = bool(polygon_valid and source_box.covers(global_polygon))
    width = source_tile_bounds[2] - source_tile_bounds[0]
    height = source_tile_bounds[3] - source_tile_bounds[1]
    has_padding = width < tile_size or height < tile_size
    if not polygon_valid:
        category = "other_invalid"
    elif has_padding:
        category = "padding_related"
    elif not source_inside:
        category = "source_raster_boundary"
    else:
        category = "internal_tile_boundary"

    row_off, col_off = int(record["row_off"]), int(record["col_off"])
    detection = GlobalDetection(
        detection_id=str(record["detection_id"]),
        scene_id=str(record["scene_id"]),
        tile_id=str(record["tile_id"]),
        tile_row_off=row_off,
        tile_col_off=col_off,
        tile_pixel_bounds=source_tile_bounds,
        tile_valid_region=valid,
        tile_size=tile_size,
        source_width=source_width,
        source_height=source_height,
        class_id=int(record["class_id"]),
        class_name=str(record["class_name"]),
        confidence=float(record["confidence"]),
        local_obb_vertices=local_vertices,  # type: ignore[arg-type]
        global_pixel_obb_vertices=global_vertices,  # type: ignore[arg-type]
        global_pixel_polygon=global_polygon,
        source_crs="EPSG:4326",
    )
    return RejectedGeometry(
        detection=detection,
        rejection_reason=str(record["rejection_reason"]),
        outside_vertex_count=outside_count,
        crossed_sides=tuple(side for side in ("left", "right", "top", "bottom") if side in sides),
        maximum_excursion_pixels=max(excursions, default=0.0),
        polygon_area=area,
        intersection_area=intersection,
        inside_tile_fraction=inside_fraction,
        centroid_local=(float(centroid.x), float(centroid.y)),
        centroid_inside_tile=centroid_inside,
        attempted_global_polygon_valid=polygon_valid,
        global_polygon_inside_source=source_inside,
        tile_edge_class=_tile_edge_class(source_tile_bounds, source_width, source_height),
        has_padding=has_padding,
        boundary_category=category,
    )


def policy_accepts(candidate: RejectedGeometry, policy: str) -> bool:
    """Return whether an offline policy admits one rejected candidate."""

    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    if policy == "current":
        return False
    common = (
        candidate.attempted_global_polygon_valid
        and candidate.global_polygon_inside_source
    )
    if policy == "centroid":
        return common and candidate.centroid_inside_tile
    threshold = 0.90 if policy == "area90" else 0.75
    return common and candidate.inside_tile_fraction >= threshold


def _global_detection(
    *,
    detection_id: str,
    scene_id: str,
    tile_id: str,
    confidence: float,
    local_vertices: Sequence[Sequence[float]],
    global_vertices: Sequence[Sequence[float]],
    source_width: int,
    source_height: int,
    tile_size: int,
    class_id: int = 0,
    class_name: str = "plane",
) -> GlobalDetection:
    from shapely.geometry import Polygon

    row_off, col_off = _tile_offsets(tile_id)
    width = min(tile_size, source_width - col_off)
    height = min(tile_size, source_height - row_off)
    local = _points(local_vertices)
    global_points = _points(global_vertices)
    return GlobalDetection(
        detection_id=detection_id,
        scene_id=scene_id,
        tile_id=tile_id,
        tile_row_off=row_off,
        tile_col_off=col_off,
        tile_pixel_bounds=(float(col_off), float(row_off), float(col_off + width), float(row_off + height)),
        tile_valid_region=(0.0, 0.0, float(width), float(height)),
        tile_size=tile_size,
        source_width=source_width,
        source_height=source_height,
        class_id=class_id,
        class_name=class_name,
        confidence=float(confidence),
        local_obb_vertices=local,  # type: ignore[arg-type]
        global_pixel_obb_vertices=global_points,  # type: ignore[arg-type]
        global_pixel_polygon=Polygon(global_points),
        source_crs="EPSG:4326",
    )


def reconstruct_accepted_pre_nms(
    inference_dir: Path, source_width: int, source_height: int, tile_size: int
) -> tuple[GlobalDetection, ...]:
    """Reconstruct accepted pre-NMS candidates from final and suppression artifacts."""

    predictions = _read_json(inference_dir / "predictions.geojson")
    suppressions = _read_json(inference_dir / "duplicate_suppressions.json")
    by_id: dict[str, GlobalDetection] = {}
    scene_id = str(_read_json(inference_dir / "inference_summary.json")["scene_id"])
    for feature in predictions["features"]:
        properties = feature["properties"]
        detection = _global_detection(
            detection_id=str(properties["detection_id"]),
            scene_id=scene_id,
            tile_id=str(properties["source_tile_id"]),
            confidence=float(properties["confidence"]),
            local_vertices=properties["local_obb_vertices"],
            global_vertices=properties["global_pixel_obb_vertices"],
            source_width=source_width,
            source_height=source_height,
            tile_size=tile_size,
            class_id=int(properties["class_id"]),
            class_name=str(properties["class_name"]),
        )
        by_id[detection.detection_id] = detection
    for item in suppressions["suppressions"]:
        detection = _global_detection(
            detection_id=str(item["suppressed_detection_id"]),
            scene_id=scene_id,
            tile_id=str(item["suppressed_tile_id"]),
            confidence=float(item["suppressed_confidence"]),
            local_vertices=item["suppressed_local_obb_vertices"],
            global_vertices=item["suppressed_global_pixel_obb_vertices"],
            source_width=source_width,
            source_height=source_height,
            tile_size=tile_size,
            class_id=int(item["class_id"]),
            class_name=str(item["class_name"]),
        )
        if detection.detection_id in by_id:
            raise BoundaryAnalysisError(f"Duplicate reconstructed candidate: {detection.detection_id}")
        by_id[detection.detection_id] = detection
    return tuple(by_id.values())


def simulate_policy(
    accepted_pre_nms: Sequence[GlobalDetection],
    rejected: Sequence[RejectedGeometry],
    truths: Sequence[Any],
    policy: str,
) -> PolicyResult:
    """Apply one offline boundary policy with unchanged NMS and matching."""

    admitted = tuple(item.detection for item in rejected if policy_accepts(item, policy))
    admitted_ids = {item.detection_id for item in admitted}
    nms = greedy_polygon_nms((*accepted_pre_nms, *admitted), GLOBAL_NMS_IOU)
    matching = greedy_confidence_match(
        [item.global_pixel_polygon for item in nms.kept],
        [item.confidence for item in nms.kept],
        [item.obb_pixel_polygon for item in truths],
        MATCH_IOU,
    )
    tp = len(matching.matches)
    fp = len(matching.false_prediction_indices)
    fn = len(matching.false_gt_indices)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    admitted_removed = sum(
        item.suppressed_detection_id in admitted_ids for item in nms.suppressed
    )
    return PolicyResult(
        policy=policy,
        admitted_rejected=len(admitted),
        admitted_removed_by_nms=admitted_removed,
        predictions=len(nms.kept),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        kept=nms.kept,
        matches=matching.matches,
        false_gt_indices=matching.false_gt_indices,
    )


def duplicate_gt_accounting(gt_indices: Iterable[int]) -> dict[str, Any]:
    counts = Counter(int(index) for index in gt_indices)
    return {
        "unique_gt": len(counts),
        "with_one_candidate": sum(count == 1 for count in counts.values()),
        "with_multiple_candidates": sum(count > 1 for count in counts.values()),
        "candidates_per_gt_distribution": dict(sorted(Counter(counts.values()).items())),
    }


def _distribution(values: Sequence[float], names: Sequence[str]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in names}
    ordered = sorted(float(value) for value in values)

    def percentile(q: float) -> float:
        position = (len(ordered) - 1) * q
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    operations = {
        "mean": lambda: statistics.fmean(ordered),
        "median": lambda: statistics.median(ordered),
        "p10": lambda: percentile(0.10),
        "p25": lambda: percentile(0.25),
        "min": lambda: min(ordered),
        "max": lambda: max(ordered),
    }
    return {name: operations[name]() for name in names}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0]) if rows else ("empty",)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _quality(
    result: PolicyResult,
    truths: Sequence[Any],
    current_matched_gt: set[int],
    rejected_by_id: Mapping[str, RejectedGeometry],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from shapely.geometry import box

    rows: list[dict[str, Any]] = []
    for prediction_index, gt_index, overlap in result.matches:
        if gt_index in current_matched_gt:
            continue
        detection = result.kept[prediction_index]
        truth = truths[gt_index]
        center_error = float(
            detection.global_pixel_polygon.centroid.distance(
                truth.diamond_pixel_polygon.centroid
            )
        )
        tile_box = box(*detection.tile_pixel_bounds)
        visible = float(truth.obb_pixel_polygon.intersection(tile_box).area / truth.obb_pixel_polygon.area)
        rejected = rejected_by_id.get(detection.detection_id)
        rows.append(
            {
                "policy": result.policy,
                "scene_id": detection.scene_id,
                "gt_index": gt_index,
                "detection_id": detection.detection_id,
                "normalized_obb_iou": overlap,
                "confidence": detection.confidence,
                "center_error_pixels": center_error,
                "gt_visible_fraction": visible,
                "polygon_inside_tile_fraction": (
                    rejected.inside_tile_fraction if rejected is not None else 1.0
                ),
            }
        )
    fields = (
        "normalized_obb_iou",
        "confidence",
        "center_error_pixels",
        "gt_visible_fraction",
        "polygon_inside_tile_fraction",
    )
    return (
        {
            field: _distribution([float(row[field]) for row in rows], ("mean", "median"))
            for field in fields
        },
        rows,
    )


def analyze_scene(
    scene_id: str,
    scene_root: Path,
    imagery_dir: Path,
    annotation_dir: Path,
) -> dict[str, Any]:
    """Analyze one completed TRAIN scene without executing inference."""

    import rasterio
    from shapely.geometry import box

    inference_dir = scene_root / "inference"
    summary = _read_json(inference_dir / "inference_summary.json")
    raster_path = imagery_dir / f"{scene_id}.tif"
    gt_path = annotation_dir / f"{scene_id}.geojson"
    with rasterio.open(raster_path) as raster:
        source_width, source_height = raster.width, raster.height
        transform = raster.transform
        to_raster = make_crs84_to_raster_transformer(raster.crs)
    truths = _load_ground_truth(_read_json(gt_path), scene_id, transform, to_raster)
    tile_size = int(summary["tile_size"])
    accepted = reconstruct_accepted_pre_nms(
        inference_dir, source_width, source_height, tile_size
    )
    rejection_value = _read_json(inference_dir / "rejected_candidates.json")
    rejected = tuple(
        classify_rejected_geometry(item, source_width, source_height, tile_size)
        for item in rejection_value["rejections"]
    )
    results = {
        policy: simulate_policy(accepted, rejected, truths, policy) for policy in POLICIES
    }

    official_ids = {
        str(feature["properties"]["detection_id"])
        for feature in _read_json(inference_dir / "predictions.geojson")["features"]
    }
    simulated_ids = {item.detection_id for item in results["current"].kept}
    if official_ids != simulated_ids:
        raise BoundaryAnalysisError(f"Policy A does not reproduce official NMS for {scene_id}")
    current_fn = set(results["current"].false_gt_indices)
    current_matched = {gt_index for _, gt_index, _ in results["current"].matches}
    current_predictions = results["current"].kept
    rejected_by_id = {item.detection.detection_id: item for item in rejected}
    fn_matches: list[dict[str, Any]] = []
    for candidate in rejected:
        if not current_fn or not candidate.attempted_global_polygon_valid:
            continue
        overlaps = [
            (polygon_iou(candidate.detection.global_pixel_polygon, truths[index].obb_pixel_polygon), index)
            for index in sorted(current_fn)
        ]
        overlap, gt_index = max(overlaps, key=lambda item: (item[0], -item[1]))
        if overlap < MATCH_IOU:
            continue
        truth = truths[gt_index]
        tile_box = box(*candidate.detection.tile_pixel_bounds)
        visible = float(truth.obb_pixel_polygon.intersection(tile_box).area / truth.obb_pixel_polygon.area)
        nearest_accepted = max(
            (
                (polygon_iou(item.global_pixel_polygon, truth.obb_pixel_polygon), item)
                for item in current_predictions
            ),
            key=lambda item: (item[0], item[1].confidence, item[1].detection_id),
            default=(0.0, None),
        )
        centroid = truth.obb_pixel_polygon.centroid
        fn_matches.append(
            {
                "scene_id": scene_id,
                "detection_id": candidate.detection.detection_id,
                "gt_index": gt_index,
                "confidence": candidate.detection.confidence,
                "iou_with_gt": overlap,
                "tile_id": candidate.detection.tile_id,
                "row_off": candidate.detection.tile_row_off,
                "col_off": candidate.detection.tile_col_off,
                "vertices_outside_count": candidate.outside_vertex_count,
                "crossed_sides": ";".join(candidate.crossed_sides),
                "maximum_excursion_pixels": candidate.maximum_excursion_pixels,
                "polygon_inside_tile_fraction": candidate.inside_tile_fraction,
                "centroid_inside_tile": candidate.centroid_inside_tile,
                "global_polygon_inside_source": candidate.global_polygon_inside_source,
                "gt_centroid_inside_tile": tile_box.covers(centroid),
                "gt_obb_visible_fraction": visible,
                "nearest_accepted_prediction_id": (
                    nearest_accepted[1].detection_id if nearest_accepted[1] is not None else None
                ),
                "nearest_accepted_prediction_iou": nearest_accepted[0],
                "accepted_prediction_already_matches_gt": nearest_accepted[0] >= MATCH_IOU,
            }
        )
    per_gt_counts = Counter(int(row["gt_index"]) for row in fn_matches)
    for row in fn_matches:
        row["rejected_matches_for_same_gt"] = per_gt_counts[int(row["gt_index"])]
        row["another_rejected_candidate_for_same_gt"] = row["rejected_matches_for_same_gt"] > 1

    policy_rows = []
    recovered_quality: dict[str, Any] = {}
    recovered_rows: list[dict[str, Any]] = []
    current = results["current"]
    current_final_ids = {item.detection_id for item in current.kept}
    for policy, result in results.items():
        recovered_gt = current_fn - set(result.false_gt_indices)
        result_ids = {item.detection_id for item in result.kept}
        admitted_ids = {
            item.detection.detection_id
            for item in rejected
            if policy_accepts(item, policy)
        }
        policy_rows.append(
            {
                "scene_id": scene_id,
                "policy": policy,
                "admitted_rejected": result.admitted_rejected,
                "admitted_removed_by_nms": result.admitted_removed_by_nms,
                "admitted_kept_after_nms": len(result_ids & admitted_ids),
                "original_final_removed_by_new_nms": len(current_final_ids - result_ids),
                "predictions": result.predictions,
                "tp": result.tp,
                "fp": result.fp,
                "fn": result.fn,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "delta_predictions": result.predictions - current.predictions,
                "delta_tp": result.tp - current.tp,
                "delta_fp": result.fp - current.fp,
                "delta_fn": result.fn - current.fn,
                "unique_gt_recovered": len(recovered_gt),
            }
        )
        if policy != "current":
            quality, rows = _quality(result, truths, current_matched, rejected_by_id)
            recovered_quality[policy] = quality
            recovered_rows.extend(rows)

    current_quality_rows = []
    for prediction_index, gt_index, overlap in current.matches:
        detection = current.kept[prediction_index]
        truth = truths[gt_index]
        current_quality_rows.append(
            {
                "normalized_obb_iou": overlap,
                "confidence": detection.confidence,
                "center_error_pixels": float(
                    detection.global_pixel_polygon.centroid.distance(
                        truth.diamond_pixel_polygon.centroid
                    )
                ),
            }
        )
    current_quality = {
        field: _distribution([float(row[field]) for row in current_quality_rows], ("mean", "median"))
        for field in ("normalized_obb_iou", "confidence", "center_error_pixels")
    }
    return {
        "scene_id": scene_id,
        "source_width": source_width,
        "source_height": source_height,
        "tile_count": int(summary["tile_count"]),
        "tile_size": tile_size,
        "padded_tile_count": sum(
            tile.has_padding
            for tile in generate_tile_windows(
                scene_id, source_width, source_height, tile_size, float(summary["tile_overlap"])
            )
        ),
        "accepted_pre_nms": accepted,
        "rejected": rejected,
        "truths": truths,
        "fn_matches": fn_matches,
        "policy_rows": policy_rows,
        "recovered_quality": recovered_quality,
        "recovered_rows": recovered_rows,
        "current_quality": current_quality,
    }


def run_analysis(
    batch_root: Path,
    imagery_dir: Path,
    annotation_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the complete artifact-only 12-scene boundary-policy analysis."""

    manifest = list(csv.DictReader((batch_root / "selection_manifest.csv").open("r", encoding="utf-8-sig")))
    if len(manifest) != 12 or any(row.get("split") != "train" for row in manifest):
        raise BoundaryAnalysisError("Expected exactly 12 TRAIN manifest rows")
    scenes = [
        analyze_scene(row["scene_id"], batch_root / "scenes" / row["scene_id"], imagery_dir, annotation_dir)
        for row in manifest
    ]
    rejected = [item for scene in scenes for item in scene["rejected"]]
    fn_matches = [row for scene in scenes for row in scene["fn_matches"]]
    policy_rows = [row for scene in scenes for row in scene["policy_rows"]]
    recovered_rows = [row for scene in scenes for row in scene["recovered_rows"]]
    composite_counts = Counter((row["scene_id"], int(row["gt_index"])) for row in fn_matches)
    gt_accounting = {
        "unique_gt": len(composite_counts),
        "with_one_candidate": sum(value == 1 for value in composite_counts.values()),
        "with_multiple_candidates": sum(value > 1 for value in composite_counts.values()),
        "candidates_per_gt_distribution": dict(sorted(Counter(composite_counts.values()).items())),
    }
    visibility = []
    for key in composite_counts:
        visibility.append(
            max(
                float(row["gt_obb_visible_fraction"])
                for row in fn_matches
                if (row["scene_id"], int(row["gt_index"])) == key
            )
        )
    visibility_bins = {
        ">=95%": sum(value >= 0.95 for value in visibility),
        "75-95%": sum(0.75 <= value < 0.95 for value in visibility),
        "50-75%": sum(0.50 <= value < 0.75 for value in visibility),
        "<50%": sum(value < 0.50 for value in visibility),
    }

    aggregate_policies = {}
    for policy in POLICIES:
        rows = [row for row in policy_rows if row["policy"] == policy]
        predictions = sum(int(row["predictions"]) for row in rows)
        tp = sum(int(row["tp"]) for row in rows)
        fp = sum(int(row["fp"]) for row in rows)
        fn = sum(int(row["fn"]) for row in rows)
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        aggregate_policies[policy] = {
            "admitted_rejected": sum(int(row["admitted_rejected"]) for row in rows),
            "admitted_removed_by_nms": sum(int(row["admitted_removed_by_nms"]) for row in rows),
            "admitted_kept_after_nms": sum(int(row["admitted_kept_after_nms"]) for row in rows),
            "original_final_removed_by_new_nms": sum(
                int(row["original_final_removed_by_new_nms"]) for row in rows
            ),
            "predictions": predictions,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "delta_predictions": sum(int(row["delta_predictions"]) for row in rows),
            "delta_tp": sum(int(row["delta_tp"]) for row in rows),
            "delta_fp": sum(int(row["delta_fp"]) for row in rows),
            "delta_fn": sum(int(row["delta_fn"]) for row in rows),
            "unique_gt_recovered": sum(int(row["unique_gt_recovered"]) for row in rows),
        }

    rejected_rows = []
    for item in rejected:
        row = asdict(item)
        row.pop("detection")
        row.update(
            {
                "scene_id": item.detection.scene_id,
                "detection_id": item.detection.detection_id,
                "tile_id": item.detection.tile_id,
                "confidence": item.detection.confidence,
                "row_off": item.detection.tile_row_off,
                "col_off": item.detection.tile_col_off,
                "crossed_sides": ";".join(item.crossed_sides),
                "centroid_local": json.dumps(item.centroid_local),
            }
        )
        rejected_rows.append(row)

    recovered_quality = {}
    for policy in POLICIES[1:]:
        rows = [row for row in recovered_rows if row["policy"] == policy]
        recovered_quality[policy] = {
            field: _distribution([float(row[field]) for row in rows], ("mean", "median"))
            for field in (
                "normalized_obb_iou",
                "confidence",
                "center_error_pixels",
                "gt_visible_fraction",
                "polygon_inside_tile_fraction",
            )
        }
    current_quality_rows = []
    for scene in scenes:
        current = next(row for row in scene["policy_rows"] if row["policy"] == "current")
        current_quality_rows.append((current["tp"], scene["current_quality"]))
    # Pooled current quality is reconstructed from retained current matches.
    pooled_current = {}
    for field in ("normalized_obb_iou", "confidence", "center_error_pixels"):
        values = []
        for scene in scenes:
            result = next(
                simulate_policy(scene["accepted_pre_nms"], scene["rejected"], scene["truths"], policy)
                for policy in ("current",)
            )
            for prediction_index, gt_index, overlap in result.matches:
                detection = result.kept[prediction_index]
                truth = scene["truths"][gt_index]
                if field == "normalized_obb_iou":
                    values.append(overlap)
                elif field == "confidence":
                    values.append(detection.confidence)
                else:
                    values.append(
                        float(detection.global_pixel_polygon.centroid.distance(truth.diamond_pixel_polygon.centroid))
                    )
        pooled_current[field] = _distribution(values, ("mean", "median"))

    summary = {
        "scope": "artifact-only fixed-setting TRAIN boundary-policy forensics",
        "scene_count": len(scenes),
        "total_rejected": len(rejected),
        "total_tiles_with_padding": sum(int(scene["padded_tile_count"]) for scene in scenes),
        "scenes_with_padding": sum(int(scene["padded_tile_count"] > 0) for scene in scenes),
        "outside_vertex_counts": dict(sorted(Counter(item.outside_vertex_count for item in rejected).items())),
        "boundary_categories": dict(sorted(Counter(item.boundary_category for item in rejected).items())),
        "centroid_inside_count": sum(item.centroid_inside_tile for item in rejected),
        "inside_tile_fraction": _distribution(
            [item.inside_tile_fraction for item in rejected],
            ("mean", "median", "p10", "p25", "min"),
        ),
        "fn_matching_rejected_candidates": len(fn_matches),
        "unique_fn_gt_accounting": gt_accounting,
        "gt_visibility_unique_gt_best_tile": visibility_bins,
        "policies": aggregate_policies,
        "recovered_tp_quality": recovered_quality,
        "current_tp_quality": pooled_current,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    _write_csv(output_dir / "rejected_candidate_geometry.csv", rejected_rows)
    _write_csv(output_dir / "fn_matching_rejected_candidates.csv", fn_matches)
    _write_csv(output_dir / "per_scene_policy_comparison.csv", policy_rows)
    _write_csv(output_dir / "recovered_tp_quality.csv", recovered_rows)
    return summary


def _default_paths() -> dict[str, Path]:
    root = find_project_root()
    rareplanes = root / "datasets" / "RarePlanes"
    return {
        "batch_root": root / "outputs" / "batch_train_evaluation" / "fixed_baseline_12",
        "imagery": rareplanes / "train" / "imagery" / "PS-RGB_cog",
        "annotations": rareplanes / "train" / "annotations" / "geojson_aircraft",
        "output": root / "outputs" / "boundary_rejection_analysis",
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description="Offline boundary-rejection forensics")
    parser.add_argument("--batch-root", type=Path, default=defaults["batch_root"])
    parser.add_argument("--imagery-dir", type=Path, default=defaults["imagery"])
    parser.add_argument("--annotation-dir", type=Path, default=defaults["annotations"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_analysis(
        args.batch_root, args.imagery_dir, args.annotation_dir, args.output_dir
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
