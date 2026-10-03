"""Fixed-confidence, TRAIN-only diagnostic probe for three RarePlanes scenes.

This module is deliberately separate from production inference.  It preserves
all DOTA-class OBB geometry at one predeclared confidence (0.05), uses the
existing tile/coordinate/NMS implementations, and never writes official
prediction or evaluation artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

try:
    from .geospatial_evaluation import _load_ground_truth, make_crs84_to_raster_transformer
    from .geospatial_inference import (
        DetectionRejected,
        GlobalDetection,
        LoadedOBBModel,
        RawOBBDetection,
        SuppressionRecord,
        build_rejected_detection,
        extract_ultralytics_obb,
        greedy_polygon_nms,
        load_obb_model,
        polygon_iou,
        raw_detection_to_global,
    )
    from .geospatial_tiling import (
        generate_tile_windows,
        local_obb_to_global_pixels,
        read_tile,
        stride_from_overlap,
        validate_tile_plan,
    )
    from .project_config import find_project_root
except ImportError:  # Support direct execution from src/.
    from geospatial_evaluation import _load_ground_truth, make_crs84_to_raster_transformer
    from geospatial_inference import (
        DetectionRejected,
        GlobalDetection,
        LoadedOBBModel,
        RawOBBDetection,
        SuppressionRecord,
        build_rejected_detection,
        extract_ultralytics_obb,
        greedy_polygon_nms,
        load_obb_model,
        polygon_iou,
        raw_detection_to_global,
    )
    from geospatial_tiling import (
        generate_tile_windows,
        local_obb_to_global_pixels,
        read_tile,
        stride_from_overlap,
        validate_tile_plan,
    )
    from project_config import find_project_root


PROBE_CONFIDENCE = 0.05
IMAGE_SIZE = 1024
TILE_SIZE = 1024
TILE_OVERLAP = 0.20
MODEL_IOU = 0.45
GLOBAL_NMS_IOU = 0.50
DEVICE = "cpu"
PLANE_CLASS_ID = 0
PLANE_CLASS_NAME = "plane"
GT_MATCH_IOU = 0.50
MEANINGFUL_IOU = 0.25
RESPONSE_COMPARABILITY_IOU = 0.05
ALLOWED_SCENE_IDS = (
    "96_10400100096C2500",
    "103_104001004890F500",
    "1_104005000FDC8D00",
)
SCENE_SUBSETS = {
    "96_10400100096C2500": "development",
    "103_104001004890F500": "holdout",
    "1_104005000FDC8D00": "holdout",
}


class ProbeError(RuntimeError):
    """Raised when the fixed diagnostic protocol is violated."""


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def enforce_fixed_protocol(
    scene_ids: Sequence[str], confidence: float, split: str = "train"
) -> None:
    """Reject any scene, split, or confidence outside the predeclared protocol."""

    if tuple(scene_ids) != ALLOWED_SCENE_IDS:
        raise ProbeError(f"Probe requires exactly these ordered scenes: {ALLOWED_SCENE_IDS}")
    if split.lower() != "train":
        raise ProbeError("Low-confidence probe is TRAIN-only")
    if not math.isclose(float(confidence), PROBE_CONFIDENCE, abs_tol=1e-12):
        raise ProbeError("Diagnostic confidence is fixed at 0.05")


def validate_single_scene(scene_id: str, confidence: float, split: str = "train") -> None:
    if scene_id not in ALLOWED_SCENE_IDS:
        raise ProbeError(f"Scene is not predeclared: {scene_id}")
    if split.lower() != "train":
        raise ProbeError("Low-confidence probe is TRAIN-only")
    if not math.isclose(float(confidence), PROBE_CONFIDENCE, abs_tol=1e-12):
        raise ProbeError("Diagnostic confidence is fixed at 0.05")


def confidence_band(confidence: float) -> str:
    value = float(confidence)
    if value < PROBE_CONFIDENCE:
        raise ValueError("Probe detections cannot be below confidence 0.05")
    if value < 0.10:
        return "0.05-0.10"
    if value < 0.15:
        return "0.10-0.15"
    if value < 0.20:
        return "0.15-0.20"
    if value < 0.25:
        return "0.20-0.25"
    return ">=0.25"


def _vertices(value: Sequence[Sequence[float]] | None) -> list[list[float]] | None:
    if value is None:
        return None
    return [[float(point[0]), float(point[1])] for point in value]


def _closed_polygon(value: Sequence[Sequence[float]] | None) -> list[list[float]] | None:
    vertices = _vertices(value)
    if not vertices:
        return vertices
    return [*vertices, vertices[0]]


def raw_detection_record(
    detection: RawOBBDetection,
    tile: Any,
    global_detection: GlobalDetection | None,
    rejection_reason: str | None,
) -> dict[str, Any]:
    attempted = None
    attempted_valid = False
    try:
        attempted = local_obb_to_global_pixels(
            detection.local_obb_vertices, tile, validate=False
        )
        from shapely.geometry import Polygon

        polygon = Polygon(attempted)
        attempted_valid = bool(
            not polygon.is_empty and polygon.is_valid and polygon.area > 0.0
        )
    except (IndexError, TypeError, ValueError):
        attempted = None
    global_vertices = (
        global_detection.global_pixel_obb_vertices if global_detection is not None else attempted
    )
    return {
        "detection_id": detection.detection_id,
        "scene_id": tile.scene_id,
        "tile_id": tile.tile_id,
        "tile_row_off": tile.row_off,
        "tile_col_off": tile.col_off,
        "tile_pixel_bounds": list(tile.source_bounds.as_tuple()),
        "tile_valid_region": list(tile.valid_region.as_tuple()),
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "confidence": detection.confidence,
        "confidence_band": confidence_band(detection.confidence),
        "tile_local_obb_vertices": _vertices(detection.local_obb_vertices),
        "global_pixel_obb_vertices": _vertices(global_vertices),
        "global_pixel_polygon": _closed_polygon(global_vertices),
        "accepted_by_production_boundary_policy": global_detection is not None,
        "rejection_reason": rejection_reason,
        "attempted_global_polygon_valid": attempted_valid,
    }


def global_detection_record(detection: GlobalDetection) -> dict[str, Any]:
    return {
        "detection_id": detection.detection_id,
        "scene_id": detection.scene_id,
        "tile_id": detection.tile_id,
        "tile_row_off": detection.tile_row_off,
        "tile_col_off": detection.tile_col_off,
        "tile_pixel_bounds": list(detection.tile_pixel_bounds),
        "tile_valid_region": list(detection.tile_valid_region),
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "confidence": detection.confidence,
        "confidence_band": confidence_band(detection.confidence),
        "tile_local_obb_vertices": _vertices(detection.local_obb_vertices),
        "global_pixel_obb_vertices": _vertices(detection.global_pixel_obb_vertices),
        "global_pixel_polygon": _closed_polygon(detection.global_pixel_obb_vertices),
    }


def best_detection_response(gt_polygon: Any, detections: Sequence[GlobalDetection]) -> dict[str, Any]:
    """Select by greatest true polygon IoU, then confidence, then stable ID."""

    best: GlobalDetection | None = None
    best_iou = 0.0
    gt_bounds = gt_polygon.bounds
    for detection in detections:
        bounds = detection.global_pixel_polygon.bounds
        if (
            bounds[2] <= gt_bounds[0]
            or gt_bounds[2] <= bounds[0]
            or bounds[3] <= gt_bounds[1]
            or gt_bounds[3] <= bounds[1]
        ):
            continue
        overlap = polygon_iou(gt_polygon, detection.global_pixel_polygon)
        if best is None or (overlap, detection.confidence, detection.detection_id) > (
            best_iou,
            best.confidence,
            best.detection_id,
        ):
            best = detection
            best_iou = overlap
    return {
        "detection_id": best.detection_id if best else None,
        "class_id": best.class_id if best else None,
        "class_name": best.class_name if best else None,
        "confidence": best.confidence if best else None,
        "iou": best_iou,
    }


def overlap_category(iou: float, prefix: str) -> str:
    if iou >= GT_MATCH_IOU:
        return f"A_{prefix}_iou_ge_0.50"
    if iou >= MEANINGFUL_IOU:
        return f"B_{prefix}_iou_0.25_to_0.50"
    if iou > 0.0:
        return f"C_{prefix}_overlap_lt_0.25"
    return f"D_no_{prefix}_overlap"


def response_classification(
    plane_iou: float,
    plane_confidence: float | None,
    nonplane_iou: float,
    nonplane_confidence: float | None,
) -> str:
    """Classify useful evidence with a fixed, descriptive comparison rule.

    IoU >= 0.25 is meaningful.  If both classes are meaningful and their IoUs
    differ by at most 0.05, the response is AMBIGUOUS; otherwise the larger-IoU
    response dominates.  Confidence is retained for reporting, not used to
    override geometry.  The confidence arguments make that choice explicit.
    """

    del plane_confidence, nonplane_confidence
    plane_useful = float(plane_iou) >= MEANINGFUL_IOU
    nonplane_useful = float(nonplane_iou) >= MEANINGFUL_IOU
    if not plane_useful and not nonplane_useful:
        return "NO_MEANINGFUL_RESPONSE"
    if plane_useful and not nonplane_useful:
        return "PLANE_DOMINANT"
    if nonplane_useful and not plane_useful:
        return "NON_PLANE_DOMINANT"
    if abs(float(plane_iou) - float(nonplane_iou)) <= RESPONSE_COMPARABILITY_IOU:
        return "AMBIGUOUS"
    return "PLANE_DOMINANT" if plane_iou > nonplane_iou else "NON_PLANE_DOMINANT"


def _distribution(values: Iterable[float]) -> dict[str, float | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(finite),
        "min": min(finite),
        "median": statistics.median(finite),
        "mean": statistics.fmean(finite),
        "max": max(finite),
    }


def analyze_gt_responses(
    scene_id: str, truths: Sequence[Any], detections: Sequence[GlobalDetection]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    planes = tuple(item for item in detections if item.class_id == PLANE_CLASS_ID)
    nonplanes = tuple(item for item in detections if item.class_id != PLANE_CLASS_ID)
    rows: list[dict[str, Any]] = []
    for truth in truths:
        plane = best_detection_response(truth.obb_pixel_polygon, planes)
        nonplane = best_detection_response(truth.obb_pixel_polygon, nonplanes)
        plane_iou = float(plane["iou"])
        nonplane_iou = float(nonplane["iou"])
        below = bool(
            plane_iou >= GT_MATCH_IOU
            and plane["confidence"] is not None
            and float(plane["confidence"]) < 0.25
        )
        rows.append(
            {
                "scene_id": scene_id,
                "dataset_split": "train",
                "dataset_subset": SCENE_SUBSETS[scene_id],
                "gt_index": truth.gt_index,
                "best_plane_detection_id": plane["detection_id"],
                "best_plane_confidence": plane["confidence"],
                "best_plane_iou": plane_iou,
                "plane_overlap_category": overlap_category(plane_iou, "plane"),
                "below_0.25_geometrically_valid_plane": below,
                "best_nonplane_detection_id": nonplane["detection_id"],
                "best_nonplane_class_id": nonplane["class_id"],
                "best_nonplane_class": nonplane["class_name"],
                "best_nonplane_confidence": nonplane["confidence"],
                "best_nonplane_iou": nonplane_iou,
                "nonplane_overlap_category": overlap_category(nonplane_iou, "nonplane"),
                "response_classification": response_classification(
                    plane_iou, plane["confidence"], nonplane_iou, nonplane["confidence"]
                ),
                "no_overlapping_detection_any_class": plane_iou == 0.0 and nonplane_iou == 0.0,
            }
        )
    below_rows = [row for row in rows if row["below_0.25_geometrically_valid_plane"]]
    summary = {
        "gt_count": len(rows),
        "plane_overlap_categories": dict(
            sorted(Counter(row["plane_overlap_category"] for row in rows).items())
        ),
        "nonplane_overlap_categories_for_gt_without_plane_match": dict(
            sorted(
                Counter(
                    row["nonplane_overlap_category"]
                    for row in rows
                    if float(row["best_plane_iou"]) < GT_MATCH_IOU
                ).items()
            )
        ),
        "response_classification": dict(
            sorted(Counter(row["response_classification"] for row in rows).items())
        ),
        "below_0.25_geometrically_valid_plane_count": len(below_rows),
        "below_0.25_geometrically_valid_plane_fraction": len(below_rows) / len(rows) if rows else 0.0,
        "below_0.25_plane_confidence_distribution": _distribution(
            float(row["best_plane_confidence"]) for row in below_rows
        ),
        "nonplane_iou_ge_0.50_classes_for_gt_without_plane_match": dict(
            sorted(
                Counter(
                    str(row["best_nonplane_class"])
                    for row in rows
                    if float(row["best_plane_iou"]) < GT_MATCH_IOU
                    and float(row["best_nonplane_iou"]) >= GT_MATCH_IOU
                ).items()
            )
        ),
        "no_overlapping_detection_any_class": sum(
            bool(row["no_overlapping_detection_any_class"]) for row in rows
        ),
    }
    return rows, summary


def _suppression_record(record: SuppressionRecord) -> dict[str, Any]:
    value = asdict(record)
    for key in (
        "suppressed_local_obb_vertices",
        "suppressed_global_pixel_obb_vertices",
        "kept_global_pixel_obb_vertices",
    ):
        value[key] = _vertices(value[key])
    return value


def run_scene_probe(
    scene_id: str,
    model_path: Path,
    raster_path: Path,
    annotation_path: Path,
    output_dir: Path,
    production_summary_path: Path,
) -> dict[str, Any]:
    """Run one scene in the current (fresh) process at the fixed protocol."""

    validate_single_scene(scene_id, PROBE_CONFIDENCE)
    if raster_path.stem != scene_id or annotation_path.stem != scene_id:
        raise ProbeError("Raster/annotation scene pairing mismatch")
    start = time.perf_counter()
    model_start = time.perf_counter()
    loaded: LoadedOBBModel = load_obb_model(model_path, DEVICE)
    model_load_seconds = time.perf_counter() - model_start
    if loaded.names.get(PLANE_CLASS_ID) != PLANE_CLASS_NAME:
        raise ProbeError("Expected DOTA class 0 to be plane")

    import numpy as np
    import rasterio

    raw_records: list[dict[str, Any]] = []
    accepted: list[GlobalDetection] = []
    rejected_records: list[dict[str, Any]] = []
    raw_class_counts: Counter[str] = Counter()

    with rasterio.open(raster_path) as dataset:
        if dataset.count != 3 or any(str(dtype) != "uint8" for dtype in dataset.dtypes):
            raise ProbeError("Expected three-band uint8 PS-RGB imagery")
        if dataset.crs is None:
            raise ProbeError("Raster CRS is missing")
        windows = tuple(
            generate_tile_windows(
                scene_id, dataset.width, dataset.height, TILE_SIZE, TILE_OVERLAP
            )
        )
        plan = validate_tile_plan(windows, dataset.width, dataset.height)
        if not plan.valid:
            raise ProbeError(f"Invalid deterministic tile plan: {plan}")
        for number, tile in enumerate(windows, 1):
            tile_read = read_tile(dataset, tile, pad_to_tile_size=True)
            hwc_rgb = np.moveaxis(tile_read.data, 0, -1)
            model_input = np.ascontiguousarray(hwc_rgb[..., ::-1])
            predictions = loaded.model.predict(
                source=model_input,
                imgsz=IMAGE_SIZE,
                conf=PROBE_CONFIDENCE,
                iou=MODEL_IOU,
                device=DEVICE,
                save=False,
                verbose=False,
            )
            if len(predictions) != 1:
                raise ProbeError(f"Expected one result for {tile.tile_id}")
            raw = extract_ultralytics_obb(predictions[0], loaded.names, tile)
            raw_class_counts.update(item.class_name for item in raw)
            for detection in raw:
                global_detection = None
                rejection_reason = None
                try:
                    global_detection = raw_detection_to_global(detection, tile, dataset.crs)
                except DetectionRejected as exc:
                    rejection_reason = exc.reason
                    rejected_records.append(asdict(build_rejected_detection(detection, tile, exc.reason)))
                else:
                    accepted.append(global_detection)
                raw_records.append(
                    raw_detection_record(
                        detection, tile, global_detection, rejection_reason
                    )
                )
            if number == 1 or number % 25 == 0 or number == len(windows):
                print(
                    f"PROBE {scene_id}: tile {number}/{len(windows)} raw_total={len(raw_records)}",
                    flush=True,
                )
            del predictions, raw, model_input, hwc_rgb, tile_read

        nms = greedy_polygon_nms(accepted, GLOBAL_NMS_IOU)
        deduplicated = nms.kept
        truths = _load_ground_truth(
            _read_json(annotation_path),
            scene_id,
            dataset.transform,
            make_crs84_to_raster_transformer(dataset.crs),
        )

    gt_rows, gt_summary = analyze_gt_responses(scene_id, truths, deduplicated)
    raw_planes = [row for row in raw_records if int(row["class_id"]) == PLANE_CLASS_ID]
    dedup_planes = [item for item in deduplicated if item.class_id == PLANE_CLASS_ID]
    plane_bands = dict.fromkeys(
        ("0.05-0.10", "0.10-0.15", "0.15-0.20", "0.20-0.25", ">=0.25"), 0
    )
    plane_bands.update(Counter(confidence_band(item.confidence) for item in dedup_planes))

    production = _read_json(production_summary_path)
    diagnostic_ge_025_raw = [row for row in raw_records if float(row["confidence"]) >= 0.25]
    diagnostic_ge_025_planes = [row for row in diagnostic_ge_025_raw if int(row["class_id"]) == 0]
    diagnostic_ge_025_dedup_planes = [
        item for item in dedup_planes if item.confidence >= 0.25
    ]
    consistency = {
        "diagnostic_raw_all_class_ge_0.25": len(diagnostic_ge_025_raw),
        "production_raw_all_class": int(production["all_class_raw_detections"]),
        "raw_all_class_difference": len(diagnostic_ge_025_raw) - int(production["all_class_raw_detections"]),
        "diagnostic_raw_plane_ge_0.25": len(diagnostic_ge_025_planes),
        "production_raw_plane": int(production["raw_target_candidates"]),
        "raw_plane_difference": len(diagnostic_ge_025_planes) - int(production["raw_target_candidates"]),
        "diagnostic_deduplicated_plane_ge_0.25": len(diagnostic_ge_025_dedup_planes),
        "production_final_plane": int(production["final_target_count"]),
        "deduplicated_plane_difference": len(diagnostic_ge_025_dedup_planes) - int(production["final_target_count"]),
    }
    summary = {
        "scene_id": scene_id,
        "dataset_split": "train",
        "dataset_subset": SCENE_SUBSETS[scene_id],
        "probe_confidence": PROBE_CONFIDENCE,
        "device": DEVICE,
        "image_size": IMAGE_SIZE,
        "tile_size": TILE_SIZE,
        "tile_overlap": TILE_OVERLAP,
        "integer_stride": stride_from_overlap(TILE_SIZE, TILE_OVERLAP),
        "model_iou": MODEL_IOU,
        "global_nms_iou": GLOBAL_NMS_IOU,
        "tile_count": len(windows),
        "gt_count": len(truths),
        "raw_all_class_detections": len(raw_records),
        "accepted_all_class_before_global_nms": len(accepted),
        "rejected_by_boundary_policy": len(rejected_records),
        "deduplicated_all_class_detections": len(deduplicated),
        "raw_class_counts": dict(sorted(raw_class_counts.items())),
        "deduplicated_class_counts": dict(
            sorted(Counter(item.class_name for item in deduplicated).items())
        ),
        "raw_plane_detections": len(raw_planes),
        "deduplicated_plane_detections": len(dedup_planes),
        "plane_confidence_bands_deduplicated": plane_bands,
        "plane_confidence_distribution_deduplicated": _distribution(
            item.confidence for item in dedup_planes
        ),
        "global_suppressions": len(nms.suppressed),
        "gt_response": gt_summary,
        "production_ge_0.25_consistency": consistency,
        "model_load_seconds": model_load_seconds,
        "runtime_seconds": time.perf_counter() - start,
    }

    scene_dir = output_dir / "scenes" / scene_id
    _write_json(scene_dir / "raw_detections.json", {"detections": raw_records})
    _write_json(
        scene_dir / "deduplicated_detections.json",
        {"detections": [global_detection_record(item) for item in deduplicated]},
    )
    _write_json(
        scene_dir / "duplicate_suppressions.json",
        {"suppressions": [_suppression_record(item) for item in nms.suppressed]},
    )
    _write_json(scene_dir / "rejected_detections.json", {"detections": rejected_records})
    _write_csv(scene_dir / "gt_response_analysis.csv", gt_rows, tuple(gt_rows[0]))
    _write_json(scene_dir / "scene_summary.json", summary)
    _write_json(scene_dir / "complete.json", {"scene_id": scene_id, "complete": True})
    return summary


def _production_summary(root: Path, scene_id: str) -> Path:
    if SCENE_SUBSETS[scene_id] == "development":
        base = root / "outputs" / "batch_train_evaluation" / "fixed_baseline_12"
    else:
        base = root / "outputs" / "boundary_policy_validation" / "train_holdout_12"
    return base / "scenes" / scene_id / "inference" / "inference_summary.json"


def _scene_command(root: Path, output_dir: Path, scene_id: str) -> list[str]:
    train = root / "datasets" / "RarePlanes" / "train"
    return [
        sys.executable,
        "-m",
        "src.low_confidence_probe",
        "run-scene",
        "--scene-id",
        scene_id,
        "--model",
        str(root / "best.pt"),
        "--raster",
        str(train / "imagery" / "PS-RGB_cog" / f"{scene_id}.tif"),
        "--annotation",
        str(train / "annotations" / "geojson_aircraft" / f"{scene_id}.geojson"),
        "--production-summary",
        str(_production_summary(root, scene_id)),
        "--output-dir",
        str(output_dir),
    ]


def aggregate_probe(output_dir: Path, process_log: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = [
        _read_json(output_dir / "scenes" / scene_id / "scene_summary.json")
        for scene_id in ALLOWED_SCENE_IDS
    ]
    gt_rows: list[dict[str, str]] = []
    for scene_id in ALLOWED_SCENE_IDS:
        path = output_dir / "scenes" / scene_id / "gt_response_analysis.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            gt_rows.extend(csv.DictReader(handle))

    below = [row for row in gt_rows if row["below_0.25_geometrically_valid_plane"] == "True"]
    nonplane_rows = [
        row
        for row in gt_rows
        if float(row["best_plane_iou"]) < GT_MATCH_IOU
        and float(row["best_nonplane_iou"]) > 0.0
    ]
    aggregate = {
        "protocol": {
            "confidence": PROBE_CONFIDENCE,
            "scenes": list(ALLOWED_SCENE_IDS),
            "split": "train",
            "device": DEVICE,
            "image_size": IMAGE_SIZE,
            "tile_size": TILE_SIZE,
            "tile_overlap": TILE_OVERLAP,
            "integer_stride": stride_from_overlap(TILE_SIZE, TILE_OVERLAP),
            "model_iou": MODEL_IOU,
            "global_nms_iou": GLOBAL_NMS_IOU,
            "official_metrics_modified": False,
        },
        "scenes": summaries,
        "total_gt": sum(int(item["gt_count"]) for item in summaries),
        "below_0.25_geometrically_valid_plane_count": len(below),
        "below_0.25_geometrically_valid_plane_fraction": len(below) / len(gt_rows),
        "below_0.25_confidence_distribution": _distribution(
            float(row["best_plane_confidence"]) for row in below
        ),
        "below_0.25_scene_distribution": dict(
            sorted(Counter(row["scene_id"] for row in below).items())
        ),
        "nonplane_overlap_class_counts_without_plane_match": dict(
            sorted(Counter(row["best_nonplane_class"] for row in nonplane_rows).items())
        ),
        "response_classification": dict(
            sorted(Counter(row["response_classification"] for row in gt_rows).items())
        ),
        "native_process_failures": sum(int(item["exit_code"]) != 0 for item in process_log),
    }
    scene_rows = []
    band_rows = []
    confusion_rows = []
    for summary in summaries:
        scene_rows.append(
            {
                "scene_id": summary["scene_id"],
                "dataset_subset": summary["dataset_subset"],
                "gt_count": summary["gt_count"],
                "raw_all_class_detections": summary["raw_all_class_detections"],
                "deduplicated_all_class_detections": summary["deduplicated_all_class_detections"],
                "raw_plane_detections": summary["raw_plane_detections"],
                "deduplicated_plane_detections": summary["deduplicated_plane_detections"],
                "runtime_seconds": summary["runtime_seconds"],
            }
        )
        for band, count in summary["plane_confidence_bands_deduplicated"].items():
            band_rows.append({"scene_id": summary["scene_id"], "confidence_band": band, "count": count})
    for row in gt_rows:
        if float(row["best_plane_iou"]) < GT_MATCH_IOU:
            confusion_rows.append(
                {
                    key: row[key]
                    for key in (
                        "scene_id",
                        "dataset_subset",
                        "gt_index",
                        "best_nonplane_detection_id",
                        "best_nonplane_class",
                        "best_nonplane_confidence",
                        "best_nonplane_iou",
                        "nonplane_overlap_category",
                    )
                }
            )
    _write_csv(output_dir / "scene_summary.csv", scene_rows, tuple(scene_rows[0]))
    _write_csv(output_dir / "gt_response_analysis.csv", gt_rows, tuple(gt_rows[0]))
    _write_csv(
        output_dir / "plane_confidence_distribution.csv", band_rows, tuple(band_rows[0])
    )
    _write_csv(output_dir / "nonplane_confusion.csv", confusion_rows, tuple(confusion_rows[0]))
    _write_json(output_dir / "aggregate_summary.json", aggregate)
    return aggregate


def run_probe_isolated(root: Path, output_dir: Path) -> dict[str, Any]:
    enforce_fixed_protocol(ALLOWED_SCENE_IDS, PROBE_CONFIDENCE, "train")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ProbeError(f"Refusing to overwrite an existing probe: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, Any]] = []
    for scene_id in ALLOWED_SCENE_IDS:
        command = _scene_command(root, output_dir, scene_id)
        started = time.time()
        monotonic = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        entry = {
            "scene_id": scene_id,
            "process_start_unix": started,
            "process_completed_unix": time.time(),
            "runtime_seconds": time.perf_counter() - monotonic,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "native_failure_or_warning": completed.returncode < 0 or completed.returncode > 255,
        }
        log.append(entry)
        _write_json(output_dir / "process_execution_log.json", log)
        print(
            f"PROCESS {scene_id}: exit={completed.returncode} runtime={entry['runtime_seconds']:.1f}s",
            flush=True,
        )
        if completed.returncode != 0:
            raise ProbeError(f"Scene process failed without retry: {scene_id}")
    return aggregate_probe(output_dir, log)


def _defaults() -> tuple[Path, Path]:
    root = find_project_root()
    return root, root / "outputs" / "low_confidence_probe" / "conf_005"


def build_parser() -> argparse.ArgumentParser:
    root, output = _defaults()
    parser = argparse.ArgumentParser(description="Fixed conf=0.05 RarePlanes TRAIN probe")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_probe = subparsers.add_parser("run-probe")
    run_probe.add_argument("--root", type=Path, default=root)
    run_probe.add_argument("--output-dir", type=Path, default=output)
    run_scene = subparsers.add_parser("run-scene")
    run_scene.add_argument("--scene-id", required=True)
    run_scene.add_argument("--model", type=Path, required=True)
    run_scene.add_argument("--raster", type=Path, required=True)
    run_scene.add_argument("--annotation", type=Path, required=True)
    run_scene.add_argument("--production-summary", type=Path, required=True)
    run_scene.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run-probe":
        summary = run_probe_isolated(args.root.resolve(), args.output_dir.resolve())
    else:
        summary = run_scene_probe(
            args.scene_id,
            args.model.resolve(),
            args.raster.resolve(),
            args.annotation.resolve(),
            args.output_dir.resolve(),
            args.production_summary.resolve(),
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
