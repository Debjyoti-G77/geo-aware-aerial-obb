"""Deterministic TRAIN sampling and fixed-setting batch measurement.

This module orchestrates the existing geospatial inference and evaluation
layers. It intentionally exposes no threshold-sweep interface: all inference
and matching settings are the frozen engineering baseline defined below.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Iterable, Mapping, Sequence

try:
    from .geospatial_evaluation import (
        EvaluationResult,
        evaluate_scene,
        write_evaluation_outputs,
    )
    from .geospatial_inference import (
        SceneInferenceSettings,
        load_obb_model,
        process_raster_scene,
    )
    from .project_config import find_project_root
except ImportError:  # Support direct execution.
    from geospatial_evaluation import (
        EvaluationResult,
        evaluate_scene,
        write_evaluation_outputs,
    )
    from geospatial_inference import (
        SceneInferenceSettings,
        load_obb_model,
        process_raster_scene,
    )
    from project_config import find_project_root


SAMPLE_SIZE = 12
REQUIRED_SCENE_ID = "2_104001003B4B5900"
CONFIDENCE = 0.25
MODEL_IOU = 0.45
GLOBAL_NMS_IOU = 0.50
MATCH_IOU = 0.50
TILE_SIZE = 1024
TILE_OVERLAP = 0.20
IMAGE_SIZE = 1024
DEVICE = "cpu"
TARGET_CLASS = "plane"


class BatchEvaluationError(RuntimeError):
    """Raised when deterministic selection or batch inputs are invalid."""


@dataclass(frozen=True)
class SceneRecord:
    scene_id: str
    split: str
    loc_id: int
    image_catalog_id: str
    width: int
    height: int
    gt_aircraft_count: int
    country: str
    state_province: str
    air_field: str
    weather: str
    cloud_cover_percentage: float
    off_nadir_angle: float
    sensor: str
    collection_time_start: str
    location_observation_count: int
    selection_rationale: str = ""

    @property
    def pixel_count(self) -> int:
        return self.width * self.height


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """Return a deterministic linearly interpolated percentile."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _float(value: Any, field: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise BatchEvaluationError(f"Invalid numeric metadata {field}={value!r}") from exc
    if not math.isfinite(converted):
        raise BatchEvaluationError(f"Non-finite metadata {field}={value!r}")
    return converted


def load_train_scene_records(
    metadata_csv: Path,
    imagery_dir: Path,
    annotation_dir: Path,
) -> tuple[SceneRecord, ...]:
    """Load paired TRAIN-only metadata without reading raster pixel arrays."""

    import rasterio

    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    train_rows = {row["image_id"]: row for row in all_rows if row.get("Train") == "1"}
    location_counts = Counter(row["loc_id"] for row in train_rows.values())
    image_paths = sorted(imagery_dir.glob("*.tif"), key=lambda path: path.stem)
    records: list[SceneRecord] = []
    for image_path in image_paths:
        scene_id = image_path.stem
        row = train_rows.get(scene_id)
        if row is None:
            raise BatchEvaluationError(f"TRAIN metadata missing for {scene_id}")
        annotation_path = annotation_dir / f"{scene_id}.geojson"
        if not annotation_path.is_file():
            raise BatchEvaluationError(f"TRAIN annotation missing for {scene_id}")
        with annotation_path.open("r", encoding="utf-8") as handle:
            annotation = json.load(handle)
        with rasterio.open(image_path) as raster:
            width, height = raster.width, raster.height
        records.append(
            SceneRecord(
                scene_id=scene_id,
                split="train",
                loc_id=int(row["loc_id"]),
                image_catalog_id=scene_id.split("_", 1)[1],
                width=width,
                height=height,
                gt_aircraft_count=len(annotation.get("features", [])),
                country=row.get("Country", ""),
                state_province=row.get("State_Province", ""),
                air_field=row.get("Air_Field", ""),
                weather=row.get("Weather", ""),
                cloud_cover_percentage=_float(
                    row.get("cloud_cover_percentage"), "cloud_cover_percentage"
                ),
                off_nadir_angle=_float(row.get("off_nadir_max"), "off_nadir_max"),
                sensor=row.get("sensor", ""),
                collection_time_start=row.get("collection_time_start", ""),
                location_observation_count=location_counts[row["loc_id"]],
            )
        )
    if set(train_rows) != {record.scene_id for record in records}:
        raise BatchEvaluationError("TRAIN metadata and imagery scene sets differ")
    return tuple(records)


def _representative(records: Sequence[SceneRecord]) -> SceneRecord:
    """Choose a deterministic multivariate medoid from one condition group."""

    if not records:
        raise ValueError("Cannot choose a representative from an empty group")
    feature_values = {
        "log_pixels": [math.log(record.pixel_count) for record in records],
        "gt": [float(record.gt_aircraft_count) for record in records],
        "off": [record.off_nadir_angle for record in records],
        "cloud": [record.cloud_cover_percentage for record in records],
    }
    medians = {key: statistics.median(values) for key, values in feature_values.items()}
    ranges = {
        key: max(values) - min(values) or 1.0 for key, values in feature_values.items()
    }

    def distance(record: SceneRecord) -> tuple[float, str]:
        values = (
            (math.log(record.pixel_count), "log_pixels"),
            (float(record.gt_aircraft_count), "gt"),
            (record.off_nadir_angle, "off"),
            (record.cloud_cover_percentage, "cloud"),
        )
        score = sum(((value - medians[key]) / ranges[key]) ** 2 for value, key in values)
        return score, record.scene_id

    return min(records, key=distance)


def _closest_to_quantile(
    records: Sequence[SceneRecord], getter: Any, quantile: float
) -> SceneRecord:
    target = percentile([float(getter(record)) for record in records], quantile)
    assert target is not None
    return min(records, key=lambda record: (abs(float(getter(record)) - target), record.scene_id))


def deterministic_select_train_scenes(
    records: Sequence[SceneRecord],
    count: int = SAMPLE_SIZE,
    required_scene_id: str = REQUIRED_SCENE_ID,
) -> tuple[SceneRecord, ...]:
    """Select a frozen metadata/geometry-diverse sample before inference."""

    train = sorted((record for record in records if record.split == "train"), key=lambda item: item.scene_id)
    if count <= 0 or count > len(train):
        raise ValueError("Invalid sample size")
    by_id = {record.scene_id: record for record in train}
    if required_scene_id not in by_id:
        raise BatchEvaluationError(f"Required TRAIN scene missing: {required_scene_id}")

    selected: list[SceneRecord] = []
    reasons: dict[str, list[str]] = defaultdict(list)

    def add(record: SceneRecord, reason: str) -> None:
        reasons[record.scene_id].append(reason)
        if record.scene_id not in {item.scene_id for item in selected} and len(selected) < count:
            selected.append(record)

    add(by_id[required_scene_id], "required previously validated baseline scene")
    add(min(train, key=lambda item: (item.gt_aircraft_count, item.scene_id)), "minimum TRAIN GT-count anchor")
    add(max(train, key=lambda item: (item.gt_aircraft_count, tuple(-ord(c) for c in item.scene_id))), "maximum TRAIN GT-count anchor")
    add(min(train, key=lambda item: (item.off_nadir_angle, item.scene_id)), "minimum off-nadir anchor")
    add(max(train, key=lambda item: (item.off_nadir_angle, tuple(-ord(c) for c in item.scene_id))), "maximum off-nadir anchor")
    add(max(train, key=lambda item: (item.cloud_cover_percentage, tuple(-ord(c) for c in item.scene_id))), "maximum catalog cloud-cover anchor")
    for weather in ("Snow", "Cloud Cover or Haze"):
        group = [record for record in train if record.weather == weather]
        if group:
            add(_representative(group), f"deterministic {weather} condition medoid")
    maximum_observations = max(record.location_observation_count for record in train)
    repeated = [record for record in train if record.location_observation_count == maximum_observations]
    add(_representative(repeated), "maximum repeat-observation-location medoid")
    add(_closest_to_quantile(train, lambda item: math.log(item.pixel_count), 0.75), "75th-percentile image-area anchor")
    add(_closest_to_quantile(train, lambda item: item.gt_aircraft_count, 0.75), "75th-percentile GT-count anchor")

    numeric = {
        "log_pixels": [math.log(record.pixel_count) for record in train],
        "gt": [float(record.gt_aircraft_count) for record in train],
        "off": [record.off_nadir_angle for record in train],
        "cloud": [record.cloud_cover_percentage for record in train],
        "aspect": [record.width / record.height for record in train],
        "observations": [float(record.location_observation_count) for record in train],
    }
    ranges = {key: max(values) - min(values) or 1.0 for key, values in numeric.items()}
    minima = {key: min(values) for key, values in numeric.items()}

    def vector(record: SceneRecord) -> tuple[float, ...]:
        raw = {
            "log_pixels": math.log(record.pixel_count),
            "gt": float(record.gt_aircraft_count),
            "off": record.off_nadir_angle,
            "cloud": record.cloud_cover_percentage,
            "aspect": record.width / record.height,
            "observations": float(record.location_observation_count),
        }
        return tuple((raw[key] - minima[key]) / ranges[key] for key in numeric)

    vectors = {record.scene_id: vector(record) for record in train}
    while len(selected) < count:
        selected_ids = {record.scene_id for record in selected}
        countries = {record.country for record in selected}
        weathers = {record.weather for record in selected}
        locations = {record.loc_id for record in selected}

        def diversity_score(record: SceneRecord) -> tuple[float, str]:
            candidate = vectors[record.scene_id]
            minimum_distance = min(
                math.dist(candidate, vectors[item.scene_id]) for item in selected
            )
            score = (
                minimum_distance
                + (0.35 if record.country not in countries else 0.0)
                + (0.20 if record.weather not in weathers else 0.0)
                + (0.15 if record.loc_id not in locations else 0.0)
            )
            return score, "".join(chr(255 - ord(c)) for c in record.scene_id)

        candidate = max(
            (record for record in train if record.scene_id not in selected_ids),
            key=diversity_score,
        )
        add(candidate, "greedy metadata/geometry diversity fill")

    return tuple(
        SceneRecord(
            **{
                **asdict(record),
                "selection_rationale": "; ".join(reasons[record.scene_id]),
            }
        )
        for record in selected
    )


MANIFEST_FIELDS = tuple(SceneRecord.__dataclass_fields__)


def write_selection_manifest(records: Sequence[SceneRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def read_selection_manifest(path: Path) -> tuple[SceneRecord, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = []
    for row in rows:
        if row.get("split") != "train":
            raise BatchEvaluationError(
                f"Manifest contains non-TRAIN scene {row.get('scene_id')!r}"
            )
        records.append(
            SceneRecord(
                scene_id=row["scene_id"],
                split=row["split"],
                loc_id=int(row["loc_id"]),
                image_catalog_id=row["image_catalog_id"],
                width=int(row["width"]),
                height=int(row["height"]),
                gt_aircraft_count=int(row["gt_aircraft_count"]),
                country=row["country"],
                state_province=row["state_province"],
                air_field=row["air_field"],
                weather=row["weather"],
                cloud_cover_percentage=float(row["cloud_cover_percentage"]),
                off_nadir_angle=float(row["off_nadir_angle"]),
                sensor=row["sensor"],
                collection_time_start=row["collection_time_start"],
                location_observation_count=int(row["location_observation_count"]),
                selection_rationale=row["selection_rationale"],
            )
        )
    if len(records) != SAMPLE_SIZE or len({record.scene_id for record in records}) != SAMPLE_SIZE:
        raise BatchEvaluationError(f"Manifest must contain exactly {SAMPLE_SIZE} unique scenes")
    return tuple(records)


def aggregate_micro(scene_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gt = sum(int(item["gt_count"]) for item in scene_metrics)
    predictions = sum(int(item["prediction_count"]) for item in scene_metrics)
    tp = sum(int(item["tp"]) for item in scene_metrics)
    fp = sum(int(item["fp"]) for item in scene_metrics)
    fn = sum(int(item["fn"]) for item in scene_metrics)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gt": gt,
        "predictions": predictions,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate_macro(scene_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for metric in ("precision", "recall", "f1"):
        values = [float(item[metric]) for item in scene_metrics]
        result[metric] = {
            "mean": statistics.fmean(values) if values else 0.0,
            "median": statistics.median(values) if values else 0.0,
        }
    result["zero_denominator_policy"] = (
        "Per-scene precision or recall with a zero denominator is 0.0; no scene is dropped."
    )
    return result


def aggregate_distribution(
    values: Sequence[float], requested: Sequence[str]
) -> dict[str, float | None]:
    if not values:
        return {name: None for name in requested}
    operations = {
        "mean": lambda: statistics.fmean(values),
        "median": lambda: statistics.median(values),
        "p25": lambda: percentile(values, 0.25),
        "p75": lambda: percentile(values, 0.75),
        "p95": lambda: percentile(values, 0.95),
        "min": lambda: min(values),
        "max": lambda: max(values),
    }
    return {name: operations[name]() for name in requested}


def aggregate_size_recall(
    gt_rows: Sequence[Mapping[str, Any]], matched_keys: set[tuple[str, int]]
) -> dict[str, dict[str, Any]]:
    totals: Counter[str] = Counter()
    matched: Counter[str] = Counter()
    for row in gt_rows:
        category = str(row.get("faa_wingspan_class") or "missing")
        totals[category] += 1
        key = (str(row["scene_id"]), int(row["gt_index"]))
        if key in matched_keys:
            matched[category] += 1
    return {
        category: {
            "gt_count": totals[category],
            "matched_gt_count": matched[category],
            "recall": matched[category] / totals[category] if totals[category] else 0.0,
        }
        for category in sorted(totals)
    }


def aggregate_pipeline_diagnostics(results: Sequence[EvaluationResult]) -> dict[str, int]:
    return {
        "suppressed_candidates": sum(
            int(result.nms_diagnostics.get("suppressed_count", 0)) for result in results
        ),
        "suppressed_matching_current_fn": sum(
            int(result.nms_diagnostics.get("suppressed_independently_matching_current_fn", 0))
            for result in results
        ),
        "rejected_candidates": sum(
            int(result.boundary_rejection_diagnostics.get("rejected_count", 0))
            for result in results
        ),
        "reconstructable_rejected_candidates": sum(
            int(result.boundary_rejection_diagnostics.get("rejected_with_reconstructable_geometry", 0))
            for result in results
        ),
        "rejected_matching_current_fn": sum(
            int(result.boundary_rejection_diagnostics.get("rejected_independently_matching_current_fn", 0))
            for result in results
        ),
    }


def _pearson(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) < 2 or len(first) != len(second):
        return None
    first_mean, second_mean = statistics.fmean(first), statistics.fmean(second)
    numerator = sum((x - first_mean) * (y - second_mean) for x, y in zip(first, second, strict=True))
    first_scale = math.sqrt(sum((x - first_mean) ** 2 for x in first))
    second_scale = math.sqrt(sum((y - second_mean) ** 2 for y in second))
    return numerator / (first_scale * second_scale) if first_scale and second_scale else None


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + end - 1) / 2.0 + 1.0
        for position in range(index, end):
            ranks[ordered[position][0]] = average_rank
        index = end
    return ranks


def _correlations(scene_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    relationships = {
        "recall_vs_off_nadir": ("recall", "off_nadir_angle"),
        "recall_vs_gt_count": ("recall", "gt_count"),
        "runtime_vs_tile_count": ("runtime_seconds", "tile_count"),
    }
    output = {}
    for name, (first_key, second_key) in relationships.items():
        first = [float(item[first_key]) for item in scene_metrics]
        second = [float(item[second_key]) for item in scene_metrics]
        output[name] = {
            "n": len(first),
            "pearson": _pearson(first, second),
            "spearman": _pearson(_ranks(first), _ranks(second)),
        }
    output["interpretation_limit"] = (
        "n=12 is descriptive only; correlations are not causal or inferential evidence."
    )
    return output


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_fixed_batch(
    manifest_path: Path,
    imagery_dir: Path,
    annotation_dir: Path,
    model_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the frozen baseline over the already-selected TRAIN manifest."""

    records = read_selection_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_selection_manifest(records, output_dir / "selection_manifest.csv")
    model = None
    model_load_seconds = 0.0

    scene_metrics: list[dict[str, Any]] = []
    evaluation_results: list[EvaluationResult] = []
    matched_rows: list[dict[str, Any]] = []
    fp_rows: list[dict[str, Any]] = []
    fn_rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []

    expected_inference_settings = {
        "tile_size": TILE_SIZE,
        "tile_overlap": TILE_OVERLAP,
        "integer_stride": 819,
        "device": DEVICE,
        "image_size": IMAGE_SIZE,
        "inference_confidence": CONFIDENCE,
        "inference_iou": MODEL_IOU,
        "global_nms_iou": GLOBAL_NMS_IOU,
        "target_class_name": TARGET_CLASS,
    }

    for scene_number, record in enumerate(records, 1):
        raster_path = imagery_dir / f"{record.scene_id}.tif"
        gt_path = annotation_dir / f"{record.scene_id}.geojson"
        scene_root = output_dir / "scenes" / record.scene_id
        inference_dir = scene_root / "inference"
        evaluation_dir = scene_root / "evaluation"
        print(f"BATCH SCENE {scene_number}/{len(records)}: {record.scene_id}", flush=True)
        predictions_path = inference_dir / "predictions.geojson"
        suppression_path = inference_dir / "duplicate_suppressions.json"
        rejection_path = inference_dir / "rejected_candidates.json"
        inference_summary_path = inference_dir / "inference_summary.json"
        reusable_paths = (
            predictions_path,
            suppression_path,
            rejection_path,
            inference_summary_path,
        )
        infer_summary: Mapping[str, Any] | None = None
        if all(path.is_file() for path in reusable_paths):
            with inference_summary_path.open("r", encoding="utf-8") as handle:
                candidate_summary = json.load(handle)
            settings_match = all(
                candidate_summary.get(key) == value
                for key, value in expected_inference_settings.items()
            )
            if candidate_summary.get("scene_id") == record.scene_id and settings_match:
                infer_summary = candidate_summary
                print(f"REUSING COMPLETE FIXED-SETTING INFERENCE: {record.scene_id}", flush=True)
        if infer_summary is None:
            if model is None:
                load_start = time.perf_counter()
                model = load_obb_model(model_path, DEVICE)
                model_load_seconds = time.perf_counter() - load_start
            inference = process_raster_scene(
                model,
                SceneInferenceSettings(
                    model_path=model_path,
                    raster_path=raster_path,
                    output_dir=inference_dir,
                    device=DEVICE,
                    image_size=IMAGE_SIZE,
                    confidence=CONFIDENCE,
                    model_iou=MODEL_IOU,
                    global_nms_iou=GLOBAL_NMS_IOU,
                    tile_size=TILE_SIZE,
                    tile_overlap=TILE_OVERLAP,
                    target_class=TARGET_CLASS,
                ),
                model_load_seconds=model_load_seconds,
            )
            model_load_seconds = 0.0
            infer_summary = asdict(inference.summary)
            predictions_path = inference.predictions_path
            suppression_path = inference.suppression_path
            rejection_path = inference.rejection_path
        evaluation = evaluate_scene(
            predictions_path,
            gt_path,
            raster_path,
            match_iou=MATCH_IOU,
            suppression_json=suppression_path,
            rejection_json=rejection_path,
        )
        write_evaluation_outputs(evaluation, evaluation_dir)
        evaluation_results.append(evaluation)
        summary = evaluation.summary
        scene_metrics.append(
            {
                "scene_id": record.scene_id,
                "loc_id": record.loc_id,
                "country": record.country,
                "air_field": record.air_field,
                "weather": record.weather,
                "cloud_cover_percentage": record.cloud_cover_percentage,
                "off_nadir_angle": record.off_nadir_angle,
                "width": record.width,
                "height": record.height,
                "tile_count": infer_summary["tile_count"],
                "runtime_seconds": infer_summary["total_runtime_seconds"],
                "all_class_raw_detections": infer_summary["all_class_raw_detections"],
                "raw_plane_candidates": infer_summary["raw_target_candidates"],
                "invalid_rejected_candidates": infer_summary["invalid_or_padding_rejected"],
                "pre_nms_planes": infer_summary["pre_nms_target_count"],
                "duplicates_suppressed": infer_summary["duplicates_suppressed"],
                "gt_count": summary["ground_truth_count"],
                "prediction_count": summary["prediction_count"],
                "tp": summary["tp"],
                "fp": summary["fp"],
                "fn": summary["fn"],
                "precision": summary["precision"],
                "recall": summary["recall"],
                "f1": summary["f1"],
                "matched_mean_iou": summary["matched_obb_iou"]["mean"],
                "matched_median_iou": summary["matched_obb_iou"]["median"],
                "mean_center_error_pixels": summary["center_error_pixels"]["mean"],
                "mean_center_error_metres": summary["center_error_metres"]["mean"],
                "suppressed_matching_current_fn": evaluation.nms_diagnostics.get(
                    "suppressed_independently_matching_current_fn", 0
                ),
                "rejected_matching_current_fn": evaluation.boundary_rejection_diagnostics.get(
                    "rejected_independently_matching_current_fn", 0
                ),
            }
        )
        matched_rows.extend({"scene_id": record.scene_id, **asdict(item)} for item in evaluation.matches)
        fp_rows.extend({"scene_id": record.scene_id, **asdict(item)} for item in evaluation.false_positives)
        for item in evaluation.false_negatives:
            value = asdict(item)
            properties = value.pop("properties")
            fn_rows.append(
                {
                    "scene_id": record.scene_id,
                    **value,
                    "faa_wingspan_class": properties.get("faa_wingspan_class"),
                    "location": properties.get("location"),
                }
            )
        with gt_path.open("r", encoding="utf-8") as handle:
            features = json.load(handle)["features"]
        for gt_index, feature in enumerate(features):
            gt_rows.append(
                {
                    "scene_id": record.scene_id,
                    "gt_index": gt_index,
                    **feature.get("properties", {}),
                }
            )

    total_runtime = sum(float(item["runtime_seconds"]) for item in scene_metrics)
    micro = aggregate_micro(scene_metrics)
    macro = aggregate_macro(scene_metrics)
    matched_keys = {(row["scene_id"], int(row["gt_index"])) for row in matched_rows}
    size_recall = aggregate_size_recall(gt_rows, matched_keys)
    pipeline = aggregate_pipeline_diagnostics(evaluation_results)
    normalized_iou = [float(row["normalized_obb_iou"]) for row in matched_rows]
    diamond_iou = [float(row["original_diamond_iou"]) for row in matched_rows]
    center_pixels = [float(row["center_error_pixels"]) for row in matched_rows]
    center_metres = [float(row["center_error_metres"]) for row in matched_rows]
    aggregate = {
        "sample_name": "fixed_baseline_12",
        "sample_size": len(records),
        "selection_policy": "fixed anchors plus deterministic metadata/geometry diversity fill",
        "fixed_settings": {
            "confidence": CONFIDENCE,
            "model_iou": MODEL_IOU,
            "global_polygon_nms_iou": GLOBAL_NMS_IOU,
            "tile_size": TILE_SIZE,
            "overlap": TILE_OVERLAP,
            "stride": 819,
            "target_class": TARGET_CLASS,
            "device": DEVICE,
            "match_iou": MATCH_IOU,
        },
        "total_tiles": sum(int(item["tile_count"]) for item in scene_metrics),
        "total_cpu_runtime_seconds": total_runtime,
        "micro": micro,
        "macro": macro,
        "matched_normalized_obb_iou": aggregate_distribution(
            normalized_iou, ("mean", "median", "p25", "p75", "min", "max")
        ),
        "original_diamond_iou": aggregate_distribution(diamond_iou, ("mean", "median")),
        "center_error_pixels": aggregate_distribution(
            center_pixels, ("mean", "median", "p95", "max")
        ),
        "center_error_metres": aggregate_distribution(
            center_metres, ("mean", "median", "p95", "max")
        ),
        "size_category_recall": size_recall,
        "pipeline_diagnostics": pipeline,
        "condition_relationships": _correlations(scene_metrics),
        "scene_metrics": scene_metrics,
    }

    _write_rows(output_dir / "per_scene_metrics.csv", scene_metrics, tuple(scene_metrics[0]))
    _write_rows(output_dir / "matched_detections.csv", matched_rows, tuple(matched_rows[0]) if matched_rows else ("scene_id",))
    _write_rows(output_dir / "false_positives.csv", fp_rows, tuple(fp_rows[0]) if fp_rows else ("scene_id",))
    _write_rows(output_dir / "false_negatives.csv", fn_rows, tuple(fn_rows[0]) if fn_rows else ("scene_id",))
    _write_json(output_dir / "pipeline_diagnostics.json", pipeline)
    _write_json(output_dir / "aggregate_summary.json", aggregate)
    return aggregate


def _default_paths() -> dict[str, Path]:
    root = find_project_root()
    rareplanes = root / "datasets" / "RarePlanes"
    return {
        "root": root,
        "metadata": rareplanes / "metadata" / "RarePlanes_Public_Metadata.csv",
        "imagery": rareplanes / "train" / "imagery" / "PS-RGB_cog",
        "annotations": rareplanes / "train" / "annotations" / "geojson_aircraft",
        "manifest": root / ".tmp" / "fixed_baseline_12_selection_manifest.csv",
        "output": root / "outputs" / "batch_train_evaluation" / "fixed_baseline_12",
        "model": root / "best.pt",
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description="Select or run the fixed 12-scene TRAIN baseline.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select", help="Create the deterministic TRAIN manifest only")
    select.add_argument("--metadata", type=Path, default=defaults["metadata"])
    select.add_argument("--imagery-dir", type=Path, default=defaults["imagery"])
    select.add_argument("--annotation-dir", type=Path, default=defaults["annotations"])
    select.add_argument("--manifest", type=Path, default=defaults["manifest"])
    run = subparsers.add_parser("run", help="Run fixed inference and evaluation from a manifest")
    run.add_argument("--manifest", type=Path, default=defaults["manifest"])
    run.add_argument("--imagery-dir", type=Path, default=defaults["imagery"])
    run.add_argument("--annotation-dir", type=Path, default=defaults["annotations"])
    run.add_argument("--model", type=Path, default=defaults["model"])
    run.add_argument("--output-dir", type=Path, default=defaults["output"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "select":
        records = load_train_scene_records(args.metadata, args.imagery_dir, args.annotation_dir)
        selected = deterministic_select_train_scenes(records)
        write_selection_manifest(selected, args.manifest)
        print(json.dumps([asdict(record) for record in selected], indent=2, ensure_ascii=False))
        print(f"Manifest: {args.manifest}")
        return 0
    aggregate = run_fixed_batch(
        args.manifest,
        args.imagery_dir,
        args.annotation_dir,
        args.model,
        args.output_dir,
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
