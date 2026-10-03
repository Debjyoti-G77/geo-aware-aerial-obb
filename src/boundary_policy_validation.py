"""Independent TRAIN holdout validation for frozen tile-boundary policies.

Production inference remains Policy A.  This module selects a location-disjoint
holdout before inference, runs each scene in a fresh child process, and applies
the already-defined Policies B/C/D only to retained diagnostic artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

try:
    from .batch_evaluation import (
        CONFIDENCE,
        DEVICE,
        GLOBAL_NMS_IOU,
        IMAGE_SIZE,
        MATCH_IOU,
        MODEL_IOU,
        TARGET_CLASS,
        TILE_OVERLAP,
        TILE_SIZE,
        SceneRecord,
        aggregate_macro,
        aggregate_micro,
        load_train_scene_records,
    )
    from .boundary_rejection_analysis import (
        POLICIES,
        analyze_scene,
        policy_accepts,
        simulate_policy,
    )
    from .geospatial_evaluation import (
        evaluate_scene,
        geodesic_distance_metres,
        make_raster_to_crs84_transformer,
        pixel_point_to_crs84,
        polygon_iou,
        write_evaluation_outputs,
    )
    from .geospatial_inference import (
        SceneInferenceSettings,
        load_obb_model,
        process_raster_scene,
    )
    from .project_config import find_project_root
except ImportError:  # Support direct execution.
    from batch_evaluation import (
        CONFIDENCE,
        DEVICE,
        GLOBAL_NMS_IOU,
        IMAGE_SIZE,
        MATCH_IOU,
        MODEL_IOU,
        TARGET_CLASS,
        TILE_OVERLAP,
        TILE_SIZE,
        SceneRecord,
        aggregate_macro,
        aggregate_micro,
        load_train_scene_records,
    )
    from boundary_rejection_analysis import (
        POLICIES,
        analyze_scene,
        policy_accepts,
        simulate_policy,
    )
    from geospatial_evaluation import (
        evaluate_scene,
        geodesic_distance_metres,
        make_raster_to_crs84_transformer,
        pixel_point_to_crs84,
        polygon_iou,
        write_evaluation_outputs,
    )
    from geospatial_inference import (
        SceneInferenceSettings,
        load_obb_model,
        process_raster_scene,
    )
    from project_config import find_project_root


HOLDOUT_SIZE = 12
DEVELOPMENT_SCENE_IDS = frozenset(
    {
        "2_104001003B4B5900",
        "33_1040010003449400",
        "96_10400100096C2500",
        "112_104001001FBAB400",
        "130_104001000F25CE00",
        "30_104001002394E000",
        "88_1040010019A43D00",
        "86_104001003D8DB300",
        "88_104001002B095600",
        "24_104001002299DB00",
        "125_10400100088C2C00",
        "18_10400100211BDD00",
    }
)
DEVELOPMENT_LOC_IDS = frozenset(
    int(scene_id.split("_", 1)[0]) for scene_id in DEVELOPMENT_SCENE_IDS
)
POLICY_LABELS = {
    "current": "A",
    "centroid": "B",
    "area90": "C",
    "area75": "D",
}
EXPECTED_STRIDE = 819
EXPECTED_SETTINGS = {
    "tile_size": TILE_SIZE,
    "tile_overlap": TILE_OVERLAP,
    "integer_stride": EXPECTED_STRIDE,
    "device": DEVICE,
    "image_size": IMAGE_SIZE,
    "inference_confidence": CONFIDENCE,
    "inference_iou": MODEL_IOU,
    "global_nms_iou": GLOBAL_NMS_IOU,
    "target_class_name": TARGET_CLASS,
}


class HoldoutValidationError(RuntimeError):
    """Raised when selection, scene outputs, or frozen settings are invalid."""


@dataclass(frozen=True)
class HoldoutRecord:
    scene_id: str
    split: str
    loc_id: int
    image_catalog_id: str
    width: int
    height: int
    image_area_pixels: int
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
    aircraft_size_composition: str
    selection_reason: str


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _size_composition(annotation_path: Path) -> str:
    counts: Counter[str] = Counter()
    for feature in _read_json(annotation_path).get("features", []):
        value = (feature.get("properties") or {}).get("faa_wingspan_class")
        counts[str(value) if value is not None else "missing"] += 1
    return json.dumps(dict(sorted(counts.items())), separators=(",", ":"))


def build_holdout_records(
    records: Sequence[SceneRecord], annotation_dir: Path
) -> tuple[HoldoutRecord, ...]:
    """Add pre-inference aircraft-size composition to TRAIN metadata."""

    output = []
    for record in records:
        if record.split != "train":
            continue
        output.append(
            HoldoutRecord(
                scene_id=record.scene_id,
                split="train",
                loc_id=record.loc_id,
                image_catalog_id=record.image_catalog_id,
                width=record.width,
                height=record.height,
                image_area_pixels=record.pixel_count,
                gt_aircraft_count=record.gt_aircraft_count,
                country=record.country,
                state_province=record.state_province,
                air_field=record.air_field,
                weather=record.weather,
                cloud_cover_percentage=record.cloud_cover_percentage,
                off_nadir_angle=record.off_nadir_angle,
                sensor=record.sensor,
                collection_time_start=record.collection_time_start,
                location_observation_count=record.location_observation_count,
                aircraft_size_composition=_size_composition(
                    annotation_dir / f"{record.scene_id}.geojson"
                ),
                selection_reason="",
            )
        )
    return tuple(output)


def _median_representative(records: Sequence[HoldoutRecord], field: str) -> HoldoutRecord:
    values = [float(getattr(record, field)) for record in records]
    target = statistics.median(values)
    return min(records, key=lambda record: (abs(float(getattr(record, field)) - target), record.scene_id))


def _size_share(record: HoldoutRecord, predicate: Any) -> float:
    counts = json.loads(record.aircraft_size_composition)
    total = sum(int(value) for value in counts.values())
    selected = sum(int(value) for key, value in counts.items() if predicate(key))
    return selected / total if total else 0.0


def select_location_disjoint_holdout(
    records: Sequence[HoldoutRecord], count: int = HOLDOUT_SIZE
) -> tuple[HoldoutRecord, ...]:
    """Select a deterministic TRAIN holdout using pre-inference metadata only."""

    eligible = sorted(
        (
            record
            for record in records
            if record.split == "train"
            and record.scene_id not in DEVELOPMENT_SCENE_IDS
            and record.loc_id not in DEVELOPMENT_LOC_IDS
        ),
        key=lambda record: record.scene_id,
    )
    eligible_locations = {record.loc_id for record in eligible}
    if len(eligible_locations) < count:
        raise HoldoutValidationError(
            f"Location-disjoint selection requires {count} locations, but only "
            f"{len(eligible_locations)} are eligible; exclusion was not relaxed."
        )

    selected: list[HoldoutRecord] = []
    reasons: dict[str, list[str]] = defaultdict(list)

    def add(record: HoldoutRecord, reason: str) -> None:
        if record.loc_id in {item.loc_id for item in selected}:
            return
        if record.scene_id in {item.scene_id for item in selected}:
            return
        if len(selected) < count:
            selected.append(record)
            reasons[record.scene_id].append(reason)

    add(min(eligible, key=lambda item: (item.gt_aircraft_count, item.scene_id)), "minimum eligible GT-count anchor")
    add(_median_representative(eligible, "gt_aircraft_count"), "median eligible GT-count anchor")
    add(max(eligible, key=lambda item: (item.gt_aircraft_count, item.scene_id)), "maximum eligible GT-count anchor")
    add(min(eligible, key=lambda item: (item.image_area_pixels, item.scene_id)), "minimum eligible image-area anchor")
    add(_median_representative(eligible, "image_area_pixels"), "median eligible image-area anchor")
    add(max(eligible, key=lambda item: (item.image_area_pixels, item.scene_id)), "maximum eligible image-area anchor")
    add(min(eligible, key=lambda item: (item.off_nadir_angle, item.scene_id)), "minimum eligible off-nadir anchor")
    add(max(eligible, key=lambda item: (item.off_nadir_angle, item.scene_id)), "maximum eligible off-nadir anchor")
    add(max(eligible, key=lambda item: (item.cloud_cover_percentage, item.scene_id)), "maximum eligible cloud-cover anchor")
    add(max(eligible, key=lambda item: (_size_share(item, lambda key: key == "1"), item.scene_id)), "maximum small-aircraft composition anchor")
    add(max(eligible, key=lambda item: (_size_share(item, lambda key: key not in {"1", "missing"}), item.scene_id)), "maximum larger-aircraft composition anchor")

    numeric_fields = (
        "gt_aircraft_count",
        "image_area_pixels",
        "off_nadir_angle",
        "cloud_cover_percentage",
    )
    bounds = {
        field: (
            min(float(getattr(record, field)) for record in eligible),
            max(float(getattr(record, field)) for record in eligible),
        )
        for field in numeric_fields
    }

    def vector(record: HoldoutRecord) -> tuple[float, ...]:
        values = []
        for field in numeric_fields:
            low, high = bounds[field]
            value = float(getattr(record, field))
            values.append((value - low) / (high - low or 1.0))
        values.extend(
            (
                _size_share(record, lambda key: key == "1"),
                _size_share(record, lambda key: key not in {"1", "missing"}),
            )
        )
        return tuple(values)

    vectors = {record.scene_id: vector(record) for record in eligible}
    while len(selected) < count:
        selected_ids = {record.scene_id for record in selected}
        selected_locations = {record.loc_id for record in selected}
        countries = {record.country for record in selected}
        weathers = {record.weather for record in selected}

        def score(record: HoldoutRecord) -> tuple[float, str]:
            minimum_distance = min(
                math.dist(vectors[record.scene_id], vectors[item.scene_id])
                for item in selected
            )
            diversity = minimum_distance
            diversity += 0.35 if record.country not in countries else 0.0
            diversity += 0.20 if record.weather not in weathers else 0.0
            return diversity, record.scene_id

        choices = [
            record
            for record in eligible
            if record.scene_id not in selected_ids and record.loc_id not in selected_locations
        ]
        add(max(choices, key=score), "greedy pre-inference metadata diversity fill")

    result = tuple(
        replace(record, selection_reason="; ".join(reasons[record.scene_id]))
        for record in selected
    )
    validate_holdout_selection(result, count)
    return result


def validate_holdout_selection(
    records: Sequence[HoldoutRecord], count: int = HOLDOUT_SIZE
) -> None:
    if len(records) != count or len({record.scene_id for record in records}) != count:
        raise HoldoutValidationError(f"Holdout must contain exactly {count} unique scenes")
    if any(record.split != "train" for record in records):
        raise HoldoutValidationError("Holdout contains a non-TRAIN scene")
    scene_overlap = {record.scene_id for record in records} & DEVELOPMENT_SCENE_IDS
    location_overlap = {record.loc_id for record in records} & DEVELOPMENT_LOC_IDS
    if scene_overlap or location_overlap:
        raise HoldoutValidationError(
            f"Development overlap is forbidden: scenes={sorted(scene_overlap)}, "
            f"locations={sorted(location_overlap)}"
        )
    if len({record.loc_id for record in records}) != count:
        raise HoldoutValidationError("Holdout must use one scene per location")


MANIFEST_FIELDS = tuple(HoldoutRecord.__dataclass_fields__)


def write_manifest(records: Sequence[HoldoutRecord], path: Path) -> None:
    validate_holdout_selection(records)
    _write_csv(path, [asdict(record) for record in records], MANIFEST_FIELDS)


def read_manifest(path: Path) -> tuple[HoldoutRecord, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = tuple(
        HoldoutRecord(
            scene_id=row["scene_id"],
            split=row["split"],
            loc_id=int(row["loc_id"]),
            image_catalog_id=row["image_catalog_id"],
            width=int(row["width"]),
            height=int(row["height"]),
            image_area_pixels=int(row["image_area_pixels"]),
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
            aircraft_size_composition=row["aircraft_size_composition"],
            selection_reason=row["selection_reason"],
        )
        for row in rows
    )
    validate_holdout_selection(records)
    return records


REQUIRED_INFERENCE_FILES = (
    "predictions.geojson",
    "inference_summary.json",
    "duplicate_suppressions.json",
    "rejected_candidates.json",
)
REQUIRED_EVALUATION_FILES = (
    "evaluation_summary.json",
    "matches.csv",
    "false_positives.csv",
    "false_negatives.csv",
    "boundary_rejection_diagnostics.json",
    "nms_diagnostics.json",
)


def scene_output_complete(scene_root: Path, scene_id: str) -> bool:
    """Accept only a complete, frozen-setting scene written by this runner."""

    inference_dir = scene_root / "inference"
    evaluation_dir = scene_root / "evaluation"
    marker = scene_root / "completed.json"
    required = [inference_dir / name for name in REQUIRED_INFERENCE_FILES]
    required.extend(evaluation_dir / name for name in REQUIRED_EVALUATION_FILES)
    if not marker.is_file() or not all(path.is_file() for path in required):
        return False
    try:
        summary = _read_json(inference_dir / "inference_summary.json")
        evaluation = _read_json(evaluation_dir / "evaluation_summary.json")
        completion = _read_json(marker)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        summary.get("scene_id") == scene_id
        and all(summary.get(key) == value for key, value in EXPECTED_SETTINGS.items())
        and evaluation.get("scene_id") == scene_id
        and evaluation.get("match_iou_threshold") == MATCH_IOU
        and completion.get("scene_id") == scene_id
        and completion.get("status") == "complete"
        and completion.get("frozen_settings") == EXPECTED_SETTINGS
    )


def run_one_scene(
    record: HoldoutRecord,
    imagery_dir: Path,
    annotation_dir: Path,
    model_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Run production Policy A inference and evaluation for exactly one scene."""

    scene_root = output_root / "scenes" / record.scene_id
    if scene_output_complete(scene_root, record.scene_id):
        return _read_json(scene_root / "completed.json")
    raster_path = imagery_dir / f"{record.scene_id}.tif"
    annotation_path = annotation_dir / f"{record.scene_id}.geojson"
    start = time.perf_counter()
    model_start = time.perf_counter()
    model = load_obb_model(model_path, DEVICE)
    model_load_seconds = time.perf_counter() - model_start
    inference = process_raster_scene(
        model,
        SceneInferenceSettings(
            model_path=model_path,
            raster_path=raster_path,
            output_dir=scene_root / "inference",
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
    evaluation = evaluate_scene(
        inference.predictions_path,
        annotation_path,
        raster_path,
        match_iou=MATCH_IOU,
        suppression_json=inference.suppression_path,
        rejection_json=inference.rejection_path,
    )
    write_evaluation_outputs(evaluation, scene_root / "evaluation")
    completion = {
        "scene_id": record.scene_id,
        "status": "complete",
        "frozen_settings": EXPECTED_SETTINGS,
        "model_load_seconds": model_load_seconds,
        "process_runtime_seconds": time.perf_counter() - start,
        "tile_count": inference.summary.tile_count,
    }
    _write_json(scene_root / "completed.json", completion)
    if not scene_output_complete(scene_root, record.scene_id):
        raise HoldoutValidationError(f"Scene output failed completion check: {record.scene_id}")
    return completion


def run_scenes_isolated(
    manifest_path: Path,
    imagery_dir: Path,
    annotation_dir: Path,
    model_path: Path,
    output_root: Path,
    max_attempts: int = 2,
) -> list[dict[str, Any]]:
    """Run each manifest scene in a fresh interpreter with bounded resume."""

    records = read_manifest(manifest_path)
    execution_path = output_root / "process_execution_log.json"
    if execution_path.is_file():
        existing = _read_json(execution_path)
        attempts = list(existing.get("attempts", []))
    else:
        attempts = []
    for index, record in enumerate(records, 1):
        scene_root = output_root / "scenes" / record.scene_id
        if scene_output_complete(scene_root, record.scene_id):
            print(f"HOLDOUT {index}/{len(records)} REUSE: {record.scene_id}", flush=True)
            continue
        for attempt_number in range(1, max_attempts + 1):
            print(
                f"HOLDOUT {index}/{len(records)} ATTEMPT {attempt_number}: {record.scene_id}",
                flush=True,
            )
            start = time.perf_counter()
            command = build_scene_command(
                record.scene_id,
                manifest_path,
                imagery_dir,
                annotation_dir,
                model_path,
                output_root,
            )
            completed = subprocess.run(command, check=False)
            complete = scene_output_complete(scene_root, record.scene_id)
            attempts.append(
                {
                    "scene_id": record.scene_id,
                    "attempt": attempt_number,
                    "exit_code": completed.returncode,
                    "complete_output": complete,
                    "wall_seconds": time.perf_counter() - start,
                }
            )
            _write_json(
                execution_path,
                {
                    "process_isolation": "one fresh Python process per scene attempt",
                    "max_attempts_per_scene": max_attempts,
                    "attempts": attempts,
                },
            )
            if complete:
                break
            if completed.returncode == 0:
                raise HoldoutValidationError(
                    f"Scene process exited zero but output is incomplete: {record.scene_id}"
                )
        if not scene_output_complete(scene_root, record.scene_id):
            raise HoldoutValidationError(
                f"Scene failed after {max_attempts} unchanged fresh-process attempts: {record.scene_id}"
            )
    return attempts


def build_scene_command(
    scene_id: str,
    manifest_path: Path,
    imagery_dir: Path,
    annotation_dir: Path,
    model_path: Path,
    output_root: Path,
) -> list[str]:
    """Build one child command containing exactly one frozen manifest scene."""

    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "run-scene",
        "--scene-id",
        scene_id,
        "--manifest",
        str(manifest_path),
        "--imagery-dir",
        str(imagery_dir),
        "--annotation-dir",
        str(annotation_dir),
        "--model",
        str(model_path),
        "--output-dir",
        str(output_root),
    ]


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
    }


def _nearest_gt_iou(detection: Any, truths: Sequence[Any]) -> float:
    return max(
        (polygon_iou(detection.global_pixel_polygon, truth.obb_pixel_polygon) for truth in truths),
        default=0.0,
    )


def _fp_iou_band(value: float) -> str:
    if value >= 0.45:
        return ">=0.45_and_<0.50"
    if value >= 0.25:
        return ">=0.25_and_<0.45"
    if value > 0.0:
        return ">0_and_<0.25"
    return "0_overlap"


def _match_quality(
    scene: Mapping[str, Any], result: Any, current_matched: set[int], raster_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import rasterio
    from shapely.geometry import box

    rejected_by_id = {item.detection.detection_id: item for item in scene["rejected"]}
    recovered = []
    current_rows = []
    with rasterio.open(raster_path) as raster:
        transformer = make_raster_to_crs84_transformer(raster.crs)
        transform = raster.transform
        for prediction_index, gt_index, overlap in result.matches:
            detection = result.kept[prediction_index]
            truth = scene["truths"][gt_index]
            prediction_center = detection.global_pixel_polygon.centroid
            truth_center = truth.diamond_pixel_polygon.centroid
            center_pixels = float(prediction_center.distance(truth_center))
            prediction_world = pixel_point_to_crs84(
                prediction_center.x, prediction_center.y, transform, transformer
            )
            truth_world = pixel_point_to_crs84(
                truth_center.x, truth_center.y, transform, transformer
            )
            row = {
                "scene_id": scene["scene_id"],
                "policy": result.policy,
                "detection_id": detection.detection_id,
                "gt_index": gt_index,
                "normalized_obb_iou": overlap,
                "confidence": detection.confidence,
                "center_error_pixels": center_pixels,
                "center_error_metres": geodesic_distance_metres(prediction_world, truth_world),
                "gt_visible_fraction": None,
                "candidate_inside_tile_fraction": None,
            }
            rejected = rejected_by_id.get(detection.detection_id)
            if rejected is not None:
                tile_box = box(*detection.tile_pixel_bounds)
                row["gt_visible_fraction"] = float(
                    truth.obb_pixel_polygon.intersection(tile_box).area
                    / truth.obb_pixel_polygon.area
                )
                row["candidate_inside_tile_fraction"] = rejected.inside_tile_fraction
            if result.policy == "current":
                current_rows.append(row)
            elif gt_index not in current_matched:
                recovered.append(row)
    return recovered, current_rows


def calculate_policy_deltas(
    aggregate: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    current = aggregate["current"]
    rows = []
    for policy in POLICIES[1:]:
        item = aggregate[policy]
        rows.append(
            {
                "policy": policy,
                "policy_label": POLICY_LABELS[policy],
                "delta_predictions": item["predictions"] - current["predictions"],
                "delta_tp": item["tp"] - current["tp"],
                "delta_fp": item["fp"] - current["fp"],
                "delta_fn": item["fn"] - current["fn"],
                "delta_micro_precision": item["precision"] - current["precision"],
                "delta_micro_recall": item["recall"] - current["recall"],
                "delta_micro_f1": item["f1"] - current["f1"],
            }
        )
    return rows


def unique_recovery_accounting(
    current_false_gt: set[int],
    alternative_false_gt: set[int],
    represented_gt: set[int],
) -> dict[str, int]:
    """Count GT identities, never candidate-to-GT links, for recovery claims."""

    recovered = current_false_gt - alternative_false_gt
    return {
        "represented_policy_a_fn_gt": len(represented_gt),
        "unique_fn_recovered": len(recovered),
        "represented_remaining_fn": len(represented_gt & alternative_false_gt),
    }


def aggregate_validation(
    manifest_path: Path,
    imagery_dir: Path,
    annotation_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Aggregate frozen offline policies after every isolated scene completes."""

    records = read_manifest(manifest_path)
    incomplete = [
        record.scene_id
        for record in records
        if not scene_output_complete(output_root / "scenes" / record.scene_id, record.scene_id)
    ]
    if incomplete:
        raise HoldoutValidationError(f"Cannot aggregate incomplete scenes: {incomplete}")

    scene_analyses = {}
    for record in records:
        print(f"AGGREGATE SCENE: {record.scene_id}", flush=True)
        scene_analyses[record.scene_id] = analyze_scene(
            record.scene_id,
            output_root / "scenes" / record.scene_id,
            imagery_dir,
            annotation_dir,
        )
    policy_results: dict[str, dict[str, Any]] = defaultdict(dict)
    per_scene_rows = []
    recovered_rows = []
    current_quality_rows = []
    additional_fp_rows = []
    recovery_accounting: dict[str, Counter[str]] = {
        policy: Counter() for policy in POLICIES[1:]
    }
    safety = Counter()

    for record in records:
        scene = scene_analyses[record.scene_id]
        truths = scene["truths"]
        results = {
            policy: simulate_policy(
                scene["accepted_pre_nms"], scene["rejected"], truths, policy
            )
            for policy in POLICIES
        }
        policy_results[record.scene_id] = results
        current = results["current"]
        current_matched = {gt_index for _, gt_index, _ in current.matches}
        current_fn = set(current.false_gt_indices)
        current_fp_ids = {
            current.kept[index].detection_id
            for index in range(len(current.kept))
            if index not in {match[0] for match in current.matches}
        }
        _, current_rows = _match_quality(
            scene,
            current,
            current_matched,
            imagery_dir / f"{record.scene_id}.tif",
        )
        current_quality_rows.extend(current_rows)

        for rejected in scene["rejected"]:
            safety[rejected.boundary_category] += 1
            safety["invalid_polygons"] += int(not rejected.attempted_global_polygon_valid)
            safety["source_raster_crossings"] += int(
                rejected.attempted_global_polygon_valid
                and not rejected.global_polygon_inside_source
            )
            safety["padding_related"] += int(rejected.has_padding)

        for policy in POLICIES:
            result = results[policy]
            current_ids = {item.detection_id for item in current.kept}
            result_ids = {item.detection_id for item in result.kept}
            admitted = [item for item in scene["rejected"] if policy_accepts(item, policy)]
            admitted_ids = {item.detection.detection_id for item in admitted}
            represented = {
                truth_index
                for truth_index in current_fn
                if any(
                    polygon_iou(item.detection.global_pixel_polygon, truths[truth_index].obb_pixel_polygon)
                    >= MATCH_IOU
                    for item in admitted
                )
            }
            if policy != "current":
                recovery_accounting[policy].update(
                    {
                        **unique_recovery_accounting(
                            current_fn, set(result.false_gt_indices), represented
                        ),
                        "admitted_removed_by_nms": result.admitted_removed_by_nms,
                        "policy_a_final_displaced": len(current_ids - result_ids),
                    }
                )
                quality, _ = _match_quality(
                    scene,
                    result,
                    current_matched,
                    imagery_dir / f"{record.scene_id}.tif",
                )
                recovered_rows.extend(quality)
                matched_prediction_indices = {match[0] for match in result.matches}
                for prediction_index, detection in enumerate(result.kept):
                    if prediction_index in matched_prediction_indices:
                        continue
                    if detection.detection_id in current_fp_ids:
                        continue
                    overlap = _nearest_gt_iou(detection, truths)
                    additional_fp_rows.append(
                        {
                            "scene_id": record.scene_id,
                            "policy": policy,
                            "detection_id": detection.detection_id,
                            "confidence": detection.confidence,
                            "nearest_gt_iou": overlap,
                            "nearest_gt_iou_band": _fp_iou_band(overlap),
                            "from_admitted_rejected": detection.detection_id in admitted_ids,
                        }
                    )
            per_scene_rows.append(
                {
                    "scene_id": record.scene_id,
                    "loc_id": record.loc_id,
                    "country": record.country,
                    "weather": record.weather,
                    "aircraft_size_composition": record.aircraft_size_composition,
                    "policy": policy,
                    "policy_label": POLICY_LABELS[policy],
                    "gt_count": len(truths),
                    "prediction_count": result.predictions,
                    "tp": result.tp,
                    "fp": result.fp,
                    "fn": result.fn,
                    "precision": result.precision,
                    "recall": result.recall,
                    "f1": result.f1,
                    "delta_tp": result.tp - current.tp,
                    "delta_fp": result.fp - current.fp,
                    "delta_fn": result.fn - current.fn,
                    "admitted_rejected": result.admitted_rejected,
                    "admitted_removed_by_nms": result.admitted_removed_by_nms,
                    "policy_a_final_displaced": len(current_ids - result_ids),
                }
            )

    aggregate: dict[str, Any] = {}
    for policy in POLICIES:
        rows = [row for row in per_scene_rows if row["policy"] == policy]
        micro = aggregate_micro(rows)
        aggregate[policy] = {
            **micro,
            "macro": aggregate_macro(rows),
            "policy_label": POLICY_LABELS[policy],
        }
    deltas = calculate_policy_deltas(aggregate)

    quality_summary = {}
    for policy in POLICIES[1:]:
        rows = [row for row in recovered_rows if row["policy"] == policy]
        quality_summary[policy] = {
            field: _distribution([float(row[field]) for row in rows if row[field] is not None])
            for field in (
                "normalized_obb_iou",
                "confidence",
                "center_error_pixels",
                "center_error_metres",
                "gt_visible_fraction",
                "candidate_inside_tile_fraction",
            )
        }
    quality_summary["policy_a_original_tp"] = {
        field: _distribution([float(row[field]) for row in current_quality_rows if row[field] is not None])
        for field in (
            "normalized_obb_iou",
            "confidence",
            "center_error_pixels",
            "center_error_metres",
        )
    }

    fp_summary = {}
    for policy in POLICIES[1:]:
        rows = [row for row in additional_fp_rows if row["policy"] == policy]
        fp_summary[policy] = {
            "count": len(rows),
            "nearest_gt_iou_bands": dict(
                sorted(Counter(str(row["nearest_gt_iou_band"]) for row in rows).items())
            ),
            "confidence": _distribution([float(row["confidence"]) for row in rows]),
        }

    area90_rows = [row for row in per_scene_rows if row["policy"] == "area90"]
    positive_tp = [row for row in area90_rows if int(row["delta_tp"]) > 0]
    positive_fp = [row for row in area90_rows if int(row["delta_fp"]) > 0]
    largest = max(area90_rows, key=lambda row: (int(row["delta_tp"]), row["scene_id"]))
    total_gain = sum(int(row["delta_tp"]) for row in area90_rows)
    policy_c_distribution = {
        "scenes_with_tp_gain": len(positive_tp),
        "scenes_with_fp_gain": len(positive_fp),
        "largest_tp_gain_scene": largest,
        "fraction_total_tp_gain_from_largest": (
            int(largest["delta_tp"]) / total_gain if total_gain > 0 else 0.0
        ),
    }
    source_safety = {
        **dict(safety),
        "alternative_admitted_source_unsafe": sum(
            not item.global_polygon_inside_source
            for scene in scene_analyses.values()
            for item in scene["rejected"]
            if any(policy_accepts(item, policy) for policy in POLICIES[1:])
        ),
    }
    comparison = {
        "scope": "independent location-disjoint RarePlanes TRAIN holdout",
        "holdout_scene_count": len(records),
        "development_scene_overlap": 0,
        "development_loc_id_overlap": 0,
        "frozen_settings": {**EXPECTED_SETTINGS, "evaluation_match_iou": MATCH_IOU},
        "policy_definitions": {
            "A": "all four vertices inside valid tile (unchanged production output)",
            "B": "centroid inside tile and global polygon inside source raster",
            "C": ">=90% polygon area inside tile and global polygon inside source raster",
            "D": ">=75% polygon area inside tile and global polygon inside source raster",
        },
        "total_tiles": sum(
            int(_read_json(output_root / "scenes" / record.scene_id / "inference" / "inference_summary.json")["tile_count"])
            for record in records
        ),
        "policies": aggregate,
        "recovery_accounting": {
            policy: dict(values) for policy, values in recovery_accounting.items()
        },
        "recovered_tp_quality": quality_summary,
        "additional_false_positives": fp_summary,
        "policy_c_scene_distribution": policy_c_distribution,
        "source_raster_safety": source_safety,
    }

    _write_csv(
        output_root / "per_scene_policy_metrics.csv",
        per_scene_rows,
        tuple(per_scene_rows[0]),
    )
    _write_json(output_root / "aggregate_policy_comparison.json", comparison)
    _write_csv(output_root / "policy_deltas.csv", deltas, tuple(deltas[0]))
    quality_fields = tuple(recovered_rows[0]) if recovered_rows else (
        "scene_id", "policy", "detection_id", "gt_index"
    )
    _write_csv(output_root / "recovered_tp_quality.csv", recovered_rows, quality_fields)
    fp_fields = tuple(additional_fp_rows[0]) if additional_fp_rows else (
        "scene_id", "policy", "detection_id"
    )
    _write_csv(output_root / "additional_false_positives.csv", additional_fp_rows, fp_fields)
    _write_json(
        output_root / "holdout_boundary_diagnostics.json",
        {
            "source_raster_safety": source_safety,
            "recovery_accounting": comparison["recovery_accounting"],
            "policy_c_scene_distribution": policy_c_distribution,
        },
    )
    return comparison


def _defaults() -> dict[str, Path]:
    root = find_project_root()
    dataset = root / "datasets" / "RarePlanes"
    output = root / "outputs" / "boundary_policy_validation" / "train_holdout_12"
    return {
        "metadata": dataset / "metadata" / "RarePlanes_Public_Metadata.csv",
        "imagery": dataset / "train" / "imagery" / "PS-RGB_cog",
        "annotations": dataset / "train" / "annotations" / "geojson_aircraft",
        "model": root / "best.pt",
        "output": output,
        "manifest": output / "selection_manifest.csv",
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = _defaults()
    parser = argparse.ArgumentParser(description="Validate frozen boundary policies on a TRAIN holdout")
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--metadata", type=Path, default=defaults["metadata"])
    select.add_argument("--imagery-dir", type=Path, default=defaults["imagery"])
    select.add_argument("--annotation-dir", type=Path, default=defaults["annotations"])
    select.add_argument("--manifest", type=Path, default=defaults["manifest"])
    run_scene = subparsers.add_parser("run-scene")
    run_scene.add_argument("--scene-id", required=True)
    for target in (run_scene,):
        target.add_argument("--manifest", type=Path, default=defaults["manifest"])
        target.add_argument("--imagery-dir", type=Path, default=defaults["imagery"])
        target.add_argument("--annotation-dir", type=Path, default=defaults["annotations"])
        target.add_argument("--model", type=Path, default=defaults["model"])
        target.add_argument("--output-dir", type=Path, default=defaults["output"])
    run = subparsers.add_parser("run-isolated")
    run.add_argument("--manifest", type=Path, default=defaults["manifest"])
    run.add_argument("--imagery-dir", type=Path, default=defaults["imagery"])
    run.add_argument("--annotation-dir", type=Path, default=defaults["annotations"])
    run.add_argument("--model", type=Path, default=defaults["model"])
    run.add_argument("--output-dir", type=Path, default=defaults["output"])
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--manifest", type=Path, default=defaults["manifest"])
    aggregate.add_argument("--imagery-dir", type=Path, default=defaults["imagery"])
    aggregate.add_argument("--annotation-dir", type=Path, default=defaults["annotations"])
    aggregate.add_argument("--output-dir", type=Path, default=defaults["output"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "select":
        base = load_train_scene_records(args.metadata, args.imagery_dir, args.annotation_dir)
        records = build_holdout_records(base, args.annotation_dir)
        selected = select_location_disjoint_holdout(records)
        write_manifest(selected, args.manifest)
        print(json.dumps([asdict(record) for record in selected], indent=2, ensure_ascii=False))
        return 0
    if args.command == "run-scene":
        records = {record.scene_id: record for record in read_manifest(args.manifest)}
        if args.scene_id not in records:
            raise HoldoutValidationError(f"Scene is absent from frozen manifest: {args.scene_id}")
        result = run_one_scene(
            records[args.scene_id],
            args.imagery_dir,
            args.annotation_dir,
            args.model,
            args.output_dir,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "run-isolated":
        attempts = run_scenes_isolated(
            args.manifest,
            args.imagery_dir,
            args.annotation_dir,
            args.model,
            args.output_dir,
        )
        print(json.dumps(attempts, indent=2, ensure_ascii=False))
        return 0
    result = aggregate_validation(
        args.manifest, args.imagery_dir, args.annotation_dir, args.output_dir
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
