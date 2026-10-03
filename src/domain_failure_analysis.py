"""Read-only diagnostics for RarePlanes TRAIN domain and scale failures.

The analysis consumes frozen development and holdout artifacts.  It never
changes inference settings or official metrics and reads rasters only through
bounded windows or a small deterministic overview sample.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

try:
    from .batch_evaluation import percentile
    from .geospatial_evaluation import (
        _load_ground_truth,
        make_crs84_to_raster_transformer,
    )
    from .geospatial_tiling import generate_tile_windows
    from .project_config import find_project_root
except ImportError:  # Support direct execution.
    from batch_evaluation import percentile
    from geospatial_evaluation import _load_ground_truth, make_crs84_to_raster_transformer
    from geospatial_tiling import generate_tile_windows
    from project_config import find_project_root


SUBSETS = ("development", "holdout")
EXTREME_FAILURE_SCENES = (
    "96_10400100096C2500",
    "88_1040010019A43D00",
    "88_104001002B095600",
)
SIZE_FIELDS = ("major_axis_pixels", "minor_axis_pixels", "pixel_obb_area")
SCALE_COMPARISON_FIELDS = (
    "major_axis_pixels",
    "minor_axis_pixels",
    "pixel_obb_area",
    "physical_wingspan",
    "physical_length",
)
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
SIZE_BIN_LABELS = ("Q1_smallest", "Q2", "Q3", "Q4_largest")


class DomainAnalysisError(RuntimeError):
    """Raised when frozen TRAIN artifacts are missing or inconsistent."""


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def describe(values: Iterable[float]) -> dict[str, float | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {name: None for name in ("mean", "median", "p10", "p25", "p75", "p90")}
    return {
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "p10": percentile(finite, 0.10),
        "p25": percentile(finite, 0.25),
        "p75": percentile(finite, 0.75),
        "p90": percentile(finite, 0.90),
    }


def quantile_summary(values: Sequence[float]) -> dict[str, float | None]:
    return {
        name: percentile(values, quantile)
        for name, quantile in zip(("p10", "p25", "median", "p75", "p90"), QUANTILES, strict=True)
    }


def minimum_rectangle_dimensions(polygon: Any) -> tuple[float, float, float, float]:
    """Return major, minor, area, and aspect ratio for a valid pixel OBB."""

    coordinates = list(polygon.exterior.coords)
    if len(coordinates) < 5 or polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise DomainAnalysisError("Invalid minimum-area rectangle")
    lengths = [
        math.dist(coordinates[index], coordinates[index + 1])
        for index in range(4)
    ]
    major = max(lengths)
    minor = min(length for length in lengths if length > 0)
    return major, minor, float(polygon.area), major / minor


def quartile_edges(values: Sequence[float]) -> tuple[float, float, float]:
    if not values:
        raise ValueError("Cannot calculate quartiles from an empty sequence")
    return (
        float(percentile(values, 0.25)),
        float(percentile(values, 0.50)),
        float(percentile(values, 0.75)),
    )


def assign_size_bin(value: float, edges: Sequence[float]) -> str:
    """Assign equal-edge values to the higher deterministic quartile bin."""

    if len(edges) != 3 or any(edges[index] > edges[index + 1] for index in range(2)):
        raise ValueError("Expected three sorted quartile edges")
    return SIZE_BIN_LABELS[bisect_right(tuple(float(edge) for edge in edges), float(value))]


def object_crop_bounds(
    polygon_bounds: Sequence[float], raster_width: int, raster_height: int, major_axis: float
) -> tuple[int, int, int, int]:
    """Return a clipped contextual crop using 1.5 major axes or eight pixels."""

    left, top, right, bottom = (float(value) for value in polygon_bounds)
    margin = max(8.0, 1.5 * float(major_axis))
    col0 = max(0, math.floor(left - margin))
    row0 = max(0, math.floor(top - margin))
    col1 = min(int(raster_width), math.ceil(right + margin))
    row1 = min(int(raster_height), math.ceil(bottom + margin))
    if col1 <= col0 or row1 <= row0:
        raise DomainAnalysisError("Object crop is empty after clipping")
    return col0, row0, col1, row1


def foreground_background_statistics(
    rgb: Any, foreground_mask: Any
) -> dict[str, float | None]:
    """Calculate simple normalized RGB/luminance foreground-background contrast."""

    import numpy as np

    array = np.asarray(rgb, dtype=np.float64)
    mask = np.asarray(foreground_mask, dtype=bool)
    if array.ndim != 3 or array.shape[0] != 3 or mask.shape != array.shape[1:]:
        raise ValueError("Expected CHW RGB and matching 2-D foreground mask")
    valid = np.all(np.isfinite(array), axis=0)
    foreground = mask & valid
    background = (~mask) & valid
    if not foreground.any() or not background.any():
        return {
            "foreground_luminance_mean": None,
            "background_luminance_mean": None,
            "signed_luminance_contrast": None,
            "absolute_luminance_contrast": None,
            "rgb_color_distance": None,
            "foreground_luminance_std": None,
            "background_luminance_std": None,
        }
    normalized = array / 255.0
    luminance = (
        0.2126 * normalized[0] + 0.7152 * normalized[1] + 0.0722 * normalized[2]
    )
    fg_lum = luminance[foreground]
    bg_lum = luminance[background]
    fg_rgb = normalized[:, foreground].mean(axis=1)
    bg_rgb = normalized[:, background].mean(axis=1)
    signed = float(fg_lum.mean() - bg_lum.mean())
    return {
        "foreground_luminance_mean": float(fg_lum.mean()),
        "background_luminance_mean": float(bg_lum.mean()),
        "signed_luminance_contrast": signed,
        "absolute_luminance_contrast": abs(signed),
        "rgb_color_distance": float(np.linalg.norm(fg_rgb - bg_rgb)),
        "foreground_luminance_std": float(fg_lum.std()),
        "background_luminance_std": float(bg_lum.std()),
    }


def nearest_neighbor_distances(points: Sequence[tuple[float, float]]) -> list[float | None]:
    if len(points) < 2:
        return [None] * len(points)
    return [
        min(math.dist(point, other) for other_index, other in enumerate(points) if index != other_index)
        for index, point in enumerate(points)
    ]


def _correlation(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) < 2 or len(first) != len(second):
        return None
    first_mean = statistics.fmean(first)
    second_mean = statistics.fmean(second)
    numerator = sum((x - first_mean) * (y - second_mean) for x, y in zip(first, second, strict=True))
    denominator = math.sqrt(
        sum((x - first_mean) ** 2 for x in first)
        * sum((y - second_mean) ** 2 for y in second)
    )
    return numerator / denominator if denominator else None


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        rank = (cursor + end - 1) / 2.0 + 1.0
        for position in range(cursor, end):
            ranks[ordered[position][0]] = rank
        cursor = end
    return ranks


def correlations(first: Sequence[float], second: Sequence[float]) -> dict[str, Any]:
    return {
        "n": len(first),
        "pearson": _correlation(first, second),
        "spearman": _correlation(_ranks(first), _ranks(second)),
    }


def _manifest_records(path: Path, subset: str) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    if len(rows) != 12:
        raise DomainAnalysisError(f"Expected 12 {subset} manifest rows")
    if any(row.get("split", "train") != "train" for row in rows):
        raise DomainAnalysisError(f"{subset} manifest contains non-TRAIN data")
    records = []
    for row in rows:
        records.append(
            {
                "dataset_subset": subset,
                "scene_id": row["scene_id"],
                "loc_id": int(row["loc_id"]),
                "country": row.get("country", ""),
                "state_province": row.get("state_province", ""),
                "air_field": row.get("air_field", ""),
                "weather": row.get("weather", ""),
                "cloud_cover_percentage": float(row.get("cloud_cover_percentage") or 0.0),
                "off_nadir_angle": float(row.get("off_nadir_angle") or 0.0),
                "sensor": row.get("sensor", ""),
                "collection_time_start": row.get("collection_time_start", ""),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "gt_aircraft_count": int(row["gt_aircraft_count"]),
            }
        )
    return records


def load_analysis_scenes(
    development_manifest: Path, holdout_manifest: Path
) -> tuple[dict[str, Any], ...]:
    records = _manifest_records(development_manifest, "development")
    records.extend(_manifest_records(holdout_manifest, "holdout"))
    ids = [record["scene_id"] for record in records]
    if len(ids) != 24 or len(set(ids)) != 24:
        raise DomainAnalysisError("Expected 24 unique TRAIN scenes with preserved provenance")
    return tuple(records)


def scene_radiometry(dataset: Any, max_sample_dimension: int = 512) -> dict[str, Any]:
    """Read a deterministic whole-scene average-resampled overview, at most 512²."""

    import numpy as np
    from rasterio.enums import Resampling

    scale = min(1.0, max_sample_dimension / max(dataset.width, dataset.height))
    sample_width = max(1, round(dataset.width * scale))
    sample_height = max(1, round(dataset.height * scale))
    array = dataset.read(
        out_shape=(dataset.count, sample_height, sample_width),
        masked=True,
        resampling=Resampling.average,
    ).astype(np.float64)
    values = array.filled(np.nan)
    valid = np.all(np.isfinite(values), axis=0)
    band_rows = {}
    for band in range(dataset.count):
        band_values = values[band][valid]
        band_rows[f"band_{band + 1}"] = {
            "mean": float(np.mean(band_values)),
            "median": float(np.median(band_values)),
            "std": float(np.std(band_values)),
            "p05": float(np.percentile(band_values, 5)),
            "p25": float(np.percentile(band_values, 25)),
            "p75": float(np.percentile(band_values, 75)),
            "p95": float(np.percentile(band_values, 95)),
        }
    normalized = values / 255.0
    luminance = 0.2126 * normalized[0] + 0.7152 * normalized[1] + 0.0722 * normalized[2]
    safe_normalized = normalized.copy()
    safe_normalized[:, ~valid] = 0.0
    maximum = np.max(safe_normalized, axis=0)
    minimum = np.min(safe_normalized, axis=0)
    saturation = np.divide(maximum - minimum, maximum, out=np.zeros_like(maximum), where=maximum > 0)
    lum_values = luminance[valid]
    saturation_values = saturation[valid]
    horizontal = np.abs(luminance[:, 1:] - luminance[:, :-1])
    vertical = np.abs(luminance[1:, :] - luminance[:-1, :])
    local_contrast = float(np.nanmean([np.nanmean(horizontal), np.nanmean(vertical)]))
    return {
        "sampling_definition": "whole-scene average resampling to max dimension 512",
        "sample_width": sample_width,
        "sample_height": sample_height,
        "bands": band_rows,
        "luminance_mean": float(np.mean(lum_values)),
        "luminance_std": float(np.std(lum_values)),
        "luminance_p05": float(np.percentile(lum_values, 5)),
        "luminance_p95": float(np.percentile(lum_values, 95)),
        "saturation_mean": float(np.mean(saturation_values)),
        "saturation_median": float(np.median(saturation_values)),
        "very_bright_fraction": float(np.mean(lum_values >= 0.90)),
        "very_dark_fraction": float(np.mean(lum_values <= 0.10)),
        "approximate_local_contrast": local_contrast,
    }


def streamed_band_ranges(dataset: Any) -> dict[str, dict[str, float | None]]:
    """Calculate exact valid-pixel band ranges using native block windows."""

    import numpy as np

    minima = [math.inf] * dataset.count
    maxima = [-math.inf] * dataset.count
    for _, window in dataset.block_windows(1):
        block = dataset.read(window=window, masked=True)
        for index in range(dataset.count):
            values = block[index].compressed()
            if values.size:
                minima[index] = min(minima[index], float(np.min(values)))
                maxima[index] = max(maxima[index], float(np.max(values)))
    return {
        f"band_{index + 1}": {
            "min": minima[index] if math.isfinite(minima[index]) else None,
            "max": maxima[index] if math.isfinite(maxima[index]) else None,
        }
        for index in range(dataset.count)
    }


def _object_contrast(dataset: Any, polygon: Any, major_axis: float) -> dict[str, Any]:
    import numpy as np
    from affine import Affine
    from rasterio.features import geometry_mask
    from rasterio.windows import Window
    from shapely.geometry import mapping

    col0, row0, col1, row1 = object_crop_bounds(
        polygon.bounds, dataset.width, dataset.height, major_axis
    )
    window = Window(col0, row0, col1 - col0, row1 - row0)
    rgb = dataset.read((1, 2, 3), window=window).astype(np.float64)
    mask = geometry_mask(
        [mapping(polygon)],
        out_shape=(row1 - row0, col1 - col0),
        # The GT polygon is in continuous source-pixel coordinates, so crop
        # pixels map to that same coordinate space through a pure translation.
        transform=Affine.translation(col0, row0),
        invert=True,
        all_touched=False,
    )
    return {
        "crop_col0": col0,
        "crop_row0": row0,
        "crop_col1": col1,
        "crop_row1": row1,
        **foreground_background_statistics(rgb, mask),
    }


def _evaluation_maps(evaluation_dir: Path) -> tuple[dict[int, dict[str, str]], dict[int, dict[str, str]]]:
    matches = {int(row["gt_index"]): row for row in _read_csv(evaluation_dir / "matches.csv")}
    false_negatives = {
        int(row["gt_index"]): row for row in _read_csv(evaluation_dir / "false_negatives.csv")
    }
    if set(matches) & set(false_negatives):
        raise DomainAnalysisError("GT index appears as both TP and FN")
    return matches, false_negatives


def _failure_category(
    gt_index: int,
    false_negative: Mapping[str, str],
    boundary_records: Sequence[Mapping[str, Any]],
    nms_records: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, bool]]:
    nearest_iou = float(false_negative["nearest_prediction_iou"]) if false_negative.get("nearest_prediction_iou") not in (None, "", "None") else 0.0
    flags = {
        "plane_final_nearby": nearest_iou > 0.0,
        "boundary_candidate": any(
            item.get("best_current_fn_gt_index") == gt_index
            and item.get("would_independently_match_current_fn") is True
            for item in boundary_records
        ),
        "nms_candidate": any(
            item.get("best_current_fn_gt_index") == gt_index
            and item.get("would_independently_match_current_fn") is True
            for item in nms_records
        ),
        "non_plane_geometry_available": False,
    }
    active = sum(flags[key] for key in ("plane_final_nearby", "boundary_candidate", "nms_candidate"))
    if active > 1:
        category = "F_ambiguous_multiple_plane_evidence"
    elif flags["boundary_candidate"]:
        category = "D_boundary_rejected_plane"
    elif flags["nms_candidate"]:
        category = "E_global_nms_removed_plane"
    elif flags["plane_final_nearby"]:
        category = "B_final_plane_below_match_iou_or_assignment_conflict"
    else:
        category = "A_no_retained_plane_evidence_nearby"
    return category, flags


def _scene_root(record: Mapping[str, Any], development_root: Path, holdout_root: Path) -> Path:
    root = development_root if record["dataset_subset"] == "development" else holdout_root
    return root / "scenes" / str(record["scene_id"])


def analyze_scene(
    record: Mapping[str, Any],
    development_root: Path,
    holdout_root: Path,
    imagery_dir: Path,
    annotation_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    import rasterio

    scene_id = str(record["scene_id"])
    scene_root = _scene_root(record, development_root, holdout_root)
    inference_dir = scene_root / "inference"
    evaluation_dir = scene_root / "evaluation"
    summary = _read_json(inference_dir / "inference_summary.json")
    evaluation = _read_json(evaluation_dir / "evaluation_summary.json")
    boundary = _read_json(evaluation_dir / "boundary_rejection_diagnostics.json")
    nms = _read_json(evaluation_dir / "nms_diagnostics.json")
    matches, false_negatives = _evaluation_maps(evaluation_dir)
    raster_path = imagery_dir / f"{scene_id}.tif"
    annotation_path = annotation_dir / f"{scene_id}.geojson"

    with rasterio.open(raster_path) as dataset:
        truths = _load_ground_truth(
            _read_json(annotation_path),
            scene_id,
            dataset.transform,
            make_crs84_to_raster_transformer(dataset.crs),
        )
        expected_tiles = len(
            tuple(
                generate_tile_windows(
                    scene_id,
                    dataset.width,
                    dataset.height,
                    int(summary["tile_size"]),
                    float(summary["tile_overlap"]),
                )
            )
        )
        radiometry = scene_radiometry(dataset)
        centroids = [
            (float(truth.obb_pixel_polygon.centroid.x), float(truth.obb_pixel_polygon.centroid.y))
            for truth in truths
        ]
        nearest_distances = nearest_neighbor_distances(centroids)
        objects = []
        failures = []
        for truth, nearest_distance in zip(truths, nearest_distances, strict=True):
            major, minor, area, aspect = minimum_rectangle_dimensions(truth.obb_pixel_polygon)
            center = truth.obb_pixel_polygon.centroid
            properties = truth.properties
            match = matches.get(truth.gt_index)
            false_negative = false_negatives.get(truth.gt_index)
            if (match is None) == (false_negative is None):
                raise DomainAnalysisError(f"GT outcome is not exactly one of TP/FN: {scene_id}:{truth.gt_index}")
            outcome = "TP" if match is not None else "FN"
            row = {
                "dataset_subset": record["dataset_subset"],
                "scene_id": scene_id,
                "loc_id": record["loc_id"],
                "gt_index": truth.gt_index,
                "role": properties.get("role"),
                "faa_wingspan_class": properties.get("faa_wingspan_class"),
                "physical_length": properties.get("length"),
                "physical_wingspan": properties.get("wingspan"),
                "major_axis_pixels": major,
                "minor_axis_pixels": minor,
                "pixel_obb_width": major,
                "pixel_obb_height": minor,
                "pixel_obb_area": area,
                "original_diamond_area": float(truth.diamond_pixel_polygon.area),
                "aspect_ratio": aspect,
                "centroid_col": float(center.x),
                "centroid_row": float(center.y),
                "nearest_raster_boundary_pixels": min(
                    float(center.x), float(center.y), dataset.width - float(center.x), dataset.height - float(center.y)
                ),
                "nearest_gt_centroid_pixels": nearest_distance,
                "local_gt_count_within_256px": sum(
                    math.dist((center.x, center.y), other) <= 256.0 and other != (center.x, center.y)
                    for other in centroids
                ),
                "outcome": outcome,
                "matched_tp": outcome == "TP",
                "prediction_confidence": float(match["confidence"]) if match else None,
                "normalized_obb_iou": float(match["normalized_obb_iou"]) if match else None,
                "center_error_pixels": float(match["center_error_pixels"]) if match else None,
                "center_error_metres": float(match["center_error_metres"]) if match else None,
                "nearest_final_prediction_iou": (
                    float(false_negative["nearest_prediction_iou"])
                    if false_negative and false_negative.get("nearest_prediction_iou") not in (None, "", "None")
                    else None
                ),
                "nearest_final_prediction_confidence": (
                    float(false_negative["nearest_prediction_confidence"])
                    if false_negative and false_negative.get("nearest_prediction_confidence") not in (None, "", "None")
                    else None
                ),
                "weather": record["weather"],
                "cloud_cover_percentage": record["cloud_cover_percentage"],
                "off_nadir_angle": record["off_nadir_angle"],
                "image_width": dataset.width,
                "image_height": dataset.height,
                "country": record["country"],
                "air_field": record["air_field"],
                "sensor": record["sensor"],
                "collection_time_start": record["collection_time_start"],
                **_object_contrast(dataset, truth.obb_pixel_polygon, major),
            }
            objects.append(row)
            if false_negative:
                category, flags = _failure_category(
                    truth.gt_index,
                    false_negative,
                    boundary.get("records", []),
                    nms.get("records", []),
                )
                failures.append(
                    {
                        "dataset_subset": record["dataset_subset"],
                        "scene_id": scene_id,
                        "loc_id": record["loc_id"],
                        "gt_index": truth.gt_index,
                        "primary_category": category,
                        **flags,
                        "nearest_final_prediction_iou": row["nearest_final_prediction_iou"],
                        "nearest_final_prediction_confidence": row["nearest_final_prediction_confidence"],
                    }
                )

        scene = {
            **record,
            "raster_width": dataset.width,
            "raster_height": dataset.height,
            "raster_dtype": ";".join(dataset.dtypes),
            "raster_band_count": dataset.count,
            "raster_colorinterp": ";".join(item.name for item in dataset.colorinterp),
            "raster_nodata": dataset.nodata,
            "raster_crs": str(dataset.crs),
            "raster_transform": tuple(dataset.transform),
            "overview_levels": ";".join(str(value) for value in dataset.overviews(1)),
            "actual_tile_count": int(summary["tile_count"]),
            "expected_tile_count": expected_tiles,
            "tile_coverage_complete": expected_tiles == int(summary["tile_count"]),
            "all_class_raw_detections": int(summary["all_class_raw_detections"]),
            "raw_plane_candidates": int(summary["raw_target_candidates"]),
            "final_plane_predictions": int(summary["final_target_count"]),
            "raw_class_counts": json.dumps(summary.get("raw_class_counts", {}), sort_keys=True),
            "gt_count": int(evaluation["ground_truth_count"]),
            "tp": int(evaluation["tp"]),
            "fp": int(evaluation["fp"]),
            "fn": int(evaluation["fn"]),
            "precision": float(evaluation["precision"]),
            "recall": float(evaluation["recall"]),
            "f1": float(evaluation["f1"]),
            "gt_per_megapixel": len(truths) / (dataset.width * dataset.height / 1_000_000.0),
            "nearest_neighbor_mean_pixels": statistics.fmean(value for value in nearest_distances if value is not None) if len(truths) > 1 else None,
            "nearest_neighbor_median_pixels": statistics.median(value for value in nearest_distances if value is not None) if len(truths) > 1 else None,
            **{key: value for key, value in radiometry.items() if key != "bands"},
        }
        if scene_id in EXTREME_FAILURE_SCENES:
            scene["exact_band_ranges"] = json.dumps(
                streamed_band_ranges(dataset), sort_keys=True
            )
        else:
            scene["exact_band_ranges"] = ""
        for band, values in radiometry["bands"].items():
            for metric, value in values.items():
                scene[f"{band}_{metric}"] = value
    return objects, scene, failures


def grouped_recall(
    objects: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in objects:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for group_key, rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        tp = sum(row["outcome"] == "TP" for row in rows)
        output.append(
            {
                **dict(zip(keys, group_key, strict=True)),
                "gt_count": len(rows),
                "tp": tp,
                "fn": len(rows) - tp,
                "recall": tp / len(rows),
            }
        )
    return output


def _summary_by_outcome(objects: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for outcome in ("TP", "FN"):
        selected = [row for row in objects if row["outcome"] == outcome]
        for field in SCALE_COMPARISON_FIELDS:
            values = [float(row[field]) for row in selected if row[field] not in (None, "")]
            description = describe(values)
            rows.append(
                {
                    "outcome": outcome,
                    "metric": field,
                    "count": len(values),
                    "mean": description["mean"],
                    "median": description["median"],
                    "p25": description["p25"],
                    "p75": description["p75"],
                }
            )
    return rows


def _contrast_summary(objects: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in objects:
        weather_group = "snow" if row["weather"] == "Snow" else "non_snow"
        groups[(str(row["outcome"]), weather_group)].append(row)
    output = []
    fields = (
        "absolute_luminance_contrast",
        "rgb_color_distance",
        "foreground_luminance_std",
        "background_luminance_std",
    )
    for (outcome, weather_group), rows in sorted(groups.items()):
        for field in fields:
            values = [float(row[field]) for row in rows if row[field] is not None]
            summary = describe(values)
            output.append(
                {
                    "outcome": outcome,
                    "weather_group": weather_group,
                    "metric": field,
                    "count": len(values),
                    "mean": summary["mean"],
                    "median": summary["median"],
                    "p25": summary["p25"],
                    "p75": summary["p75"],
                }
            )
    return output


def _off_nadir_summary(scenes: Sequence[Mapping[str, Any]], objects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    off = [float(scene["off_nadir_angle"]) for scene in scenes]
    recall = [float(scene["recall"]) for scene in scenes]
    scene_major = [
        statistics.median(float(row["major_axis_pixels"]) for row in objects if row["scene_id"] == scene["scene_id"])
        for scene in scenes
    ]
    center_error = []
    center_off = []
    for scene in scenes:
        values = [
            float(row["center_error_pixels"])
            for row in objects
            if row["scene_id"] == scene["scene_id"] and row["outcome"] == "TP"
        ]
        if values:
            center_error.append(statistics.mean(values))
            center_off.append(float(scene["off_nadir_angle"]))
    return {
        "recall_vs_off_nadir": correlations(off, recall),
        "median_major_axis_vs_off_nadir": correlations(off, scene_major),
        "matched_center_error_vs_off_nadir": correlations(center_off, center_error),
        "interpretation_limit": "Scene-level n=24 is small; correlations are descriptive, not causal.",
    }


def _scene_domain_summary(
    scenes: Sequence[Mapping[str, Any]], objects: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    descriptors = (
        "luminance_mean",
        "luminance_std",
        "saturation_mean",
        "very_bright_fraction",
        "very_dark_fraction",
        "approximate_local_contrast",
    )
    weather_groups = {}
    for label, selected in (
        ("snow", [scene for scene in scenes if scene["weather"] == "Snow"]),
        ("non_snow", [scene for scene in scenes if scene["weather"] != "Snow"]),
    ):
        gt = sum(int(scene["gt_count"]) for scene in selected)
        tp = sum(int(scene["tp"]) for scene in selected)
        weather_groups[label] = {
            "scene_count": len(selected),
            "gt": gt,
            "tp": tp,
            "fn": gt - tp,
            "micro_recall": tp / gt if gt else 0.0,
            **{
                f"{field}_scene_mean": statistics.fmean(float(scene[field]) for scene in selected)
                for field in descriptors
            },
        }
    recall = [float(scene["recall"]) for scene in scenes]
    radiometry_correlations = {
        f"recall_vs_{field}": correlations(
            [float(scene[field]) for scene in scenes], recall
        )
        for field in descriptors
    }
    density = [float(scene["gt_per_megapixel"]) for scene in scenes]
    nn_scenes = [scene for scene in scenes if scene["nearest_neighbor_median_pixels"] is not None]
    object_crowding = {}
    for outcome in ("TP", "FN"):
        selected = [row for row in objects if row["outcome"] == outcome]
        object_crowding[outcome] = {
            "nearest_neighbor_pixels": describe(
                float(row["nearest_gt_centroid_pixels"])
                for row in selected
                if row["nearest_gt_centroid_pixels"] is not None
            ),
            "local_gt_count_within_256px": describe(
                float(row["local_gt_count_within_256px"]) for row in selected
            ),
        }
    return {
        "weather_groups": weather_groups,
        "radiometry_correlations": radiometry_correlations,
        "density": {
            "recall_vs_gt_per_megapixel": correlations(density, recall),
            "recall_vs_median_nearest_neighbor_pixels": correlations(
                [float(scene["nearest_neighbor_median_pixels"]) for scene in nn_scenes],
                [float(scene["recall"]) for scene in nn_scenes],
            ),
            "object_level_tp_vs_fn": object_crowding,
        },
    }


def run_analysis(
    development_manifest: Path,
    holdout_manifest: Path,
    development_root: Path,
    holdout_root: Path,
    imagery_dir: Path,
    annotation_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    records = load_analysis_scenes(development_manifest, holdout_manifest)
    objects: list[dict[str, Any]] = []
    scenes = []
    failures = []
    for index, record in enumerate(records, 1):
        print(f"DOMAIN ANALYSIS {index}/24: {record['dataset_subset']} {record['scene_id']}", flush=True)
        scene_objects, scene, scene_failures = analyze_scene(
            record, development_root, holdout_root, imagery_dir, annotation_dir
        )
        objects.extend(scene_objects)
        scenes.append(scene)
        failures.extend(scene_failures)

    size_distributions = {
        field: quantile_summary([float(row[field]) for row in objects]) for field in SIZE_FIELDS
    }
    edges = quartile_edges([float(row["major_axis_pixels"]) for row in objects])
    for row in objects:
        row["major_axis_size_bin"] = assign_size_bin(float(row["major_axis_pixels"]), edges)

    size_recall = grouped_recall(objects, ("major_axis_size_bin",))
    faa_recall = grouped_recall(objects, ("faa_wingspan_class",))
    faa_size_recall = grouped_recall(objects, ("faa_wingspan_class", "major_axis_size_bin"))
    weather_size = grouped_recall(objects, ("weather", "major_axis_size_bin"))
    snow_size = grouped_recall(
        [
            {**row, "snow_group": "snow" if row["weather"] == "Snow" else "non_snow"}
            for row in objects
        ],
        ("snow_group", "major_axis_size_bin"),
    )
    faa_weather_size = grouped_recall(
        [
            {
                **row,
                "faa_group": (
                    "class_1" if str(row["faa_wingspan_class"]) == "1" else "classes_2_plus"
                ),
                "snow_group": "snow" if row["weather"] == "Snow" else "non_snow",
            }
            for row in objects
        ],
        ("faa_group", "snow_group", "major_axis_size_bin"),
    )
    outcome_scale = _summary_by_outcome(objects)
    contrast_summary = _contrast_summary(objects)
    failure_counts = dict(sorted(Counter(row["primary_category"] for row in failures).items()))

    class_one_bins = [row for row in faa_size_recall if str(row["faa_wingspan_class"]) == "1"]
    other_bins = [
        row
        for row in grouped_recall(
            [
                {**item, "faa_group": "class_1" if str(item["faa_wingspan_class"]) == "1" else "classes_2_plus"}
                for item in objects
            ],
            ("faa_group", "major_axis_size_bin"),
        )
        if row["faa_group"] == "classes_2_plus"
    ]

    extreme = {}
    for scene_id in EXTREME_FAILURE_SCENES:
        scene = next(item for item in scenes if item["scene_id"] == scene_id)
        candidate_controls = [
            item
            for item in scenes
            if item["scene_id"] not in EXTREME_FAILURE_SCENES and item["recall"] >= 0.5
        ]
        scene_major = statistics.median(
            float(row["major_axis_pixels"]) for row in objects if row["scene_id"] == scene_id
        )
        control = min(
            candidate_controls,
            key=lambda item: abs(
                statistics.median(
                    float(row["major_axis_pixels"])
                    for row in objects
                    if row["scene_id"] == item["scene_id"]
                )
                - scene_major
            ),
        )
        extreme[scene_id] = {
            "imagery_integrity": {
                "opens": True,
                "bands": scene["raster_band_count"],
                "colorinterp": scene["raster_colorinterp"],
                "dtype": scene["raster_dtype"],
                "nodata": scene["raster_nodata"],
                "crs": scene["raster_crs"],
                "transform": scene["raster_transform"],
                "overview_levels": scene["overview_levels"],
            },
            "tile_coverage_complete": scene["tile_coverage_complete"],
            "tile_count": scene["actual_tile_count"],
            "all_class_raw_detections": scene["all_class_raw_detections"],
            "raw_plane_candidates": scene["raw_plane_candidates"],
            "final_plane_predictions": scene["final_plane_predictions"],
            "raw_class_counts": json.loads(scene["raw_class_counts"]),
            "exact_band_ranges": json.loads(scene["exact_band_ranges"]),
            "recall": scene["recall"],
            "median_major_axis_pixels": scene_major,
            "radiometry": {
                key: scene[key]
                for key in (
                    "luminance_mean",
                    "luminance_std",
                    "saturation_mean",
                    "very_bright_fraction",
                    "very_dark_fraction",
                    "approximate_local_contrast",
                )
            },
            "comparable_successful_scene": control["scene_id"],
        }

    summary = {
        "scope": "24 frozen RarePlanes TRAIN scenes; development and location-disjoint holdout retained separately",
        "scene_counts": dict(sorted(Counter(row["dataset_subset"] for row in scenes).items())),
        "total_gt": len(objects),
        "total_tp": sum(row["outcome"] == "TP" for row in objects),
        "total_fn": sum(row["outcome"] == "FN" for row in objects),
        "size_distributions": size_distributions,
        "major_axis_quartile_edges": edges,
        "size_bin_recall": size_recall,
        "faa_class_recall": faa_recall,
        "faa_class_size_recall": faa_size_recall,
        "class_1_size_bins": class_one_bins,
        "classes_2_plus_size_bins": other_bins,
        "weather_size_recall": weather_size,
        "snow_vs_non_snow_size_recall": snow_size,
        "faa_weather_size_recall": faa_weather_size,
        "failure_category_counts": failure_counts,
        "non_plane_confusion": {
            "object_level_reconstructable": False,
            "reason": "Frozen artifacts retain per-scene raw class counts but not raw non-plane geometry.",
            "scene_raw_class_counts_available": True,
        },
        "off_nadir": _off_nadir_summary(scenes, objects),
        "scene_domain": _scene_domain_summary(scenes, objects),
        "extreme_failure_scenes": extreme,
        "low_confidence_probe": {
            "run": False,
            "reason": "Optional probe deferred; no new inference was needed for the requested frozen-artifact diagnosis.",
        },
        "radiometry_definitions": {
            "sampling": "whole-scene average resampling to maximum dimension 512",
            "luminance": "0.2126 R + 0.7152 G + 0.0722 B after uint8/255 normalization",
            "very_bright": "luminance >= 0.90",
            "very_dark": "luminance <= 0.10",
            "local_contrast": "mean absolute adjacent-pixel luminance difference in overview sample",
        },
        "object_contrast_definition": (
            "GT OBB foreground versus all valid pixels outside the OBB in a clipped crop; "
            "crop margin=max(8 pixels, 1.5*GT major axis)."
        ),
    }

    _write_csv(output_dir / "object_level_metrics.csv", objects, tuple(objects[0]))
    _write_csv(output_dir / "scene_level_metrics.csv", scenes, tuple(scenes[0]))
    _write_csv(output_dir / "size_bin_recall.csv", size_recall, tuple(size_recall[0]))
    _write_csv(output_dir / "weather_size_recall.csv", weather_size, tuple(weather_size[0]))
    _write_csv(output_dir / "tp_fn_scale_summary.csv", outcome_scale, tuple(outcome_scale[0]))
    radiometry_fields = [
        key for key in scenes[0] if key.startswith("band_") or key in {
            "dataset_subset", "scene_id", "loc_id", "weather", "recall", "luminance_mean",
            "luminance_std", "luminance_p05", "luminance_p95", "saturation_mean",
            "saturation_median", "very_bright_fraction", "very_dark_fraction",
            "approximate_local_contrast",
        }
    ]
    _write_csv(output_dir / "radiometry_summary.csv", scenes, radiometry_fields)
    _write_csv(output_dir / "object_contrast_summary.csv", contrast_summary, tuple(contrast_summary[0]))
    _write_csv(output_dir / "fn_failure_categories.csv", failures, tuple(failures[0]))
    class_rows = [
        {
            "dataset_subset": scene["dataset_subset"],
            "scene_id": scene["scene_id"],
            "loc_id": scene["loc_id"],
            "all_class_raw_detections": scene["all_class_raw_detections"],
            "raw_plane_candidates": scene["raw_plane_candidates"],
            "raw_class_counts": scene["raw_class_counts"],
            "object_level_geometry_available": False,
        }
        for scene in scenes
    ]
    _write_csv(output_dir / "class_confusion_summary.csv", class_rows, tuple(class_rows[0]))
    _write_json(output_dir / "aggregate_summary.json", summary)
    return summary


def _defaults() -> dict[str, Path]:
    root = find_project_root()
    dataset = root / "datasets" / "RarePlanes" / "train"
    development = root / "outputs" / "batch_train_evaluation" / "fixed_baseline_12"
    holdout = root / "outputs" / "boundary_policy_validation" / "train_holdout_12"
    return {
        "development_manifest": development / "selection_manifest.csv",
        "holdout_manifest": holdout / "selection_manifest.csv",
        "development_root": development,
        "holdout_root": holdout,
        "imagery": dataset / "imagery" / "PS-RGB_cog",
        "annotations": dataset / "annotations" / "geojson_aircraft",
        "output": root / "outputs" / "domain_failure_analysis" / "train_24",
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = _defaults()
    parser = argparse.ArgumentParser(description="Diagnose fixed TRAIN domain and scale failures")
    parser.add_argument("--development-manifest", type=Path, default=defaults["development_manifest"])
    parser.add_argument("--holdout-manifest", type=Path, default=defaults["holdout_manifest"])
    parser.add_argument("--development-root", type=Path, default=defaults["development_root"])
    parser.add_argument("--holdout-root", type=Path, default=defaults["holdout_root"])
    parser.add_argument("--imagery-dir", type=Path, default=defaults["imagery"])
    parser.add_argument("--annotation-dir", type=Path, default=defaults["annotations"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_analysis(
        args.development_manifest,
        args.holdout_manifest,
        args.development_root,
        args.holdout_root,
        args.imagery_dir,
        args.annotation_dir,
        args.output_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
