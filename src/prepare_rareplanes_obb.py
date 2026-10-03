"""Prepare a leakage-safe RarePlanes TRAIN-only Ultralytics OBB dataset.

Official aircraft diamonds are preserved in provenance and converted to
minimum-area rectangles for one-class (``0 = plane``) OBB training.  Split
selection uses only TRAIN metadata/annotations and excludes every previously
analysed location from the fresh validation-location candidate pool.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

try:
    from .geospatial_evaluation import _load_ground_truth, make_crs84_to_raster_transformer
    from .geospatial_tiling import TileWindow, generate_tile_windows, read_tile, stride_from_overlap
    from .project_config import find_project_root
except ImportError:  # Support direct execution from src/.
    from geospatial_evaluation import _load_ground_truth, make_crs84_to_raster_transformer
    from geospatial_tiling import TileWindow, generate_tile_windows, read_tile, stride_from_overlap
    from project_config import find_project_root


TILE_SIZE = 1024
TILE_OVERLAP = 0.20
VALIDATION_FRACTION = 0.18
SPLIT_SEED = 20261001
BACKGROUND_TO_POSITIVE_RATIO = 0.25
FULL_VISIBILITY_TOLERANCE = 1e-7
OUTPUT_CLASS_ID = 0
OUTPUT_CLASS_NAME = "plane"
# Frozen union of the 12 development and 12 prior-holdout TRAIN scenes. Two
# development scenes share loc_id 88, so the evidence-based union has 23 IDs.
# This fallback makes preparation reproducible from a clean clone where the
# original ignored diagnostic output manifests are intentionally absent.
ANALYZED_LOCATION_IDS = frozenset(
    {
        1, 2, 3, 4, 9, 14, 18, 24, 30, 33, 51, 59,
        86, 88, 90, 96, 103, 111, 112, 117, 118, 125, 130,
    }
)


class PreparationError(RuntimeError):
    """Raised when preparation would violate the frozen dataset protocol."""


@dataclass(frozen=True)
class SceneRecord:
    scene_id: str
    loc_id: int
    country: str
    state_province: str
    air_field: str
    weather: str
    sensor: str
    collection_time_start: str
    collection_time_end: str
    cloud_cover_percentage: float
    off_nadir: float
    split: str = "train"


@dataclass(frozen=True)
class ObjectRecord:
    scene_id: str
    loc_id: int
    gt_index: int
    faa_class: int | None
    diamond_vertices: tuple[tuple[float, float], ...]
    obb_vertices: tuple[tuple[float, float], ...]
    obb_area: float
    major_axis: float


@dataclass(frozen=True)
class Assignment:
    object_record: ObjectRecord
    tile: TileWindow | None
    visibility_fraction: float
    reason: str


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
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


def _stable_hash(*parts: Any) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_rectangle_vertices(polygon: Any) -> tuple[tuple[float, float], ...]:
    """Return four cyclic corners, clockwise in image coordinates, topmost first."""

    coordinates = [(float(x), float(y)) for x, y in list(polygon.exterior.coords)[:-1]]
    if len(coordinates) != 4 or polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise PreparationError("Expected a valid four-corner minimum-area rectangle")
    signed_twice_area = sum(
        coordinates[index][0] * coordinates[(index + 1) % 4][1]
        - coordinates[(index + 1) % 4][0] * coordinates[index][1]
        for index in range(4)
    )
    # Positive shoelace orientation is visually clockwise because raster y grows downward.
    if signed_twice_area < 0:
        coordinates.reverse()
    start = min(range(4), key=lambda index: (coordinates[index][1], coordinates[index][0]))
    return tuple(coordinates[start:] + coordinates[:start])


def diamond_to_minimum_rectangle(diamond: Any) -> tuple[tuple[float, float], ...]:
    rectangle = diamond.minimum_rotated_rectangle
    return canonical_rectangle_vertices(rectangle)


def rectangle_dimensions(vertices: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    from shapely.geometry import Polygon

    polygon = Polygon(vertices)
    lengths = [
        math.dist(vertices[index], vertices[(index + 1) % 4]) for index in range(4)
    ]
    return max(lengths), min(lengths), float(polygon.area)


def visibility_fraction(vertices: Sequence[Sequence[float]], tile: TileWindow) -> float:
    from shapely.geometry import Polygon, box

    polygon = Polygon(vertices)
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        return 0.0
    footprint = box(
        tile.col_off,
        tile.row_off,
        tile.col_off + tile.width,
        tile.row_off + tile.height,
    )
    return float(polygon.intersection(footprint).area / polygon.area)


def _corners_inside(vertices: Sequence[Sequence[float]], tile: TileWindow) -> bool:
    left, top = tile.col_off, tile.row_off
    right, bottom = left + tile.width, top + tile.height
    epsilon = 1e-8
    return all(
        left - epsilon <= float(x) <= right + epsilon
        and top - epsilon <= float(y) <= bottom + epsilon
        for x, y in vertices
    )


def assign_object_to_tile(
    object_record: ObjectRecord, windows: Sequence[TileWindow]
) -> Assignment:
    """Assign exactly one fully visible owner tile; never clip a training OBB."""

    from shapely.geometry import Polygon

    polygon = Polygon(object_record.obb_vertices)
    centroid = polygon.centroid
    candidates: list[tuple[float, str, TileWindow, float]] = []
    best_visibility = 0.0
    for tile in windows:
        visible = visibility_fraction(object_record.obb_vertices, tile)
        best_visibility = max(best_visibility, visible)
        if visible >= 1.0 - FULL_VISIBILITY_TOLERANCE and _corners_inside(
            object_record.obb_vertices, tile
        ):
            tile_center_x = tile.col_off + tile.width / 2.0
            tile_center_y = tile.row_off + tile.height / 2.0
            distance = (centroid.x - tile_center_x) ** 2 + (centroid.y - tile_center_y) ** 2
            candidates.append((distance, tile.tile_id, tile, visible))
    if not candidates:
        return Assignment(object_record, None, best_visibility, "omitted_no_fully_visible_tile")
    _, _, owner, visible = min(candidates, key=lambda item: (item[0], item[1]))
    return Assignment(
        object_record,
        owner,
        visible,
        "single_owner_fully_visible_closest_tile_center",
    )


def global_to_normalized_tile_label(
    vertices: Sequence[Sequence[float]], tile: TileWindow
) -> tuple[float, ...]:
    """Translate source pixels to an unresized, padded 1024-square chip label."""

    values: list[float] = []
    for x, y in vertices:
        local_x = float(x) - tile.col_off
        local_y = float(y) - tile.row_off
        normalized_x = local_x / tile.tile_size
        normalized_y = local_y / tile.tile_size
        if not math.isfinite(normalized_x) or not math.isfinite(normalized_y):
            raise PreparationError("Non-finite normalized OBB coordinate")
        if not -1e-9 <= normalized_x <= 1.0 + 1e-9 or not -1e-9 <= normalized_y <= 1.0 + 1e-9:
            raise PreparationError("Training OBB coordinate falls outside [0, 1]")
        values.extend((min(1.0, max(0.0, normalized_x)), min(1.0, max(0.0, normalized_y))))
    if len(values) != 8:
        raise PreparationError("Ultralytics OBB labels require exactly four vertices")
    return tuple(values)


def deterministic_background_sample(
    tile_ids: Sequence[str], positive_count: int, split: str
) -> tuple[str, ...]:
    limit = min(len(tile_ids), math.floor(positive_count * BACKGROUND_TO_POSITIVE_RATIO))
    ordered = sorted(tile_ids, key=lambda tile_id: (_stable_hash(SPLIT_SEED, split, tile_id), tile_id))
    return tuple(ordered[:limit])


def _manifest_loc_ids(path: Path) -> set[int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 12 or any(row.get("split", "train").lower() != "train" for row in rows):
        raise PreparationError(f"Expected a 12-scene TRAIN manifest: {path}")
    return {int(row["loc_id"]) for row in rows}


def analyzed_location_ids(development_manifest: Path, holdout_manifest: Path) -> set[int]:
    if not development_manifest.exists() and not holdout_manifest.exists():
        return set(ANALYZED_LOCATION_IDS)
    if not development_manifest.exists() or not holdout_manifest.exists():
        raise PreparationError(
            "Development and prior-holdout manifests must either both exist or both be absent"
        )
    development = _manifest_loc_ids(development_manifest)
    holdout = _manifest_loc_ids(holdout_manifest)
    if development & holdout:
        raise PreparationError("Development and prior holdout location IDs overlap")
    # The 24 analysed scenes represent 23 locations because two development
    # scenes share one loc_id.  Exclude the evidence-based union, not an
    # assumed one-location-per-scene count.
    values = development | holdout
    if len(values) != 23:
        raise PreparationError(
            f"Expected 23 unique locations across 24 analysed TRAIN scenes, found {len(values)}"
        )
    if values != ANALYZED_LOCATION_IDS:
        raise PreparationError("Analysed TRAIN manifests do not match frozen location provenance")
    return values


def load_train_scene_records(metadata_path: Path, imagery_dir: Path) -> tuple[SceneRecord, ...]:
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        metadata = {row["image_id"]: row for row in csv.DictReader(handle)}
    image_ids = sorted(path.stem for path in imagery_dir.glob("*.tif"))
    records = []
    for scene_id in image_ids:
        row = metadata.get(scene_id)
        if row is None:
            raise PreparationError(f"TRAIN raster lacks metadata: {scene_id}")
        if row.get("Train") != "1":
            raise PreparationError(f"Non-TRAIN raster encountered: {scene_id}")
        records.append(
            SceneRecord(
                scene_id=scene_id,
                loc_id=int(row["loc_id"]),
                country=row.get("Country", ""),
                state_province=row.get("State_Province", ""),
                air_field=row.get("Air_Field", ""),
                weather=row.get("Weather", ""),
                sensor=row.get("sensor", ""),
                collection_time_start=row.get("collection_time_start", ""),
                collection_time_end=row.get("collection_time_end", ""),
                cloud_cover_percentage=float(row.get("cloud_cover_percentage") or 0.0),
                off_nadir=float(row.get("off_nadir_max") or 0.0),
            )
        )
    if len(records) != 187:
        raise PreparationError(f"Expected 187 RarePlanes TRAIN rasters, found {len(records)}")
    return tuple(records)


def _normalized(values: Mapping[int, float]) -> dict[int, float]:
    low, high = min(values.values()), max(values.values())
    if high == low:
        return {key: 0.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def _allocate_weather_counts(groups: Mapping[str, Sequence[int]], target: int) -> dict[str, int]:
    total = sum(len(items) for items in groups.values())
    exact = {weather: target * len(items) / total for weather, items in groups.items()}
    allocated = {weather: math.floor(value) for weather, value in exact.items()}
    if target >= len(groups):
        for weather, items in groups.items():
            if items and allocated[weather] == 0:
                allocated[weather] = 1
    while sum(allocated.values()) > target:
        choices = [weather for weather in allocated if allocated[weather] > 1]
        weather = min(choices, key=lambda key: (exact[key] - allocated[key], key))
        allocated[weather] -= 1
    while sum(allocated.values()) < target:
        choices = [weather for weather, items in groups.items() if allocated[weather] < len(items)]
        weather = max(choices, key=lambda key: (exact[key] - allocated[key], key))
        allocated[weather] += 1
    return allocated


def select_validation_locations(
    scenes: Sequence[SceneRecord],
    objects_by_scene: Mapping[str, Sequence[ObjectRecord]],
    excluded_loc_ids: set[int],
) -> tuple[set[int], dict[str, Any]]:
    """Select diverse metadata-only validation locations with a fixed seed."""

    by_loc: dict[int, list[SceneRecord]] = defaultdict(list)
    for scene in scenes:
        by_loc[scene.loc_id].append(scene)
    eligible = sorted(set(by_loc) - set(excluded_loc_ids))
    if not eligible:
        raise PreparationError("No fresh TRAIN locations remain for validation")
    aircraft_counts = {
        loc_id: sum(len(objects_by_scene[scene.scene_id]) for scene in by_loc[loc_id])
        for loc_id in eligible
    }
    class_one_share = {}
    for loc_id in eligible:
        objects = [obj for scene in by_loc[loc_id] for obj in objects_by_scene[scene.scene_id]]
        class_one_share[loc_id] = (
            sum(obj.faa_class == 1 for obj in objects) / len(objects) if objects else 0.0
        )
    off_nadir = {
        loc_id: sum(scene.off_nadir for scene in by_loc[loc_id]) / len(by_loc[loc_id])
        for loc_id in eligible
    }
    log_aircraft = {key: math.log1p(value) for key, value in aircraft_counts.items()}
    normalized_aircraft = _normalized(log_aircraft)
    normalized_class_one = _normalized(class_one_share)
    normalized_off_nadir = _normalized(off_nadir)
    weather_by_loc = {
        loc_id: Counter(scene.weather for scene in by_loc[loc_id]).most_common(1)[0][0]
        for loc_id in eligible
    }
    country_by_loc = {
        loc_id: Counter(scene.country for scene in by_loc[loc_id]).most_common(1)[0][0]
        for loc_id in eligible
    }
    weather_groups: dict[str, list[int]] = defaultdict(list)
    for loc_id in eligible:
        weather_groups[weather_by_loc[loc_id]].append(loc_id)
    target = max(1, round(len(eligible) * VALIDATION_FRACTION))
    allocations = _allocate_weather_counts(weather_groups, target)
    selected: set[int] = set()
    for weather in sorted(weather_groups):
        candidates = sorted(weather_groups[weather])
        needed = allocations[weather]
        group_selected: list[int] = []
        if needed:
            group_selected.append(
                min(candidates, key=lambda loc: (_stable_hash(SPLIT_SEED, weather, loc), loc))
            )
        while len(group_selected) < needed:
            remaining = [loc for loc in candidates if loc not in group_selected]

            def minimum_distance(candidate: int) -> tuple[float, str, int]:
                distances = []
                for chosen in group_selected:
                    distance = (
                        abs(normalized_aircraft[candidate] - normalized_aircraft[chosen])
                        + abs(normalized_class_one[candidate] - normalized_class_one[chosen])
                        + abs(normalized_off_nadir[candidate] - normalized_off_nadir[chosen])
                        + (0.5 if country_by_loc[candidate] != country_by_loc[chosen] else 0.0)
                    )
                    distances.append(distance)
                return min(distances), _stable_hash(SPLIT_SEED, candidate), -candidate

            group_selected.append(max(remaining, key=minimum_distance))
        selected.update(group_selected)
    if len(selected) != target or selected & excluded_loc_ids:
        raise PreparationError("Validation location selection violated its frozen constraints")
    details = {
        "algorithm": (
            "weather-proportional allocation followed by fixed-seed greedy metadata diversity "
            "over log aircraft count, FAA-class-1 share, off-nadir, and country"
        ),
        "seed": SPLIT_SEED,
        "validation_fraction_of_eligible_locations": VALIDATION_FRACTION,
        "eligible_location_count": len(eligible),
        "target_validation_location_count": target,
        "excluded_analyzed_location_ids": sorted(excluded_loc_ids),
        "selected_validation_location_ids": sorted(selected),
        "weather_allocation": dict(sorted(allocations.items())),
    }
    return selected, details


def load_scene_objects(
    scene: SceneRecord, raster_path: Path, annotation_path: Path
) -> tuple[ObjectRecord, ...]:
    import rasterio

    with rasterio.open(raster_path) as dataset:
        truths = _load_ground_truth(
            _read_json(annotation_path),
            scene.scene_id,
            dataset.transform,
            make_crs84_to_raster_transformer(dataset.crs),
        )
    objects = []
    for truth in truths:
        diamond = tuple(
            (float(x), float(y)) for x, y in list(truth.diamond_pixel_polygon.exterior.coords)[:-1]
        )
        rectangle = diamond_to_minimum_rectangle(truth.diamond_pixel_polygon)
        major, _, area = rectangle_dimensions(rectangle)
        raw_class = truth.properties.get("faa_wingspan_class")
        faa_class = int(raw_class) if raw_class not in (None, "") else None
        objects.append(
            ObjectRecord(
                scene_id=scene.scene_id,
                loc_id=scene.loc_id,
                gt_index=truth.gt_index,
                faa_class=faa_class,
                diamond_vertices=diamond,
                obb_vertices=rectangle,
                obb_area=area,
                major_axis=major,
            )
        )
    return tuple(objects)


def build_plan(
    scenes: Sequence[SceneRecord],
    objects_by_scene: Mapping[str, Sequence[ObjectRecord]],
    validation_loc_ids: set[int],
    imagery_dir: Path,
) -> dict[str, Any]:
    import rasterio

    scene_plans = []
    assignments_by_scene: dict[str, tuple[Assignment, ...]] = {}
    windows_by_scene: dict[str, tuple[TileWindow, ...]] = {}
    positive_by_split: dict[str, set[str]] = {"train": set(), "val": set()}
    empty_by_split: dict[str, list[str]] = {"train": [], "val": []}
    tile_lookup: dict[str, tuple[str, TileWindow]] = {}
    for number, scene in enumerate(scenes, 1):
        split = "val" if scene.loc_id in validation_loc_ids else "train"
        with rasterio.open(imagery_dir / f"{scene.scene_id}.tif") as dataset:
            windows = tuple(
                generate_tile_windows(
                    scene.scene_id, dataset.width, dataset.height, TILE_SIZE, TILE_OVERLAP
                )
            )
            dimensions = (dataset.width, dataset.height)
        assignments = tuple(
            assign_object_to_tile(object_record, windows)
            for object_record in objects_by_scene[scene.scene_id]
        )
        positive = {assignment.tile.tile_id for assignment in assignments if assignment.tile}
        for tile in windows:
            tile_lookup[tile.tile_id] = (split, tile)
            if tile.tile_id in positive:
                positive_by_split[split].add(tile.tile_id)
            else:
                empty_by_split[split].append(tile.tile_id)
        assignments_by_scene[scene.scene_id] = assignments
        windows_by_scene[scene.scene_id] = windows
        scene_plans.append(
            {
                **asdict(scene),
                "split": split,
                "raster_width": dimensions[0],
                "raster_height": dimensions[1],
                "window_count": len(windows),
                "positive_window_count": len(positive),
                "gt_count": len(assignments),
                "represented_gt": sum(assignment.tile is not None for assignment in assignments),
                "omitted_gt": sum(assignment.tile is None for assignment in assignments),
            }
        )
        print(f"PLAN {number}/{len(scenes)} {scene.scene_id}", flush=True)
    backgrounds = {
        split: set(
            deterministic_background_sample(
                empty_by_split[split], len(positive_by_split[split]), split
            )
        )
        for split in ("train", "val")
    }
    selected_tiles = {
        split: positive_by_split[split] | backgrounds[split] for split in ("train", "val")
    }
    return {
        "scene_plans": scene_plans,
        "assignments_by_scene": assignments_by_scene,
        "windows_by_scene": windows_by_scene,
        "tile_lookup": tile_lookup,
        "positive_tile_ids": positive_by_split,
        "background_tile_ids": backgrounds,
        "selected_tile_ids": selected_tiles,
        "empty_tile_counts": {split: len(empty_by_split[split]) for split in empty_by_split},
    }


def estimate_png_bytes(
    plan: Mapping[str, Any], imagery_dir: Path, sample_count: int = 24
) -> dict[str, Any]:
    import cv2
    import numpy as np
    import rasterio

    selected = sorted(
        set().union(*plan["selected_tile_ids"].values()),
        key=lambda tile_id: (_stable_hash(SPLIT_SEED, "png-estimate", tile_id), tile_id),
    )
    sampled = selected[: min(sample_count, len(selected))]
    sizes = []
    grouped: dict[str, list[TileWindow]] = defaultdict(list)
    for tile_id in sampled:
        _, tile = plan["tile_lookup"][tile_id]
        grouped[tile.scene_id].append(tile)
    for scene_id, tiles in grouped.items():
        with rasterio.open(imagery_dir / f"{scene_id}.tif") as dataset:
            for tile in tiles:
                rgb = np.moveaxis(read_tile(dataset, tile, pad_to_tile_size=True).data, 0, -1)
                ok, encoded = cv2.imencode(".png", np.ascontiguousarray(rgb[..., ::-1]))
                if not ok:
                    raise PreparationError(f"PNG estimate encoding failed: {tile.tile_id}")
                sizes.append(int(encoded.size))
    average = sum(sizes) / len(sizes) if sizes else 0.0
    selected_count = sum(len(values) for values in plan["selected_tile_ids"].values())
    return {
        "sample_count": len(sizes),
        "sample_png_bytes": sizes,
        "average_png_bytes": average,
        "selected_tile_count": selected_count,
        "estimated_image_bytes": math.ceil(average * selected_count),
        "estimated_total_bytes_with_10_percent_overhead": math.ceil(average * selected_count * 1.10),
    }


def _format_label(values: Sequence[float]) -> str:
    return f"{OUTPUT_CLASS_ID} " + " ".join(f"{value:.8f}" for value in values)


def materialize_dataset(
    scenes: Sequence[SceneRecord],
    plan: Mapping[str, Any],
    imagery_dir: Path,
    dataset_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import cv2
    import numpy as np
    import rasterio

    if dataset_root.exists() and any(dataset_root.iterdir()):
        raise PreparationError(f"Refusing to overwrite non-empty generated dataset: {dataset_root}")
    for split in ("train", "val"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    scene_by_id = {scene.scene_id: scene for scene in scenes}
    assignments_by_tile: dict[str, list[Assignment]] = defaultdict(list)
    coverage_rows = []
    object_rows = []
    for scene_id, assignments in plan["assignments_by_scene"].items():
        split = "val" if scene_by_id[scene_id].loc_id in {
            int(row["loc_id"])
            for row in plan["scene_plans"]
            if row["split"] == "val"
        } else "train"
        for assignment in assignments:
            object_record = assignment.object_record
            tile_count = 1 if assignment.tile is not None else 0
            coverage_rows.append(
                {
                    "split": split,
                    "scene_id": scene_id,
                    "loc_id": object_record.loc_id,
                    "gt_index": object_record.gt_index,
                    "represented": assignment.tile is not None,
                    "representation_count": tile_count,
                    "assignment_reason": assignment.reason,
                    "best_visibility_fraction": assignment.visibility_fraction,
                }
            )
            if assignment.tile is not None:
                assignments_by_tile[assignment.tile.tile_id].append(assignment)

    tile_rows = []
    selected = set().union(*plan["selected_tile_ids"].values())
    for number, scene_plan in enumerate(plan["scene_plans"], 1):
        scene_id = scene_plan["scene_id"]
        scene = scene_by_id[scene_id]
        split = scene_plan["split"]
        raster_path = imagery_dir / f"{scene_id}.tif"
        with rasterio.open(raster_path) as dataset:
            for tile in plan["windows_by_scene"][scene_id]:
                if tile.tile_id not in selected:
                    continue
                assignments = sorted(
                    assignments_by_tile.get(tile.tile_id, []),
                    key=lambda assignment: assignment.object_record.gt_index,
                )
                image_name = f"{tile.tile_id}.png"
                label_name = f"{tile.tile_id}.txt"
                image_path = dataset_root / "images" / split / image_name
                label_path = dataset_root / "labels" / split / label_name
                tile_read = read_tile(dataset, tile, pad_to_tile_size=True)
                rgb = np.moveaxis(tile_read.data, 0, -1)
                if not cv2.imwrite(str(image_path), np.ascontiguousarray(rgb[..., ::-1])):
                    raise PreparationError(f"Failed to write PNG: {image_path}")
                lines = []
                for assignment in assignments:
                    obj = assignment.object_record
                    normalized = global_to_normalized_tile_label(obj.obb_vertices, tile)
                    lines.append(_format_label(normalized))
                    local_vertices = tuple(
                        (x - tile.col_off, y - tile.row_off) for x, y in obj.obb_vertices
                    )
                    object_rows.append(
                        {
                            "split": split,
                            "scene_id": scene_id,
                            "loc_id": scene.loc_id,
                            "tile_id": tile.tile_id,
                            "gt_index": obj.gt_index,
                            "class_id": OUTPUT_CLASS_ID,
                            "source_diamond_vertices": json.dumps(obj.diamond_vertices),
                            "derived_global_obb_vertices": json.dumps(obj.obb_vertices),
                            "local_tile_obb_vertices": json.dumps(local_vertices),
                            "normalized_obb_vertices": json.dumps(normalized),
                            "visibility_fraction": assignment.visibility_fraction,
                            "assignment_reason": assignment.reason,
                        }
                    )
                label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                tile_rows.append(
                    {
                        "split": split,
                        "scene_id": scene_id,
                        "loc_id": scene.loc_id,
                        "source_raster": str(raster_path.relative_to(find_project_root())),
                        "tile_id": tile.tile_id,
                        "row_off": tile.row_off,
                        "col_off": tile.col_off,
                        "width": tile.width,
                        "height": tile.height,
                        "tile_size": tile.tile_size,
                        "object_count": len(assignments),
                        "tile_kind": "positive" if assignments else "background",
                        "weather": scene.weather,
                        "country": scene.country,
                        "state_province": scene.state_province,
                        "air_field": scene.air_field,
                        "sensor": scene.sensor,
                        "collection_time_start": scene.collection_time_start,
                        "off_nadir": scene.off_nadir,
                    }
                )
        print(f"WRITE {number}/{len(plan['scene_plans'])} {scene_id}", flush=True)
    return tile_rows, object_rows, coverage_rows


def audit_generated_dataset(
    dataset_root: Path,
    tile_rows: Sequence[Mapping[str, Any]],
    object_rows: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from shapely.geometry import Polygon

    split_report = {}
    malformed = 0
    out_of_range = 0
    missing_pairs = 0
    for split in ("train", "val"):
        images = sorted((dataset_root / "images" / split).glob("*.png"))
        labels = sorted((dataset_root / "labels" / split).glob("*.txt"))
        image_stems, label_stems = {path.stem for path in images}, {path.stem for path in labels}
        missing_pairs += len(image_stems ^ label_stems)
        instances = 0
        for path in labels:
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                try:
                    values = [float(value) for value in parts]
                    valid = (
                        len(values) == 9
                        and values[0] == OUTPUT_CLASS_ID
                        and all(math.isfinite(value) for value in values)
                    )
                    if not valid:
                        malformed += 1
                        continue
                    coords = values[1:]
                    if any(value < 0.0 or value > 1.0 for value in coords):
                        out_of_range += 1
                    polygon = Polygon(list(zip(coords[::2], coords[1::2], strict=True)))
                    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
                        malformed += 1
                    instances += 1
                except (TypeError, ValueError):
                    malformed += 1
        rows = [row for row in tile_rows if row["split"] == split]
        split_report[split] = {
            "images": len(images),
            "labels": len(labels),
            "positive_tiles": sum(row["tile_kind"] == "positive" for row in rows),
            "background_tiles": sum(row["tile_kind"] == "background" for row in rows),
            "obb_instances": instances,
        }
    train_rows = [row for row in tile_rows if row["split"] == "train"]
    val_rows = [row for row in tile_rows if row["split"] == "val"]
    coverage = {}
    for split in ("train", "val"):
        rows = [row for row in coverage_rows if row["split"] == split]
        distribution = Counter(int(row["representation_count"]) for row in rows)
        coverage[split] = {
            "gt_total": len(rows),
            "represented_gt": sum(bool(row["represented"]) for row in rows),
            "omitted_gt": sum(not bool(row["represented"]) for row in rows),
            "representation_rate": sum(bool(row["represented"]) for row in rows) / len(rows) if rows else 0.0,
            "representation_count_distribution": dict(sorted(distribution.items())),
        }
    return {
        "splits": split_report,
        "malformed_labels": malformed,
        "out_of_range_labels": out_of_range,
        "missing_image_label_pairs": missing_pairs,
        "train_val_loc_id_overlap": sorted(
            {int(row["loc_id"]) for row in train_rows}
            & {int(row["loc_id"]) for row in val_rows}
        ),
        "train_val_scene_overlap": sorted(
            {str(row["scene_id"]) for row in train_rows}
            & {str(row["scene_id"]) for row in val_rows}
        ),
        "duplicate_gt_assignments": sum(
            int(row["representation_count"]) > 1 for row in coverage_rows
        ),
        "gt_coverage": coverage,
        "object_manifest_rows": len(object_rows),
    }


def generate_visual_audit(
    dataset_root: Path, tile_rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> list[str]:
    import cv2
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for split in ("train", "val"):
        positives = [row for row in tile_rows if row["split"] == split and row["tile_kind"] == "positive"]
        ordered = sorted(
            positives,
            key=lambda row: (
                0 if row["weather"] == "Snow" else 1,
                -int(row["object_count"]),
                _stable_hash(SPLIT_SEED, "visual", row["tile_id"]),
            ),
        )
        selected = []
        candidates = []
        snow = [row for row in ordered if row["weather"] == "Snow"]
        clear = [row for row in ordered if row["weather"] != "Snow"]
        if snow:
            candidates.append(snow[0])
        if clear:
            candidates.append(clear[0])
        candidates.extend(sorted(positives, key=lambda row: -int(row["object_count"]))[:2])
        candidates.extend(sorted(positives, key=lambda row: int(row["object_count"]))[:2])
        candidates.extend(ordered)
        for row in candidates:
            if row["tile_id"] not in {item["tile_id"] for item in selected}:
                selected.append(row)
            if len(selected) == 6:
                break
        for index, row in enumerate(selected, 1):
            image_path = dataset_root / "images" / split / f"{row['tile_id']}.png"
            label_path = dataset_root / "labels" / split / f"{row['tile_id']}.txt"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise PreparationError(f"Visual audit image unreadable: {image_path}")
            for line in label_path.read_text(encoding="utf-8").splitlines():
                values = [float(value) for value in line.split()[1:]]
                points = np.array(
                    [(round(values[i] * TILE_SIZE), round(values[i + 1] * TILE_SIZE)) for i in range(0, 8, 2)],
                    dtype=np.int32,
                )
                cv2.polylines(image, [points], True, (0, 0, 255), 2, cv2.LINE_AA)
            target = output_dir / f"{split}_{index:02d}_{row['tile_id']}.png"
            if not cv2.imwrite(str(target), image):
                raise PreparationError(f"Failed to write visual audit image: {target}")
            written.append(str(target))
    return written


def ultralytics_dataset_smoke(yaml_path: Path) -> dict[str, Any]:
    from ultralytics.cfg import DEFAULT_CFG
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(yaml_path, autodownload=False)
    datasets = {}
    for split in ("train", "val"):
        dataset = YOLODataset(
            img_path=data[split],
            imgsz=TILE_SIZE,
            cache=False,
            augment=False,
            hyp=DEFAULT_CFG,
            prefix=f"{split}: ",
            batch_size=1,
            data=data,
            task="obb",
        )
        datasets[split] = {
            "images": len(dataset.im_files),
            "labels": len(dataset.labels),
            "instances": sum(len(label["cls"]) for label in dataset.labels),
        }
    return {"class_count": len(data["names"]), "names": data["names"], "splits": datasets}


def prepare_dataset(
    root: Path,
    dataset_root: Path,
    yaml_path: Path,
    audit_output: Path,
    *,
    plan_only: bool = False,
) -> dict[str, Any]:
    source = root / "datasets" / "RarePlanes"
    train_root = source / "train"
    imagery = train_root / "imagery" / "PS-RGB_cog"
    annotations = train_root / "annotations" / "geojson_aircraft"
    metadata = source / "metadata" / "RarePlanes_Public_Metadata.csv"
    development_manifest = root / "outputs" / "batch_train_evaluation" / "fixed_baseline_12" / "selection_manifest.csv"
    holdout_manifest = root / "outputs" / "boundary_policy_validation" / "train_holdout_12" / "selection_manifest.csv"
    scenes = load_train_scene_records(metadata, imagery)
    objects_by_scene = {}
    for index, scene in enumerate(scenes, 1):
        objects_by_scene[scene.scene_id] = load_scene_objects(
            scene,
            imagery / f"{scene.scene_id}.tif",
            annotations / f"{scene.scene_id}.geojson",
        )
        print(f"GEOMETRY {index}/{len(scenes)} {scene.scene_id}", flush=True)
    excluded = analyzed_location_ids(development_manifest, holdout_manifest)
    validation_locs, selection = select_validation_locations(scenes, objects_by_scene, excluded)
    plan = build_plan(scenes, objects_by_scene, validation_locs, imagery)
    estimate = estimate_png_bytes(plan, imagery)
    disk_before = shutil.disk_usage(root)
    projected_free = disk_before.free - estimate["estimated_total_bytes_with_10_percent_overhead"]
    split_rows = [
        {**row, "analyzed_location": int(row["loc_id"]) in excluded}
        for row in plan["scene_plans"]
    ]
    plan_summary = {
        "protocol": {
            "source_split": "train",
            "class_names": {0: OUTPUT_CLASS_NAME},
            "tile_size": TILE_SIZE,
            "tile_overlap": TILE_OVERLAP,
            "integer_stride": stride_from_overlap(TILE_SIZE, TILE_OVERLAP),
            "background_to_positive_ratio": BACKGROUND_TO_POSITIVE_RATIO,
            "object_assignment": "one closest-center tile with full rectangular OBB visibility; no clipping",
        },
        "selection": selection,
        "splits": {},
        "storage_estimate": {
            **estimate,
            "free_bytes_before": disk_before.free,
            "projected_free_bytes": projected_free,
            "minimum_required_free_bytes_after": 50 * 1024**3,
        },
    }
    for split in ("train", "val"):
        selected_scenes = [row for row in split_rows if row["split"] == split]
        scene_ids = {row["scene_id"] for row in selected_scenes}
        split_objects = [obj for scene_id in scene_ids for obj in objects_by_scene[scene_id]]
        plan_summary["splits"][split] = {
            "locations": len({int(row["loc_id"]) for row in selected_scenes}),
            "scenes": len(selected_scenes),
            "gt_aircraft": len(split_objects),
            "positive_tiles": len(plan["positive_tile_ids"][split]),
            "background_tiles_retained": len(plan["background_tile_ids"][split]),
            "empty_tiles_skipped": plan["empty_tile_counts"][split] - len(plan["background_tile_ids"][split]),
            "weather_scene_distribution": dict(sorted(Counter(row["weather"] for row in selected_scenes).items())),
            "faa_class_distribution": dict(sorted(Counter(obj.faa_class for obj in split_objects).items(), key=lambda item: str(item[0]))),
            "snow_scenes": sum(row["weather"] == "Snow" for row in selected_scenes),
            "snow_aircraft": sum(
                len(objects_by_scene[row["scene_id"]])
                for row in selected_scenes
                if row["weather"] == "Snow"
            ),
            "snow_positive_tiles": sum(
                row["positive_window_count"]
                for row in selected_scenes
                if row["weather"] == "Snow"
            ),
        }
    audit_output.mkdir(parents=True, exist_ok=True)
    _write_json(audit_output / "preparation_plan.json", plan_summary)
    _write_csv(audit_output / "split_manifest.csv", split_rows, tuple(split_rows[0]))
    if projected_free < 50 * 1024**3:
        raise PreparationError("Projected generated dataset would leave less than 50 GiB free")
    if plan_only:
        return plan_summary

    tile_rows, object_rows, coverage_rows = materialize_dataset(
        scenes, plan, imagery, dataset_root
    )
    _write_csv(dataset_root / "manifests" / "tiles.csv", tile_rows, tuple(tile_rows[0]))
    _write_csv(dataset_root / "manifests" / "objects.csv", object_rows, tuple(object_rows[0]))
    _write_csv(dataset_root / "manifests" / "gt_coverage.csv", coverage_rows, tuple(coverage_rows[0]))
    _write_csv(dataset_root / "manifests" / "split_scenes.csv", split_rows, tuple(split_rows[0]))
    audit = audit_generated_dataset(dataset_root, tile_rows, object_rows, coverage_rows)
    visual_paths = generate_visual_audit(dataset_root, tile_rows, audit_output / "visual_samples")
    smoke = ultralytics_dataset_smoke(yaml_path)
    disk_after = shutil.disk_usage(root)
    final = {
        **plan_summary,
        "audit": audit,
        "visual_audit_paths": visual_paths,
        "ultralytics_smoke_test": smoke,
        "actual_generated_bytes": sum(
            path.stat().st_size for path in dataset_root.rglob("*") if path.is_file()
        ),
        "free_bytes_after": disk_after.free,
    }
    _write_json(audit_output / "dataset_audit.json", final)
    return final


def _defaults() -> dict[str, Path]:
    root = find_project_root()
    return {
        "root": root,
        "dataset": root / "datasets" / "RarePlanes_yolo_obb",
        "yaml": root / "configs" / "rareplanes_obb.yaml",
        "audit": root / "outputs" / "rareplanes_dataset_preparation",
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = _defaults()
    parser = argparse.ArgumentParser(description="Prepare RarePlanes TRAIN-only YOLO OBB chips")
    parser.add_argument("--root", type=Path, default=defaults["root"])
    parser.add_argument("--dataset-root", type=Path, default=defaults["dataset"])
    parser.add_argument("--yaml", type=Path, default=defaults["yaml"])
    parser.add_argument("--audit-output", type=Path, default=defaults["audit"])
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_dataset(
        args.root.resolve(),
        args.dataset_root.resolve(),
        args.yaml.resolve(),
        args.audit_output.resolve(),
        plan_only=args.plan_only,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
