from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest

import yaml
from shapely.geometry import Polygon


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from geospatial_tiling import TileWindow  # noqa: E402
from prepare_rareplanes_obb import (  # noqa: E402
    ANALYZED_LOCATION_IDS,
    BACKGROUND_TO_POSITIVE_RATIO,
    OUTPUT_CLASS_ID,
    ObjectRecord,
    PreparationError,
    SceneRecord,
    analyzed_location_ids,
    assign_object_to_tile,
    audit_generated_dataset,
    canonical_rectangle_vertices,
    deterministic_background_sample,
    diamond_to_minimum_rectangle,
    global_to_normalized_tile_label,
    select_validation_locations,
    visibility_fraction,
)


def tile(tile_id: str, col: int, row: int, size: int = 100) -> TileWindow:
    return TileWindow(
        scene_id="1_scene",
        tile_id=tile_id,
        row_off=row,
        col_off=col,
        width=size,
        height=size,
        tile_size=size,
        source_width=max(200, col + size),
        source_height=max(200, row + size),
        row_index=row // size,
        col_index=col // size,
    )


def object_record(vertices=((20, 20), (40, 20), (40, 30), (20, 30))) -> ObjectRecord:
    return ObjectRecord(
        scene_id="1_scene",
        loc_id=1,
        gt_index=0,
        faa_class=1,
        diamond_vertices=((30, 15), (45, 25), (30, 35), (15, 25)),
        obb_vertices=tuple(vertices),
        obb_area=200.0,
        major_axis=20.0,
    )


class PrepareRarePlanesOBBTests(unittest.TestCase):
    def test_diamond_to_minimum_area_rectangle(self) -> None:
        diamond = Polygon([(5, 0), (10, 5), (5, 10), (0, 5)])
        rectangle = diamond_to_minimum_rectangle(diamond)
        polygon = Polygon(rectangle)
        self.assertEqual(len(rectangle), 4)
        self.assertAlmostEqual(polygon.area, 50.0)
        self.assertTrue(polygon.equals(diamond))

    def test_vertex_order_is_cyclic_clockwise_in_image_space(self) -> None:
        polygon = Polygon([(10, 10), (0, 10), (0, 0), (10, 0)])
        vertices = canonical_rectangle_vertices(polygon)
        self.assertEqual(vertices[0], (0.0, 0.0))
        signed = sum(
            vertices[i][0] * vertices[(i + 1) % 4][1]
            - vertices[(i + 1) % 4][0] * vertices[i][1]
            for i in range(4)
        )
        self.assertGreater(signed, 0.0)

    def test_global_to_tile_local_normalized_label(self) -> None:
        owner = tile("tile-a", 100, 200, 100)
        values = global_to_normalized_tile_label(
            ((110, 210), (190, 210), (190, 290), (110, 290)), owner
        )
        self.assertEqual(values, (0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))

    def test_out_of_range_label_is_rejected(self) -> None:
        with self.assertRaisesRegex(Exception, "outside"):
            global_to_normalized_tile_label(
                ((-1, 1), (2, 1), (2, 2), (-1, 2)), tile("tile-a", 0, 0)
            )

    def test_visibility_calculation(self) -> None:
        owner = tile("tile-a", 0, 0)
        self.assertAlmostEqual(
            visibility_fraction(((90, 10), (110, 10), (110, 30), (90, 30)), owner),
            0.5,
        )

    def test_object_has_one_fully_visible_owner(self) -> None:
        obj = object_record()
        windows = (tile("b", 0, 0), tile("a", 10, 0))
        assignment = assign_object_to_tile(obj, windows)
        self.assertIsNotNone(assignment.tile)
        self.assertEqual(assignment.tile.tile_id, "b")
        self.assertEqual(assignment.reason, "single_owner_fully_visible_closest_tile_center")

    def test_truncated_object_is_omitted_without_clipping(self) -> None:
        obj = object_record(((90, 20), (110, 20), (110, 30), (90, 30)))
        assignment = assign_object_to_tile(obj, (tile("only", 0, 0),))
        self.assertIsNone(assignment.tile)
        self.assertAlmostEqual(assignment.visibility_fraction, 0.5)

    def test_background_sampling_is_deterministic_and_bounded(self) -> None:
        candidates = tuple(f"tile-{index}" for index in range(20))
        first = deterministic_background_sample(candidates, 8, "train")
        second = deterministic_background_sample(tuple(reversed(candidates)), 8, "train")
        self.assertEqual(first, second)
        self.assertEqual(len(first), int(8 * BACKGROUND_TO_POSITIVE_RATIO))

    def test_validation_locations_are_disjoint_and_exclude_analyzed(self) -> None:
        scenes = tuple(
            SceneRecord(
                scene_id=f"{loc}_scene",
                loc_id=loc,
                country="A" if loc % 2 else "B",
                state_province="",
                air_field="",
                weather="Snow" if loc % 3 == 0 else "Clear Skies",
                sensor="WV03",
                collection_time_start="",
                collection_time_end="",
                cloud_cover_percentage=0.0,
                off_nadir=float(loc),
            )
            for loc in range(1, 31)
        )
        objects = {
            scene.scene_id: tuple(
                ObjectRecord(
                    scene.scene_id,
                    scene.loc_id,
                    index,
                    1 if index % 2 else 2,
                    ((0, 0), (1, 0), (1, 1), (0, 1)),
                    ((0, 0), (1, 0), (1, 1), (0, 1)),
                    1.0,
                    1.0,
                )
                for index in range(scene.loc_id % 5 + 1)
            )
            for scene in scenes
        }
        excluded = {1, 2, 3, 4}
        selected, details = select_validation_locations(scenes, objects, excluded)
        selected_again, _ = select_validation_locations(scenes, objects, excluded)
        self.assertEqual(selected, selected_again)
        self.assertFalse(selected & excluded)
        self.assertEqual(len(selected), details["target_validation_location_count"])
        self.assertFalse(selected & (set(range(1, 31)) - selected - excluded))

    def test_frozen_analyzed_locations_support_clean_clone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = analyzed_location_ids(root / "development.csv", root / "holdout.csv")
        self.assertEqual(actual, set(ANALYZED_LOCATION_IDS))
        self.assertEqual(len(actual), 23)

    def test_partial_analyzed_provenance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "development.csv").write_text("scene_id,loc_id,split\n", encoding="utf-8")
            with self.assertRaisesRegex(PreparationError, "both exist or both be absent"):
                analyzed_location_ids(root / "development.csv", root / "holdout.csv")

    def test_source_contains_no_official_test_path_reference(self) -> None:
        source = (SRC_DIR / "prepare_rareplanes_obb.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("rareplanes/test", source.replace("\\", "/"))
        self.assertNotIn("instances_test", source)

    def test_manifest_provenance_fields(self) -> None:
        scene = SceneRecord(
            "1_scene", 1, "Country", "State", "Airport", "Snow", "WV03", "start", "end", 2.0, 10.0
        )
        fields = asdict(scene)
        for required in (
            "scene_id",
            "loc_id",
            "weather",
            "sensor",
            "collection_time_start",
            "off_nadir",
        ):
            self.assertIn(required, fields)

    def test_gt_coverage_and_label_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "val"):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
                (root / "images" / split / f"{split}.png").write_bytes(b"placeholder")
                (root / "labels" / split / f"{split}.txt").write_text(
                    "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", encoding="utf-8"
                )
            tiles = [
                {"split": "train", "loc_id": 1, "scene_id": "a", "tile_kind": "positive"},
                {"split": "val", "loc_id": 2, "scene_id": "b", "tile_kind": "positive"},
            ]
            coverage = [
                {"split": "train", "represented": True, "representation_count": 1},
                {"split": "val", "represented": False, "representation_count": 0},
            ]
            audit = audit_generated_dataset(root, tiles, [{}, {}], coverage)
            self.assertEqual(audit["malformed_labels"], 0)
            self.assertEqual(audit["missing_image_label_pairs"], 0)
            self.assertEqual(audit["gt_coverage"]["train"]["represented_gt"], 1)
            self.assertEqual(audit["gt_coverage"]["val"]["omitted_gt"], 1)
            self.assertEqual(audit["train_val_loc_id_overlap"], [])

    def test_dataset_yaml_is_one_class_and_train_val_only(self) -> None:
        path = REPOSITORY_ROOT / "configs" / "rareplanes_obb.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(data["names"], {0: "plane"})
        self.assertEqual(data["train"], "images/train")
        self.assertEqual(data["val"], "images/val")
        self.assertNotIn("test", data)
        self.assertFalse(Path(data["path"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
