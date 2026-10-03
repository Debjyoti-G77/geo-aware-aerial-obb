from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from shapely.geometry import Polygon


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from domain_failure_analysis import (  # noqa: E402
    DomainAnalysisError,
    _failure_category,
    _object_contrast,
    assign_size_bin,
    foreground_background_statistics,
    grouped_recall,
    load_analysis_scenes,
    minimum_rectangle_dimensions,
    nearest_neighbor_distances,
    object_crop_bounds,
    quartile_edges,
)


class DomainFailureAnalysisTests(unittest.TestCase):
    def test_pixel_space_gt_size_extraction(self) -> None:
        polygon = Polygon([(0, 0), (12, 0), (12, 4), (0, 4)])
        major, minor, area, aspect = minimum_rectangle_dimensions(polygon)
        self.assertAlmostEqual(major, 12.0)
        self.assertAlmostEqual(minor, 4.0)
        self.assertAlmostEqual(area, 48.0)
        self.assertAlmostEqual(aspect, 3.0)

    def test_deterministic_size_bin_assignment(self) -> None:
        values = [1, 2, 3, 4, 5, 6, 7, 8]
        edges = quartile_edges(values)
        self.assertEqual(edges, quartile_edges(tuple(reversed(values))))
        labels = [assign_size_bin(value, edges) for value in values]
        self.assertEqual(labels, sorted(labels, key=("Q1_smallest", "Q2", "Q3", "Q4_largest").index))
        self.assertEqual(assign_size_bin(edges[0], edges), "Q2")

    def test_tp_fn_join_logic_is_exclusive(self) -> None:
        matches = {0: {"confidence": "0.9"}}
        false_negatives = {1: {"nearest_prediction_iou": "0.1"}}
        self.assertFalse(set(matches) & set(false_negatives))
        self.assertEqual(set(matches) | set(false_negatives), {0, 1})

    def test_weather_size_stratification(self) -> None:
        rows = [
            {"weather": "Snow", "size": "Q1", "outcome": "TP"},
            {"weather": "Snow", "size": "Q1", "outcome": "FN"},
            {"weather": "Clear", "size": "Q1", "outcome": "TP"},
        ]
        result = grouped_recall(rows, ("weather", "size"))
        snow = next(row for row in result if row["weather"] == "Snow")
        self.assertEqual((snow["gt_count"], snow["tp"], snow["fn"]), (2, 1, 1))
        self.assertAlmostEqual(snow["recall"], 0.5)

    def test_object_crop_bounds_are_clipped(self) -> None:
        self.assertEqual(object_crop_bounds((2, 3, 12, 13), 100, 80, 10), (0, 0, 27, 28))
        self.assertEqual(object_crop_bounds((90, 70, 99, 79), 100, 80, 10), (75, 55, 100, 80))

    def test_foreground_background_contrast(self) -> None:
        rgb = np.zeros((3, 4, 4), dtype=np.uint8)
        rgb[:, 1:3, 1:3] = 255
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        result = foreground_background_statistics(rgb, mask)
        self.assertAlmostEqual(result["foreground_luminance_mean"], 1.0)
        self.assertAlmostEqual(result["background_luminance_mean"], 0.0)
        self.assertAlmostEqual(result["absolute_luminance_contrast"], 1.0)
        self.assertAlmostEqual(result["rgb_color_distance"], 3 ** 0.5)

    def test_object_contrast_uses_source_pixel_polygon_coordinates(self) -> None:
        from affine import Affine
        from rasterio.io import MemoryFile

        data = np.zeros((3, 20, 20), dtype=np.uint8)
        data[:, 5:10, 5:10] = 255
        polygon = Polygon([(5, 5), (10, 5), (10, 10), (5, 10)])
        with MemoryFile() as memory:
            with memory.open(
                driver="GTiff",
                width=20,
                height=20,
                count=3,
                dtype="uint8",
                transform=Affine(2, 0, 100, 0, -2, 200),
            ) as dataset:
                dataset.write(data)
                result = _object_contrast(dataset, polygon, 5.0)
        self.assertIsNotNone(result["foreground_luminance_mean"])
        self.assertGreater(result["absolute_luminance_contrast"], 0.9)

    def test_density_nearest_neighbor_calculation(self) -> None:
        result = nearest_neighbor_distances([(0, 0), (3, 4), (13, 4)])
        self.assertEqual(result, [5.0, 5.0, 10.0])
        self.assertEqual(nearest_neighbor_distances([(1, 1)]), [None])

    def test_failure_category_accounting(self) -> None:
        fn = {"nearest_prediction_iou": "0.2"}
        category, flags = _failure_category(4, fn, [], [])
        self.assertTrue(flags["plane_final_nearby"])
        self.assertTrue(category.startswith("B_"))
        boundary = [{"best_current_fn_gt_index": 4, "would_independently_match_current_fn": True}]
        category, _ = _failure_category(4, {"nearest_prediction_iou": ""}, boundary, [])
        self.assertTrue(category.startswith("D_"))

    def test_train_only_and_provenance_enforcement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            development = root / "development.csv"
            holdout = root / "holdout.csv"
            fields = ("scene_id", "split", "loc_id", "width", "height", "gt_aircraft_count")
            for path, prefix in ((development, "d"), (holdout, "h")):
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    for index in range(12):
                        writer.writerow(
                            {
                                "scene_id": f"{prefix}{index}",
                                "split": "train",
                                "loc_id": index + (0 if prefix == "d" else 100),
                                "width": 100,
                                "height": 100,
                                "gt_aircraft_count": 1,
                            }
                        )
            records = load_analysis_scenes(development, holdout)
            self.assertEqual(sum(row["dataset_subset"] == "development" for row in records), 12)
            self.assertEqual(sum(row["dataset_subset"] == "holdout" for row in records), 12)

            with development.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["split"] = "test"
            with development.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(DomainAnalysisError, "non-TRAIN"):
                load_analysis_scenes(development, holdout)


if __name__ == "__main__":
    unittest.main()
