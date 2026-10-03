from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from affine import Affine
from shapely import affinity
from shapely.geometry import Polygon, box


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from geospatial_evaluation import (  # noqa: E402
    _build_boundary_diagnostics,
    _build_nms_diagnostics,
    EvaluationResult,
    FalseNegative,
    FalsePositive,
    GroundTruth,
    Match,
    descriptive_statistics,
    geodesic_distance_metres,
    geographic_polygon_to_pixel,
    greedy_confidence_match,
    minimum_rotated_rectangle,
    polygon_iou,
    precision_recall_f1,
    round_trip_error,
    write_evaluation_outputs,
)


class MatchingTests(unittest.TestCase):
    def test_perfect_obb_match(self):
        polygon = box(10, 20, 40, 60)
        result = greedy_confidence_match([polygon], [0.9], [polygon], 0.50)
        self.assertEqual(result.matches, ((0, 0, 1.0),))
        self.assertEqual(result.false_prediction_indices, ())
        self.assertEqual(result.false_gt_indices, ())

    def test_iou_below_threshold(self):
        result = greedy_confidence_match(
            [box(0, 0, 10, 10)], [0.9], [box(9, 0, 19, 10)], 0.50
        )
        self.assertEqual(result.matches, ())
        self.assertEqual(result.false_prediction_indices, (0,))
        self.assertEqual(result.false_gt_indices, (0,))

    def test_duplicate_predictions_for_one_gt(self):
        gt = box(0, 0, 10, 10)
        result = greedy_confidence_match([gt, gt], [0.9, 0.8], [gt], 0.50)
        self.assertEqual(result.matches, ((0, 0, 1.0),))
        self.assertEqual(result.false_prediction_indices, (1,))

    def test_one_prediction_cannot_match_multiple_gt(self):
        polygon = box(0, 0, 10, 10)
        result = greedy_confidence_match([polygon], [0.9], [polygon, polygon], 0.50)
        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.false_gt_indices, (1,))

    def test_one_gt_cannot_match_multiple_predictions(self):
        polygon = box(0, 0, 10, 10)
        result = greedy_confidence_match(
            [polygon, polygon, polygon], [0.9, 0.8, 0.7], [polygon], 0.50
        )
        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.false_prediction_indices, (1, 2))

    def test_confidence_order_controls_greedy_matching(self):
        gt = box(0, 0, 10, 10)
        lower_confidence_perfect = gt
        higher_confidence_partial = box(0, 0, 7, 10)
        result = greedy_confidence_match(
            [lower_confidence_perfect, higher_confidence_partial],
            [0.8, 0.9],
            [gt],
            0.50,
        )
        self.assertEqual(result.matches[0][0], 1)
        self.assertAlmostEqual(result.matches[0][2], 0.7)
        self.assertEqual(result.false_prediction_indices, (0,))

    def test_rotated_polygon_iou(self):
        original = affinity.rotate(box(-10, -2, 10, 2), 31, origin=(0, 0))
        shifted = affinity.translate(original, xoff=1.0)
        overlap = polygon_iou(original, shifted)
        self.assertGreater(overlap, 0.5)
        self.assertLess(overlap, 1.0)
        self.assertAlmostEqual(polygon_iou(original, original), 1.0)

    def test_empty_prediction_case(self):
        result = greedy_confidence_match([], [], [box(0, 0, 1, 1)], 0.50)
        self.assertEqual(result.matches, ())
        self.assertEqual(result.false_prediction_indices, ())
        self.assertEqual(result.false_gt_indices, (0,))

    def test_empty_ground_truth_case(self):
        result = greedy_confidence_match([box(0, 0, 1, 1)], [0.9], [], 0.50)
        self.assertEqual(result.matches, ())
        self.assertEqual(result.false_prediction_indices, (0,))
        self.assertEqual(result.false_gt_indices, ())


class GeometryTests(unittest.TestCase):
    def test_gt_diamond_to_minimum_rotated_rectangle(self):
        diamond = Polygon([(0, 2), (3, 0), (8, 2), (3, 4)])
        rectangle = minimum_rotated_rectangle(diamond)
        self.assertTrue(rectangle.is_valid)
        self.assertGreater(rectangle.area, diamond.area)
        self.assertEqual(len(rectangle.exterior.coords), 5)

    def test_crs84_world_to_continuous_pixel_and_round_trip(self):
        transform = Affine(0.01, 0.0, 5.0, 0.0, -0.01, 50.0)
        world = Polygon([(5.1, 49.9), (5.2, 49.9), (5.2, 49.8), (5.1, 49.8)])
        pixel = geographic_polygon_to_pixel(world, transform, None)
        self.assertEqual(tuple(round(value, 8) for value in pixel.bounds), (10.0, 10.0, 20.0, 20.0))
        self.assertLess(round_trip_error(world, pixel, transform, None), 1e-12)

    def test_center_distance_calculations(self):
        first = box(0, 0, 2, 2).centroid
        second = box(3, 4, 5, 6).centroid
        self.assertAlmostEqual(first.distance(second), 5.0)
        distance = geodesic_distance_metres((0.0, 0.0), (0.001, 0.0))
        self.assertGreater(distance, 111.0)
        self.assertLess(distance, 112.0)


class MetricTests(unittest.TestCase):
    def test_tp_fp_fn_precision_recall_f1(self):
        precision, recall, f1 = precision_recall_f1(tp=2, fp=1, fn=2)
        self.assertAlmostEqual(precision, 2 / 3)
        self.assertAlmostEqual(recall, 1 / 2)
        self.assertAlmostEqual(f1, 4 / 7)

    def test_descriptive_statistics(self):
        values = descriptive_statistics([0.5, 0.7, 0.9])
        self.assertAlmostEqual(values["mean"], 0.7)
        self.assertAlmostEqual(values["median"], 0.7)
        self.assertEqual(descriptive_statistics([])["mean"], None)


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        first = box(0, 0, 10, 10)
        second = box(20, 20, 30, 30)
        self.truths = (
            GroundTruth(0, first, first, first, {}),
            GroundTruth(1, second, second, second, {}),
        )
        self.official_matches = (
            Match("kept", 0, 0.9, 1.0, 1.0, 0.0, 0.0, "tile-a"),
        )

    def test_evaluator_compares_suppressed_geometry_to_current_fn(self):
        suppression = {
            "suppressions": [
                {
                    "suppressed_detection_id": "suppressed",
                    "kept_detection_id": "kept",
                    "suppressed_global_pixel_obb_vertices": [
                        [20, 20], [30, 20], [30, 30], [20, 30]
                    ],
                    "kept_global_pixel_obb_vertices": [
                        [0, 0], [10, 0], [10, 10], [0, 10]
                    ],
                }
            ]
        }
        result = _build_nms_diagnostics(
            suppression,
            self.official_matches,
            (),
            self.truths,
            (1,),
            0.50,
        )
        record = result["records"][0]
        self.assertTrue(record["geometry_reconstructable"])
        self.assertEqual(record["best_gt_index"], 1)
        self.assertEqual(record["best_gt_normalized_obb_iou"], 1.0)
        self.assertTrue(record["would_independently_match_current_fn"])

    def test_suppressed_candidate_checks_unmatched_gt_even_when_not_best(self):
        matched = box(0, 0, 10, 10)
        unmatched = box(2, 0, 12, 10)
        truths = (
            GroundTruth(0, matched, matched, matched, {}),
            GroundTruth(1, unmatched, unmatched, unmatched, {}),
        )
        suppression = {
            "suppressions": [
                {
                    "suppressed_detection_id": "suppressed",
                    "kept_detection_id": "kept",
                    "suppressed_global_pixel_obb_vertices": [
                        [0, 0], [10, 0], [10, 10], [0, 10]
                    ],
                }
            ]
        }
        result = _build_nms_diagnostics(
            suppression, self.official_matches, (), truths, (1,), 0.50
        )
        record = result["records"][0]
        self.assertEqual(record["best_gt_index"], 0)
        self.assertEqual(record["best_current_fn_gt_index"], 1)
        self.assertGreater(record["best_current_fn_normalized_obb_iou"], 0.50)
        self.assertTrue(record["would_independently_match_current_fn"])

    def test_evaluator_compares_rejected_geometry_without_changing_metrics(self):
        prediction = box(0, 0, 10, 10)
        before = greedy_confidence_match([prediction], [0.9], [item.obb_pixel_polygon for item in self.truths], 0.50)
        rejection = {
            "rejections": [
                {
                    "detection_id": "rejected",
                    "attempted_global_polygon_valid": True,
                    "attempted_global_pixel_obb_vertices": [
                        [20, 20], [30, 20], [30, 30], [20, 30]
                    ],
                }
            ]
        }
        diagnostic = _build_boundary_diagnostics(
            rejection, self.truths, before.false_gt_indices, 0.50
        )
        record = diagnostic["records"][0]
        self.assertTrue(record["geometry_reconstructable"])
        self.assertEqual(record["best_gt_index"], 1)
        self.assertTrue(record["would_independently_match_current_fn"])
        after = greedy_confidence_match([prediction], [0.9], [item.obb_pixel_polygon for item in self.truths], 0.50)
        self.assertEqual(before, after)
        self.assertEqual((len(after.matches), len(after.false_prediction_indices), len(after.false_gt_indices)), (1, 0, 1))

    def test_evaluator_does_not_fabricate_invalid_rejected_geometry(self):
        rejection = {
            "rejections": [
                {
                    "detection_id": "invalid",
                    "attempted_global_polygon_valid": False,
                    "attempted_global_pixel_obb_vertices": [
                        [1, 1], [2, 2], [3, 3], [4, 4]
                    ],
                }
            ]
        }
        diagnostic = _build_boundary_diagnostics(
            rejection, self.truths, (1,), 0.50
        )
        record = diagnostic["records"][0]
        self.assertFalse(record["geometry_reconstructable"])
        self.assertIsNone(record["best_gt_index"])
        self.assertIsNone(record["best_gt_normalized_obb_iou"])


class OutputTests(unittest.TestCase):
    def test_structured_artifacts_are_written(self):
        result = EvaluationResult(
            summary={"scene_id": "scene", "tp": 1, "fp": 1, "fn": 1},
            matches=(
                Match("prediction-1", 0, 0.9, 0.8, 0.6, 2.0, 0.4, "tile-1"),
            ),
            false_positives=(FalsePositive("prediction-2", 0.4, 0.2, 1, "tile-2"),),
            false_negatives=(
                FalseNegative(
                    2,
                    "Small Civil Transport/Utility",
                    1,
                    (10.0, 20.0),
                    (5.0, 50.0),
                    7.0,
                    11.0,
                    {"role": "Small Civil Transport/Utility"},
                ),
            ),
            nms_diagnostics={"available": True},
            boundary_rejection_diagnostics={"available": True},
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_evaluation_outputs(result, Path(directory))
            self.assertEqual(len(paths), 6)
            self.assertTrue(all(path.is_file() for path in paths))
            with paths[0].open(encoding="utf-8") as source:
                self.assertEqual(json.load(source)["scene_id"], "scene")


if __name__ == "__main__":
    unittest.main()
