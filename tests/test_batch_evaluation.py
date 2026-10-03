from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from batch_evaluation import (  # noqa: E402
    BatchEvaluationError,
    REQUIRED_SCENE_ID,
    SceneRecord,
    aggregate_distribution,
    aggregate_macro,
    aggregate_micro,
    aggregate_pipeline_diagnostics,
    aggregate_size_recall,
    deterministic_select_train_scenes,
    percentile,
    read_selection_manifest,
    write_selection_manifest,
)


def make_scene(index: int, *, split: str = "train") -> SceneRecord:
    scene_id = REQUIRED_SCENE_ID if index == 0 else f"{index}_CAT{index:04d}"
    weather = ("Clear Skies", "Snow", "Cloud Cover or Haze")[index % 3]
    return SceneRecord(
        scene_id=scene_id,
        split=split,
        loc_id=index,
        image_catalog_id=scene_id.split("_", 1)[1],
        width=2000 + index * 137,
        height=1800 + index * 89,
        gt_aircraft_count=index * 3,
        country=f"country-{index % 5}",
        state_province=f"state-{index}",
        air_field=f"airfield-{index}",
        weather=weather,
        cloud_cover_percentage=float(index * 4),
        off_nadir_angle=5.0 + index,
        sensor="WV03",
        collection_time_start="2020-01-01",
        location_observation_count=1 + index % 4,
    )


class SelectionTests(unittest.TestCase):
    def test_selection_is_deterministic_and_train_only(self):
        records = tuple(make_scene(index) for index in range(20))
        first = deterministic_select_train_scenes(records)
        second = deterministic_select_train_scenes(tuple(reversed(records)))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual(len({item.scene_id for item in first}), 12)
        self.assertIn(REQUIRED_SCENE_ID, {item.scene_id for item in first})
        self.assertTrue(all(item.split == "train" for item in first))
        self.assertTrue(all(item.selection_rationale for item in first))

    def test_manifest_rejects_test_split(self):
        records = [make_scene(index) for index in range(12)]
        records[5] = make_scene(5, split="test")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            write_selection_manifest(records, path)
            with self.assertRaisesRegex(BatchEvaluationError, "non-TRAIN"):
                read_selection_manifest(path)


class AggregationTests(unittest.TestCase):
    def test_micro_metrics_pool_counts(self):
        metrics = (
            {"gt_count": 3, "prediction_count": 3, "tp": 2, "fp": 1, "fn": 1},
            {"gt_count": 2, "prediction_count": 1, "tp": 1, "fp": 0, "fn": 1},
        )
        result = aggregate_micro(metrics)
        self.assertEqual(result["gt"], 5)
        self.assertEqual(result["predictions"], 4)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (3, 1, 2))
        self.assertAlmostEqual(result["precision"], 0.75)
        self.assertAlmostEqual(result["recall"], 0.6)

    def test_macro_metrics_include_zero_denominators(self):
        metrics = (
            {"precision": 1.0, "recall": 0.5, "f1": 2 / 3},
            {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        )
        result = aggregate_macro(metrics)
        self.assertAlmostEqual(result["precision"]["mean"], 0.5)
        self.assertAlmostEqual(result["recall"]["median"], 0.25)
        self.assertIn("no scene is dropped", result["zero_denominator_policy"])

    def test_micro_empty_counts_are_zero_not_dropped(self):
        metrics = (
            {"gt_count": 0, "prediction_count": 0, "tp": 0, "fp": 0, "fn": 0},
        )
        result = aggregate_micro(metrics)
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["f1"], 0.0)

    def test_percentile_and_distribution(self):
        self.assertEqual(percentile([0, 10, 20, 30], 0.25), 7.5)
        self.assertAlmostEqual(percentile([0, 10, 20, 30], 0.95), 28.5)
        result = aggregate_distribution(
            [0.0, 10.0, 20.0, 30.0], ("mean", "median", "p25", "p75", "p95", "min", "max")
        )
        self.assertEqual(result["mean"], 15.0)
        self.assertEqual(result["p75"], 22.5)

    def test_size_category_recall(self):
        rows = (
            {"scene_id": "a", "gt_index": 0, "faa_wingspan_class": 1},
            {"scene_id": "a", "gt_index": 1, "faa_wingspan_class": 1},
            {"scene_id": "b", "gt_index": 0, "faa_wingspan_class": 2},
        )
        result = aggregate_size_recall(rows, {("a", 0), ("b", 0)})
        self.assertEqual(result["1"]["gt_count"], 2)
        self.assertEqual(result["1"]["matched_gt_count"], 1)
        self.assertEqual(result["1"]["recall"], 0.5)
        self.assertEqual(result["2"]["recall"], 1.0)

    def test_pipeline_diagnostic_aggregation(self):
        results = (
            SimpleNamespace(
                nms_diagnostics={
                    "suppressed_count": 4,
                    "suppressed_independently_matching_current_fn": 1,
                },
                boundary_rejection_diagnostics={
                    "rejected_count": 2,
                    "rejected_with_reconstructable_geometry": 2,
                    "rejected_independently_matching_current_fn": 0,
                },
            ),
            SimpleNamespace(
                nms_diagnostics={
                    "suppressed_count": 3,
                    "suppressed_independently_matching_current_fn": 0,
                },
                boundary_rejection_diagnostics={
                    "rejected_count": 1,
                    "rejected_with_reconstructable_geometry": 1,
                    "rejected_independently_matching_current_fn": 1,
                },
            ),
        )
        result = aggregate_pipeline_diagnostics(results)
        self.assertEqual(result["suppressed_candidates"], 7)
        self.assertEqual(result["suppressed_matching_current_fn"], 1)
        self.assertEqual(result["rejected_candidates"], 3)
        self.assertEqual(result["reconstructable_rejected_candidates"], 3)
        self.assertEqual(result["rejected_matching_current_fn"], 1)


if __name__ == "__main__":
    unittest.main()
