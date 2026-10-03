from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from boundary_policy_validation import (  # noqa: E402
    DEVELOPMENT_LOC_IDS,
    DEVELOPMENT_SCENE_IDS,
    EXPECTED_SETTINGS,
    HOLDOUT_SIZE,
    HoldoutRecord,
    HoldoutValidationError,
    build_scene_command,
    calculate_policy_deltas,
    scene_output_complete,
    select_location_disjoint_holdout,
    unique_recovery_accounting,
    validate_holdout_selection,
)
from boundary_rejection_analysis import policy_accepts  # noqa: E402


def record(index: int, **changes: object) -> HoldoutRecord:
    value = HoldoutRecord(
        scene_id=f"{200 + index}_catalog{index:03d}",
        split="train",
        loc_id=200 + index,
        image_catalog_id=f"catalog{index:03d}",
        width=2000 + index * 100,
        height=1500 + index * 50,
        image_area_pixels=(2000 + index * 100) * (1500 + index * 50),
        gt_aircraft_count=index + 1,
        country=f"country-{index % 4}",
        state_province="state",
        air_field=f"airport-{index}",
        weather=("Clear Skies", "Snow", "Cloud Cover or Haze")[index % 3],
        cloud_cover_percentage=float(index * 3),
        off_nadir_angle=float(5 + index),
        sensor="WV03",
        collection_time_start="2020-01-01",
        location_observation_count=1,
        aircraft_size_composition=json.dumps({"1": index + 1, "3": index % 3}),
        selection_reason="",
    )
    return replace(value, **changes)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class BoundaryPolicyValidationTests(unittest.TestCase):
    def test_selection_is_deterministic_train_only_and_location_disjoint(self) -> None:
        records = tuple(record(index) for index in range(24))
        first = select_location_disjoint_holdout(records)
        second = select_location_disjoint_holdout(tuple(reversed(records)))
        self.assertEqual(first, second)
        self.assertEqual(len(first), HOLDOUT_SIZE)
        self.assertTrue(all(item.split == "train" for item in first))
        self.assertFalse({item.scene_id for item in first} & DEVELOPMENT_SCENE_IDS)
        self.assertFalse({item.loc_id for item in first} & DEVELOPMENT_LOC_IDS)
        self.assertEqual(len({item.loc_id for item in first}), HOLDOUT_SIZE)

    def test_selection_excludes_development_scene_and_location(self) -> None:
        records = [record(index) for index in range(20)]
        records.extend(
            (
                record(30, scene_id=next(iter(DEVELOPMENT_SCENE_IDS))),
                record(31, loc_id=next(iter(DEVELOPMENT_LOC_IDS))),
            )
        )
        selected = select_location_disjoint_holdout(records)
        self.assertNotIn(next(iter(DEVELOPMENT_SCENE_IDS)), {item.scene_id for item in selected})
        self.assertNotIn(next(iter(DEVELOPMENT_LOC_IDS)), {item.loc_id for item in selected})

    def test_selection_refuses_to_relax_location_exclusion(self) -> None:
        records = tuple(record(index, loc_id=300 + index % 11) for index in range(20))
        with self.assertRaisesRegex(HoldoutValidationError, "exclusion was not relaxed"):
            select_location_disjoint_holdout(records)

    def test_validate_rejects_non_train_and_development_overlap(self) -> None:
        records = tuple(record(index) for index in range(HOLDOUT_SIZE))
        with self.assertRaisesRegex(HoldoutValidationError, "non-TRAIN"):
            validate_holdout_selection((replace(records[0], split="test"), *records[1:]))
        with self.assertRaisesRegex(HoldoutValidationError, "overlap"):
            validate_holdout_selection(
                (replace(records[0], scene_id=next(iter(DEVELOPMENT_SCENE_IDS))), *records[1:])
            )

    def test_frozen_policy_definitions(self) -> None:
        cases = (
        ("current", 1.0, True, True, False),
        ("centroid", 0.1, True, True, True),
        ("centroid", 1.0, False, True, False),
        ("area90", 0.90, False, True, True),
        ("area90", 0.8999, True, True, False),
        ("area75", 0.75, False, True, True),
        ("area75", 1.0, True, False, False),
        )
        for policy, fraction, centroid, source, expected in cases:
            with self.subTest(policy=policy, fraction=fraction):
                candidate = SimpleNamespace(
                    attempted_global_polygon_valid=True,
                    global_polygon_inside_source=source,
                    centroid_inside_tile=centroid,
                    inside_tile_fraction=fraction,
                )
                self.assertIs(policy_accepts(candidate, policy), expected)

    def test_policy_delta_calculation(self) -> None:
        aggregate = {
        "current": {"predictions": 10, "tp": 6, "fp": 4, "fn": 4, "precision": 0.6, "recall": 0.6, "f1": 0.6},
        "centroid": {"predictions": 12, "tp": 7, "fp": 5, "fn": 3, "precision": 7 / 12, "recall": 0.7, "f1": 0.636},
        "area90": {"predictions": 11, "tp": 7, "fp": 4, "fn": 3, "precision": 7 / 11, "recall": 0.7, "f1": 0.66},
        "area75": {"predictions": 12, "tp": 8, "fp": 4, "fn": 2, "precision": 2 / 3, "recall": 0.8, "f1": 0.727},
        }
        rows = calculate_policy_deltas(aggregate)
        self.assertEqual([row["policy"] for row in rows], ["centroid", "area90", "area75"])
        self.assertEqual(rows[1]["delta_tp"], 1)
        self.assertEqual(rows[1]["delta_fp"], 0)
        self.assertEqual(rows[1]["delta_fn"], -1)

    def test_unique_recovery_counts_gt_identities_not_candidate_links(self) -> None:
        result = unique_recovery_accounting(
            current_false_gt={1, 2, 3, 4},
            alternative_false_gt={2, 4},
            represented_gt={1, 2, 3},
        )
        self.assertEqual(
            result,
            {
                "represented_policy_a_fn_gt": 3,
                "unique_fn_recovered": 2,
                "represented_remaining_fn": 1,
            },
        )

    def test_process_isolation_command_targets_exactly_one_scene(self) -> None:
        command = build_scene_command(
            "scene_one",
            Path("manifest.csv"),
            Path("imagery"),
            Path("annotations"),
            Path("best.pt"),
            Path("outputs"),
        )
        self.assertEqual(command[2], "run-scene")
        self.assertEqual(command.count("--scene-id"), 1)
        self.assertEqual(command[command.index("--scene-id") + 1], "scene_one")

    def test_incomplete_scene_output_is_not_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scene_id = "200_catalog"
            scene_root = Path(directory) / scene_id
            _write_json(scene_root / "inference" / "inference_summary.json", {"scene_id": scene_id})
            self.assertFalse(scene_output_complete(scene_root, scene_id))

    def test_complete_scene_requires_exact_frozen_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scene_id = "200_catalog"
            scene_root = Path(directory) / scene_id
            inference = scene_root / "inference"
            evaluation = scene_root / "evaluation"
            for name in (
                "predictions.geojson",
                "duplicate_suppressions.json",
                "rejected_candidates.json",
            ):
                _write_json(inference / name, {})
            _write_json(inference / "inference_summary.json", {"scene_id": scene_id, **EXPECTED_SETTINGS})
            for name in ("matches.csv", "false_positives.csv", "false_negatives.csv"):
                (evaluation / name).parent.mkdir(parents=True, exist_ok=True)
                (evaluation / name).write_text("", encoding="utf-8")
            _write_json(evaluation / "evaluation_summary.json", {"scene_id": scene_id, "match_iou_threshold": 0.5})
            _write_json(evaluation / "boundary_rejection_diagnostics.json", {})
            _write_json(evaluation / "nms_diagnostics.json", {})
            _write_json(scene_root / "completed.json", {"scene_id": scene_id, "status": "complete", "frozen_settings": EXPECTED_SETTINGS})
            self.assertTrue(scene_output_complete(scene_root, scene_id))
            changed = dict(EXPECTED_SETTINGS)
            changed["inference_confidence"] = 0.26
            _write_json(scene_root / "completed.json", {"scene_id": scene_id, "status": "complete", "frozen_settings": changed})
            self.assertFalse(scene_output_complete(scene_root, scene_id))

    def test_production_policy_constants_remain_frozen(self) -> None:
        self.assertEqual(EXPECTED_SETTINGS, {
        "tile_size": 1024,
        "tile_overlap": 0.20,
        "integer_stride": 819,
        "device": "cpu",
        "image_size": 1024,
        "inference_confidence": 0.25,
        "inference_iou": 0.45,
        "global_nms_iou": 0.50,
        "target_class_name": "plane",
        })


if __name__ == "__main__":
    unittest.main()
