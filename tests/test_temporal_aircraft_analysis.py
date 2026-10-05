from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest

from pyproj import Transformer
from shapely.geometry import Polygon, box, mapping
from shapely.ops import transform


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from temporal_aircraft_analysis import (  # noqa: E402
    CorrespondenceParameters,
    TemporalAnalysisError,
    TemporalDetection,
    TemporalScene,
    analyze_temporal_scenes,
    build_activity_summary,
    choose_local_metric_crs,
    discover_compatible_output,
    load_temporal_detections,
    match_consecutive_detections,
    parse_acquisition_time,
    resolve_or_generate_selected_outputs,
    select_temporal_validation_location,
    serialize_correspondences,
    serialize_presence_changes,
    serialize_temporal_detections,
    validate_inference_summary,
    validate_temporal_scenes,
    write_temporal_outputs,
)


UTC = timezone.utc
T0 = datetime(2020, 1, 1, tzinfo=UTC)
TO_METRIC = Transformer.from_crs("OGC:CRS84", "EPSG:32618", always_xy=True)
TO_GEOGRAPHIC = Transformer.from_crs("EPSG:32618", "OGC:CRS84", always_xy=True)
BASE_X, BASE_Y = TO_METRIC.transform(-75.0, 40.0)


def detection(
    detection_id: str,
    scene_id: str,
    time: datetime,
    x_offset: float,
    *,
    y_offset: float = 0.0,
    size: float = 20.0,
) -> TemporalDetection:
    projected = box(
        BASE_X + x_offset,
        BASE_Y + y_offset,
        BASE_X + x_offset + size,
        BASE_Y + y_offset + size,
    )
    geographic = transform(TO_GEOGRAPHIC.transform, projected)
    return TemporalDetection(
        detection_id=detection_id,
        location_id=84,
        scene_id=scene_id,
        acquisition_time=time,
        confidence=0.8,
        geographic_polygon=geographic,
        projected_polygon=projected,
    )


def write_predictions(path: Path, scene_id: str, items: list[TemporalDetection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": item.detection_id,
                "geometry": mapping(item.geographic_polygon),
                "properties": {
                    "detection_id": item.detection_id,
                    "scene_id": scene_id,
                    "confidence": item.confidence,
                },
            }
            for item in items
        ],
    }
    path.write_text(json.dumps(collection), encoding="utf-8")


class TemporalAircraftAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = CorrespondenceParameters(100.0, 15.0, "test", 20.0)

    def scene(self, scene_id: str, time: datetime, path: Path = Path("predictions.geojson")) -> TemporalScene:
        return TemporalScene(84, scene_id, time, path)

    def test_same_location_validation(self) -> None:
        scenes = (self.scene("a", T0), self.scene("b", T0 + timedelta(days=1)))
        self.assertEqual(validate_temporal_scenes(scenes), scenes)

    def test_generic_analysis_is_not_hard_locked_to_frozen_validation(self) -> None:
        scenes = (
            TemporalScene(999, "a", T0, Path("a.geojson")),
            TemporalScene(999, "b", T0 + timedelta(days=1), Path("b.geojson")),
        )
        self.assertEqual(validate_temporal_scenes(scenes), scenes)

    def test_mixed_locations_are_rejected(self) -> None:
        scenes = (
            self.scene("a", T0),
            TemporalScene(10, "b", T0 + timedelta(days=1), Path("b.geojson")),
        )
        with self.assertRaisesRegex(TemporalAnalysisError, "same location"):
            validate_temporal_scenes(scenes)

    def test_chronological_order_is_required(self) -> None:
        scenes = (self.scene("b", T0 + timedelta(days=1)), self.scene("a", T0))
        with self.assertRaisesRegex(TemporalAnalysisError, "chronological"):
            validate_temporal_scenes(scenes)

    def test_geographic_to_metric_crs(self) -> None:
        first = detection("a", "a", T0, 0.0)
        second = detection("b", "b", T0, 100.0)
        crs = choose_local_metric_crs([first.geographic_polygon, second.geographic_polygon])
        self.assertEqual(crs.to_epsg(), 32618)
        projected = transform(
            Transformer.from_crs("OGC:CRS84", crs, always_xy=True).transform,
            second.geographic_polygon,
        )
        self.assertAlmostEqual(projected.centroid.x - first.projected_polygon.centroid.x, 100.0, places=3)

    def test_one_to_one_matching_prefers_lower_cost(self) -> None:
        source = [detection("s1", "a", T0, 0.0), detection("s2", "a", T0, 80.0)]
        destination = [detection("d1", "b", T0 + timedelta(days=1), 3.0), detection("d2", "b", T0 + timedelta(days=1), 83.0)]
        matches, appeared, disappeared = match_consecutive_detections(source, destination, self.parameters)
        self.assertEqual(
            {(item.source_detection_id, item.destination_detection_id) for item in matches},
            {("s1", "d1"), ("s2", "d2")},
        )
        self.assertFalse(appeared)
        self.assertFalse(disappeared)

    def test_persistent_classification(self) -> None:
        matches, _, _ = match_consecutive_detections(
            [detection("s", "a", T0, 0.0)],
            [detection("d", "b", T0 + timedelta(days=1), 5.0)],
            self.parameters,
        )
        self.assertEqual(matches[0].category, "persistent")

    def test_displaced_classification(self) -> None:
        matches, _, _ = match_consecutive_detections(
            [detection("s", "a", T0, 0.0)],
            [detection("d", "b", T0 + timedelta(days=1), 40.0)],
            self.parameters,
        )
        self.assertEqual(matches[0].category, "displaced")

    def test_appeared_classification(self) -> None:
        destination = [detection("d", "b", T0 + timedelta(days=1), 0.0)]
        matches, appeared, disappeared = match_consecutive_detections([], destination, self.parameters)
        self.assertFalse(matches)
        self.assertEqual([item.detection_id for item in appeared], ["d"])
        self.assertFalse(disappeared)

    def test_disappeared_classification(self) -> None:
        source = [detection("s", "a", T0, 0.0)]
        matches, appeared, disappeared = match_consecutive_detections(source, [], self.parameters)
        self.assertFalse(matches)
        self.assertFalse(appeared)
        self.assertEqual([item.detection_id for item in disappeared], ["s"])

    def test_no_duplicate_assignments(self) -> None:
        source = [detection("s", "a", T0, 0.0)]
        destination = [
            detection("d1", "b", T0 + timedelta(days=1), 1.0),
            detection("d2", "b", T0 + timedelta(days=1), 2.0),
        ]
        matches, appeared, _ = match_consecutive_detections(source, destination, self.parameters)
        self.assertEqual(len(matches), 1)
        self.assertEqual(len({item.source_detection_id for item in matches}), 1)
        self.assertEqual(len(appeared), 1)

    def test_matching_is_deterministic(self) -> None:
        source = [detection("s2", "a", T0, 40.0), detection("s1", "a", T0, 0.0)]
        destination = [detection("d2", "b", T0 + timedelta(days=1), 42.0), detection("d1", "b", T0 + timedelta(days=1), 2.0)]
        first = match_consecutive_detections(source, destination, self.parameters)
        second = match_consecutive_detections(tuple(reversed(source)), tuple(reversed(destination)), self.parameters)
        self.assertEqual(first, second)

    def test_geojson_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = detection("a1", "a", T0, 0.0)
            b = detection("b1", "b", T0 + timedelta(days=1), 5.0)
            write_predictions(root / "a.geojson", "a", [a])
            write_predictions(root / "b.geojson", "b", [b])
            result = analyze_temporal_scenes(
                (self.scene("a", T0, root / "a.geojson"), self.scene("b", T0 + timedelta(days=1), root / "b.geojson")),
                max_distance_m=100.0,
                persistent_distance_m=15.0,
            )
            for collection in (
                serialize_temporal_detections(result),
                serialize_correspondences(result),
                serialize_presence_changes(result),
            ):
                self.assertEqual(collection["type"], "FeatureCollection")
                self.assertNotIn("crs", collection)
            outputs = write_temporal_outputs(result, root / "out")
            self.assertTrue(all(path.is_file() for path in outputs.values()))

    def test_empty_detection_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_predictions(root / "a.geojson", "a", [])
            write_predictions(root / "b.geojson", "b", [detection("b1", "b", T0 + timedelta(days=1), 0.0)])
            scenes = (self.scene("a", T0, root / "a.geojson"), self.scene("b", T0 + timedelta(days=1), root / "b.geojson"))
            _, loaded = load_temporal_detections(scenes)
            self.assertEqual(loaded["a"], ())
            result = analyze_temporal_scenes(scenes, max_distance_m=100.0, persistent_distance_m=15.0)
            self.assertEqual(len(result.pairs[0].appeared), 1)

    def test_two_timestamp_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_predictions(root / "a.geojson", "a", [detection("a1", "a", T0, 0.0)])
            write_predictions(root / "b.geojson", "b", [detection("b1", "b", T0 + timedelta(days=1), 5.0)])
            result = analyze_temporal_scenes(
                (self.scene("a", T0, root / "a.geojson"), self.scene("b", T0 + timedelta(days=1), root / "b.geojson")),
                max_distance_m=100.0,
                persistent_distance_m=15.0,
            )
            summary = build_activity_summary(result)
            self.assertEqual(summary["detection_count_trajectory"], [1, 1])
            self.assertEqual(summary["persistent_correspondences"], 1)
            self.assertEqual(summary["pair_statistics"][0]["matched_total"], 1)
            self.assertTrue(summary["pair_statistics"][0]["conservation"]["earlier_valid"])
            self.assertTrue(summary["pair_statistics"][0]["conservation"]["later_valid"])
            self.assertIsNone(result.activity_grid)

    def test_three_timestamp_analysis_builds_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenes = []
            for index, offset in enumerate((0.0, 5.0, 10.0)):
                scene_id = chr(ord("a") + index)
                timestamp = T0 + timedelta(days=index)
                path = root / f"{scene_id}.geojson"
                write_predictions(path, scene_id, [detection(f"{scene_id}1", scene_id, timestamp, offset)])
                scenes.append(self.scene(scene_id, timestamp, path))
            result = analyze_temporal_scenes(scenes, max_distance_m=100.0, persistent_distance_m=15.0)
            self.assertEqual(len(result.pairs), 2)
            self.assertIsNotNone(result.activity_grid)
            self.assertGreaterEqual(len(result.activity_grid["features"]), 1)

    def test_metadata_only_selection_maximizes_timestamp_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            imagery = root / "imagery"
            imagery.mkdir()
            rows = []
            for loc_id, count in ((10, 2), (84, 3), (23, 2)):
                for index in range(count):
                    scene_id = f"{loc_id}_scene_{index}"
                    (imagery / f"{scene_id}.tif").touch()
                    rows.append(
                        {
                            "image_id": scene_id,
                            "loc_id": loc_id,
                            "Train": "1",
                            "collection_time_start": f"2020-01-{index + 1:02d}T00:00:00Z",
                        }
                    )
            rows.append({"image_id": "84_test", "loc_id": 84, "Train": "0", "collection_time_start": "2020-01-04T00:00:00Z"})
            metadata = root / "metadata.csv"
            with metadata.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            split_manifest = root / "split_scenes.csv"
            with split_manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("scene_id", "loc_id", "split"))
                writer.writeheader()
                for row in rows[:-1]:
                    writer.writerow(
                        {
                            "scene_id": row["image_id"],
                            "loc_id": row["loc_id"],
                            "split": "val",
                        }
                    )
            selected, scenes, rule = select_temporal_validation_location(
                metadata,
                imagery,
                split_manifest_path=split_manifest,
                raster_validator=lambda _: True,
            )
            self.assertEqual(selected, 84)
            self.assertEqual(len(scenes), 3)
            self.assertFalse(rule["performance_data_used"])

    def test_inference_summary_policy_validation_and_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene_id = "84_scene"
            output = root / scene_id
            output.mkdir()
            summary = {
                "scene_id": scene_id,
                "model_name": "rareplanes_finetuned_best.pt",
                "model_task": "obb",
                "inference_confidence": 0.25,
                "inference_iou": 0.45,
                "global_nms_iou": 0.50,
                "image_size": 1024,
                "tile_size": 1024,
                "tile_overlap": 0.20,
                "target_class_name": "plane",
                "device": "cpu",
            }
            validate_inference_summary(summary, scene_id)
            (output / "inference_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (output / "predictions.geojson").write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
            (root / "completed.json").write_text(
                json.dumps(
                    {
                        "model_sha256": "4111cdab67bcc3c53623e4bd08aeb5943e4a49225c27c62172036b815991eced",
                        "settings": {"boundary_policy": "strict_all_obb_vertices_inside_valid_source_region"},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(discover_compatible_output(root, scene_id), output / "predictions.geojson")
            summary["inference_confidence"] = 0.10
            with self.assertRaisesRegex(TemporalAnalysisError, "incompatible"):
                validate_inference_summary(summary, scene_id)

    def test_output_resolver_reuses_compatible_results_without_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene_rows = []
            for index, scene_id in enumerate(("84_a", "84_b")):
                scene_output = root / "prior" / scene_id
                scene_output.mkdir(parents=True)
                summary = {
                    "scene_id": scene_id,
                    "model_name": "rareplanes_finetuned_best.pt",
                    "model_task": "obb",
                    "inference_confidence": 0.25,
                    "inference_iou": 0.45,
                    "global_nms_iou": 0.50,
                    "image_size": 1024,
                    "tile_size": 1024,
                    "tile_overlap": 0.20,
                    "target_class_name": "plane",
                    "device": "cpu",
                }
                (scene_output / "inference_summary.json").write_text(json.dumps(summary), encoding="utf-8")
                (scene_output / "predictions.geojson").write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
                (scene_output.parent / "completed.json").write_text(
                    json.dumps(
                        {
                            "model_sha256": "4111cdab67bcc3c53623e4bd08aeb5943e4a49225c27c62172036b815991eced",
                            "settings": {"boundary_policy": "strict_all_obb_vertices_inside_valid_source_region"},
                        }
                    ),
                    encoding="utf-8",
                )
                raster = root / f"{scene_id}.tif"
                raster.touch()
                scene_rows.append(
                    {
                        "scene_id": scene_id,
                        "acquisition_time": T0 + timedelta(days=index),
                        "raster_path": raster,
                    }
                )
            scenes = resolve_or_generate_selected_outputs(
                scene_rows,
                location_id=84,
                predictions_root=root / "prior",
                generated_root=root / "new",
                model_path=root / "missing.pt",
            )
            self.assertEqual([scene.source_status for scene in scenes], ["reused", "reused"])
            self.assertFalse((root / "new").exists())

    def test_timestamp_parsing_normalizes_utc(self) -> None:
        self.assertEqual(parse_acquisition_time("2020-01-01T00:00:00Z"), T0)
        self.assertEqual(
            parse_acquisition_time("1/30/17 16:12"),
            datetime(2017, 1, 30, 16, 12, tzinfo=UTC),
        )


if __name__ == "__main__":
    unittest.main()
