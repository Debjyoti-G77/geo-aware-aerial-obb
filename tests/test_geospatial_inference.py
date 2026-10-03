from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from affine import Affine
from shapely import affinity
from shapely.geometry import Polygon, box


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from geospatial_inference import (  # noqa: E402
    _suppression_dict,
    DetectionRejected,
    GeospatialInferenceError,
    RawOBBDetection,
    SceneInferenceSettings,
    add_geographic_geometry,
    build_rejected_detection,
    build_feature_collection,
    extract_ultralytics_obb,
    greedy_polygon_nms,
    partition_target_detections,
    polygon_iou,
    raw_detection_to_global,
    resolve_target_class,
    validate_feature_collection,
    validate_model_metadata,
)
from geospatial_tiling import TileWindow  # noqa: E402


def make_tile(tile_id: str, col_off: int = 0, row_off: int = 0) -> TileWindow:
    return TileWindow(
        scene_id="synthetic_scene",
        tile_id=tile_id,
        row_off=row_off,
        col_off=col_off,
        width=400,
        height=400,
        tile_size=400,
        source_width=1000,
        source_height=1000,
        row_index=0,
        col_index=0,
    )


def make_raw(
    detection_id: str,
    vertices,
    *,
    class_id: int = 0,
    class_name: str = "plane",
    confidence: float = 0.9,
) -> RawOBBDetection:
    return RawOBBDetection(
        detection_id=detection_id,
        detection_index=0,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        local_obb_vertices=tuple(tuple(map(float, point)) for point in vertices),
    )


class FakeOBB:
    def __init__(self):
        self.xyxyxyxy = np.array(
            [
                [[10.25, 20.5], [50.75, 21.0], [49.5, 60.25], [9.0, 59.75]],
                [[100.0, 100.0], [130.0, 100.0], [130.0, 130.0], [100.0, 130.0]],
            ],
            dtype=np.float32,
        )
        self.conf = np.array([0.91, 0.73], dtype=np.float32)
        self.cls = np.array([0.0, 1.0], dtype=np.float32)


class FakeResult:
    obb = FakeOBB()


class FakeModel:
    def __init__(self, task, names):
        self.task = task
        self.names = names


class ModelCompatibilityTests(unittest.TestCase):
    def test_valid_18_class_obb_model_resolves_plane(self):
        names = {0: "plane", **{index: f"class-{index}" for index in range(1, 18)}}
        loaded = validate_model_metadata(
            FakeModel("obb", names), Path("best.pt")
        )
        self.assertEqual(len(loaded.names), 18)
        self.assertEqual(resolve_target_class("plane", loaded.names), (0, "plane"))

    def test_valid_one_class_obb_model_resolves_plane(self):
        loaded = validate_model_metadata(
            FakeModel("obb", {0: "plane"}), Path("rareplanes.pt")
        )
        self.assertEqual(loaded.names, {0: "plane"})
        self.assertEqual(resolve_target_class("plane", loaded.names), (0, "plane"))

    def test_missing_requested_target_class_fails_clearly(self):
        loaded = validate_model_metadata(
            FakeModel("obb", {0: "ship", 1: "helicopter"}), Path("other.pt")
        )
        with self.assertRaisesRegex(
            GeospatialInferenceError,
            "Target class 'plane' is not present exactly once",
        ):
            resolve_target_class("plane", loaded.names)

    def test_non_obb_model_remains_incompatible(self):
        with self.assertRaisesRegex(
            GeospatialInferenceError, "Incompatible model task 'detect'; expected 'obb'"
        ):
            validate_model_metadata(
                FakeModel("detect", {0: "plane"}), Path("detect.pt")
            )

    def test_model_without_class_metadata_remains_incompatible(self):
        with self.assertRaisesRegex(GeospatialInferenceError, "no classes defined"):
            validate_model_metadata(FakeModel("obb", {}), Path("empty.pt"))


class ExtractionAndConversionTests(unittest.TestCase):
    def test_extracts_float_pixel_vertices_from_ultralytics_obb_property(self):
        tile = make_tile("tile_a", col_off=100, row_off=200)
        detections = extract_ultralytics_obb(
            FakeResult(), {0: "plane", 1: "ship"}, tile
        )
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0].class_name, "plane")
        self.assertAlmostEqual(detections[0].local_obb_vertices[0][0], 10.25)

        targets, discarded = partition_target_detections(detections, 0)
        self.assertEqual(len(targets), 1)
        self.assertEqual(len(discarded), 1)
        candidate = raw_detection_to_global(targets[0], tile, "EPSG:4326")
        self.assertAlmostEqual(candidate.global_pixel_obb_vertices[0][0], 110.25)
        self.assertAlmostEqual(candidate.global_pixel_obb_vertices[0][1], 220.5)
        self.assertEqual(candidate.tile_id, "tile_a")

    def test_plane_filter_retains_only_class_zero(self):
        detections = (
            make_raw("plane", ((1, 1), (2, 1), (2, 2), (1, 2))),
            make_raw(
                "ship",
                ((3, 3), (4, 3), (4, 4), (3, 4)),
                class_id=1,
                class_name="ship",
            ),
        )
        targets, discarded = partition_target_detections(detections, 0)
        self.assertEqual([item.detection_id for item in targets], ["plane"])
        self.assertEqual([item.detection_id for item in discarded], ["ship"])

    def test_degenerate_and_padded_polygons_are_rejected(self):
        tile = make_tile("tile_a")
        degenerate = make_raw("degenerate", ((1, 1), (2, 2), (3, 3), (4, 4)))
        with self.assertRaises(DetectionRejected):
            raw_detection_to_global(degenerate, tile, "EPSG:4326")

        padded = make_raw("padded", ((390, 390), (410, 390), (410, 410), (390, 410)))
        with self.assertRaisesRegex(
            DetectionRejected, "vertex_outside_valid_tile_region"
        ):
            raw_detection_to_global(padded, tile, "EPSG:4326")

    def test_rejected_candidate_retains_safe_attempted_geometry(self):
        tile = make_tile("tile_a", col_off=100, row_off=200)
        raw = make_raw(
            "outside",
            ((390, 390), (410, 390), (410, 410), (390, 410)),
        )
        rejected = build_rejected_detection(
            raw, tile, "vertex_outside_valid_tile_region"
        )
        self.assertEqual(rejected.scene_id, "synthetic_scene")
        self.assertEqual(rejected.local_obb_vertices, raw.local_obb_vertices)
        self.assertEqual(
            rejected.attempted_global_pixel_obb_vertices[0], (490.0, 590.0)
        )
        self.assertTrue(rejected.attempted_global_polygon_valid)
        self.assertEqual(rejected.tile_valid_region, (0.0, 0.0, 400.0, 400.0))
        self.assertEqual(
            rejected.tile_source_image_bounds, (100.0, 200.0, 500.0, 600.0)
        )
        self.assertEqual(rejected.rejection_reason, "vertex_outside_valid_tile_region")

    def test_rejected_invalid_geometry_is_not_marked_reconstructable(self):
        tile = make_tile("tile_a", col_off=10, row_off=20)
        raw = make_raw("degenerate", ((1, 1), (2, 2), (3, 3), (4, 4)))
        rejected = build_rejected_detection(raw, tile, "invalid_global_polygon")
        self.assertEqual(rejected.local_obb_vertices, raw.local_obb_vertices)
        self.assertIsNotNone(rejected.attempted_global_pixel_obb_vertices)
        self.assertFalse(rejected.attempted_global_polygon_valid)


class PolygonNMSTests(unittest.TestCase):
    def test_overlapping_tile_duplicate_keeps_higher_confidence(self):
        first_tile = make_tile("tile_a", col_off=0)
        second_tile = make_tile("tile_b", col_off=50)
        first = raw_detection_to_global(
            make_raw("high", ((100, 100), (200, 100), (200, 200), (100, 200)), confidence=0.95),
            first_tile,
            "EPSG:4326",
        )
        second = raw_detection_to_global(
            make_raw("low", ((52, 102), (152, 102), (152, 202), (52, 202)), confidence=0.80),
            second_tile,
            "EPSG:4326",
        )
        result = greedy_polygon_nms((second, first), threshold=0.50)
        self.assertEqual([item.detection_id for item in result.kept], ["high"])
        self.assertEqual(len(result.suppressed), 1)
        record = result.suppressed[0]
        self.assertEqual(record.suppressed_detection_id, "low")
        self.assertEqual(record.kept_detection_id, "high")
        self.assertEqual(record.suppressed_tile_id, "tile_b")
        self.assertEqual(record.kept_tile_id, "tile_a")
        self.assertGreater(record.polygon_iou, 0.90)
        self.assertEqual(
            record.suppressed_local_obb_vertices, second.local_obb_vertices
        )
        self.assertEqual(
            record.suppressed_global_pixel_obb_vertices,
            second.global_pixel_obb_vertices,
        )
        self.assertEqual(
            record.kept_global_pixel_obb_vertices,
            first.global_pixel_obb_vertices,
        )
        serialized = _suppression_dict(
            record, Affine(0.001, 0, 5, 0, -0.001, 50), "EPSG:4326"
        )
        self.assertEqual(
            serialized["suppressed_geographic_crs84_polygon"]["type"], "Polygon"
        )
        self.assertEqual(
            serialized["kept_geographic_crs84_polygon"]["type"], "Polygon"
        )

    def test_nearby_distinct_aircraft_remain_separate(self):
        tile = make_tile("tile")
        first = raw_detection_to_global(
            make_raw("one", ((20, 20), (80, 20), (80, 50), (20, 50))),
            tile,
            "EPSG:4326",
        )
        second = raw_detection_to_global(
            make_raw("two", ((90, 20), (150, 20), (150, 50), (90, 50))),
            tile,
            "EPSG:4326",
        )
        result = greedy_polygon_nms((first, second), threshold=0.50)
        self.assertEqual(len(result.kept), 2)
        self.assertEqual(len(result.suppressed), 0)

    def test_overlapping_different_classes_are_not_compared_as_duplicates(self):
        tile = make_tile("tile")
        vertices = ((20, 20), (80, 20), (80, 50), (20, 50))
        plane = raw_detection_to_global(
            make_raw("plane", vertices, class_id=0, class_name="plane"),
            tile,
            "EPSG:4326",
        )
        ship = raw_detection_to_global(
            make_raw(
                "ship",
                vertices,
                class_id=1,
                class_name="ship",
                confidence=0.8,
            ),
            tile,
            "EPSG:4326",
        )
        result = greedy_polygon_nms((plane, ship), threshold=0.50)
        self.assertEqual(len(result.kept), 2)
        self.assertEqual(len(result.suppressed), 0)

    def test_rotated_polygon_iou_is_not_axis_aligned_iou(self):
        base = box(-100, -10, 100, 10)
        first_shape = affinity.translate(affinity.rotate(base, 45), 200, 200)
        second_shape = affinity.translate(affinity.rotate(base, -45), 200, 200)
        first_vertices = list(first_shape.exterior.coords)[:4]
        second_vertices = list(second_shape.exterior.coords)[:4]
        tile = make_tile("tile")
        first = raw_detection_to_global(
            make_raw("slash", first_vertices, confidence=0.9), tile, "EPSG:4326"
        )
        second = raw_detection_to_global(
            make_raw("backslash", second_vertices, confidence=0.8), tile, "EPSG:4326"
        )
        bounds_iou = polygon_iou(box(*first.global_pixel_polygon.bounds), box(*second.global_pixel_polygon.bounds))
        oriented_iou = polygon_iou(
            first.global_pixel_polygon, second.global_pixel_polygon
        )
        self.assertGreater(bounds_iou, 0.99)
        self.assertLess(oriented_iou, 0.50)
        result = greedy_polygon_nms((first, second), threshold=0.50)
        self.assertEqual(len(result.kept), 2)

    def test_zero_overlap_is_not_suppressed(self):
        self.assertEqual(
            polygon_iou(
                Polygon(((0, 0), (1, 0), (1, 1), (0, 1))),
                Polygon(((10, 10), (11, 10), (11, 11), (10, 11))),
            ),
            0.0,
        )

    def test_invalid_polygon_iou_is_safe(self):
        bow_tie = Polygon(((0, 0), (2, 2), (0, 2), (2, 0)))
        valid = Polygon(((0, 0), (3, 0), (3, 3), (0, 3)))
        self.assertFalse(bow_tie.is_valid)
        self.assertEqual(polygon_iou(bow_tie, valid), 0.0)


class GeoJSONTests(unittest.TestCase):
    def test_geojson_is_closed_lon_lat_and_contains_provenance(self):
        tile = make_tile("tile_geo")
        candidate = raw_detection_to_global(
            make_raw("geo", ((100, 100), (200, 100), (200, 200), (100, 200))),
            tile,
            "EPSG:4326",
        )
        transform = Affine.translation(-75.0, 40.0) @ Affine.scale(0.001, -0.001)
        detection = add_geographic_geometry(candidate, transform, "EPSG:4326")
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = SceneInferenceSettings(
                model_path=Path(temp_dir) / "best.pt",
                raster_path=Path(temp_dir) / "scene.tif",
                output_dir=Path(temp_dir) / "output",
            )
            collection = build_feature_collection(
                (detection,),
                model_name="best.pt",
                model_task="obb",
                settings=settings,
            )

        self.assertNotIn("crs", collection)
        feature = collection["features"][0]
        self.assertEqual(feature["geometry"]["type"], "Polygon")
        ring = feature["geometry"]["coordinates"][0]
        self.assertEqual(len(ring), 5)
        self.assertEqual(ring[0], ring[-1])
        self.assertAlmostEqual(ring[0][0], -74.9)
        self.assertAlmostEqual(ring[0][1], 39.9)
        required = {
            "detection_id",
            "scene_id",
            "class_id",
            "class_name",
            "confidence",
            "source_tile_id",
            "model_name",
            "model_task",
            "inference_conf",
            "inference_iou",
            "tile_size",
            "tile_overlap",
            "global_nms_iou",
        }
        self.assertTrue(required.issubset(feature["properties"]))
        footprint = Polygon(((-75, 40), (-74, 40), (-74, 39), (-75, 39)))
        validation = validate_feature_collection(collection, 1, footprint)
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["feature_count"], 1)
        json.dumps(collection, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
