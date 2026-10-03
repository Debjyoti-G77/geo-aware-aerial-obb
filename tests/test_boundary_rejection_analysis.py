from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from boundary_rejection_analysis import (  # noqa: E402
    _global_detection,
    classify_rejected_geometry,
    duplicate_gt_accounting,
    policy_accepts,
    simulate_policy,
)


def rejected_record(
    local_vertices,
    *,
    row_off=100,
    col_off=100,
    width=100,
    height=100,
    detection_id="scene__r000100_c000100__d0000",
):
    global_vertices = [
        [x + col_off, y + row_off] for x, y in local_vertices
    ]
    return {
        "detection_id": detection_id,
        "scene_id": "scene",
        "tile_id": "scene__r000100_c000100",
        "class_id": 0,
        "class_name": "plane",
        "confidence": 0.8,
        "rejection_reason": "vertex_outside_valid_tile_region",
        "local_obb_vertices": local_vertices,
        "attempted_global_pixel_obb_vertices": global_vertices,
        "attempted_global_polygon_valid": True,
        "tile_valid_region": [0.0, 0.0, float(width), float(height)],
        "tile_source_image_bounds": [
            float(col_off),
            float(row_off),
            float(col_off + width),
            float(row_off + height),
        ],
        "row_off": row_off,
        "col_off": col_off,
    }


class GeometryClassificationTests(unittest.TestCase):
    def test_one_vertex_slightly_outside_with_centroid_inside(self):
        item = classify_rejected_geometry(
            rejected_record([[-1, 10], [40, 10], [40, 40], [10, 40]]),
            500,
            500,
            tile_size=100,
        )
        self.assertEqual(item.outside_vertex_count, 1)
        self.assertEqual(item.crossed_sides, ("left",))
        self.assertAlmostEqual(item.maximum_excursion_pixels, 1.0)
        self.assertTrue(item.centroid_inside_tile)
        self.assertEqual(item.boundary_category, "internal_tile_boundary")

    def test_two_vertices_outside_and_centroid_on_inclusive_boundary(self):
        item = classify_rejected_geometry(
            rejected_record([[-10, 10], [-10, 20], [10, 20], [10, 10]]),
            500,
            500,
            tile_size=100,
        )
        self.assertEqual(item.outside_vertex_count, 2)
        self.assertTrue(item.centroid_inside_tile)
        self.assertAlmostEqual(item.inside_tile_fraction, 0.5)

    def test_centroid_outside(self):
        item = classify_rejected_geometry(
            rejected_record([[-30, 10], [-30, 30], [-10, 30], [-10, 10]]),
            500,
            500,
            tile_size=100,
        )
        self.assertFalse(item.centroid_inside_tile)
        self.assertFalse(policy_accepts(item, "centroid"))

    def test_area_fraction_policy_thresholds(self):
        ninety = classify_rejected_geometry(
            rejected_record([[-10, 10], [90, 10], [90, 20], [-10, 20]]),
            500,
            500,
            tile_size=100,
        )
        seventy_five = classify_rejected_geometry(
            rejected_record([[-25, 10], [75, 10], [75, 20], [-25, 20]]),
            500,
            500,
            tile_size=100,
        )
        self.assertAlmostEqual(ninety.inside_tile_fraction, 0.9)
        self.assertTrue(policy_accepts(ninety, "area90"))
        self.assertAlmostEqual(seventy_five.inside_tile_fraction, 0.75)
        self.assertFalse(policy_accepts(seventy_five, "area90"))
        self.assertTrue(policy_accepts(seventy_five, "area75"))

    def test_source_raster_boundary_is_distinct_from_internal_boundary(self):
        internal = classify_rejected_geometry(
            rejected_record([[-5, 10], [10, 10], [10, 20], [-5, 20]]),
            500,
            500,
            tile_size=100,
        )
        source_record = rejected_record(
            [[-5, 10], [10, 10], [10, 20], [-5, 20]], row_off=0, col_off=0
        )
        source_record["tile_id"] = "scene__r000000_c000000"
        source = classify_rejected_geometry(source_record, 500, 500, tile_size=100)
        self.assertEqual(internal.boundary_category, "internal_tile_boundary")
        self.assertEqual(source.boundary_category, "source_raster_boundary")
        self.assertFalse(policy_accepts(source, "centroid"))


class AccountingAndSimulationTests(unittest.TestCase):
    def test_duplicate_gt_accounting(self):
        result = duplicate_gt_accounting([1, 2, 2, 3, 3, 3])
        self.assertEqual(result["unique_gt"], 3)
        self.assertEqual(result["with_one_candidate"], 1)
        self.assertEqual(result["with_multiple_candidates"], 2)
        self.assertEqual(result["candidates_per_gt_distribution"], {1: 1, 2: 1, 3: 1})

    def test_offline_policy_adds_candidate_without_changing_current(self):
        from shapely.geometry import Polygon

        accepted = _global_detection(
            detection_id="scene__r000000_c000000__d0000",
            scene_id="scene",
            tile_id="scene__r000000_c000000",
            confidence=0.9,
            local_vertices=[[10, 10], [30, 10], [30, 30], [10, 30]],
            global_vertices=[[10, 10], [30, 10], [30, 30], [10, 30]],
            source_width=500,
            source_height=500,
            tile_size=100,
        )
        rejected = classify_rejected_geometry(
            rejected_record(
                [[-1, 10], [19, 10], [19, 30], [-1, 30]],
                detection_id="scene__r000100_c000100__d0001",
            ),
            500,
            500,
            tile_size=100,
        )
        truths = (
            SimpleNamespace(
                obb_pixel_polygon=Polygon([[10, 10], [30, 10], [30, 30], [10, 30]])
            ),
            SimpleNamespace(
                obb_pixel_polygon=Polygon([[99, 110], [119, 110], [119, 130], [99, 130]])
            ),
        )
        current = simulate_policy((accepted,), (rejected,), truths, "current")
        centroid = simulate_policy((accepted,), (rejected,), truths, "centroid")
        self.assertEqual((current.predictions, current.tp, current.fp, current.fn), (1, 1, 0, 1))
        self.assertEqual((centroid.predictions, centroid.tp, centroid.fp, centroid.fn), (2, 2, 0, 0))
        self.assertEqual(current.admitted_rejected, 0)
        self.assertEqual(centroid.admitted_rejected, 1)


if __name__ == "__main__":
    unittest.main()
