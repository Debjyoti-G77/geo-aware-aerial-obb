from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

from shapely.geometry import Polygon


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from geospatial_inference import GlobalDetection, greedy_polygon_nms  # noqa: E402
from low_confidence_probe import (  # noqa: E402
    ALLOWED_SCENE_IDS,
    PROBE_CONFIDENCE,
    ProbeError,
    analyze_gt_responses,
    best_detection_response,
    confidence_band,
    enforce_fixed_protocol,
    response_classification,
    validate_single_scene,
)


def detection(
    detection_id: str,
    class_id: int,
    class_name: str,
    confidence: float,
    vertices: tuple[tuple[float, float], ...],
    tile_id: str = "scene__r000_c000",
) -> GlobalDetection:
    polygon = Polygon(vertices)
    return GlobalDetection(
        detection_id=detection_id,
        scene_id=ALLOWED_SCENE_IDS[0],
        tile_id=tile_id,
        tile_row_off=0,
        tile_col_off=0,
        tile_pixel_bounds=(0, 0, 1024, 1024),
        tile_valid_region=(0, 0, 1024, 1024),
        tile_size=1024,
        source_width=2048,
        source_height=2048,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        local_obb_vertices=vertices,  # type: ignore[arg-type]
        global_pixel_obb_vertices=vertices,  # type: ignore[arg-type]
        global_pixel_polygon=polygon,
        source_crs="EPSG:4326",
    )


class Truth:
    def __init__(self, index: int, polygon: Polygon):
        self.gt_index = index
        self.obb_pixel_polygon = polygon


class LowConfidenceProbeTests(unittest.TestCase):
    def test_fixed_confidence_and_exact_scene_enforcement(self) -> None:
        enforce_fixed_protocol(ALLOWED_SCENE_IDS, PROBE_CONFIDENCE, "train")
        with self.assertRaisesRegex(ProbeError, "exactly"):
            enforce_fixed_protocol(ALLOWED_SCENE_IDS[:2], PROBE_CONFIDENCE, "train")
        with self.assertRaisesRegex(ProbeError, "fixed"):
            enforce_fixed_protocol(ALLOWED_SCENE_IDS, 0.10, "train")

    def test_train_only_enforcement(self) -> None:
        with self.assertRaisesRegex(ProbeError, "TRAIN-only"):
            enforce_fixed_protocol(ALLOWED_SCENE_IDS, PROBE_CONFIDENCE, "test")
        with self.assertRaisesRegex(ProbeError, "TRAIN-only"):
            validate_single_scene(ALLOWED_SCENE_IDS[0], PROBE_CONFIDENCE, "validation")

    def test_plane_confidence_bands(self) -> None:
        expected = {
            0.05: "0.05-0.10",
            0.0999: "0.05-0.10",
            0.10: "0.10-0.15",
            0.15: "0.15-0.20",
            0.20: "0.20-0.25",
            0.25: ">=0.25",
            0.90: ">=0.25",
        }
        for value, band in expected.items():
            self.assertEqual(confidence_band(value), band)

    def test_best_plane_response_uses_iou_then_confidence(self) -> None:
        gt = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        lower_iou = detection("low-iou", 0, "plane", 0.99, ((0, 0), (8, 0), (8, 10), (0, 10)))
        higher_iou = detection("high-iou", 0, "plane", 0.06, ((0, 0), (9, 0), (9, 10), (0, 10)))
        best = best_detection_response(gt, (lower_iou, higher_iou))
        self.assertEqual(best["detection_id"], "high-iou")
        self.assertAlmostEqual(best["iou"], 0.9)

    def test_best_nonplane_response_preserves_class(self) -> None:
        gt = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        vehicle = detection("vehicle", 9, "small-vehicle", 0.12, ((0, 0), (10, 0), (10, 10), (0, 10)))
        response = best_detection_response(gt, (vehicle,))
        self.assertEqual(response["class_name"], "small-vehicle")
        self.assertEqual(response["class_id"], 9)

    def test_below_025_geometrically_valid_signal_count(self) -> None:
        gt = Truth(0, Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]))
        plane = detection("plane", 0, "plane", 0.12, ((0, 0), (10, 0), (10, 10), (0, 10)))
        rows, summary = analyze_gt_responses(ALLOWED_SCENE_IDS[0], (gt,), (plane,))
        self.assertTrue(rows[0]["below_0.25_geometrically_valid_plane"])
        self.assertEqual(summary["below_0.25_geometrically_valid_plane_count"], 1)

    def test_nonplane_class_aggregation(self) -> None:
        truths = (
            Truth(0, Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])),
            Truth(1, Polygon([(20, 0), (30, 0), (30, 10), (20, 10)])),
        )
        detections = (
            detection("v1", 9, "small-vehicle", 0.2, ((0, 0), (10, 0), (10, 10), (0, 10))),
            detection("v2", 9, "small-vehicle", 0.3, ((20, 0), (30, 0), (30, 10), (20, 10))),
        )
        _, summary = analyze_gt_responses(ALLOWED_SCENE_IDS[0], truths, detections)
        self.assertEqual(
            summary["nonplane_iou_ge_0.50_classes_for_gt_without_plane_match"],
            {"small-vehicle": 2},
        )

    def test_response_categories_are_mutually_exclusive(self) -> None:
        cases = {
            "NO_MEANINGFUL_RESPONSE": (0.1, 0.1),
            "PLANE_DOMINANT": (0.8, 0.2),
            "NON_PLANE_DOMINANT": (0.2, 0.8),
            "AMBIGUOUS": (0.60, 0.56),
        }
        for expected, (plane_iou, nonplane_iou) in cases.items():
            self.assertEqual(
                response_classification(plane_iou, 0.1, nonplane_iou, 0.2), expected
            )

    def test_raw_and_deduplicated_counts_are_distinct(self) -> None:
        first = detection("a", 0, "plane", 0.9, ((0, 0), (10, 0), (10, 10), (0, 10)))
        second = replace(first, detection_id="b", confidence=0.8, tile_id="scene__r000_c001")
        result = greedy_polygon_nms((first, second), 0.50)
        self.assertEqual(len((first, second)), 2)
        self.assertEqual(len(result.kept), 1)
        self.assertEqual(len(result.suppressed), 1)

    def test_production_configuration_remains_frozen(self) -> None:
        config = (REPOSITORY_ROOT / "configs" / "project.yaml").read_text(encoding="utf-8")
        self.assertIn("confidence: 0.25", config)
        self.assertIn("iou: 0.45", config)
        self.assertIn("global_nms_iou: 0.50", config)
        self.assertNotIn("confidence: 0.05", config)


if __name__ == "__main__":
    unittest.main()
