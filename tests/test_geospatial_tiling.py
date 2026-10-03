from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from geospatial_tiling import (  # noqa: E402
    GeometryError,
    build_geospatial_detection,
    generate_tile_windows,
    global_pixels_to_world,
    local_obb_to_global_pixels,
    points_within_valid_tile_region,
    read_tile,
    stride_from_overlap,
    tiling_parameters_from_config,
    validate_tile_plan,
    world_to_global_pixels,
)
def assert_complete_pixel_coverage(
    testcase: unittest.TestCase,
    windows,
    width: int,
    height: int,
) -> None:
    """Independent small-raster oracle for the analytical plan validator."""

    import numpy as np

    coverage = np.zeros((height, width), dtype=np.uint16)
    for window in windows:
        coverage[
            window.row_off : window.row_off + window.height,
            window.col_off : window.col_off + window.width,
        ] += 1
    testcase.assertTrue(np.all(coverage >= 1))


class TileCoverageTests(unittest.TestCase):
    def check_plan(self, width: int, height: int, tile_size: int, overlap: float):
        windows = list(
            generate_tile_windows("scene", width, height, tile_size, overlap)
        )
        result = validate_tile_plan(windows, width, height)
        self.assertTrue(result.valid)
        self.assertEqual(len(windows), len({window.tile_id for window in windows}))
        self.assertEqual(len(windows), len(set(windows)))
        self.assertEqual(max(window.col_off + window.width for window in windows), width)
        self.assertEqual(max(window.row_off + window.height for window in windows), height)
        assert_complete_pixel_coverage(self, windows, width, height)
        return windows

    def test_default_stride_uses_documented_round_half_up(self):
        self.assertEqual(stride_from_overlap(1024, 0.20), 819)
        self.assertEqual(stride_from_overlap(5, 0.50), 3)

    def test_dimensions_exactly_follow_stride_pattern(self):
        windows = self.check_plan(340, 260, 100, 0.20)
        self.assertEqual(sorted({window.col_off for window in windows}), [0, 80, 160, 240])
        self.assertEqual(sorted({window.row_off for window in windows}), [0, 80, 160])

    def test_non_divisible_dimensions_anchor_final_windows(self):
        windows = self.check_plan(355, 279, 100, 0.20)
        self.assertEqual(
            sorted({window.col_off for window in windows}), [0, 80, 160, 240, 255]
        )
        self.assertEqual(
            sorted({window.row_off for window in windows}), [0, 80, 160, 179]
        )

    def test_dimensions_just_above_tile_size(self):
        windows = self.check_plan(101, 103, 100, 0.20)
        self.assertEqual(sorted({window.col_off for window in windows}), [0, 1])
        self.assertEqual(sorted({window.row_off for window in windows}), [0, 3])

    def test_raster_smaller_than_tile_retains_valid_dimensions(self):
        windows = self.check_plan(73, 61, 100, 0.20)
        self.assertEqual(len(windows), 1)
        tile = windows[0]
        self.assertEqual((tile.width, tile.height), (73, 61))
        self.assertTrue(tile.has_padding)
        self.assertEqual(tile.valid_region.as_tuple(), (0.0, 0.0, 73.0, 61.0))


class CoordinateTransformTests(unittest.TestCase):
    def setUp(self):
        self.tile = next(generate_tile_windows("known", 600, 500, 200, 0.0))
        windows = list(generate_tile_windows("known", 600, 500, 200, 0.0))
        self.tile = next(
            window
            for window in windows
            if window.col_off == 200 and window.row_off == 200
        )

    def test_local_obb_to_global_pixels(self):
        local = ((10, 20), (90, 25), (85, 70), (5, 65))
        actual = local_obb_to_global_pixels(local, self.tile)
        self.assertEqual(
            actual,
            ((210.0, 220.0), (290.0, 225.0), (285.0, 270.0), (205.0, 265.0)),
        )

    def test_padded_coordinates_are_rejected(self):
        tile = next(generate_tile_windows("small", 73, 61, 100, 0.20))
        valid = ((2, 2), (72, 2), (72, 60), (2, 60))
        padded = ((2, 2), (80, 2), (80, 60), (2, 60))
        self.assertTrue(points_within_valid_tile_region(valid, tile))
        self.assertFalse(points_within_valid_tile_region(padded, tile))
        with self.assertRaises(GeometryError):
            local_obb_to_global_pixels(padded, tile)

    def test_partial_window_padding_has_an_explicit_valid_mask(self):
        import numpy as np

        tile = next(generate_tile_windows("small", 73, 61, 100, 0.20))

        class FakeDataset:
            width = 73
            height = 61
            count = 3
            nodata = None

            @staticmethod
            def read(*, window, boundless):
                self.assertFalse(boundless)
                self.assertEqual(
                    (window.col_off, window.row_off, window.width, window.height),
                    (0, 0, 73, 61),
                )
                return np.ones((3, 61, 73), dtype=np.uint8)

        tile_read = read_tile(FakeDataset(), tile, pad_to_tile_size=True)
        self.assertEqual(tile_read.data.shape, (3, 100, 100))
        self.assertEqual(tile_read.valid_mask.shape, (100, 100))
        self.assertEqual(int(tile_read.valid_mask.sum()), 73 * 61)
        self.assertTrue(np.all(tile_read.data[:, :61, :73] == 1))
        self.assertTrue(np.all(tile_read.data[:, 61:, :] == 0))
        self.assertTrue(np.all(tile_read.data[:, :, 73:] == 0))

    def test_real_train_cog_affine_and_round_trip(self):
        try:
            import rasterio
        except ImportError as exc:
            self.skipTest(f"Rasterio unavailable: {exc}")

        train_dir = (
            REPOSITORY_ROOT
            / "datasets"
            / "RarePlanes"
            / "train"
            / "imagery"
            / "PS-RGB_cog"
        )
        raster_paths = sorted(train_dir.glob("*.tif")) + sorted(train_dir.glob("*.tiff"))
        if not raster_paths:
            self.skipTest("No RarePlanes TRAIN COG is available")

        with rasterio.open(raster_paths[0]) as dataset:
            points = (
                (0.0, 0.0),
                (dataset.width / 2.0, dataset.height / 2.0),
                (float(dataset.width), float(dataset.height)),
            )
            world = global_pixels_to_world(points, dataset.transform)
            expected = tuple(dataset.transform @ point for point in points)
            for actual_point, expected_point in zip(world, expected):
                self.assertAlmostEqual(actual_point[0], expected_point[0], places=12)
                self.assertAlmostEqual(actual_point[1], expected_point[1], places=12)

            round_trip = world_to_global_pixels(world, dataset.transform)
            for actual_point, expected_point in zip(round_trip, points):
                self.assertTrue(
                    math.isclose(actual_point[0], expected_point[0], abs_tol=1e-7)
                )
                self.assertTrue(
                    math.isclose(actual_point[1], expected_point[1], abs_tol=1e-7)
                )

    def test_detection_record_contains_overlap_provenance(self):
        from affine import Affine

        detection = build_geospatial_detection(
            tile=self.tile,
            class_id=4,
            class_name="plane",
            confidence=0.75,
            local_obb_vertices=((10, 20), (90, 25), (85, 70), (5, 65)),
            transform=Affine.translation(-75.0, 40.0) @ Affine.scale(0.001, -0.001),
            source_crs="EPSG:4326",
        )
        self.assertEqual(detection.tile_id, self.tile.tile_id)
        self.assertEqual(detection.row_off, 200)
        self.assertEqual(detection.col_off, 200)
        self.assertEqual(detection.output_crs, "OGC:CRS84")
        self.assertEqual(detection.tile_bounds_source_pixels, (200.0, 200.0, 400.0, 400.0))
        self.assertEqual(len(detection.tile_geographic_polygon), 4)
        self.assertAlmostEqual(detection.geographic_polygon_vertices[0][0], -74.79)
        self.assertAlmostEqual(detection.geographic_polygon_vertices[0][1], 39.78)


class ConfigTests(unittest.TestCase):
    def test_project_tiling_defaults_are_present(self):
        # The GIS-only conda environment deliberately has no PyYAML. Verify the
        # existing project config without adding packages to that environment.
        config_text = (REPOSITORY_ROOT / "configs" / "project.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("  tile_size: 1024", config_text)
        self.assertIn("  overlap: 0.2", config_text)

    def test_existing_config_mapping_interface_is_reused(self):
        self.assertEqual(
            tiling_parameters_from_config(
                {"tiling": {"tile_size": 1024, "overlap": 0.2}}
            ),
            (1024, 0.2),
        )


if __name__ == "__main__":
    unittest.main()
