from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from export_rareplanes_onnx import (  # noqa: E402
    OBBDetection,
    ONNXExportError,
    compare_detections,
    export_filename,
    fixed_export_arguments,
    normalize_names,
    sha256_file,
    validate_input_shape,
    validate_obb_metadata,
    verify_checkpoint,
)


def detection(
    index: int,
    confidence: float,
    vertices: tuple[tuple[float, float], ...],
    class_id: int = 0,
    class_name: str = "plane",
) -> OBBDetection:
    return OBBDetection(index, class_id, class_name, confidence, vertices)


class ExportConfigurationTests(unittest.TestCase):
    def test_export_arguments_are_fixed_cpu_fp32_raw_onnx(self) -> None:
        arguments = fixed_export_arguments()
        self.assertEqual(arguments["format"], "onnx")
        self.assertEqual(arguments["imgsz"], 1024)
        self.assertEqual(arguments["batch"], 1)
        self.assertFalse(arguments["dynamic"])
        self.assertEqual(arguments["device"], "cpu")
        self.assertIsNone(arguments["quantize"])
        self.assertFalse(arguments["simplify"])
        self.assertIsNone(arguments["nms"])
        self.assertNotIn("engine", arguments.values())

    def test_export_filename_is_checkpoint_independent(self) -> None:
        self.assertEqual(export_filename(Path("another_model.pt")), "another_model.onnx")

    def test_valid_one_and_multi_class_obb_metadata(self) -> None:
        names, target = validate_obb_metadata("obb", {0: "plane"}, "plane")
        self.assertEqual(names, {0: "plane"})
        self.assertEqual(target, 0)
        names, target = validate_obb_metadata("obb", {0: "ship", 1: "plane"}, "plane")
        self.assertEqual(target, 1)
        self.assertEqual(names[1], "plane")

    def test_incompatible_task_or_target_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ONNXExportError, "OBB checkpoint"):
            validate_obb_metadata("detect", {0: "plane"}, "plane")
        with self.assertRaisesRegex(ONNXExportError, "absent"):
            validate_obb_metadata("obb", {0: "ship"}, "plane")
        with self.assertRaisesRegex(ONNXExportError, "ambiguous"):
            validate_obb_metadata("obb", {0: "plane", 1: "plane"}, "plane")

    def test_name_normalization_accepts_sequences(self) -> None:
        self.assertEqual(normalize_names(["plane", "ship"]), {0: "plane", 1: "ship"})

    def test_checkpoint_hash_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            digest = sha256_file(checkpoint)
            self.assertEqual(verify_checkpoint(checkpoint, digest.upper()), digest)
            with self.assertRaisesRegex(ONNXExportError, "SHA-256 mismatch"):
                verify_checkpoint(checkpoint, "0" * 64)

    def test_missing_checkpoint_fails(self) -> None:
        with self.assertRaises(FileNotFoundError):
            verify_checkpoint(Path("does-not-exist.pt"))

    def test_fixed_input_shape_validation(self) -> None:
        self.assertEqual(validate_input_shape([1, 3, 1024, 1024]), [1, 3, 1024, 1024])
        with self.assertRaisesRegex(ONNXExportError, "Unexpected fixed"):
            validate_input_shape(["batch", 3, 1024, 1024])


class ParityMatchingTests(unittest.TestCase):
    def test_strong_overlap_matches_and_reports_confidence_difference(self) -> None:
        pytorch = [detection(0, 0.90, ((0, 0), (10, 0), (10, 10), (0, 10)))]
        onnx = [detection(0, 0.89, ((0.1, 0), (10.1, 0), (10.1, 10), (0.1, 10)))]
        result = compare_detections(pytorch, onnx)
        self.assertEqual(result["matched_detection_count"], 1)
        self.assertGreater(result["polygon_iou"]["min"], 0.95)
        self.assertAlmostEqual(result["confidence_difference"]["max"], 0.01)
        self.assertEqual(result["unmatched_pytorch_indices"], [])
        self.assertEqual(result["unmatched_onnx_indices"], [])

    def test_rotated_polygon_matching_uses_polygon_geometry(self) -> None:
        diamond = ((5, 0), (10, 5), (5, 10), (0, 5))
        shifted = ((5.2, 0), (10.2, 5), (5.2, 10), (0.2, 5))
        result = compare_detections(
            [detection(0, 0.8, diamond)], [detection(0, 0.79, shifted)]
        )
        self.assertEqual(result["matched_detection_count"], 1)
        self.assertGreater(result["polygon_iou"]["mean"], 0.9)

    def test_unmatched_and_cross_class_detections_remain_unmatched(self) -> None:
        vertices = ((0, 0), (10, 0), (10, 10), (0, 10))
        pytorch = [detection(4, 0.9, vertices)]
        onnx = [detection(8, 0.9, vertices, class_id=1, class_name="ship")]
        result = compare_detections(pytorch, onnx)
        self.assertEqual(result["matched_detection_count"], 0)
        self.assertEqual(result["unmatched_pytorch_indices"], [4])
        self.assertEqual(result["unmatched_onnx_indices"], [8])

    def test_zero_detection_comparison_is_well_defined(self) -> None:
        result = compare_detections([], [])
        self.assertEqual(result["pytorch_detection_count"], 0)
        self.assertEqual(result["onnx_detection_count"], 0)
        self.assertIsNone(result["confidence_difference"]["mean"])


if __name__ == "__main__":
    unittest.main()
