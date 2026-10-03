"""Fixed-shape ONNX export and CPU parity validation for OBB checkpoints.

The exported graph contains raw OBB head output (no embedded NMS). PyTorch and
ONNX parity therefore uses the same Ultralytics predictor post-processing with
identical confidence and IoU settings.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import tempfile
import time
from typing import Any, Mapping, Sequence

try:
    from .project_config import find_project_root
except ImportError:  # Support direct execution from src/.
    from project_config import find_project_root


IMAGE_SIZE = 1024
BATCH_SIZE = 1
DEFAULT_CONFIDENCE = 0.25
DEFAULT_MODEL_IOU = 0.45
DEFAULT_MATCH_IOU = 0.50
DEFAULT_TARGET_CLASS = "plane"
PROVENANCE_FILENAME = "export_provenance.json"


class ONNXExportError(RuntimeError):
    """Raised when export or validation violates the fixed deployment protocol."""


@dataclass(frozen=True)
class OBBDetection:
    """Post-processed OBB used for backend parity matching."""

    index: int
    class_id: int
    class_name: str
    confidence: float
    vertices: tuple[tuple[float, float], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: Path, expected_sha256: str | None = None) -> str:
    """Verify checkpoint existence and, when supplied, its expected SHA-256."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    actual = sha256_file(path)
    if expected_sha256 and actual.lower() != expected_sha256.lower():
        raise ONNXExportError(
            f"Checkpoint SHA-256 mismatch: expected {expected_sha256.lower()}, found {actual}"
        )
    return actual


def normalize_names(names: Mapping[int, str] | Sequence[str]) -> dict[int, str]:
    if isinstance(names, Mapping):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def validate_obb_metadata(
    task: str, names: Mapping[int, str] | Sequence[str], target_class: str
) -> tuple[dict[int, str], int]:
    """Validate only the model capabilities required by this exporter."""
    if task != "obb":
        raise ONNXExportError(f"Expected an OBB checkpoint, found task={task!r}")
    normalized = normalize_names(names)
    if not normalized:
        raise ONNXExportError("Checkpoint class metadata is empty")
    matches = [class_id for class_id, name in normalized.items() if name == target_class]
    if not matches:
        raise ONNXExportError(f"Target class {target_class!r} is absent from checkpoint metadata")
    if len(matches) != 1:
        raise ONNXExportError(f"Target class {target_class!r} is ambiguous in checkpoint metadata")
    return normalized, matches[0]


def fixed_export_arguments() -> dict[str, Any]:
    """Return arguments supported by the installed Ultralytics ONNX exporter."""
    return {
        "format": "onnx",
        "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "dynamic": False,
        "device": "cpu",
        "quantize": None,
        "opset": 18,
        "simplify": False,
        "nms": None,
    }


def export_filename(source: Path) -> str:
    """Derive a checkpoint-independent ONNX filename."""
    return f"{source.stem}.onnx"


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_input_shape(shape: Sequence[Any]) -> list[int]:
    expected = [BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE]
    actual = [int(value) if isinstance(value, int) else value for value in shape]
    if actual != expected:
        raise ONNXExportError(f"Unexpected fixed ONNX input shape: {actual}; expected {expected}")
    return expected


def validate_onnx_structure(onnx_path: Path, input_tensor: Any | None = None) -> dict[str, Any]:
    """Check the graph and execute one CPU raw-output inference when input is supplied."""
    import numpy as np
    import onnx
    import onnxruntime as ort

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ONNXExportError(f"Expected one ONNX input, found {len(inputs)}")
    input_shape = validate_input_shape(inputs[0].shape)
    outputs = session.get_outputs()
    if not outputs:
        raise ONNXExportError("ONNX graph has no outputs")

    result: dict[str, Any] = {
        "onnx_checker": "passed",
        "providers": session.get_providers(),
        "input": {"name": inputs[0].name, "shape": input_shape, "type": inputs[0].type},
        "outputs": [
            {"name": output.name, "shape": list(output.shape), "type": output.type}
            for output in outputs
        ],
    }
    if input_tensor is not None:
        array = np.asarray(input_tensor, dtype=np.float32)
        if list(array.shape) != input_shape:
            raise ONNXExportError(
                f"Runtime input shape {list(array.shape)} does not match {input_shape}"
            )
        raw = session.run(None, {inputs[0].name: array})
        if not raw:
            raise ONNXExportError("ONNX Runtime produced no output tensors")
        if not all(np.isfinite(value).all() for value in raw):
            raise ONNXExportError("ONNX Runtime output contains NaN or Inf")
        result["real_input_execution"] = {
            "passed": True,
            "output_shapes": [list(value.shape) for value in raw],
            "finite": True,
        }
    return result


def image_to_tensor(image_path: Path) -> Any:
    """Load an exact 1024 RGB chip as normalized NCHW float32 for raw ORT validation."""
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ONNXExportError(f"Could not read parity image: {image_path}")
    if image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ONNXExportError(
            f"Parity chip must be exactly {IMAGE_SIZE}x{IMAGE_SIZE}, found {image.shape[1]}x{image.shape[0]}"
        )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def extract_obb_detections(result: Any) -> list[OBBDetection]:
    """Extract post-processed four-corner OBB predictions from one Ultralytics result."""
    obb = getattr(result, "obb", None)
    if obb is None:
        raise ONNXExportError("Ultralytics result does not contain OBB predictions")
    names = normalize_names(result.names)
    vertices = obb.xyxyxyxy.cpu().numpy()
    confidences = obb.conf.cpu().numpy()
    classes = obb.cls.cpu().numpy()
    detections: list[OBBDetection] = []
    for index, (corners, confidence, class_value) in enumerate(
        zip(vertices, confidences, classes, strict=True)
    ):
        class_id = int(class_value)
        detections.append(
            OBBDetection(
                index=index,
                class_id=class_id,
                class_name=names[class_id],
                confidence=float(confidence),
                vertices=tuple((float(point[0]), float(point[1])) for point in corners),
            )
        )
    return detections


def _polygon_iou(first: OBBDetection, second: OBBDetection) -> float:
    from shapely.geometry import Polygon

    a, b = Polygon(first.vertices), Polygon(second.vertices)
    if not a.is_valid or not b.is_valid or a.area <= 0 or b.area <= 0:
        return 0.0
    union = a.union(b).area
    return float(a.intersection(b).area / union) if union > 0 else 0.0


def compare_detections(
    pytorch: Sequence[OBBDetection],
    onnx_detections: Sequence[OBBDetection],
    *,
    match_iou: float = DEFAULT_MATCH_IOU,
) -> dict[str, Any]:
    """Greedily match same-class post-processed OBBs by true polygon IoU."""
    candidates = sorted(
        (
            (_polygon_iou(pt, ort), pt_index, ort_index)
            for pt_index, pt in enumerate(pytorch)
            for ort_index, ort in enumerate(onnx_detections)
            if pt.class_id == ort.class_id
        ),
        reverse=True,
    )
    used_pt: set[int] = set()
    used_ort: set[int] = set()
    matches: list[dict[str, Any]] = []
    for iou, pt_index, ort_index in candidates:
        if iou < match_iou:
            break
        if pt_index in used_pt or ort_index in used_ort:
            continue
        pt, ort = pytorch[pt_index], onnx_detections[ort_index]
        used_pt.add(pt_index)
        used_ort.add(ort_index)
        matches.append(
            {
                "pytorch_index": pt.index,
                "onnx_index": ort.index,
                "class_id": pt.class_id,
                "polygon_iou": iou,
                "pytorch_confidence": pt.confidence,
                "onnx_confidence": ort.confidence,
                "absolute_confidence_difference": abs(pt.confidence - ort.confidence),
            }
        )
    confidence_differences = [row["absolute_confidence_difference"] for row in matches]
    polygon_ious = [row["polygon_iou"] for row in matches]
    return {
        "match_iou_threshold": match_iou,
        "pytorch_detection_count": len(pytorch),
        "onnx_detection_count": len(onnx_detections),
        "matched_detection_count": len(matches),
        "unmatched_pytorch_indices": [
            detection.index for index, detection in enumerate(pytorch) if index not in used_pt
        ],
        "unmatched_onnx_indices": [
            detection.index
            for index, detection in enumerate(onnx_detections)
            if index not in used_ort
        ],
        "confidence_difference": {
            "mean": statistics.fmean(confidence_differences) if confidence_differences else None,
            "max": max(confidence_differences) if confidence_differences else None,
        },
        "polygon_iou": {
            "mean": statistics.fmean(polygon_ious) if polygon_ious else None,
            "min": min(polygon_ious) if polygon_ious else None,
            "max": max(polygon_ious) if polygon_ious else None,
        },
        "matches": matches,
    }


def _predict(model: Any, image_path: Path, confidence: float, model_iou: float) -> Any:
    results = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=confidence,
        iou=model_iou,
        device="cpu",
        verbose=False,
    )
    if len(results) != 1:
        raise ONNXExportError(f"Expected one prediction result, found {len(results)}")
    return results[0]


def _benchmark(
    model: Any,
    image_path: Path,
    confidence: float,
    model_iou: float,
    warmup_runs: int,
    timed_runs: int,
) -> dict[str, Any]:
    if warmup_runs < 0 or timed_runs <= 0:
        raise ValueError("warmup_runs must be non-negative and timed_runs must be positive")
    for _ in range(warmup_runs):
        _predict(model, image_path, confidence, model_iou)
    timings = []
    for _ in range(timed_runs):
        started = time.perf_counter()
        _predict(model, image_path, confidence, model_iou)
        timings.append((time.perf_counter() - started) * 1000.0)
    return {
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "wall_latency_ms": {
            "average": statistics.fmean(timings),
            "median": statistics.median(timings),
            "minimum": min(timings),
            "maximum": max(timings),
        },
    }


def export_checkpoint(
    source: Path,
    output_dir: Path,
    *,
    expected_sha256: str | None = None,
    target_class: str = DEFAULT_TARGET_CLASS,
    overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Export one compatible OBB checkpoint without modifying the source file."""
    from ultralytics import YOLO, __version__ as ultralytics_version
    import onnx
    import onnxruntime
    import torch

    source = source.resolve()
    output_dir = output_dir.resolve()
    source_hash = verify_checkpoint(source, expected_sha256)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / export_filename(source)
    provenance_path = output_dir / PROVENANCE_FILENAME
    existing = [path for path in (output_path, provenance_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing export artifacts: {existing}")

    with tempfile.TemporaryDirectory(prefix="onnx_export_", dir=output_dir) as temporary:
        temporary_checkpoint = Path(temporary) / "source.pt"
        shutil.copy2(source, temporary_checkpoint)
        model = YOLO(str(temporary_checkpoint))
        names, target_class_id = validate_obb_metadata(model.task, model.names, target_class)
        exported = Path(model.export(**fixed_export_arguments())).resolve()
        if not exported.is_file():
            raise ONNXExportError(f"Ultralytics did not create the expected ONNX file: {exported}")
        os.replace(exported, output_path)

    if sha256_file(source) != source_hash:
        raise ONNXExportError("Source checkpoint changed during export")
    structure = validate_onnx_structure(output_path)
    root = find_project_root(source.parent)
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "path": _portable_path(source, root),
            "sha256": source_hash,
            "task": "obb",
            "classes": names,
            "target_class": target_class,
            "target_class_id": target_class_id,
        },
        "export": {
            "arguments": fixed_export_arguments(),
            "path": _portable_path(output_path, root),
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
        "software": {
            "ultralytics": ultralytics_version,
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
        "structural_validation": structure,
    }
    _write_json(provenance_path, provenance)
    return output_path, provenance


def run_parity_validation(
    source: Path,
    onnx_path: Path,
    image_path: Path,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    model_iou: float = DEFAULT_MODEL_IOU,
    match_iou: float = DEFAULT_MATCH_IOU,
    warmup_runs: int = 1,
    timed_runs: int = 3,
) -> dict[str, Any]:
    """Compare both backends on one bounded chip after identical post-processing."""
    from ultralytics import YOLO

    image_path = image_path.resolve()
    tensor = image_to_tensor(image_path)
    raw_validation = validate_onnx_structure(onnx_path, tensor)
    pytorch_model = YOLO(str(source.resolve()))
    onnx_model = YOLO(str(onnx_path.resolve()), task="obb")
    pytorch_result = _predict(pytorch_model, image_path, confidence, model_iou)
    onnx_result = _predict(onnx_model, image_path, confidence, model_iou)
    pytorch_detections = extract_obb_detections(pytorch_result)
    onnx_detections = extract_obb_detections(onnx_result)
    parity = compare_detections(
        pytorch_detections, onnx_detections, match_iou=match_iou
    )
    pytorch_latency = _benchmark(
        pytorch_model, image_path, confidence, model_iou, warmup_runs, timed_runs
    )
    onnx_latency = _benchmark(
        onnx_model, image_path, confidence, model_iou, warmup_runs, timed_runs
    )
    pt_average = pytorch_latency["wall_latency_ms"]["average"]
    ort_average = onnx_latency["wall_latency_ms"]["average"]
    root = find_project_root(image_path.parent)
    return {
        "image": _portable_path(image_path, root),
        "settings": {
            "imgsz": IMAGE_SIZE,
            "confidence": confidence,
            "model_iou": model_iou,
            "match_polygon_iou": match_iou,
            "device": "cpu",
            "postprocessing": "Ultralytics OBB predictor for both backends",
        },
        "onnx_raw_runtime_validation": raw_validation["real_input_execution"],
        "parity": parity,
        "latency": {
            "pytorch": pytorch_latency,
            "onnxruntime": onnx_latency,
            "relative_speedup_onnx_vs_pytorch": pt_average / ort_average,
            "scope": "single-chip laptop CPU wall-clock microbenchmark",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    root = find_project_root()
    parser = argparse.ArgumentParser(description="Export and validate a fixed-shape OBB ONNX model")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=root / "outputs" / "deployment" / "rareplanes_onnx"
    )
    parser.add_argument("--expected-sha256")
    parser.add_argument("--target-class", default=DEFAULT_TARGET_CLASS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--parity-image", type=Path)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--iou", type=float, default=DEFAULT_MODEL_IOU)
    parser.add_argument("--match-iou", type=float, default=DEFAULT_MATCH_IOU)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timed-runs", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    onnx_path, provenance = export_checkpoint(
        args.model,
        args.output_dir,
        expected_sha256=args.expected_sha256,
        target_class=args.target_class,
        overwrite=args.overwrite,
    )
    if args.parity_image:
        parity = run_parity_validation(
            args.model,
            onnx_path,
            args.parity_image,
            confidence=args.confidence,
            model_iou=args.iou,
            match_iou=args.match_iou,
            warmup_runs=args.warmup_runs,
            timed_runs=args.timed_runs,
        )
        provenance["parity_validation"] = parity
        _write_json(args.output_dir.resolve() / PROVENANCE_FILENAME, provenance)
    print(json.dumps(provenance, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
