"""Portable ordinary-image inference for the YOLO26s-OBB baseline."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    from .project_config import (
        ConfigError,
        config_value,
        ensure_directory,
        find_project_root,
        load_project_config,
        resolve_project_path,
    )
except ImportError:  # Support direct execution: python src/inference_gpu.py
    from project_config import (
        ConfigError,
        config_value,
        ensure_directory,
        find_project_root,
        load_project_config,
        resolve_project_path,
    )


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif"}


class InferenceDependencyError(RuntimeError):
    """Raised when optional inference dependencies are unavailable."""


@dataclass(frozen=True)
class InferenceSettings:
    model_path: Path
    input_path: Path
    output_path: Path
    device: str
    confidence: float
    iou: float
    image_size: int


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without loading ML dependencies."""
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO26s-OBB inference on an ordinary-image directory and "
            "write annotated images plus CSV summaries."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/project.yaml",
        help="Project YAML path, resolved relative to the repository root.",
    )
    parser.add_argument(
        "--model",
        help="Model path. Overrides paths.model from the project configuration.",
    )
    parser.add_argument(
        "--input",
        default="test_images",
        help="Directory containing input images (default: test_images).",
    )
    parser.add_argument(
        "--output",
        help=(
            "Output directory. Overrides paths.output_root from the project "
            "configuration."
        ),
    )
    parser.add_argument(
        "--device",
        help="Ultralytics device, such as cpu, 0, or 0,1.",
    )
    parser.add_argument(
        "--confidence",
        "--conf",
        dest="confidence",
        type=float,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        help="Intersection-over-union threshold.",
    )
    parser.add_argument(
        "--image-size",
        "--imgsz",
        dest="image_size",
        type=int,
        help="Inference image size in pixels.",
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> InferenceSettings:
    """Combine command-line overrides with repository-relative configuration."""
    project_root = find_project_root()
    config = load_project_config(args.config, project_root=project_root)

    model = args.model or config_value(config, "paths", "model")
    output = args.output or config_value(config, "paths", "output_root")
    device = args.device or str(config_value(config, "runtime", "device"))
    confidence = (
        args.confidence
        if args.confidence is not None
        else float(config_value(config, "inference", "confidence"))
    )
    iou = (
        args.iou
        if args.iou is not None
        else float(config_value(config, "inference", "iou"))
    )
    image_size = (
        args.image_size
        if args.image_size is not None
        else int(config_value(config, "runtime", "image_size"))
    )

    if not 0.0 <= confidence <= 1.0:
        raise ConfigError("Inference confidence must be between 0 and 1.")
    if not 0.0 <= iou <= 1.0:
        raise ConfigError("Inference IoU must be between 0 and 1.")
    if image_size <= 0:
        raise ConfigError("Runtime image_size must be greater than zero.")

    return InferenceSettings(
        model_path=resolve_project_path(model, project_root=project_root),
        input_path=resolve_project_path(args.input, project_root=project_root),
        output_path=resolve_project_path(output, project_root=project_root),
        device=device,
        confidence=confidence,
        iou=iou,
        image_size=image_size,
    )


def discover_images(input_path: Path) -> list[Path]:
    """Return supported images in one directory, matching baseline behavior."""
    return sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def validate_inputs(settings: InferenceSettings) -> list[Path]:
    """Validate required inputs before loading the model or creating outputs."""
    if not settings.model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {settings.model_path}")
    if not settings.input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {settings.input_path}")
    if not settings.input_path.is_dir():
        raise NotADirectoryError(
            f"Input path must be a directory: {settings.input_path}"
        )

    image_files = discover_images(settings.input_path)
    if not image_files:
        extensions = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise FileNotFoundError(
            f"No supported images found in {settings.input_path}. "
            f"Expected one of: {extensions}"
        )
    return image_files


def run_inference(settings: InferenceSettings) -> int:
    """Run baseline ordinary-image inference using resolved settings."""
    image_files = validate_inputs(settings)

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise InferenceDependencyError(
            "Inference dependencies are unavailable. Install requirements.txt "
            "before running inference."
        ) from exc

    model = YOLO(str(settings.model_path))
    output_path = ensure_directory(settings.output_path)
    annotated_dir = ensure_directory(output_path / "annotated")

    device_label = (
        "CPU" if settings.device.lower() == "cpu" else settings.device
    )
    print(f"Model:       {settings.model_path}")
    print(f"Test images: {len(image_files)} files")
    print(f"Output:      {annotated_dir}")
    print(
        f"Device:      {device_label}  |  imgsz: {settings.image_size}px  |  "
        f"conf: {settings.confidence}  |  iou: {settings.iou}"
    )
    print("=" * 60)
    print(f"\nModel loaded: {len(model.names)} classes")
    print(f"Classes: {list(model.names.values())}\n")

    print(f"Warming up {device_label}...")
    model.predict(
        source=str(image_files[0]),
        imgsz=settings.image_size,
        conf=settings.confidence,
        device=settings.device,
        verbose=False,
    )
    print("Warmup done.\n")

    summary_rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []

    for index, image_path in enumerate(image_files, 1):
        print(f"[{index}/{len(image_files)}] {image_path.name}")

        original_image = cv2.imread(str(image_path))
        image_height, image_width = (
            original_image.shape[:2] if original_image is not None else (0, 0)
        )

        start = time.perf_counter()
        predictions = model.predict(
            source=str(image_path),
            imgsz=settings.image_size,
            conf=settings.confidence,
            iou=settings.iou,
            device=settings.device,
            verbose=False,
            save=False,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        result = predictions[0]

        annotated_image = result.plot()
        cv2.imwrite(str(annotated_dir / image_path.name), annotated_image)

        detection_count = 0
        class_counts: dict[str, int] = {}

        if result.obb is not None and len(result.obb.cls) > 0:
            class_indices = result.obb.cls.cpu().numpy()
            confidences = result.obb.conf.cpu().numpy()
            polygons = result.obb.xyxyxyxy.cpu().numpy()
            detection_count = len(class_indices)

            for detection_index in range(detection_count):
                class_index = int(class_indices[detection_index])
                class_name = model.names[class_index]
                confidence = float(confidences[detection_index])
                points = polygons[detection_index]
                xs, ys = points[:, 0], points[:, 1]
                point_count = len(xs)
                area = abs(
                    sum(
                        xs[k] * ys[(k + 1) % point_count]
                        - xs[(k + 1) % point_count] * ys[k]
                        for k in range(point_count)
                    )
                ) / 2

                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                detection_rows.append(
                    {
                        "image": image_path.name,
                        "class_idx": class_index,
                        "class_name": class_name,
                        "confidence": round(confidence, 4),
                        "box_area_px": round(float(area), 1),
                    }
                )

        print(
            f"  Latency: {latency_ms:.1f}ms  |  "
            f"Detections: {detection_count}"
        )
        for class_name, count in sorted(
            class_counts.items(), key=lambda item: -item[1]
        ):
            print(f"    {class_name}: {count}")

        summary_rows.append(
            {
                "image": image_path.name,
                "width_px": image_width,
                "height_px": image_height,
                "latency_ms": round(latency_ms, 2),
                "detections": detection_count,
                "classes_found": str(class_counts),
            }
        )

    summary_csv = output_path / "inference_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image",
                "width_px",
                "height_px",
                "latency_ms",
                "detections",
                "classes_found",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    detections_csv = output_path / "inference_detections.csv"
    with detections_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image",
                "class_idx",
                "class_name",
                "confidence",
                "box_area_px",
            ],
        )
        writer.writeheader()
        writer.writerows(detection_rows)

    latencies = [float(row["latency_ms"]) for row in summary_rows]
    average_latency = sum(latencies) / len(latencies)

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Images processed : {len(summary_rows)}")
    print(
        "Total detections : "
        f"{sum(int(row['detections']) for row in summary_rows)}"
    )
    print(f"Avg latency      : {average_latency:.1f} ms/image")
    print(f"Min latency      : {min(latencies):.1f} ms")
    print(f"Max latency      : {max(latencies):.1f} ms")
    print(f"Throughput       : {1000 / average_latency:.2f} images/sec")
    print(f"\nAnnotated images : {annotated_dir}")
    print(f"Summary CSV      : {summary_csv}")
    print(f"Detections CSV   : {detections_csv}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and run inference."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = settings_from_args(args)
        return run_inference(settings)
    except (
        ConfigError,
        FileNotFoundError,
        NotADirectoryError,
        InferenceDependencyError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
