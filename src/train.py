"""Historical DOTA baseline training utility.

This preserves the original DOTA training recipe for provenance. It is not
the current RarePlanes fine-tuning workflow; use ``train_rareplanes_obb.py``
for that. Importing this module is safe and never starts training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the historical DOTA OBB baseline recipe")
    parser.add_argument("--model", default="yolo26s-obb.pt")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "src" / "dota2_obb.yaml")
    parser.add_argument(
        "--project",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "legacy_dota_training",
    )
    parser.add_argument("--name", default="yolo26s-obb-dotav2-a100")
    parser.add_argument("--device", default="0")
    return parser


def legacy_training_arguments(args: argparse.Namespace) -> dict[str, object]:
    """Return the original recipe with only paths/device supplied portably."""

    return {
        "data": str(args.data.resolve()),
        "epochs": 150,
        "patience": 50,
        "imgsz": 1024,
        "batch": 32,
        "workers": 8,
        "device": args.device,
        "half": True,
        "optimizer": "auto",
        "cos_lr": True,
        "warmup_epochs": 3,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mixup": 0.15,
        "degrees": 10.0,
        "fliplr": 0.5,
        "flipud": 0.5,
        "save": True,
        "save_period": 5,
        "plots": True,
        "val": True,
        "verbose": True,
        "project": str(args.project.resolve()),
        "name": args.name,
        "seed": 42,
        "deterministic": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(**legacy_training_arguments(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
