"""Explicit resume helper for historical DOTA baseline training only.

The checkpoint must be supplied by the caller and must contain optimizer state
for a true Ultralytics resume. This module is unrelated to the completed
RarePlanes run recorded in ``configs/rareplanes_finetune_baseline.yaml``.
Importing it is safe and never resumes training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resume a historical DOTA baseline run")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to a resumable last.pt containing optimizer state",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")

    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    model.train(resume=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
