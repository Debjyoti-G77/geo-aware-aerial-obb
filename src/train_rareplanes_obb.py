"""Reproducible launcher and CPU readiness checks for RarePlanes OBB fine-tuning.

This module deliberately delegates training to the installed Ultralytics API.
It adds path/config validation, immutable-source safeguards, provenance capture,
and bounded CPU smoke checks; it does not implement a custom trainer.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml

try:
    from project_config import find_project_root
except ImportError:  # pragma: no cover - supports ``python src/...``
    from src.project_config import find_project_root


DEFAULT_EXPERIMENT_CONFIG = Path("configs/rareplanes_finetune_baseline.yaml")
EXPECTED_TASK = "obb"
EXPECTED_NAMES = {0: "plane"}
EXPECTED_ULTRALYTICS_VERSION = "8.4.165"
EXPECTED_RUN_NAME = "rareplanes_baseline_01_full"
SOURCE_CHECKPOINT_SHA256 = "5102e1493e3d1f2d0bb331d246e060badb1709f405cd17cd4a558cac9faefbef"
FINAL_BEST_EPOCH = 59
FINAL_CHECKPOINT_SHA256 = "4111cdab67bcc3c53623e4bd08aeb5943e4a49225c27c62172036b815991eced"
EXPECTED_TRAINING_RECIPE = {
    "task": "obb",
    "imgsz": 1024,
    "epochs": 100,
    "patience": 20,
    "batch": 8,
    "device": 0,
    "workers": 8,
    "seed": 20261001,
    "deterministic": True,
    "optimizer": "AdamW",
    "lr0": 0.002,
    "lrf": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "cos_lr": False,
    "mosaic": 1.0,
    "close_mosaic": 10,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "fliplr": 0.5,
    "flipud": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "degrees": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "amp": True,
    "freeze": None,
    "resume": False,
}
PROVENANCE_INPUTS = {
    "dataset_yaml": Path("configs/rareplanes_obb.yaml"),
    "split_manifest": Path("outputs/rareplanes_dataset_preparation/split_manifest.csv"),
    "object_manifest": Path("datasets/RarePlanes_yolo_obb/manifests/objects.csv"),
    "dataset_audit": Path("outputs/rareplanes_dataset_preparation/dataset_audit.json"),
    "source_model": Path("best.pt"),
}


class TrainingConfigError(ValueError):
    """Raised when an experiment is unsafe or inconsistent with the baseline."""


@dataclass(frozen=True)
class ExperimentPaths:
    """Resolved paths for a validated experiment."""

    root: Path
    config: Path
    model: Path
    data_yaml: Path
    dataset_root: Path
    output_dir: Path


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TrainingConfigError(f"Could not read YAML: {path}") from exc
    except yaml.YAMLError as exc:
        raise TrainingConfigError(f"Invalid YAML: {path}") from exc
    if not isinstance(value, dict):
        raise TrainingConfigError(f"YAML root must be a mapping: {path}")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingConfigError(f"{label} must be a mapping")
    return value


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_completed_run(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate historical facts without turning them into training arguments."""
    completed = _mapping(config.get("completed_run"), "completed_run")
    resume = _mapping(completed.get("resume_checkpoint"), "completed_run.resume_checkpoint")
    change = _mapping(
        resume.get("only_checkpoint_change"),
        "completed_run.resume_checkpoint.only_checkpoint_change",
    )
    checkpoint = _mapping(completed.get("best_checkpoint"), "completed_run.best_checkpoint")

    expected_values = {
        "status": "complete",
        "run_name": EXPECTED_RUN_NAME,
        "ultralytics_version": EXPECTED_ULTRALYTICS_VERSION,
        "initial_epochs_completed": 10,
        "resumed_from_epoch": 11,
        "stopped_epoch": 79,
        "best_epoch": FINAL_BEST_EPOCH,
    }
    for key, expected in expected_values.items():
        if completed.get(key) != expected:
            raise TrainingConfigError(f"completed_run.{key} must be {expected!r}")
    if resume.get("source") != "last.pt":
        raise TrainingConfigError("Completed-run resume source must be last.pt")
    if resume.get("automatic") is not False:
        raise TrainingConfigError("Historical resume must not enable automatic resume")
    if resume.get("optimizer_state_preserved") is not True:
        raise TrainingConfigError("Completed-run provenance must record preserved optimizer state")
    if change != {"field": "train_args.patience", "value": 20}:
        raise TrainingConfigError("Completed-run checkpoint modification provenance is inconsistent")
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    if checkpoint_path.is_absolute() or checkpoint_path.as_posix() != "rareplanes_finetuned_best.pt":
        raise TrainingConfigError("Completed checkpoint provenance path must remain repository-relative")
    if str(checkpoint.get("sha256", "")).lower() != FINAL_CHECKPOINT_SHA256:
        raise TrainingConfigError("Completed checkpoint SHA-256 provenance is inconsistent")
    return deepcopy(completed)


def ultralytics_version_status(
    config: Mapping[str, Any], actual_version: str
) -> dict[str, Any]:
    """Compare the installed Ultralytics version with the frozen recipe version."""
    framework = _mapping(config.get("framework"), "framework")
    expected = str(framework.get("ultralytics_version", ""))
    return {
        "expected": expected,
        "actual": str(actual_version),
        "compatible": str(actual_version) == expected,
    }


def require_ultralytics_compatibility(config: Mapping[str, Any], actual_version: str) -> None:
    """Reject training under an unverified Ultralytics version."""
    status = ultralytics_version_status(config, actual_version)
    if not status["compatible"]:
        raise TrainingConfigError(
            "Ultralytics version mismatch: "
            f"expected {status['expected']}, found {status['actual']}"
        )


def load_experiment_config(
    config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG,
    *,
    project_root: str | Path | None = None,
    require_files: bool = False,
) -> tuple[dict[str, Any], ExperimentPaths]:
    """Load and validate the recipe without requiring external runtime assets.

    ``require_files`` is retained for callers that want the complete readiness
    check in one call. The default is suitable for a clean source checkout,
    where the intentionally external source checkpoint and generated dataset
    are absent.
    """
    root = Path(project_root).resolve() if project_root else find_project_root()
    config_file = _resolve(root, config_path)
    if not config_file.is_file():
        raise TrainingConfigError(f"Experiment config not found: {config_file}")
    config = _read_yaml(config_file)

    experiment = _mapping(config.get("experiment"), "experiment")
    framework = _mapping(config.get("framework"), "framework")
    model_cfg = _mapping(config.get("model"), "model")
    data_cfg = _mapping(config.get("data"), "data")
    training = _mapping(config.get("training"), "training")
    reporting = _mapping(config.get("reporting"), "reporting")
    completed_run = _validate_completed_run(config)

    configured_model_path = Path(str(model_cfg.get("path", "")))
    if configured_model_path.is_absolute() or configured_model_path.as_posix() != "best.pt":
        raise TrainingConfigError("Source model path must remain repository-relative best.pt")
    if str(model_cfg.get("sha256", "")).lower() != SOURCE_CHECKPOINT_SHA256:
        raise TrainingConfigError("Source model SHA-256 metadata is inconsistent")

    model = _resolve(root, configured_model_path)
    data_yaml = _resolve(root, data_cfg.get("yaml", ""))
    output_dir = _resolve(root, experiment.get("output_dir", ""))
    allowed_output_root = (root / "outputs" / "training").resolve()
    if not _is_relative_to(output_dir, allowed_output_root):
        raise TrainingConfigError("Training output must remain under outputs/training")
    if output_dir == model or _is_relative_to(model, output_dir):
        raise TrainingConfigError("Training output cannot contain or overwrite the source model")
    if training.get("task") != EXPECTED_TASK or model_cfg.get("expected_task") != EXPECTED_TASK:
        raise TrainingConfigError("The baseline requires task=obb")
    if framework.get("ultralytics_version") != EXPECTED_ULTRALYTICS_VERSION:
        raise TrainingConfigError(
            f"The recipe requires Ultralytics {EXPECTED_ULTRALYTICS_VERSION}"
        )
    if model_cfg.get("source_classes") != 18:
        raise TrainingConfigError("The source checkpoint must declare 18 DOTA classes")
    if completed_run["run_name"] != experiment.get("name"):
        raise TrainingConfigError("Recipe and completed-run names must match")
    if int(training.get("seed", -1)) != int(experiment.get("seed", -2)):
        raise TrainingConfigError("Experiment and training seeds must match")
    if int(training.get("batch", 0)) <= 0:
        raise TrainingConfigError("An explicit positive batch size is required")
    if bool(reporting.get("use_rareplanes_test")):
        raise TrainingConfigError("RarePlanes TEST is forbidden for this experiment")
    for key, expected_value in EXPECTED_TRAINING_RECIPE.items():
        if training.get(key) != expected_value:
            raise TrainingConfigError(
                f"training.{key} must remain {expected_value!r} for the frozen recipe"
            )

    expected = data_cfg.get("expected_classes")
    expected = {int(key): str(value) for key, value in _mapping(expected, "data.expected_classes").items()}
    if expected != EXPECTED_NAMES:
        raise TrainingConfigError("The training target must be exactly {0: 'plane'}")

    if not data_yaml.is_file():
        raise TrainingConfigError(f"Dataset YAML not found: {data_yaml}")

    dataset = _read_yaml(data_yaml)
    if "test" in dataset:
        raise TrainingConfigError("Dataset YAML must not define a TEST split")
    names = {int(key): str(value) for key, value in _mapping(dataset.get("names"), "dataset names").items()}
    if names != EXPECTED_NAMES:
        raise TrainingConfigError("Dataset YAML must contain only class 0=plane")
    for split in ("train", "val"):
        value = str(dataset.get(split, "")).replace("\\", "/").lower()
        if not value or "rareplanes/test" in value or value.startswith("test/"):
            raise TrainingConfigError(f"Unsafe or missing dataset {split} path")
    dataset_root = _resolve(root, dataset.get("path", ""))

    paths = ExperimentPaths(root, config_file, model, data_yaml, dataset_root, output_dir)
    if require_files:
        require_runtime_assets(config, paths)
    return config, paths


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_runtime_assets(config: Mapping[str, Any], paths: ExperimentPaths) -> None:
    """Require and authenticate assets needed by readiness or training actions."""

    if not paths.model.is_file():
        raise TrainingConfigError(f"Source model not found: {paths.model}")
    expected_hash = str(_mapping(config.get("model"), "model").get("sha256", "")).lower()
    actual_hash = sha256_file(paths.model)
    if actual_hash != expected_hash:
        raise TrainingConfigError(
            "Source model SHA-256 mismatch: "
            f"expected {expected_hash}, found {actual_hash}"
        )
    if not paths.data_yaml.is_file():
        raise TrainingConfigError(f"Dataset YAML not found: {paths.data_yaml}")
    dataset = _read_yaml(paths.data_yaml)
    for split in ("train", "val"):
        split_path = _resolve(paths.dataset_root, dataset.get(split, ""))
        if not split_path.is_dir():
            raise TrainingConfigError(f"Dataset {split} directory not found: {split_path}")


def build_dataset_provenance(paths: ExperimentPaths) -> dict[str, Any]:
    """Build a compact remote-transfer manifest without hashing every chip."""
    generated_files = sorted(path for path in paths.dataset_root.rglob("*") if path.is_file())
    files = [path for path in generated_files if path.suffix.lower() not in {".cache", ".pyc"}]
    excluded = [path for path in generated_files if path not in files]
    missing = [str(rel) for rel in PROVENANCE_INPUTS.values() if not (paths.root / rel).is_file()]
    if missing:
        raise TrainingConfigError(f"Missing provenance inputs: {missing}")
    audit = json.loads((paths.root / PROVENANCE_INPUTS["dataset_audit"]).read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "dataset_root": str(paths.dataset_root.relative_to(paths.root)).replace("\\", "/"),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "generated_file_count_including_derived_caches": len(generated_files),
        "generated_total_bytes_including_derived_caches": sum(path.stat().st_size for path in generated_files),
        "excluded_regenerable_files": [
            str(path.relative_to(paths.dataset_root)).replace("\\", "/") for path in excluded
        ],
        "preparation_protocol": audit["protocol"],
        "split_selection": audit["selection"],
        "dataset_audit": audit["audit"],
        "hashes": {
            label: {
                "path": str(relative).replace("\\", "/"),
                "sha256": sha256_file(paths.root / relative),
            }
            for label, relative in PROVENANCE_INPUTS.items()
        },
    }


def verify_dataset_provenance(manifest: Mapping[str, Any], root: Path) -> list[str]:
    """Return reproducibility mismatches for a provenance manifest."""
    mismatches: list[str] = []
    hashes = manifest.get("hashes", {})
    if not isinstance(hashes, Mapping):
        return ["hashes mapping is missing"]
    for label, record in hashes.items():
        if not isinstance(record, Mapping):
            mismatches.append(f"{label}: malformed hash record")
            continue
        path = _resolve(root, str(record.get("path", "")))
        if not path.is_file():
            mismatches.append(f"{label}: missing file")
        elif sha256_file(path) != record.get("sha256"):
            mismatches.append(f"{label}: SHA-256 mismatch")
    dataset_root = _resolve(root, str(manifest.get("dataset_root", "")))
    files = (
        [
            path
            for path in dataset_root.rglob("*")
            if path.is_file() and path.suffix.lower() not in {".cache", ".pyc"}
        ]
        if dataset_root.is_dir()
        else []
    )
    if len(files) != int(manifest.get("file_count", -1)):
        mismatches.append("dataset file count mismatch")
    if sum(path.stat().st_size for path in files) != int(manifest.get("total_bytes", -1)):
        mismatches.append("dataset byte size mismatch")
    return mismatches


def _git_head(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def environment_snapshot(config: Mapping[str, Any], paths: ExperimentPaths) -> dict[str, Any]:
    """Capture software, hardware, Git, and exact experiment settings."""
    import torch
    import ultralytics

    cuda_available = torch.cuda.is_available()
    train_image_count = len(list((paths.dataset_root / "images" / "train").glob("*.png")))
    return {
        "git_head": _git_head(paths.root),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "ultralytics_version_status": ultralytics_version_status(
            config, ultralytics.__version__
        ),
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "experiment_config": deepcopy(dict(config)),
        "completed_run_provenance": _validate_completed_run(config),
        "resolved_framework_settings": resolve_framework_settings(config, train_image_count),
        "resolved_paths": {
            "model": str(paths.model.relative_to(paths.root)).replace("\\", "/"),
            "data_yaml": str(paths.data_yaml.relative_to(paths.root)).replace("\\", "/"),
            "output_dir": str(paths.output_dir.relative_to(paths.root)).replace("\\", "/"),
        },
    }


def resolve_framework_settings(config: Mapping[str, Any], train_image_count: int) -> dict[str, Any]:
    """Resolve relevant Ultralytics settings for the explicit frozen recipe."""
    from ultralytics.cfg import DEFAULT_CFG, get_cfg

    training = dict(config["training"])
    resolved = get_cfg(DEFAULT_CFG, overrides=training)
    keys = (
        "optimizer",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "cos_lr",
        "mosaic",
        "close_mosaic",
        "mixup",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "flipud",
        "fliplr",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "patience",
        "amp",
        "workers",
        "nbs",
    )
    settings = {key: getattr(resolved, key) for key in keys}
    batch = int(resolved.batch)
    epochs = int(resolved.epochs)
    iterations = math.ceil(train_image_count / max(batch, int(resolved.nbs))) * epochs
    accumulate = max(round(int(resolved.nbs) / batch), 1)
    effective_decay = float(resolved.weight_decay) * batch * accumulate / int(resolved.nbs)
    if resolved.optimizer == "auto":
        if iterations > 10_000:
            selected, learning_rate, selected_momentum = "MuSGD", 0.01, 0.9
        else:
            selected = "AdamW"
            learning_rate = round(0.002 * 5 / (4 + len(EXPECTED_NAMES)), 6)
            selected_momentum = 0.9
    else:
        selected = str(resolved.optimizer)
        learning_rate = float(resolved.lr0)
        selected_momentum = float(resolved.momentum)
    settings["optimizer_resolution"] = {
        "selection_iterations": iterations,
        "selected_optimizer": selected,
        "selected_lr0": learning_rate,
        "selected_momentum": selected_momentum,
        "configured_lr0_ignored_by_auto": resolved.optimizer == "auto",
        "configured_momentum_ignored_by_auto": resolved.optimizer == "auto",
        "gradient_accumulation": accumulate,
        "effective_weight_decay": effective_decay,
    }
    settings["lr_schedule"] = "cosine" if resolved.cos_lr else "linear"
    return settings


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_readiness_provenance(config: Mapping[str, Any], paths: ExperimentPaths, output: Path) -> dict[str, Any]:
    """Write the compact dataset/environment manifest to an ignored location."""
    require_runtime_assets(config, paths)
    dataset = build_dataset_provenance(paths)
    snapshot = environment_snapshot(config, paths)
    value = {"dataset": dataset, "environment": snapshot}
    write_json(output, value)
    return value


def _head_classification_keys(model: Any) -> set[str]:
    from ultralytics.nn.modules import Detect

    target_nc = len(model.names)
    return {
        f"{name}.{attr}.{index}.{len(sequence) - 1}.{parameter}"
        for name, module in model.named_modules()
        if isinstance(module, Detect)
        for attr in ("cv3", "one2one_cv3")
        for index, sequence in enumerate(getattr(module, attr, ()))
        if getattr(sequence[-1], "out_channels", None) == target_nc
        for parameter in ("weight", "bias")
    }


def _parameter_group(key: str, backbone_layers: int, head_index: int) -> str:
    try:
        layer = int(key.split(".")[1])
    except (IndexError, ValueError):
        return "other"
    if layer < backbone_layers:
        return "backbone"
    if layer < head_index:
        return "neck"
    if ".cv2." in key or ".one2one_cv2." in key:
        return "obb_regression"
    if ".cv4." in key or ".one2one_cv4." in key:
        return "obb_angle"
    if ".cv3." in key or ".one2one_cv3." in key:
        return "classification"
    return "obb_head_other"


def build_one_class_model(model_path: Path, *, seed: int = 20261001) -> tuple[Any, Any]:
    """Construct the same nc=1 model used by OBBTrainer and load standard weights."""
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import OBBModel

    torch.manual_seed(seed)
    source = YOLO(str(model_path)).model.float()
    if source.task != EXPECTED_TASK or len(source.names) != 18 or source.names.get(0) != "plane":
        raise TrainingConfigError("best.pt is not the expected 18-class DOTA OBB model")
    target = OBBModel(deepcopy(source.yaml), nc=1, ch=3, verbose=False)
    target.names = EXPECTED_NAMES.copy()  # OBBTrainer.set_model_names_for_load behavior
    target.load(source, verbose=False)  # standard Ultralytics load/remap path
    target.names = EXPECTED_NAMES.copy()
    target.nc = 1  # OBBTrainer.set_model_attributes behavior after weight loading
    target.task = EXPECTED_TASK
    return source, target


def inspect_model_transfer(model_path: Path, *, seed: int = 20261001) -> tuple[dict[str, Any], Any]:
    """Empirically audit exact standard Ultralytics 18-class to 1-class transfer."""
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import OBBModel

    torch.manual_seed(seed)
    source = YOLO(str(model_path)).model.float()
    target = OBBModel(deepcopy(source.yaml), nc=1, ch=3, verbose=False)
    target.names = EXPECTED_NAMES.copy()
    initial = {key: value.detach().clone() for key, value in target.state_dict().items()}
    source_state = {key: value.detach().clone() for key, value in source.state_dict().items()}
    target.load(source, verbose=False)
    target.names = EXPECTED_NAMES.copy()
    target.nc = 1  # OBBTrainer.set_model_attributes behavior after weight loading
    target.task = EXPECTED_TASK
    final = target.state_dict()

    parameter_names = set(dict(target.named_parameters()))
    source_parameter_names = set(dict(source.named_parameters()))
    remap_keys = _head_classification_keys(target)
    exact_keys = {
        key for key in parameter_names if key in source_state and source_state[key].shape == final[key].shape
    } - remap_keys
    remapped_keys = {
        key
        for key in remap_keys
        if key in source_state
        and source_state[key].shape[1:] == final[key].shape[1:]
        and final[key].shape[0] == 1
        and torch.equal(final[key], source_state[key][0:1])
    }
    new_keys = parameter_names - exact_keys - remapped_keys
    shape_skipped_source = {
        key
        for key in source_parameter_names
        if key in final and source_state[key].shape != final[key].shape
    }
    exact_verified = all(torch.equal(final[key], source_state[key]) for key in exact_keys)
    remap_changed = all(not torch.equal(final[key], initial[key]) for key in remapped_keys)

    def remap_control(class_name: str, source_index: int | None) -> bool:
        torch.manual_seed(seed)
        control = OBBModel(deepcopy(source.yaml), nc=1, ch=3, verbose=False)
        control.names = {0: class_name}
        control_initial = {
            key: value.detach().clone() for key, value in control.state_dict().items()
        }
        control_keys = _head_classification_keys(control)
        control.load(source, verbose=False)
        control_state = control.state_dict()
        if source_index is None:
            return all(torch.equal(control_state[key], control_initial[key]) for key in control_keys)
        return all(
            torch.equal(control_state[key], source_state[key][source_index : source_index + 1])
            for key in control_keys
        )

    ship_name_control = remap_control("ship", 1)
    unknown_name_control = remap_control("not-a-dota-class", None)

    backbone_layers = len(source.yaml.get("backbone", ()))
    head_index = len(target.model) - 1
    transferred_by_group: dict[str, int] = {}
    tensors_by_group: dict[str, int] = {}
    for key in sorted(exact_keys | remapped_keys):
        group = _parameter_group(key, backbone_layers, head_index)
        transferred_by_group[group] = transferred_by_group.get(group, 0) + int(final[key].numel())
        tensors_by_group[group] = tensors_by_group.get(group, 0) + 1

    audit = {
        "source": {
            "architecture": source.__class__.__name__,
            "head": source.model[-1].__class__.__name__,
            "task": source.task,
            "parameter_count": sum(parameter.numel() for parameter in source.parameters()),
            "parameter_tensors": len(source_parameter_names),
            "class_count": len(source.names),
            "class_names": source.names,
            "plane_class_index": next(key for key, value in source.names.items() if value == "plane"),
        },
        "target": {
            "architecture": target.__class__.__name__,
            "head": target.model[-1].__class__.__name__,
            "task": target.task,
            "parameter_count": sum(parameter.numel() for parameter in target.parameters()),
            "parameter_tensors": len(parameter_names),
            "class_count": target.nc,
            "class_names": target.names,
        },
        "transfer": {
            "exact_parameter_tensors": len(exact_keys),
            "remapped_parameter_tensors": len(remapped_keys),
            "transferred_parameter_tensors": len(exact_keys | remapped_keys),
            "newly_initialized_parameter_tensors": len(new_keys),
            "shape_skipped_source_parameter_tensors": len(shape_skipped_source),
            "transferred_parameter_count": sum(final[key].numel() for key in exact_keys | remapped_keys),
            "newly_initialized_parameter_count": sum(final[key].numel() for key in new_keys),
            "exact_transfer_verified": exact_verified,
            "plane_row_remapped_by_name": bool(remapped_keys) and remap_changed,
            "name_remap_control": {
                "target_ship_received_source_row_1": ship_name_control,
                "unknown_name_retained_initial_classifier": unknown_name_control,
            },
            "remapped_keys": sorted(remapped_keys),
            "newly_initialized_keys": sorted(new_keys),
            "shape_skipped_source_keys": sorted(shape_skipped_source),
            "parameter_count_by_module_group": transferred_by_group,
            "tensor_count_by_module_group": tensors_by_group,
        },
    }
    return audit, target


def _dataset(config: Mapping[str, Any], paths: ExperimentPaths, split: str, *, augment: bool) -> Any:
    from ultralytics.cfg import DEFAULT_CFG, get_cfg
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(paths.data_yaml), autodownload=False)
    hyp = get_cfg(DEFAULT_CFG, overrides={**dict(config["training"]), "data": str(paths.data_yaml)})
    return YOLODataset(
        img_path=data[split],
        imgsz=int(config["training"]["imgsz"]),
        batch_size=1,
        augment=augment,
        hyp=hyp,
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix=f"{split}: ",
        data=data,
        task="obb",
    )


def _batch_summary(dataset: Any, split: str) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader, Subset

    positive = next(index for index, label in enumerate(dataset.labels) if len(label["cls"]) > 0)
    background = next(index for index, label in enumerate(dataset.labels) if len(label["cls"]) == 0)
    loader = DataLoader(
        Subset(dataset, [positive, background]),
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    batch = next(iter(loader))
    classes = batch["cls"].reshape(-1)
    boxes = batch["bboxes"]
    summary = {
        "split": split,
        "image_shape": list(batch["img"].shape),
        "image_dtype": str(batch["img"].dtype),
        "instances": int(classes.numel()),
        "class_ids": sorted({int(value) for value in classes.tolist()}),
        "box_shape": list(boxes.shape),
        "finite": bool(torch.isfinite(boxes).all() and torch.isfinite(classes).all()),
        "normalized_range_valid": bool(boxes.numel() == 0 or ((boxes[:, :4] >= 0).all() and (boxes[:, :4] <= 1).all())),
        "batch_indices": sorted({int(value) for value in batch["batch_idx"].tolist()}),
        "positive_index": positive,
        "background_index": background,
        "positive_image": str(dataset.im_files[positive]),
        "background_image": str(dataset.im_files[background]),
        "dataloader_batches": len(loader),
    }
    positive_loader = DataLoader(
        Subset(dataset, [positive]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    positive_batch = next(iter(positive_loader))
    return summary, positive_batch


def run_cpu_smoke(config: Mapping[str, Any], paths: ExperimentPaths) -> dict[str, Any]:
    """Run bounded one-class forward, batch-collate, and loss checks without updates."""
    require_runtime_assets(config, paths)
    import torch

    audit, target = inspect_model_transfer(paths.model, seed=int(config["training"]["seed"]))
    train_dataset = _dataset(config, paths, "train", augment=False)
    val_dataset = _dataset(config, paths, "val", augment=False)
    train_summary, loss_batch = _batch_summary(train_dataset, "train")
    val_summary, _ = _batch_summary(val_dataset, "val")

    image = loss_batch["img"].float() / 255.0
    target.eval()
    with torch.no_grad():
        output = target(image)
    output_tensor = output[0] if isinstance(output, tuple) else output
    if isinstance(output_tensor, dict):
        shapes = {key: list(value.shape) if hasattr(value, "shape") else str(type(value)) for key, value in output_tensor.items()}
    elif hasattr(output_tensor, "shape"):
        shapes = list(output_tensor.shape)
    else:
        shapes = str(type(output_tensor))

    from ultralytics.cfg import DEFAULT_CFG, get_cfg

    target.args = get_cfg(DEFAULT_CFG, overrides={**dict(config["training"]), "data": str(paths.data_yaml)})
    target.names = EXPECTED_NAMES.copy()
    target.nc = 1
    target.train()
    normalized_batch = dict(loss_batch)
    normalized_batch["img"] = image
    loss, components = target(normalized_batch)
    if isinstance(components, Mapping):
        losses = {
            str(key).removesuffix("_loss"): float(value.detach().cpu())
            for key, value in components.items()
        }
    else:
        component_names = ("box", "cls", "dfl")
        component_values = [float(value) for value in components.detach().cpu().reshape(-1)]
        losses = {
            component_names[index] if index < len(component_names) else f"component_{index}": value
            for index, value in enumerate(component_values)
        }
    loss_vector = [float(value) for value in loss.detach().cpu().reshape(-1)]
    losses["total"] = sum(loss_vector)
    losses["all_finite"] = all(math.isfinite(value) for value in losses.values())

    return {
        "transfer": audit,
        "forward": {
            "input_shape": list(image.shape),
            "input_dtype": str(image.dtype),
            "target_nc": target.nc,
            "task": target.task,
            "output": shapes,
            "cpu_only": True,
        },
        "train_batch": train_summary,
        "val_batch": val_summary,
        "loss": {**losses, "raw_loss_vector": loss_vector},
        "optimizer_steps": 0,
    }


def _training_arguments(config: Mapping[str, Any], paths: ExperimentPaths) -> dict[str, Any]:
    args = deepcopy(dict(config["training"]))
    if args.get("resume") is not False:
        raise TrainingConfigError("Fresh recipe training must not resume automatically")
    args.update(
        {
            "data": str(paths.data_yaml),
            "project": str(paths.output_dir.parent),
            "name": paths.output_dir.name,
            "exist_ok": True,
        }
    )
    return args


def launch_training(config: Mapping[str, Any], paths: ExperimentPaths) -> Any:
    """Launch the guarded real GPU run through standard Ultralytics."""
    require_runtime_assets(config, paths)
    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise TrainingConfigError("CUDA is required for full training; use --smoke-cpu for local validation")
    require_ultralytics_compatibility(config, ultralytics.__version__)
    if paths.output_dir.exists() and any(paths.output_dir.iterdir()):
        raise TrainingConfigError(f"Refusing to overwrite non-empty run directory: {paths.output_dir}")
    source_hash = sha256_file(paths.model)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_provenance = build_dataset_provenance(paths)
    write_json(paths.output_dir / "dataset_provenance.json", dataset_provenance)
    write_json(paths.output_dir / "environment_provenance.json", environment_snapshot(config, paths))
    shutil.copy2(paths.config, paths.output_dir / "experiment_config.yaml")

    model = YOLO(str(paths.model))
    result = model.train(**_training_arguments(config, paths))
    if sha256_file(paths.model) != source_hash:
        raise RuntimeError("Source best.pt changed during training")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RarePlanes one-class OBB baseline launcher")
    parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT_CONFIG))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="launch the full CUDA training run")
    mode.add_argument("--smoke-cpu", action="store_true", help="run one bounded CPU forward/loss smoke check")
    mode.add_argument("--inspect-transfer", action="store_true", help="audit 18-to-1 state-dict transfer only")
    parser.add_argument(
        "--output",
        default="outputs/training_readiness/readiness_report.json",
        help="ignored JSON path for inspection/smoke/provenance results",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, paths = load_experiment_config(args.config)
    output = _resolve(paths.root, args.output)
    if args.train:
        launch_training(config, paths)
        return 0
    if args.smoke_cpu:
        value = run_cpu_smoke(config, paths)
    elif args.inspect_transfer:
        require_runtime_assets(config, paths)
        value, _ = inspect_model_transfer(paths.model, seed=int(config["training"]["seed"]))
    else:
        value = write_readiness_provenance(config, paths, output)
        print(f"Validated configuration; provenance written to {output}")
        return 0
    write_json(output, value)
    print(json.dumps(value, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
