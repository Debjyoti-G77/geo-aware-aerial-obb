from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_rareplanes_obb import (  # noqa: E402
    EXPECTED_NAMES,
    EXPECTED_TRAINING_RECIPE,
    EXPECTED_ULTRALYTICS_VERSION,
    FINAL_BEST_EPOCH,
    FINAL_CHECKPOINT_SHA256,
    SOURCE_CHECKPOINT_SHA256,
    ExperimentPaths,
    TrainingConfigError,
    _training_arguments,
    _validate_completed_run,
    build_dataset_provenance,
    build_parser,
    environment_snapshot,
    launch_training,
    load_experiment_config,
    require_runtime_assets,
    require_ultralytics_compatibility,
    resolve_framework_settings,
    sha256_file,
    verify_dataset_provenance,
)


def write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def baseline_config(*, output: str = "outputs/training/rareplanes_baseline_01_full") -> dict:
    training = dict(EXPECTED_TRAINING_RECIPE)
    training["bgr"] = 0.0
    return {
        "experiment": {
            "name": "rareplanes_baseline_01_full",
            "seed": 20261001,
            "output_dir": output,
        },
        "framework": {"ultralytics_version": "8.4.165"},
        "model": {
            "path": "best.pt",
            "sha256": SOURCE_CHECKPOINT_SHA256,
            "expected_task": "obb",
            "source_classes": 18,
        },
        "data": {"yaml": "configs/rareplanes_obb.yaml", "expected_classes": {0: "plane"}},
        "training": training,
        "reporting": {"use_rareplanes_test": False, "snow_subset": "descriptive_only_one_scene"},
        "completed_run": {
            "status": "complete",
            "run_name": "rareplanes_baseline_01_full",
            "execution_environment": "google_colab",
            "accelerator": "NVIDIA Tesla T4",
            "ultralytics_version": "8.4.165",
            "initial_epochs_completed": 10,
            "resumed_from_epoch": 11,
            "resume_checkpoint": {
                "source": "last.pt",
                "automatic": False,
                "optimizer_state_preserved": True,
                "only_checkpoint_change": {
                    "field": "train_args.patience",
                    "value": 20,
                },
            },
            "stopped_epoch": 79,
            "best_epoch": 59,
            "best_checkpoint": {
                "path": "rareplanes_finetuned_best.pt",
                "sha256": FINAL_CHECKPOINT_SHA256,
            },
            "validation_metrics": {
                "precision": 0.892,
                "recall": 0.976,
                "map50": 0.959,
                "map50_95": 0.815,
            },
        },
    }


def create_project(root: Path, config: dict | None = None, dataset: dict | None = None) -> Path:
    (root / "best.pt").write_bytes(b"model")
    (root / "datasets" / "RarePlanes_yolo_obb" / "images" / "train").mkdir(parents=True)
    (root / "datasets" / "RarePlanes_yolo_obb" / "images" / "val").mkdir(parents=True)
    write_yaml(
        root / "configs" / "rareplanes_obb.yaml",
        dataset
        or {
            "path": "datasets/RarePlanes_yolo_obb",
            "train": "images/train",
            "val": "images/val",
            "names": {0: "plane"},
        },
    )
    config_path = root / "configs" / "experiment.yaml"
    write_yaml(config_path, config or baseline_config())
    return config_path


class TrainingConfigurationTests(unittest.TestCase):
    def test_experiment_config_parses_and_freezes_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = create_project(root)
            config, paths = load_experiment_config(config_path, project_root=root)
            self.assertEqual(config["training"]["batch"], 8)
            self.assertEqual(config["training"]["seed"], 20261001)
            self.assertEqual(config["training"]["optimizer"], "AdamW")
            self.assertEqual(config["training"]["patience"], 20)
            self.assertEqual(paths.model, (root / "best.pt").resolve())
            self.assertEqual(config["model"]["sha256"], SOURCE_CHECKPOINT_SHA256)
            self.assertEqual(
                paths.output_dir,
                (root / "outputs/training/rareplanes_baseline_01_full").resolve(),
            )

    def test_source_best_pt_cannot_be_an_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = create_project(root, baseline_config(output="best.pt"))
            with self.assertRaisesRegex(TrainingConfigError, "outputs/training"):
                load_experiment_config(config_path, project_root=root)

    def test_output_path_is_isolated_under_training_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = create_project(root, baseline_config(output="outputs/not-training/run"))
            with self.assertRaisesRegex(TrainingConfigError, "outputs/training"):
                load_experiment_config(config_path, project_root=root)

    def test_dataset_yaml_rejects_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = {
                "path": "datasets/RarePlanes_yolo_obb",
                "train": "images/train",
                "val": "images/val",
                "test": "../RarePlanes/test/imagery",
                "names": {0: "plane"},
            }
            config_path = create_project(root, dataset=dataset)
            with self.assertRaisesRegex(TrainingConfigError, "TEST"):
                load_experiment_config(config_path, project_root=root)

    def test_dataset_must_be_one_class_plane(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = {
                "path": "datasets/RarePlanes_yolo_obb",
                "train": "images/train",
                "val": "images/val",
                "names": {0: "plane", 1: "ship"},
            }
            config_path = create_project(root, dataset=dataset)
            with self.assertRaisesRegex(TrainingConfigError, "only class"):
                load_experiment_config(config_path, project_root=root)

    def test_mismatched_seed_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = baseline_config()
            config["training"]["seed"] = 1
            config_path = create_project(root, config)
            with self.assertRaisesRegex(TrainingConfigError, "seeds must match"):
                load_experiment_config(config_path, project_root=root)

    def test_recipe_cannot_enable_automatic_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = baseline_config()
            config["training"]["resume"] = True
            config_path = create_project(root, config)
            with self.assertRaisesRegex(TrainingConfigError, "training.resume"):
                load_experiment_config(config_path, project_root=root)

    def test_source_checkpoint_path_and_hash_metadata_are_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = baseline_config()
            config["model"]["path"] = "another.pt"
            config_path = create_project(root, config)
            with self.assertRaisesRegex(TrainingConfigError, "repository-relative best.pt"):
                load_experiment_config(config_path, project_root=root)

            config = baseline_config()
            config["model"]["sha256"] = "0" * 64
            write_yaml(config_path, config)
            with self.assertRaisesRegex(TrainingConfigError, "metadata is inconsistent"):
                load_experiment_config(config_path, project_root=root)

    def test_runtime_readiness_requires_source_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = create_project(root)
            (root / "best.pt").unlink()
            config, paths = load_experiment_config(config_path, project_root=root)
            with self.assertRaisesRegex(TrainingConfigError, "Source model not found"):
                require_runtime_assets(config, paths)

    def test_runtime_readiness_enforces_source_checkpoint_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = create_project(root)
            config, paths = load_experiment_config(config_path, project_root=root)
            with self.assertRaisesRegex(TrainingConfigError, "SHA-256 mismatch"):
                require_runtime_assets(config, paths)
            with mock.patch(
                "train_rareplanes_obb.sha256_file",
                return_value=SOURCE_CHECKPOINT_SHA256,
            ):
                require_runtime_assets(config, paths)

    def test_historical_resume_is_provenance_only(self) -> None:
        config = baseline_config()
        completed = _validate_completed_run(config)
        self.assertFalse(completed["resume_checkpoint"]["automatic"])
        self.assertTrue(completed["resume_checkpoint"]["optimizer_state_preserved"])
        self.assertEqual(completed["resumed_from_epoch"], 11)
        self.assertEqual(completed["best_epoch"], FINAL_BEST_EPOCH)
        self.assertEqual(
            completed["best_checkpoint"]["sha256"], FINAL_CHECKPOINT_SHA256
        )

    def test_historical_provenance_cannot_enable_automatic_resume(self) -> None:
        config = baseline_config()
        config["completed_run"]["resume_checkpoint"]["automatic"] = True
        with self.assertRaisesRegex(TrainingConfigError, "automatic resume"):
            _validate_completed_run(config)

    def test_ultralytics_version_expectation_is_explicit(self) -> None:
        config = baseline_config()
        require_ultralytics_compatibility(config, EXPECTED_ULTRALYTICS_VERSION)
        with self.assertRaisesRegex(TrainingConfigError, "version mismatch"):
            require_ultralytics_compatibility(config, "8.4.164")

    def test_real_training_requires_cuda(self) -> None:
        config = baseline_config()
        root = Path("C:/safe/project")
        paths = ExperimentPaths(
            root=root,
            config=root / "configs/experiment.yaml",
            model=root / "best.pt",
            data_yaml=root / "configs/data.yaml",
            dataset_root=root / "datasets/generated",
            output_dir=root / "outputs/training/run",
        )
        with mock.patch("train_rareplanes_obb.require_runtime_assets"):
            with mock.patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(TrainingConfigError, "CUDA is required"):
                    launch_training(config, paths)

    def test_cpu_smoke_mode_is_explicitly_available(self) -> None:
        args = build_parser().parse_args(["--smoke-cpu"])
        self.assertTrue(args.smoke_cpu)
        self.assertFalse(args.train)

    def test_training_arguments_are_deterministic_and_explicit(self) -> None:
        config = baseline_config()
        root = Path("C:/project")
        paths = ExperimentPaths(
            root=root,
            config=root / "configs/experiment.yaml",
            model=root / "best.pt",
            data_yaml=root / "configs/data.yaml",
            dataset_root=root / "datasets/generated",
            output_dir=root / "outputs/training/rareplanes_baseline_01",
        )
        first = _training_arguments(config, paths)
        second = _training_arguments(config, paths)
        self.assertEqual(first, second)
        self.assertEqual(first["batch"], 8)
        self.assertEqual(first["seed"], 20261001)
        self.assertEqual(first["device"], 0)
        self.assertEqual(first["imgsz"], 1024)
        self.assertEqual(first["task"], "obb")
        self.assertEqual(first["optimizer"], "AdamW")
        self.assertEqual(first["lr0"], 0.002)
        self.assertEqual(first["momentum"], 0.9)
        self.assertEqual(first["patience"], 20)
        self.assertFalse(first["resume"])
        self.assertNotIn("completed_run", first)

    def test_framework_explicit_optimizer_resolution_is_recorded(self) -> None:
        resolved = resolve_framework_settings(baseline_config(), train_image_count=2751)
        optimizer = resolved["optimizer_resolution"]
        self.assertEqual(optimizer["selection_iterations"], 4300)
        self.assertEqual(optimizer["selected_optimizer"], "AdamW")
        self.assertEqual(optimizer["selected_lr0"], 0.002)
        self.assertEqual(optimizer["selected_momentum"], 0.9)
        self.assertFalse(optimizer["configured_lr0_ignored_by_auto"])
        self.assertEqual(optimizer["gradient_accumulation"], 8)


class ProvenanceTests(unittest.TestCase):
    def create_provenance_project(self, root: Path) -> tuple[dict, ExperimentPaths]:
        config_path = create_project(root)
        config, paths = load_experiment_config(config_path, project_root=root)
        dataset_file = paths.dataset_root / "images/train/chip.png"
        dataset_file.write_bytes(b"chip")
        inputs = {
            root / "outputs/rareplanes_dataset_preparation/split_manifest.csv": "split",
            root / "datasets/RarePlanes_yolo_obb/manifests/objects.csv": "objects",
        }
        for path, value in inputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8")
        audit_path = root / "outputs/rareplanes_dataset_preparation/dataset_audit.json"
        audit_path.write_text(
            json.dumps(
                {
                    "protocol": {"tile_size": 1024, "tile_overlap": 0.2},
                    "selection": {"seed": 20261001, "selected_validation_location_ids": [7, 10]},
                    "audit": {"train_val_loc_id_overlap": []},
                }
            ),
            encoding="utf-8",
        )
        return config, paths

    def test_provenance_snapshot_and_hash_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _config, paths = self.create_provenance_project(root)
            manifest = build_dataset_provenance(paths)
            self.assertGreaterEqual(manifest["file_count"], 2)
            self.assertEqual(manifest["split_selection"]["seed"], 20261001)
            self.assertEqual(verify_dataset_provenance(manifest, root), [])
            self.assertEqual(len(manifest["hashes"]["source_model"]["sha256"]), 64)

            (root / "best.pt").write_bytes(b"changed")
            self.assertIn("source_model: SHA-256 mismatch", verify_dataset_provenance(manifest, root))

    def test_environment_snapshot_records_fixed_seed_and_versions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, paths = self.create_provenance_project(root)
            with mock.patch("train_rareplanes_obb._git_head", return_value="abc123"):
                snapshot = environment_snapshot(config, paths)
            self.assertEqual(snapshot["git_head"], "abc123")
            self.assertEqual(snapshot["experiment_config"]["training"]["seed"], 20261001)
            self.assertTrue(snapshot["ultralytics_version_status"]["compatible"])
            self.assertEqual(snapshot["completed_run_provenance"]["best_epoch"], 59)
            self.assertIn("torch", snapshot)
            self.assertIn("ultralytics", snapshot)

    def test_sha256_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "value.bin"
            path.write_bytes(b"rareplanes")
            self.assertEqual(sha256_file(path), sha256_file(path))


class RepositoryBaselineTests(unittest.TestCase):
    def test_committed_experiment_config_has_required_baseline(self) -> None:
        config, paths = load_experiment_config(project_root=REPOSITORY_ROOT)
        self.assertEqual(config["data"]["expected_classes"], EXPECTED_NAMES)
        self.assertEqual(config["training"]["epochs"], 100)
        self.assertEqual(config["training"]["patience"], 20)
        self.assertEqual(config["training"]["batch"], 8)
        self.assertEqual(config["training"]["imgsz"], 1024)
        self.assertEqual(config["training"]["optimizer"], "AdamW")
        self.assertEqual(config["training"]["lr0"], 0.002)
        self.assertEqual(config["training"]["momentum"], 0.9)
        self.assertFalse(config["training"]["resume"])
        self.assertEqual(config["experiment"]["name"], "rareplanes_baseline_01_full")
        self.assertEqual(config["framework"]["ultralytics_version"], "8.4.165")
        self.assertEqual(config["completed_run"]["best_epoch"], 59)
        self.assertEqual(config["model"]["path"], "best.pt")
        self.assertEqual(config["model"]["sha256"], SOURCE_CHECKPOINT_SHA256)
        self.assertEqual(
            config["completed_run"]["best_checkpoint"]["sha256"],
            FINAL_CHECKPOINT_SHA256,
        )
        self.assertEqual(paths.data_yaml.name, "rareplanes_obb.yaml")


if __name__ == "__main__":
    unittest.main()
