from __future__ import annotations

import importlib.util
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LegacyDotaTrainingTests(unittest.TestCase):
    def test_imports_do_not_load_or_call_ultralytics(self) -> None:
        fake = types.ModuleType("ultralytics")

        def forbidden_yolo(*_args, **_kwargs):
            raise AssertionError("YOLO must not be created while importing a legacy utility")

        fake.YOLO = forbidden_yolo
        with patch.dict(sys.modules, {"ultralytics": fake}):
            load_module("train.py", "legacy_dota_train_import_test")
            load_module("resume.py", "legacy_dota_resume_import_test")

    def test_runtime_sources_have_no_workspace_assumption(self) -> None:
        for filename in ("train.py", "resume.py"):
            source = (ROOT / "src" / filename).read_text(encoding="utf-8")
            self.assertNotIn("/workspace/", source)
            self.assertIn('if __name__ == "__main__":', source)

    def test_legacy_recipe_uses_repository_relative_defaults(self) -> None:
        module = load_module("train.py", "legacy_dota_train_defaults_test")
        args = module.build_parser().parse_args([])
        values = module.legacy_training_arguments(args)
        self.assertEqual(Path(values["data"]), ROOT / "src" / "dota2_obb.yaml")
        self.assertEqual(Path(values["project"]), ROOT / "outputs" / "legacy_dota_training")
        self.assertEqual(values["epochs"], 150)
        self.assertEqual(values["optimizer"], "auto")

    def test_resume_requires_an_explicit_checkpoint(self) -> None:
        module = load_module("resume.py", "legacy_dota_resume_parser_test")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            module.build_parser().parse_args([])


if __name__ == "__main__":
    unittest.main()
