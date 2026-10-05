from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from api_service import (  # noqa: E402
    InferenceRuntime,
    RasterUploadError,
    ServiceSettings,
    create_app,
    validate_uploaded_raster,
)
from geospatial_inference import (  # noqa: E402
    GeospatialInferenceError,
    LoadedOBBModel,
)


MODEL_METADATA = {
    "backend": "pt",
    "model_filename": "model.pt",
    "model_sha256": "a" * 64,
    "task": "obb",
    "class_names": {0: "plane"},
    "target_class_id": 0,
    "target_class": "plane",
    "device": "cpu",
    "image_size": 1024,
    "tile_size": 1024,
    "overlap": 0.2,
    "confidence": 0.25,
    "model_iou": 0.45,
    "global_nms_iou": 0.5,
}


class FakeSummary:
    def to_dict(self) -> dict:
        return {
            "scene_id": "input",
            "raster_width": 2048,
            "raster_height": 1024,
            "raster_bands": 3,
            "raster_dtype": "uint8",
            "source_crs": "EPSG:4326",
            "output_crs": "OGC:CRS84",
            "tile_count": 3,
            "tile_size": 1024,
            "tile_overlap": 0.2,
            "integer_stride": 819,
            "image_size": 1024,
            "inference_confidence": 0.25,
            "inference_iou": 0.45,
            "global_nms_iou": 0.5,
            "all_class_raw_detections": 4,
            "raw_class_counts": {"plane": 4},
            "raw_target_candidates": 4,
            "non_target_discarded": 0,
            "invalid_or_padding_rejected": 1,
            "rejection_counts": {"outside_valid_tile_region": 1},
            "pre_nms_target_count": 3,
            "duplicates_suppressed": 1,
            "final_target_count": 2,
            "scene_processing_seconds": 1.25,
            "total_runtime_seconds": 1.25,
            "spatial_validation": {"valid": True},
        }


class FakeRuntime:
    def __init__(self, *, inference_error: Exception | None = None):
        self.inference_error = inference_error
        self.input_paths: list[Path] = []
        self.output_paths: list[Path] = []
        self.infer_calls = 0

    def model_metadata(self) -> dict:
        return dict(MODEL_METADATA)

    def infer(self, raster_path: Path, output_dir: Path):
        self.infer_calls += 1
        self.input_paths.append(raster_path)
        self.output_paths.append(output_dir)
        if self.inference_error is not None:
            raise self.inference_error
        output_dir.mkdir(parents=True)
        predictions = output_dir / "predictions.geojson"
        predictions.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[[-80.1, 26.0], [-80.0, 26.0], [-80.0, 26.1], [-80.1, 26.0]]],
                            },
                            "properties": {"confidence": 0.9},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(summary=FakeSummary(), predictions_path=predictions)


def accepting_validator(path: Path) -> dict:
    if not path.is_file():
        raise AssertionError("Temporary upload was not created")
    return {"width": 1, "height": 1, "source_crs": "EPSG:4326"}


class APIServiceTests(unittest.TestCase):
    def test_live_health_does_not_require_model(self) -> None:
        client = TestClient(create_app(ServiceSettings(model_path=None)))
        response = client.get("/health/live")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_ready_health_reports_not_ready_without_model(self) -> None:
        client = TestClient(create_app(ServiceSettings(model_path=None)))
        response = client.get("/health/ready")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "model_not_configured")

    def test_model_metadata_endpoint_is_non_sensitive(self) -> None:
        client = TestClient(
            create_app(ServiceSettings(model_path=None), runtime=FakeRuntime())
        )
        response = client.get("/model")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["target_class"], "plane")
        self.assertNotIn("path", json.dumps(response.json()).lower())

    def test_empty_upload_is_rejected(self) -> None:
        client = TestClient(
            create_app(
                ServiceSettings(model_path=None),
                runtime=FakeRuntime(),
                raster_validator=accepting_validator,
            )
        )
        response = client.post("/infer", files={"file": ("empty.tif", b"", "image/tiff")})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "empty_upload")

    def test_unsupported_upload_is_rejected(self) -> None:
        client = TestClient(create_app(ServiceSettings(model_path=None), runtime=FakeRuntime()))
        response = client.post("/infer", files={"file": ("image.jpg", b"value", "image/jpeg")})
        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["detail"]["code"], "unsupported_file")

    def test_upload_size_limit(self) -> None:
        settings = ServiceSettings(model_path=None, max_upload_bytes=3)
        client = TestClient(
            create_app(settings, runtime=FakeRuntime(), raster_validator=accepting_validator)
        )
        response = client.post("/infer", files={"file": ("large.tif", b"1234", "image/tiff")})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"]["code"], "upload_too_large")

    def test_invalid_raster_returns_clean_client_error(self) -> None:
        def reject(_: Path) -> dict:
            raise RasterUploadError("invalid_raster", "Unreadable geospatial raster", 422)

        client = TestClient(create_app(ServiceSettings(model_path=None), runtime=FakeRuntime(), raster_validator=reject))
        response = client.post("/infer", files={"file": ("bad.tif", b"not-a-tiff", "image/tiff")})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["code"], "invalid_raster")
        self.assertNotIn("Traceback", response.text)

    def test_successful_mocked_inference_response_and_geojson(self) -> None:
        runtime = FakeRuntime()
        client = TestClient(
            create_app(ServiceSettings(model_path=None), runtime=runtime, raster_validator=accepting_validator)
        )
        response = client.post("/infer", files={"file": ("scene.tif", b"tiff", "image/tiff")})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(set(payload), {"scene", "model", "inference", "geojson"})
        self.assertEqual(payload["geojson"]["type"], "FeatureCollection")
        self.assertEqual(payload["inference"]["final_target_count"], 2)
        self.assertEqual(runtime.infer_calls, 1)

    def test_temporary_upload_is_cleaned_after_response(self) -> None:
        runtime = FakeRuntime()
        client = TestClient(
            create_app(ServiceSettings(model_path=None), runtime=runtime, raster_validator=accepting_validator)
        )
        response = client.post("/infer", files={"file": ("scene.tif", b"tiff", "image/tiff")})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(runtime.input_paths[0].exists())
        self.assertFalse(runtime.output_paths[0].exists())

    def test_response_does_not_expose_server_paths_or_trust_client_path(self) -> None:
        runtime = FakeRuntime()
        client = TestClient(
            create_app(ServiceSettings(model_path=None), runtime=runtime, raster_validator=accepting_validator)
        )
        response = client.post(
            "/infer",
            files={"file": ("../../secret/scene.tif", b"tiff", "image/tiff")},
        )
        payload = response.json()
        self.assertEqual(payload["scene"]["filename"], "scene.tif")
        self.assertNotIn(str(runtime.input_paths[0].parent), json.dumps(payload))

    def test_model_sha_mismatch_produces_readiness_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.pt"
            model.write_bytes(b"model")
            settings = ServiceSettings(model, expected_sha256="0" * 64)
            client = TestClient(create_app(settings))
            response = client.get("/health/ready")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["error"]["code"], "model_sha_mismatch")

    def test_target_class_compatibility_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.pt"
            model.write_bytes(b"model")
            expected_hash = hashlib.sha256(b"model").hexdigest()

            def loader(_: Path, __: str) -> LoadedOBBModel:
                return LoadedOBBModel(object(), {0: "ship"}, "obb", "model.pt")

            runtime = InferenceRuntime(
                ServiceSettings(model, expected_sha256=expected_hash),
                model_loader=loader,
            )
            client = TestClient(create_app(runtime.settings, runtime=runtime))
            response = client.get("/health/ready")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["error"]["code"], "model_incompatible")

    def test_inference_exception_is_controlled(self) -> None:
        runtime = FakeRuntime(inference_error=GeospatialInferenceError("sensitive internal detail"))
        client = TestClient(
            create_app(ServiceSettings(model_path=None), runtime=runtime, raster_validator=accepting_validator)
        )
        response = client.post("/infer", files={"file": ("scene.tif", b"tiff", "image/tiff")})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"]["code"], "inference_failed")
        self.assertNotIn("sensitive internal detail", response.text)

    def test_runtime_loads_model_once_across_repeated_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.pt"
            model.write_bytes(b"model")
            expected_hash = hashlib.sha256(b"model").hexdigest()
            calls = []

            def loader(path: Path, device: str) -> LoadedOBBModel:
                calls.append((path, device))
                return LoadedOBBModel(object(), {0: "plane"}, "obb", "model.pt")

            runtime = InferenceRuntime(
                ServiceSettings(model, expected_sha256=expected_hash),
                model_loader=loader,
            )
            first = runtime.model_metadata()
            second = runtime.model_metadata()
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 1)
            self.assertEqual(runtime.load_count, 1)

    def test_missing_model_returns_classified_readiness_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "missing.pt"
            client = TestClient(create_app(ServiceSettings(model)))
            response = client.get("/health/ready")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["error"]["code"], "model_missing")
            self.assertNotIn(temporary, response.text)

    def test_loader_error_does_not_expose_internal_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.pt"
            model.write_bytes(b"model")

            def loader(path: Path, _: str):
                raise RuntimeError(f"Native loading error at {path}")

            runtime = InferenceRuntime(ServiceSettings(model), model_loader=loader)
            response = TestClient(create_app(runtime.settings, runtime=runtime)).get("/health/ready")
            self.assertEqual(response.status_code, 503)
            self.assertNotIn(temporary, response.text)
            self.assertNotIn("Native loading error", response.text)

    def test_non_obb_model_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.pt"
            model.write_bytes(b"model")
            runtime = InferenceRuntime(
                ServiceSettings(model),
                model_loader=lambda *_: LoadedOBBModel(object(), {0: "plane"}, "detect", "model.pt"),
            )
            response = TestClient(create_app(runtime.settings, runtime=runtime)).get("/health/ready")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["error"]["code"], "model_incompatible")

    def test_unexpected_inference_error_is_sanitized_and_cleans_upload(self) -> None:
        runtime = FakeRuntime(inference_error=RuntimeError("sensitive native error"))
        client = TestClient(create_app(ServiceSettings(None), runtime=runtime, raster_validator=accepting_validator))
        response = client.post("/infer", files={"file": ("scene.tif", b"tiff", "image/tiff")})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("sensitive native error", response.text)
        self.assertFalse(runtime.input_paths[0].parent.exists())

    def test_real_validator_rejects_corrupt_upload(self) -> None:
        client = TestClient(create_app(ServiceSettings(None), runtime=FakeRuntime()))
        response = client.post("/infer", files={"file": ("bad.tif", b"not-a-tiff", "image/tiff")})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("Traceback", response.text)

    def test_validator_rejects_singular_transform(self) -> None:
        from affine import Affine
        dataset = SimpleNamespace(
            driver="GTiff", width=10, height=10, count=3,
            crs="EPSG:4326", transform=Affine(1, 1, 0, 1, 1, 0),
            dtypes=("uint8", "uint8", "uint8"),
        )
        with patch("rasterio.open") as opened:
            opened.return_value.__enter__.return_value = dataset
            with self.assertRaises(RasterUploadError) as raised:
                validate_uploaded_raster(Path("synthetic.tif"))
            self.assertEqual(raised.exception.code, "missing_georeferencing")

    def test_inference_requests_reuse_one_loaded_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.pt"
            model.write_bytes(b"model")
            loads = []
            runner = FakeRuntime()

            def loader(path: Path, device: str):
                loads.append((path, device))
                return LoadedOBBModel(object(), {0: "plane"}, "obb", "model.pt")

            def scene_runner(_loaded, settings, **_kwargs):
                return runner.infer(settings.raster_path, settings.output_dir)

            runtime = InferenceRuntime(ServiceSettings(model), model_loader=loader, scene_runner=scene_runner)
            client = TestClient(create_app(runtime.settings, runtime=runtime, raster_validator=accepting_validator))
            for _ in range(2):
                response = client.post("/infer", files={"file": ("scene.tif", b"tiff", "image/tiff")})
                self.assertEqual(response.status_code, 200)
            self.assertEqual(len(loads), 1)
            self.assertEqual(runner.infer_calls, 2)


if __name__ == "__main__":
    unittest.main()
