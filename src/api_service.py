"""FastAPI wrapper around the validated full-scene geospatial OBB pipeline.

The service deliberately delegates model loading and inference to
``geospatial_inference``.  It adds deployment configuration, upload safety,
model lifecycle management, serialization, and HTTP error boundaries; it does
not implement a second inference path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile
import threading
from typing import Any, Callable, Mapping

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

try:
    from .geospatial_inference import (
        GeospatialInferenceError,
        LoadedOBBModel,
        SceneInferenceSettings,
        load_obb_model,
        process_raster_scene,
        resolve_target_class,
    )
except ImportError:  # Direct import when src/ is placed on sys.path.
    from geospatial_inference import (
        GeospatialInferenceError,
        LoadedOBBModel,
        SceneInferenceSettings,
        load_obb_model,
        process_raster_scene,
        resolve_target_class,
    )


DEFAULT_MAX_UPLOAD_BYTES = 512 * 1024 * 1024
SUPPORTED_SUFFIXES = {".tif", ".tiff"}
SUPPORTED_CONTENT_TYPES = {
    "application/geotiff",
    "application/octet-stream",
    "application/x-geotiff",
    "image/geotiff",
    "image/tiff",
}


class ServiceError(RuntimeError):
    """A safe, classified service error suitable for an HTTP response."""

    def __init__(self, code: str, message: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class RasterUploadError(ServiceError):
    pass


@dataclass(frozen=True)
class ServiceSettings:
    model_path: Path | None
    expected_sha256: str | None = None
    backend: str = "auto"
    device: str = "cpu"
    target_class: str = "plane"
    image_size: int = 1024
    tile_size: int = 1024
    tile_overlap: float = 0.20
    confidence: float = 0.25
    model_iou: float = 0.45
    global_nms_iou: float = 0.50
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES

    def __post_init__(self) -> None:
        backend = self.backend.lower()
        if backend not in {"auto", "pt", "onnx"}:
            raise ValueError("GEO_OBB_BACKEND must be auto, pt, or onnx")
        if self.device.lower() != "cpu":
            raise ValueError("The validated geospatial service currently requires CPU")
        if self.expected_sha256 is not None:
            digest = self.expected_sha256.lower().strip()
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("GEO_OBB_MODEL_SHA256 must be a 64-character hexadecimal digest")
        if self.max_upload_bytes <= 0:
            raise ValueError("GEO_OBB_MAX_UPLOAD_BYTES must be positive")

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> "ServiceSettings":
        values = os.environ if environment is None else environment
        model_value = values.get("GEO_OBB_MODEL_PATH", "").strip()
        hash_value = values.get("GEO_OBB_MODEL_SHA256", "").strip()
        return cls(
            model_path=Path(model_value).expanduser() if model_value else None,
            expected_sha256=hash_value or None,
            backend=values.get("GEO_OBB_BACKEND", "auto"),
            device=values.get("GEO_OBB_DEVICE", "cpu"),
            max_upload_bytes=int(values.get("GEO_OBB_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_backend(settings: ServiceSettings) -> str:
    if settings.model_path is None:
        raise ServiceError("model_not_configured", "Model path is not configured", 503)
    suffix = settings.model_path.suffix.lower()
    if suffix not in {".pt", ".onnx"}:
        raise ServiceError("unsupported_model", "Configured model must be a .pt or .onnx OBB model", 503)
    inferred = suffix.removeprefix(".")
    if settings.backend.lower() != "auto" and settings.backend.lower() != inferred:
        raise ServiceError("backend_mismatch", "Configured backend does not match the model file", 503)
    return inferred


class InferenceRuntime:
    """Lazily load one model and serialize GDAL/model inference requests."""

    def __init__(
        self,
        settings: ServiceSettings,
        *,
        model_loader: Callable[[Path, str], LoadedOBBModel] = load_obb_model,
        scene_runner: Callable[..., Any] = process_raster_scene,
    ):
        self.settings = settings
        self._model_loader = model_loader
        self._scene_runner = scene_runner
        self._initialization_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._loaded_model: LoadedOBBModel | None = None
        self._model_sha256: str | None = None
        self._backend: str | None = None
        self._load_count = 0

    @property
    def load_count(self) -> int:
        return self._load_count

    def ensure_ready(self) -> LoadedOBBModel:
        if self._loaded_model is not None:
            return self._loaded_model
        with self._initialization_lock:
            if self._loaded_model is not None:
                return self._loaded_model
            model_path = self.settings.model_path
            backend = _resolved_backend(self.settings)
            if model_path is None or not model_path.is_file():
                filename = model_path.name if model_path else "<not configured>"
                raise ServiceError("model_missing", f"Configured model is unavailable: {filename}", 503)
            try:
                actual_sha256 = sha256_file(model_path)
            except OSError as exc:
                raise ServiceError("model_unreadable", "Configured model cannot be read", 503) from exc
            if (
                self.settings.expected_sha256 is not None
                and actual_sha256.lower() != self.settings.expected_sha256.lower()
            ):
                raise ServiceError("model_sha_mismatch", "Configured model SHA256 does not match", 503)
            try:
                for dependency in ("rasterio", "pyproj", "shapely", "numpy"):
                    importlib.import_module(dependency)
            except ImportError as exc:
                raise ServiceError("runtime_unavailable", "A required geospatial runtime dependency is unavailable", 503) from exc
            try:
                loaded = self._model_loader(model_path, self.settings.device)
            except Exception as exc:
                raise ServiceError("model_incompatible", "Configured model could not be loaded as a compatible OBB model", 503) from exc
            if loaded.task != "obb" or not loaded.names:
                raise ServiceError("model_incompatible", "Configured model requires OBB task and non-empty class metadata", 503)
            try:
                resolve_target_class(self.settings.target_class, loaded.names)
            except (GeospatialInferenceError, ValueError) as exc:
                raise ServiceError("model_incompatible", "Configured target class is missing or ambiguous in model metadata", 503) from exc
            self._loaded_model = loaded
            self._model_sha256 = actual_sha256
            self._backend = backend
            self._load_count += 1
            return loaded

    def model_metadata(self) -> dict[str, Any]:
        loaded = self.ensure_ready()
        target_id, target_name = resolve_target_class(self.settings.target_class, loaded.names)
        return {
            "backend": self._backend,
            "model_filename": loaded.model_name,
            "model_sha256": self._model_sha256,
            "task": loaded.task,
            "class_names": loaded.names,
            "target_class_id": target_id,
            "target_class": target_name,
            "device": self.settings.device,
            "image_size": self.settings.image_size,
            "tile_size": self.settings.tile_size,
            "overlap": self.settings.tile_overlap,
            "confidence": self.settings.confidence,
            "model_iou": self.settings.model_iou,
            "global_nms_iou": self.settings.global_nms_iou,
        }

    def infer(self, raster_path: Path, output_dir: Path) -> Any:
        loaded = self.ensure_ready()
        settings = SceneInferenceSettings(
            model_path=self.settings.model_path or Path(loaded.model_name),
            raster_path=raster_path,
            output_dir=output_dir,
            device=self.settings.device,
            image_size=self.settings.image_size,
            confidence=self.settings.confidence,
            model_iou=self.settings.model_iou,
            global_nms_iou=self.settings.global_nms_iou,
            tile_size=self.settings.tile_size,
            tile_overlap=self.settings.tile_overlap,
            target_class=self.settings.target_class,
        )
        with self._inference_lock:
            return self._scene_runner(loaded, settings, model_load_seconds=0.0)


def validate_uploaded_raster(path: Path) -> dict[str, Any]:
    """Perform bounded metadata-only validation before model inference."""

    try:
        import rasterio

        with rasterio.open(path) as dataset:
            if dataset.driver != "GTiff":
                raise RasterUploadError("unsupported_raster", "Raster driver must be GeoTIFF", 422)
            if dataset.width <= 0 or dataset.height <= 0:
                raise RasterUploadError("invalid_raster", "Raster dimensions are invalid", 422)
            if dataset.count != 3:
                raise RasterUploadError("invalid_raster", "Raster must contain exactly three bands", 422)
            transform = dataset.transform
            determinant = transform.a * transform.e - transform.b * transform.d
            if (
                dataset.crs is None
                or transform.is_identity
                or not all(math.isfinite(value) for value in transform)
                or not math.isfinite(determinant)
                or determinant == 0
            ):
                raise RasterUploadError("missing_georeferencing", "Raster requires a valid CRS and affine transform", 422)
            if any(str(dtype) != "uint8" for dtype in dataset.dtypes):
                raise RasterUploadError("invalid_raster", "Raster bands must be uint8", 422)
            return {
                "driver": dataset.driver,
                "width": dataset.width,
                "height": dataset.height,
                "bands": dataset.count,
                "dtype": str(dataset.dtypes[0]),
                "source_crs": str(dataset.crs),
            }
    except RasterUploadError:
        raise
    except Exception as exc:
        raise RasterUploadError("invalid_raster", "Upload is not a readable georeferenced raster", 422) from exc


def _safe_client_filename(value: str | None) -> str:
    if not value:
        return "upload.tif"
    return PurePosixPath(PureWindowsPath(value).name).name or "upload.tif"


async def _save_upload(upload: UploadFile, destination: Path, limit: int) -> int:
    total = 0
    with destination.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise RasterUploadError(
                    "upload_too_large",
                    f"Upload exceeds the configured {limit}-byte limit",
                    413,
                )
            handle.write(chunk)
    if total == 0:
        raise RasterUploadError("empty_upload", "Uploaded raster is empty", 400)
    return total


def _safe_error(error: ServiceError) -> dict[str, Any]:
    return {"status": "not_ready", "error": {"code": error.code, "message": error.message}}


def _response_from_result(
    result: Any,
    *,
    client_filename: str,
    model_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    summary = result.summary.to_dict()
    geojson = json.loads(result.predictions_path.read_text(encoding="utf-8"))
    inference_keys = (
        "tile_count",
        "tile_size",
        "tile_overlap",
        "integer_stride",
        "image_size",
        "inference_confidence",
        "inference_iou",
        "global_nms_iou",
        "all_class_raw_detections",
        "raw_class_counts",
        "raw_target_candidates",
        "non_target_discarded",
        "invalid_or_padding_rejected",
        "rejection_counts",
        "pre_nms_target_count",
        "duplicates_suppressed",
        "final_target_count",
        "scene_processing_seconds",
        "total_runtime_seconds",
        "spatial_validation",
    )
    return {
        "scene": {
            "filename": client_filename,
            "scene_id": summary["scene_id"],
            "source_crs": summary["source_crs"],
            "output_crs": summary["output_crs"],
            "width": summary["raster_width"],
            "height": summary["raster_height"],
            "bands": summary["raster_bands"],
            "dtype": summary["raster_dtype"],
        },
        "model": dict(model_metadata),
        "inference": {key: summary[key] for key in inference_keys},
        "geojson": geojson,
    }


def create_app(
    settings: ServiceSettings | None = None,
    *,
    runtime: Any | None = None,
    raster_validator: Callable[[Path], Mapping[str, Any]] = validate_uploaded_raster,
) -> FastAPI:
    service_settings = settings or ServiceSettings.from_environment()
    service_runtime = runtime or InferenceRuntime(service_settings)
    app = FastAPI(
        title="Geo-Aware Aerial OBB Inference API",
        version="1.0.0",
        description="Local/portfolio API for full-scene geospatial aircraft OBB inference.",
    )
    app.state.runtime = service_runtime
    app.state.settings = service_settings

    @app.get("/health/live")
    def health_live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def health_ready() -> Any:
        try:
            metadata = service_runtime.model_metadata()
        except ServiceError as exc:
            return JSONResponse(status_code=exc.status_code, content=_safe_error(exc))
        return {
            "status": "ready",
            "backend": metadata["backend"],
            "model_filename": metadata["model_filename"],
            "task": metadata["task"],
            "target_class": metadata["target_class"],
        }

    @app.get("/model")
    def model_metadata() -> Any:
        try:
            return service_runtime.model_metadata()
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": exc.message}) from None

    @app.post("/infer")
    async def infer(file: UploadFile = File(...)) -> Any:
        try:
            filename = _safe_client_filename(file.filename)
            suffix = Path(filename).suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                raise HTTPException(status_code=415, detail={"code": "unsupported_file", "message": "Only .tif and .tiff uploads are supported"})
            content_type = (file.content_type or "application/octet-stream").lower()
            if content_type not in SUPPORTED_CONTENT_TYPES:
                raise HTTPException(status_code=415, detail={"code": "unsupported_media_type", "message": "Unsupported raster media type"})
            with tempfile.TemporaryDirectory(prefix="geo-obb-api-") as temporary:
                temporary_root = Path(temporary)
                upload_path = temporary_root / f"input{suffix}"
                await _save_upload(file, upload_path, service_settings.max_upload_bytes)
                await run_in_threadpool(raster_validator, upload_path)
                result = await run_in_threadpool(service_runtime.infer, upload_path, temporary_root / "output")
                metadata = service_runtime.model_metadata()
                return _response_from_result(result, client_filename=filename, model_metadata=metadata)
        except RasterUploadError as exc:
            raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": exc.message}) from None
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": exc.message}) from None
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=500, detail={"code": "inference_failed", "message": "Geospatial inference failed"}) from None
        finally:
            await file.close()

    return app


app = create_app()
