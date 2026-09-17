"""FastAPI application factory and HTTP endpoints."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .config import Settings
from .domain import CLASS_NAMES, SUPPORTED_MEDIA_TYPES
from .errors import AppError
from .inference import PredictionService
from .preprocessing import ImageDecodingError
from .schemas import (
    ClassItem,
    ClassListResponse,
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    RankedPrediction,
)

LOGGER = logging.getLogger("fashion_classifier.api")


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", uuid4()))


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an application instance with explicit, testable dependencies."""
    runtime_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = None
        app.state.model_error = None
        if runtime_settings.model_path.is_file():
            try:
                app.state.predictor = PredictionService.from_checkpoint(
                    runtime_settings.model_path,
                    max_image_pixels=runtime_settings.max_image_pixels,
                )
                LOGGER.info("Loaded model from %s", runtime_settings.model_path)
            except Exception as exc:  # Startup remains inspectable through the health endpoint.
                app.state.model_error = str(exc)
                LOGGER.exception("Could not load model from %s", runtime_settings.model_path)
        else:
            app.state.model_error = f"Checkpoint not found at {runtime_settings.model_path}."
            LOGGER.warning(app.state.model_error)
        yield

    app = FastAPI(
        title="Fashion Image Classifier API",
        summary="Classify one image into the ten Fashion-MNIST categories.",
        version="1.1.0",
        lifespan=lifespan,
    )
    app.state.settings = runtime_settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(runtime_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
        request.state.request_id = request.headers.get("X-Request-ID", str(uuid4()))
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        payload = {
            "error": {
                "code": exc.code,
                "message": exc.message,
                "request_id": _request_id(request),
            }
        }
        return JSONResponse(status_code=exc.status_code, content=payload)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        LOGGER.info("Request validation failed: %s", exc)
        payload = {
            "error": {
                "code": "invalid_request",
                "message": "The request does not match the API contract.",
                "request_id": _request_id(request),
            }
        }
        return JSONResponse(status_code=422, content=payload)

    @app.get("/health", response_model=HealthResponse, tags=["Operations"])
    async def health(request: Request) -> HealthResponse:
        predictor: PredictionService | None = request.app.state.predictor
        if predictor is None:
            return HealthResponse(
                status="degraded",
                model_ready=False,
                model_version=None,
                detail=request.app.state.model_error or "Model is not ready.",
            )
        return HealthResponse(
            status="ok",
            model_ready=True,
            model_version=predictor.model_version,
            detail="Model is ready.",
        )

    @app.get(
        "/api/v1/classes",
        response_model=ClassListResponse,
        tags=["Classification"],
    )
    async def list_classes() -> ClassListResponse:
        return ClassListResponse(
            classes=[ClassItem(id=index, name=name) for index, name in enumerate(CLASS_NAMES)]
        )

    @app.post(
        "/api/v1/predict",
        response_model=PredictionResponse,
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["Classification"],
    )
    async def predict(
        request: Request, file: Annotated[UploadFile, File()]
    ) -> PredictionResponse:
        predictor: PredictionService | None = request.app.state.predictor
        if predictor is None:
            raise AppError(
                503,
                "model_unavailable",
                "The model is not ready. Train a checkpoint and restart the API.",
            )
        if file.content_type not in SUPPORTED_MEDIA_TYPES:
            raise AppError(
                400,
                "unsupported_media_type",
                "Upload a PNG, JPEG, or WebP image.",
            )

        try:
            image_bytes = await file.read(runtime_settings.max_upload_bytes + 1)
        finally:
            await file.close()
        if len(image_bytes) > runtime_settings.max_upload_bytes:
            raise AppError(
                413,
                "file_too_large",
                f"The image must not exceed {runtime_settings.max_upload_bytes} bytes.",
            )

        try:
            prediction = await run_in_threadpool(predictor.predict_bytes, image_bytes)
        except ImageDecodingError as exc:
            raise AppError(400, "invalid_image", str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("Prediction failed for request %s", _request_id(request))
            raise AppError(
                500,
                "prediction_failed",
                "The image could not be classified due to an internal error.",
            ) from exc

        return PredictionResponse(
            request_id=_request_id(request),
            class_id=prediction.class_id,
            class_name=prediction.class_name,
            confidence=prediction.confidence,
            is_unknown=prediction.is_unknown,
            rejection_reason=prediction.rejection_reason,
            top_predictions=[RankedPrediction(**item) for item in prediction.top_predictions],
            model_version=prediction.model_version,
            inference_ms=prediction.inference_ms,
        )

    return app


app = create_app()
