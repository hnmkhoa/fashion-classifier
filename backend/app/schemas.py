"""Public API response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassItem(BaseModel):
    id: int
    name: str


class ClassListResponse(BaseModel):
    classes: list[ClassItem]


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_version: str | None
    detail: str


class RankedPrediction(BaseModel):
    class_id: int
    class_name: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    request_id: str
    class_id: int | None
    class_name: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    is_unknown: bool
    rejection_reason: str | None
    top_predictions: list[RankedPrediction]
    model_version: str
    inference_ms: float = Field(ge=0.0)


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody
