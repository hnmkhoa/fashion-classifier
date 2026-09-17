"""Checkpoint validation and model inference orchestration."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import torch

from .domain import (
    CHECKPOINT_SCHEMA_VERSION,
    CLASS_NAMES,
    INPUT_SIZE,
    MODEL_ARCHITECTURE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
)
from .model import FashionCNN
from .preprocessing import prepare_image_bytes

_MIN_ACCEPTED_PROBABILITY = 0.55
_MIN_ACCEPTED_MARGIN = 0.15
_UNKNOWN_CLASS_NAME = "Unknown"


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint violates the serving contract."""


class RankedPredictionData(TypedDict):
    class_id: int
    class_name: str
    probability: float


@dataclass(frozen=True, slots=True)
class Prediction:
    class_id: int | None
    class_name: str
    confidence: float | None
    is_unknown: bool
    rejection_reason: str | None
    top_predictions: list[RankedPredictionData]
    model_version: str
    inference_ms: float


def _require_checkpoint_contract(checkpoint: dict[str, Any]) -> None:
    required_keys = {
        "schema_version",
        "architecture",
        "model_version",
        "model_state_dict",
        "class_names",
        "normalization",
        "input_size",
    }
    missing = sorted(required_keys.difference(checkpoint))
    if missing:
        raise CheckpointCompatibilityError(
            f"Checkpoint is missing required keys: {', '.join(missing)}."
        )
    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointCompatibilityError("Checkpoint schema version is not supported.")
    if checkpoint["architecture"] != MODEL_ARCHITECTURE:
        raise CheckpointCompatibilityError("Checkpoint architecture is not supported.")
    if tuple(checkpoint["class_names"]) != CLASS_NAMES:
        raise CheckpointCompatibilityError(
            "Checkpoint class order does not match the API contract."
        )
    if tuple(checkpoint["input_size"]) != INPUT_SIZE:
        raise CheckpointCompatibilityError(
            "Checkpoint input size does not match the model contract."
        )

    normalization = checkpoint["normalization"]
    if not isinstance(normalization, dict):
        raise CheckpointCompatibilityError("Checkpoint normalization metadata is invalid.")
    try:
        mean_matches = math.isclose(
            float(normalization["mean"]), NORMALIZATION_MEAN, rel_tol=0.0, abs_tol=1e-12
        )
        std_matches = math.isclose(
            float(normalization["std"]), NORMALIZATION_STD, rel_tol=0.0, abs_tol=1e-12
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointCompatibilityError(
            "Checkpoint normalization metadata is invalid."
        ) from exc
    if not mean_matches or not std_matches:
        raise CheckpointCompatibilityError(
            "Checkpoint normalization does not match inference preprocessing."
        )


class PredictionService:
    """Own a loaded model and expose deterministic prediction behavior."""

    def __init__(self, model: FashionCNN, model_version: str, max_image_pixels: int) -> None:
        self._model = model.eval()
        self.model_version = model_version
        self.max_image_pixels = max_image_pixels

    @classmethod
    def from_checkpoint(cls, path: Path, max_image_pixels: int) -> PredictionService:
        """Load a trusted local weights-only checkpoint and validate its metadata."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise CheckpointCompatibilityError("Checkpoint root must be a dictionary.")
        _require_checkpoint_contract(checkpoint)

        model = FashionCNN(num_classes=len(CLASS_NAMES))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return cls(
            model=model,
            model_version=str(checkpoint["model_version"]),
            max_image_pixels=max_image_pixels,
        )

    def predict_bytes(self, image_bytes: bytes, top_k: int = 3) -> Prediction:
        """Preprocess one image and return a ranked prediction."""
        prepared = prepare_image_bytes(image_bytes, self.max_image_pixels)
        started = time.perf_counter()
        with torch.inference_mode():
            probabilities = torch.softmax(self._model(prepared.tensor), dim=1)[0]
        inference_ms = (time.perf_counter() - started) * 1000.0

        count = min(max(top_k, 1), len(CLASS_NAMES))
        values, indices = torch.topk(probabilities, k=count)
        ranked: list[RankedPredictionData] = [
            RankedPredictionData(
                class_id=int(class_id),
                class_name=CLASS_NAMES[int(class_id)],
                probability=float(probability),
            )
            for probability, class_id in zip(values.tolist(), indices.tolist(), strict=True)
        ]
        winner = ranked[0]
        top_two = torch.topk(probabilities, k=2).values.tolist()
        probability_margin = top_two[0] - top_two[1]
        rejection_reason = prepared.rejection_reason
        if rejection_reason is None and winner["probability"] < _MIN_ACCEPTED_PROBABILITY:
            rejection_reason = "No class reached the minimum model probability."
        elif rejection_reason is None and probability_margin < _MIN_ACCEPTED_MARGIN:
            rejection_reason = "The two most likely classes are too close to distinguish."

        is_unknown = rejection_reason is not None
        return Prediction(
            class_id=None if is_unknown else winner["class_id"],
            class_name=_UNKNOWN_CLASS_NAME if is_unknown else winner["class_name"],
            confidence=None if is_unknown else winner["probability"],
            is_unknown=is_unknown,
            rejection_reason=rejection_reason,
            top_predictions=ranked,
            model_version=self.model_version,
            inference_ms=inference_ms,
        )
