from pathlib import Path
from typing import Any

import pytest
import torch

from backend.app.config import Settings
from backend.app.domain import (
    CHECKPOINT_SCHEMA_VERSION,
    CLASS_NAMES,
    INPUT_SIZE,
    MODEL_ARCHITECTURE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
)
from backend.app.model import FashionCNN


def checkpoint_payload(**overrides: Any) -> dict[str, Any]:
    model = FashionCNN()
    output_layer = model.classifier[-1]
    assert isinstance(output_layer, torch.nn.Linear)
    with torch.no_grad():
        output_layer.weight.zero_()
        output_layer.bias.fill_(-5.0)
        output_layer.bias[7] = 5.0

    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "architecture": MODEL_ARCHITECTURE,
        "model_version": "test-model-v1",
        "model_state_dict": model.state_dict(),
        "class_names": list(CLASS_NAMES),
        "normalization": {"mean": NORMALIZATION_MEAN, "std": NORMALIZATION_STD},
        "input_size": list(INPUT_SIZE),
        "training": {"seed": 42},
        "metrics": {},
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def valid_checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "fashion_cnn.pt"
    torch.save(checkpoint_payload(), path)
    return path


def make_settings(model_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "model_path": model_path,
        "max_upload_bytes": 5 * 1024 * 1024,
        "max_image_pixels": 1_000_000,
        "cors_origins": ("http://localhost:5173",),
    }
    values.update(overrides)
    return Settings(**values)
