import io
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from backend.app.inference import CheckpointCompatibilityError, PredictionService
from backend.app.model import FashionCNN
from backend.tests.conftest import checkpoint_payload


def fashion_like_image_bytes() -> bytes:
    image = Image.new("L", (28, 28), color=0)
    ImageDraw.Draw(image).ellipse((5, 8, 22, 20), fill=255)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def service_with_probabilities(probabilities: list[float]) -> PredictionService:
    model = FashionCNN()
    output_layer = model.classifier[-1]
    assert isinstance(output_layer, torch.nn.Linear)
    assert output_layer.bias is not None
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        output_layer.bias.copy_(torch.log(torch.tensor(probabilities)))
    return PredictionService(model, "threshold-test", max_image_pixels=1_000)


def test_checkpoint_rejects_changed_label_order(tmp_path: Path) -> None:
    path = tmp_path / "invalid.pt"
    torch.save(checkpoint_payload(class_names=["changed"] * 10), path)

    with pytest.raises(CheckpointCompatibilityError, match="class order"):
        PredictionService.from_checkpoint(path, max_image_pixels=10_000)


def test_checkpoint_rejects_changed_normalization(tmp_path: Path) -> None:
    path = tmp_path / "invalid.pt"
    torch.save(checkpoint_payload(normalization={"mean": 0.5, "std": 0.5}), path)

    with pytest.raises(CheckpointCompatibilityError, match="normalization"):
        PredictionService.from_checkpoint(path, max_image_pixels=10_000)


def test_prediction_rejects_low_maximum_probability() -> None:
    service = service_with_probabilities([0.20, *([0.80 / 9] * 9)])

    prediction = service.predict_bytes(fashion_like_image_bytes())

    assert prediction.is_unknown is True
    assert prediction.class_id is None
    assert prediction.confidence is None
    assert prediction.rejection_reason == "No class reached the minimum model probability."


def test_prediction_rejects_small_top_two_margin() -> None:
    service = service_with_probabilities([0.56, 0.43, *([0.01 / 8] * 8)])

    prediction = service.predict_bytes(fashion_like_image_bytes())

    assert prediction.is_unknown is True
    assert prediction.rejection_reason == (
        "The two most likely classes are too close to distinguish."
    )
