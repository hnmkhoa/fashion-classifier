import pytest
import torch

from backend.app.model import FashionCNN


def test_model_returns_ten_logits() -> None:
    model = FashionCNN()
    output = model(torch.zeros(4, 1, 28, 28))

    assert output.shape == (4, 10)


def test_model_rejects_an_incompatible_shape() -> None:
    model = FashionCNN()

    with pytest.raises(ValueError, match="Expected input shape"):
        model(torch.zeros(1, 3, 28, 28))

