import io

import pytest
import torch
from PIL import Image, ImageDraw

from backend.app.domain import NORMALIZATION_MEAN, NORMALIZATION_STD
from backend.app.preprocessing import (
    ImageDecodingError,
    prepare_image_bytes,
    preprocess_image_bytes,
)


def image_bytes(size: tuple[int, int] = (56, 28)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(255, 255, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def object_image_bytes(background: int, foreground: int) -> bytes:
    image = Image.new("L", (120, 80), color=background)
    draw = ImageDraw.Draw(image)
    draw.ellipse((30, 20, 90, 60), fill=foreground)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_preprocessing_returns_expected_shape_and_finite_values() -> None:
    tensor = preprocess_image_bytes(image_bytes(), max_image_pixels=10_000)

    assert tensor.shape == (1, 1, 28, 28)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_preprocessing_rejects_non_image_bytes() -> None:
    with pytest.raises(ImageDecodingError, match="could not be decoded"):
        preprocess_image_bytes(b"plain text", max_image_pixels=10_000)


def test_preprocessing_rejects_too_many_pixels() -> None:
    with pytest.raises(ImageDecodingError, match="pixel limit"):
        preprocess_image_bytes(image_bytes((100, 100)), max_image_pixels=9_999)


def test_light_and_dark_backgrounds_produce_equivalent_foregrounds() -> None:
    dark_on_light = preprocess_image_bytes(
        object_image_bytes(background=255, foreground=0), max_image_pixels=20_000
    )
    light_on_dark = preprocess_image_bytes(
        object_image_bytes(background=0, foreground=255), max_image_pixels=20_000
    )

    assert torch.allclose(dark_on_light, light_on_dark, atol=1e-6)
    expected_background = (0.0 - NORMALIZATION_MEAN) / NORMALIZATION_STD
    assert dark_on_light[0, 0, 0, 0].item() == pytest.approx(expected_background)
    assert dark_on_light.max().item() > 1.0


def test_native_fashion_mnist_input_is_not_recropped() -> None:
    image = Image.new("L", (28, 28), color=0)
    ImageDraw.Draw(image).rectangle((8, 8, 19, 19), fill=128)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    tensor = preprocess_image_bytes(buffer.getvalue(), max_image_pixels=1_000)

    expected_center = ((128.0 / 255.0) - NORMALIZATION_MEAN) / NORMALIZATION_STD
    assert tensor[0, 0, 10, 10].item() == pytest.approx(expected_center)


def test_complex_scene_is_marked_out_of_domain() -> None:
    image = Image.new("L", (120, 80), color=0)
    draw = ImageDraw.Draw(image)
    for y in range(0, 80, 10):
        for x in range(0, 120, 10):
            draw.rectangle((x, y, x + 9, y + 9), fill=255 if (x + y) % 20 else 0)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    prepared = prepare_image_bytes(buffer.getvalue(), max_image_pixels=20_000)

    assert prepared.rejection_reason is not None
    assert "background is too complex" in prepared.rejection_reason
