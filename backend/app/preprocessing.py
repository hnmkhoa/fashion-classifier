"""Safe image decoding and deterministic inference preprocessing."""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from .domain import INPUT_HEIGHT, INPUT_WIDTH, NORMALIZATION_MEAN, NORMALIZATION_STD

_FASHION_MNIST_BORDER_MAX = 32
_FOREGROUND_CANVAS_SIZE = 24
_MAX_BACKGROUND_VARIATION = 28.0
_MIN_FOREGROUND_FRACTION = 0.01
_MAX_FOREGROUND_FRACTION = 0.70


class ImageDecodingError(ValueError):
    """Raised when uploaded bytes cannot be decoded under the image contract."""


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """A model-ready tensor and an optional reason to reject it as out of domain."""

    tensor: torch.Tensor
    rejection_reason: str | None


def decode_image(image_bytes: bytes, max_image_pixels: int) -> Image.Image:
    """Decode image bytes, enforce a pixel limit, and normalize EXIF orientation."""
    if not image_bytes:
        raise ImageDecodingError("The uploaded image is empty.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as source:
                width, height = source.size
                if width <= 0 or height <= 0 or width * height > max_image_pixels:
                    raise ImageDecodingError(
                        f"Decoded image exceeds the {max_image_pixels:,}-pixel limit."
                    )
                source.load()
                oriented = ImageOps.exif_transpose(source)
                if oriented is None:
                    oriented = source
                return oriented.convert("L")
    except ImageDecodingError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageDecodingError("The image dimensions are too large to process safely.") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageDecodingError("The uploaded file could not be decoded as an image.") from exc


def _border_pixels(pixels: np.ndarray) -> np.ndarray:
    """Return the outer image border used to estimate a product-photo background."""
    return np.concatenate(
        (pixels[0, :], pixels[-1, :], pixels[1:-1, 0], pixels[1:-1, -1])
    )


def _foreground_canvas(grayscale: Image.Image) -> tuple[Image.Image, str | None]:
    """Map either light or dark uniform backgrounds to the Fashion-MNIST visual domain."""
    pixels = np.asarray(grayscale, dtype=np.float32)
    border = _border_pixels(pixels)
    background = float(np.median(border))
    background_variation = float(np.percentile(np.abs(border - background), 90.0))
    contrast = np.abs(pixels - background)
    peak = float(np.percentile(contrast, 99.5))
    if peak < 8.0:
        return (
            Image.new("L", (INPUT_WIDTH, INPUT_HEIGHT), color=0),
            "No distinct garment foreground was found.",
        )

    threshold = max(float(_FASHION_MNIST_BORDER_MAX), peak * 0.10)
    foreground_mask = contrast >= threshold
    if not np.any(foreground_mask):
        return (
            Image.new("L", (INPUT_WIDTH, INPUT_HEIGHT), color=0),
            "No distinct garment foreground was found.",
        )

    foreground_fraction = float(np.mean(foreground_mask))
    rejection_reason = None
    if background_variation > _MAX_BACKGROUND_VARIATION:
        rejection_reason = (
            "The image background is too complex for this Fashion-MNIST model."
        )
    elif foreground_fraction < _MIN_FOREGROUND_FRACTION:
        rejection_reason = "The garment occupies too little of the image."
    elif foreground_fraction > _MAX_FOREGROUND_FRACTION:
        rejection_reason = "The scene contains too much foreground detail."

    # Keep native Fashion-MNIST samples pixel-identical to the evaluation pipeline.
    if grayscale.size == (INPUT_WIDTH, INPUT_HEIGHT) and background <= 16:
        return grayscale.copy(), rejection_reason

    y_coordinates, x_coordinates = np.nonzero(foreground_mask)
    left = int(x_coordinates.min())
    right = int(x_coordinates.max()) + 1
    top = int(y_coordinates.min())
    bottom = int(y_coordinates.max()) + 1
    margin = max(1, int(round(max(right - left, bottom - top) * 0.08)))
    left = max(0, left - margin)
    right = min(pixels.shape[1], right + margin)
    top = max(0, top - margin)
    bottom = min(pixels.shape[0], bottom + margin)

    scaled = np.clip(contrast * (255.0 / peak), 0.0, 255.0).astype(np.uint8)
    foreground = Image.fromarray(scaled[top:bottom, left:right])
    contained = ImageOps.contain(
        foreground,
        (_FOREGROUND_CANVAS_SIZE, _FOREGROUND_CANVAS_SIZE),
        method=Image.Resampling.LANCZOS,
    )
    canvas = Image.new("L", (INPUT_WIDTH, INPUT_HEIGHT), color=0)
    offset = ((INPUT_WIDTH - contained.width) // 2, (INPUT_HEIGHT - contained.height) // 2)
    canvas.paste(contained, offset)
    return canvas, rejection_reason


def prepare_image(image: Image.Image) -> PreparedImage:
    """Build a tensor and assess whether an upload resembles the training domain."""
    grayscale = image.convert("L")
    canvas, rejection_reason = _foreground_canvas(grayscale)

    pixels = np.asarray(canvas, dtype=np.float32) / 255.0
    normalized = (pixels - NORMALIZATION_MEAN) / NORMALIZATION_STD
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    return PreparedImage(tensor=tensor, rejection_reason=rejection_reason)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Adapt an upload to the Fashion-MNIST domain and add a batch dimension."""
    return prepare_image(image).tensor


def prepare_image_bytes(image_bytes: bytes, max_image_pixels: int) -> PreparedImage:
    """Decode uploaded bytes and return both model input and domain assessment."""
    return prepare_image(decode_image(image_bytes, max_image_pixels))


def preprocess_image_bytes(image_bytes: bytes, max_image_pixels: int) -> torch.Tensor:
    """Convert uploaded bytes into one model-ready tensor."""
    return prepare_image_bytes(image_bytes, max_image_pixels).tensor
