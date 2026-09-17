"""Shared domain constants used by both training and inference."""

from typing import Final

CLASS_NAMES: Final[tuple[str, ...]] = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

INPUT_CHANNELS: Final[int] = 1
INPUT_HEIGHT: Final[int] = 28
INPUT_WIDTH: Final[int] = 28
INPUT_SIZE: Final[tuple[int, int, int]] = (INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)

# These statistics are the commonly used population values for Fashion-MNIST.
# They are part of the checkpoint contract and must not drift between training and serving.
NORMALIZATION_MEAN: Final[float] = 0.2860405969887955
NORMALIZATION_STD: Final[float] = 0.35302424451492237

MODEL_ARCHITECTURE: Final[str] = "fashion_cnn_v1"
CHECKPOINT_SCHEMA_VERSION: Final[int] = 1

SUPPORTED_MEDIA_TYPES: Final[frozenset[str]] = frozenset(
    {"image/jpeg", "image/png", "image/webp"}
)

