"""Environment-backed application settings with dependency-free parsing."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _positive_int_from_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _path_from_env(name: str, default: Path) -> Path:
    value = Path(os.getenv(name, str(default))).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings resolved when an application instance is created."""

    model_path: Path
    max_upload_bytes: int
    max_image_pixels: int
    cors_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> Settings:
        origins = tuple(
            origin.strip()
            for origin in os.getenv(
                "CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173,"
                "http://localhost:3000,http://127.0.0.1:3000,"
                "http://192.168.1.27:5173",
            ).split(",")
            if origin.strip()
        )
        return cls(
            model_path=_path_from_env("MODEL_PATH", PROJECT_ROOT / "artifacts/fashion_cnn.pt"),
            max_upload_bytes=_positive_int_from_env("MAX_UPLOAD_BYTES", 5 * 1024 * 1024),
            max_image_pixels=_positive_int_from_env("MAX_IMAGE_PIXELS", 40_000_000),
            cors_origins=origins,
        )
