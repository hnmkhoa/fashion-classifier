import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from backend.app.domain import CLASS_NAMES
from backend.app.main import create_app
from backend.tests.conftest import make_settings


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    image = Image.new("L", (28, 28), color=0)
    ImageDraw.Draw(image).ellipse((5, 8, 22, 20), fill=255)
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def complex_scene_bytes() -> bytes:
    buffer = io.BytesIO()
    image = Image.new("L", (120, 80), color=0)
    draw = ImageDraw.Draw(image)
    for y in range(0, 80, 10):
        for x in range(0, 120, 10):
            draw.rectangle((x, y, x + 9, y + 9), fill=255 if (x + y) % 20 else 0)
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_is_degraded_when_checkpoint_is_missing(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path / "missing.pt"))

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_ready"] is False


def test_classes_follow_canonical_order(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path / "missing.pt"))

    with TestClient(app) as client:
        response = client.get("/api/v1/classes")

    assert response.status_code == 200
    assert [item["name"] for item in response.json()["classes"]] == list(CLASS_NAMES)


@pytest.mark.parametrize(
    "origin",
    ("http://localhost:5173", "http://127.0.0.1:5173", "http://192.168.1.27:5173"),
)
def test_local_frontend_origins_are_allowed_by_cors(tmp_path: Path, origin: str) -> None:
    app = create_app(
        make_settings(
            tmp_path / "missing.pt",
            cors_origins=("http://localhost:5173", "http://127.0.0.1:5173", "http://192.168.1.27:5173"),
        )
    )

    with TestClient(app) as client:
        response = client.options(
            "/api/v1/predict",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_valid_prediction_returns_ranked_output(valid_checkpoint: Path) -> None:
    app = create_app(make_settings(valid_checkpoint))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            files={"file": ("sample.png", png_bytes(), "image/png")},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["class_name"] in CLASS_NAMES
    assert payload["is_unknown"] is False
    assert payload["rejection_reason"] is None
    assert payload["model_version"] == "test-model-v1"
    assert len(payload["top_predictions"]) == 3
    assert payload["confidence"] == payload["top_predictions"][0]["probability"]
    probabilities = [item["probability"] for item in payload["top_predictions"]]
    assert probabilities == sorted(probabilities, reverse=True)


def test_complex_scene_is_returned_as_unknown(valid_checkpoint: Path) -> None:
    app = create_app(make_settings(valid_checkpoint))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            files={"file": ("scene.png", complex_scene_bytes(), "image/png")},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["class_id"] is None
    assert payload["class_name"] == "Unknown"
    assert payload["confidence"] is None
    assert payload["is_unknown"] is True
    assert "background is too complex" in payload["rejection_reason"]
    assert len(payload["top_predictions"]) == 3


def test_invalid_image_has_structured_error(valid_checkpoint: Path) -> None:
    app = create_app(make_settings(valid_checkpoint))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            files={"file": ("sample.png", b"not an image", "image/png")},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_image"
    assert response.json()["error"]["request_id"]


def test_upload_larger_than_limit_is_rejected_before_decoding(valid_checkpoint: Path) -> None:
    app = create_app(make_settings(valid_checkpoint, max_upload_bytes=4))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            files={"file": ("sample.png", b"12345", "image/png")},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_prediction_is_unavailable_without_checkpoint(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path / "missing.pt"))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            files={"file": ("sample.png", png_bytes(), "image/png")},
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"
