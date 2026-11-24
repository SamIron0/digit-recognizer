"""
Work package 8 — Backend API tests
Owner: Okpara Chidozie (Seancss-23)

Run from the repo root:
    python -m pytest tests/test_api.py -v

These pass with the dummy model/stub preprocess AND with the real ones,
because they only check the response shape, not a specific digit.
"""

import base64
import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # Using the context manager runs the startup (lifespan) handler.
    with TestClient(app) as c:
        yield c


def make_png_b64(draw_digit: bool = True, size: int = 280) -> str:
    """A 280x280 canvas-style PNG: black ink on white, like #9 exports."""
    img = Image.new("L", (size, size), 255)
    if draw_digit:
        d = ImageDraw.Draw(img)
        d.line([(90, 60), (190, 60), (130, 220)], fill=0, width=18)  # rough "7"
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def assert_prediction_shape(data: dict):
    assert set(data) == {"digit", "confidence", "probabilities"}
    assert isinstance(data["digit"], int) and 0 <= data["digit"] <= 9
    assert isinstance(data["confidence"], float) and 0.0 <= data["confidence"] <= 1.0
    assert len(data["probabilities"]) == 10
    assert all(isinstance(p, float) for p in data["probabilities"])
    assert abs(sum(data["probabilities"]) - 1.0) < 0.01
    assert data["confidence"] == max(data["probabilities"])


# ---- /health ---------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---- /predict happy path ---------------------------------------------------

def test_predict_valid_png(client):
    r = client.post("/predict", json={"image": make_png_b64()})
    assert r.status_code == 200
    assert_prediction_shape(r.json())


def test_predict_accepts_data_url_prefix(client):
    # canvas.toDataURL() includes this prefix; the API should handle it.
    r = client.post("/predict", json={"image": "data:image/png;base64," + make_png_b64()})
    assert r.status_code == 200
    assert_prediction_shape(r.json())


def test_predict_transparent_png(client):
    img = Image.new("RGBA", (280, 280), (0, 0, 0, 0))
    ImageDraw.Draw(img).line([(140, 40), (140, 240)], fill=(0, 0, 0, 255), width=18)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    r = client.post("/predict", json={"image": base64.b64encode(buf.getvalue()).decode()})
    assert r.status_code == 200
    assert_prediction_shape(r.json())


def test_cors_header_present(client):
    r = client.get("/health", headers={"Origin": "http://localhost:5500"})
    assert r.headers.get("access-control-allow-origin") == "*"


# ---- /predict bad input -> 4xx with {"error": ...}, never 500 ---------------

@pytest.mark.parametrize(
    "payload",
    [
        {"image": "this is not base64!!!"},  # malformed base64
        {"image": base64.b64encode(b"hello, not an image").decode()},  # not an image
        {"image": ""},  # empty
    ],
)
def test_predict_bad_image_returns_400(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 400
    assert "error" in r.json()


@pytest.mark.parametrize("payload", [{}, {"img": "abc"}, {"image": 123}])
def test_predict_bad_body_returns_422(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert "error" in r.json()


def test_blank_canvas_returns_400_when_preprocess_rejects_it(client, monkeypatch):
    # #7's preprocess raises ValueError on a blank canvas; simulate that.
    def raises(_):
        raise ValueError("No digit found in image")

    monkeypatch.setattr(app.state, "preprocess", raises)
    r = client.post("/predict", json={"image": make_png_b64(draw_digit=False)})
    assert r.status_code == 400
    assert r.json() == {"error": "No digit found in image"}
