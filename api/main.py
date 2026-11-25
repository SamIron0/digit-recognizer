"""
Work package 8 — Backend API
Owner: Okpara Chidozie (Seancss-23)

FastAPI app that serves the digit recognizer.

    GET  /health   -> {"status": "ok", "model_loaded": bool, "preprocess": "real" | "stub"}
    POST /predict  -> {"digit": 7, "confidence": 0.97, "probabilities": [10 floats]}

Errors always come back as {"error": "message"} with a 4xx status, never a 500
for bad input.

Run locally:
    uvicorn api.main:app --reload
Then:
    curl http://localhost:8000/health

Stand-ins (from the team playbook):
- If models/digit_cnn.keras doesn't exist yet (#5 / #4 not merged), a dummy
  model returns fixed probabilities that always predict 7.
- If src.preprocess.preprocess isn't implemented yet (#7 not merged), a stub
  returns np.zeros((1, 28, 28, 1)).
Once the real pieces land on main, this file picks them up automatically —
no code changes needed here.
"""

import base64
import binascii
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

logger = logging.getLogger("digit-api")

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "digit_cnn.keras"

# Reject anything bigger than ~5 MB of base64. A 280x280 PNG is a few KB.
MAX_IMAGE_CHARS = 5_000_000


# ---------------------------------------------------------------------------
# Stand-ins used until the real model / preprocessing exist
# ---------------------------------------------------------------------------

class DummyModel:
    """Mimics model.predict() so the API works before #5 ships a real model."""

    def predict(self, x, verbose=0):
        probs = np.full((1, 10), 0.003, dtype=np.float32)
        probs[0, 7] = 0.973  # sums to 1.0, always predicts 7
        return probs


def stub_preprocess(image: np.ndarray) -> np.ndarray:
    """Placeholder from the playbook until #7's preprocess() is merged."""
    return np.zeros((1, 28, 28, 1), dtype=np.float32)


def load_preprocess() -> tuple[Callable[[np.ndarray], np.ndarray], str]:
    """Use #7's preprocess() if it exists, otherwise the stub."""
    try:
        from src.preprocess import preprocess  # type: ignore

        return preprocess, "real"
    except ImportError:
        logger.warning("src.preprocess.preprocess not found, using stub")
        return stub_preprocess, "stub"


def load_model():
    """Load the Keras model once. Falls back to DummyModel if it's missing."""
    if not MODEL_PATH.exists():
        logger.warning("%s not found, using DummyModel", MODEL_PATH)
        return DummyModel(), False
    try:
        # Imported here so tests and the dummy mode don't need TensorFlow.
        from tensorflow import keras

        model = keras.models.load_model(MODEL_PATH)
        logger.info("Loaded model from %s", MODEL_PATH)
        return model, True
    except Exception:  # corrupt file, TF missing, etc.
        logger.exception("Could not load %s, using DummyModel", MODEL_PATH)
        return DummyModel(), False


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup, not per request.
    app.state.model, app.state.model_loaded = load_model()
    app.state.preprocess, app.state.preprocess_kind = load_preprocess()
    yield


app = FastAPI(title="Handwritten Digit Recognizer API", lifespan=lifespan)

# Without CORS the browser blocks #10's fetch() calls from the frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    image: str  # base64 PNG, with or without the "data:image/png;base64," prefix


class BadInput(Exception):
    """Raised for anything the client sent wrong. Becomes a 400."""


@app.exception_handler(BadInput)
async def bad_input_handler(request: Request, exc: BadInput):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    # Missing "image" field, wrong type, invalid JSON, etc.
    return JSONResponse(
        status_code=422,
        content={"error": 'Request body must be JSON like {"image": "<base64 PNG>"}'},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_image(data: str) -> np.ndarray:
    """base64 string -> grayscale numpy array (H, W), uint8."""
    data = data.strip()
    if not data:
        raise BadInput("Image is empty")
    if len(data) > MAX_IMAGE_CHARS:
        raise BadInput("Image is too large")

    # The canvas's toDataURL() gives "data:image/png;base64,...." — strip it here
    # so the frontend can send it either way.
    if data.startswith("data:"):
        _, _, data = data.partition(",")

    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise BadInput("Image is not valid base64")

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise BadInput("Data is not a valid image")

    # Transparent pixels would turn black on convert("L"), so paste onto white
    # first (an unfilled canvas exports as transparent).
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)

    return np.array(img.convert("L"))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": bool(request.app.state.model_loaded),
        "preprocess": request.app.state.preprocess_kind,
    }


@app.post("/predict")
def predict(body: PredictRequest, request: Request):
    gray = decode_image(body.image)

    try:
        x = request.app.state.preprocess(gray)
    except ValueError as e:
        # #7 raises ValueError for a blank canvas
        raise BadInput(str(e) or "No digit found in image")

    x = np.asarray(x, dtype=np.float32).reshape(1, 28, 28, 1)
    probs = np.asarray(request.app.state.model.predict(x, verbose=0)).reshape(-1)

    # Cast numpy types to plain Python so FastAPI can serialize them.
    digit = int(np.argmax(probs))
    return {
        "digit": digit,
        "confidence": round(float(probs[digit]), 4),
        "probabilities": [round(float(p), 4) for p in probs],
    }
