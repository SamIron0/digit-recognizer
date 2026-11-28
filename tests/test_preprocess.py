"""Tests for src.preprocess.preprocess."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import preprocess


def _stroke(size, box, ink, background):
    img = np.full((size, size), background, np.uint8)
    r0, c0, r1, c1 = box
    img[r0:r1, c0:c1] = ink
    return img


def test_output_shape_dtype_and_range():
    for background, ink in ((255, 0), (0, 255)):
        out = preprocess(_stroke(40, (10, 16, 30, 24), ink, background))
        assert out.shape == (1, 28, 28, 1)
        assert out.dtype == np.float32
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0
        img = out[0, :, :, 0]
        assert img[0, 0] == 0.0
        assert img[14, 14] > 0.5

    floated = _stroke(40, (10, 16, 30, 24), 0, 255).astype(np.float32) / 255.0
    assert preprocess(floated).shape == (1, 28, 28, 1)


@pytest.mark.parametrize("value", [0, 255])
def test_blank_image_raises(value):
    with pytest.raises(ValueError, match="No digit"):
        preprocess(np.full((28, 28), value, np.uint8))


def test_tiny_and_full_frame_scale_and_center():
    # Both strokes are taller than they are wide, so the long side is the height.
    for box in ((2, 2, 10, 5), (1, 30, 99, 40)):
        out = preprocess(_stroke(100, box, 0, 255))
        img = out[0, :, :, 0]
        rows = np.flatnonzero((img > 0.05).any(axis=1))
        assert 18 <= rows[-1] - rows[0] + 1 <= 22
        ys, xs = np.ogrid[:28, :28]
        mass = float(img.sum())
        cy = float((ys * img).sum() / mass)
        cx = float((xs * img).sum() / mass)
        assert abs(cy - 14) < 1.5
        assert abs(cx - 14) < 1.5


def test_sample_images():
    samples = Path(__file__).resolve().parent.parent / "samples"
    paths = [p for p in samples.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    assert paths
    for path in paths:
        out = preprocess(np.asarray(Image.open(path).convert("L")))
        assert out.shape == (1, 28, 28, 1)
        assert out.dtype == np.float32
        assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0
