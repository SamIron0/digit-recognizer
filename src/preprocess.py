"""Turn a grayscale digit into a MNIST-style (1, 28, 28, 1) tensor."""

import numpy as np
from PIL import Image


def preprocess(image: np.ndarray) -> np.ndarray:
    """Crop, scale, and center a digit. Blank images raise ValueError.

    Accepts a grayscale array of any size (uint8 0-255 or float 0-1).
    Dark ink on a light background is inverted. The result is float32 in
    0-1, white digit on black, longer side 20px, center of mass at (14, 14).
    """
    img = np.asarray(image)
    as_float = img.dtype.kind == "f"
    img = img.astype(np.float32)
    if as_float and img.max() <= 1.0:
        img *= 255.0

    # MNIST is white ink on black. A bright mean means the background is light.
    if img.mean() > 127:
        img = 255.0 - img

    img = np.where(img > 127, img, 0.0)
    rows = np.flatnonzero(img.any(axis=1))
    cols = np.flatnonzero(img.any(axis=0))
    if rows.size == 0:
        raise ValueError("No digit found in image")
    cropped = img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

    height, width = cropped.shape
    if height >= width:
        new_h, new_w = 20, max(1, round(width * 20 / height))
    else:
        new_w, new_h = 20, max(1, round(height * 20 / width))
    resized = np.asarray(
        Image.fromarray(cropped.astype(np.uint8)).resize(
            (new_w, new_h), Image.Resampling.LANCZOS
        ),
        dtype=np.float32,
    )

    # Shift so the center of mass lands on pixel (14, 14). No wrap-around.
    ys, xs = np.ogrid[:new_h, :new_w]
    mass = float(resized.sum())
    if mass == 0:
        raise ValueError("No digit found in image")
    top = round(14 - float((ys * resized).sum() / mass))
    left = round(14 - float((xs * resized).sum() / mass))
    canvas = np.zeros((28, 28), dtype=np.float32)
    y0, x0 = max(top, 0), max(left, 0)
    y1, x1 = min(top + new_h, 28), min(left + new_w, 28)
    canvas[y0:y1, x0:x1] = resized[y0 - top : y1 - top, x0 - left : x1 - left]

    return (canvas / 255.0).reshape(1, 28, 28, 1)
