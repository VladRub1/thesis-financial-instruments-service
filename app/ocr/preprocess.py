"""Best-effort image preprocessing for scanned documents."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

TARGET_DPI = 300


def to_grayscale(img: Image.Image) -> Image.Image:
    return img.convert("L")


def binarize(img: Image.Image, threshold: int = 0) -> Image.Image:
    """Otsu-style binarization using numpy (no OpenCV/scipy)."""
    arr = np.array(img.convert("L"), dtype=np.uint8)
    if threshold <= 0:
        # simple Otsu via histogram
        hist, _ = np.histogram(arr.ravel(), bins=256, range=(0, 256))
        total = arr.size
        cum_sum = np.cumsum(hist * np.arange(256))
        cum_count = np.cumsum(hist)
        mean_bg = np.divide(cum_sum, cum_count, out=np.zeros_like(cum_sum, dtype=float), where=cum_count > 0)
        mean_fg = np.divide(
            cum_sum[-1] - cum_sum,
            total - cum_count,
            out=np.zeros_like(cum_sum, dtype=float),
            where=(total - cum_count) > 0,
        )
        weight_bg = cum_count / total
        weight_fg = 1.0 - weight_bg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        threshold = int(np.argmax(variance))
    binary = ((arr > threshold) * 255).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def normalize_dpi(img: Image.Image, current_dpi: int | None, target_dpi: int = TARGET_DPI) -> Image.Image:
    if current_dpi and current_dpi != target_dpi:
        scale = target_dpi / current_dpi
        new_size = (int(img.width * scale), int(img.height * scale))
        return img.resize(new_size, Image.LANCZOS)
    return img


def sharpen(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.SHARPEN)


def preprocess(img: Image.Image, current_dpi: int | None = None) -> Image.Image:
    """Full preprocessing pipeline: grayscale → sharpen → DPI normalize."""
    img = to_grayscale(img)
    img = sharpen(img)
    img = normalize_dpi(img, current_dpi)
    return img
