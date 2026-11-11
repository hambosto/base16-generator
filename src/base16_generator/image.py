from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .color import Array, Rgb, srgb_to_oklch

MIN_ALPHA = 128


def _read_visible_rgb(path: Path, resize: int) -> Rgb:
    with Image.open(path) as image:
        image.thumbnail((resize, resize), Image.Resampling.NEAREST)
        rgba = np.asarray(image.convert("RGBA")).reshape(-1, 4)

    visible = rgba[rgba[:, 3] >= MIN_ALPHA, :3]
    if len(visible) == 0:
        raise ValueError("image has no visible pixels")
    return visible


def load_oklch(path: Path, resize: int) -> Array:
    if resize < 1:
        raise ValueError("resize must be a positive integer")
    return srgb_to_oklch(_read_visible_rgb(path, resize))
