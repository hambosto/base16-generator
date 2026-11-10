from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .color import Array, srgb_to_oklch


def load_oklch(path: Path, resize: int | None = None) -> Array:
    with Image.open(path) as image:
        if resize is not None:
            image.thumbnail((resize, resize), Image.Resampling.NEAREST)
        rgba = image.convert("RGBA")

    pixels = np.asarray(rgba).reshape(-1, 4)
    visible = pixels[pixels[:, 3] > 0, :3]
    if len(visible) == 0:
        raise ValueError("image has no visible pixels")
    return srgb_to_oklch(visible)
