from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .color import Array, srgb_to_oklch

MIN_ALPHA = 128


def load_oklch(path: Path, resize: int) -> Array:
    if resize < 1:
        raise ValueError("resize must be a positive integer")
    with Image.open(path) as image:
        image.thumbnail((resize, resize), Image.Resampling.NEAREST)
        rgba = np.asarray(image.convert("RGBA")).reshape(-1, 4)

    visible = rgba[rgba[:, 3] >= MIN_ALPHA, :3]
    if len(visible) == 0:
        raise ValueError("image has no visible pixels")
    return srgb_to_oklch(visible)
