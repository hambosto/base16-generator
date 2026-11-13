from __future__ import annotations

from pathlib import Path

import numpy as np

from .accents import accent_colors
from .color import Array, enforce_contrast, oklch_to_srgb
from .image import load_oklch
from .neutrals import NUM_NEUTRALS, neutral_colors
from .variants import VARIANTS, Variant

HEADROOM = 0.9

MIN_CONTRAST = np.column_stack(
    [
        [1.0, 1.0, 1.0, 3.0, 4.5, 7.0, 7.0, 7.0] + [4.5] * 8,
        [1.0, 1.0, 1.0, 1.5, 4.5, 4.5, 4.5, 4.5] + [3.0] * 8,
    ]
)
BACKGROUND_ROWS = (0, 2)


def build_palette(
    lch: Array, contrast: float = 0.5, variant: Variant = VARIANTS["dark"]
) -> dict[str, str]:
    if not 0.0 <= contrast <= 1.0:
        raise ValueError("contrast must be between 0.0 and 1.0")

    colors = enforce_contrast(
        np.vstack(
            [
                neutral_colors(lch, variant, contrast),
                accent_colors(lch, variant, contrast),
            ]
        ),
        MIN_CONTRAST,
        BACKGROUND_ROWS,
        toward=(1.0 + variant.sign) / 2.0,
        headroom=HEADROOM,
    )
    colors[:NUM_NEUTRALS, 0] = variant.sign * np.maximum.accumulate(
        variant.sign * colors[:NUM_NEUTRALS, 0]
    )
    return {
        f"base{i:02X}": rgb.tobytes().hex()
        for i, rgb in enumerate(oklch_to_srgb(colors))
    }


def generate_palette(
    path: str | Path,
    *,
    contrast: float = 0.5,
    variant: str = "dark",
    resize: int = 256,
) -> dict[str, str]:
    return build_palette(load_oklch(Path(path), resize), contrast, VARIANTS[variant])
