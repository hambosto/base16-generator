from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from .accents import accent_colors
from .color import Array, enforce_contrast, oklch_to_srgb
from .image import load_oklch
from .neutrals import NUM_NEUTRALS, neutral_colors
from .variants import VARIANTS, Variant

DEFAULT_RICHNESS: Final = 0.8
HEADROOM: Final = 0.9

MIN_CONTRAST: Final = np.column_stack(
    [
        [1.0, 1.0, 1.0, 3.0, 4.5, 7.0, 7.0, 7.0] + [4.5] * 8,
        [1.0, 1.0, 1.0, 1.5, 4.5, 4.5, 4.5, 4.5] + [3.0] * 8,
    ]
)

REFERENCE_ROWS: Final = (0, 2)


def _check_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")


def _keep_ramp_monotonic(colors: Array, sign: float) -> Array:
    colors[:NUM_NEUTRALS, 0] = sign * np.maximum.accumulate(
        sign * colors[:NUM_NEUTRALS, 0]
    )
    return colors


def _to_hex(colors: Array) -> dict[str, str]:
    return {
        f"base{i:02X}": rgb.tobytes().hex()
        for i, rgb in enumerate(oklch_to_srgb(colors))
    }


def build_palette(
    lch: Array,
    contrast: float = 0.5,
    variant: Variant = VARIANTS["dark"],
    richness: float = DEFAULT_RICHNESS,
) -> dict[str, str]:
    _check_unit_interval("contrast", contrast)
    _check_unit_interval("richness", richness)

    colors = np.vstack(
        [
            neutral_colors(lch, variant, contrast),
            accent_colors(lch, variant, contrast, richness),
        ]
    )
    colors = enforce_contrast(
        colors,
        MIN_CONTRAST,
        REFERENCE_ROWS,
        toward=(1.0 + variant.sign) / 2.0,
        headroom=HEADROOM,
    )
    return _to_hex(_keep_ramp_monotonic(colors, variant.sign))


def generate_palette(
    path: str | Path,
    *,
    contrast: float = 0.5,
    variant: str = "dark",
    richness: float = DEFAULT_RICHNESS,
    resize: int = 256,
) -> dict[str, str]:
    return build_palette(
        load_oklch(Path(path), resize), contrast, VARIANTS[variant], richness
    )
