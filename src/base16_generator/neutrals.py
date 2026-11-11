from __future__ import annotations

from typing import Final

import numpy as np

from .color import Array
from .variants import Variant

RAMP: Final = np.array([0.0, 0.08, 0.19, 0.38, 0.62, 0.82, 0.92, 1.0])
NUM_NEUTRALS: Final = len(RAMP)
HALF: Final = NUM_NEUTRALS // 2

TINT: Final = 1.0
CHROMA_CAP: Final = np.repeat([0.045, 0.035], HALF)


def _tint(pixels: Array) -> tuple[float, float]:
    weight = pixels[:, 1]
    rad = np.radians(pixels[:, 2])
    hue = np.degrees(
        np.arctan2((weight * np.sin(rad)).sum(), (weight * np.cos(rad)).sum())
    )
    return float(hue % 360.0), float(weight.mean())


def _split_by_lightness(lch: Array, sign: float) -> tuple[Array, Array]:
    relative = (lch[:, 0] - np.median(lch[:, 0])) * sign
    foreground = lch[relative > 0.0]
    return lch[relative <= 0.0], foreground if len(foreground) else lch


def _lightness_ramp(variant: Variant, contrast: float) -> Array:
    low = np.interp(contrast, [0.0, 1.0], variant.background)
    high = np.interp(contrast, [0.0, 1.0], variant.foreground)
    return low + (high - low) * RAMP


def neutral_colors(lch: Array, variant: Variant, contrast: float) -> Array:
    background, foreground = (
        _tint(side) for side in _split_by_lightness(lch, variant.sign)
    )
    hue = np.repeat([background[0], foreground[0]], HALF)
    chroma = np.repeat([background[1], foreground[1]], HALF)
    return np.column_stack(
        [_lightness_ramp(variant, contrast), np.minimum(TINT * chroma, CHROMA_CAP), hue]
    )
