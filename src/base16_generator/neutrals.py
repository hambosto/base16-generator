from __future__ import annotations

import numpy as np

from .color import Array
from .variants import Variant

RAMP = np.array([0.0, 0.08, 0.19, 0.38, 0.62, 0.82, 0.92, 1.0])
NUM_NEUTRALS = len(RAMP)
HALF = NUM_NEUTRALS // 2

TINT = 0.7
CHROMA_CAP = np.repeat([0.04, 0.025], HALF)
CONTRAST_SPREAD = 0.1
MIN_SPAN = 0.62


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


def _end_of_ramp(pixels: Array, bounds: tuple[float, float], shift: float) -> float:
    start, stop = bounds
    tone = np.median(pixels[:, 0])
    return float(np.clip(start + (stop - start) * tone + shift, start, stop))


def neutral_colors(lch: Array, variant: Variant, contrast: float) -> Array:
    shadows, highlights = _split_by_lightness(lch, variant.sign)
    nudge = variant.sign * CONTRAST_SPREAD * (contrast - 0.5)
    low = _end_of_ramp(shadows, variant.background_bounds, -nudge)
    high = _end_of_ramp(highlights, variant.foreground_bounds, nudge)
    high = low + variant.sign * max(variant.sign * (high - low), MIN_SPAN)

    (shadow_hue, shadow_chroma), (highlight_hue, highlight_chroma) = (
        _tint(shadows),
        _tint(highlights),
    )
    return np.column_stack(
        [
            low + (high - low) * RAMP,
            np.minimum(
                TINT * np.repeat([shadow_chroma, highlight_chroma], HALF), CHROMA_CAP
            ),
            np.repeat([shadow_hue, highlight_hue], HALF),
        ]
    )
