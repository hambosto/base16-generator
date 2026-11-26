from __future__ import annotations

import numpy as np

from .color import Array, circular_mean
from .variants import Variant

NUM_NEUTRALS = 8
HALF = NUM_NEUTRALS // 2

TINT = 0.7
CHROMA_CAP = np.repeat([0.04, 0.025], HALF)
CONTRAST_SPREAD = 0.1
MIN_SPAN = 0.62
MAX_BASE_LIGHTNESS = 0.99


def _tint(pixels: Array) -> tuple[float, float]:
    return circular_mean(pixels[:, 2], pixels[:, 1]), float(pixels[:, 1].mean())


def _split_by_lightness(lch: Array, sign: float) -> tuple[Array, Array]:
    relative = (lch[:, 0] - np.median(lch[:, 0])) * sign
    foreground = lch[relative > 0.0]
    return lch[relative <= 0.0], foreground if len(foreground) else lch


def _end_of_ramp(pixels: Array, bounds: tuple[float, float], shift: float) -> float:
    start, stop = bounds
    tone = np.median(pixels[:, 0])
    return float(np.clip(start + (stop - start) * tone + shift, start, stop))


def neutral_colors(
    lch: Array,
    variant: Variant,
    contrast: float,
    base_lift: float = 0.0,
    ramp_lift: float = 0.0,
) -> Array:
    shadows, highlights = _split_by_lightness(lch, variant.sign)
    nudge = variant.sign * CONTRAST_SPREAD * (contrast - 0.5)
    low = _end_of_ramp(shadows, variant.background_bounds, -nudge)
    if variant.sign < 0 and base_lift:
        low = min(MAX_BASE_LIGHTNESS, low + base_lift)
    high = _end_of_ramp(highlights, variant.foreground_bounds, nudge)
    high = low + variant.sign * max(variant.sign * (high - low), MIN_SPAN)
    ramp = np.array(variant.neutral_ramp)
    if variant.sign < 0 and ramp_lift:
        ramp[:3] = ramp[:3] * (1.0 - ramp_lift)

    (shadow_hue, shadow_chroma), (highlight_hue, highlight_chroma) = (
        _tint(shadows),
        _tint(highlights),
    )
    return np.column_stack(
        [
            low + (high - low) * ramp,
            np.minimum(
                TINT * np.repeat([shadow_chroma, highlight_chroma], HALF), CHROMA_CAP
            ),
            np.repeat([shadow_hue, highlight_hue], HALF),
        ]
    )
