from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

from .color import (
    Array,
    enforce_contrast,
    oklch_to_srgb,
    srgb_to_oklch,
    wrap_degrees,
)

SLOT_HUES = np.array([23.0, 54.0, 85.0, 137.0, 195.0, 247.0, 318.0])
NUM_SLOTS = len(SLOT_HUES)
_GAP_ABOVE = np.diff(np.append(SLOT_HUES, SLOT_HUES[0] + 360.0))
_GAP_BELOW = np.roll(_GAP_ABOVE, 1)
REACH = 0.75

VIVID_CHROMA = 0.025
MIN_SLOT_SHARE = 0.02
HUE_SMOOTHING = 6.0
HUE_WINDOW = 14.0
TONE_PERCENTILE = 80.0
PRESENT_SHARE = 0.005
MIN_SEPARATION = 16.0
DEFAULT_RICHNESS = 0.8
ACCENT_CHROMA = (0.065, 0.19)
HEADROOM = 0.9
CONTRAST_SHIFT = 0.08
BROWN_CHROMA = 0.8

NEUTRAL_TINT = 1.0
NEUTRAL_CAP = np.array([0.045] * 4 + [0.035] * 4)
RAMP = np.array([0.0, 0.08, 0.19, 0.38, 0.62, 0.82, 0.92, 1.0])

MIN_CONTRAST = np.column_stack(
    [
        [1.0, 1.0, 1.0, 3.0, 4.5, 7.0, 7.0, 7.0] + [4.5] * 8,
        [1.0, 1.0, 1.0, 1.5, 4.5, 4.5, 4.5, 4.5] + [3.0] * 8,
    ]
)

_EDGES = np.arange(-41.0, 42.0)
_CENTERS = _EDGES[:-1] + 0.5
_KERNEL = np.exp(-0.5 * (np.arange(-18, 19) / HUE_SMOOTHING) ** 2)


class Variant(NamedTuple):
    sign: float
    background: tuple[float, float]
    foreground: tuple[float, float]
    accent_bounds: tuple[float, float]
    rich_lightness: tuple[float, ...]
    rich_chroma: tuple[float, ...]
    brown_step: float


VARIANTS = {
    "dark": Variant(
        sign=1.0,
        background=(0.27, 0.17),
        foreground=(0.93, 0.985),
        accent_bounds=(0.60, 0.93),
        rich_lightness=(0.66, 0.74, 0.81, 0.75, 0.75, 0.70, 0.70),
        rich_chroma=(0.160, 0.131, 0.127, 0.128, 0.097, 0.110, 0.128),
        brown_step=0.12,
    ),
    "light": Variant(
        sign=-1.0,
        background=(0.95, 0.99),
        foreground=(0.30, 0.16),
        accent_bounds=(0.42, 0.60),
        rich_lightness=(0.57, 0.62, 0.65, 0.60, 0.62, 0.56, 0.57),
        rich_chroma=(0.161, 0.138, 0.127, 0.119, 0.096, 0.123, 0.145),
        brown_step=0.04,
    ),
}


def load_oklch(path: Path, resize: int) -> Array:
    if resize < 1:
        raise ValueError("resize must be a positive integer")
    with Image.open(path) as image:
        image.thumbnail((resize, resize), Image.Resampling.NEAREST)
        rgba = np.asarray(image.convert("RGBA")).reshape(-1, 4)

    visible = rgba[rgba[:, 3] >= 128, :3]
    if len(visible) == 0:
        raise ValueError("image has no visible pixels")
    return srgb_to_oklch(visible)


def _tint(pixels: Array) -> tuple[float, float]:
    rad = np.radians(pixels[:, 2])
    weight = pixels[:, 1]
    hue = np.degrees(
        np.arctan2((weight * np.sin(rad)).sum(), (weight * np.cos(rad)).sum())
    )
    return float(hue % 360.0), float(weight.mean())


def build_palette(
    lch: Array,
    contrast: float = 0.5,
    variant: Variant = VARIANTS["dark"],
    richness: float = DEFAULT_RICHNESS,
) -> dict[str, str]:
    if not 0.0 <= contrast <= 1.0:
        raise ValueError("contrast must be between 0.0 and 1.0")
    if not 0.0 <= richness <= 1.0:
        raise ValueError("richness must be between 0.0 and 1.0")
    sign = variant.sign

    vivid = lch[lch[:, 1] >= VIVID_CHROMA]
    hues = SLOT_HUES.copy()
    shares = np.zeros(NUM_SLOTS)
    found = np.zeros(NUM_SLOTS, dtype=bool)
    rich_lightness = np.array(variant.rich_lightness)
    rich_chroma = np.array(variant.rich_chroma)
    slot_lightness = rich_lightness.copy()
    slot_chroma = rich_chroma.copy()

    if len(vivid):
        offset = wrap_degrees(vivid[:, 2, None] - SLOT_HUES)
        nearest = np.abs(offset).argmin(axis=1)
        for k in range(NUM_SLOTS):
            voters = nearest == k
            shares[k] = vivid[voters, 1].sum() / vivid[:, 1].sum()
            if shares[k] < MIN_SLOT_SHARE:
                continue
            histogram, _ = np.histogram(
                offset[voters, k], bins=_EDGES, weights=vivid[voters, 1]
            )
            peak = _CENTERS[np.convolve(histogram, _KERNEL, mode="same").argmax()]
            members = voters & (np.abs(offset[:, k] - peak) <= HUE_WINDOW)
            hues[k] = SLOT_HUES[k] + np.average(
                offset[members, k], weights=vivid[members, 1]
            )
            slot_lightness[k] = np.percentile(vivid[members, 0], TONE_PERCENTILE)
            slot_chroma[k] = np.percentile(vivid[members, 1], TONE_PERCENTILE)
            found[k] = True

    if found.any():
        shift_lightness = np.median((slot_lightness - rich_lightness)[found])
        shift_chroma = np.median((slot_chroma - rich_chroma)[found])
        histogram, _ = np.histogram(
            vivid[:, 2], bins=72, range=(0.0, 360.0), weights=vivid[:, 1]
        )
        present = (
            np.flatnonzero(histogram >= PRESENT_SHARE * histogram.sum()) + 0.5
        ) * 5.0
        for k in np.flatnonzero(~found):
            gap = wrap_degrees(present - SLOT_HUES[k])
            toward = gap[np.abs(gap).argmin()]
            hues[k] = SLOT_HUES[k] + np.clip(
                toward, -REACH * _GAP_BELOW[k] / 2.0, REACH * _GAP_ABOVE[k] / 2.0
            )
            slot_lightness[k] = rich_lightness[k] + shift_lightness
            slot_chroma[k] = rich_chroma[k] + shift_chroma

    for _ in range(NUM_SLOTS):
        for k in range(NUM_SLOTS):
            j = (k + 1) % NUM_SLOTS
            gap = hues[j] - hues[k] + (360.0 if j == 0 else 0.0)
            if gap < MIN_SEPARATION:
                if shares[k] < shares[j]:
                    hues[k] -= MIN_SEPARATION - gap
                else:
                    hues[j] += MIN_SEPARATION - gap
    hues %= 360.0

    shift = sign * CONTRAST_SHIFT * (contrast - 0.5)
    lightness_gap = slot_lightness - rich_lightness
    chroma_gap = slot_chroma - rich_chroma
    lightness = rich_lightness + np.where(
        sign * lightness_gap > 0, lightness_gap, (1.0 - richness) * lightness_gap
    )
    chroma = rich_chroma + np.where(
        chroma_gap > 0, chroma_gap, (1.0 - richness) * chroma_gap
    )

    accents = np.empty((8, 3))
    accents[:7, 0] = np.clip(lightness + shift, *variant.accent_bounds)
    accents[:7, 1] = np.clip(chroma, *ACCENT_CHROMA)
    accents[:7, 2] = hues
    accents[7] = (
        np.clip(accents[1, 0] - variant.brown_step, *variant.accent_bounds),
        BROWN_CHROMA * accents[1, 1],
        hues[0] + wrap_degrees(hues[1] - hues[0]) / 2.0,
    )

    median = np.median(lch[:, 0])
    background_side = lch[(lch[:, 0] - median) * sign <= 0.0]
    foreground_side = lch[(lch[:, 0] - median) * sign > 0.0]
    if len(foreground_side) == 0:
        foreground_side = lch
    background_hue, background_chroma = _tint(background_side)
    foreground_hue, foreground_chroma = _tint(foreground_side)

    low = np.interp(contrast, [0.0, 1.0], variant.background)
    high = np.interp(contrast, [0.0, 1.0], variant.foreground)
    neutrals = np.column_stack(
        [
            low + (high - low) * RAMP,
            np.minimum(
                NEUTRAL_TINT
                * np.array([background_chroma] * 4 + [foreground_chroma] * 4),
                NEUTRAL_CAP,
            ),
            np.array([background_hue] * 4 + [foreground_hue] * 4),
        ]
    )

    colors = enforce_contrast(
        np.vstack([neutrals, accents]),
        MIN_CONTRAST,
        reference_rows=(0, 2),
        toward=(1.0 + sign) / 2.0,
        headroom=HEADROOM,
    )
    colors[:8, 0] = sign * np.maximum.accumulate(sign * colors[:8, 0])

    rgb = oklch_to_srgb(colors)
    return {f"base{i:02X}": rgb[i].tobytes().hex() for i in range(16)}


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
