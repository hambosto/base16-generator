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

SLOT_HUES = np.array([29.0, 62.0, 110.0, 142.0, 195.0, 255.0, 310.0])
NUM_SLOTS = len(SLOT_HUES)

VIVID_CHROMA = 0.025
MIN_SLOT_SHARE = 0.02
HUE_SMOOTHING = 6.0
HUE_WINDOW = 14.0
HARMONIZE = 12.0
MIN_SEPARATION = 16.0
ACCENT_CHROMA = (0.065, 0.17)
HEADROOM = 0.9
FALLBACK_CHROMA = 0.08
CONTRAST_SHIFT = 0.08
BROWN_STEP = 0.06

NEUTRAL_TINT = 0.8
NEUTRAL_CAP = np.array([0.05] * 4 + [0.03] * 4)
RAMP = np.array([0.0, 0.055, 0.13, 0.36, 0.62, 0.85, 0.93, 1.0])

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
    fallback: tuple[float, ...]


VARIANTS = {
    "dark": Variant(
        sign=1.0,
        background=(0.28, 0.16),
        foreground=(0.92, 0.985),
        accent_bounds=(0.60, 0.88),
        fallback=(0.66, 0.74, 0.85, 0.78, 0.78, 0.70, 0.70),
    ),
    "light": Variant(
        sign=-1.0,
        background=(0.94, 0.99),
        foreground=(0.30, 0.14),
        accent_bounds=(0.42, 0.60),
        fallback=(0.54, 0.56, 0.54, 0.52, 0.52, 0.48, 0.48),
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


def build_palette(lch: Array, contrast: float, variant: Variant) -> dict[str, str]:
    if not 0.0 <= contrast <= 1.0:
        raise ValueError("contrast must be between 0.0 and 1.0")
    sign = variant.sign

    vivid = lch[lch[:, 1] >= VIVID_CHROMA]
    hues = SLOT_HUES.copy()
    shares = np.zeros(NUM_SLOTS)
    found = np.zeros(NUM_SLOTS, dtype=bool)
    slot_lightness = np.array(variant.fallback)
    slot_chroma = np.full(NUM_SLOTS, FALLBACK_CHROMA)

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
            slot_lightness[k] = vivid[members, 0].mean()
            slot_chroma[k] = vivid[members, 1].mean()
            found[k] = True

    if found.any():
        dominant = hues[shares.argmax()]
        for k in np.flatnonzero(~found):
            turn = np.clip(
                0.5 * wrap_degrees(dominant - SLOT_HUES[k]), -HARMONIZE, HARMONIZE
            )
            hues[k] = SLOT_HUES[k] + turn
            slot_lightness[k] = np.median(slot_lightness[found])
            slot_chroma[k] = np.median(slot_chroma[found])

    for _ in range(3):
        for k in range(NUM_SLOTS):
            j = (k + 1) % NUM_SLOTS
            gap = (hues[j] - hues[k]) % 360.0
            if gap < MIN_SEPARATION:
                if shares[k] < shares[j]:
                    hues[k] -= MIN_SEPARATION - gap
                else:
                    hues[j] += MIN_SEPARATION - gap

    hues %= 360.0
    shift = sign * CONTRAST_SHIFT * (contrast - 0.5)
    accents = np.empty((8, 3))
    accents[:7, 0] = np.clip(slot_lightness + shift, *variant.accent_bounds)
    accents[:7, 1] = np.clip(slot_chroma, *ACCENT_CHROMA)
    accents[:7, 2] = hues
    accents[7] = (
        np.clip(accents[1, 0] - sign * BROWN_STEP, *variant.accent_bounds),
        np.clip(0.5 * accents[1, 1], 0.04, 0.09),
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
    resize: int = 256,
) -> dict[str, str]:
    return build_palette(load_oklch(Path(path), resize), contrast, VARIANTS[variant])
