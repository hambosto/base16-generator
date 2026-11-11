from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

import numpy as np

from .color import Array, Mask, wrap_degrees
from .variants import Variant

SLOT_HUES: Final = np.array([23.0, 54.0, 85.0, 137.0, 195.0, 247.0, 318.0])
NUM_SLOTS: Final = len(SLOT_HUES)
RED, ORANGE = 0, 1

_GAP_ABOVE = np.diff(np.append(SLOT_HUES, SLOT_HUES[0] + 360.0))
_GAP_BELOW = np.roll(_GAP_ABOVE, 1)

VIVID_CHROMA: Final = 0.025
MIN_SLOT_SHARE: Final = 0.02
PRESENT_SHARE: Final = 0.005
HUE_BINS: Final = 72
HUE_SMOOTHING: Final = 6.0
HUE_WINDOW: Final = 14.0
TONE_PERCENTILE: Final = 80.0
REACH: Final = 0.75
MIN_SEPARATION: Final = 16.0
ACCENT_CHROMA: Final = (0.065, 0.19)
CONTRAST_SHIFT: Final = 0.08
BROWN_CHROMA: Final = 0.8

_EDGES = np.arange(-41.0, 42.0)
_CENTERS = _EDGES[:-1] + 0.5
_KERNEL = np.exp(-0.5 * (np.arange(-18, 19) / HUE_SMOOTHING) ** 2)


@dataclass(frozen=True, slots=True)
class Slots:
    hue: Array
    lightness: Array
    chroma: Array
    share: Array
    found: Mask


def _targets(variant: Variant) -> tuple[Array, Array]:
    return np.array(variant.rich_lightness), np.array(variant.rich_chroma)


def _dominant_members(offsets: Array, weights: Array, voters: Mask) -> Mask:
    histogram, _ = np.histogram(offsets[voters], bins=_EDGES, weights=weights[voters])
    peak = _CENTERS[np.convolve(histogram, _KERNEL, mode="same").argmax()]
    members: Mask = voters & (np.abs(offsets - peak) <= HUE_WINDOW)
    return members


def _find_slots(vivid: Array, variant: Variant) -> Slots:
    rich_lightness, rich_chroma = _targets(variant)
    slots = Slots(
        hue=SLOT_HUES.copy(),
        lightness=rich_lightness.copy(),
        chroma=rich_chroma.copy(),
        share=np.zeros(NUM_SLOTS),
        found=np.zeros(NUM_SLOTS, dtype=bool),
    )
    if len(vivid) == 0:
        return slots

    weights = vivid[:, 1]
    offset = wrap_degrees(vivid[:, 2, None] - SLOT_HUES)
    nearest = np.abs(offset).argmin(axis=1)
    for k in range(NUM_SLOTS):
        voters = nearest == k
        slots.share[k] = weights[voters].sum() / weights.sum()
        if slots.share[k] < MIN_SLOT_SHARE:
            continue
        members = _dominant_members(offset[:, k], weights, voters)
        slots.hue[k] = SLOT_HUES[k] + np.average(
            offset[members, k], weights=weights[members]
        )
        slots.lightness[k] = np.percentile(vivid[members, 0], TONE_PERCENTILE)
        slots.chroma[k] = np.percentile(weights[members], TONE_PERCENTILE)
        slots.found[k] = True
    return slots


def _present_hues(vivid: Array) -> Array:
    histogram, _ = np.histogram(
        vivid[:, 2], bins=HUE_BINS, range=(0.0, 360.0), weights=vivid[:, 1]
    )
    bins = np.flatnonzero(histogram >= PRESENT_SHARE * histogram.sum())
    return ((bins + 0.5) * (360.0 / HUE_BINS)).astype(np.float64)


def _fill_missing(slots: Slots, vivid: Array, variant: Variant) -> Slots:
    rich_lightness, rich_chroma = _targets(variant)
    lightness_shift = np.median((slots.lightness - rich_lightness)[slots.found])
    chroma_shift = np.median((slots.chroma - rich_chroma)[slots.found])
    present = _present_hues(vivid)

    hue = slots.hue.copy()
    lightness = slots.lightness.copy()
    chroma = slots.chroma.copy()
    for k in np.flatnonzero(~slots.found):
        gap = wrap_degrees(present - SLOT_HUES[k])
        toward = gap[np.abs(gap).argmin()]
        hue[k] = SLOT_HUES[k] + np.clip(
            toward, -REACH * _GAP_BELOW[k] / 2.0, REACH * _GAP_ABOVE[k] / 2.0
        )
        lightness[k] = rich_lightness[k] + lightness_shift
        chroma[k] = rich_chroma[k] + chroma_shift
    return replace(slots, hue=hue, lightness=lightness, chroma=chroma)


def _separate(slots: Slots) -> Slots:
    hue = slots.hue.copy()
    for _ in range(NUM_SLOTS):
        for k in range(NUM_SLOTS):
            j = (k + 1) % NUM_SLOTS
            gap = hue[j] - hue[k] + (360.0 if j == 0 else 0.0)
            if gap >= MIN_SEPARATION:
                continue
            if slots.share[k] < slots.share[j]:
                hue[k] -= MIN_SEPARATION - gap
            else:
                hue[j] += MIN_SEPARATION - gap
    return replace(slots, hue=hue % 360.0)


def _lift(value: Array, target: Array, richness: float, direction: float) -> Array:
    gap = value - target
    return target + np.where(direction * gap > 0, gap, (1.0 - richness) * gap)


def _brown(accents: Array, variant: Variant) -> Array:
    red, orange = accents[RED], accents[ORANGE]
    return np.array(
        [
            np.clip(orange[0] - variant.brown_step, *variant.accent_bounds),
            BROWN_CHROMA * orange[1],
            red[2] + wrap_degrees(orange[2] - red[2]) / 2.0,
        ]
    )


def _assemble(
    slots: Slots, variant: Variant, contrast: float, richness: float
) -> Array:
    rich_lightness, rich_chroma = _targets(variant)
    lightness = _lift(slots.lightness, rich_lightness, richness, variant.sign)
    chroma = _lift(slots.chroma, rich_chroma, richness, 1.0)
    shift = variant.sign * CONTRAST_SHIFT * (contrast - 0.5)

    accents = np.empty((NUM_SLOTS + 1, 3))
    accents[:NUM_SLOTS, 0] = np.clip(lightness + shift, *variant.accent_bounds)
    accents[:NUM_SLOTS, 1] = np.clip(chroma, *ACCENT_CHROMA)
    accents[:NUM_SLOTS, 2] = slots.hue
    accents[NUM_SLOTS] = _brown(accents, variant)

    return accents


def accent_colors(
    lch: Array, variant: Variant, contrast: float, richness: float
) -> Array:
    vivid = lch[lch[:, 1] >= VIVID_CHROMA]
    slots = _find_slots(vivid, variant)
    if slots.found.any():
        slots = _fill_missing(slots, vivid, variant)
    return _assemble(_separate(slots), variant, contrast, richness)
