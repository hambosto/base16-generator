from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .color import Array, Mask, wrap_degrees
from .variants import Variant

SLOT_HUES = np.array([23.0, 54.0, 85.0, 137.0, 195.0, 247.0, 318.0])
NUM_SLOTS = len(SLOT_HUES)
RED, ORANGE = 0, 1

DRIFT_BELOW = np.array([26.0, 27.0, 34.0, 34.0, 59.0, 69.0, 38.0])
DRIFT_ABOVE = np.array([18.0, 36.0, 32.0, 22.0, 53.0, 49.0, 59.0])

VIVID_CHROMA = 0.025
MIN_SLOT_SHARE = 0.02
PEAK_RADIUS = 14.0
PEAK_SMOOTHING = 6.0
TONE_PERCENTILE = 90.0
WARP = 0.7
DRIFT_SCALE = 1.2
MIN_SEPARATION = 12.0
MAX_SEPARATION = 40.0
MIN_CHROMA = 0.065
MIN_NEIGHBOUR_DE = 0.075
CHROMA_GAIN = 1.6
CONTRAST_SHIFT = 0.08
BROWN_CHROMA = 0.9

_EDGES = np.arange(-41.0, 42.0)
_CENTERS = _EDGES[:-1] + 0.5
_KERNEL = np.exp(-0.5 * (np.arange(-18, 19) / PEAK_SMOOTHING) ** 2)


@dataclass(frozen=True)
class Slots:
    hue: Array
    lightness: Array
    chroma: Array
    share: Array
    found: Mask


def _peak_members(offsets: Array, weights: Array, voters: Mask) -> Mask:
    histogram, _ = np.histogram(offsets[voters], bins=_EDGES, weights=weights[voters])
    peak = _CENTERS[np.convolve(histogram, _KERNEL, mode="same").argmax()]
    members: Mask = voters & (np.abs(offsets - peak) <= PEAK_RADIUS)
    return members


def _find_slots(vivid: Array) -> Slots:
    hue = SLOT_HUES.copy()
    lightness = np.full(NUM_SLOTS, np.nan)
    chroma = np.full(NUM_SLOTS, np.nan)
    share = np.zeros(NUM_SLOTS)
    if len(vivid) == 0:
        return Slots(hue, lightness, chroma, share, share > 0)

    weights = vivid[:, 1]
    offset = wrap_degrees(vivid[:, 2, None] - SLOT_HUES)
    nearest = np.abs(offset).argmin(axis=1)
    for k in range(NUM_SLOTS):
        voters = nearest == k
        share[k] = weights[voters].sum() / weights.sum()
        if share[k] < MIN_SLOT_SHARE:
            continue
        members = _peak_members(offset[:, k], weights, voters)
        hue[k] += np.average(offset[members, k], weights=weights[members])
        lightness[k] = np.percentile(vivid[members, 0], TONE_PERCENTILE)
        chroma[k] = CHROMA_GAIN * np.percentile(weights[members], TONE_PERCENTILE)
    return Slots(hue, lightness, chroma, share, share >= MIN_SLOT_SHARE)


def _warped_hues(vivid: Array) -> Array:
    weights = vivid[:, 1]
    rad = np.radians(vivid[:, 2])
    x = (weights * np.cos(rad)).sum() / weights.sum()
    y = (weights * np.sin(rad)).sum() / weights.sum()
    dominant = np.degrees(np.arctan2(y, x))
    compression = 1.0 - WARP * np.hypot(x, y)
    warped: Array = dominant + compression * wrap_degrees(SLOT_HUES - dominant)
    return warped


def _fill_hues(slots: Slots, vivid: Array) -> Array:
    drift = np.clip(
        wrap_degrees(_warped_hues(vivid) - SLOT_HUES),
        -DRIFT_SCALE * DRIFT_BELOW,
        DRIFT_SCALE * DRIFT_ABOVE,
    )
    return np.where(slots.found, slots.hue, SLOT_HUES + drift)


def _isotonic(values: Array, weights: Array) -> Array:
    blocks: list[tuple[float, float, int]] = []
    for value, weight in zip(values, weights, strict=True):
        blocks.append((value, weight, 1))
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            (mean_b, weight_b, size_b), (mean_a, weight_a, size_a) = (
                blocks.pop(),
                blocks.pop(),
            )
            total = weight_a + weight_b
            blocks.append(
                (
                    (mean_a * weight_a + mean_b * weight_b) / total,
                    total,
                    size_a + size_b,
                )
            )
    return np.concatenate([np.full(size, mean) for mean, _, size in blocks])


def _needed_gaps(lightness: Array, chroma: Array) -> Array:
    next_chroma = np.roll(chroma, -1)
    tone = (np.roll(lightness, -1) - lightness) ** 2 + (next_chroma - chroma) ** 2
    half_chord = np.sqrt(np.maximum(MIN_NEIGHBOUR_DE**2 - tone, 0.0))
    ratio = np.minimum(half_chord / (chroma + next_chroma), 1.0)
    return np.clip(2.0 * np.degrees(np.arcsin(ratio)), MIN_SEPARATION, MAX_SEPARATION)


def _separate(hue: Array, slots: Slots, gaps: Array) -> Array:
    cut = int(np.diff(np.append(hue, hue[0] + 360.0)).argmax() + 1) % NUM_SLOTS
    line = np.concatenate([hue[cut:], hue[:cut] + 360.0])
    weights = np.where(slots.found, 100.0 * (1.0 + slots.share), 1.0)
    steps = np.concatenate([[0.0], np.cumsum(np.roll(gaps, -cut)[: NUM_SLOTS - 1])])
    fitted = _isotonic(line - steps, np.roll(weights, -cut)) + steps
    separated: Array = np.roll(fitted, cut) % 360.0
    return separated


def _tone(
    own: Array, found: Mask, bounds: tuple[float, float], shape: tuple[float, ...]
) -> Array:
    level = np.clip(own[found].mean(), *bounds) if found.any() else sum(bounds) / 2.0
    target = level + np.array(shape)
    half = (bounds[1] - bounds[0]) / 2.0
    return np.clip(np.where(found, own, target), target - half, target + half)


def accent_colors(lch: Array, variant: Variant, contrast: float) -> Array:
    vivid = lch[lch[:, 1] >= VIVID_CHROMA]
    slots = _find_slots(vivid)
    if slots.found.any():
        slots = replace(slots, hue=_fill_hues(slots, vivid))
    shift = variant.sign * CONTRAST_SHIFT * (contrast - 0.5)
    lightness = np.clip(
        _tone(
            slots.lightness,
            slots.found,
            variant.lightness_range,
            variant.lightness_shape,
        )
        + shift,
        *variant.accent_bounds,
    )
    chroma = _tone(
        slots.chroma, slots.found, variant.chroma_range, variant.chroma_shape
    )
    hue = _separate(slots.hue, slots, _needed_gaps(lightness, chroma))

    accents = np.empty((NUM_SLOTS + 1, 3))
    accents[:NUM_SLOTS, 0] = lightness
    accents[:NUM_SLOTS, 1] = chroma
    accents[:NUM_SLOTS, 2] = hue
    accents[NUM_SLOTS] = (
        np.clip(accents[ORANGE, 0] - variant.brown_step, *variant.accent_bounds),
        BROWN_CHROMA * accents[ORANGE, 1],
        hue[RED] + wrap_degrees(hue[ORANGE] - hue[RED]) / 2.0,
    )
    return accents
