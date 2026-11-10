from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from .clustering import NUM_CLUSTERS, Clustering, cluster
from .color import (
    Array,
    enforce_contrast,
    linear_luminance,
    oklch_to_srgb,
)
from .image import load_oklch
from .variants import DARK, VARIANTS, Variant

NUM_ACCENTS: Final = NUM_CLUSTERS + 1
ZONE_MERGE_THRESHOLD: Final = 50.0
BACKGROUND_SLOTS: Final = 4
NEUTRAL_CHROMA_RATIO: Final = 0.5
MAX_BACKGROUND_CHROMA: Final = 0.04
MAX_FOREGROUND_CHROMA: Final = 0.025
MIN_ACCENT_CHROMA: Final = 0.07
MAX_ACCENT_CHROMA: Final = 0.16
CHROMA_RATIO_RANGE: Final = (
    0.6,
    1.4,
)
MIN_BROWN_CHROMA: Final = 0.04
MAX_BROWN_CHROMA: Final = 0.09
BROWN_CHROMA_RATIO: Final = 0.5
LIGHTNESS_FOLLOW: Final = 0.5
MAX_LIGHTNESS_OFFSET: Final = 0.06
CONTRAST_LIGHTNESS_SHIFT: Final = 0.08

MIN_CONTRAST: Final[Array] = np.column_stack(
    [
        [1.0, 1.0, 1.0, 3.0, 4.5, 7.0, 7.0, 7.0] + [4.5] * NUM_ACCENTS,
        [1.0, 1.0, 1.0, 1.5, 4.5, 4.5, 4.5, 4.5] + [3.0] * NUM_ACCENTS,
    ]
)

LABELS: Final = tuple(f"base{i:02X}" for i in range(16))


def _blend(centers: Array, weights: Array) -> Array:
    weights = weights / weights.sum()
    rad = np.radians(centers[:, 2])
    hue = (
        np.degrees(
            np.arctan2((np.sin(rad) * weights).sum(), (np.cos(rad) * weights).sum())
        )
        % 360.0
    )
    return np.array(
        [(centers[:, 0] * weights).sum(), (centers[:, 1] * weights).sum(), hue]
    )


def _accents(result: Clustering, contrast: float, variant: Variant) -> Array:
    centers = result.centers
    baseline = np.array(variant.accent_lightness)
    low, high = variant.accent_range
    shift = variant.direction * CONTRAST_LIGHTNESS_SHIFT * (contrast - 0.5)

    mean = centers[:, 1].mean()
    ratio = (
        np.clip(centers[:, 1] / mean, *CHROMA_RATIO_RANGE)
        if mean > 0
        else np.ones(NUM_CLUSTERS)
    )
    target = np.clip(
        result.mean_chroma * (1.0 + 3.0 * contrast),
        MIN_ACCENT_CHROMA,
        MAX_ACCENT_CHROMA,
    )

    nudge = np.clip(
        LIGHTNESS_FOLLOW * (centers[:, 0] - centers[:, 0].mean()),
        -MAX_LIGHTNESS_OFFSET,
        MAX_LIGHTNESS_OFFSET,
    )

    accents = np.empty((NUM_ACCENTS, 3))
    accents[:NUM_CLUSTERS, 0] = np.clip(
        baseline[:NUM_CLUSTERS] + nudge + shift, low, high
    )
    accents[:NUM_CLUSTERS, 1] = np.minimum(target * ratio, MAX_ACCENT_CHROMA)
    accents[:NUM_CLUSTERS, 2] = centers[:, 2]

    brown_hue = _blend(accents[:2], np.ones(2))[2]
    brown_chroma = np.clip(
        BROWN_CHROMA_RATIO * accents[1, 1], MIN_BROWN_CHROMA, MAX_BROWN_CHROMA
    )
    accents[NUM_CLUSTERS] = (
        np.clip(baseline[NUM_CLUSTERS] + shift, low, high),
        brown_chroma,
        brown_hue,
    )
    return accents


def _neutral_tints(result: Clustering, direction: float) -> tuple[Array, Array]:
    centers, counts = result.centers, result.counts

    populated = np.flatnonzero(result.populated)
    if populated.size == 0:
        grey = np.array([0.5, 0.0, 0.0])
        return grey, grey

    by_hue = [
        int(k) for k in populated[np.argsort(centers[populated, 2], kind="stable")]
    ]
    zones = [[by_hue[0]]]
    for k in by_hue[1:]:
        if centers[k, 2] - centers[zones[-1][-1], 2] <= ZONE_MERGE_THRESHOLD:
            zones[-1].append(k)
        else:
            zones.append([k])

    seam_gap = 360.0 - centers[zones[-1][-1], 2] + centers[zones[0][0], 2]
    if len(zones) > 1 and seam_gap <= ZONE_MERGE_THRESHOLD:
        zones[0] = zones.pop() + zones[0]

    if len(zones) == 1:
        zone = zones[0]
        background = _blend(centers[zone], counts[zone].astype(np.float64))
        return background, centers[max(zone, key=lambda k: direction * centers[k, 0])]

    zones.sort(key=lambda zone: counts[zone].sum(), reverse=True)
    first, second = (
        _blend(centers[zone], counts[zone].astype(np.float64)) for zone in zones[:2]
    )
    return first, first if direction * (first[0] - second[0]) > 0 else second


def build_palette(
    result: Clustering, contrast: float = 0.5, variant: Variant = DARK
) -> dict[str, str]:
    contrast = float(np.clip(contrast, 0.0, 1.0))
    accents = _accents(result, contrast, variant)
    background, foreground = _neutral_tints(result, variant.direction)

    neutrals = np.empty((len(variant.neutral_lightness), 3))
    for i, lightness in enumerate(variant.neutral_lightness):
        if i < BACKGROUND_SLOTS:
            tint, cap = background, MAX_BACKGROUND_CHROMA
        else:
            tint, cap = foreground, MAX_FOREGROUND_CHROMA
        neutrals[i] = (lightness, min(tint[1] * NEUTRAL_CHROMA_RATIO, cap), tint[2])

    colors = np.vstack([neutrals, accents])
    anchors = oklch_to_srgb(colors[[0, 2]])
    anchor_luminance = linear_luminance(anchors)
    colors = enforce_contrast(
        colors, MIN_CONTRAST, anchor_luminance, toward=(1.0 + variant.direction) / 2.0
    )

    palette: dict[str, str] = {}
    for label, (r, g, b) in zip(LABELS, oklch_to_srgb(colors)):
        palette[label] = f"{int(r):02x}{int(g):02x}{int(b):02x}"
    return palette


def generate_palette(
    image_path: str | Path,
    *,
    contrast: float = 0.5,
    resize: int | None = 256,
    variant: str = "dark",
) -> dict[str, str]:
    return build_palette(
        cluster(load_oklch(Path(image_path), resize)), contrast, VARIANTS[variant]
    )
