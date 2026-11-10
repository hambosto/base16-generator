from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .color import Array, wrap_degrees

SLOT_HUES: Final[Array] = np.array([29.0, 62.0, 110.0, 142.0, 195.0, 255.0, 310.0])
NUM_CLUSTERS: Final = len(SLOT_HUES)

_NEXT_HUES: Final[Array] = np.roll(SLOT_HUES, -1) + np.where(
    np.arange(NUM_CLUSTERS) == NUM_CLUSTERS - 1, 360.0, 0.0
)

SECTOR_ABOVE: Final[Array] = (_NEXT_HUES - SLOT_HUES) / 2.0
SECTOR_BELOW: Final[Array] = np.roll(SECTOR_ABOVE, 1)
SECTOR_SHIFTS: Final = (-15.0, 0.0, 15.0)

MAX_ITERS: Final = 100
TOLERANCE: Final = 1e-4
CHROMA_THRESHOLD: Final = 0.01
MIN_SCALE: Final = 1e-6
UNIFORM_HUE_STD: Final = 360.0 / math.sqrt(12.0)
UNASSIGNED: Final = -1


@dataclass(frozen=True, slots=True)
class Clustering:
    centers: Array
    active: NDArray[np.bool_]
    counts: NDArray[np.intp]
    mean_chroma: float

    @property
    def populated(self) -> NDArray[np.bool_]:
        return self.active & (self.counts > 0)


def _sq_distance(points: Array, center: Array, scale: Array) -> Array:
    delta = points - center
    delta[..., 2] = wrap_degrees(delta[..., 2])
    return np.sum((delta / scale) ** 2, axis=-1)


def _nearest(
    points: Array, centers: Array, active: NDArray[np.bool_], scale: Array
) -> NDArray[np.intp]:
    best = np.full(len(points), np.inf)
    nearest = np.zeros(len(points), dtype=np.intp)
    for k in np.flatnonzero(active):
        dist = _sq_distance(points, centers[k], scale)
        closer = dist < best
        best[closer] = dist[closer]
        nearest[closer] = k
    return nearest


def _kmeans(
    points: Array, start: Array, scale: Array, shift: float
) -> tuple[Array, NDArray[np.bool_], NDArray[np.intp]]:
    anchors = (SLOT_HUES + shift) % 360.0

    centers = np.empty((NUM_CLUSTERS, 3))
    centers[:, :2] = start
    centers[:, 2] = anchors
    active = np.ones(NUM_CLUSTERS, dtype=bool)
    labels = np.full(len(points), UNASSIGNED, dtype=np.intp)

    rad = np.radians(points[:, 2])
    features = np.column_stack([points[:, 0], points[:, 1], np.sin(rad), np.cos(rad)])

    for _ in range(MAX_ITERS):
        if not active.any():
            break

        free = np.flatnonzero(labels == UNASSIGNED)
        nearest = _nearest(points[free], centers, active, scale)
        counts = np.bincount(nearest, minlength=NUM_CLUSTERS)
        sums = np.column_stack(
            [
                np.bincount(nearest, weights=features[free, d], minlength=NUM_CLUSTERS)
                for d in range(4)
            ]
        )

        settled = True
        for k in np.flatnonzero(active & (counts > 0)):
            lightness = sums[k, 0] / counts[k]
            chroma = sums[k, 1] / counts[k]
            hue = float(np.degrees(np.arctan2(sums[k, 2], sums[k, 3])) % 360.0)
            drift = float(wrap_degrees(hue - anchors[k]))

            if -SECTOR_BELOW[k] <= drift <= SECTOR_ABOVE[k]:
                new = np.array([lightness, chroma, hue])
                if _sq_distance(new, centers[k], scale) > TOLERANCE**2:
                    settled = False
                centers[k] = new
                continue

            edge = anchors[k] + (SECTOR_ABOVE[k] if drift > 0 else -SECTOR_BELOW[k])
            centers[k] = (lightness, chroma, edge % 360.0)
            active[k] = False
            labels[free[nearest == k]] = k
            settled = False

        if settled:
            break

    leftover = np.flatnonzero(labels == UNASSIGNED)
    if leftover.size:
        labels[leftover] = _nearest(points[leftover], centers, active, scale)
    return centers, active, labels


def cluster(lch: Array) -> Clustering:
    vivid = lch[lch[:, 1] >= CHROMA_THRESHOLD]
    if len(vivid):
        points, mean_chroma = vivid, float(vivid[:, 1].mean())
    else:
        points, mean_chroma = lch, 0.0

    rad = np.radians(points[:, 2])
    resultant = np.hypot(np.sin(rad).mean(), np.cos(rad).mean())
    circular_std = math.degrees(
        math.sqrt(-2.0 * math.log(float(np.clip(resultant, 1e-12, 1.0))))
    )
    scale = np.maximum(
        [points[:, 0].std(), points[:, 1].std(), min(circular_std, UNIFORM_HUE_STD)],
        MIN_SCALE,
    )
    start = points[:, :2].mean(axis=0)

    best: Clustering | None = None
    best_score = (-1, -1)
    for shift in SECTOR_SHIFTS:
        centers, active, labels = _kmeans(points, start, scale, shift)
        counts = np.bincount(labels, minlength=NUM_CLUSTERS)
        result = Clustering(centers, active, counts, mean_chroma)
        kept = result.populated
        score = (int(kept.sum()), int(counts[kept].sum()))
        if score > best_score:
            best, best_score = result, score

    assert best is not None
    return best
