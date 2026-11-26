from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np

from .color import (
    Array,
    Mask,
    adjacent_min,
    circular_mean,
    max_chroma,
    oklch_to_srgb,
    srgb_to_oklch,
    wrap_degrees,
)
from .variants import Variant
from .vision import DEFICIENCIES, simulated_lab

SLOT_HUES = np.array([23.0, 54.0, 85.0, 137.0, 195.0, 247.0, 318.0])
NUM_SLOTS = len(SLOT_HUES)
RED, ORANGE, GREEN = 0, 1, 3
CYAN = 4
BLUE = 5

DRIFT_BELOW = np.array([78.0, 54.0, 68.0, 34.0, 59.0, 138.0, 38.0])
DRIFT_ABOVE = np.array([54.0, 72.0, 64.0, 22.0, 53.0, 98.0, 59.0])

VIVID_CHROMA = 0.025
POOL_CHROMA = 0.015
POOL_MIN_FRACTION = 0.002
WARP_COHERENCE = 0.9
VIVID_MIN_SAMPLE = 32
MIN_SLOT_SHARE = 0.02
BLUE_MIN_SHARE = 0.10
SNAP_RED_SKIP_LOW = 65.0
SNAP_RED_SKIP_HIGH = 110.0
RELIEF_MAX_DRAG = 25.0
NUDGE_FRACTION = 0.5
NUDGE_MAX_MOVE = 8.0
NUDGE_MIN_MOVE = 0.5
CYAN_SPREAD = 0.2
MIN_HUE_GAP = 11.0
MIN_ROUND_TRIP_GAP = 12.0
MIN_VALID_DE = 0.04
SNAP_TARGETS = {RED: 42.0, GREEN: 172.0}
SNAP_BANDS = {RED: (45.0, 200.0), GREEN: (175.0, 200.0)}
GREEN_PULL_START = 165.0
GREEN_PULL_LIMIT = 45.0
GREEN_PULL_FACTOR = 0.7
SURVIVING_CHROMA_FACTOR = 0.9
SHARE_THRESHOLD = np.array([MIN_SLOT_SHARE] * 5 + [BLUE_MIN_SHARE, MIN_SLOT_SHARE])
LIGHT_SHARE_THRESHOLD = np.array([0.02, 0.02, 0.10, 0.02, 0.10, 0.20, 0.02])
PEAK_RADIUS = 14.0
PEAK_SMOOTHING = 6.0
TONE_PERCENTILE = 90.0
WARP = 0.7
CAST = 0.3
DRIFT_SCALE = 1.3
MIN_SEPARATION = 14.0
MAX_SEPARATION = 40.0
MIN_CHROMA = 0.065
MIN_NEIGHBOUR_DE = 0.085
CHROMA_GAIN = 1.6
LIGHT_CHROMA_GAIN = 1.15
DARK_CHROMA_GAIN = 1.12
VISION_MIN_DE = 0.03
RED_GREEN_MIN_DE = 0.10
REPEL_ROUNDS = 3
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


def _slot_fit(
    offsets: Array,
    weights: Array,
    voters: Mask,
    levels: Array,
    chroma_values: Array,
) -> tuple[float, float, float]:
    members = _peak_members(offsets, weights, voters)
    return (
        float(np.average(offsets[members], weights=weights[members])),
        float(np.percentile(levels[members], TONE_PERCENTILE)),
        CHROMA_GAIN * float(np.percentile(chroma_values[members], TONE_PERCENTILE)),
    )


def _find_slots(
    vivid: Array, pool: Array, thresholds: Array = SHARE_THRESHOLD
) -> Slots:
    hue = SLOT_HUES.copy()
    lightness = np.full(NUM_SLOTS, np.nan)
    chroma = np.full(NUM_SLOTS, np.nan)
    share = np.zeros(NUM_SLOTS)
    if len(pool) == 0:
        return Slots(hue, lightness, chroma, share, share > 0)

    pool_weights = np.maximum(pool[:, 1] - POOL_CHROMA, 0.0)
    pool_total = float(pool_weights.sum())
    pool_offset = wrap_degrees(pool[:, 2, None] - SLOT_HUES)
    pool_nearest = np.abs(pool_offset).argmin(axis=1)
    weights = vivid[:, 1] if len(vivid) else np.array([])
    vivid_total = float(weights.sum()) if len(vivid) else 0.0
    offset = wrap_degrees(vivid[:, 2, None] - SLOT_HUES) if len(vivid) else np.array([])
    nearest = np.abs(offset).argmin(axis=1) if len(vivid) else np.array([], dtype=int)
    for k in range(NUM_SLOTS):
        pool_voters = pool_nearest == k
        share[k] = (
            pool_weights[pool_voters].sum() / pool_total if pool_total > 0 else 0.0
        )
        if share[k] < thresholds[k]:
            continue
        voters = nearest == k
        vivid_share = weights[voters].sum() / vivid_total if len(vivid) else 0.0
        if vivid_share >= MIN_SLOT_SHARE:
            fitted = _slot_fit(offset[:, k], weights, voters, vivid[:, 0], weights)
        else:
            fitted = _slot_fit(
                pool_offset[:, k], pool_weights, pool_voters, pool[:, 0], pool[:, 1]
            )
        hue[k] += fitted[0]
        lightness[k] = fitted[1]
        chroma[k] = fitted[2]
    return Slots(hue, lightness, chroma, share, share >= thresholds)


def _resultant(pixels: Array) -> tuple[float, float]:
    weights = pixels[:, 1]
    rad = np.radians(pixels[:, 2])
    total = weights.sum()
    return (
        float((weights * np.cos(rad)).sum() / total),
        float((weights * np.sin(rad)).sum() / total),
    )


def _warped_hues(vivid: Array) -> Array:
    x, y = _resultant(vivid)
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
    warped: Array = SLOT_HUES + drift
    if not slots.found.any():
        return warped
    matched = slots.hue[slots.found]
    lean = np.zeros(NUM_SLOTS)
    for k in range(NUM_SLOTS):
        if slots.found[k]:
            continue
        offsets = wrap_degrees(matched - warped[k])
        lean[k] = CAST * offsets[np.abs(offsets).argmin()]
    filled: Array = np.where(slots.found, slots.hue, warped + lean)
    return filled


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


def _hungarian(cost: Array) -> list[int]:
    n = len(cost)
    size = max(len(cost), len(cost[0]) if n else 0)
    padded = np.full((size, size), 1e6)
    padded[: len(cost), : len(cost[0])] = cost
    u = np.zeros(size + 1)
    v = np.zeros(size + 1)
    p = np.zeros(size + 1, dtype=int)
    way = np.zeros(size + 1, dtype=int)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                cur = padded[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    assignment = [0] * len(cost)
    for j in range(1, size + 1):
        if p[j] and p[j] <= len(cost):
            assignment[p[j] - 1] = j - 1
    return assignment


def _cluster_centres(lch: Array) -> list[float]:
    vivid = lch[lch[:, 1] >= VIVID_CHROMA]
    if len(vivid) == 0:
        return []
    histogram, _ = np.histogram(vivid[:, 2], bins=np.arange(361.0), weights=vivid[:, 1])
    smoothed = np.convolve(
        np.concatenate([histogram[-18:], histogram, histogram[:18]]),
        _KERNEL,
        mode="same",
    )[18:-18]
    order = np.argsort(smoothed)[::-1]
    centres: list[float] = []
    for peak in order:
        if smoothed[peak] <= 0:
            break
        if any(abs(wrap_degrees(float(peak) - c)) < PEAK_RADIUS for c in centres):
            continue
        members = vivid[np.abs(wrap_degrees(vivid[:, 2] - peak)) <= PEAK_RADIUS]
        if len(members) == 0:
            continue
        centre = circular_mean(members[:, 2], members[:, 1])
        if any(abs(wrap_degrees(centre - c)) < PEAK_RADIUS for c in centres):
            continue
        centres.append(centre)
        if len(centres) >= 12:
            break
    return centres


def _assignment_cost(
    unfound: list[int],
    filled: Array,
    centres: list[float],
    found_hues: list[float],
    cyan_spread: float = 0.0,
) -> Array:
    cost = np.full((len(unfound), len(centres)), 1e6)
    for i, k in enumerate(unfound):
        for j, centre in enumerate(centres):
            if any(abs(wrap_degrees(centre - h)) < MIN_HUE_GAP for h in found_hues):
                continue
            band = SNAP_BANDS.get(k)
            if band and band[0] <= centre <= band[1]:
                continue
            distance = abs(float(wrap_degrees(filled[k] - centre)))
            if k == CYAN and cyan_spread:
                distance -= cyan_spread * abs(
                    float(wrap_degrees(centre - SLOT_HUES[CYAN]))
                )
            cost[i, j] = distance
    return cost


def _restore_until_ok(
    assigned: Array,
    unfound: list[int],
    hue: Array,
    cost: Array,
    order: list[int],
    lightness: Array,
    chroma: Array,
) -> Array:
    for i in sorted(
        range(len(unfound)),
        key=lambda i: cost[i, order[i]] if order[i] >= 0 else -1.0,
        reverse=True,
    ):
        assigned[unfound[i]] = hue[unfound[i]]
        if _hue_move_ok(assigned, lightness, chroma):
            return assigned
    return hue


def _assign_hues(
    hue: Array,
    slots: Slots,
    filled: Array,
    lch: Array,
    lightness: Array,
    chroma: Array,
    centres: list[float] | None = None,
    cyan_spread: float = 0.0,
) -> Array:
    unfound = [k for k in range(NUM_SLOTS) if not slots.found[k]]
    if not unfound:
        return hue
    if centres is None:
        centres = _cluster_centres(lch)
    if len(centres) < len(unfound):
        return hue
    found_hues = [float(hue[k]) for k in range(NUM_SLOTS) if slots.found[k]]
    cost = _assignment_cost(unfound, filled, centres, found_hues, cyan_spread)
    order = _hungarian(cost)
    assigned = hue.copy()
    for i, k in enumerate(unfound):
        if order[i] >= 0 and cost[i, order[i]] < 1e6:
            assigned[k] = centres[order[i]]
    if _hue_move_ok(assigned, lightness, chroma):
        return assigned
    return _restore_until_ok(assigned, unfound, hue, cost, order, lightness, chroma)


def _separate(hue: Array, slots: Slots, gaps: Array) -> Array:
    cut = int(np.diff(np.append(hue, hue[0] + 360.0)).argmax() + 1) % NUM_SLOTS
    line = np.concatenate([hue[cut:], hue[:cut] + 360.0])
    weights = np.where(slots.found, 100.0 * (1.0 + slots.share), 1.0)
    steps = np.concatenate([[0.0], np.cumsum(np.roll(gaps, -cut)[: NUM_SLOTS - 1])])
    fitted = _isotonic(line - steps, np.roll(weights, -cut)) + steps
    separated: Array = np.roll(fitted, cut) % 360.0
    return separated


def _tone(
    own: Array, slots: Slots, bounds: tuple[float, float], shape: tuple[float, ...]
) -> Array:
    if slots.found.any():
        level = float(
            np.clip(
                np.average(own[slots.found], weights=slots.share[slots.found]),
                *bounds,
            )
        )
    else:
        level = sum(bounds) / 2.0
    target = level + np.array(shape)
    half = (bounds[1] - bounds[0]) / 2.0
    return np.clip(np.where(slots.found, own, target), target - half, target + half)


def _repel_lightness(
    lightness: Array, chroma: Array, hue: Array, bounds: tuple[float, float]
) -> Array:
    fitted = lightness.copy()
    for _ in range(REPEL_ROUNDS):
        lch = np.column_stack([fitted, chroma, hue])
        shortfalls: list[tuple[int, int, float]] = []
        for matrix in DEFICIENCIES.values():
            lab = simulated_lab(lch, matrix)
            for k, j in combinations(range(NUM_SLOTS), 2):
                red_green = {k, j} == {RED, GREEN}
                target = RED_GREEN_MIN_DE if red_green else VISION_MIN_DE
                distance = float(np.sqrt(((lab[k] - lab[j]) ** 2).sum()))
                if distance < target:
                    shortfalls.append((k, j, target - distance))
        if not shortfalls:
            break
        for k, j, shortfall in shortfalls:
            high, low = (k, j) if fitted[k] >= fitted[j] else (j, k)
            fitted[high] = min(fitted[high] + shortfall, bounds[1])
            fitted[low] = max(fitted[low] - shortfall, bounds[0])
    return np.clip(fitted, *bounds)


def _dominant_resultant(pixels: Array) -> float:
    weights = pixels[:, 1]
    if len(pixels) == 0 or weights.sum() == 0:
        return 0.0
    x, y = _resultant(pixels)
    return float(np.hypot(x, y))


def find_slots(lch: Array, thresholds: Array | None = None) -> Slots:
    vivid = lch[lch[:, 1] >= VIVID_CHROMA]
    pool = lch[lch[:, 1] >= POOL_CHROMA]
    slots = _find_slots(
        vivid, pool, SHARE_THRESHOLD if thresholds is None else thresholds
    )
    if len(pool) < POOL_MIN_FRACTION * len(lch):
        slots = replace(slots, found=np.zeros(NUM_SLOTS, dtype=bool))
    return slots


def _warp_source(lch: Array, slots: Slots) -> Array:
    vivid = lch[lch[:, 1] >= VIVID_CHROMA]
    pool = lch[lch[:, 1] >= POOL_CHROMA]
    if len(vivid) >= VIVID_MIN_SAMPLE and _dominant_resultant(vivid) >= WARP_COHERENCE:
        return vivid
    if len(pool):
        return pool
    return lch


def _tone_vivid_flag(lch: Array, slots: Slots) -> bool:
    vivid = lch[lch[:, 1] >= VIVID_CHROMA]
    if len(vivid) < VIVID_MIN_SAMPLE:
        return False
    weights = vivid[:, 1]
    offset = wrap_degrees(vivid[:, 2, None] - SLOT_HUES)
    nearest = np.abs(offset).argmin(axis=1)
    return any(
        slots.found[k]
        and (nearest == k).sum() >= VIVID_MIN_SAMPLE
        and weights[nearest == k].sum() / weights.sum() >= MIN_SLOT_SHARE
        for k in range(NUM_SLOTS)
    )


def canonical_inputs(
    lch: Array, thresholds: Array | None = None
) -> tuple[Slots, Array | None, bool, list[float]]:
    slots = find_slots(lch, thresholds)
    source = _warp_source(lch, slots) if slots.found.any() else None
    return slots, source, _tone_vivid_flag(lch, slots), _cluster_centres(lch)


def _hue_move_ok(hue: Array, lightness: Array, chroma: Array) -> bool:
    ordered = np.sort(hue)
    gaps = np.diff(np.append(ordered, ordered[0] + 360.0))
    if gaps.min() < MIN_HUE_GAP:
        return False
    round_tripped = srgb_to_oklch(
        oklch_to_srgb(np.column_stack([lightness, chroma, hue]))
    )[:, 2]
    ordered = np.sort(round_tripped)
    gaps = np.diff(np.append(ordered, ordered[0] + 360.0))
    if gaps.min() < MIN_ROUND_TRIP_GAP:
        return False
    rad = np.radians(hue)
    lab = np.column_stack([lightness, chroma * np.cos(rad), chroma * np.sin(rad)])
    return adjacent_min(lab) >= MIN_VALID_DE


def _try_move(
    hue: Array, slot: int, value: float, lightness: Array, chroma: Array
) -> Array | None:
    trial = hue.copy()
    trial[slot] = value
    return trial if _hue_move_ok(trial, lightness, chroma) else None


def _relieve_drag(
    hue: Array, slots: Slots, filled: Array, lightness: Array, chroma: Array
) -> Array:
    if not slots.found.any():
        return hue
    relieved = hue.copy()
    order = sorted(
        (k for k in range(NUM_SLOTS) if not slots.found[k] and k not in (RED, GREEN)),
        key=lambda k: abs(wrap_degrees(relieved[k] - filled[k])),
        reverse=True,
    )
    for k in order:
        drag = wrap_degrees(relieved[k] - filled[k])
        if abs(drag) <= RELIEF_MAX_DRAG:
            continue
        moved = _try_move(
            relieved,
            k,
            (filled[k] + np.sign(drag) * RELIEF_MAX_DRAG) % 360.0,
            lightness,
            chroma,
        )
        if moved is not None:
            relieved = moved
    return relieved


def _coverage_nudge(hue: Array, lch: Array, lightness: Array, chroma: Array) -> Array:
    vivid = lch[lch[:, 1] >= VIVID_CHROMA]
    if len(vivid) == 0:
        return hue
    nearest = np.abs(wrap_degrees(vivid[:, 2, None] - hue[None, :])).argmin(axis=1)
    nudged = hue.copy()
    for k in range(NUM_SLOTS):
        if k in (RED, GREEN, BLUE):
            continue
        members = nearest == k
        if not members.any():
            continue
        mean = circular_mean(vivid[members, 2], vivid[members, 1])
        move = float(
            np.clip(
                NUDGE_FRACTION * wrap_degrees(mean - nudged[k]),
                -NUDGE_MAX_MOVE,
                NUDGE_MAX_MOVE,
            )
        )
        if abs(move) < NUDGE_MIN_MOVE:
            continue
        moved = _try_move(nudged, k, (nudged[k] + move) % 360.0, lightness, chroma)
        if moved is not None:
            nudged = moved
    return nudged


def _snap_semantic(hue: Array, slots: Slots, lightness: Array, chroma: Array) -> Array:
    snapped = hue.copy()
    for slot, target in SNAP_TARGETS.items():
        low, high = SNAP_BANDS[slot]
        if slots.found[slot] or not low <= snapped[slot] <= high:
            continue
        if slot == RED and SNAP_RED_SKIP_LOW <= snapped[slot] < SNAP_RED_SKIP_HIGH:
            continue
        others = [float(snapped[k]) for k in range(NUM_SLOTS) if k != slot]
        candidates = [target]
        candidates += [(h - MIN_HUE_GAP) % 360.0 for h in others]
        candidates += [(h + MIN_HUE_GAP) % 360.0 for h in others]
        for candidate in candidates:
            if low <= candidate <= high:
                continue
            moved = _try_move(snapped, slot, candidate, lightness, chroma)
            if moved is None:
                continue
            snapped = moved
            break
    return snapped


def _accent_tones(
    slots: Slots,
    tone_slots: Slots,
    variant: Variant,
    contrast: float,
    light_lift: float,
    light_tone_shift: float,
) -> tuple[Array, Array]:
    shift = variant.sign * CONTRAST_SHIFT * (contrast - 0.5)
    lightness = np.clip(
        _tone(
            slots.lightness,
            tone_slots,
            variant.lightness_range,
            variant.lightness_shape,
        )
        + shift,
        *variant.accent_bounds,
    )
    if light_lift:
        lightness = np.clip(lightness + light_lift, *variant.accent_bounds)
    if variant.sign < 0 and light_tone_shift:
        lightness = np.clip(lightness + light_tone_shift, *variant.accent_bounds)
    chroma = np.maximum(
        _tone(slots.chroma, tone_slots, variant.chroma_range, variant.chroma_shape),
        MIN_CHROMA,
    )
    chroma = chroma * (LIGHT_CHROMA_GAIN if variant.sign < 0 else DARK_CHROMA_GAIN)
    return _repel_lightness(lightness, chroma, slots.hue, variant.accent_bounds), chroma


def _accent_hues(
    slots: Slots,
    lch: Array,
    lightness: Array,
    chroma: Array,
    variant: Variant,
    separation: str,
    centres: list[float] | None,
    cyan_spread: float,
    relief: bool,
    nudge: bool,
    snap: bool,
) -> Array:
    surviving = np.minimum(
        chroma,
        SURVIVING_CHROMA_FACTOR
        * max_chroma(np.column_stack([lightness, chroma, slots.hue])),
    )
    separated_input = slots.hue.copy()
    if variant.sign > 0 and not slots.found[GREEN]:
        excess = np.clip(
            separated_input[GREEN] - GREEN_PULL_START, 0.0, GREEN_PULL_LIMIT
        )
        separated_input[GREEN] -= GREEN_PULL_FACTOR * excess
    hue = _separate(separated_input, slots, _needed_gaps(lightness, surviving))
    if separation == "assign":
        hue = _assign_hues(
            hue,
            slots,
            separated_input,
            lch,
            lightness,
            chroma,
            centres,
            cyan_spread if variant.sign < 0 else 0.0,
        )
    if relief and separation == "isotonic":
        hue = _relieve_drag(hue, slots, separated_input, lightness, chroma)
    if nudge:
        hue = _coverage_nudge(hue, lch, lightness, chroma)
    if snap:
        hue = _snap_semantic(hue, slots, lightness, chroma)
    return hue


def accent_colors(
    lch: Array,
    variant: Variant,
    contrast: float,
    slots: Slots | None = None,
    source: Array | None = None,
    tone_vivid: bool | None = None,
    snap: bool = True,
    relief: bool = True,
    nudge: bool = True,
    light_lift: float = 0.0,
    light_tone_shift: float = 0.0,
    separation: str = "isotonic",
    centres: list[float] | None = None,
    cyan_spread: float = 0.0,
) -> Array:
    if slots is None:
        slots = find_slots(lch, LIGHT_SHARE_THRESHOLD if variant.sign < 0 else None)
    if slots.found.any():
        if source is None:
            source = _warp_source(lch, slots)
        slots = replace(slots, hue=_fill_hues(slots, source))
    if tone_vivid is None:
        tone_vivid = _tone_vivid_flag(lch, slots)
    tone_slots = (
        slots if tone_vivid else replace(slots, found=np.zeros(NUM_SLOTS, dtype=bool))
    )
    lightness, chroma = _accent_tones(
        slots, tone_slots, variant, contrast, light_lift, light_tone_shift
    )
    hue = _accent_hues(
        slots,
        lch,
        lightness,
        chroma,
        variant,
        separation,
        centres,
        cyan_spread,
        relief,
        nudge,
        snap,
    )

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
