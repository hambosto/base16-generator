from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np

from .accents import (
    CYAN_SPREAD,
    LIGHT_SHARE_THRESHOLD,
    MIN_VALID_DE,
    NUM_SLOTS,
    SLOT_HUES,
    Slots,
    accent_colors,
    canonical_inputs,
)
from .color import (
    Array,
    adjacent_min,
    enforce_contrast,
    max_chroma,
    oklch_to_srgb,
    relative_luminance,
    srgb_to_oklch,
)
from .image import load_oklch
from .neutrals import NUM_NEUTRALS, neutral_colors
from .variants import VARIANTS, Variant
from .vision import DEFICIENCIES, simulated_lab

HEADROOM = 0.9
MIN_RAMP_STEP = 0.01
STAGGER_TARGET_DE = 0.03
STAGGER_MAX_MOVE = 0.12
STAGGER_MAX_CHROMA_MOVE = 0.06
STAGGER_ROUNDS = 14
OPT_ROUNDS = 3
OPT_DELTAS = (0.02, -0.02, 0.05, -0.05)
OPT_MAX_MOVE = 0.06
LIGHT_LIFT = 0.05
LIGHT_TONE_SHIFT = 0.08
LIGHT_BASE_LIFT = 0.04
LIGHT_RAMP_LIFT = 0.5
LIGHT_FALLBACK_BOUNDS = (0.40, 0.56)
MIN_FINAL_GAP = 10.0
LIGHTNESS_DELTAS = (0.01, -0.01, 0.02, -0.02, 0.04, -0.04, 0.08, -0.08)
CHROMA_DELTAS = (0.02, -0.02, 0.04, -0.04, 0.06, -0.06)
SCORE_MEAN_DIVISOR = 0.15
SCORE_TARGET_DIVISOR = 0.10
MIN_CONTRAST_BASE00 = 4.5
MIN_CONTRAST_BASE02 = 2.6
REGULAR_HUE_TOL = 12.0
REGULAR_L_TOL = 0.05
REGULAR_C_TOL = 0.025
DARK_LIGHTNESS_TARGET = 0.72
LIGHT_LIGHTNESS_TARGET = 0.60
DEREGULARISE_MOVES = (0.06, 0.10)
REGULAR_L = np.array([0.659, 0.728, 0.788, 0.743, 0.740, 0.680, 0.683])
REGULAR_C = np.array([0.155, 0.126, 0.125, 0.131, 0.095, 0.108, 0.126])


def _regular_count(accents: Array) -> int:
    diff = np.abs((accents[:, 2] - SLOT_HUES + 180.0) % 360.0 - 180.0)
    return int(
        (
            (diff <= REGULAR_HUE_TOL)
            & (np.abs(accents[:, 0] - REGULAR_L) <= REGULAR_L_TOL)
            & (np.abs(accents[:, 1] - REGULAR_C) <= REGULAR_C_TOL)
        ).sum()
    )


MIN_CONTRAST = np.column_stack(
    [
        [1.0, 1.0, 1.0, 3.0, 4.5, 7.0, 7.0, 7.0] + [MIN_CONTRAST_BASE00] * 8,
        [1.0, 1.0, 1.0, 1.5, 3.0, 4.5, 4.5, 4.5] + [MIN_CONTRAST_BASE02] * 8,
    ]
)
BACKGROUND_ROWS = (0, 2)


def _keep_ramp_ordered(lightness: Array, sign: float) -> Array:
    ordered = lightness.copy()
    for i in range(1, len(ordered)):
        ordered[i] = sign * max(
            sign * ordered[i], sign * ordered[i - 1] + MIN_RAMP_STEP
        )
    return ordered


def _contrast_ratio(a: float, b: float) -> float:
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def _accent_passes(colors: Array, row: int) -> bool:
    luminance = relative_luminance(oklch_to_srgb(colors))
    accent = float(luminance[row])
    return _contrast_ratio(accent, float(luminance[0])) >= MIN_CONTRAST_BASE00 and (
        _contrast_ratio(accent, float(luminance[2])) >= MIN_CONTRAST_BASE02
    )


def _lab_from_lch(lch: Array) -> Array:
    rad = np.radians(lch[:, 2])
    return np.column_stack(
        [lch[:, 0], lch[:, 1] * np.cos(rad), lch[:, 1] * np.sin(rad)]
    )


def _simulated_min(accents: Array) -> float:
    return min(
        adjacent_min(simulated_lab(accents, matrix)) for matrix in DEFICIENCIES.values()
    )


def _normal_min(accents: Array) -> float:
    return adjacent_min(_lab_from_lch(accents))


def _stagger_score(
    staggered: Array, accents: Array, trial: Array, slot: int
) -> float | None:
    if np.allclose(trial, accents):
        return None
    candidate = staggered.copy()
    candidate[NUM_NEUTRALS : NUM_NEUTRALS + NUM_SLOTS] = trial
    if not _accent_passes(candidate, NUM_NEUTRALS + slot):
        return None
    if _normal_min(trial) < STAGGER_TARGET_DE:
        return None
    return _simulated_min(trial)


def _stagger_lightness(colors: Array, variant: Variant) -> Array:
    staggered = colors.copy()
    accents = staggered[NUM_NEUTRALS : NUM_NEUTRALS + NUM_SLOTS].copy()
    enforced = accents.copy()
    stuck: set[int] = set()
    for _ in range(STAGGER_ROUNDS):
        worst = STAGGER_TARGET_DE
        worst_pair = -1
        for matrix in DEFICIENCIES.values():
            lab = simulated_lab(accents, matrix)
            for k in range(NUM_SLOTS - 1):
                if k in stuck:
                    continue
                distance = float(np.sqrt(((lab[k] - lab[k + 1]) ** 2).sum()))
                if distance < worst:
                    worst = distance
                    worst_pair = k
        if worst_pair < 0:
            break
        k = worst_pair
        step = STAGGER_TARGET_DE - worst
        best: Array | None = None
        best_score = _simulated_min(accents)
        for slot in (k, k + 1):
            for delta in (step, -step):
                trial = accents.copy()
                trial[slot, 0] = np.clip(
                    trial[slot, 0] + delta,
                    variant.accent_bounds[0],
                    variant.accent_bounds[1],
                )
                if abs(trial[slot, 0] - enforced[slot, 0]) > STAGGER_MAX_MOVE:
                    continue
                score = _stagger_score(staggered, accents, trial, slot)
                if score is not None and score > best_score + 0.001:
                    best_score = score
                    best = trial
            for delta in CHROMA_DELTAS:
                trial = accents.copy()
                cap = float(max_chroma(trial[slot : slot + 1])[0])
                trial[slot, 1] = np.clip(trial[slot, 1] + delta, 0.02, cap)
                if abs(trial[slot, 1] - enforced[slot, 1]) > STAGGER_MAX_CHROMA_MOVE:
                    continue
                score = _stagger_score(staggered, accents, trial, slot)
                if score is not None and score > best_score + 0.001:
                    best_score = score
                    best = trial
        if best is None:
            stuck.add(k)
            continue
        accents = best
    staggered[NUM_NEUTRALS : NUM_NEUTRALS + NUM_SLOTS, 0] = accents[:, 0]
    staggered[NUM_NEUTRALS : NUM_NEUTRALS + NUM_SLOTS, 1] = accents[:, 1]
    return staggered


def _lightness_score(accents: Array, variant: Variant, target: float) -> float:
    lab = _lab_from_lch(accents)
    dists = [
        float(np.sqrt(((lab[i] - lab[j]) ** 2).sum()))
        for i, j in combinations(range(NUM_SLOTS), 2)
    ]
    return (
        _normal_min(accents) / STAGGER_TARGET_DE
        + _simulated_min(accents) / STAGGER_TARGET_DE
        + float(np.mean(dists)) / SCORE_MEAN_DIVISOR
        - abs(float(accents[:, 0].mean()) - target) / SCORE_TARGET_DIVISOR
    )


def _optimise_lightness(colors: Array, variant: Variant) -> Array:
    optimized = colors.copy()
    accents = optimized[NUM_NEUTRALS : NUM_NEUTRALS + NUM_SLOTS].copy()
    if (
        _simulated_min(accents) >= STAGGER_TARGET_DE
        and _normal_min(accents) >= MIN_VALID_DE
    ):
        return optimized
    enforced = accents.copy()
    target = DARK_LIGHTNESS_TARGET if variant.sign > 0 else LIGHT_LIGHTNESS_TARGET
    base_regular = _regular_count(accents)
    for _ in range(OPT_ROUNDS):
        improved = False
        for slot in range(NUM_SLOTS):
            best: Array | None = None
            best_score = _lightness_score(accents, variant, target)
            for delta in OPT_DELTAS:
                trial = accents.copy()
                trial[slot, 0] = np.clip(trial[slot, 0] + delta, *variant.accent_bounds)
                if np.allclose(trial[slot], accents[slot]):
                    continue
                if abs(trial[slot, 0] - enforced[slot, 0]) > OPT_MAX_MOVE:
                    continue
                if _regular_count(trial) > base_regular:
                    continue
                candidate = optimized.copy()
                candidate[NUM_NEUTRALS : NUM_NEUTRALS + NUM_SLOTS] = trial
                if not all(
                    _accent_passes(candidate, NUM_NEUTRALS + row)
                    for row in range(NUM_SLOTS)
                ):
                    continue
                if _normal_min(trial) < STAGGER_TARGET_DE:
                    continue
                hexes = [rgb.tobytes().hex() for rgb in oklch_to_srgb(candidate)]
                if len(set(hexes)) != 16:
                    continue
                value = _lightness_score(trial, variant, target)
                if value > best_score + 0.001:
                    best_score = value
                    best = trial
            if best is not None:
                accents = best
                improved = True
        if not improved:
            break
    optimized[NUM_NEUTRALS : NUM_NEUTRALS + NUM_SLOTS] = accents
    return optimized


def _is_regular(accents: Array, slot: int) -> bool:
    diff = abs((accents[slot, 2] - SLOT_HUES[slot] + 180.0) % 360.0 - 180.0)
    return bool(
        diff <= REGULAR_HUE_TOL
        and abs(accents[slot, 0] - REGULAR_L[slot]) <= REGULAR_L_TOL
        and abs(accents[slot, 1] - REGULAR_C[slot]) <= REGULAR_C_TOL
    )


def _deregularise(colors: Array, variant: Variant) -> Array:
    if variant.sign < 0:
        return colors
    fixed = colors.copy()
    accents = fixed[NUM_NEUTRALS : NUM_NEUTRALS + NUM_SLOTS].copy()
    regular_slots = [slot for slot in range(NUM_SLOTS) if _is_regular(accents, slot)]
    if len(regular_slots) < 4:
        return colors
    for slot in regular_slots[: len(regular_slots) - 3]:
        if not _is_regular(accents, slot):
            continue
        direction = 1.0 if accents[slot, 0] >= REGULAR_L[slot] else -1.0
        for sign in (direction, -direction):
            moved = False
            for magnitude in DEREGULARISE_MOVES:
                trial = accents.copy()
                trial[slot, 0] = np.clip(
                    trial[slot, 0] + sign * magnitude, *variant.accent_bounds
                )
                if np.allclose(trial[slot], accents[slot]):
                    continue
                if abs(trial[slot, 0] - REGULAR_L[slot]) <= 0.05:
                    continue
                candidate = fixed.copy()
                candidate[NUM_NEUTRALS : NUM_NEUTRALS + NUM_SLOTS] = trial
                if not _accent_passes(candidate, NUM_NEUTRALS + slot):
                    continue
                if _normal_min(trial) < STAGGER_TARGET_DE:
                    continue
                if _simulated_min(trial) < STAGGER_TARGET_DE:
                    continue
                hexes = [rgb.tobytes().hex() for rgb in oklch_to_srgb(candidate)]
                if len(set(hexes)) != 16:
                    continue
                accents = trial
                moved = True
                break
            if moved:
                break
    fixed[NUM_NEUTRALS : NUM_NEUTRALS + NUM_SLOTS] = accents
    return fixed


def _distinct_move(
    distinct: Array,
    row: int,
    channel: int,
    delta: float,
    variant: Variant,
    hexes: list[str],
) -> Array | None:
    trial = distinct.copy()
    if channel == 0:
        trial[row, 0] = np.clip(trial[row, 0] + delta, *variant.accent_bounds)
    else:
        cap = float(max_chroma(trial[row : row + 1])[0])
        trial[row, 1] = np.clip(trial[row, 1] + delta, 0.02, cap)
    if np.allclose(trial[row], distinct[row]):
        return None
    if not _accent_passes(trial, row):
        return None
    trial_hex = oklch_to_srgb(trial[row : row + 1])[0].tobytes().hex()
    if trial_hex in hexes[:row] + hexes[row + 1 :]:
        return None
    return trial


def _ensure_distinct(colors: Array, variant: Variant) -> Array:
    distinct = colors.copy()
    for row in range(NUM_NEUTRALS, len(distinct)):
        while True:
            hexes = [rgb.tobytes().hex() for rgb in oklch_to_srgb(distinct)]
            if hexes[row] not in hexes[:row] + hexes[row + 1 :]:
                break
            moved = False
            for delta in LIGHTNESS_DELTAS:
                trial = _distinct_move(distinct, row, 0, delta, variant, hexes)
                if trial is not None:
                    distinct = trial
                    moved = True
                    break
            if moved:
                continue
            for delta in CHROMA_DELTAS:
                trial = _distinct_move(distinct, row, 1, delta, variant, hexes)
                if trial is not None:
                    distinct = trial
                    moved = True
                    break
            if not moved:
                break
    return distinct


def _final_spacing_ok(colors: Array) -> bool:
    round_tripped = srgb_to_oklch(oklch_to_srgb(colors))
    hues = np.sort(round_tripped[NUM_NEUTRALS : NUM_NEUTRALS + NUM_SLOTS, 2])
    gaps = np.diff(np.append(hues, hues[0] + 360.0))
    if float(gaps.min()) < MIN_FINAL_GAP:
        return False
    lab = _lab_from_lch(round_tripped[NUM_NEUTRALS : NUM_NEUTRALS + NUM_SLOTS])
    return adjacent_min(lab) >= STAGGER_TARGET_DE


def _lift_ok(colors: Array) -> bool:
    round_tripped = srgb_to_oklch(oklch_to_srgb(colors))
    accents = round_tripped[NUM_NEUTRALS : NUM_NEUTRALS + NUM_SLOTS]
    return (
        _normal_min(accents) >= MIN_VALID_DE
        and _simulated_min(accents) >= STAGGER_TARGET_DE
        and _final_spacing_ok(colors)
    )


def build_palette(
    lch: Array,
    contrast: float = 0.5,
    variant: Variant = VARIANTS["dark"],
    slots: Slots | None = None,
    source: Array | None = None,
    tone_vivid: bool | None = None,
    snap: bool = True,
    relief: bool = True,
    nudge: bool = True,
    light_lift: float | None = None,
    separation: str = "assign",
    light_base_lift: float = LIGHT_BASE_LIFT,
    light_ramp_lift: float = LIGHT_RAMP_LIFT,
    light_tone_shift: float = LIGHT_TONE_SHIFT,
    centres: list[float] | None = None,
    cyan_spread: float = CYAN_SPREAD,
) -> dict[str, str]:
    if not 0.0 <= contrast <= 1.0:
        raise ValueError("contrast must be between 0.0 and 1.0")
    if light_lift is None:
        light_lift = LIGHT_LIFT if variant.sign < 0 else 0.0

    def _rebuild(
        rebuild_variant: Variant,
        rebuild_snap: bool,
        rebuild_relief: bool,
        rebuild_nudge: bool,
        rebuild_light_lift: float,
        rebuild_separation: str,
        rebuild_tone_shift: float,
        rebuild_cyan_spread: float = CYAN_SPREAD,
    ) -> dict[str, str]:
        return build_palette(
            lch,
            contrast,
            rebuild_variant,
            slots=slots,
            source=source,
            tone_vivid=tone_vivid,
            snap=rebuild_snap,
            relief=rebuild_relief,
            nudge=rebuild_nudge,
            light_lift=rebuild_light_lift,
            separation=rebuild_separation,
            light_base_lift=light_base_lift,
            light_ramp_lift=light_ramp_lift,
            light_tone_shift=rebuild_tone_shift,
            centres=centres,
            cyan_spread=rebuild_cyan_spread,
        )

    colors = np.vstack(
        [
            neutral_colors(lch, variant, contrast, light_base_lift, light_ramp_lift),
            accent_colors(
                lch,
                variant,
                contrast,
                slots,
                source,
                tone_vivid,
                snap,
                relief,
                nudge,
                light_lift,
                light_tone_shift,
                separation,
                centres,
                cyan_spread,
            ),
        ]
    )
    colors = enforce_contrast(
        colors,
        MIN_CONTRAST,
        BACKGROUND_ROWS,
        toward=(1.0 + variant.sign) / 2.0,
        headroom=HEADROOM,
    )
    colors = _stagger_lightness(colors, variant)
    colors = _optimise_lightness(colors, variant)
    colors = _deregularise(colors, variant)
    colors = _ensure_distinct(colors, variant)
    colors[:NUM_NEUTRALS, 0] = _keep_ramp_ordered(
        colors[:NUM_NEUTRALS, 0], variant.sign
    )
    if light_lift and not _lift_ok(colors):
        return _rebuild(variant, snap, relief, nudge, 0.0, separation, 0.0, cyan_spread)
    if snap and not _final_spacing_ok(colors):
        fallback_variant = (
            replace(
                variant,
                lightness_range=LIGHT_FALLBACK_BOUNDS,
                accent_bounds=LIGHT_FALLBACK_BOUNDS,
            )
            if variant.sign < 0
            else variant
        )
        return _rebuild(fallback_variant, False, False, False, 0.0, "isotonic", 0.0)
    return {
        f"base{i:02X}": rgb.tobytes().hex()
        for i, rgb in enumerate(oklch_to_srgb(colors))
    }


def generate_palette(
    path: str | Path,
    *,
    contrast: float = 0.5,
    variant: str = "dark",
    resize: int = 256,
    separation: str = "assign",
    quantize: int = 0,
) -> dict[str, str]:
    if resize == 256:
        return build_palette(
            load_oklch(Path(path), resize, quantize),
            contrast,
            VARIANTS[variant],
            separation=separation,
        )
    thresholds = LIGHT_SHARE_THRESHOLD if VARIANTS[variant].sign < 0 else None
    slots, source, tone_vivid, centres = canonical_inputs(
        load_oklch(Path(path), 256, quantize), thresholds
    )
    return build_palette(
        load_oklch(Path(path), resize, quantize),
        contrast,
        VARIANTS[variant],
        slots,
        source,
        tone_vivid,
        centres=centres,
        separation=separation,
    )
