from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Rgb = NDArray[np.uint8]
Mask = NDArray[np.bool_]
Angle = TypeVar("Angle", float, Array)

GAMUT_TOLERANCE = 1e-4
BISECTIONS = 20
CHROMA_CEILING = 0.4

_RGB_TO_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_LMS_TO_LAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_LMS_TO_RGB = np.linalg.inv(_RGB_TO_LMS)
_LAB_TO_LMS = np.linalg.inv(_LMS_TO_LAB)
_LUMA = np.array([0.2126, 0.7152, 0.0722])


def wrap_degrees(angle: Angle) -> Angle:
    return (angle + 180.0) % 360.0 - 180.0


def _decode_srgb(rgb: Rgb) -> Array:
    encoded = rgb / 255.0
    return np.where(
        encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4
    )


def _encode_srgb(linear: Array) -> Rgb:
    linear = np.clip(linear, 0.0, 1.0)
    encoded = np.where(
        linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1.0 / 2.4) - 0.055
    )
    return np.rint(encoded * 255.0).astype(np.uint8)


def srgb_to_oklch(rgb: Rgb) -> Array:
    lab = np.cbrt(_decode_srgb(rgb) @ _RGB_TO_LMS.T) @ _LMS_TO_LAB.T
    lightness, a, b = np.moveaxis(lab, -1, 0)
    hue = np.degrees(np.arctan2(b, a)) % 360.0
    return np.stack([lightness, np.hypot(a, b), hue], axis=-1)


def _oklch_to_linear(lch: Array) -> Array:
    lightness, chroma, hue = np.moveaxis(lch, -1, 0)
    rad = np.radians(hue)
    lab = np.stack([lightness, chroma * np.cos(rad), chroma * np.sin(rad)], axis=-1)
    return (lab @ _LAB_TO_LMS.T) ** 3 @ _LMS_TO_RGB.T


def _in_gamut(lch: Array) -> Mask:
    linear = _oklch_to_linear(lch)
    return np.all(
        (linear >= -GAMUT_TOLERANCE) & (linear <= 1.0 + GAMUT_TOLERANCE), axis=-1
    )


def max_chroma(lch: Array) -> Array:
    probe = np.array(lch, dtype=np.float64)
    probe[..., 0] = np.clip(probe[..., 0], 0.0, 1.0)
    low = np.zeros(probe.shape[:-1])
    high = np.full(probe.shape[:-1], CHROMA_CEILING)
    for _ in range(BISECTIONS):
        mid = (low + high) / 2.0
        probe[..., 1] = mid
        fits = _in_gamut(probe)
        low = np.where(fits, mid, low)
        high = np.where(fits, high, mid)
    return low


def oklch_to_srgb(lch: Array) -> Rgb:
    colors = np.array(lch, dtype=np.float64)
    colors[..., 0] = np.clip(colors[..., 0], 0.0, 1.0)
    colors[..., 1] = np.clip(colors[..., 1], 0.0, max_chroma(colors))
    return _encode_srgb(_oklch_to_linear(colors))


def relative_luminance(rgb: Rgb) -> Array:
    luminance: Array = _decode_srgb(rgb) @ _LUMA
    return luminance


def contrast_ratios(rgb: Rgb, references: Array) -> Array:
    luminance = relative_luminance(rgb)[:, None]
    lighter = np.maximum(luminance, references)
    darker = np.minimum(luminance, references)
    return (lighter + 0.05) / (darker + 0.05)


def _cap_chroma(colors: Array, headroom: float) -> Array:
    capped = colors.copy()
    capped[:, 1] = np.minimum(capped[:, 1], headroom * max_chroma(capped))
    return capped


def _nearest_passing_lightness(
    colors: Array, toward: float, passes: Callable[[Array], Mask]
) -> Array:
    near = colors[:, 0].copy()
    far = np.full(len(colors), toward)
    trial = colors.copy()
    for _ in range(BISECTIONS):
        trial[:, 0] = (near + far) / 2.0
        ok = passes(trial)
        far = np.where(ok, trial[:, 0], far)
        near = np.where(ok, near, trial[:, 0])
    return far


def enforce_contrast(
    colors: Array,
    minimums: Array,
    reference_rows: tuple[int, ...],
    toward: float,
    headroom: float = 1.0,
) -> Array:
    colors = _cap_chroma(colors, headroom)
    references = relative_luminance(oklch_to_srgb(colors[list(reference_rows)]))

    def passes(candidate: Array) -> Mask:
        ratios = contrast_ratios(oklch_to_srgb(candidate), references)
        return np.all(ratios >= minimums, axis=1)

    short = ~passes(colors)
    if not short.any():
        return colors

    lightness = _nearest_passing_lightness(
        colors, toward, lambda trial: passes(_cap_chroma(trial, headroom))
    )
    fixed = colors.copy()
    fixed[short, 0] = lightness[short]
    return _cap_chroma(fixed, headroom)
