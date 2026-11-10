from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

GAMUT_TOLERANCE: Final = 1e-4
BISECTIONS: Final = 24

_LINEAR_TO_LMS: Final[Array] = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)

_LMS_TO_LINEAR: Final[Array] = np.array(
    [
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ]
)

_LMS_CBRT_TO_LAB: Final[Array] = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)

_LAB_TO_LMS_CBRT: Final[Array] = np.array(
    [
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ]
)

_LUMA: Final[Array] = np.array([0.2126, 0.7152, 0.0722])


def wrap_degrees(angle: Array | float) -> Array | float:
    return (angle + 180.0) % 360.0 - 180.0


def _srgb_to_linear(rgb: NDArray[np.uint8]) -> Array:
    encoded = rgb / 255.0
    return np.where(
        encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4
    )


def linear_luminance(rgb: NDArray[np.uint8]) -> Array:
    return _srgb_to_linear(rgb) @ _LUMA


def srgb_to_oklch(rgb: NDArray[np.uint8]) -> Array:
    lab = np.cbrt(_srgb_to_linear(rgb) @ _LINEAR_TO_LMS.T) @ _LMS_CBRT_TO_LAB.T
    lightness, a, b = np.moveaxis(lab, -1, 0)
    hue = np.degrees(np.arctan2(b, a)) % 360.0
    return np.stack([lightness, np.hypot(a, b), hue], axis=-1)


def _oklch_to_linear(oklch: Array) -> Array:
    lightness, chroma, hue = np.moveaxis(oklch, -1, 0)
    rad = np.radians(hue)
    lab = np.stack([lightness, chroma * np.cos(rad), chroma * np.sin(rad)], axis=-1)
    return (lab @ _LAB_TO_LMS_CBRT.T) ** 3 @ _LMS_TO_LINEAR.T


def _in_gamut(oklch: Array) -> NDArray[np.bool_]:
    linear = _oklch_to_linear(oklch)
    return np.all(
        (linear >= -GAMUT_TOLERANCE) & (linear <= 1.0 + GAMUT_TOLERANCE), axis=-1
    )


def oklch_to_srgb(oklch: Array) -> NDArray[np.uint8]:
    colors = np.array(oklch, dtype=np.float64)
    colors[..., 0] = np.clip(colors[..., 0], 0.0, 1.0)
    colors[..., 1] = np.maximum(colors[..., 1], 0.0)

    outside = ~_in_gamut(colors)
    if outside.any():
        low = np.zeros_like(colors[..., 1])
        high = colors[..., 1].copy()
        trial = colors.copy()
        for _ in range(BISECTIONS):
            mid = (low + high) / 2.0
            trial[..., 1] = mid
            fits = _in_gamut(trial)
            low = np.where(fits, mid, low)
            high = np.where(fits, high, mid)
        colors[..., 1] = np.where(outside, low, colors[..., 1])

    linear = np.clip(_oklch_to_linear(colors), 0.0, 1.0)
    encoded = np.where(
        linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055
    )
    return np.rint(encoded * 255.0).astype(np.uint8)


def contrast_ratios(rgb: NDArray[np.uint8], background_luminance: Array) -> Array:
    luminance = linear_luminance(rgb)[:, None]
    lighter = np.maximum(luminance, background_luminance)
    darker = np.minimum(luminance, background_luminance)
    return (lighter + 0.05) / (darker + 0.05)


def meets_contrast(
    colors: Array, minimums: Array, background_luminance: Array
) -> NDArray[np.bool_]:
    ratios = contrast_ratios(oklch_to_srgb(colors), background_luminance)
    return np.all(ratios >= minimums, axis=1)


def enforce_contrast(
    colors: Array, minimums: Array, background_luminance: Array, toward: float
) -> Array:
    short = ~meets_contrast(colors, minimums, background_luminance)
    if not short.any():
        return colors

    near = colors[:, 0].copy()
    far = np.full(len(colors), toward)
    trial = colors.copy()
    for _ in range(BISECTIONS):
        mid = (near + far) / 2.0
        trial[:, 0] = mid
        passes = meets_contrast(trial, minimums, background_luminance)
        far = np.where(passes, mid, far)
        near = np.where(passes, near, mid)

    fixed = colors.copy()
    fixed[:, 0] = np.where(short, far, colors[:, 0])
    return fixed
