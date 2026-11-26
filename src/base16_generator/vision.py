from __future__ import annotations

import numpy as np

from .color import Array, Rgb, _oklch_to_linear, decode_srgb, encode_srgb, srgb_to_oklch

PROTAN = np.array([[0.567, 0.433, 0.0], [0.558, 0.442, 0.0], [0.0, 0.242, 0.758]])
DEUTAN = np.array([[0.625, 0.375, 0.0], [0.7, 0.3, 0.0], [0.0, 0.3, 0.7]])
TRITAN = np.array([[0.95, 0.05, 0.0], [0.0, 0.433, 0.567], [0.0, 0.475, 0.525]])
DEFICIENCIES = {"protanopia": PROTAN, "deuteranopia": DEUTAN, "tritanopia": TRITAN}


def _lab_from_rgb(rgb: Rgb) -> Array:
    decoded = srgb_to_oklch(rgb)
    rad = np.radians(decoded[:, 2])
    return np.column_stack(
        [decoded[:, 0], decoded[:, 1] * np.cos(rad), decoded[:, 1] * np.sin(rad)]
    )


def simulated_lab(lch: Array, matrix: Array) -> Array:
    linear = _oklch_to_linear(lch) @ matrix.T
    rgb = encode_srgb(np.clip(linear, 0.0, 1.0))
    return _lab_from_rgb(rgb)


def simulated_lab_from_rgb(rgb: Rgb, matrix: Array) -> Array:
    linear = decode_srgb(rgb) @ matrix.T
    encoded = encode_srgb(np.clip(linear, 0.0, 1.0))
    return _lab_from_rgb(encoded)
