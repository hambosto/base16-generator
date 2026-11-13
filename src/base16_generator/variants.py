from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    sign: float
    background_bounds: tuple[float, float]
    foreground_bounds: tuple[float, float]
    accent_bounds: tuple[float, float]
    lightness_range: tuple[float, float]
    chroma_range: tuple[float, float]
    lightness_shape: tuple[float, ...]
    chroma_shape: tuple[float, ...]
    brown_step: float


VARIANTS = {
    "dark": Variant(
        sign=1.0,
        background_bounds=(0.14, 0.30),
        foreground_bounds=(0.84, 0.985),
        accent_bounds=(0.60, 0.93),
        lightness_range=(0.62, 0.80),
        chroma_range=(0.08, 0.173),
        lightness_shape=(-0.067, 0.014, 0.083, 0.028, 0.019, -0.031, -0.027),
        chroma_shape=(0.032, 0.003, -0.001, 0.0, -0.031, -0.018, 0.0),
        brown_step=0.12,
    ),
    "light": Variant(
        sign=-1.0,
        background_bounds=(0.93, 0.99),
        foreground_bounds=(0.14, 0.32),
        accent_bounds=(0.40, 0.56),
        lightness_range=(0.40, 0.56),
        chroma_range=(0.10, 0.177),
        lightness_shape=(-0.030, 0.023, 0.049, 0.0, 0.016, -0.044, -0.033),
        chroma_shape=(0.030, 0.007, -0.004, -0.012, -0.036, -0.009, 0.014),
        brown_step=0.04,
    ),
}
