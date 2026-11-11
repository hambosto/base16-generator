from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Variant:
    sign: float
    background: tuple[float, float]
    foreground: tuple[float, float]
    accent_bounds: tuple[float, float]
    rich_lightness: tuple[float, ...]
    rich_chroma: tuple[float, ...]
    brown_step: float


VARIANTS = {
    "dark": Variant(
        sign=1.0,
        background=(0.27, 0.17),
        foreground=(0.93, 0.985),
        accent_bounds=(0.60, 0.93),
        rich_lightness=(0.66, 0.74, 0.81, 0.75, 0.75, 0.70, 0.70),
        rich_chroma=(0.160, 0.131, 0.127, 0.128, 0.097, 0.110, 0.128),
        brown_step=0.12,
    ),
    "light": Variant(
        sign=-1.0,
        background=(0.95, 0.99),
        foreground=(0.30, 0.16),
        accent_bounds=(0.42, 0.60),
        rich_lightness=(0.57, 0.62, 0.65, 0.60, 0.62, 0.56, 0.57),
        rich_chroma=(0.161, 0.138, 0.127, 0.119, 0.096, 0.123, 0.145),
        brown_step=0.04,
    ),
}
