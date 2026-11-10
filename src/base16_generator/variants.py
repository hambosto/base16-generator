from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class Variant:
    neutral_lightness: tuple[float, ...]
    accent_lightness: tuple[float, ...]
    accent_range: tuple[float, float]
    direction: float


DARK: Final = Variant(
    neutral_lightness=(0.22, 0.26, 0.32, 0.48, 0.68, 0.86, 0.91, 0.96),
    accent_lightness=(0.66, 0.74, 0.85, 0.78, 0.78, 0.70, 0.70, 0.62),
    accent_range=(0.60, 0.88),
    direction=1.0,
)

LIGHT: Final = Variant(
    neutral_lightness=(0.97, 0.93, 0.88, 0.68, 0.50, 0.34, 0.27, 0.20),
    accent_lightness=(0.54, 0.56, 0.54, 0.52, 0.52, 0.48, 0.48, 0.46),
    accent_range=(0.35, 0.60),
    direction=-1.0,
)

VARIANTS: Final = {"dark": DARK, "light": LIGHT}
