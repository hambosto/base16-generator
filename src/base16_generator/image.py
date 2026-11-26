from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from .color import Array, srgb_to_oklch

MIN_ALPHA = 128
MAX_PIXELS = 100_000_000


def _convert_profile(image: Image.Image, profile: bytes) -> Image.Image:
    try:
        source = ImageCms.getOpenProfile(BytesIO(profile))
        target = ImageCms.createProfile("sRGB")
    except Exception:
        return image
    if image.mode not in ("RGBA", "RGB", "CMYK"):
        return image
    try:
        if image.mode == "RGBA":
            alpha = image.getchannel("A")
            rgb = ImageCms.profileToProfile(
                image.convert("RGB"), source, target, outputMode="RGB"
            )
            if rgb is None:
                return image
            converted = rgb.convert("RGBA")
            converted.putalpha(alpha)
            return converted
        rgb = ImageCms.profileToProfile(image, source, target, outputMode="RGB")
        return rgb if rgb is not None else image
    except Exception:
        return image


def load_oklch(path: Path, resize: int, quantize: int = 0) -> Array:
    if resize < 1:
        raise ValueError("resize must be a positive integer")
    if quantize < 0:
        raise ValueError("quantize must be zero or a positive integer")
    try:
        opened = Image.open(path)
    except Image.DecompressionBombError as error:
        raise ValueError(f"image too large: exceeds {MAX_PIXELS} pixels") from error
    with opened as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError(
                f"image too large: {image.width}x{image.height}"
                f" exceeds {MAX_PIXELS} pixels"
            )
        image.thumbnail((resize, resize), Image.Resampling.LANCZOS)
        profile = image.info.get("icc_profile")
        sampled: Image.Image = image
        if isinstance(profile, bytes) and profile:
            sampled = _convert_profile(image, profile)
        if quantize:
            rgb = sampled.convert("RGB").quantize(
                colors=quantize, method=Image.Quantize.MEDIANCUT
            )
            quantised = rgb.convert("RGB")
            if sampled.mode == "RGBA":
                quantised = quantised.convert("RGBA")
                quantised.putalpha(sampled.getchannel("A"))
            sampled = quantised
        rgba = np.asarray(sampled.convert("RGBA")).reshape(-1, 4)

    visible = rgba[rgba[:, 3] >= MIN_ALPHA, :3]
    if len(visible) == 0:
        raise ValueError("image has no visible pixels")
    return srgb_to_oklch(visible)
