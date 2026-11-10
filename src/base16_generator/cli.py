from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .palette import generate_palette
from .scheme import format_scheme
from .variants import VARIANTS


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="base16-generator",
        description="Base16 palette from an image via OKLCH constrained k-means.",
    )
    parser.add_argument("-i", "--image", type=Path, required=True, help="path to image")
    parser.add_argument(
        "-c", "--contrast", type=float, default=0.5, help="0.0..1.0 (default 0.5)"
    )
    parser.add_argument(
        "-v",
        "--variant",
        choices=sorted(VARIANTS),
        default="dark",
        help="dark or light scheme (default dark)",
    )
    parser.add_argument("-n", "--name", help="scheme name (default: image file name)")
    parser.add_argument(
        "-r",
        "--resize",
        type=int,
        default=256,
        help="cap the longest side (NEAREST) (default 256)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="write to file instead of stdout"
    )
    args = parser.parse_args(argv)

    if not 0.0 <= args.contrast <= 1.0:
        parser.error("--contrast must be between 0.0 and 1.0")
    if args.resize is not None and args.resize < 1:
        parser.error("--resize must be a positive integer")

    try:
        palette = generate_palette(
            args.image,
            contrast=args.contrast,
            resize=args.resize,
            variant=args.variant,
        )
        text = format_scheme(palette, args.name or args.image.stem, args.variant)

        if args.output:
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
    except (OSError, ValueError) as exc:
        print(f"base16-generator: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
