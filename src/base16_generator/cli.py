from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .palette import DEFAULT_RICHNESS, VARIANTS, generate_palette


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="base16-generator",
        description="Base16 scheme from an image via OKLCH hue-sector clustering.",
    )
    parser.add_argument("-i", "--image", type=Path, required=True, help="path to image")
    parser.add_argument(
        "-c", "--contrast", type=float, default=0.5, help="0.0..1.0 (default 0.5)"
    )
    parser.add_argument(
        "-s",
        "--richness",
        type=float,
        default=DEFAULT_RICHNESS,
        help="0.0 = accents exactly as the image has them, 1.0 = the typical "
        f"lightness and chroma of well-known schemes (default {DEFAULT_RICHNESS})",
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
        help="cap the longest side, NEAREST (default 256)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="write to file instead of stdout"
    )
    args = parser.parse_args(argv)

    try:
        palette = generate_palette(
            args.image,
            contrast=args.contrast,
            variant=args.variant,
            richness=args.richness,
            resize=args.resize,
        )
    except (OSError, ValueError) as exc:
        print(f"base16-generator: {exc}", file=sys.stderr)
        return 1

    scheme = {
        "scheme": args.name or args.image.stem,
        "author": "Base16 Generator (https://github.com/hambosto/base16-generator)",
        **{label: value.upper() for label, value in palette.items()},
    }
    text = json.dumps(scheme, indent=2) + "\n"

    try:
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
    except OSError as exc:
        print(f"base16-generator: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
