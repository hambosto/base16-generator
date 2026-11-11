from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .palette import DEFAULT_RICHNESS, generate_palette
from .variants import VARIANTS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="base16-generator",
        description="Base16 scheme from an image via OKLCH hue-family clustering.",
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
    return parser


def _format_scheme(palette: dict[str, str], variant: str) -> str:
    scheme = {
        "scheme": variant,
        "author": "Base16 Generator (https://github.com/hambosto/base16-generator)",
        **{label: value.upper() for label, value in palette.items()},
    }
    return json.dumps(scheme, indent=2) + "\n"


def _emit(text: str, output: Path | None) -> None:
    if output is None:
        sys.stdout.write(text)
    else:
        output.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        palette = generate_palette(
            args.image,
            contrast=args.contrast,
            variant=args.variant,
            richness=args.richness,
            resize=args.resize,
        )
        _emit(_format_scheme(palette, args.variant), args.output)
    except (OSError, ValueError) as exc:
        print(f"base16-generator: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())