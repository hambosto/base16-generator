from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .palette import generate_palette
from .variants import VARIANTS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="base16-generator",
        description="Base16 scheme from the colors of an image.",
    )
    parser.add_argument("-i", "--image", type=Path, required=True)
    parser.add_argument(
        "-c", "--contrast", type=float, default=0.5, help="0.0 to 1.0 (default 0.5)"
    )
    parser.add_argument(
        "-v",
        "--variant",
        choices=sorted(VARIANTS),
        default="dark",
        help="(default dark)",
    )
    parser.add_argument(
        "-r",
        "--resize",
        type=int,
        default=256,
        help="cap the longest side before sampling (default 256)",
    )
    parser.add_argument("-o", "--output", type=Path, help="write here, not to stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        palette = generate_palette(
            args.image, contrast=args.contrast, variant=args.variant, resize=args.resize
        )
        scheme = {
            "scheme": args.variant,
            "author": "Base16 Generator (https://github.com/hambosto/base16-generator)",
            **{label: value.upper() for label, value in palette.items()},
        }
        text = json.dumps(scheme, indent=2) + "\n"
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
    except (OSError, ValueError) as error:
        print(f"base16-generator: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
