from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .palette import VARIANTS, generate_palette


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
            resize=args.resize,
        )
    except (OSError, ValueError) as exc:
        print(f"iridion: {exc}", file=sys.stderr)
        return 1

    lines = [
        'system: "base16"',
        f"name: {json.dumps(args.name or args.image.stem)}",
        'author: "iridion"',
        f'variant: "{args.variant}"',
        "palette:",
        *(f'  {label}: "{value}"' for label, value in palette.items()),
    ]
    text = "\n".join(lines) + "\n"

    try:
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
    except OSError as exc:
        print(f"iridion: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
