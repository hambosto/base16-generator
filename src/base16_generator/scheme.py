from __future__ import annotations

import json
import re

DEFAULT_SLUG: str = "base16-generator"


def format_scheme(palette: dict[str, str], name: str, variant: str) -> str:
    scheme = {
        "system": "base16",
        "name": name,
        "slug": re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or DEFAULT_SLUG,
        "author": "base16-generator",
        "variant": variant,
        "palette": palette,
    }
    return json.dumps(scheme, indent=2) + "\n"
