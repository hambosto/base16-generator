# base16-generator

Generates a base16 color scheme from an image. Vivid pixels vote for seven
hue slots in OKLCH, neutrals are tinted from the image, and the finished
palette is contrast-checked and gamut-mapped to sRGB, so it holds up as a
terminal theme.

## Install

Needs [uv](https://docs.astral.sh/uv/) and Python 3.14+. From a checkout of
this repo:

```sh
uv tool install .
```

Or run it without installing:

```sh
uv run base16-generator -i wallpaper.png
```

## Usage

```sh
base16-generator -i wallpaper.png
```

Prints the scheme as JSON to stdout:

```json
{
  "scheme": "wallpaper",
  "author": "Base16 Generator (https://github.com/hambosto/base16-generator)",
  "base00": "1A1B26",
  "base01": "...",
  "base0F": "..."
}
```

Use `-o` to write a file and `-n` to set the scheme name:

```sh
base16-generator -i wallpaper.png -v light -c 0.7 -s 0.6 -n "My Theme" -o my-theme.json
```

Flags:

```text
-i, --image       image to read (required)
-v, --variant     dark or light (default: dark)
-c, --contrast    0.0 to 1.0 (default: 0.5)
-s, --richness    0.0 to 1.0 (default: 0.8)
-n, --name        scheme name (default: image filename)
-r, --resize      downscale before sampling (default: 256)
-o, --output      write to file instead of stdout
```

- `--contrast` shifts accent lightness and widens the neutral ramp; higher
  values push the palette further apart.
- `--richness` blends sampled accent lightness and chroma toward the values
  of well-known schemes: `0.0` keeps the image's own tones, `1.0` uses the
  typical values.

### As a library

```python
from base16_generator import generate_palette

palette = generate_palette(
    "wallpaper.png", variant="light", contrast=0.7, richness=0.6
)
```

This returns a dict mapping `base00`–`base0F` to lowercase hex strings without
a leading `#`. For more control, `load_oklch` and `build_palette` are exported
too:

```python
from base16_generator import VARIANTS, build_palette, load_oklch

palette = build_palette(
    load_oklch("wallpaper.png", resize=256),
    contrast=0.7,
    variant=VARIANTS["light"],
)
```

## Development

```sh
uv sync    # install dependencies
uv build   # build sdist and wheel
```
