# base16-generator

Generates a base16 color scheme from an image. Vivid pixels vote for seven
hue slots in OKLCH, neutrals are tinted from the image, and the finished
palette is contrast-checked and gamut-mapped to sRGB, so it holds up as a
terminal theme.

## Install

Needs Python 3.14+. From a checkout of this repo:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
pip install .
```

Or run it without installing:

```sh
pip install -e .
base16-generator -i wallpaper.png
```

## Usage

```sh
base16-generator -i wallpaper.png
```

Prints the scheme as JSON to stdout:

```json
{
  "scheme": "dark",
  "author": "base16-generator",
  "base00": "1A1B26",
  "base01": "...",
  "base0F": "..."
}
```

Use `-o` to write a file instead of printing it:

```sh
base16-generator -i wallpaper.png -v light -c 0.7 -o my-theme.json
```

Flags:

```text
-i, --image       image to read (required)
-v, --variant     dark or light (default: dark)
-c, --contrast    0.0 to 1.0 (default: 0.5)
-r, --resize      cap the longest side before sampling (default: 256)
-o, --output      write to file instead of stdout
```

- `--contrast` shifts accent lightness and widens the neutral ramp; higher
  values push the palette further apart.
- `--variant` sets which side of the ramp is the background. The `scheme`
  field in the output is the variant, `"dark"` or `"light"`.
- Images larger than 100 megapixels are rejected with a one-line error.
  An embedded ICC profile, if present, is converted to sRGB first;
  a corrupt profile falls back to the raw pixel values.

### As a library

```python
from base16_generator import generate_palette

palette = generate_palette("wallpaper.png", variant="light", contrast=0.7)
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
python3.14 -m venv .venv   # create the virtual environment
source .venv/bin/activate  # use it
pip install -e ".[dev]"    # install the package plus lint tools
ruff check src/
ruff format --check src/
mypy src/                  # strict
```
