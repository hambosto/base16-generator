# base16-generator

Generates a base16 color scheme from an image. The palette is picked in
OKLCH space and contrast-checked, so it holds up as a terminal theme.

## Install

Needs [uv](https://docs.astral.sh/uv/). From a checkout of this repo:

```sh
uv tool install .
```

## Nix

A flake with [uv2nix](https://pyproject-nix.github.io/uv2nix/) is included.
Dependencies come straight from `uv.lock` as prebuilt wheels — no `uv sync`
needed.

Add this repo to your flake's `inputs` (e.g. in NixOS or home-manager configs):

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    base16-generator = {
      url = "github:hambosto/base16-generator";
      # Optional: share our nixpkgs and avoid building a second one
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
}
```

Then pull the package into an output. NixOS:

```nix
environment.systemPackages = [
  inputs.base16-generator.packages.${pkgs.system}.default
];
```

home-manager uses the same pattern:

```nix
home.packages = [
  inputs.base16-generator.packages.${pkgs.system}.default
];
```

Output available from this flake:

```text
overlays.default         adds pkgs.base16-generator to any package set
packages.<system>.default     virtualenv with the CLI
packages.<system>.application same, with the virtualenv hidden
apps.<system>.default         runs the CLI directly (nix run)
devShells.<system>.default    editable dev shell
```

Instead of referencing the package directly, you can import it through the
overlay (`nixpkgs.overlays = [ inputs.base16-generator.overlays.default ];`)
and then use `pkgs.base16-generator` everywhere in that config.

To try it once without declaring anything:

```sh
nix run github:hambosto/base16-generator -- -i wallpaper.png
```

For this repo's own development, `nix develop` provides a shell with the
package installed in editable mode, so the CLI picks up source changes without
rebuilds.

## Usage

```sh
base16-generator -i wallpaper.png
```

Prints the scheme as JSON to stdout. Use `-o` to write a file:

```sh
base16-generator -i wallpaper.png -v light -c 0.7 -n "My Theme" -o my-theme.json
```

Flags:

```text
-i, --image       image to read (required)
-v, --variant     dark or light (default: dark)
-c, --contrast    0.0 to 1.0 (default: 0.5)
-n, --name        scheme name (default: image filename)
-r, --resize      downscale before sampling (default: 256)
-o, --output      write to file instead of stdout
```

### As a library

```python
from base16_generator import generate_palette

palette = generate_palette("wallpaper.png", variant="light", contrast=0.7)
```

## Development

```sh
uv sync    # install dependencies
uv build   # build sdist and wheel
```

With Nix, `nix develop` gives you the same setup from `uv.lock` instead — the
package installed editable, no `uv sync`.
