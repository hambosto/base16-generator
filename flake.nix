{
  description = "Base16 palette from an image via OKLCH constrained k-means";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;

      # Systems with prebuilt numpy/pillow wheels in uv.lock.
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];
      forAllSystems = lib.genAttrs systems;

      # Load the uv workspace & generate overlays from uv.lock.
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      # Production overlay: prefer prebuilt binary wheels.
      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };

      # Dev overlay: make the local package editable, pointing at the
      # source tree through the REPO_ROOT environment variable.
      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
        members = [ "base16-generator" ];
      };

      # Restrict local package sources to files the uv_build backend needs so
      # that packages (especially editables) only rebuild when they change.
      pyprojectOverrides = final: prev: {
        base16-generator = prev.base16-generator.overrideAttrs (old: {
          src = lib.fileset.toSource rec {
            root = ./.;
            fileset = lib.fileset.unions [
              (root + "/pyproject.toml")
              (root + "/README.md")
              (root + "/src")
            ];
          };
        });
      };

      pythonSets = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};

          # Auto-select an interpreter compatible with the lock's
          # requires-python constraint (>=3.14).
          python = lib.head (
            pyproject-nix.lib.util.filterPythonInterpreters {
              inherit (workspace) requires-python;
              inherit (pkgs) pythonInterpreters;
            }
          );
        in
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (
            lib.composeManyExtensions [
              # Build systems (uv_build) required when packages are built from source.
              pyproject-build-systems.overlays.wheel
              # Packages generated from uv.lock.
              overlay
              pyprojectOverrides
            ]
          )
      );
      # Per-system packages, shared between the packages & overlays outputs.
      forSystem =
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pythonSet = pythonSets.${system};
          inherit (pkgs.callPackages pyproject-nix.build.util { }) mkApplication;
        in
        {
          default = pythonSet.mkVirtualEnv "base16-generator-env" workspace.deps.default;

          # Same environment, but hiding the virtualenv plumbing such as
          # interpreter, activation scripts and pyvenv.cfg.
          application = mkApplication {
            venv = pythonSet.mkVirtualEnv "base16-generator-app-env" workspace.deps.default;
            package = pythonSet.base16-generator;
          };
        };

      # Overlay for consumers: import into any package set, then use
      # pkgs.base16-generator & pkgs.base16-generator-application.
      pkgsOverlay =
        final: prev:
        let
          # Read the system from prev.stdenv, NOT final.stdenv.
          # Forcing final.stdenv from an overlay attribute runs into
          # infinite recursion in the stdenv bootstrap fold.
          system = prev.stdenv.hostPlatform.system;
        in
        assert lib.assertMsg (lib.elem system systems)
          "base16-generator overlay: system '${system}' is not supported; \
          supported systems: ${lib.concatStringsSep ", " systems}";
        {
          base16-generator = (forSystem system).default;
          base16-generator-application = (forSystem system).application;
        };
    in
    {
      overlays.default = pkgsOverlay;

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pythonSet = pythonSets.${system}.overrideScope editableOverlay;
          virtualenv = pythonSet.mkVirtualEnv "base16-generator-dev-env" workspace.deps.all;
        in
        {
          default = pkgs.mkShell {
            packages = [
              virtualenv
              pkgs.uv
            ];

            env = {
              # Don't let uv manage a virtual environment, uv2nix provisions it.
              UV_NO_SYNC = "1";
              UV_PYTHON = pythonSet.python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
            };

            shellHook = ''
              unset PYTHONPATH
              export REPO_ROOT=$(git rev-parse --show-toplevel)
            '';
          };
        }
      );

      packages = forAllSystems forSystem;

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.application}/bin/base16-generator";
        };
      });
    };
}
