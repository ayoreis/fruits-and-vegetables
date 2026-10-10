{
  description = "Fruits & vegetables shelf monitoring";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [ pkgs.python3 ];

          # pip wheels (numpy, opencv) link against these shared libraries.
          LD_LIBRARY_PATH = pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isLinux (
            pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib # libstdc++
              pkgs.zlib
            ]
          );

          # Rebuild the venv when it was made with a different Python, and reinstall when requirements change.
          shellHook = ''
            if [ "$(readlink -f .venv/bin/python 2>/dev/null)" != "$(readlink -f "$(command -v python)")" ]; then
              rm -rf .venv
              python -m venv .venv
            fi
            if [ requirements.txt -nt .venv/.installed ]; then
              .venv/bin/pip install -q -r requirements.txt && touch .venv/.installed
            fi
            source .venv/bin/activate
          '';
        };
      });
    };
}
