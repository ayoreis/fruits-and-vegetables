{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  packages = [ pkgs.python3 ];

  # pip wheels (numpy, opencv) link against these shared libraries.
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib # libstdc++
    pkgs.zlib
  ];

  shellHook = ''
    if [ ! -d .venv ]; then
      python -m venv .venv
      .venv/bin/pip install -q -r requirements.txt
    fi
    source .venv/bin/activate
  '';
}
