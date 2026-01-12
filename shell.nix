{ pkgs ? (import <nixpkgs> {}).pkgs }:
with pkgs;
mkShell {
  packages = [
    uv
    python3
   python3Packages.numpy # Use Nix' numpy with proper c++ lib pathing
  ];
  buildInputs = [
  ];
  shellHook = ''
    # fixes libstdc++ issues and libgl.so issues
    LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib/:/run/opengl-driver/lib/
  '';
}
