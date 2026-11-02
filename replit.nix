{ pkgs }:
let
  fresh = import (fetchTarball
    "https://github.com/NixOS/nixpkgs/archive/nixpkgs-unstable.tar.gz")
    { config.allowUnfree = true; };
in {
    deps = [
      fresh.claude-code
      pkgs.htop
      pkgs.vim
      pkgs.texliveFull
      pkgs.python310
      pkgs.bibtex-tidy
    ];
}
