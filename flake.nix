{
  description = "H-GCL Elliptic++: Mac CPU and NixOS CUDA environments";
  inputs.nixpkgs.url = "https://channels.nixos.org/nixos-26.05/nixexprs.tar.xz";
  outputs = { nixpkgs, ... }:
    let
      systems = [ "aarch64-darwin" "x86_64-linux" ];
      eachSystem = f: nixpkgs.lib.genAttrs systems (system: f system);
    in {
      devShells = eachSystem (system:
        let
          pkgs = import nixpkgs { inherit system; };
          tools = p: [ p.python311 p.uv p.git p.bashInteractive p.coreutils p.cacert p.tmux ];
          # Linux wheels expect the conventional loader/library layout. This is
          # an unprivileged FHS environment, not Docker or a system configuration.
          linuxEnv = pkgs.buildFHSEnv {
            name = "hgcl-lab";
            targetPkgs = p: tools p ++ [ p.zlib p.stdenv.cc.cc.lib ];
            runScript = "bash";
            profile = ''
              export HGCL_NIX_SYSTEM=x86_64-linux
              export UV_PYTHON_DOWNLOADS=never
              export LD_LIBRARY_PATH=/run/opengl-driver/lib:/run/opengl-driver-32/lib:''${LD_LIBRARY_PATH:-}
            '';
          };
        in {
          default = if system == "x86_64-linux" then linuxEnv.env else pkgs.mkShell {
            packages = tools pkgs;
            shellHook = ''
              export HGCL_NIX_SYSTEM=aarch64-darwin
              export UV_PYTHON_DOWNLOADS=never
            '';
          };
        });
    };
}
