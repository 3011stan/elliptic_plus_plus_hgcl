#!/usr/bin/env bash
# Always invoke from the repository, inside nix develop. No sudo or global installs.
set -euo pipefail
cd "$(dirname "$0")/../.."
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_PYTHON_DOWNLOADS=never
case "${HGCL_NIX_SYSTEM:-}" in
  x86_64-linux) hgcl_env=.venv-lab; hgcl_lock=requirements/lab-cuda.lock ;;
  aarch64-darwin) hgcl_env=.venv-nix; hgcl_lock=requirements/mac-cpu.lock ;;
  *) echo 'Entre primeiro no ambiente com nix develop.' >&2; exit 2 ;;
esac
[[ -f flake.lock ]] || { echo 'Gere flake.lock no laboratório antes de instalar.' >&2; exit 2; }
case "${1:-}" in
  lock)
    [[ "$HGCL_NIX_SYSTEM" == x86_64-linux ]] || { echo 'O lock CUDA é gerado no NixOS.' >&2; exit 2; }
    [[ ! -e "$hgcl_lock" ]] || { echo 'Lock existente: não será substituído.' >&2; exit 2; }
    hgcl_tmp=$(mktemp requirements/.lab-cuda.XXXXXX)
    trap 'rm -f "$hgcl_tmp"' EXIT
    uv pip compile pyproject.toml requirements/lab-cuda.in --extra dev \
      --python "$(command -v python3.11)" --generate-hashes --no-build \
      --custom-compile-command "bash scripts/lab/environment.sh lock" --output-file "$hgcl_tmp"
    mv "$hgcl_tmp" "$hgcl_lock"
    ;;
  install)
    [[ -f "$hgcl_lock" ]] || { echo 'Gere o lock Python antes de instalar.' >&2; exit 2; }
    [[ -d "$hgcl_env" ]] || uv venv --python "$(command -v python3.11)" "$hgcl_env"
    uv pip sync --python "$hgcl_env/bin/python" --require-hashes --no-build "$hgcl_lock"
    if [[ "$HGCL_NIX_SYSTEM" == x86_64-linux ]]; then
      # Linux build backend is included in the generated hash lock.
      uv pip install --python "$hgcl_env/bin/python" --no-deps --no-build-isolation --editable .
    else
      # Preserve the Mac installation procedure, including isolated build tooling.
      uv pip install --python "$hgcl_env/bin/python" --no-deps --editable .
    fi
    uv pip check --python "$hgcl_env/bin/python"
    ;;
  *) echo 'Uso: bash scripts/lab/environment.sh lock|install' >&2; exit 2 ;;
esac
