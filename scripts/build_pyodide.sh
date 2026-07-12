#!/usr/bin/env bash
# Build a Pyodide (WASM/Emscripten) wheel for fastsim.
#
# The FMI feature pulls in libloading (runtime dlopen) and is excluded here via
# --no-default-features; everything else (solvers, JIT tape interpreter, blocks)
# is pure Rust and compiles to wasm32-unknown-emscripten. The codegen feature IS
# included (minijinja is pure Rust / WASM-safe) so sim.to_c() works in the browser.
#
# Prerequisites (see README "Pyodide build"):
#   - rustup with a nightly toolchain + rust-src + wasm32-unknown-emscripten target
#   - pyodide-build (pip) with the matching cross-build env installed
#   - emscripten matching PYODIDE_VERSION (emcc on PATH, or EMSDK_DIR set)
set -euo pipefail

PYODIDE_VERSION="${PYODIDE_VERSION:-0.29.4}"
EMSDK_DIR="${EMSDK_DIR:-$HOME/Projects/TEMP/emsdk}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Prefer a local .venv if pyodide-build is installed there.
if [ -d "$REPO_ROOT/.venv/bin" ] && [ -x "$REPO_ROOT/.venv/bin/pyodide" ]; then
  PATH="$REPO_ROOT/.venv/bin:$PATH"
fi

# Rust: nightly is required for -Z build-std (Pyodide builds the std for wasm).
export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-nightly}"
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

# Emscripten: use emcc if already on PATH (e.g. CI), else source emsdk.
if ! command -v emcc >/dev/null 2>&1; then
  if [ -f "$EMSDK_DIR/emsdk_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$EMSDK_DIR/emsdk_env.sh"
  else
    echo "error: emcc not found and EMSDK_DIR ($EMSDK_DIR) has no emsdk_env.sh" >&2
    exit 1
  fi
fi

# macOS framework Python can't verify TLS certs without certifi's bundle.
if [ -z "${SSL_CERT_FILE:-}" ] && python -c "import certifi" >/dev/null 2>&1; then
  SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
  export SSL_CERT_FILE
fi

# Install the cross-build env for the pinned Pyodide version (idempotent, cached).
pyodide xbuildenv install "$PYODIDE_VERSION"

# Drop the fmi feature (and libloading) for the WASM build; keep python bindings
# and codegen (sim.to_c() in the browser).
export MATURIN_PEP517_ARGS="--no-default-features --features python,codegen"

cd "$REPO_ROOT"
echo "Building fastsim Pyodide wheel (pyodide $PYODIDE_VERSION, emcc $(emcc --version | head -1))"
pyodide build

echo
echo "Built wheel(s):"
ls -1 "$REPO_ROOT"/dist/*wasm32*.whl 2>/dev/null || ls -1 "$REPO_ROOT"/dist/*.whl
