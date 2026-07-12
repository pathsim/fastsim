"""Faithful clang->wasm compile + run for the generated C.

This is the faithful half of the codegen verification: it reproduces the in-app
pipeline that ``test_codegen_clang_matrix.py`` can only approximate with a native
gcc compile.

  1. ``Simulation.to_c`` generates the C (+ a `buffer`-exporting harness that
     mirrors ``pathview-fastsim/.../clang/harness.ts``).
  2. clang (via the pip-installable ``ziglang`` package - no system toolchain or
     Docker needed) compiles it to WASM. The browser links freestanding with
     ``-Wl,--allow-undefined`` so libm comes from the JS ``env`` table; zig's
     linker rejects that flag, so we target ``wasm32-wasi`` (reactor), which
     statically links libm instead. Same clang codegen, self-contained module.
  3. Node instantiates and runs it (calling the reactor ``_initialize``) with the
     production import table available (``run_wasm.mjs``, mirroring
     ``wasmRunner.ts``), so a clang-only codegen issue surfaces as in the browser.
  4. The state trajectory is compared to fastsim's fixed-step reference within
     tolerance (permutation-invariant pairing; see ``codegen_common``).

The ``env`` import NAMES (the ``fma``/``memcpy`` missing-import class) are covered
deterministically by the import-contract guard in ``test_codegen_clang_matrix.py``;
this layer adds real clang->wasm codegen + real WASM execution of the trajectory.

Gated on ziglang + Node; skips cleanly otherwise. ``pip install ziglang`` enables
it locally and on CI.

Run: ``python -m pytest tests/python/test_codegen_wasm_matrix.py -v``
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from codegen_common import (
    KEYS, SOLVER_CLS, SYSTEMS, TRAJ_COMBOS, TRAJ_DOUBLE_ONLY,
    gen_main_c_buffer, match_worst, reference, state_count,
)

_HERE = os.path.dirname(__file__)
_RUNNER = os.path.join(_HERE, "wasm_run", "run_wasm.mjs")
_NODE = shutil.which("node")


def _have_zig():
    try:
        return subprocess.run([sys.executable, "-m", "ziglang", "version"],
                              capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False


_ZIG = _have_zig()
_WASM_READY = bool(_NODE) and _ZIG
_SKIP_REASON = "needs `pip install ziglang` + Node for the faithful clang->wasm run"

# clang->wasm via zig. wasm32-wasi reactor (zig's linker rejects the browser's
# freestanding `--allow-undefined`); libm is statically linked, the module is
# self-contained. Export run + buffer like worker.ts.
_ZIG_FLAGS = [
    "-target", "wasm32-wasi", "-mexec-model=reactor",
    "-Wl,--export=run", "-Wl,--export=buffer", "-O2", "-std=c99",
]

# Faithful spot-check subset (the gcc layer is exhaustive over all systems/combos).
_WASM_SYSTEMS = ["ode", "mathchain", "lorenz", "statespace", "pid", "event"]


def _compile_wasm(files, duration, dt, tmp_path):
    for fn, src in files.items():
        (tmp_path / fn).write_text(src)
    inc = next((n for n in files if n.endswith("_solver.h")), "model.h")
    (tmp_path / "main.c").write_text(gen_main_c_buffer(files["model.h"], duration, dt, inc))
    cfiles = [str(tmp_path / f) for f in files if f.endswith(".c")] + [str(tmp_path / "main.c")]
    out = tmp_path / "out.wasm"
    cmd = [sys.executable, "-m", "ziglang", "cc", *_ZIG_FLAGS, "-o", str(out), *cfiles]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        pytest.fail(f"clang->wasm failed:\n{r.stderr.strip()[:600]}")
    assert out.exists() and out.stat().st_size > 0, "no wasm produced"
    return out


def _run_wasm(wasm_path, n_state):
    r = subprocess.run([_NODE, _RUNNER, str(wasm_path), str(n_state)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"node wasm run failed: {r.stderr.strip()[:400]}"
    data = json.loads(r.stdout)
    time = np.asarray(data["time"], dtype=float)
    states = np.asarray(data["states"], dtype=float).T if data["states"] else np.empty((len(time), 0))
    return time, states


@pytest.mark.skipif(not _WASM_READY, reason=_SKIP_REASON)
@pytest.mark.parametrize("system", _WASM_SYSTEMS)
@pytest.mark.parametrize("combo", TRAJ_COMBOS, ids=lambda c: "_".join(c[k] for k in KEYS))
def test_wasm_compiles_and_matches_reference(system, combo, tmp_path):
    if combo["numeric"] == "float" and system in TRAJ_DOUBLE_ONLY:
        pytest.skip(f"{system}: float codegen vs double reference is too sensitive (noise)")
    factory, duration = SYSTEMS[system]
    files = factory().to_c(**combo)
    dt = factory().dt
    n_state = state_count(files["model.h"])
    wasm = _compile_wasm(files, duration, dt, tmp_path)
    tc, xc = _run_wasm(wasm, n_state)
    tr, xr = reference(factory(), duration, dt, SOLVER_CLS[combo["solver"]])
    worst, max_abs = match_worst(xc, xr, combo["numeric"])
    assert worst <= 1.0, (
        f"{system} {combo}: wasm trajectory mismatch worst ratio={worst:.3g} (max abs={max_abs:.3g})"
    )
