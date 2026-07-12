"""Codegen permutation, import-contract and trajectory tests against a native C compiler.

For every ``(numeric, reductions, structure, layout, solver, api)`` combination
and a range of representative systems this checks three things:

1. **Generation** - ``Simulation.to_c`` must never raise (the always-on guard).

2. **Import contract** - the set of *external* functions the generated C calls
   becomes an ``env.*`` import when the app links with clang
   ``-nostdlib -Wl,--allow-undefined`` (see ``worker.ts``). Each one must be
   provided by the browser runner's import table (``WASM_IMPORT_NAMES`` in
   ``pathview-fastsim/src/lib/codegen/clang/wasmRunner.ts``). This is the guard
   that catches the missing-``fma``/``memcpy`` import class of bug - a native
   ``-c`` compile cannot see it because it links libc/libm directly.

3. **Trajectory** - where a C compiler is available, all generated ``.c`` plus a
   harness are compiled and run, and the integrated STATE trajectory is compared
   against fastsim's own fixed-step reference (same solver + dt) within
   ``atol + rtol*|ref|`` - the same verify the in-app "Validate" performs.

``gcc`` is the local compile-and-run engine; the import guard mirrors the
clang->wasm contract so the wasm-import class is covered without a wasm
toolchain. The faithful clang->wasm compile + run lives in
``test_codegen_wasm_matrix.py`` (gated on Docker + Node). Generation + import
contract are pure-Python and always run; compilation is the local deep check.

Run: ``python -m pytest tests/python/test_codegen_clang_matrix.py -v``
"""
import os
import shutil
import subprocess

import numpy as np
import pytest

from codegen_common import (
    COMBOS, KEYS, PROVIDED_IMPORTS, SOLVER_CLS, SYSTEMS, TRAJ_COMBOS, TRAJ_DOUBLE_ONLY,
    combo_id, external_calls, gen_main_c_print, match_worst, reference, state_count,
)

# Compiler resolution: `$FASTSIM_CC` (an explicit path — the machine's PATH `gcc`
# may be an ancient MSYS2 build that ICEs on doubles) wins, then a PATH gcc/cc.
# No hardcoded machine-specific path. Set `$FASTSIM_REQUIRE_CC=1` on a machine
# meant to verify (CI / release gate) so a missing compiler FAILS instead of the
# whole compile-and-run layer silently skipping.
def _resolve_cc():
    """Resolve the compiler spec. May be multi-word ("zig cc") — whitespace-
    split, first token is the program (same semantics as the Rust side's
    `codegen::verify::cc_command`)."""
    env_cc = os.environ.get("FASTSIM_CC")
    if env_cc:
        prog = env_cc.split()[0]
        if os.path.exists(prog) or shutil.which(prog):
            return env_cc
        return None
    return shutil.which("gcc") or shutil.which("cc")


_CC = _resolve_cc()
_REQUIRE_CC = bool(os.environ.get("FASTSIM_REQUIRE_CC"))

if _REQUIRE_CC and not _CC:
    raise RuntimeError(
        "FASTSIM_REQUIRE_CC is set but no working C compiler was found; "
        "point $FASTSIM_CC at a C99 compiler with libm."
    )


def _cc_env(work_dir=None):
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(_CC.split()[0]) + os.pathsep + env.get("PATH", "")
    # zig: isolate the caches per build dir — concurrent zig processes deadlock
    # on the shared global cache lock (mirrors Rust `cc_command_in`).
    if "zig" in _CC and work_dir is not None:
        env["ZIG_LOCAL_CACHE_DIR"] = str(work_dir / ".zig-local-cache")
        env["ZIG_GLOBAL_CACHE_DIR"] = str(work_dir / ".zig-global-cache")
    return env


def _compile_and_run(files, duration, dt, tmp_path):
    """Compile all generated .c + harness with the local compiler and run it.
    Returns (times, states); pytest.skip on a launch failure, pytest.fail on a
    genuine compile error."""
    for fn, src in files.items():
        (tmp_path / fn).write_text(src)
    inc = next((n for n in files if n.endswith("_solver.h")), "model.h")
    (tmp_path / "main.c").write_text(gen_main_c_print(files["model.h"], duration, dt, inc))
    cfiles = [str(tmp_path / f) for f in files if f.endswith(".c")] + [str(tmp_path / "main.c")]
    exe = str(tmp_path / "model_run.exe")
    env = _cc_env(tmp_path)
    r = subprocess.run([*_CC.split(), "-std=c99", "-O2", "-I", str(tmp_path), *cfiles, "-lm", "-o", exe],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        if "error:" not in r.stderr:
            pytest.skip(f"C compiler will not run here: {r.stderr.strip()[:80]}")
        pytest.fail("compile failed:\n" + r.stderr.strip())
    out = subprocess.run([exe], capture_output=True, text=True, env=env)
    assert out.returncode == 0, f"run crashed (code {out.returncode}): {out.stderr.strip()[:200]}"
    rows = [[float(x) for x in line.split()] for line in out.stdout.strip().splitlines()]
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1:]


@pytest.mark.parametrize("system", list(SYSTEMS))
@pytest.mark.parametrize("combo", COMBOS, ids=combo_id)
def test_generates_and_imports_supported(system, combo):
    """Every combo generates C, and every external symbol it calls is provided by
    the browser runner's import table. This is the missing-import guard."""
    opts = dict(zip(KEYS, combo))
    files = SYSTEMS[system][0]().to_c(**opts)
    assert files, f"{system} {opts}: no files"
    missing = external_calls(files) - PROVIDED_IMPORTS
    assert not missing, (
        f"{system} {opts}: generated C imports {sorted(missing)} not provided by "
        f"WASM_IMPORT_NAMES (wasmRunner.ts) - instantiation would fail with "
        f'"function import requires a callable". Add it to the runner + PROVIDED_IMPORTS.'
    )


@pytest.mark.skipif(not _CC, reason="no C compiler found")
@pytest.mark.parametrize("system", list(SYSTEMS))
@pytest.mark.parametrize("combo", TRAJ_COMBOS, ids=lambda c: "_".join(c[k] for k in KEYS))
def test_compiles_and_matches_reference(system, combo, tmp_path):
    """Compile + run the generated C and compare its state trajectory to fastsim's
    fixed-step reference within tolerance (the in-app Validate, headless)."""
    if combo["numeric"] == "float" and system in TRAJ_DOUBLE_ONLY:
        pytest.skip(f"{system}: float codegen vs double reference is too sensitive (noise)")
    factory, duration = SYSTEMS[system]
    files = factory().to_c(**combo)
    dt = factory().dt
    tc, xc = _compile_and_run(files, duration, dt, tmp_path)
    tr, xr = reference(factory(), duration, dt, SOLVER_CLS[combo["solver"]])
    worst, max_abs = match_worst(xc, xr, combo["numeric"])
    assert worst <= 1.0, (
        f"{system} {combo}: trajectory mismatch worst ratio={worst:.3g} (max abs={max_abs:.3g})"
    )
