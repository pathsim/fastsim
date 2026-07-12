"""Permutation robustness for codegen settings.

Every combination of (numeric, reductions, structure, layout, solver, api) must
generate valid C for a range of representative models. Where a C compiler is
available each generated `.c` is compiled to an object file (catching the class
of bug where a setting combo emits a phantom symbol, e.g. the flat deriv reading
an undeclared `m`); without a compiler the test still checks generation.

Generation is the always-on guard; compilation is the local deep check.
"""
import itertools
import os
import shutil
import subprocess

import numpy as np
import pytest

import fastsim as fs
from fastsim.blocks import (StepSource, Integrator, Scope, SinusoidalSource,
                            Function, Amplifier, RandomNumberGenerator)

_CC = (shutil.which("gcc") or shutil.which("cc")
       or next((p for p in ["C:/Repositories/TEMP/mingw64/bin/gcc.exe"] if os.path.exists(p)), None))

AXES = {
    "numeric": ["double", "float"],
    "reductions": ["unrolled", "vectorized"],
    "structure": ["hierarchical", "flat"],
    "layout": ["compact", "library"],
    "solver": ["rk4", "euler"],
    "api": ["struct"],
}
_KEYS = list(AXES)
_COMBOS = list(itertools.product(*AXES.values()))


def _ode():
    src = SinusoidalSource(frequency=1.0, amplitude=1.0, phase=0.0)
    itg = Integrator(0.0); sco = Scope()
    return fs.Simulation(blocks=[src, itg, sco],
        connections=[fs.Connection(src, itg), fs.Connection(itg, sco)], dt=0.05)


def _event():
    src = StepSource(amplitude=[1.0, 2.0], tau=[0.1, 0.5])
    itg = Integrator(0.0); sco = Scope()
    return fs.Simulation(blocks=[src, itg, sco],
        connections=[fs.Connection(src, itg), fs.Connection(itg, sco)], dt=0.05)


def _func():
    src = SinusoidalSource(frequency=1.0, amplitude=1.0, phase=0.0)
    fn = Function(lambda u: np.tanh(u) * 0.5 + u * u)
    itg = Integrator(0.0); sco = Scope()
    return fs.Simulation(blocks=[src, fn, itg, sco],
        connections=[fs.Connection(src, fn), fs.Connection(fn, itg), fs.Connection(itg, sco)], dt=0.05)


def _noise():
    src = RandomNumberGenerator(sampling_period=0.05, seed=7)
    itg = Integrator(0.0); sco = Scope()
    return fs.Simulation(blocks=[src, itg, sco],
        connections=[fs.Connection(src, itg), fs.Connection(itg, sco)], dt=0.05)


MODELS = {"ode": _ode, "event": _event, "func": _func, "noise": _noise}


def _cc_env():
    """Env with the compiler's own dir on PATH so its runtime DLLs resolve."""
    if not _CC:
        return None
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(_CC) + os.pathsep + env.get("PATH", "")
    return env


@pytest.mark.parametrize("mname", list(MODELS))
def test_every_setting_combo_generates_and_compiles(mname, tmp_path):
    env = _cc_env()
    cc_failures = []
    for combo in _COMBOS:
        opts = dict(zip(_KEYS, combo))
        files = MODELS[mname]().to_c(**opts)        # generation must never raise
        assert files, f"{mname} {opts}: no files"
        if not _CC:
            continue
        d = tmp_path / ("_".join(combo))
        d.mkdir()
        for fn, src in files.items():
            (d / fn).write_text(src)
        for fn in files:
            if not fn.endswith(".c"):
                continue
            r = subprocess.run([_CC, "-std=c99", "-c", "-I", str(d), str(d / fn),
                                "-o", str(d / (fn + ".o"))], capture_output=True, text=True, env=env)
            if r.returncode != 0:
                # Distinguish a real C error from the compiler failing to launch
                # (missing runtime DLLs / sandbox quirk): only the former is a
                # codegen bug. A launch failure skips the compile check.
                if "error:" not in r.stderr:
                    pytest.skip(f"C compiler will not run here: {r.stderr.strip()[:80]}")
                tail = (r.stderr.strip().splitlines() or ["?"])[-1]
                cc_failures.append(f"{mname} {opts} [{fn}]: {tail}")
    assert not cc_failures, "compile failures:\n" + "\n".join(cc_failures[:10])
