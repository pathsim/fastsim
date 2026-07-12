"""`Simulation.verify_c`: SIL verification of the generated C from Python.

Skips when no local C compiler is available — unless FASTSIM_REQUIRE_CC is
set (CI), in which case a missing compiler fails loudly.
"""

import os

import pytest

from fastsim import Simulation, Connection
from fastsim.blocks import Integrator, Amplifier
from fastsim._fastsim import find_c_compiler

_CC = find_c_compiler()
_REQUIRE_CC = os.environ.get("FASTSIM_REQUIRE_CC", "") not in ("", "0")
if _CC is None and _REQUIRE_CC:
    raise RuntimeError(
        "FASTSIM_REQUIRE_CC is set but no working C compiler was found; "
        "point $FASTSIM_CC at a C99 compiler with libm."
    )

pytestmark = pytest.mark.skipif(_CC is None, reason="no working C compiler found")


def _oscillator():
    int_v, int_x, amp = Integrator(0.0), Integrator(1.0), Amplifier(-1.0)
    return Simulation(
        blocks=[int_v, int_x, amp],
        connections=[
            Connection(int_v, int_x),
            Connection(int_x, amp),
            Connection(amp, int_v),
        ],
        log=False,
    )


def test_verify_c_oscillator_passes():
    rep = _oscillator().verify_c("osc", duration=2.0, dt=1e-3)
    assert rep["passed"], rep
    assert rep["max_scaled_error"] <= 1.0
    assert rep["n_steps"] == 2000
    assert rep["n_states"] == 2
    assert rep["compiler"]
    assert rep["build_dir"] is None  # cleaned up by default
    assert any(f.endswith(".c") for f in rep["files"])


def test_verify_c_keep_build_keeps_sources():
    rep = _oscillator().verify_c("osc", duration=0.1, dt=1e-2, keep_build=True)
    assert rep["passed"]
    assert rep["build_dir"] and os.path.isdir(rep["build_dir"])
    names = os.listdir(rep["build_dir"])
    assert "osc.c" in names and "main.c" in names
    import shutil

    shutil.rmtree(rep["build_dir"], ignore_errors=True)


def test_verify_c_library_layout():
    rep = _oscillator().verify_c(
        "osc", duration=0.5, dt=1e-3, layout="library", structure="hierarchical"
    )
    assert rep["passed"], rep
    assert any(f.endswith("_solver.c") for f in rep["files"])


def test_verify_c_rejects_adaptive_solver():
    with pytest.raises(RuntimeError, match="adaptive"):
        _oscillator().verify_c("osc", solver="rkdp54")
