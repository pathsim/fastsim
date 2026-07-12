"""End-to-end Python tests for the FMI 3.0 FMU block bindings.

Drives Reference-FMUs through the public `fastsim._fastsim.ModelExchangeFMU` /
`CoSimulationFMU` API via a full Simulation, mirroring the pathsim usage
pattern so that `import fastsim as pathsim` works as a drop-in.
"""

import math
import os
import pytest

from fastsim._fastsim import (
    CoSimulationFMU,
    Connection,
    ModelExchangeFMU,
    Scope,
    Simulation,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures", "fmi")


def fmu(name: str) -> str:
    return os.path.join(FIXTURES, f"{name}.fmu")


# ---------------------------------------------------------------------------
# Model Exchange — Dahlquist analytical reference: x(t) = exp(-t)
# ---------------------------------------------------------------------------

def test_dahlquist_me_analytical():
    f = ModelExchangeFMU(fmu("Dahlquist"))
    s = Scope()
    sim = Simulation([f, s], [Connection(f, s)], dt=1e-3)
    sim.run(1.0)

    _, chans = s.read()
    x_final = chans[0][-1]
    assert abs(x_final - math.exp(-1.0)) < 1e-3


def test_me_start_value_override():
    # Change decay rate k=1 → 2. Expect x(1) = exp(-2) ≈ 0.1353.
    f = ModelExchangeFMU(fmu("Dahlquist"), start_values={"k": 2.0})
    s = Scope()
    sim = Simulation([f, s], [Connection(f, s)], dt=1e-3)
    sim.run(1.0)

    _, chans = s.read()
    assert abs(chans[0][-1] - math.exp(-2.0)) < 2e-3


def test_me_unknown_start_value_raises():
    with pytest.raises(ValueError):
        ModelExchangeFMU(fmu("Dahlquist"), start_values={"nope": 1.0})


# ---------------------------------------------------------------------------
# Model Exchange — BouncingBall state event
# ---------------------------------------------------------------------------

def test_bouncingball_me_event():
    f = ModelExchangeFMU(fmu("BouncingBall"), tolerance=1e-10)
    s = Scope(labels=["h", "v"])
    sim = Simulation(
        [f, s],
        [Connection(f[0], s[0]), Connection(f[1], s[1])],
        dt=1e-3,
    )
    sim.run(1.5)

    _, chans = s.read()
    h = chans[0]
    v = chans[1]
    assert min(h) > -5e-3, f"ball went through floor: min h = {min(h)}"
    assert max(v) > 1.0, f"expected upward velocity after bounce, max v = {max(v)}"


# ---------------------------------------------------------------------------
# Co-Simulation — BouncingBall with eventMode + earlyReturn
# ---------------------------------------------------------------------------

def test_bouncingball_cs_event_mode():
    f = CoSimulationFMU(fmu("BouncingBall"), dt=1e-2)
    s = Scope(labels=["h", "v"])
    sim = Simulation(
        [f, s],
        [Connection(f[0], s[0]), Connection(f[1], s[1])],
        dt=1e-2,
    )
    sim.run(1.5)

    _, chans = s.read()
    assert min(chans[0]) > -1e-2
    assert max(chans[1]) > 1.0


# ---------------------------------------------------------------------------
# Clocks FMU (ScheduledExecution only) — must be rejected cleanly
# ---------------------------------------------------------------------------

def test_clocks_fmu_rejected_for_me():
    with pytest.raises(RuntimeError):
        ModelExchangeFMU(fmu("Clocks"))


def test_clocks_fmu_rejected_for_cs():
    with pytest.raises(RuntimeError):
        CoSimulationFMU(fmu("Clocks"), dt=0.1)
