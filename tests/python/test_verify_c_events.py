"""SIL verification over event-carrying models, one per event mechanism.

``verify_c`` compiles the emitted C and compares it against the engine state by
state, so it is the check that catches a model which exports cleanly and then
computes something else. The class of bug it exists for is real: a
simulation-level event used to be dropped silently by ``to_c`` / ``to_fmu``,
and a bouncing ball fell through the floor.

Blocks reach the C through three different event mechanisms, and the emitted
code differs for each:

* **scheduled / periodic** — a due-time comparison advances a counter
  (``StepSource``, ``PulseSource``, ``ClockSource``, and every sampled block);
* **sampled memory** — a schedule event that also writes a memory slot
  (``SampleHold``, ``Delay``, ``FIR``, ``TappedDelay``, ``DiscreteIntegrator``);
* **threshold / state-dependent** — a guard over the current state
  (``Relay``, ``Comparator``, ``Switch``, ``Counter``, ``Backlash``).

This sweeps one model per block so a mechanism cannot regress unnoticed. The
per-block SiL parity in ``test_codegen_block_coverage.py`` compiles the same
blocks, but drives its own harness; this drives ``verify_c``, the API a user
would reach for, and so guards that path too.
"""
import numpy as np
import pytest

import fastsim as fs
from fastsim import Connection
from fastsim import blocks as B
from fastsim.solvers import RK4

from codegen_common import needs_cc

pytestmark = needs_cc

DT = 1e-3
DURATION = 1.0

# name -> (block factory, driver factory or None for a sinusoid)
SCHEDULED = {
    "StepSource": (lambda: B.Amplifier(1.0),
                   lambda: B.StepSource(amplitude=[1.0, 2.0, 0.5], tau=[0.1, 0.5, 0.8])),
    "PulseSource": (lambda: B.Amplifier(1.0), lambda: B.PulseSource(amplitude=1.0, T=0.25)),
    "SquareWaveSource": (lambda: B.Amplifier(1.0),
                         lambda: B.SquareWaveSource(amplitude=1.0, frequency=4.0)),
    "ClockSource": (lambda: B.Amplifier(1.0), lambda: B.ClockSource(T=0.1, tau=0.0)),
}

SAMPLED = {
    "SampleHold": (lambda: B.SampleHold(T=0.05, tau=0.0), None),
    "ZeroOrderHold": (lambda: B.ZeroOrderHold(T=0.05, tau=0.0), None),
    "Delay": (lambda: B.Delay(tau=0.1, sampling_period=0.02), None),
    "DiscreteIntegrator": (lambda: B.DiscreteIntegrator(T=0.05, tau=0.0, initial_value=[0.0]), None),
    "FIR": (lambda: B.FIR(coeffs=[0.5, 0.5], T=0.05, tau=0.0), None),
    "TappedDelay": (lambda: B.TappedDelay(N=2, T=0.05, tau=0.0), None),
}

THRESHOLD = {
    "Relay": (lambda: B.Relay(threshold_up=0.5, threshold_down=-0.5,
                              value_up=1.0, value_down=-1.0), None),
    "Comparator": (lambda: B.Comparator(threshold=0.0), None),
    "Deadband": (lambda: B.Deadband(lower=-0.5, upper=0.5), None),
    "Backlash": (lambda: B.Backlash(width=0.5, f_max=100.0), None),
    "RateLimiter": (lambda: B.RateLimiter(rate=1.0, f_max=100.0), None),
    "Switch": (lambda: B.Switch(switch_state=0), None),
    "Counter": (lambda: B.Counter(start=0.0, threshold=5.0), None),
}

ALL_CASES = {**SCHEDULED, **SAMPLED, **THRESHOLD}


def _model(block_factory, driver_factory):
    """driver -> block -> Integrator -> Scope.

    The integrator is what makes the check meaningful: `verify_c` compares
    continuous states, so the block's output has to reach one. It also
    accumulates any timing error rather than letting it cancel.
    """
    src = driver_factory() if driver_factory else B.SinusoidalSource(frequency=2.0, amplitude=1.0)
    blk = block_factory()
    integ = B.Integrator(0.0)
    sco = B.Scope()
    return fs.Simulation(
        [src, blk, integ, sco],
        [Connection(src, blk), Connection(blk, integ), Connection(integ, sco)],
        Solver=RK4, dt=DT, log=False,
    )


@pytest.mark.parametrize("name", sorted(ALL_CASES))
def test_generated_c_matches_the_engine(name):
    """The compiled C reproduces the engine's states, sample for sample."""
    block_factory, driver_factory = ALL_CASES[name]
    report = _model(block_factory, driver_factory).verify_c(
        name.lower(), duration=DURATION, dt=DT)

    assert report["passed"], (
        f"{name}: generated C diverges from the engine — worst scaled error "
        f"{report['max_scaled_error']:.3e} on state {report['worst_state']} at "
        f"t={report['worst_time']:.4f}")


def test_the_sweep_covers_every_mechanism():
    """A sweep that quietly lost a category would still be green."""
    assert SCHEDULED and SAMPLED and THRESHOLD
    assert len(ALL_CASES) == len(SCHEDULED) + len(SAMPLED) + len(THRESHOLD), "name collision"
    assert len(ALL_CASES) >= 15


def test_simulation_level_events_are_refused():
    """A global event has no static lowering, and must not export silently.

    Its guard and action are host closures: generated C could only leave them
    out, and would then integrate a different model. This is the failure that
    made a bouncing ball fall through the floor.
    """
    from fastsim.events import ZeroCrossingDown

    acc = B.Constant(-9.81)
    vel = B.Integrator(0.0)
    pos = B.Integrator(1.0)
    sco = B.Scope()

    def bounce(t):
        *_, v = vel()
        vel.engine.set(-0.8 * v)
        pos.engine.set(abs(pos()[-1]))

    evt = ZeroCrossingDown(func_evt=lambda t: pos()[-1], func_act=bounce, tolerance=1e-6)
    sim = fs.Simulation(
        [acc, vel, pos, sco],
        [Connection(acc, vel), Connection(vel, pos), Connection(pos, sco)],
        events=[evt], Solver=RK4, dt=DT, log=False,
    )

    with pytest.raises(Exception, match="simulation-level"):
        sim.to_c("ball")


def test_block_internal_events_still_export():
    """Not a blanket refusal: an event that lives in a block lowers fine."""
    files = _model(lambda: B.SampleHold(T=0.05, tau=0.0), None).to_c("held")
    joined = "".join(files.values())
    assert "handle_events" in joined, "a sampled block should emit event handling"
