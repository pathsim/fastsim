"""PulseSource edge timing.

Regression: the `dt / t_rise` singularity at `t_rise == 0` used to be handled by
clamping the time constants to 1e-12. But the phase start times are DERIVED from
them (`t_start_fall = tau + t_rise + T*duty`), so clamping moved the whole edge
off its exact instant:

  * a pulse with T=1, duty=0.5 still read `amplitude` AT t = 0.5, because its
    falling edge had been pushed to 0.5 + 1e-12;
  * at a period boundary `dt / 1e-12` landed mid-ramp on the floating-point
    residue (dt ~ 6.7e-16) and emitted ~6.7e-4 — a value the pulse never takes.

A zero-length edge is now treated as instantaneous in the output equation and
the time constants are left alone.
"""
import numpy as np

import fastsim as fs
from fastsim.blocks import PulseSource, Scope
from fastsim.solvers import RK4

DT = 0.001


def _run(duration=2.0, **kwargs):
    src = PulseSource(**kwargs)
    sco = Scope()
    sim = fs.Simulation([src, sco], [fs.Connection(src, sco)],
                        Solver=RK4, dt=DT, log=False)
    sim.run(duration=duration, reset=True, adaptive=False)
    t, (y,) = sco.read()
    return np.asarray(t), np.asarray(y)


def test_ideal_pulse_is_exactly_two_valued():
    """With zero rise/fall the output only ever takes 0 or `amplitude`."""
    _, y = _run(amplitude=2.0, T=1.0)
    extra = sorted(set(np.round(y, 12)) - {0.0, 2.0})
    assert not extra, f"pulse emitted values outside {{0, amplitude}}: {extra}"


def test_falling_edge_lands_on_its_exact_instant():
    """duty=0.5, T=1 is `amplitude` on [0, 0.5) and 0 on [0.5, 1).

    Accumulated stepping puts a sample at 1.4999999999999456 — a few hundred
    ulp of clock drift below the nominal edge. Schedule detection absorbs that
    drift (pathsim#249): a step ending within the drift tolerance of the edge
    IS the edge, so that sample already reads the low phase. Without the
    allowance the edge fired one full step late. What must hold is that the
    first sample at or after the drift-corrected edge is low, and the one
    before it is high.
    """
    t, y = _run(amplitude=1.0, T=1.0)
    for edge in (0.5, 1.5):
        # the same drift allowance the schedule uses
        tol = 1e-10 * edge
        after = np.nonzero(t >= edge - tol)[0]
        assert len(after) and after[0] > 0, f"no samples bracket the edge at {edge}"
        i = int(after[0])
        assert t[i] - edge < DT, "no sample close enough after the edge"
        assert y[i] == 0.0, f"t={t[i]!r}: expected the low phase at the edge, got {y[i]}"
        assert y[i - 1] == 1.0, f"t={t[i-1]!r}: expected the high plateau before the edge"


def test_duty_cycle_sets_the_high_fraction():
    """The share of samples at `amplitude` follows `duty`, not `duty` shifted by
    a rise-time floor."""
    for duty in (0.25, 0.5, 0.75):
        t, y = _run(duration=4.0, amplitude=1.0, T=1.0, duty=duty)
        frac = float(np.mean(y == 1.0))
        assert abs(frac - duty) < 2 * DT, f"duty={duty}: high fraction {frac:.4f}"


def test_finite_rise_time_still_ramps():
    """A real (non-zero) rise time keeps its linear ramp — the instantaneous
    path must not swallow the general case."""
    t, y = _run(duration=1.0, amplitude=1.0, T=1.0, t_rise=0.1, duty=0.4)
    mid = int(np.argmin(np.abs(t - 0.05)))          # halfway up the ramp
    assert 0.4 < y[mid] < 0.6, f"expected ~0.5 mid-ramp, got {y[mid]}"
    top = int(np.argmin(np.abs(t - 0.15)))          # on the plateau
    assert y[top] == 1.0, f"expected the plateau at t=0.15, got {y[top]}"
