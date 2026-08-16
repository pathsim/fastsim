"""Sampling instants of the discrete blocks.

Driving a sampling block with the ramp ``u(t) = t`` makes its behaviour
self-reporting: whatever it holds IS the instant it sampled. That turns "when
does this block sample, and how long does it hold" into a direct measurement,
with no engine internals involved.

Regression this locks in: ``Delay`` in discrete mode trimmed its ring buffer to
``N = round(tau / sampling_period)`` entries and emitted the front. With N
entries the front is only ``N - 1`` periods old, so the block delayed by
``tau - sampling_period``. The ring has to hold the sample just taken PLUS the
N in flight.
"""
import numpy as np
import pytest

import fastsim as fs
from fastsim.solvers import RK4

DT = 0.01
DURATION = 0.6
TS = 0.05


def _held(block_name, **kwargs):
    """Run `source(t) -> block -> scope` and return (times, held values)."""
    src = fs.blocks.Source(func=lambda t: t)
    blk = getattr(fs.blocks, block_name)(**kwargs)
    sco = fs.blocks.Scope()
    sim = fs.Simulation([src, blk, sco],
                        [fs.Connection(src, blk), fs.Connection(blk, sco[0])],
                        Solver=RK4, dt=DT, log=False)
    sim.run(duration=DURATION, reset=True, adaptive=False)
    t, (y,) = sco.read()
    return np.asarray(t), np.asarray(y)


def _transitions(t, y):
    return [(t[i], y[i]) for i in range(1, len(y)) if abs(y[i] - y[i - 1]) > 1e-12]


@pytest.mark.parametrize("name,kwargs", [
    ("SampleHold", dict(T=TS, tau=0.0)),
    ("ZeroOrderHold", dict(T=TS, tau=0.0)),
])
def test_hold_blocks_hold_the_instant_they_sampled(name, kwargs):
    """A hold block outputs the input value at its own sampling instant, so on a
    ramp the held value equals the time it was taken."""
    t, y = _held(name, **kwargs)
    for t_change, held in _transitions(t, y):
        assert held <= t_change + 1e-12, (
            f"{name}: held {held} at t={t_change} — a hold cannot report the future")
        assert t_change - held < TS + DT, (
            f"{name}: held {held} at t={t_change} is more than one period stale")


def test_delay_holds_a_sample_tau_old():
    """`Delay(tau, sampling_period)` must lag by tau, not by tau - one period.

    The held value is the instant the sample was taken, so `t_change - held` is
    the realised delay. It lands within one integration step of tau.
    """
    tau = 0.10
    t, y = _held("Delay", tau=tau, sampling_period=TS)
    changes = _transitions(t, y)
    assert changes, "Delay never produced a sample"
    for t_change, held in changes:
        realised = t_change - held
        assert abs(realised - tau) <= TS / 2 + DT, (
            f"Delay realised {realised:.4f}s of delay at t={t_change:.3f} "
            f"(sample from t={held:.3f}), expected {tau}s")


def test_delay_ring_length_follows_tau():
    """More periods of tau means proportionally more delay — guards against a
    fix that hardcodes one particular ring length."""
    for n in (1, 2, 3):
        tau = n * TS
        t, y = _held("Delay", tau=tau, sampling_period=TS)
        changes = _transitions(t, y)
        assert changes, f"Delay(tau={tau}) never produced a sample"
        realised = np.mean([tc - h for tc, h in changes])
        assert abs(realised - tau) <= TS / 2 + DT, (
            f"Delay(tau={tau}, Ts={TS}) realised {realised:.4f}s on average")
