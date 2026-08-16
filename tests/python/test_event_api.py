"""The public surface of the event types, against pathsim.

pathsim gives every event the same methods, because they all inherit one base
class. fastsim's six wrappers each spelled their own list, and the lists had
drifted: ``Condition`` and ``ScheduleList`` had neither ``on()`` nor ``off()``,
only the three ``ZeroCrossing`` variants could be iterated, and nothing exposed
``tolerance``. ``examples_event/example_bouncingball_switched.py`` needs
``Condition.off()``, so the gap was load-bearing, not cosmetic.

The second half of this file covers the reason ``on``/``off`` do not go through
the event's cell: actions routinely switch tracking on the very event being
resolved (both examples below do), and the simulation holds a mutable borrow of
that event for the whole of ``resolve``.
"""
import numpy as np
import pytest

import fastsim as fs
import fastsim.events as fe
from fastsim.solvers import RK4

try:
    import pathsim as ps
    import pathsim.events as pe
    HAS_PATHSIM = True
except ImportError:                                       # pragma: no cover
    HAS_PATHSIM = False


# The six event types, each with a constructor that needs no simulation.
CTORS = {
    "ZeroCrossing":     lambda m: m.ZeroCrossing(func_evt=lambda t: t - 1.0),
    "ZeroCrossingUp":   lambda m: m.ZeroCrossingUp(func_evt=lambda t: t - 1.0),
    "ZeroCrossingDown": lambda m: m.ZeroCrossingDown(func_evt=lambda t: t - 1.0),
    "Schedule":         lambda m: m.Schedule(t_start=0.0, t_period=1.0),
    "ScheduleList":     lambda m: m.ScheduleList(times_evt=[1.0, 2.0]),
    "Condition":        lambda m: m.Condition(func_evt=lambda t: t > 1.0),
}

# Every event answers to these, whatever its type.
SHARED_SURFACE = ["on", "off", "reset", "buffer", "estimate", "detect", "resolve"]

# Deliberately not mirrored from pathsim:
#   func_evt / func_act — fastsim compiles the callables into Rust closures at
#     construction, so an attribute here could be read but not meaningfully
#     rebound, which is worse than its absence.
#   to_checkpoint / load_checkpoint — pathsim's per-object NPZ protocol;
#     fastsim checkpoints at the simulation level.


@pytest.mark.parametrize("name", sorted(CTORS))
@pytest.mark.parametrize("method", SHARED_SURFACE)
def test_every_event_has_the_shared_surface(name, method):
    """No event type may be missing a method another one has."""
    evt = CTORS[name](fe)
    assert callable(getattr(evt, method, None)), (
        f"{name} has no {method}() — the six wrappers must share one surface")


@pytest.mark.parametrize("name", sorted(CTORS))
def test_every_event_supports_the_container_protocol(name):
    """`len()`, `bool()` and iteration work before anything has been resolved."""
    evt = CTORS[name](fe)
    assert len(evt) == 0
    assert bool(evt) is True
    assert list(iter(evt)) == []


@pytest.mark.parametrize("name", sorted(CTORS))
def test_on_off_toggles_truthiness(name):
    evt = CTORS[name](fe)
    evt.off()
    assert not evt
    evt.on()
    assert evt


@pytest.mark.parametrize("name", sorted(CTORS))
def test_tolerance_is_readable_and_writable(name):
    evt = CTORS[name](fe)
    before = evt.tolerance
    assert isinstance(before, float)
    evt.tolerance = 1e-9
    assert evt.tolerance == 1e-9


@pytest.mark.skipif(not HAS_PATHSIM, reason="pathsim not installed")
@pytest.mark.parametrize("name", sorted(CTORS))
def test_surface_matches_pathsim(name):
    """Whatever pathsim exposes publicly, fastsim exposes too.

    Except the four documented above; they are asserted as known-absent so this
    test fails if one of them ever appears without the list being updated.
    """
    known_absent = {"func_evt", "func_act", "to_checkpoint", "load_checkpoint"}
    if name == "ScheduleList":
        # pathsim's ScheduleList subclasses Schedule and so carries t_start,
        # t_period and t_end. Its own detect() reads none of them — the schedule
        # comes from times_evt — so mirroring them would add three attributes
        # that look like knobs and do nothing.
        known_absent |= {"t_start", "t_period", "t_end"}
    f, p = CTORS[name](fe), CTORS[name](pe)
    public = lambda o: {a for a in dir(o) if not a.startswith("_")}
    missing = public(p) - public(f)
    unexpected = missing - known_absent
    assert not unexpected, f"{name} is missing {sorted(unexpected)} relative to pathsim"
    assert missing <= known_absent
    for dunder in ("__len__", "__iter__", "__bool__"):
        assert hasattr(f, dunder), f"{name} has no {dunder}"


# -- reentrancy ------------------------------------------------------------------------

def _bouncing_ball(engine, events_from):
    """Ball dropped onto a floor, with a Condition that ends event tracking.

    Mirrors the shape of pathsim's `example_bouncingball_switched.py`: the
    Condition's own action deactivates the Condition *and* the contact event,
    from inside the action, while both are being resolved.
    """
    B = engine.blocks
    Iv = B.Integrator(0.0)                      # velocity
    Ix = B.Integrator(1.0)                      # height
    g = B.Constant(-9.81)
    sco = B.Scope()
    blocks = [Iv, Ix, g, sco]
    connections = [engine.Connection(g, Iv), engine.Connection(Iv, Ix),
                   engine.Connection(Ix, sco[0])]

    def contact_evt(t):
        *_, x = Ix()
        return x

    def contact_act(t):
        *_, x = Ix()
        *_, v = Iv()
        Ix.engine.set(abs(x))
        Iv.engine.set(-0.8 * v)

    bounce = events_from.ZeroCrossingDown(func_evt=contact_evt,
                                          func_act=contact_act, tolerance=1e-4)

    def stop_act(t):
        bounce.off()      # a different event
        stop.off()        # ... and this one, while it is being resolved

    stop = events_from.Condition(func_evt=lambda t: len(bounce) >= 3,
                                 func_act=stop_act)

    sim = engine.Simulation(blocks, connections, events=[bounce, stop],
                            dt=0.01, log=False)
    return sim, bounce, stop, sco


def test_action_may_switch_its_own_event_off():
    """`off()` from inside the action of the event being resolved.

    The simulation holds a mutable borrow of the event across `resolve`, so this
    is the case the shared activation flag exists for. Before it, `Condition`
    simply had no `off()` to call.
    """
    sim, bounce, stop, _ = _bouncing_ball(fs, fe)
    sim.run(5.0, reset=True, adaptive=True)

    assert len(stop) == 1, "the Condition should have fired exactly once"
    assert not stop, "its action turned it off"
    assert not bounce, "its action turned the contact event off too"
    assert len(bounce) == 3, (
        f"contact tracking should stop at the third bounce, got {len(bounce)}")


@pytest.mark.skipif(not HAS_PATHSIM, reason="pathsim not installed")
def test_self_switching_events_agree_with_pathsim():
    """Same model, same event bookkeeping, in both engines."""
    f_sim, f_bounce, f_stop, f_sco = _bouncing_ball(fs, fe)
    p_sim, p_bounce, p_stop, p_sco = _bouncing_ball(ps, pe)
    f_sim.run(5.0, reset=True, adaptive=True)
    p_sim.run(5.0, reset=True, adaptive=True)

    assert len(f_bounce) == len(p_bounce)
    assert len(f_stop) == len(p_stop)
    assert bool(f_bounce) == bool(p_bounce)
    assert bool(f_stop) == bool(p_stop)

    # The recorded event times, not just their count.
    assert np.allclose(list(f_bounce), list(p_bounce), rtol=1e-3, atol=1e-4), (
        f"bounce times differ: {list(f_bounce)} vs {list(p_bounce)}")


def test_reset_reactivates_and_clears():
    sim, bounce, stop, _ = _bouncing_ball(fs, fe)
    sim.run(5.0, reset=True, adaptive=True)
    assert len(bounce) and not bounce

    bounce.reset()
    assert len(bounce) == 0
    assert bounce, "reset reactivates"
    assert list(iter(bounce)) == []


def test_iteration_yields_the_resolved_times():
    """`iter(event)` is the recorded event times, in order."""
    fired = []
    sched = fe.ScheduleList(times_evt=[0.1, 0.2, 0.3], func_act=fired.append)
    src, sco = fs.blocks.Constant(1.0), fs.blocks.Scope()
    sim = fs.Simulation([src, sco], [fs.Connection(src, sco)], events=[sched],
                        Solver=RK4, dt=0.01, log=False)
    sim.run(0.5, reset=True, adaptive=False)

    times = list(iter(sched))
    assert len(times) == len(sched) == 3
    assert times == sorted(times)
    assert np.allclose(times, [0.1, 0.2, 0.3], atol=1e-9)
    assert np.allclose(times, fired, atol=1e-12), "iteration matches what fired"


# -- scheduled firing times ------------------------------------------------------------

def _fire_times(event, duration, dt=0.01):
    src, sco = fs.blocks.Constant(1.0), fs.blocks.Scope()
    sim = fs.Simulation([src, sco], [fs.Connection(src, sco)], events=[event],
                        Solver=RK4, dt=dt, log=False)
    sim.run(duration, reset=True, adaptive=False)
    return list(iter(event))


@pytest.mark.parametrize("times", [
    [0.1, 0.2, 0.3],
    [0.15, 0.35],
    [0.07, 0.13, 0.29],
])
def test_schedule_list_fires_at_the_requested_times(times):
    """A scheduled event fires when it was asked to, to the last bit.

    The reference is the request itself, not the other engine: a time schedule
    has an exact answer. Off the step grid these used to land a full ``dt``
    early — ``detect`` reported an exact hit as ratio 0 (start of the step) when
    the hit is at ``t``, the step's end. Fixed here and upstream (pathsim#248).
    """
    got = _fire_times(fe.ScheduleList(times_evt=list(times)), duration=0.5)
    assert len(got) == len(times)
    assert np.allclose(got, times, atol=1e-12), f"{got} != {times}"


@pytest.mark.parametrize("t_start, t_period, duration, want", [
    (0.0, 0.1, 0.35, [0.0, 0.1, 0.2, 0.3]),
    (0.05, 0.1, 0.35, [0.05, 0.15, 0.25, 0.35]),
    (0.0, 0.07, 0.35, [0.0, 0.07, 0.14, 0.21, 0.28, 0.35]),
])
def test_schedule_fires_on_its_period(t_start, t_period, duration, want):
    got = _fire_times(fe.Schedule(t_start=t_start, t_period=t_period), duration)
    assert len(got) == len(want)
    assert np.allclose(got, want, atol=1e-12), f"{got} != {want}"


def test_schedule_timing_attributes_take_effect():
    """The setters reach the fields the simulation reads, not a copy."""
    sched = fe.Schedule(t_start=0.0, t_period=1.0)
    assert sched.t_start == 0.0 and sched.t_period == 1.0 and sched.t_end is None

    sched.t_start, sched.t_period, sched.t_end = 0.05, 0.1, 0.25
    assert (sched.t_start, sched.t_period, sched.t_end) == (0.05, 0.1, 0.25)

    got = _fire_times(sched, duration=0.5)
    assert np.allclose(got, [0.05, 0.15, 0.25], atol=1e-12), (
        f"{got}: the schedule should follow the reassigned period and stop at t_end")


def test_schedule_list_times_are_readable_and_sorted_on_assignment():
    sched = fe.ScheduleList(times_evt=[0.3, 0.1])
    assert sched.times_evt == [0.1, 0.3], "the constructor sorts"

    sched.times_evt = [0.25, 0.05, 0.15]
    assert sched.times_evt == [0.05, 0.15, 0.25], "so does assignment"

    got = _fire_times(sched, duration=0.4)
    assert np.allclose(got, [0.05, 0.15, 0.25], atol=1e-12)
