########################################################################################
##
##              Block-internal event forwarding through port() (issue #17 phase 4)
##
##   A ported mixed-signal pathsim block keeps its internal events: they are
##   translated to fastsim events (ZeroCrossing family, Schedule, Condition) and
##   attached to the block as block-internal events, so any Simulation tracks
##   them automatically. The wrapped block stays alive on the shim path so the
##   event callbacks (which close over it) read the current state, and func_act
##   state mutations are pushed back into fastsim's engine.
##
########################################################################################

# IMPORTS ==============================================================================

import unittest

import numpy as np
import pytest

from fastsim import Simulation, Connection, port
from fastsim.blocks import Scope
from fastsim.solvers import SSPRK22

pytestmark = pytest.mark.pathsim  # auto-skips when pathsim is not installed

_LN2 = float(np.log(2.0))  # dx/dt=-x from 1 crosses 0.5 every ln(2)


def _decay_hybrid(event_factory):
    """pathsim DynamicalSystem: dx/dt=-x from x0=1 with an internal event built
    by ``event_factory(block)``.
    """
    from pathsim.blocks import DynamicalSystem as PsDynamicalSystem

    blk = PsDynamicalSystem(
        func_dyn=lambda x, u, t: -x,
        func_alg=lambda x, u, t: x,
        initial_value=1.0,
    )
    blk.events.append(event_factory(blk))
    return blk


def _run(block, duration, dt=0.005):
    sco = Scope()
    sim = Simulation(
        blocks=[block, sco],
        connections=[Connection(block, sco)],
        Solver=SSPRK22, dt=dt, log=False,
    )
    sim.run(duration)
    _, [y] = sco.read()
    return np.asarray(y)


# TESTCASES ============================================================================

class TestZeroCrossingForwarding(unittest.TestCase):

    def test_event_fires_and_resets(self):
        from pathsim.events import ZeroCrossingDown
        # Reset to 1 when x crosses 0.5 downward -> sawtooth bounded in [0.5, 1.0].
        block = port(_decay_hybrid(lambda b: ZeroCrossingDown(
            func_evt=lambda t: b.state[0] - 0.5,
            func_act=lambda t: setattr(b, "state", np.array([1.0])),
        )))
        y = _run(block, 3.5)
        self.assertGreater(np.min(y), 0.49)   # without the event: ~exp(-3.5)
        self.assertLessEqual(np.max(y), 1.0 + 1e-9)

    def test_event_times_match_analytic(self):
        from pathsim.events import ZeroCrossingDown
        fired = []
        def act(t, b):
            fired.append(t)
            b.state = np.array([1.0])
        block = port(_decay_hybrid(lambda b: ZeroCrossingDown(
            func_evt=lambda t: b.state[0] - 0.5,
            func_act=lambda t, b=b: act(t, b),
        )))
        _run(block, 3.5)
        self.assertGreaterEqual(len(fired), 4)
        for k, t in enumerate(fired, start=1):
            self.assertAlmostEqual(t, k * _LN2, delta=0.02)

    def test_forwarding_is_logged(self):
        from pathsim.events import ZeroCrossingDown
        with self.assertLogs("fastsim.port", level="INFO") as cm:
            port(_decay_hybrid(lambda b: ZeroCrossingDown(func_evt=lambda t: b.state[0] - 0.5)))
        self.assertTrue(any("events" in m and "forwarded" in m for m in cm.output))


class TestScheduleAndCondition(unittest.TestCase):

    def test_schedule_event_times(self):
        from pathsim.events import Schedule
        fired = []
        block = port(_decay_hybrid(lambda b: Schedule(
            t_start=0.5, t_period=0.5,
            func_act=lambda t: (fired.append(t), setattr(b, "state", np.array([1.0])))[1],
        )))
        _run(block, 2.1)
        self.assertGreaterEqual(len(fired), 4)
        for k, t in enumerate(fired, start=1):
            self.assertAlmostEqual(t, 0.5 * k, delta=0.02)

    def test_condition_event_fires(self):
        from pathsim.events import Condition
        fired = []
        block = port(_decay_hybrid(lambda b: Condition(
            func_evt=lambda t: b.state[0] < 0.5,
            func_act=lambda t: fired.append(t),
        )))
        _run(block, 2.0)
        # x decays past 0.5 around t=ln(2)≈0.693 -> condition fires once.
        self.assertEqual(len(fired), 1)
        self.assertAlmostEqual(fired[0], _LN2, delta=0.05)


class TestEventLifecycle(unittest.TestCase):

    def test_events_are_block_internal(self):
        # Ported events live on the block, not just the simulation: a fresh
        # Simulation built from the same block re-detects them after reset.
        from pathsim.events import ZeroCrossingDown
        fired = []
        block = port(_decay_hybrid(lambda b: ZeroCrossingDown(
            func_evt=lambda t: b.state[0] - 0.5,
            func_act=lambda t: (fired.append(t), setattr(b, "state", np.array([1.0])))[1],
        )))
        _run(block, 1.5)
        n_first = len(fired)
        self.assertGreaterEqual(n_first, 2)
        # reset() must reset the block-internal event history too.
        _run(block, 1.5)  # fresh Simulation -> reset -> should fire again
        self.assertGreaterEqual(len(fired) - n_first, 2)


class TestUnsupportedEvent(unittest.TestCase):
    """Unknown event types (no exact translation) must warn, not silently vanish."""

    def test_unsupported_event_warns(self):
        from pathsim.blocks import DynamicalSystem as PsDynamicalSystem
        from pathsim.events._event import Event

        blk = PsDynamicalSystem(
            func_dyn=lambda x, u, t: -x,
            func_alg=lambda x, u, t: x,
            initial_value=1.0,
        )
        blk.events.append(Event(func_evt=lambda t: 0.0))  # base Event: unknown type

        with self.assertLogs("fastsim.port", level="WARNING") as cm:
            port(blk)
        self.assertTrue(any("unsupported" in m.lower() for m in cm.output))


# RUN TESTS LOCALLY ====================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
