########################################################################################
##
##              Cooperative streaming generator (run_streaming)
##
##   The fastsim.Simulation facade exposes run_streaming() as a Python generator
##   that advances the engine in sim-time chunks (run_begin + run_until) and
##   yields once per tick. This mirrors pathsim's generator API so a live UI
##   (e.g. pathview) can step it with next(), extract incremental Scope data
##   between ticks, inject mutations, and stop cooperatively.
##
##   These tests verify, against the native fastsim engine (no pathsim needed):
##     - the generator yields the expected number of frames
##     - Scope.read(incremental=True) returns only new samples and the pieces
##       concatenate to the full (non-incremental) read with no gaps/dupes
##     - chunked stepping is BIT-IDENTICAL to the blocking run (same timesteps)
##     - stop() between ticks terminates the generator promptly
##
########################################################################################

# IMPORTS ==============================================================================

import unittest

import numpy as np

from fastsim import Simulation, Connection
from fastsim.blocks import Source, Scope
from fastsim.solvers import RKDP54


# HELPERS ==============================================================================

_SOLVER_KW = dict(Solver=RKDP54, tolerance_lte_abs=1e-8, tolerance_lte_rel=0.0, log=False)


def _make_sim(dt=0.01):
    """Source(sin) -> Scope, adaptive solver. Returns (sim, scope)."""
    src = Source(lambda t: np.sin(t))
    sco = Scope()
    sim = Simulation([src, sco], [Connection(src, sco)], dt=dt, **_SOLVER_KW)
    return sim, sco


# TESTCASES ============================================================================

class TestFrameCount(unittest.TestCase):
    """duration * tickrate ticks, plus one final yield."""

    def test_yields_and_terminates(self):
        sim, _ = _make_sim()
        frames = list(sim.run_streaming(duration=1.0, reset=True, tickrate=10))
        # At least one chunk yield plus the final yield.
        self.assertGreaterEqual(len(frames), 2)

    def test_walltime_paced_not_simtime(self):
        # Regression guard: yields are WALL-CLOCK paced, not one per
        # duration*tickrate sim-time tick. A fast sim (here it completes in
        # milliseconds) must yield far fewer than duration*tickrate (=300)
        # times, else every tick pays the costly extract callback.
        sim, _ = _make_sim()
        frames = list(sim.run_streaming(duration=30.0, reset=True, tickrate=10))
        self.assertLess(
            len(frames), 50, f"expected wall-clock pacing, got {len(frames)} frames"
        )

    def test_callback_return_is_yielded(self):
        sim, sco = _make_sim()
        gen = sim.run_streaming(
            duration=1.0, reset=True, tickrate=5,
            func_callback=lambda: sco.read(incremental=True),
        )
        first = next(gen)
        # callback result is the (times, channels) tuple from Scope.read
        self.assertIsInstance(first, tuple)
        self.assertEqual(len(first), 2)
        list(gen)  # drain

    def test_no_callback_yields_none(self):
        sim, _ = _make_sim()
        frames = list(sim.run_streaming(duration=0.5, reset=True, tickrate=4))
        self.assertTrue(all(f is None for f in frames))


class TestIncrementalRead(unittest.TestCase):
    """Incremental pieces concatenate to the full read, no gaps or duplicates."""

    def test_incremental_concatenates_to_full(self):
        sim, sco = _make_sim()
        times = []
        for frame in sim.run_streaming(
            duration=1.0, reset=True, tickrate=10,
            func_callback=lambda: sco.read(incremental=True),
        ):
            t, _ = frame
            times.extend(np.asarray(t).tolist())

        full_t, _ = sco.read(incremental=False)
        np.testing.assert_array_equal(np.asarray(times), np.asarray(full_t))

    def test_incremental_cursor_resets_on_reset(self):
        sim, sco = _make_sim()
        list(sim.run_streaming(duration=0.5, reset=True, tickrate=10))
        # A fresh run with reset=True must re-yield from the start.
        first_times = []
        for frame in sim.run_streaming(
            duration=0.5, reset=True, tickrate=10,
            func_callback=lambda: sco.read(incremental=True),
        ):
            t, _ = frame
            first_times.extend(np.asarray(t).tolist())
        full_t, _ = sco.read(incremental=False)
        np.testing.assert_array_equal(np.asarray(first_times), np.asarray(full_t))


class TestParityWithBlockingRun(unittest.TestCase):
    """Chunked stepping reproduces the blocking run's timesteps exactly."""

    def test_scope_times_bit_identical(self):
        sim_a, sco_a = _make_sim()
        sim_a.run(1.0, reset=True)
        ta, cha = sco_a.read()

        sim_b, sco_b = _make_sim()
        list(sim_b.run_streaming(duration=1.0, reset=True, tickrate=10))
        tb, chb = sco_b.read()

        np.testing.assert_array_equal(ta, tb)
        self.assertEqual(len(cha), len(chb))
        for a, b in zip(cha, chb):
            np.testing.assert_array_equal(a, b)


class TestCooperativeStop(unittest.TestCase):
    """stop() between ticks ends the generator without running to completion."""

    def test_stop_terminates_early(self):
        sim, _ = _make_sim()
        gen = sim.run_streaming(duration=10.0, reset=True, tickrate=10)
        next(gen)  # advance one tick
        self.assertLess(sim.time, 10.0)
        sim.stop()
        self.assertFalse(sim.active)
        # Generator exits via the active check; only the final yield remains.
        remaining = list(gen)
        self.assertEqual(len(remaining), 1)
        self.assertLess(sim.time, 10.0)


if __name__ == "__main__":
    unittest.main()
