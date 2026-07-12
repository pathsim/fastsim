########################################################################################
##
##              Testing StopSimulation + callback-error propagation
##              (pathsim drop-in parity)
##
########################################################################################

import unittest

import fastsim
from fastsim import Simulation, Connection, StopSimulation
from fastsim.blocks import Source, Scope


class TestStopSimulationImports(unittest.TestCase):

    def test_exported_from_top_level_and_exceptions(self):
        from fastsim import StopSimulation as S1
        from fastsim.exceptions import StopSimulation as S2
        self.assertIs(S1, S2)
        self.assertTrue(issubclass(S1, Exception))


class TestStopSimulationSystem(unittest.TestCase):

    def _system(self, src_func):
        src = Source(src_func)
        sco = Scope()
        sim = Simulation([src, sco], [Connection(src, sco)], dt=0.01)
        return sim

    def test_stop_simulation_halts_cleanly(self):
        """A block raising StopSimulation stops the run without propagating."""
        def src(t):
            if t > 0.5:
                raise StopSimulation(f"threshold reached at t={t:.3f}")
            return t

        sim = self._system(src)
        # Must NOT raise — StopSimulation is caught and terminates the run.
        sim.run(10.0)

        self.assertFalse(sim.active, "sim should be inactive after StopSimulation")
        self.assertLess(sim.time, 1.0, f"sim should stop near t=0.5, got {sim.time}")

    def test_other_exception_is_reraised(self):
        """A non-StopSimulation error in a callback propagates out of run()."""
        def src(t):
            if t > 0.5:
                raise ValueError("boom")
            return t

        sim = self._system(src)
        with self.assertRaises(ValueError):
            sim.run(10.0)

    def test_clean_run_is_unaffected(self):
        """A run with no raising callback completes normally."""
        sim = self._system(lambda t: t)
        sim.run(1.0)
        self.assertGreaterEqual(sim.time, 0.99)


if __name__ == "__main__":
    unittest.main()
