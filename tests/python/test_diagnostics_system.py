########################################################################################
##
##              Testing diagnostics exposure (issue #29): the structured
##              convergence data the engine gathers every step is reachable
##              after a run instead of being discarded into a log string.
##
########################################################################################

import unittest

from fastsim import Simulation, Connection
from fastsim.blocks import Constant, Scope


class TestDiagnostics(unittest.TestCase):

    def test_run_summary_always_available(self):
        """run_summary carries the numeric outcome without diagnostics=True."""
        cns = Constant(1.0)
        sco = Scope()
        sim = Simulation([cns, sco], [Connection(cns, sco)], dt=0.01, log=False)
        sim.run(0.1)
        s = sim.run_summary
        for key in ("converged", "max_residual", "worst_block", "truncated_at"):
            self.assertIn(key, s)
        self.assertTrue(s["converged"])

    def test_diagnostics_snapshot_with_flag(self):
        """With diagnostics=True, the per-timestep snapshot is exposed with its
        iteration counts and a human-readable summary."""
        cns = Constant(1.0)
        sco = Scope()
        sim = Simulation([cns, sco], [Connection(cns, sco)],
                         dt=0.01, log=False, diagnostics=True)
        sim.run(0.1)
        d = sim.diagnostics
        self.assertIsNotNone(d)
        self.assertGreaterEqual(d.solve_iterations, 0)
        self.assertIsInstance(d.summary(), str)


if __name__ == "__main__":
    unittest.main()
