########################################################################################
##
##              Testing fail-open numeric outcomes (issue #27): non-convergence
##              and truncation populate RunStats truthfully and emit an
##              unconditional, catchable FastSimConvergenceWarning instead of
##              returning a green summary over garbage.
##
########################################################################################

import unittest
import warnings

import numpy as np

from fastsim import Simulation, Connection
from fastsim.blocks import Constant, Scope, Adder, Function
from fastsim.exceptions import FastSimConvergenceWarning


class TestFailOpen(unittest.TestCase):

    def _nonconverging_sim(self):
        # A nonlinear feedthrough loop that cannot converge within a tiny
        # iteration budget: y = 1 + f(y), f oscillatory. iterations_max=2 gives
        # a single fixed-point pass, far from the NLS_COEF criterion.
        cns = Constant(1.0)
        add = Adder()
        fnc = Function(lambda x: 2.0 * np.sin(8.0 * x) + x)
        sco = Scope()
        return Simulation(
            [cns, add, fnc, sco],
            [Connection(cns, add[0]), Connection(add, fnc),
             Connection(fnc, add[1]), Connection(add, sco)],
            dt=0.05, iterations_max=2, log=False,
        )

    def test_nonconverging_loop_warns(self):
        """An algebraic loop that does not converge must emit a
        FastSimConvergenceWarning (visible AND catchable, regardless of log)."""
        sim = self._nonconverging_sim()
        with self.assertWarns(FastSimConvergenceWarning):
            sim.run(0.2)

    def test_runstats_reports_nonconvergence(self):
        """RunStats carries the structured outcome fields with truthful values."""
        sim = self._nonconverging_sim()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stats = sim.run(0.2)
        for key in ("converged", "max_residual", "truncated_at", "worst_block"):
            self.assertIn(key, stats)
        self.assertEqual(stats["converged"], 0.0)
        self.assertGreater(stats["max_residual"], 0.0)

    def test_runstats_clean_run_reports_converged(self):
        """A well-behaved run reports converged=1.0 and emits no warning."""
        cns = Constant(1.0)
        sco = Scope()
        sim = Simulation([cns, sco], [Connection(cns, sco)], dt=0.01, log=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FastSimConvergenceWarning)
            stats = sim.run(0.1)
        self.assertEqual(stats["converged"], 1.0)
        self.assertTrue(np.isnan(stats["truncated_at"]))


if __name__ == "__main__":
    unittest.main()
