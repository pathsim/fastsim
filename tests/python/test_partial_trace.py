########################################################################################
##
##              Independent per-operator JIT tracing (issue #17 / "trace everything")
##
##   _trace_dynamical_system no longer bails to a full Python-callback block when
##   ONE operator is untraceable: each of func_dyn / func_alg is wrapped in its
##   own LazyTraced, so a traceable op_dyn is JIT-compiled even if op_alg must
##   fall back to Python (and vice versa). Verified by correctness on a block
##   whose output branches on time (untraceable) while its dynamics trace.
##
########################################################################################

# IMPORTS ==============================================================================

import unittest

import numpy as np

from fastsim import Simulation, Connection
from fastsim.blocks import Scope, DynamicalSystem
from fastsim.solvers import RKDP54


# TESTCASES ============================================================================

class TestPartialTrace(unittest.TestCase):
    """dx/dt = -x, x(0)=1  ->  x(t)=exp(-t).
    Output y = x for t<1, else 2*x — the branch on t is NOT traceable, so op_alg
    falls back to Python while op_dyn is JIT-compiled.
    """

    def _build(self):
        return DynamicalSystem(
            func_dyn=lambda x, u, t: -x,                       # traceable
            func_alg=lambda x, u, t: x * (2.0 if t >= 1.0 else 1.0),  # branch on t -> not traceable
            initial_value=1.0,
        )

    def test_runs_correctly_with_partial_trace(self):
        ds = self._build()
        sco = Scope()
        sim = Simulation(
            blocks=[ds, sco],
            connections=[Connection(ds, sco)],
            Solver=RKDP54, tolerance_lte_abs=1e-9, tolerance_lte_rel=0.0,
            log=False,
        )
        sim.run(2.0)
        time, [y] = sco.read()
        time, y = np.asarray(time), np.asarray(y)

        # Piecewise reference: exp(-t) before t=1, 2*exp(-t) after.
        ref = np.where(time < 1.0, np.exp(-time), 2.0 * np.exp(-time))
        # Allow a couple of samples straddling the t=1 discontinuity to differ.
        err = np.abs(ref - y)
        self.assertLess(np.median(err), 1e-6)
        self.assertAlmostEqual(y[-1], 2.0 * np.exp(-2.0), places=5)

    def test_block_reports_jit_compiled(self):
        # op_dyn traces -> the block is built by the JIT path (partial), so the
        # block reports jit_compiled even though op_alg runs in Python.
        ds = self._build()
        self.assertTrue(ds.jit_compiled)


# RUN TESTS LOCALLY ====================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
