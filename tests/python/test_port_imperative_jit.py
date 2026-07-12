########################################################################################
##
##              Imperative-idiom RHS now JIT-accelerates through port() (issue #17)
##
##   Many real toolbox ODE/DynamicalSystem blocks build their derivative with the
##   imperative idiom `dx = np.zeros(n); dx[i] = ...; return dx`. The JIT tracer
##   now supports it (monkeypatched array constructors + __setitem__), so such a
##   ported block reaches Tier 1 (full Rust speed) instead of the Python shim.
##
########################################################################################

# IMPORTS ==============================================================================

import unittest

import numpy as np
import pytest

from fastsim import Simulation, Connection, port
from fastsim.blocks import Scope
from fastsim.blocks import ODE as FsODE
from fastsim.solvers import RKCK54

pytestmark = pytest.mark.pathsim  # auto-skips when pathsim is not installed


def _imperative_rhs(x, u, t):
    # Harmonic oscillator: dx0 = x1, dx1 = -x0  (built imperatively)
    dx = np.zeros(2)
    dx[0] = x[1]
    dx[1] = -x[0]
    return dx


def _functional_rhs(x, u, t):
    return np.array([x[1], -x[0]])


# TESTCASES ============================================================================

class TestImperativePortJIT(unittest.TestCase):

    def test_ported_imperative_block_is_jit(self):
        from pathsim.blocks import ODE as PsODE
        ported = port(PsODE(func=_imperative_rhs, initial_value=[1.0, 0.0]))
        # Tier 1 acceleration with a *successful* JIT trace (not a Python fallback).
        self.assertTrue(getattr(ported, "jit_compiled", False))

    def test_parity_imperative_vs_functional(self):
        from pathsim.blocks import ODE as PsODE

        def run(block):
            sco = Scope()
            sim = Simulation(
                blocks=[block, sco], connections=[Connection(block, sco)],
                Solver=RKCK54, tolerance_lte_abs=1e-9, tolerance_lte_rel=0.0,
                log=False,
            )
            sim.run(6.0)
            t, [y] = sco.read()
            return np.asarray(t), np.asarray(y)

        ti, yi = run(port(PsODE(func=_imperative_rhs, initial_value=[1.0, 0.0])))
        tf, yf = run(FsODE(func=_functional_rhs, initial_value=[1.0, 0.0]))
        np.testing.assert_array_equal(ti, tf)
        np.testing.assert_allclose(yi, yf, rtol=0.0, atol=1e-12)


# RUN TESTS LOCALLY ====================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
