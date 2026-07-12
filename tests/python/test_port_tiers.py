########################################################################################
##
##              Tiered classification of fastsim.port.port (issue #17)
##
##   Verifies port() picks the right strategy per block and logs the decision:
##     Tier 0  fastsim block        -> passthrough (identity)
##     Tier 1  clean op_dyn block   -> accelerate (operator extraction + JIT)
##     Tier 3  custom / no operator -> engine-shim fallback
##
########################################################################################

# IMPORTS ==============================================================================

import logging
import unittest

import numpy as np
import pytest

from fastsim import Simulation, Connection, port
from fastsim.blocks import Source, Scope
from fastsim.blocks import Integrator as FsIntegrator
from fastsim.blocks import ODE as FsODE
from fastsim.solvers import RKDP54

pytestmark = pytest.mark.pathsim  # auto-skips when pathsim is not installed


# TESTCASES ============================================================================

class TestTierClassification(unittest.TestCase):

    def test_tier0_fastsim_passthrough(self):
        blk = FsIntegrator(0.0)
        # passthrough is a no-op -> logs at DEBUG (not INFO) to keep native
        # systems quiet; only real porting logs at INFO.
        with self.assertLogs("fastsim.port", level="DEBUG") as cm:
            out = port(blk)
        self.assertIs(out, blk, "fastsim block must pass through unchanged")
        self.assertTrue(any("passthrough" in m for m in cm.output))

    def test_tier1_dynamicalsystem_accelerates(self):
        from pathsim.blocks import DynamicalSystem as PsDynamicalSystem
        blk = PsDynamicalSystem(
            func_dyn=lambda x, u, t: -2.0 * x + u,
            func_alg=lambda x, u, t: x,
            initial_value=1.0,
        )
        with self.assertLogs("fastsim.port", level="INFO") as cm:
            out = port(blk)
        self.assertTrue(any("accelerate" in m for m in cm.output))
        # The decay RHS is JIT-traceable -> full Rust acceleration, no shim.
        self.assertTrue(getattr(out, "jit_compiled", False))

    def test_tier1_ode_accelerates(self):
        from pathsim.blocks import ODE as PsODE
        blk = PsODE(func=lambda x, u, t: -x, initial_value=1.0)
        with self.assertLogs("fastsim.port", level="INFO") as cm:
            out = port(blk)
        self.assertTrue(any("accelerate" in m for m in cm.output))
        self.assertTrue(getattr(out, "jit_compiled", False))

    def test_tier3_integrator_shim(self):
        # Integrator overrides step/update and exposes no op_dyn -> shim fallback.
        from pathsim.blocks import Integrator as PsIntegrator
        with self.assertLogs("fastsim.port", level="INFO") as cm:
            port(PsIntegrator(0.0))
        self.assertTrue(any("shim fallback" in m for m in cm.output))
        self.assertTrue(any("no op_dyn" in m for m in cm.output))

    def test_tier3_custom_override_shim(self):
        # A subclass overriding an engine hook must NOT be accelerated, even
        # though it inherits op_dyn — its custom step could diverge.
        from pathsim.blocks import ODE as PsODE

        class CustomODE(PsODE):
            def step(self, t, dt):
                return super().step(t, dt)

        with self.assertLogs("fastsim.port", level="INFO") as cm:
            port(CustomODE(func=lambda x, u, t: -x, initial_value=1.0))
        self.assertTrue(any("shim fallback" in m for m in cm.output))
        self.assertTrue(any("overrides step" in m for m in cm.output))


class TestTier1ODEParity(unittest.TestCase):
    """A Tier-1 ported pathsim ODE must match the fastsim-native ODE.

    System: dx/dt = -x + sin(t), x(0) = 0.
    """

    def _run(self, block):
        src = Source(lambda t: np.sin(t))
        sco = Scope()
        sim = Simulation(
            blocks=[src, block, sco],
            connections=[Connection(src, block), Connection(block, sco)],
            Solver=RKDP54, tolerance_lte_abs=1e-7, tolerance_lte_rel=0.0,
            log=False,
        )
        sim.run(10.0)
        t, [y] = sco.read()
        return np.asarray(t), np.asarray(y)

    def test_parity(self):
        from pathsim.blocks import ODE as PsODE
        tn, yn = self._run(FsODE(func=lambda x, u, t: -x + u, initial_value=0.0))
        tp, yp = self._run(port(PsODE(func=lambda x, u, t: -x + u, initial_value=0.0)))
        np.testing.assert_array_equal(tn, tp)
        np.testing.assert_allclose(yn, yp, rtol=0.0, atol=1e-12)


# RUN TESTS LOCALLY ====================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main(verbosity=2)
