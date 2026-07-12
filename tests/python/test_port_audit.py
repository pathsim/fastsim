########################################################################################
##
##              Regression tests for the port() audit fixes (issue #17)
##
##   F1: the accelerate (Tier-1) path must not mutate the input pathsim block.
##   F2: the shim warns (not silently zero) when a block's step() does not route
##       its RHS through engine.step/solve.
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


# TESTCASES ============================================================================

class TestNoInputMutation(unittest.TestCase):
    """F1: porting via the accelerate path must leave the source block pristine."""

    def test_accelerate_does_not_mutate_engine(self):
        from pathsim.blocks import DynamicalSystem as PsDynamicalSystem
        ds = PsDynamicalSystem(
            func_dyn=lambda x, u, t: -x,
            func_alg=lambda x, u, t: x,
            initial_value=1.0,
        )
        self.assertIsNone(ds.engine)
        port(ds)  # Tier 1 — only inspects the block to extract operators
        self.assertIsNone(ds.engine, "accelerate path must not leave a shim engine")

    def test_ode_accelerate_does_not_mutate_engine(self):
        from pathsim.blocks import ODE as PsODE
        ode = PsODE(func=lambda x, u, t: -x, initial_value=1.0)
        self.assertIsNone(ode.engine)
        port(ode)
        self.assertIsNone(ode.engine)


class TestShimContractWarning(unittest.TestCase):
    """F2: a block whose step() ignores the engine -> shim captures nothing -> warn."""

    def test_warns_when_step_skips_engine(self):
        from pathsim.blocks import DynamicalSystem as PsDynamicalSystem

        class BadStep(PsDynamicalSystem):
            def __init__(self):
                super().__init__(
                    func_dyn=lambda x, u, t: -x,
                    func_alg=lambda x, u, t: x,
                    initial_value=1.0,
                )
            def step(self, t, dt):
                # Never calls self.engine.step/solve -> shim._f stays None.
                return True, 0.0, None

        block = port(BadStep())  # overrides step -> shim path
        sco = Scope()
        with self.assertLogs("fastsim.port", level="WARNING") as cm:
            sim = Simulation(
                blocks=[block, sco], connections=[Connection(block, sco)],
                Solver=SSPRK22, dt=0.01, log=False,
            )
            sim.run(0.1)
        self.assertTrue(any("did not call engine.step" in m for m in cm.output))


# RUN TESTS LOCALLY ====================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
