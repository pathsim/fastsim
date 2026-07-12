########################################################################################
##
##              Auto-porting Simulation facade (issue #17)
##
##   The fastsim.Simulation facade runs every block in its `blocks` list through
##   port(): fastsim blocks pass through, pathsim blocks are accelerated or
##   shimmed. These tests drive autonomous blocks (no input connection needed),
##   so a pathsim block can be handed to the facade un-ported and the facade
##   does the porting end-to-end — verified by behaviour (state decays correctly)
##   and by the logged decision.
##
########################################################################################

# IMPORTS ==============================================================================

import unittest

import numpy as np
import pytest

from fastsim import Simulation, Connection
from fastsim.blocks import Source, Scope
from fastsim.blocks import ODE as FsODE
from fastsim.solvers import RKDP54

pytestmark = pytest.mark.pathsim  # auto-skips when pathsim is not installed

_SOLVER_KW = dict(Solver=RKDP54, tolerance_lte_abs=1e-8, tolerance_lte_rel=0.0)


# TESTCASES ============================================================================

class TestAutoPortEndToEnd(unittest.TestCase):
    """Autonomous decay dx/dt = -x, x(0) = 1  ->  x(5) = exp(-5)."""

    decay_ref = np.exp(-5.0)

    def _state_after_run(self, sim):
        sim.run(5.0)
        return sim.blocks[0].state[0]

    def test_autoport_accelerate(self):
        from pathsim.blocks import ODE as PsODE
        block = PsODE(func=lambda x, u, t: -x, initial_value=1.0)
        with self.assertLogs("fastsim.port", level="INFO") as cm:
            sim = Simulation(blocks=[block], **_SOLVER_KW)  # log defaults True
        self.assertTrue(any("accelerate" in m for m in cm.output))
        self.assertAlmostEqual(self._state_after_run(sim), self.decay_ref, places=6)

    def test_autoport_shim(self):
        from pathsim.blocks import ODE as PsODE

        class CustomODE(PsODE):
            # Overriding an engine hook forces the Tier-3 shim path.
            def step(self, t, dt):
                return super().step(t, dt)

        block = CustomODE(func=lambda x, u, t: -x, initial_value=1.0)
        with self.assertLogs("fastsim.port", level="INFO") as cm:
            sim = Simulation(blocks=[block], **_SOLVER_KW)
        self.assertTrue(any("shim fallback" in m for m in cm.output))
        self.assertAlmostEqual(self._state_after_run(sim), self.decay_ref, places=6)

    def test_fastsim_block_passthrough(self):
        # Verify behaviour (a fastsim block passes through the facade and runs
        # correctly). The passthrough log is DEBUG now and the facade raises the
        # port logger to INFO, so the message isn't observable here; the log
        # itself is covered by the direct-port() tests in test_port_tiers/class.
        block = FsODE(func=lambda x, u, t: -x, initial_value=1.0)
        sim = Simulation(blocks=[block], **_SOLVER_KW)
        self.assertAlmostEqual(self._state_after_run(sim), self.decay_ref, places=6)

    def test_mixed_fastsim_and_pathsim(self):
        from pathsim.blocks import ODE as PsODE
        fs = FsODE(func=lambda x, u, t: -x, initial_value=1.0)
        ps = PsODE(func=lambda x, u, t: -2.0 * x, initial_value=1.0)
        sim = Simulation(blocks=[fs, ps], log=False, **_SOLVER_KW)
        sim.run(5.0)
        blocks = sim.blocks
        self.assertAlmostEqual(blocks[0].state[0], np.exp(-5.0), places=6)
        self.assertAlmostEqual(blocks[1].state[0], np.exp(-10.0), places=6)


class TestFacadeTransparency(unittest.TestCase):
    """A port-first system (block pre-ported, then connected) runs through the
    facade and reproduces the analytical solution. Confirms the facade keeps the
    full Simulation API transparent.

    System: dx/dt = -x + sin(t), x(0) = 0
    Analytic: x(t) = 0.5*(sin(t) - cos(t)) + 0.5*exp(-t)
    """

    def test_port_first_matches_analytic(self):
        from fastsim import port
        from pathsim.blocks import ODE as PsODE

        ode = port(PsODE(func=lambda x, u, t: -x + u, initial_value=0.0))
        src = Source(lambda t: np.sin(t))
        sco = Scope()
        sim = Simulation(
            blocks=[src, ode, sco],
            connections=[Connection(src, ode), Connection(ode, sco)],
            log=False, **_SOLVER_KW,
        )
        sim.run(15.0)
        time, [y] = sco.read()
        ref = 0.5 * (np.sin(time) - np.cos(time)) + 0.5 * np.exp(-time)
        self.assertLess(np.max(np.abs(ref - y)), 1e-5)

    def test_contains_delegates(self):
        # __contains__ must forward to the inner Rust simulation.
        block = FsODE(func=lambda x, u, t: -x, initial_value=1.0)
        sim = Simulation(blocks=[block], log=False)
        self.assertIn(block, sim)


# RUN TESTS LOCALLY ====================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
