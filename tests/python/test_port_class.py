########################################################################################
##
##              Class-level porting of pathsim toolbox blocks (issue #17)
##
##   The toolbox workflow: `MyBlock = port(MyBlock)` once, then use instances
##   normally with fastsim Connections. Adaptable thin subclasses are rebased
##   onto fastsim (full speed); classes overriding engine hooks get a
##   shim-wrapper class. Either way `MyBlock(...)` yields a fastsim block.
##
########################################################################################

# IMPORTS ==============================================================================

import unittest

import numpy as np
import pytest

from fastsim import Simulation, Connection, port
from fastsim import _fastsim
from fastsim.blocks import Source, Scope
from fastsim.blocks import ODE as FsODE
from fastsim.solvers import RKDP54

pytestmark = pytest.mark.pathsim  # auto-skips when pathsim is not installed

_SOLVER_KW = dict(Solver=RKDP54, tolerance_lte_abs=1e-8, tolerance_lte_rel=0.0)


# A "toolbox" block: thin domain subclass that only computes parameters for the
# generic base in __init__ (the adaptable, fully-accelerable case).
def _make_toolbox_classes():
    from pathsim.blocks import DynamicalSystem as PsDynamicalSystem

    class Decay(PsDynamicalSystem):
        """Thin toolbox block: dx/dt = -a*x, y = x."""
        def __init__(self, a=1.0, x0=1.0):
            super().__init__(
                func_dyn=lambda x, u, t: -a * x,
                func_alg=lambda x, u, t: x,
                initial_value=x0,
            )

    class CustomDecay(PsDynamicalSystem):
        """Toolbox block that overrides an engine hook -> not adaptable."""
        def __init__(self, a=1.0, x0=1.0):
            super().__init__(
                func_dyn=lambda x, u, t: -a * x,
                func_alg=lambda x, u, t: x,
                initial_value=x0,
            )
        def step(self, t, dt):
            return super().step(t, dt)

    return Decay, CustomDecay


# A thin ALGEBRAIC toolbox block: subclasses the generic Function base (no
# state, no initial_value) and only forwards a computed func to super().__init__.
# This is the common toolbox shape (e.g. pathsim-chem thermodynamics blocks) and
# must rebase onto fastsim's Function — not fall into the state-only shim.
def _make_algebraic_toolbox_class():
    from pathsim.blocks import Function as PsFunction

    class Scale(PsFunction):
        """Thin algebraic toolbox block: y = gain * u."""
        def __init__(self, gain=2.0):
            self.gain = gain
            super().__init__(func=lambda u: gain * u)

    return Scale


# TESTCASES ============================================================================

class TestClassLevelPort(unittest.TestCase):

    def test_adaptable_class_accelerates(self):
        Decay, _ = _make_toolbox_classes()
        with self.assertLogs("fastsim.port", level="INFO") as cm:
            Ported = port(Decay)
        self.assertTrue(any("class accelerate" in m for m in cm.output))
        # Instances are real fastsim blocks (usable directly in Connection).
        d = Ported(a=2.0, x0=1.0)
        self.assertIsInstance(d, _fastsim.Block)

    def test_custom_hook_class_shim_wraps(self):
        _, CustomDecay = _make_toolbox_classes()
        with self.assertLogs("fastsim.port", level="INFO") as cm:
            Ported = port(CustomDecay)
        self.assertTrue(any("class shim-wrap" in m for m in cm.output))
        d = Ported(a=2.0, x0=1.0)
        self.assertIsInstance(d, _fastsim.Block)

    def test_algebraic_class_accelerates(self):
        # A thin Function subclass (algebraic, no state) must rebase via adapt,
        # NOT fall into the state-only shim (regression: issue #17 algebraic).
        Scale = _make_algebraic_toolbox_class()
        with self.assertLogs("fastsim.port", level="INFO") as cm:
            Ported = port(Scale)
        self.assertTrue(any("class accelerate" in m for m in cm.output))
        s = Ported(gain=3.0)
        self.assertIsInstance(s, _fastsim.Block)

    def test_fastsim_class_passthrough(self):
        # passthrough is a no-op -> logs at DEBUG, not INFO.
        with self.assertLogs("fastsim.port", level="DEBUG") as cm:
            out = port(FsODE)
        self.assertIs(out, FsODE)
        self.assertTrue(any("class passthrough" in m for m in cm.output))


class TestClassLevelEndToEnd(unittest.TestCase):
    """Ported toolbox classes run in a normal fastsim system (fastsim
    Connections) and reproduce dx/dt = -a*x  ->  x(t) = x0*exp(-a*t).
    """

    def _run(self, block):
        sco = Scope()
        sim = Simulation(
            blocks=[block, sco],
            connections=[Connection(block, sco)],
            log=False, **_SOLVER_KW,
        )
        sim.run(5.0)
        _, [y] = sco.read()
        return y[-1]

    def test_adaptable_end_to_end(self):
        Decay, _ = _make_toolbox_classes()
        Ported = port(Decay)
        last = self._run(Ported(a=2.0, x0=1.0))
        self.assertAlmostEqual(last, np.exp(-10.0), places=5)

    def test_custom_hook_end_to_end(self):
        _, CustomDecay = _make_toolbox_classes()
        Ported = port(CustomDecay)
        last = self._run(Ported(a=2.0, x0=1.0))
        self.assertAlmostEqual(last, np.exp(-10.0), places=5)

    def test_algebraic_end_to_end(self):
        # Ramp source -> ported algebraic block (y = gain*u) -> scope: y == gain*t.
        Scale = _make_algebraic_toolbox_class()
        Ported = port(Scale)
        src = Source(lambda t: t)
        blk = Ported(gain=3.0)
        sco = Scope()
        sim = Simulation(
            blocks=[src, blk, sco],
            connections=[Connection(src, blk), Connection(blk, sco)],
            log=False, **_SOLVER_KW,
        )
        sim.run(2.0)
        t, [y] = sco.read()
        self.assertTrue(np.allclose(y, 3.0 * np.asarray(t), atol=1e-9))


# RUN TESTS LOCALLY ====================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
