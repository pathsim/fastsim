########################################################################################
##
##              Phase-1 spike for issue #17: porting arbitrary pathsim blocks
##
##   Validates the engine-shim / RHS-capture bridge (fastsim.port.port):
##   a pathsim block instance, wrapped so that fastsim's own (Rust) engine
##   integrates it, must produce results bit-identical to the equivalent
##   fastsim-native block — across explicit, adaptive and implicit solvers.
##
##   Bit-parity is the decisive check: it proves fastsim owns the stage loop
##   and the wrapped block's RHS is the only thing flowing across the bridge.
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

import pytest

from fastsim import Simulation, Connection, port
from fastsim.blocks import Source, Scope
from fastsim.blocks import Integrator as FsIntegrator
from fastsim.blocks import DynamicalSystem as FsDynamicalSystem

from fastsim.solvers import SSPRK22, RKDP54, ESDIRK43

pytestmark = pytest.mark.pathsim  # auto-skips when pathsim is not installed


# HELPERS ==============================================================================

def _run_and_read(integ_block, solver, duration, *, source=None, **solver_kwargs):
    """Build src -> block -> scope, run, return (time, output) arrays."""
    src = Source(source if source is not None else (lambda t: np.sin(t)))
    sco = Scope()
    sim = Simulation(
        blocks=[src, integ_block, sco],
        connections=[Connection(src, integ_block), Connection(integ_block, sco)],
        Solver=solver,
        log=False,
        **solver_kwargs,
    )
    sim.run(duration)
    time, [out] = sco.read()
    return np.asarray(time), np.asarray(out)


# TESTCASES ============================================================================

class TestPortIntegratorParity(unittest.TestCase):
    """A ported pathsim Integrator must match the fastsim-native Integrator
    bit-for-bit, since both reduce to dx/dt = u integrated by the same engine.
    """

    def _native(self):
        return FsIntegrator(0.0)

    def _ported(self):
        from pathsim.blocks import Integrator as PsIntegrator
        return port(PsIntegrator(0.0))

    def test_explicit_fixed_step(self):
        tn, yn = _run_and_read(self._native(), SSPRK22, 10.0)
        tp, yp = _run_and_read(self._ported(), SSPRK22, 10.0)
        np.testing.assert_array_equal(tn, tp)
        np.testing.assert_array_equal(yn, yp)

    def test_adaptive_explicit(self):
        # Adaptive stepping exercises the stage loop + error-driven dt control;
        # identical RHS => identical error estimates => identical timesteps.
        kw = dict(tolerance_lte_abs=1e-7, tolerance_lte_rel=0.0)
        tn, yn = _run_and_read(self._native(), RKDP54, 10.0, **kw)
        tp, yp = _run_and_read(self._ported(), RKDP54, 10.0, **kw)
        np.testing.assert_array_equal(tn, tp)
        np.testing.assert_array_equal(yn, yp)

    def test_implicit(self):
        # Implicit solver routes the RHS through engine.solve(f, J, dt), the
        # shim's second capture point.
        kw = dict(tolerance_lte_abs=1e-7, tolerance_lte_rel=0.0)
        tn, yn = _run_and_read(self._native(), ESDIRK43, 10.0, **kw)
        tp, yp = _run_and_read(self._ported(), ESDIRK43, 10.0, **kw)
        np.testing.assert_array_equal(tn, tp)
        np.testing.assert_array_equal(yn, yp)


class TestPortDynamicalSystemParity(unittest.TestCase):
    """A ported pathsim DynamicalSystem (with op_dyn/op_alg) must match the
    fastsim-native DynamicalSystem. Exercises the op_dyn capture path and the
    func_alg output bridge (y = f_alg(x, u, t)).

    System: dx/dt = -a*x + u, y = 2*x, x(0) = x0.
    """

    a = 2.0
    x0 = 1.5

    def _native(self):
        return FsDynamicalSystem(
            func_dyn=lambda x, u, t: -self.a * x + u,
            func_alg=lambda x, u, t: 2.0 * x,
            initial_value=self.x0,
        )

    def _ported(self):
        from pathsim.blocks import DynamicalSystem as PsDynamicalSystem
        return port(PsDynamicalSystem(
            func_dyn=lambda x, u, t: -self.a * x + u,
            func_alg=lambda x, u, t: 2.0 * x,
            initial_value=self.x0,
        ))

    def test_explicit_fixed_step(self):
        tn, yn = _run_and_read(self._native(), SSPRK22, 8.0)
        tp, yp = _run_and_read(self._ported(), SSPRK22, 8.0)
        np.testing.assert_array_equal(tn, tp)
        np.testing.assert_array_equal(yn, yp)

    def test_adaptive_explicit(self):
        kw = dict(tolerance_lte_abs=1e-7, tolerance_lte_rel=0.0)
        tn, yn = _run_and_read(self._native(), RKDP54, 8.0, **kw)
        tp, yp = _run_and_read(self._ported(), RKDP54, 8.0, **kw)
        np.testing.assert_array_equal(tn, tp)
        np.testing.assert_array_equal(yn, yp)


# RUN TESTS LOCALLY ====================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
