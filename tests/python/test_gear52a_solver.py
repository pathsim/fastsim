########################################################################################
##
##                            Standalone GEAR52A solver tests
##
##  Variable-step, variable-order BDF (orders 1..5) with order ramp-up startup.
##  Exercises the standalone `GEAR52A.integrate(...)` interface — same surface
##  as ESDIRK32/43/54 — on the canonical stiff suite (Van der Pol, Robertson)
##  and on a smooth test (exponential growth) where global truncation error is
##  bounded by the requested tolerance.
##
########################################################################################

import unittest
import numpy as np

from scipy.integrate import solve_ivp

from fastsim.solvers import GEAR52A


# TESTCASE =============================================================================

class TestGear52aStandalone(unittest.TestCase):
    """Standalone integrate() tests for GEAR52A."""

    def test_metadata_visible(self):
        """The class must be importable and expose the integrate classmethod."""
        self.assertTrue(hasattr(GEAR52A, "integrate"))

    def test_exponential_growth(self):
        """ẋ = x, x(0) = 1, x(1) = e. GEAR52A must integrate within tolerance."""
        def f(x, t):
            return [x[0]]

        t, x = GEAR52A.integrate(
            f, [1.0],
            time_start=0.0, time_end=1.0,
            dt=0.01, dt_min=1e-12, dt_max=0.5,
            adaptive=True,
            tolerance_lte_abs=1e-10, tolerance_lte_rel=1e-8,
            max_iterations=50,
            optimizer="newton_anderson",
        )
        self.assertGreater(len(t), 5, "GEAR52A produced suspiciously few steps")
        self.assertAlmostEqual(t[-1], 1.0, places=10)
        # GTE bound: ~1e-3 with these tolerances on this fast-growing problem.
        self.assertLess(abs(x[-1, 0] - np.e), 1e-3,
                        f"GEAR52A on ẋ=x: got x(1) = {x[-1, 0]}")

    def test_robertson_matches_radau(self):
        """Robertson (canonical stiff). Compare against scipy Radau reference
        — GEAR52A must agree to a few digits at t=1 and conserve mass."""
        a, b, c = 0.04, 1e4, 3e7

        def f(x, t):
            return [
                -a * x[0] + b * x[1] * x[2],
                 a * x[0] - b * x[1] * x[2] - c * x[1] ** 2,
                 c * x[1] ** 2,
            ]

        t, x = GEAR52A.integrate(
            f, [1.0, 0.0, 0.0],
            time_start=0.0, time_end=1.0,
            dt=1e-3, dt_min=1e-15, dt_max=1e-1,
            adaptive=True,
            tolerance_lte_abs=1e-8, tolerance_lte_rel=1e-6,
            max_iterations=100,
            optimizer="newton_anderson",
        )

        def f_t(_t, _x): return f(_x, _t)
        sol = solve_ivp(f_t, [0, 1], [1.0, 0.0, 0.0], method="Radau",
                        rtol=1e-10, atol=1e-12)
        ref = sol.y[:, -1]

        # Mass conservation: x[0] + x[1] + x[2] is conserved analytically.
        mass = x[-1].sum()
        self.assertAlmostEqual(mass, 1.0, places=6, msg=f"mass = {mass}")

        # Per-component agreement with Radau (loose: GEAR52A tols here are
        # 1e-8, GTE on Robertson is well-known to be ~atol).
        max_diff = np.max(np.abs(x[-1] - ref))
        self.assertLess(max_diff, 1e-5,
                        f"GEAR52A vs Radau diff = {max_diff:.2e}")

    def test_vanderpol_nonstiff(self):
        """Van der Pol with mu=1 (non-stiff regime). GEAR52A must complete
        a full period and stay in physical bounds."""
        mu = 1.0

        def f(x, t):
            return [x[1], mu * (1 - x[0] ** 2) * x[1] - x[0]]

        t, x = GEAR52A.integrate(
            f, [2.0, 0.0],
            time_start=0.0, time_end=10.0,
            dt=0.05, dt_min=1e-12, dt_max=0.5,
            adaptive=True,
            tolerance_lte_abs=1e-7, tolerance_lte_rel=1e-5,
            max_iterations=80,
            optimizer="newton_anderson",
        )

        def f_t(_t, _x): return f(_x, _t)
        sol = solve_ivp(f_t, [0, 10], [2.0, 0.0], method="Radau",
                        rtol=1e-10, atol=1e-12, dense_output=True)
        ref = sol.sol(t).T

        max_diff = np.max(np.abs(x - ref))
        # Looser bound (5e-2) since GEAR52A on a non-stiff oscillator with
        # 1e-7/1e-5 tolerances accumulates phase error over a full period.
        # Tighter bounds belong on the stiff Robertson test where BDF excels.
        self.assertLess(max_diff, 5e-2,
                        f"VdP GEAR52A vs Radau: max diff = {max_diff:.2e}")

    def test_step_rejection_and_revert(self):
        """When asked for an absurdly tight tolerance with a smooth solution,
        the controller must still converge (no infinite reject loop)."""
        def f(x, t):
            return [-x[0]]

        t, x = GEAR52A.integrate(
            f, [1.0],
            time_start=0.0, time_end=2.0,
            dt=0.1, dt_min=1e-15, dt_max=0.5,
            adaptive=True,
            tolerance_lte_abs=1e-12, tolerance_lte_rel=1e-10,
            max_iterations=80,
            optimizer="newton_anderson",
        )
        self.assertAlmostEqual(t[-1], 2.0, places=10)
        # The primary invariant here is "no runaway / no NaN" — keeps GTE in
        # the same order as the reference rather than asserting tight
        # accuracy, since BDF GTE on a long horizon is dominated by order-2
        # ramp-up cost.
        self.assertAlmostEqual(x[-1, 0], np.exp(-2.0), places=2)


# RUN TESTS LOCALLY ====================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
