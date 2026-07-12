"""Tests for SemiExplicitDAE — inner-Newton reduction to a pure ODE in x.

The block's `.state` exposes only the differential state `x`; the converged
algebraic values `z` appear on the block's output (ports `n_x .. n_x+n_z`).
Tests exercise both explicit (RKDP54) and implicit (ESDIRK43/54) solvers
since the reduction makes any solver usable.
"""

import unittest

import numpy as np

from fastsim import Simulation
from fastsim.blocks import SemiExplicitDAE
from fastsim.solvers import ESDIRK43, ESDIRK54, RKDP54


class TestSemiExplicitExplicitSolver(unittest.TestCase):
    """Index-1 DAE integrated by an explicit RK solver — only possible
    because the reduced form eliminates z from the outer problem."""

    def test_pinned_constraint_rkdp54(self):
        # ẋ = -x + z,  0 = x - z  →  z = x,  ẋ = 0  →  stationary at x=1.
        dae = SemiExplicitDAE(
            f_dyn=lambda x, z, u, t: np.array([-x[0] + z[0]]),
            f_alg=lambda x, z, u, t: np.array([x[0] - z[0]]),
            x0=[1.0],
            z0=[1.0],
        )
        sim = Simulation(blocks=[dae], connections=[])
        sim._set_solver(RKDP54, tolerance_lte_abs=1e-8, tolerance_lte_rel=1e-8)
        sim.run(1.0)

        x = dae.state[0]
        self.assertAlmostEqual(x, 1.0, places=4)

    def test_driven_constraint_rkdp54(self):
        # ẋ = z,  0 = z - sin(t)  →  z(t) = sin(t),  x(t) = x0 + 1 - cos(t).
        dae = SemiExplicitDAE(
            f_dyn=lambda x, z, u, t: np.array([z[0]]),
            f_alg=lambda x, z, u, t: np.array([z[0] - np.sin(t)]),
            x0=[0.0],
            z0=[0.0],
            jac_z=lambda x, z, u, t: np.array([[1.0]]),
        )
        sim = Simulation(blocks=[dae], connections=[])
        sim._set_solver(RKDP54, tolerance_lte_abs=1e-10, tolerance_lte_rel=1e-10)
        sim.run(np.pi)
        self.assertAlmostEqual(dae.state[0], 2.0, places=4)


class TestSemiExplicitImplicitSolver(unittest.TestCase):
    """Same system, implicit solver — must give the same trajectory."""

    def test_pinned_constraint_esdirk43(self):
        dae = SemiExplicitDAE(
            f_dyn=lambda x, z, u, t: np.array([-x[0] + z[0]]),
            f_alg=lambda x, z, u, t: np.array([x[0] - z[0]]),
            x0=[1.0],
            z0=[1.0],
        )
        sim = Simulation(blocks=[dae], connections=[])
        sim._set_solver(ESDIRK43, tolerance_lte_abs=1e-8, tolerance_lte_rel=1e-8)
        sim.run(1.0)
        self.assertAlmostEqual(dae.state[0], 1.0, places=4)

    def test_driven_constraint_esdirk54(self):
        dae = SemiExplicitDAE(
            f_dyn=lambda x, z, u, t: np.array([z[0]]),
            f_alg=lambda x, z, u, t: np.array([z[0] - np.sin(t)]),
            x0=[0.0],
            z0=[0.0],
        )
        sim = Simulation(blocks=[dae], connections=[])
        sim._set_solver(ESDIRK54, tolerance_lte_abs=1e-9, tolerance_lte_rel=1e-9)
        sim.run(np.pi)
        self.assertAlmostEqual(dae.state[0], 2.0, places=3)


class TestSemiExplicitMultiState(unittest.TestCase):
    """2 differential + 2 algebraic states."""

    def test_two_dyn_two_alg(self):
        # ẋ1 = -x1 + z1,  ẋ2 = -2·x2 + z2
        # 0 = z1 - 1.0,   0 = z2 - 2.0     (constants)
        # → ẋ1 = -x1 + 1, ẋ2 = -2·x2 + 2   → steady state x1 = 1, x2 = 1.
        dae = SemiExplicitDAE(
            f_dyn=lambda x, z, u, t: np.array([-x[0] + z[0], -2 * x[1] + z[1]]),
            f_alg=lambda x, z, u, t: np.array([z[0] - 1.0, z[1] - 2.0]),
            x0=[0.0, 0.0],
            z0=[1.0, 2.0],
        )
        sim = Simulation(blocks=[dae], connections=[])
        sim._set_solver(RKDP54, tolerance_lte_abs=1e-8, tolerance_lte_rel=1e-8)
        sim.run(5.0)

        # Analytic at T=5: x1 = 1 - exp(-5),  x2 = 1 - exp(-10).
        self.assertAlmostEqual(dae.state[0], 1.0 - np.exp(-5.0), places=4)
        self.assertAlmostEqual(dae.state[1], 1.0 - np.exp(-10.0), places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
