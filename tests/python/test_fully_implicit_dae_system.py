"""Tests for FullyImplicitDAE block — F(x, ẋ, u, t) = 0."""

import unittest

import numpy as np

from fastsim import Simulation
from fastsim.blocks import FullyImplicitDAE
from fastsim.solvers import DIRK3, ESDIRK43


class TestFullyImplicitScalar(unittest.TestCase):

    def test_exp_decay(self):
        # F = ẋ + x = 0  →  ẋ = -x.
        dae = FullyImplicitDAE(
            func=lambda x, xdot, u, t: np.array([xdot[0] + x[0]]),
            initial_value=[1.0],
        )
        sim = Simulation(blocks=[dae], connections=[])
        sim._set_solver(ESDIRK43, tolerance_lte_abs=1e-10, tolerance_lte_rel=1e-10)
        sim.run(1.0)
        self.assertAlmostEqual(dae.state[0], np.exp(-1.0), places=5)

    def test_mass_scaled(self):
        # F = 2·ẋ + x = 0  →  ẋ = -x/2  →  x(1) = exp(-0.5).
        dae = FullyImplicitDAE(
            func=lambda x, xdot, u, t: np.array([2.0 * xdot[0] + x[0]]),
            initial_value=[1.0],
        )
        sim = Simulation(blocks=[dae], connections=[])
        sim._set_solver(ESDIRK43, tolerance_lte_abs=1e-10, tolerance_lte_rel=1e-10)
        sim.run(1.0)
        self.assertAlmostEqual(dae.state[0], np.exp(-0.5), places=5)


class TestFullyImplicitIndex1(unittest.TestCase):
    """Index-1 DAEs need DIRK (no explicit first stage)."""

    def test_pinned_constraint(self):
        # F1 = ẋ + x - z = 0  →  ẋ = -x + z
        # F2 = x - z = 0       (algebraic, no ẋ-dependence → singular ∂F/∂ẋ)
        # Consistent init (1, 1) → stays at (1, 1).
        def f(x, xdot, u, t):
            return np.array([xdot[0] + x[0] - x[1], x[0] - x[1]])

        dae = FullyImplicitDAE(
            func=f,
            initial_value=[1.0, 1.0],
            jac_x=lambda x, xdot, u, t: np.array([[1.0, -1.0], [1.0, -1.0]]),
            jac_xdot=lambda x, xdot, u, t: np.array([[1.0, 0.0], [0.0, 0.0]]),
        )
        sim = Simulation(blocks=[dae], connections=[])
        sim._set_solver(DIRK3)
        sim.run(1.0)

        x, z = dae.state
        self.assertAlmostEqual(x, 1.0, places=3)
        self.assertAlmostEqual(z, 1.0, places=3)
        self.assertAlmostEqual(abs(x - z), 0.0, places=3)


class TestFullyImplicitNumericalJacobian(unittest.TestCase):
    """Verify the central-difference Jacobian fallback works."""

    def test_no_analytical_jac(self):
        # Same as exp_decay but without supplying Jacobians.
        dae = FullyImplicitDAE(
            func=lambda x, xdot, u, t: np.array([xdot[0] + x[0]]),
            initial_value=[1.0],
            # jac_x, jac_xdot omitted → central differences
        )
        sim = Simulation(blocks=[dae], connections=[])
        sim._set_solver(ESDIRK43, tolerance_lte_abs=1e-8, tolerance_lte_rel=1e-8)
        sim.run(1.0)
        self.assertAlmostEqual(dae.state[0], np.exp(-1.0), places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
