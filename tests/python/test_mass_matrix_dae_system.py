"""Tests for MassMatrixDAE block — verifies that the constant-mass-matrix
DAE form integrates correctly via the MassMatrixStageBuilder."""

import unittest

import numpy as np

from fastsim import Simulation
from fastsim.blocks import MassMatrixDAE
from fastsim.solvers import ESDIRK43, ESDIRK54


class TestMassMatrixIdentity(unittest.TestCase):
    """M = I must give the same trajectory as a plain ODE."""

    def test_identity_recovers_exp_decay(self):
        M = np.eye(1)
        dae = MassMatrixDAE(
            func=lambda x, u, t: np.array([-x[0]]),
            mass=M,
            initial_value=[1.0],
        )
        sim = Simulation(blocks=[dae], connections=[])
        sim._set_solver(ESDIRK43, tolerance_lte_abs=1e-10, tolerance_lte_rel=1e-10)
        sim.run(1.0)

        x_final = dae.state[0]
        self.assertAlmostEqual(x_final, np.exp(-1.0), places=5)


class TestMassMatrixDiagonal(unittest.TestCase):
    """Diagonal M rescales the effective dynamics: M·ẋ = -x → ẋ = -M⁻¹·x."""

    def test_diagonal_scales_dynamics(self):
        M = np.array([[2.0]])
        dae = MassMatrixDAE(
            func=lambda x, u, t: np.array([-x[0]]),
            mass=M,
            initial_value=[1.0],
        )
        sim = Simulation(blocks=[dae], connections=[])
        sim._set_solver(ESDIRK43, tolerance_lte_abs=1e-10, tolerance_lte_rel=1e-10)
        sim.run(1.0)

        # M = 2 means ẋ = -x/2 → x(1) = exp(-0.5)
        self.assertAlmostEqual(dae.state[0], np.exp(-0.5), places=5)


class TestIndex1DAE(unittest.TestCase):
    """Index-1 DAE with algebraic constraint: singular M pins z = x."""

    def test_singular_mass_holds_constraint(self):
        # ẋ = -x + z,  0 = x - z  → z = x, ẋ = 0 → stationary
        M = np.array([[1.0, 0.0],
                      [0.0, 0.0]])

        def f(x, u, t):
            return np.array([-x[0] + x[1], x[0] - x[1]])

        dae = MassMatrixDAE(func=f, mass=M, initial_value=[1.0, 1.0])
        sim = Simulation(blocks=[dae], connections=[])
        sim._set_solver(ESDIRK43, tolerance_lte_abs=1e-8, tolerance_lte_rel=1e-8)
        sim.run(1.0)

        x, z = dae.state
        self.assertAlmostEqual(x, 1.0, places=4)
        self.assertAlmostEqual(z, 1.0, places=4)
        self.assertAlmostEqual(abs(x - z), 0.0, places=4)

    def test_index1_driven_constraint(self):
        """Driven DAE: ẋ = -x + z,  0 = z - cos(t).
        Constraint forces z(t) = cos(t).  ODE for x: ẋ = -x + cos(t).
        Analytical: x(t) = 0.5·(sin(t) + cos(t)) + C·exp(-t), C chosen so x(0)=0.5.
        """
        M = np.array([[1.0, 0.0],
                      [0.0, 0.0]])

        def f(x, u, t):
            return np.array([-x[0] + x[1], x[1] - np.cos(t)])

        # x(0) = 0.5, z(0) = cos(0) = 1.0
        dae = MassMatrixDAE(func=f, mass=M, initial_value=[0.5, 1.0])
        sim = Simulation(blocks=[dae], connections=[])
        sim._set_solver(ESDIRK54, tolerance_lte_abs=1e-8, tolerance_lte_rel=1e-8)
        sim.run(np.pi)  # t = π

        x, z = dae.state
        # At t = π: z = cos(π) = -1
        self.assertAlmostEqual(z, -1.0, places=4)
        # Analytic x(t) = 0.5·(sin(t) + cos(t)) + 0.5·exp(-t)·[x(0) - 0.5]
        # With x(0)=0.5, the constant drops out → x(t) = 0.5·(sin + cos)
        expected_x = 0.5 * (np.sin(np.pi) + np.cos(np.pi))
        self.assertAlmostEqual(x, expected_x, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
