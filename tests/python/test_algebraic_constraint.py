"""Tests for the native AlgebraicConstraint block — solve F(x, u) = 0 for x each
evaluation (warmstarted Newton, AD Jacobian, dynamic inputs).
"""

import unittest

import numpy as np

from fastsim import Simulation, Connection
from fastsim.blocks import AlgebraicConstraint, Scope, Constant


def _solved(block):
    return np.asarray(block.outputs, dtype=float).reshape(-1)


class TestAlgebraicConstraint(unittest.TestCase):

    def test_scalar_root(self):
        # x^2 - 2 = 0  ->  x = sqrt(2)
        ac = AlgebraicConstraint(lambda x, u: np.array([x[0] ** 2 - 2.0]), [1.0])
        Simulation([ac, Scope()], [Connection(ac, Scope())], log=False).run(0.005)
        self.assertAlmostEqual(_solved(ac)[0], np.sqrt(2.0), places=10)

    def test_vector_nonlinear(self):
        # {x^2 + y^2 = 1, x = y}  ->  x = y = 1/sqrt(2)
        res = lambda x, u: np.array([x[0] ** 2 + x[1] ** 2 - 1.0, x[0] - x[1]])
        ac = AlgebraicConstraint(res, [0.6, 0.4])
        Simulation([ac, Scope()], [Connection(ac, Scope())], log=False).run(0.005)
        np.testing.assert_allclose(_solved(ac), [1 / np.sqrt(2)] * 2, atol=1e-9)

    def test_equilibrium_with_dynamic_input(self):
        # B = K*A and A + B = c(input)  ->  A = c/(1+K), B = K*c/(1+K)
        K = 3.0
        def res(x, u):
            a, b = x
            return np.array([b - K * a, a + b - u[0]])
        ac = AlgebraicConstraint(res, [0.5, 0.5])
        c = Constant(2.0)
        Simulation([c, ac, Scope()],
                   [Connection(c, ac[0]), Connection(ac, Scope())], log=False).run(0.005)
        np.testing.assert_allclose(_solved(ac), [0.5, 1.5], atol=1e-10)

    def test_docstring_and_info(self):
        # Unified onto the registry docstring + info() like every other block.
        self.assertGreater(len(AlgebraicConstraint.__doc__ or ""), 200)
        self.assertIn("F(x, u) = 0", AlgebraicConstraint.__doc__)
        info = AlgebraicConstraint.info()
        self.assertEqual(info["type"], "AlgebraicConstraint")
        self.assertEqual(set(info["parameters"]), {"residual", "x0"})

    def test_quasi_steady_state(self):
        # Feeding a zeroed rate F := -k (x - u) recovers the QSSA: x -> u.
        k = 5.0
        ac = AlgebraicConstraint(lambda x, u: np.array([-k * (x[0] - u[0])]), [0.0])
        c = Constant(1.7)
        Simulation([c, ac, Scope()],
                   [Connection(c, ac[0]), Connection(ac, Scope())], log=False).run(0.005)
        self.assertAlmostEqual(_solved(ac)[0], 1.7, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
