"""Tests for the native BVP1D block — scipy.solve_bvp rebuilt with tracer + AD,
plus free parameters and interior (multipoint) conditions.
"""

import unittest

import numpy as np

from fastsim import Simulation, Connection
from fastsim.blocks import BVP1D, Scope, Constant
from fastsim.jit import jacobian


class TestAutodiffMinusZero(unittest.TestCase):
    """Regression: `x - 0.0` must keep its gradient (the AD Jacobian relies on it)."""

    def test_jacobian_through_minus_zero(self):
        f = lambda x, t: np.array([(x[0] - 0.0) * 3.0, x[1] * 2.0])
        J = np.asarray(jacobian(f)(np.array([1.0, 1.0]), 0.0)).reshape(2, 2)
        np.testing.assert_allclose(J, [[3.0, 0.0], [0.0, 2.0]], atol=1e-12)


class TestBVP1D(unittest.TestCase):
    """Native scipy.solve_bvp: collocation + AD Jacobian + mesh refinement."""

    def test_boundary_layer_4th_order(self):
        eps = 1e-2
        exact = lambda x: (np.exp(x / eps) - 1) / (np.exp(1 / eps) - 1)
        fun = lambda x, y, p, inp: np.array([y[1], y[1] / eps])
        bc = lambda ya, yb, p, inp: np.array([ya[0], yb[0] - 1.0])
        xo = np.linspace(0, 1, 50)
        b = BVP1D(fun, bc, n_eq=2, domain=(0, 1), n_mesh=11,
                  initial=lambda x: np.vstack([x, np.ones_like(x)]), x_out=xo)
        Simulation([b, Scope()], [Connection(b, Scope())], log=False).run(0.005)
        self.assertLess(np.max(np.abs(b.solution()[0] - exact(xo))), 1e-5)

    def test_multifield_bcr_matches_scipy(self):
        from scipy.integrate import solve_bvp
        Bo_l, phi_l, Bo_g, phi_g, psi, nu, y_in = 10., 1., 8., .8, .15, 1., 0.

        def theta(a, b, x):
            return a - np.sqrt(np.maximum(0., (1 - psi * x) * b / nu))

        def fun(x, y, p, inp):
            a, da, b, db = y
            th = theta(a, b, x)
            return np.array([da, Bo_l * (phi_l * th - da), db,
                             (Bo_g / (1 - psi * x)) * ((1 + 2 * psi / Bo_g) * db - phi_g * th)])

        bc = lambda ya, yb, p, inp: np.array(
            [ya[1], yb[0] + yb[1] / Bo_l - 1., ya[2] - ya[3] / Bo_g - y_in, yb[3]])
        xo = np.linspace(0, 1, 60)
        blk = BVP1D(fun, bc, n_eq=4, domain=(0, 1), n_mesh=11,
                    initial=lambda x: np.vstack([0.5 + 0 * x, 0 * x, 0.5 + 0 * x, 0 * x]),
                    x_out=xo)
        Simulation([blk, Scope()], [Connection(blk, Scope())], log=False).run(0.005)
        ref = solve_bvp(
            lambda x, S: np.vstack([S[1], Bo_l * (phi_l * theta(S[0], S[2], x) - S[1]), S[3],
                                    (Bo_g / (1 - psi * x)) * ((1 + 2 * psi / Bo_g) * S[3] - phi_g * theta(S[0], S[2], x))]),
            lambda Sa, Sb: np.array([Sa[1], Sb[0] + Sb[1] / Bo_l - 1, Sa[2] - Sa[3] / Bo_g - y_in, Sb[3]]),
            np.linspace(0, 1, 11),
            np.vstack([np.full(11, .5), np.zeros(11), np.full(11, .5), np.zeros(11)]), tol=1e-8)
        sol = blk.solution()
        self.assertLess(np.max(np.abs(sol[0] - ref.sol(xo)[0])), 1e-5)
        self.assertLess(np.max(np.abs(sol[2] - ref.sol(xo)[2])), 1e-5)

    def test_bc_from_inputs(self):
        fun = lambda x, y, p, inp: np.array([y[1], 0.0 * y[0]])
        bc = lambda ya, yb, p, inp: np.array([ya[0], yb[0] - inp[0]])
        xo = np.linspace(0, 1, 20)
        blk = BVP1D(fun, bc, n_eq=2, domain=(0, 1), n_mesh=6, x_out=xo)
        g = Constant(0.7)
        Simulation([g, blk, Scope()],
                   [Connection(g, blk[0]), Connection(blk, Scope())], log=False).run(0.005)
        self.assertLess(np.max(np.abs(blk.solution()[0] - 0.7 * xo)), 1e-6)


class TestFreeParameters(unittest.TestCase):
    """Free parameter p: eigenvalue problem u'' + λ u = 0, u(0)=u(1)=0, u'(0)=1."""

    def test_eigenvalue(self):
        fun = lambda x, y, p, inp: np.array([y[1], -p[0] * y[0]])
        bc = lambda ya, yb, p, inp: np.array([ya[0], yb[0], ya[1] - 1.0])  # 3 = n+k
        b = BVP1D(fun, bc, n_eq=2, n_params=1, p0=[8.0], n_mesh=41,
                  initial=lambda x: np.vstack([np.sin(np.pi * x) / np.pi, np.cos(np.pi * x)]))
        Simulation([b, Scope()], [Connection(b, Scope())], log=False).run(0.005)
        self.assertAlmostEqual(b.parameters()[0], np.pi ** 2, places=3)


class TestInteriorConditions(unittest.TestCase):
    """Interior/multipoint condition: -u'' = 1, u(0)=0, u(0.5)=0.1."""

    def test_multipoint(self):
        fun = lambda x, y, p, inp: np.array([y[1], -1.0])
        bc = lambda ya, yb, p, inp: np.array([ya[0]])                 # 1 BC
        icond = lambda yp, p, inp: np.array([yp[0] - 0.1])            # u(0.5)=0.1
        xo = np.linspace(0, 1, 40)
        b = BVP1D(fun, bc, n_eq=2, n_mesh=41, x_out=xo,
                  x_ports=[0.5], interior_conditions=icond)
        Simulation([b, Scope()], [Connection(b, Scope())], log=False).run(0.005)
        exact = -xo ** 2 / 2 + 0.45 * xo
        self.assertLess(np.max(np.abs(b.solution()[0] - exact)), 1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
