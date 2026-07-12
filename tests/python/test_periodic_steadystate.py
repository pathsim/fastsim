########################################################################################
##
##                  Testing periodic steady-state shooting solver
##
##   Verifies Simulation.periodic_steady_state() converges to the limit
##   cycle of a periodically-driven dynamical system, both with explicit
##   and implicit inner solvers.
##
########################################################################################

import math
import unittest

import numpy as np

from fastsim import Simulation, Connection
from fastsim.blocks import Source, Scope, DynamicalSystem
from fastsim.solvers import RKDP54, ESDIRK43


class TestPeriodicSteadyStateLowpass(unittest.TestCase):
    """
    Linear lowpass with sinusoidal forcing.

    ODE:    dx/dt = -x + sin(omega*t)
    Steady state (analytical):
        x(t) = A*sin(omega*t) + B*cos(omega*t)
        A = 1/(1+omega^2),  B = -omega/(1+omega^2)
        => x(0) = B = -omega/(1+omega^2)

    With omega = 1: x(0) = -0.5.
    """

    def _build_system(self):
        omega = 1.0
        period = 2.0 * math.pi / omega
        x0_expected = -omega / (1.0 + omega * omega)

        src = Source(lambda t: math.sin(omega * t))
        plant = DynamicalSystem(
            func_dyn=lambda x, u, t: -x + u,
            func_alg=lambda x, u, t: x,
            initial_value=0.0,
        )
        sco = Scope(labels=["state"])

        sim = Simulation(
            blocks=[src, plant, sco],
            connections=[
                Connection(src, plant),
                Connection(plant, sco),
            ],
            Solver=RKDP54,
            tolerance_lte_abs=1e-10,
            tolerance_lte_rel=1e-8,
            log=False,
        )
        return sim, sco, period, x0_expected

    def test_lowpass_rkdp54(self):
        sim, sco, period, x0_expected = self._build_system()
        sim.periodic_steady_state(period=period, Solver=RKDP54,
                                  tolerance_lte_abs=1e-10, tolerance_lte_rel=1e-8,
                                  reset=True)
        time, [state] = sco.read()
        self.assertAlmostEqual(state[0], x0_expected, places=3,
            msg=f"x(0) should be {x0_expected:.4f}, got {state[0]:.4f}")
        # Limit-cycle closure: x(T) == x(0)
        self.assertAlmostEqual(state[-1], state[0], places=3,
            msg=f"x(T) - x(0) = {state[-1] - state[0]:.4e}")

    def test_lowpass_esdirk43(self):
        sim, sco, period, x0_expected = self._build_system()
        sim.periodic_steady_state(period=period, Solver=ESDIRK43,
                                  tolerance_lte_abs=1e-10, tolerance_lte_rel=1e-8,
                                  reset=True)
        time, [state] = sco.read()
        self.assertAlmostEqual(state[0], x0_expected, places=3)
        self.assertAlmostEqual(state[-1], state[0], places=3)


class TestPeriodicSteadyStateVanDerPol(unittest.TestCase):
    """
    Driven Van der Pol oscillator.

    System: x'' - mu*(1 - x^2)*x' + x = A*sin(omega*t)
    State:  x1 = x, x2 = x'
            dx1/dt = x2
            dx2/dt = mu*(1 - x1^2)*x2 - x1 + A*sin(omega*t)

    Compare PSS result vs. long-transient (40 periods) reference.
    """

    def test_vdp_locks_to_forcing(self):
        mu = 1.0
        A_force = 1.2
        omega = 1.0
        period = 2.0 * math.pi / omega

        def f_dyn(x, u, t):
            return np.array([x[1], mu*(1 - x[0]**2)*x[1] - x[0] + u[0]])
        def f_alg(x, u, t):
            return np.array([x[0]])

        # --- Reference: 40-period transient ---
        src_ref = Source(lambda t: A_force * math.sin(omega * t))
        vdp_ref = DynamicalSystem(
            func_dyn=f_dyn, func_alg=f_alg, initial_value=[0.5, 0.0],
        )
        sim_ref = Simulation(
            blocks=[src_ref, vdp_ref],
            connections=[Connection(src_ref, vdp_ref)],
            Solver=RKDP54, tolerance_lte_abs=1e-10, tolerance_lte_rel=1e-8,
            log=False,
        )
        sim_ref.run(duration=40 * period, reset=True)
        ref_state = np.array(vdp_ref.state)

        # --- PSS from zero ---
        src_pss = Source(lambda t: A_force * math.sin(omega * t))
        vdp_pss = DynamicalSystem(
            func_dyn=f_dyn, func_alg=f_alg, initial_value=[0.5, 0.0],
        )
        sim_pss = Simulation(
            blocks=[src_pss, vdp_pss],
            connections=[Connection(src_pss, vdp_pss)],
            Solver=RKDP54, tolerance_lte_abs=1e-10, tolerance_lte_rel=1e-8,
            log=False,
        )
        sim_pss.periodic_steady_state(period=period, Solver=RKDP54,
                                      tolerance_lte_abs=1e-10, tolerance_lte_rel=1e-8,
                                      reset=True)
        pss_state = np.array(vdp_pss.state)

        diff = np.linalg.norm(pss_state - ref_state)
        self.assertLess(diff, 5e-2,
            f"PSS state {pss_state} vs ref {ref_state}, diff={diff:.4e}")


if __name__ == "__main__":
    unittest.main()
