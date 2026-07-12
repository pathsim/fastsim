########################################################################################
##
##          Testing bouncing ball energy conservation and advanced event behavior
##
########################################################################################

import unittest
import numpy as np

from fastsim import Simulation, Connection
from fastsim.blocks import Integrator, Constant, Scope
from fastsim.events import ZeroCrossing

from fastsim.solvers import (
    RKF21, RKBS32, RKF45, RKCK54, RKDP54, RKV65, RKF78, RKDP87
    )


class TestBouncingBallEnergy(unittest.TestCase):
    """Test energy conservation for elastic bouncing ball (b=1.0).

    With perfect elasticity, total energy E = mgh + 0.5*mv^2 should be conserved.
    The ball should bounce indefinitely with constant period.
    """

    def setUp(self):
        self.g = 9.81
        self.h = 1.0
        self.b = 1.0  # perfect elasticity

        Ix = Integrator(self.h)
        Iv = Integrator()
        Cn = Constant(-self.g)
        Sc = Scope()

        blocks = [Ix, Iv, Cn, Sc]
        connections = [
            Connection(Cn, Iv),
            Connection(Iv, Ix),
            Connection(Ix, Sc[0]),
            Connection(Iv, Sc[1]),
        ]

        def func_evt(t):
            *_, x = Ix()
            return x

        def func_act(t):
            *_, x = Ix()
            *_, v = Iv()
            Ix.engine.set(abs(x))
            Iv.engine.set(-self.b * v)

        self.E1 = ZeroCrossing(func_evt=func_evt, func_act=func_act, tolerance=1e-6)
        self.Sc = Sc
        self.Sim = Simulation(blocks, connections, [self.E1], log=False)

    def test_energy_conservation(self):
        """With b=1.0, total energy should be conserved across bounces."""
        for SOL in [RKDP54, RKCK54, RKV65]:
            with self.subTest(SOL=str(SOL)):
                self.Sim.reset()
                self.Sim._set_solver(SOL, tolerance_lte_abs=1e-8)
                self.Sim.run(10)

                time, [x, v] = self.Sc.read()

                # Energy at each sample: E = g*x + 0.5*v^2
                E = self.g * np.array(x) + 0.5 * np.array(v)**2
                E0 = self.g * self.h  # initial energy (v0=0)

                # Energy should stay within 1% of initial
                self.assertTrue(np.all(np.abs(E - E0) / E0 < 0.01),
                    f"Energy drift > 1%: max deviation = {np.max(np.abs(E - E0) / E0):.4f}")

    def test_bounce_period_consistency(self):
        """All bounce periods should be identical for elastic bouncing."""
        for SOL in [RKDP54, RKCK54]:
            with self.subTest(SOL=str(SOL)):
                self.Sim.reset()
                self.Sim._set_solver(SOL, tolerance_lte_abs=1e-8)
                self.Sim.run(10)

                bounce_times = list(self.E1)
                if len(bounce_times) < 4:
                    self.fail("Too few bounces detected")

                # All periods should match T = 2*sqrt(2h/g)
                T = 2 * np.sqrt(2 * self.h / self.g)
                periods = np.diff(bounce_times)

                for i, p in enumerate(periods):
                    self.assertAlmostEqual(p, T, places=3,
                        msg=f"Period {i}: {p:.6f} != {T:.6f}")

    def test_position_never_negative(self):
        """Ball position must never go below ground (x >= 0)."""
        for SOL in [RKF21, RKBS32, RKDP54, RKDP87]:
            with self.subTest(SOL=str(SOL)):
                self.Sim.reset()
                self.Sim._set_solver(SOL)
                self.Sim.run(10)

                time, [x, v] = self.Sc.read()
                self.assertTrue(np.min(x) >= -1e-4,
                    f"Ball went underground: min x = {np.min(x):.6f}")


class TestBouncingBallDissipation(unittest.TestCase):
    """Test that inelastic bouncing ball loses energy correctly."""

    def setUp(self):
        self.g = 9.81
        self.h = 2.0
        self.b = 0.8  # lose 20% velocity each bounce

        Ix = Integrator(self.h)
        Iv = Integrator()
        Cn = Constant(-self.g)
        Sc = Scope()

        blocks = [Ix, Iv, Cn, Sc]
        connections = [
            Connection(Cn, Iv),
            Connection(Iv, Ix),
            Connection(Ix, Sc[0]),
            Connection(Iv, Sc[1]),
        ]

        def func_evt(t):
            *_, x = Ix()
            return x

        def func_act(t):
            *_, x = Ix()
            *_, v = Iv()
            Ix.engine.set(abs(x))
            Iv.engine.set(-self.b * v)

        self.E1 = ZeroCrossing(func_evt=func_evt, func_act=func_act, tolerance=1e-6)
        self.Sc = Sc
        self.Sim = Simulation(blocks, connections, [self.E1], log=False)

    def test_energy_dissipation(self):
        """Ball should settle — final height much less than initial."""
        for SOL in [RKDP54, RKCK54]:
            with self.subTest(SOL=str(SOL)):
                self.Sim.reset()
                self.Sim._set_solver(SOL, tolerance_lte_abs=1e-6)
                self.Sim.run(10)

                time, [x, v] = self.Sc.read()
                time = np.array(time)
                x = np.array(x)

                early_max = np.max(x[time < 2])
                late_max = np.max(x[time > 8]) if np.any(time > 8) else 0.0
                self.assertGreater(early_max, 1.0)
                self.assertLess(late_max, early_max * 0.5,
                    f"Ball didn't dissipate: late={late_max:.3f} vs early={early_max:.3f}")

    def test_many_bounces_detected(self):
        """With b=0.8, many bounces should be detected before settling."""
        self.Sim.reset()
        self.Sim._set_solver(RKDP54, tolerance_lte_abs=1e-6)
        self.Sim.run(10)

        n_bounces = len(self.E1)
        self.assertGreater(n_bounces, 5, f"Expected >5 bounces, got {n_bounces}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
