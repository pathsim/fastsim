########################################################################################
##
##              Testing Volterra-Lotka with event-based population thresholds
##
########################################################################################

import unittest
import numpy as np

from fastsim import Simulation, Connection
from fastsim.blocks import ODE, Scope
from fastsim.events import ZeroCrossing

from fastsim.solvers import RKDP54, RKCK54, RKBS32


class TestVolterraLotkaEventSystem(unittest.TestCase):
    """Test zero-crossing events on a Volterra-Lotka predator-prey system.

    The system oscillates periodically. We place a threshold crossing on
    the prey population and verify events are detected consistently.
    """

    def setUp(self):
        a, b, c, d = 1.5, 1.0, 3.0, 1.0
        self.threshold = 2.0

        def lotka_volterra(x, u, t):
            return np.array([
                a*x[0] - b*x[0]*x[1],
                -c*x[1] + d*x[0]*x[1],
            ])

        self.ode = ODE(lotka_volterra, initial_value=[1.0, 1.0])
        self.sco = Scope()

        blocks = [self.ode, self.sco]
        connections = [
            Connection(self.ode[0], self.sco[0]),
            Connection(self.ode[1], self.sco[1]),
        ]

        # Detect when prey population crosses threshold
        def evt_prey(t):
            *_, x = self.ode()
            return x[0] - self.threshold

        self.E1 = ZeroCrossing(func_evt=evt_prey, tolerance=1e-4)
        self.Sim = Simulation(blocks, connections, [self.E1], log=False)

    def test_events_detected(self):
        """Crossings should be detected for oscillating population."""
        for SOL in [RKDP54, RKCK54, RKBS32]:
            with self.subTest(SOL=str(SOL)):
                self.Sim.reset()
                self.Sim._set_solver(SOL, tolerance_lte_abs=1e-6)
                self.Sim.run(20)

                n_events = len(self.E1)
                self.assertGreater(n_events, 3, f"Too few crossings: {n_events}")

    def test_periodic_crossings(self):
        """Crossing times should be periodic (Volterra-Lotka has fixed period)."""
        self.Sim.reset()
        self.Sim._set_solver(RKDP54, tolerance_lte_abs=1e-8)
        self.Sim.run(30)

        event_times = list(self.E1)
        if len(event_times) < 6:
            self.fail(f"Not enough crossings: {len(event_times)}")

        # Every second crossing is the same phase (up then down)
        # So period = diff between every other event
        periods = [event_times[i+2] - event_times[i] for i in range(len(event_times) - 2)]
        mean_period = np.mean(periods)

        # All periods should match within 5%
        for i, p in enumerate(periods):
            self.assertAlmostEqual(p / mean_period, 1.0, places=1,
                msg=f"Period {i}: {p:.4f} vs mean {mean_period:.4f}")

    def test_population_stays_positive(self):
        """Both populations must remain positive (biological constraint)."""
        self.Sim.reset()
        self.Sim._set_solver(RKDP54, tolerance_lte_abs=1e-6)
        self.Sim.run(20)

        time, [prey, pred] = self.sco.read()
        self.assertTrue(np.min(prey) > -0.01,
            f"Prey went negative: min = {np.min(prey):.4f}")
        self.assertTrue(np.min(pred) > -0.01,
            f"Predator went negative: min = {np.min(pred):.4f}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
