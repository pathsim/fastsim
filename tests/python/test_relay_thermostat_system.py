########################################################################################
##
##                    Testing relay-controlled thermostat system
##
##   Thermal plant with relay hysteresis controller. Verifies event-driven
##   switching behavior produces correct temperature oscillation pattern.
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from fastsim import Simulation, Connection
from fastsim.blocks import Integrator, Amplifier, Adder, Constant, Relay, Scope

from fastsim.solvers import (
    RKBS32, RKCK54, RKDP54, RKV65, RKDP87,
    ESDIRK32, ESDIRK43, ESDIRK54
    )


# TESTCASE =============================================================================

class TestRelayThermostatSystem(unittest.TestCase):
    """
    Thermostat system: relay controller with hysteresis driving a first-order
    thermal plant.

    System:
        heater = Relay(threshold_up=22, threshold_down=18, value_up=0, value_down=50)
        dT/dt = -alpha*(T - T_ambient) + heater_output / C

    When temperature rises above 22 -> heater OFF (value_up=0)
    When temperature drops below 18 -> heater ON (value_down=50)

    The system should oscillate between the two thresholds.
    """

    def setUp(self):

        #thermal parameters
        self.alpha = 0.5  # heat loss coefficient
        self.T_amb = 10.0  # ambient temperature
        self.C = 5.0  # thermal capacity

        #initial temperature (between thresholds)
        self.T0 = 20.0

        #blocks
        self.Int = Integrator(self.T0)  # temperature state
        Amp = Amplifier(-self.alpha)  # heat loss: -alpha * T
        Amb = Constant(self.alpha * self.T_amb)  # ambient contribution: alpha * T_amb
        Htr = Amplifier(1.0 / self.C)  # heater gain: heater / C
        Add = Adder()  # sum: -alpha*T + alpha*T_amb + heater/C

        self.Rly = Relay(
            threshold_up=22.0,
            threshold_down=18.0,
            value_up=0.0,     # heater off when T > 22
            value_down=50.0   # heater on when T < 18
            )

        self.Sco = Scope(labels=["temperature", "heater"])

        blocks = [self.Int, Amp, Amb, Htr, Add, self.Rly, self.Sco]

        #connections: T -> Amp, Amp -> Add[0], Amb -> Add[1], Rly -> Htr -> Add[2], Add -> Int
        connections = [
            Connection(self.Int, Amp, self.Rly, self.Sco[0]),
            Connection(Amp, Add[0]),
            Connection(Amb, Add[1]),
            Connection(self.Rly, Htr, self.Sco[1]),
            Connection(Htr, Add[2]),
            Connection(Add, self.Int)
            ]

        self.Sim = Simulation(
            blocks,
            connections,
            dt=0.01,
            log=False
            )


    def test_thermostat_oscillation(self):
        """Test that temperature oscillates between thresholds"""

        self.Sim.run(duration=30, reset=True)

        time, [temp, heater] = self.Sco.read()

        #after initial transient (t>5), temperature should stay within bounds
        mask = time > 5
        temp_steady = temp[mask]

        #temperature should oscillate within reasonable bounds around thresholds
        self.assertTrue(np.min(temp_steady) > 16.0,
            f"Temperature dropped too low: {np.min(temp_steady):.2f}")
        self.assertTrue(np.max(temp_steady) < 24.0,
            f"Temperature rose too high: {np.max(temp_steady):.2f}")

        #heater should have switched multiple times
        heater_steady = heater[mask]
        switches = np.sum(np.abs(np.diff(heater_steady)) > 1)
        self.assertGreater(switches, 2, "Heater should have switched multiple times")


    def test_thermostat_with_adaptive_solvers(self):
        """Test thermostat with different adaptive solvers"""

        for SOL in [RKBS32, RKCK54, RKDP87]:

            with self.subTest(SOL=str(SOL)):

                self.Sim.reset()
                self.Sim._set_solver(SOL, tolerance_lte_abs=1e-6)
                self.Sim.run(duration=20, reset=True)

                time, [temp, _] = self.Sco.read()

                #temperature should stay bounded
                mask = time > 5
                self.assertTrue(np.min(temp[mask]) > 16.0)
                self.assertTrue(np.max(temp[mask]) < 24.0)


    def test_thermostat_generates_c_and_matches_the_reference(self):
        """The relay's zero-crossing events lower to C, and the compiled
        trajectory matches the reference over the full fixed-step run."""

        files = self.Sim.to_c("thermostat", solver="rk4")
        self.assertIn("thermostat.c", files)
        self.assertIn("thermostat.h", files)
        #the relay contributes two zero-cross events (rising/falling threshold)
        self.assertIn("thermostat_handle_events", files["thermostat.c"])

        report = self.Sim.verify_c("thermostat", duration=30.0, dt=0.01, solver="rk4")
        self.assertTrue(report["passed"],
            f"C run diverged from the reference: {report['max_scaled_error']:.3e} "
            f"at t={report['worst_time']}")


    def test_to_c_inherits_the_simulation_tolerances(self):
        """An adaptive solver's emitted error scale carries THIS simulation's
        LTE tolerances, so the generated C accepts the same steps the reference
        does. Explicit atol/rtol override them."""

        import re

        def scale(src):
            m = re.search(r"double scale = ([^;]+);", src)
            self.assertIsNotNone(m, "adaptive solver emitted no error scale")
            return m.group(1)

        self.Sim.set_solver(RKBS32, tolerance_lte_abs=1e-9, tolerance_lte_rel=1.5e-7)
        inherited = scale(self.Sim.to_c("m", solver="rkbs32")["m.c"])
        self.assertIn("1e-9", inherited)
        self.assertIn("1.5e-7", inherited)

        overridden = scale(self.Sim.to_c("m", solver="rkbs32", atol=2e-5, rtol=3e-4)["m.c"])
        self.assertIn("2e-5", overridden)
        self.assertIn("0.0003", overridden)

        #a fixed-step tableau has no error control and must not carry them
        self.assertNotIn("1e-9", self.Sim.to_c("m", solver="rk4")["m.c"])

        for bad in ({"atol": 0.0}, {"rtol": -1.0}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.Sim.to_c("m", solver="rkbs32", **bad)


    def test_set_solver_keeps_the_simulation_tolerances(self):
        """Swapping the solver without naming tolerances must not silently
        retune the model to the engine defaults."""

        sim = Simulation(
            [self.Int, self.Rly, self.Sco], [], dt=0.01, log=False,
            Solver=RKBS32, tolerance_lte_abs=1e-9, tolerance_lte_rel=1.5e-7,
            )
        sim.set_solver(RKDP54)
        src = sim.to_c("m", solver="rkdp54")["m.c"]
        self.assertIn("1e-9", src)
        self.assertIn("1.5e-7", src)


    def test_thermostat_with_implicit_solvers(self):
        """Test thermostat with implicit adaptive solvers"""

        for SOL in [ESDIRK32, ESDIRK43]:

            with self.subTest(SOL=str(SOL)):

                self.Sim.reset()
                self.Sim._set_solver(SOL, tolerance_lte_abs=1e-6)
                self.Sim.run(duration=20, reset=True)

                time, [temp, _] = self.Sco.read()

                #temperature should stay bounded
                mask = time > 5
                self.assertTrue(np.min(temp[mask]) > 16.0)
                self.assertTrue(np.max(temp[mask]) < 24.0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
