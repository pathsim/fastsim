########################################################################################
##
##                    Testing codegen through Subsystem boundaries
##
##   Every other codegen test builds a flat model, so subsystem flattening went
##   unexercised: interface splices were resolved per PORT while connections
##   address per ELEMENT, which silently collapsed every channel of a multi-input
##   subsystem onto one signal, and the flat block order was taken from the
##   per-scope IR schedule, which is not a valid order once a subsystem's
##   children are spliced in at the parent's position.
##
########################################################################################

# IMPORTS ==============================================================================

import unittest

import numpy as np

from fastsim import Simulation, Connection, Subsystem, Interface
from fastsim.blocks import (Adder, Amplifier, Constant, Integrator, Relay, Scope,
                            SinusoidalSource)
from fastsim.solvers import RK4


# TESTCASE =============================================================================

class TestCodegenSubsystem(unittest.TestCase):
    """Generated C must match the reference for models built out of subsystems."""

    def _verify(self, sim, duration=0.05, dt=0.001):
        return sim.verify_c("m", duration=duration, dt=dt, solver="rk4")

    def test_two_input_subsystem_keeps_its_channels_apart(self):
        """dx/dt = u0 + u1 with the adder inside a subsystem.

        Both subsystem inputs land on one width-2 interface port, so resolving
        the splice per port made the second connection overwrite the first and
        both adder channels read the same constant.
        """
        iface, add, integ = Interface(), Adder(), Integrator(initial_value=0.0)
        sub = Subsystem(
            blocks=[iface, add, integ],
            connections=[
                Connection(iface[0], add[0]),
                Connection(iface[1], add[1]),
                Connection(add[0], integ[0]),
                Connection(integ[0], iface[0]),
            ])
        c0, c1, sco = Constant(value=1.0), Constant(value=100.0), Scope()
        sim = Simulation(
            [sub, c0, c1, sco],
            [Connection(c0[0], sub[0]), Connection(c1[0], sub[1]),
             Connection(sub[0], sco[0])],
            Solver=RK4, dt=0.001, log=False)

        rep = self._verify(sim)
        self.assertTrue(rep["passed"],
            f"subsystem inputs mismatched: {rep['max_scaled_error']:.3e}")

        # The slope must be u0 + u1, not 2*u1 (the collapsed-channel symptom).
        sim2 = Simulation(
            [sub, c0, c1, sco],
            [Connection(c0[0], sub[0]), Connection(c1[0], sub[1]),
             Connection(sub[0], sco[0])],
            Solver=RK4, dt=0.001, log=False)
        t, states, _ = sim2.compile().run(0.002, True, False)
        self.assertAlmostEqual((states[1][0] - states[0][0]) / 0.001, 101.0, places=6)

    def test_subsystem_driven_by_a_parent_level_source(self):
        """A subsystem fed by a parent-level source, the simplest shape where
        inlining reorders the algebraic pass. (The thermostat case below is what
        actually catches a mis-ordered pass — this one guards the plumbing.)"""
        iface, gain = Interface(), Amplifier(gain=3.0)
        sub = Subsystem(
            blocks=[iface, gain],
            connections=[Connection(iface[0], gain[0]), Connection(gain[0], iface[0])])
        src = Constant(value=2.0)
        integ, sco = Integrator(initial_value=0.0), Scope()
        sim = Simulation(
            [sub, src, integ, sco],
            [Connection(src[0], sub[0]), Connection(sub[0], integ[0]),
             Connection(integ[0], sco[0])],
            Solver=RK4, dt=0.001, log=False)

        rep = self._verify(sim)
        self.assertTrue(rep["passed"],
            f"ordering mismatch: {rep['max_scaled_error']:.3e}")

    def test_nested_subsystems(self):
        """Splices resolve through more than one level of nesting."""
        inner_if, inner_gain = Interface(), Amplifier(gain=2.0)
        inner = Subsystem(
            blocks=[inner_if, inner_gain],
            connections=[Connection(inner_if[0], inner_gain[0]),
                         Connection(inner_gain[0], inner_if[0])])
        outer_if, outer_add = Interface(), Adder()
        outer = Subsystem(
            blocks=[outer_if, inner, outer_add],
            connections=[
                Connection(outer_if[0], inner[0]),
                Connection(outer_if[1], outer_add[1]),
                Connection(inner[0], outer_add[0]),
                Connection(outer_add[0], outer_if[0]),
            ])
        c0, c1 = Constant(value=1.0), Constant(value=10.0)
        integ, sco = Integrator(initial_value=0.0), Scope()
        sim = Simulation(
            [outer, c0, c1, integ, sco],
            [Connection(c0[0], outer[0]), Connection(c1[0], outer[1]),
             Connection(outer[0], integ[0]), Connection(integ[0], sco[0])],
            Solver=RK4, dt=0.001, log=False)

        rep = self._verify(sim)
        self.assertTrue(rep["passed"],
            f"nested subsystem mismatch: {rep['max_scaled_error']:.3e}")

        # 2*u0 + u1 = 2*1 + 10 = 12
        sim2 = Simulation(
            [outer, c0, c1, integ, sco],
            [Connection(c0[0], outer[0]), Connection(c1[0], outer[1]),
             Connection(outer[0], integ[0]), Connection(integ[0], sco[0])],
            Solver=RK4, dt=0.001, log=False)
        _, states, _ = sim2.compile().run(0.002, True, False)
        self.assertAlmostEqual((states[1][0] - states[0][0]) / 0.001, 12.0, places=6)

    def test_relay_thermostat_with_a_house_subsystem(self):
        """The editor's thermostat example: a relay's zero-crossing events
        driving a plant wrapped in a subsystem. The generated C must reproduce
        the switching, not run with the heater stuck."""
        T_0, T_a, T_d, C = 23.0, 10.0, 5.0, 1.0
        H, T_hi, T_lo = 15.0 * C, 24.0, 21.0

        house_if = Interface()
        flux = Adder(operations="-++")
        capacity = Amplifier(gain=C)
        heat = Integrator(initial_value=T_0)
        house = Subsystem(
            blocks=[house_if, flux, capacity, heat],
            connections=[
                Connection(heat[0], capacity[1], house_if[0]),
                Connection(capacity[1], flux[0]),
                Connection(capacity[0], flux[1]),
                Connection(flux[0], heat[0]),
                Connection(house_if[1], capacity[0]),
                Connection(house_if[0], flux[2]),
            ])
        relay = Relay(threshold_up=T_hi, threshold_down=T_lo, value_up=0.0, value_down=H)
        amb_const, amb_day = Constant(value=T_a), SinusoidalSource(amplitude=T_d, frequency=1 / 24)
        adder = Adder()
        temp, htr = Scope(labels=["T"]), Scope(labels=["heater"])
        conns = [
            Connection(relay[0], house[0], htr[0]),
            Connection(house[0], relay[0], temp[0]),
            Connection(amb_const[0], adder[1]),
            Connection(amb_day[0], adder[0]),
            Connection(adder[0], house[1]),
        ]
        sim = Simulation([house, relay, amb_const, amb_day, adder, temp, htr], conns,
                         Solver=RK4, dt=0.01, log=False)

        rep = sim.verify_c("thermostat", duration=24.0, dt=0.01, solver="rk4")
        self.assertTrue(rep["passed"],
            f"thermostat C diverged: {rep['max_scaled_error']:.3e} at t={rep['worst_time']}")

        # And it really does cycle — a stuck heater would also "match" a broken
        # reference, so assert the physics too.
        sim2 = Simulation([house, relay, amb_const, amb_day, adder, temp, htr], conns,
                          Solver=RK4, dt=0.01, log=False)
        sim2.run(duration=24, reset=True)
        _, (T,) = temp.read()
        _, (heater,) = htr.read()
        self.assertGreater(int(np.sum(np.abs(np.diff(heater)) > 1)), 5,
            "heater should switch repeatedly")
        self.assertGreater(T.min(), T_lo - 2.0)
        self.assertLess(T.max(), T_hi + 1.0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
