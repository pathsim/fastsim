########################################################################################
##
##              Testing SimError surfacing: user-configuration and runtime
##              errors raise clean, CATCHABLE Python exceptions instead of
##              panicking the extension (issues #27, #28).
##
########################################################################################

import unittest

from fastsim import Simulation, Connection, Subsystem, Interface
from fastsim.blocks import Constant, Scope, Divider, LUT1D
from fastsim.exceptions import (
    FastSimError,
    InvalidBlockParameterError,
    PortConnectionError,
)
from fastsim.solvers import SSPRK22


class TestErrorHandling(unittest.TestCase):

    def test_bad_port_alias_raises_at_run(self):
        """A connection to a non-existent port alias used to panic deep in the
        data-transfer hot path; it must now surface as a clean exception when
        the run resolves the graph (no process abort)."""
        c = Constant(1.0)
        sco = Scope()
        # 'nonexistent' is not a declared input alias on Scope.
        sim = Simulation([c, sco], [Connection(c, sco["nonexistent"])], dt=0.01)
        with self.assertRaises(ValueError):
            sim.run(1.0)

    def test_bad_port_alias_is_fastsim_error(self):
        """Issue #27/#33: a port typo is a hard error catchable both as the
        historical ValueError AND the new FastSimError/PortConnectionError."""
        c = Constant(1.0)
        sco = Scope()
        sim = Simulation([c, sco], [Connection(c, sco["nope"])], dt=0.01)
        with self.assertRaises(PortConnectionError):
            sim.run(1.0)

    def test_divider_unknown_op_raises(self):
        """Divider with an op character outside '*'/'/' raises at construction
        rather than panicking."""
        with self.assertRaises(ValueError):
            Divider("x")

    def test_subsystem_multiple_interfaces_raises(self):
        """A Subsystem with more than one Interface block raises rather than
        panicking."""
        with self.assertRaises(ValueError):
            Subsystem([Interface(), Interface(), Constant(1.0)], [])

    # -- issue #28: constructor asserts -> catchable ValueError -----------------

    def test_bad_lut_raises_value_error(self):
        """LUT1D with mismatched points/values lengths must raise ValueError
        (previously an uncatchable assert panic)."""
        with self.assertRaises(ValueError):
            LUT1D([0.0, 1.0, 2.0], [0.0, 1.0])
        # And catchable via the new hierarchy.
        with self.assertRaises(InvalidBlockParameterError):
            LUT1D([0.0], [0.0])
        with self.assertRaises(FastSimError):
            LUT1D([0.0, 2.0, 1.0], [0.0, 1.0, 2.0])  # not strictly increasing

    # -- issue #28: Divider runtime zero-denominator -> catchable ---------------

    def test_zero_denominator_run_fails_catchably(self):
        """A data-dependent zero denominator under zero_div='raise' used to
        panic uncatchably at a timestep; it must now stop the run and surface a
        catchable exception."""
        num = Constant(1.0)
        den = Constant(0.0)
        div = Divider("*/", zero_div="raise")  # in0 numerator, in1 denominator
        sco = Scope()
        sim = Simulation(
            [num, den, div, sco],
            [Connection(num, div[0]), Connection(den, div[1]), Connection(div, sco)],
            dt=0.01, log=False,
        )
        with self.assertRaises(ValueError):
            sim.run(1.0)

    # -- issue #28: raising Python ODE callback in integrate() ------------------

    def test_raising_ode_callback_surfaces_in_integrate(self):
        """A user ODE callback that raises inside Solver.integrate() must
        surface the ORIGINAL Python exception, not an uncatchable panic."""
        class Boom(RuntimeError):
            pass

        def f(x, t):
            raise Boom("callback exploded")

        with self.assertRaises(Boom):
            SSPRK22.integrate(f, [1.0], time_start=0.0, time_end=0.1, dt=0.01)


if __name__ == "__main__":
    unittest.main()
