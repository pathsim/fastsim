########################################################################################
##
##              Testing API-consistency items (issue #33): exception hierarchy,
##              ndarray returns, public set_solver alias, IR to_json for
##              from_dict-built modules, and the extract_vec_f64 length check.
##
########################################################################################

import json
import unittest

import numpy as np

from fastsim import Simulation, Connection, ir
from fastsim.blocks import Constant, Scope, Integrator, ODE
from fastsim.solvers import RKDP54
from fastsim.exceptions import (
    FastSimError,
    FastSimValueError,
    SolverError,
    ConvergenceError,
    AlgebraicLoopError,
    StepSizeError,
    SingularJacobianError,
    TruncatedTrajectoryError,
    InvalidBlockParameterError,
    PortConnectionError,
    FastSimWarning,
    FastSimConvergenceWarning,
)


class TestExceptionHierarchy(unittest.TestCase):

    def test_dual_bases_preserve_builtin_compat(self):
        # Parameter/config errors keep ValueError compat.
        for cls in (FastSimValueError, InvalidBlockParameterError, PortConnectionError):
            self.assertTrue(issubclass(cls, FastSimError))
            self.assertTrue(issubclass(cls, ValueError))
        # Solver failures keep RuntimeError compat.
        for cls in (SolverError, ConvergenceError, AlgebraicLoopError,
                    StepSizeError, SingularJacobianError, TruncatedTrajectoryError):
            self.assertTrue(issubclass(cls, FastSimError))
            self.assertTrue(issubclass(cls, RuntimeError))
        # Solver subclasses nest under SolverError.
        for cls in (ConvergenceError, AlgebraicLoopError, StepSizeError):
            self.assertTrue(issubclass(cls, SolverError))

    def test_warning_hierarchy(self):
        self.assertTrue(issubclass(FastSimConvergenceWarning, FastSimWarning))
        self.assertTrue(issubclass(FastSimWarning, UserWarning))


class TestNdarrayReturns(unittest.TestCase):

    def _run_small(self):
        c = Constant(1.0)
        integ = Integrator(0.0)
        sco = Scope()
        sim = Simulation(
            [c, integ, sco],
            [Connection(c, integ), Connection(integ, sco)],
            dt=0.01, log=False,
        )
        sim.run(0.05)
        return c, integ

    def test_inputs_outputs_are_ndarrays(self):
        c, integ = self._run_small()
        self.assertIsInstance(c.outputs, np.ndarray)
        self.assertIsInstance(integ.inputs, np.ndarray)
        self.assertIsInstance(integ.outputs, np.ndarray)

    def test_stateless_state_is_empty_array_not_none(self):
        c, integ = self._run_small()
        self.assertIsInstance(c.state, np.ndarray)
        self.assertEqual(c.state.size, 0)      # stateless -> empty, not None
        self.assertIsInstance(integ.state, np.ndarray)
        self.assertEqual(integ.state.size, 1)  # one continuous state


class TestSetSolverAlias(unittest.TestCase):

    def test_public_and_underscore_set_solver(self):
        c = Constant(1.0)
        sco = Scope()
        sim = Simulation([c, sco], [Connection(c, sco)], dt=0.01, log=False)
        sim.set_solver(RKDP54)     # public name (mirrors compiled path)
        sim._set_solver(RKDP54)    # backwards-compatible underscore alias
        sim.run(0.02)


class TestIrToJson(unittest.TestCase):

    def test_to_json_for_from_dict_module(self):
        c = Constant(1.0)
        integ = Integrator(0.0)
        sco = Scope()
        sim = Simulation(
            [c, integ, sco],
            [Connection(c, integ), Connection(integ, sco)],
            dt=0.01, log=False,
        )
        # A module built from a parsed dict has no cached _raw_json, so to_json
        # must serialize the dataclass tree (previously NotImplementedError).
        d = json.loads(sim.to_ir().to_json())
        m = ir.Module.from_dict(d)
        self.assertIsNone(m.__dict__.get("_raw_json"))
        s = m.to_json()
        d2 = json.loads(s)
        # Round-trips: re-parse yields the same top-level shape.
        self.assertEqual(d2["name"], d["name"])
        ir.Module.from_dict(d2)  # must not raise


class TestExtractVecLengthCheck(unittest.TestCase):

    def test_wrong_length_ode_return_raises_clearly(self):
        # func returns a 2-vector but the state has 1 component. The float(np...)
        # calls keep the callback opaque (not traced), so it goes through the
        # extract_vec_f64 length-check boundary.
        def f(x, u, t):
            return [float(np.sin(x[0])), float(np.cos(x[0]))]

        blk = ODE(func=f, initial_value=[1.0])
        sim = Simulation([blk], [], dt=0.01, log=False)
        with self.assertRaisesRegex(ValueError, "state component"):
            sim.run(0.05)


if __name__ == "__main__":
    unittest.main()
