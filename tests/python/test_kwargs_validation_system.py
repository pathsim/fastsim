########################################################################################
##
##              Testing kwargs handling (issue #31): the LTE tolerance knobs are
##              explicit named parameters, and a typo'd solver kwarg raises
##              TypeError instead of being silently dropped.
##
########################################################################################

import inspect
import unittest

from fastsim import Simulation, Subsystem, Interface, Connection
from fastsim.blocks import Constant, Scope, Integrator


class TestKwargsValidation(unittest.TestCase):

    def test_promoted_tolerances_visible_in_signature(self):
        """tolerance_lte_abs / tolerance_lte_rel are explicit params, so they
        show up in inspect.signature (no longer hidden in **solver_kwargs)."""
        sig = inspect.signature(Simulation.__init__)
        self.assertIn("tolerance_lte_abs", sig.parameters)
        self.assertIn("tolerance_lte_rel", sig.parameters)

    def test_promoted_tolerances_accepted(self):
        """Explicit LTE tolerances construct without error and run cleanly."""
        cns = Constant(1.0)
        sco = Scope()
        sim = Simulation(
            [cns, sco], [Connection(cns, sco)], dt=0.01, log=False,
            tolerance_lte_abs=1e-8, tolerance_lte_rel=1e-9,
        )
        stats = sim.run(0.05)
        self.assertEqual(stats["converged"], 1.0)

    def test_typo_kwarg_raises_typeerror(self):
        """A mistyped solver kwarg must raise TypeError, not silently vanish."""
        with self.assertRaises(TypeError):
            Simulation([Constant(1.0)], [], dt=0.01, tolerance_lte_abz=1e-6)

    def test_subsystem_typo_kwarg_raises_typeerror(self):
        """The Subsystem double-swallow is closed: a typo raises TypeError."""
        with self.assertRaises(TypeError):
            Subsystem([Interface(), Constant(1.0)], [], bogus_knob=1.0)

    def test_tolerance_fpi_still_accepted_with_warning(self):
        """The retired tolerance_fpi is accepted for source compatibility and
        emits a DeprecationWarning (only when explicitly supplied)."""
        with self.assertWarns(DeprecationWarning):
            Simulation([Constant(1.0), Scope()], [], dt=0.01, log=False,
                       tolerance_fpi=1e-9)

    def test_default_construction_no_deprecation_warning(self):
        """A default construction must NOT spam a tolerance_fpi deprecation."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            Simulation([Constant(1.0), Scope()], [], dt=0.01, log=False)


if __name__ == "__main__":
    unittest.main()
