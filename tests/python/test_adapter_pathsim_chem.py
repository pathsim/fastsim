"""Tests for `fastsim.adapter.adapt` using pathsim-chem as the reference toolbox.

These tests require `pathsim_chem` to be installed. If it isn't, they are
skipped so the main suite still passes on a clean install.
"""

import unittest

import numpy as np

import fastsim
from fastsim import Simulation, Connection
from fastsim.blocks import Source, Scope
from fastsim.solvers import RKDP54


try:
    from pathsim_chem.tritium import ResidenceTime, Process
    _HAS_PATHSIM_CHEM = True
except ImportError:
    _HAS_PATHSIM_CHEM = False


@unittest.skipUnless(_HAS_PATHSIM_CHEM, "pathsim-chem not installed")
class TestResidenceTime(unittest.TestCase):

    def test_clone_preserves_fastsim_base(self):
        """Adapted class MRO contains fastsim.DynamicalSystem (not pathsim's)."""
        import fastsim.blocks as fs
        import pathsim.blocks as ps
        ResFS = fastsim.adapt(ResidenceTime)

        self.assertIs(ResFS.__mro__[1], fs.DynamicalSystem)
        self.assertNotIn(ps.DynamicalSystem, ResFS.__mro__)
        # Original untouched
        self.assertIs(ResidenceTime.__mro__[1], ps.DynamicalSystem)

    def test_runs_in_fastsim_simulation(self):
        """Adapted ResidenceTime with constant source converges to src·tau."""
        ResFS = fastsim.adapt(ResidenceTime)
        tau, src = 2.0, 5.0
        rt = ResFS(tau=tau, initial_value=0.0, source_term=src)
        sco = Scope()
        sim = Simulation([rt, sco], [Connection(rt, sco)], log=False)
        sim._set_solver(RKDP54, tolerance_lte_abs=1e-10)
        sim.run(10.0)
        _, ch = sco.read()
        expected = src * tau * (1 - np.exp(-10.0 / tau))
        self.assertAlmostEqual(ch[0][-1], expected, places=3)


@unittest.skipUnless(_HAS_PATHSIM_CHEM, "pathsim-chem not installed")
class TestProcess(unittest.TestCase):

    def test_multi_level_hierarchy(self):
        """Process -> ResidenceTime -> DynamicalSystem is fully rebased."""
        import fastsim.blocks as fs
        ProcFS = fastsim.adapt(Process)
        self.assertIn(fs.DynamicalSystem, ProcFS.__mro__)

    def test_process_reaches_steady_state(self):
        """Process with unit input → x -> tau, x/tau -> 1."""
        ProcFS = fastsim.adapt(Process)
        tau2, u = 1.5, 1.0
        src = Source(lambda t: u)
        proc = ProcFS(tau=tau2, initial_value=0.0, source_term=0.0)
        sco = Scope()
        sim = Simulation(
            [src, proc, sco],
            [Connection(src, proc),
             Connection(proc[0], sco[0]),
             Connection(proc[1], sco[1])],
            log=False,
        )
        sim._set_solver(RKDP54, tolerance_lte_abs=1e-10)
        sim.run(20.0)
        _, ch = sco.read()
        self.assertAlmostEqual(ch[0][-1], tau2 * u, places=3)
        self.assertAlmostEqual(ch[1][-1], u, places=3)


@unittest.skipUnless(_HAS_PATHSIM_CHEM, "pathsim-chem not installed")
class TestOverrideSafety(unittest.TestCase):

    def test_overriding_reset_is_rejected(self):
        """A toolbox class that shadows a fastsim base method must refuse adaptation."""
        from pathsim.blocks.dynsys import DynamicalSystem as PsDynSys

        class Evil(PsDynSys):
            def __init__(self, tau=1.0):
                super().__init__(
                    func_dyn=lambda x, u, t: -x / tau,
                    func_alg=lambda x, u, t: x,
                    initial_value=0.0,
                )
            def reset(self):
                pass

        with self.assertRaises(TypeError) as ctx:
            fastsim.adapt(Evil)
        self.assertIn("reset", str(ctx.exception))

    def test_override_non_strict_warns(self):
        import warnings
        from pathsim.blocks.dynsys import DynamicalSystem as PsDynSys

        class Evil2(PsDynSys):
            def __init__(self, tau=1.0):
                super().__init__(
                    func_dyn=lambda x, u, t: -x / tau,
                    func_alg=lambda x, u, t: x,
                    initial_value=0.0,
                )
            def reset(self):
                pass

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fastsim.adapt(Evil2, strict=False)
            self.assertTrue(any("reset" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
