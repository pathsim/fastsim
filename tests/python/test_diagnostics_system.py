########################################################################################
##
##              Testing diagnostics exposure (issue #29): the structured
##              convergence data the engine gathers every step is reachable
##              after a run instead of being discarded into a log string.
##
########################################################################################

import unittest

from fastsim import Simulation, Connection
from fastsim.blocks import Constant, Scope


class TestDiagnostics(unittest.TestCase):

    def test_run_summary_always_available(self):
        """run_summary carries the numeric outcome without diagnostics=True."""
        cns = Constant(1.0)
        sco = Scope()
        sim = Simulation([cns, sco], [Connection(cns, sco)], dt=0.01, log=False)
        sim.run(0.1)
        s = sim.run_summary
        for key in ("converged", "max_residual", "worst_block", "truncated_at"):
            self.assertIn(key, s)
        self.assertTrue(s["converged"])

    def test_diagnostics_snapshot_with_flag(self):
        """With diagnostics=True, the per-timestep snapshot is exposed with its
        iteration counts and a human-readable summary."""
        cns = Constant(1.0)
        sco = Scope()
        sim = Simulation([cns, sco], [Connection(cns, sco)],
                         dt=0.01, log=False, diagnostics=True)
        sim.run(0.1)
        d = sim.diagnostics
        self.assertIsNotNone(d)
        self.assertGreaterEqual(d.solve_iterations, 0)
        self.assertIsInstance(d.summary(), str)

    def test_diagnostics_history_records_every_step(self):
        """diagnostics='history' keeps one snapshot per accepted step, in
        time order; plain True records only the live snapshot."""
        from fastsim.blocks import Integrator

        def build(diag):
            itg = Integrator(1.0)
            sco = Scope()
            return Simulation([itg, sco], [Connection(itg, sco)],
                              dt=0.01, log=False, diagnostics=diag)

        sim = build("history")
        stats = sim.run(0.1, adaptive=False)
        hist = sim.diagnostics_history
        self.assertIsNotNone(hist)
        self.assertEqual(len(hist), int(stats["total_steps"]))
        times = [d.time for d in hist]
        self.assertEqual(times, sorted(times))
        # every entry is a full snapshot
        self.assertGreaterEqual(hist[-1].solve_iterations, 0)

        # plain True: live snapshot only, no history
        sim2 = build(True)
        sim2.run(0.1, adaptive=False)
        self.assertIsNotNone(sim2.diagnostics)
        self.assertIsNone(sim2.diagnostics_history)

    def test_diagnostics_history_cleared_on_reset(self):
        from fastsim.blocks import Integrator
        itg = Integrator(1.0)
        sco = Scope()
        sim = Simulation([itg, sco], [Connection(itg, sco)],
                         dt=0.01, log=False, diagnostics="history")
        sim.run(0.1, adaptive=False)
        self.assertGreater(len(sim.diagnostics_history), 0)
        sim.reset()
        self.assertEqual(len(sim.diagnostics_history), 0)

    def test_diagnostics_rejects_unknown_string(self):
        cns = Constant(1.0)
        sco = Scope()
        with self.assertRaises(ValueError):
            Simulation([cns, sco], [Connection(cns, sco)],
                       log=False, diagnostics="everything")

    def test_compiled_stats_after_run(self):
        """CompiledSimulation.stats mirrors the interpreted stats dict at
        compiled-path scope: accepted/rejected steps and tape evaluations."""
        from fastsim.blocks import Integrator, Amplifier
        itg = Integrator(1.0)
        amp = Amplifier(-0.5)
        sco = Scope()
        sim = Simulation([itg, amp, sco],
                         [Connection(itg, amp, sco), Connection(amp, itg)],
                         dt=0.01, log=False)
        compiled = sim.compile()
        times, _states, _rec = compiled.run(1.0, adaptive=True)
        s = compiled.stats
        self.assertEqual(int(s["total_steps"]), len(times) - 1)
        self.assertGreaterEqual(s["rejected_steps"], 0)
        # every RK step costs at least one derivative evaluation
        self.assertGreater(s["total_evals"], s["total_steps"])


if __name__ == "__main__":
    unittest.main()
