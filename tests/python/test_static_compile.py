########################################################################################
##
##                  Testing the static compile surface (sim.compile())
##
##  sim.compile() -> fastsim.CompiledSimulation: the whole model fused into one
##  dx/dt tape over a global state vector. Exercises signal recording (scope
##  taps), live parameter retuning, and the compiled->block state handoff.
##
########################################################################################

import math
import unittest

from fastsim import Simulation, Connection
from fastsim.blocks import Integrator, Amplifier, Scope, Adder, SinusoidalSource


def _decay_with_scope():
    """x' = -x, x(0)=1; a scope observes x.  x(t) = exp(-t)."""
    i = Integrator(1.0)
    amp = Amplifier(-1.0)
    sco = Scope()
    sim = Simulation(
        [i, amp, sco],
        [Connection(i, amp), Connection(amp, i), Connection(i, sco)],
    )
    return sim, i


class TestStaticCompile(unittest.TestCase):
    def test_compile_run_and_record(self):
        sim, _ = _decay_with_scope()
        c = sim.compile()
        self.assertEqual(c.n_state, 1)
        self.assertEqual(len(c.state_labels), 1)

        c.dt = 0.01
        times, states, rec = c.run(2.0)
        # state matches analytic exp(-t)
        tf = times[-1]
        self.assertAlmostEqual(states[-1][0], math.exp(-tf), delta=2e-3)
        # the scope's observed signal was recorded (one tap == x trajectory)
        self.assertEqual(len(rec), 1)
        (label, series), = rec.items()
        self.assertEqual(len(series), len(times))
        for r, s in zip(series, states):
            self.assertAlmostEqual(r, s[0], delta=1e-9)

    def test_records_source_signal(self):
        # A scope watching a time-varying source records sin(t), even though the
        # source has no state.
        src = SinusoidalSource(1.0, 1.0, 0.0)  # amplitude, frequency, phase
        i = Integrator(0.0)
        sco = Scope()
        sim = Simulation([src, i, sco], [Connection(src, i), Connection(src, sco)])
        c = sim.compile()
        # tap evaluated at a point: signal == amplitude*sin(2*pi*f*t + phase)
        t = 0.3
        val = c.eval_taps([0.0], t)[0]
        self.assertAlmostEqual(val, math.sin(2 * math.pi * 1.0 * t), delta=1e-9)

    def test_parameters_stay_live(self):
        sim, _ = _decay_with_scope()
        c = sim.compile()
        self.assertAlmostEqual(c.deriv([1.0], 0.0)[0], -1.0, delta=1e-12)
        gain = next(n for n in c.param_names if "gain" in n)
        self.assertTrue(c.set_param(gain, -3.0))
        self.assertAlmostEqual(c.deriv([1.0], 0.0)[0], -3.0, delta=1e-12)

    def test_compile_inherits_solver(self):
        # compile() must carry over the source simulation's solver, tolerances
        # and dt, so a compiled run integrates the same problem with the same
        # method — not silently fall back to the default explicit RKBS32 (which
        # on a stiff model is stability-bound and diverges in step count).
        from fastsim.solvers import ESDIRK43
        i = Integrator(1.0)
        amp = Amplifier(-1.0)
        sim = Simulation(
            [i, amp], [Connection(i, amp), Connection(amp, i)],
            dt=0.02, Solver=ESDIRK43,
            tolerance_lte_abs=1e-7, tolerance_lte_rel=1e-4, log=False,
        )
        c = sim.compile()
        self.assertEqual(c.solver, "ESDIRK43")
        self.assertEqual(c.tolerance_lte_abs, 1e-7)
        self.assertEqual(c.tolerance_lte_rel, 1e-4)
        self.assertEqual(c.dt, 0.02)

    def test_compile_inherits_default_solver(self):
        # With no explicit Solver, the source sim's default propagates too (a
        # real factory name, not the compiled-side RKBS32 fallback).
        sim, _ = _decay_with_scope()
        self.assertEqual(sim.compile().solver, sim.solver)

    def test_traced_function_block_compiles(self):
        # A traced Function block (JIT op-graph) must fuse into the compiled
        # system, not be treated as opaque. x' = -x via Function(lambda u: -u).
        from fastsim.blocks import Function
        i = Integrator(1.0)
        f = Function(lambda u: -u)
        sim = Simulation([i, f], [Connection(i, f), Connection(f, i)])
        c = sim.compile()
        self.assertEqual(c.n_state, 1)
        self.assertAlmostEqual(c.deriv([2.0], 0.0)[0], -2.0, delta=1e-9)
        c.dt = 0.01
        c.run(1.0)
        tf, xf = c.times[-1], c.states[-1][0]
        self.assertAlmostEqual(xf, math.exp(-tf), delta=2e-3)

    def test_traced_ode_block_compiles(self):
        # A traced ODE block: dx/dt = -2x, y = x.
        from fastsim.blocks import ODE
        ode = ODE(lambda x, u, t: [-2.0 * x[0]], initial_value=[1.0])
        sco = Scope()
        sim = Simulation([ode, sco], [Connection(ode, sco)])
        c = sim.compile()
        self.assertEqual(c.n_state, 1)
        self.assertAlmostEqual(c.deriv([3.0], 0.0)[0], -6.0, delta=1e-9)
        c.dt = 0.01
        c.run(1.0)
        tf, xf = c.times[-1], c.states[-1][0]
        self.assertAlmostEqual(xf, math.exp(-2 * tf), delta=2e-3)

    def test_discrete_event_block_compiles(self):
        # A sample-hold has a memory slot + a Schedule event; it must compile and
        # run via the event-aware loop (no longer rejected).
        from fastsim.blocks import SampleHold
        src = SinusoidalSource(1.0, 1.0, 0.0)
        sh = SampleHold(0.1)
        i = Integrator(0.0)
        sim = Simulation([src, sh, i], [Connection(src, sh), Connection(sh, i)])
        c = sim.compile()
        self.assertGreaterEqual(c.n_mem, 1, "sample-hold contributes discrete memory")
        c.dt = 0.01
        times, states, _rec = c.run(1.0)
        # ~one segment per sample interval (the integrator is piecewise-linear).
        self.assertGreaterEqual(len(times), 10)
        self.assertLessEqual(abs(times[-1] - 1.0), 0.05)
        self.assertTrue(all(abs(s[0]) < 1e3 for s in states), "trajectory stays finite")

    def test_subsystem_compiles_to_block(self):
        # A Subsystem implementing a first-order lag (dx/dt = u - x, y = x)
        # compiles to a single fused block that drops back into a Simulation and
        # matches the per-block subsystem (and the analytic 1 - e^{-t}).
        from fastsim import Subsystem, Interface
        from fastsim.blocks import Adder, Constant

        def lag():
            iface = Interface()
            err = Adder("+-")  # u - x
            i = Integrator(0.0)  # x' = (u - x), y = x
            sub = Subsystem(
                blocks=[iface, err, i],
                connections=[
                    Connection(iface, err[0]),
                    Connection(i, err[1]),
                    Connection(err, i),
                    Connection(i, iface),
                ],
            )
            return sub, i

        dur = 2.0

        # Reference: per-block subsystem.
        u_r, (sub_r, _) = Constant(1.0), lag()
        ref = Simulation([u_r, sub_r], [Connection(u_r, sub_r)], dt=0.01, log=False)
        ref.run(dur)
        y_ref = sub_r.outputs[0]

        # Assemble once, then compile the subsystem into a block.
        u0, (sub0, _) = Constant(1.0), lag()
        asm = Simulation([u0, sub0], [Connection(u0, sub0)], dt=0.01, log=False)
        asm.run(0.01)
        block = sub0.compile()
        self.assertEqual(block.type_name, "CompiledSubsystem")

        # Compiled block in a fresh sim (same dt).
        u_c = Constant(1.0)
        cmp = Simulation([u_c, block], [Connection(u_c, block)], dt=0.01, log=False)
        cmp.run(dur)
        y_cmp = block.outputs[0]

        self.assertAlmostEqual(y_cmp, y_ref, delta=1e-3)
        self.assertAlmostEqual(y_cmp, 1.0 - math.exp(-dur), delta=1e-2)

    def test_compile_rejects_non_subsystem(self):
        # compile() is only defined for Subsystem blocks.
        i = Integrator(1.0)
        with self.assertRaises(ValueError):
            i.compile()

    def test_rejects_unsupported_with_reason(self):
        # purely algebraic loop -> ValueError mentioning the reason
        src = SinusoidalSource(1.0, 1.0, 0.0)
        err = Adder("+-")
        kp = Amplifier(0.5)
        sim = Simulation(
            [src, err, kp],
            [Connection(src, err), Connection(err, kp), Connection(kp, err)],
        )
        with self.assertRaises(ValueError):
            sim.compile()


if __name__ == "__main__":
    unittest.main()
