########################################################################################
##
##                      Testing the hierarchical IR export (sim.to_ir())
##
##  Exercises the first-class Python IR surface (fastsim.ir.Module) against real
##  models: per-block op-graphs, typed extern blocks, nested subsystems with
##  interface-referencing connections, and lossless JSON round-trips.
##
########################################################################################

import unittest

from fastsim import Simulation, Connection, Interface, Subsystem, ir
from fastsim.blocks import (
    Integrator, Amplifier, Adder, Scope, SinusoidalSource, Multiplier,
)

INTERFACE = 0xFFFFFFFF


def _vdp_subsystem_sim():
    """A van-der-Pol-style nested subsystem driving a scope (mirrors the
    structure of test_vanderpol_system.py)."""
    If = Interface()
    I1 = Integrator(2.0)
    I2 = Integrator(0.0)
    A = Amplifier(-1.0)
    M = Multiplier()
    Add = Adder("++")
    sub_blocks = [If, I1, I2, A, M, Add]
    sub_conn = [
        Connection(If, I1),
        Connection(I1, I2),
        Connection(I2, A),
        Connection(A, Add),
        Connection(Add, If),
        Connection(I1, M),
    ]
    VDP = Subsystem(sub_blocks, sub_conn)
    Sco = Scope()
    return Simulation([VDP, Sco], [Connection(VDP, Sco)])


class TestIRExport(unittest.TestCase):
    def test_flat_model_ops_and_extern(self):
        src = SinusoidalSource(1.0, 1.0, 0.0)
        amp = Amplifier(2.0)
        sco = Scope()
        sim = Simulation([src, amp, sco], [Connection(src, amp), Connection(amp, sco)])
        m = sim.to_ir("flat")

        self.assertIsInstance(m, ir.Module)
        names = {b.type_name for b in m.blocks()}
        self.assertIn("Amplifier", names)
        # amplifier carries a real op-graph (not extern)
        amp_b = next(b for b in m.blocks() if b.type_name == "Amplifier")
        self.assertFalse(amp_b.is_extern)
        self.assertTrue(amp_b.regions.alg.ops, "amplifier should have alg ops")
        # scope is a sink -> typed extern
        self.assertTrue(m.extern_blocks())
        self.assertIn("Scope", {b.type_name for b in m.extern_blocks()})

    def test_param_op_graph(self):
        amp = Amplifier(3.5)
        sco = Scope()
        sim = Simulation([amp, sco], [Connection(amp, sco)])
        m = sim.to_ir()
        amp_b = next(b for b in m.blocks() if b.type_name == "Amplifier")
        # gain is a runtime-mutable Param
        self.assertTrue(any(p.name == "gain" for p in amp_b.params))
        kinds = [o.kind for o in amp_b.regions.alg.ops]
        self.assertIn("Param", kinds)
        self.assertIn("Binary", kinds)

    def test_nested_subsystem(self):
        sim = _vdp_subsystem_sim()
        m = sim.to_ir("vdp")

        subs = list(m.root.subsystems())
        self.assertEqual(len(subs), 1)
        s = subs[0]
        # integrators live inside the subsystem
        inner = {b.type_name for b in s.blocks()}
        self.assertIn("Integrator", inner)
        # an Integrator carries both alg (y=x) and dyn (dx/dt) regions
        integ = next(b for b in s.blocks() if b.type_name == "Integrator")
        self.assertTrue(integ.regions.alg.ops)
        self.assertTrue(integ.regions.dyn.writes, "integrator should have a dyn region")
        # internal connections reference the interface sentinel
        self.assertTrue(
            any(c.src.is_interface or any(t.is_interface for t in c.targets)
                for c in s.connections),
            "subsystem connections should reference INTERFACE",
        )
        # outer-facing interface mirrors the wrapper's I/O
        self.assertTrue(s.interface.inputs or s.interface.outputs)

    def test_json_roundtrip_stable(self):
        sim = _vdp_subsystem_sim()
        m = sim.to_ir("vdp")
        j1 = m.to_json()
        m2 = ir.Module.from_json(j1)
        self.assertEqual(j1, m2.to_json(), "IR JSON round-trip must be stable")
        self.assertEqual(m.summary(), m2.summary())

    def test_summary_runs(self):
        sim = _vdp_subsystem_sim()
        s = sim.to_ir().summary()
        self.assertIn("Module", s)
        self.assertIn("blocks", s)

    def test_schedule_feedforward(self):
        """A feedforward chain populates topo + depth groups, no SCCs."""
        src = SinusoidalSource(1.0, 1.0, 0.0)
        amp = Amplifier(2.0)
        integ = Integrator(0.0)
        sco = Scope()
        sim = Simulation(
            [src, amp, integ, sco],
            [Connection(src, amp), Connection(amp, integ), Connection(integ, sco)],
        )
        sched = sim.to_ir("ff").root.schedule
        self.assertEqual(len(sched.topo), 4, "topo covers every child")
        self.assertTrue(sched.groups, "feedforward exposes depth groups")
        self.assertFalse(sched.sccs, "feedforward has no algebraic loops")
        self.assertFalse(sched.back_edges)

    def test_schedule_algebraic_loop(self):
        """A purely algebraic feedback loop surfaces as one SCC with a back-edge.

        src -> err(+ -) -> kp(0.5) -> err, kp -> scope. The err/kp pair is an
        algebraic loop (no integrator in the cycle)."""
        src = SinusoidalSource(1.0, 1.0, 0.0)
        err = Adder("+-")
        kp = Amplifier(0.5)
        sco = Scope()
        sim = Simulation(
            [src, err, kp, sco],
            [Connection(src, err), Connection(err, kp),
             Connection(kp, err), Connection(kp, sco)],
        )
        sched = sim.to_ir("loop").root.schedule
        self.assertEqual(len(sched.sccs), 1, "one algebraic loop")
        scc = sched.sccs[0]
        self.assertEqual(len(scc.blocks), 2, "loop = {err, kp}")
        self.assertTrue(scc.back_edges, "SCC reports a back-edge cut")
        self.assertTrue(sched.back_edges, "global back-edge set populated")
        # back-edges reference real connections in this scope.
        conn_ids = {c.id for c in sim.to_ir("loop").root.connections}
        for be in sched.back_edges:
            self.assertIn(be, conn_ids)

    def test_opaque_block_event_surfaced(self):
        """A Scope with a sampling period is opaque, but its Schedule event is
        surfaced as an opaque event carrying the period/phase."""
        src = SinusoidalSource(1.0, 1.0, 0.0)
        sco = Scope(sampling_period=0.25)  # -> finite sampling period -> Schedule event
        sim = Simulation([src, sco], [Connection(src, sco)])
        m = sim.to_ir("opaque_evt")
        sco_b = next(b for b in m.blocks() if b.type_name == "Scope")
        self.assertTrue(sco_b.is_extern, "scope is opaque")
        self.assertEqual(len(sco_b.events), 1, "sampling event surfaced")
        e = sco_b.events[0]
        self.assertTrue(e.opaque)
        self.assertEqual(e.kind.kind, "Schedule")

    def test_json_roundtrip_with_events(self):
        src = SinusoidalSource(1.0, 1.0, 0.0)
        sco = Scope(sampling_period=0.25)
        sim = Simulation([src, sco], [Connection(src, sco)])
        m = sim.to_ir("evt")
        m2 = ir.Module.from_json(m.to_json())
        self.assertEqual(m.to_json(), m2.to_json())


if __name__ == "__main__":
    unittest.main()
