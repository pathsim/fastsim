//! `compile`: optional **static** system compilation.
//!
//! Bakes a whole connected model (via its stable [`ir::schema::Module`]) into a
//! single fused op-graph computing `dX/dt = F(X, t)` over one global continuous
//! state vector, then integrates it with the existing solver. This trades the
//! mutable per-block runtime for a flat tape: no per-block dispatch, no
//! block-boundary copies, and cross-block CSE — faster, but static (parameters
//! and structure are frozen at compile time).
//!
//! This is the opt-in counterpart to the per-block runtime, *not* a replacement:
//! the runtime stays the source of truth and supports everything (mutation,
//! opaque blocks, simulation-level events). The static path covers the
//! acyclic, fully op-expressible subset and rejects the rest with a clear
//! [`CompileError`] rather than silently mis-modelling it.
//!
//! Discrete and event-driven blocks *are* supported: a block's memory joins the
//! global discrete-memory vector `M` and its block-internal events (zero-cross,
//! schedule, condition) drive an event-aware run loop, writing back into `X`/`M`
//! (see `validate_flat` and the splicer in `splice.rs`).
//!
//! Sinks (Scope, recorders) are not part of the dynamics; instead the signals
//! they observe become recorded output *taps* (`tap_labels` / `record_taps`),
//! the static-compile equivalent of scope traces.
//!
//! Scope: flat models, nested subsystems (flattened), and discrete/event/memory
//! blocks expressible as ops. Rejected: opaque/extern blocks whose math lowers
//! to an `Op::Call` (RNG/noise, arbitrary Python/Rust callables, FMU, DAE);
//! algebraic loops; state-less models; simulation-level (global) events;
//! non-scalar parameters; and the event-effect writes the static path does not
//! model (`Write::Output` output latches and a `StateDeriv` write inside an
//! event effect).
//!
//! [`ir::schema::Module`]: crate::ir::schema::Module

mod splice;
mod runtime;
pub use runtime::CompiledSimulation;

use std::rc::Rc;

use crate::ir::schema::{Block, Module, Op};
use crate::ssa::tape::InterpretedFn;
use splice::{EvtKind, WriteTarget};

/// A block-internal event lowered for the compiled run loop: a guard scalar tape
/// (`ZeroCross`/`Condition`), an effect tape (writes), and where each effect
/// output goes in the global `X`/`M` vectors.
// Owned tapes (not `Rc`) so a `CompiledSimulation` is `Send` + `Clone` for the
// batch API — each run gets its own guard/effect scratch (issue #45).
#[derive(Clone)]
struct CompiledEvent {
    kind: EvtKind,
    guard: Option<InterpretedFn>,
    effect: InterpretedFn,
    targets: Vec<WriteTarget>,
}


/// Why a model could not be statically compiled. The per-block runtime handles
/// all of these; the static path is a deliberately narrow fast lane.
#[derive(Debug, Clone, PartialEq)]
pub enum CompileError {
    /// The model has an algebraic loop (a `schedule.sccs` entry). A flat tape is
    /// a DAG; loops need an inner fixed-point solve (future work).
    AlgebraicLoop,
    /// A non-sink block whose math is not op-expressible (RNG, arbitrary
    /// callable, FMU). Carries the block's `type_name`. (Sinks are not rejected;
    /// their observed signals become recorded taps.)
    OpaqueBlock(String),
    /// Simulation-level (global) events are present (opaque host actions).
    GlobalEvents,
    /// The model has no continuous state — nothing to integrate.
    NoState,
    /// Any other currently-unsupported construct, with a human-readable reason.
    Unsupported(String),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::AlgebraicLoop => write!(f, "model has an algebraic loop (not supported by static compile v1)"),
            CompileError::OpaqueBlock(n) => write!(f, "block '{n}' has no static op-graph (opaque: RNG/noise, arbitrary Python/Rust callable, or DAE) and cannot be compiled"),
            CompileError::GlobalEvents => write!(f, "model has simulation-level (global) events"),
            CompileError::NoState => write!(f, "model has no continuous state to integrate"),
            CompileError::Unsupported(r) => write!(f, "unsupported: {r}"),
        }
    }
}

impl std::error::Error for CompileError {}


/// Statically compile a model snapshot into a [`CompiledSimulation`].
pub fn compile(module: &Module) -> Result<CompiledSimulation, CompileError> {
    if !module.events.is_empty() {
        return Err(CompileError::GlobalEvents);
    }

    // Flatten nested subsystems into a single flat scope (blocks + connections),
    // then validate the static subset.
    let flat = crate::compile::flatten::flatten(module)?;
    validate_flat(&flat.blocks)?;
    if !flat.has_acyclic_schedule {
        return Err(CompileError::AlgebraicLoop);
    }

    let block_refs: Vec<&Block> = flat.blocks.iter().collect();
    let spliced = splice::splice(&block_refs, &flat.connections, None)?;

    let tap_nodes: Vec<u32> = spliced.taps.iter().map(|(_, n)| *n).collect();

    // Derivative graph (outputs = dX/dt), then its Jacobian ∂F/∂x via symbolic
    // autodiff for implicit solvers — both lowered to tapes.
    let mut deriv_graph = spliced.graph.clone();
    deriv_graph.outputs = spliced.deriv_outputs;
    // The symbolic Jacobian ∂F/∂x is the (O(n²)) bulk of compile, and only
    // implicit solvers ever use it — but the solver is chosen *after* compile
    // (`set_solver`). So retain the derivative graph and differentiate LAZILY on
    // first implicit use; explicit-solver runs never pay for it. Probe the slot
    // now (cheap) to keep the compile-time error for a state-less model.
    if deriv_graph.signature.slot("x").is_none() {
        return Err(CompileError::Unsupported("could not differentiate dX/dt wrt state".into()));
    }
    let jac_src = deriv_graph.clone();
    optimize_to_fixpoint(&mut deriv_graph);
    let fused = InterpretedFn::from_graph(deriv_graph);
    let taps = build_tape(&spliced.graph, tap_nodes);
    let tap_labels: Vec<String> = spliced.taps.into_iter().map(|(l, _)| l).collect();

    // One guard tape + one effect tape per block-internal event, over (x, m, t).
    let events: Vec<CompiledEvent> = spliced
        .events
        .into_iter()
        .map(|ce| CompiledEvent {
            guard: ce.guard_node.map(|n| build_tape(&spliced.graph, vec![n])),
            effect: build_tape(&spliced.graph, ce.effect_outputs),
            targets: ce.effect_targets,
            kind: ce.kind,
        })
        .collect();

    let x0 = spliced.x0;
    let m0 = spliced.m0;
    Ok(CompiledSimulation {
        fused,
        jac_src,
        jac: std::cell::RefCell::new(None),
        jac_const: std::cell::Cell::new(None),
        taps,
        tap_labels,
        n_state: spliced.n_state,
        n_mem: spliced.n_mem,
        x: x0.clone(),
        m: m0.clone(),
        rec_times: vec![0.0],
        rec_states: x0.clone(),
        rec_mems: m0.clone(),
        output_stride: 1,
        max_samples: None,
        accepted_steps: 0,
        x0,
        m0,
        state_labels: spliced.state_labels,
        events,
        dt: crate::constants::SIM_TIMESTEP,
        solver: "RKBS32".to_string(),
        atol: crate::constants::SOL_TOLERANCE_LTE_ABS,
        rtol: crate::constants::SOL_TOLERANCE_LTE_REL,
        time: 0.0,
        logger: crate::utils::logger::Logger::disabled(),
        rec_cache: std::cell::RefCell::new(None),
        last_stats: Default::default(),
    })
}

/// Run the op-graph optimizer to convergence (or the safety bound), in place.
fn optimize_to_fixpoint(g: &mut crate::ssa::graph::Graph) {
    for _ in 0..crate::constants::COMPILE_OPTIMIZE_MAX_PASSES {
        if !crate::ssa::optimize::optimize(g) {
            break;
        }
    }
}

/// Materialize a tight tape for a given output set from a shared graph: clone,
/// retarget outputs, run DCE/folding to convergence, lower to an `InterpretedFn`.
fn build_tape(graph: &crate::ssa::graph::Graph, outputs: Vec<u32>) -> InterpretedFn {
    let mut g = graph.clone();
    g.outputs = outputs;
    optimize_to_fixpoint(&mut g);
    InterpretedFn::from_graph(g)
}

/// Compile a [`Subsystem`] into a single fused runtime [`Block`] — a
/// `DynamicalSystem` with extras. The subsystem's interior is flattened and
/// spliced over `(x, m, u, t)`, yielding three tapes baked from one shared
/// graph: `dx/dt = f_dyn(x, u, t)`, the subsystem outputs `y = f_alg(x, u, t)`,
/// and an exact symbolic Jacobian `∂(dx/dt)/∂x`. The subsystem's block-internal
/// events (zero-crossings, schedules, conditions) are captured as the new
/// block's own events, sharing the block's discrete memory `m`. The result drops
/// straight into a per-block [`Simulation`] like any leaf block.
///
/// Rejects the same constructs as [`compile`] (algebraic loops, opaque blocks,
/// no continuous state) with a precise [`CompileError`].
///
/// [`Subsystem`]: crate::ir::schema::Subsystem
/// [`Block`]: crate::blocks::block::Block
/// [`Simulation`]: crate::simulation::Simulation
pub fn compile_block(
    sub: &crate::ir::schema::Subsystem,
) -> Result<crate::blocks::block::BlockRef, CompileError> {
    use crate::blocks::block::{Block as RtBlock, BlockRole as RtRole};
    use crate::ssa::graph::Node;
    use crate::solvers::solver::Solver;
    use crate::utils::fastcell::FastCell;
    use crate::utils::register::Register;

    // Flatten the subsystem interior (keeping its own interface live), validate
    // the static subset, then splice over (x, m, u, t).
    let flat = flatten::flatten_subsystem(sub)?;
    validate_flat(&flat.blocks)?;
    if !flat.has_acyclic_schedule {
        return Err(CompileError::AlgebraicLoop);
    }

    let block_refs: Vec<&Block> = flat.blocks.iter().collect();
    let iface = splice::IfaceCfg {
        n_in: flat.n_in,
        n_out: flat.n_out,
        output_srcs: flat.output_srcs,
    };
    let spliced = splice::splice(&block_refs, &flat.connections, Some(iface))?;

    let n_state = spliced.n_state;
    let n_mem = spliced.n_mem;
    let n_in = flat.n_in;
    let n_out = flat.n_out;

    // Derivative graph (dx/dt) and its symbolic Jacobian ∂F/∂x.
    let mut deriv_graph = spliced.graph.clone();
    deriv_graph.outputs = spliced.deriv_outputs.clone();
    // Symbolic Jacobian ∂F/∂x. Represent it SPARSELY (coordinate pattern +
    // values-only tape, sparse-LU Newton step) only when it is large enough,
    // genuinely sparse, AND state-dependent. Small, dense, or constant Jacobians
    // keep the dense path: a constant Jacobian factors once per dt through the
    // dense solver's byte-identical cache, which a per-step sparse rebuild would
    // forfeit, and tiny/dense matrices are faster dense than through faer-sparse.
    // A structurally all-zero Jacobian carries no analytical information, so
    // `jac_dyn` stays unset ("no Jacobian").
    let dyn_tape = build_tape(&spliced.graph, spliced.deriv_outputs.clone());

    // Output (algebraic) graph: detect real u-feedthrough on the DCE'd graph so
    // the block declares passthrough only when an output actually reads `u`
    // (avoids spurious algebraic-loop edges in the enclosing system).
    let mut alg_graph = spliced.graph.clone();
    alg_graph.outputs = spliced.iface_outputs.clone();
    optimize_to_fixpoint(&mut alg_graph);
    let u_lo = (n_state + n_mem) as u32;
    let u_hi = u_lo + n_in as u32;
    let has_feedthrough = alg_graph
        .nodes
        .iter()
        .any(|nd| matches!(nd, Node::Input(i) if *i >= u_lo && *i < u_hi));
    let alg_tape = InterpretedFn::from_graph(alg_graph);

    let x0 = spliced.x0.clone();
    let m0 = spliced.m0.clone();
    let m_cell: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(m0.clone()));

    // -- Assemble the runtime block --
    let mut b = RtBlock::new(None, None);
    b.type_name = "CompiledSubsystem";
    b.inputs = Register::new(Some(n_in.max(1)), None);
    b.outputs = Register::new(Some(n_out.max(1)), None);
    b.role = RtRole { is_dyn: true, is_src: false, is_rec: false };
    // Opaque fused block (tapes, no SSA `ops`): declare its u->y feedthrough.
    b.opaque_feedthrough = has_feedthrough;
    b.initial_value = Some(x0.clone());
    b.engine = Some(Solver::with_defaults(&x0));
    let len_val = if has_feedthrough { 1 } else { 0 };
    b.len_fn = Some(Box::new(move |_| len_val));

    // dx/dt = f_dyn(x, u, t) over the live discrete memory m.
    {
        let m = m_cell.clone();
        b.f_dyn = Some(Box::new(move |x, u, t, out| {
            let m = m.borrow();
            out.extend(dyn_tape.call(&[x, &m[..], u, &[t]]));
        }));
    }
    // ∂(dx/dt)/∂x via the dynamic-path Operator: the AD Jacobian (sparse-aware,
    // gated exactly like the interpreted and CompiledSimulation paths) is derived
    // from the derivative graph; the live discrete-memory cell is read as a fixed
    // input at solve time. Replaces the old detached `jac_dyn` + `jac_pattern`.
    b.dyn_op = Some(crate::blocks::operator::Operator::jac_only(
        crate::blocks::blockops::RegionGraph::Fixed(deriv_graph),
    ));
    b.mem_ref = Some(m_cell.clone());
    // y = f_alg(x, u, t) — the subsystem outputs.
    if n_out > 0 {
        let m = m_cell.clone();
        b.f_alg = Some(Box::new(move |x, u, t, out| {
            let m = m.borrow();
            out.extend(alg_tape.call(&[x, &m[..], u, &[t]]));
        }));
    }

    let blk_ref = Rc::new(FastCell::new(b));

    // Block-internal events: guards read the live (x, m, u, t); effects write the
    // discrete memory m and/or reset continuous state x.
    install_compiled_events(&blk_ref, &m_cell, &spliced.graph, &spliced.events);

    // Reset: rewind discrete memory to m0 (engine + events handled by Block::reset).
    {
        let m = m_cell.clone();
        let m0r = m0.clone();
        blk_ref.borrow_mut().reset_fn = Some(Box::new(move |blk| {
            blk.inputs.reset();
            blk.outputs.reset();
            if let (Some(eng), Some(iv)) = (blk.engine.as_mut(), blk.initial_value.as_ref()) {
                eng.reset(Some(iv));
            }
            let mm = m.borrow_mut();
            mm.clear();
            mm.extend_from_slice(&m0r);
        }));
    }

    Ok(blk_ref)
}

/// Assemble one runtime `SimEvent` from a compiled event's `kind`, a lazily-built
/// guard evaluator (invoked only for ZeroCross/Condition), and an effect-action
/// factory. Both event-lowering paths — whole-system [`CompiledSimulation::build_sim_events`]
/// and the per-block [`install_compiled_events`] — share this dispatch and differ
/// only in how guard/effect read and write state, captured by the two closures.
fn assemble_sim_event(
    kind: &EvtKind,
    make_guard: impl FnOnce() -> Box<dyn Fn(f64) -> f64>,
    make_act: impl FnOnce() -> Box<dyn FnMut(f64)>,
) -> crate::simulation::SimEventRef {
    use crate::constants::EVT_TOLERANCE;
    use crate::events::condition::Condition;
    use crate::events::schedule::{Schedule, ScheduleList};
    use crate::events::zerocrossing::{CrossingDirection, ZeroCrossing};
    use crate::ir::schema::{Direction, ScheduleTimes};
    use crate::utils::fastcell::FastCell;

    match kind {
        EvtKind::ZeroCross(dir) => {
            let guard = make_guard();
            let direction = match dir {
                Direction::Rising => CrossingDirection::Up,
                Direction::Falling => CrossingDirection::Down,
                Direction::Both => CrossingDirection::Both,
            };
            Rc::new(FastCell::new(ZeroCrossing::with_direction(
                direction, guard, Some(make_act()), EVT_TOLERANCE,
            )))
        }
        EvtKind::Condition => {
            let guard = make_guard();
            Rc::new(FastCell::new(Condition::new(
                move |t| guard(t) != 0.0, Some(make_act()), EVT_TOLERANCE,
            )))
        }
        EvtKind::Schedule(times) => match times {
            ScheduleTimes::Periodic { period, phase } => Rc::new(FastCell::new(
                Schedule::new(*phase, None, *period, Some(make_act()), EVT_TOLERANCE),
            )),
            ScheduleTimes::Fixed(ts) => Rc::new(FastCell::new(ScheduleList::new(
                ts.clone(), Some(make_act()), EVT_TOLERANCE,
            ))),
        },
    }
}

/// Build runtime `SimEvent`s for a compiled subsystem block and push them onto
/// the block. Guard/effect tapes read the *live* block state: `x` from the
/// engine, `u` from the input register, `m` from the shared memory cell. Mirrors
/// [`CompiledSimulation::build_sim_events`], but bound to a runtime block instead
/// of a standalone `(x, m)` scratch.
fn install_compiled_events(
    blk_ref: &crate::blocks::block::BlockRef,
    m_cell: &Rc<crate::utils::fastcell::FastCell<Vec<f64>>>,
    graph: &crate::ssa::graph::Graph,
    specs: &[splice::CompiledEventSpec],
) {
    use crate::utils::fastcell::FastCell;

    // Read the live (x, u) out of the block, plus m, and evaluate `tape` at t.
    fn eval_live(
        blk: &crate::blocks::block::BlockRef,
        m_cell: &Rc<FastCell<Vec<f64>>>,
        tape: &InterpretedFn,
        t: f64,
    ) -> Vec<f64> {
        let b = blk.borrow();
        let x: &[f64] = b.engine.as_ref().map(|e| e.get()).unwrap_or(&[]);
        let u: &[f64] = &b.inputs._data;
        let m = m_cell.borrow();
        tape.call(&[x, &m[..], u, &[t]])
    }

    let mut events = Vec::with_capacity(specs.len());
    for ce in specs {
        // Guard closure: eval the guard tape at the live block state.
        let make_guard = || -> Box<dyn Fn(f64) -> f64> {
            let guard = Rc::new(build_tape(
                graph,
                vec![ce.guard_node.expect("ZeroCross/Condition has a guard")],
            ));
            let blk = blk_ref.clone();
            let mc = m_cell.clone();
            Box::new(move |t: f64| eval_live(&blk, &mc, &guard, t)[0])
        };
        // Effect closure: eval at the live (pre-write) state, then apply writes
        // to m (memory) and x (discrete state reset).
        let make_act = || -> Box<dyn FnMut(f64)> {
            let eff = Rc::new(build_tape(graph, ce.effect_outputs.clone()));
            let targets = ce.effect_targets.clone();
            let blk = blk_ref.clone();
            let mc = m_cell.clone();
            Box::new(move |t: f64| {
                let outs = eval_live(&blk, &mc, &eff, t);
                let mut state_writes: Vec<(usize, f64)> = Vec::new();
                {
                    let m = mc.borrow_mut();
                    for (val, tgt) in outs.iter().zip(targets.iter()) {
                        match tgt {
                            splice::WriteTarget::Mem(i) => m[*i] = *val,
                            splice::WriteTarget::State(i) => state_writes.push((*i, *val)),
                        }
                    }
                }
                if !state_writes.is_empty() {
                    let b = blk.borrow_mut();
                    if let Some(eng) = b.engine.as_mut() {
                        let mut xs = eng.get().to_vec();
                        for (i, v) in state_writes {
                            xs[i] = v;
                        }
                        eng.set(&xs);
                    }
                }
            })
        };
        events.push(assemble_sim_event(&ce.kind, make_guard, make_act));
    }
    blk_ref.borrow_mut().events = events;
}

/// Reject blocks the static path cannot model, with a precise reason. Pure
/// sinks (Scope, recorders) are *skipped*: they have no outputs feeding the ODE,
/// so they contribute nothing to `dX/dt` and their sampling event is irrelevant
/// to the continuous dynamics. Memory / event blocks (discrete, relay,
/// comparator, ...) are now supported: their memory joins the global `M` vector
/// and their events drive the event-aware run loop.
fn validate_flat(blocks: &[Block]) -> Result<(), CompileError> {
    use crate::ir::schema::BlockRole;
    for b in blocks {
        if b.role == BlockRole::Sink {
            continue;
        }
        // Opaque math (RNG, FMU, untraceable callables) appears as a `Call` op in
        // any region; the splicer also rejects `Call` inside event guards/effects.
        let has_call = b
            .regions
            .alg
            .ops
            .iter()
            .chain(b.regions.dyn_.ops.iter())
            .any(|o| matches!(o, Op::Call { .. }));
        if has_call {
            return Err(CompileError::OpaqueBlock(b.type_name.clone()));
        }
    }
    Ok(())
}

mod flatten;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::block::BlockRef;
    use crate::blocks::constructors::{amplifier, integrator, sample_hold, scope, sinusoidal_source};
    use crate::connection::Connection as RtConn;
    use crate::ir::builder::module_from_sim;
    use crate::simulation::Simulation;
    use crate::utils::portreference::PortReference;
    use std::rc::Rc;

    fn conn(a: &BlockRef, b: &BlockRef) -> Rc<RtConn> {
        Rc::new(RtConn::new(
            PortReference::new(a.clone(), None),
            vec![PortReference::new(b.clone(), None)],
        ))
    }

    /// Harmonic oscillator x1'=x2, x2'=-x1: i1(x1) <- i2(x2), i1 -> amp(-1) -> i2.
    fn oscillator() -> CompiledSimulation {
        let i1 = integrator(1.0); // x1(0) = 1
        let i2 = integrator(0.0); // x2(0) = 0
        let amp = amplifier(-1.0);
        let mut sim = Simulation::with_defaults(
            vec![i1.clone(), i2.clone(), amp.clone()],
            vec![conn(&i2, &i1), conn(&i1, &amp), conn(&amp, &i2)],
        );
        sim.run(0.01, false, false); // assemble (resolves shape-poly widths)
        compile(&module_from_sim(&sim, "osc")).expect("oscillator compiles")
    }

    #[test]
    fn pointwise_oscillator_rhs() {
        let c = oscillator();
        assert_eq!(c.n_state, 2);
        assert_eq!(c.x0, vec![1.0, 0.0]);
        // F([a,b]) = [b, -a]
        for (x, expect) in [
            ([1.0, 0.0], [0.0, -1.0]),
            ([0.3, 0.7], [0.7, -0.3]),
            ([-2.0, 5.0], [5.0, 2.0]),
        ] {
            let f = c.deriv(&x, 0.0);
            assert!((f[0] - expect[0]).abs() < 1e-12, "dx1: {f:?} vs {expect:?}");
            assert!((f[1] - expect[1]).abs() < 1e-12, "dx2: {f:?} vs {expect:?}");
        }
    }

    #[test]
    fn trajectory_oscillator_matches_analytic() {
        // x1(t) = cos t, x2(t) = -sin t.
        let mut c = oscillator();
        c.dt = 0.01;
        c.run(std::f64::consts::PI, true, true);
        let tf = *c.times().last().unwrap();
        let states = c.states();
        let xf = states.last().unwrap();
        assert!((tf - std::f64::consts::PI).abs() < 1e-6, "t_end {tf}");
        assert!((xf[0] - tf.cos()).abs() < 2e-3, "x1 {} vs {}", xf[0], tf.cos());
        assert!((xf[1] - (-tf.sin())).abs() < 2e-3, "x2 {} vs {}", xf[1], -tf.sin());
    }

    #[test]
    fn pointwise_and_trajectory_exponential_decay() {
        // x' = -2 x, x(0) = 1 -> x(t) = exp(-2t).
        let i = integrator(1.0);
        let k = amplifier(-2.0);
        let mut sim = Simulation::with_defaults(
            vec![i.clone(), k.clone()],
            vec![conn(&i, &k), conn(&k, &i)],
        );
        sim.run(0.01, false, false);
        let mut c = compile(&module_from_sim(&sim, "decay")).unwrap();
        assert_eq!(c.n_state, 1);
        // F([a]) = [-2a]
        assert!((c.deriv(&[1.0], 0.0)[0] - (-2.0)).abs() < 1e-12);
        assert!((c.deriv(&[3.5], 0.0)[0] - (-7.0)).abs() < 1e-12);
        // trajectory
        c.dt = 0.01;
        c.run(1.0, true, true);
        let tf = *c.times().last().unwrap();
        let xf = c.states().last().unwrap()[0];
        assert!((xf - (-2.0 * tf).exp()).abs() < 2e-3, "x {xf} vs {}", (-2.0 * tf).exp());
    }

    /// Issue #44: flat trajectory storage is consistent (rows reconstruct the
    /// flat buffer), and `output_stride` downsamples while always keeping the
    /// initial point and the true final state (so re-runs stay continuous).
    #[test]
    fn flat_storage_and_output_stride_downsampling() {
        let i = integrator(1.0);
        let k = amplifier(-2.0);
        let sim = Simulation::with_defaults(
            vec![i.clone(), k.clone()],
            vec![conn(&i, &k), conn(&k, &i)],
        );
        // Full trajectory (stride 1).
        let mut c = compile(&module_from_sim(&sim, "decay")).unwrap();
        c.dt = 0.01;
        c.run(1.0, true, false);
        let full = c.states();
        // Flat buffer is row-major with stride n_state, parallel to times.
        assert_eq!(c.states_flat().len(), c.times().len() * c.n_state);
        assert_eq!(full.len(), c.times().len());
        let full_final = *full.last().unwrap().first().unwrap();
        let full_final_t = *c.times().last().unwrap();

        // Downsample: keep 1 of every 5 accepted steps.
        c.set_output_stride(5);
        c.run(1.0, true, false);
        let ds = c.states();
        assert_eq!(c.states_flat().len(), c.times().len() * c.n_state);
        // Far fewer samples than the full run, but not empty.
        assert!(ds.len() < full.len() / 2 && ds.len() >= 2, "downsampled len {}", ds.len());
        // Endpoints preserved: same initial and (bit-exact) final state/time as
        // the full run — the deterministic stepping is unchanged, only recording.
        assert_eq!(*ds.first().unwrap().first().unwrap(), 1.0);
        assert_eq!(*ds.last().unwrap().first().unwrap(), full_final);
        assert_eq!(*c.times().last().unwrap(), full_final_t);
    }

    #[test]
    fn flattens_nested_subsystem() {
        // Outer integrator i feeds a subsystem S = amp(-0.5); S feeds i back.
        // After flattening: i' = -0.5*i  ->  x(t) = exp(-0.5 t).
        use crate::subsystem::{interface, subsystem};
        let iface = interface();
        let inner_amp = amplifier(-0.5);
        let s = subsystem(
            vec![iface.clone(), inner_amp.clone()],
            vec![conn(&iface, &inner_amp), conn(&inner_amp, &iface)],
            10,
        )
        .unwrap();
        let i = integrator(1.0);
        let mut sim = Simulation::with_defaults(
            vec![i.clone(), s.clone()],
            vec![conn(&i, &s), conn(&s, &i)],
        );
        sim.run(0.01, false, false);

        let mut c = compile(&module_from_sim(&sim, "nested_decay")).expect("nested model compiles");
        assert_eq!(c.n_state, 1, "one integrator state after flattening");
        // F([a]) = [-0.5 a] — the inner amplifier was inlined.
        assert!((c.deriv(&[1.0], 0.0)[0] - (-0.5)).abs() < 1e-12);
        assert!((c.deriv(&[4.0], 0.0)[0] - (-2.0)).abs() < 1e-12);
        // trajectory vs analytic exp(-0.5 t)
        c.dt = 0.01;
        c.run(2.0, true, true);
        let tf = *c.times().last().unwrap();
        let xf = c.states().last().unwrap()[0];
        assert!((xf - (-0.5 * tf).exp()).abs() < 2e-3, "x {xf} vs {}", (-0.5 * tf).exp());
    }

    #[test]
    fn compiled_x0_is_initial_state() {
        // Global state vector x0 reflects the blocks' construction-time initials
        // in flatten (block) order.
        let i1 = integrator(1.0);
        let i2 = integrator(0.0);
        let amp = amplifier(-1.0);
        let mut sim = Simulation::with_defaults(
            vec![i1.clone(), i2.clone(), amp.clone()],
            vec![conn(&i2, &i1), conn(&i1, &amp), conn(&amp, &i2)],
        );
        sim.run(0.01, false, false);
        let c = compile(&module_from_sim(&sim, "osc")).unwrap();
        assert_eq!(c.x0, vec![1.0, 0.0]);
        assert_eq!(c.state_labels.len(), 2);
    }

    #[test]
    fn nested_state_layout() {
        use crate::subsystem::{interface, subsystem};
        // outer integrator (init 1.0) + subsystem's inner integrator (init 3.0).
        // Flatten DFS order: outer first, then the inner -> x0 = [1.0, 3.0].
        let iface = interface();
        let inner_i = integrator(3.0);
        let s = subsystem(
            vec![iface.clone(), inner_i.clone()],
            vec![conn(&iface, &inner_i), conn(&inner_i, &iface)],
            10,
        )
        .unwrap();
        let outer = integrator(1.0);
        let mut sim = Simulation::with_defaults(
            vec![outer.clone(), s.clone()],
            vec![conn(&outer, &s), conn(&s, &outer)],
        );
        sim.run(0.01, false, false);
        let c = compile(&module_from_sim(&sim, "nested")).unwrap();
        assert_eq!(c.n_state, 2, "outer + inner integrator");
        assert_eq!(c.x0, vec![1.0, 3.0], "outer then inner, construction initials");
    }

    #[test]
    fn equivalence_compiled_vs_per_block() {
        // Same model, same solver (SSPRK22), same fixed dt: the fused tape and
        // the per-block runtime must integrate to the same state.
        let i1 = integrator(1.0);
        let i2 = integrator(0.0);
        let amp = amplifier(-1.0);
        let mut sim = Simulation::with_defaults(
            vec![i1.clone(), i2.clone(), amp.clone()],
            vec![conn(&i2, &i1), conn(&i1, &amp), conn(&amp, &i2)],
        );
        let dt = sim.dt;
        let dur = 1.0;
        let mut c = compile(&module_from_sim(&sim, "osc")).unwrap();

        sim.run(dur, false, false); // per-block; integrator output == its state
        let pb_x1 = i1.borrow().outputs.get_single(0);
        let pb_x2 = i2.borrow().outputs.get_single(0);

        // Match the per-block solver/dt, then compare the compiled sample closest
        // to `dur` (integrate overshoots t_end by up to one fixed step).
        c.solver = "SSPRK22".to_string();
        c.dt = dt;
        c.run(dur, true, false);
        let idx = c
            .times()
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (**a - dur).abs().total_cmp(&(**b - dur).abs()))
            .map(|(i, _)| i)
            .unwrap();
        let states = c.states();
        let xf = &states[idx];
        assert!((xf[0] - pb_x1).abs() < 1e-6, "x1 compiled {} vs per-block {pb_x1}", xf[0]);
        assert!((xf[1] - pb_x2).abs() < 1e-6, "x2 compiled {} vs per-block {pb_x2}", xf[1]);
    }

    #[test]
    fn accepts_model_with_sink() {
        // A scope (sink) is skipped: x' = -x, x(0)=1, scope just observes.
        let i = integrator(1.0);
        let amp = amplifier(-1.0);
        let sco = scope(Some(0.1), 0.0, vec![]); // sampling event -> must be ignored
        let mut sim = Simulation::with_defaults(
            vec![i.clone(), amp.clone(), sco.clone()],
            vec![conn(&i, &amp), conn(&amp, &i), conn(&i, &sco)],
        );
        sim.run(0.01, false, false);
        let mut c = compile(&module_from_sim(&sim, "withsink")).expect("sink is recorded, model compiles");
        assert_eq!(c.n_state, 1);
        assert!((c.deriv(&[2.0], 0.0)[0] - (-2.0)).abs() < 1e-12);
        // the scope observes the integrator output (== state x): one tap that
        // records x along the trajectory.
        assert_eq!(c.tap_labels().len(), 1, "scope input -> one recorded signal");
        assert_eq!(c.eval_taps(&[2.5], 0.0), vec![2.5], "tap == observed signal");
        c.dt = 0.01;
        c.run(1.0, true, true);
        let rec = c.recordings();
        assert_eq!(rec.len(), 1);
        assert_eq!(rec[0].len(), c.times().len());
        // recorded signal equals the state trajectory (scope watches x).
        let states = c.states();
        for (r, s) in rec[0].iter().zip(states.iter()) {
            assert!((r - s[0]).abs() < 1e-12);
        }
    }

    #[test]
    fn implicit_solver_uses_native_jacobian() {
        // x' = -2 x, x(0)=1 -> exp(-2t), integrated with an implicit ESDIRK
        // method (exercises the compile-time symbolic Jacobian path).
        let i = integrator(1.0);
        let k = amplifier(-2.0);
        let mut sim = Simulation::with_defaults(
            vec![i.clone(), k.clone()],
            vec![conn(&i, &k), conn(&k, &i)],
        );
        sim.run(0.01, false, false);
        let mut c = compile(&module_from_sim(&sim, "decay")).unwrap();
        c.set_solver("ESDIRK43", 1e-9, 1e-7);
        c.run(1.0, true, true);
        let tf = *c.times().last().unwrap();
        let xf = c.states().last().unwrap()[0];
        assert!((xf - (-2.0 * tf).exp()).abs() < 1e-4, "implicit x {xf} vs {}", (-2.0 * tf).exp());
    }

    #[test]
    fn parameters_stay_live_after_compile() {
        // amp gain is a baked Param: retuning it changes the dynamics without
        // recompiling. decay i' = gain*i.
        let i = integrator(1.0);
        let amp = amplifier(-1.0);
        let mut sim = Simulation::with_defaults(
            vec![i.clone(), amp.clone()],
            vec![conn(&i, &amp), conn(&amp, &i)],
        );
        sim.run(0.01, false, false);
        let mut c = compile(&module_from_sim(&sim, "decay")).unwrap();
        assert!((c.deriv(&[1.0], 0.0)[0] - (-1.0)).abs() < 1e-12);
        // find the gain param label and retune to -3.0
        let gain = c.param_names().iter().find(|n| n.contains("gain")).cloned().expect("gain param");
        assert!(c.set_param(&gain, -3.0));
        assert!((c.deriv(&[1.0], 0.0)[0] - (-3.0)).abs() < 1e-12, "retuned gain takes effect");
    }

    /// Issue #45: the parallel batch API runs one parameter set per thread over
    /// independent clones and returns results bit-identical to running the sweep
    /// serially (deterministic, no shared state).
    #[cfg(feature = "parallel")]
    #[test]
    fn run_batch_matches_serial_sweep() {
        // i' = gain*i, i(0)=1 -> i(t) = exp(gain*t). Sweep the gain.
        let i = integrator(1.0);
        let amp = amplifier(-1.0);
        let sim = Simulation::with_defaults(
            vec![i.clone(), amp.clone()],
            vec![conn(&i, &amp), conn(&amp, &i)],
        );
        let mut base = compile(&module_from_sim(&sim, "decay")).unwrap();
        base.dt = 0.01;
        let gain = base.param_names().iter().find(|n| n.contains("gain")).cloned().unwrap();

        let gains = [-0.5_f64, -1.0, -2.0, -3.0, -4.0, -5.0, -0.25, -1.5];
        let param_sets: Vec<Vec<(String, f64)>> =
            gains.iter().map(|&g| vec![(gain.clone(), g)]).collect();

        // Serial reference.
        let mut serial: Vec<Vec<f64>> = Vec::new();
        for ps in &param_sets {
            let mut c = base.clone();
            for (n, v) in ps { c.set_param(n, *v); }
            c.run(1.0, true, true);
            serial.push(c.state().to_vec());
        }

        // Parallel batch — must match the serial finals bit-for-bit.
        let batch = base.run_batch(&param_sets, 1.0, true);
        assert_eq!(batch.len(), gains.len());
        for (k, (b, s)) in batch.iter().zip(serial.iter()).enumerate() {
            assert_eq!(b, s, "batch run {k} (gain {}) differs from serial", gains[k]);
        }
        // Sanity: monotonic in gain magnitude (faster decay -> smaller final).
        assert!(batch[4][0] < batch[0][0], "steeper decay should end lower");
    }

    #[test]
    fn rejects_algebraic_loop() {
        // src -> err(+-) -> kp(0.5) -> err : purely algebraic loop.
        let src = sinusoidal_source(1.0, 1.0, 0.0);
        let err = crate::blocks::constructors::adder(Some("+-"));
        let kp = amplifier(0.5);
        let mut sim = Simulation::with_defaults(
            vec![src.clone(), err.clone(), kp.clone()],
            vec![conn(&src, &err), conn(&err, &kp), conn(&kp, &err)],
        );
        sim.run(0.01, false, false);
        match compile(&module_from_sim(&sim, "loop")) {
            Err(CompileError::AlgebraicLoop) => {}
            other => panic!("expected AlgebraicLoop, got {:?}", other.err()),
        }
    }

    #[test]
    fn zerocross_relay_matches_per_block() {
        // Oscillator x1'=x2, x2'=-x1 drives a relay (zero-crossing hysteresis);
        // the relay output feeds an integrator. The compiled event loop must
        // locate the crossings and match the per-block runtime.
        use crate::blocks::constructors::relay;
        let i1 = integrator(1.0);
        let i2 = integrator(0.0);
        let amp = amplifier(-1.0);
        let r = relay(0.0, 0.0, 1.0, -1.0); // switches on x1 sign
        let i3 = integrator(0.0); // integrates the relay output
        let mut sim = Simulation::with_defaults(
            vec![i1.clone(), i2.clone(), amp.clone(), r.clone(), i3.clone()],
            vec![conn(&i2, &i1), conn(&i1, &amp), conn(&amp, &i2), conn(&i1, &r), conn(&r, &i3)],
        );
        let mut c = compile(&module_from_sim(&sim, "relay")).expect("relay compiles");
        assert!(!c.events.is_empty(), "relay contributes zero-crossing events");

        let dur = 3.0;
        sim.run(dur, false, true);
        let pb = i3.borrow().outputs.get_single(0);

        c.run(dur, true, true);
        let xf = c.states().last().unwrap()[2]; // i3 state (3rd integrator)
        assert!((xf - pb).abs() < 5e-2, "compiled {xf} vs per-block {pb}");
    }

    #[test]
    fn tableau_less_implicit_with_events_runs() {
        // A tableau-less implicit solver (GEAR52A) routes through `take_step`'s
        // no-tableau path in the event loop. Smoke test: a discrete model runs
        // and stays finite.
        let src = sinusoidal_source(1.0, 1.0, 0.0);
        let sh = sample_hold(0.1, 0.0);
        let i = integrator(0.0);
        let sim = Simulation::with_defaults(
            vec![src.clone(), sh.clone(), i.clone()],
            vec![conn(&src, &sh), conn(&sh, &i)],
        );
        let mut c = compile(&module_from_sim(&sim, "disc")).unwrap();
        c.set_solver("GEAR52A", 1e-7, 1e-5);
        c.dt = 0.01;
        c.run(1.0, true, true);
        let xf = c.states().last().unwrap()[0];
        assert!(xf.is_finite() && xf.abs() < 10.0, "GEAR52A+events finite: {xf}");
    }

    #[test]
    fn implicit_solver_with_events_matches_per_block() {
        // Same discrete model as the explicit test, but integrated with an
        // implicit ESDIRK method — event handling is identical, only the inner
        // stage step differs (Newton vs explicit `step`).
        let src = sinusoidal_source(1.0, 1.0, 0.0);
        let sh = sample_hold(0.1, 0.0);
        let i = integrator(0.0);
        let mut sim = Simulation::with_defaults(
            vec![src.clone(), sh.clone(), i.clone()],
            vec![conn(&src, &sh), conn(&sh, &i)],
        );
        let dt = sim.dt;
        let mut c = compile(&module_from_sim(&sim, "disc")).unwrap();

        let dur = 1.0;
        sim.run(dur, false, false);
        let pb = i.borrow().outputs.get_single(0);

        c.set_solver("ESDIRK43", 1e-9, 1e-7);
        c.dt = dt;
        c.run(dur, true, false);
        let idx = c
            .times()
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (**a - dur).abs().total_cmp(&(**b - dur).abs()))
            .map(|(i, _)| i)
            .unwrap();
        let xf = c.states()[idx][0];
        assert!((xf - pb).abs() < 1e-4, "implicit+events {xf} vs per-block {pb}");
    }

    #[test]
    fn discrete_block_matches_per_block() {
        // src -> sample_hold(0.1) -> integrator. The sample-hold's Schedule event
        // updates a memory slot; the integrator integrates the staircase. The
        // compiled event loop must match the per-block runtime (same solver/dt).
        let src = sinusoidal_source(1.0, 1.0, 0.0);
        let sh = sample_hold(0.1, 0.0);
        let i = integrator(0.0);
        let mut sim = Simulation::with_defaults(
            vec![src.clone(), sh.clone(), i.clone()],
            vec![conn(&src, &sh), conn(&sh, &i)],
        );
        let dt = sim.dt;
        let mut c = compile(&module_from_sim(&sim, "disc")).expect("discrete block compiles");
        assert!(c.n_mem >= 1, "sample-hold contributes a memory slot");

        let dur = 1.0;
        sim.run(dur, false, false);
        let pb = i.borrow().outputs.get_single(0);

        c.solver = "SSPRK22".to_string();
        c.dt = dt;
        c.run(dur, true, false);
        let idx = c
            .times()
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (**a - dur).abs().total_cmp(&(**b - dur).abs()))
            .map(|(i, _)| i)
            .unwrap();
        let xf = c.states()[idx][0];
        assert!((xf - pb).abs() < 1e-6, "compiled {xf} vs per-block {pb}");
    }

    /// Extract the (single) `Child::Subsystem` IR node from an assembled sim.
    fn extract_subsystem(sim: &Simulation, name: &str) -> crate::ir::schema::Subsystem {
        use crate::ir::schema::Child;
        module_from_sim(sim, name)
            .root
            .children
            .into_iter()
            .find_map(|c| match c {
                Child::Subsystem(s) => Some(s),
                _ => None,
            })
            .expect("subsystem child present")
    }

    #[test]
    fn compiles_subsystem_to_block() {
        // Subsystem = first-order lag  dx/dt = u - x,  y = x. Compiling it to a
        // block and dropping it into a fresh sim must match the per-block
        // subsystem driven by the same input (and the analytic 1 - e^{-t}).
        use crate::blocks::constructors::{adder, constant};
        use crate::subsystem::{interface, subsystem};
        use crate::utils::portreference::Port;

        // i -> err input port 1 (the `- x` term); err input port 0 is `u`.
        let conn_to = |a: &crate::blocks::block::BlockRef, b: &crate::blocks::block::BlockRef, p: usize| {
            Rc::new(RtConn::new(
                PortReference::new(a.clone(), None),
                vec![PortReference::new(b.clone(), Some(vec![Port::Index(p)]))],
            ))
        };
        let lag = || {
            let iface = interface();
            let err = adder(Some("+-")); // u - x
            let i = integrator(0.0); // x' = (u - x), y = x
            subsystem(
                vec![iface.clone(), err.clone(), i.clone()],
                vec![conn(&iface, &err), conn_to(&i, &err, 1), conn(&err, &i), conn(&i, &iface)],
                10,
            )
            .unwrap()
        };

        // Assemble once to get the IR subsystem, then compile it to a block.
        let u0 = constant(1.0);
        let sub0 = lag();
        let sco0 = scope(None, 0.0, vec![]);
        let mut asm = Simulation::with_defaults(
            vec![u0.clone(), sub0.clone(), sco0.clone()],
            vec![conn(&u0, &sub0), conn(&sub0, &sco0)],
        );
        asm.run(0.01, false, false);
        let sub_ir = extract_subsystem(&asm, "lag");
        let block = compile_block(&sub_ir).expect("subsystem compiles to a block");

        let dur = 2.0;
        // Reference: per-block subsystem.
        let ur = constant(1.0);
        let subr = lag();
        let scor = scope(None, 0.0, vec![]);
        let mut ref_sim = Simulation::with_defaults(
            vec![ur.clone(), subr.clone(), scor.clone()],
            vec![conn(&ur, &subr), conn(&subr, &scor)],
        );
        ref_sim.run(dur, false, false);
        let y_ref = subr.borrow().outputs.get_single(0);

        // Compiled block in a fresh sim (same default solver / dt).
        let uc = constant(1.0);
        let scoc = scope(None, 0.0, vec![]);
        let mut cmp_sim = Simulation::with_defaults(
            vec![uc.clone(), block.clone(), scoc.clone()],
            vec![conn(&uc, &block), conn(&block, &scoc)],
        );
        cmp_sim.run(dur, false, false);
        let y_cmp = block.borrow().outputs.get_single(0);

        assert!((y_cmp - y_ref).abs() < 1e-5, "compiled {y_cmp} vs per-block {y_ref}");
        let analytic = 1.0 - (-(ref_sim.time)).exp();
        assert!((y_cmp - analytic).abs() < 1e-2, "compiled {y_cmp} vs analytic {analytic}");
    }

    #[test]
    fn compiles_two_state_subsystem() {
        // Damped harmonic oscillator subsystem (force u -> position x):
        //   v' = u - w^2 x - 2 z w v,  x' = v.  Two states + internal feedback.
        use crate::blocks::constructors::{adder, constant};
        use crate::subsystem::{interface, subsystem};
        use crate::utils::portreference::Port;

        let conn_to = |a: &BlockRef, b: &BlockRef, p: usize| {
            Rc::new(RtConn::new(
                PortReference::new(a.clone(), None),
                vec![PortReference::new(b.clone(), Some(vec![Port::Index(p)]))],
            ))
        };
        let (w, zeta) = (2.0_f64, 0.1_f64);
        let plant = || {
            let iface = interface();
            let acc = adder(Some("+--")); // u - w^2 x - 2 z w v
            let v = integrator(0.0);
            let x = integrator(0.0);
            let kx = amplifier(w * w);
            let kv = amplifier(2.0 * zeta * w);
            subsystem(
                vec![iface.clone(), acc.clone(), v.clone(), x.clone(), kx.clone(), kv.clone()],
                vec![
                    conn(&iface, &acc),
                    conn_to(&kx, &acc, 1),
                    conn_to(&kv, &acc, 2),
                    conn(&acc, &v),
                    conn(&v, &x),
                    conn(&x, &kx),
                    conn(&v, &kv),
                    conn(&x, &iface),
                ],
                10,
            )
            .unwrap()
        };

        // Compile straight after construction, with NO assembling run. The
        // subsystem resolves its internal port widths in `assemble_graph` at
        // construction, so the shape-poly Adder is captured at width 3 here;
        // before that fix this silently dropped the feedback terms.
        let sub0 = plant();
        let sub_ir = crate::ir::builder::subsystem_to_ir(&sub0).expect("subsystem ir");
        let block = compile_block(&sub_ir).expect("two-state subsystem compiles");

        let dur = 6.0;
        let ur = constant(1.0);
        let subr = plant();
        let scor = scope(None, 0.0, vec![]);
        let mut ref_sim = Simulation::with_defaults(
            vec![ur.clone(), subr.clone(), scor.clone()],
            vec![conn(&ur, &subr), conn(&subr, &scor)],
        );
        ref_sim.run(dur, false, false);
        let y_ref = subr.borrow().outputs.get_single(0);

        let uc = constant(1.0);
        let scoc = scope(None, 0.0, vec![]);
        let mut cmp_sim = Simulation::with_defaults(
            vec![uc.clone(), block.clone(), scoc.clone()],
            vec![conn(&uc, &block), conn(&block, &scoc)],
        );
        cmp_sim.run(dur, false, false);
        let y_cmp = block.borrow().outputs.get_single(0);

        // Steady state is u/w^2 = 0.25; underdamped, so it rings down to it.
        assert!((y_cmp - y_ref).abs() < 1e-3, "compiled {y_cmp} vs per-block {y_ref}");
    }

    #[test]
    fn compiled_block_captures_internal_events() {
        // Subsystem with an internal zero-crossing: u -> relay -> integrator -> y.
        // The relay's event lives inside the subsystem; the compiled block must
        // capture it and match the per-block subsystem.
        use crate::blocks::constructors::relay;
        use crate::subsystem::{interface, subsystem};

        let switched = || {
            let iface = interface();
            let r = relay(0.0, 0.0, 1.0, -1.0); // switches on input sign
            let i = integrator(0.0); // integrates the relay output
            subsystem(
                vec![iface.clone(), r.clone(), i.clone()],
                vec![conn(&iface, &r), conn(&r, &i), conn(&i, &iface)],
                10,
            )
            .unwrap()
        };

        let u0 = sinusoidal_source(1.0, 1.0, 0.0);
        let sub0 = switched();
        let sco0 = scope(None, 0.0, vec![]);
        let mut asm = Simulation::with_defaults(
            vec![u0.clone(), sub0.clone(), sco0.clone()],
            vec![conn(&u0, &sub0), conn(&sub0, &sco0)],
        );
        asm.run(0.01, false, false);
        let sub_ir = extract_subsystem(&asm, "switched");
        let block = compile_block(&sub_ir).expect("subsystem with event compiles");
        assert!(block.borrow().has_events(), "internal zero-crossing captured");

        let dur = 3.0;
        let ur = sinusoidal_source(1.0, 1.0, 0.0);
        let subr = switched();
        let scor = scope(None, 0.0, vec![]);
        let mut ref_sim = Simulation::with_defaults(
            vec![ur.clone(), subr.clone(), scor.clone()],
            vec![conn(&ur, &subr), conn(&subr, &scor)],
        );
        ref_sim.run(dur, false, true);
        let y_ref = subr.borrow().outputs.get_single(0);

        let uc = sinusoidal_source(1.0, 1.0, 0.0);
        let scoc = scope(None, 0.0, vec![]);
        let mut cmp_sim = Simulation::with_defaults(
            vec![uc.clone(), block.clone(), scoc.clone()],
            vec![conn(&uc, &block), conn(&block, &scoc)],
        );
        cmp_sim.run(dur, false, true);
        let y_cmp = block.borrow().outputs.get_single(0);

        assert!((y_cmp - y_ref).abs() < 5e-2, "compiled {y_cmp} vs per-block {y_ref}");
    }

    /// A compiled run must END at the same time as the interpreted run of the
    /// same model — for fixed and adaptive stepping, with and without events.
    /// Regression: the continuous compiled loop used `while time < t_end + dt`
    /// (one full extra step past the interpreted end) and the event-aware loop
    /// clamped every fixed step onto `t_end` (ending early instead).
    #[test]
    fn run_end_time_matches_interpreted() {
        let dur = 1.0;
        let dt = 0.3; // deliberately does not divide `dur`

        // -- fixed-step, pure-continuous: x' = -x --
        let i = integrator(1.0);
        let amp = amplifier(-1.0);
        let mut sim = Simulation::with_defaults(
            vec![i.clone(), amp.clone()],
            vec![conn(&i, &amp), conn(&amp, &i)],
        );
        sim.dt = dt;
        let mut c = compile(&module_from_sim(&sim, "endtime")).unwrap();
        c.dt = dt;
        sim.run(dur, false, false);
        c.run(dur, true, false);
        assert!(
            (c.time() - sim.time).abs() < 1e-12,
            "fixed-step end: compiled {} vs interpreted {}",
            c.time(),
            sim.time
        );
        // pathsim-parity contract: first step that reaches or passes t_end.
        assert!(c.time() >= dur && c.time() < dur + dt + 1e-12);

        // -- adaptive, pure-continuous: lands exactly on t_end --
        let i2 = integrator(1.0);
        let amp2 = amplifier(-1.0);
        let sim2 = Simulation::with_defaults(
            vec![i2.clone(), amp2.clone()],
            vec![conn(&i2, &amp2), conn(&amp2, &i2)],
        );
        let mut c2 = compile(&module_from_sim(&sim2, "endtime_ad")).unwrap();
        c2.dt = 0.1;
        c2.set_solver("RKBS32", 1e-8, 1e-6);
        c2.run(dur, true, true);
        assert!(
            (c2.time() - dur).abs() < 1e-9,
            "adaptive end: compiled {} != t_end {dur}",
            c2.time()
        );

        // -- fixed-step WITH block-internal events (sample-hold): same end
        //    semantics as the event-free loop and the interpreted engine --
        let src = sinusoidal_source(1.0, 1.0, 0.0);
        let sh = sample_hold(0.25, 0.0);
        let i3 = integrator(0.0);
        let mut sim3 = Simulation::with_defaults(
            vec![src.clone(), sh.clone(), i3.clone()],
            vec![conn(&src, &sh), conn(&sh, &i3)],
        );
        sim3.dt = dt;
        let mut c3 = compile(&module_from_sim(&sim3, "endtime_evt")).unwrap();
        assert!(!c3.events.is_empty(), "sample-hold contributes events");
        c3.dt = dt;
        sim3.run(dur, false, false);
        c3.run(dur, true, false);
        assert!(
            (c3.time() - sim3.time).abs() < 1e-12,
            "fixed-step+events end: compiled {} vs interpreted {}",
            c3.time(),
            sim3.time
        );
    }
}
