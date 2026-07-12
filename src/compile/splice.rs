//! Graph splicing: fuse a flat IR scope into one `jit::graph::Graph` over a
//! global continuous-state vector `X` and a global discrete-memory vector `M`.
//!
//! Each block's IR op-graph is inlined into a shared graph. `Input` ops resolve
//! through `connections` directly to the producing block's output ops (no
//! block-boundary copies); `State` ops map to a slot in `X`; `Memory` ops map to
//! a slot in `M`; `Time`/`Param`/`Const`/compute ops map straight across.
//! Hash-consing gives cross-block CSE for free.
//!
//! The flat input layout is `("x", n_state)`, `("m", n_mem)`, `("t", 1)`, so any
//! tape (derivative, taps, event guards, event effects) is evaluated as
//! `call(&[&x, &m, &[t]])`. `M` is held constant between events and mutated by
//! event effects; blocks without events/memory leave `n_mem = 0`.

use std::collections::{HashMap, HashSet};

use crate::ir::schema::{
    Block, BlockRole, Connection, Direction, EventKind, Op, ParamValue, ScheduleTimes, Write,
};
use crate::ssa::graph::{Graph, InputSignature, Node};

use super::CompileError;

/// Which op-region of a block a translation walk is reading from. Replaces an
/// earlier arithmetic region-id encoding so the cases are self-documenting and
/// a bad id is a compile error, not an opaque slice panic.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Region {
    /// Algebraic region (`alg`): output writes.
    Alg,
    /// Dynamic region (`dyn_`): state-derivative writes.
    Dyn,
    /// Event `k`'s guard expression.
    Guard(usize),
    /// Event `k`'s effect region.
    Effect(usize),
}

/// A destination an event effect writes to.
#[derive(Clone, Copy)]
pub(super) enum WriteTarget {
    /// Global discrete-memory index.
    Mem(usize),
    /// Global continuous-state index (discrete reset of `X`).
    State(usize),
}

/// Event firing kind, carrying just what the runtime `SimEvent` needs.
#[derive(Clone)]
pub(super) enum EvtKind {
    ZeroCross(Direction),
    Schedule(ScheduleTimes),
    Condition,
}

/// One block-internal event, lowered for the compiled run loop.
pub(super) struct CompiledEventSpec {
    pub kind: EvtKind,
    /// Guard scalar node (`ZeroCross`/`Condition`); `None` for `Schedule`.
    pub guard_node: Option<u32>,
    /// Effect output nodes (one per write), parallel to `effect_targets`.
    pub effect_outputs: Vec<u32>,
    pub effect_targets: Vec<WriteTarget>,
}

/// Subsystem-interface configuration when splicing a flat scope *into a block*
/// (rather than a whole system). Adds a `u` (external-input) slot the inner
/// blocks read through `BlockId::INTERFACE`-sourced connections, and names the
/// inner producers feeding each subsystem output element.
pub(super) struct IfaceCfg {
    /// Number of external-input scalar elements (the `u` slot width).
    pub n_in: usize,
    /// Number of subsystem-output scalar elements (the `y` width).
    pub n_out: usize,
    /// Per global output element, the inner producer `(leaf_block, elem)` that
    /// drives it; `(BlockId::INTERFACE.0, elem)` for a direct input passthrough,
    /// `None` for an unconnected output (folds to constant 0).
    pub output_srcs: Vec<Option<(u32, u32)>>,
}

/// Result of splicing a flat scope. `graph` holds every node; the output sets
/// (`deriv_outputs`, `taps`, event guard/effect nodes) index into it.
pub(super) struct Spliced {
    pub graph: Graph,
    pub n_state: usize,
    pub n_mem: usize,
    pub x0: Vec<f64>,
    pub m0: Vec<f64>,
    pub state_labels: Vec<String>,
    /// One node per global state element: `dX/dt` (global state order).
    pub deriv_outputs: Vec<u32>,
    /// Recorded signals (the inputs each sink observes): `(label, node)`.
    pub taps: Vec<(String, u32)>,
    /// Block-internal events, in block order.
    pub events: Vec<CompiledEventSpec>,
    /// One node per subsystem-output element: `y = f_alg(x, m, u, t)`. Empty for
    /// whole-system splicing (no interface).
    pub iface_outputs: Vec<u32>,
}

fn elems_or0(elems: &Option<Vec<u32>>) -> Vec<u32> {
    elems.clone().unwrap_or_else(|| vec![0])
}

/// The resolved producer of one input element: which block output feeds it (or
/// the subsystem interface). Precomputed so `resolve_input` is O(1) instead of
/// rescanning every connection per element.
#[derive(Clone, Copy)]
struct InputSrc {
    block: u32,
    port: u32,
    elem: u32,
    interface: bool,
}

/// Build the `(target_block, target_port, target_elem) -> producer` index from
/// the connection list. First connection wins (matching the original in-order
/// scan), so later duplicates do not override an earlier wiring.
fn build_input_index(conns: &[Connection]) -> HashMap<(u32, u32, u32), InputSrc> {
    let mut idx = HashMap::new();
    for c in conns {
        let s_idx = elems_or0(&c.src.elems);
        let interface = c.src.block == crate::ir::schema::BlockId::INTERFACE;
        for tgt in &c.targets {
            for (j, &d) in elems_or0(&tgt.elems).iter().enumerate() {
                let se = s_idx.get(j).copied().unwrap_or(0);
                idx.entry((tgt.block.0, tgt.port, d)).or_insert(InputSrc {
                    block: c.src.block.0,
                    port: c.src.port,
                    elem: se,
                    interface,
                });
            }
        }
    }
    idx
}

struct Splicer<'a> {
    blocks: &'a [&'a Block],
    input_index: HashMap<(u32, u32, u32), InputSrc>,
    state_offset: Vec<usize>,
    param_offset: Vec<usize>,
    /// Per block, per memory slot: global start index in `M`.
    mem_global: Vec<Vec<usize>>,
    /// Flat base of the `m` / `t` slots (= `n_state` / `n_state + n_mem`).
    mem_base: u32,
    t_flat: u32,
    /// Flat base of the `u` (external-input) slot when splicing a *subsystem*
    /// into a block: connections whose source is `BlockId::INTERFACE` resolve to
    /// `u[elem]` instead of an inner producer. `None` for whole-system splicing.
    iface_u_base: Option<u32>,
    g: Graph,
    memo: HashMap<(u32, Region, u32), u32>,
    in_progress: HashSet<(u32, Region, u32)>,
}

impl<'a> Splicer<'a> {
    fn resolve_output(&mut self, block: u32, port: u32, out_elem: u32) -> Result<u32, CompileError> {
        let src = self.blocks[block as usize].regions.alg.writes.iter().find_map(|w| match w {
            Write::Output { port: p, elem, src } if *p == port && *elem == out_elem => Some(src.0),
            _ => None,
        });
        match src {
            Some(local) => self.translate(block, Region::Alg, local),
            None => Err(CompileError::Unsupported(format!(
                "block '{}' produces no output {port}:{out_elem}",
                self.blocks[block as usize].type_name
            ))),
        }
    }

    fn resolve_input(&mut self, block: u32, port: u32, elem: u32) -> Result<u32, CompileError> {
        match self.input_index.get(&(block, port, elem)).copied() {
            Some(src) => {
                // Subsystem splicing: a connection sourced at the interface is an
                // external input — resolve to the `u` slot, not an inner producer.
                if src.interface {
                    if let Some(u_base) = self.iface_u_base {
                        return Ok(self.g.input(u_base + src.elem));
                    }
                }
                self.resolve_output(src.block, src.port, src.elem)
            }
            None => Ok(self.g.constant(0.0)),
        }
    }

    /// The op slice for a region id: alg, dyn, or event `k`'s guard / effect.
    fn region_op(&self, block: u32, region: Region, local: u32) -> Op {
        let b = self.blocks[block as usize];
        let ops: &[Op] = match region {
            Region::Alg => &b.regions.alg.ops,
            Region::Dyn => &b.regions.dyn_.ops,
            Region::Guard(k) => match &b.events[k].kind {
                EventKind::ZeroCross { guard, .. } | EventKind::Condition { guard } => &guard.ops,
                EventKind::Schedule { .. } => &[],
            },
            Region::Effect(k) => &b.events[k].effect.ops,
        };
        ops[local as usize].clone()
    }

    /// Inline one local op of `block`'s region into the fused graph.
    fn translate(&mut self, block: u32, region: Region, local: u32) -> Result<u32, CompileError> {
        let key = (block, region, local);
        if let Some(&n) = self.memo.get(&key) {
            return Ok(n);
        }
        if !self.in_progress.insert(key) {
            return Err(CompileError::AlgebraicLoop);
        }
        let op = self.region_op(block, region, local);
        let node = match op {
            Op::Const(v) => self.g.constant(v),
            Op::Time => self.g.input(self.t_flat),
            Op::Input { port, elem } => self.resolve_input(block, port, elem)?,
            Op::State { id } => self.g.input((self.state_offset[block as usize] + id.idx()) as u32),
            Op::Param { id } => self.g.param((self.param_offset[block as usize] + id.idx()) as u32),
            Op::Memory { slot, offset } => {
                let idx = self.mem_global[block as usize][slot.idx()] + offset as usize;
                self.g.input(self.mem_base + idx as u32)
            }
            Op::Binary { op, a, b } => {
                let a = self.translate(block, region, a.0)?;
                let b = self.translate(block, region, b.0)?;
                self.g.binary(op, a, b)
            }
            Op::Unary { op, a } => {
                let a = self.translate(block, region, a.0)?;
                self.g.unary(op, a)
            }
            Op::Cmp { op, a, b } => {
                let a = self.translate(block, region, a.0)?;
                let b = self.translate(block, region, b.0)?;
                self.g.cmp(op, a, b)
            }
            Op::Select { c, t, e } => {
                let c = self.translate(block, region, c.0)?;
                let t = self.translate(block, region, t.0)?;
                let e = self.translate(block, region, e.0)?;
                self.g.select(c, t, e)
            }
            Op::Fma { a, b, c } => {
                let a = self.translate(block, region, a.0)?;
                let b = self.translate(block, region, b.0)?;
                let c = self.translate(block, region, c.0)?;
                self.g.add(Node::Fma(a, b, c))
            }
            Op::Reduce { op, args } => {
                let mut nodes = Vec::with_capacity(args.len());
                for a in &args {
                    nodes.push(self.translate(block, region, a.0)?);
                }
                self.g.reduce(op, nodes)
            }
            Op::Dot { a, b } => {
                let mut a_nodes = Vec::with_capacity(a.len());
                for n in &a {
                    a_nodes.push(self.translate(block, region, n.0)?);
                }
                let mut b_nodes = Vec::with_capacity(b.len());
                for n in &b {
                    b_nodes.push(self.translate(block, region, n.0)?);
                }
                self.g.dot(a_nodes, b_nodes)
            }
            Op::Lut1d { input, points, values, clamp } => {
                let x = self.translate(block, region, input.0)?;
                crate::blocks::constructors::table::lut1d_to_graph(&mut self.g, x, &points, &values, clamp)
            }
            Op::Call { .. } => {
                return Err(CompileError::OpaqueBlock(
                    self.blocks[block as usize].type_name.clone(),
                ))
            }
        };
        self.in_progress.remove(&key);
        self.memo.insert(key, node);
        Ok(node)
    }
}

pub(super) fn splice(
    blocks: &[&Block],
    conns: &[Connection],
    iface: Option<IfaceCfg>,
) -> Result<Spliced, CompileError> {
    // Global state / memory / parameter layout (block order).
    let mut state_offset = vec![0usize; blocks.len()];
    let mut param_offset = vec![0usize; blocks.len()];
    let mut mem_global = vec![Vec::new(); blocks.len()];
    let (mut n_state, mut n_param, mut n_mem) = (0usize, 0usize, 0usize);
    let mut param_defaults = Vec::new();
    let mut param_names = Vec::new();
    let (mut x0, mut m0) = (Vec::new(), Vec::new());
    let mut state_labels = Vec::new();

    for (i, b) in blocks.iter().enumerate() {
        state_offset[i] = n_state;
        for sv in &b.state {
            x0.push(sv.init);
            state_labels.push(format!("{}.{}", b.name, sv.name));
        }
        n_state += b.state.len();

        for slot in &b.memory {
            mem_global[i].push(n_mem);
            m0.extend_from_slice(&slot.init);
            n_mem += slot.size as usize;
        }

        param_offset[i] = n_param;
        for p in &b.params {
            param_names.push(format!("{}.{}", b.name, p.name));
            match &p.value {
                ParamValue::Scalar(v) => param_defaults.push(*v),
                _ => {
                    return Err(CompileError::Unsupported(format!(
                        "block '{}' has a non-scalar parameter '{}'",
                        b.type_name, p.name
                    )))
                }
            }
        }
        n_param += b.params.len();
    }

    if n_state == 0 {
        return Err(CompileError::NoState);
    }

    let n_in = iface.as_ref().map(|c| c.n_in).unwrap_or(0);
    // Flat input layout: `x`, `m`, then (for a subsystem) `u`, then `t`.
    let sig = match &iface {
        Some(_) => InputSignature::from_named_sizes([
            ("x", n_state), ("m", n_mem), ("u", n_in), ("t", 1),
        ]),
        None => InputSignature::from_named_sizes([("x", n_state), ("m", n_mem), ("t", 1)]),
    };
    let iface_u_base = iface.as_ref().map(|_| (n_state + n_mem) as u32);
    let t_flat = (n_state + n_mem + n_in) as u32;
    let mut g = Graph::new(sig);
    g.n_params = n_param;
    g.param_defaults = param_defaults;
    g.param_names = param_names;

    // Collect derivative work first so the translate recursion doesn't alias the
    // block iteration.
    let mut deriv_work: Vec<(usize, usize, u32)> = Vec::new();
    for (i, b) in blocks.iter().enumerate() {
        for w in &b.regions.dyn_.writes {
            if let Write::StateDeriv { id, src } = w {
                deriv_work.push((i, id.idx(), src.0));
            }
        }
    }

    let mut sp = Splicer {
        blocks,
        input_index: build_input_index(conns),
        state_offset,
        param_offset,
        mem_global,
        mem_base: n_state as u32,
        t_flat,
        iface_u_base,
        g,
        memo: HashMap::new(),
        in_progress: HashSet::new(),
    };

    let mut deriv: Vec<Option<u32>> = vec![None; n_state];
    for (i, sid, src) in deriv_work {
        let node = sp.translate(i as u32, Region::Dyn, src)?;
        deriv[sp.state_offset[i] + sid] = Some(node);
    }
    let deriv_outputs: Vec<u32> = deriv
        .into_iter()
        .enumerate()
        .map(|(k, o)| {
            o.ok_or_else(|| CompileError::Unsupported(format!("global state {k} has no derivative")))
        })
        .collect::<Result<_, _>>()?;

    // Block-internal events: guard scalar + effect writes, lowered over (x, m, t).
    let mut events: Vec<CompiledEventSpec> = Vec::new();
    for i in 0..blocks.len() {
        for k in 0..blocks[i].events.len() {
            let ev = &blocks[i].events[k];
            let kind = match &ev.kind {
                EventKind::ZeroCross { direction, .. } => EvtKind::ZeroCross(*direction),
                EventKind::Schedule { times } => EvtKind::Schedule(times.clone()),
                EventKind::Condition { .. } => EvtKind::Condition,
            };
            let guard_node = match &ev.kind {
                EventKind::ZeroCross { guard, .. } | EventKind::Condition { guard } => {
                    if guard.ops.is_empty() {
                        None
                    } else {
                        // IR convention: the guard's scalar value is its last op.
                        let last = guard.ops.len() as u32 - 1;
                        Some(sp.translate(i as u32, Region::Guard(k), last)?)
                    }
                }
                EventKind::Schedule { .. } => None,
            };
            // Effect writes — collect (block, slot/state) first so we don't alias.
            let writes: Vec<Write> = ev.effect.writes.clone();
            let mut effect_outputs = Vec::new();
            let mut effect_targets = Vec::new();
            for w in &writes {
                match w {
                    Write::MemoryWrite { slot, offset, src } => {
                        let node = sp.translate(i as u32, Region::Effect(k), src.0)?;
                        effect_outputs.push(node);
                        let idx = sp.mem_global[i][slot.idx()] + *offset as usize;
                        effect_targets.push(WriteTarget::Mem(idx));
                    }
                    Write::StateWrite { id, src } => {
                        let node = sp.translate(i as u32, Region::Effect(k), src.0)?;
                        effect_outputs.push(node);
                        effect_targets.push(WriteTarget::State(sp.state_offset[i] + id.idx()));
                    }
                    // Output latches in event effects are not modelled by the
                    // static compile (would need per-step output latches). Reject
                    // loudly rather than silently dropping the write and mis-modelling.
                    Write::Output { .. } => {
                        return Err(CompileError::Unsupported(
                            "event effect writes a block output (output latch); not modelled by static compile".into(),
                        ));
                    }
                    Write::StateDeriv { .. } => {
                        return Err(CompileError::Unsupported(
                            "event effect contains a state-derivative write (only legal in dyn regions)".into(),
                        ));
                    }
                }
            }
            events.push(CompiledEventSpec { kind, guard_node, effect_outputs, effect_targets });
        }
    }

    // Output taps: the signals each sink observes (deduped by node).
    let mut taps: Vec<(String, u32)> = Vec::new();
    let mut tapped: HashSet<u32> = HashSet::new();
    let sink_ids: Vec<usize> = (0..blocks.len())
        .filter(|&i| blocks[i].role == BlockRole::Sink)
        .collect();
    for i in sink_ids {
        let inw: u32 = blocks[i].ports.inputs.iter().map(|p| p.size).sum();
        for elem in 0..inw {
            let node = sp.resolve_input(i as u32, 0, elem)?;
            if tapped.insert(node) {
                taps.push((format!("{}#{}.in{}", blocks[i].name, i, elem), node));
            }
        }
    }

    // Subsystem outputs: one node per output element, `y = f_alg(x, m, u, t)`.
    // Each is the inner producer feeding that interface-output element (or the
    // `u` slot directly for an input passthrough).
    let iface_outputs: Vec<u32> = match &iface {
        None => Vec::new(),
        Some(cfg) => {
            let u_base = iface_u_base.expect("interface splicing has a u slot");
            let mut outs = Vec::with_capacity(cfg.n_out);
            for src in &cfg.output_srcs {
                let node = match *src {
                    None => sp.g.constant(0.0),
                    Some((block, elem)) if block == crate::ir::schema::BlockId::INTERFACE.0 => {
                        sp.g.input(u_base + elem)
                    }
                    Some((block, elem)) => sp.resolve_output(block, 0, elem)?,
                };
                outs.push(node);
            }
            outs
        }
    };

    Ok(Spliced {
        graph: sp.g,
        n_state,
        n_mem,
        x0,
        m0,
        state_labels,
        deriv_outputs,
        taps,
        events,
        iface_outputs,
    })
}

