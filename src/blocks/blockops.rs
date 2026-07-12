//! Per-block operator atomization: the lowest IR layer.
//!
//! This module owns the shared SSA graph-building utilities (`region_graph_xu`,
//! `mem_read_alg_graph`), the slot-naming convention (`SlotKind`), the
//! shape-poly graph wrappers (`RegionGraph`, `ShapeLazyGraph`), and the
//! discrete-block specs (`MemSpec`, `EventSpec`, `Lut1dSpec`). A block's
//! algebraic / dynamic behaviour lives in its per-path `Operator`s (see
//! `blocks::operator`), the single source of truth from which BOTH the runtime
//! closures and the hierarchical IR (`ir::builder`) are derived, so the two can
//! never diverge.
//!
//! Graphs use a shared input-slot naming convention so the IR builder can
//! decode flat inputs back into structured reads:
//!
//!   - `"x"`: continuous state vector
//!   - `"u"`: input ports, flat-concatenated (matches the runtime's
//!     `inputs.to_array()`); the builder splits it back into ports using the
//!     block's port sizes
//!   - `"mem{k}"`: memory slot k (later WP)
//!   - `"t"`: simulation time (scalar)
//!
//! Shape-polymorphic blocks (Amplifier, Adder, elementwise math) cannot fix
//! their graph at construction: the input width is only known after connection
//! resolution. `ShapeLazyGraph` mirrors the Python-side `LazyTraced` for
//! Rust-native graph builders: it rebuilds + caches the lowered graph keyed on
//! the input width, with an alloc-free fast path once the width is stable.

use std::cell::RefCell;
use std::rc::Rc;

use smallvec::SmallVec;

use crate::blocks::block::BlockFn;
use crate::ssa::build::GraphBuilder;
use crate::ssa::graph::{Graph, InputSignature};
use crate::ssa::tape::InterpretedFn;

/// Build a memory-read alg graph: `y[i] = mem[slot][i]` for `i in 0..n`
/// (identity read of one memory slot to the output). Shared by discrete blocks
/// whose algebraic output is just the held memory.
pub fn mem_read_alg_graph(slot: u32, n: usize) -> Graph {
    let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([(memory_slot_name(slot), n)])));
    let outs: Vec<u32> = {
        let gb = GraphBuilder::new(&cell);
        (0..n as u32).map(|i| gb.input(i)).collect()
    };
    let mut g = cell.into_inner();
    g.outputs = outs;
    g
}

/// Build a region op-graph with `("x", ns)` and `("u", ni)` input slots plus
/// `param_names` scalar params (defaults from `param_defaults`), wiring the
/// body through a `GraphBuilder`. `build` receives
/// `(builder, param_nodes, x_nodes, u_nodes, &mut out_nodes)` and is the
/// `GraphBuilder` instantiation of the block's generic `eval` (the native
/// closure is the `F64Builder` instantiation of the same `eval`). Shared by
/// every custom dynamic / stateful block so they stay single-source.
pub fn region_graph_xu(
    ns: usize,
    ni: usize,
    param_defaults: Vec<f64>,
    param_names: &[&str],
    build: impl FnOnce(&GraphBuilder, &[u32], &[u32], &[u32], &mut Vec<u32>),
) -> Graph {
    let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("x", ns), ("u", ni)])));
    {
        let mut g = cell.borrow_mut();
        g.n_params = param_defaults.len();
        g.param_defaults = param_defaults;
        g.param_names = param_names.iter().map(|s| s.to_string()).collect();
    }
    let out = {
        let gb = GraphBuilder::new(&cell);
        let x: Vec<_> = (0..ns as u32).map(|i| gb.input(i)).collect();
        let u: Vec<_> = (0..ni as u32).map(|i| gb.input(ns as u32 + i)).collect();
        let params: Vec<_> = (0..param_names.len() as u32).map(|i| gb.param(i)).collect();
        let mut out = Vec::new();
        build(&gb, &params, &x, &u, &mut out);
        out
    };
    let mut g = cell.into_inner();
    g.outputs = out;
    g
}

// ======================================================================================
// Slot classification (shape-independent: depends on slot name, not size)
// ======================================================================================

/// Semantic kind of a region input slot, decoded from its name. The slot-naming
/// convention is owned here (see the module header): `"x"` continuous state,
/// `"t"` time, `"u"` (and anything else) the flat input vector, `"mem{k}"`
/// memory slot `k`. Both the runtime slot plan (`SlotSource`) and the IR's
/// structured-read decoding (`ir::builder::decode_input`) classify through this
/// one authority, so the producers and consumers of the convention cannot drift.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SlotKind {
    State,
    Time,
    Input,
    Memory(u32),
}

/// Memory-slot input name for slot index `k` (the `"mem{k}"` convention). The
/// format twin of the `slot_kind` parse below; use it when building signatures.
pub fn memory_slot_name(k: u32) -> String {
    format!("mem{k}")
}

/// Classify a region input slot by name. The single authority for the
/// `"x"`/`"t"`/`"u"`/`"mem{k}"` convention.
pub fn slot_kind(name: &str) -> SlotKind {
    match name {
        "x" => SlotKind::State,
        "t" => SlotKind::Time,
        n if n.starts_with("mem") => SlotKind::Memory(n["mem".len()..].parse().unwrap_or(0)),
        // "u" and anything else default to the flat input vector.
        _ => SlotKind::Input,
    }
}

/// Which `(x, u, t)` argument feeds a given input slot on the runtime hot path,
/// decided once from the slot name (via [`slot_kind`]) so the hot path is a
/// plain index. Memory slots are not fed through this path.
#[derive(Clone, Copy)]
enum SlotSource { State, Input, Time }

fn classify_slot(name: &str) -> SlotSource {
    match slot_kind(name) {
        SlotKind::State => SlotSource::State,
        SlotKind::Time => SlotSource::Time,
        SlotKind::Input | SlotKind::Memory(_) => SlotSource::Input,
    }
}

// ======================================================================================
// LoweredGraph: a compiled region + precomputed slot plan, alloc-free to call
// ======================================================================================

/// A region graph compiled to an `InterpretedFn` with its input-slot plan
/// precomputed. `eval` assembles the slot slices from `(x, u, t)` into a
/// stack-backed `SmallVec` (no heap for the usual <= 4 slots) and evaluates
/// via `call_into`, which reuses its own scratch and writes into `out`.
pub struct LoweredGraph {
    compiled: Rc<InterpretedFn>,
    sources: SmallVec<[SlotSource; 4]>,
    n_out: usize,
}

impl LoweredGraph {
    pub fn from_graph(graph: &Graph) -> Self {
        let sources = graph
            .signature
            .slots
            .iter()
            .map(|s| classify_slot(&s.name))
            .collect();
        let compiled = Rc::new(InterpretedFn::from_graph(graph.clone()));
        let n_out = compiled.n_out;
        Self { compiled, sources, n_out }
    }

    #[inline]
    pub fn eval(&self, x: &[f64], u: &[f64], t: f64, out: &mut Vec<f64>) {
        let tb = [t];
        let mut slots: SmallVec<[&[f64]; 4]> = SmallVec::with_capacity(self.sources.len());
        for s in &self.sources {
            slots.push(match s {
                SlotSource::State => x,
                SlotSource::Input => u,
                SlotSource::Time => &tb[..],
            });
        }
        // Caller clears `out`; size it to the region's output arity and let
        // `call_into` overwrite every element.
        out.resize(self.n_out, 0.0);
        self.compiled.call_into(&slots, out);
    }
}

// ======================================================================================
// ShapeLazyGraph: Rust-native analogue of LazyTraced for graph builders
// ======================================================================================

struct ShapeEntry {
    /// Input width (`u.len()`) this entry was built for.
    width: usize,
    /// Source graph, kept for IR derivation. `None` when the builder could not
    /// produce an op-graph at this width (e.g. a Python callable with
    /// data-dependent control flow): the block then behaves as opaque.
    graph: Option<Graph>,
}

/// Builds + caches a region graph keyed on input width, for IR derivation of
/// shape-polymorphic blocks. The runtime path does NOT go through here (under
/// the 2b design the runtime closure is the native `eval::<F64Builder>`); this
/// only resolves the op-graph at the connected width when the IR is built.
///
/// The builder is fallible (`-> Option<Graph>`): Rust-native graph builders
/// always succeed (`new`), but a re-traced Python callable can fail to lower at
/// a given width (`new_fallible`), in which case the block falls back to its
/// opaque runtime closure instead of panicking.
pub struct ShapeLazyGraph {
    builder: Box<dyn Fn(usize) -> Option<Graph>>,
    cache: RefCell<Option<ShapeEntry>>,
}

impl ShapeLazyGraph {
    /// Infallible builder (Rust-native graphs): always yields an op-graph.
    pub fn new(builder: impl Fn(usize) -> Graph + 'static) -> Rc<Self> {
        Rc::new(Self { builder: Box::new(move |w| Some(builder(w))), cache: RefCell::new(None) })
    }

    /// Fallible builder (re-traced Python callable): `None` at a width means the
    /// callable is not op-traceable there, so the block stays opaque.
    pub fn new_fallible(builder: impl Fn(usize) -> Option<Graph> + 'static) -> Rc<Self> {
        Rc::new(Self { builder: Box::new(builder), cache: RefCell::new(None) })
    }

    /// Resolve the source graph at the given input width, rebuilding + caching
    /// on a width change. Called at IR-build time, never on the hot path.
    /// `None` when the builder cannot lower at this width (opaque fallback).
    pub fn resolve_graph(&self, width: usize) -> Option<Graph> {
        let stale = match &*self.cache.borrow() {
            Some(e) => e.width != width,
            None => true,
        };
        if stale {
            let graph = (self.builder)(width);
            *self.cache.borrow_mut() = Some(ShapeEntry { width, graph });
        }
        self.cache.borrow().as_ref().unwrap().graph.clone()
    }
}

// ======================================================================================
// RegionGraph: a region is either shape-fixed at construction or shape-lazy
// ======================================================================================

#[derive(Clone)]
pub enum RegionGraph {
    /// Graph fully determined at construction (state-sized blocks, sources).
    Fixed(Graph),
    /// Graph parameterized by input width, resolved after connection layout.
    Lazy(Rc<ShapeLazyGraph>),
}

impl RegionGraph {
    /// Concrete source graph for IR derivation. `input_width` is used only for
    /// `Lazy` regions (the connected input element count). `None` when a `Lazy`
    /// region cannot lower at this width (the block falls back to opaque).
    pub fn resolve(&self, input_width: usize) -> Option<Graph> {
        match self {
            RegionGraph::Fixed(g) => Some(g.clone()),
            RegionGraph::Lazy(slg) => slg.resolve_graph(input_width),
        }
    }
}

// ======================================================================================
// Discrete-block specs (memory slots, events) + LUT structure
// ======================================================================================

/// A discrete memory slot (block-local persistent state updated by events).
#[derive(Clone)]
pub struct MemSpec {
    pub name: String,
    pub init: Vec<f64>,
}

/// Zero-crossing direction for `EventKindSpec::ZeroCross`.
#[derive(Clone, Copy)]
pub enum DirSpec {
    Rising,
    Falling,
    Both,
}

/// How a discrete event fires.
#[derive(Clone)]
pub enum EventKindSpec {
    /// First fire at `phase`, then every `period`.
    SchedulePeriodic { period: f64, phase: f64 },
    /// Fire at the given absolute times.
    ScheduleFixed(Vec<f64>),
    /// Fire when `guard` crosses zero in `direction`. The guard graph has one
    /// scalar output (its last op).
    ZeroCross { guard: Graph, direction: DirSpec },
    /// Fire while `guard`'s scalar output is non-zero.
    Condition { guard: Graph },
}

/// Where one effect-graph output is written.
#[derive(Clone, Copy)]
pub struct MemTarget {
    pub slot: u32,
    pub offset: u32,
}

/// A block-internal event: a guard kind plus an effect graph whose outputs are
/// written into memory slots (`effect.outputs[i]` -> `targets[i]`). The effect
/// graph reads inputs (`"u"`), memory (`"mem{k}"`), and time (`"t"`).
#[derive(Clone)]
pub struct EventSpec {
    pub kind: EventKindSpec,
    pub effect: Graph,
    pub targets: Vec<MemTarget>,
}

/// Resolved discrete spec at a given input width: the alg graph (reads memory),
/// the memory slots, and the events that update them.
pub type DiscreteResolved = (Graph, Vec<MemSpec>, Vec<EventSpec>);

/// A 1-D lookup table's structure (breakpoints, values, extrapolation mode).
/// Carried alongside the select-chain `alg` graph so the IR builder can emit a
/// single structured `Op::Lut1d` (table) instead of the unrolled select chain.
#[derive(Clone)]
pub struct Lut1dSpec {
    pub points: Vec<f64>,
    pub values: Vec<f64>,
    pub clamp: bool,
}

// ======================================================================================
// BlockFn lowering bridges (the single-source runtime derivations)
// ======================================================================================

/// Lower a shape-fixed region graph to a runtime `BlockFn`, alloc-free. Used
/// for already-traced group-1 blocks (Source/Function/ODE), which have no
/// native closure alternative and legitimately run as a tape.
pub fn block_fn_from_graph(graph: &Graph) -> BlockFn {
    let lowered = LoweredGraph::from_graph(graph);
    Box::new(move |x, u, t, out| lowered.eval(x, u, t, out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssa::graph::{BinOp, Graph, InputSignature, Node};

    /// `y = gain * u` with gain as a mutable Param node lowers and matches.
    #[test]
    fn block_fn_from_graph_amplifier_param() {
        let sig = InputSignature::from_named_sizes([("u", 1usize)]);
        let mut g = Graph::new(sig);
        g.n_params = 1;
        g.param_defaults = vec![2.5];
        g.param_names = vec!["gain".into()];
        let u = g.input(0);
        let gain = g.param(0);
        let y = g.binary(BinOp::Mul, u, gain);
        g.outputs.push(y);

        let f = block_fn_from_graph(&g);
        let mut out = Vec::new();
        f(&[], &[4.0], 0.0, &mut out);
        assert_eq!(out, vec![10.0]);
    }

    /// A time-only source graph (`y = t`) lowers and reads the `t` slot.
    #[test]
    fn block_fn_from_graph_time_source() {
        let sig = InputSignature::from_named_sizes([("t", 1usize)]);
        let mut g = Graph::new(sig);
        let t = g.add(Node::Input(0));
        g.outputs.push(t);

        let f = block_fn_from_graph(&g);
        let mut out = Vec::new();
        f(&[], &[], 0.75, &mut out);
        assert_eq!(out, vec![0.75]);
    }

    /// Shape-lazy IR resolution rebuilds a width-correct graph on width change.
    #[test]
    fn shape_lazy_resolves_per_width() {
        let slg = ShapeLazyGraph::new(|n: usize| {
            let sig = InputSignature::from_named_sizes([("u", n)]);
            let mut g = Graph::new(sig);
            let three = g.constant(3.0);
            for i in 0..n {
                let ui = g.input(i as u32);
                let yi = g.binary(BinOp::Mul, ui, three);
                g.outputs.push(yi);
            }
            g
        });

        assert_eq!(slg.resolve_graph(2).unwrap().outputs.len(), 2);
        assert_eq!(slg.resolve_graph(3).unwrap().outputs.len(), 3);
        // Re-resolving the same width returns the cached graph.
        assert_eq!(slg.resolve_graph(3).unwrap().outputs.len(), 3);
    }

    /// A fallible builder that cannot lower at a width yields `None` (opaque
    /// fallback) instead of panicking.
    #[test]
    fn shape_lazy_fallible_returns_none() {
        let slg = ShapeLazyGraph::new_fallible(|_n: usize| None);
        assert!(slg.resolve_graph(2).is_none());
        assert!(RegionGraph::Lazy(slg).resolve(2).is_none());
    }
}
