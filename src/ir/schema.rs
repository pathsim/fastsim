//! IR v1 schema: pure data, no runtime, serde-serializable.
//!
//! A `Module` is the serializable, language-agnostic snapshot of a live
//! `Simulation`: `Module -> Subsystem (recursive) -> Block -> Regions{alg, dyn}
//! -> Region{ops, writes}`. Block regions mirror pathsim's `op_alg` / `op_dyn`
//! operators, lowered one level further to scalar SSA `Op`s.
//!
//! The op vocabulary is a 1:1 mirror of `jit::graph` (`Const`, `Input`,
//! `Param`, `Binary`, `Unary`, `Cmp`, `Select`, `Fma`, `Reduce`, `Dot`) extended with
//! IR-level reads (`Time`, `State`, `Memory`) and an `extern Call` escape hatch
//! for opaque blocks (RNG, lookup tables, FMU). Keeping the kinds aligned with
//! `jit::graph` makes tape lowering (WP-later) a trivial map.
//!
//! IDs are `u32` index newtypes, never pointers, so a `Module` JSON-roundtrips
//! losslessly.

use serde::{Deserialize, Serialize};

pub const IR_VERSION: u32 = 1;

// ======================================================================================
// ID newtypes: type-safe at call sites, bare integers in JSON.
// ======================================================================================

macro_rules! id_type {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub u32);

        impl $name {
            pub const fn new(v: u32) -> Self { Self(v) }
            pub const fn idx(self) -> usize { self.0 as usize }
        }
    };
}

id_type!(SubsystemId);
id_type!(BlockId);
id_type!(ConnectionId);
id_type!(NodeId);
id_type!(StateId);
id_type!(MemorySlotId);
id_type!(ParamId);
id_type!(EventId);
id_type!(ExternId);

impl BlockId {
    /// Sentinel `BlockId` used in `PortRef` to refer to the enclosing
    /// subsystem's `Interface` ports rather than a regular child block.
    pub const INTERFACE: BlockId = BlockId(u32::MAX);
}

// ======================================================================================
// Top-level module
// ======================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Module {
    pub ir_version: u32,
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    pub root: Subsystem,
    /// Simulation-level (global) events not attached to any block: standalone
    /// `ZeroCross` / `Schedule` / `Condition` registered on the `Simulation`.
    /// Their guards/actions are host closures, so they are always `opaque`
    /// (only the statically-known kind/timing is recorded).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<Event>,
    /// Declarations for every `Op::Call` extern referenced anywhere in the
    /// module (RNG, lookup tables, FMU, ...).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extern_decls: Vec<ExternDecl>,
}

impl Module {
    pub fn new(name: impl Into<String>, root: Subsystem) -> Self {
        Self {
            ir_version: IR_VERSION,
            name: name.into(),
            description: String::new(),
            root,
            events: Vec::new(),
            extern_decls: Vec::new(),
        }
    }
}

/// Opaque extern referenced by `Op::Call`. The IR carries only its shape;
/// backends that cannot supply the extern (codegen, eval) flag the block as
/// non-lowerable rather than fail the whole module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternDecl {
    pub id: ExternId,
    pub name: String,
    pub arity_in: u32,
    pub arity_out: u32,
}

// ======================================================================================
// Subsystem (recursive)
// ======================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subsystem {
    pub id: SubsystemId,
    pub name: String,
    pub interface: Interface,
    /// Direct children only; `Subsystem` nests recursively via `Child`.
    pub children: Vec<Child>,
    pub connections: Vec<Connection>,
    pub schedule: Schedule,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Child {
    Block(Block),
    Subsystem(Subsystem),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Interface {
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Port {
    pub name: String,
    /// Number of scalar elements on this port (1 for SISO).
    #[serde(default = "one_u32")]
    pub size: u32,
}

fn one_u32() -> u32 { 1 }

// ======================================================================================
// Block
// ======================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub id: BlockId,
    pub name: String,
    /// Source block kind for provenance / pretty-printing ("Amplifier",
    /// "Integrator", ...). No semantic meaning: backends rely on
    /// `regions` / `role` / `state` / `memory` for behaviour.
    pub type_name: String,
    pub role: BlockRole,
    pub ports: Ports,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub params: Vec<Param>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub state: Vec<StateVar>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub memory: Vec<MemorySlot>,
    pub regions: Regions,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<Event>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockRole {
    /// Algebraic feedthrough: y = f(u, t). No state. DAG-depth driven.
    Algebraic,
    /// Has continuous state integrated by a solver. May also feed through.
    Dynamic,
    /// No inputs (Constant, sinusoidal source, ...). Always at depth 0.
    Source,
    /// No outputs (Scope, recorder). Pure sink.
    Sink,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Ports {
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Param {
    pub id: ParamId,
    pub name: String,
    pub value: ParamValue,
}

/// Parameter values keep their structure so backends can emit them as
/// scalars, vectors, or matrices without re-flattening. A snapshot of the
/// live block's parameter at build time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParamValue {
    Scalar(f64),
    Vector(Vec<f64>),
    Matrix { rows: u32, cols: u32, data: Vec<f64> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateVar {
    pub id: StateId,
    pub name: String,
    pub init: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySlot {
    pub id: MemorySlotId,
    pub name: String,
    pub size: u32,
    pub init: Vec<f64>,
}

// ======================================================================================
// Regions: SSA ops + side-effect writes (mirrors pathsim op_alg / op_dyn).
// ======================================================================================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Regions {
    /// Output equation: y = f(x, u, t, mem). pathsim `op_alg`.
    #[serde(default, skip_serializing_if = "Region::is_empty")]
    pub alg: Region,
    /// State derivative: dx/dt = g(x, u, t, mem). pathsim `op_dyn`.
    #[serde(default, skip_serializing_if = "Region::is_empty")]
    pub dyn_: Region,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Region {
    /// SSA nodes. `NodeId(i)` refers to `ops[i]`. Reads only; effects happen
    /// through `writes` after all `ops` execute.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ops: Vec<Op>,
    /// Side-effect writes, applied in order after `ops`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub writes: Vec<Write>,
}

impl Region {
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty() && self.writes.is_empty()
    }
}

// ======================================================================================
// Op vocabulary: mirrors jit/graph.rs plus IR-level reads + extern escape.
// ======================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Op {
    // -- Reads (zero data operands) --
    Const(f64),
    Time,
    Input { port: u32, elem: u32 },
    Param { id: ParamId },
    State { id: StateId },
    Memory { slot: MemorySlotId, offset: u32 },

    // -- Compute (mirror jit::graph::Node) --
    Binary { op: BinOpKind, a: NodeId, b: NodeId },
    Unary { op: UnaryOpKind, a: NodeId },
    Cmp { op: CmpKind, a: NodeId, b: NodeId },
    Select { c: NodeId, t: NodeId, e: NodeId },
    Fma { a: NodeId, b: NodeId, c: NodeId },

    // -- Structured: variadic reduction over an operand list --
    Reduce { op: ReduceKind, args: Vec<NodeId> },

    // -- Structured: fused dot product Σ aᵢ·bᵢ over two equal-length lists --
    Dot { a: Vec<NodeId>, b: Vec<NodeId> },

    // -- Structured: 1-D piecewise-linear lookup table over a fixed breakpoint
    // grid. `input` is the lookup coordinate; segment `k` is the highest index
    // with `points[k] <= input`, then `y = values[k] + t·(values[k+1]-values[k])`
    // with `t = (input - points[k]) / (points[k+1] - points[k])`. `clamp` holds
    // the output flat past either end (otherwise the boundary segment continues
    // linearly). Carries the table inline so codegen can emit `static const`
    // arrays + a counted search instead of an O(N) unrolled `select` chain.
    Lut1d { input: NodeId, points: Vec<f64>, values: Vec<f64>, clamp: bool },

    // -- Escape hatch (RNG, opaque externs) --
    Call { id: ExternId, args: Vec<NodeId>, out_idx: u32 },
}

// The op sub-vocabularies (binary/unary/cmp/reduce kinds) ARE the jit graph's
// enums, re-exported here under their IR names. There is no separate IR copy:
// one enum, used by both the runtime graph and the serializable IR, so a new op
// is declared once and the two representations cannot drift. The jit enums
// carry the serde derives needed for IR (de)serialization.
pub use crate::ssa::graph::{
    BinOp as BinOpKind, CmpOp as CmpKind, ReduceOp as ReduceKind, UnaryOp as UnaryOpKind,
};

// ======================================================================================
// Side-effect writes
// ======================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Write {
    /// `outputs[port][elem] = ops[src]`. Used by `alg` regions and event effects.
    Output { port: u32, elem: u32, src: NodeId },
    /// `dx/dt[id] = ops[src]`. Only legal in `dyn` regions.
    StateDeriv { id: StateId, src: NodeId },
    /// Discrete state mutation `x[id] = ops[src]`. Only legal in event effects.
    StateWrite { id: StateId, src: NodeId },
    /// `memory[slot][offset] = ops[src]`.
    MemoryWrite { slot: MemorySlotId, offset: u32, src: NodeId },
}

// ======================================================================================
// Events (block-internal)
// ======================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: EventId,
    pub kind: EventKind,
    /// Discrete actions executed when the event resolves. Empty for `opaque`
    /// events (the action is host code).
    #[serde(default, skip_serializing_if = "Region::is_empty")]
    pub effect: Region,
    /// True when the guard and/or action is host code (RNG, scope recording,
    /// arbitrary callback) not expressible as ops. The `effect` and any guard
    /// `Region` inside `kind` are then empty/advisory; `kind` still carries the
    /// statically-known structure (Schedule timing, ZeroCross direction).
    #[serde(default, skip_serializing_if = "is_false")]
    pub opaque: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventKind {
    /// Fires when `guard`'s output crosses zero in `direction`. `guard` must
    /// produce exactly one scalar value; the IR convention is that the value
    /// is the last `Op` in `guard.ops` (no `writes` needed).
    ZeroCross { guard: Region, direction: Direction },
    /// Fires at given simulation times.
    Schedule { times: ScheduleTimes },
    /// Fires while `guard`'s scalar output is non-zero.
    Condition { guard: Region },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Rising, Falling, Both,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScheduleTimes {
    Fixed(Vec<f64>),
    Periodic { period: f64, phase: f64 },
}

// ======================================================================================
// Connections + schedule
// ======================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    pub id: ConnectionId,
    pub src: PortRef,
    pub targets: Vec<PortRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortRef {
    /// `BlockId::INTERFACE` refers to the enclosing subsystem's interface.
    pub block: BlockId,
    pub port: u32,
    /// `None` = all elements of the port; `Some(idx)` = MIMO element slicing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elems: Option<Vec<u32>>,
}

/// Precomputed evaluation plan for one subsystem scope.
///
/// **Derived/advisory.** Everything here is recomputable from `connections` +
/// block roles (that is exactly what fastsim's `utils::schedule` does, and what the
/// builder reuses verbatim). `connections` remain the source of truth for the
/// dataflow wiring; the `Schedule` only spares a consumer from reimplementing
/// the graph analysis (topological order, depth layering, Tarjan SCCs,
/// back-edge classification).
///
/// Block ids here refer to the scope's direct children, plus the
/// `BlockId::INTERFACE` sentinel where the enclosing subsystem's interface
/// participates in the order (it forwards the subsystem's inputs).
///
/// "Depth" throughout is *algebraic feedthrough* depth, not naive topological
/// distance: dynamic and source blocks sit at depth 0 because their outputs
/// depend on state/time, not on the current-step inputs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Schedule {
    /// Full linear evaluation order: DAG depths first, then algebraic-loop
    /// members. Covers every child (and the interface, if present).
    pub topo: Vec<BlockId>,
    /// Acyclic part, grouped by algebraic-feedthrough depth (each group can in
    /// principle be evaluated in parallel; backends decide whether to exploit
    /// that). Loop members live in `sccs`, not here.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<DagGroup>,
    /// Strongly-connected components: each is an algebraic loop solved
    /// iteratively (Anderson-accelerated fixed point in the runtime).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sccs: Vec<Scc>,
    /// Connection IDs cut as back-edges to break the loops (deduped union of
    /// every SCC's cut set).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub back_edges: Vec<ConnectionId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagGroup {
    pub depth: u32,
    pub blocks: Vec<BlockId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scc {
    pub blocks: Vec<BlockId>,
    pub back_edges: Vec<ConnectionId>,
}

// ======================================================================================
// Tests
// ======================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// `Source(sin(t)) -> Amplifier(2.0) -> Integrator(0)` mini model.
    fn mini_model() -> Module {
        // src: y = sin(t)
        let src = Block {
            id: BlockId(1),
            name: "src".into(),
            type_name: "Source".into(),
            role: BlockRole::Source,
            ports: Ports {
                inputs: vec![],
                outputs: vec![Port { name: "out".into(), size: 1 }],
            },
            params: vec![],
            state: vec![],
            memory: vec![],
            regions: Regions {
                alg: Region {
                    ops: vec![
                        Op::Time,
                        Op::Unary { op: UnaryOpKind::Sin, a: NodeId(0) },
                    ],
                    writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(1) }],
                },
                dyn_: Region::default(),
            },
            events: vec![],
        };

        // amp: y = gain * u, gain as a Param node (runtime-mutable, pathsim-faithful)
        let amp = Block {
            id: BlockId(2),
            name: "amp".into(),
            type_name: "Amplifier".into(),
            role: BlockRole::Algebraic,
            ports: Ports {
                inputs: vec![Port { name: "in".into(), size: 1 }],
                outputs: vec![Port { name: "out".into(), size: 1 }],
            },
            params: vec![Param {
                id: ParamId(0),
                name: "gain".into(),
                value: ParamValue::Scalar(2.0),
            }],
            state: vec![],
            memory: vec![],
            regions: Regions {
                alg: Region {
                    ops: vec![
                        Op::Input { port: 0, elem: 0 },
                        Op::Param { id: ParamId(0) },
                        Op::Binary { op: BinOpKind::Mul, a: NodeId(0), b: NodeId(1) },
                    ],
                    writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(2) }],
                },
                dyn_: Region::default(),
            },
            events: vec![],
        };

        // int: y = x, dx/dt = u
        let integ = Block {
            id: BlockId(3),
            name: "int".into(),
            type_name: "Integrator".into(),
            role: BlockRole::Dynamic,
            ports: Ports {
                inputs: vec![Port { name: "in".into(), size: 1 }],
                outputs: vec![Port { name: "out".into(), size: 1 }],
            },
            params: vec![],
            state: vec![StateVar { id: StateId(0), name: "x".into(), init: 0.0 }],
            memory: vec![],
            regions: Regions {
                alg: Region {
                    ops: vec![Op::State { id: StateId(0) }],
                    writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
                },
                dyn_: Region {
                    ops: vec![Op::Input { port: 0, elem: 0 }],
                    writes: vec![Write::StateDeriv { id: StateId(0), src: NodeId(0) }],
                },
            },
            events: vec![],
        };

        let conns = vec![
            Connection {
                id: ConnectionId(0),
                src: PortRef { block: BlockId(1), port: 0, elems: None },
                targets: vec![PortRef { block: BlockId(2), port: 0, elems: None }],
            },
            Connection {
                id: ConnectionId(1),
                src: PortRef { block: BlockId(2), port: 0, elems: None },
                targets: vec![PortRef { block: BlockId(3), port: 0, elems: None }],
            },
            Connection {
                id: ConnectionId(2),
                src: PortRef { block: BlockId(3), port: 0, elems: None },
                targets: vec![PortRef { block: BlockId::INTERFACE, port: 0, elems: None }],
            },
        ];

        let root = Subsystem {
            id: SubsystemId(0),
            name: "root".into(),
            interface: Interface {
                inputs: vec![],
                outputs: vec![Port { name: "y".into(), size: 1 }],
            },
            children: vec![
                Child::Block(src),
                Child::Block(amp),
                Child::Block(integ),
            ],
            connections: conns,
            schedule: Schedule {
                topo: vec![BlockId(1), BlockId(2), BlockId(3)],
                groups: vec![
                    DagGroup { depth: 0, blocks: vec![BlockId(1)] },
                    DagGroup { depth: 1, blocks: vec![BlockId(2)] },
                    DagGroup { depth: 2, blocks: vec![BlockId(3)] },
                ],
                sccs: vec![],
                back_edges: vec![],
            },
        };

        Module::new("demo", root)
    }

    #[test]
    fn module_json_roundtrip_mini_model() {
        let m1 = mini_model();
        let json = serde_json::to_string_pretty(&m1).expect("serialize");
        let m2: Module = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(m2.ir_version, IR_VERSION);
        assert_eq!(m2.name, "demo");
        assert_eq!(m2.root.children.len(), 3);
        assert_eq!(m2.root.connections.len(), 3);

        // Re-serialize and compare textually: any schema asymmetry (e.g. a
        // Default-skipped field that came back populated) trips the test.
        let json2 = serde_json::to_string_pretty(&m2).expect("re-serialize");
        assert_eq!(json, json2, "JSON roundtrip is not stable");
    }

    #[test]
    fn block_interface_sentinel_roundtrips() {
        let r = PortRef { block: BlockId::INTERFACE, port: 0, elems: None };
        let s = serde_json::to_string(&r).unwrap();
        let r2: PortRef = serde_json::from_str(&s).unwrap();
        assert_eq!(r2.block, BlockId::INTERFACE);
    }

    #[test]
    fn op_variants_roundtrip() {
        let cases = vec![
            Op::Const(1.5),
            Op::Time,
            Op::Input { port: 2, elem: 1 },
            Op::Param { id: ParamId(0) },
            Op::State { id: StateId(0) },
            Op::Memory { slot: MemorySlotId(0), offset: 3 },
            Op::Binary { op: BinOpKind::Mul, a: NodeId(3), b: NodeId(4) },
            Op::Unary { op: UnaryOpKind::Sin, a: NodeId(0) },
            Op::Cmp { op: CmpKind::Gt, a: NodeId(0), b: NodeId(1) },
            Op::Select { c: NodeId(0), t: NodeId(1), e: NodeId(2) },
            Op::Fma { a: NodeId(0), b: NodeId(1), c: NodeId(2) },
            Op::Reduce { op: ReduceKind::Sum, args: vec![NodeId(0), NodeId(1), NodeId(2)] },
            Op::Dot { a: vec![NodeId(0), NodeId(1)], b: vec![NodeId(2), NodeId(3)] },
            Op::Call { id: ExternId(0), args: vec![NodeId(0)], out_idx: 0 },
        ];
        for op in &cases {
            let s = serde_json::to_string(op).unwrap();
            let _: Op = serde_json::from_str(&s).expect(&s);
        }
    }

    #[test]
    fn empty_collections_skipped_in_json() {
        let b = Block {
            id: BlockId(0), name: "n".into(), type_name: "Constant".into(),
            role: BlockRole::Source,
            ports: Ports::default(),
            params: vec![], state: vec![], memory: vec![],
            regions: Regions::default(),
            events: vec![],
        };
        let json = serde_json::to_string(&b).unwrap();
        assert!(!json.contains("\"events\""), "empty events should skip");
        assert!(!json.contains("\"params\""), "empty params should skip");
        assert!(!json.contains("\"state\""), "empty state should skip");
    }
}
