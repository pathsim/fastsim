//! Build IR `schema` artifacts from the live runtime: convert a `jit::graph`
//! SSA graph into a `schema::Region`, and a runtime `Block`'s per-path
//! operators into a `schema::Block`.
//!
//! The graph and the region share node ordering (one schema `Op` per graph
//! node, in index order), so a graph `NodeId` maps to the schema `NodeId` of
//! the same index. Flat graph inputs are decoded back into structured reads
//! (`Input`/`State`/`Memory`/`Time`) using the graph signature's slot names
//! (the `blockops` convention).
//!
//! Port handling here is single-port (the whole `"u"` slot is input port 0, all
//! outputs are output port 0). True MIMO port splitting lands with subsystems
//! (WP8).

use std::rc::Rc;

use crate::blocks::block::{Block, BlockRef, BlockRole as RtRole};
use crate::blocks::blockops::{DiscreteResolved, EventSpec, Lut1dSpec, MemSpec, RegionGraph};
use crate::ssa::graph::{Graph, InputSignature, Node};
use crate::ir::schema::*;
use crate::simulation::Simulation;

/// Which side of a block a region produces, so outputs become the right writes.
#[derive(Clone, Copy)]
pub enum RegionRole {
    /// `alg`: outputs are `Write::Output` (mapped to output ports).
    Alg,
    /// `dyn`: outputs are `Write::StateDeriv` (one per state element).
    Dyn,
}

/// Per-block layout needed to decode flat graph indices into structured reads.
pub struct GraphDecode<'a> {
    /// Element count of each input port (the `"u"` slot is their concatenation).
    pub input_port_sizes: &'a [u32],
    /// Element count of each output port (for `alg` output writes).
    pub output_port_sizes: &'a [u32],
}

/// Lower a graph's nodes to schema `Op`s (index-aligned: schema `NodeId(i)`
/// is graph node `i`). Shared by every region kind.
fn ops_from_graph(graph: &Graph, dec: &GraphDecode) -> Vec<Op> {
    graph.nodes.iter().map(|n| node_to_op(n, &graph.signature, dec)).collect()
}

/// Build an event-effect `Region`: ops from the effect graph, plus a
/// `MemoryWrite` per output mapping `effect.outputs[i]` -> `targets[i]`.
pub fn region_from_graph_mem(graph: &Graph, dec: &GraphDecode, targets: &[crate::blocks::blockops::MemTarget]) -> Region {
    let ops = ops_from_graph(graph, dec);
    let writes = graph
        .outputs
        .iter()
        .zip(targets.iter())
        .map(|(&nid, t)| Write::MemoryWrite { slot: MemorySlotId(t.slot), offset: t.offset, src: NodeId(nid) })
        .collect();
    Region { ops, writes }
}

/// Convert a `jit::graph::Graph` into a `schema::Region`. Emits one `Op` per
/// graph node (index-aligned), then maps the graph outputs to writes per `role`.
pub fn region_from_graph(graph: &Graph, role: RegionRole, dec: &GraphDecode) -> Region {
    let ops = ops_from_graph(graph, dec);

    let writes = match role {
        RegionRole::Alg => graph
            .outputs
            .iter()
            .enumerate()
            .map(|(i, &nid)| {
                let (port, elem) = split_port(i as u32, dec.output_port_sizes);
                Write::Output { port, elem, src: NodeId(nid) }
            })
            .collect(),
        RegionRole::Dyn => graph
            .outputs
            .iter()
            .enumerate()
            .map(|(i, &nid)| Write::StateDeriv { id: StateId(i as u32), src: NodeId(nid) })
            .collect(),
    };

    Region { ops, writes }
}

fn node_to_op(node: &Node, sig: &InputSignature, dec: &GraphDecode) -> Op {
    match node {
        Node::Const(bits) => Op::Const(f64::from_bits(*bits)),
        Node::Input(flat) => decode_input(*flat, sig, dec),
        Node::Param(idx) => Op::Param { id: ParamId(*idx) },
        Node::Binary(op, a, b) => Op::Binary { op: *op, a: NodeId(*a), b: NodeId(*b) },
        Node::Unary(op, a) => Op::Unary { op: *op, a: NodeId(*a) },
        Node::Cmp(op, a, b) => Op::Cmp { op: *op, a: NodeId(*a), b: NodeId(*b) },
        Node::Select(c, t, e) => Op::Select { c: NodeId(*c), t: NodeId(*t), e: NodeId(*e) },
        Node::Fma(a, b, c) => Op::Fma { a: NodeId(*a), b: NodeId(*b), c: NodeId(*c) },
        Node::Reduce(op, args) => Op::Reduce {
            op: *op,
            args: args.iter().map(|&a| NodeId(a)).collect(),
        },
        Node::Dot(a, b) => Op::Dot {
            a: a.iter().map(|&n| NodeId(n)).collect(),
            b: b.iter().map(|&n| NodeId(n)).collect(),
        },
    }
}

/// Decode a flat input index into a structured read via the slot it falls in.
/// The slot-name convention is classified by `blockops::slot_kind`, the single
/// authority shared with the runtime slot plan (so they cannot drift).
fn decode_input(flat: u32, sig: &InputSignature, dec: &GraphDecode) -> Op {
    use crate::blocks::blockops::SlotKind;
    let (slot_idx, elem) = sig.decode(flat as usize);
    let name = sig.slots.get(slot_idx).map(|s| s.name.as_str()).unwrap_or("u");
    match crate::blocks::blockops::slot_kind(name) {
        SlotKind::State => Op::State { id: StateId(elem as u32) },
        SlotKind::Time => Op::Time,
        SlotKind::Memory(k) => Op::Memory { slot: MemorySlotId(k), offset: elem as u32 },
        // "u": split the concatenated input vector back into (port, elem).
        SlotKind::Input => {
            let (port, port_elem) = split_port(elem as u32, dec.input_port_sizes);
            Op::Input { port, elem: port_elem }
        }
    }
}

/// Map a flat element index across a list of port sizes to `(port, elem)`.
/// Falls back to port 0 when `sizes` is empty (single-port convention).
fn split_port(flat: u32, sizes: &[u32]) -> (u32, u32) {
    let mut acc = 0u32;
    for (p, &sz) in sizes.iter().enumerate() {
        if flat < acc + sz {
            return (p as u32, flat - acc);
        }
        acc += sz;
    }
    (0, flat)
}

fn map_role(r: RtRole) -> BlockRole {
    if r.is_rec {
        BlockRole::Sink
    } else if r.is_src {
        BlockRole::Source
    } else if r.is_dyn {
        BlockRole::Dynamic
    } else {
        BlockRole::Algebraic
    }
}

/// The op-graph pieces the IR builder lowers for one leaf block, borrowed from
/// a runtime block's per-path operators (the single source of truth). Replaces
/// the former `BlockOps` aggregate now that the `Operator` is the block's sole
/// graph representation.
pub struct LeafOps<'a> {
    pub type_name: &'a str,
    /// Algebraic region: `y = f(x, u, t, mem)`.
    pub alg: &'a RegionGraph,
    /// Dynamic region: `dx/dt = g(x, u, t)`. `None` for pure-algebraic blocks.
    pub dyn_: Option<&'a RegionGraph>,
    /// Continuous-state initial values (length == the `"x"` slot size).
    pub state_init: &'a [f64],
    /// Discrete memory slots (sampled / event-driven blocks).
    pub memory: &'a [MemSpec],
    /// Block-internal events (Schedule / ZeroCross / Condition).
    pub events: &'a [EventSpec],
    /// Shape-poly discrete blocks: resolves `(alg, memory, events)` at the
    /// connected input width, used in place of `alg`/`memory`/`events`.
    pub discrete_builder: Option<&'a Rc<dyn Fn(usize) -> Option<DiscreteResolved>>>,
    /// Lookup-table structure: when set, `alg` lowers to a single `Op::Lut1d`.
    pub lut1d: Option<&'a Lut1dSpec>,
}

/// Build a `schema::Block` from a runtime block's per-path operators.
/// `input_width` is the connected input element count (drives shape-lazy
/// resolution and port decoding). Returns `None` if the block carries no
/// op-graph (opaque/unported): those surface as an extern `Op::Call`.
pub fn block_ops_to_ir_block(block: &Block, id: BlockId) -> Option<Block_> {
    let input_width = block.inputs.len() as u32;
    let output_width = block.outputs.len() as u32;
    // A shape-poly discrete block carries no fixed `alg_op` (its alg comes from
    // the resolver), so a placeholder graph stands in and `build_from_ops` uses
    // the resolver instead.
    if block.alg_op.is_none() && block.op_discrete_builder.is_none() {
        return None; // opaque (no op-graph): surfaces as an extern `Op::Call`
    }
    let placeholder;
    let alg: &RegionGraph = match block.alg_op.as_ref().and_then(|o| o.graph_ref()) {
        Some(rg) => rg,
        None => {
            placeholder = RegionGraph::Fixed(Graph::new(InputSignature::empty()));
            &placeholder
        }
    };
    let state_init = block.initial_value.as_deref().unwrap_or(&[]);
    let ops = LeafOps {
        type_name: block.op_type_name.unwrap_or(block.type_name),
        alg,
        dyn_: block.dyn_op.as_ref().and_then(|o| o.graph_ref()),
        state_init,
        memory: &block.op_memory,
        events: &block.op_events,
        discrete_builder: block.op_discrete_builder.as_ref(),
        lut1d: block.alg_op.as_ref().and_then(|o| o.lut1d.as_ref()),
    };
    // `None` here means a `Lazy` region could not lower at the connected width
    // (opaque): the block surfaces as an extern `Op::Call`, same as any unported
    // block, instead of panicking.
    build_from_ops(&ops, &block.role, id, input_width, output_width)
}

// Alias so the doc signature reads clearly without colliding with runtime Block.
type Block_ = crate::ir::schema::Block;

/// Core construction, separated so tests can drive it without a runtime Block.
/// Returns `None` when a `Lazy` region cannot lower at this width (a Python
/// callable with data-dependent control flow): the block is then opaque and the
/// caller surfaces it as an extern `Op::Call` instead.
pub fn build_from_ops(
    ops: &LeafOps,
    role: &RtRole,
    id: BlockId,
    input_width: u32,
    output_width: u32,
) -> Option<Block_> {
    let in_sizes = vec![input_width];

    // Shape-poly discrete blocks resolve (alg, memory, events) at the input width.
    let (alg_graph, mem_specs, evt_specs) = match ops.discrete_builder {
        // A discrete builder that returns `None` at this width is not
        // op-traceable here (e.g. a JIT-traced Wrapper whose Python effect does
        // not trace) -> opaque (the caller emits an extern Call).
        Some(build) => build(input_width as usize)?,
        // A non-resolvable `Lazy` alg region means the block is not op-traceable
        // here -> opaque (the caller emits an extern Call).
        None => (ops.alg.resolve(input_width as usize)?, ops.memory.to_vec(), ops.events.to_vec()),
    };
    // The algebraic graph is the single source of truth for the block's output
    // arity: a block may declare fewer output ports than its region actually
    // writes (e.g. an n-bit ADC whose bit ports resolve lazily), and undersizing
    // the output layout overflows the `sig[]` buffer. Take the larger of the two.
    let output_width = output_width.max(alg_graph.outputs.len() as u32);
    let out_sizes = vec![output_width];
    let dec = GraphDecode { input_port_sizes: &in_sizes, output_port_sizes: &out_sizes };
    // A LUT1D block lowers to a single structured `Op::Lut1d` (the table) rather
    // than tracing its equivalent select-chain graph, so table-aware backends
    // emit a real lookup table. The runtime closure and `splice` are unaffected.
    let alg = if let Some(lut) = &ops.lut1d {
        Region {
            ops: vec![
                Op::Input { port: 0, elem: 0 },
                Op::Lut1d {
                    input: NodeId(0),
                    points: lut.points.clone(),
                    values: lut.values.clone(),
                    clamp: lut.clamp,
                },
            ],
            writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(1) }],
        }
    } else {
        region_from_graph(&alg_graph, RegionRole::Alg, &dec)
    };

    let dyn_ = match &ops.dyn_ {
        Some(rg) => {
            let g = rg.resolve(input_width as usize)?;
            region_from_graph(&g, RegionRole::Dyn, &dec)
        }
        None => Region::default(),
    };

    // Params come from the alg graph (the dyn graph shares the same set).
    let params = alg_graph
        .param_names
        .iter()
        .zip(alg_graph.param_defaults.iter())
        .enumerate()
        .map(|(i, (name, &v))| Param {
            id: ParamId(i as u32),
            name: name.clone(),
            value: ParamValue::Scalar(v),
        })
        .collect();

    let state = ops
        .state_init
        .iter()
        .enumerate()
        .map(|(i, &init)| StateVar { id: StateId(i as u32), name: format!("x{i}"), init })
        .collect();

    let ports = Ports {
        inputs: if input_width > 0 {
            vec![Port { name: "in".into(), size: input_width }]
        } else {
            vec![]
        },
        outputs: if output_width > 0 {
            vec![Port { name: "out".into(), size: output_width }]
        } else {
            vec![]
        },
    };

    // Discrete memory slots + the events that update them.
    let memory = mem_specs
        .iter()
        .enumerate()
        .map(|(i, m)| MemorySlot {
            id: MemorySlotId(i as u32),
            name: m.name.clone(),
            size: m.init.len() as u32,
            init: m.init.clone(),
        })
        .collect();
    let events = evt_specs
        .iter()
        .enumerate()
        .map(|(i, e)| Event {
            id: EventId(i as u32),
            kind: map_event_kind(&e.kind, &dec),
            effect: region_from_graph_mem(&e.effect, &dec, &e.targets),
            opaque: false,
        })
        .collect();

    Some(Block_ {
        id,
        name: ops.type_name.to_string(),
        type_name: ops.type_name.to_string(),
        role: map_role(*role),
        ports,
        params,
        state,
        memory,
        regions: Regions { alg, dyn_ },
        events,
    })
}

fn map_event_kind(k: &crate::blocks::blockops::EventKindSpec, dec: &GraphDecode) -> EventKind {
    use crate::blocks::blockops::{DirSpec, EventKindSpec as K};
    // A guard region is the graph's ops with no writes (its value is the last op).
    let guard_region = |g: &Graph| Region { ops: ops_from_graph(g, dec), writes: vec![] };
    match k {
        K::SchedulePeriodic { period, phase } => {
            EventKind::Schedule { times: ScheduleTimes::Periodic { period: *period, phase: *phase } }
        }
        K::ScheduleFixed(ts) => EventKind::Schedule { times: ScheduleTimes::Fixed(ts.clone()) },
        K::ZeroCross { guard, direction } => EventKind::ZeroCross {
            guard: guard_region(guard),
            direction: match direction {
                DirSpec::Rising => Direction::Rising,
                DirSpec::Falling => Direction::Falling,
                DirSpec::Both => Direction::Both,
            },
        },
        K::Condition { guard } => EventKind::Condition { guard: guard_region(guard) },
    }
}

/// IR block for a runtime block carrying no `ops` (opaque: Scope, RNG, FMU,
/// event-driven and arbitrary-callable blocks). Represented honestly as a
/// single `extern Op::Call` whose output is the block's outputs: the IR records
/// the block exists with its arity, but makes no claim about its math. Returns
/// the block plus the `ExternDecl` to register on the module.
/// Build an opaque IR `Event` from a runtime event's statically-known
/// descriptor: the kind/timing is recorded, the guard/action stay opaque
/// (`opaque = true`, empty guard/effect regions). Shared by opaque blocks and
/// simulation-level events.
fn opaque_event(id: EventId, d: &crate::events::eventtype::EventDescriptor) -> Event {
    use crate::events::eventtype::{CrossDir, EventDescriptor as D};
    let kind = match d {
        D::SchedulePeriodic { period, phase } => {
            EventKind::Schedule { times: ScheduleTimes::Periodic { period: *period, phase: *phase } }
        }
        D::ScheduleFixed { times } => EventKind::Schedule { times: ScheduleTimes::Fixed(times.clone()) },
        D::ZeroCross { direction } => EventKind::ZeroCross {
            guard: Region::default(),
            direction: match direction {
                CrossDir::Rising => Direction::Rising,
                CrossDir::Falling => Direction::Falling,
                CrossDir::Both => Direction::Both,
            },
        },
        D::Condition => EventKind::Condition { guard: Region::default() },
    };
    Event { id, kind, effect: Region::default(), opaque: true }
}

/// Lower a block's runtime events into opaque IR events (kind/timing known,
/// guard/action host code). Used for opaque blocks whose math isn't op-bearing.
fn opaque_block_events(b: &Block) -> Vec<Event> {
    b.events
        .iter()
        .enumerate()
        .map(|(i, e)| opaque_event(EventId(i as u32), &e.borrow().ir_descriptor()))
        .collect()
}

fn opaque_block(b: &Block, id: BlockId, extern_id: ExternId) -> (Block_, ExternDecl) {
    let inw = b.inputs.len() as u32;
    let outw = b.outputs.len() as u32;

    // alg region: read all inputs, then one Call per output element.
    let mut ops: Vec<Op> = (0..inw).map(|e| Op::Input { port: 0, elem: e }).collect();
    let arg_ids: Vec<NodeId> = (0..inw).map(NodeId).collect();
    let mut writes = Vec::with_capacity(outw as usize);
    for k in 0..outw {
        let call_id = NodeId(ops.len() as u32);
        ops.push(Op::Call { id: extern_id, args: arg_ids.clone(), out_idx: k });
        writes.push(Write::Output { port: 0, elem: k, src: call_id });
    }
    let regions = if outw > 0 {
        Regions { alg: Region { ops, writes }, dyn_: Region::default() }
    } else {
        // pure sink (Scope): no outputs, nothing to call.
        Regions::default()
    };

    let block = Block_ {
        id,
        name: b.type_name.to_string(),
        type_name: b.type_name.to_string(),
        role: map_role(b.role),
        ports: Ports {
            inputs: if inw > 0 { vec![Port { name: "in".into(), size: inw }] } else { vec![] },
            outputs: if outw > 0 { vec![Port { name: "out".into(), size: outw }] } else { vec![] },
        },
        params: vec![],
        state: vec![],
        memory: vec![],
        regions,
        events: opaque_block_events(b),
    };
    let decl = ExternDecl {
        id: extern_id,
        name: b.type_name.to_string(),
        arity_in: inw,
        arity_out: outw,
    };
    (block, decl)
}

/// A PortReference's resolved channel indices as element slicing. `[0]` (the
/// SISO default) maps to `None` ("whole port"); anything else is explicit.
fn elems_of(indices: Vec<usize>) -> Option<Vec<u32>> {
    if indices == [0] {
        None
    } else {
        Some(indices.into_iter().map(|i| i as u32).collect())
    }
}

/// Outer-facing interface of a subsystem wrapper, from its port counts.
fn wrapper_interface(b: &Block) -> Interface {
    let inw = b.inputs.len() as u32;
    let outw = b.outputs.len() as u32;
    Interface {
        inputs: if inw > 0 { vec![Port { name: "in".into(), size: inw }] } else { vec![] },
        outputs: if outw > 0 { vec![Port { name: "out".into(), size: outw }] } else { vec![] },
    }
}

/// Recursively build a `Subsystem` IR node from a block list + connections.
/// `interface` (when `Some`) is the inner Interface block: it is excluded from
/// `children` and connections to/from it are rewritten to `BlockId::INTERFACE`.
#[allow(clippy::too_many_arguments)]
fn build_scope(
    id: SubsystemId,
    name: String,
    blocks: &[BlockRef],
    connections: &[crate::connection::ConnectionRef],
    interface: Option<&BlockRef>,
    next_id: &mut u32,
    extern_decls: &mut Vec<ExternDecl>,
) -> Subsystem {
    let mut children = Vec::with_capacity(blocks.len());
    for (i, blk) in blocks.iter().enumerate() {
        let guard = blk.borrow();
        let bid = BlockId(i as u32);
        let child = if let Some(inner_rc) = guard.subsystem_inner.clone() {
            *next_id += 1;
            let sub_id = SubsystemId(*next_id);
            let inner = inner_rc.borrow();
            let mut nested = build_scope(
                sub_id,
                format!("{name}/sub{i}"),
                &inner.blocks,
                &inner.connections,
                Some(&inner.interface),
                next_id,
                extern_decls,
            );
            // The subsystem's outer-facing interface is the wrapper block's I/O.
            nested.interface = wrapper_interface(guard);
            Child::Subsystem(nested)
        } else {
            match block_ops_to_ir_block(guard, bid) {
                Some(b) => Child::Block(b),
                None => {
                    let eid = ExternId(extern_decls.len() as u32);
                    let (b, decl) = opaque_block(guard, bid, eid);
                    extern_decls.push(decl);
                    Child::Block(b)
                }
            }
        };
        children.push(child);
    }

    // Block identity -> BlockId within this scope; the interface is the sentinel.
    let idx_of = |b: &BlockRef| -> Option<BlockId> {
        if let Some(iface) = interface {
            if Rc::ptr_eq(b, iface) {
                return Some(BlockId::INTERFACE);
            }
        }
        blocks.iter().position(|x| Rc::ptr_eq(x, b)).map(|i| BlockId(i as u32))
    };

    let mut conns = Vec::with_capacity(connections.len());
    for (i, c) in connections.iter().enumerate() {
        let Some(sb) = idx_of(&c.source.block) else { continue };
        let src = PortRef { block: sb, port: 0, elems: elems_of(c.source._get_output_indices()) };
        let targets = c
            .targets
            .iter()
            .filter_map(|t| {
                idx_of(&t.block).map(|tb| PortRef { block: tb, port: 0, elems: elems_of(t._get_input_indices()) })
            })
            .collect();
        conns.push(Connection { id: ConnectionId(i as u32), src, targets });
    }

    let schedule = build_schedule(blocks, connections, interface);
    Subsystem {
        id,
        name,
        interface: Interface::default(),
        children,
        connections: conns,
        schedule,
    }
}

/// Derive the IR `Schedule` for one scope from its connections + block roles,
/// reusing the runtime's graph analysis (`utils::schedule::Graph`) verbatim so the
/// IR schedule and the live evaluation order can never drift.
///
/// Node-id convention mirrors `subsystem.rs`: regular blocks are nodes
/// `0..blocks.len()` (== their `BlockId`), the optional interface is the last
/// node (`blocks.len()`) and maps to `BlockId::INTERFACE`. The schedule is
/// *derived/advisory* metadata; `connections` remain the source of truth and a
/// consumer may recompute everything here from them.
fn build_schedule(
    blocks: &[BlockRef],
    connections: &[crate::connection::ConnectionRef],
    interface: Option<&BlockRef>,
) -> Schedule {
    let iface_node = blocks.len();

    // Build the schedule through the SAME port-granular assembly the runtime
    // uses (`assemble_graph_from`), not a coarse block-level rebuild, so the IR
    // schedule cannot drift from the live evaluation order. Node convention:
    // regular blocks `0..n`, the optional interface appended last (== `iface_node`).
    let mut all = blocks.to_vec();
    if let Some(iface) = interface {
        all.push(iface.clone());
    }
    let g = crate::simulation::assemble_graph_from(&all, connections);

    // Graph node id -> IR BlockId (interface node -> sentinel).
    let to_bid = |node: usize| -> BlockId {
        if interface.is_some() && node == iface_node {
            BlockId::INTERFACE
        } else {
            BlockId(node as u32)
        }
    };

    let topo: Vec<BlockId> = g.topo_order().into_iter().map(to_bid).collect();

    let groups: Vec<DagGroup> = g
        .dag_iter()
        .enumerate()
        .filter(|(_, (blks, _))| !blks.is_empty())
        .map(|(depth, (blks, _))| DagGroup {
            depth: depth as u32,
            blocks: blks.iter().map(|&n| to_bid(n)).collect(),
        })
        .collect();

    let sccs: Vec<Scc> = g
        .algebraic_loops()
        .map(|(blks, back)| Scc {
            blocks: blks.iter().map(|&n| to_bid(n)).collect(),
            back_edges: back.iter().map(|&c| ConnectionId(c as u32)).collect(),
        })
        .collect();

    // Global back-edge list (deduped): the union of every SCC's cut set.
    let mut back_raw: Vec<usize> = g.loop_closing_connections().to_vec();
    back_raw.sort_unstable();
    back_raw.dedup();
    let back_edges: Vec<ConnectionId> =
        back_raw.into_iter().map(|c| ConnectionId(c as u32)).collect();

    Schedule { topo, groups, sccs, back_edges }
}

/// Build a standalone IR [`Subsystem`] from a runtime subsystem wrapper block,
/// recursing into nested subsystems. The subsystem's interface reflects the
/// wrapper's resolved I/O widths, so the block must be assembled (part of a sim
/// that has been run, or otherwise port-resolved) before calling. Returns `None`
/// for a non-subsystem (leaf) block.
///
/// This is the per-subsystem analogue of [`module_from_sim`]: it yields exactly
/// the `Child::Subsystem` node that `module_from_sim` would embed, ready to feed
/// to [`crate::compile::compile_block`].
pub fn subsystem_to_ir(block: &BlockRef) -> Option<Subsystem> {
    let guard = block.borrow();
    let inner_rc = guard.subsystem_inner.clone()?;
    let inner = inner_rc.borrow();
    let mut next_id = 0u32;
    let mut extern_decls: Vec<ExternDecl> = Vec::new();
    let mut sub = build_scope(
        SubsystemId(0),
        "subsystem".to_string(),
        &inner.blocks,
        &inner.connections,
        Some(&inner.interface),
        &mut next_id,
        &mut extern_decls,
    );
    // The outer-facing interface is the wrapper block's resolved I/O.
    sub.interface = wrapper_interface(guard);
    Some(sub)
}

/// Build a `Module` whose root is a single subsystem *with its interface live*
/// (the open-system analogue of [`module_from_sim`]). The root interface's input
/// ports become external inputs (`u[]`) and its output ports observable outputs,
/// so codegen / FMU export treat the subsystem as an open block `f(x, u, t)`
/// rather than a closed system. Returns `None` for a non-subsystem block.
pub fn module_from_subsystem(block: &BlockRef, name: impl Into<String>) -> Option<Module> {
    subsystem_to_ir(block).map(|sub| Module::new(name, sub))
}

/// Build a `Module` snapshot from an assembled `Simulation`, recursing into
/// nested subsystems and mapping MIMO channel slicing. Blocks with `ops` become
/// full IR blocks; others become typed `extern` blocks.
pub fn module_from_sim(sim: &Simulation, name: impl Into<String>) -> Module {
    let mut extern_decls: Vec<ExternDecl> = Vec::new();
    let mut next_id = 0u32;
    let root = build_scope(
        SubsystemId(0),
        "root".to_string(),
        &sim.blocks,
        &sim.connections,
        None,
        &mut next_id,
        &mut extern_decls,
    );
    // Simulation-level (global) events: standalone events registered on the
    // sim, not attached to any block. Guards/actions are host closures, so they
    // are recorded as opaque (kind/timing only).
    let events = sim
        .events
        .iter()
        .enumerate()
        .map(|(i, e)| opaque_event(EventId(i as u32), &e.borrow().ir_descriptor()))
        .collect();

    let mut m = Module::new(name, root);
    m.events = events;
    m.extern_decls = extern_decls;
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::constructors::core::{
        adder_eval, adder_graph, amplifier_eval, amplifier_graph, constant_eval, constant_graph,
        integrator_alg_eval, integrator_alg_graph, integrator_dyn_eval, integrator_dyn_graph,
    };
    use crate::blocks::constructors::sources::{sinusoidal_eval, sinusoidal_source_graph};
    use crate::ir::eval::{eval_region, EvalCtx};
    use crate::ssa::build::F64Builder;

    fn dec1(inw: u32, outw: u32) -> ([u32; 1], [u32; 1]) {
        ([inw], [outw])
    }

    // ----------------------------------------------------------------------------------
    // Op-vocabulary parity: jit interpret == jit tape == ir eval, per sub-enum variant.
    //
    // The shared op set is transcribed into three numeric backends (the jit
    // recursive `interpret`, the flat `Tape` interpreter, and `ir::eval`). This
    // sweep locks them together so the upcoming sub-enum merge (R2) and the
    // f64-semantics consolidation (R3) cannot silently diverge: a wrong mapping
    // or a desynced math function trips it immediately. Each `all_*()` lists its
    // enum's variants exactly once; the inner `_guard` match is exhaustive, so
    // adding a variant to the enum fails to compile until it is listed here too.
    // ----------------------------------------------------------------------------------
    use crate::ssa::graph::{BinOp, CmpOp, ReduceOp, UnaryOp};
    use crate::ssa::tape::InterpretedFn;

    macro_rules! variant_list {
        ($name:ident, $ty:ty, $($v:ident),+ $(,)?) => {
            fn $name() -> Vec<$ty> {
                fn _guard(o: $ty) { match o { $(<$ty>::$v => {}),+ } }
                vec![$(<$ty>::$v),+]
            }
        };
    }
    variant_list!(all_bin, BinOp, Add, Sub, Mul, Div, Pow, Mod, Min, Max, Atan2, Hypot);
    variant_list!(
        all_unary, UnaryOp,
        Neg, Sin, Cos, Tan, Atan, Sinh, Cosh, Tanh, Exp, Log, Log10, Abs, Sqrt, Sign, Floor,
        Asin, Acos, Asinh, Acosh, Atanh, Ceil, Round, Trunc, Log2, Log1p, Expm1, Cbrt,
        Erf, Erfc, Lgamma, Tgamma, Digamma, RandUniform,
    );
    variant_list!(all_cmp, CmpOp, Gt, Ge, Lt, Le, Eq, Ne);
    variant_list!(all_reduce, ReduceOp, Sum, Product, Min, Max);

    /// Agreement predicate: exact, with a tiny relative slack, and NaN==NaN
    /// counts as agreement (out-of-domain ops produce NaN identically in all
    /// three backends, which is itself the property we want to lock).
    fn parity_close(a: f64, b: f64) -> bool {
        (a.is_nan() && b.is_nan()) || a == b || (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
    }

    /// Evaluate one finished graph (with its mirror region) three ways and
    /// assert all backends agree element-wise.
    fn assert_three_way(label: &str, g: Graph, region: &Region, u: &[f64]) {
        let interp = g.interpret(&[u], &[]);
        let f = InterpretedFn::from_graph(g);
        let tape = f.call(&[u]);
        let ctx = EvalCtx { inputs: &[u], state: &[], memory: &[], params: &[], t: 0.0 };
        let ir = eval_region(region, &ctx).unwrap();
        assert_eq!(tape.len(), interp.len(), "{label}: tape arity");
        assert_eq!(ir.len(), interp.len(), "{label}: ir arity");
        for i in 0..interp.len() {
            assert!(parity_close(interp[i], tape[i]), "{label}: jit interpret {} != tape {}", interp[i], tape[i]);
            assert!(parity_close(interp[i], ir[i]), "{label}: jit interpret {} != ir eval {}", interp[i], ir[i]);
        }
    }

    /// In-domain probe for a unary op (avoids NaN where the function is
    /// restricted, and picks a negative point for `Sign`). New variants fall to
    /// the default and will surface loudly if the default is out of domain.
    fn unary_probe(op: UnaryOp) -> f64 {
        use UnaryOp::*;
        match op {
            Acosh => 1.7,
            Asin | Acos | Atanh => 0.4,
            Log | Log10 | Log2 | Sqrt | Cbrt | Lgamma | Tgamma | Digamma => 1.3,
            Sign => -0.7,
            _ => 0.5,
        }
    }

    #[test]
    fn op_vocab_unary_parity() {
        for op in all_unary() {
            let x = unary_probe(op);
            let mut g = Graph::with_single_input("u", 1);
            let n0 = g.input(0);
            let nr = g.unary(op, n0);
            g.outputs.push(nr);
            let region = region_from_graph(
                &g, RegionRole::Alg,
                &GraphDecode { input_port_sizes: &[1], output_port_sizes: &[1] },
            );
            assert_three_way(&format!("unary {op:?}"), g, &region, &[x]);
        }
    }

    #[test]
    fn op_vocab_binary_parity() {
        for op in all_bin() {
            for (a, b) in [(2.5_f64, 1.3_f64), (1.3, 2.5), (-1.7, 0.9)] {
                let mut g = Graph::with_single_input("u", 2);
                let na = g.input(0);
                let nb = g.input(1);
                let nr = g.binary(op, na, nb);
                g.outputs.push(nr);
                let region = region_from_graph(
                    &g, RegionRole::Alg,
                    &GraphDecode { input_port_sizes: &[2], output_port_sizes: &[1] },
                );
                assert_three_way(&format!("binary {op:?} ({a},{b})"), g, &region, &[a, b]);
            }
        }
    }

    #[test]
    fn op_vocab_cmp_parity() {
        for op in all_cmp() {
            for (a, b) in [(2.5_f64, 1.3_f64), (1.3, 2.5), (1.3, 1.3)] {
                let mut g = Graph::with_single_input("u", 2);
                let na = g.input(0);
                let nb = g.input(1);
                let nr = g.cmp(op, na, nb);
                g.outputs.push(nr);
                let region = region_from_graph(
                    &g, RegionRole::Alg,
                    &GraphDecode { input_port_sizes: &[2], output_port_sizes: &[1] },
                );
                assert_three_way(&format!("cmp {op:?} ({a},{b})"), g, &region, &[a, b]);
            }
        }
    }

    #[test]
    fn op_vocab_reduce_parity() {
        for op in all_reduce() {
            let mut g = Graph::with_single_input("u", 3);
            let args: Vec<u32> = (0..3).map(|i| g.input(i)).collect();
            let nr = g.reduce(op, args);
            g.outputs.push(nr);
            let region = region_from_graph(
                &g, RegionRole::Alg,
                &GraphDecode { input_port_sizes: &[3], output_port_sizes: &[1] },
            );
            assert_three_way(&format!("reduce {op:?}"), g, &region, &[1.5, -2.0, 3.0]);
        }
    }

    /// Amplifier: graph -> Region -> eval must equal the native generic eval.
    #[test]
    fn amplifier_ir_matches_native() {
        let gain = 2.5;
        let n = 3;
        let g = amplifier_graph(gain, n);
        let (ins, outs) = dec1(n as u32, n as u32);
        let dec = GraphDecode { input_port_sizes: &ins, output_port_sizes: &outs };
        let region = region_from_graph(&g, RegionRole::Alg, &dec);

        let u = [1.0, -2.0, 3.5];
        let mut native = Vec::new();
        amplifier_eval(&F64Builder, gain, &u, &mut native);

        let ctx = EvalCtx { inputs: &[&u], state: &[], memory: &[], params: &[gain], t: 0.0 };
        let ir = eval_region(&region, &ctx).unwrap();
        assert_eq!(ir, native);
    }

    #[test]
    fn adder_ir_matches_native() {
        let ops = [1.0, -1.0];
        let n = 2;
        let g = adder_graph(&ops, n);
        let (ins, outs) = dec1(n as u32, 1);
        let dec = GraphDecode { input_port_sizes: &ins, output_port_sizes: &outs };
        let region = region_from_graph(&g, RegionRole::Alg, &dec);

        let u = [3.0, 1.25];
        let mut native = Vec::new();
        adder_eval(&F64Builder, &ops, &u, &mut native);

        let ctx = EvalCtx { inputs: &[&u], state: &[], memory: &[], params: &[], t: 0.0 };
        assert_eq!(eval_region(&region, &ctx).unwrap(), native);
    }

    #[test]
    fn constant_ir_matches_native() {
        let g = constant_graph(3.5);
        let dec = GraphDecode { input_port_sizes: &[], output_port_sizes: &[1] };
        let region = region_from_graph(&g, RegionRole::Alg, &dec);
        let mut native = Vec::new();
        constant_eval(3.5, &mut native);
        let ctx = EvalCtx { inputs: &[], state: &[], memory: &[], params: &[3.5], t: 0.0 };
        assert_eq!(eval_region(&region, &ctx).unwrap(), native);
    }

    #[test]
    fn integrator_ir_matches_native() {
        let n = 2;
        let alg_g = integrator_alg_graph(n);
        let dyn_g = integrator_dyn_graph(n);
        let (ins, outs) = dec1(n as u32, n as u32);
        let dec = GraphDecode { input_port_sizes: &ins, output_port_sizes: &outs };
        let alg = region_from_graph(&alg_g, RegionRole::Alg, &dec);
        let dyn_ = region_from_graph(&dyn_g, RegionRole::Dyn, &dec);

        let x = [0.7, -1.3];
        let u = [2.0, 5.0];
        // alg: y = x (reads state)
        let mut native_alg = Vec::new();
        integrator_alg_eval(&x, &mut native_alg);
        let ctx_alg = EvalCtx { inputs: &[&u], state: &x, memory: &[], params: &[], t: 0.0 };
        assert_eq!(eval_region(&alg, &ctx_alg).unwrap(), native_alg);
        // dyn: dx/dt = u
        let mut native_dyn = Vec::new();
        integrator_dyn_eval(&u, &mut native_dyn);
        let ctx_dyn = EvalCtx { inputs: &[&u], state: &x, memory: &[], params: &[], t: 0.0 };
        assert_eq!(eval_region(&dyn_, &ctx_dyn).unwrap(), native_dyn);
    }

    /// Source variant: evaluate the IR alg region across several `t` and
    /// compare to the runtime `f_alg` (sources read time, not inputs).
    fn assert_source_ir_matches_runtime(blk: &crate::blocks::block::BlockRef, ts: &[f64]) {
        let b = blk.borrow();
        let ir = block_ops_to_ir_block(b, BlockId(0)).expect("source has ops");
        let params: Vec<f64> = ir
            .params
            .iter()
            .map(|p| match p.value {
                ParamValue::Scalar(v) => v,
                _ => 0.0,
            })
            .collect();
        for &t in ts {
            let ctx = EvalCtx { inputs: &[], state: &[], memory: &[], params: &params, t };
            let ir_out = eval_region(&ir.regions.alg, &ctx).unwrap();
            let mut rt = Vec::new();
            b.f_alg.as_ref().unwrap()(&[], &[], t, &mut rt);
            assert_eq!(ir_out.len(), rt.len(), "{} arity", ir.type_name);
            for (a, e) in ir_out.iter().zip(rt.iter()) {
                assert!((a - e).abs() < 1e-12, "{} @t={t}: ir={a} rt={e}", ir.type_name);
            }
        }
    }

    /// Dynamic block: both alg (y) and dyn (dx/dt) IR regions must match the
    /// runtime `f_alg`/`f_dyn` at a given state `x` and input `u`.
    fn assert_dynamic_ir_matches_runtime(blk: &crate::blocks::block::BlockRef, x: &[f64], u: &[f64]) {
        let b = blk.borrow();
        let ir = block_ops_to_ir_block(b, BlockId(0)).expect("dynamic block has ops");
        let params: Vec<f64> = ir
            .params
            .iter()
            .map(|p| match p.value {
                ParamValue::Scalar(v) => v,
                _ => 0.0,
            })
            .collect();
        let ctx = EvalCtx { inputs: &[u], state: x, memory: &[], params: &params, t: 0.0 };

        let ir_alg = eval_region(&ir.regions.alg, &ctx).unwrap();
        let mut rt_alg = Vec::new();
        b.f_alg.as_ref().unwrap()(x, u, 0.0, &mut rt_alg);
        assert_eq!(ir_alg.len(), rt_alg.len(), "{} alg arity", ir.type_name);
        for (a, e) in ir_alg.iter().zip(rt_alg.iter()) {
            assert!((a - e).abs() < 1e-12, "{} alg: ir={a} rt={e}", ir.type_name);
        }

        let ir_dyn = eval_region(&ir.regions.dyn_, &ctx).unwrap();
        let mut rt_dyn = Vec::new();
        b.f_dyn.as_ref().unwrap()(x, u, 0.0, &mut rt_dyn);
        assert_eq!(ir_dyn.len(), rt_dyn.len(), "{} dyn arity", ir.type_name);
        for (a, e) in ir_dyn.iter().zip(rt_dyn.iter()) {
            assert!((a - e).abs() < 1e-12, "{} dyn: ir={a} rt={e}", ir.type_name);
        }
    }

    #[test]
    fn statespace_and_lti_ir_match_runtime() {
        use crate::blocks::constructors::{pt1, pt2, statespace};
        // 2x2 MIMO state space with feedthrough.
        let ss = statespace(
            vec![-1.0, 0.5, -0.3, -2.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.1, 0.0, 0.0, 0.2],
            2, 2, 2, None,
        );
        assert_dynamic_ir_matches_runtime(&ss, &[0.4, -0.2], &[1.0, -1.0]);
        // PT1 / PT2 delegate to statespace, so they inherit ops.
        assert_dynamic_ir_matches_runtime(&pt1(2.0, 0.5), &[0.7], &[1.5]);
        assert_dynamic_ir_matches_runtime(&pt2(1.0, 0.5, 0.7), &[0.3, -0.1], &[2.0]);
        // ctrl: lead_lag/pid delegate to statespace; differentiator + AWPID are custom.
        use crate::blocks::constructors::{anti_windup_pid, differentiator, lead_lag, pid};
        assert_dynamic_ir_matches_runtime(&lead_lag(2.0, 0.3, 0.1), &[0.5], &[1.0]);
        assert_dynamic_ir_matches_runtime(&pid(1.5, 0.5, 0.05, 100.0), &[0.2, -0.4], &[0.8]);
        assert_dynamic_ir_matches_runtime(&differentiator(50.0), &[0.3], &[1.2]);
        // rate_limiter / backlash are dynamic (clamp on the derivative).
        use crate::blocks::constructors::{backlash, rate_limiter};
        assert_dynamic_ir_matches_runtime(&rate_limiter(1.0, 100.0), &[0.2], &[5.0]);
        assert_dynamic_ir_matches_runtime(&rate_limiter(1.0, 100.0), &[0.2], &[0.21]);
        assert_dynamic_ir_matches_runtime(&backlash(0.5, 100.0), &[0.3], &[0.4]);
        assert_dynamic_ir_matches_runtime(&backlash(0.5, 100.0), &[0.3], &[1.0]);
        // AWPID: exercise both the unsaturated and saturated regimes.
        assert_dynamic_ir_matches_runtime(&anti_windup_pid(1.0, 0.5, 0.1, 50.0, 2.0, (-1.0, 1.0)), &[0.1, 0.2], &[0.3]);
        assert_dynamic_ir_matches_runtime(&anti_windup_pid(1.0, 0.5, 0.1, 50.0, 2.0, (-1.0, 1.0)), &[0.1, 5.0], &[3.0]);
    }

    /// L1 unification guard: every vector / matrix-vector block must lower to a
    /// fused `Dot` / `Reduce` op, never a hand-rolled scalar `Mul`/`Add` chain.
    /// This fails loudly the moment a block regresses off the shared
    /// `Builder::dot` / `Builder::reduce` path, keeping the vector tier as
    /// single-source as the scalar op vocabulary.
    #[test]
    fn vector_blocks_emit_fused_ops() {
        use crate::blocks::constructors::{adder, matrix_block, multiplier, norm_block, statespace};

        // Size inputs (shape-lazy blocks need width > 1 to fuse), pull the IR
        // block, and report whether the chosen region carries a Dot/Reduce.
        fn fused(blk: &crate::blocks::block::BlockRef, width: usize, dyn_region: bool) -> bool {
            blk.borrow_mut().inputs.resize(width);
            let b = blk.borrow();
            let ir = block_ops_to_ir_block(b, BlockId(0)).expect("block has ops");
            let region = if dyn_region { &ir.regions.dyn_ } else { &ir.regions.alg };
            region.ops.iter().any(|o| matches!(o, Op::Dot { .. } | Op::Reduce { .. }))
        }

        // StateSpace dx/dt = Ax+Bu: one Dot per row in the dyn region.
        let ss = statespace(
            vec![-1.0, 0.5, -0.3, -2.0], vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0], vec![0.0; 4], 2, 2, 2, None,
        );
        assert!(fused(&ss, 2, true), "StateSpace dyn must fuse to Dot");
        // Matrix y = M*u: one Dot per row (alg).
        assert!(
            fused(&matrix_block(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3), 3, false),
            "Matrix must fuse to Dot"
        );
        // Norm = sqrt(u . u): a single Dot.
        assert!(fused(&norm_block(), 3, false), "Norm must fuse to Dot");
        // Weighted Adder -> Dot; plain Adder -> Reduce(Sum); Multiplier -> Reduce(Product).
        assert!(fused(&adder(Some("+-+")), 3, false), "weighted Adder must fuse to Dot");
        assert!(fused(&adder(None), 3, false), "plain Adder must fuse to Reduce");
        assert!(fused(&multiplier(), 3, false), "Multiplier must fuse to Reduce");
    }

    /// ChirpSource is dynamic (state = integrated phase): check alg (reads x)
    /// and dyn (reads t) IR regions across a (t, x) grid.
    #[test]
    fn chirp_source_ir_matches_runtime() {
        use crate::blocks::constructors::chirp_source;
        let blk = chirp_source(2.0, 1.0, 3.0, 0.5, 0.3);
        let b = blk.borrow();
        let ir = block_ops_to_ir_block(b, BlockId(0)).unwrap();
        let params: Vec<f64> = ir
            .params
            .iter()
            .map(|p| match p.value {
                ParamValue::Scalar(v) => v,
                _ => 0.0,
            })
            .collect();
        for &t in &[0.0, 0.1, 0.37, 0.5, 0.9, 1.3] {
            for &x in &[0.0, 0.5, -1.2, 3.0] {
                let ctx = EvalCtx { inputs: &[&[]], state: &[x], memory: &[], params: &params, t };
                let ir_alg = eval_region(&ir.regions.alg, &ctx).unwrap();
                let mut rt_alg = Vec::new();
                b.f_alg.as_ref().unwrap()(&[x], &[], t, &mut rt_alg);
                assert!((ir_alg[0] - rt_alg[0]).abs() < 1e-12, "chirp alg @t={t},x={x}");
                let ir_dyn = eval_region(&ir.regions.dyn_, &ctx).unwrap();
                let mut rt_dyn = Vec::new();
                b.f_dyn.as_ref().unwrap()(&[x], &[], t, &mut rt_dyn);
                assert!((ir_dyn[0] - rt_dyn[0]).abs() < 1e-12, "chirp dyn @t={t},x={x}");
            }
        }
    }

    #[test]
    fn time_sources_ir_match_runtime() {
        use crate::blocks::constructors::{
            clock_source, gaussian_pulse_source, sinusoidal_source, square_wave_source,
            triangle_wave_source,
        };
        let ts = [0.0, 0.1, 0.37, 0.5, 0.9, 1.25, 2.6];
        assert_source_ir_matches_runtime(&sinusoidal_source(1.5, 2.0, 0.3), &ts);
        assert_source_ir_matches_runtime(&triangle_wave_source(1.0, 2.0, 0.0), &ts);
        assert_source_ir_matches_runtime(&square_wave_source(1.5, 2.0, 0.4), &ts);
        assert_source_ir_matches_runtime(&clock_source(0.3), &ts);
        assert_source_ir_matches_runtime(&gaussian_pulse_source(2.0, 5.0, 0.5), &ts);
    }

    /// End-to-end: a constructed block's IR (via `block_ops_to_ir_block`),
    /// evaluated by `ir::eval`, matches its live runtime `f_alg`. This is the
    /// per-block verification primitive WP9 will sweep over the whole library.
    fn assert_block_ir_matches_runtime(blk: &crate::blocks::block::BlockRef, u: &[f64]) {
        {
            let b = blk.borrow_mut();
            b.inputs.resize(u.len());
        }
        let b = blk.borrow();
        let ir = block_ops_to_ir_block(b, BlockId(0)).expect("block has ops");
        let params: Vec<f64> = ir
            .params
            .iter()
            .map(|p| match p.value {
                ParamValue::Scalar(v) => v,
                _ => 0.0,
            })
            .collect();
        let ctx = EvalCtx { inputs: &[u], state: &[], memory: &[], params: &params, t: 0.0 };
        let ir_out = eval_region(&ir.regions.alg, &ctx).unwrap();

        let mut rt = Vec::new();
        b.f_alg.as_ref().unwrap()(&[], u, 0.0, &mut rt);

        assert_eq!(ir_out.len(), rt.len(), "arity mismatch for {}", ir.type_name);
        for (a, e) in ir_out.iter().zip(rt.iter()) {
            assert!((a - e).abs() < 1e-12, "{}: ir={a} rt={e}", ir.type_name);
        }
    }

    #[test]
    fn unary_and_multiplier_ir_match_runtime() {
        use crate::blocks::constructors::{cos_block, multiplier, sin_block, sqrt_block};
        assert_block_ir_matches_runtime(&sin_block(), &[0.7]);
        assert_block_ir_matches_runtime(&cos_block(), &[1.3]);
        assert_block_ir_matches_runtime(&sqrt_block(), &[2.0]);
        assert_block_ir_matches_runtime(&multiplier(), &[2.0, 3.0, -1.5]);
    }

    #[test]
    fn param_and_logic_blocks_ir_match_runtime() {
        use crate::blocks::constructors::{
            atan2_block, clip_block, equal_block, greater_than, less_than, logic_and, logic_not,
            logic_or, matrix_block, mod_block, norm_block, polynomial, pow_block, pow_prod,
            rescale_block,
        };
        // param-unary
        assert_block_ir_matches_runtime(&pow_block(2.0), &[3.0]);
        assert_block_ir_matches_runtime(&clip_block(-1.0, 1.0), &[5.0]);
        assert_block_ir_matches_runtime(&clip_block(-1.0, 1.0), &[0.5]);
        assert_block_ir_matches_runtime(&rescale_block(2.0, 1.0, true, -3.0, 3.0), &[5.0]);
        assert_block_ir_matches_runtime(&rescale_block(2.0, 1.0, false, 0.0, 0.0), &[5.0]);
        // binary logic / comparison
        assert_block_ir_matches_runtime(&greater_than(), &[2.0, 1.0]);
        assert_block_ir_matches_runtime(&less_than(), &[2.0, 1.0]);
        assert_block_ir_matches_runtime(&equal_block(0.1), &[1.0, 1.05]);
        assert_block_ir_matches_runtime(&equal_block(0.1), &[1.0, 2.0]);
        assert_block_ir_matches_runtime(&logic_and(), &[1.0, 0.0]);
        assert_block_ir_matches_runtime(&logic_or(), &[0.0, 1.0]);
        assert_block_ir_matches_runtime(&logic_not(), &[1.0]);
        assert_block_ir_matches_runtime(&mod_block(), &[7.0, 3.0]);
        assert_block_ir_matches_runtime(&atan2_block(), &[1.0, 1.0]);
        // misc
        assert_block_ir_matches_runtime(&norm_block(), &[3.0, 4.0]);
        assert_block_ir_matches_runtime(&polynomial(vec![1.0, 2.0, 3.0]), &[2.0, -1.0]);
        assert_block_ir_matches_runtime(&pow_prod(vec![2.0, 3.0]), &[2.0, 3.0]);
        assert_block_ir_matches_runtime(&matrix_block(vec![1.0, 2.0, 3.0, 4.0], 2, 2), &[1.5, -2.0]);
        use crate::blocks::constructors::{deadband, lut1d, mod_single, ExtrapMode};
        assert_block_ir_matches_runtime(&deadband(-1.0, 1.0), &[2.5]);
        assert_block_ir_matches_runtime(&deadband(-1.0, 1.0), &[0.3]);
        // mod_single == rem_euclid, incl. negative input.
        for x in [0.3, 1.7, 2.0, -0.4, -3.2] {
            assert_block_ir_matches_runtime(&mod_single(1.5), &[x]);
        }
        // lut1d: interior, left-extrap/clamp, right-extrap/clamp.
        let pts = vec![0.0, 1.0, 2.0, 4.0];
        let vals = vec![0.0, 10.0, 30.0, 20.0];
        for x in [-0.5, 0.0, 0.5, 1.0, 1.5, 3.0, 4.0, 5.5] {
            assert_block_ir_matches_runtime(&lut1d(pts.clone(), vals.clone(), ExtrapMode::Extrapolate).unwrap(), &[x]);
            assert_block_ir_matches_runtime(&lut1d(pts.clone(), vals.clone(), ExtrapMode::Clamp).unwrap(), &[x]);
        }
        // divider: num/den over an ops string; Warn (nonzero den) and Clamp.
        use crate::blocks::constructors::{divider, ZeroDiv};
        assert_block_ir_matches_runtime(&divider(Some("*/"), ZeroDiv::Warn).unwrap(), &[6.0, 3.0]);
        assert_block_ir_matches_runtime(&divider(Some("**/"), ZeroDiv::Warn).unwrap(), &[6.0, 4.0, 3.0]);
        assert_block_ir_matches_runtime(&divider(Some("*/"), ZeroDiv::Clamp).unwrap(), &[1.0, 0.0]);
        assert_block_ir_matches_runtime(&divider(None, ZeroDiv::Warn).unwrap(), &[2.0, 3.0, 4.0]);
    }

    /// The op-expressible block catalog: every block that carries a static
    /// op-graph (`ops.is_some()`). Shared by the completeness gate and the
    /// feedthrough-parity guard so both cover the same full set.
    fn op_expressible_blocks() -> Vec<crate::blocks::block::BlockRef> {
        use crate::blocks::constructors::*;
        vec![
            integrator(0.0), integrator_vec(&[0.0, 0.0]), amplifier(2.0), adder(Some("+-")),
            constant(1.0), multiplier(),
            sin_block(), cos_block(), tan_block(), exp_block(), log_block(), log10_block(),
            sqrt_block(), abs_block(), sinh_block(), cosh_block(), tanh_block(), atan_block(),
            alias_block(),
            pow_block(2.0), clip_block(-1.0, 1.0), rescale_block(1.0, 0.0, true, -1.0, 1.0),
            greater_than(), less_than(), equal_block(0.1), logic_and(), logic_or(), logic_not(),
            atan2_block(), mod_block(),
            norm_block(), polynomial(vec![1.0, 2.0]), pow_prod(vec![2.0, 3.0]),
            matrix_block(vec![1.0, 0.0, 0.0, 1.0], 2, 2), deadband(-1.0, 1.0),
            sinusoidal_source(1.0, 1.0, 0.0), triangle_wave_source(1.0, 1.0, 0.0),
            square_wave_source(1.0, 1.0, 0.0), clock_source(0.5), gaussian_pulse_source(1.0, 1.0, 0.5),
            statespace(vec![-1.0], vec![1.0], vec![1.0], vec![0.0], 1, 1, 1, None),
            pt1(1.0, 1.0), pt2(1.0, 1.0, 0.5), lead_lag(1.0, 0.5, 0.3), pid(1.0, 0.5, 0.1, 100.0),
            differentiator(50.0), anti_windup_pid(1.0, 0.5, 0.1, 50.0, 2.0, (-1.0, 1.0)),
            chirp_source(1.0, 1.0, 5.0, 1.0, 0.0),
            transfer_function_num_den(&[1.0], &[1.0, 1.0]),
            butter_lowpass(5.0, 2).unwrap(), butter_highpass(5.0, 2).unwrap(),
            butter_bandpass(1.0, 5.0, 2).unwrap(), butter_bandstop(1.0, 5.0, 2).unwrap(), allpass_filter(5.0, 2).unwrap(),
            discrete_integrator(0.1, 0.0, vec![0.0]),
            sample_hold(0.1, 0.0),
            discrete_derivative(0.1, 0.0),
            adc(4, -1.0, 1.0, 0.1, 0.0), dac(4, -1.0, 1.0, 0.1, 0.0),
            tapped_delay(3, 0.1, 0.0).unwrap(),
            discrete_state_space(vec![0.5], vec![1.0], vec![2.0], vec![0.0], 1, 1, 1, 0.1, 0.0, None).unwrap(),
            discrete_transfer_function(&[1.0], &[1.0, 0.5], 0.1, 0.0).unwrap(),
            fir(vec![0.5, 0.25], 0.1, 0.0),
            relay(1.0, -1.0, 10.0, -10.0), comparator(0.0, (-1.0, 1.0)),
            switch(3, Some(1)),
            rate_limiter(1.0, 100.0), backlash(0.5, 100.0),
            counter(0.0, 10.0), counter_up(0.0, 10.0), counter_down(0.0, 10.0),
            step_source(vec![1.0, 2.0, 3.0], vec![0.0, 1.0, 2.0]).unwrap(),
            delay(0.1, 0.01),
            first_order_hold(0.1, 0.0),
            pulse_source(2.0, 1.0, 0.1, 0.2, 0.0, 0.5),
            mod_single(2.0),
            lut1d(vec![0.0, 1.0, 2.0], vec![0.0, 10.0, 30.0], ExtrapMode::Extrapolate).unwrap(),
            lut1d(vec![0.0, 1.0, 2.0], vec![0.0, 10.0, 30.0], ExtrapMode::Clamp).unwrap(),
            divider(Some("*/"), ZeroDiv::Warn).unwrap(),
            divider(Some("*/"), ZeroDiv::Clamp).unwrap(),
        ]
    }

    #[test]
    fn completeness_gate_all_blocks_classified() {
        use crate::blocks::constructors::*;
        let has_ops = |b: &crate::blocks::block::BlockRef| {
            let blk = b.borrow();
            blk.alg_op.is_some() || blk.op_discrete_builder.is_some()
        };

        // --- op-expressible blocks: MUST carry ops ---
        let with_ops = op_expressible_blocks();
        for b in &with_ops {
            assert!(has_ops(b), "block '{}' should carry ops", b.borrow().type_name);
        }

        // --- intentionally opaque (no static op-graph): MUST NOT carry ops ---
        // Arbitrary-callable factories / Python-traced (group 1) / non-op math:
        let opaque: Vec<crate::blocks::block::BlockRef> = vec![
            math_block("M", f64::sin), logic_block("L", |a, b| a + b),
            source(|t| t), function(|u| u.to_vec()),
            dynamical_function(|u, _t| u.to_vec()),
            dynamical_system(|x, _u, _t| x.to_vec(), |x, _u, _t| x.to_vec(), &[0.0], false, None),
            ode(|x, _u, _t| vec![-x[0]], &[1.0], None),
            // Sinks / RNG / spectral:
            scope(None, 0.0, vec![]), spectrum(vec![1.0], 0.0, 0.5, 1, vec![]),
            white_noise(1.0, None, None, None), pink_noise(1.0, None, 3, None, None),
            random_number_generator(None, None),
            // Arbitrary-callable discrete (opaque function):
            wrapper(|u| u.to_vec(), 0.1, 0.0),
        ];
        for b in &opaque {
            assert!(!has_ops(b), "block '{}' is classified opaque but carries ops", b.borrow().type_name);
        }
        // Not constructed here (exotic args, but covered by mechanism/classification):
        // transfer_function / _zpg / _prc / _prc_mimo (Complex64 -> statespace, ops via delegation),
        // mass_matrix_dae / semi_explicit_dae / fully_implicit_dae (group-1 traced, opaque).
        // chirp/sinusoidal_phase_noise carry NOMINAL ops (deterministic skeleton;
        // the RNG phase noise is a runtime-only input, taken at 0 for compile/codegen).
    }

    /// WP6: a discrete block's Memory + Event IR. DiscreteIntegrator has two
    /// memory slots (state, held); each period the event sets held' = state and
    /// state' = state + period*u; the alg region outputs the held value.
    #[test]
    fn discrete_integrator_memory_event_ir() {
        use crate::blocks::constructors::discrete_integrator;
        let blk = discrete_integrator(0.1, 0.0, vec![1.0, 2.0]);
        {
            let b = blk.borrow_mut();
            b.inputs.resize(2);
        }
        let b = blk.borrow();
        let ir = block_ops_to_ir_block(b, BlockId(0)).unwrap();
        assert_eq!(ir.memory.len(), 2, "state + held slots");
        assert_eq!(ir.events.len(), 1);
        assert!(matches!(
            ir.events[0].kind,
            EventKind::Schedule { times: ScheduleTimes::Periodic { .. } }
        ));

        let state = [5.0, 6.0];
        let held = [3.0, 4.0];
        let u = [10.0, 20.0];
        let ctx = EvalCtx { inputs: &[&u], state: &[], memory: &[&state, &held], params: &[], t: 0.0 };
        // alg outputs the held slot.
        assert_eq!(eval_region(&ir.regions.alg, &ctx).unwrap(), held.to_vec());
        // event effect: [held'=state(2), state'=state+0.1*u(2)].
        let eff = eval_region(&ir.events[0].effect, &ctx).unwrap();
        assert_eq!(&eff[0..2], &[5.0, 6.0]);
        assert!((eff[2] - (5.0 + 0.1 * 10.0)).abs() < 1e-12);
        assert!((eff[3] - (6.0 + 0.1 * 20.0)).abs() < 1e-12);
    }

    /// WP6 shape-poly discrete: SampleHold resolved at width 2 has one `held`
    /// memory slot (size 2); each period held' = u; alg outputs held.
    #[test]
    fn sample_hold_memory_event_ir() {
        use crate::blocks::constructors::sample_hold;
        let blk = sample_hold(0.2, 0.05);
        {
            let b = blk.borrow_mut();
            b.inputs.resize(2);
        }
        let b = blk.borrow();
        let ir = block_ops_to_ir_block(b, BlockId(0)).unwrap();
        assert_eq!(ir.memory.len(), 1);
        assert_eq!(ir.memory[0].size, 2);
        assert_eq!(ir.events.len(), 1);

        let held = [7.0, 8.0];
        let u = [1.5, -2.5];
        let ctx = EvalCtx { inputs: &[&u], state: &[], memory: &[&held], params: &[], t: 0.0 };
        // alg outputs the held slot.
        assert_eq!(eval_region(&ir.regions.alg, &ctx).unwrap(), held.to_vec());
        // event effect: held' = u.
        assert_eq!(eval_region(&ir.events[0].effect, &ctx).unwrap(), u.to_vec());
    }

    /// WP6 quantizers: ADC splits a sample into bits, DAC reassembles bits into
    /// an analog value. Verifies the event effects against hand-computed codes.
    #[test]
    fn adc_dac_memory_event_ir() {
        use crate::blocks::constructors::{adc, dac};
        // ADC: 4 bits over [-1, 1]. u=0.3 -> scaled 0.65 -> code floor(10.4)=10 = 1010b.
        let a = adc(4, -1.0, 1.0, 0.1, 0.0);
        let ab = a.borrow();
        let air = block_ops_to_ir_block(ab, BlockId(0)).unwrap();
        let ctx = EvalCtx { inputs: &[&[0.3]], state: &[], memory: &[], params: &[], t: 0.0 };
        let bits = eval_region(&air.events[0].effect, &ctx).unwrap();
        assert_eq!(bits, vec![0.0, 1.0, 0.0, 1.0]); // 10 = 1010 (LSB first)

        // DAC: bits 1010b (code 10) -> -1 + 2*10/15.
        let d = dac(4, -1.0, 1.0, 0.1, 0.0);
        let db = d.borrow();
        let dir = block_ops_to_ir_block(db, BlockId(0)).unwrap();
        let ctx2 = EvalCtx { inputs: &[&[0.0, 1.0, 0.0, 1.0]], state: &[], memory: &[], params: &[], t: 0.0 };
        let out = eval_region(&dir.events[0].effect, &ctx2).unwrap();
        assert!((out[0] - (-1.0 + 2.0 * 10.0 / 15.0)).abs() < 1e-12);
    }

    /// WP6 discrete state space: held_out' = C*state + D*u, state' = A*state + B*u.
    #[test]
    fn discrete_state_space_memory_event_ir() {
        use crate::blocks::constructors::discrete_state_space;
        // 1-state: A=0.5, B=1, C=2, D=0.
        let blk = discrete_state_space(vec![0.5], vec![1.0], vec![2.0], vec![0.0], 1, 1, 1, 0.1, 0.0, None).unwrap();
        let b = blk.borrow();
        let ir = block_ops_to_ir_block(b, BlockId(0)).unwrap();
        assert_eq!(ir.memory.len(), 2);
        let state = [3.0];
        let u = [4.0];
        let ctx = EvalCtx { inputs: &[&u], state: &[], memory: &[&state, &[0.0]], params: &[], t: 0.0 };
        let eff = eval_region(&ir.events[0].effect, &ctx).unwrap();
        // [held = C*x + D*u = 6, state' = A*x + B*u = 5.5]
        assert!((eff[0] - 6.0).abs() < 1e-12);
        assert!((eff[1] - 5.5).abs() < 1e-12);
    }

    /// WP6 FIR: held[i] = sum_k coeffs[k]*newbuf[k][i]; buffer shifts in u.
    #[test]
    fn fir_memory_event_ir() {
        use crate::blocks::constructors::fir;
        // coeffs [0.5, 0.25], SISO. old buffer [2, 3], u = 4.
        let blk = fir(vec![0.5, 0.25], 0.1, 0.0);
        {
            let b = blk.borrow_mut();
            b.inputs.resize(1);
        }
        let b = blk.borrow();
        let ir = block_ops_to_ir_block(b, BlockId(0)).unwrap();
        let buffer = [2.0, 3.0];
        let u = [4.0];
        let ctx = EvalCtx { inputs: &[&u], state: &[], memory: &[&buffer, &[0.0]], params: &[], t: 0.0 };
        let eff = eval_region(&ir.events[0].effect, &ctx).unwrap();
        // held = 0.5*4 + 0.25*2 = 2.5; new buffer = [4, 2].
        assert!((eff[0] - 2.5).abs() < 1e-12);
        assert!((eff[1] - 4.0).abs() < 1e-12);
        assert!((eff[2] - 2.0).abs() < 1e-12);
    }

    /// WP6 zero-crossing hysteresis: Relay holds an output toggled by two
    /// zero-crossing events; Comparator is a pure algebraic select.
    #[test]
    fn relay_and_comparator_ir() {
        use crate::blocks::constructors::{comparator, relay};
        // Relay: 2 ZeroCross events, effects = value_up / value_down.
        let r = relay(1.0, -1.0, 10.0, -10.0);
        let rb = r.borrow();
        let rir = block_ops_to_ir_block(rb, BlockId(0)).unwrap();
        assert_eq!(rir.events.len(), 2);
        assert!(matches!(rir.events[0].kind, EventKind::ZeroCross { direction: Direction::Rising, .. }));
        assert!(matches!(rir.events[1].kind, EventKind::ZeroCross { direction: Direction::Falling, .. }));
        let empty: EvalCtx = EvalCtx { inputs: &[], state: &[], memory: &[], params: &[], t: 0.0 };
        assert_eq!(eval_region(&rir.events[0].effect, &empty).unwrap(), vec![10.0]);
        assert_eq!(eval_region(&rir.events[1].effect, &empty).unwrap(), vec![-10.0]);
        // alg reads the held `out` slot.
        let out = [3.0];
        let ctx = EvalCtx { inputs: &[], state: &[], memory: &[&out], params: &[], t: 0.0 };
        assert_eq!(eval_region(&rir.regions.alg, &ctx).unwrap(), vec![3.0]);

        // Comparator: algebraic select(u >= thr, hi, lo).
        let c = comparator(0.0, (-1.0, 1.0));
        let cb = c.borrow();
        let cir = block_ops_to_ir_block(cb, BlockId(0)).unwrap();
        let params = [0.0, 1.0, -1.0];
        let hi_ctx = EvalCtx { inputs: &[&[0.5]], state: &[], memory: &[], params: &params, t: 0.0 };
        assert_eq!(eval_region(&cir.regions.alg, &hi_ctx).unwrap(), vec![1.0]);
        let lo_ctx = EvalCtx { inputs: &[&[-0.5]], state: &[], memory: &[], params: &params, t: 0.0 };
        assert_eq!(eval_region(&cir.regions.alg, &lo_ctx).unwrap(), vec![-1.0]);
    }

    /// WP6 StepSource: each scheduled time advances out -> amplitudes[cnt], cnt++.
    #[test]
    fn step_source_memory_event_ir() {
        use crate::blocks::constructors::step_source;
        let blk = step_source(vec![1.0, 2.0, 3.0], vec![0.0, 1.0, 2.0]).unwrap();
        let b = blk.borrow();
        let ir = block_ops_to_ir_block(b, BlockId(0)).unwrap();
        assert_eq!(ir.memory.len(), 2);
        assert!(matches!(ir.events[0].kind, EventKind::Schedule { times: ScheduleTimes::Fixed(_) }));
        // cnt=0, out=0 -> out'=amps[0]=1, cnt'=1.
        let ctx = EvalCtx { inputs: &[], state: &[], memory: &[&[0.0], &[0.0]], params: &[], t: 0.0 };
        assert_eq!(eval_region(&ir.events[0].effect, &ctx).unwrap(), vec![1.0, 1.0]);
        // cnt=2, out=2 -> out'=amps[2]=3, cnt'=3.
        let ctx2 = EvalCtx { inputs: &[], state: &[], memory: &[&[2.0], &[2.0]], params: &[], t: 0.0 };
        assert_eq!(eval_region(&ir.events[0].effect, &ctx2).unwrap(), vec![3.0, 3.0]);
    }

    /// WP6 FirstOrderHold: time-interpolating alg (linear extrapolation once two
    /// samples seen) + event that shifts the sample pair.
    #[test]
    fn first_order_hold_memory_event_ir() {
        use crate::blocks::constructors::first_order_hold;
        let blk = first_order_hold(0.1, 0.0);
        {
            let b = blk.borrow_mut();
            b.inputs.resize(1);
        }
        let b = blk.borrow();
        let ir = block_ops_to_ir_block(b, BlockId(0)).unwrap();
        assert_eq!(ir.memory.len(), 4);
        // n>=2: y = u_curr + (u_curr-u_prev)/T*(t-t_curr) = 3 + 20*0.2 = 7.
        let mem: &[&[f64]] = &[&[1.0], &[3.0], &[0.5], &[2.0]];
        let ctx = EvalCtx { inputs: &[], state: &[], memory: mem, params: &[], t: 0.7 };
        assert!((eval_region(&ir.regions.alg, &ctx).unwrap()[0] - 7.0).abs() < 1e-12);
        // n<2: held at u_curr = 3.
        let mem1: &[&[f64]] = &[&[1.0], &[3.0], &[0.5], &[1.0]];
        let ctx1 = EvalCtx { inputs: &[], state: &[], memory: mem1, params: &[], t: 0.7 };
        assert!((eval_region(&ir.regions.alg, &ctx1).unwrap()[0] - 3.0).abs() < 1e-12);
        // event: [u_prev'=u_curr=3, u_curr'=u=5, t_curr'=t=0.9, n'=n+1=3].
        // memory by slot: [0]=u_prev, [1]=u_curr=3, [2]=t_curr, [3]=n=2.
        let mem2: &[&[f64]] = &[&[0.0], &[3.0], &[0.0], &[2.0]];
        let ctx2 = EvalCtx { inputs: &[&[5.0]], state: &[], memory: mem2, params: &[], t: 0.9 };
        let eff = eval_region(&ir.events[0].effect, &ctx2).unwrap();
        assert_eq!(eff, vec![3.0, 5.0, 0.9, 3.0]);
    }

    /// WP6 PulseSource: phase-indexed trapezoid interpolated over time.
    #[test]
    fn pulse_source_memory_event_ir() {
        use crate::blocks::constructors::pulse_source;
        let blk = pulse_source(2.0, 1.0, 0.1, 0.2, 0.0, 0.5);
        let b = blk.borrow();
        let ir = block_ops_to_ir_block(b, BlockId(0)).unwrap();
        assert_eq!(ir.memory.len(), 2);
        assert_eq!(ir.events.len(), 4);
        // rising (phase 1), phase_start 0, t=0.05, t_rise=0.1 -> 2*min(0.5,1)=1.
        let rise = EvalCtx { inputs: &[], state: &[], memory: &[&[1.0], &[0.0]], params: &[], t: 0.05 };
        assert!((eval_region(&ir.regions.alg, &rise).unwrap()[0] - 1.0).abs() < 1e-12);
        // high (phase 2) -> amplitude 2.
        let high = EvalCtx { inputs: &[], state: &[], memory: &[&[2.0], &[0.0]], params: &[], t: 0.5 };
        assert!((eval_region(&ir.regions.alg, &high).unwrap()[0] - 2.0).abs() < 1e-12);
        // low (phase 0) -> 0.
        let low = EvalCtx { inputs: &[], state: &[], memory: &[&[0.0], &[0.0]], params: &[], t: 0.5 };
        assert!((eval_region(&ir.regions.alg, &low).unwrap()[0]).abs() < 1e-12);
    }

    /// WP8: a nested subsystem (y = 2*u) becomes a `Child::Subsystem` whose
    /// internal connections reference `BlockId::INTERFACE`, with the wrapper's
    /// I/O as its interface. The whole module JSON-roundtrips.
    #[test]
    fn nested_subsystem_ir() {
        use crate::blocks::constructors::{amplifier, scope, sinusoidal_source};
        use crate::connection::Connection as RtConn;
        use crate::simulation::Simulation;
        use crate::subsystem::{interface, subsystem};
        use crate::utils::portreference::PortReference;
        use std::rc::Rc;

        let conn = |a: &crate::blocks::block::BlockRef, b: &crate::blocks::block::BlockRef| {
            Rc::new(RtConn::new(
                PortReference::new(a.clone(), None),
                vec![PortReference::new(b.clone(), None)],
            ))
        };
        // Inner: interface -> amp(2) -> interface.
        let iface = interface();
        let amp = amplifier(2.0);
        let sub = subsystem(
            vec![iface.clone(), amp.clone()],
            vec![conn(&iface, &amp), conn(&amp, &iface)],
            10,
        )
        .unwrap();
        // Outer: src -> sub -> scope.
        let src = sinusoidal_source(1.0, 1.0, 0.0);
        let sco = scope(None, 0.0, vec![]);
        let mut sim = Simulation::with_defaults(
            vec![src.clone(), sub.clone(), sco.clone()],
            vec![conn(&src, &sub), conn(&sub, &sco)],
        );
        sim.run(0.03, false, false);

        let m = module_from_sim(&sim, "nested");
        assert_eq!(m.root.children.len(), 3);
        let subc = m.root.children.iter().find_map(|c| match c {
            Child::Subsystem(s) => Some(s),
            _ => None,
        });
        let s = subc.expect("subsystem child present");
        // inner amplifier is a child; interface excluded.
        assert!(s.children.iter().any(|c| matches!(c, Child::Block(b) if b.type_name == "Amplifier")));
        // inner connections reference the interface sentinel.
        let refs_iface = s.connections.iter().any(|c| {
            c.src.block == BlockId::INTERFACE || c.targets.iter().any(|t| t.block == BlockId::INTERFACE)
        });
        assert!(refs_iface, "inner connections should reference INTERFACE");
        // outer-facing interface mirrors the wrapper's I/O.
        assert!(!s.interface.inputs.is_empty() || !s.interface.outputs.is_empty());

        let json = serde_json::to_string_pretty(&m).unwrap();
        let m2: Module = serde_json::from_str(&json).unwrap();
        assert_eq!(json, serde_json::to_string_pretty(&m2).unwrap(), "nested module roundtrip");
    }

    /// WP8: MIMO channel slicing -> PortRef.elems.
    #[test]
    fn mimo_elems_mapping() {
        use crate::blocks::constructors::matrix_block;
        use crate::utils::portreference::{Port, PortReference};
        let blk = matrix_block(vec![1.0, 0.0, 0.0, 1.0], 2, 2); // 2 outputs
        // default (whole) -> None
        let pr_all = PortReference::new(blk.clone(), None);
        assert_eq!(elems_of(pr_all._get_output_indices()), None);
        // explicit channel 1 -> Some([1])
        let pr_one = PortReference::new(blk.clone(), Some(vec![Port::Index(1)]));
        assert_eq!(elems_of(pr_one._get_output_indices()), Some(vec![1]));
    }

    /// Build a Module from an assembled control-loop sim and JSON-roundtrip it.
    #[test]
    fn module_from_sim_roundtrips() {
        use crate::blocks::constructors::{adder, amplifier, integrator, scope, sinusoidal_source};
        use crate::connection::Connection as RtConn;
        use crate::simulation::Simulation;
        use crate::utils::portreference::PortReference;
        use std::rc::Rc;

        let src = sinusoidal_source(1.0, 1.0, 0.0);
        let err = adder(Some("+-"));
        let kp = amplifier(4.0);
        let plant = integrator(0.0);
        let sco = scope(None, 0.0, vec![]);
        let conn = |a: &crate::blocks::block::BlockRef, b: &crate::blocks::block::BlockRef| {
            Rc::new(RtConn::new(
                PortReference::new(a.clone(), None),
                vec![PortReference::new(b.clone(), None)],
            ))
        };
        let conns = vec![
            conn(&src, &err), conn(&err, &kp), conn(&kp, &plant),
            conn(&plant, &err), conn(&plant, &sco),
        ];
        let mut sim = Simulation::with_defaults(
            vec![src, err, kp, plant, sco],
            conns,
        );
        sim.run(0.05, false, false);

        let m = module_from_sim(&sim, "control_loop");
        assert_eq!(m.root.children.len(), 5);
        assert_eq!(m.root.connections.len(), 5);
        // The scope is opaque -> represented as one extern Call decl.
        assert_eq!(m.extern_decls.len(), 1, "scope should be one extern");
        assert_eq!(m.extern_decls[0].name, "Scope");
        // Every non-sink child produces its outputs (ops or extern Call).
        for child in &m.root.children {
            if let Child::Block(b) = child {
                if !b.ports.outputs.is_empty() {
                    assert!(
                        !b.regions.alg.writes.is_empty(),
                        "block '{}' has outputs but no producing region",
                        b.type_name
                    );
                }
            }
        }

        let json = serde_json::to_string_pretty(&m).unwrap();
        let m2: Module = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string_pretty(&m2).unwrap();
        assert_eq!(json, json2, "module JSON roundtrip not stable");
    }

    /// Schedule on a pure feedforward chain: src -> amp -> integrator -> scope.
    /// No algebraic loops; `groups` lay out algebraic-feedthrough depth (the amp
    /// resolves one depth after the source/integrator), `topo` covers everyone.
    #[test]
    fn schedule_feedforward_groups() {
        use crate::blocks::constructors::{amplifier, integrator, scope, sinusoidal_source};
        use crate::connection::Connection as RtConn;
        use crate::simulation::Simulation;
        use crate::utils::portreference::PortReference;
        use std::rc::Rc;

        let conn = |a: &crate::blocks::block::BlockRef, b: &crate::blocks::block::BlockRef| {
            Rc::new(RtConn::new(
                PortReference::new(a.clone(), None),
                vec![PortReference::new(b.clone(), None)],
            ))
        };
        let src = sinusoidal_source(1.0, 1.0, 0.0);
        let amp = amplifier(2.0);
        let plant = integrator(0.0);
        let sco = scope(None, 0.0, vec![]);
        let mut sim = Simulation::with_defaults(
            vec![src.clone(), amp.clone(), plant.clone(), sco.clone()],
            vec![conn(&src, &amp), conn(&amp, &plant), conn(&plant, &sco)],
        );
        sim.run(0.03, false, false);

        let m = module_from_sim(&sim, "ff");
        let sched = &m.root.schedule;
        assert!(sched.sccs.is_empty(), "feedforward has no algebraic loops");
        assert!(sched.back_edges.is_empty());
        // topo covers all four children exactly once.
        assert_eq!(sched.topo.len(), 4);
        // amplifier (BlockId 1) is the only algebraic-feedthrough block -> it sits
        // strictly deeper than the source it reads from.
        let depth_of = |bid: BlockId| {
            sched.groups.iter().find(|g| g.blocks.contains(&bid)).map(|g| g.depth)
        };
        let amp_depth = depth_of(BlockId(1)).expect("amp scheduled");
        let src_depth = depth_of(BlockId(0)).expect("src scheduled");
        assert!(amp_depth > src_depth, "amp resolves after its source");
    }

    /// Schedule on a *purely algebraic* loop: src -> err(+ -) -> kp -> err. `err`
    /// and `kp` form one SCC; the feedback connection is reported as a back-edge.
    /// (Contrast with `module_from_sim_roundtrips`, where the feedback passes
    /// through an integrator and is therefore not an algebraic loop.)
    #[test]
    fn schedule_algebraic_loop_scc() {
        use crate::blocks::constructors::{adder, amplifier, scope, sinusoidal_source};
        use crate::connection::Connection as RtConn;
        use crate::simulation::Simulation;
        use crate::utils::portreference::PortReference;
        use std::rc::Rc;

        let conn = |a: &crate::blocks::block::BlockRef, b: &crate::blocks::block::BlockRef| {
            Rc::new(RtConn::new(
                PortReference::new(a.clone(), None),
                vec![PortReference::new(b.clone(), None)],
            ))
        };
        let src = sinusoidal_source(1.0, 1.0, 0.0); // BlockId 0
        let err = adder(Some("+-")); // BlockId 1
        let kp = amplifier(0.5); // BlockId 2 (gain < 1 -> contractive loop)
        let sco = scope(None, 0.0, vec![]); // BlockId 3
        let conns = vec![
            conn(&src, &err),
            conn(&err, &kp),
            conn(&kp, &err), // algebraic feedback closes the loop
            conn(&kp, &sco),
        ];
        let mut sim = Simulation::with_defaults(
            vec![src.clone(), err.clone(), kp.clone(), sco.clone()],
            conns,
        );
        sim.run(0.03, false, false);

        let m = module_from_sim(&sim, "alg_loop");
        let sched = &m.root.schedule;
        assert_eq!(sched.sccs.len(), 1, "one algebraic loop");
        let scc = &sched.sccs[0];
        let mut members = scc.blocks.clone();
        members.sort_by_key(|b| b.0);
        assert_eq!(members, vec![BlockId(1), BlockId(2)], "loop = {{err, kp}}");
        assert!(!scc.back_edges.is_empty(), "SCC must report a back-edge cut");
        // the global back-edge list mirrors the SCC's cut (deduped union).
        assert!(!sched.back_edges.is_empty());
        for be in &scc.back_edges {
            assert!(sched.back_edges.contains(be), "SCC cut is in the global set");
        }
        // every back-edge id is a real connection in this scope.
        let conn_ids: std::collections::HashSet<u32> =
            m.root.connections.iter().map(|c| c.id.0).collect();
        for be in &sched.back_edges {
            assert!(conn_ids.contains(&be.0), "back-edge references a real connection");
        }
        // topo still covers all four children.
        assert_eq!(sched.topo.len(), 4);
    }

    /// Opaque block events are made visible: a `scope` with a sampling period
    /// carries a runtime Schedule event. The IR represents the (otherwise
    /// opaque) scope with that event surfaced as an opaque Schedule.
    #[test]
    fn opaque_block_event_surfaced() {
        use crate::blocks::constructors::{scope, sinusoidal_source};
        use crate::connection::Connection as RtConn;
        use crate::simulation::Simulation;
        use crate::utils::portreference::PortReference;
        use std::rc::Rc;

        let conn = |a: &crate::blocks::block::BlockRef, b: &crate::blocks::block::BlockRef| {
            Rc::new(RtConn::new(
                PortReference::new(a.clone(), None),
                vec![PortReference::new(b.clone(), None)],
            ))
        };
        let src = sinusoidal_source(1.0, 1.0, 0.0);
        let sco = scope(Some(0.25), 0.1, vec![]); // sampling period -> Schedule event
        let mut sim = Simulation::with_defaults(vec![src.clone(), sco.clone()], vec![conn(&src, &sco)]);
        sim.run(0.5, false, false);

        let m = module_from_sim(&sim, "opaque_evt");
        let sco_b = m
            .root
            .children
            .iter()
            .find_map(|c| match c {
                Child::Block(b) if b.type_name == "Scope" => Some(b),
                _ => None,
            })
            .expect("scope present");
        // scope is opaque -> registered as an extern.
        assert!(m.extern_decls.iter().any(|d| d.name == "Scope"), "scope is opaque/extern");
        assert_eq!(sco_b.events.len(), 1, "sampling event surfaced");
        let e = &sco_b.events[0];
        assert!(e.opaque, "opaque-block event is flagged opaque");
        assert!(e.effect.is_empty(), "opaque event has no op effect");
        match &e.kind {
            EventKind::Schedule { times: ScheduleTimes::Periodic { period, phase } } => {
                assert!((period - 0.25).abs() < 1e-12);
                assert!((phase - 0.1).abs() < 1e-12);
            }
            other => panic!("expected periodic schedule, got {other:?}"),
        }
        // module roundtrips with the opaque event present.
        let json = serde_json::to_string_pretty(&m).unwrap();
        let m2: Module = serde_json::from_str(&json).unwrap();
        assert_eq!(json, serde_json::to_string_pretty(&m2).unwrap());
    }

    /// Simulation-level (global) events land in `Module.events` as opaque
    /// events carrying only their statically-known kind/timing.
    #[test]
    fn global_simulation_events_in_module() {
        use crate::blocks::constructors::{integrator, scope};
        use crate::connection::Connection as RtConn;
        use crate::events::schedule::ScheduleList;
        use crate::simulation::Simulation;
        use crate::utils::fastcell::FastCell;
        use crate::utils::portreference::PortReference;
        use std::rc::Rc;

        let conn = |a: &crate::blocks::block::BlockRef, b: &crate::blocks::block::BlockRef| {
            Rc::new(RtConn::new(
                PortReference::new(a.clone(), None),
                vec![PortReference::new(b.clone(), None)],
            ))
        };
        let plant = integrator(0.0);
        let sco = scope(None, 0.0, vec![]);
        let mut sim = Simulation::with_defaults(vec![plant.clone(), sco.clone()], vec![conn(&plant, &sco)]);
        // A standalone global Schedule firing at fixed times.
        let evt = ScheduleList::from_times(vec![0.1, 0.2, 0.4]);
        sim.add_event(Rc::new(FastCell::new(evt)));

        let m = module_from_sim(&sim, "global_evt");
        assert_eq!(m.events.len(), 1, "global event recorded on the module");
        let e = &m.events[0];
        assert!(e.opaque);
        match &e.kind {
            EventKind::Schedule { times: ScheduleTimes::Fixed(ts) } => {
                assert_eq!(ts, &vec![0.1, 0.2, 0.4]);
            }
            other => panic!("expected fixed schedule, got {other:?}"),
        }
    }

    #[test]
    fn sinusoidal_ir_matches_native() {
        let (freq, amp, phase) = (1.5, 2.0, 0.3);
        let g = sinusoidal_source_graph(freq, amp, phase);
        let dec = GraphDecode { input_port_sizes: &[], output_port_sizes: &[1] };
        let region = region_from_graph(&g, RegionRole::Alg, &dec);

        let t = 0.42;
        let mut native = Vec::new();
        sinusoidal_eval(&F64Builder, freq, amp, phase, t, &mut native);
        let ctx = EvalCtx {
            inputs: &[],
            state: &[],
            memory: &[],
            params: &[freq, amp, phase],
            t,
        };
        let ir = eval_region(&region, &ctx).unwrap();
        assert!((ir[0] - native[0]).abs() < 1e-15, "ir={} native={}", ir[0], native[0]);
    }
}
