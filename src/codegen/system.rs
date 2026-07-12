//! Whole-system emission: an IR `Module` to a complete, runnable C model
//! (state vector, per-block functions, `model_deriv`, a tableau-driven integrator
//! loop, `model_init` / `model_run`). Everything is derived from the IR alone
//! (connections + schedule), never from `crate::compile`.
//!
//! The IR is first *flattened* ([`flatten`]) into a list of leaf blocks and
//! connections over flat indices — this is where nested subsystems are inlined
//! and their interface connections spliced. [`build_plan`] then resolves the
//! signal/state layout and element-level wiring, and computes the algebraic
//! evaluation order from the dependency graph (so it is correct after
//! flattening, independent of the IR's advisory schedule).
//!
//! `Structure::Hierarchical` emits one named function per block (`alg_i`,
//! `deriv_i`) wired through a `sig[]` buffer, with a provenance comment per
//! block. `Structure::Flat` (one fused `dx/dt`) builds on the same plan and is
//! not yet emitted.
//!
//! Current scope: single-element-per-input-port blocks emit any element count
//! (MIMO within one port); multiple *input ports* and opaque externs surface as
//! `Unsupported`.

use crate::ir::schema::{
    BinOpKind, Block, BlockId, BlockRole, Child, Direction, EventKind, Module, NodeId, Op,
    ParamValue, ReduceKind, Region, ScheduleTimes, Subsystem, UnaryOpKind, Write,
};

use std::collections::HashSet;

use serde::Serialize;

use super::{
    fmt_lit, lower_one, render, solver, CTarget, CodegenError, CodegenOptions, GeneratedFile, Layout,
    Leaves, Numeric, Reductions, Structure, Target,
};

type R<T> = Result<T, CodegenError>;

fn unsupported(s: impl Into<String>) -> CodegenError {
    CodegenError::Unsupported(s.into())
}

/// A connection over flattened leaf-block indices: no interface, no subsystems.
struct FlatConn {
    /// `(flat block index, output port, optional channel elems)`.
    src: (usize, u32, Option<Vec<u32>>),
    targets: Vec<(usize, u32, Option<Vec<u32>>)>,
}

/// One external-input wire: a root-interface input element feeds a leaf block's
/// input element. The block reads `m->u[ext_idx]` there instead of an internal
/// signal. Only present when the module's root carries an interface (a subsystem
/// exported as an open system).
struct ExtWire {
    /// Flat index into the external input vector `u[]`.
    ext_idx: usize,
    /// Flat (post-flatten) block index.
    block: usize,
    /// Flat input-element index on that block (concatenated input ports).
    in_elem: usize,
}

/// A flattened model: leaf blocks (with path-qualified provenance names) and
/// connections over their flat indices. `n_input` / `ext_wiring` / `input_names`
/// are non-empty only when the root has an interface (subsystem export).
struct Flat<'a> {
    blocks: Vec<&'a Block>,
    names: Vec<String>,
    conns: Vec<FlatConn>,
    /// Number of external input elements (root-interface inputs, flattened).
    n_input: usize,
    /// External-input wiring: which leaf input elements read `u[]`.
    ext_wiring: Vec<ExtWire>,
    /// Name per external input element (`port` or `port[elem]`).
    input_names: Vec<String>,
    /// Leaf source endpoints driving the root-interface output ports. The signals
    /// they reference are the FMU's true outputs; others are locals.
    iface_out: Vec<Endpoint>,
    /// Whether the root carries an interface (so non-interface block outputs are
    /// `local`, not `output`). A closed model exposes all block outputs.
    has_root_interface: bool,
}

/// A flat endpoint `(flat block index, port, optional elems)`.
type Endpoint = (usize, u32, Option<Vec<u32>>);

/// How a child resolves after flattening: a single leaf, or a subsystem whose
/// interface ports splice to inner leaf endpoints. A subsystem's maps already
/// hold *leaf* endpoints (its own inner subsystems are resolved recursively), so
/// a parent splices uniformly regardless of nesting depth.
enum ChildRef {
    Leaf(usize),
    Sub {
        /// Inner leaf targets fed by each interface *input* port.
        in_targets: Vec<Vec<Endpoint>>,
        /// Inner leaf source driving each interface *output* port.
        out_src: Vec<Option<Endpoint>>,
    },
}

/// Flatten an IR module into leaf blocks + flat connections, inlining nested
/// subsystems to arbitrary depth and splicing their interface connections.
fn flatten(module: &Module) -> R<Flat<'_>> {
    let mut blocks: Vec<&Block> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut conns: Vec<FlatConn> = Vec::new();
    // Capture the root's resolved child refs so root-interface inputs can be
    // wired to leaf input elements through arbitrary nesting (a closed model with
    // no root interface simply has no interface-input connections to resolve).
    let mut root_refs: Vec<ChildRef> = Vec::new();
    flatten_scope(&module.root, "", &mut blocks, &mut names, &mut conns, Some(&mut root_refs))?;

    // Resolve external-input wiring + interface-output sources against the
    // *pre-drop* flat indices (the ones `root_refs` holds), then drop sinks and
    // remap the surviving block indices.
    let (n_input, ext_pre, input_names) = root_external_inputs(&module.root, &blocks, &root_refs)?;
    let out_pre = root_interface_outputs(&module.root, &root_refs)?;
    let has_root_interface =
        !module.root.interface.inputs.is_empty() || !module.root.interface.outputs.is_empty();
    let (blocks, names, conns, new_index) = drop_sink_blocks(blocks, names, conns);
    let ext_wiring: Vec<ExtWire> = ext_pre
        .into_iter()
        .filter_map(|w| new_index[w.block].map(|nb| ExtWire { block: nb, ..w }))
        .collect();
    let iface_out: Vec<Endpoint> = out_pre
        .into_iter()
        .filter_map(|(b, p, e)| new_index[b].map(|nb| (nb, p, e)))
        .collect();
    Ok(Flat { blocks, names, conns, n_input, ext_wiring, input_names, iface_out, has_root_interface })
}

/// Resolve the leaf source endpoint driving each root-interface *output* port
/// (over pre-sink-drop flat indices). The signals these endpoints reference are
/// the FMU's true outputs (`causality=output`); other block outputs are locals.
fn root_interface_outputs(root: &Subsystem, child_refs: &[ChildRef]) -> R<Vec<Endpoint>> {
    let mut out = Vec::new();
    for c in &root.connections {
        for t in &c.targets {
            if t.block != BlockId::INTERFACE {
                continue;
            }
            if c.src.block == BlockId::INTERFACE {
                return Err(unsupported("interface pass-through connection"));
            }
            out.push(resolve_src(child_refs, c.src.block, c.src.port, &c.src.elems)?);
        }
    }
    Ok(out)
}

/// Wire the root interface's inputs to leaf block input elements (the external
/// input vector `u[]`). Returns `(n_input, wiring, names)`, with `wiring` over
/// *pre-sink-drop* flat block indices. Empty when the root has no interface
/// inputs (a closed model). Nesting and fan-out are handled by `resolve_target`,
/// the same recursive splice resolution used for internal connections: a root
/// interface input may feed a direct leaf, a nested subsystem (to any depth), or
/// several targets at once.
fn root_external_inputs(
    root: &Subsystem,
    blocks: &[&Block],
    child_refs: &[ChildRef],
) -> R<(usize, Vec<ExtWire>, Vec<String>)> {
    let in_ports = &root.interface.inputs;
    if in_ports.is_empty() {
        return Ok((0, Vec::new(), Vec::new()));
    }
    // Flat external-input layout: concatenate the interface input ports.
    let mut port_off = Vec::with_capacity(in_ports.len());
    let mut n_input = 0usize;
    let mut input_names = Vec::new();
    for p in in_ports {
        port_off.push(n_input);
        for e in 0..p.size {
            input_names.push(if p.size == 1 { p.name.clone() } else { format!("{}[{e}]", p.name) });
        }
        n_input += p.size as usize;
    }

    let mut wiring = Vec::new();
    for c in &root.connections {
        if c.src.block != BlockId::INTERFACE {
            continue; // internal / interface-output connections: handled by flatten_scope
        }
        let k = c.src.port as usize;
        let ksize = in_ports
            .get(k)
            .ok_or_else(|| unsupported("interface input port out of range"))?
            .size;
        let off_k = port_off[k];
        let s_elems: Vec<u32> = c.src.elems.clone().unwrap_or_else(|| (0..ksize).collect());
        for t in &c.targets {
            if t.block == BlockId::INTERFACE {
                return Err(unsupported("interface pass-through connection"));
            }
            // Resolve through nested subsystems to leaf endpoints (one for a leaf
            // target, several for a subsystem input that fans in). Each endpoint's
            // input elements pair positionally with this interface port's elements.
            for (eb, eport, eelems) in resolve_target(child_refs, t.block, t.port, &t.elems)? {
                let e_sizes: Vec<u32> = blocks[eb].ports.inputs.iter().map(|p| p.size).collect();
                let e_pre = port_offset(&e_sizes, eport)?;
                let e_size = *e_sizes
                    .get(eport as usize)
                    .ok_or_else(|| unsupported("interface-input target port out of range"))?;
                let e_elems: Vec<u32> = eelems.unwrap_or_else(|| (0..e_size).collect());
                for (se, te) in s_elems.iter().zip(e_elems.iter()) {
                    wiring.push(ExtWire {
                        ext_idx: off_k + *se as usize,
                        block: eb,
                        in_elem: e_pre + *te as usize,
                    });
                }
            }
        }
    }
    Ok((n_input, wiring, input_names))
}

/// Drop passive sink blocks (e.g. Scope: `BlockRole::Sink`) and their incoming
/// connections, remapping flat indices. Sinks are pure recorders: they have no
/// outputs feeding the dynamics and lower to an opaque `Op::Call`, but contribute
/// nothing to the generated C, so codegen drops them rather than erroring (the
/// same way opaque events are dropped during collection).
#[allow(clippy::type_complexity)]
fn drop_sink_blocks(
    blocks: Vec<&Block>,
    names: Vec<String>,
    conns: Vec<FlatConn>,
) -> (Vec<&Block>, Vec<String>, Vec<FlatConn>, Vec<Option<usize>>) {
    let keep: Vec<bool> = blocks.iter().map(|b| !matches!(b.role, BlockRole::Sink)).collect();
    // Old flat index -> new index among the kept blocks (identity when no sinks).
    let mut new_index = vec![None; blocks.len()];
    let mut next = 0usize;
    for (i, &k) in keep.iter().enumerate() {
        if k {
            new_index[i] = Some(next);
            next += 1;
        }
    }
    if keep.iter().all(|&k| k) {
        return (blocks, names, conns, new_index);
    }

    let kept_blocks: Vec<&Block> =
        blocks.iter().zip(&keep).filter(|(_, &k)| k).map(|(b, _)| *b).collect();
    let kept_names: Vec<String> =
        names.into_iter().zip(&keep).filter(|(_, &k)| k).map(|(n, _)| n).collect();

    let mut kept_conns = Vec::with_capacity(conns.len());
    for c in conns {
        let (sbi, sport, ssel) = c.src;
        // A sink has no outputs, so it is never a source; guard anyway.
        let Some(nsbi) = new_index[sbi] else { continue };
        let targets: Vec<_> = c
            .targets
            .into_iter()
            .filter_map(|(tbi, tport, tsel)| new_index[tbi].map(|nt| (nt, tport, tsel)))
            .collect();
        if targets.is_empty() {
            continue;
        }
        kept_conns.push(FlatConn { src: (nsbi, sport, ssel), targets });
    }

    (kept_blocks, kept_names, kept_conns, new_index)
}

/// Flatten one scope (the root or a subsystem) in place: register its children
/// (recursing into nested subsystems, so their leaves and inner connections land
/// in the shared accumulators), route its internal connections into `conns`, and
/// return how this scope's own interface ports splice to inner leaf endpoints.
fn flatten_scope<'a>(
    scope: &'a Subsystem,
    prefix: &str,
    blocks: &mut Vec<&'a Block>,
    names: &mut Vec<String>,
    conns: &mut Vec<FlatConn>,
    capture_child_refs: Option<&mut Vec<ChildRef>>,
) -> R<ChildRef> {
    let qualify = |name: &str| -> String {
        if prefix.is_empty() { name.to_string() } else { format!("{prefix}/{name}") }
    };

    // A real (port-granular) algebraic loop in this scope cannot be statically
    // ordered: reject it here, using the authoritative IR schedule rather than a
    // separate cyclicity check.
    if !scope.schedule.sccs.is_empty() {
        return Err(unsupported("algebraic loop (cannot order the algebraic pass)"));
    }

    // Emit children in the scope's schedule topo order (the IR schedule is the
    // single source of truth, built port-granular by `assemble_graph_from`), so
    // the flattened block list is itself a valid global evaluation order:
    // subsystems inline in their own schedule order at their parent position,
    // and `build_plan` consumes the order directly instead of recomputing it.
    // `child_refs` stays indexed by child position (== `BlockId`) for connection
    // resolution. The interface sentinel is skipped; any child missing from the
    // topo (shouldn't happen) is appended in declaration order.
    let topo_order: Vec<usize> = scope
        .schedule
        .topo
        .iter()
        .filter_map(|bid| {
            if *bid == BlockId::INTERFACE { return None; }
            let i = bid.0 as usize;
            (i < scope.children.len()).then_some(i)
        })
        .collect();
    let mut child_refs: Vec<Option<ChildRef>> = (0..scope.children.len()).map(|_| None).collect();
    for i in topo_order.into_iter().chain(0..scope.children.len()) {
        if child_refs[i].is_some() {
            continue;
        }
        let cref = match &scope.children[i] {
            Child::Block(b) => {
                let fi = blocks.len();
                blocks.push(b);
                names.push(qualify(&b.name));
                ChildRef::Leaf(fi)
            }
            Child::Subsystem(s) => {
                flatten_scope(s, &qualify(&s.name), blocks, names, conns, None)?
            }
        };
        child_refs[i] = Some(cref);
    }
    let child_refs: Vec<ChildRef> =
        child_refs.into_iter().map(|c| c.expect("every child emitted")).collect();

    // Route this scope's connections: interface-touching ones build the splice
    // maps; the rest are inlined into `conns` with both ends resolved to leaves.
    let mut in_targets: Vec<Vec<Endpoint>> = vec![Vec::new(); scope.interface.inputs.len()];
    let mut out_src: Vec<Option<Endpoint>> = vec![None; scope.interface.outputs.len()];
    for c in &scope.connections {
        let src_iface = c.src.block == BlockId::INTERFACE;
        for t in &c.targets {
            let tgt_iface = t.block == BlockId::INTERFACE;
            match (src_iface, tgt_iface) {
                // interface input -> inner: record the inner leaf targets it feeds
                (true, false) => {
                    let eps = resolve_target(&child_refs, t.block, t.port, &t.elems)?;
                    in_targets
                        .get_mut(c.src.port as usize)
                        .ok_or_else(|| unsupported("interface input port out of range"))?
                        .extend(eps);
                }
                // inner -> interface output: record the inner leaf source driving it
                (false, true) => {
                    let ep = resolve_src(&child_refs, c.src.block, c.src.port, &c.src.elems)?;
                    *out_src
                        .get_mut(t.port as usize)
                        .ok_or_else(|| unsupported("interface output port out of range"))? = Some(ep);
                }
                // inner -> inner: inline directly
                (false, false) => {
                    let src = resolve_src(&child_refs, c.src.block, c.src.port, &c.src.elems)?;
                    let targets = resolve_target(&child_refs, t.block, t.port, &t.elems)?;
                    conns.push(FlatConn { src, targets });
                }
                (true, true) => return Err(unsupported("interface pass-through connection")),
            }
        }
    }

    // Hand the resolved child refs to the caller (only the root asks). They map
    // each child `BlockId` to its leaf / subsystem splice, so root-interface
    // inputs can be resolved to leaf input elements through arbitrary nesting via
    // `resolve_target` — the same machinery used for internal connections.
    if let Some(cap) = capture_child_refs {
        *cap = child_refs;
    }

    Ok(ChildRef::Sub { in_targets, out_src })
}

/// Resolve a connection *source* through `child_refs` to a single leaf endpoint,
/// following a subsystem's driven output port inward.
fn resolve_src(
    child_refs: &[ChildRef],
    block: BlockId,
    port: u32,
    elems: &Option<Vec<u32>>,
) -> R<Endpoint> {
    match child_refs
        .get(block.0 as usize)
        .ok_or_else(|| unsupported("connection from unknown block"))?
    {
        ChildRef::Leaf(fi) => Ok((*fi, port, elems.clone())),
        ChildRef::Sub { out_src, .. } => out_src
            .get(port as usize)
            .and_then(|o| o.clone())
            .ok_or_else(|| unsupported("subsystem output port is not driven internally")),
    }
}

/// Resolve a connection *target* through `child_refs` to leaf endpoints (a
/// subsystem input port fans out to the inner targets it feeds).
fn resolve_target(
    child_refs: &[ChildRef],
    block: BlockId,
    port: u32,
    elems: &Option<Vec<u32>>,
) -> R<Vec<Endpoint>> {
    match child_refs
        .get(block.0 as usize)
        .ok_or_else(|| unsupported("connection to unknown block"))?
    {
        ChildRef::Leaf(fi) => Ok(vec![(*fi, port, elems.clone())]),
        ChildRef::Sub { in_targets, .. } => in_targets
            .get(port as usize)
            .cloned()
            .ok_or_else(|| unsupported("subsystem input port out of range")),
    }
}

/// Resolved wiring + layout for a flat system, derived purely from the IR.
///
/// Signals are per scalar element: a block's output ports are concatenated into
/// `out[]`/`sig[]` and its input ports into `u[]`, matching `emit_region_fn`'s
/// element addressing.
struct Plan<'a> {
    blocks: Vec<&'a Block>,
    /// Path-qualified provenance name per block.
    names: Vec<String>,
    /// Algebraic-pass evaluation order (flat indices).
    topo: Vec<usize>,
    /// First signal index of each block's outputs in `sig[]` (`None` = no outputs).
    out_off: Vec<Option<usize>>,
    n_sig: usize,
    /// State offset of each dynamic block in `x[]` (`None` = no state).
    state_off: Vec<Option<usize>>,
    n_state: usize,
    /// Input-port element sizes per block (to size the gathered `u[]`).
    in_sizes: Vec<Vec<u32>>,
    /// Per block, per input element (concatenated ports), the feeding signal
    /// index (`None` = unconnected or fed by an external input — see `ext_input`).
    input_src: Vec<Vec<Option<usize>>>,
    /// Per block, per input element, the external-input index it reads (`Some`
    /// overrides `input_src`). Non-empty only for an open system (subsystem with
    /// a root interface). Resolves to `m->u[idx]` in the struct API.
    ext_input: Vec<Vec<Option<usize>>>,
    /// Total external input element count (size of the `u[]` array).
    n_input: usize,
    /// Name per external input element (for FMI input variables).
    input_names: Vec<String>,
    /// Signal indices (into `sig[]`) that drive root-interface output ports; the
    /// FMU's true outputs. Empty for a closed model (then all outputs qualify).
    iface_output_sigs: Vec<usize>,
    /// Whether the root has an interface: then only `iface_output_sigs` are FMI
    /// outputs and other block outputs are locals; otherwise all are outputs.
    has_root_interface: bool,
    /// Per block, the global `mem[]` offset of each memory slot (indexed by
    /// `MemorySlotId`). Empty for blocks without discrete memory.
    mem_slot_off: Vec<Vec<usize>>,
    /// Total discrete-memory element count (size of the `mem[]` array).
    n_mem: usize,
}

/// Sum of the first `port` entries of `sizes` (a port's flat element offset).
pub(crate) fn port_offset(sizes: &[u32], port: u32) -> R<usize> {
    let p = port as usize;
    if p >= sizes.len() {
        return Err(unsupported(format!("port index {port} out of range")));
    }
    Ok(sizes[..p].iter().map(|&s| s as usize).sum())
}

fn total_elems(sizes: &[u32]) -> usize {
    sizes.iter().map(|&s| s as usize).sum()
}

/// Resolve layout + wiring + evaluation order from a flattened model.
/// An opaque block lowers (some of) its outputs to an `Op::Call` extern — it
/// carries no static op-graph for its math (RNG/noise, an arbitrary
/// Python/Rust callable, or a DAE). The C backend cannot emit it.
fn block_is_opaque(b: &Block) -> bool {
    let has_call = |r: &Region| r.ops.iter().any(|o| matches!(o, Op::Call { .. }));
    has_call(&b.regions.alg) || has_call(&b.regions.dyn_)
}

fn build_plan<'a>(flat: Flat<'a>) -> R<Plan<'a>> {
    let blocks = flat.blocks;
    let names = flat.names;
    let n = blocks.len();

    // Reject opaque blocks up front with a clear, block-named error. Without
    // this, the first `Op::Call` only surfaces deep in rvalue lowering as a
    // cryptic "Op::Call (opaque extern)" with no hint which block is at fault.
    let opaque: Vec<String> = blocks
        .iter()
        .zip(names.iter())
        .filter(|(b, _)| block_is_opaque(b))
        .map(|(b, name)| {
            if name.is_empty() || name.as_str() == b.type_name {
                b.type_name.clone()
            } else {
                format!("{} ({})", name, b.type_name)
            }
        })
        .collect();
    if !opaque.is_empty() {
        return Err(unsupported(format!(
            "{count} block(s) have no static op-graph and cannot be lowered to C \
             (opaque: RNG/noise, arbitrary Python/Rust callable, or DAE): {list}. \
             Use op-expressible blocks, or remove these before code generation.",
            count = opaque.len(),
            list = opaque.join(", "),
        )));
    }

    let in_sizes: Vec<Vec<u32>> =
        blocks.iter().map(|b| b.ports.inputs.iter().map(|p| p.size).collect()).collect();
    let out_sizes: Vec<Vec<u32>> =
        blocks.iter().map(|b| b.ports.outputs.iter().map(|p| p.size).collect()).collect();

    // Output-signal layout (block order, element-wise) and state layout.
    let mut out_off = Vec::with_capacity(n);
    let mut n_sig = 0usize;
    let mut state_off = Vec::with_capacity(n);
    let mut n_state = 0usize;
    for (i, b) in blocks.iter().enumerate() {
        let n_out = total_elems(&out_sizes[i]);
        out_off.push(if n_out > 0 {
            let o = n_sig;
            n_sig += n_out;
            Some(o)
        } else {
            None
        });
        // Any block carrying continuous state needs integration — not only
        // `Dynamic` ones. A stateful source (e.g. the chirp's phase integrator)
        // maps to IR `Source` for scheduling (no inputs, depth 0) yet still owns
        // a state to advance; gating on the state itself (Sinks are already
        // dropped) keeps both classes covered.
        state_off.push(if !b.state.is_empty() && !matches!(b.role, BlockRole::Sink) {
            let o = n_state;
            n_state += b.state.len();
            Some(o)
        } else {
            None
        });
    }

    // Discrete-memory layout: concatenate each block's slots (in id order) into
    // a global mem[] array; `mem_slot_off[block][slot_id]` is the slot's base.
    let mut mem_slot_off: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut n_mem = 0usize;
    for b in &blocks {
        let max_id = b.memory.iter().map(|s| s.id.0 as usize).max();
        let mut offs = vec![0usize; max_id.map_or(0, |m| m + 1)];
        let mut slots: Vec<_> = b.memory.iter().collect();
        slots.sort_by_key(|s| s.id.0);
        for s in slots {
            offs[s.id.0 as usize] = n_mem;
            n_mem += s.size as usize;
        }
        mem_slot_off.push(offs);
    }

    // Which block owns each signal index (for dependency ordering).
    let mut sig_owner = vec![usize::MAX; n_sig];
    for (i, off) in out_off.iter().enumerate() {
        if let Some(o) = off {
            for k in 0..total_elems(&out_sizes[i]) {
                sig_owner[o + k] = i;
            }
        }
    }

    // Element-level input wiring from connections, honouring `elems` slicing.
    let mut input_src: Vec<Vec<Option<usize>>> =
        in_sizes.iter().map(|s| vec![None; total_elems(s)]).collect();
    for c in &flat.conns {
        let (sbi, sport, ref s_sel) = c.src;
        let s_base = out_off
            .get(sbi)
            .copied()
            .flatten()
            .ok_or_else(|| unsupported("connection from a block with no outputs"))?;
        let s_pre = port_offset(&out_sizes[sbi], sport)?;
        let s_size = out_sizes[sbi][sport as usize];
        let s_elems: Vec<u32> = s_sel.clone().unwrap_or_else(|| (0..s_size).collect());
        for (tbi, tport, t_sel) in &c.targets {
            let t_pre = port_offset(&in_sizes[*tbi], *tport)?;
            let t_size = in_sizes[*tbi][*tport as usize];
            let t_elems: Vec<u32> = t_sel.clone().unwrap_or_else(|| (0..t_size).collect());
            for (se, te) in s_elems.iter().zip(t_elems.iter()) {
                let ssig = s_base + s_pre + *se as usize;
                let tin = t_pre + *te as usize;
                if let Some(slot) = input_src.get_mut(*tbi).and_then(|v| v.get_mut(tin)) {
                    *slot = Some(ssig);
                }
            }
        }
    }

    // External-input wiring (root-interface inputs → leaf input elements). These
    // override `input_src` at the same input element and resolve to `m->u[]`.
    let mut ext_input: Vec<Vec<Option<usize>>> =
        in_sizes.iter().map(|s| vec![None; total_elems(s)]).collect();
    for w in &flat.ext_wiring {
        if let Some(slot) = ext_input.get_mut(w.block).and_then(|v| v.get_mut(w.in_elem)) {
            *slot = Some(w.ext_idx);
        }
    }

    // Interface-output signal indices (the FMU's true outputs): convert the
    // resolved leaf endpoints to `sig[]` indices via the output layout.
    let mut iface_output_sigs = Vec::new();
    for (b, port, elems) in &flat.iface_out {
        let base = out_off
            .get(*b)
            .copied()
            .flatten()
            .ok_or_else(|| unsupported("interface output driven by a block with no outputs"))?;
        let pre = port_offset(&out_sizes[*b], *port)?;
        let size = out_sizes[*b][*port as usize];
        let es: Vec<u32> = elems.clone().unwrap_or_else(|| (0..size).collect());
        for e in es {
            iface_output_sigs.push(base + pre + e as usize);
        }
    }

    // Algebraic-pass order: the flattener already emitted blocks in the IR
    // schedule's topo order (the single, port-granular source of truth, see
    // `flatten_scope`), so the flat index order IS the evaluation order. Real
    // algebraic loops were rejected per-scope during flattening. No recompute.
    let topo: Vec<usize> = (0..n).collect();

    Ok(Plan {
        blocks, names, topo, out_off, n_sig, state_off, n_state, in_sizes, input_src,
        ext_input, n_input: flat.n_input, input_names: flat.input_names,
        iface_output_sigs, has_root_interface: flat.has_root_interface,
        mem_slot_off, n_mem,
    })
}

/// The top-of-file comment banner prepended to every generated C file: what the
/// file is, which model it came from, the generator version, and a do-not-edit
/// notice. A one-liner per file name keeps each file self-describing.
fn file_banner(file_name: &str, model_name: &str) -> String {
    // Files are named `<base>.{h,c}` / `<base>_blocks.*` / `<base>_solver.*`
    // (see `file_base`), so classify by suffix.
    let summary = if file_name.ends_with("_blocks.h") {
        "Per-block region-function prototypes."
    } else if file_name.ends_with("_blocks.c") {
        "Per-block region functions: algebraic outputs and state derivatives."
    } else if file_name.ends_with("_solver.h") {
        "Tableau-driven integrator interface."
    } else if file_name.ends_with("_solver.c") {
        "Tableau-driven integrator implementation."
    } else if file_name.ends_with(".h") {
        "Public interface: model dimensions, state layout, and entry-point prototypes."
    } else if file_name.ends_with(".c") {
        "Model implementation: block equations, dx/dt, outputs, init, and stepping."
    } else {
        "Generated C source."
    };
    let name = if model_name.is_empty() { "model" } else { model_name };
    format!(
        "/*\n\
         \x20* {file_name} - {summary}\n\
         \x20*\n\
         \x20* Model:     {name}\n\
         \x20* Generator: fastsim {ver} (C99 code generation)\n\
         \x20*\n\
         {LICENSE_BANNER}\
         \x20*\n\
         \x20* Auto-generated from a fastsim block diagram. Do not edit by hand:\n\
         \x20* changes are overwritten on regeneration. Edit the model instead.\n\
         \x20*/\n",
        ver = env!("CARGO_PKG_VERSION"),
    )
}

/// License notice stamped into every generated file (see [`file_banner`]). The
/// generated C is "Output" under fastsim's license: free for noncommercial use,
/// but shipping it in a commercial product needs a commercial license. Keeping
/// this on the artifact itself — not just in the repo's LICENSE — is what makes
/// the term travel with code that gets handed onward.
const LICENSE_BANNER: &str = "\
\x20* License:   PolyForm Noncommercial 1.0.0. This generated code is \"Output\"\n\
\x20*            under fastsim's license: free for noncommercial use, but using\n\
\x20*            or distributing it in a commercial product requires a\n\
\x20*            commercial license. Contact: info@pathsim.org\n";


/// Generate the full set of C files a build needs, named for the file each
/// becomes on disk. Every model is emitted through the struct (`model_t`) API.
/// `Layout::Compact` yields `model.h` (the `model_t` typedef, dimensions,
/// `<NAME>_SIG_*` ids, and entry-point prototypes) + `model.c`; `Layout::Library`
/// additionally splits out `solver.{c,h}` (the tableau-driven integrator) and, under
/// `Structure::Hierarchical`, `blocks.{c,h}` (the per-block `<name>_blk_i_alg` /
/// `<name>_blk_i_deriv` functions). Every extern symbol and include guard is
/// prefixed with the model name; see `doc/codegen.md` for the emitted-API contract.
pub fn generate(module: &Module, opts: &CodegenOptions) -> R<Vec<GeneratedFile>> {
    let file = |name: &str, contents: String| {
        // Always end with a newline: a generated file without a trailing newline
        // trips `-Wnewline-eof` for everyone who compiles the downloaded C.
        let body = format!("{}{}", file_banner(name, &module.name), contents);
        GeneratedFile {
            name: name.to_string(),
            contents: if body.ends_with('\n') { body } else { format!("{body}\n") },
        }
    };

    let ctx = build_struct_ctx(module, opts)?;
    // File names carry the model name (`file_base`) so two generated models
    // coexist in one build directory; the templates emit their internal
    // `#include`s from the same base, so the set is consistent by construction.
    let base = ctx.file_base.clone();
    let mut files = vec![
        file(&format!("{base}.h"), render("model_struct.h", &ctx)?),
        file(&format!("{base}.c"), render("model_struct.c", &ctx)?),
    ];
    if opts.layout == Layout::Library {
        // Hierarchical: the per-block functions move to their own files.
        if !ctx.block_protos.is_empty() {
            files.push(file(&format!("{base}_blocks.h"), render("blocks_struct.h", &ctx)?));
            files.push(file(&format!("{base}_blocks.c"), render("blocks_struct.c", &ctx)?));
        }
        // The tableau-driven integrator is split out (shareable/swappable).
        files.push(file(&format!("{base}_solver.h"), render("solver_struct.h", &ctx)?));
        files.push(file(&format!("{base}_solver.c"), render("solver_struct.c", &ctx)?));
    }
    if opts.scaffold && opts.numeric.frac().is_some() {
        return Err(CodegenError::Unsupported(
            "scaffold under fixed point (the demo driver prints a floating CSV); \
             generate the scaffold with numeric=\"double\"/\"float\" or write \
             your own driver over <name>_step"
                .into(),
        ));
    }
    if opts.scaffold {
        // Build scaffold: CMakeLists + an EDITABLE demo driver over
        // `<name>_step`. Emitted RAW (no do-not-edit banner -- these files
        // carry their own "editable starting point" headers) and AFTER the
        // model sources, so `model_sources` below is exactly the .c set the
        // static library needs.
        #[derive(Serialize)]
        struct ScaffoldCtx<'a> {
            name: &'a str,
            file_base: &'a str,
            version: &'a str,
            entry_header: String,
            model_sources: Vec<String>,
            /// Signal names printed by the demo CSV: states + block outputs
            /// (parameters are constants -- noise in a trajectory).
            print_sigs: Vec<String>,
        }
        let entry_header = if opts.layout == Layout::Library {
            format!("{base}_solver.h")
        } else {
            format!("{base}.h")
        };
        let sc = ScaffoldCtx {
            name: &ctx.name,
            file_base: &base,
            version: env!("CARGO_PKG_VERSION"),
            entry_header,
            model_sources: files
                .iter()
                .filter(|f| f.name.ends_with(".c"))
                .map(|f| f.name.clone())
                .collect(),
            print_sigs: ctx
                .sigs
                .iter()
                .filter(|s| s.id < ctx.n_state + ctx.n_sig)
                .map(|s| s.name.clone())
                .collect(),
        };
        files.push(GeneratedFile {
            name: format!("{base}_main.c"),
            contents: render("scaffold_main.c", &sc)?,
        });
        files.push(GeneratedFile {
            name: "CMakeLists.txt".to_string(),
            contents: render("scaffold_cmake", &sc)?,
        });
    }
    if opts.trace {
        // Model-to-code trace map + static metrics, derived AFTER emission so
        // function definitions can be resolved to file/line in the actual
        // output (see `CodegenOptions::trace`).
        let trace = build_trace_json(module, opts, &files)?;
        files.push(GeneratedFile { name: format!("{base}_trace.json"), contents: trace });
    }
    if opts.a2l {
        files.push(GeneratedFile { name: format!("{base}.a2l"), contents: build_a2l(module, opts)? });
    }
    Ok(files)
}


/// One field of the emitted `<name>_t` struct with its computed offset.
#[derive(Debug, Clone, PartialEq, Eq)]
struct StructField {
    name: String,
    /// C type as emitted (`double`/`float`, `size_t`, `int`).
    ctype: &'static str,
    /// Array element count (1 for scalars).
    count: usize,
    /// Byte offset of the field within the struct.
    offset: usize,
    /// Size of ONE element in bytes.
    elem_size: usize,
}

/// The emitted `<name>_t` field inventory with byte offsets, computed by the
/// standard natural-alignment layout rules (each field aligned to its own
/// size; total size padded to the widest alignment). For the simple field
/// types the struct uses (`double`/`float`, `size_t`, `int`) these rules are
/// shared by all mainstream ABIs (x86-64 SysV, MSVC x64, ARM AAPCS/AAPCS64),
/// so the offsets are what `offsetof` yields there. `size_t` is assumed
/// 8 bytes (64-bit target); document accordingly where exposed.
///
/// MUST mirror the field ORDER of `templates/model_struct.h.jinja` — the
/// `struct_layout_matches_emitted_header` test parses the emitted header and
/// pins the two against each other.
fn model_struct_fields(
    plan: &Plan,
    events: &[EventSpec],
    tableau: &crate::solvers::tableaus::Tableau,
    numeric: Numeric,
) -> (Vec<StructField>, usize) {
    let real: &'static str = numeric.real();
    let real_size: usize = if numeric.real() == "double" { 8 } else { 4 };
    let mut fields: Vec<(String, &'static str, usize, usize)> = Vec::new(); // name, ctype, count, elem_size
    let mut push = |name: String, ctype: &'static str, count: usize, elem: usize| {
        fields.push((name, ctype, count, elem));
    };

    push("time".into(), real, 1, real_size);
    if plan.n_state > 0 {
        push("x".into(), real, plan.n_state, real_size);
    }
    if plan.n_sig > 0 {
        push("sig".into(), real, plan.n_sig, real_size);
    }
    let n_param: usize = plan.blocks.iter().map(|b| flatten_params(b).len()).sum();
    if n_param > 0 {
        push("p".into(), real, n_param, real_size);
    }
    if plan.n_input > 0 {
        push("u".into(), real, plan.n_input, real_size);
    }
    if plan.n_mem > 0 {
        push("mem".into(), real, plan.n_mem, real_size);
    }
    for ev in events {
        let suffix = ev.suffix();
        match ev {
            EventSpec::Periodic { .. } => push(format!("next_{suffix}"), real, 1, real_size),
            EventSpec::Fixed { .. } => push(format!("fi_{suffix}"), "size_t", 1, 8),
            EventSpec::ZeroCross { .. } | EventSpec::Condition { .. } => {
                push(format!("prev_{suffix}"), real, 1, real_size);
                push(format!("init_{suffix}"), "int", 1, 4);
            }
        }
    }
    if !events.is_empty() {
        push("fs_started".into(), "int", 1, 4);
    }
    if tableau.is_adaptive() {
        push("fs_h".into(), real, 1, real_size);
    }

    // Natural-alignment layout: align each field to its element size.
    let mut out = Vec::with_capacity(fields.len());
    let mut off = 0usize;
    let mut max_align = 1usize;
    for (name, ctype, count, elem) in fields {
        let align = elem; // natural alignment == element size for these types
        max_align = max_align.max(align);
        off = off.div_ceil(align) * align;
        out.push(StructField { name, ctype, count, offset: off, elem_size: elem });
        off += elem * count;
    }
    let total = off.div_ceil(max_align) * max_align;
    (out, total)
}

/// Build the `<name>.a2l` contents (see [`CodegenOptions::a2l`]): every
/// addressable variable of the generated model as an ASAP2 MEASUREMENT
/// (time, states, block outputs, external inputs, discrete memory) or
/// CHARACTERISTIC (tunable parameters), addressed as `SYMBOL_LINK
/// "<name>" <offset>` with offsets from [`model_struct_fields`]. Names come
/// from the same disambiguated inventory as the `SIG_*` enum, so the A2L and
/// the generated C agree by construction.
fn build_a2l(module: &Module, opts: &CodegenOptions) -> R<String> {
    #[derive(Serialize)]
    struct Entry {
        name: String,
        desc: String,
        offset: usize,
        writable: bool,
    }
    #[derive(Serialize)]
    struct A2lCtx {
        name: String,
        file_base: String,
        symbol: String,
        version: &'static str,
        real: &'static str,
        dtype: &'static str,
        lo: String,
        hi: String,
        /// Fixed point: the LINEAR conversion factor `2^-frac` (phys = a*int);
        /// `None` emits the identity conversion.
        compu_scale: Option<String>,
        measurements: Vec<Entry>,
        characteristics: Vec<Entry>,
    }

    let flat = flatten(module)?;
    let plan = build_plan(flat)?;
    let events = collect_events(&plan)?;
    let model_name = c_ident(&module.name);
    let layout = build_layout(&plan, &model_name);
    let (fields, _total) = model_struct_fields(&plan, &events, opts.solver.tableau(), opts.numeric);
    let real_size: usize = if opts.numeric.real() == "double" { 8 } else { 4 };
    let field_off = |nm: &str| fields.iter().find(|f| f.name == nm).map(|f| f.offset);

    let (x_off, sig_off, p_off, u_off, mem_off) =
        (field_off("x"), field_off("sig"), field_off("p"), field_off("u"), field_off("mem"));

    let mut measurements = Vec::new();
    let mut characteristics = Vec::new();
    measurements.push(Entry {
        name: "time".into(),
        desc: "simulation time".into(),
        offset: field_off("time").unwrap_or(0),
        writable: false,
    });
    for v in &layout.vars {
        match v.kind {
            VarKind::State => {
                if let Some(base) = x_off {
                    measurements.push(Entry {
                        name: v.name.clone(),
                        desc: "continuous state".into(),
                        offset: base + v.signal_id * real_size,
                        writable: true,
                    });
                }
            }
            VarKind::Output | VarKind::Local => {
                if let Some(base) = sig_off {
                    measurements.push(Entry {
                        name: v.name.clone(),
                        desc: "block output signal".into(),
                        offset: base + (v.signal_id - plan.n_state) * real_size,
                        writable: false,
                    });
                }
            }
            VarKind::Param => {
                if let Some(base) = p_off {
                    characteristics.push(Entry {
                        name: v.name.clone(),
                        desc: "tunable parameter".into(),
                        offset: base + (v.signal_id - plan.n_state - plan.n_sig) * real_size,
                        writable: true,
                    });
                }
            }
            VarKind::Input => {
                if let Some(base) = u_off {
                    measurements.push(Entry {
                        name: v.name.clone(),
                        desc: "external input".into(),
                        offset: base + v.signal_id * real_size,
                        writable: true,
                    });
                }
            }
        }
    }
    // Discrete memory: not in the signal-id space; name per block/slot/element,
    // disambiguated exactly like `build_layout` disambiguates its vars.
    if let Some(base) = mem_off {
        let mut used: HashSet<String> =
            layout.vars.iter().map(|v| v.name.clone()).collect();
        used.insert("time".into());
        for (i, b) in plan.blocks.iter().enumerate() {
            for (slot_idx, m) in b.memory.iter().enumerate() {
                let off0 = plan.mem_slot_off[i].get(slot_idx).copied().unwrap_or(0);
                for e in 0..m.size as usize {
                    let stem = if m.size > 1 {
                        format!("{}_{}_{e}", c_ident(&plan.names[i]), c_ident(&m.name))
                    } else {
                        format!("{}_{}", c_ident(&plan.names[i]), c_ident(&m.name))
                    };
                    let mut nm = stem.clone();
                    let mut k = 1;
                    while !used.insert(nm.clone()) {
                        nm = format!("{stem}_{k}");
                        k += 1;
                    }
                    measurements.push(Entry {
                        name: nm,
                        desc: "discrete memory".into(),
                        offset: base + (off0 + e) * real_size,
                        writable: false,
                    });
                }
            }
        }
    }

    let float = opts.numeric == Numeric::Float;
    // Fixed point: raw int32 storage with a LINEAR conversion (phys = 2^-frac
    // * int) — exactly the classic A2L fixed-point pattern, so the tool shows
    // physical values while reading Q ints.
    let (dtype, lo, hi, compu_scale) = match opts.numeric.frac() {
        Some(frac) => (
            "SLONG",
            "-2147483648".to_string(),
            "2147483647".to_string(),
            Some(format!("{:e}", 1.0 / (1i64 << frac) as f64)),
        ),
        None if float => ("FLOAT32_IEEE", "-3.4E38".to_string(), "3.4E38".to_string(), None),
        None => ("FLOAT64_IEEE", "-1.7E308".to_string(), "1.7E308".to_string(), None),
    };
    let ctx = A2lCtx {
        symbol: model_name.clone(),
        name: model_name,
        file_base: file_base(&module.name),
        version: env!("CARGO_PKG_VERSION"),
        real: opts.numeric.real(),
        dtype,
        lo,
        hi,
        compu_scale,
        measurements,
        characteristics,
    };
    render("a2l", &ctx)
}

/// Build the `<name>_trace.json` contents (see [`CodegenOptions::trace`]):
/// the model-to-code trace map — block → emitted functions (resolved to
/// file/line in the actual output), block → states/outputs/params with their
/// `SIG_*` ids, block → events — plus static metrics (packed RAM estimate,
/// integrator stack estimate, IR op counts, per-step work). Everything here
/// is derived from the same plan/layout the emitter used, so map and code
/// agree by construction; only the line numbers are recovered by scanning the
/// emitted text (definitions are flush-left, calls are indented).
fn build_trace_json(
    module: &Module,
    opts: &CodegenOptions,
    files: &[GeneratedFile],
) -> R<String> {
    use serde_json::{json, Value};

    let flat = flatten(module)?;
    let plan = build_plan(flat)?;
    let events = collect_events(&plan)?;
    let model_name = c_ident(&module.name);
    let layout = build_layout(&plan, &model_name);
    let t = opts.solver.tableau();
    let real_bytes: usize = if opts.numeric.real() == "double" { 8 } else { 4 };
    let hierarchical = opts.structure == Structure::Hierarchical;

    // Parameter offsets, exactly as build_layout computes them.
    let flat_params: Vec<Vec<f64>> = plan.blocks.iter().map(|b| flatten_params(b)).collect();
    let mut param_off = Vec::with_capacity(plan.blocks.len());
    let mut n_param = 0usize;
    for fp in &flat_params {
        param_off.push(n_param);
        n_param += fp.len();
    }

    // Resolve a symbol DEFINITION to {file, line}: definitions are emitted
    // flush-left (possibly behind `static `), calls are indented. Scan `.c`
    // files first so a header declaration never shadows the definition.
    let def_of = |symbol: &str| -> Option<Value> {
        let needle = format!("{symbol}(");
        let mut ordered: Vec<&GeneratedFile> = files.iter().filter(|f| f.name.ends_with(".c")).collect();
        ordered.extend(files.iter().filter(|f| !f.name.ends_with(".c")));
        for f in ordered {
            for (i, line) in f.contents.lines().enumerate() {
                if !line.starts_with(' ') && !line.starts_with('/') && !line.starts_with('*')
                    && line.contains(&needle)
                {
                    return Some(json!({"symbol": symbol, "file": f.name, "line": i + 1}));
                }
            }
        }
        None
    };

    // -- per-block map ------------------------------------------------------------------
    let var = |idx: usize| -> Value {
        let v = &layout.vars[idx];
        json!({"name": v.name, "signal_id": v.signal_id, "start": v.start})
    };
    let mut blocks = Vec::new();
    let (mut alg_ops_total, mut deriv_ops_total, mut event_ops_total) = (0usize, 0usize, 0usize);
    for (i, b) in plan.blocks.iter().enumerate() {
        let alg_ops = b.regions.alg.ops.len();
        let deriv_ops = b.regions.dyn_.ops.len();
        alg_ops_total += alg_ops;
        deriv_ops_total += deriv_ops;

        let mut functions = Vec::new();
        if hierarchical {
            for suffix in ["alg", "deriv"] {
                if let Some(d) = def_of(&format!("{model_name}_blk_{i}_{suffix}")) {
                    functions.push(d);
                }
            }
        }
        let states: Vec<Value> = plan.state_off[i]
            .map(|off| (0..b.state.len()).map(|k| var(off + k)).collect())
            .unwrap_or_default();
        let outputs: Vec<Value> = plan.out_off[i]
            .map(|off| {
                let n_out: usize = b.ports.outputs.iter().map(|p| p.size as usize).sum();
                (0..n_out).map(|k| var(plan.n_state + off + k)).collect()
            })
            .unwrap_or_default();
        let params: Vec<Value> = (0..flat_params[i].len())
            .map(|k| var(plan.n_state + plan.n_sig + param_off[i] + k))
            .collect();
        let memory: Vec<Value> = b
            .memory
            .iter()
            .enumerate()
            .map(|(slot, m)| {
                json!({"name": m.name, "size": m.size,
                       "mem_offset": plan.mem_slot_off[i].get(slot)})
            })
            .collect();
        let block_events: Vec<Value> = events
            .iter()
            .filter(|ev| ev.block() == i)
            .map(|ev| {
                let suffix = ev.suffix();
                event_ops_total += ev.effect().ops.len();
                json!({
                    "suffix": suffix,
                    "effect": def_of(&format!("effect_{suffix}")),
                    "guard": ev.guard().map(|_| def_of(&format!("guard_{suffix}"))),
                    "effect_ops": ev.effect().ops.len(),
                })
            })
            .collect();

        blocks.push(json!({
            "name": plan.names[i],
            "type": b.type_name,
            "functions": functions,
            "inlined_into": if hierarchical { Value::Null } else {
                json!([format!("{model_name}_outputs"), format!("{model_name}_deriv")])
            },
            "states": states,
            "outputs": outputs,
            "params": params,
            "memory": memory,
            "events": block_events,
            "ops": {"alg": alg_ops, "deriv": deriv_ops},
        }));
    }

    // -- static metrics -----------------------------------------------------------------
    // model_t size, PACKED (no padding/ABI assumptions — a lower bound):
    // time + x[] + sig[] + p[] + mem[] + u[] + per-event bookkeeping (+ fs_h).
    let mut event_field_bytes = 0usize;
    for ev in &events {
        event_field_bytes += match ev {
            EventSpec::Periodic { .. } => real_bytes,
            EventSpec::Fixed { .. } => 8, // size_t index
            EventSpec::ZeroCross { .. } | EventSpec::Condition { .. } => real_bytes + 4,
        };
    }
    if !events.is_empty() {
        event_field_bytes += 4; // fs_started
    }
    let adaptive = t.is_adaptive();
    let struct_bytes = real_bytes
        * (1 + plan.n_state + plan.n_sig + n_param + plan.n_mem + plan.n_input)
        + event_field_bytes
        + if adaptive { real_bytes } else { 0 };
    // Integrator kernel locals: x0[n] + k[s][n] (the dominant stack user).
    let kernel_stack = real_bytes * plan.n_state * (1 + t.s)
        + if adaptive { real_bytes * plan.n_state } else { 0 };
    // Butcher constants in ROM: c[s] + a[s][s] (+ tr[s] when adaptive).
    let tableau_bytes = real_bytes * (t.s + t.s * t.s + if adaptive { t.s } else { 0 });
    // One step evaluates deriv (= outputs + block derivs) once per stage.
    let per_step_ops = t.s * (alg_ops_total + deriv_ops_total);

    // Exact struct layout (natural alignment, 64-bit size_t) — shared with
    // the A2L emitter, so trace map and calibration map cannot diverge.
    let (struct_fields, struct_total) =
        model_struct_fields(&plan, &events, t, opts.numeric);
    let struct_layout: Vec<Value> = struct_fields
        .iter()
        .map(|f| json!({
            "name": f.name, "ctype": f.ctype, "count": f.count,
            "offset": f.offset, "elem_size": f.elem_size,
        }))
        .collect();

    let entry_points: Vec<Value> = ["init", "step", "run", "handle_events", "outputs", "deriv", "get_signal", "set_signal", "jvp"]
        .iter()
        .filter_map(|ep| def_of(&format!("{model_name}_{ep}")))
        .collect();

    let signals: Vec<Value> = layout
        .vars
        .iter()
        .map(|v| json!({
            "name": v.name, "signal_id": v.signal_id,
            "kind": format!("{:?}", v.kind), "settable": v.settable(),
            "start": v.start,
        }))
        .collect();

    let doc = json!({
        "model": module.name,
        "generator": format!("fastsim {}", env!("CARGO_PKG_VERSION")),
        "ir_version": module.ir_version,
        "options": {
            "numeric": format!("{:?}", opts.numeric),
            "structure": format!("{:?}", opts.structure),
            "layout": format!("{:?}", opts.layout),
            "solver": t.name,
        },
        "files": files.iter().map(|f| f.name.clone()).collect::<Vec<_>>(),
        "entry_points": entry_points,
        "blocks": blocks,
        "signals": signals,
        "struct_layout": struct_layout,
        "metrics": {
            "n_state": plan.n_state,
            "n_sig": plan.n_sig,
            "n_param": n_param,
            "n_mem": plan.n_mem,
            "n_input": plan.n_input,
            "n_events": events.len(),
            "solver_stages": t.s,
            "model_struct_bytes_packed": struct_bytes,
            "model_struct_bytes_aligned64": struct_total,
            "integrator_stack_bytes": kernel_stack,
            "tableau_const_bytes": tableau_bytes,
            "ir_ops": {"alg": alg_ops_total, "deriv": deriv_ops_total, "events": event_ops_total},
            "per_step_ops_estimate": per_step_ops,
            "notes": "byte figures are packed lower bounds (no struct padding, \
compiler- and ABI-independent); per_step_ops counts IR ops (stages x (alg + deriv)), \
a proxy for FLOPs and code size, not a cycle count",
        },
    });
    serde_json::to_string_pretty(&doc)
        .map_err(|e| CodegenError::Template(format!("trace serialization: {e}")))
}

/// An emittable event with its block/index provenance. Opaque events (host-side
/// recording / RNG, not expressible as ops) are dropped during collection.
enum EventSpec<'a> {
    Periodic { block: usize, idx: usize, period: f64, phase: f64, effect: &'a Region },
    Fixed { block: usize, idx: usize, times: &'a [f64], effect: &'a Region },
    ZeroCross { block: usize, idx: usize, guard: &'a Region, direction: Direction, effect: &'a Region },
    Condition { block: usize, idx: usize, guard: &'a Region, effect: &'a Region },
}

impl<'a> EventSpec<'a> {
    fn block(&self) -> usize {
        match self {
            Self::Periodic { block, .. }
            | Self::Fixed { block, .. }
            | Self::ZeroCross { block, .. }
            | Self::Condition { block, .. } => *block,
        }
    }
    fn idx(&self) -> usize {
        match self {
            Self::Periodic { idx, .. }
            | Self::Fixed { idx, .. }
            | Self::ZeroCross { idx, .. }
            | Self::Condition { idx, .. } => *idx,
        }
    }
    fn effect(&self) -> &'a Region {
        match self {
            Self::Periodic { effect, .. }
            | Self::Fixed { effect, .. }
            | Self::ZeroCross { effect, .. }
            | Self::Condition { effect, .. } => effect,
        }
    }
    /// The scalar guard region, for the kinds that test one (zero-cross, condition).
    fn guard(&self) -> Option<&'a Region> {
        match self {
            Self::ZeroCross { guard, .. } | Self::Condition { guard, .. } => Some(guard),
            _ => None,
        }
    }
    /// `block_idx`, used to name this event's file-static variables uniquely.
    fn suffix(&self) -> String {
        format!("{}_{}", self.block(), self.idx())
    }
}

/// Does this event read any block input (via its effect or guard)? Determines
/// whether `model_handle_events` must compute the `sig[]` algebraic pass.
fn event_reads_input(ev: &EventSpec) -> bool {
    region_reads_input(ev.effect()) || ev.guard().is_some_and(region_reads_input)
}

/// Collect every emittable event. All four expressible kinds (periodic / fixed
/// schedule, zero-cross, condition) are emitted. An opaque event (its guard or
/// effect is host code lowering to an `Op::Call`, not ops) is rejected loudly,
/// mirroring the opaque-block rejection in `build_plan`: silently dropping it
/// would emit code that omits a real event and is therefore wrong.
fn collect_events<'a>(plan: &Plan<'a>) -> R<Vec<EventSpec<'a>>> {
    let mut evs = Vec::new();
    for (i, b) in plan.blocks.iter().enumerate() {
        for (e, ev) in b.events.iter().enumerate() {
            if ev.opaque {
                let name = plan.names.get(i).filter(|n| !n.is_empty()).map(String::as_str).unwrap_or(&b.type_name);
                return Err(unsupported(format!(
                    "block '{name}' ({}) event {e} is opaque (its guard/effect is host code, \
                     not a static op-graph) and cannot be lowered to C. \
                     Use an op-expressible event, or remove this block before code generation.",
                    b.type_name,
                )));
            }
            evs.push(match &ev.kind {
                EventKind::Schedule { times: ScheduleTimes::Periodic { period, phase } } => {
                    EventSpec::Periodic { block: i, idx: e, period: *period, phase: *phase, effect: &ev.effect }
                }
                EventKind::Schedule { times: ScheduleTimes::Fixed(times) } => {
                    EventSpec::Fixed { block: i, idx: e, times, effect: &ev.effect }
                }
                EventKind::ZeroCross { guard, direction } => {
                    EventSpec::ZeroCross { block: i, idx: e, guard, direction: *direction, effect: &ev.effect }
                }
                EventKind::Condition { guard } => {
                    EventSpec::Condition { block: i, idx: e, guard, effect: &ev.effect }
                }
            });
        }
    }
    Ok(evs)
}

/// Does a region read any block input (vs only state/time/params)? Used to
/// decide whether to emit the per-call input gather at all.
fn region_reads_input(r: &Region) -> bool {
    r.ops.iter().any(|op| matches!(op, Op::Input { .. }))
}

/// Block parameters flattened to scalar values in id order (the layout both the
/// hierarchical `P_i[]` array and the flat driver's inlined literals use).
fn flatten_params(b: &Block) -> Vec<f64> {
    let mut sorted: Vec<_> = b.params.iter().collect();
    sorted.sort_by_key(|p| p.id.0);
    let mut vals: Vec<f64> = Vec::new();
    for p in sorted {
        match &p.value {
            ParamValue::Scalar(v) => vals.push(*v),
            ParamValue::Vector(vs) => vals.extend(vs),
            ParamValue::Matrix { data, .. } => vals.extend(data),
        }
    }
    vals
}


// ======================================================================================
// Struct model (rtModel-style): one `model_t` with get/set signal accessors
// ======================================================================================

/// Struct-mode leaf resolution: every read targets the `model_t* m` instance.
/// Inputs come from the signal store `m->sig[]` (the producing block's output),
/// states from `m->x[]`, parameters from `m->p[]` (runtime-settable), memory
/// from `m->mem[]`. Node-refs are global temps within the current function.
struct StructLeaves<'a> {
    target: &'a CTarget,
    l2g: &'a [usize],
    input_src: &'a [Option<usize>],
    ext_input: &'a [Option<usize>],
    in_sizes: &'a [u32],
    state_base: usize,
    param_base: usize,
    mem_off: &'a [usize],
}

impl Leaves for StructLeaves<'_> {
    fn temp(&self, node: NodeId) -> String {
        self.target.temp(self.l2g[node.0 as usize] as u32)
    }
    fn constant(&self, c: f64) -> String {
        self.target.literal(c)
    }
    fn time(&self) -> String {
        "m->time".to_string()
    }
    fn input(&self, port: u32, elem: u32) -> R<String> {
        let flat = port_offset(self.in_sizes, port)? + elem as usize;
        // An external input (root interface) takes precedence over an internal
        // signal: it reads the FMU's `u[]` vector.
        if let Some(ext) = self.ext_input.get(flat).copied().flatten() {
            return Ok(format!("m->u[{ext}]"));
        }
        match self.input_src.get(flat).copied().flatten() {
            Some(sig) => Ok(format!("m->sig[{sig}]")),
            None => Ok(self.target.literal(0.0)),
        }
    }
    fn state(&self, id: u32) -> String {
        format!("m->x[{}]", self.state_base + id as usize)
    }
    fn param(&self, id: u32) -> String {
        format!("m->p[{}]", self.param_base + id as usize)
    }
    fn memory(&self, slot: usize, offset: u32) -> R<String> {
        let base = self
            .mem_off
            .get(slot)
            .copied()
            .ok_or_else(|| unsupported("struct: memory slot has no layout"))?;
        Ok(format!("m->mem[{}]", base + offset as usize))
    }
}

/// One named, addressable signal in the struct API (an `enum` entry + a get/set
/// case). States and outputs are readable; states and parameters settable.
#[derive(Serialize)]
struct SigName {
    name: String,
    id: usize,
    settable: bool,
}

/// The role of a [`LayoutVar`] in the struct model's addressable signal space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarKind {
    /// A continuous state (`m->x[]`): readable and settable.
    State,
    /// A block output signal (`m->sig[]`) that drives a root-interface output
    /// port (or any output in a closed model): an FMI `output`. Readable only.
    Output,
    /// A block output signal (`m->sig[]`) that is *not* a root-interface output:
    /// an FMI `local` (observable, but not part of the model's interface).
    Local,
    /// A tunable parameter (`m->p[]`): readable and settable.
    Param,
    /// An external input (`m->u[]`) of an open system (subsystem export):
    /// settable by the importer, not part of the `get_signal` id space. Its
    /// `signal_id` is the flat index into `u[]`.
    Input,
}

/// One addressable variable of the struct API: the same identity the generated
/// `<NAME>_SIG_<name>` enum and `*_get_signal`/`*_set_signal` accessors use. Its
/// `signal_id` is the id those accessors take. Consumers that need a stable
/// external addressing (e.g. the FMU exporter's value references) derive from
/// this so they stay consistent with the emitted C by construction.
#[derive(Debug, Clone)]
pub struct LayoutVar {
    pub name: String,
    pub signal_id: usize,
    pub kind: VarKind,
    /// Initial value seeded by `model_init` (states and params). `None` for
    /// outputs, which are computed.
    pub start: Option<f64>,
}

impl LayoutVar {
    pub fn settable(&self) -> bool {
        matches!(self.kind, VarKind::State | VarKind::Param)
    }
}

/// The addressable layout of a struct-API model: dimensions plus the named
/// variables (states, then outputs, then params, matching the generated
/// `get_signal` id ranges). Produced by [`struct_layout`] from the same plan
/// that drives emission, so a `signal_id` here is exactly the id the generated
/// `*_get_signal` / `*_set_signal` take.
#[derive(Debug, Clone)]
pub struct ModelLayout {
    pub name: String,
    pub n_state: usize,
    pub n_sig: usize,
    pub n_param: usize,
    pub vars: Vec<LayoutVar>,
    /// Number of external input elements (`m->u[]`); non-zero only for an open
    /// system (a subsystem exported with its interface inputs live).
    pub n_input: usize,
    /// Whether the model carries discrete events (periodic/fixed/zero-cross/
    /// condition). Consumers that only handle the continuous ODE (e.g. the FMU
    /// exporter's first phase) check this to reject event models up front.
    pub has_events: bool,
    /// Whether an analytic `model_jvp` (forward-mode ∂ẋ/∂x · seed) is emitted for
    /// this model. False if any op is non-differentiable (LUT, opaque call,
    /// `fmod`, min/max reduction). The FMU advertises directional derivatives
    /// only when this holds.
    pub jvp: bool,
}

impl ModelLayout {
    pub fn states(&self) -> impl Iterator<Item = &LayoutVar> {
        self.vars.iter().filter(|v| v.kind == VarKind::State)
    }
    pub fn outputs(&self) -> impl Iterator<Item = &LayoutVar> {
        self.vars.iter().filter(|v| v.kind == VarKind::Output)
    }
    /// Observable internal signals (`local` in FMI): block outputs that do not
    /// drive a root-interface output port.
    pub fn locals(&self) -> impl Iterator<Item = &LayoutVar> {
        self.vars.iter().filter(|v| v.kind == VarKind::Local)
    }
    pub fn params(&self) -> impl Iterator<Item = &LayoutVar> {
        self.vars.iter().filter(|v| v.kind == VarKind::Param)
    }
    /// External inputs in `u[]` order; their `signal_id` is the flat `u[]` index.
    pub fn inputs(&self) -> impl Iterator<Item = &LayoutVar> {
        self.vars.iter().filter(|v| v.kind == VarKind::Input)
    }
}

/// Compute the addressable variable layout for the struct API: states (ids
/// `0..n_state`), then block outputs (`n_state..n_state+n_sig`), then params
/// (`n_state+n_sig..`), with C-identifier names disambiguated exactly as the
/// emitted `<NAME>_SIG_*` enum. This is the single source `build_struct_ctx`
/// builds its `sigs` from, so the enum and any external map agree.
fn build_layout(plan: &Plan, name: &str) -> ModelLayout {
    let flat_params: Vec<Vec<f64>> = plan.blocks.iter().map(|b| flatten_params(b)).collect();
    let mut param_off = Vec::with_capacity(plan.blocks.len());
    let mut n_param = 0usize;
    for fp in &flat_params {
        param_off.push(n_param);
        n_param += fp.len();
    }
    let out_sizes: Vec<Vec<u32>> = plan
        .blocks
        .iter()
        .map(|b| b.ports.outputs.iter().map(|p| p.size).collect())
        .collect();

    let mut vars: Vec<LayoutVar> = Vec::new();
    let mut used = HashSet::new();
    let mut add = |base: String, signal_id: usize, kind: VarKind, start: Option<f64>| {
        let mut nm = base.clone();
        let mut k = 1;
        while !used.insert(nm.clone()) {
            nm = format!("{base}_{k}");
            k += 1;
        }
        vars.push(LayoutVar { name: nm, signal_id, kind, start });
    };

    // States first (ids 0..n_state), matching the `m->x[]` order.
    for (i, b) in plan.blocks.iter().enumerate() {
        if let Some(off) = plan.state_off[i] {
            let multi = b.state.len() > 1;
            for k in 0..b.state.len() {
                let id = c_ident(&plan.names[i]);
                let base = if multi { format!("{id}_x{k}") } else { id };
                add(base, off + k, VarKind::State, Some(b.state[k].init));
            }
        }
    }
    // Then block outputs (ids n_state..n_state+n_sig), matching `m->sig[]`. An
    // output that drives a root-interface output port (or any output in a closed
    // model) is an FMI `output`; the rest are `local` (observable, not interface).
    for (i, out_size) in out_sizes.iter().enumerate() {
        if let Some(ooff) = plan.out_off[i] {
            let n_out = total_elems(out_size);
            let multi = n_out > 1;
            for k in 0..n_out {
                let sig_idx = ooff + k;
                let kind = if !plan.has_root_interface || plan.iface_output_sigs.contains(&sig_idx) {
                    VarKind::Output
                } else {
                    VarKind::Local
                };
                let id = c_ident(&plan.names[i]);
                let base = if multi { format!("{id}_y{k}") } else { format!("{id}_y") };
                add(base, plan.n_state + sig_idx, kind, None);
            }
        }
    }
    // Then params (ids n_state+n_sig..), matching `m->p[]`.
    for (i, fp) in flat_params.iter().enumerate() {
        for k in 0..fp.len() {
            let base = format!("{}_p{k}", c_ident(&plan.names[i]));
            add(base, plan.n_state + plan.n_sig + param_off[i] + k, VarKind::Param, Some(fp[k]));
        }
    }
    // External inputs (open system): not in the signal id space; `signal_id` is
    // the flat `u[]` index. Names come from the root interface input ports.
    for (idx, nm) in plan.input_names.iter().enumerate() {
        add(c_ident(nm), idx, VarKind::Input, None);
    }

    ModelLayout {
        name: name.to_owned(),
        n_state: plan.n_state,
        n_sig: plan.n_sig,
        n_param,
        vars,
        n_input: plan.n_input,
        has_events: false,
        jvp: false,
    }
}

/// The addressable layout of a module compiled in the struct (`ModelApi::Struct`)
/// shape: the dimensions and named variables behind `*_get_signal`/`*_set_signal`.
/// Errors with the same `Unsupported` as the struct emitter for a stateless model
/// (the struct API needs continuous state). Lets a caller (e.g. the FMU exporter)
/// map external identities onto the generated C without re-deriving the layout.
pub fn struct_layout(module: &Module, _opts: &CodegenOptions) -> R<ModelLayout> {
    let plan = build_plan(flatten(module)?)?;
    if plan.n_state == 0 {
        return Err(unsupported("Struct API needs continuous state"));
    }
    let mut layout = build_layout(&plan, &c_ident(&module.name));
    layout.has_events = !collect_events(&plan)?.is_empty();
    layout.jvp = jvp_supported(&plan);
    Ok(layout)
}

/// One event of a struct-API model, as the export layer needs it: the `suffix`
/// that names its generated `guard_<suffix>` / `effect_<suffix>` functions and
/// `m->{next,fi,prev,init}_<suffix>` fields, plus the kind-specific schedule /
/// crossing data. State events (zero-cross, condition) are *event indicators*;
/// time events (periodic, fixed) drive `nextEventTime`.
#[derive(Debug, Clone)]
pub struct EventInfo {
    pub suffix: String,
    pub kind: EventKindInfo,
    /// Whether the effect writes a continuous state (`Write::StateWrite`). Lets
    /// the FMU report `valuesOfContinuousStatesChanged` precisely (a memory-only
    /// effect changes discrete state, not the continuous state).
    pub modifies_state: bool,
}

#[derive(Debug, Clone)]
pub enum EventKindInfo {
    Periodic { period: f64, phase: f64 },
    Fixed { times: Vec<f64> },
    ZeroCross { direction: Direction },
    Condition,
}

impl EventInfo {
    /// State events carry a guard `z(x, t)` an FMI host monitors for sign
    /// changes (an "event indicator"); time events do not.
    pub fn is_indicator(&self) -> bool {
        matches!(self.kind, EventKindInfo::ZeroCross { .. } | EventKindInfo::Condition)
    }
}

/// The events of a struct-API model, in emission order. Produced from the same
/// `collect_events` the struct emitter uses, so the suffixes match the generated
/// `guard_`/`effect_` functions and struct fields exactly.
#[derive(Debug, Clone, Default)]
pub struct EventLayout {
    pub events: Vec<EventInfo>,
}

impl EventLayout {
    pub fn indicators(&self) -> impl Iterator<Item = &EventInfo> {
        self.events.iter().filter(|e| e.is_indicator())
    }
    pub fn n_indicators(&self) -> usize {
        self.indicators().count()
    }
}

/// The event layout of a module compiled in the struct shape: kinds, suffixes
/// and schedule/crossing data for every emittable event. Errors the same way the
/// struct emitter does on an opaque event. Lets the FMU exporter build the FMI
/// event interface (indicators + discrete-state update) over the generated
/// `guard_`/`effect_` functions without re-deriving the events.
pub fn event_layout(module: &Module, _opts: &CodegenOptions) -> R<EventLayout> {
    let plan = build_plan(flatten(module)?)?;
    let evs = collect_events(&plan)?;
    let events = evs
        .iter()
        .map(|ev| EventInfo {
            suffix: ev.suffix(),
            kind: match ev {
                EventSpec::Periodic { period, phase, .. } => {
                    EventKindInfo::Periodic { period: *period, phase: *phase }
                }
                EventSpec::Fixed { times, .. } => EventKindInfo::Fixed { times: times.to_vec() },
                EventSpec::ZeroCross { direction, .. } => {
                    EventKindInfo::ZeroCross { direction: *direction }
                }
                EventSpec::Condition { .. } => EventKindInfo::Condition,
            },
            modifies_state: ev
                .effect()
                .writes
                .iter()
                .any(|w| matches!(w, Write::StateWrite { .. })),
        })
        .collect();
    Ok(EventLayout { events })
}

#[derive(Serialize)]
struct StructCtx {
    real: &'static str,
    eq_tol: String,
    /// The `fastsim_rand_uniform` C helper (see `codegen::RNG_HELPER_C`).
    rng_helper: &'static str,
    /// The `fastsim_digamma` C helper (see `codegen::DIGAMMA_HELPER_C`).
    digamma_helper: &'static str,
    name: String,
    /// Base name of the emitted files (`<file_base>.h` etc.); the templates use
    /// it for the internal `#include`s so they always match the actual names.
    file_base: String,
    n_state: usize,
    n_sig: usize,
    n_param: usize,
    n_mem: usize,
    n_input: usize,
    has_sig: bool,
    has_param: bool,
    has_mem: bool,
    has_input: bool,
    /// No continuous state: model_run is a pure time/event stepper, no deriv.
    is_discrete: bool,
    /// Library layout: per-block functions move to blocks.c, the integrator to
    /// solver.c, and the driver functions get external (not static) linkage.
    library: bool,
    /// Per-block `blk_i_alg` / `blk_i_deriv` functions (Hierarchical; empty Flat).
    block_fns: String,
    /// External prototypes for the per-block functions (Library; for blocks.h).
    block_protos: String,
    outputs_body: String,
    deriv_body: String,
    init_body: String,
    sigs: Vec<SigName>,
    /// The generated continuous integrator (stage kernel + `<name>_step` +
    /// `model_run`), emitted from the chosen Butcher tableau. Empty when the model
    /// is pure-discrete (the template emits the time/event driver instead).
    solver_body: String,
    /// Extra `model_t` fields the integrator needs (adaptive solvers carry `fs_h`).
    solver_fields: Vec<String>,
    /// `0.5 * dt`, numeric-aware (`(dt >> 1)` under fixed point) — the step
    /// tolerance used by the event schedulers and the discrete run loop.
    half_dt: String,
    /// Fractional bits under fixed point (`None` for double/float): drives
    /// the `<NAME>_Q_*` conversion macros in the emitted header.
    fixed_frac: Option<u8>,
    has_events: bool,
    need_sig_handle: bool,
    event_fns: String,
    event_fields: Vec<String>,
    events: Vec<StructEventCtx>,
    /// Whether an analytic `model_jvp` (forward-mode directional derivative
    /// ∂ẋ/∂x · seed) was emitted. False if any op is non-differentiable.
    has_jvp: bool,
    /// The `model_jvp` body (empty when `has_jvp` is false).
    jvp_body: String,
}

/// Base file name for the generated sources: the sanitized model name, or
/// `model` when the name is empty/degenerate. The emitted files are
/// `<base>.h` / `<base>.c` (+ `<base>_blocks.*` / `<base>_solver.*` under the
/// Library layout) and every internal `#include` uses these names, so two
/// generated models can live in ONE build directory without file collisions —
/// the file-level counterpart of the model-name symbol prefixing (issue #34).
pub fn file_base(model_name: &str) -> String {
    let s = c_ident(model_name);
    if s.is_empty() || s == "_" { "model".to_string() } else { s }
}

/// Sanitize a (possibly path-qualified) block name into a C identifier.
fn c_ident(name: &str) -> String {
    let mut s: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if s.is_empty() || s.as_bytes()[0].is_ascii_digit() {
        s.insert(0, '_');
    }
    s
}

/// Lower one region in struct mode into the running statement body, returning the
/// local-node -> global-temp map. Uses [`lower_one`], so LUT/vectorized ops work.
fn lower_struct_region(
    plan: &Plan,
    target: &CTarget,
    block_i: usize,
    param_off: &[usize],
    region: &Region,
    g: &mut usize,
    body: &mut String,
) -> R<Vec<usize>> {
    let mut l2g = vec![0usize; region.ops.len()];
    for (local, op) in region.ops.iter().enumerate() {
        let id = *g;
        let leaves = StructLeaves {
            target,
            l2g: &l2g[..local],
            input_src: &plan.input_src[block_i],
            ext_input: &plan.ext_input[block_i],
            in_sizes: &plan.in_sizes[block_i],
            state_base: plan.state_off[block_i].unwrap_or(0),
            param_base: param_off[block_i],
            mem_off: &plan.mem_slot_off[block_i],
        };
        let stmt = lower_one(op, id, target, &leaves)?;
        match stmt.block {
            Some(blk) => body.push_str(&blk),
            None => body.push_str(&format!("    const {} v{id} = {};\n", target.scalar_ty(), stmt.expr)),
        }
        l2g[local] = id;
        *g += 1;
    }
    Ok(l2g)
}

// ======================================================================================
// Analytic directional derivative (forward-mode AD) for the struct model
// ======================================================================================
//
// `model_jvp(m, seed, d_dxdt)` computes the Jacobian-vector product ∂ẋ/∂x · seed
// exactly, by forward-mode AD: it recomputes the alg + dyn passes and, alongside
// each primal value `vN`, a tangent `dN` carrying the directional derivative.
// State leaves seed their tangent from `seed[]`; signal inputs from a local
// `d_sig[]`; everything else is constant (tangent 0). The derivative rules below
// mirror `ssa::autodiff`, but emit C directly (verified against finite
// differences in the codegen tests), so the FMU ships an exact Jacobian a host's
// implicit integrator can use.

/// Whether an op can take part in the forward-AD JVP. Only a genuinely opaque
/// extern call (RNG/noise/DAE/Python: no known derivative) disqualifies the
/// model. Everything else — including `fmod`, min/max reductions, and `Lut1d`
/// (piecewise-linear) — has a well-defined a.e. tangent emitted below.
fn op_jvp_supported(op: &Op) -> bool {
    !matches!(op, Op::Call { .. })
}

/// Whether every op in the model's alg + dyn regions supports the JVP, so an
/// analytic `model_jvp` can be emitted. Shared by `struct_layout` (sets the
/// capability flag) and `build_struct_ctx` (decides emission), so the two agree.
fn jvp_supported(plan: &Plan) -> bool {
    plan.blocks.iter().all(|b| {
        b.regions.alg.ops.iter().all(op_jvp_supported)
            && b.regions.dyn_.ops.iter().all(op_jvp_supported)
    })
}

/// Tangent leaf resolution: the forward-AD twin of [`StructLeaves`]. A state's
/// tangent is its `seed[]` entry; a signal input's tangent is the producing
/// block's `d_sig[]` entry; constants/params/time/memory have tangent 0.
struct TangentLeaves<'a> {
    l2g: &'a [usize],
    input_src: &'a [Option<usize>],
    ext_input: &'a [Option<usize>],
    in_sizes: &'a [u32],
    state_base: usize,
    param_base: usize,
    zero: String,
}

impl TangentLeaves<'_> {
    /// Tangent temp `dN` of an earlier node (same numbering as its primal `vN`).
    fn temp(&self, node: NodeId) -> String {
        format!("d{}", self.l2g[node.0 as usize])
    }
    fn state(&self, id: u32) -> String {
        format!("x_seed[{}]", self.state_base + id as usize)
    }
    fn param(&self, id: u32) -> String {
        format!("p_seed[{}]", self.param_base + id as usize)
    }
    fn input(&self, port: u32, elem: u32) -> R<String> {
        let flat = port_offset(self.in_sizes, port)? + elem as usize;
        // An external input's tangent is its `u_seed[]` entry; an internal
        // signal's is the producing block's `d_sig[]` entry.
        if let Some(ext) = self.ext_input.get(flat).copied().flatten() {
            return Ok(format!("u_seed[{ext}]"));
        }
        match self.input_src.get(flat).copied().flatten() {
            Some(sig) => Ok(format!("d_sig[{sig}]")),
            None => Ok(self.zero.clone()),
        }
    }
}

/// Tangent of a binary op via the product/quotient/chain rules. `vid` is the
/// op's own primal temp (`vN`), reused where the derivative references the
/// result (pow, hypot). Mirrors `ssa::autodiff`.
fn tangent_binary(
    t: &CTarget,
    op: BinOpKind,
    va: &str,
    vb: &str,
    da: &str,
    db: &str,
    vid: &str,
) -> R<String> {
    use BinOpKind as B;
    Ok(match op {
        B::Add => t.binary(B::Add, da, db),
        B::Sub => t.binary(B::Sub, da, db),
        // d(a*b) = a*db + b*da
        B::Mul => t.binary(B::Add, &t.binary(B::Mul, va, db), &t.binary(B::Mul, vb, da)),
        // d(a/b) = (da - (a/b)*db) / b   ((a/b) == vid)
        B::Div => t.binary(B::Div, &t.binary(B::Sub, da, &t.binary(B::Mul, vid, db)), vb),
        // d(a^b) = a^b * (db*ln(a) + b*da/a)
        B::Pow => {
            let l = t.binary(B::Mul, db, &t.unary(UnaryOpKind::Log, va)?);
            let r = t.binary(B::Div, &t.binary(B::Mul, vb, da), va);
            t.binary(B::Mul, vid, &t.binary(B::Add, &l, &r))
        }
        // d(hypot(a,b)) = (a*da + b*db) / hypot(a,b)   (hypot == vid)
        B::Hypot => {
            let num = t.binary(B::Add, &t.binary(B::Mul, va, da), &t.binary(B::Mul, vb, db));
            t.binary(B::Div, &num, vid)
        }
        // d(atan2(a,b)) = (b*da - a*db) / (a^2 + b^2)
        B::Atan2 => {
            let num = t.binary(B::Sub, &t.binary(B::Mul, vb, da), &t.binary(B::Mul, va, db));
            let den = t.binary(B::Add, &t.binary(B::Mul, va, va), &t.binary(B::Mul, vb, vb));
            t.binary(B::Div, &num, &den)
        }
        // min/max pass the tangent of whichever operand is selected.
        B::Min => format!("({va} <= {vb} ? {da} : {db})"),
        B::Max => format!("({va} >= {vb} ? {da} : {db})"),
        // fmod(a,b) = a - trunc(a/b)*b, so a.e. d = da - trunc(a/b)*db
        // (the integer quotient trunc(a/b) is locally constant).
        B::Mod => {
            let q = t.unary(UnaryOpKind::Trunc, &t.binary(B::Div, va, vb))?;
            t.binary(B::Sub, da, &t.binary(B::Mul, &q, db))
        }
    })
}

/// Tangent of a unary op: `f'(a) * da`, mirroring `ssa::autodiff`. `vid` is the
/// op's own primal temp, reused where `f'` is cheap in terms of `f(a)` (exp,
/// sqrt, tanh, tan, expm1, cbrt, tgamma). Zero-derivative ops return `0`.
fn tangent_unary(t: &CTarget, op: UnaryOpKind, va: &str, da: &str, vid: &str) -> R<String> {
    use UnaryOpKind as U;
    let lit = |x: f64| t.literal(x);
    let mul = |a: &str, b: &str| t.binary(BinOpKind::Mul, a, b);
    let div = |a: &str, b: &str| t.binary(BinOpKind::Div, a, b);
    let add = |a: &str, b: &str| t.binary(BinOpKind::Add, a, b);
    let sub = |a: &str, b: &str| t.binary(BinOpKind::Sub, a, b);
    let un = |o: U, a: &str| t.unary(o, a);
    Ok(match op {
        U::Neg => t.unary(U::Neg, da)?,
        U::Sin => mul(&un(U::Cos, va)?, da),
        U::Cos => mul(&t.unary(U::Neg, &un(U::Sin, va)?)?, da),
        // (1 + tan^2) * da  (tan == vid)
        U::Tan => mul(&add(&lit(1.0), &mul(vid, vid)), da),
        U::Atan => div(da, &add(&lit(1.0), &mul(va, va))),
        U::Sinh => mul(&un(U::Cosh, va)?, da),
        U::Cosh => mul(&un(U::Sinh, va)?, da),
        // (1 - tanh^2) * da  (tanh == vid)
        U::Tanh => mul(&sub(&lit(1.0), &mul(vid, vid)), da),
        U::Exp => mul(vid, da),                       // exp(a) == vid
        U::Log => div(da, va),
        U::Log10 => div(da, &mul(va, &lit(std::f64::consts::LN_10))),
        U::Log2 => div(da, &mul(va, &lit(std::f64::consts::LN_2))),
        U::Log1p => div(da, &add(&lit(1.0), va)),
        U::Abs => mul(&un(U::Sign, va)?, da),
        U::Sqrt => div(da, &mul(&lit(2.0), vid)),     // 2*sqrt(a) == 2*vid
        U::Asin => div(da, &un(U::Sqrt, &sub(&lit(1.0), &mul(va, va)))?),
        U::Acos => t.unary(U::Neg, &div(da, &un(U::Sqrt, &sub(&lit(1.0), &mul(va, va)))?))?,
        U::Asinh => div(da, &un(U::Sqrt, &add(&lit(1.0), &mul(va, va)))?),
        U::Acosh => div(da, &un(U::Sqrt, &sub(&mul(va, va), &lit(1.0)))?),
        U::Atanh => div(da, &sub(&lit(1.0), &mul(va, va))),
        U::Expm1 => mul(&add(vid, &lit(1.0)), da),    // exp(a) = expm1(a)+1
        U::Cbrt => div(da, &mul(&lit(3.0), &mul(vid, vid))), // 3*cbrt(a)^2
        // (2/sqrt(pi)) * exp(-a^2) * da
        U::Erf => {
            let pref = mul(&lit(2.0 / std::f64::consts::PI.sqrt()), &un(U::Exp, &t.unary(U::Neg, &mul(va, va))?)?);
            mul(&pref, da)
        }
        U::Erfc => {
            let pref = mul(&lit(-2.0 / std::f64::consts::PI.sqrt()), &un(U::Exp, &t.unary(U::Neg, &mul(va, va))?)?);
            mul(&pref, da)
        }
        U::Lgamma => mul(&un(U::Digamma, va)?, da),
        U::Tgamma => mul(&mul(vid, &un(U::Digamma, va)?), da),
        // Piecewise-constant or non-differentiable: tangent 0.
        U::Sign | U::Floor | U::Ceil | U::Round | U::Trunc | U::RandUniform | U::Digamma => lit(0.0),
    })
}

/// Tangent of one op, given the primal leaves (for operand primals and the op's
/// own `vN`) and the tangent leaves (for operand tangents). Returns the C rvalue
/// for `dN`.
fn tangent_expr(
    op: &Op,
    id: usize,
    t: &CTarget,
    primal: &StructLeaves,
    tan: &TangentLeaves,
) -> R<String> {
    let vp = |n: &NodeId| primal.temp(*n);
    let vt = |n: &NodeId| tan.temp(*n);
    let vid = t.temp(id as u32);
    Ok(match op {
        Op::Const(_) | Op::Time | Op::Memory { .. } => tan.zero.clone(),
        Op::Param { id } => tan.param(id.0),
        Op::State { id } => tan.state(id.0),
        Op::Input { port, elem } => tan.input(*port, *elem)?,
        Op::Binary { op, a, b } => tangent_binary(t, *op, &vp(a), &vp(b), &vt(a), &vt(b), &vid)?,
        Op::Unary { op, a } => tangent_unary(t, *op, &vp(a), &vt(a), &vid)?,
        Op::Cmp { .. } => tan.zero.clone(),
        // d(c ? a : b) = c ? da : db
        Op::Select { c, t: th, e } => format!("({} != {} ? {} : {})", vp(c), t.literal(0.0), vt(th), vt(e)),
        // d(a*b + c) = a*db + b*da + dc
        Op::Fma { a, b, c } => {
            let ab = t.binary(BinOpKind::Add, &t.binary(BinOpKind::Mul, &vp(a), &vt(b)), &t.binary(BinOpKind::Mul, &vp(b), &vt(a)));
            t.binary(BinOpKind::Add, &ab, &vt(c))
        }
        Op::Reduce { op, args } => match op {
            // d(Σ aᵢ) = Σ daᵢ
            ReduceKind::Sum => t.reduce(ReduceKind::Sum, &args.iter().map(&vt).collect::<Vec<_>>()),
            // d(Π aᵢ) = Σⱼ (daⱼ · Πᵢ≠ⱼ aᵢ)
            ReduceKind::Product => {
                if args.is_empty() {
                    t.literal(0.0)
                } else {
                    let terms: Vec<String> = (0..args.len())
                        .map(|j| {
                            let others: Vec<String> = args
                                .iter()
                                .enumerate()
                                .filter(|(i, _)| *i != j)
                                .map(|(_, n)| vp(n))
                                .collect();
                            let prod = t.reduce(ReduceKind::Product, &others);
                            t.binary(BinOpKind::Mul, &vt(&args[j]), &prod)
                        })
                        .collect();
                    t.reduce(ReduceKind::Sum, &terms)
                }
            }
            // Subgradient of the running extremum: the tangent flows from the
            // operand currently selected (mirrors `ssa::autodiff`).
            ReduceKind::Min | ReduceKind::Max => {
                if args.is_empty() {
                    t.literal(0.0)
                } else {
                    let is_min = matches!(op, ReduceKind::Min);
                    let cmp = if is_min { "<" } else { ">" };
                    let ext = if is_min { BinOpKind::Min } else { BinOpKind::Max };
                    let mut running = vp(&args[0]);
                    let mut d = vt(&args[0]);
                    for ak in &args[1..] {
                        let (pak, dak) = (vp(ak), vt(ak));
                        d = format!("({pak} {cmp} {running} ? {dak} : {d})");
                        running = t.binary(ext, &running, &pak);
                    }
                    d
                }
            }
        },
        // d(Σ aᵢ·bᵢ) = Σ (aᵢ·dbᵢ + daᵢ·bᵢ)
        Op::Dot { a, b } => {
            let terms: Vec<String> = a
                .iter()
                .zip(b.iter())
                .map(|(ai, bi)| {
                    t.binary(BinOpKind::Add, &t.binary(BinOpKind::Mul, &vp(ai), &vt(bi)), &t.binary(BinOpKind::Mul, &vt(ai), &vp(bi)))
                })
                .collect();
            t.reduce(ReduceKind::Sum, &terms)
        }
        Op::Lut1d { .. } => return Err(unsupported("JVP of Lut1d")),
        Op::Call { .. } => return Err(unsupported("JVP of opaque Call")),
    })
}

/// Lower one region for the JVP: emit each op's primal `vN` (reusing the struct
/// leaves) and its tangent `dN` (forward-AD). Returns the local→global map. Uses
/// an unrolled target so reductions stay plain `vN` temps the tangent can read.
fn lower_struct_region_jvp(
    plan: &Plan,
    target: &CTarget,
    block_i: usize,
    param_off: &[usize],
    region: &Region,
    g: &mut usize,
    body: &mut String,
) -> R<Vec<usize>> {
    let mut l2g = vec![0usize; region.ops.len()];
    let state_base = plan.state_off[block_i].unwrap_or(0);
    for (local, op) in region.ops.iter().enumerate() {
        let id = *g;
        let primal = StructLeaves {
            target,
            l2g: &l2g[..local],
            input_src: &plan.input_src[block_i],
            ext_input: &plan.ext_input[block_i],
            in_sizes: &plan.in_sizes[block_i],
            state_base,
            param_base: param_off[block_i],
            mem_off: &plan.mem_slot_off[block_i],
        };
        let tan = TangentLeaves {
            l2g: &l2g[..local],
            input_src: &plan.input_src[block_i],
            ext_input: &plan.ext_input[block_i],
            in_sizes: &plan.in_sizes[block_i],
            state_base,
            param_base: param_off[block_i],
            zero: target.literal(0.0),
        };
        let stmt = lower_one(op, id, target, &primal)?;
        match stmt.block {
            Some(blk) => body.push_str(&blk),
            None => body.push_str(&format!("    const {} v{id} = {};\n", target.scalar_ty(), stmt.expr)),
        }
        // A LUT emits a block for both primal and tangent (segment search +
        // slope); every other op's tangent is a single rvalue.
        if let Op::Lut1d { input, points, values, clamp } = op {
            let blk = target.lut1d_tangent_block(
                id,
                &primal.temp(*input),
                &tan.temp(*input),
                points,
                values,
                *clamp,
            )?;
            body.push_str(&blk);
        } else {
            let dexpr = tangent_expr(op, id, target, &primal, &tan)?;
            body.push_str(&format!("    const {} d{id} = {};\n", target.scalar_ty(), dexpr));
        }
        l2g[local] = id;
        *g += 1;
    }
    Ok(l2g)
}

/// Build the `model_jvp` body: forward-mode AD over the alg pass (filling a local
/// `d_sig[]`) then the dyn pass (filling `d_dxdt[]`). Returns `None` if the model
/// has a non-differentiable op (the caller then omits the function and the
/// capability). Uses `Reductions::Unrolled` so operand primals are plain temps.
fn build_struct_jvp(plan: &Plan, opts: &CodegenOptions, param_off: &[usize], model_name: &str) -> R<Option<String>> {
    // No continuous state -> no ∂ẋ/∂x to emit (pure-discrete model).
    if plan.n_state == 0 || !jvp_supported(plan) {
        return Ok(None);
    }
    let target = CTarget::new(&CodegenOptions { reductions: Reductions::Unrolled, ..opts.clone() })?;
    let out_sizes: Vec<Vec<u32>> = plan
        .blocks
        .iter()
        .map(|b| b.ports.outputs.iter().map(|p| p.size).collect())
        .collect();

    let mut body = String::new();
    // Refresh primal signals once; the tangent pass recomputes primal temps it
    // needs locally for derivative coefficients. `d_sig` / `d_dxdt` are caller-
    // provided output buffers (so output sensitivities are observable too).
    if plan.n_sig > 0 {
        body.push_str(&format!("    {model_name}_outputs(m);\n"));
    }

    let mut g = 0usize;
    // Tangent alg pass: d_sig[].
    for &i in &plan.topo {
        let alg = &plan.blocks[i].regions.alg;
        let (Some(ooff), false) = (plan.out_off[i], alg.is_empty()) else { continue };
        body.push_str(&format!("    /* {} (jvp) */\n", plan.names[i]));
        let l2g = lower_struct_region_jvp(plan, &target, i, param_off, alg, &mut g, &mut body)?;
        for w in &alg.writes {
            if let Write::Output { port, elem, src } = w {
                let sig = ooff + port_offset(&out_sizes[i], *port)? + *elem as usize;
                body.push_str(&format!("    d_sig[{sig}] = d{};\n", l2g[src.0 as usize]));
            }
        }
    }
    // Tangent dyn pass: d_dxdt[].
    for (i, b) in plan.blocks.iter().enumerate() {
        let Some(soff) = plan.state_off[i] else { continue };
        if b.regions.dyn_.is_empty() {
            continue;
        }
        body.push_str(&format!("    /* d/dt {} (jvp) */\n", plan.names[i]));
        let l2g = lower_struct_region_jvp(plan, &target, i, param_off, &b.regions.dyn_, &mut g, &mut body)?;
        for w in &b.regions.dyn_.writes {
            if let Write::StateDeriv { id, src } = w {
                body.push_str(&format!("    d_dxdt[{}] = d{};\n", soff + id.idx(), l2g[src.0 as usize]));
            }
        }
    }
    Ok(Some(body))
}


/// Build the shared render context for the struct API (drives both
/// `model_struct.c` and `model_struct.h`): signal/state/param/memory layout,
/// the lowered `model_outputs` / `model_deriv` / `model_init` bodies, the named
/// addressable signals, and the event functions/fields.
fn build_struct_ctx(module: &Module, opts: &CodegenOptions) -> R<StructCtx> {
    let plan = build_plan(flatten(module)?)?;
    let events = collect_events(&plan)?;
    // n_state == 0 is allowed: a pure-discrete model (memory + events, no
    // continuous integration) emits a time-stepping model_run and no deriv.
    let target = CTarget::new(opts)?;
    let lit = |x: f64| fmt_lit(x, opts.numeric);
    let model_name = c_ident(&module.name);

    // The addressable variable layout (states, outputs, params) is the single
    // source for the `sigs` enum below *and* for any external map (FMU export),
    // so the generated C and that map agree by construction.
    let layout = build_layout(&plan, &model_name);

    // Parameter layout: concatenate every block's flattened params into m->p[].
    let flat_params: Vec<Vec<f64>> = plan.blocks.iter().map(|b| flatten_params(b)).collect();
    let mut param_off = Vec::with_capacity(plan.blocks.len());
    let mut n_param = 0usize;
    for fp in &flat_params {
        param_off.push(n_param);
        n_param += fp.len();
    }
    let out_sizes: Vec<Vec<u32>> = plan
        .blocks
        .iter()
        .map(|b| b.ports.outputs.iter().map(|p| p.size).collect())
        .collect();

    // Hierarchical emits one `blk_i_alg` / `blk_i_deriv` per block (taking
    // `model_t* m`); model_outputs / model_deriv then call them in topological
    // order. Flat fuses every block into the two driver functions (one shared
    // temp numbering). `block_fns` holds the per-block functions (Hierarchical).
    let hierarchical = opts.structure == Structure::Hierarchical;
    let library = opts.layout == Layout::Library;
    let real = opts.numeric.real();
    // Library moves the per-block functions to blocks.c (external linkage, with
    // prototypes in blocks.h); Compact keeps them file-static inside model.c.
    let blk_linkage = if library { "" } else { "static " };
    let mut block_fns = String::new();
    let mut block_protos = String::new();

    // model_outputs body: alg pass in topological order, storing m->sig[].
    let mut outputs_body = String::new();
    let mut g = 0usize;
    for &i in &plan.topo {
        let alg = &plan.blocks[i].regions.alg;
        let (Some(ooff), false) = (plan.out_off[i], alg.is_empty()) else { continue };
        if hierarchical {
            let mut fb = String::new();
            let mut gi = 0usize;
            let l2g = lower_struct_region(&plan, &target, i, &param_off, alg, &mut gi, &mut fb)?;
            for w in &alg.writes {
                if let Write::Output { port, elem, src } = w {
                    let sig = ooff + port_offset(&out_sizes[i], *port)? + *elem as usize;
                    fb.push_str(&format!("    m->sig[{sig}] = v{};\n", l2g[src.0 as usize]));
                }
            }
            let sig = format!("void {model_name}_blk_{i}_alg({model_name}_t * restrict m)");
            block_fns.push_str(&format!("/* alg of {} */\n{blk_linkage}{sig} {{\n{fb}}}\n\n", plan.names[i]));
            if library {
                block_protos.push_str(&format!("/* alg of {} */\n{sig};\n", plan.names[i]));
            }
            outputs_body.push_str(&format!("    {model_name}_blk_{i}_alg(m);\n"));
        } else {
            outputs_body.push_str(&format!("    /* {} */\n", plan.names[i]));
            let l2g = lower_struct_region(&plan, &target, i, &param_off, alg, &mut g, &mut outputs_body)?;
            for w in &alg.writes {
                if let Write::Output { port, elem, src } = w {
                    let sig = ooff + port_offset(&out_sizes[i], *port)? + *elem as usize;
                    outputs_body.push_str(&format!("    m->sig[{sig}] = v{};\n", l2g[src.0 as usize]));
                }
            }
        }
    }

    // model_deriv body: refresh signals, then each dynamic block's dxdt.
    let mut deriv_body = format!("    {model_name}_outputs(m);\n");
    let mut g = 0usize;
    for (i, b) in plan.blocks.iter().enumerate() {
        let Some(soff) = plan.state_off[i] else { continue };
        if b.regions.dyn_.is_empty() {
            continue;
        }
        if hierarchical {
            let mut fb = String::new();
            let mut gi = 0usize;
            let l2g = lower_struct_region(&plan, &target, i, &param_off, &b.regions.dyn_, &mut gi, &mut fb)?;
            for w in &b.regions.dyn_.writes {
                if let Write::StateDeriv { id, src } = w {
                    fb.push_str(&format!("    dxdt[{}] = v{};\n", soff + id.idx(), l2g[src.0 as usize]));
                }
            }
            let sig = format!("void {model_name}_blk_{i}_deriv({model_name}_t * restrict m, {real}* restrict dxdt)");
            block_fns.push_str(&format!("/* d/dt {} */\n{blk_linkage}{sig} {{\n{fb}}}\n\n", plan.names[i]));
            if library {
                block_protos.push_str(&format!("/* d/dt {} */\n{sig};\n", plan.names[i]));
            }
            deriv_body.push_str(&format!("    {model_name}_blk_{i}_deriv(m, dxdt);\n"));
        } else {
            deriv_body.push_str(&format!("    /* d/dt {} */\n", plan.names[i]));
            let l2g = lower_struct_region(&plan, &target, i, &param_off, &b.regions.dyn_, &mut g, &mut deriv_body)?;
            for w in &b.regions.dyn_.writes {
                if let Write::StateDeriv { id, src } = w {
                    deriv_body.push_str(&format!("    dxdt[{}] = v{};\n", soff + id.idx(), l2g[src.0 as usize]));
                }
            }
        }
    }

    // model_jvp body: analytic forward-mode directional derivative ∂ẋ/∂x · seed,
    // or none if the model has a non-differentiable op.
    let jvp_body = build_struct_jvp(&plan, opts, &param_off, &model_name)?;

    // model_init body: time, states, params, memory.
    let mut init_body = format!("    m->time = {};\n", lit(0.0));
    for (i, b) in plan.blocks.iter().enumerate() {
        if let Some(off) = plan.state_off[i] {
            for (k, sv) in b.state.iter().enumerate() {
                init_body.push_str(&format!("    m->x[{}] = {};\n", off + k, lit(sv.init)));
            }
        }
    }
    for (i, fp) in flat_params.iter().enumerate() {
        for (k, v) in fp.iter().enumerate() {
            init_body.push_str(&format!("    m->p[{}] = {};\n", param_off[i] + k, lit(*v)));
        }
    }
    for (i, b) in plan.blocks.iter().enumerate() {
        for slot in &b.memory {
            let base = plan.mem_slot_off[i][slot.id.0 as usize];
            for (e, v) in slot.init.iter().enumerate() {
                init_body.push_str(&format!("    m->mem[{}] = {};\n", base + e, lit(*v)));
            }
        }
    }

    // Named, addressable signals: derived from `layout` (states, then outputs,
    // then params) so the `<NAME>_SIG_*` enum matches the layout id-for-id.
    // External inputs are addressed through `m->u[]`, not the signal id space, so
    // they are excluded from the `get_signal`/`set_signal` enum.
    let sigs: Vec<SigName> = layout
        .vars
        .iter()
        .filter(|v| v.kind != VarKind::Input)
        .map(|v| SigName { name: v.name.clone(), id: v.signal_id, settable: v.settable() })
        .collect();

    // Events: effect/guard functions, per-event template contexts, struct
    // counter fields, and their init lines (appended to model_init).
    let ev = build_struct_events(&plan, &target, &param_off, &model_name, &events, &lit)?;
    for line in &ev.inits {
        init_body.push_str(line);
        init_body.push('\n');
    }

    // Continuous integrator: emit the stage kernel + `<name>_step` + `<name>_run` from
    // the chosen Butcher tableau (the runtime's own registry). Only relevant when
    // there is continuous state; the pure-discrete driver lives in the template.
    let (solver_body, solver_fields) = if plan.n_state > 0 {
        let t = opts.solver.tableau();
        let scx = solver::SolverCtx {
            name: &model_name,
            n_state: plan.n_state,
            real: opts.numeric.real(),
            numeric: opts.numeric,
            has_events: !events.is_empty(),
            has_sig: plan.n_sig > 0,
        };
        init_body.push_str(&solver::init_body(t));
        (solver::emit(t, &scx)?, solver::struct_fields(t, opts.numeric.real()))
    } else {
        (String::new(), Vec::new())
    };

    let ctx = StructCtx {
        real: opts.numeric.real(),
        eq_tol: fmt_lit(crate::constants::JIT_FLOAT_EQ_TOL, Numeric::Double),
        rng_helper: crate::codegen::RNG_HELPER_C,
        digamma_helper: crate::codegen::DIGAMMA_HELPER_C,
        file_base: file_base(&module.name),
        name: model_name,
        n_state: plan.n_state,
        n_sig: plan.n_sig,
        n_param,
        n_mem: plan.n_mem,
        n_input: plan.n_input,
        has_sig: plan.n_sig > 0,
        has_param: n_param > 0,
        has_mem: plan.n_mem > 0,
        has_input: plan.n_input > 0,
        is_discrete: plan.n_state == 0,
        library,
        block_fns,
        block_protos,
        outputs_body,
        deriv_body,
        init_body,
        sigs,
        solver_body,
        solver_fields,
        fixed_frac: opts.numeric.frac(),
        half_dt: match opts.numeric.frac() {
            Some(_) => "(dt >> 1)".to_string(),
            None => format!("{} * dt", lit(0.5)),
        },
        has_events: !events.is_empty(),
        need_sig_handle: ev.need_sig,
        event_fns: ev.fns,
        event_fields: ev.fields,
        events: ev.ctxs,
        has_jvp: jvp_body.is_some(),
        jvp_body: jvp_body.unwrap_or_default(),
    };
    Ok(ctx)
}

/// One event, ready for the struct-mode `model_handle_events` template loop. The
/// counters (`next`/`fi`/`prev`/`init`) live in the model struct (`m->...`); the
/// effect and guard are emitted as functions taking `m`.
#[derive(Serialize)]
struct StructEventCtx {
    kind: &'static str,
    suffix: String,
    phase: String,
    period: String,
    times: Vec<String>,
    /// Zero-cross firing test over `cur_<suffix>` / `m->prev_<suffix>`.
    cross_cond: String,
}

/// Emit a struct-mode event-effect function `effect_<suffix>(model_t* m)`: it
/// reads/writes the instance through `m->` and stores into `m->mem` / `m->x`.
fn emit_struct_effect(
    model: &str,
    suffix: &str,
    region: &Region,
    block_i: usize,
    plan: &Plan,
    param_off: &[usize],
    target: &CTarget,
) -> R<String> {
    let mut body = String::new();
    let mut g = 0usize;
    let l2g = lower_struct_region(plan, target, block_i, param_off, region, &mut g, &mut body)?;
    let mem_off = &plan.mem_slot_off[block_i];
    let state_base = plan.state_off[block_i].unwrap_or(0);
    for w in &region.writes {
        match w {
            Write::MemoryWrite { slot, offset, src } => {
                let base = mem_off
                    .get(slot.idx())
                    .copied()
                    .ok_or_else(|| unsupported("struct effect: memory slot has no layout"))?;
                body.push_str(&format!("    m->mem[{}] = v{};\n", base + *offset as usize, l2g[src.0 as usize]));
            }
            Write::StateWrite { id, src } => {
                body.push_str(&format!("    m->x[{}] = v{};\n", state_base + id.idx(), l2g[src.0 as usize]));
            }
            _ => return Err(unsupported("struct event effect writes an output (only memory/state)")),
        }
    }
    Ok(format!("static void effect_{suffix}({model}_t * restrict m) {{\n{body}}}\n"))
}

/// Emit a struct-mode event-guard function `guard_<suffix>(const model_t* m)`
/// returning the scalar test value (the last op of the guard region).
fn emit_struct_guard(
    model: &str,
    suffix: &str,
    region: &Region,
    block_i: usize,
    plan: &Plan,
    param_off: &[usize],
    target: &CTarget,
) -> R<String> {
    let mut body = String::new();
    let mut g = 0usize;
    let l2g = lower_struct_region(plan, target, block_i, param_off, region, &mut g, &mut body)?;
    let result = l2g.last().copied().ok_or_else(|| unsupported("struct event guard has no ops"))?;
    Ok(format!(
        "static {} guard_{suffix}(const {model}_t * restrict m) {{\n{body}    return v{result};\n}}\n",
        target.scalar_ty()
    ))
}

/// Build everything the struct-mode event handling needs: the effect/guard
/// functions, the per-event template contexts, the struct counter fields, the
/// `model_init` seeding lines, and whether a signal refresh is needed.
struct StructEvents {
    fns: String,
    ctxs: Vec<StructEventCtx>,
    fields: Vec<String>,
    inits: Vec<String>,
    need_sig: bool,
}

fn build_struct_events(
    plan: &Plan,
    target: &CTarget,
    param_off: &[usize],
    model: &str,
    events: &[EventSpec],
    lit: &impl Fn(f64) -> String,
) -> R<StructEvents> {
    let t = target.scalar_ty();
    let mut fns = String::new();
    let mut ctxs = Vec::new();
    let mut fields = Vec::new();
    let mut inits = Vec::new();
    let mut need_sig = false;
    for ev in events {
        let suffix = ev.suffix();
        let block = ev.block();
        need_sig |= event_reads_input(ev);
        fns.push_str(&emit_struct_effect(model, &suffix, ev.effect(), block, plan, param_off, target)?);
        if let Some(guard) = ev.guard() {
            fns.push_str(&emit_struct_guard(model, &suffix, guard, block, plan, param_off, target)?);
        }
        let mut phase = String::new();
        let mut period = String::new();
        let mut times = Vec::new();
        let mut cross_cond = String::new();
        let kind = match ev {
            EventSpec::Periodic { period: p, phase: ph, .. } => {
                phase = lit(*ph);
                period = lit(*p);
                fields.push(format!("    {t} next_{suffix};"));
                inits.push(format!("    m->next_{suffix} = {};", lit(*ph)));
                "periodic"
            }
            EventSpec::Fixed { times: ts, .. } => {
                times = ts.iter().map(|x| lit(*x)).collect();
                fields.push(format!("    size_t fi_{suffix};"));
                inits.push(format!("    m->fi_{suffix} = 0;"));
                "fixed"
            }
            EventSpec::ZeroCross { direction, .. } => {
                cross_cond = match direction {
                    Direction::Both => format!("cur_{suffix} * m->prev_{suffix} < 0"),
                    Direction::Rising => format!("cur_{suffix} > 0 && m->prev_{suffix} <= 0"),
                    Direction::Falling => format!("cur_{suffix} < 0 && m->prev_{suffix} >= 0"),
                };
                fields.push(format!("    {t} prev_{suffix};"));
                fields.push(format!("    int init_{suffix};"));
                inits.push(format!("    m->init_{suffix} = 0;"));
                "zerocross"
            }
            EventSpec::Condition { .. } => {
                fields.push(format!("    {t} prev_{suffix};"));
                fields.push(format!("    int init_{suffix};"));
                inits.push(format!("    m->init_{suffix} = 0;"));
                "condition"
            }
        };
        ctxs.push(StructEventCtx { kind, suffix, phase, period, times, cross_cond });
    }
    if !events.is_empty() {
        // First-step guard for `<name>_step`: events due at the CURRENT time
        // (phase-0 schedules) fire once before the first step, mirroring
        // `<name>_run`'s initial `handle_events`. `run` sets it too, so mixing
        // the two drivers never re-fires the initial pass.
        fields.push("    int fs_started;  /* initial events fired (step/run drivers) */".to_string());
        inits.push("    m->fs_started = 0;".to_string());
    }
    Ok(StructEvents { fns, ctxs, fields, inits, need_sig })
}



#[cfg(test)]
mod opaque_block_tests {
    use super::generate;
    use crate::blocks::block::BlockRef;
    use crate::blocks::constructors::{
        chirp_phase_noise_source, scope, sinusoidal_phase_noise_source, white_noise,
    };
    use crate::codegen::CodegenOptions;
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

    /// `src -> scope` lowered to a Module, then codegen'd.
    fn gen_source(src: BlockRef) -> Result<Vec<super::GeneratedFile>, crate::codegen::CodegenError> {
        let sink = scope(None, 0.0, vec![]);
        let mut sim = Simulation::with_defaults(vec![src.clone(), sink.clone()], vec![conn(&src, &sink)]);
        sim.run(0.01, false, false); // assemble (resolve shape-poly widths)
        generate(&module_from_sim(&sim, "m"), &CodegenOptions::default())
    }

    #[test]
    fn phase_noise_sources_lower_to_nominal_c() {
        // The deterministic chirp/sinusoid skeletons now carry ops, so codegen
        // succeeds (the RNG phase noise is taken at its zero nominal).
        let chirp = chirp_phase_noise_source(1.0, 1.0, 2.0, 1.0, 0.0, 0.1, 0.05, None, Some(7));
        let files = gen_source(chirp).expect("ChirpPhaseNoiseSource lowers to C");
        assert!(files.iter().any(|f| f.name == "m.c"));

        let sine = sinusoidal_phase_noise_source(2.0, 1.0, 0.0, 0.1, 0.05, None, Some(7));
        gen_source(sine).expect("SinusoidalPhaseNoiseSource lowers to C");
    }

    #[test]
    fn genuinely_opaque_block_errors_with_block_name() {
        // A pure RNG source has no static op-graph: codegen must reject it with a
        // clear, block-named message (not a cryptic "Op::Call").
        let noise = white_noise(1.0, None, None, Some(7));
        let err = gen_source(noise).expect_err("WhiteNoise has no op-graph");
        let msg = format!("{err}");
        assert!(msg.contains("WhiteNoise"), "error names the block: {msg}");
        assert!(msg.contains("opaque"), "error states the reason: {msg}");
        assert!(!msg.contains("Op::Call"), "no cryptic Op::Call leak: {msg}");
    }
}

