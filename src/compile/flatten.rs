//! Flatten a hierarchical IR `Module` into a single flat scope (leaf blocks +
//! normalized connections) for the splicer.
//!
//! Subsystems are pure relays: an interface forwards the subsystem's external
//! inputs to inner consumers and inner producers to its external outputs. We
//! collect every leaf block (DFS, fresh contiguous `BlockId`s) and then, for
//! each leaf-consumer input element, trace the connection graph backward —
//! descending into / popping out of subsystems through `BlockId::INTERFACE` —
//! to the ultimate leaf producer, emitting one direct leaf→leaf connection.
//! Arbitrary nesting and chained interfaces fall out of the recursion.
//!
//! Acyclicity is recomputed on the flattened graph (`utils::schedule`); a dataflow
//! cycle found while tracing also marks the model as non-acyclic. (The splicer
//! additionally guards against cycles, so a missed classification cannot hang.)

use std::collections::HashSet;

use crate::ir::schema::{Block, BlockId, BlockRole, Child, Connection, ConnectionId, Module, PortRef, Subsystem};
use crate::utils::schedule::{Schedule, NodeRole};

use super::CompileError;

pub(super) struct Flat {
    pub blocks: Vec<Block>,
    pub connections: Vec<Connection>,
    pub has_acyclic_schedule: bool,
}

enum ChildKind {
    /// Leaf block with its flat (global) id.
    Leaf(u32),
    /// Nested subsystem at this scope index.
    Sub(usize),
}

struct Scope {
    /// One entry per local child (indexed by the scope-local `BlockId`).
    children: Vec<ChildKind>,
    /// Connections of this scope (local refs / `INTERFACE`).
    connections: Vec<Connection>,
    /// `(parent_scope, this_subsystem's local id in parent)`; `None` for root.
    parent: Option<(usize, u32)>,
}

fn elems_or0(elems: &Option<Vec<u32>>) -> Vec<u32> {
    elems.clone().unwrap_or_else(|| vec![0])
}

struct Flattener {
    scopes: Vec<Scope>,
    /// When true, a producer trace that bottoms out at the *root* scope's
    /// interface yields `(BlockId::INTERFACE.0, elem)` instead of `None` — so a
    /// subsystem can be spliced into a block with a live `u` (external-input)
    /// boundary rather than having its inputs folded to zero.
    preserve_root_iface: bool,
}

impl Flattener {
    /// Trace `(scope, block, elem)` on the *producer* side back to the leaf that
    /// drives it. `visiting` breaks dataflow cycles (sets `saw_cycle`).
    fn resolve(
        &self,
        scope: usize,
        block: BlockId,
        elem: u32,
        visiting: &mut HashSet<(usize, u32, u32)>,
        saw_cycle: &mut bool,
    ) -> Option<(u32, u32)> {
        let key = (scope, block.0, elem);
        if !visiting.insert(key) {
            *saw_cycle = true;
            return None;
        }
        let res = if block == BlockId::INTERFACE {
            // External input of this scope: pop to the parent and resolve what
            // feeds this subsystem's input element.
            match self.scopes[scope].parent {
                Some((pscope, child)) => {
                    self.find_producer(pscope, BlockId(child), elem, visiting, saw_cycle)
                }
                // Root external input: unconnected (whole-system) or the
                // subsystem's own interface boundary (subsystem-compile).
                None if self.preserve_root_iface => Some((BlockId::INTERFACE.0, elem)),
                None => None,
            }
        } else {
            match self.scopes[scope].children[block.idx()] {
                ChildKind::Leaf(flat) => Some((flat, elem)),
                // Subsystem output element: descend to its internal feeder of
                // the interface output.
                ChildKind::Sub(sub) => {
                    self.find_producer(sub, BlockId::INTERFACE, elem, visiting, saw_cycle)
                }
            }
        };
        visiting.remove(&key);
        res
    }

    /// Find the connection in `scope` delivering `elem` to `target_block`, and
    /// resolve its source to a leaf.
    fn find_producer(
        &self,
        scope: usize,
        target_block: BlockId,
        elem: u32,
        visiting: &mut HashSet<(usize, u32, u32)>,
        saw_cycle: &mut bool,
    ) -> Option<(u32, u32)> {
        for c in &self.scopes[scope].connections {
            let s_idx = elems_or0(&c.src.elems);
            for tgt in &c.targets {
                if tgt.block != target_block {
                    continue;
                }
                let d_idx = elems_or0(&tgt.elems);
                if let Some(j) = d_idx.iter().position(|&d| d == elem) {
                    let se = s_idx.get(j).copied().unwrap_or(0);
                    return self.resolve(scope, c.src.block, se, visiting, saw_cycle);
                }
            }
        }
        None
    }
}

/// DFS: register a scope, clone its leaf blocks with fresh flat ids, recurse
/// into nested subsystems. Returns the new scope's index.
fn build_scopes(
    sub: &Subsystem,
    parent: Option<(usize, u32)>,
    scopes: &mut Vec<Scope>,
    leaves: &mut Vec<Block>,
) -> usize {
    let my_idx = scopes.len();
    scopes.push(Scope { children: Vec::new(), connections: sub.connections.clone(), parent });

    let mut kinds = Vec::with_capacity(sub.children.len());
    for (local, ch) in sub.children.iter().enumerate() {
        match ch {
            Child::Block(b) => {
                let flat = leaves.len() as u32;
                let mut clone = b.clone();
                clone.id = BlockId(flat);
                leaves.push(clone);
                kinds.push(ChildKind::Leaf(flat));
            }
            Child::Subsystem(s) => {
                let sub_idx = build_scopes(s, Some((my_idx, local as u32)), scopes, leaves);
                kinds.push(ChildKind::Sub(sub_idx));
            }
        }
    }
    scopes[my_idx].children = kinds;
    my_idx
}

/// Shared core for both flatten entry points: build the scope tree for `sub`,
/// then emit one direct leaf→leaf connection per leaf-consumer input element
/// (tracing each source back through nested interfaces). With
/// `preserve_root_iface`, a trace bottoming out at the root interface yields an
/// `INTERFACE`-sourced connection (the live `u` boundary) instead of being
/// dropped. Returns the built `Flattener` (for any follow-up interface-output
/// tracing), the leaf blocks, the normalized connections, and whether a dataflow
/// cycle was seen.
fn flatten_scopes(
    sub: &Subsystem,
    preserve_root_iface: bool,
) -> (Flattener, Vec<Block>, Vec<Connection>, bool) {
    let mut scopes: Vec<Scope> = Vec::new();
    let mut leaves: Vec<Block> = Vec::new();
    build_scopes(sub, None, &mut scopes, &mut leaves);

    let fl = Flattener { scopes, preserve_root_iface };
    let mut saw_cycle = false;
    let mut connections: Vec<Connection> = Vec::new();
    let mut cid = 0u32;

    for scope in 0..fl.scopes.len() {
        for c in &fl.scopes[scope].connections {
            let s_idx = elems_or0(&c.src.elems);
            for tgt in &c.targets {
                let flat_consumer = match tgt.block {
                    BlockId::INTERFACE => continue, // subsystem output: relay
                    b => match fl.scopes[scope].children[b.idx()] {
                        ChildKind::Leaf(f) => f,
                        ChildKind::Sub(_) => continue, // subsystem input: relay
                    },
                };
                let d_idx = elems_or0(&tgt.elems);
                for (j, &d) in d_idx.iter().enumerate() {
                    let se = s_idx.get(j).copied().unwrap_or(0);
                    let mut visiting = HashSet::new();
                    if let Some((pf, pe)) =
                        fl.resolve(scope, c.src.block, se, &mut visiting, &mut saw_cycle)
                    {
                        connections.push(Connection {
                            id: ConnectionId(cid),
                            src: PortRef { block: BlockId(pf), port: 0, elems: Some(vec![pe]) },
                            targets: vec![PortRef {
                                block: BlockId(flat_consumer),
                                port: 0,
                                elems: Some(vec![d]),
                            }],
                        });
                        cid += 1;
                    }
                }
            }
        }
    }

    (fl, leaves, connections, saw_cycle)
}

pub(super) fn flatten(module: &Module) -> Result<Flat, CompileError> {
    let (_fl, leaves, connections, saw_cycle) = flatten_scopes(&module.root, false);
    let has_acyclic_schedule = acyclic(&leaves, &connections, saw_cycle);
    Ok(Flat { blocks: leaves, connections, has_acyclic_schedule })
}

/// Recompute acyclicity over flattened leaf→leaf edges. Interface-sourced edges
/// (`BlockId::INTERFACE`) are external inputs, not internal feedback, so they are
/// skipped — they cannot close an internal algebraic loop.
fn acyclic(leaves: &[Block], connections: &[Connection], saw_cycle: bool) -> bool {
    let roles: Vec<NodeRole> = leaves
        .iter()
        .map(|b| NodeRole {
            is_alg: b.role == BlockRole::Algebraic,
        })
        .collect();
    let edges: Vec<(usize, usize, usize)> = connections
        .iter()
        .enumerate()
        .filter(|(_, c)| c.src.block != BlockId::INTERFACE)
        .map(|(i, c)| (c.src.block.idx(), c.targets[0].block.idx(), i))
        .collect();
    let g = Schedule::new(&roles, &edges);
    !g.has_loops && !saw_cycle
}

/// Result of flattening a `Subsystem` *for compilation into a single block*: leaf
/// blocks + normalized connections (interior leaf→leaf plus `INTERFACE`→leaf for
/// external inputs), the inner producer of each interface-output element, and the
/// external boundary widths.
pub(super) struct FlatSub {
    pub blocks: Vec<Block>,
    pub connections: Vec<Connection>,
    /// Per global output element: the inner producer `(leaf, elem)` driving it,
    /// `(INTERFACE.0, elem)` for an input passthrough, `None` if unconnected.
    pub output_srcs: Vec<Option<(u32, u32)>>,
    pub n_in: usize,
    pub n_out: usize,
    pub has_acyclic_schedule: bool,
}

/// Flatten a `Subsystem` into a flat scope that keeps its *own* interface live:
/// inner blocks read external inputs through `INTERFACE`-sourced connections (the
/// `u` slot), and each interface-output element is traced back to its inner
/// producer. Nested child subsystems are fully inlined.
pub(super) fn flatten_subsystem(sub: &Subsystem) -> Result<FlatSub, CompileError> {
    let n_in: usize = sub.interface.inputs.iter().map(|p| p.size as usize).sum();
    let n_out: usize = sub.interface.outputs.iter().map(|p| p.size as usize).sum();

    // Interior connections (one direct edge per leaf-consumer input element),
    // keeping the subsystem's own interface live as the `u` boundary.
    let (fl, leaves, connections, mut saw_cycle) = flatten_scopes(sub, true);

    // Interface outputs: trace each root `INTERFACE`-targeted element back to the
    // inner leaf that produces it (global output-element order).
    let mut output_srcs: Vec<Option<(u32, u32)>> = vec![None; n_out];
    for c in &fl.scopes[0].connections {
        let s_idx = elems_or0(&c.src.elems);
        for tgt in &c.targets {
            if tgt.block != BlockId::INTERFACE {
                continue;
            }
            let d_idx = elems_or0(&tgt.elems);
            for (j, &d) in d_idx.iter().enumerate() {
                let se = s_idx.get(j).copied().unwrap_or(0);
                let mut visiting = HashSet::new();
                if let Some(p) = fl.resolve(0, c.src.block, se, &mut visiting, &mut saw_cycle) {
                    if (d as usize) < n_out {
                        output_srcs[d as usize] = Some(p);
                    }
                }
            }
        }
    }

    let has_acyclic_schedule = acyclic(&leaves, &connections, saw_cycle);

    Ok(FlatSub { blocks: leaves, connections, output_srcs, n_in, n_out, has_acyclic_schedule })
}
