// SSA Graph IR for JIT compilation.
//
// A directed acyclic graph where each node produces a single f64 value.
// Hash-consing at insertion time gives automatic CSE (common subexpression
// elimination) — identical subexpressions share the same NodeId.
//
// Inputs are name-addressed via an `InputSignature`: each slot carries a
// `(name, size)` pair.  Internally every input reference collapses into a
// single flat index (`Node::Input(flat_idx)`).  Downstream consumers (Tape,
// AD, Tracer) consult the signature to recover slot-level information.
//
// Parameters (`Node::Param`) are a separate namespace — their values are
// set once and don't change between calls (unlike inputs, which are
// supplied at every call).

use std::collections::HashMap;
use std::fmt;

/// Index into the graph's node array.
pub type NodeId = u32;

// The op vocabulary and its canonical f64 semantics live in `super::op` (the op
// manifest). Re-exported here so existing `graph::BinOp` / `graph::apply_*`
// paths (and the IR's re-export of them) keep resolving unchanged.
pub use super::op::{
    apply_binary, apply_cmp, apply_unary, digamma, rand_uniform, BinOp, CmpOp, ReduceOp, UnaryOp,
};

/// A single node in the SSA graph.
#[derive(Clone, PartialEq)]
pub enum Node {
    /// Literal constant.
    Const(u64),
    /// Load from the flat input space at `flat_idx`.
    /// `InputSignature` maps the flat index back to `(slot_name, element)`.
    Input(u32),
    /// Load mutable parameter by index.
    Param(u32),
    /// Binary operation.
    Binary(BinOp, NodeId, NodeId),
    /// Unary operation.
    Unary(UnaryOp, NodeId),
    /// Comparison (result 0.0 or 1.0).
    Cmp(CmpOp, NodeId, NodeId),
    /// Conditional select: if cond != 0 then a else b.
    Select(NodeId, NodeId, NodeId),
    /// Fused multiply-add: a * b + c (single rounding).
    Fma(NodeId, NodeId, NodeId),
    /// Variadic reduction over an operand list (sum, product, min, max).
    Reduce(ReduceOp, Vec<NodeId>),
    /// Fused dot product `Σ aᵢ·bᵢ` over two equal-length operand lists,
    /// accumulated with `mul_add` (one rounding per term). The structured form
    /// of a matrix-vector row: one node instead of an FMA chain, and the tape
    /// walks a single fused loop.
    Dot(Vec<NodeId>, Vec<NodeId>),
}

impl Node {
    /// Visit each child `NodeId` (operand) of this node, in order. Leaves
    /// (`Const`/`Input`/`Param`) have none. The single source of truth for "what
    /// does this node depend on": reachability, DCE and structural SSA queries
    /// use this instead of re-matching every variant.
    #[inline]
    pub fn for_each_child(&self, mut f: impl FnMut(NodeId)) {
        match self {
            Node::Const(_) | Node::Input(_) | Node::Param(_) => {}
            Node::Unary(_, a) => f(*a),
            Node::Binary(_, a, b) | Node::Cmp(_, a, b) => {
                f(*a);
                f(*b);
            }
            Node::Select(c, t, e) | Node::Fma(c, t, e) => {
                f(*c);
                f(*t);
                f(*e);
            }
            Node::Reduce(_, args) => args.iter().for_each(|&a| f(a)),
            Node::Dot(a, b) => {
                a.iter().for_each(|&x| f(x));
                b.iter().for_each(|&x| f(x));
            }
        }
    }

    /// Mutable visit of each child `NodeId` (for operand remapping after DCE /
    /// node merging). Same traversal as [`Node::for_each_child`].
    #[inline]
    pub fn for_each_child_mut(&mut self, mut f: impl FnMut(&mut NodeId)) {
        match self {
            Node::Const(_) | Node::Input(_) | Node::Param(_) => {}
            Node::Unary(_, a) => f(a),
            Node::Binary(_, a, b) | Node::Cmp(_, a, b) => {
                f(a);
                f(b);
            }
            Node::Select(c, t, e) | Node::Fma(c, t, e) => {
                f(c);
                f(t);
                f(e);
            }
            Node::Reduce(_, args) => args.iter_mut().for_each(f),
            Node::Dot(a, b) => {
                a.iter_mut().for_each(&mut f);
                b.iter_mut().for_each(f);
            }
        }
    }
}

impl Eq for Node {}
impl std::hash::Hash for Node {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Node::Const(bits) => bits.hash(state),
            Node::Input(i) | Node::Param(i) => i.hash(state),
            Node::Binary(op, a, b) => { op.hash(state); a.hash(state); b.hash(state); }
            Node::Unary(op, a) => { op.hash(state); a.hash(state); }
            Node::Cmp(op, a, b) => { op.hash(state); a.hash(state); b.hash(state); }
            Node::Select(c, t, e) => { c.hash(state); t.hash(state); e.hash(state); }
            Node::Fma(a, b, c) => { a.hash(state); b.hash(state); c.hash(state); }
            Node::Reduce(op, args) => { op.hash(state); args.hash(state); }
            Node::Dot(a, b) => { a.hash(state); b.hash(state); }
        }
    }
}

// ======================================================================================
// InputSignature — ordered list of named input slots
// ======================================================================================

/// Description of one input slot (positional, ordered).
#[derive(Clone, Debug)]
pub struct InputSlot {
    pub name: String,
    pub size: usize,
    pub offset: usize,  // computed: flat start position
}

/// Full input specification for a graph.
#[derive(Clone, Debug, Default)]
pub struct InputSignature {
    pub slots: Vec<InputSlot>,
    pub total_size: usize,
}

impl InputSignature {
    /// Build a signature from `(name, size)` pairs in order.  Offsets are
    /// computed automatically.
    pub fn from_named_sizes<S: Into<String>>(entries: impl IntoIterator<Item = (S, usize)>) -> Self {
        let mut slots = Vec::new();
        let mut offset = 0usize;
        for (name, size) in entries {
            slots.push(InputSlot { name: name.into(), size, offset });
            offset += size;
        }
        Self { total_size: offset, slots }
    }

    /// Empty signature (no inputs).
    pub fn empty() -> Self { Self::default() }

    /// Look up a slot by name.
    pub fn slot(&self, name: &str) -> Option<&InputSlot> {
        self.slots.iter().find(|s| s.name == name)
    }

    /// Flat index → `(slot_idx, element_idx)`.  Used by tape building.
    pub fn decode(&self, flat: usize) -> (usize, usize) {
        for (i, s) in self.slots.iter().enumerate() {
            if flat >= s.offset && flat < s.offset + s.size {
                return (i, flat - s.offset);
            }
        }
        // Out-of-range (shouldn't happen for validated graphs); return sentinel.
        (usize::MAX, 0)
    }

    /// Number of slots.
    pub fn len(&self) -> usize { self.slots.len() }
    pub fn is_empty(&self) -> bool { self.slots.is_empty() }
}

// ======================================================================================
// Graph
// ======================================================================================

/// SSA computation graph with hash-consing.
#[derive(Clone)]
pub struct Graph {
    /// All nodes, indexed by NodeId.
    pub nodes: Vec<Node>,
    /// Hash-consing map: node → existing NodeId (for CSE).
    pub dedup: HashMap<Node, NodeId>,
    /// Output node IDs (one per output element).
    pub outputs: Vec<NodeId>,
    /// Named input layout.
    pub signature: InputSignature,
    /// Number of mutable parameters.
    pub n_params: usize,
    /// Parameter default values.
    pub param_defaults: Vec<f64>,
    /// Parameter names.
    pub param_names: Vec<String>,
}

impl Graph {
    pub fn new(signature: InputSignature) -> Self {
        Self {
            nodes: Vec::with_capacity(64),
            dedup: HashMap::with_capacity(64),
            outputs: Vec::new(),
            signature,
            n_params: 0,
            param_defaults: Vec::new(),
            param_names: Vec::new(),
        }
    }

    /// Convenience: create with a single `(name, size)` slot.
    pub fn with_single_input<S: Into<String>>(name: S, size: usize) -> Self {
        Self::new(InputSignature::from_named_sizes([(name, size)]))
    }

    /// Add a node, returning its ID. If an identical node exists, returns that ID (CSE).
    pub fn add(&mut self, node: Node) -> NodeId {
        if let Some(&id) = self.dedup.get(&node) {
            return id;
        }
        let id = self.nodes.len() as NodeId;
        self.dedup.insert(node.clone(), id);
        self.nodes.push(node);
        id
    }

    pub fn constant(&mut self, v: f64) -> NodeId {
        self.add(Node::Const(v.to_bits()))
    }

    pub fn input(&mut self, flat_idx: u32) -> NodeId {
        self.add(Node::Input(flat_idx))
    }

    pub fn param(&mut self, idx: u32) -> NodeId {
        self.add(Node::Param(idx))
    }

    pub fn binary(&mut self, op: BinOp, a: NodeId, b: NodeId) -> NodeId {
        self.add(Node::Binary(op, a, b))
    }

    pub fn unary(&mut self, op: UnaryOp, a: NodeId) -> NodeId {
        self.add(Node::Unary(op, a))
    }

    pub fn cmp(&mut self, op: CmpOp, a: NodeId, b: NodeId) -> NodeId {
        self.add(Node::Cmp(op, a, b))
    }

    pub fn select(&mut self, cond: NodeId, then_val: NodeId, else_val: NodeId) -> NodeId {
        self.add(Node::Select(cond, then_val, else_val))
    }

    pub fn reduce(&mut self, op: ReduceOp, args: Vec<NodeId>) -> NodeId {
        self.add(Node::Reduce(op, args))
    }

    pub fn dot(&mut self, a: Vec<NodeId>, b: Vec<NodeId>) -> NodeId {
        debug_assert_eq!(a.len(), b.len(), "Dot operand lists must be equal length");
        self.add(Node::Dot(a, b))
    }

    pub fn is_const(&self, id: NodeId, val: f64) -> bool {
        matches!(self.nodes.get(id as usize), Some(Node::Const(bits)) if *bits == val.to_bits())
    }

    pub fn const_value(&self, id: NodeId) -> Option<f64> {
        match self.nodes.get(id as usize) {
            Some(Node::Const(bits)) => Some(f64::from_bits(*bits)),
            _ => None,
        }
    }

    pub fn len(&self) -> usize { self.nodes.len() }
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }

    /// Interpret the graph (straightforward recursive eval, for tests).
    /// `inputs` are in slot order and must match the signature's slot sizes.
    pub fn interpret(&self, inputs: &[&[f64]], params: &[f64]) -> Vec<f64> {
        let mut values: Vec<f64> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let v: f64 = match node {
                Node::Const(bits) => f64::from_bits(*bits),
                Node::Input(flat) => {
                    let (slot, elem) = self.signature.decode(*flat as usize);
                    inputs.get(slot)
                        .and_then(|s| s.get(elem))
                        .copied()
                        .unwrap_or(0.0)
                }
                Node::Param(i) => params.get(*i as usize).copied().unwrap_or(0.0),
                Node::Binary(op, a, b) => {
                    apply_binary(*op, values[*a as usize], values[*b as usize])
                }
                Node::Unary(op, a) => {
                    apply_unary(*op, values[*a as usize])
                }
                Node::Cmp(op, a, b) => {
                    apply_cmp(*op, values[*a as usize], values[*b as usize])
                }
                Node::Select(c, th, el) => {
                    if values[*c as usize] != 0.0 { values[*th as usize] } else { values[*el as usize] }
                }
                Node::Fma(a, b, c) => {
                    values[*a as usize].mul_add(values[*b as usize], values[*c as usize])
                }
                Node::Reduce(op, args) => {
                    let xs: Vec<f64> = args.iter().map(|&a| values[a as usize]).collect();
                    super::op::reduce(*op, &xs)
                }
                Node::Dot(a, b) => {
                    let av: Vec<f64> = a.iter().map(|&i| values[i as usize]).collect();
                    let bv: Vec<f64> = b.iter().map(|&i| values[i as usize]).collect();
                    super::op::dot(&av, &bv)
                }
            };
            values.push(v);
        }
        self.outputs.iter().map(|&id| values[id as usize]).collect()
    }
}



impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Graph({} nodes, {} outputs, slots=[",
               self.nodes.len(), self.outputs.len())?;
        for (i, s) in self.signature.slots.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}={}", s.name, s.size)?;
        }
        write!(f, "], params={})", self.n_params)
    }
}

