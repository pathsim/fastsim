// Optimization passes on the SSA Graph IR.
//
// These run on the graph before it is lowered to the flat tape
// (`InterpretedFn::from_graph`), simplifying it at trace time so the tape is
// as small and cheap as possible.
//
// Passes:
// 1. Constant folding — evaluate Const op Const at compile time
// 2. Strength reduction — x^2→x*x, x^0.5→sqrt(x), x^1→x, x^0→1
// 3. Algebraic identities — x+0→x, x*1→x, x*0→0, x-x→0
// 4. FMA detection — a*b+c → fma(a,b,c)

use super::graph::*;

/// Run all optimization passes on a graph. Modifies in place.
/// Returns true if any changes were made.
pub fn optimize(graph: &mut Graph) -> bool {
    let mut any_changed = false;
    // Iterate until fixpoint
    loop {
        let mut changed = false;
        for id in 0..graph.nodes.len() {
            changed |= constant_fold(graph, id as NodeId);
            changed |= strength_reduce(graph, id as NodeId);
            changed |= algebraic_simplify(graph, id as NodeId);
        }
        if !changed { break; }
        any_changed = true;
    }
    // FMA detection (single pass, after other opts stabilize)
    for id in 0..graph.nodes.len() {
        any_changed |= detect_fma(graph, id as NodeId);
    }
    any_changed
}

/// Remove nodes not reachable from outputs, compact the node array, update
/// all references (outputs, binary/unary/cmp/select/fma operands) and rebuild
/// the hash-cons map.  Returns true if anything was removed.
///
/// NOTE: this renumbers NodeIds.  Callers holding IDs from before DCE must
/// not use them afterwards — prefer calling this immediately before tape
/// construction rather than as part of the general `optimize()` pipeline.
pub fn dead_code_elimination(graph: &mut Graph) -> bool {
    let n = graph.nodes.len();
    if n == 0 { return false; }

    // 1. Reachability via DFS from outputs.
    let mut live = vec![false; n];
    let mut stack: Vec<NodeId> = graph.outputs.clone();
    while let Some(id) = stack.pop() {
        let i = id as usize;
        if i >= n || live[i] { continue; }
        live[i] = true;
        graph.nodes[i].for_each_child(|c| stack.push(c));
    }

    if live.iter().all(|&b| b) { return false; }

    // 2. Build old→new index remap.
    let mut remap = vec![0u32; n];
    let mut new_nodes: Vec<Node> = Vec::with_capacity(live.iter().filter(|&&b| b).count());
    for (i, keep) in live.iter().enumerate() {
        if *keep {
            remap[i] = new_nodes.len() as u32;
            new_nodes.push(graph.nodes[i].clone());
        }
    }

    // 3. Rewrite operand references.
    for node in &mut new_nodes {
        node.for_each_child_mut(|c| *c = remap[*c as usize]);
    }

    // 4. Remap outputs.
    for out in &mut graph.outputs {
        *out = remap[*out as usize];
    }

    // 5. Rebuild hash-consing map (old IDs are stale).
    graph.nodes = new_nodes;
    graph.dedup.clear();
    for (i, node) in graph.nodes.iter().enumerate() {
        graph.dedup.entry(node.clone()).or_insert(i as u32);
    }

    true
}

// ======================================================================================
// Canonicalize: tape-lowering rebuild (value numbering + folds)
// ======================================================================================

/// Tape-lowering canonicalization: rebuild the live graph through a FRESH
/// hash-consing map, with commutative operand ordering and structural folds
/// applied on the way in. This is the additive tier on top of `optimize()`:
///
/// - `optimize()`'s in-place rewrites leave duplicate node content behind
///   (`nodes[id] = nodes[x].clone()`) and a stale `dedup` map, so AD-emitted
///   expressions miss CSE. The rebuild re-numbers values and merges all
///   duplicates.
/// - Commutative `Binary` operands (`Add`/`Mul`/`Min`/`Max`/`Hypot`) and the
///   symmetric comparisons (`Eq`/`Ne`) are sorted by operand id, so `a*b` and
///   `b*a` value-number identically (bit-exact: IEEE `+`/`*`/`min`/`max` are
///   commutative).
/// - Folds the variants `optimize()` does not cover: `Cmp`/`Select`/`Fma`/
///   `Reduce`/`Dot` constant folding, `Select(c, a, a)` -> `a`, branch
///   selection under a constant condition, and single-operand `Reduce`/`Dot`
///   collapse.
/// - Subsumes `dead_code_elimination`: only nodes reachable from the outputs
///   are rebuilt (run to a fixpoint, since a constant-condition `Select` fold
///   can orphan its dead branch mid-rebuild).
///
/// Every transform is VALUE-EXACT (at most the sign of a zero can change,
/// which no comparison or arithmetic observes): no reassociation, no
/// re-grouping of the fused 4-lane `Reduce`/`Dot` folds. That matters because
/// a regrouped sum shifting by one ULP can flip a downstream tolerance-banded
/// comparison at its boundary — a 0/1 jump, not an ULP error. Reassociating
/// transforms (Add-chain -> Reduce bundling) belong to a separate, explicitly
/// opt-in tier.
///
/// Invoked ONLY at tape lowering (`InterpretedFn::from_graph`) — the runtime
/// and the static compile path. Codegen consumes the IR built from the
/// un-canonicalized graphs (`ir::builder::region_from_graph`) and is
/// deliberately untouched, keeping the emitted C readable and stable.
///
/// NOTE: renumbers NodeIds (like DCE) — callers holding old ids must not use
/// them afterwards. Idempotent: once `canonicalize` returns, a further run
/// reports no changes.
pub fn canonicalize(graph: &mut Graph) -> bool {
    let mut any = false;
    // Fixpoint: a Select fold can orphan its dead branch (inserted before the
    // fold was visible), which only the next pass's liveness sweep removes.
    // Convergence: the rebuild's left-to-right DFS keeps sorted operand order
    // stable across passes (see `canonicalize_once`), so a pass either folds/
    // merges (shrinks) or reproduces the graph. The hard cap is a production
    // safety net only — stopping early is always sound (every pass is
    // semantics-preserving), and the idempotence fuzzer pins convergence.
    for _ in 0..crate::constants::COMPILE_OPTIMIZE_MAX_PASSES {
        if !canonicalize_once(graph) {
            break;
        }
        any = true;
    }
    any
}

fn canonicalize_once(graph: &mut Graph) -> bool {
    let n = graph.nodes.len();
    if n == 0 {
        return false;
    }

    // Rebuild live nodes in post-order DFS from the outputs, so children are
    // always rebuilt before their parents. (Index order is NOT assumed:
    // `reassociate_chains` re-materializes operand nodes ABOVE their
    // consumer, and only reachable nodes are rebuilt — DCE for free.)
    //
    // Children must be visited LEFT-TO-RIGHT (first operand first): the first
    // operand then receives the smaller rebuilt id, so an operand pair sorted
    // by the commutative canonicalization KEEPS its order on the next pass.
    // Right-first traversal would re-invert the ids each pass and make the
    // sort ping-pong — a fixpoint loop that never terminates.
    let mut new = Graph::new(graph.signature.clone());
    new.n_params = graph.n_params;
    new.param_defaults = graph.param_defaults.clone();
    new.param_names = graph.param_names.clone();
    let mut remap = vec![u32::MAX; n];
    let mut visited = vec![false; n];
    let mut children: smallvec::SmallVec<[NodeId; 4]> = smallvec::SmallVec::new();
    let mut stack: Vec<(NodeId, bool)> =
        graph.outputs.iter().rev().map(|&o| (o, false)).collect();
    while let Some((id, children_done)) = stack.pop() {
        let i = id as usize;
        if children_done {
            if remap[i] == u32::MAX {
                let mut node = graph.nodes[i].clone();
                node.for_each_child_mut(|c| *c = remap[*c as usize]);
                remap[i] = insert_canonical(&mut new, node);
            }
        } else {
            if visited[i] {
                continue;
            }
            visited[i] = true;
            stack.push((id, true));
            // Push reversed so the FIRST child pops (and rebuilds) first.
            children.clear();
            graph.nodes[i].for_each_child(|c| children.push(c));
            for &c in children.iter().rev() {
                stack.push((c, false));
            }
        }
    }
    let new_outputs: Vec<NodeId> =
        graph.outputs.iter().map(|&o| remap[o as usize]).collect();

    let changed = new.nodes != graph.nodes || new_outputs != graph.outputs;
    graph.nodes = new.nodes;
    graph.dedup = new.dedup;
    graph.outputs = new_outputs;
    changed
}

/// Insert one node (children already remapped into `g`) with canonical
/// operand order and structural folds applied. May recurse one level when a
/// fold rewrites the node kind (e.g. `Fma(a, 1, c)` -> `Add(a, c)`).
fn insert_canonical(g: &mut Graph, mut node: Node) -> NodeId {
    // Commutative / symmetric operand ordering. Only ops that are bit-exact
    // commutative under IEEE qualify: `+`/`*` (including signed zeros),
    // `hypot`, and the |a-b|-banded Eq/Ne. `Min`/`Max` are excluded — on a
    // `±0.0` tie they return the SECOND operand, and `Sign`/`atan2`/`1/x`
    // observe a zero's sign, so swapping could flip a downstream value.
    match &mut node {
        Node::Binary(BinOp::Add | BinOp::Mul | BinOp::Hypot, a, b) if *a > *b => {
            std::mem::swap(a, b)
        }
        Node::Cmp(CmpOp::Eq | CmpOp::Ne, a, b) if *a > *b => std::mem::swap(a, b),
        _ => {}
    }

    match &node {
        // Binary/Unary constant folding with the same domain gates as the
        // fixed-point pass (never bake a NaN).
        Node::Binary(op, a, b) => {
            if let (Some(va), Some(vb)) = (g.const_value(*a), g.const_value(*b)) {
                let div_by_zero = matches!(op, BinOp::Div | BinOp::Mod) && vb == 0.0;
                if !div_by_zero {
                    let r = apply_binary(*op, va, vb);
                    if !r.is_nan() {
                        return g.constant(r);
                    }
                }
            }
        }
        Node::Unary(op, a) => {
            if let Some(va) = g.const_value(*a) {
                if unary_fold_in_domain(*op, va) {
                    let r = apply_unary(*op, va);
                    if !r.is_nan() {
                        return g.constant(r);
                    }
                }
            }
        }
        Node::Cmp(op, a, b) => {
            if let (Some(va), Some(vb)) = (g.const_value(*a), g.const_value(*b)) {
                return g.constant(apply_cmp(*op, va, vb));
            }
        }
        Node::Select(c, t, e) => {
            if let Some(cv) = g.const_value(*c) {
                return if cv != 0.0 { *t } else { *e };
            }
            if t == e {
                return *t;
            }
        }
        Node::Fma(a, b, c) => {
            if let (Some(va), Some(vb), Some(vc)) =
                (g.const_value(*a), g.const_value(*b), g.const_value(*c))
            {
                let r = va.mul_add(vb, vc);
                if !r.is_nan() {
                    return g.constant(r);
                }
            }
            // fma(a, 1, c) computes a*1 + c with one rounding == Add exactly.
            if g.is_const(*a, 1.0) {
                return insert_canonical(g, Node::Binary(BinOp::Add, *b, *c));
            }
            if g.is_const(*b, 1.0) {
                return insert_canonical(g, Node::Binary(BinOp::Add, *a, *c));
            }
        }
        Node::Reduce(op, args) => {
            // Single operand: `combine(identity, x)`. Only Product (1*x) is
            // exactly `x` for EVERY input. Sum differs in a zero's sign
            // (`0.0 + (-0.0)` is `+0.0`, observable through Sign/atan2/1/x);
            // Min/Max IGNORE a NaN operand (`max(-inf, NaN)` is `-inf`), so
            // collapsing them would turn an identity into a NaN.
            if args.len() == 1 && matches!(op, ReduceOp::Product) {
                return args[0];
            }
            if let Some(vals) = args
                .iter()
                .map(|&a| g.const_value(a))
                .collect::<Option<Vec<f64>>>()
            {
                return g.constant(super::op::reduce(*op, &vals));
            }
            // NO identity-element drops here: removing an operand re-groups
            // the 4-lane fold (ULP shifts) — see the module-level note.
        }
        Node::Dot(a, b) => {
            if let (Some(av), Some(bv)) = (
                a.iter().map(|&i| g.const_value(i)).collect::<Option<Vec<f64>>>(),
                b.iter().map(|&i| g.const_value(i)).collect::<Option<Vec<f64>>>(),
            ) {
                return g.constant(super::op::dot(&av, &bv));
            }
            // NO singleton-to-Mul collapse: the kernel computes `0 + a*b`,
            // which differs from `a*b` in a zero's sign (see Reduce above).
            // NO zero-coefficient drops (re-grouping; see module-level note).
        }
        _ => {}
    }
    g.add(node)
}

// ======================================================================================
// Reassociate: Add/Fma accumulation chains -> fused Reduce/Dot (tape-only tier)
// ======================================================================================

/// The full tape-lowering pipeline applied by `InterpretedFn::from_graph`:
/// value-exact canonicalization, then chain reassociation, then a cleanup
/// canonicalize (drops the dead chain interiors). Exposed as one function so
/// tests can run the exact pipeline on a reference copy.
pub fn lower_for_tape(graph: &mut Graph) -> bool {
    let a = canonicalize(graph);
    let b = reassociate_chains(graph);
    let c = if b { canonicalize(graph) } else { false };
    a || b || c
}

/// Bundle Add/Fma accumulation chains into fused `Reduce::Sum` / `Dot` nodes.
///
/// After `optimize()`, an AD product-rule sum or a fan-in accumulation is a
/// deep chain of `Fma(a, b, Fma(c, d, …))` / `Binary(Add)` nodes: O(N) tape
/// ops with a serial dependency chain. This pass collapses a chain of
/// [`JIT_REASSOC_MIN_TERMS`]+ terms into ONE structured node — a `Dot` when
/// every term is a product, a `Reduce::Sum` otherwise — which the tape
/// evaluates with the 4-lane (AVX2-dispatched) kernels.
///
/// This is the explicitly REASSOCIATING tier (unlike `canonicalize`, which is
/// value-exact): regrouping a sum shifts results by ULPs, and an Fma term's
/// single rounding becomes the Dot kernel's two. That is the same tolerance
/// class as the FMA fusion in `optimize()` — with one principled exception:
/// a ULP shift through a DISCRETIZING op is a finite jump, not an ULP error
/// (a tolerance-banded `Eq` flips at its band edge, `floor` at an integer,
/// `RandUniform` re-keys entirely). So chains whose value can reach a
/// comparison operand, a `Select` condition, `Floor`/`Ceil`/`Round`/`Trunc`/
/// `Sign`/`RandUniform`, or a `Mod` operand are left untouched: ULP shifts
/// are only introduced where every downstream path is continuous.
///
/// Tape-lowering only (`lower_for_tape`); codegen consumes the IR from the
/// un-transformed graphs and keeps emitting the readable unrolled form.
pub fn reassociate_chains(graph: &mut Graph) -> bool {
    use crate::constants::JIT_REASSOC_MIN_TERMS;

    let n = graph.nodes.len();
    if n == 0 {
        return false;
    }

    // ---- use counts (a chain interior must have exactly one consumer) ----
    let mut use_count = vec![0u32; n];
    for node in &graph.nodes {
        node.for_each_child(|c| use_count[c as usize] += 1);
    }
    let mut is_output = vec![false; n];
    for &o in &graph.outputs {
        is_output[o as usize] = true;
    }

    // ---- ULP-sensitivity: ancestor closure of discretizing-op operands ----
    // Seed: every node feeding a discontinuous consumer position. Closure:
    // everything those seeds depend on (a shifted ancestor shifts the seed).
    let mut sensitive = vec![false; n];
    let mut stack: Vec<NodeId> = Vec::new();
    for node in &graph.nodes {
        match node {
            Node::Cmp(_, a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            Node::Select(c, _, _) => stack.push(*c),
            Node::Unary(
                UnaryOp::Floor
                | UnaryOp::Ceil
                | UnaryOp::Round
                | UnaryOp::Trunc
                | UnaryOp::Sign
                | UnaryOp::RandUniform,
                a,
            ) => stack.push(*a),
            Node::Binary(BinOp::Mod, a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            _ => {}
        }
    }
    while let Some(id) = stack.pop() {
        let i = id as usize;
        if sensitive[i] {
            continue;
        }
        sensitive[i] = true;
        graph.nodes[i].for_each_child(|c| stack.push(c));
    }

    // ---- chain detection ----
    // A node is chain INTERIOR if it is an Add/Fma consumed exactly once and
    // not an output (its value is unobservable, so regrouping it is free).
    let is_interior = |i: usize, g: &Graph| -> bool {
        use_count[i] == 1
            && !is_output[i]
            && matches!(g.nodes[i], Node::Binary(BinOp::Add, _, _) | Node::Fma(..))
    };
    // A chain ROOT is an Add/Fma that is NOT itself an interior of a larger
    // chain (its consumers are observable or non-chain ops).
    let mut roots: Vec<usize> = Vec::new();
    for i in 0..n {
        if matches!(graph.nodes[i], Node::Binary(BinOp::Add, _, _) | Node::Fma(..))
            && !is_interior(i, graph)
        {
            roots.push(i);
        }
    }

    let mut changed = false;
    for root in roots {
        if sensitive[root] {
            continue;
        }
        // Collect the chain's terms: products `(a, b)` from Fma steps and
        // single-use Mul leaves, plain nodes otherwise.
        let mut products: Vec<(NodeId, NodeId)> = Vec::new();
        let mut plains: Vec<NodeId> = Vec::new();
        let mut walk: Vec<NodeId> = vec![root as NodeId];
        let mut first = true;
        while let Some(id) = walk.pop() {
            let i = id as usize;
            let in_chain = first || is_interior(i, graph);
            first = false;
            match (&graph.nodes[i], in_chain) {
                (Node::Binary(BinOp::Add, a, b), true) => {
                    walk.push(*a);
                    walk.push(*b);
                }
                (Node::Fma(a, b, acc), true) => {
                    products.push((*a, *b));
                    walk.push(*acc);
                }
                // A single-use Mul leaf joins the product list (it dies with
                // the chain), a shared one stays a plain term.
                (Node::Binary(BinOp::Mul, a, b), _)
                    if use_count[i] == 1 && !is_output[id as usize] =>
                {
                    products.push((*a, *b));
                }
                _ => plains.push(id),
            }
        }

        let n_terms = products.len() + plains.len();
        if n_terms < JIT_REASSOC_MIN_TERMS {
            continue;
        }

        let new_node = if plains.is_empty() {
            // Pure product chain -> one fused Dot.
            let (a, b): (Vec<NodeId>, Vec<NodeId>) = products.into_iter().unzip();
            Node::Dot(a, b)
        } else {
            // Mixed chain -> Reduce::Sum; product terms re-materialize as Mul
            // nodes (hash-consed, so an existing Mul is reused).
            let mut terms = plains;
            for (a, b) in products {
                terms.push(graph.binary(BinOp::Mul, a, b));
            }
            Node::Reduce(ReduceOp::Sum, terms)
        };
        graph.nodes[root] = new_node;
        // The dedup entry for the old root content is now stale; the cleanup
        // canonicalize() in `lower_for_tape` rebuilds the map.
        changed = true;
    }
    changed
}

/// Whether a unary op is defined at `va` for compile-time folding. Out-of-domain
/// constants (e.g. `log` of a non-positive value) are left as runtime ops rather
/// than baked into a NaN. Every other op folds unconditionally.
fn unary_fold_in_domain(op: UnaryOp, va: f64) -> bool {
    match op {
        UnaryOp::Log | UnaryOp::Log10 | UnaryOp::Log2 => va > 0.0,
        UnaryOp::Sqrt => va >= 0.0,
        UnaryOp::Asin | UnaryOp::Acos => va.abs() <= 1.0,
        UnaryOp::Acosh => va >= 1.0,
        UnaryOp::Atanh => va.abs() < 1.0,
        UnaryOp::Log1p => va > -1.0,
        _ => true,
    }
}

/// Constant folding: if a binary/unary op's operands are all constants, evaluate
/// it at compile time. The arithmetic itself routes through the canonical
/// `apply_binary`/`apply_unary` (graph.rs); this pass only adds the domain gates
/// that keep an undefined fold (div-by-zero, log of a negative) a runtime op.
fn constant_fold(graph: &mut Graph, id: NodeId) -> bool {
    let node = graph.nodes[id as usize].clone();
    let result = match node {
        Node::Binary(op, a, b) => {
            let (Some(va), Some(vb)) = (graph.const_value(a), graph.const_value(b)) else {
                return false;
            };
            if matches!(op, BinOp::Div | BinOp::Mod) && vb == 0.0 {
                return false;
            }
            apply_binary(op, va, vb)
        }
        Node::Unary(op, a) => {
            let Some(va) = graph.const_value(a) else { return false };
            if !unary_fold_in_domain(op, va) {
                return false;
            }
            apply_unary(op, va)
        }
        _ => return false,
    };
    if result.is_finite() || result == f64::INFINITY || result == f64::NEG_INFINITY {
        graph.nodes[id as usize] = Node::Const(result.to_bits());
        return true;
    }
    false
}

/// Strength reduction: replace expensive operations with cheaper equivalents.
fn strength_reduce(graph: &mut Graph, id: NodeId) -> bool {
    let node = graph.nodes[id as usize].clone();
    match node {
        // x^2 → x*x (eliminates powf extern call!)
        Node::Binary(BinOp::Pow, x, c) if graph.is_const(c, 2.0) => {
            graph.nodes[id as usize] = Node::Binary(BinOp::Mul, x, x);
            true
        }
        // x^3 → x*x*x — only safe if x*x already exists (lower ID)
        Node::Binary(BinOp::Pow, x, c) if graph.is_const(c, 3.0) => {
            // Check if Mul(x,x) already exists via dedup
            let x2_node = Node::Binary(BinOp::Mul, x, x);
            if let Some(&x2_id) = graph.dedup.get(&x2_node) {
                if x2_id < id {
                    graph.nodes[id as usize] = Node::Binary(BinOp::Mul, x2_id, x);
                    return true;
                }
            }
            false
        }
        // x^0.5 → sqrt(x)
        Node::Binary(BinOp::Pow, x, c) if graph.is_const(c, 0.5) => {
            graph.nodes[id as usize] = Node::Unary(UnaryOp::Sqrt, x);
            true
        }
        // x^1 → x
        Node::Binary(BinOp::Pow, x, c) if graph.is_const(c, 1.0) => {
            graph.nodes[id as usize] = graph.nodes[x as usize].clone();
            true
        }
        // x^0 → 1.0
        Node::Binary(BinOp::Pow, _x, c) if graph.is_const(c, 0.0) => {
            graph.nodes[id as usize] = Node::Const(1.0_f64.to_bits());
            true
        }
        // x^(-1) → 1/x (only if Const(1.0) already in graph at lower ID)
        Node::Binary(BinOp::Pow, x, c) if graph.is_const(c, -1.0) => {
            let one_node = Node::Const(1.0_f64.to_bits());
            if let Some(&one_id) = graph.dedup.get(&one_node) {
                if one_id < id {
                    graph.nodes[id as usize] = Node::Binary(BinOp::Div, one_id, x);
                    return true;
                }
            }
            false
        }
        // --x → x
        Node::Unary(UnaryOp::Neg, a) => {
            if let Node::Unary(UnaryOp::Neg, inner) = graph.nodes[a as usize] {
                graph.nodes[id as usize] = graph.nodes[inner as usize].clone();
                return true;
            }
            false
        }
        _ => false,
    }
}

/// Algebraic identity simplification.
fn algebraic_simplify(graph: &mut Graph, id: NodeId) -> bool {
    let node = graph.nodes[id as usize].clone();
    match node {
        // x + 0 → x
        Node::Binary(BinOp::Add, x, c) if graph.is_const(c, 0.0) => {
            graph.nodes[id as usize] = graph.nodes[x as usize].clone();
            true
        }
        // 0 + x → x
        Node::Binary(BinOp::Add, c, x) if graph.is_const(c, 0.0) => {
            graph.nodes[id as usize] = graph.nodes[x as usize].clone();
            true
        }
        // x - 0 → x
        Node::Binary(BinOp::Sub, x, c) if graph.is_const(c, 0.0) => {
            graph.nodes[id as usize] = graph.nodes[x as usize].clone();
            true
        }
        // x - x → 0
        Node::Binary(BinOp::Sub, a, b) if a == b => {
            graph.nodes[id as usize] = Node::Const(0.0_f64.to_bits());
            true
        }
        // x * 0 → 0
        Node::Binary(BinOp::Mul, _x, c) if graph.is_const(c, 0.0) => {
            graph.nodes[id as usize] = Node::Const(0.0_f64.to_bits());
            true
        }
        // 0 * x → 0
        Node::Binary(BinOp::Mul, c, _x) if graph.is_const(c, 0.0) => {
            graph.nodes[id as usize] = Node::Const(0.0_f64.to_bits());
            true
        }
        // x * 1 → x
        Node::Binary(BinOp::Mul, x, c) if graph.is_const(c, 1.0) => {
            graph.nodes[id as usize] = graph.nodes[x as usize].clone();
            true
        }
        // 1 * x → x
        Node::Binary(BinOp::Mul, c, x) if graph.is_const(c, 1.0) => {
            graph.nodes[id as usize] = graph.nodes[x as usize].clone();
            true
        }
        // x / 1 → x
        Node::Binary(BinOp::Div, x, c) if graph.is_const(c, 1.0) => {
            graph.nodes[id as usize] = graph.nodes[x as usize].clone();
            true
        }
        // x / x → 1  (for safe x; we skip — introduces NaN at x=0)
        // -(-x) → x
        Node::Unary(UnaryOp::Neg, inner) => {
            if let Node::Unary(UnaryOp::Neg, x) = graph.nodes[inner as usize] {
                graph.nodes[id as usize] = graph.nodes[x as usize].clone();
                return true;
            }
            false
        }
        // abs(abs(x)) → abs(x);   abs(-x) → abs(x);   abs(sqrt(x)) → sqrt(x).
        Node::Unary(UnaryOp::Abs, inner) => {
            match graph.nodes[inner as usize] {
                Node::Unary(UnaryOp::Abs, _) | Node::Unary(UnaryOp::Sqrt, _) => {
                    graph.nodes[id as usize] = graph.nodes[inner as usize].clone();
                    true
                }
                Node::Unary(UnaryOp::Neg, x) => {
                    // abs(-x) = abs(x)
                    let new_node = Node::Unary(UnaryOp::Abs, x);
                    graph.nodes[id as usize] = new_node;
                    true
                }
                _ => false,
            }
        }
        // log(exp(x)) → x.
        Node::Unary(UnaryOp::Log, inner) => {
            if let Node::Unary(UnaryOp::Exp, x) = graph.nodes[inner as usize] {
                graph.nodes[id as usize] = graph.nodes[x as usize].clone();
                return true;
            }
            false
        }
        // sqrt(x) * sqrt(x) → x  (valid for x ≥ 0, which sqrt's domain guarantees).
        Node::Binary(BinOp::Mul, a, b) if a == b => {
            if let Node::Unary(UnaryOp::Sqrt, x) = graph.nodes[a as usize] {
                graph.nodes[id as usize] = graph.nodes[x as usize].clone();
                return true;
            }
            false
        }
        _ => false,
    }
}

/// FMA detection: a*b + c → fma(a, b, c).
fn detect_fma(graph: &mut Graph, id: NodeId) -> bool {
    let node = graph.nodes[id as usize].clone();
    // (a * b) + c → fma(a, b, c)
    if let Node::Binary(BinOp::Add, mul_id, c) = node {
        if let Node::Binary(BinOp::Mul, a, b) = graph.nodes[mul_id as usize] {
            graph.nodes[id as usize] = Node::Fma(a, b, c);
            return true;
        }
        // c + (a * b) → fma(a, b, c)
        if let Node::Binary(BinOp::Mul, a, b) = graph.nodes[c as usize] {
            graph.nodes[id as usize] = Node::Fma(a, b, mul_id);
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_fold() {
        let mut g = Graph::new(InputSignature::empty());
        let a = g.constant(2.0);
        let b = g.constant(3.0);
        let sum = g.binary(BinOp::Add, a, b);
        g.outputs.push(sum);

        optimize(&mut g);
        assert!(g.is_const(sum, 5.0), "2+3 should fold to 5");
    }

    #[test]
    fn test_constant_fold_chain() {
        let mut g = Graph::new(InputSignature::empty());
        let a = g.constant(2.0);
        let b = g.constant(3.0);
        let c = g.constant(4.0);
        let ab = g.binary(BinOp::Mul, a, b); // 6
        let abc = g.binary(BinOp::Add, ab, c); // 10
        g.outputs.push(abc);

        optimize(&mut g);
        assert!(g.is_const(ab, 6.0));
        assert!(g.is_const(abc, 10.0));
    }

    #[test]
    fn test_strength_reduce_pow2() {
        let mut g = Graph::with_single_input("x", 1);
        let x = g.add(Node::Input(0));
        let two = g.constant(2.0);
        let pow = g.binary(BinOp::Pow, x, two);
        g.outputs.push(pow);

        optimize(&mut g);
        // Should be Mul(x, x) now
        assert!(matches!(g.nodes[pow as usize], Node::Binary(BinOp::Mul, a, b) if a == x && b == x));
    }

    #[test]
    fn test_strength_reduce_sqrt() {
        let mut g = Graph::with_single_input("x", 1);
        let x = g.add(Node::Input(0));
        let half = g.constant(0.5);
        let pow = g.binary(BinOp::Pow, x, half);
        g.outputs.push(pow);

        optimize(&mut g);
        assert!(matches!(g.nodes[pow as usize], Node::Unary(UnaryOp::Sqrt, a) if a == x));
    }

    #[test]
    fn test_algebraic_identity() {
        let mut g = Graph::with_single_input("x", 1);
        let x = g.add(Node::Input(0));
        let zero = g.constant(0.0);
        let one = g.constant(1.0);

        let x_plus_0 = g.binary(BinOp::Add, x, zero);
        let x_times_1 = g.binary(BinOp::Mul, x, one);
        let x_times_0 = g.binary(BinOp::Mul, x, zero);
        let x_minus_x = g.binary(BinOp::Sub, x, x);

        g.outputs.extend_from_slice(&[x_plus_0, x_times_1, x_times_0, x_minus_x]);

        optimize(&mut g);

        // x+0 → Input(0)
        assert!(matches!(g.nodes[x_plus_0 as usize], Node::Input(0)));
        // x*1 → Input(0)
        assert!(matches!(g.nodes[x_times_1 as usize], Node::Input(0)));
        // x*0 → 0
        assert!(g.is_const(x_times_0, 0.0));
        // x-x → 0
        assert!(g.is_const(x_minus_x, 0.0));
    }

    #[test]
    fn test_fma_detection() {
        let mut g = Graph::new(InputSignature::from_named_sizes([("x", 1), ("u", 1)]));
        let x = g.add(Node::Input(0));  // x[0]
        let u = g.add(Node::Input(1));  // u[0]
        let c = g.constant(5.0);
        let mul = g.binary(BinOp::Mul, x, u);
        let add = g.binary(BinOp::Add, mul, c);
        g.outputs.push(add);

        optimize(&mut g);
        assert!(matches!(g.nodes[add as usize], Node::Fma(a, b, cc) if a == x && b == u && cc == c));
    }

    #[test]
    fn test_optimize_robertson_pow() {
        // Robertson ODE has x[1]^2 which should become x[1]*x[1].
        // Direct Graph construction — no Stack-IR involved.
        let mut g = Graph::new(InputSignature::from_named_sizes([("x", 3)]));
        let c = g.constant(3e7);
        let x1 = g.input(1);
        let two = g.constant(2.0);
        let x1_2 = g.binary(BinOp::Pow, x1, two);
        let out = g.binary(BinOp::Mul, c, x1_2);
        g.outputs.push(out);

        optimize(&mut g);

        // Verify correctness: 3e7 * 0.5^2 = 3e7 * 0.25 = 7500000
        let result = g.interpret(&[&[0.0, 0.5, 0.0]], &[]);
        assert!((result[0] - 7500000.0).abs() < 1e-4);

        // Verify no Pow nodes remain (should be Mul after strength reduction)
        let has_pow = g.nodes.iter().any(|n| matches!(n, Node::Binary(BinOp::Pow, _, _)));
        assert!(!has_pow, "Pow should be eliminated by strength reduction");
    }

    // ============================================================
    // Fuzz / property-based tests for optimizer correctness
    // ============================================================

    /// Build a random expression graph and verify optimize preserves semantics.
    fn verify_optimize_preserves_semantics(seed: u64) {
        let mut rng = SimpleRng(seed);
        let mut g = Graph::new(InputSignature::from_named_sizes([("x", 2), ("u", 1), ("t", 1)]));

        // Build random expression tree (depth 3-5).  Flat offsets:
        //   x[0]=0, x[1]=1, u[0]=2, t=3.
        let x0 = g.add(Node::Input(0));
        let x1 = g.add(Node::Input(1));
        let u0 = g.add(Node::Input(2));
        let t = g.add(Node::Input(3));
        let leaves = [x0, x1, u0, t];

        // Build 8 random intermediate nodes
        let mut pool: Vec<NodeId> = leaves.to_vec();
        let ops = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow];
        let uops = [UnaryOp::Neg, UnaryOp::Abs, UnaryOp::Sin, UnaryOp::Sqrt];

        for _ in 0..8 {
            let kind = rng.next() % 4;
            let node = match kind {
                0 => {
                    // Binary op on two random pool elements
                    let a = pool[rng.next() as usize % pool.len()];
                    let b = pool[rng.next() as usize % pool.len()];
                    let op = ops[rng.next() as usize % ops.len()];
                    g.binary(op, a, b)
                }
                1 => {
                    // Unary op
                    let a = pool[rng.next() as usize % pool.len()];
                    let op = uops[rng.next() as usize % uops.len()];
                    g.unary(op, a)
                }
                2 => {
                    // Constant
                    let vals = [0.0, 1.0, 2.0, 0.5, -1.0, 3.2, 1e-6];
                    g.constant(vals[rng.next() as usize % vals.len()])
                }
                _ => {
                    // Identity: x + 0, x * 1 (things optimizer should simplify)
                    let a = pool[rng.next() as usize % pool.len()];
                    let zero = g.constant(0.0);
                    g.binary(BinOp::Add, a, zero)
                }
            };
            pool.push(node);
        }

        // Pick 2 outputs
        let o1 = pool[rng.next() as usize % pool.len()];
        let o2 = pool[rng.next() as usize % pool.len()];
        g.outputs = vec![o1, o2];

        // Test with multiple input vectors
        let test_inputs = [
            ([1.0, 2.0], [0.5], 0.1),
            ([0.0, 0.0], [1.0], 0.0),
            ([-1.0, 3.0], [0.7], 2.5),
            ([0.5, -0.5], [0.1], 1.0),
        ];

        let before: Vec<Vec<f64>> = test_inputs.iter()
            .map(|(x, u, t)| g.interpret(&[&x[..], &u[..], &[*t]], &[]))
            .collect();

        // Optimize
        let mut g_opt = g.clone();
        optimize(&mut g_opt);

        // Verify semantics preserved
        for (i, (x, u, t)) in test_inputs.iter().enumerate() {
            let after = g_opt.interpret(&[&x[..], &u[..], &[*t]], &[]);
            for (j, (b, a)) in before[i].iter().zip(after.iter()).enumerate() {
                if b.is_nan() && a.is_nan() { continue; }
                if b.is_infinite() && a.is_infinite() && b.signum() == a.signum() { continue; }
                let tol = 1e-10 * b.abs().max(1.0);
                assert!(
                    (b - a).abs() < tol,
                    "seed={} input={} output={}: before={} after={}", seed, i, j, b, a
                );
            }
        }
    }

    /// Simple deterministic PRNG for reproducible tests.
    struct SimpleRng(u64);
    impl SimpleRng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    #[test]
    fn test_optimize_fuzz_500_seeds() {
        for seed in 0..500 {
            verify_optimize_preserves_semantics(seed);
        }
    }

    #[test]
    fn test_optimize_edge_case_all_constants() {
        let mut g = Graph::new(InputSignature::empty());
        let a = g.constant(2.0);
        let b = g.constant(3.0);
        let c = g.constant(4.0);
        let ab = g.binary(BinOp::Add, a, b);
        let abc = g.binary(BinOp::Mul, ab, c);
        let sin_abc = g.unary(UnaryOp::Sin, abc);
        g.outputs.push(sin_abc);

        let before = g.interpret(&[], &[]);
        optimize(&mut g);
        let after = g.interpret(&[], &[]);

        // Should fold to a single constant: sin(20) = 0.9129...
        assert!((before[0] - after[0]).abs() < 1e-10);
        assert!(g.is_const(sin_abc, (20.0_f64).sin()));
    }

    #[test]
    fn test_optimize_identity_chains() {
        // x + 0 - 0 * 1 / 1 should all simplify
        let mut g = Graph::with_single_input("x", 1);
        let x = g.add(Node::Input(0));
        let zero = g.constant(0.0);
        let one = g.constant(1.0);

        let a = g.binary(BinOp::Add, x, zero);   // x + 0 → x
        let b = g.binary(BinOp::Sub, a, zero);    // x - 0 → x
        let c = g.binary(BinOp::Mul, b, one);     // x * 1 → x
        let d = g.binary(BinOp::Div, c, one);     // x / 1 → x
        let e = g.binary(BinOp::Pow, d, one);     // x ^ 1 → x
        g.outputs.push(e);

        let before = g.interpret(&[&[7.0]], &[]);
        optimize(&mut g);
        let after = g.interpret(&[&[7.0]], &[]);
        assert!((before[0] - 7.0).abs() < 1e-10);
        assert!((after[0] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_optimize_pow_variants() {
        let mut g = Graph::with_single_input("x", 1);
        let x = g.add(Node::Input(0));
        let c0 = g.constant(0.0);
        let c1 = g.constant(1.0);
        let c2 = g.constant(2.0);
        let chalf = g.constant(0.5);

        let p0 = g.binary(BinOp::Pow, x, c0);     // x^0 → 1
        let p1 = g.binary(BinOp::Pow, x, c1);     // x^1 → x
        let p2 = g.binary(BinOp::Pow, x, c2);     // x^2 → x*x
        let phalf = g.binary(BinOp::Pow, x, chalf); // x^0.5 → sqrt

        g.outputs = vec![p0, p1, p2, phalf];

        optimize(&mut g);

        let result = g.interpret(&[&[9.0]], &[]);
        assert!((result[0] - 1.0).abs() < 1e-10, "x^0 should be 1");
        assert!((result[1] - 9.0).abs() < 1e-10, "x^1 should be x");
        assert!((result[2] - 81.0).abs() < 1e-10, "x^2 should be 81");
        assert!((result[3] - 3.0).abs() < 1e-10, "x^0.5 should be 3");

        // Verify no Pow nodes for 0, 1, 2, 0.5
        let pow_count = g.nodes.iter().filter(|n| matches!(n, Node::Binary(BinOp::Pow, _, _))).count();
        assert_eq!(pow_count, 0, "All Pow should be eliminated");
    }

    #[test]
    fn test_optimize_double_neg() {
        let mut g = Graph::with_single_input("x", 1);
        let x = g.add(Node::Input(0));
        let neg1 = g.unary(UnaryOp::Neg, x);
        let neg2 = g.unary(UnaryOp::Neg, neg1);
        g.outputs.push(neg2);

        optimize(&mut g);
        let result = g.interpret(&[&[42.0]], &[]);
        assert!((result[0] - 42.0).abs() < 1e-10, "--x should be x");
    }

    #[test]
    fn test_optimize_with_params() {
        let mut g = Graph::with_single_input("x", 1);
        g.n_params = 2;
        g.param_defaults = vec![3.0, 7.0];
        g.param_names = vec!["a".into(), "b".into()];

        let x = g.add(Node::Input(0));
        let a = g.add(Node::Param(0));
        let b = g.add(Node::Param(1));
        let zero = g.constant(0.0);

        // a*x + b + 0 → should simplify the +0
        let ax = g.binary(BinOp::Mul, a, x);
        let axb = g.binary(BinOp::Add, ax, b);
        let axb0 = g.binary(BinOp::Add, axb, zero);
        g.outputs.push(axb0);

        optimize(&mut g);

        // a*x + b = 3*5 + 7 = 22
        let result = g.interpret(&[&[5.0]], &[3.0, 7.0]);
        assert!((result[0] - 22.0).abs() < 1e-10);
    }

    // ============================================================
    // Full pipeline verification: Stack IR → Graph → optimize → interpret
    // ============================================================

    #[test]
    fn test_dce_drops_unreachable_nodes() {
        let mut g = Graph::with_single_input("x", 1);
        let x = g.add(Node::Input(0));
        let seven = g.constant(7.0);
        let _dead1 = g.binary(BinOp::Mul, x, seven);
        let _dead2 = g.unary(UnaryOp::Sin, x);
        let one = g.constant(1.0);
        let live = g.binary(BinOp::Add, x, one);
        g.outputs.push(live);

        let n_before = g.nodes.len();
        dead_code_elimination(&mut g);
        assert!(g.nodes.len() < n_before, "DCE should shrink the node list");
        // Live computation still evaluates correctly.
        let r = g.interpret(&[&[2.0]], &[]);
        assert!((r[0] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_algebraic_neg_neg() {
        let mut g = Graph::with_single_input("x", 1);
        let x = g.add(Node::Input(0));
        let neg1 = g.unary(UnaryOp::Neg, x);
        let neg2 = g.unary(UnaryOp::Neg, neg1);
        g.outputs.push(neg2);
        optimize(&mut g);
        // After optimize --x → x  (checked via evaluation; IDs may have shifted).
        let r = g.interpret(&[&[3.5]], &[]);
        assert!((r[0] - 3.5).abs() < 1e-12);
    }

    #[test]
    fn test_algebraic_log_exp() {
        let mut g = Graph::with_single_input("x", 1);
        let x = g.add(Node::Input(0));
        let ex = g.unary(UnaryOp::Exp, x);
        let lx = g.unary(UnaryOp::Log, ex);
        g.outputs.push(lx);
        optimize(&mut g);
        let r = g.interpret(&[&[-1.7]], &[]);
        assert!((r[0] - (-1.7)).abs() < 1e-12);
    }

    #[test]
    fn test_algebraic_sqrt_squared() {
        let mut g = Graph::with_single_input("x", 1);
        let x = g.add(Node::Input(0));
        let sx = g.unary(UnaryOp::Sqrt, x);
        let sq = g.binary(BinOp::Mul, sx, sx);
        g.outputs.push(sq);
        optimize(&mut g);
        let r = g.interpret(&[&[4.2]], &[]);
        assert!((r[0] - 4.2).abs() < 1e-12);
    }

    #[test]
    fn test_algebraic_abs_neg() {
        let mut g = Graph::with_single_input("x", 1);
        let x = g.add(Node::Input(0));
        let neg = g.unary(UnaryOp::Neg, x);
        let abs_neg = g.unary(UnaryOp::Abs, neg);
        g.outputs.push(abs_neg);
        optimize(&mut g);
        let r = g.interpret(&[&[-3.0]], &[]);
        assert!((r[0] - 3.0).abs() < 1e-12);
    }

    // Note: the original three-way full-pipeline test compared the Stack-IR
    // interpreter against the graph path.  Stack-IR has been retired, so
    // the individual passes are exercised by the fuzz test above, which
    // round-trips 500 random graphs through `optimize()` and verifies that
    // evaluation results are preserved.

    // ============================================================
    // canonicalize(): value-numbering rebuild + folds
    // ============================================================

    #[test]
    fn canonicalize_merges_commuted_operands() {
        // a*b and b*a are distinct nodes under plain hash-consing; the
        // canonical operand order value-numbers them into one.
        let mut g = Graph::new(InputSignature::from_named_sizes([("x", 2)]));
        let a = g.input(0);
        let b = g.input(1);
        let ab = g.binary(BinOp::Mul, a, b);
        let ba = g.binary(BinOp::Mul, b, a);
        let s = g.binary(BinOp::Add, ab, ba); // 2ab, computed twice
        g.outputs = vec![s];
        assert_ne!(ab, ba, "precondition: hash-consing alone does not merge");

        canonicalize(&mut g);
        // One Mul, one Add (plus the two inputs) — the commuted twin is gone.
        let muls = g.nodes.iter()
            .filter(|n| matches!(n, Node::Binary(BinOp::Mul, _, _))).count();
        assert_eq!(muls, 1, "commuted Mul twins should merge");
        let r = g.interpret(&[&[3.0, 5.0]], &[]);
        assert!((r[0] - 30.0).abs() < 1e-12);
    }

    #[test]
    fn canonicalize_merges_optimize_clone_duplicates() {
        // optimize() rewrites `x + 0 -> x` by CLONING node x's content in
        // place, leaving two nodes with identical content. The rebuild merges
        // them so the tape evaluates the expression once.
        let mut g = Graph::new(InputSignature::from_named_sizes([("x", 1)]));
        let x = g.input(0);
        let sin = g.unary(UnaryOp::Sin, x);
        let zero = g.constant(0.0);
        let sin_plus_0 = g.binary(BinOp::Add, sin, zero);
        // Consume both the original and the (to-be-cloned) alias.
        let prod = g.binary(BinOp::Mul, sin, sin_plus_0);
        g.outputs = vec![prod];

        optimize(&mut g); // sin_plus_0 becomes a clone of the Sin node
        let sin_count_before = g.nodes.iter()
            .filter(|n| matches!(n, Node::Unary(UnaryOp::Sin, _))).count();
        assert!(sin_count_before >= 2, "precondition: optimize leaves a clone");

        canonicalize(&mut g);
        let sin_count = g.nodes.iter()
            .filter(|n| matches!(n, Node::Unary(UnaryOp::Sin, _))).count();
        assert_eq!(sin_count, 1, "duplicate Sin clones should merge");
        let r = g.interpret(&[&[0.5]], &[]);
        let expect = 0.5_f64.sin() * 0.5_f64.sin();
        assert!((r[0] - expect).abs() < 1e-15);
    }

    #[test]
    fn canonicalize_folds_select_and_cmp() {
        let mut g = Graph::new(InputSignature::from_named_sizes([("x", 1)]));
        let x = g.input(0);
        let c1 = g.constant(1.0);
        let c2 = g.constant(2.0);
        // Cmp of constants folds; Select under it picks the branch; the
        // same-branch select collapses without a condition.
        let cond = g.cmp(CmpOp::Gt, c2, c1);     // -> 1.0
        let sel = g.select(cond, x, c1);          // -> x
        let same = g.select(x, sel, sel);         // -> sel -> x
        g.outputs = vec![same];

        canonicalize(&mut g);
        assert!(!g.nodes.iter().any(|n| matches!(n, Node::Select(..) | Node::Cmp(..))),
            "constant-cond and same-branch selects should fold away");
        let r = g.interpret(&[&[7.5]], &[]);
        assert!((r[0] - 7.5).abs() < 1e-15);
    }

    // ============================================================
    // reassociate_chains(): Add/Fma chains -> fused Reduce/Dot
    // ============================================================

    /// Build an n-term mul-add chain the way optimize() leaves an AD row:
    /// `Fma(a1, b1, Fma(a2, b2, ... Fma(an, bn, seed)))`.
    fn fma_chain(g: &mut Graph, terms: &[(NodeId, NodeId)], seed: NodeId) -> NodeId {
        let mut acc = seed;
        for &(a, b) in terms {
            acc = g.add(Node::Fma(a, b, acc));
        }
        acc
    }

    #[test]
    fn reassociate_pure_product_chain_to_dot() {
        // 5 products + a zero seed -> one Dot after the pipeline.
        let mut g = Graph::new(InputSignature::from_named_sizes([("x", 5), ("u", 5)]));
        let terms: Vec<(NodeId, NodeId)> = (0..5)
            .map(|i| {
                let xi = g.input(i as u32);
                let ui = g.input(5 + i as u32);
                (xi, ui)
            })
            .collect();
        let muls: Vec<NodeId> = terms.iter()
            .map(|&(a, b)| g.binary(BinOp::Mul, a, b))
            .collect();
        // Plain Add chain over single-use Muls (the pre-FMA trace shape).
        let mut acc = muls[0];
        for &m in &muls[1..] {
            acc = g.binary(BinOp::Add, acc, m);
        }
        g.outputs = vec![acc];

        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let u = [0.5, -1.0, 2.0, 0.25, -2.0];
        let before = g.interpret(&[&x, &u], &[]);
        lower_for_tape(&mut g);
        assert!(g.nodes.iter().any(|n| matches!(n, Node::Dot(..))),
            "product chain should bundle into a Dot");
        assert!(!g.nodes.iter().any(|n| matches!(n, Node::Binary(BinOp::Add, _, _) | Node::Fma(..))),
            "chain interiors should be gone");
        let after = g.interpret(&[&x, &u], &[]);
        assert!((before[0] - after[0]).abs() < 1e-9 * before[0].abs().max(1.0));
    }

    #[test]
    fn reassociate_fma_chain_to_dot() {
        // The post-optimize() shape: nested Fma accumulation.
        let mut g = Graph::new(InputSignature::from_named_sizes([("x", 4), ("u", 4)]));
        let terms: Vec<(NodeId, NodeId)> = (0..4)
            .map(|i| {
                let xi = g.input(i as u32);
                let ui = g.input(4 + i as u32);
                (xi, ui)
            })
            .collect();
        let zero = g.constant(0.0);
        let chain = fma_chain(&mut g, &terms, zero);
        g.outputs = vec![chain];

        let x = [1.0, 2.0, 3.0, 4.0];
        let u = [0.5, -1.0, 2.0, 0.25];
        let before = g.interpret(&[&x, &u], &[]);
        lower_for_tape(&mut g);
        // zero seed is a plain term -> Reduce over [zero?]... the seed const
        // joins as a plain term, so this lowers to Reduce(Sum) or Dot+const;
        // either way the chain interiors must be gone and the value preserved.
        assert!(g.nodes.iter().any(|n| matches!(n, Node::Dot(..) | Node::Reduce(ReduceOp::Sum, _))),
            "fma chain should bundle");
        assert!(!g.nodes.iter().any(|n| matches!(n, Node::Fma(..))),
            "fma interiors should be gone");
        let after = g.interpret(&[&x, &u], &[]);
        assert!((before[0] - after[0]).abs() < 1e-9 * before[0].abs().max(1.0));
    }

    #[test]
    fn reassociate_mixed_chain_to_reduce() {
        // Plain leaves mixed in -> Reduce::Sum.
        let mut g = Graph::new(InputSignature::from_named_sizes([("x", 5)]));
        let xs: Vec<NodeId> = (0..5).map(|i| g.input(i)).collect();
        let s = g.unary(UnaryOp::Sin, xs[0]);
        let mut acc = g.binary(BinOp::Add, s, xs[1]);
        acc = g.binary(BinOp::Add, acc, xs[2]);
        acc = g.binary(BinOp::Add, acc, xs[3]);
        acc = g.binary(BinOp::Add, acc, xs[4]);
        g.outputs = vec![acc];

        let x = [0.3, 1.0, 2.0, 3.0, 4.0];
        let before = g.interpret(&[&x], &[]);
        lower_for_tape(&mut g);
        assert!(g.nodes.iter().any(|n| matches!(n, Node::Reduce(ReduceOp::Sum, args) if args.len() == 5)),
            "5-term add chain should bundle into one Reduce");
        let after = g.interpret(&[&x], &[]);
        assert!((before[0] - after[0]).abs() < 1e-12 * before[0].abs().max(1.0));
    }

    #[test]
    fn reassociate_respects_discretizing_gate() {
        // The same chain, but its value feeds a banded comparison — the gate
        // must leave it untouched (a regrouped ULP could flip the band edge).
        let mut g = Graph::new(InputSignature::from_named_sizes([("x", 5)]));
        let xs: Vec<NodeId> = (0..5).map(|i| g.input(i)).collect();
        let mut acc = g.binary(BinOp::Add, xs[0], xs[1]);
        acc = g.binary(BinOp::Add, acc, xs[2]);
        acc = g.binary(BinOp::Add, acc, xs[3]);
        acc = g.binary(BinOp::Add, acc, xs[4]);
        let c = g.constant(10.0);
        let eq = g.cmp(CmpOp::Eq, acc, c);
        g.outputs = vec![eq];

        lower_for_tape(&mut g);
        assert!(!g.nodes.iter().any(|n| matches!(n, Node::Reduce(..))),
            "a chain feeding a comparison must not be reassociated");
    }

    #[test]
    fn reassociate_keeps_short_chains() {
        // Below JIT_REASSOC_MIN_TERMS the unrolled form stays.
        let mut g = Graph::new(InputSignature::from_named_sizes([("x", 3)]));
        let xs: Vec<NodeId> = (0..3).map(|i| g.input(i)).collect();
        let a1 = g.binary(BinOp::Add, xs[0], xs[1]);
        let a2 = g.binary(BinOp::Add, a1, xs[2]);
        g.outputs = vec![a2];

        lower_for_tape(&mut g);
        assert!(!g.nodes.iter().any(|n| matches!(n, Node::Reduce(..) | Node::Dot(..))),
            "3-term chain stays unrolled");
    }

    #[test]
    fn canonicalize_reduce_and_dot_folds() {
        let mut g = Graph::new(InputSignature::from_named_sizes([("x", 2)]));
        let x0 = g.input(0);
        let x1 = g.input(1);
        let c2 = g.constant(2.0);
        let c3 = g.constant(3.0);
        // Only Product singletons collapse (Sum: sign-of-zero; Min/Max: NaN
        // is ignored by the fold, so the identity would become a NaN);
        // all-const forms fold.
        let single_prod = g.reduce(ReduceOp::Product, vec![x0]);
        let single_max = g.reduce(ReduceOp::Max, vec![x1]);
        let const_red = g.reduce(ReduceOp::Sum, vec![c2, c3]);
        let const_dot = g.dot(vec![c2, c3], vec![c3, c2]);
        g.outputs = vec![single_prod, single_max, const_red, const_dot];

        canonicalize(&mut g);
        assert!(!g.nodes.iter().any(|n| matches!(n, Node::Dot(..))),
            "the all-const dot should fold");
        assert!(!g.nodes.iter().any(|n| matches!(n, Node::Reduce(ReduceOp::Product, _))),
            "the single-arg Product should collapse");
        assert!(g.nodes.iter().any(|n| matches!(n, Node::Reduce(ReduceOp::Max, _))),
            "the single-arg Max must stay (max(-inf, NaN) is -inf, not NaN)");
        let r = g.interpret(&[&[3.0, 4.0]], &[]);
        assert!((r[0] - 3.0).abs() < 1e-12, "single-arg product collapses: {r:?}");
        assert!((r[1] - 4.0).abs() < 1e-12, "singleton max value: {r:?}");
        assert!((r[2] - 5.0).abs() < 1e-12, "const reduce folds: {r:?}");
        assert!((r[3] - 12.0).abs() < 1e-12, "const dot folds: {r:?}");
    }
}
