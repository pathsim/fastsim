//! Tape: the SSA graph's fast evaluator.
//!
//! `Tape` is a flat, cache-friendly lowering of an `ssa::graph::Graph` with
//! liveness-driven work-slot reuse; `InterpretedFn` is the public
//! compiled-function handle backed by it. This is the IR's own executable form
//! (the recursive `Graph::interpret` is the slow reference oracle, this is the
//! fast path), built from the same op vocabulary. The differential fuzzer below
//! pins `interpret == tape` bit-exact.

use super::graph::*;
use crate::constants::JIT_FLOAT_EQ_TOL;
use super::op::code as op;
use super::op::{binary_opcode, cmp_opcode, reduce_code, unary_opcode};

// ======================================================================================
// Compiled tape (cache-friendly linear evaluation)
// ======================================================================================


#[derive(Clone, Copy)]
struct TapeOp {
    opcode: u8,
    _pad: [u8; 3],
    /// Destination work slot this op writes to. With liveness-based slot reuse
    /// this is no longer the op's position — a slot freed by a dying operand is
    /// recycled, so `dst` may alias an earlier op's slot (in-place evaluation).
    dst: u32,
    arg0: u32,
    arg1: u32,
    arg2: u32,
}

#[derive(Clone)]
struct Tape {
    ops: Vec<TapeOp>,
    consts: Vec<f64>,
    /// Flattened operand lists for variadic `REDUCE`/`DOT` ops. Entries are work
    /// slots (post-allocation), referenced as `[arg1 .. arg1 + arg2]`.
    arg_pool: Vec<u32>,
    outputs: Vec<u32>,
    n_work: usize,
    /// Size of the per-eval gather scratch: `2 * max(Dot/Reduce arity)`. `Dot`
    /// gathers its two operand lists into contiguous halves and `Reduce` its one
    /// list, so the canonical `op::dot` / `op::reduce` (4-lane, auto-vectorisable)
    /// run over a contiguous slice instead of the scattered work-slot gather.
    scratch_len: usize,
    /// Per input slot: minimum slice length a caller must supply, i.e.
    /// `1 + max(element index read)` over the tape's `INPUT` ops (0 = the slot
    /// is never read). `eval` indexes inputs with `get_unchecked`, so
    /// `InterpretedFn::call`/`call_into` verify the caller's slices against
    /// this BEFORE evaluating — an undersized slice is a caller bug and must
    /// fail loudly instead of reading out of bounds.
    min_input_lens: Vec<u32>,
}

/// Liveness-driven work-slot allocation. Returns `slot_of[node] = work slot`,
/// reusing a slot as soon as its node is dead (LIFO free list). Output nodes are
/// pinned live-to-end so their slots are never recycled. The peak slot count is
/// `max(slot_of) + 1`, typically far below `nodes.len()` for deep expressions.
fn allocate_work_slots(graph: &Graph) -> Vec<u32> {
    let n = graph.nodes.len();
    if n == 0 { return Vec::new(); }

    // last_use[c] = highest op index that reads node c (its own index if never
    // read), or u32::MAX if c is an output (pinned, never freed).
    let mut last_use: Vec<u32> = (0..n as u32).collect();
    for (i, node) in graph.nodes.iter().enumerate() {
        node.for_each_child(|c| {
            let c = c as usize;
            if (i as u32) > last_use[c] { last_use[c] = i as u32; }
        });
    }
    for &o in &graph.outputs { last_use[o as usize] = u32::MAX; }

    let mut slot_of = vec![0u32; n];
    let mut free: Vec<u32> = Vec::new();
    let mut n_slots: u32 = 0;
    let mut dying: Vec<usize> = Vec::with_capacity(3);
    for i in 0..n {
        // Free operands that die at this step *before* picking dst, so the op
        // can reuse a dying operand's slot (in-place). Dedup so a node used
        // twice (e.g. `x*x`) frees its single slot only once.
        dying.clear();
        graph.nodes[i].for_each_child(|c| {
            let c = c as usize;
            if last_use[c] == i as u32 && !dying.contains(&c) { dying.push(c); }
        });
        for &c in &dying { free.push(slot_of[c]); }

        slot_of[i] = free.pop().unwrap_or_else(|| { let s = n_slots; n_slots += 1; s });
    }
    slot_of
}

impl Tape {
    fn from_graph(graph: &Graph) -> Self {
        let n = graph.nodes.len();
        let mut ops = Vec::with_capacity(n);
        let mut consts = Vec::new();
        let mut arg_pool: Vec<u32> = Vec::new();
        let mut min_input_lens: Vec<u32> = vec![0; graph.signature.slots.len()];

        // ---- liveness + linear-scan slot allocation (CasADi-style) ----
        //
        // Instead of one work slot per node (`n_work == n`), reuse a slot once
        // its node is dead. `last_use[c]` is the position of the last op that
        // reads node `c`; output nodes are pinned live-to-end. A LIFO free list
        // recycles slots: a slot freed by a dying operand is preferentially
        // handed to the op that consumed it, giving in-place evaluation
        // (`w[a] = w[a] op w[b]`) and a far smaller, cache-friendlier buffer.
        let slot_of = allocate_work_slots(graph);
        let n_slots = slot_of.iter().copied().max().map(|m| m as usize + 1).unwrap_or(0);

        for (i, node) in graph.nodes.iter().enumerate() {
            let dst = slot_of[i];
            let s = |nid: NodeId| slot_of[nid as usize]; // node id -> work slot
            let tape_op = match node {
                Node::Const(bits) => {
                    let ci = consts.len() as u32;
                    consts.push(f64::from_bits(*bits));
                    TapeOp { opcode: op::CONST, _pad: [0; 3], dst, arg0: ci, arg1: 0, arg2: 0 }
                }
                Node::Input(flat) => {
                    let (slot, elem) = graph.signature.decode(*flat as usize);
                    // `decode` returns a `usize::MAX` sentinel for a flat index
                    // outside the signature. Emitting that into the tape would
                    // turn a graph-construction bug into an unchecked
                    // out-of-bounds read at eval time — fail at build instead.
                    assert!(
                        slot != usize::MAX,
                        "input node {} is outside the graph signature (total size {})",
                        flat, graph.signature.total_size
                    );
                    if slot >= min_input_lens.len() {
                        min_input_lens.resize(slot + 1, 0);
                    }
                    min_input_lens[slot] = min_input_lens[slot].max(elem as u32 + 1);
                    TapeOp { opcode: op::INPUT, _pad: [0; 3], dst, arg0: slot as u32, arg1: elem as u32, arg2: 0 }
                }
                Node::Param(i) => {
                    TapeOp { opcode: op::PARAM, _pad: [0; 3], dst, arg0: *i, arg1: 0, arg2: 0 }
                }
                Node::Binary(bop, a, b) => {
                    let opc = binary_opcode(*bop);
                    TapeOp { opcode: opc, _pad: [0; 3], dst, arg0: s(*a), arg1: s(*b), arg2: 0 }
                }
                Node::Unary(uop, a) => {
                    let opc = unary_opcode(*uop);
                    TapeOp { opcode: opc, _pad: [0; 3], dst, arg0: s(*a), arg1: 0, arg2: 0 }
                }
                Node::Cmp(cop, a, b) => {
                    let opc = cmp_opcode(*cop);
                    TapeOp { opcode: opc, _pad: [0; 3], dst, arg0: s(*a), arg1: s(*b), arg2: 0 }
                }
                Node::Select(c, th, el) => {
                    TapeOp { opcode: op::SELECT, _pad: [0; 3], dst, arg0: s(*c), arg1: s(*th), arg2: s(*el) }
                }
                Node::Fma(a, b, c) => {
                    TapeOp { opcode: op::FMA, _pad: [0; 3], dst, arg0: s(*a), arg1: s(*b), arg2: s(*c) }
                }
                Node::Reduce(rop, args) => {
                    let off = arg_pool.len() as u32;
                    arg_pool.extend(args.iter().map(|&a| s(a)));
                    let code = reduce_code(*rop);
                    TapeOp { opcode: op::REDUCE, _pad: [0; 3], dst, arg0: code, arg1: off, arg2: args.len() as u32 }
                }
                Node::Dot(a, b) => {
                    let off = arg_pool.len() as u32;
                    arg_pool.extend(a.iter().map(|&x| s(x)));
                    arg_pool.extend(b.iter().map(|&x| s(x)));
                    TapeOp { opcode: op::DOT, _pad: [0; 3], dst, arg0: a.len() as u32, arg1: off, arg2: 0 }
                }
            };
            ops.push(tape_op);
        }

        let max_arity = graph
            .nodes
            .iter()
            .map(|n| match n {
                Node::Reduce(_, args) => args.len(),
                Node::Dot(a, _) => a.len(),
                _ => 0,
            })
            .max()
            .unwrap_or(0);

        Tape {
            ops,
            consts,
            arg_pool,
            outputs: graph.outputs.iter().map(|&id| slot_of[id as usize]).collect(),
            n_work: n_slots,
            scratch_len: 2 * max_arity,
            min_input_lens,
        }
    }

    /// Evaluate the tape into the work buffer.
    ///
    /// # Safety invariants relied on by `unsafe { get_unchecked }` below
    ///
    /// - `w.len() >= self.n_work` — `InterpretedFn` sizes the work vector to
    ///   `tape.n_work` (the peak live-slot count from `allocate_work_slots`).
    ///   Each op writes to `w[op.dst]` with `op.dst < self.n_work`.
    /// - For every `Binary` / `Unary` / `Cmp` / `Select` / `Fma` op,
    ///   `arg{0,1,2}` are work slots of operands that are still live when this
    ///   op runs, so they are `< self.n_work` and in-bounds.  This is
    ///   guaranteed by the liveness analysis (an operand's slot is only freed
    ///   at or after its last use) and the SSA acyclicity from `Graph::add`,
    ///   preserved by `dead_code_elimination`.
    /// - `Const` ops use `arg0` as an index into `self.consts`; the
    ///   lowering in `Tape::from_graph` always pushes the const before
    ///   referencing it.
    /// - `Input` ops carry `(slot, elem)` resolved via `signature.decode`;
    ///   the caller is responsible for supplying matching slots.  An
    ///   out-of-range slot is a tape-construction bug.
    /// - `Param` ops carry an index into `self.params` (capped to the
    ///   graph's `n_params` at lowering time).
    ///
    /// Bounds-checked indexing previously dominated per-op cost (each op
    /// performed 2-3 array indexes); `get_unchecked` here roughly halves
    /// the dispatch overhead on transcendental-light workloads.
    #[inline]
    fn eval(&self, w: &mut [f64], scratch: &mut [f64], inputs: &[&[f64]], params: &[f64]) {
        unsafe {
            for op in self.ops.iter() {
                let a0 = op.arg0 as usize;
                let a1 = op.arg1 as usize;
                let value = match op.opcode {
                    op::CONST => *self.consts.get_unchecked(a0),
                    op::INPUT => *inputs.get_unchecked(a0).get_unchecked(a1),
                    op::PARAM => *params.get_unchecked(a0),
                    op::ADD => *w.get_unchecked(a0) + *w.get_unchecked(a1),
                    op::SUB => *w.get_unchecked(a0) - *w.get_unchecked(a1),
                    op::MUL => *w.get_unchecked(a0) * *w.get_unchecked(a1),
                    op::DIV => *w.get_unchecked(a0) / *w.get_unchecked(a1),
                    op::POW => w.get_unchecked(a0).powf(*w.get_unchecked(a1)),
                    op::MOD => *w.get_unchecked(a0) % *w.get_unchecked(a1),
                    op::MIN => w.get_unchecked(a0).min(*w.get_unchecked(a1)),
                    op::MAX => w.get_unchecked(a0).max(*w.get_unchecked(a1)),
                    op::ATAN2 => w.get_unchecked(a0).atan2(*w.get_unchecked(a1)),
                    op::HYPOT => w.get_unchecked(a0).hypot(*w.get_unchecked(a1)),
                    op::NEG  => -*w.get_unchecked(a0),
                    op::SIN  => w.get_unchecked(a0).sin(),
                    op::COS  => w.get_unchecked(a0).cos(),
                    op::TAN  => w.get_unchecked(a0).tan(),
                    op::ATAN => w.get_unchecked(a0).atan(),
                    op::SINH => w.get_unchecked(a0).sinh(),
                    op::COSH => w.get_unchecked(a0).cosh(),
                    op::TANH => w.get_unchecked(a0).tanh(),
                    op::EXP  => w.get_unchecked(a0).exp(),
                    op::LOG  => w.get_unchecked(a0).ln(),
                    op::LOG10 => w.get_unchecked(a0).log10(),
                    op::ABS   => w.get_unchecked(a0).abs(),
                    op::SQRT  => w.get_unchecked(a0).sqrt(),
                    op::SIGN  => crate::ssa::op::numpy_sign(*w.get_unchecked(a0)),
                    op::FLOOR => w.get_unchecked(a0).floor(),
                    op::ASIN  => w.get_unchecked(a0).asin(),
                    op::ACOS  => w.get_unchecked(a0).acos(),
                    op::ASINH => w.get_unchecked(a0).asinh(),
                    op::ACOSH => w.get_unchecked(a0).acosh(),
                    op::ATANH => w.get_unchecked(a0).atanh(),
                    op::CEIL  => w.get_unchecked(a0).ceil(),
                    op::ROUND => w.get_unchecked(a0).round(),
                    op::TRUNC => w.get_unchecked(a0).trunc(),
                    op::LOG2  => w.get_unchecked(a0).log2(),
                    op::LOG1P => w.get_unchecked(a0).ln_1p(),
                    op::EXPM1 => w.get_unchecked(a0).exp_m1(),
                    op::CBRT  => w.get_unchecked(a0).cbrt(),
                    op::ERF    => libm::erf(*w.get_unchecked(a0)),
                    op::ERFC   => libm::erfc(*w.get_unchecked(a0)),
                    op::LGAMMA => libm::lgamma(*w.get_unchecked(a0)),
                    op::TGAMMA => libm::tgamma(*w.get_unchecked(a0)),
                    op::DIGAMMA => digamma(*w.get_unchecked(a0)),
                    op::RAND_UNIFORM => rand_uniform(*w.get_unchecked(a0)),
                    op::CMP_GT => if *w.get_unchecked(a0) >  *w.get_unchecked(a1) { 1.0 } else { 0.0 },
                    op::CMP_GE => if *w.get_unchecked(a0) >= *w.get_unchecked(a1) { 1.0 } else { 0.0 },
                    op::CMP_LT => if *w.get_unchecked(a0) <  *w.get_unchecked(a1) { 1.0 } else { 0.0 },
                    op::CMP_LE => if *w.get_unchecked(a0) <= *w.get_unchecked(a1) { 1.0 } else { 0.0 },
                    op::CMP_EQ => if (*w.get_unchecked(a0) - *w.get_unchecked(a1)).abs() <  JIT_FLOAT_EQ_TOL { 1.0 } else { 0.0 },
                    op::CMP_NE => if (*w.get_unchecked(a0) - *w.get_unchecked(a1)).abs() >= JIT_FLOAT_EQ_TOL { 1.0 } else { 0.0 },
                    op::SELECT => {
                        let a2 = op.arg2 as usize;
                        if *w.get_unchecked(a0) != 0.0 { *w.get_unchecked(a1) } else { *w.get_unchecked(a2) }
                    }
                    op::FMA => {
                        let a2 = op.arg2 as usize;
                        w.get_unchecked(a0).mul_add(*w.get_unchecked(a1), *w.get_unchecked(a2))
                    }
                    op::REDUCE => {
                        // arg0 = reduce-op code, arg1 = pool offset, arg2 = count.
                        // Gather the scattered operands into a contiguous scratch,
                        // then run the canonical 4-lane reduction (shared with the
                        // interpreter and native builder, so bit-for-bit identical).
                        let start = a1;
                        let count = op.arg2 as usize;
                        for j in 0..count {
                            let k = *self.arg_pool.get_unchecked(start + j) as usize;
                            *scratch.get_unchecked_mut(j) = *w.get_unchecked(k);
                        }
                        let rop = match op.arg0 {
                            op::reduce::SUM => crate::ssa::op::ReduceOp::Sum,
                            op::reduce::PRODUCT => crate::ssa::op::ReduceOp::Product,
                            op::reduce::MIN => crate::ssa::op::ReduceOp::Min,
                            _ => crate::ssa::op::ReduceOp::Max,
                        };
                        crate::ssa::op::reduce(rop, scratch.get_unchecked(0..count))
                    }
                    op::DOT => {
                        // arg0 = term count k; arg1 = a-list offset; b-list at arg1+k.
                        // Gather both operand lists into the two contiguous halves of
                        // the scratch, then the canonical 4-lane `dot`.
                        let k = op.arg0 as usize;
                        let a_off = a1;
                        let b_off = a1 + k;
                        for j in 0..k {
                            let ai = *self.arg_pool.get_unchecked(a_off + j) as usize;
                            let bi = *self.arg_pool.get_unchecked(b_off + j) as usize;
                            *scratch.get_unchecked_mut(j) = *w.get_unchecked(ai);
                            *scratch.get_unchecked_mut(k + j) = *w.get_unchecked(bi);
                        }
                        crate::ssa::op::dot(scratch.get_unchecked(0..k), scratch.get_unchecked(k..2 * k))
                    }
                    _ => std::hint::unreachable_unchecked(),
                };
                *w.get_unchecked_mut(op.dst as usize) = value;
            }
        }
    }
}

// ======================================================================================
// InterpretedFn — public API backed by the flat tape
// ======================================================================================

/// Compiled function backed by a flat tape interpreter.
/// Pre-allocates a work vector for zero-alloc evaluation.
///
/// `Clone` produces an independent copy with its own work/scratch buffers, so
/// clones can be evaluated concurrently on different threads (the batch API's
/// per-run copies — issue #45).
#[derive(Clone)]
pub struct InterpretedFn {
    tape: Tape,
    pub signature: InputSignature,
    pub n_out: usize,
    pub params: Vec<f64>,
    pub param_names: Vec<String>,
    work: std::cell::RefCell<Vec<f64>>,
    /// Reused gather scratch for `Dot`/`Reduce` (sized `tape.scratch_len`).
    scratch: std::cell::RefCell<Vec<f64>>,
}

impl InterpretedFn {
    pub fn from_graph(graph: Graph) -> Self {
        // Tape-lowering pipeline: canonicalize (value-numbering rebuild,
        // commutative ordering, Select/Cmp/Fma/Reduce/Dot folds, DCE), then
        // Add/Fma-chain reassociation into fused Reduce/Dot kernels (gated
        // away from discretizing consumers), then cleanup. Tape-lowering
        // only: codegen consumes the IR from the un-transformed graphs and
        // stays readable/stable. Works on a local copy — the caller's graph
        // is untouched (NodeIds stay stable for tests/AD).
        let mut compacted = graph;
        super::optimize::lower_for_tape(&mut compacted);

        let tape = Tape::from_graph(&compacted);
        let n_work = tape.n_work;
        let scratch_len = tape.scratch_len;
        Self {
            n_out: tape.outputs.len(),
            signature: compacted.signature.clone(),
            params: compacted.param_defaults.clone(),
            param_names: compacted.param_names.clone(),
            work: std::cell::RefCell::new(vec![0.0; n_work]),
            scratch: std::cell::RefCell::new(vec![0.0; scratch_len]),
            tape,
        }
    }

    /// Verify the caller's input slices against the tape's per-slot read
    /// bounds. `eval` indexes inputs with `get_unchecked` (the documented
    /// safety contract), so this check is what makes `call`/`call_into` safe
    /// public APIs: an undersized or missing slot panics with a precise
    /// message instead of reading out of bounds. Cost is a handful of
    /// comparisons per call — noise next to the tape evaluation itself.
    #[inline]
    fn check_inputs(&self, inputs: &[&[f64]]) {
        for (slot, &need) in self.tape.min_input_lens.iter().enumerate() {
            if need == 0 {
                continue; // slot never read by this tape
            }
            let got = inputs.get(slot).map(|s| s.len()).unwrap_or(0);
            if got < need as usize {
                let name = self
                    .signature
                    .slots
                    .get(slot)
                    .map(|s| s.name.as_str())
                    .unwrap_or("?");
                panic!(
                    "input slot {slot} ('{name}') too short: got {got} element(s), \
                     the compiled function reads {need} (traced signature: {:?})",
                    self.signature
                        .slots
                        .iter()
                        .map(|s| (s.name.as_str(), s.size))
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    /// Evaluate with inputs in slot order.  Returns a new `Vec<f64>`.
    ///
    /// Panics if an input slice is shorter than what the compiled function
    /// reads from it (see `check_inputs`).
    pub fn call(&self, inputs: &[&[f64]]) -> Vec<f64> {
        self.check_inputs(inputs);
        let mut w = self.work.borrow_mut();
        let mut sc = self.scratch.borrow_mut();
        self.tape.eval(&mut w, &mut sc, inputs, &self.params);
        self.tape.outputs.iter().map(|&id| w[id as usize]).collect()
    }

    /// Evaluate into a caller-provided output buffer (no allocation).
    ///
    /// Panics if an input slice is shorter than what the compiled function
    /// reads from it (see `check_inputs`).
    pub fn call_into(&self, inputs: &[&[f64]], out: &mut [f64]) {
        self.check_inputs(inputs);
        let mut w = self.work.borrow_mut();
        let mut sc = self.scratch.borrow_mut();
        self.tape.eval(&mut w, &mut sc, inputs, &self.params);
        for (i, &id) in self.tape.outputs.iter().enumerate() {
            if i < out.len() { out[i] = w[id as usize]; }
        }
    }

    pub fn set_param(&mut self, index: usize, value: f64) {
        if index < self.params.len() {
            self.params[index] = value;
        }
    }

    pub fn set_param_by_name(&mut self, name: &str, value: f64) -> bool {
        if let Some(idx) = self.param_names.iter().position(|n| n == name) {
            self.params[idx] = value;
            true
        } else { false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_basic_interpret() {
        let sig = InputSignature::from_named_sizes([("x", 2)]);
        let mut g = Graph::new(sig);
        let x0 = g.input(0);
        let x1 = g.input(1);
        let two = g.constant(2.0);
        let prod = g.binary(BinOp::Mul, x0, two);
        let sum = g.binary(BinOp::Add, prod, x1);
        g.outputs.push(sum);
        let r = g.interpret(&[&[3.0, 4.0]], &[]);
        assert_eq!(r, vec![10.0]);
    }

    #[test]
    fn graph_multi_slot() {
        // slots: x=[a], y=[b, c] → output = a*b + c
        let sig = InputSignature::from_named_sizes([("x", 1), ("y", 2)]);
        let mut g = Graph::new(sig);
        let a = g.input(0);     // x[0]
        let b = g.input(1);     // y[0]
        let c = g.input(2);     // y[1]
        let ab = g.binary(BinOp::Mul, a, b);
        let res = g.binary(BinOp::Add, ab, c);
        g.outputs.push(res);
        assert_eq!(g.interpret(&[&[3.0], &[4.0, 5.0]], &[]), vec![17.0]);
    }

    #[test]
    fn graph_dedup() {
        let sig = InputSignature::from_named_sizes([("x", 1)]);
        let mut g = Graph::new(sig);
        let a = g.constant(1.0);
        let b = g.constant(1.0);
        assert_eq!(a, b);
    }

    #[test]
    fn tape_matches_interpret() {
        let sig = InputSignature::from_named_sizes([("x", 2)]);
        let mut g = Graph::new(sig);
        let x = g.input(0);
        let y = g.input(1);
        let s = g.binary(BinOp::Add, x, y);
        g.outputs.push(s);
        let f = InterpretedFn::from_graph(g);
        let out = f.call(&[&[1.5, 2.5]]);
        assert_eq!(out, vec![4.0]);
    }

    #[test]
    fn reduce_interpret_and_tape() {
        // All four reductions over the same 4-element input, checked against
        // hand values and cross-checked interpret == compiled tape.
        let sig = InputSignature::from_named_sizes([("x", 4)]);
        let mut g = Graph::new(sig);
        let xs: Vec<NodeId> = (0..4).map(|i| g.input(i)).collect();
        let s = g.reduce(ReduceOp::Sum, xs.clone());
        let p = g.reduce(ReduceOp::Product, xs.clone());
        let mn = g.reduce(ReduceOp::Min, xs.clone());
        let mx = g.reduce(ReduceOp::Max, xs.clone());
        g.outputs = vec![s, p, mn, mx];

        let input = [1.5, -2.0, 3.0, 0.5];
        let interp = g.interpret(&[&input], &[]);
        assert_eq!(interp, vec![3.0, -4.5, -2.0, 3.0]);

        let f = InterpretedFn::from_graph(g);
        assert_eq!(f.call(&[&input]), interp);
    }

    #[test]
    fn dot_interpret_and_tape() {
        // y = Σ aᵢ·xᵢ with a = [2, -1, 0.5], x from inputs.
        let sig = InputSignature::from_named_sizes([("x", 3)]);
        let mut g = Graph::new(sig);
        let a: Vec<NodeId> = [2.0, -1.0, 0.5].iter().map(|&c| g.constant(c)).collect();
        let x: Vec<NodeId> = (0..3).map(|i| g.input(i)).collect();
        let d = g.dot(a, x);
        g.outputs = vec![d];
        let input = [3.0, 4.0, 8.0];
        // 2*3 + (-1)*4 + 0.5*8 = 6 - 4 + 4 = 6
        let interp = g.interpret(&[&input], &[]);
        assert_eq!(interp, vec![6.0]);
        let f = InterpretedFn::from_graph(g);
        assert_eq!(f.call(&[&input]), interp);
    }

    #[test]
    fn reduce_dce_keeps_operands() {
        // A reduce reachable from outputs must keep all its operands alive
        // through dead-code elimination (operand renumbering included).
        let sig = InputSignature::from_named_sizes([("x", 3)]);
        let mut g = Graph::new(sig);
        let _dead = g.constant(99.0); // unreachable, forces a renumber
        let xs: Vec<NodeId> = (0..3).map(|i| g.input(i)).collect();
        let s = g.reduce(ReduceOp::Sum, xs);
        g.outputs = vec![s];
        let f = InterpretedFn::from_graph(g);
        assert_eq!(f.call(&[&[10.0, 20.0, 30.0]]), vec![60.0]);
    }

    #[test]
    #[should_panic(expected = "input slot 0 ('x') too short")]
    fn call_rejects_undersized_input_slot() {
        // The tape reads x[0..2]; a 1-element slice must panic loudly instead
        // of reading out of bounds (the eval loop uses get_unchecked).
        let sig = InputSignature::from_named_sizes([("x", 2)]);
        let mut g = Graph::new(sig);
        let a = g.input(0);
        let b = g.input(1);
        let s = g.binary(BinOp::Add, a, b);
        g.outputs.push(s);
        let f = InterpretedFn::from_graph(g);
        let _ = f.call(&[&[1.0]]);
    }

    #[test]
    #[should_panic(expected = "too short")]
    fn call_rejects_missing_input_slot() {
        // Two slots traced, only one supplied.
        let sig = InputSignature::from_named_sizes([("x", 1), ("t", 1)]);
        let mut g = Graph::new(sig);
        let a = g.input(0);
        let t = g.input(1);
        let s = g.binary(BinOp::Mul, a, t);
        g.outputs.push(s);
        let f = InterpretedFn::from_graph(g);
        let _ = f.call(&[&[1.0]]);
    }

    #[test]
    fn call_accepts_partially_used_slot() {
        // The tape only reads x[0] out of a 3-wide slot; a 1-element slice is
        // memory-safe and must keep working (no false positive).
        let sig = InputSignature::from_named_sizes([("x", 3)]);
        let mut g = Graph::new(sig);
        let a = g.input(0);
        let two = g.constant(2.0);
        let s = g.binary(BinOp::Mul, a, two);
        g.outputs.push(s);
        let f = InterpretedFn::from_graph(g);
        assert_eq!(f.call(&[&[4.0]]), vec![8.0]);
    }

    #[test]
    fn call_accepts_unread_trailing_slot() {
        // Slot 't' is never read → callers may omit it entirely.
        let sig = InputSignature::from_named_sizes([("x", 1), ("t", 1)]);
        let mut g = Graph::new(sig);
        let a = g.input(0);
        let s = g.binary(BinOp::Add, a, a);
        g.outputs.push(s);
        let f = InterpretedFn::from_graph(g);
        assert_eq!(f.call(&[&[3.0]]), vec![6.0]);
    }

    #[test]
    fn signature_decode() {
        let sig = InputSignature::from_named_sizes([("x", 3), ("t", 1)]);
        assert_eq!(sig.decode(0), (0, 0));
        assert_eq!(sig.decode(1), (0, 1));
        assert_eq!(sig.decode(2), (0, 2));
        assert_eq!(sig.decode(3), (1, 0));
        assert_eq!(sig.slot("x").unwrap().offset, 0);
        assert_eq!(sig.slot("t").unwrap().offset, 3);
    }

    // ==================================================================
    // Differential fuzzer: random SSA DAGs over the full op set, with the
    // recursive `interpret` as the reference oracle.
    //
    // Two invariants are checked across many random graphs:
    //   1. `interpret == tape` BIT-EXACT. The tape interpreter inlines its
    //      own arithmetic in the hot loop instead of calling the canonical
    //      `apply_*` functions; this catches any drift between the two.
    //   2. `interpret(raw) ≈ interpret(optimize(raw))` APPROX. Optimization
    //      (FMA fusion, x²→x·x strength reduction) legitimately changes the
    //      last bit, so this leg uses a relative/absolute tolerance.
    // ==================================================================

    /// Deterministic xorshift64* PRNG — no external dep, fully reproducible.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self { Rng(seed | 1) }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: usize) -> usize { (self.next_u64() % n as u64) as usize }
        /// A float in roughly [-range, range], occasionally exact small ints
        /// so `Eq`/`Mod`/`Pow` hit interesting cases.
        fn val(&mut self, range: f64) -> f64 {
            match self.below(8) {
                0 => 0.0,
                1 => 1.0,
                2 => -1.0,
                3 => 2.0,
                _ => {
                    let u = (self.next_u64() as f64) / (u64::MAX as f64); // [0,1]
                    (u * 2.0 - 1.0) * range
                }
            }
        }
    }

    /// Build a random valid SSA graph. Every operand references an earlier
    /// node, so the result is always a well-formed DAG.
    fn random_graph(rng: &mut Rng) -> Graph {
        let slot_sizes = [("x", 3usize), ("u", 2), ("t", 1)];
        let n_inputs: usize = slot_sizes.iter().map(|(_, s)| *s).sum();
        let n_params = 2usize;
        let mut g = Graph::new(InputSignature::from_named_sizes(slot_sizes));
        g.n_params = n_params;
        g.param_defaults = (0..n_params).map(|_| rng.val(3.0)).collect();
        g.param_names = (0..n_params).map(|i| format!("p{i}")).collect();

        // Seed pool: every input, every param, a few constants.
        let mut pool: Vec<NodeId> = Vec::new();
        for i in 0..n_inputs { pool.push(g.input(i as u32)); }
        for i in 0..n_params { pool.push(g.param(i as u32)); }
        for _ in 0..3 { pool.push(g.constant(rng.val(3.0))); }

        let n_extra = 8 + rng.below(40);
        for _ in 0..n_extra {
            let pick = |rng: &mut Rng, pool: &[NodeId]| pool[rng.below(pool.len())];
            let node = match rng.below(9) {
                0 => {
                    const OPS: [BinOp; 10] = [
                        BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow,
                        BinOp::Mod, BinOp::Min, BinOp::Max, BinOp::Atan2, BinOp::Hypot];
                    g.binary(OPS[rng.below(10)], pick(rng, &pool), pick(rng, &pool))
                }
                1 => {
                    const OPS: [UnaryOp; 32] = [
                        UnaryOp::Neg, UnaryOp::Sin, UnaryOp::Cos, UnaryOp::Tan, UnaryOp::Atan,
                        UnaryOp::Sinh, UnaryOp::Cosh, UnaryOp::Tanh, UnaryOp::Exp, UnaryOp::Log,
                        UnaryOp::Log10, UnaryOp::Abs, UnaryOp::Sqrt, UnaryOp::Sign, UnaryOp::Floor,
                        UnaryOp::Asin, UnaryOp::Acos, UnaryOp::Asinh, UnaryOp::Acosh, UnaryOp::Atanh,
                        UnaryOp::Ceil, UnaryOp::Round, UnaryOp::Trunc, UnaryOp::Log2, UnaryOp::Log1p,
                        UnaryOp::Expm1, UnaryOp::Cbrt, UnaryOp::Erf, UnaryOp::Erfc, UnaryOp::Lgamma,
                        UnaryOp::Tgamma, UnaryOp::RandUniform];
                    g.unary(OPS[rng.below(32)], pick(rng, &pool))
                }
                2 => {
                    const OPS: [CmpOp; 6] = [
                        CmpOp::Gt, CmpOp::Ge, CmpOp::Lt, CmpOp::Le, CmpOp::Eq, CmpOp::Ne];
                    g.cmp(OPS[rng.below(6)], pick(rng, &pool), pick(rng, &pool))
                }
                3 => g.select(pick(rng, &pool), pick(rng, &pool), pick(rng, &pool)),
                4 => g.add(Node::Fma(pick(rng, &pool), pick(rng, &pool), pick(rng, &pool))),
                5 | 6 => {
                    const OPS: [ReduceOp; 4] =
                        [ReduceOp::Sum, ReduceOp::Product, ReduceOp::Min, ReduceOp::Max];
                    let k = 1 + rng.below(5);
                    let args = (0..k).map(|_| pick(rng, &pool)).collect();
                    g.reduce(OPS[rng.below(4)], args)
                }
                7 => {
                    let k = 1 + rng.below(5);
                    let a = (0..k).map(|_| pick(rng, &pool)).collect();
                    let b = (0..k).map(|_| pick(rng, &pool)).collect();
                    g.dot(a, b)
                }
                _ => g.constant(rng.val(3.0)),
            };
            pool.push(node);
        }

        let n_out = 1 + rng.below(4);
        g.outputs = (0..n_out).map(|_| pool[rng.below(pool.len())]).collect();
        g
    }

    fn random_inputs(rng: &mut Rng) -> Vec<Vec<f64>> {
        vec![
            (0..3).map(|_| rng.val(3.0)).collect(),
            (0..2).map(|_| rng.val(3.0)).collect(),
            vec![rng.val(3.0)],
        ]
    }

    /// NaN-tolerant bit equality: identical bits, or both NaN (any payload).
    fn bits_eq(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    /// Approximate equality that treats NaN/±inf sensibly. Optimization may
    /// shift the last bit via FMA fusion, so allow a relative + absolute band.
    fn approx_eq(a: f64, b: f64) -> bool {
        if a.is_nan() && b.is_nan() { return true; }
        if a == b { return true; } // catches ±inf with matching sign
        let diff = (a - b).abs();
        diff <= 1e-9 + 1e-6 * a.abs().max(b.abs())
    }

    #[test]
    fn rand_uniform_is_in_range_deterministic_and_roughly_flat() {
        // Deterministic: same key → same draw.
        assert_eq!(rand_uniform(3.0), rand_uniform(3.0));
        // Distinct keys decorrelate (sanity, not a strict guarantee).
        assert_ne!(rand_uniform(3.0), rand_uniform(4.0));
        // Range [0, 1) and a rough uniform mean over a sweep.
        let mut sum = 0.0;
        let n = 100_000;
        for i in 0..n {
            let u = rand_uniform(i as f64);
            assert!((0.0..1.0).contains(&u), "rand_uniform out of range: {u}");
            sum += u;
        }
        let mean = sum / n as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean {mean} not near 0.5");
    }

    #[test]
    fn rand_uniform_traces_through_tape() {
        // A one-node RandUniform graph evaluates identically interpret vs tape.
        let sig = InputSignature::from_named_sizes([("k", 1)]);
        let mut g = Graph::new(sig);
        let k = g.input(0);
        let u = g.unary(UnaryOp::RandUniform, k);
        g.outputs = vec![u];
        let interp = g.interpret(&[&[42.0]], &[]);
        let f = InterpretedFn::from_graph(g);
        assert_eq!(f.call(&[&[42.0]]), interp);
        assert_eq!(interp[0], rand_uniform(42.0));
    }

    #[test]
    fn liveness_reuses_slots_in_linear_chain() {
        // A 50-deep accumulator chain: each intermediate dies as soon as the
        // next op reads it, so the live set never exceeds a handful of slots.
        let sig = InputSignature::from_named_sizes([("x", 1)]);
        let mut g = Graph::new(sig);
        let mut acc = g.input(0);
        for _ in 0..50 {
            let c = g.constant(1.5);
            acc = g.binary(BinOp::Add, acc, c);
        }
        g.outputs = vec![acc];
        let n_nodes = g.len();
        let slots = allocate_work_slots(&g);
        let n_slots = slots.iter().copied().max().unwrap() as usize + 1;
        assert!(n_slots <= 4, "linear chain should reuse slots: {n_slots} slots for {n_nodes} nodes");
        // And the result is still correct end-to-end through the reused tape.
        let f = InterpretedFn::from_graph(g);
        assert_eq!(f.call(&[&[10.0]]), vec![10.0 + 50.0 * 1.5]);
    }

    #[test]
    fn fuzz_tape_matches_interpret_bit_exact() {
        // `from_graph` runs the lowering pipeline (canonicalize + chain
        // reassociation) before building the tape, so the bit-exact oracle is
        // the recursive interpreter on the SAME lowered graph (the pipeline
        // is idempotent — from_graph's second run is a no-op). This pins the
        // tape's inlined hot-loop arithmetic against the canonical `apply_*`
        // semantics; raw-vs-lowered value preservation has its own
        // (tolerance-based) leg below.
        for seed in 0..2000u64 {
            let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1));
            let g = random_graph(&mut rng);
            let params = g.param_defaults.clone();
            let inputs = random_inputs(&mut rng);
            let in_refs: Vec<&[f64]> = inputs.iter().map(|v| v.as_slice()).collect();

            let mut canon = g.clone();
            super::super::optimize::lower_for_tape(&mut canon);
            let reference = canon.interpret(&in_refs, &params);
            let f = InterpretedFn::from_graph(g);
            let got = f.call(&in_refs);

            assert_eq!(reference.len(), got.len(), "seed {seed}: output arity");
            for (i, (&r, &t)) in reference.iter().zip(got.iter()).enumerate() {
                assert!(bits_eq(r, t),
                    "seed {seed} out[{i}]: interpret={r:?} ({:#x}) != tape={t:?} ({:#x})",
                    r.to_bits(), t.to_bits());
            }
        }
    }

    #[test]
    fn fuzz_canonicalize_preserves_semantics() {
        // Raw vs canonicalized evaluation. canonicalize() is value-exact by
        // design, but the tolerance leg also covers any future drift.
        for seed in 0..2000u64 {
            let mut rng = Rng::new(seed.wrapping_mul(0xA076_1D64_78BD_642F).wrapping_add(1));
            let g = random_graph(&mut rng);
            let params = g.param_defaults.clone();
            let inputs = random_inputs(&mut rng);
            let in_refs: Vec<&[f64]> = inputs.iter().map(|v| v.as_slice()).collect();

            let reference = g.interpret(&in_refs, &params);
            let mut canon = g.clone();
            let _ = super::super::optimize::canonicalize(&mut canon);
            let got = canon.interpret(&in_refs, &params);

            assert_eq!(reference.len(), got.len(), "seed {seed}: output arity");
            for (i, (&r, &c)) in reference.iter().zip(got.iter()).enumerate() {
                assert!(approx_eq(r, c),
                    "seed {seed} out[{i}]: raw={r:?} != canonical={c:?}");
            }
        }
    }

    #[test]
    fn fuzz_lower_for_tape_preserves_semantics() {
        // Raw vs the FULL lowering pipeline (canonicalize + chain
        // reassociation). Reassociation shifts ULPs, and its discretizing-
        // consumer gate guarantees those shifts only flow through continuous
        // ops — so a tolerance comparison is sound (no 0/1 flips possible).
        for seed in 0..2000u64 {
            let mut rng = Rng::new(seed.wrapping_mul(0x6C62_272E_07BB_0142).wrapping_add(1));
            let g = random_graph(&mut rng);
            let params = g.param_defaults.clone();
            let inputs = random_inputs(&mut rng);
            let in_refs: Vec<&[f64]> = inputs.iter().map(|v| v.as_slice()).collect();

            let reference = g.interpret(&in_refs, &params);
            let mut lowered = g.clone();
            let _ = super::super::optimize::lower_for_tape(&mut lowered);
            let got = lowered.interpret(&in_refs, &params);

            assert_eq!(reference.len(), got.len(), "seed {seed}: output arity");
            for (i, (&r, &c)) in reference.iter().zip(got.iter()).enumerate() {
                assert!(approx_eq(r, c),
                    "seed {seed} out[{i}]: raw={r:?} != lowered={c:?}");
            }
        }
    }

    #[test]
    fn fuzz_lower_for_tape_idempotent() {
        // A second pipeline run over an already-lowered graph must be a
        // structural no-op — `from_graph` relies on this (the bit-exact leg
        // interprets a once-lowered clone as its oracle).
        for seed in 0..500u64 {
            let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(7));
            let mut g = random_graph(&mut rng);
            super::super::optimize::lower_for_tape(&mut g);
            let nodes_after_first = g.nodes.clone();
            let outputs_after_first = g.outputs.clone();
            let changed = super::super::optimize::lower_for_tape(&mut g);
            assert!(!changed, "seed {seed}: second lower_for_tape reported changes");
            assert!(g.nodes == nodes_after_first && g.outputs == outputs_after_first,
                "seed {seed}: second lower_for_tape altered the graph");
        }
    }

    #[test]
    fn fuzz_optimize_preserves_semantics() {
        for seed in 0..2000u64 {
            let mut rng = Rng::new(seed.wrapping_mul(0xD1B5_4A32_D192_ED03).wrapping_add(1));
            let g = random_graph(&mut rng);
            let params = g.param_defaults.clone();
            let inputs = random_inputs(&mut rng);
            let in_refs: Vec<&[f64]> = inputs.iter().map(|v| v.as_slice()).collect();

            let reference = g.interpret(&in_refs, &params);
            let mut opt = g.clone();
            super::super::optimize::optimize(&mut opt);
            let got = opt.interpret(&in_refs, &params);

            assert_eq!(reference.len(), got.len(), "seed {seed}: output arity");
            for (i, (&r, &o)) in reference.iter().zip(got.iter()).enumerate() {
                assert!(approx_eq(r, o),
                    "seed {seed} out[{i}]: raw={r:?} != optimized={o:?}");
            }
        }
    }
}
