//! `Builder`: a numeric backend abstraction that lets a block's math be
//! written ONCE and instantiated two ways (the "2b" single-source approach):
//!
//!   - `F64Builder` — operations are plain `f64` arithmetic. Monomorphised and
//!     inlined, a generic block `eval::<F64Builder>` compiles to exactly the
//!     hand-written native closure, so the runtime hot path keeps native speed.
//!   - `GraphBuilder` — operations record SSA nodes into a `jit::graph::Graph`.
//!     `eval::<GraphBuilder>` yields the op-graph that drives IR / codegen /
//!     verification.
//!
//! Both backends defer to the canonical `graph::apply_binary` / `apply_unary` /
//! `apply_cmp`, so native runtime and lowered graph agree by construction (not
//! by hand-kept mirroring).

use std::cell::RefCell;

use crate::ssa::graph::{
    apply_binary, apply_cmp, apply_unary, BinOp, CmpOp, Graph, NodeId, ReduceOp, UnaryOp,
};

/// Numeric backend. `N` is the value type (`f64` for native, `NodeId` for the
/// graph). All methods take `&self` so `GraphBuilder` can mutate via interior
/// mutability while `F64Builder` stays a stateless unit.
pub trait Builder {
    type N: Copy;

    fn cst(&self, v: f64) -> Self::N;

    // -- binary --
    fn add(&self, a: Self::N, b: Self::N) -> Self::N;
    fn sub(&self, a: Self::N, b: Self::N) -> Self::N;
    fn mul(&self, a: Self::N, b: Self::N) -> Self::N;
    fn div(&self, a: Self::N, b: Self::N) -> Self::N;
    fn powf(&self, a: Self::N, b: Self::N) -> Self::N;
    fn modulo(&self, a: Self::N, b: Self::N) -> Self::N;
    fn min(&self, a: Self::N, b: Self::N) -> Self::N;
    fn max(&self, a: Self::N, b: Self::N) -> Self::N;
    fn atan2(&self, a: Self::N, b: Self::N) -> Self::N;
    fn hypot(&self, a: Self::N, b: Self::N) -> Self::N;

    // -- unary --
    fn neg(&self, a: Self::N) -> Self::N;
    fn sin(&self, a: Self::N) -> Self::N;
    fn cos(&self, a: Self::N) -> Self::N;
    fn tan(&self, a: Self::N) -> Self::N;
    fn atan(&self, a: Self::N) -> Self::N;
    fn sinh(&self, a: Self::N) -> Self::N;
    fn cosh(&self, a: Self::N) -> Self::N;
    fn tanh(&self, a: Self::N) -> Self::N;
    fn exp(&self, a: Self::N) -> Self::N;
    fn ln(&self, a: Self::N) -> Self::N;
    fn log10(&self, a: Self::N) -> Self::N;
    fn abs(&self, a: Self::N) -> Self::N;
    fn sqrt(&self, a: Self::N) -> Self::N;
    fn sign(&self, a: Self::N) -> Self::N;
    fn floor(&self, a: Self::N) -> Self::N;
    /// Stateless PRNG: hash the key to a uniform `[0, 1)` (see `rand_uniform`).
    fn rand_uniform(&self, a: Self::N) -> Self::N;
    fn asin(&self, a: Self::N) -> Self::N;
    fn acos(&self, a: Self::N) -> Self::N;
    fn asinh(&self, a: Self::N) -> Self::N;
    fn acosh(&self, a: Self::N) -> Self::N;
    fn atanh(&self, a: Self::N) -> Self::N;
    fn ceil(&self, a: Self::N) -> Self::N;
    fn round(&self, a: Self::N) -> Self::N;
    fn trunc(&self, a: Self::N) -> Self::N;
    fn log2(&self, a: Self::N) -> Self::N;
    fn ln_1p(&self, a: Self::N) -> Self::N;
    fn exp_m1(&self, a: Self::N) -> Self::N;
    fn cbrt(&self, a: Self::N) -> Self::N;
    fn erf(&self, a: Self::N) -> Self::N;
    fn erfc(&self, a: Self::N) -> Self::N;
    fn lgamma(&self, a: Self::N) -> Self::N;
    fn tgamma(&self, a: Self::N) -> Self::N;

    // -- comparisons (result 0.0 / 1.0) --
    fn gt(&self, a: Self::N, b: Self::N) -> Self::N;
    fn ge(&self, a: Self::N, b: Self::N) -> Self::N;
    fn lt(&self, a: Self::N, b: Self::N) -> Self::N;
    fn le(&self, a: Self::N, b: Self::N) -> Self::N;
    fn eq(&self, a: Self::N, b: Self::N) -> Self::N;
    fn ne(&self, a: Self::N, b: Self::N) -> Self::N;

    /// Conditional select: `cond != 0 ? t : e`.
    fn select(&self, cond: Self::N, t: Self::N, e: Self::N) -> Self::N;

    // -- vector ops (fused) --

    /// Fused dot product `Σ aᵢ·bᵢ` over two equal-length operand lists. An empty
    /// list is `0`, a single pair is one product. This is the canonical way for
    /// a block to express a matrix-vector row or an inner product: the native
    /// backend folds with `mul_add` and the graph backend records one `Dot`
    /// node, so both agree and downstream AD / optimization / codegen see the
    /// structure instead of an unrolled scalar chain.
    fn dot(&self, a: &[Self::N], b: &[Self::N]) -> Self::N;

    /// Fused reduction of `xs` under `op`. An empty list is the op's identity, a
    /// single element passes through. Native folds with `op.combine`; the graph
    /// backend records one `Reduce` node.
    fn reduce(&self, op: ReduceOp, xs: &[Self::N]) -> Self::N;
}

// ======================================================================================
// Native f64 backend
// ======================================================================================

/// Canonical native dot product over two contiguous `f64` slices, used by
/// `F64Builder::dot` and the native matvec closures (`affine_mv_into`,
/// `matrix_mv_into`).
///
/// Four independent `mul + add` accumulators break the reduction's dependency
/// chain so LLVM lowers the body to packed multiply/add (SSE2 is baseline on
/// x86-64, so this widens without needing an `+fma` target feature). We
/// deliberately use plain `a*b + acc`, NOT `f64::mul_add`: without a hardware
/// FMA target feature `mul_add` lowers to a libm `fma()` *call* per element,
/// which is far slower than two native instructions. The four-lane split
/// reassociates the sum, so the result can differ from the serial graph `Dot`
/// by a few ULPs (within the 1e-12 parity tolerance, the tradeoff every BLAS
/// makes); tails of length < 4 stay on the serial fold.
#[inline]
pub(crate) fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    crate::ssa::op::dot(a, b)
}

/// Stateless backend evaluating directly in `f64`. Mirrors the tape
/// interpreter so a generic `eval::<F64Builder>` matches it bit-for-bit.
#[derive(Clone, Copy, Default)]
pub struct F64Builder;

// The native math is NOT re-spelled here: each method names its op and defers to
// the canonical `apply_*` (graph.rs). The op variant is a compile-time constant
// and the `apply_*` are `#[inline]`, so monomorphised `eval::<F64Builder>` folds
// back to the bare `a.sin()` / `a + b` and keeps native hot-path speed, while the
// semantics live in exactly one place.
impl Builder for F64Builder {
    type N = f64;

    #[inline] fn cst(&self, v: f64) -> f64 { v }

    #[inline] fn add(&self, a: f64, b: f64) -> f64 { apply_binary(BinOp::Add, a, b) }
    #[inline] fn sub(&self, a: f64, b: f64) -> f64 { apply_binary(BinOp::Sub, a, b) }
    #[inline] fn mul(&self, a: f64, b: f64) -> f64 { apply_binary(BinOp::Mul, a, b) }
    #[inline] fn div(&self, a: f64, b: f64) -> f64 { apply_binary(BinOp::Div, a, b) }
    #[inline] fn powf(&self, a: f64, b: f64) -> f64 { apply_binary(BinOp::Pow, a, b) }
    #[inline] fn modulo(&self, a: f64, b: f64) -> f64 { apply_binary(BinOp::Mod, a, b) }
    #[inline] fn min(&self, a: f64, b: f64) -> f64 { apply_binary(BinOp::Min, a, b) }
    #[inline] fn max(&self, a: f64, b: f64) -> f64 { apply_binary(BinOp::Max, a, b) }
    #[inline] fn atan2(&self, a: f64, b: f64) -> f64 { apply_binary(BinOp::Atan2, a, b) }
    #[inline] fn hypot(&self, a: f64, b: f64) -> f64 { apply_binary(BinOp::Hypot, a, b) }

    #[inline] fn neg(&self, a: f64) -> f64 { apply_unary(UnaryOp::Neg, a) }
    #[inline] fn sin(&self, a: f64) -> f64 { apply_unary(UnaryOp::Sin, a) }
    #[inline] fn cos(&self, a: f64) -> f64 { apply_unary(UnaryOp::Cos, a) }
    #[inline] fn tan(&self, a: f64) -> f64 { apply_unary(UnaryOp::Tan, a) }
    #[inline] fn atan(&self, a: f64) -> f64 { apply_unary(UnaryOp::Atan, a) }
    #[inline] fn sinh(&self, a: f64) -> f64 { apply_unary(UnaryOp::Sinh, a) }
    #[inline] fn cosh(&self, a: f64) -> f64 { apply_unary(UnaryOp::Cosh, a) }
    #[inline] fn tanh(&self, a: f64) -> f64 { apply_unary(UnaryOp::Tanh, a) }
    #[inline] fn exp(&self, a: f64) -> f64 { apply_unary(UnaryOp::Exp, a) }
    #[inline] fn ln(&self, a: f64) -> f64 { apply_unary(UnaryOp::Log, a) }
    #[inline] fn log10(&self, a: f64) -> f64 { apply_unary(UnaryOp::Log10, a) }
    #[inline] fn abs(&self, a: f64) -> f64 { apply_unary(UnaryOp::Abs, a) }
    #[inline] fn sqrt(&self, a: f64) -> f64 { apply_unary(UnaryOp::Sqrt, a) }
    #[inline] fn sign(&self, a: f64) -> f64 { apply_unary(UnaryOp::Sign, a) }
    #[inline] fn floor(&self, a: f64) -> f64 { apply_unary(UnaryOp::Floor, a) }
    #[inline] fn rand_uniform(&self, a: f64) -> f64 { apply_unary(UnaryOp::RandUniform, a) }
    #[inline] fn asin(&self, a: f64) -> f64 { apply_unary(UnaryOp::Asin, a) }
    #[inline] fn acos(&self, a: f64) -> f64 { apply_unary(UnaryOp::Acos, a) }
    #[inline] fn asinh(&self, a: f64) -> f64 { apply_unary(UnaryOp::Asinh, a) }
    #[inline] fn acosh(&self, a: f64) -> f64 { apply_unary(UnaryOp::Acosh, a) }
    #[inline] fn atanh(&self, a: f64) -> f64 { apply_unary(UnaryOp::Atanh, a) }
    #[inline] fn ceil(&self, a: f64) -> f64 { apply_unary(UnaryOp::Ceil, a) }
    #[inline] fn round(&self, a: f64) -> f64 { apply_unary(UnaryOp::Round, a) }
    #[inline] fn trunc(&self, a: f64) -> f64 { apply_unary(UnaryOp::Trunc, a) }
    #[inline] fn log2(&self, a: f64) -> f64 { apply_unary(UnaryOp::Log2, a) }
    #[inline] fn ln_1p(&self, a: f64) -> f64 { apply_unary(UnaryOp::Log1p, a) }
    #[inline] fn exp_m1(&self, a: f64) -> f64 { apply_unary(UnaryOp::Expm1, a) }
    #[inline] fn cbrt(&self, a: f64) -> f64 { apply_unary(UnaryOp::Cbrt, a) }
    #[inline] fn erf(&self, a: f64) -> f64 { apply_unary(UnaryOp::Erf, a) }
    #[inline] fn erfc(&self, a: f64) -> f64 { apply_unary(UnaryOp::Erfc, a) }
    #[inline] fn lgamma(&self, a: f64) -> f64 { apply_unary(UnaryOp::Lgamma, a) }
    #[inline] fn tgamma(&self, a: f64) -> f64 { apply_unary(UnaryOp::Tgamma, a) }

    #[inline] fn gt(&self, a: f64, b: f64) -> f64 { apply_cmp(CmpOp::Gt, a, b) }
    #[inline] fn ge(&self, a: f64, b: f64) -> f64 { apply_cmp(CmpOp::Ge, a, b) }
    #[inline] fn lt(&self, a: f64, b: f64) -> f64 { apply_cmp(CmpOp::Lt, a, b) }
    #[inline] fn le(&self, a: f64, b: f64) -> f64 { apply_cmp(CmpOp::Le, a, b) }
    #[inline] fn eq(&self, a: f64, b: f64) -> f64 { apply_cmp(CmpOp::Eq, a, b) }
    #[inline] fn ne(&self, a: f64, b: f64) -> f64 { apply_cmp(CmpOp::Ne, a, b) }

    #[inline] fn select(&self, cond: f64, t: f64, e: f64) -> f64 { if cond != 0.0 { t } else { e } }

    #[inline]
    fn dot(&self, a: &[f64], b: &[f64]) -> f64 {
        dot_f64(a, b)
    }

    #[inline]
    fn reduce(&self, op: ReduceOp, xs: &[f64]) -> f64 {
        crate::ssa::op::reduce(op, xs)
    }
}

// ======================================================================================
// Graph-recording backend
// ======================================================================================

/// Records operations as SSA nodes into a `Graph` via interior mutability.
/// Inputs and params are minted with `input` / `param`; constants and ops go
/// through the `Builder` trait. After `eval`, take the graph with `finish`.
pub struct GraphBuilder<'a> {
    g: &'a RefCell<Graph>,
}

impl<'a> GraphBuilder<'a> {
    pub fn new(g: &'a RefCell<Graph>) -> Self {
        Self { g }
    }

    /// Mint a read of flat input slot index `flat_idx`.
    pub fn input(&self, flat_idx: u32) -> NodeId {
        self.g.borrow_mut().input(flat_idx)
    }

    /// Mint a read of parameter index `idx`.
    pub fn param(&self, idx: u32) -> NodeId {
        self.g.borrow_mut().param(idx)
    }

    #[inline]
    fn bin(&self, op: BinOp, a: NodeId, b: NodeId) -> NodeId {
        self.g.borrow_mut().binary(op, a, b)
    }

    #[inline]
    fn un(&self, op: UnaryOp, a: NodeId) -> NodeId {
        self.g.borrow_mut().unary(op, a)
    }

    #[inline]
    fn cmp(&self, op: CmpOp, a: NodeId, b: NodeId) -> NodeId {
        self.g.borrow_mut().cmp(op, a, b)
    }
}

impl Builder for GraphBuilder<'_> {
    type N = NodeId;

    fn cst(&self, v: f64) -> NodeId { self.g.borrow_mut().constant(v) }

    fn add(&self, a: NodeId, b: NodeId) -> NodeId { self.bin(BinOp::Add, a, b) }
    fn sub(&self, a: NodeId, b: NodeId) -> NodeId { self.bin(BinOp::Sub, a, b) }
    fn mul(&self, a: NodeId, b: NodeId) -> NodeId { self.bin(BinOp::Mul, a, b) }
    fn div(&self, a: NodeId, b: NodeId) -> NodeId { self.bin(BinOp::Div, a, b) }
    fn powf(&self, a: NodeId, b: NodeId) -> NodeId { self.bin(BinOp::Pow, a, b) }
    fn modulo(&self, a: NodeId, b: NodeId) -> NodeId { self.bin(BinOp::Mod, a, b) }
    fn min(&self, a: NodeId, b: NodeId) -> NodeId { self.bin(BinOp::Min, a, b) }
    fn max(&self, a: NodeId, b: NodeId) -> NodeId { self.bin(BinOp::Max, a, b) }
    fn atan2(&self, a: NodeId, b: NodeId) -> NodeId { self.bin(BinOp::Atan2, a, b) }
    fn hypot(&self, a: NodeId, b: NodeId) -> NodeId { self.bin(BinOp::Hypot, a, b) }

    fn neg(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Neg, a) }
    fn sin(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Sin, a) }
    fn cos(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Cos, a) }
    fn tan(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Tan, a) }
    fn atan(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Atan, a) }
    fn sinh(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Sinh, a) }
    fn cosh(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Cosh, a) }
    fn tanh(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Tanh, a) }
    fn exp(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Exp, a) }
    fn ln(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Log, a) }
    fn log10(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Log10, a) }
    fn abs(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Abs, a) }
    fn sqrt(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Sqrt, a) }
    fn sign(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Sign, a) }
    fn floor(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Floor, a) }
    fn rand_uniform(&self, a: NodeId) -> NodeId { self.un(UnaryOp::RandUniform, a) }
    fn asin(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Asin, a) }
    fn acos(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Acos, a) }
    fn asinh(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Asinh, a) }
    fn acosh(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Acosh, a) }
    fn atanh(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Atanh, a) }
    fn ceil(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Ceil, a) }
    fn round(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Round, a) }
    fn trunc(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Trunc, a) }
    fn log2(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Log2, a) }
    fn ln_1p(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Log1p, a) }
    fn exp_m1(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Expm1, a) }
    fn cbrt(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Cbrt, a) }
    fn erf(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Erf, a) }
    fn erfc(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Erfc, a) }
    fn lgamma(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Lgamma, a) }
    fn tgamma(&self, a: NodeId) -> NodeId { self.un(UnaryOp::Tgamma, a) }

    fn gt(&self, a: NodeId, b: NodeId) -> NodeId { self.cmp(CmpOp::Gt, a, b) }
    fn ge(&self, a: NodeId, b: NodeId) -> NodeId { self.cmp(CmpOp::Ge, a, b) }
    fn lt(&self, a: NodeId, b: NodeId) -> NodeId { self.cmp(CmpOp::Lt, a, b) }
    fn le(&self, a: NodeId, b: NodeId) -> NodeId { self.cmp(CmpOp::Le, a, b) }
    fn eq(&self, a: NodeId, b: NodeId) -> NodeId { self.cmp(CmpOp::Eq, a, b) }
    fn ne(&self, a: NodeId, b: NodeId) -> NodeId { self.cmp(CmpOp::Ne, a, b) }

    fn select(&self, cond: NodeId, t: NodeId, e: NodeId) -> NodeId {
        self.g.borrow_mut().select(cond, t, e)
    }

    // Collapse the trivial arities the way the tracer's `coeff_dot`/`node_dot`
    // do, so blocks and traced functions yield identical lean graphs; only the
    // genuine multi-term case records a `Dot`/`Reduce` node.
    fn dot(&self, a: &[NodeId], b: &[NodeId]) -> NodeId {
        match a.len() {
            0 => self.cst(0.0),
            1 => self.mul(a[0], b[0]),
            _ => self.g.borrow_mut().dot(a.to_vec(), b.to_vec()),
        }
    }

    fn reduce(&self, op: ReduceOp, xs: &[NodeId]) -> NodeId {
        match xs.len() {
            0 => self.cst(op.identity()),
            1 => xs[0],
            _ => self.g.borrow_mut().reduce(op, xs.to_vec()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssa::graph::{InputSignature};
    use crate::ssa::tape::InterpretedFn;

    /// A small generic expression evaluated natively and via the graph must
    /// agree (this is the property the whole 2b approach rests on).
    fn expr<B: Builder>(b: &B, x: B::N) -> B::N {
        // y = sin(2*x) + 3
        let two = b.cst(2.0);
        let three = b.cst(3.0);
        let s = b.sin(b.mul(two, x));
        b.add(s, three)
    }

    #[test]
    fn native_matches_graph() {
        let x = 0.7_f64;
        let native = expr(&F64Builder, x);

        let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", 1usize)])));
        let gb = GraphBuilder::new(&cell);
        let xn = gb.input(0);
        let yn = expr(&gb, xn);
        let mut g = cell.into_inner();
        g.outputs.push(yn);
        let compiled = InterpretedFn::from_graph(g);
        let mut out = vec![0.0];
        compiled.call_into(&[&[x]], &mut out);

        assert!((native - out[0]).abs() < 1e-15, "native={native} graph={}", out[0]);
    }

    /// The fused vector ops must agree native vs graph too: `reduce` and `dot`
    /// over the same inputs, native fold == graph `Reduce`/`Dot` lowered tape.
    fn vec_expr<B: Builder>(b: &B, xs: &[B::N]) -> B::N {
        // y = sum(xs) + dot(xs, xs)
        let s = b.reduce(ReduceOp::Sum, xs);
        let d = b.dot(xs, xs);
        b.add(s, d)
    }

    #[test]
    fn native_matches_graph_vector_ops() {
        let xs = [0.5_f64, -1.5, 2.0, 3.25];
        let native = vec_expr(&F64Builder, &xs);

        let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", xs.len())])));
        let gb = GraphBuilder::new(&cell);
        let xn: Vec<_> = (0..xs.len() as u32).map(|i| gb.input(i)).collect();
        let yn = vec_expr(&gb, &xn);
        let mut g = cell.into_inner();
        g.outputs.push(yn);
        let compiled = InterpretedFn::from_graph(g);
        let mut out = vec![0.0];
        compiled.call_into(&[&xs], &mut out);

        assert!((native - out[0]).abs() < 1e-12, "native={native} graph={}", out[0]);
    }
}
