// Math and logic block constructors.
//
// Math blocks: elementwise unary transforms `y[i] = op(u[i])` over the whole
// input vector (sin, cos, abs, sqrt, pow, clip, mod, rescale, …), matching
// pathsim's `Math` blocks (a numpy ufunc on the array). Plus the small family of
// matrix / norm / power-product helpers that logically belong to the math group
// but need a little more state (norm / pow_prod reduce a vector to a scalar).
// Logic blocks: 2-input 1-output scalar predicates (greater_than, equal, …).
// `logic_not` is a 1-input block implemented via `math_block`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::blocks::block::{Block, BlockRef};
use crate::blocks::blockops::ShapeLazyGraph;
use smallvec::SmallVec;

use crate::ssa::build::{dot_f64, Builder, F64Builder, GraphBuilder};
use crate::ssa::graph::{Graph, InputSignature};
use crate::utils::fastcell::FastCell;
use crate::utils::register::Register;


/// Define a unary math block in the 2b single-source style: one local generic
/// `eval` over a `Builder` drives both the native `f_alg` closure and the IR
/// op-graph. `$method` is a `jit::build::Builder` method (e.g. `sin`, `ln`,
/// `sqrt`). Elementwise `y[i] = op(u[i])` over the whole input vector, matching
/// pathsim's `Math` blocks (a numpy ufunc applied to the array). The op-graph is
/// shape-lazy: it is built for the actual connected width once layout is known.
macro_rules! unary_math_block {
    ($ctor:ident, $name:literal, $method:ident) => {
        pub fn $ctor() -> BlockRef {
            fn eval<B: Builder>(b: &B, u: &[B::N], out: &mut Vec<B::N>) {
                out.clear();
                for &ui in u {
                    out.push(b.$method(ui));
                }
            }
            let mut blk = Block::default_block();
            blk.type_name = $name;
            blk.f_alg = Some(Box::new(|_x, u: &[f64], _t, out: &mut Vec<f64>| {
                eval(&F64Builder, u, out);
            }));
            let slg = ShapeLazyGraph::new(|n| {
                let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", n)])));
                let gb = GraphBuilder::new(&cell);
                let u: Vec<_> = (0..n as u32).map(|i| gb.input(i)).collect();
                let mut out = Vec::new();
                eval(&gb, &u, &mut out);
                let mut g = cell.into_inner();
                g.outputs = out;
                g
            });
            blk.set_alg_lazy($name, slg);
            Rc::new(FastCell::new(blk))
        }
    };
}

/// Define a two-input logic/comparison block (`y = f(u0, u1)`, single output) in
/// the 2b single-source style. The body is an expression over a `Builder` `$bb`
/// and the two inputs `$aa`, `$cc`, used for both the native closure and graph.
macro_rules! binary_logic_block {
    ($ctor:ident, $name:literal, |$bb:ident, $aa:ident, $cc:ident| $body:expr) => {
        pub fn $ctor() -> BlockRef {
            fn build<B: Builder>($bb: &B, $aa: B::N, $cc: B::N) -> B::N {
                $body
            }
            let mut blk = Block::new(
                Some(HashMap::from([("a".to_string(), 0), ("b".to_string(), 1)])),
                Some(HashMap::from([("y".to_string(), 0)])),
            );
            blk.type_name = $name;
            blk.inputs.resize(2);
            blk.f_alg = Some(Box::new(|_x, u: &[f64], _t, out: &mut Vec<f64>| {
                out.clear();
                out.push(build(&F64Builder, u[0], u[1]));
            }));
            let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", 2usize)])));
            let y = {
                let gb = GraphBuilder::new(&cell);
                build(&gb, gb.input(0), gb.input(1))
            };
            let mut g = cell.into_inner();
            g.outputs.push(y);
            blk.set_alg($name, g);
            Rc::new(FastCell::new(blk))
        }
    };
}

// Math blocks: y = func(u) — overrides update
// ======================================================================================

pub fn math_block(name: &'static str, func: impl Fn(f64) -> f64 + 'static) -> BlockRef {
    let mut b = Block::default_block();
    b.type_name = name;
    let func = Box::new(func);
    // Elementwise `y[i] = func(u[i])` (opaque: an arbitrary Rust closure, so no
    // op-graph). Matches pathsim's elementwise `Math` semantics.
    b.f_alg = Some(Box::new(move |_x, u: &[f64], _t, out: &mut Vec<f64>| {
        out.clear();
        out.extend(u.iter().map(|&ui| func(ui)));
    }));

    Rc::new(FastCell::new(b))
}

// Elementwise unary math blocks (2b single source: native closure + op-graph
// from one generic `build`). `y[i] = op(u[i])`.
unary_math_block!(sin_block, "Sin", sin);
unary_math_block!(cos_block, "Cos", cos);
unary_math_block!(exp_block, "Exp", exp);
unary_math_block!(abs_block, "Abs", abs);
unary_math_block!(sqrt_block, "Sqrt", sqrt);
unary_math_block!(log_block, "Log", ln);
unary_math_block!(tanh_block, "Tanh", tanh);
unary_math_block!(tan_block, "Tan", tan);
unary_math_block!(atan_block, "Atan", atan);
unary_math_block!(sinh_block, "Sinh", sinh);
unary_math_block!(cosh_block, "Cosh", cosh);
unary_math_block!(log10_block, "Log10", log10);
/// Pow (elementwise): y[i] = u[i]^exp.
pub fn pow_block(exp: f64) -> BlockRef {
    fn eval<B: Builder>(b: &B, u: &[B::N], e: B::N, out: &mut Vec<B::N>) {
        out.clear();
        for &ui in u {
            out.push(b.powf(ui, e));
        }
    }
    let mut blk = Block::default_block();
    blk.type_name = "Pow";
    blk.f_alg = Some(Box::new(move |_x, u: &[f64], _t, out: &mut Vec<f64>| {
        eval(&F64Builder, u, exp, out);
    }));
    let slg = ShapeLazyGraph::new(move |n| {
        let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", n)])));
        {
            let mut g = cell.borrow_mut();
            g.n_params = 1;
            g.param_defaults = vec![exp];
            g.param_names = vec!["exp".into()];
        }
        let gb = GraphBuilder::new(&cell);
        let e = gb.param(0);
        let u: Vec<_> = (0..n as u32).map(|i| gb.input(i)).collect();
        let mut out = Vec::new();
        eval(&gb, &u, e, &mut out);
        let mut g = cell.into_inner();
        g.outputs = out;
        g
    });
    blk.set_alg_lazy("Pow", slg);
    Rc::new(FastCell::new(blk))
}

/// Clip (elementwise): y[i] = clamp(u[i], min, max) == min(max(u[i], min), max).
pub fn clip_block(min: f64, max: f64) -> BlockRef {
    fn eval<B: Builder>(b: &B, u: &[B::N], lo: B::N, hi: B::N, out: &mut Vec<B::N>) {
        out.clear();
        for &ui in u {
            out.push(b.min(b.max(ui, lo), hi));
        }
    }
    let mut blk = Block::default_block();
    blk.type_name = "Clip";
    blk.f_alg = Some(Box::new(move |_x, u: &[f64], _t, out: &mut Vec<f64>| {
        eval(&F64Builder, u, min, max, out);
    }));
    let slg = ShapeLazyGraph::new(move |n| {
        let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", n)])));
        {
            let mut g = cell.borrow_mut();
            g.n_params = 2;
            g.param_defaults = vec![min, max];
            g.param_names = vec!["min".into(), "max".into()];
        }
        let gb = GraphBuilder::new(&cell);
        let (lo, hi) = (gb.param(0), gb.param(1));
        let u: Vec<_> = (0..n as u32).map(|i| gb.input(i)).collect();
        let mut out = Vec::new();
        eval(&gb, &u, lo, hi, &mut out);
        let mut g = cell.into_inner();
        g.outputs = out;
        g
    });
    blk.set_alg_lazy("Clip", slg);
    Rc::new(FastCell::new(blk))
}

/// Rescale: y = u*scale + offset, optionally saturated to [out_lo, out_hi].
pub fn rescale_block(scale: f64, offset: f64, saturate: bool, out_lo: f64, out_hi: f64) -> BlockRef {
    fn eval<B: Builder>(
        b: &B, u: &[B::N], scale: B::N, offset: B::N, saturate: bool, lo: B::N, hi: B::N,
        out: &mut Vec<B::N>,
    ) {
        out.clear();
        for &ui in u {
            let y = b.add(b.mul(ui, scale), offset);
            out.push(if saturate { b.min(b.max(y, lo), hi) } else { y });
        }
    }
    let mut blk = Block::default_block();
    blk.type_name = "Rescale";
    blk.f_alg = Some(Box::new(move |_x, u: &[f64], _t, out: &mut Vec<f64>| {
        eval(&F64Builder, u, scale, offset, saturate, out_lo, out_hi, out);
    }));
    let slg = ShapeLazyGraph::new(move |n| {
        let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", n)])));
        {
            let mut g = cell.borrow_mut();
            g.n_params = 4;
            g.param_defaults = vec![scale, offset, out_lo, out_hi];
            g.param_names = vec!["scale".into(), "offset".into(), "out_lo".into(), "out_hi".into()];
        }
        let gb = GraphBuilder::new(&cell);
        let (sc, off, lo, hi) = (gb.param(0), gb.param(1), gb.param(2), gb.param(3));
        let u: Vec<_> = (0..n as u32).map(|i| gb.input(i)).collect();
        let mut out = Vec::new();
        eval(&gb, &u, sc, off, saturate, lo, hi, &mut out);
        let mut g = cell.into_inner();
        g.outputs = out;
        g
    });
    blk.set_alg_lazy("Rescale", slg);
    Rc::new(FastCell::new(blk))
}

// Atan2 — two-input arctangent: y = atan2(u0, u1)
binary_logic_block!(atan2_block, "Atan2", |b, a, c| b.atan2(a, c));

// Mod — modulo: y = u0 % u1 (0 when the divisor is 0)
binary_logic_block!(mod_block, "Mod", |b, a, c| {
    let z = b.cst(0.0);
    b.select(b.ne(c, z), b.modulo(a, c), z)
});

/// Mod (single-input, elementwise): y[i] = u[i] mod modulus (Euclidean).
pub fn mod_single(modulus: f64) -> BlockRef {
    // rem_euclid(x, m) == { let r = x % m; if r < 0 { r + |m| } else { r } }.
    fn eval<B: Builder>(b: &B, u: &[B::N], m: B::N, out: &mut Vec<B::N>) {
        out.clear();
        for &ui in u {
            let r = b.modulo(ui, m);
            let adj = b.add(r, b.abs(m));
            out.push(b.select(b.lt(r, b.cst(0.0)), adj, r));
        }
    }
    let mut blk = Block::default_block();
    blk.type_name = "Mod";
    blk.f_alg = Some(Box::new(move |_x, u: &[f64], _t, out: &mut Vec<f64>| {
        eval(&F64Builder, u, modulus, out);
    }));
    let slg = ShapeLazyGraph::new(move |n| {
        let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", n)])));
        {
            let mut g = cell.borrow_mut();
            g.n_params = 1;
            g.param_defaults = vec![modulus];
            g.param_names = vec!["modulus".into()];
        }
        let gb = GraphBuilder::new(&cell);
        let m = gb.param(0);
        let u: Vec<_> = (0..n as u32).map(|i| gb.input(i)).collect();
        let mut out = Vec::new();
        eval(&gb, &u, m, &mut out);
        let mut g = cell.into_inner();
        g.outputs = out;
        g
    });
    blk.set_alg_lazy("Mod", slg);
    Rc::new(FastCell::new(blk))
}

/// Norm (MISO, shape-poly): y = sqrt(sum u_i^2).
pub(crate) fn norm_eval<B: Builder>(b: &B, u: &[B::N], out: &mut Vec<B::N>) {
    out.clear();
    // ||u|| = sqrt(u · u): one fused dot product, then sqrt.
    out.push(b.sqrt(b.dot(u, u)));
}

fn norm_graph(n: usize) -> Graph {
    let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", n)])));
    let gb = GraphBuilder::new(&cell);
    let u: Vec<_> = (0..n as u32).map(|i| gb.input(i)).collect();
    let mut out = Vec::new();
    norm_eval(&gb, &u, &mut out);
    let mut g = cell.into_inner();
    g.outputs = out;
    g
}

/// Norm — vector norm (MISO): y = ||u||
pub fn norm_block() -> BlockRef {
    let mut b = Block::default_block();
    b.type_name = "Norm";
    b.f_alg = Some(Box::new(|_x, u, _t, out| norm_eval(&F64Builder, u, out)));
    b.set_alg_lazy("Norm", ShapeLazyGraph::new(norm_graph));
    Rc::new(FastCell::new(b))
}

/// Polynomial — element-wise polynomial in the input.
///
/// Coefficients follow the numpy.polyval convention (highest power of u
/// first):
///
///   y_i = c[0]·u_i^n + c[1]·u_i^{n-1} + … + c[n-1]·u_i + c[n]
///
/// Vector inputs are evaluated channel-wise.
/// Polynomial (shape-poly, channel-wise Horner): y_i = (((c0*u_i + c1)*u_i + ...).
pub(crate) fn polynomial_eval<B: Builder>(b: &B, coeffs: &[f64], u: &[B::N], out: &mut Vec<B::N>) {
    out.clear();
    for &ui in u {
        let mut y = b.cst(0.0);
        for &c in coeffs {
            y = b.add(b.mul(y, ui), b.cst(c));
        }
        out.push(y);
    }
}

pub fn polynomial(coeffs: Vec<f64>) -> BlockRef {
    let mut b = Block::default_block();
    b.type_name = "Polynomial";
    let coeffs_native = coeffs.clone();
    b.f_alg = Some(Box::new(move |_x, u, _t, out| {
        polynomial_eval(&F64Builder, &coeffs_native, u, out)
    }));
    let slg = ShapeLazyGraph::new(move |n| {
        let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", n)])));
        let gb = GraphBuilder::new(&cell);
        let u: Vec<_> = (0..n as u32).map(|i| gb.input(i)).collect();
        let mut out = Vec::new();
        polynomial_eval(&gb, &coeffs, &u, &mut out);
        let mut g = cell.into_inner();
        g.outputs = out;
        g
    });
    b.set_alg_lazy("Polynomial", slg);
    Rc::new(FastCell::new(b))
}

/// PowProd (fixed width = powers.len()): y = prod(u_i ^ p_i).
pub(crate) fn pow_prod_eval<B: Builder>(b: &B, powers: &[f64], u: &[B::N], out: &mut Vec<B::N>) {
    out.clear();
    let m = u.len().min(powers.len());
    let mut acc: Option<B::N> = None;
    for i in 0..m {
        let term = b.powf(u[i], b.cst(powers[i]));
        acc = Some(match acc {
            None => term,
            Some(a) => b.mul(a, term),
        });
    }
    out.push(acc.unwrap_or_else(|| b.cst(1.0)));
}

pub fn pow_prod(powers: Vec<f64>) -> BlockRef {
    let n = powers.len();
    let mut b = Block::default_block();
    b.type_name = "PowProd";
    b.inputs.resize(n);
    let powers_native = powers.clone();
    b.f_alg = Some(Box::new(move |_x, u, _t, out| {
        pow_prod_eval(&F64Builder, &powers_native, u, out)
    }));
    // Fixed width: build the graph once at construction.
    let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", n)])));
    let out = {
        let gb = GraphBuilder::new(&cell);
        let u: Vec<_> = (0..n as u32).map(|i| gb.input(i)).collect();
        let mut out = Vec::new();
        pow_prod_eval(&gb, &powers, &u, &mut out);
        out
    };
    let mut g = cell.into_inner();
    g.outputs = out;
    b.set_alg("PowProd", g);
    Rc::new(FastCell::new(b))
}

/// Matrix — matrix multiplication: y = M * u
/// Matrix (MIMO, fixed rows x cols): y = M*u, M row-major flat.
pub(crate) fn matrix_eval<B: Builder>(
    b: &B, m: &[f64], rows: usize, cols: usize, u: &[B::N], out: &mut Vec<B::N>,
) {
    out.clear();
    let cc = cols.min(u.len());
    // GRAPH path (built once): one fused `Dot` node per output row, zeros skipped
    // for a lean graph. The native runtime uses the streaming `matrix_mv_into`.
    let mut coeffs: SmallVec<[B::N; 8]> = SmallVec::new();
    let mut vals: SmallVec<[B::N; 8]> = SmallVec::new();
    for r in 0..rows {
        coeffs.clear();
        vals.clear();
        for c in 0..cc {
            let co = m[r * cols + c];
            if co != 0.0 {
                coeffs.push(b.cst(co));
                vals.push(u[c]);
            }
        }
        out.push(b.dot(&coeffs, &vals));
    }
}

/// Allocation-free native `y = M*u` for the runtime hot path: each row is a
/// 4-lane `dot_f64` over the contiguous matrix row and input slice (no per-call
/// collection, vectorizes for dense `M`). The op-graph form uses `matrix_eval`;
/// eval-parity guards they agree to 1e-12.
pub(crate) fn matrix_mv_into(out: &mut Vec<f64>, m: &[f64], rows: usize, cols: usize, u: &[f64]) {
    out.clear();
    let cc = cols.min(u.len());
    for r in 0..rows {
        out.push(dot_f64(&m[r * cols..r * cols + cc], &u[..cc]));
    }
}

pub fn matrix_block(m: Vec<f64>, rows: usize, cols: usize) -> BlockRef {
    let mut b = Block::default_block();
    b.type_name = "Matrix";
    b.inputs.resize(cols);
    b.outputs = Register::new(Some(rows), None);
    let m_native = m.clone();
    b.f_alg = Some(Box::new(move |_x, u, _t, out| {
        matrix_mv_into(out, &m_native, rows, cols, u)
    }));
    // Fixed shape: build the op-graph once (cols inputs, rows outputs).
    let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", cols)])));
    let out_nodes = {
        let gb = GraphBuilder::new(&cell);
        let u: Vec<_> = (0..cols as u32).map(|i| gb.input(i)).collect();
        let mut out = Vec::new();
        matrix_eval(&gb, &m, rows, cols, &u, &mut out);
        out
    };
    let mut g = cell.into_inner();
    g.outputs = out_nodes;
    b.set_alg("Matrix", g);
    Rc::new(FastCell::new(b))
}

// ======================================================================================
// Logic blocks: y = func(u0, u1) — overrides update
// ======================================================================================

pub fn logic_block(name: &'static str, func: impl Fn(f64, f64) -> f64 + 'static) -> BlockRef {
    let mut b = Block::new(
        Some(HashMap::from([("a".to_string(), 0), ("b".to_string(), 1)])),
        Some(HashMap::from([("y".to_string(), 0)])),
    );
    b.type_name = name;
    b.inputs.resize(2);

    let func = Box::new(func);
    b.f_alg = Some(Box::new(move |_x, u, _t, out| out.push(func(u[0], u[1]))));

    Rc::new(FastCell::new(b))
}

// GreaterThan: y = 1 if a > b, else 0
binary_logic_block!(greater_than, "GreaterThan", |b, a, c| b.gt(a, c));
// LessThan: y = 1 if a < b, else 0
binary_logic_block!(less_than, "LessThan", |b, a, c| b.lt(a, c));
// LogicAnd: y = 1 if a > 0.5 AND b > 0.5, else 0 (product of the two bools)
binary_logic_block!(logic_and, "LogicAnd", |b, a, c| {
    let h = b.cst(0.5);
    b.mul(b.gt(a, h), b.gt(c, h))
});
// LogicOr: y = 1 if a > 0.5 OR b > 0.5, else 0 (max of the two bools)
binary_logic_block!(logic_or, "LogicOr", |b, a, c| {
    let h = b.cst(0.5);
    b.max(b.gt(a, h), b.gt(c, h))
});

/// Equal: y = 1 if |a - b| < tolerance, else 0
pub fn equal_block(tolerance: f64) -> BlockRef {
    fn build<B: Builder>(b: &B, a: B::N, c: B::N, tol: B::N) -> B::N {
        b.lt(b.abs(b.sub(a, c)), tol)
    }
    let mut blk = Block::new(
        Some(HashMap::from([("a".to_string(), 0), ("b".to_string(), 1)])),
        Some(HashMap::from([("y".to_string(), 0)])),
    );
    blk.type_name = "Equal";
    blk.inputs.resize(2);
    blk.f_alg = Some(Box::new(move |_x, u: &[f64], _t, out: &mut Vec<f64>| {
        out.clear();
        out.push(build(&F64Builder, u[0], u[1], tolerance));
    }));
    let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", 2usize)])));
    {
        let mut g = cell.borrow_mut();
        g.n_params = 1;
        g.param_defaults = vec![tolerance];
        g.param_names = vec!["tolerance".into()];
    }
    let y = {
        let gb = GraphBuilder::new(&cell);
        build(&gb, gb.input(0), gb.input(1), gb.param(0))
    };
    let mut g = cell.into_inner();
    g.outputs.push(y);
    blk.set_alg("Equal", g);
    Rc::new(FastCell::new(blk))
}

/// LogicNot: y = 1 if u <= 0.5, else 0  (== 1 - [u > 0.5])
pub fn logic_not() -> BlockRef {
    fn build<B: Builder>(b: &B, u0: B::N) -> B::N {
        let h = b.cst(0.5);
        let one = b.cst(1.0);
        b.sub(one, b.gt(u0, h))
    }
    let mut blk = Block::default_block();
    blk.type_name = "LogicNot";
    blk.f_alg = Some(Box::new(|_x, u: &[f64], _t, out: &mut Vec<f64>| {
        out.clear();
        out.push(build(&F64Builder, u[0]));
    }));
    let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", 1usize)])));
    let y = {
        let gb = GraphBuilder::new(&cell);
        build(&gb, gb.input(0))
    };
    let mut g = cell.into_inner();
    g.outputs.push(y);
    blk.set_alg("LogicNot", g);
    Rc::new(FastCell::new(blk))
}

/// Alias: y = u (pass-through).  Equivalent to an identity `math_block`.
pub fn alias_block() -> BlockRef {
    let mut blk = Block::default_block();
    blk.type_name = "Alias";
    blk.f_alg = Some(Box::new(|_x, u: &[f64], _t, out: &mut Vec<f64>| {
        out.clear();
        out.push(u[0]);
    }));
    // Identity SISO: output is input element 0.
    let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", 1usize)])));
    let y = GraphBuilder::new(&cell).input(0);
    let mut g = cell.into_inner();
    g.outputs.push(y);
    blk.set_alg("Alias", g);
    Rc::new(FastCell::new(blk))
}


#[cfg(test)]
mod tests {
    use super::*;

    /// Math blocks are elementwise over the input vector (pathsim `Math` parity):
    /// both the native `f_alg` and the resolved op-graph produce `y[i] = op(u[i])`,
    /// not just `y[0] = op(u[0])` (the old SISO behaviour silently dropped the
    /// tail of a vector input).
    #[test]
    fn math_blocks_are_elementwise() {
        let u = [0.5_f64, 1.0, -0.3, 2.0];

        // Native closure output.
        let native = |blk: &BlockRef| -> Vec<f64> {
            let b = blk.borrow();
            let mut out = Vec::new();
            (b.f_alg.as_ref().unwrap())(&[], &u, 0.0, &mut out);
            out
        };
        // Resolved op-graph output at the connected width (from the alg operator).
        let graph = |blk: &BlockRef| -> Vec<f64> {
            let b = blk.borrow();
            let g = b.alg_op.as_ref().unwrap().graph_ref().unwrap().resolve(u.len()).unwrap();
            let params = g.param_defaults.clone();
            g.interpret(&[&u], &params)
        };

        // sin: native is elementwise sin, and the op-graph matches it.
        let sin = sin_block();
        let sn = native(&sin);
        assert_eq!(sn.len(), 4, "Sin must emit one output per input element");
        for (i, &ui) in u.iter().enumerate() {
            assert!((sn[i] - ui.sin()).abs() < 1e-15, "Sin[{i}] not elementwise");
        }
        assert_eq!(graph(&sin), sn, "Sin op-graph must match native elementwise");

        // pow value check.
        let p = pow_block(2.0);
        let pn = native(&p);
        assert_eq!(pn.len(), 4);
        for (i, &ui) in u.iter().enumerate() {
            assert!((pn[i] - ui.powf(2.0)).abs() < 1e-12, "Pow[{i}] not elementwise");
        }

        // The rest: emit n outputs, op-graph matches native elementwise.
        let blocks = [
            pow_block(2.0),
            clip_block(-0.4, 1.5),
            mod_single(0.7),
            rescale_block(2.0, 0.1, false, 0.0, 0.0),
            cos_block(),
            abs_block(),
        ];
        for blk in &blocks {
            let nv = native(blk);
            let gv = graph(blk);
            let name = blk.borrow().type_name;
            assert_eq!(nv.len(), 4, "{name} native not elementwise");
            assert_eq!(gv.len(), 4, "{name} op-graph not elementwise");
            for i in 0..4 {
                assert!((gv[i] - nv[i]).abs() < 1e-12, "{name} op-graph != native at {i}");
            }
        }
    }
}
