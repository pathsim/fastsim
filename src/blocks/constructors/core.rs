// Core primitive block constructors: integrators, basic arithmetic,
// constants, sources, and the function block.  These are all very small
// leaf blocks — they don't depend on any other submodule's helpers.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;


use crate::blocks::block::{Block, BlockRef, BlockRole};
use crate::blocks::blockops::ShapeLazyGraph;
use crate::error::SimError;
use crate::ssa::build::{Builder, F64Builder, GraphBuilder};
use smallvec::SmallVec;

use crate::ssa::graph::{Graph, InputSignature, ReduceOp};
use crate::solvers::solver::Solver;
use crate::utils::fastcell::FastCell;
use crate::utils::register::Register;

use super::out_port_map;

// --------------------------------------------------------------------------
// Block math, written ONCE generically over a `Builder` (the 2b single source
// of truth). `eval::<F64Builder>` is the native runtime closure (monomorphised
// to the hand-written math, no tape overhead); `eval::<GraphBuilder>` records
// the op-graph for the IR. Slot names follow the `blockops` convention
// ("x" state, "u" inputs, "t" time).
// --------------------------------------------------------------------------

/// Constant source: y = value (no inputs; a plain copy of the value).
pub(crate) fn constant_eval<N: Copy>(value: N, out: &mut Vec<N>) {
    out.clear();
    out.push(value);
}

/// Amplifier (shape-poly): y[i] = gain * u[i].
pub(crate) fn amplifier_eval<B: Builder>(b: &B, gain: B::N, u: &[B::N], out: &mut Vec<B::N>) {
    out.clear();
    for &ui in u {
        out.push(b.mul(ui, gain));
    }
}

/// Adder (shape-poly): y = sum(coeff_i * u[i]). Empty `ops` means a plain sum
/// (all coefficients 1); inputs beyond `ops.len()` get coefficient 0.
pub(crate) fn adder_eval<B: Builder>(b: &B, ops: &[f64], u: &[B::N], out: &mut Vec<B::N>) {
    out.clear();
    if ops.is_empty() {
        // Plain sum of inputs: one fused reduction (identical add chain).
        out.push(b.reduce(ReduceOp::Sum, u));
    } else {
        // Weighted sum `Σ ops[i]·u[i]`: a fused dot product (missing weights
        // default to 0, matching the old per-term behaviour).
        let coeffs: SmallVec<[B::N; 8]> =
            (0..u.len()).map(|i| b.cst(ops.get(i).copied().unwrap_or(0.0))).collect();
        out.push(b.dot(&coeffs, u));
    }
}

/// Multiplier (shape-poly): y = prod(u[i]). Empty input yields 1.
pub(crate) fn multiplier_eval<B: Builder>(b: &B, u: &[B::N], out: &mut Vec<B::N>) {
    out.clear();
    // Product of inputs: one fused reduction (identical mul chain, identity 1).
    out.push(b.reduce(ReduceOp::Product, u));
}

/// Integrator output region: y = x (identity copy of the state vector).
pub(crate) fn integrator_alg_eval<N: Copy>(x: &[N], out: &mut Vec<N>) {
    out.clear();
    out.extend_from_slice(x);
}

/// Integrator derivative region: dx/dt = u (identity copy of the input).
pub(crate) fn integrator_dyn_eval<N: Copy>(u: &[N], out: &mut Vec<N>) {
    out.clear();
    out.extend_from_slice(u);
}

// -- op-graph wrappers: instantiate the generic math with `GraphBuilder` --

pub(crate) fn constant_graph(value: f64) -> Graph {
    let cell = RefCell::new(Graph::new(InputSignature::empty()));
    {
        let mut g = cell.borrow_mut();
        g.n_params = 1;
        g.param_defaults = vec![value];
        g.param_names = vec!["value".into()];
    }
    let gb = GraphBuilder::new(&cell);
    let v = gb.param(0);
    let mut out = Vec::new();
    constant_eval(v, &mut out);
    let mut g = cell.into_inner();
    g.outputs = out;
    g
}

pub(crate) fn amplifier_graph(gain: f64, n: usize) -> Graph {
    let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", n)])));
    {
        let mut g = cell.borrow_mut();
        g.n_params = 1;
        g.param_defaults = vec![gain];
        g.param_names = vec!["gain".into()];
    }
    let gb = GraphBuilder::new(&cell);
    let gain_n = gb.param(0);
    let u: Vec<_> = (0..n as u32).map(|i| gb.input(i)).collect();
    let mut out = Vec::new();
    amplifier_eval(&gb, gain_n, &u, &mut out);
    let mut g = cell.into_inner();
    g.outputs = out;
    g
}

pub(crate) fn adder_graph(ops: &[f64], n: usize) -> Graph {
    let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", n)])));
    let gb = GraphBuilder::new(&cell);
    let u: Vec<_> = (0..n as u32).map(|i| gb.input(i)).collect();
    let mut out = Vec::new();
    adder_eval(&gb, ops, &u, &mut out);
    let mut g = cell.into_inner();
    g.outputs = out;
    g
}

pub(crate) fn multiplier_graph(n: usize) -> Graph {
    let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", n)])));
    let gb = GraphBuilder::new(&cell);
    let u: Vec<_> = (0..n as u32).map(|i| gb.input(i)).collect();
    let mut out = Vec::new();
    multiplier_eval(&gb, &u, &mut out);
    let mut g = cell.into_inner();
    g.outputs = out;
    g
}

pub(crate) fn integrator_alg_graph(n: usize) -> Graph {
    let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("x", n)])));
    let gb = GraphBuilder::new(&cell);
    let x: Vec<_> = (0..n as u32).map(|i| gb.input(i)).collect();
    let mut out = Vec::new();
    integrator_alg_eval(&x, &mut out);
    let mut g = cell.into_inner();
    g.outputs = out;
    g
}

pub(crate) fn integrator_dyn_graph(n: usize) -> Graph {
    let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", n)])));
    let gb = GraphBuilder::new(&cell);
    let u: Vec<_> = (0..n as u32).map(|i| gb.input(i)).collect();
    let mut out = Vec::new();
    integrator_dyn_eval(&u, &mut out);
    let mut g = cell.into_inner();
    g.outputs = out;
    g
}

// ======================================================================================
// Integrator: dx/dt = u, y = x — overrides len, update, solve, step
// ======================================================================================

/// Integrator (vector): dx/dt = u, y = x
pub fn integrator_vec(initial_value: &[f64]) -> BlockRef {
    let n = initial_value.len();
    let mut b = Block::default_block();
    b.type_name = "Integrator";
    b.role = BlockRole { is_dyn: true, is_src: false, is_rec: false };
    b.initial_value = Some(initial_value.to_vec());
    b.engine = Some(Solver::with_defaults(initial_value));
    b.inputs = Register::new(Some(n), None);
    b.outputs = Register::new(Some(n), None);
    b.len_fn = Some(Box::new(|_| 0));
    // Single source of truth: derive both runtime closures from the op-graphs.
    let alg = integrator_alg_graph(n);
    let dyn_ = integrator_dyn_graph(n);
    b.f_alg = Some(Box::new(|x, _u, _t, out| integrator_alg_eval(x, out)));   // y = x
    b.f_dyn = Some(Box::new(|_x, u, _t, out| integrator_dyn_eval(u, out)));   // dx/dt = u
    b.set_dynamic("Integrator", alg, dyn_);
    Rc::new(FastCell::new(b))
}

/// Integrator (scalar): dx/dt = u, y = x
pub fn integrator(initial_value: f64) -> BlockRef {
    integrator_vec(&[initial_value])
}

// ======================================================================================
// Amplifier: y = gain * u — overrides update
// ======================================================================================

/// Amplifier: y = gain * u
pub fn amplifier(gain: f64) -> BlockRef {
    let mut b = Block::default_block();
    b.type_name = "Amplifier";
    // Native closure (full speed) and IR graph both derive from amplifier_eval.
    // Shape-poly: the input width is only known after connection layout, so the
    // IR graph is built lazily + cached; the runtime needs no width.
    b.f_alg = Some(Box::new(move |_x, u, _t, out| amplifier_eval(&F64Builder, gain, u, out)));
    let slg = ShapeLazyGraph::new(move |n| amplifier_graph(gain, n));
    b.set_alg_lazy("Amplifier", slg);
    Rc::new(FastCell::new(b))
}

// ======================================================================================
// Adder: y = sum(op_i * u_i) — overrides update
// ======================================================================================

/// Adder: y = sum(op_i * u_i) based on operations string ('+', '-', '0')
pub fn adder(operations: Option<&str>) -> BlockRef {
    let mut b = Block::new(None, Some(out_port_map()));
    b.type_name = "Adder";

    let ops_map: HashMap<char, f64> = HashMap::from([('+', 1.0), ('-', -1.0), ('0', 0.0)]);
    let ops_array: Vec<f64> = match operations {
        Some(ops_str) => ops_str.chars().map(|ch| ops_map[&ch]).collect(),
        None => Vec::new(),
    };

    let ops_native = ops_array.clone();
    b.f_alg = Some(Box::new(move |_x, u, _t, out| adder_eval(&F64Builder, &ops_native, u, out)));
    let slg = ShapeLazyGraph::new(move |n| adder_graph(&ops_array, n));
    b.set_alg_lazy("Adder", slg);

    Rc::new(FastCell::new(b))
}

// ======================================================================================
// Multiplier: y = prod(u_i) — overrides update
// ======================================================================================

/// Multiplier: y = prod(u_i)
pub fn multiplier() -> BlockRef {
    let mut b = Block::new(None, Some(out_port_map()));
    b.type_name = "Multiplier";
    b.f_alg = Some(Box::new(|_x, u, _t, out| multiplier_eval(&F64Builder, u, out)));
    let slg = ShapeLazyGraph::new(multiplier_graph);
    b.set_alg_lazy("Multiplier", slg);
    Rc::new(FastCell::new(b))
}

// ======================================================================================
// Divider: combined multiply / divide (MISO) — mirrors pathsim Divider
// ======================================================================================

/// Policy applied when a denominator input is zero.
///
/// Matches pathsim `Divider`'s `zero_div` parameter semantics.
#[derive(Debug, Clone, Copy, Default)]
pub enum ZeroDiv {
    /// Propagate ±inf (IEEE 754 division by zero, no warning in Rust).
    #[default]
    Warn,
    /// Panic with a clear message when any denominator input is exactly 0.
    Raise,
    /// Clamp denominator magnitude to `f64::EPSILON` (preserving sign) so the
    /// output stays large-but-finite.
    Clamp,
}

/// Divider: `y = ∏ u_i^{op_i}` with `op ∈ {+1, -1}` per input port.
///
/// `operations` is a string of `*` / `/` characters, one per input port.
/// Default `"*/"` matches pathsim: first input multiplied, second divided.
/// If `None`, behaves exactly like `multiplier` (all inputs multiplied).
///
/// Inputs beyond the length of `operations` default to `*` (multiply), which
/// matches how pathsim pads its `_ops_array`.
/// Divider math (shape-poly): num = prod(u_i where op>0), den = prod(u_i where
/// op<0); empty ops => product of all. `Warn`/`Raise` divide as-is (Raise's
/// zero guard is a runtime-only panic, not representable in ops); `Clamp`
/// nudges a near-zero denominator to ±EPSILON.
pub(crate) fn divider_build<B: Builder>(b: &B, ops: &[f64], zero_div: ZeroDiv, u: &[B::N], out: &mut Vec<B::N>) {
    out.clear();
    let (mut num, mut den): (Option<B::N>, Option<B::N>) = (None, None);
    let acc = |slot: &mut Option<B::N>, v: B::N| {
        *slot = Some(match *slot {
            None => v,
            Some(a) => b.mul(a, v),
        });
    };
    if ops.is_empty() {
        for &ui in u {
            acc(&mut num, ui);
        }
    } else {
        for (i, &ui) in u.iter().enumerate() {
            let op = ops.get(i).copied().unwrap_or(1.0);
            if op > 0.0 {
                acc(&mut num, ui);
            } else if op < 0.0 {
                acc(&mut den, ui);
            }
        }
    }
    let num = num.unwrap_or_else(|| b.cst(1.0));
    let den_raw = den.unwrap_or_else(|| b.cst(1.0));
    let den = match zero_div {
        ZeroDiv::Clamp => {
            // Replace a near-zero denominator by ±EPSILON. The sign comes from
            // an explicit `den < 0` select — NOT `eps·sign(den)`, which is 0 at
            // an exactly-zero denominator now that `sign` has numpy semantics
            // (sign(0) = 0) and would clamp 0 to 0. An exact zero takes the
            // else-branch (+eps), matching pathsim's `_safe_den(0) -> +eps`.
            let eps = b.cst(f64::EPSILON);
            let neg_eps = b.cst(-f64::EPSILON);
            let zero = b.cst(0.0);
            let signed_eps = b.select(b.lt(den_raw, zero), neg_eps, eps);
            b.select(b.lt(b.abs(den_raw), eps), signed_eps, den_raw)
        }
        _ => den_raw,
    };
    out.push(b.div(num, den));
}

pub fn divider(operations: Option<&str>, zero_div: ZeroDiv) -> Result<BlockRef, SimError> {
    let mut b = Block::new(None, Some(out_port_map()));
    b.type_name = "Divider";

    // Decode operations once; store as SmallVec of exponents (±1.0). A 0.0
    // exponent is possible if the caller uses pathsim's legacy '0' marker; we
    // accept it too for drop-in parity.
    let ops: Vec<f64> = match operations {
        None => Vec::new(),
        Some(s) => {
            let mut v = Vec::with_capacity(s.len());
            for c in s.chars() {
                v.push(match c {
                    '*' => 1.0,
                    '/' => -1.0,
                    '0' => 0.0,
                    other => {
                        return Err(SimError::UnknownOp {
                            op: other.to_string(),
                            expected: "'*' / '/'",
                        })
                    }
                });
            }
            v
        }
    };

    // IR op-graph (shape-poly). Runtime keeps its bespoke closure below (notably
    // the Raise zero-guard panic, which has no op representation).
    let ops_graph = ops.clone();
    let slg = crate::blocks::blockops::ShapeLazyGraph::new(move |n| {
        let cell = std::cell::RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", n)])));
        let out = {
            let gb = GraphBuilder::new(&cell);
            let u: Vec<_> = (0..n as u32).map(|i| gb.input(i)).collect();
            let mut out = Vec::new();
            divider_build(&gb, &ops_graph, zero_div, &u, &mut out);
            out
        };
        let mut g = cell.into_inner();
        g.outputs = out;
        g
    });
    b.set_alg_lazy("Divider", slg);

    // NOTE: this runtime closure duplicates `divider_build`'s num/den/zero-div
    // arithmetic because the `Raise` zero-guard panic has no op representation.
    // Divider is the one block that breaks the single-source 2b pattern, so the
    // two must be kept in sync: any change to `divider_build` belongs here too.
    b.f_alg = Some(Box::new(move |_x, u, _t, out| {
        let y = if ops.is_empty() {
            u.iter().product::<f64>()
        } else {
            let mut num = 1.0_f64;
            let mut den = 1.0_f64;
            for (i, &ui) in u.iter().enumerate() {
                // Inputs past the ops string default to multiply.
                let op = ops.get(i).copied().unwrap_or(1.0);
                if op > 0.0 { num *= ui; }
                else if op < 0.0 { den *= ui; }
                // op == 0.0 → drop
            }
            let den = match zero_div {
                ZeroDiv::Warn => den,
                ZeroDiv::Raise => {
                    if den == 0.0 {
                        // Data-dependent zero denominator under zero_div='raise'.
                        // Record a catchable fault and stop the run cooperatively
                        // instead of panicking uncatchably across PyO3 (issue #28).
                        crate::simulation::record_runtime_fault(
                            crate::error::SimError::InvalidBlockParam(
                                "Divider: denominator evaluated to zero under zero_div='raise'"
                                    .to_string(),
                            ),
                        );
                        // Finite fallback so this partial step can finish; the run
                        // stops immediately and the fault is re-raised afterward.
                        1.0
                    } else {
                        den
                    }
                }
                ZeroDiv::Clamp => if den == 0.0 { f64::EPSILON }
                                  else if den.abs() < f64::EPSILON { f64::EPSILON.copysign(den) }
                                  else { den },
            };
            num / den
        };
        out.push(y);
    }));

    Ok(Rc::new(FastCell::new(b)))
}

// ======================================================================================
// Constant: y = value — overrides len, update
// ======================================================================================

/// Constant source: y = value
pub fn constant(value: f64) -> BlockRef {
    let mut b = Block::new(
        Some(HashMap::new()),
        Some(out_port_map()),
    );
    b.type_name = "Constant";
    b.role = BlockRole { is_dyn: false, is_src: true, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));
    b.f_alg = Some(Box::new(move |_x, _u, _t, out| constant_eval(value, out)));
    b.set_alg("Constant", constant_graph(value));
    Rc::new(FastCell::new(b))
}

// ======================================================================================
// Source: y = func(t) — overrides len, update
// ======================================================================================

/// Source: y = func(t)
pub fn source(func: impl Fn(f64) -> f64 + 'static) -> BlockRef {
    let mut b = Block::new(
        Some(HashMap::new()),
        Some(out_port_map()),
    );
    b.type_name = "Source";
    b.role = BlockRole { is_dyn: false, is_src: true, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));
    let func = Box::new(func);
    b.f_alg = Some(Box::new(move |_x, _u, t, out| out.push(func(t))));
    Rc::new(FastCell::new(b))
}


// ======================================================================================
// Function: y = func(*u) — overrides update
// ======================================================================================

/// Function: y = func(u)
pub fn function(func: impl Fn(&[f64]) -> Vec<f64> + 'static) -> BlockRef {
    let mut b = Block::default_block();
    b.type_name = "Function";
    b.f_alg = Some(Box::new(move |_x, u, _t, out| out.extend(func(u))));
    Rc::new(FastCell::new(b))
}

