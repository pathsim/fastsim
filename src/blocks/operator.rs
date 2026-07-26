//! `Operator`: one computational path (algebraic or dynamic) of a block.
//!
//! A block's behaviour splits into at most two operators: the algebraic output
//! `y = f_alg(x, u, t)` and the dynamic derivative `dx/dt = f_dyn(x, u, t)`.
//! Each `Operator` owns, for its path, everything the rest of the system needs:
//!
//!   - the **SSA op-graph** (`RegionGraph`, shape-fixed or shape-lazy) — the
//!     source of truth for IR / static compile / codegen,
//!   - the **native evaluation closure** (the fast interpreted path), built from
//!     the same generic `eval` as the graph (the "2b single source" pattern),
//!   - the **Jacobian**, derived by forward/reverse AD *from the graph itself*
//!     (sparse-aware, gated, width-cached) — NOT a hand-written callback, so it
//!     can never desync from the function, and it is rebuilt together with the
//!     graph when a shape-polymorphic block reshapes,
//!   - codegen lowering hints (`lut1d`).
//!
//! This replaces the block's previously-scattered `f_alg` / `f_dyn` / `jac_dyn`
//! closures plus the parallel `jac_pattern` attribute: the sparsity now lives
//! WITH the Jacobian tape, mirroring pathsim's `Operator(func, jac)` abstraction
//! (but the AD is over our SSA graph, and the Jacobian is recomputed per Newton
//! step rather than linearized-and-reused).
//!
//! Genuinely opaque operators (FMU, RNG, arbitrary Python that does not trace)
//! carry no graph; their Jacobian falls back to a central-difference `num_jac`.

use std::cell::RefCell;
use std::rc::Rc;

use smallvec::SmallVec;

use crate::blocks::block::BlockFn;
use crate::blocks::blockops::{Lut1dSpec, RegionGraph};
use crate::constants::{
    LINSOLVE_SPARSE_MAX_DENSITY, LINSOLVE_SPARSE_MIN_DIM, NUM_JAC_REL, NUM_JAC_TOL,
};
use crate::solvers::solver::{Jacobian, SparseJac};
use crate::ssa::autodiff::{
    jacobian_is_constant, jacobian_sparse_wrt_slot, jacobian_wrt_param, jacobian_wrt_slot,
};
use crate::ssa::graph::{Graph, InputSignature};
use crate::ssa::tape::InterpretedFn;
use crate::utils::numerical::num_jac;

/// How the AD Jacobian is presented to the solver, decided once per width.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum JacMode {
    /// `n == 1`: a single scalar entry.
    Scalar,
    /// Small / dense / constant: scatter the nonzeros into a dense `n×n` matrix.
    Dense,
    /// Large + genuinely sparse + state-dependent: carry the coordinate pattern.
    Sparse,
}

/// Width-keyed Jacobian evaluator built from the operator's graph. Holds a tape
/// that evaluates ONLY the structurally-nonzero entries (in pattern order); the
/// dense presentation scatters them. Rebuilt when the input width changes.
struct JacTape {
    width: usize,
    tape: InterpretedFn,
    /// Structurally-invariant coordinate pattern, shared with every emitted
    /// `SparseJac` via `Rc` (no per-call pattern clone — issue #43).
    rows: Rc<[u32]>,
    cols: Rc<[u32]>,
    n: usize,
    mode: JacMode,
    /// Reused buffer for the values-only tape evaluation, so a Jacobian call
    /// does not allocate a fresh `Vec` for the nonzeros each time (issue #43).
    val_scratch: RefCell<Vec<f64>>,
}

/// One computational path of a block (algebraic or dynamic). See module docs.
pub struct Operator {
    /// SSA op-graph for this path. `None` for opaque operators (numerical Jacobian).
    graph: Option<RegionGraph>,
    /// Fast native evaluation `out = f(x, u, t)`.
    eval_fn: BlockFn,
    /// Analytical Jacobian for opaque operators that DO carry one (e.g. an FMU's
    /// directional derivatives): `out = dense row-major ∂f/∂x`. Ignored when a
    /// `graph` is present (then AD over the graph supplies the Jacobian).
    jac_fn: Option<BlockFn>,
    /// Lazily-built, width-keyed AD Jacobian (rebuilt with the graph on reshape).
    jac: RefCell<Option<JacTape>>,
    /// codegen LUT lowering hint, not recoverable from the graph alone.
    pub lut1d: Option<Lut1dSpec>,
}

impl Operator {
    /// A traceable operator: native closure + the SSA graph it was lowered from
    /// (the AD Jacobian is derived from this graph on demand).
    pub fn traceable(graph: RegionGraph, eval_fn: BlockFn) -> Self {
        Self { graph: Some(graph), eval_fn, jac_fn: None, jac: RefCell::new(None), lut1d: None }
    }

    /// An opaque operator (no op-graph): the Jacobian is computed by central
    /// finite differences over the native closure.
    pub fn opaque(eval_fn: BlockFn) -> Self {
        Self { graph: None, eval_fn, jac_fn: None, jac: RefCell::new(None), lut1d: None }
    }

    /// An opaque operator that carries its own analytical Jacobian (dense
    /// row-major `∂f/∂x`), e.g. an FMU exposing directional derivatives.
    pub fn opaque_with_jac(eval_fn: BlockFn, jac_fn: BlockFn) -> Self {
        Self { graph: None, eval_fn, jac_fn: Some(jac_fn), jac: RefCell::new(None), lut1d: None }
    }

    /// A graph-only operator: holds the SSA op-graph (for IR / codegen / AD
    /// Jacobian); evaluation stays with the block's own native `f_alg`/`f_dyn`
    /// closure (Rust closures are not `Clone`, so the operator does not own one).
    /// This is the per-path unit a block stores as its sole graph representation.
    pub fn graph_only(graph: RegionGraph) -> Self {
        Self {
            graph: Some(graph),
            eval_fn: Box::new(|_, _, _, _| {}),
            jac_fn: None,
            jac: RefCell::new(None),
            lut1d: None,
        }
    }

    /// Alias for the dyn-path Jacobian call sites (the compiled-subsystem fuse
    /// installs its derivative graph here): identical to [`Self::graph_only`].
    pub fn jac_only(graph: RegionGraph) -> Self {
        Self::graph_only(graph)
    }

    /// The op-graph behind this operator (for IR / codegen), if any.
    pub fn graph_ref(&self) -> Option<&RegionGraph> {
        self.graph.as_ref()
    }

    /// Attach a 1-D LUT codegen lowering hint.
    pub fn with_lut1d(mut self, lut: Option<Lut1dSpec>) -> Self {
        self.lut1d = lut;
        self
    }

    /// `true` if this operator carries an op-graph (i.e. is not opaque).
    pub fn is_traceable(&self) -> bool {
        self.graph.is_some()
    }

    /// Native evaluation `out = f(x, u, t)`.
    #[inline]
    pub fn eval(&self, x: &[f64], u: &[f64], t: f64, out: &mut Vec<f64>) {
        (self.eval_fn)(x, u, t, out);
    }

    /// The SSA op-graph resolved at the connected input `width` (for IR / compile
    /// / codegen). `None` for opaque operators.
    pub fn resolve(&self, width: usize) -> Option<Graph> {
        self.graph.as_ref().and_then(|g| g.resolve(width))
    }

    /// Jacobian of this path's output w.r.t. the state slot `"x"`, at `(x, u, t)`.
    ///
    /// Traceable operators differentiate the resolved graph via AD: only the
    /// structurally-nonzero entries are evaluated, and the representation
    /// (`Scalar` / `Matrix` / `Sparse`) is gated on size, density and
    /// state-dependence (same gate as the static compile path). The tape is
    /// cached and rebuilt only when the input width changes. Opaque operators
    /// fall back to `num_jac`. Returns `None` when the Jacobian is structurally
    /// zero (no state dependence), so the solver does plain functional iteration.
    pub fn jacobian_wrt_state(
        &self,
        x: &[f64],
        u: &[f64],
        t: f64,
        mem: &[f64],
    ) -> Option<Jacobian> {
        let width = u.len();
        // Resolve the op-graph at this width; an opaque operator (no graph) or a
        // `Lazy` region that cannot lower here (data-dependent Python callable)
        // both fall back to the analytical or numerical Jacobian.
        let resolved = self.graph.as_ref().and_then(|g| g.resolve(width));
        let Some(graph) = resolved else {
            // An analytical Jacobian closure if one was supplied (e.g. an FMU's
            // directional derivatives), else central-difference `num_jac`.
            if let Some(jf) = &self.jac_fn {
                return assemble_dense_jac(jf, x, u, t);
            }
            return Some(num_jac(&|xx, uu, tt, o| (self.eval_fn)(xx, uu, tt, o), x, u, t));
        };

        let stale = match &*self.jac.borrow() {
            Some(j) => j.width != width,
            None => true,
        };
        if stale {
            *self.jac.borrow_mut() = build_jac_tape(&graph, "x", width);
        }

        let jac = self.jac.borrow();
        let jt = jac.as_ref()?; // None → structurally-zero Jacobian
        let t_arr = [t];
        let inputs = slot_inputs(&jt.tape.signature, x, u, &t_arr, mem)?;
        // Evaluate the structural nonzeros into the reused scratch (no per-call
        // `Vec` allocation for the values — issue #43).
        let mut values = jt.val_scratch.borrow_mut();
        values.clear();
        values.resize(jt.tape.n_out, 0.0);
        jt.tape.call_into(&inputs, &mut values);
        Some(match jt.mode {
            JacMode::Scalar => Jacobian::Scalar(values.first().copied().unwrap_or(0.0)),
            // The pattern is shared via `Rc` (refcount bump); only the values
            // are copied out into the returned owned `SparseJac`.
            JacMode::Sparse => Jacobian::Sparse(SparseJac {
                n: jt.n,
                rows: jt.rows.clone(),
                cols: jt.cols.clone(),
                values: values.clone(),
            }),
            JacMode::Dense => {
                let mut m = vec![0.0; jt.n * jt.n];
                for k in 0..values.len() {
                    m[jt.rows[k] as usize * jt.n + jt.cols[k] as usize] = values[k];
                }
                Jacobian::Matrix(m, jt.n)
            }
        })
    }

    /// Rectangular dense Jacobian `∂out/∂<wrt>` at `(x, u, t)`, row-major
    /// `n_out × n_slot`. Returns `None` when the slot is absent or the operator
    /// can neither trace nor evaluate.
    ///
    /// This is a one-shot query for operating-point linearization, deliberately
    /// NOT built on [`Self::jacobian_wrt_state`]. That path feeds the implicit
    /// solver, and its `Jacobian` representation is square throughout: the
    /// sparsity gate keys on a single dimension `n`, and the dense presentation
    /// scatters into `n × n`. That holds for `∂(dx/dt)/∂x` but cannot express
    /// `B = ∂(dx/dt)/∂u` (`nx × nu`) or `C = ∂y/∂x` (`ny × nx`). Linearization
    /// is not on any hot path, so it takes the plain dense route instead of
    /// widening the solver's representation.
    ///
    /// Traceable operators differentiate the resolved graph by AD;
    /// `jacobian_wrt_flat_range` emits the dense row-major `n_out × size`
    /// layout directly. Opaque operators fall back to central differences over
    /// the native closure.
    pub fn dense_jacobian_wrt(
        &self,
        wrt: &str,
        x: &[f64],
        u: &[f64],
        t: f64,
        mem: &[f64],
    ) -> Option<(Vec<f64>, usize, usize)> {
        let width = u.len();

        // Opaque (or a `Lazy` region that cannot lower at this width): central
        // differences over the native closure.
        let Some(graph) = self.graph.as_ref().and_then(|g| g.resolve(width)) else {
            return self.finite_difference_wrt(wrt, x, u, t);
        };

        let n_cols = graph.signature.slot(wrt)?.size;
        let n_rows = graph.outputs.len();
        if n_rows == 0 || n_cols == 0 {
            return Some((Vec::new(), n_rows, n_cols));
        }

        let tape = InterpretedFn::from_graph(jacobian_wrt_slot(&graph, wrt)?);
        let t_arr = [t];
        let inputs = slot_inputs(&tape.signature, x, u, &t_arr, mem)?;

        let mut values = vec![0.0; tape.n_out];
        tape.call_into(&inputs, &mut values);

        Some((values, n_rows, n_cols))
    }

    /// Central-difference fallback for [`Self::dense_jacobian_wrt`] on opaque
    /// operators. Only `"x"` and `"u"` are perturbable — the remaining slots
    /// (time, discrete memory) are held fixed for an operating-point Jacobian.
    fn finite_difference_wrt(
        &self,
        wrt: &str,
        x: &[f64],
        u: &[f64],
        t: f64,
    ) -> Option<(Vec<f64>, usize, usize)> {
        // A `graph_only` operator carries a no-op `eval_fn` (the block keeps its
        // own closure), so differencing it would silently yield zeros.
        if self.graph.is_some() {
            return None;
        }

        let mut probe = Vec::new();
        (self.eval_fn)(x, u, t, &mut probe);
        let n_rows = probe.len();

        let n_cols = match wrt {
            "x" => x.len(),
            "u" => u.len(),
            _ => return None,
        };
        if n_rows == 0 || n_cols == 0 {
            return Some((Vec::new(), n_rows, n_cols));
        }

        let mut jac = vec![0.0; n_rows * n_cols];
        let (mut xp, mut up) = (x.to_vec(), u.to_vec());
        let (mut f_p, mut f_m) = (Vec::new(), Vec::new());

        for j in 0..n_cols {
            let base = if wrt == "x" { x[j] } else { u[j] };
            let h = (NUM_JAC_REL * base.abs()).max(NUM_JAC_TOL);

            let slot = if wrt == "x" { &mut xp[j] } else { &mut up[j] };
            *slot = base + h;
            f_p.clear();
            (self.eval_fn)(&xp, &up, t, &mut f_p);

            let slot = if wrt == "x" { &mut xp[j] } else { &mut up[j] };
            *slot = base - h;
            f_m.clear();
            (self.eval_fn)(&xp, &up, t, &mut f_m);

            let slot = if wrt == "x" { &mut xp[j] } else { &mut up[j] };
            *slot = base;

            for i in 0..n_rows {
                let (p, m) = (f_p.get(i).copied()?, f_m.get(i).copied()?);
                jac[i * n_cols + j] = (p - m) / (2.0 * h);
            }
        }

        Some((jac, n_rows, n_cols))
    }
}

impl Operator {
    /// Names of the mutable parameters this path exposes, in index order.
    /// Empty for opaque operators and for graphs that carry no parameters.
    pub fn param_names(&self, width: usize) -> Vec<String> {
        self.graph
            .as_ref()
            .and_then(|g| g.resolve(width))
            .map(|g| g.param_names.clone())
            .unwrap_or_default()
    }

    /// Index of the named parameter on this path, if it has one.
    pub fn param_index(&self, name: &str, width: usize) -> Option<u32> {
        self.param_names(width)
            .iter()
            .position(|n| n == name)
            .map(|i| i as u32)
    }

    /// Sensitivity `∂out/∂p` of this path's outputs w.r.t. the parameter `idx`,
    /// as a dense column of length `n_out`.
    ///
    /// Parameters live in their own namespace in the SSA graph
    /// (`Node::Param`), so this differentiates against a *design* quantity —
    /// `∂y/∂gain` — rather than against the state or the inputs. Opaque
    /// operators have no parameters to differentiate and return `None`.
    pub fn dense_jacobian_wrt_param(
        &self,
        idx: u32,
        x: &[f64],
        u: &[f64],
        t: f64,
        mem: &[f64],
    ) -> Option<Vec<f64>> {
        let width = u.len();
        let graph = self.graph.as_ref().and_then(|g| g.resolve(width))?;

        let n_out = graph.outputs.len();
        if n_out == 0 {
            return Some(Vec::new());
        }

        let tape = InterpretedFn::from_graph(jacobian_wrt_param(&graph, idx)?);
        let t_arr = [t];
        let inputs = slot_inputs(&tape.signature, x, u, &t_arr, mem)?;

        let mut values = vec![0.0; tape.n_out];
        tape.call_into(&inputs, &mut values);
        Some(values)
    }
}

/// Build the width-keyed Jacobian tape from a resolved graph: a values-only tape
/// over the structurally-nonzero `∂out/∂wrt` entries plus their coordinate
/// pattern, and the gated presentation mode. `None` when the slot is absent or
/// the Jacobian is structurally all-zero.
fn build_jac_tape(graph: &Graph, wrt: &str, width: usize) -> Option<JacTape> {
    let (jg, rows, cols) = jacobian_sparse_wrt_slot(graph, wrt)?;
    if rows.is_empty() {
        return None;
    }
    let n = graph.signature.slot(wrt)?.size;
    let nnz = rows.len();
    let mode = if n == 1 {
        JacMode::Scalar
    } else if n >= LINSOLVE_SPARSE_MIN_DIM
        && (nnz as f64) <= LINSOLVE_SPARSE_MAX_DENSITY * (n * n) as f64
        && !jacobian_is_constant(graph, wrt)
    {
        // Large, genuinely sparse, state-dependent: carry the pattern. Constant
        // Jacobians stay dense so the linear solver's byte-identical
        // factorization cache keeps firing (a per-step sparse rebuild would
        // forfeit it); small / dense ones are faster dense.
        JacMode::Sparse
    } else {
        JacMode::Dense
    };
    Some(JacTape {
        width,
        tape: InterpretedFn::from_graph(jg),
        rows: rows.into(),
        cols: cols.into(),
        n,
        mode,
        val_scratch: RefCell::new(Vec::new()),
    })
}

/// Map a Jacobian tape's input signature to slot-ordered slices. The convention
/// matches the `blockops` slot naming / the static compile: `"x"` state, `"u"` flat inputs, `"t"`
/// scalar time, and the discrete memory — either flat `"m"` (the fused compile
/// signature) or per-slot `"mem{k}"` (block-internal discrete blocks). Memory is
/// a fixed input for the continuous-step Jacobian (`∂(dx/dt)/∂x` does not
/// differentiate it). Returns `None` for an unrecognised slot.
fn slot_inputs<'a>(
    sig: &InputSignature,
    x: &'a [f64],
    u: &'a [f64],
    t: &'a [f64],
    mem: &'a [f64],
) -> Option<SmallVec<[&'a [f64]; 4]>> {
    // Stack-backed for the usual <= 4 slots (x, u, t, m) — no heap allocation
    // of the input-slice vector per Jacobian evaluation (issue #43).
    let mut inputs: SmallVec<[&'a [f64]; 4]> = SmallVec::with_capacity(sig.slots.len());
    let mut mem_off = 0usize;
    for slot in &sig.slots {
        let name = slot.name.as_str();
        if name == "x" {
            inputs.push(x);
        } else if name == "u" {
            inputs.push(u);
        } else if name == "t" {
            inputs.push(t);
        } else if name == "m" {
            // The flat memory must cover the declared slot size; if it does not
            // (no live memory supplied), bail so the caller falls back rather than
            // reading out of range.
            if mem.len() < slot.size {
                return None;
            }
            inputs.push(&mem[..slot.size]);
        } else if name.starts_with("mem") {
            // Per-slot discrete memory `mem{k}`: carve the k-th slice out of the
            // flat `mem` by this slot's declared size, in signature order.
            let end = (mem_off + slot.size).min(mem.len());
            inputs.push(mem.get(mem_off..end).unwrap_or(&[]));
            mem_off = end;
        } else {
            return None;
        }
    }
    Some(inputs)
}

/// Assemble a `Jacobian` from an analytical dense-`∂f/∂x` closure (opaque
/// operators that carry one, e.g. an FMU). Mirrors the legacy `compute_jacobian`
/// dense path: `Scalar` for 1-D, `Matrix` otherwise; `None` if it produces nothing.
fn assemble_dense_jac(jac_fn: &BlockFn, x: &[f64], u: &[f64], t: f64) -> Option<Jacobian> {
    let mut buf = Vec::new();
    jac_fn(x, u, t, &mut buf);
    if buf.len() == x.len() * x.len() && !buf.is_empty() {
        if x.len() == 1 {
            Some(Jacobian::Scalar(buf[0]))
        } else {
            let n = x.len();
            Some(Jacobian::Matrix(buf, n))
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::blockops::RegionGraph;
    use crate::ssa::build::{Builder, GraphBuilder};
    use crate::ssa::graph::{Graph, InputSignature};

    /// Build a fixed `RegionGraph` for `dx/dt = f(x)` over `n` states, plus the
    /// matching native closure, from one generic builder body.
    fn op_from_state_rhs(
        n: usize,
        build: impl Fn(&GraphBuilder, &[u32], &mut Vec<u32>) + 'static + Copy,
        native: impl Fn(&[f64], &mut Vec<f64>) + 'static,
    ) -> Operator {
        let cell = std::cell::RefCell::new(Graph::new(InputSignature::from_named_sizes([
            ("x", n),
            ("u", 1),
            ("t", 1),
        ])));
        let outs = {
            let gb = GraphBuilder::new(&cell);
            let xs: Vec<u32> = (0..n as u32).map(|i| gb.input(i)).collect();
            let mut out = Vec::new();
            build(&gb, &xs, &mut out);
            out
        };
        let mut g = cell.into_inner();
        g.outputs = outs;
        let eval: BlockFn = Box::new(move |x, _u, _t, out| native(x, out));
        Operator::traceable(RegionGraph::Fixed(g), eval)
    }

    #[test]
    fn dense_small_jacobian_matches_hand_values() {
        // n = 2: dx0 = x0*x1, dx1 = -sin(x1). J = [[x1, x0], [0, -cos(x1)]].
        let op = op_from_state_rhs(
            2,
            |gb, xs, out| {
                out.push(gb.mul(xs[0], xs[1]));
                let s = gb.sin(xs[1]);
                out.push(gb.neg(s));
            },
            |x, out| {
                out.clear();
                out.push(x[0] * x[1]);
                out.push(-x[1].sin());
            },
        );
        let x = [0.7, -1.3];
        let j = op.jacobian_wrt_state(&x, &[0.0], 0.0, &[]).unwrap();
        match j {
            Jacobian::Matrix(m, n) => {
                assert_eq!(n, 2);
                let expect = [x[1], x[0], 0.0, -x[1].cos()];
                for i in 0..4 {
                    assert!((m[i] - expect[i]).abs() < 1e-12, "entry {i}: {} != {}", m[i], expect[i]);
                }
            }
            other => panic!("expected dense Matrix for n=2, got {other:?}"),
        }
    }

    #[test]
    fn large_sparse_jacobian_is_sparse_and_correct() {
        // n = 64 bidiagonal nonlinear: dx_i = -x_i + (i>0 ? x_{i-1}*x_i : 0).
        // J is bidiagonal (self + previous), state-dependent → Sparse mode.
        let n = 64usize;
        let op = op_from_state_rhs(
            n,
            move |gb, xs, out| {
                for i in 0..n {
                    let neg = gb.neg(xs[i]);
                    if i == 0 {
                        out.push(neg);
                    } else {
                        let coup = gb.mul(xs[i - 1], xs[i]);
                        out.push(gb.add(neg, coup));
                    }
                }
            },
            move |x, out| {
                out.clear();
                for i in 0..n {
                    let mut v = -x[i];
                    if i > 0 {
                        v += x[i - 1] * x[i];
                    }
                    out.push(v);
                }
            },
        );
        let x: Vec<f64> = (0..n).map(|i| 0.1 + 0.01 * i as f64).collect();
        let j = op.jacobian_wrt_state(&x, &[0.0], 0.0, &[]).unwrap();
        match j {
            Jacobian::Sparse(sj) => {
                assert_eq!(sj.n, n);
                // Bidiagonal: n diagonal + (n-1) sub-diagonal = 2n-1 nonzeros.
                assert_eq!(sj.values.len(), 2 * n - 1);
                // Spot-check: ∂(dx_i)/∂x_i = -1 + (i>0 ? x_{i-1} : 0);
                //             ∂(dx_i)/∂x_{i-1} = x_i.
                let dense = sj.to_dense();
                for i in 0..n {
                    let diag = -1.0 + if i > 0 { x[i - 1] } else { 0.0 };
                    assert!((dense[i * n + i] - diag).abs() < 1e-12, "diag {i}");
                    if i > 0 {
                        assert!((dense[i * n + (i - 1)] - x[i]).abs() < 1e-12, "sub {i}");
                    }
                }
            }
            other => panic!("expected Sparse for n=64 bidiagonal, got {other:?}"),
        }
    }

    #[test]
    fn opaque_operator_uses_numerical_jacobian() {
        // No graph → central-difference num_jac over the native closure.
        // f(x) = -2*x (scalar) → J = -2.
        let op = Operator::opaque(Box::new(|x, _u, _t, out: &mut Vec<f64>| {
            out.clear();
            out.push(-2.0 * x[0]);
        }));
        match op.jacobian_wrt_state(&[3.0], &[], 0.0, &[]).unwrap() {
            Jacobian::Scalar(v) => assert!((v + 2.0).abs() < 1e-6, "got {v}"),
            other => panic!("expected Scalar, got {other:?}"),
        }
    }

    /// Build a fixed `RegionGraph` over `("x", nx), ("u", nu)` from a builder body.
    fn op_from_xu(
        nx: usize,
        nu: usize,
        build: impl Fn(&GraphBuilder, &[u32], &[u32], &mut Vec<u32>),
    ) -> Operator {
        let cell = std::cell::RefCell::new(Graph::new(InputSignature::from_named_sizes([
            ("x", nx),
            ("u", nu),
        ])));
        let outs = {
            let gb = GraphBuilder::new(&cell);
            let xs: Vec<u32> = (0..nx as u32).map(|i| gb.input(i)).collect();
            let us: Vec<u32> = (0..nu as u32).map(|i| gb.input(nx as u32 + i)).collect();
            let mut out = Vec::new();
            build(&gb, &xs, &us, &mut out);
            out
        };
        let mut g = cell.into_inner();
        g.outputs = outs;
        Operator::graph_only(RegionGraph::Fixed(g))
    }

    /// The rectangular query must handle `n_out != n_slot`, which the square
    /// `Jacobian` representation behind `jacobian_wrt_state` cannot express.
    #[test]
    fn dense_jacobian_wrt_is_rectangular() {
        // 3 states, 2 inputs, 1 output: y = x0*u0 + x1 + 2*u1  (x2 unused).
        // ∂y/∂x = [u0, 1, 0]   (1x3)
        // ∂y/∂u = [x0, 2]      (1x2)
        let op = op_from_xu(3, 2, |gb, xs, us, out| {
            let a = gb.mul(xs[0], us[0]);
            let b = gb.add(a, xs[1]);
            let two = gb.cst(2.0);
            let c = gb.mul(two, us[1]);
            out.push(gb.add(b, c));
        });

        let (x, u) = ([0.7, -1.3, 5.0], [2.5, 4.0]);

        let (jx, rows, cols) = op.dense_jacobian_wrt("x", &x, &u, 0.0, &[]).unwrap();
        assert_eq!((rows, cols), (1, 3));
        for (got, want) in jx.iter().zip([u[0], 1.0, 0.0]) {
            assert!((got - want).abs() < 1e-12, "d/dx: {got} != {want}");
        }

        let (ju, rows, cols) = op.dense_jacobian_wrt("u", &x, &u, 0.0, &[]).unwrap();
        assert_eq!((rows, cols), (1, 2));
        for (got, want) in ju.iter().zip([x[0], 2.0]) {
            assert!((got - want).abs() < 1e-12, "d/du: {got} != {want}");
        }
    }

    /// Row-major layout: entry `(i, j)` sits at `i * n_cols + j`.
    #[test]
    fn dense_jacobian_wrt_is_row_major() {
        // 2 states, 1 input, 2 outputs: y0 = 3*x1, y1 = x0*u0.
        // ∂y/∂x = [[0, 3], [u0, 0]]
        let op = op_from_xu(2, 1, |gb, xs, us, out| {
            let three = gb.cst(3.0);
            out.push(gb.mul(three, xs[1]));
            out.push(gb.mul(xs[0], us[0]));
        });

        let (x, u) = ([1.5, -2.0], [4.0]);
        let (jx, rows, cols) = op.dense_jacobian_wrt("x", &x, &u, 0.0, &[]).unwrap();

        assert_eq!((rows, cols), (2, 2));
        let at = |i: usize, j: usize| jx[i * cols + j];
        assert!((at(0, 0) - 0.0).abs() < 1e-12);
        assert!((at(0, 1) - 3.0).abs() < 1e-12);
        assert!((at(1, 0) - u[0]).abs() < 1e-12);
        assert!((at(1, 1) - 0.0).abs() < 1e-12);
    }

    /// Opaque operators fall back to central differences, also rectangularly.
    #[test]
    fn dense_jacobian_wrt_opaque_fallback() {
        // f(x, u) = [x0 * u0, -3 * u0]  →  ∂f/∂u = [[x0], [-3]] (2x1)
        let op = Operator::opaque(Box::new(|x: &[f64], u: &[f64], _t, out: &mut Vec<f64>| {
            out.clear();
            out.push(x[0] * u[0]);
            out.push(-3.0 * u[0]);
        }));

        let (ju, rows, cols) = op.dense_jacobian_wrt("u", &[2.0], &[5.0], 0.0, &[]).unwrap();
        assert_eq!((rows, cols), (2, 1));
        assert!((ju[0] - 2.0).abs() < 1e-6, "got {}", ju[0]);
        assert!((ju[1] + 3.0).abs() < 1e-6, "got {}", ju[1]);
    }

    /// A `graph_only` operator carries a no-op closure; differencing it would
    /// silently return zeros, so the fallback must refuse instead.
    #[test]
    fn dense_jacobian_wrt_refuses_unknown_slot_on_graph_only() {
        let op = op_from_xu(1, 1, |gb, xs, us, out| {
            out.push(gb.mul(xs[0], us[0]));
        });
        assert!(op.dense_jacobian_wrt("nope", &[1.0], &[1.0], 0.0, &[]).is_none());
    }
}
