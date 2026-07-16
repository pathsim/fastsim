// Rust-based function tracer for JIT compilation.
//
// Instead of parsing Python source code (AST approach), we call the user's
// function with symbolic Tracer objects. Every arithmetic operation is
// recorded into an SSA Graph via __add__, __mul__, etc.
//
// This handles all callables (lambdas, closures, decorated, interactive)
// without needing inspect.getsource(). numpy interception via __array_ufunc__.
//
// Module layout: this file holds the shared graph handle (`SharedGraph`), the
// scalar `JitTracer` + its operand enums, the `where`/`clip` helpers, and the
// trace driver (`trace_with_signature`, `_trace_*`). The N-D `JitTracerArray`
// and its numpy-protocol surface live in the `array` submodule.

// The operator overloads return `Py<PyAny>` and frequently `.into()` an
// already-`Py<PyAny>` value for uniformity; clippy flags these as useless
// conversions but they keep the surface consistent.
#![allow(clippy::useless_conversion)]
// `__array_priority__` is a numpy dunder constant whose name is fixed by the
// numpy protocol and cannot be upper-cased.
#![allow(non_upper_case_globals)]

use pyo3::prelude::*;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::types::{PyDict, PyTuple};
use std::cell::RefCell;
use std::rc::Rc;

use crate::ssa::graph::*;
use super::ufunc_table::{binop_for_ufunc, cmpop_for_ufunc, unary_op_for_ufunc};

mod array;
pub use array::JitTracerArray;

/// Shared graph builder that multiple Tracer instances write into.
///
/// `Rc<RefCell>` rather than `Arc<Mutex>`: tracing always runs on the Python
/// thread under the GIL, and the tracer pyclasses are `unsendable`, so there
/// is no cross-thread access to synchronise — the former mutex paid an atomic
/// lock per emitted node for nothing.
#[derive(Clone)]
struct SharedGraph(Rc<RefCell<Graph>>);

impl SharedGraph {
    fn new(signature: InputSignature) -> Self {
        Self(Rc::new(RefCell::new(Graph::new(signature))))
    }
    /// Run `f` with mutable access to the graph — the batch-emission door:
    /// per-element loops emit their nodes under ONE borrow instead of one
    /// borrow per node.
    fn with<R>(&self, f: impl FnOnce(&mut Graph) -> R) -> R {
        f(&mut self.0.borrow_mut())
    }
    fn add(&self, node: Node) -> NodeId {
        self.0.borrow_mut().add(node)
    }
    /// Mint `n` consecutive `Input(base..base+n)` reads under a single borrow.
    fn add_input_range(&self, base: u32, n: usize) -> Vec<NodeId> {
        let mut g = self.0.borrow_mut();
        (0..n as u32).map(|i| g.add(Node::Input(base + i))).collect()
    }
    fn constant(&self, v: f64) -> NodeId {
        self.0.borrow_mut().constant(v)
    }
    fn binary(&self, op: BinOp, a: NodeId, b: NodeId) -> NodeId {
        self.0.borrow_mut().binary(op, a, b)
    }
    fn unary(&self, op: UnaryOp, a: NodeId) -> NodeId {
        self.0.borrow_mut().unary(op, a)
    }
    fn cmp(&self, op: CmpOp, a: NodeId, b: NodeId) -> NodeId {
        self.0.borrow_mut().cmp(op, a, b)
    }
    fn select(&self, c: NodeId, th: NodeId, el: NodeId) -> NodeId {
        self.0.borrow_mut().select(c, th, el)
    }
}

/// A symbolic f64 value that records operations into the SSA graph.
#[pyclass(name = "JitTracer", from_py_object, unsendable)]
#[derive(Clone)]
pub struct JitTracer {
    node_id: NodeId,
    graph: SharedGraph,
}

impl JitTracer {
    fn new(graph: &SharedGraph, node_id: NodeId) -> Self {
        Self { node_id, graph: graph.clone() }
    }

    fn ensure_id(&self, other: &TracerOrFloat) -> NodeId {
        match other {
            TracerOrFloat::Tracer(t) => t.node_id,
            TracerOrFloat::Float(v) => self.graph.constant(*v),
        }
    }

    fn binop(&self, op: BinOp, other: &TracerOrFloat) -> Self {
        let rhs = self.ensure_id(other);
        Self::new(&self.graph, self.graph.binary(op, self.node_id, rhs))
    }

    fn rbinop(&self, op: BinOp, other: &TracerOrFloat) -> Self {
        let lhs = self.ensure_id(other);
        Self::new(&self.graph, self.graph.binary(op, lhs, self.node_id))
    }
}

/// Accepts either a JitTracer or a plain float.
#[derive(FromPyObject)]
pub enum TracerOrFloat {
    Tracer(JitTracer),
    Float(f64),
}

/// Accepts any operand valid in an element-wise `JitTracerArray` binary op.
/// Order matters for FromPyObject: try most specific (tracer types) first so a
/// list-looking Python object isn't eagerly extracted as `NdArray`.
#[derive(FromPyObject)]
enum ArrayBinArg {
    TracerArray(JitTracerArray),
    Tracer(JitTracer),
    Float(f64),
    /// numpy arrays or plain Python lists with numeric content
    NdArray(Vec<f64>),
}

#[pymethods]
impl JitTracer {
    // --- Arithmetic ---
    fn __add__(&self, other: TracerOrFloat) -> Self { self.binop(BinOp::Add, &other) }
    fn __radd__(&self, other: TracerOrFloat) -> Self { self.rbinop(BinOp::Add, &other) }
    fn __sub__(&self, other: TracerOrFloat) -> Self { self.binop(BinOp::Sub, &other) }
    fn __rsub__(&self, other: TracerOrFloat) -> Self { self.rbinop(BinOp::Sub, &other) }
    fn __mul__(&self, other: TracerOrFloat) -> Self { self.binop(BinOp::Mul, &other) }
    fn __rmul__(&self, other: TracerOrFloat) -> Self { self.rbinop(BinOp::Mul, &other) }
    fn __truediv__(&self, other: TracerOrFloat) -> Self { self.binop(BinOp::Div, &other) }
    fn __rtruediv__(&self, other: TracerOrFloat) -> Self { self.rbinop(BinOp::Div, &other) }
    fn __pow__(&self, other: TracerOrFloat, _modulo: Option<&Bound<'_, PyAny>>) -> Self {
        self.binop(BinOp::Pow, &other)
    }
    fn __rpow__(&self, other: TracerOrFloat, _modulo: Option<&Bound<'_, PyAny>>) -> Self {
        self.rbinop(BinOp::Pow, &other)
    }
    // Python `%` is floored modulo — see `floored_mod` (`fmod` keeps raw Mod).
    fn __mod__(&self, other: TracerOrFloat) -> Self {
        let rhs = self.ensure_id(&other);
        Self::new(&self.graph, floored_mod(&self.graph, self.node_id, rhs))
    }
    fn __rmod__(&self, other: TracerOrFloat) -> Self {
        let lhs = self.ensure_id(&other);
        Self::new(&self.graph, floored_mod(&self.graph, lhs, self.node_id))
    }
    // Floor division `a // b` = floor(a / b). Common for stepwise PRNG keys
    // (`t // dt`) and binning; lowers to Div + Floor so it traces.
    fn __floordiv__(&self, other: TracerOrFloat) -> Self {
        let q = self.binop(BinOp::Div, &other);
        Self::new(&self.graph, self.graph.unary(UnaryOp::Floor, q.node_id))
    }
    fn __rfloordiv__(&self, other: TracerOrFloat) -> Self {
        let q = self.rbinop(BinOp::Div, &other);
        Self::new(&self.graph, self.graph.unary(UnaryOp::Floor, q.node_id))
    }
    fn __neg__(&self) -> Self {
        Self::new(&self.graph, self.graph.unary(UnaryOp::Neg, self.node_id))
    }
    fn __pos__(&self) -> Self { self.clone() }
    fn __abs__(&self) -> Self {
        Self::new(&self.graph, self.graph.unary(UnaryOp::Abs, self.node_id))
    }

    // --- Comparisons (return Tracer with 0.0/1.0) ---
    fn __gt__(&self, other: TracerOrFloat) -> Self {
        let rhs = self.ensure_id(&other);
        Self::new(&self.graph, self.graph.cmp(CmpOp::Gt, self.node_id, rhs))
    }
    fn __ge__(&self, other: TracerOrFloat) -> Self {
        let rhs = self.ensure_id(&other);
        Self::new(&self.graph, self.graph.cmp(CmpOp::Ge, self.node_id, rhs))
    }
    fn __lt__(&self, other: TracerOrFloat) -> Self {
        let rhs = self.ensure_id(&other);
        Self::new(&self.graph, self.graph.cmp(CmpOp::Lt, self.node_id, rhs))
    }
    fn __le__(&self, other: TracerOrFloat) -> Self {
        let rhs = self.ensure_id(&other);
        Self::new(&self.graph, self.graph.cmp(CmpOp::Le, self.node_id, rhs))
    }
    fn __eq__(&self, other: TracerOrFloat) -> Self {
        let rhs = self.ensure_id(&other);
        Self::new(&self.graph, self.graph.cmp(CmpOp::Eq, self.node_id, rhs))
    }
    fn __ne__(&self, other: TracerOrFloat) -> Self {
        let rhs = self.ensure_id(&other);
        Self::new(&self.graph, self.graph.cmp(CmpOp::Ne, self.node_id, rhs))
    }

    // --- Bool (raises error during trace; guide user toward np.where) ---
    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "JitTracer cannot be used in if/else during JIT tracing. \
             Rewrite the branch as `np.where(cond, then_val, else_val)`."
        ))
    }
    fn __float__(&self) -> PyResult<f64> {
        Err(PyTypeError::new_err("JitTracer cannot be converted to float during tracing."))
    }

    // --- numpy ufunc interception ---
    #[pyo3(signature = (ufunc, method, *args, **_kwargs))]
    fn __array_ufunc__(
        &self,
        py: Python<'_>,
        ufunc: &Bound<'_, PyAny>,
        method: &str,
        args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        if method != "__call__" {
            return Ok(py.NotImplemented().into());
        }
        let name: String = ufunc.getattr("__name__")?.extract()?;

        // Composite ufuncs: expressed via existing ops (see `unary_composite`),
        // so they trace, AD and codegen for free.
        if matches!(name.as_str(),
            "deg2rad" | "radians" | "rad2deg" | "degrees" | "square"
            | "reciprocal" | "exp2" | "expit")
        {
            let t = self.extract_tracer_arg(args, 0)?;
            let r = unary_composite(&t.graph, &name, t.node_id)
                .expect("name matched the composite list");
            return Ok(Py::new(py, JitTracer::new(&t.graph, r))?.into_any().into());
        }

        // Unary / binary / comparison dispatch all share the same ufunc-name
        // tables as the array tracer (`tracer::ufunc_table`), so scalar and
        // array paths stay in lockstep (e.g. `not_equal` works on both).
        if let Some(op) = unary_op_for_ufunc(&name) {
            let tracer = self.extract_tracer_arg(args, 0)?;
            let result = JitTracer::new(&tracer.graph, tracer.graph.unary(op, tracer.node_id));
            return Ok(Py::new(py, result)?.into_any().into());
        }

        // Binary ufuncs (arithmetic, comparison, binary composites). A scalar
        // tracer can meet an ARRAY operand here — `np.minimum(x[0], x)` lands
        // on the first input's `__array_ufunc__`, i.e. this one. Promote to
        // the array elementwise path (operand order preserved via `swap`)
        // instead of erroring out of the trace.
        let bin_op = binop_for_ufunc(&name);
        let cmp_op = cmpop_for_ufunc(&name);
        let is_bin_composite = matches!(name.as_str(),
            "copysign" | "logaddexp" | "heaviside" | "remainder" | "mod");
        if (bin_op.is_some() || cmp_op.is_some() || is_bin_composite) && args.len() == 2 {
            let a = self.classify_operand(&args.get_item(0)?)?;
            let b = self.classify_operand(&args.get_item(1)?)?;
            let g = &self.graph;
            return match (a, b) {
                (UfuncOperand::Scalar(x), UfuncOperand::Scalar(y)) => {
                    let r = if let Some(op) = bin_op {
                        g.binary(op, x, y)
                    } else if let Some(op) = cmp_op {
                        g.cmp(op, x, y)
                    } else {
                        binary_composite(g, &name, x, y)
                            .expect("name matched the composite list")
                    };
                    Ok(Py::new(py, JitTracer::new(g, r))?.into_any().into())
                }
                (UfuncOperand::Array(ta), UfuncOperand::Scalar(y)) => {
                    let other = ArrayBinArg::Tracer(JitTracer::new(g, y));
                    ta.ufunc_elementwise(py, &name, bin_op, cmp_op, other, false)
                }
                (UfuncOperand::Scalar(x), UfuncOperand::Array(ta)) => {
                    let other = ArrayBinArg::Tracer(JitTracer::new(g, x));
                    ta.ufunc_elementwise(py, &name, bin_op, cmp_op, other, true)
                }
                (UfuncOperand::Array(ta), UfuncOperand::Array(tb)) => {
                    let other = ArrayBinArg::TracerArray(tb);
                    ta.ufunc_elementwise(py, &name, bin_op, cmp_op, other, false)
                }
            };
        }

        Ok(py.NotImplemented().into())
    }

    /// NEP 18: intercept np.clip, np.where on scalar tracers
    #[pyo3(signature = (func, _types, args, _kwargs=None))]
    fn __array_function__(
        &self,
        py: Python<'_>,
        func: &Bound<'_, PyAny>,
        _types: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let name: String = func.getattr("__name__")?.extract()?;
        match name.as_str() {
            "clip" => {
                if args.len() < 3 { return Ok(py.NotImplemented().into()); }
                let lo = self.ensure_id(&args.get_item(1)?.extract::<TracerOrFloat>()?);
                let hi = self.ensure_id(&args.get_item(2)?.extract::<TracerOrFloat>()?);
                let clamped_lo = self.graph.binary(BinOp::Max, self.node_id, lo);
                let clamped = self.graph.binary(BinOp::Min, clamped_lo, hi);
                Ok(Py::new(py, JitTracer::new(&self.graph, clamped))?.into_any().into())
            }
            // Full where dispatch (shared with the array tracer): scalar or
            // array condition (traced or constant), scalar or array branches,
            // size-1 broadcasting. NOTE: numpy dispatches here whenever ANY
            // argument is a tracer — the condition is args[0], not `self`.
            "where" => array::np_where_dispatch(py, &self.graph, args),
            // np.searchsorted over a constant grid on a scalar tracer value
            // (numpy dispatches here because the grid is a plain array).
            "searchsorted" => array::np_searchsorted_dispatch(py, &self.graph, args, _kwargs),
            // np.interp over a constant ascending grid lowers to a select
            // chain (see `emit_interp`). Extra kwargs (left/right/period) are
            // not supported and fall through to NotImplemented.
            "interp" if _kwargs.is_none_or(|k| k.is_empty()) => {
                let Some((xp, fp)) = extract_interp_grids(args) else {
                    return Ok(py.NotImplemented().into());
                };
                let Ok(x) = args.get_item(0)?.extract::<JitTracer>() else {
                    return Ok(py.NotImplemented().into());
                };
                let r = emit_interp(&x.graph, x.node_id, &xp, &fp);
                Ok(Py::new(py, JitTracer::new(&x.graph, r))?.into_any().into())
            }
            // np.atleast_1d on a scalar tracer: a 1-element array view.
            "atleast_1d" => {
                if args.len() != 1 { return Ok(py.NotImplemented().into()); }
                let Ok(t) = args.get_item(0)?.extract::<JitTracer>() else {
                    return Ok(py.NotImplemented().into());
                };
                let arr = JitTracerArray::from_nodes(&t.graph, vec![t.node_id]);
                Ok(Py::new(py, arr)?.into_any().into())
            }
            // np.stack / np.hstack / np.vstack / np.concatenate / np.asarray /
            // np.array on a list of tracers → build a JitTracerArray from nodes.
            "stack" | "hstack" | "vstack" | "concatenate" | "asarray" | "array" => {
                if args.is_empty() { return Ok(py.NotImplemented().into()); }
                let seq = args.get_item(0)?;
                self.collect_tracer_sequence(py, &seq)
                    .map(|ids| JitTracerArray::from_nodes(&self.graph, ids))
                    .and_then(|ta| Ok(Py::new(py, ta)?.into_any().into()))
                    .or_else(|_| Ok(py.NotImplemented().into()))
            }
            _ => Ok(py.NotImplemented().into()),
        }
    }

    /// Extract a sequence of tracer/float/nested-tracer-array into a flat list
    /// of node ids. Used by np.stack / np.concatenate / np.array interception.
    fn collect_tracer_sequence(&self, _py: Python<'_>, seq: &Bound<'_, PyAny>) -> PyResult<Vec<NodeId>> {
        let mut out = Vec::new();
        let iter = seq.try_iter()?;
        for item in iter {
            let item = item?;
            if let Ok(t) = item.extract::<JitTracer>() {
                out.push(t.node_id);
            } else if let Ok(ta) = item.extract::<JitTracerArray>() {
                for i in 0..ta.size { out.push(ta.get_node(i)); }
            } else if let Ok(v) = item.extract::<f64>() {
                out.push(self.graph.constant(v));
            } else {
                return Err(PyTypeError::new_err(
                    "tracer sequence: expected JitTracer / JitTracerArray / float"));
            }
        }
        Ok(out)
    }

    fn __repr__(&self) -> String {
        format!("JitTracer(node={})", self.node_id)
    }
}

/// A binary-ufunc operand as the scalar tracer sees it: a scalar graph node
/// (tracer or constant) or an array-like (tracer array / numeric sequence).
pub(super) enum UfuncOperand {
    Scalar(NodeId),
    Array(JitTracerArray),
}

impl JitTracer {
    /// Extract a JitTracer from a ufunc argument (may be Tracer or float).
    fn extract_tracer_arg(&self, args: &Bound<'_, PyTuple>, idx: usize) -> PyResult<JitTracer> {
        let arg = args.get_item(idx)?;
        if let Ok(t) = arg.extract::<JitTracer>() {
            Ok(t)
        } else if let Ok(v) = arg.extract::<f64>() {
            Ok(JitTracer::new(&self.graph, self.graph.constant(v)))
        } else {
            Err(PyTypeError::new_err("Expected JitTracer or float"))
        }
    }

    /// Classify one binary-ufunc operand (see [`UfuncOperand`]). Constants are
    /// minted into the graph; plain numeric sequences become constant arrays.
    fn classify_operand(&self, arg: &Bound<'_, PyAny>) -> PyResult<UfuncOperand> {
        if let Ok(t) = arg.extract::<JitTracer>() {
            return Ok(UfuncOperand::Scalar(t.node_id));
        }
        if let Ok(v) = arg.extract::<f64>() {
            return Ok(UfuncOperand::Scalar(self.graph.constant(v)));
        }
        if let Ok(ta) = arg.extract::<JitTracerArray>() {
            return Ok(UfuncOperand::Array(ta));
        }
        if let Ok(vs) = arg.extract::<Vec<f64>>() {
            let nodes = vs.iter().map(|&v| self.graph.constant(v)).collect();
            return Ok(UfuncOperand::Array(JitTracerArray::from_nodes(&self.graph, nodes)));
        }
        Err(PyTypeError::new_err(
            "unsupported ufunc operand during JIT trace (expected tracer, float, or numeric array)"
        ))
    }
}


/// fastsim.where(cond, then_val, else_val) — branchless conditional.
#[pyfunction]
#[pyo3(name = "where_")]
pub fn jit_where(cond: &JitTracer, then_val: TracerOrFloat, else_val: TracerOrFloat) -> JitTracer {
    let th = cond.ensure_id(&then_val);
    let el = cond.ensure_id(&else_val);
    JitTracer::new(&cond.graph, cond.graph.select(cond.node_id, th, el))
}

/// fastsim.clip(x, lo, hi) — clamp value.
#[pyfunction]
#[pyo3(name = "clip")]
pub fn jit_clip(x: &JitTracer, lo: TracerOrFloat, hi: TracerOrFloat) -> JitTracer {
    let lo_id = x.ensure_id(&lo);
    let hi_id = x.ensure_id(&hi);
    let clamped_lo = x.graph.binary(BinOp::Max, x.node_id, lo_id);
    let clamped = x.graph.binary(BinOp::Min, clamped_lo, hi_id);
    JitTracer::new(&x.graph, clamped)
}

/// fastsim.random_uniform(key) — stateless, traceable uniform in `[0, 1)`.
///
/// A *pure function* of `key`: same key → same draw, every run. Unlike
/// `np.random.*` (untraceable hidden state) this lowers to a single SSA node and
/// JITs. For per-step noise feed a stepwise key, e.g. `random_uniform(floor(t/dt))`.
#[pyfunction]
#[pyo3(name = "random_uniform")]
pub fn jit_random_uniform(key: &JitTracer) -> JitTracer {
    JitTracer::new(&key.graph, key.graph.unary(UnaryOp::RandUniform, key.node_id))
}

/// fastsim.random_normal(key) — stateless, traceable standard normal `N(0, 1)`.
///
/// Box-Muller over two decorrelated uniform draws (`key` and `key + 0.5`),
/// composed from the `RandUniform` node and ordinary math ops — so it traces,
/// codegens via the same path, and replays bit-for-bit.
#[pyfunction]
#[pyo3(name = "random_normal")]
pub fn jit_random_normal(key: &JitTracer) -> JitTracer {
    let g = &key.graph;
    // u1 = U(key), u2 = U(key + 0.5)  (the offset decorrelates the two streams)
    let half = g.constant(0.5);
    let key2 = g.binary(BinOp::Add, key.node_id, half);
    let u1 = g.unary(UnaryOp::RandUniform, key.node_id);
    let u2 = g.unary(UnaryOp::RandUniform, key2);
    // r = sqrt(-2 ln(max(u1, tiny)))   (guard u1 == 0 → -inf)
    let tiny = g.constant(f64::MIN_POSITIVE);
    let u1c = g.binary(BinOp::Max, u1, tiny);
    let lnu = g.unary(UnaryOp::Log, u1c);
    let neg2 = g.constant(-2.0);
    let m2ln = g.binary(BinOp::Mul, neg2, lnu);
    let r = g.unary(UnaryOp::Sqrt, m2ln);
    // z = r * cos(2π u2)
    let twopi = g.constant(std::f64::consts::TAU);
    let ang = g.binary(BinOp::Mul, twopi, u2);
    let c = g.unary(UnaryOp::Cos, ang);
    JitTracer::new(g, g.binary(BinOp::Mul, r, c))
}


// ======================================================================================
// Composite ufuncs: lowered to existing ops at trace time (no dedicated opcode),
// so they trace, AD and codegen through the ops they expand to. Shared by the
// scalar (`JitTracer`) and array (`JitTracerArray`) ufunc paths.
// ======================================================================================

/// Emit a unary composite ufunc by name, or `None` if the name is not one.
/// The `&mut Graph` core lets array loops emit N elements under one borrow.
pub(super) fn unary_composite_g(g: &mut Graph, name: &str, x: NodeId) -> Option<NodeId> {
    Some(match name {
        // numpy aliases the same conversion under two ufunc names each.
        "deg2rad" | "radians" => {
            let c = g.constant(std::f64::consts::PI / 180.0);
            g.binary(BinOp::Mul, x, c)
        }
        "rad2deg" | "degrees" => {
            let c = g.constant(180.0 / std::f64::consts::PI);
            g.binary(BinOp::Mul, x, c)
        }
        // `np.square` stays `x*x`, keeping the optimizer's reuse and the
        // existing Mul derivative rule.
        "square" => g.binary(BinOp::Mul, x, x),
        "reciprocal" => {
            let one = g.constant(1.0);
            g.binary(BinOp::Div, one, x)
        }
        "exp2" => {
            let two = g.constant(2.0);
            g.binary(BinOp::Pow, two, x)
        }
        // scipy.special.expit (logistic sigmoid): 1 / (1 + exp(-x)).
        "expit" => {
            let one = g.constant(1.0);
            let nx = g.unary(UnaryOp::Neg, x);
            let e = g.unary(UnaryOp::Exp, nx);
            let d = g.binary(BinOp::Add, one, e);
            g.binary(BinOp::Div, one, d)
        }
        _ => return None,
    })
}

fn unary_composite(g: &SharedGraph, name: &str, x: NodeId) -> Option<NodeId> {
    g.with(|g| unary_composite_g(g, name, x))
}

/// Emit a binary composite ufunc by name, or `None` if the name is not one.
/// The `&mut Graph` core lets array loops emit N elements under one borrow.
pub(super) fn binary_composite_g(g: &mut Graph, name: &str, a: NodeId, b: NodeId) -> Option<NodeId> {
    Some(match name {
        // copysign(a, b) = signbit(b) ? -|a| : |a|. NOT `|a| * sign(b)`: `sign`
        // now has numpy semantics (sign(0) = 0), which would zero the result
        // for b == ±0. IEEE signbit is composed from existing ops as
        // `b < 0 || 1/b < 0` — the reciprocal leg is what catches b == -0.0
        // (1/-0.0 = -inf), keeping numpy's sign-of-zero behaviour exact.
        "copysign" => {
            let zero = g.constant(0.0);
            let one = g.constant(1.0);
            let abs_a = g.unary(UnaryOp::Abs, a);
            let neg_abs_a = g.unary(UnaryOp::Neg, abs_a);
            let b_neg = g.cmp(CmpOp::Lt, b, zero);
            let recip = g.binary(BinOp::Div, one, b);
            let recip_neg = g.cmp(CmpOp::Lt, recip, zero);
            let signbit = g.binary(BinOp::Max, b_neg, recip_neg);
            g.select(signbit, neg_abs_a, abs_a)
        }
        // Overflow-safe log(exp(a) + exp(b)) = max(a,b) + log1p(exp(-|a-b|)),
        // the same rearrangement numpy's C kernel uses.
        "logaddexp" => {
            let m = g.binary(BinOp::Max, a, b);
            let d = g.binary(BinOp::Sub, a, b);
            let ad = g.unary(UnaryOp::Abs, d);
            let nd = g.unary(UnaryOp::Neg, ad);
            let e = g.unary(UnaryOp::Exp, nd);
            let l = g.unary(UnaryOp::Log1p, e);
            g.binary(BinOp::Add, m, l)
        }
        // Floored modulo (`np.remainder` / `np.mod`); `fmod` stays the raw op.
        "remainder" | "mod" => floored_mod_g(g, a, b),
        // heaviside(a, h0): 0 for a<0, h0 at a==0, 1 for a>0. Built from strict
        // Gt/Lt comparisons only — no tolerance-banded Eq involved.
        "heaviside" => {
            let zero = g.constant(0.0);
            let one = g.constant(1.0);
            let gt = g.cmp(CmpOp::Gt, a, zero);
            let lt = g.cmp(CmpOp::Lt, a, zero);
            let at_zero = g.select(lt, zero, b);
            g.select(gt, one, at_zero)
        }
        _ => return None,
    })
}

fn binary_composite(g: &SharedGraph, name: &str, a: NodeId, b: NodeId) -> Option<NodeId> {
    g.with(|g| binary_composite_g(g, name, a, b))
}

/// Emit Python/numpy FLOORED modulo (`a % b`, `np.remainder`): the result's
/// sign follows the DIVISOR, while the raw `Mod` op is C `fmod` (sign follows
/// the dividend). Composes numpy's own fixup from existing ops:
/// `m = fmod(a, b); if m != 0 and sign(m) != sign(b): m += b`. The `m != 0`
/// guard must be EXACT (numpy's is): near-exact multiples leave `m` within a
/// few ULP of zero, where the tolerance-banded `Ne` would wrongly suppress
/// the fixup (a finite jump of `b`). `Gt(|m|, 0)` is exact and band-free.
/// `np.fmod` keeps the raw `Mod` lowering.
pub(super) fn floored_mod_g(g: &mut Graph, a: NodeId, b: NodeId) -> NodeId {
    let m = g.binary(BinOp::Mod, a, b);
    let zero = g.constant(0.0);
    let m_neg = g.cmp(CmpOp::Lt, m, zero);
    let b_neg = g.cmp(CmpOp::Lt, b, zero);
    let mismatch = g.cmp(CmpOp::Ne, m_neg, b_neg);
    let abs_m = g.unary(UnaryOp::Abs, m);
    let nonzero = g.cmp(CmpOp::Gt, abs_m, zero);
    let fix = g.binary(BinOp::Mul, nonzero, mismatch);
    let m_plus_b = g.binary(BinOp::Add, m, b);
    g.select(fix, m_plus_b, m)
}

fn floored_mod(g: &SharedGraph, a: NodeId, b: NodeId) -> NodeId {
    g.with(|g| floored_mod_g(g, a, b))
}

/// Emit `np.interp(x, xp, fp)` for one scalar node: piecewise-linear
/// interpolation over ascending breakpoints, clamped to `fp[0]` / `fp[last]`
/// outside the grid (numpy's default `left`/`right`). Lowered as a select
/// chain — segment `i`'s line wins while `x >= xp[i]`, the final select pins
/// the right clamp — so it traces, ADs (piecewise gradients) and codegens
/// through existing ops.
pub(super) fn emit_interp_g(g: &mut Graph, x: NodeId, xp: &[f64], fp: &[f64]) -> NodeId {
    debug_assert_eq!(xp.len(), fp.len());
    debug_assert!(!xp.is_empty());
    let mut y = g.constant(fp[0]);
    for i in 0..xp.len().saturating_sub(1) {
        let slope = (fp[i + 1] - fp[i]) / (xp[i + 1] - xp[i]);
        let xi = g.constant(xp[i]);
        let dx = g.binary(BinOp::Sub, x, xi);
        let sl = g.constant(slope);
        let t = g.binary(BinOp::Mul, dx, sl);
        let fpi = g.constant(fp[i]);
        let yi = g.binary(BinOp::Add, fpi, t);
        let cond = g.cmp(CmpOp::Ge, x, xi);
        y = g.select(cond, yi, y);
    }
    let last = xp.len() - 1;
    let xl = g.constant(xp[last]);
    let fl = g.constant(fp[last]);
    let cond = g.cmp(CmpOp::Ge, x, xl);
    g.select(cond, fl, y)
}

fn emit_interp(g: &SharedGraph, x: NodeId, xp: &[f64], fp: &[f64]) -> NodeId {
    g.with(|g| emit_interp_g(g, x, xp, fp))
}

/// Extract `np.interp`'s constant breakpoint grids (`xp`, `fp`) from call args.
/// `None` when they are not plain numeric sequences (e.g. traced values — the
/// grid must be constant for the select-chain lowering).
pub(super) fn extract_interp_grids(
    args: &Bound<'_, PyTuple>,
) -> Option<(Vec<f64>, Vec<f64>)> {
    if args.len() < 3 {
        return None;
    }
    let xp: Vec<f64> = args.get_item(1).ok()?.extract().ok()?;
    let fp: Vec<f64> = args.get_item(2).ok()?.extract().ok()?;
    if xp.is_empty() || xp.len() != fp.len() {
        return None;
    }
    // numpy requires ascending xp; a non-ascending grid would silently change
    // meaning under the select chain, so refuse to lower it.
    if xp.windows(2).any(|w| w[1] <= w[0]) {
        return None;
    }
    Some((xp, fp))
}

/// One positional argument to the traced function.
pub enum TraceArg<'a> {
    /// Symbolic array of `size` elements; slot name for diagnostics + AD lookup.
    Array { name: &'a str, size: usize },
    /// Symbolic scalar; slot name for diagnostics + AD lookup.
    Scalar { name: &'a str },
}

/// Trace `func(arg0, arg1, …)` into an SSA graph.  The signature specifies the
/// positional arguments in order; each `Array` becomes a `JitTracerArray` with
/// a dedicated input slot, each `Scalar` becomes a `JitTracer`.
///
/// This is the single generic entry point — ODE, Function, DAE, MassMatrix
/// blocks and the standalone `jit(func)` all use it with different signatures.
/// Python-callable installed over numpy's `zeros`/`empty`/`ones`/`full` during
/// a trace, so the imperative `np.zeros(n)` idiom records constant nodes into
/// the graph instead of allocating a real ndarray (which would reject tracer
/// assignment with "cannot be converted to float").
#[pyclass(unsendable)]
struct ArrayCtor {
    graph: SharedGraph,
    fill: f64,
    is_full: bool,
    /// The real numpy constructor, for signatures the patch does not model
    /// (integer/bool dtypes, order=, like=): delegating yields a plain
    /// ndarray, which still traces for read-only use — instead of silently
    /// minting f64 tracer nodes under a non-float dtype.
    orig: Py<PyAny>,
}

#[pymethods]
impl ArrayCtor {
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        // kwargs policy: model only a float dtype; delegate anything else.
        if let Some(k) = kwargs {
            for (key, val) in k.iter() {
                let is_float_dtype = key.extract::<String>().is_ok_and(|k| k == "dtype")
                    && py.import("numpy")
                        .and_then(|np| np.call_method1("dtype", (val,)))
                        .and_then(|d| d.getattr("kind"))
                        .and_then(|kind| kind.extract::<String>())
                        .is_ok_and(|kind| kind == "f");
                if !is_float_dtype {
                    return self.orig.bind(py).call(args, kwargs).map(|r| r.unbind());
                }
            }
        }
        if args.is_empty() {
            return Err(PyValueError::new_err("array constructor requires a shape"));
        }
        let shape_obj = args.get_item(0)?;
        let shape: Vec<usize> = if let Ok(n) = shape_obj.extract::<usize>() {
            vec![n]
        } else if let Ok(t) = shape_obj.extract::<Vec<usize>>() {
            t
        } else {
            // Unmodelled shape argument: hand it to the real constructor.
            return self.orig.bind(py).call(args, kwargs).map(|r| r.unbind());
        };
        let size: usize = shape.iter().product();
        let fill_node = if self.is_full {
            let fv = args.get_item(1)
                .map_err(|_| PyValueError::new_err("np.full requires a fill_value"))?;
            if let Ok(t) = fv.extract::<JitTracer>() { t.node_id }
            else if let Ok(v) = fv.extract::<f64>() { self.graph.constant(v) }
            else { return Err(PyTypeError::new_err("np.full: unsupported fill_value")); }
        } else {
            self.graph.constant(self.fill)
        };
        let nodes = vec![fill_node; size];
        Ok(Py::new(py, JitTracerArray::from_nodes_shape(&self.graph, nodes, shape))?
            .into_any().into())
    }
}

/// Python-callable installed over numpy's constant array FACTORIES
/// (`arange`/`linspace`/`eye`/`diag`) during a trace. They carry no array
/// argument, so `__array_function__` cannot intercept them — but their results
/// are common ASSIGNMENT TARGETS (`a = np.arange(3.0); a[0] = expr`), and a
/// real ndarray rejects tracer assignment. The patch returns a constant-node
/// `JitTracerArray` instead. Signatures the patch does not model (dtype=,
/// retstep=, …) delegate to the saved original — the result is then a plain
/// constant ndarray, which still traces for read-only use.
#[pyclass(unsendable)]
struct ConstFactoryCtor {
    graph: SharedGraph,
    kind: &'static str,
    orig: Py<PyAny>,
}

impl ConstFactoryCtor {
    fn const_array(
        &self,
        py: Python<'_>,
        vals: Vec<f64>,
        shape: Vec<usize>,
    ) -> PyResult<Py<PyAny>> {
        let nodes: Vec<NodeId> = vals.iter().map(|&v| self.graph.constant(v)).collect();
        Ok(Py::new(py, JitTracerArray::from_nodes_shape(&self.graph, nodes, shape))?
            .into_any().into())
    }

    fn delegate(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        self.orig.bind(py).call(args, kwargs).map(|r| r.unbind())
    }
}

#[pymethods]
impl ConstFactoryCtor {
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        match self.kind {
            "arange" => {
                if kwargs.is_some_and(|k| !k.is_empty()) || args.is_empty() || args.len() > 3 {
                    return self.delegate(py, args, kwargs);
                }
                let mut vals = [0.0f64, 0.0, 1.0]; // start, stop, step
                for (i, slot) in (0..args.len()).zip(if args.len() == 1 { [1usize, 0, 0] } else { [0usize, 1, 2] }) {
                    match args.get_item(i)?.extract::<f64>() {
                        Ok(v) => vals[slot] = v,
                        Err(_) => return self.delegate(py, args, kwargs),
                    }
                }
                let (start, stop, step) = (vals[0], vals[1], vals[2]);
                if step == 0.0 || !step.is_finite() {
                    return self.delegate(py, args, kwargs);
                }
                // numpy's count: ceil((stop - start) / step), clamped at 0.
                let n = ((stop - start) / step).ceil().max(0.0) as usize;
                let out: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
                self.const_array(py, out, vec![n])
            }
            "linspace" => {
                // Positional (start, stop[, num]); `num` may also be a kwarg.
                // Any other kwarg (retstep, endpoint=False, axis, dtype) delegates.
                if args.len() < 2 || args.len() > 3 {
                    return self.delegate(py, args, kwargs);
                }
                let mut num: Option<usize> = None;
                if let Some(k) = kwargs {
                    for (key, val) in k.iter() {
                        match key.extract::<String>().as_deref() {
                            Ok("num") => match val.extract::<usize>() {
                                Ok(v) => num = Some(v),
                                Err(_) => return self.delegate(py, args, kwargs),
                            },
                            _ => return self.delegate(py, args, kwargs),
                        }
                    }
                }
                let (Ok(start), Ok(stop)) = (
                    args.get_item(0)?.extract::<f64>(),
                    args.get_item(1)?.extract::<f64>(),
                ) else {
                    return self.delegate(py, args, kwargs);
                };
                if args.len() == 3 {
                    match args.get_item(2)?.extract::<usize>() {
                        Ok(v) => num = Some(v),
                        Err(_) => return self.delegate(py, args, kwargs),
                    }
                }
                let n = num.unwrap_or(50);
                let mut out = Vec::with_capacity(n);
                if n == 1 {
                    out.push(start);
                } else if n > 1 {
                    let step = (stop - start) / (n - 1) as f64;
                    for i in 0..n {
                        out.push(start + i as f64 * step);
                    }
                    // numpy pins the endpoint exactly.
                    out[n - 1] = stop;
                }
                let len = out.len();
                self.const_array(py, out, vec![len])
            }
            "eye" => {
                if kwargs.is_some_and(|k| !k.is_empty()) || args.is_empty() || args.len() > 2 {
                    return self.delegate(py, args, kwargs);
                }
                let Ok(n) = args.get_item(0)?.extract::<usize>() else {
                    return self.delegate(py, args, kwargs);
                };
                let m = if args.len() == 2 {
                    match args.get_item(1)?.extract::<usize>() {
                        Ok(v) => v,
                        Err(_) => return self.delegate(py, args, kwargs),
                    }
                } else { n };
                let mut vals = vec![0.0; n * m];
                for i in 0..n.min(m) {
                    vals[i * m + i] = 1.0;
                }
                self.const_array(py, vals, vec![n, m])
            }
            "diag" => {
                if kwargs.is_some_and(|k| !k.is_empty()) || args.len() != 1 {
                    return self.delegate(py, args, kwargs);
                }
                let arg = args.get_item(0)?;
                // Traced 1-D vector -> n x n matrix with tracer nodes on the
                // diagonal (a genuinely useful traced idiom, e.g. M(x) builds).
                if let Ok(ta) = arg.extract::<JitTracerArray>() {
                    let n = ta.size;
                    let zero = self.graph.constant(0.0);
                    let mut nodes = vec![zero; n * n];
                    for i in 0..n {
                        nodes[i * n + i] = ta.get_node(i);
                    }
                    return Ok(Py::new(py, JitTracerArray::from_nodes_shape(
                        &self.graph, nodes, vec![n, n]))?.into_any().into());
                }
                if let Ok(vs) = arg.extract::<Vec<f64>>() {
                    let n = vs.len();
                    let mut vals = vec![0.0; n * n];
                    for (i, v) in vs.iter().enumerate() {
                        vals[i * n + i] = *v;
                    }
                    return self.const_array(py, vals, vec![n, n]);
                }
                self.delegate(py, args, kwargs)
            }
            _ => self.delegate(py, args, kwargs),
        }
    }
}

thread_local! {
    /// Nesting depth of active traces. The numpy constructor monkeypatches are
    /// installed only by the OUTERMOST trace: a nested trace re-patching would
    /// save the outer patch as "original" and restore it wrongly on exit.
    static TRACE_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

pub fn trace_with_signature(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
    args: &[TraceArg],
) -> PyResult<Option<crate::ssa::graph::Graph>> {
    // Build the InputSignature from the arg list.
    let slots: Vec<(String, usize)> = args.iter().map(|a| match a {
        TraceArg::Array { name, size } => (name.to_string(), *size),
        TraceArg::Scalar { name } => (name.to_string(), 1),
    }).collect();
    let sig = InputSignature::from_named_sizes(slots);
    let shared = SharedGraph::new(sig.clone());

    // Materialise each arg as a Python object backed by the flat input space.
    let py_args: Vec<Py<PyAny>> = args.iter().enumerate().map(|(i, a)| {
        let slot = &sig.slots[i];
        match a {
            TraceArg::Array { .. } => {
                let arr = JitTracerArray::from_flat_range(
                    &shared, slot.offset as u32, slot.size);
                Py::new(py, arr).unwrap().into_any()
            }
            TraceArg::Scalar { .. } => {
                let node = shared.add(Node::Input(slot.offset as u32));
                let tr = JitTracer::new(&shared, node);
                Py::new(py, tr).unwrap().into_any()
            }
        }
    }).collect();

    // Truncate to the function's actual arity: a Python callable expecting
    // fewer parameters than the spec would raise TypeError.  Introspect via
    // `inspect.signature`; on failure fall back to the full arg list.
    let n_params: usize = py.import("inspect")
        .ok()
        .and_then(|m| m.call_method1("signature", (func,)).ok())
        .and_then(|s| s.getattr("parameters").ok())
        .and_then(|p| p.call_method0("__len__").ok())
        .and_then(|l| l.extract().ok())
        .unwrap_or(py_args.len());
    let take = n_params.min(py_args.len());
    let py_args_tuple = PyTuple::new(py, &py_args[..take])?;

    // Monkeypatch numpy's array constructors for the duration of the call so the
    // imperative `dx = np.zeros(n); dx[i] = ...` idiom traces. These carry no
    // array argument, so __array_function__ cannot intercept them. Restored in
    // every path before returning (success or error). Only the OUTERMOST trace
    // patches (see `TRACE_DEPTH`): a nested trace would capture the outer
    // patch as its "original".
    let outermost = TRACE_DEPTH.with(|d| { let v = d.get(); d.set(v + 1); v == 0 });
    let np = if outermost { py.import("numpy").ok() } else { None };
    let mut saved: Vec<(&'static str, Py<PyAny>)> = Vec::new();
    if let Some(np) = np.as_ref() {
        for (name, fill, is_full) in [
            ("zeros", 0.0, false), ("empty", 0.0, false),
            ("ones", 1.0, false), ("full", 0.0, true),
        ] {
            if let Ok(orig) = np.getattr(name) {
                let ctor = ArrayCtor {
                    graph: shared.clone(),
                    fill,
                    is_full,
                    orig: orig.clone().unbind(),
                };
                if let Ok(ctor) = Py::new(py, ctor) {
                    if np.setattr(name, ctor).is_ok() {
                        saved.push((name, orig.unbind()));
                    }
                }
            }
        }
        // Constant factories (`np.arange(3.0)[0] = expr` idioms): patched to
        // return constant-node tracer arrays; unsupported signatures delegate
        // to the saved original (see `ConstFactoryCtor`).
        for name in ["arange", "linspace", "eye", "diag"] {
            if let Ok(orig) = np.getattr(name) {
                let ctor = ConstFactoryCtor {
                    graph: shared.clone(),
                    kind: name,
                    orig: orig.clone().unbind(),
                };
                if let Ok(ctor) = Py::new(py, ctor) {
                    if np.setattr(name, ctor).is_ok() {
                        saved.push((name, orig.unbind()));
                    }
                }
            }
        }
    }
    let call_result = func.call1(py_args_tuple);
    if let Some(np) = np.as_ref() {
        for (name, orig) in &saved {
            let _ = np.setattr(*name, orig.bind(py));
        }
    }
    TRACE_DEPTH.with(|d| d.set(d.get() - 1));
    let result = call_result?;

    // Extract outputs *before* locking the shared graph: the Python result
    // may still carry lazy input references whose iteration internally calls
    // SharedGraph::add, which would deadlock against an outer lock.
    let outputs = extract_outputs(py, &result, &shared)?;
    if outputs.is_empty() { return Ok(None); }

    let mut graph = {
        let mut g = shared.0.borrow_mut();
        g.outputs = outputs;
        g.clone()
    };

    crate::ssa::optimize::optimize(&mut graph);
    Ok(Some(graph))
}

/// Fixed op-graph for an identity output `y = x` over an `n`-element state — the
/// `alg` region of an ODE block (output equals the integrated state).
pub(crate) fn state_identity_graph(n: usize) -> crate::ssa::graph::Graph {
    use crate::ssa::graph::{Graph, InputSignature};
    let mut g = Graph::new(InputSignature::from_named_sizes([("x", n)]));
    let outs: Vec<u32> = (0..n as u32).map(|i| g.input(i)).collect();
    g.outputs = outs;
    g
}

/// Python-exposed: trace + optimize → ODE block with analytical Jacobian.
#[pyfunction]
#[pyo3(signature = (func, initial_value=None))]
pub fn _trace_ode(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
    initial_value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<crate::pybindings::py::PyBlock>> {
    use crate::pybindings::py::lazy::{LazyTraced, SigArg};
    use crate::pybindings::py::PyBlock;
    let iv = crate::pybindings::py::extract_initial_value(initial_value)?;
    let n_x = iv.len();

    // Classify probe: does the callback trace at all? A shape-mismatch
    // error (ValueError from numpy ops like `betas @ u`) is fine — the
    // LazyTraced wrapper will retrace with the correct `u.len()` at runtime.
    // A TypeError or other fatal error (e.g. `if` on a tracer) propagates
    // so that `_trace_or_none` on the Python side falls back to the
    // non-JIT factory, matching the pre-lazy behavior. On success the probe
    // graph SEEDS the LazyTraced cache, so a block whose resolved width
    // matches the probe (the SISO case) never traces twice.
    let probe = trace_with_signature(py, func, &[
        TraceArg::Array { name: "x", size: n_x },
        TraceArg::Array { name: "u", size: 1 },
        TraceArg::Scalar { name: "t" },
    ]);
    let seed = match probe {
        Ok(Some(g)) => Some(g),
        Ok(None) => return Ok(None),
        Err(e) => {
            // ValueError and IndexError usually signal a shape mismatch
            // from operations like `betas @ u` or `u[i]` with unknown
            // runtime size. The LazyTraced cache will retrace with the
            // real shape. TypeErrors (bare `if`, unsupported ops) and
            // other exceptions are structural and must propagate so that
            // the outer `_trace_or_none` falls back to the Python callback.
            let is_shape_error = e.is_instance_of::<pyo3::exceptions::PyValueError>(py)
                || e.is_instance_of::<pyo3::exceptions::PyIndexError>(py);
            if !is_shape_error {
                return Err(e);
            }
            None
        }
    };

    let callable = func.clone().unbind();
    let traced = LazyTraced::new(
        callable,
        vec![SigArg::Array("x"), SigArg::Array("u"), SigArg::Scalar("t")],
        Some("x"),
    );
    if let Some(g) = seed { traced.seed(g); }

    let mut blk = crate::blocks::block::Block::default_block();
    blk.type_name = "ODE";
    blk.role = crate::blocks::block::BlockRole {
        is_dyn: true, is_src: false, is_rec: false,
    };
    blk.initial_value = Some(iv.to_vec());
    blk.engine = Some(crate::solvers::solver::Solver::with_defaults(&iv));
    blk.len_fn = Some(Box::new(|_| 0));

    let traced_ir = traced.clone();
    let traced_dyn = traced.clone();
    blk.f_dyn = Some(Box::new(move |x, u, t, out| {
        traced_dyn.call_into(&[x, u, &[t]], out);
    }));
    blk.f_alg = Some(Box::new(|x, _u, _t, out| out.extend_from_slice(x)));

    let traced_jac = traced;
    blk.jac_dyn = Some(Box::new(move |x, u, t, out| {
        traced_jac.call_jacobian_into(&[x, u, &[t]], out);
    }));

    // Retain the op atomization for the IR / static compile: alg is the identity
    // `y = x`, dyn is the traced derivative `f(x, u, t)` (shape-lazy on `u`,
    // served from the SAME trace cache as the runtime tape).
    let dyn_lazy = traced_ir.op_graph(move |w| vec![n_x, w, 1]);
    blk.op_type_name = Some("ODE");
    blk.alg_op = Some(crate::blocks::operator::Operator::graph_only(
        crate::blocks::blockops::RegionGraph::Fixed(state_identity_graph(n_x)),
    ));
    blk.dyn_op = Some(crate::blocks::operator::Operator::graph_only(
        crate::blocks::blockops::RegionGraph::Lazy(dyn_lazy),
    ));

    Ok(Some(PyBlock::wrap(std::rc::Rc::new(crate::utils::fastcell::FastCell::new(blk)))))
}

/// Python-exposed: Function block with lazy-rejit trace over `u`.
#[pyfunction]
pub fn _trace_function_block(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
) -> PyResult<Option<crate::pybindings::py::PyBlock>> {
    use crate::pybindings::py::lazy::{LazyTraced, SigArg};
    use crate::pybindings::py::PyBlock;

    // Classify probe (see _trace_ode above); the probe graph seeds the cache.
    let probe = trace_with_signature(py, func, &[
        TraceArg::Array { name: "u", size: 1 },
    ]);
    let seed = match probe {
        Ok(Some(g)) => Some(g),
        Ok(None) => return Ok(None),
        Err(e) => {
            // ValueError and IndexError usually signal a shape mismatch
            // from operations like `betas @ u` or `u[i]` with unknown
            // runtime size. The LazyTraced cache will retrace with the
            // real shape. TypeErrors (bare `if`, unsupported ops) and
            // other exceptions are structural and must propagate so that
            // the outer `_trace_or_none` falls back to the Python callback.
            let is_shape_error = e.is_instance_of::<pyo3::exceptions::PyValueError>(py)
                || e.is_instance_of::<pyo3::exceptions::PyIndexError>(py);
            if !is_shape_error {
                return Err(e);
            }
            None
        }
    };

    let callable = func.clone().unbind();
    let traced = LazyTraced::new(callable, vec![SigArg::Array("u")], None);
    if let Some(g) = seed { traced.seed(g); }
    let mut b = crate::blocks::block::Block::default_block();
    b.type_name = "Function";
    let t = traced.clone();
    b.f_alg = Some(Box::new(move |_x, u, _t, out| {
        t.call_into(&[u], out);
    }));
    // Retain the traced graph as the block's op atomization so the IR / static
    // compile can fuse it (shape-lazy on `u`, served from the same trace cache
    // as the runtime tape).
    b.set_alg_lazy("Function", traced.op_graph(|w| vec![w]));
    Ok(Some(PyBlock::wrap(std::rc::Rc::new(crate::utils::fastcell::FastCell::new(b)))))
}

/// Python-exposed: Source block (t-only) — eager trace is safe because
/// the signature has no arrays that can change shape at runtime.
#[pyfunction]
pub fn _trace_source(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
) -> PyResult<Option<crate::pybindings::py::PyBlock>> {
    use crate::pybindings::py::PyBlock;
    let graph = match trace_with_signature(py, func, &[
        TraceArg::Scalar { name: "t" },
    ])? {
        Some(g) => g, None => return Ok(None),
    };
    let mut b = crate::blocks::block::Block::new(
        Some(std::collections::HashMap::new()),
        Some(std::collections::HashMap::from([("out".to_string(), 0)])),
    );
    b.type_name = "Source";
    b.role = crate::blocks::block::BlockRole {
        is_dyn: false, is_src: true, is_rec: false,
    };
    b.len_fn = Some(Box::new(|_| 0));
    // Derive the runtime closure from the traced graph, then retain the graph
    // itself as the block's op atomization (single source of truth for the IR).
    b.f_alg = Some(crate::blocks::blockops::block_fn_from_graph(&graph));
    b.set_alg("Source", graph);
    Ok(Some(PyBlock::wrap(std::rc::Rc::new(crate::utils::fastcell::FastCell::new(b)))))
}

/// Extract output NodeIds from a Python return value (Tracer, list, tuple, np.array).
#[allow(clippy::only_used_in_recursion)]
fn extract_outputs(py: Python<'_>, result: &Bound<'_, PyAny>, graph: &SharedGraph) -> PyResult<Vec<NodeId>> {
    // Single Tracer
    if let Ok(t) = result.extract::<JitTracer>() {
        return Ok(vec![t.node_id]);
    }

    // List or tuple of Tracers/floats
    if let Ok(items) = result.extract::<Vec<Bound<'_, PyAny>>>() {
        let mut ids = Vec::with_capacity(items.len());
        for item in &items {
            if let Ok(t) = item.extract::<JitTracer>() {
                ids.push(t.node_id);
            } else if let Ok(v) = item.extract::<f64>() {
                ids.push(graph.constant(v));
            } else {
                return Err(PyTypeError::new_err(
                    format!("Expected JitTracer or float in output, got {}", item.get_type().name()?)
                ));
            }
        }
        return Ok(ids);
    }

    // Try np.array unwrap
    if let Ok(list) = result.call_method0("tolist") {
        return extract_outputs(py, &list, graph);
    }

    // Single float
    if let Ok(v) = result.extract::<f64>() {
        return Ok(vec![graph.constant(v)]);
    }

    Ok(vec![])
}

/// Register tracer classes and functions in the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<JitTracer>()?;
    m.add_class::<JitTracerArray>()?;
    m.add_function(pyo3::wrap_pyfunction!(jit_where, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(jit_clip, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(jit_random_uniform, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(jit_random_normal, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(_trace_ode, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(_trace_function_block, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(_trace_source, m)?)?;
    Ok(())
}
