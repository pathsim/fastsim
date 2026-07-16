// JitTracerArray: the symbolic N-D array tracer and its full numpy-protocol
// surface (__array_ufunc__ / __array_function__, indexing, reductions, view
// ops). Split out of the tracer god-file; the scalar tracer, shared graph,
// and trace driver live in the parent module.

// Operator overloads `.into()` already-`Py<PyAny>` values for uniformity.
#![allow(clippy::useless_conversion)]
// `__array_priority__` is a numpy dunder constant; its name is fixed by numpy.
#![allow(non_upper_case_globals)]

use pyo3::prelude::*;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::types::{PyDict, PySlice, PyTuple};

use crate::ssa::graph::*;
use super::super::ndshape::{bcast_src_flat, shape_with_kept_axis, strides_row_major};
use super::super::ufunc_table::{binop_for_ufunc, cmpop_for_ufunc, unary_op_for_ufunc};
use super::{ArrayBinArg, JitTracer, SharedGraph, TracerOrFloat};

/// Where a `JitTracerArray`'s element nodes come from. Replaces the former
/// two-`Option` (`base`/`node_ids`) encoding so the "exactly one is set"
/// invariant is a type guarantee — no more "has no data source" panics.
#[derive(Clone)]
enum Source {
    /// A contiguous range of flat input slots: element `i` is `Input(base + i)`.
    Input { base: u32 },
    /// Pre-computed graph nodes (e.g. a matmul result or a mutated array).
    Nodes(Vec<NodeId>),
}

/// A symbolic array (an input slot, or a computed array like a matmul result).
#[pyclass(name = "JitTracerArray", from_py_object, unsendable)]
#[derive(Clone)]
pub struct JitTracerArray {
    graph: SharedGraph,
    /// Source of this array's element nodes (input slots or computed nodes).
    source: Source,
    pub(super) size: usize,
    /// Python-surface tensor shape.  Product(shape) == size.  Default is
    /// `[size]` (1-D).  The underlying graph is scalar-SSA and always flat
    /// row-major; shape is purely a view for reshape / transpose / N-D
    /// metadata ops on the Python side.
    shape: Vec<usize>,
}

impl JitTracerArray {
    /// Create a tracer array backed by a contiguous range of flat input indices.
    pub(super) fn from_flat_range(graph: &SharedGraph, base: u32, size: usize) -> Self {
        Self { graph: graph.clone(), source: Source::Input { base }, size, shape: vec![size] }
    }

    /// Create from pre-computed node IDs (for matmul results etc.)
    pub(super) fn from_nodes(graph: &SharedGraph, node_ids: Vec<NodeId>) -> Self {
        let size = node_ids.len();
        Self { graph: graph.clone(), source: Source::Nodes(node_ids), size, shape: vec![size] }
    }

    /// Create from pre-computed node IDs with an explicit N-D shape.  The
    /// product of `shape` must equal `node_ids.len()`.  Used by view ops
    /// (reshape, transpose, squeeze, unsqueeze) and by N-D constructions.
    pub(super) fn from_nodes_shape(graph: &SharedGraph, node_ids: Vec<NodeId>, shape: Vec<usize>) -> Self {
        let size = node_ids.len();
        debug_assert_eq!(size, shape.iter().product::<usize>(),
            "shape product {:?} != node count {}", shape, size);
        Self { graph: graph.clone(), source: Source::Nodes(node_ids), size, shape }
    }

    /// Materialize node IDs as a `Vec<NodeId>` of length `size`, regardless of
    /// whether the array is a `from_flat_range` (lazy) or `from_nodes` instance.
    /// Used by view ops that need to permute / slice the flat layout.
    fn materialize(&self) -> Vec<NodeId> {
        match &self.source {
            Source::Nodes(ids) => ids.clone(),
            Source::Input { base } => self.graph.add_input_range(*base, self.size),
        }
    }

    /// Parse a Python indexer into a normalized `AxisIdx` list. Supports
    /// `int` (negative ok), `slice` (any step sign), constant integer
    /// lists/arrays (fancy indexing — a pure node permutation at trace time),
    /// `Ellipsis`, `None`/newaxis, and tuples thereof. Missing trailing axes
    /// default to `:`. Boolean masks are rejected: a mask's output SHAPE is
    /// data-dependent, which no trace can represent.
    fn parse_indexers(&self, idx: &Bound<'_, PyAny>) -> PyResult<Vec<AxisIdx>> {
        let raw: Vec<Bound<'_, PyAny>> = if let Ok(tup) = idx.cast::<PyTuple>() {
            tup.iter().collect()
        } else {
            vec![idx.clone()]
        };
        let ndim = self.shape.len();

        // First pass: locate the (single) Ellipsis and count axis-consuming
        // entries, so the Ellipsis can expand to the right number of `:`.
        let is_ellipsis = |item: &Bound<'_, PyAny>| {
            item.get_type().name().map(|n| n.to_string()).unwrap_or_default() == "ellipsis"
        };
        let mut ellipsis_at: Option<usize> = None;
        let mut consuming = 0usize;
        for (i, item) in raw.iter().enumerate() {
            if is_ellipsis(item) {
                if ellipsis_at.is_some() {
                    return Err(PyIndexError::new_err(
                        "an index can only have a single ellipsis ('...')"));
                }
                ellipsis_at = Some(i);
            } else if !item.is_none() {
                consuming += 1;
            }
        }
        if consuming > ndim {
            return Err(PyIndexError::new_err(format!(
                "too many indices: array is {}-D but got {}", ndim, consuming)));
        }

        let mut indexers: Vec<AxisIdx> = Vec::new();
        let mut axis = 0usize; // input axis consumed so far
        for (i, item) in raw.iter().enumerate() {
            if Some(i) == ellipsis_at {
                // Expand to full slices over the axes no explicit entry covers.
                for _ in 0..(ndim - consuming) {
                    indexers.push(AxisIdx::Slice {
                        start: 0, len: self.shape[axis], step: 1,
                    });
                    axis += 1;
                }
                continue;
            }
            if item.is_none() {
                indexers.push(AxisIdx::NewAxis);
                continue;
            }
            let dim = self.shape[axis];
            if let Ok(i) = item.extract::<isize>() {
                let ni = if i < 0 { dim as isize + i } else { i };
                if ni < 0 || ni >= dim as isize {
                    return Err(PyIndexError::new_err(format!(
                        "index {} out of range for axis {} with size {}", i, axis, dim
                    )));
                }
                indexers.push(AxisIdx::Int(ni as usize));
            } else if let Ok(s) = item.cast::<PySlice>() {
                let info = s.indices(dim as isize)?;
                indexers.push(AxisIdx::Slice {
                    start: info.start,
                    len: info.slicelength,
                    step: info.step,
                });
            } else if item.extract::<Vec<bool>>().is_ok()
                || item.getattr("dtype").is_ok_and(|d| {
                    d.getattr("kind").and_then(|k| k.extract::<String>()).is_ok_and(|k| k == "b")
                })
            {
                return Err(PyTypeError::new_err(
                    "boolean mask indexing is data-dependent (output shape depends on \
                     runtime values) and cannot be traced; rewrite with np.where"));
            } else if let Ok(list) = item.extract::<Vec<isize>>() {
                // Constant integer fancy index: a node permutation at trace time.
                let mut picks = Vec::with_capacity(list.len());
                for &i in &list {
                    let ni = if i < 0 { dim as isize + i } else { i };
                    if ni < 0 || ni >= dim as isize {
                        return Err(PyIndexError::new_err(format!(
                            "index {} out of range for axis {} with size {}", i, axis, dim
                        )));
                    }
                    picks.push(ni as usize);
                }
                indexers.push(AxisIdx::Fancy(picks));
            } else {
                return Err(PyTypeError::new_err(format!(
                    "unsupported index type on axis {} (int / slice / Ellipsis / None / \
                     constant int list supported)", axis
                )));
            }
            axis += 1;
        }
        // Pad with full slices on remaining axes.
        for axis in axis..ndim {
            indexers.push(AxisIdx::Slice {
                start: 0,
                len: self.shape[axis],
                step: 1,
            });
        }
        Ok(indexers)
    }

    /// Output shape produced by parsed indexers (`Int` drops the axis,
    /// `NewAxis` inserts a 1). Empty means a scalar result.
    fn indexed_shape(indexers: &[AxisIdx]) -> Vec<usize> {
        let mut out_shape = Vec::new();
        for ix in indexers {
            match ix {
                AxisIdx::Slice { len, .. } => out_shape.push(*len),
                AxisIdx::Fancy(v) => out_shape.push(v.len()),
                AxisIdx::NewAxis => out_shape.push(1),
                AxisIdx::Int(_) => {}
            }
        }
        out_shape
    }

    /// Apply parsed indexers to produce (output shape, output node_ids).
    /// Scalar result (all `Int` on all axes) yields empty `out_shape`.
    fn apply_indexers(&self, indexers: &[AxisIdx]) -> (Vec<usize>, Vec<NodeId>) {
        let self_nodes = self.materialize();
        let nodes = self.indexed_positions(indexers)
            .into_iter()
            .map(|p| self_nodes[p])
            .collect();
        (Self::indexed_shape(indexers), nodes)
    }

    /// Flat positions selected by `indexers`, in row-major output order.
    /// The single index walker behind both `__getitem__` and `__setitem__`.
    fn indexed_positions(&self, indexers: &[AxisIdx]) -> Vec<usize> {
        let self_strides = strides_row_major(&self.shape);
        let out_shape = Self::indexed_shape(indexers);
        let out_size: usize = out_shape.iter().product::<usize>().max(1);
        let n_out = out_shape.len();
        let mut positions = Vec::with_capacity(out_size);
        let mut out_idx = vec![0usize; n_out];
        for _ in 0..out_size {
            let mut flat = 0isize;
            let mut od = 0usize;
            let mut axis = 0usize;
            for ix in indexers {
                match ix {
                    AxisIdx::Int(i) => {
                        flat += (i * self_strides[axis]) as isize;
                        axis += 1;
                    }
                    AxisIdx::Slice { start, step, .. } => {
                        // Signed walk handles negative steps (x[::-1]).
                        flat += (start + out_idx[od] as isize * step)
                            * self_strides[axis] as isize;
                        od += 1;
                        axis += 1;
                    }
                    AxisIdx::Fancy(picks) => {
                        flat += (picks[out_idx[od]] * self_strides[axis]) as isize;
                        od += 1;
                        axis += 1;
                    }
                    AxisIdx::NewAxis => {
                        od += 1; // size-1 output axis, consumes no input axis
                    }
                }
            }
            positions.push(flat as usize);
            for d in (0..n_out).rev() {
                out_idx[d] += 1;
                if out_idx[d] < out_shape[d] { break; }
                out_idx[d] = 0;
            }
        }
        positions
    }

    /// Coerce a Python RHS (scalar tracer/float, or array/list) into exactly
    /// `count` node ids for assignment into `count` positions. Size-1 broadcasts.
    fn coerce_value_nodes(&self, value: &Bound<'_, PyAny>, count: usize) -> PyResult<Vec<NodeId>> {
        if let Ok(t) = value.extract::<JitTracer>() {
            return Ok(vec![t.node_id; count]);
        }
        if let Ok(v) = value.extract::<f64>() {
            return Ok(vec![self.graph.constant(v); count]);
        }
        if let Ok(arr) = value.extract::<JitTracerArray>() {
            let nodes = arr.materialize();
            if nodes.len() == count { return Ok(nodes); }
            if nodes.len() == 1 { return Ok(vec![nodes[0]; count]); }
            return Err(PyValueError::new_err(format!(
                "could not assign array of size {} into {} position(s)", nodes.len(), count)));
        }
        if let Ok(vals) = value.extract::<Vec<f64>>() {
            if vals.len() == count {
                return Ok(vals.iter().map(|&v| self.graph.constant(v)).collect());
            }
            if vals.len() == 1 {
                return Ok(vec![self.graph.constant(vals[0]); count]);
            }
            return Err(PyValueError::new_err(format!(
                "could not assign array of size {} into {} position(s)", vals.len(), count)));
        }
        Err(PyTypeError::new_err("unsupported value type in JIT array assignment"))
    }

    /// Get the NodeId for element i.
    pub(super) fn get_node(&self, i: usize) -> NodeId {
        match &self.source {
            Source::Nodes(ids) => ids[i],
            Source::Input { base } => self.graph.add(Node::Input(base + i as u32)),
        }
    }

    /// Element-wise binary op with N-D numpy broadcasting.
    /// `swap=true` flips operand order (needed for r-methods).
    fn array_binop(
        &self,
        py: Python<'_>,
        op: BinOp,
        other: ArrayBinArg,
        swap: bool,
    ) -> PyResult<Py<PyAny>> {
        self.array_elementwise(py, other, swap, |g, a, b| g.binary(op, a, b))
    }

    /// Element-wise unary op: apply `op` to every element, preserving shape.
    /// Emits all N nodes under one graph borrow.
    fn array_unary(&self, py: Python<'_>, op: UnaryOp) -> PyResult<Py<PyAny>> {
        let ids = self.materialize();
        let nodes: Vec<NodeId> = self.graph.with(|g| {
            ids.iter().map(|&x| g.unary(op, x)).collect()
        });
        Ok(Py::new(py, JitTracerArray::from_nodes_shape(&self.graph, nodes, self.shape.clone()))?.into_any().into())
    }

    /// Element-wise comparison (returns a JitTracerArray of 0.0/1.0 nodes).
    fn array_cmpop(
        &self,
        py: Python<'_>,
        op: CmpOp,
        other: ArrayBinArg,
        swap: bool,
    ) -> PyResult<Py<PyAny>> {
        self.array_elementwise(py, other, swap, |g, a, b| g.cmp(op, a, b))
    }

    /// Core N-D broadcasting walker.  Produces one output element per broadcast
    /// position by calling `emit(graph, self_node, other_node)` (or swapped).
    /// Scalar operands (Float, Tracer) take a fast path with no index
    /// expansion. All element nodes are emitted under ONE graph borrow.
    fn array_elementwise(
        &self,
        py: Python<'_>,
        other: ArrayBinArg,
        swap: bool,
        emit: impl Fn(&mut Graph, NodeId, NodeId) -> NodeId,
    ) -> PyResult<Py<PyAny>> {
        // Scalar-fast-path: output shape = self.shape, no index machinery.
        let scalar_node: Option<NodeId> = match &other {
            ArrayBinArg::Float(v) => Some(self.graph.constant(*v)),
            ArrayBinArg::Tracer(t) => Some(t.node_id),
            _ => None,
        };
        if let Some(c) = scalar_node {
            let ids = self.materialize();
            let nodes: Vec<NodeId> = self.graph.with(|g| {
                ids.iter().map(|&x| {
                    let (a, b) = if swap { (c, x) } else { (x, c) };
                    emit(g, a, b)
                }).collect()
            });
            return Ok(Py::new(py, JitTracerArray::from_nodes_shape(&self.graph, nodes, self.shape.clone()))?.into_any().into());
        }

        // N-D broadcasting path.  Materialize the other operand's shape and
        // a flat node_id lookup.
        let (other_shape, other_nodes): (Vec<usize>, Vec<NodeId>) = match other {
            ArrayBinArg::TracerArray(ta) => (ta.shape.clone(), ta.materialize()),
            ArrayBinArg::NdArray(vs) => {
                let shape = vec![vs.len()];
                let nodes: Vec<NodeId> = self.graph.with(|g| {
                    vs.iter().map(|v| g.constant(*v)).collect()
                });
                (shape, nodes)
            }
            _ => unreachable!("scalar case handled above"),
        };

        let out_shape = broadcast_nd(&self.shape, &other_shape)?;
        let out_size: usize = out_shape.iter().product();
        let self_strides = strides_row_major(&self.shape);
        let other_strides = strides_row_major(&other_shape);
        let self_nodes = self.materialize();
        let n_out = out_shape.len();

        let nodes: Vec<NodeId> = self.graph.with(|g| {
            let mut nodes = Vec::with_capacity(out_size);
            let mut idx = vec![0usize; n_out];
            for _ in 0..out_size {
                let self_flat = bcast_src_flat(&idx, &self.shape, &self_strides);
                let other_flat = bcast_src_flat(&idx, &other_shape, &other_strides);
                let x = self_nodes[self_flat];
                let y = other_nodes[other_flat];
                let (a, b) = if swap { (y, x) } else { (x, y) };
                nodes.push(emit(g, a, b));
                // row-major increment
                for d in (0..n_out).rev() {
                    idx[d] += 1;
                    if idx[d] < out_shape[d] { break; }
                    idx[d] = 0;
                }
            }
            nodes
        });
        Ok(Py::new(py, JitTracerArray::from_nodes_shape(&self.graph, nodes, out_shape))?.into_any().into())
    }
}

impl JitTracerArray {
    /// Elementwise lowering for a named binary ufunc: arithmetic (`BinOp`),
    /// comparison (`CmpOp`), or a binary composite (copysign / logaddexp /
    /// heaviside). Also the entry point for the scalar tracer's mixed-operand
    /// promotion (`np.minimum(x[0], x)` lands on the scalar tracer first);
    /// `swap` flips operand order for scalar-first calls.
    pub(super) fn ufunc_elementwise(
        &self,
        py: Python<'_>,
        name: &str,
        bin_op: Option<BinOp>,
        cmp_op: Option<CmpOp>,
        other: ArrayBinArg,
        swap: bool,
    ) -> PyResult<Py<PyAny>> {
        if let Some(op) = bin_op {
            return self.array_binop(py, op, other, swap);
        }
        if let Some(op) = cmp_op {
            return self.array_cmpop(py, op, other, swap);
        }
        let name = name.to_string();
        self.array_elementwise(py, other, swap, move |g, a, b| {
            super::binary_composite_g(g, &name, a, b)
                .expect("caller matched the composite name list")
        })
    }

    /// Reduce to a scalar (axis=None) or along one axis, with numpy
    /// `keepdims` semantics. Shared by the `np.sum(x, ...)` function arm and
    /// the `x.sum(...)` method form.
    fn reduce_dispatch(
        &self,
        py: Python<'_>,
        op: BinOp,
        identity: f64,
        axis: Option<isize>,
        keepdims: bool,
    ) -> PyResult<Py<PyAny>> {
        match axis {
            None => {
                let scalar = reduce_array(self, op, identity);
                if keepdims {
                    // All-1s shape of same rank as input.
                    let shape = vec![1usize; self.shape.len().max(1)];
                    Ok(Py::new(py, JitTracerArray::from_nodes_shape(
                        &self.graph, vec![scalar.node_id], shape))?.into_any().into())
                } else {
                    Ok(Py::new(py, scalar)?.into_any().into())
                }
            }
            Some(a) => {
                let (mut out_shape, out_nodes) = reduce_along_axis(self, op, identity, a)?;
                if keepdims {
                    out_shape = shape_with_kept_axis(&out_shape, self.shape.len(), a);
                }
                if out_shape.is_empty() {
                    Ok(Py::new(py, JitTracer::new(&self.graph, out_nodes[0]))?.into_any().into())
                } else {
                    Ok(Py::new(py, JitTracerArray::from_nodes_shape(
                        &self.graph, out_nodes, out_shape))?.into_any().into())
                }
            }
        }
    }

    /// Arithmetic mean (axis-aware, `keepdims`). Shared by the `np.mean`
    /// function arm and the `x.mean(...)` method form.
    fn mean_dispatch(
        &self,
        py: Python<'_>,
        axis: Option<isize>,
        keepdims: bool,
    ) -> PyResult<Py<PyAny>> {
        match axis {
            None => {
                let sum = reduce_array(self, BinOp::Add, 0.0);
                let n = self.graph.constant(self.size as f64);
                let mean = self.graph.binary(BinOp::Div, sum.node_id, n);
                if keepdims {
                    let shape = vec![1usize; self.shape.len().max(1)];
                    return Ok(Py::new(py, JitTracerArray::from_nodes_shape(
                        &self.graph, vec![mean], shape))?.into_any().into());
                }
                Ok(Py::new(py, JitTracer::new(&self.graph, mean))?.into_any().into())
            }
            Some(a) => {
                let (mut out_shape, sum_nodes) = reduce_along_axis(self, BinOp::Add, 0.0, a)?;
                let ax = if a < 0 { self.shape.len() as isize + a } else { a } as usize;
                let n_node = self.graph.constant(self.shape[ax] as f64);
                let mean_nodes: Vec<NodeId> = sum_nodes.iter()
                    .map(|&s| self.graph.binary(BinOp::Div, s, n_node))
                    .collect();
                if keepdims {
                    out_shape = shape_with_kept_axis(&out_shape, self.shape.len(), a);
                }
                if out_shape.is_empty() {
                    return Ok(Py::new(py, JitTracer::new(&self.graph, mean_nodes[0]))?.into_any().into());
                }
                Ok(Py::new(py, JitTracerArray::from_nodes_shape(
                    &self.graph, mean_nodes, out_shape))?.into_any().into())
            }
        }
    }
}

/// Shared `np.where(cond, a, b)` lowering for both tracer types: traced or
/// constant condition (scalar or vector), scalar or array branches, size-1
/// broadcasting. A constant condition picks its branch at trace time (no
/// `Select` emitted — numpy would resolve it eagerly too); a traced condition
/// lowers to per-element `Select` nodes. The condition is `args[0]`, NOT the
/// dispatching object: numpy lands here whenever ANY argument is a tracer.
pub(super) fn np_where_dispatch(
    py: Python<'_>,
    graph: &SharedGraph,
    args: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    if args.len() < 3 {
        return Ok(py.NotImplemented().into());
    }
    /// One `np.where` operand with its N-D shape: traced nodes or eager
    /// constants, both flat row-major. A scalar has shape `[]`.
    enum Operand {
        Nodes(Vec<usize>, Vec<NodeId>),
        Consts(Vec<usize>, Vec<f64>),
    }
    impl Operand {
        fn shape(&self) -> &[usize] {
            match self { Operand::Nodes(s, _) | Operand::Consts(s, _) => s }
        }
    }
    let classify = |arg: &Bound<'_, PyAny>| -> Option<Operand> {
        if let Ok(t) = arg.extract::<JitTracer>() {
            Some(Operand::Nodes(vec![], vec![t.node_id]))
        } else if let Ok(ta) = arg.extract::<JitTracerArray>() {
            Some(Operand::Nodes(ta.shape.clone(), ta.materialize()))
        } else if let Ok(v) = arg.extract::<f64>() {
            Some(Operand::Consts(vec![], vec![v]))
        } else if let Ok(vs) = arg.extract::<Vec<f64>>() {
            Some(Operand::Consts(vec![vs.len()], vs))
        } else {
            None
        }
    };
    let (Some(cond), Some(then_op), Some(else_op)) = (
        classify(&args.get_item(0)?),
        classify(&args.get_item(1)?),
        classify(&args.get_item(2)?),
    ) else {
        return Ok(py.NotImplemented().into());
    };

    // Full numpy broadcasting across all three operands; the result carries
    // the broadcast shape (the former flat-only path returned 1-D and lost
    // 2-D shapes).
    let out_shape = broadcast_nd(&broadcast_nd(cond.shape(), then_op.shape())?, else_op.shape())?;
    let out_size: usize = out_shape.iter().product::<usize>().max(1);
    let scalar_result = out_shape.is_empty();
    let n_out = out_shape.len();

    let node_at = |op: &Operand, idx: &[usize]| -> NodeId {
        match op {
            Operand::Nodes(shape, nodes) => {
                let flat = bcast_src_flat(idx, shape, &strides_row_major(shape));
                nodes[flat]
            }
            Operand::Consts(shape, vals) => {
                let flat = bcast_src_flat(idx, shape, &strides_row_major(shape));
                graph.constant(vals[flat])
            }
        }
    };

    let mut nodes = Vec::with_capacity(out_size);
    let mut idx = vec![0usize; n_out];
    for _ in 0..out_size {
        let n = match &cond {
            Operand::Consts(shape, vals) => {
                // A constant condition picks its branch at trace time (numpy
                // would resolve it eagerly too).
                let flat = bcast_src_flat(&idx, shape, &strides_row_major(shape));
                if vals[flat] != 0.0 { node_at(&then_op, &idx) } else { node_at(&else_op, &idx) }
            }
            Operand::Nodes(shape, cn) => {
                let flat = bcast_src_flat(&idx, shape, &strides_row_major(shape));
                let t = node_at(&then_op, &idx);
                let e = node_at(&else_op, &idx);
                graph.select(cn[flat], t, e)
            }
        };
        nodes.push(n);
        for d in (0..n_out).rev() {
            idx[d] += 1;
            if idx[d] < out_shape[d] { break; }
            idx[d] = 0;
        }
    }
    if scalar_result {
        Ok(Py::new(py, JitTracer::new(graph, nodes[0]))?.into_any().into())
    } else {
        Ok(Py::new(py, JitTracerArray::from_nodes_shape(graph, nodes, out_shape))?.into_any().into())
    }
}

/// Per-axis indexer used by `JitTracerArray::__getitem__` / `__setitem__`.
/// `Slice` carries the resolved (start, len, step) triple from
/// `PySlice::indices` — `len` is the output element count along the axis and
/// `step` may be negative (`x[::-1]`). `Fancy` holds normalized constant
/// integer picks; `NewAxis` (`None`) inserts a size-1 output axis without
/// consuming an input axis.
enum AxisIdx {
    Int(usize),
    Slice { start: isize, len: usize, step: isize },
    Fancy(Vec<usize>),
    NewAxis,
}

/// Row-major stride vector for a shape.  `strides[i]` equals the product of
/// all trailing dimensions after axis `i`, so `flat_index = Σ idx[i] * strides[i]`.
/// An empty shape (scalar) returns an empty vector.
/// N-D numpy broadcasting rule.  Right-align the shapes, pad missing axes with
/// 1 on the left, and for each axis require equal sizes or one-side-is-1.
/// The result shape takes the maximum of each axis.  An empty shape (scalar)
/// broadcasts against anything.
fn broadcast_nd(a: &[usize], b: &[usize]) -> PyResult<Vec<usize>> {
    let n = a.len().max(b.len());
    let pa = n - a.len();
    let pb = n - b.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let ai = if i < pa { 1 } else { a[i - pa] };
        let bi = if i < pb { 1 } else { b[i - pb] };
        let size = if ai == bi { ai }
                   else if ai == 1 { bi }
                   else if bi == 1 { ai }
                   else {
                       return Err(pyo3::exceptions::PyValueError::new_err(
                           format!("shape mismatch (not broadcastable): {:?} vs {:?}", a, b)
                       ));
                   };
        out.push(size);
    }
    Ok(out)
}

/// N-D concatenation of (shape, flat row-major nodes) items along `axis`.
/// numpy semantics: all items share rank and every non-`axis` extent; the
/// output extent on `axis` is the sum. Shared by the `concatenate`,
/// `hstack` (axis 1) and `vstack` (axis 0 after 2-D promotion) arms.
fn concat_nd(
    items: &[(Vec<usize>, Vec<NodeId>)],
    axis: isize,
) -> PyResult<(Vec<usize>, Vec<NodeId>)> {
    let ref_shape = items[0].0.clone();
    let rank = ref_shape.len();
    if rank == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "np.concatenate: 0-D arrays cannot be concatenated"
        ));
    }
    let ax = if axis < 0 { rank as isize + axis } else { axis };
    if ax < 0 || ax >= rank as isize {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("np.concatenate: axis {} out of bounds for rank {}", axis, rank)
        ));
    }
    let ax = ax as usize;
    let mut out_shape = ref_shape.clone();
    out_shape[ax] = 0;
    for (s, _) in items {
        if s.len() != rank {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("np.concatenate: rank mismatch {} vs {}", rank, s.len())
            ));
        }
        for d in 0..rank {
            if d == ax { out_shape[ax] += s[d]; }
            else if s[d] != ref_shape[d] {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("np.concatenate: shape mismatch on axis {} ({} vs {})", d, ref_shape[d], s[d])
                ));
            }
        }
    }
    let out_size: usize = out_shape.iter().product();
    let out_strides = strides_row_major(&out_shape);
    let mut nodes = vec![0u32; out_size];
    let mut axis_offset = 0usize;
    for (src_shape, src_nodes) in items {
        let src_strides = strides_row_major(src_shape);
        let src_size: usize = src_shape.iter().product();
        let mut src_idx = vec![0usize; rank];
        for _ in 0..src_size {
            let mut src_flat = 0usize;
            let mut dst_flat = 0usize;
            for d in 0..rank {
                src_flat += src_idx[d] * src_strides[d];
                let out_pos = if d == ax { src_idx[d] + axis_offset } else { src_idx[d] };
                dst_flat += out_pos * out_strides[d];
            }
            nodes[dst_flat] = src_nodes[src_flat];
            for d in (0..rank).rev() {
                src_idx[d] += 1;
                if src_idx[d] < src_shape[d] { break; }
                src_idx[d] = 0;
            }
        }
        axis_offset += src_shape[ax];
    }
    Ok((out_shape, nodes))
}

/// Invert a dense constant matrix by Gauss--Jordan with partial pivoting.
/// `None` for a singular (or non-square) matrix. Trace-time only: the cost is
/// O(n^3) floats once, so `np.linalg.solve(A_const, b)` can lower to n fused
/// dot products over `b`'s nodes.
fn invert_matrix(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    if n == 0 || a.iter().any(|r| r.len() != n) { return None; }
    // Augmented [A | I], eliminated in place.
    let mut m: Vec<Vec<f64>> = (0..n).map(|i| {
        let mut row = a[i].clone();
        row.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
        row
    }).collect();
    for col in 0..n {
        // Partial pivot: largest |entry| at or below the diagonal.
        let pivot = (col..n).max_by(|&i, &j| {
            m[i][col].abs().partial_cmp(&m[j][col].abs()).unwrap()
        })?;
        if m[pivot][col].abs() < 1e-300 { return None; } // singular
        m.swap(col, pivot);
        let p = m[col][col];
        for v in m[col].iter_mut() { *v /= p; }
        for row in 0..n {
            if row == col { continue; }
            let f = m[row][col];
            if f == 0.0 { continue; }
            for k in 0..2 * n {
                m[row][k] -= f * m[col][k];
            }
        }
    }
    Some(m.into_iter().map(|row| row[n..].to_vec()).collect())
}

/// Lower `np.var`/`np.std` (population variance with numpy's ddof correction).
/// Shared by the `__array_function__` arm and the `x.var()`/`x.std()` method
/// forms. The squared-diff sum is ONE fused `Dot(d, d)` node.
fn var_std(
    py: Python<'_>, ta: &JitTracerArray, ddof: usize, is_std: bool,
) -> PyResult<Py<PyAny>> {
    if ddof >= ta.size {
        return Err(PyValueError::new_err(format!(
            "var/std: ddof {} >= size {}", ddof, ta.size)));
    }
    let n_node = ta.graph.constant(ta.size as f64);
    let denom = ta.graph.constant((ta.size - ddof) as f64);
    let sum = reduce_array(ta, BinOp::Add, 0.0).node_id;
    let mean = ta.graph.binary(BinOp::Div, sum, n_node);
    let ids = ta.materialize();
    let diffs: Vec<NodeId> = ta.graph.with(|g| {
        ids.iter().map(|&x| g.binary(BinOp::Sub, x, mean)).collect()
    });
    let ssd = node_dot(&ta.graph, diffs.clone(), diffs);
    let var = ta.graph.binary(BinOp::Div, ssd, denom);
    let result = if is_std { ta.graph.unary(UnaryOp::Sqrt, var) } else { var };
    Ok(Py::new(py, JitTracer::new(&ta.graph, result))?.into_any().into())
}

/// Shared `np.searchsorted(const_grid, v, side=...)` lowering for both tracer
/// types: the insertion index is a count, #\{grid[i] < v\} (side='left') or
/// #\{grid[i] <= v\} ('right'), lowered as a fused Reduce over per-breakpoint
/// Cmp nodes. The grid must be constant (like np.interp).
pub(super) fn np_searchsorted_dispatch(
    py: Python<'_>,
    graph: &SharedGraph,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    if args.len() < 2 || !kwargs_subset(kwargs, &["side"]) {
        return Ok(py.NotImplemented().into());
    }
    let Ok(grid) = args.get_item(0)?.extract::<Vec<f64>>() else {
        return Ok(py.NotImplemented().into());
    };
    let mut right = false;
    if args.len() >= 3 {
        match args.get_item(2)?.extract::<String>() {
            Ok(s) if s == "left" => {}
            Ok(s) if s == "right" => right = true,
            _ => return Ok(py.NotImplemented().into()),
        }
    }
    if let Some(k) = kwargs {
        if let Ok(Some(v)) = k.get_item("side") {
            match v.extract::<String>() {
                Ok(s) if s == "left" => {}
                Ok(s) if s == "right" => right = true,
                _ => return Ok(py.NotImplemented().into()),
            }
        }
    }
    let emit_one = |g: &mut Graph, v: NodeId| -> NodeId {
        let cmps: Vec<NodeId> = grid.iter().map(|&gp| {
            let c = g.constant(gp);
            if right { g.cmp(CmpOp::Le, c, v) } else { g.cmp(CmpOp::Lt, c, v) }
        }).collect();
        match cmps.len() {
            0 => g.constant(0.0),
            1 => cmps[0],
            _ => g.add(Node::Reduce(ReduceOp::Sum, cmps)),
        }
    };
    let v_arg = args.get_item(1)?;
    if let Ok(t) = v_arg.extract::<JitTracer>() {
        let node = graph.with(|g| emit_one(g, t.node_id));
        return Ok(Py::new(py, JitTracer::new(graph, node))?.into_any().into());
    }
    if let Ok(ta) = v_arg.extract::<JitTracerArray>() {
        let ids = ta.materialize();
        let nodes: Vec<NodeId> = graph.with(|g| {
            ids.iter().map(|&v| emit_one(g, v)).collect()
        });
        return Ok(Py::new(py, JitTracerArray::from_nodes_shape(
            graph, nodes, ta.shape.clone()))?.into_any().into());
    }
    Ok(py.NotImplemented().into())
}

/// Running accumulation (`cumsum`/`cumprod`) over a 1-D tracer array. Shared
/// by the `np.*` function arms and the `x.cumsum()`/`x.cumprod()` methods.
fn cumulative(ta: &JitTracerArray, op: BinOp) -> Vec<NodeId> {
    let ids = ta.materialize();
    if ids.is_empty() { return ids; }
    let mut acc = ids[0];
    let mut nodes = Vec::with_capacity(ids.len());
    nodes.push(acc);
    ta.graph.with(|g| {
        for &x in &ids[1..] {
            acc = g.binary(op, acc, x);
            nodes.push(acc);
        }
    });
    nodes
}

/// n-th first difference over a 1-D tracer array (numpy `np.diff` core).
/// `order >= len` yields an empty vec, exactly like numpy.
fn diff_nodes(ta: &JitTracerArray, order: usize) -> Vec<NodeId> {
    let mut nodes = ta.materialize();
    ta.graph.with(|g| {
        for _ in 0..order {
            if nodes.is_empty() { break; }
            nodes = (0..nodes.len().saturating_sub(1))
                .map(|i| g.binary(BinOp::Sub, nodes[i + 1], nodes[i]))
                .collect();
        }
    });
    nodes
}

/// Lower `np.argmax`/`np.argmin` as a running subgradient-style fold: carry
/// `(best_value, best_index)` through per-element `Select` pairs. Strict
/// comparison keeps numpy's first-occurrence-wins tie rule. The index is
/// returned as an f64 tracer (the graph is scalar f64 throughout).
fn arg_reduce(py: Python<'_>, ta: &JitTracerArray, is_max: bool) -> PyResult<Py<PyAny>> {
    if ta.size == 0 {
        return Err(PyValueError::new_err("argmax/argmin of an empty array"));
    }
    let ids = ta.materialize();
    let node = ta.graph.with(|g| {
        let mut best_val = ids[0];
        let mut best_idx = g.constant(0.0);
        for (i, &x) in ids.iter().enumerate().skip(1) {
            let better = if is_max {
                g.cmp(CmpOp::Gt, x, best_val)
            } else {
                g.cmp(CmpOp::Lt, x, best_val)
            };
            let idx_c = g.constant(i as f64);
            best_val = g.select(better, x, best_val);
            best_idx = g.select(better, idx_c, best_idx);
        }
        best_idx
    });
    Ok(Py::new(py, JitTracer::new(&ta.graph, node))?.into_any().into())
}

/// The variadic reduction matching an accumulation `BinOp`, if one exists.
/// Only the four ops that the numpy reduction paths use map across.
fn reduce_op_for(op: BinOp) -> Option<ReduceOp> {
    match op {
        BinOp::Add => Some(ReduceOp::Sum),
        BinOp::Mul => Some(ReduceOp::Product),
        BinOp::Min => Some(ReduceOp::Min),
        BinOp::Max => Some(ReduceOp::Max),
        _ => None,
    }
}

/// Collapse a list of element nodes into a single value. For a reducible op
/// (sum/product/min/max) this emits ONE `Reduce` node instead of an N-deep
/// `Binary` chain — the whole point of the structured op: the trace graph
/// stays O(1) per reduction, not O(N), and the tape walks one tight loop.
/// Trivial 0/1-element cases collapse to the identity constant / the operand,
/// matching the previous left-fold's result bit-for-bit (`0+a == a`, etc.).
fn fold_nodes(graph: &SharedGraph, op: BinOp, identity: f64, nodes: Vec<NodeId>) -> NodeId {
    match nodes.len() {
        0 => graph.constant(identity),
        1 => nodes[0],
        _ => match reduce_op_for(op) {
            Some(rop) => graph.add(Node::Reduce(rop, nodes)),
            None => {
                // Non-reducible op: fall back to a left-associative chain.
                let mut acc = nodes[0];
                for &n in &nodes[1..] {
                    acc = graph.binary(op, acc, n);
                }
                acc
            }
        },
    }
}

/// Fold an array into a scalar tracer with `op`. Empty array returns `identity`.
fn reduce_array(ta: &JitTracerArray, op: BinOp, identity: f64) -> JitTracer {
    let nodes = ta.materialize();
    JitTracer::new(&ta.graph, fold_nodes(&ta.graph, op, identity, nodes))
}

/// Build a fused dot product `Σ cᵢ·xᵢ` from (coefficient, value-node) pairs.
/// Emits ONE `Dot` node (an FMA loop in the tape) instead of an Add-chain of
/// products — the matrix-vector row primitive. Empty (all coeffs zero) → 0;
/// a single term → a plain `Mul` (the optimizer drops a unit coefficient).
fn coeff_dot(graph: &SharedGraph, terms: Vec<(f64, NodeId)>) -> NodeId {
    match terms.len() {
        0 => graph.constant(0.0),
        1 => {
            let (c, x) = terms[0];
            let cn = graph.constant(c);
            graph.binary(BinOp::Mul, cn, x)
        }
        _ => {
            let a: Vec<NodeId> = terms.iter().map(|&(c, _)| graph.constant(c)).collect();
            let b: Vec<NodeId> = terms.iter().map(|&(_, x)| x).collect();
            graph.add(Node::Dot(a, b))
        }
    }
}

/// Fused dot product of two value-node lists (both operands symbolic).
fn node_dot(graph: &SharedGraph, a: Vec<NodeId>, b: Vec<NodeId>) -> NodeId {
    match a.len() {
        0 => graph.constant(0.0),
        1 => graph.binary(BinOp::Mul, a[0], b[0]),
        _ => graph.add(Node::Dot(a, b)),
    }
}

/// Symbolic `@` where *both* operands are traced (e.g. a state-dependent matrix
/// `M(x) @ v`). The constant-matrix path lives in `__matmul__`/`__rmatmul__`;
/// this covers the all-symbolic 1-D/2-D combinations, each result entry a fused
/// `node_dot` row. Returns a scalar `JitTracer` for the 1-D·1-D case, otherwise
/// a `JitTracerArray` with the contracted shape.
fn symbolic_matmul(
    py: Python<'_>,
    a: &JitTracerArray,
    b: &JitTracerArray,
) -> PyResult<Py<PyAny>> {
    let g = &a.graph;
    let dim_err = |what: &str| pyo3::exceptions::PyValueError::new_err(what.to_string());
    // Materialize both operands once; rows/cols are slices of the flat vecs.
    let an = a.materialize();
    let bn = b.materialize();
    let row = |v: &[NodeId], base: usize, len: usize| -> Vec<NodeId> {
        v[base..base + len].to_vec()
    };
    let col = |v: &[NodeId], j: usize, rows: usize, cols: usize| -> Vec<NodeId> {
        (0..rows).map(|i| v[i * cols + j]).collect()
    };
    match (a.shape.len(), b.shape.len()) {
        // 1-D · 1-D → scalar dot.
        (1, 1) => {
            if a.size != b.size {
                return Err(dim_err(&format!(
                    "matmul size mismatch: ({},) @ ({},)", a.size, b.size)));
            }
            let d = node_dot(g, row(&an, 0, a.size), row(&bn, 0, b.size));
            Ok(Py::new(py, JitTracer::new(g, d))?.into_any().into())
        }
        // (m×k) · (k,) → (m,)
        (2, 1) => {
            let (m, k) = (a.shape[0], a.shape[1]);
            if k != b.size {
                return Err(dim_err(&format!(
                    "matmul shape mismatch: ({m},{k}) @ ({},)", b.size)));
            }
            let bv = row(&bn, 0, k);
            let out: Vec<NodeId> = (0..m)
                .map(|i| node_dot(g, row(&an, i * k, k), bv.clone()))
                .collect();
            Ok(Py::new(py, JitTracerArray::from_nodes(g, out))?.into_any().into())
        }
        // (k,) · (k×n) → (n,)
        (1, 2) => {
            let (k, n) = (b.shape[0], b.shape[1]);
            if a.size != k {
                return Err(dim_err(&format!(
                    "matmul shape mismatch: ({},) @ ({k},{n})", a.size)));
            }
            let av = row(&an, 0, k);
            let out: Vec<NodeId> = (0..n)
                .map(|j| node_dot(g, av.clone(), col(&bn, j, k, n)))
                .collect();
            Ok(Py::new(py, JitTracerArray::from_nodes(g, out))?.into_any().into())
        }
        // (m×k) · (k×n) → (m×n)
        (2, 2) => {
            let (m, k) = (a.shape[0], a.shape[1]);
            let (k2, n) = (b.shape[0], b.shape[1]);
            if k != k2 {
                return Err(dim_err(&format!(
                    "matmul shape mismatch: ({m},{k}) @ ({k2},{n})")));
            }
            let mut out = Vec::with_capacity(m * n);
            for i in 0..m {
                let arow = row(&an, i * k, k);
                for j in 0..n {
                    out.push(node_dot(g, arow.clone(), col(&bn, j, k, n)));
                }
            }
            Ok(Py::new(py, JitTracerArray::from_nodes_shape(g, out, vec![m, n]))?
                .into_any().into())
        }
        _ => Err(dim_err("matmul: only 1-D and 2-D operands are supported")),
    }
}

/// Reduce an array along a single axis.  Output shape has that axis removed.
/// 1-D input with `axis=0` reduces to a shape `[]` result — wrapped as a
/// 1-element array; callers check `out_shape.is_empty()` to know whether to
/// return a scalar `JitTracer` instead.
fn reduce_along_axis(
    ta: &JitTracerArray, op: BinOp, identity: f64, axis: isize,
) -> PyResult<(Vec<usize>, Vec<NodeId>)> {
    let n = ta.shape.len();
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cannot reduce along axis of a 0-D array"
        ));
    }
    let ax = if axis < 0 { n as isize + axis } else { axis };
    if ax < 0 || ax >= n as isize {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("axis {} out of bounds for {}-D array", axis, n)
        ));
    }
    let axis = ax as usize;

    let out_shape: Vec<usize> = ta.shape.iter().enumerate()
        .filter_map(|(i, &s)| if i == axis { None } else { Some(s) })
        .collect();
    let in_strides = strides_row_major(&ta.shape);
    let out_size: usize = out_shape.iter().product::<usize>().max(1);
    let reduce_len = ta.shape[axis];
    let axis_stride = in_strides[axis];

    let in_nodes = ta.materialize();
    let mut out_nodes = Vec::with_capacity(out_size);
    let mut out_idx = vec![0usize; out_shape.len()];
    for _ in 0..out_size {
        // Flat index in the input, with the reduced axis held at 0.
        let mut base_flat = 0usize;
        let mut od = 0usize;
        for d in 0..n {
            if d == axis { continue; }
            base_flat += out_idx[od] * in_strides[d];
            od += 1;
        }
        let terms: Vec<NodeId> = (0..reduce_len)
            .map(|k| in_nodes[base_flat + k * axis_stride])
            .collect();
        out_nodes.push(fold_nodes(&ta.graph, op, identity, terms));
        for d in (0..out_shape.len()).rev() {
            out_idx[d] += 1;
            if out_idx[d] < out_shape[d] { break; }
            out_idx[d] = 0;
        }
    }
    Ok((out_shape, out_nodes))
}

/// Extract the `axis` parameter from a numpy reduction call — positional
/// (`np.sum(arr, 0)`) or keyword (`np.sum(arr, axis=0)`).  `Ok(None)` means
/// "reduce everything to a scalar" (numpy default).
fn extract_axis(
    args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<isize>> {
    if args.len() >= 2 {
        // Positional axis — may be `None` (keep scalar reduction).
        let a = args.get_item(1)?;
        if a.is_none() { return Ok(None); }
        return a.extract::<isize>().map(Some);
    }
    if let Some(k) = kwargs {
        if let Ok(Some(v)) = k.get_item("axis") {
            if v.is_none() { return Ok(None); }
            return v.extract::<isize>().map(Some);
        }
    }
    Ok(None)
}

/// Extract the `keepdims` kwarg (numpy reductions).  Default: `false`.
fn extract_keepdims(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<bool> {
    if let Some(k) = kwargs {
        if let Ok(Some(v)) = k.get_item("keepdims") {
            return v.extract::<bool>();
        }
    }
    Ok(false)
}

/// True when every kwarg key is in `allowed`. The kwargs POLICY for every
/// `__array_function__` arm: a numpy parameter the arm does not model
/// (`out=`, `where=`, `initial=`, `dtype=`, ...) must NOT be silently
/// ignored — ignoring one produces a value that diverges from eager numpy
/// with no error (a silent miscompile). On an unknown kwarg the arm returns
/// `NotImplemented`; numpy raises `TypeError`, the trace fails, and the block
/// falls back to the opaque Python callback — fail-open.
fn kwargs_subset(kwargs: Option<&Bound<'_, PyDict>>, allowed: &[&str]) -> bool {
    match kwargs {
        None => true,
        Some(k) => k.iter().all(|(key, _)| {
            key.extract::<String>().is_ok_and(|s| allowed.contains(&s.as_str()))
        }),
    }
}

#[pymethods]
impl JitTracerArray {
    /// Higher priority than ndarray (0.0) so A @ tracer calls our __rmatmul__
    #[classattr]
    const __array_priority__: f64 = 20.0;

    /// Intercept numpy ufuncs (np.matmul is a ufunc, np.add, etc.)
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

        // ---- Composite ufuncs: elementwise compositions of existing ops, so
        // they trace + AD + codegen for free (see `super::unary_composite`). ----
        if matches!(name.as_str(),
            "deg2rad" | "radians" | "rad2deg" | "degrees" | "square"
            | "reciprocal" | "exp2" | "expit")
        {
            if args.len() != 1 { return Ok(py.NotImplemented().into()); }
            if let Ok(ta) = args.get_item(0)?.extract::<JitTracerArray>() {
                let ids = ta.materialize();
                let nodes: Vec<NodeId> = ta.graph.with(|g| {
                    ids.iter().map(|&x| super::unary_composite_g(g, &name, x)
                        .expect("name matched the composite list")).collect()
                });
                return Ok(Py::new(py, JitTracerArray::from_nodes_shape(
                    &ta.graph, nodes, ta.shape.clone()))?.into_any().into());
            }
            return Ok(py.NotImplemented().into());
        }

        // ---- Binary composite ufuncs (copysign / logaddexp / heaviside /
        //      floored remainder) ----
        if matches!(name.as_str(),
            "copysign" | "logaddexp" | "heaviside" | "remainder" | "mod")
        {
            if args.len() != 2 { return Ok(py.NotImplemented().into()); }
            let arg0 = args.get_item(0)?;
            let arg1 = args.get_item(1)?;
            if let Ok(ta) = arg0.extract::<JitTracerArray>() {
                let other: ArrayBinArg = arg1.extract()?;
                return ta.ufunc_elementwise(py, &name, None, None, other, false);
            }
            if let Ok(ta) = arg1.extract::<JitTracerArray>() {
                let other: ArrayBinArg = arg0.extract()?;
                return ta.ufunc_elementwise(py, &name, None, None, other, true);
            }
            return Ok(py.NotImplemented().into());
        }

        // ---- Unary ufuncs (np.sin, np.exp, ...): element-wise on the array ----
        if let Some(op) = unary_op_for_ufunc(&name) {
            if args.len() != 1 { return Ok(py.NotImplemented().into()); }
            let arg0 = args.get_item(0)?;
            if let Ok(ta) = arg0.extract::<JitTracerArray>() {
                return ta.array_unary(py, op);
            }
            return Ok(py.NotImplemented().into());
        }

        // ---- Binary ufuncs (np.add, np.minimum, np.arctan2, ...) ----
        if let Some(op) = binop_for_ufunc(&name) {
            if args.len() != 2 { return Ok(py.NotImplemented().into()); }
            let arg0 = args.get_item(0)?;
            let arg1 = args.get_item(1)?;
            if let Ok(ta) = arg0.extract::<JitTracerArray>() {
                let other: ArrayBinArg = arg1.extract()?;
                return ta.array_binop(py, op, other, false);
            }
            if let Ok(ta) = arg1.extract::<JitTracerArray>() {
                let other: ArrayBinArg = arg0.extract()?;
                return ta.array_binop(py, op, other, true);
            }
            return Ok(py.NotImplemented().into());
        }

        // ---- Comparison ufuncs (np.greater, np.equal, ...): return 0/1 array ----
        if let Some(op) = cmpop_for_ufunc(&name) {
            if args.len() != 2 { return Ok(py.NotImplemented().into()); }
            let arg0 = args.get_item(0)?;
            let arg1 = args.get_item(1)?;
            if let Ok(ta) = arg0.extract::<JitTracerArray>() {
                let other: ArrayBinArg = arg1.extract()?;
                return ta.array_cmpop(py, op, other, false);
            }
            if let Ok(ta) = arg1.extract::<JitTracerArray>() {
                let other: ArrayBinArg = arg0.extract()?;
                return ta.array_cmpop(py, op, other, true);
            }
            return Ok(py.NotImplemented().into());
        }

        // ---- Structural ops that don't fit the elementwise pattern ----
        match name.as_str() {
            "matmul" => {
                if args.len() != 2 { return Ok(py.NotImplemented().into()); }
                let arg0 = args.get_item(0)?;
                let arg1 = args.get_item(1)?;
                if let Ok(ta) = arg1.extract::<JitTracerArray>() {
                    ta.__rmatmul__(py, &arg0)
                } else if let Ok(ta) = arg0.extract::<JitTracerArray>() {
                    ta.__matmul__(py, &arg1)
                } else {
                    Ok(py.NotImplemented().into())
                }
            }
            _ => Ok(py.NotImplemented().into()),
        }
    }

    /// Indexing entry point.  Supports:
    /// - int (positive or negative)
    /// - slice `a:b:c` (negative steps included, e.g. `x[::-1]`)
    /// - constant integer lists/arrays (fancy indexing — node permutation)
    /// - `Ellipsis` and `None`/newaxis
    /// - tuple of the above (missing trailing axes default to `:`)
    ///
    /// Returns `JitTracer` when the result is scalar (all-int index on all
    /// axes of the array's rank), `JitTracerArray` otherwise.  Boolean masks
    /// are rejected: their output shape is data-dependent and untraceable.
    fn __getitem__(&self, py: Python<'_>, idx: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let indexers = self.parse_indexers(idx)?;
        let (out_shape, out_nodes) = self.apply_indexers(&indexers);
        if out_shape.is_empty() {
            // Scalar — preserves the existing 1-D `arr[i] → JitTracer` contract.
            Ok(Py::new(py, JitTracer::new(&self.graph, out_nodes[0]))?.into_any().into())
        } else {
            Ok(Py::new(py, JitTracerArray::from_nodes_shape(&self.graph, out_nodes, out_shape))?.into_any().into())
        }
    }

    /// In-place element / slice assignment — unlocks the common imperative ODE
    /// idiom `dx = np.zeros(n); dx[i] = ...; return dx`. Materializes lazy input
    /// arrays into an owned node list so writes don't alias the input space.
    fn __setitem__(&mut self, idx: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if !matches!(self.source, Source::Nodes(_)) {
            self.source = Source::Nodes(self.materialize());
        }
        let indexers = self.parse_indexers(idx)?;
        let positions = self.indexed_positions(&indexers);
        let value_nodes = self.coerce_value_nodes(value, positions.len())?;
        let ids = match &mut self.source {
            Source::Nodes(ids) => ids,
            Source::Input { .. } => unreachable!("materialized to Nodes above"),
        };
        for (pos, node) in positions.iter().zip(value_nodes) {
            ids[*pos] = node;
        }
        Ok(())
    }

    fn __len__(&self) -> usize {
        // numpy semantics: len(arr) is the size of the leading axis for N-D,
        // total size for 1-D.  Preserved for 1-D backwards compat.
        if self.shape.len() <= 1 { self.size } else { self.shape[0] }
    }

    // ---- N-D view ops (shape metadata; graph nodes unchanged or permuted) ----

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.shape)
    }

    #[getter]
    fn ndim(&self) -> usize { self.shape.len() }

    #[getter]
    fn size(&self) -> usize { self.size }

    /// `x.T` — transpose with reversed axes (numpy convention).
    #[getter]
    #[pyo3(name = "T")]
    fn transpose_default(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.transpose(py, None)
    }

    /// Reshape to the given shape.  One axis may be `-1` (inferred from the
    /// remaining product).  Zero-cost: keeps the flat node IDs, swaps metadata.
    #[pyo3(signature = (*shape))]
    fn reshape(&self, py: Python<'_>, shape: Vec<i64>) -> PyResult<Py<PyAny>> {
        // PyO3 already unpacks both reshape(h, w) and reshape((h, w)) into a
        // plain Vec<i64> via the `*shape` signature, so no flattening is needed.
        let mut new_shape: Vec<usize> = Vec::with_capacity(shape.len());
        let mut inferred: Option<usize> = None;
        let mut known_product: usize = 1;
        for (i, d) in shape.iter().enumerate() {
            if *d == -1 {
                if inferred.is_some() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "reshape: only one dimension can be -1"
                    ));
                }
                inferred = Some(i);
                new_shape.push(0);  // placeholder
            } else if *d >= 0 {
                new_shape.push(*d as usize);
                known_product *= *d as usize;
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("reshape: invalid dimension {}", d)
                ));
            }
        }
        if let Some(i) = inferred {
            if known_product == 0 || !self.size.is_multiple_of(known_product) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("cannot reshape array of size {} into shape {:?}", self.size, shape)
                ));
            }
            new_shape[i] = self.size / known_product;
        }
        let prod: usize = new_shape.iter().product();
        if prod != self.size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("cannot reshape array of size {} into shape {:?}", self.size, new_shape)
            ));
        }
        let nodes = self.materialize();
        Ok(Py::new(py, JitTracerArray::from_nodes_shape(&self.graph, nodes, new_shape))?.into_any().into())
    }

    /// Transpose / permute axes.  `axes=None` reverses all axes (numpy default).
    /// For 1-D arrays the result is identical to the input (numpy behavior).
    #[pyo3(signature = (axes=None))]
    fn transpose(&self, py: Python<'_>, axes: Option<Vec<isize>>) -> PyResult<Py<PyAny>> {
        let n = self.shape.len();
        let axes: Vec<usize> = match axes {
            None => (0..n).rev().collect(),
            Some(v) => {
                if v.len() != n {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        format!("axes don't match array: got {} axes, array is {}-D", v.len(), n)
                    ));
                }
                let mut out = Vec::with_capacity(n);
                let mut seen = vec![false; n];
                for a in v {
                    let a = if a < 0 { (n as isize + a) as usize } else { a as usize };
                    if a >= n || seen[a] {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            format!("axes must be a permutation of 0..{}", n)
                        ));
                    }
                    seen[a] = true;
                    out.push(a);
                }
                out
            }
        };

        // 1-D or 0-D: transpose is a no-op.
        if n <= 1 {
            return Ok(Py::new(py, self.clone())?.into_any().into());
        }

        let old_strides = strides_row_major(&self.shape);
        let new_shape: Vec<usize> = axes.iter().map(|&a| self.shape[a]).collect();
        let old_nodes = self.materialize();

        let mut new_nodes = Vec::with_capacity(self.size);
        let mut idx = vec![0usize; n];
        for _ in 0..self.size {
            let mut old_flat = 0usize;
            for d in 0..n {
                old_flat += idx[d] * old_strides[axes[d]];
            }
            new_nodes.push(old_nodes[old_flat]);
            // Row-major increment over new_shape (carries from last axis).
            for d in (0..n).rev() {
                idx[d] += 1;
                if idx[d] < new_shape[d] { break; }
                idx[d] = 0;
            }
        }
        Ok(Py::new(py, JitTracerArray::from_nodes_shape(&self.graph, new_nodes, new_shape))?.into_any().into())
    }

    /// Squeeze — drop size-1 axes.  `axis=None` drops all size-1 axes;
    /// integer axis drops that specific axis only.
    #[pyo3(signature = (axis=None))]
    fn squeeze(&self, py: Python<'_>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
        let n = self.shape.len();
        let new_shape: Vec<usize> = match axis {
            None => self.shape.iter().copied().filter(|&s| s != 1).collect(),
            Some(a) => {
                let a = if a < 0 { (n as isize + a) as usize } else { a as usize };
                if a >= n {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        format!("axis {} out of bounds for {}-D array", a, n)
                    ));
                }
                if self.shape[a] != 1 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        format!("cannot squeeze axis {} with size {}", a, self.shape[a])
                    ));
                }
                self.shape.iter().enumerate()
                    .filter_map(|(i, &s)| if i == a { None } else { Some(s) })
                    .collect()
            }
        };
        // Squeeze on fully-1 shape leaves a single-element 1-D array (not 0-D)
        let new_shape = if new_shape.is_empty() { vec![1] } else { new_shape };
        let nodes = self.materialize();
        Ok(Py::new(py, JitTracerArray::from_nodes_shape(&self.graph, nodes, new_shape))?.into_any().into())
    }

    /// Insert a size-1 axis at `axis` (numpy `expand_dims`).
    fn unsqueeze(&self, py: Python<'_>, axis: isize) -> PyResult<Py<PyAny>> {
        let n = self.shape.len();
        let a = if axis < 0 { (n as isize + 1 + axis) as usize } else { axis as usize };
        if a > n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("axis {} out of bounds for insert into {}-D array", axis, n)
            ));
        }
        let mut new_shape = self.shape.clone();
        new_shape.insert(a, 1);
        let nodes = self.materialize();
        Ok(Py::new(py, JitTracerArray::from_nodes_shape(&self.graph, nodes, new_shape))?.into_any().into())
    }

    // ---- numpy-style METHOD forms (x.sum(), x.mean(axis=0), x.dot(v), …).
    //      Same lowering as the np.* function forms — both funnel through
    //      `reduce_dispatch` / `mean_dispatch` / the matmul paths, so the two
    //      spellings can never diverge. ----

    /// numpy-style `x.sum(axis=None, keepdims=False)` over the traced array.
    #[pyo3(signature = (axis=None, keepdims=false))]
    fn sum(&self, py: Python<'_>, axis: Option<isize>, keepdims: bool) -> PyResult<Py<PyAny>> {
        self.reduce_dispatch(py, BinOp::Add, 0.0, axis, keepdims)
    }

    /// numpy-style `x.prod(axis=None, keepdims=False)` over the traced array.
    #[pyo3(signature = (axis=None, keepdims=false))]
    fn prod(&self, py: Python<'_>, axis: Option<isize>, keepdims: bool) -> PyResult<Py<PyAny>> {
        self.reduce_dispatch(py, BinOp::Mul, 1.0, axis, keepdims)
    }

    /// numpy-style `x.min(axis=None, keepdims=False)` over the traced array.
    #[pyo3(signature = (axis=None, keepdims=false))]
    fn min(&self, py: Python<'_>, axis: Option<isize>, keepdims: bool) -> PyResult<Py<PyAny>> {
        self.reduce_dispatch(py, BinOp::Min, f64::INFINITY, axis, keepdims)
    }

    /// numpy-style `x.max(axis=None, keepdims=False)` over the traced array.
    #[pyo3(signature = (axis=None, keepdims=false))]
    fn max(&self, py: Python<'_>, axis: Option<isize>, keepdims: bool) -> PyResult<Py<PyAny>> {
        self.reduce_dispatch(py, BinOp::Max, f64::NEG_INFINITY, axis, keepdims)
    }

    /// numpy-style `x.mean(axis=None, keepdims=False)` over the traced array.
    #[pyo3(signature = (axis=None, keepdims=false))]
    fn mean(&self, py: Python<'_>, axis: Option<isize>, keepdims: bool) -> PyResult<Py<PyAny>> {
        self.mean_dispatch(py, axis, keepdims)
    }

    /// numpy-style `x.var(ddof=0)` — same lowering as `np.var(x, ddof=...)`.
    #[pyo3(signature = (ddof=0))]
    fn var(&self, py: Python<'_>, ddof: usize) -> PyResult<Py<PyAny>> {
        var_std(py, self, ddof, false)
    }

    /// numpy-style `x.std(ddof=0)` — same lowering as `np.std(x, ddof=...)`.
    #[pyo3(signature = (ddof=0))]
    fn std(&self, py: Python<'_>, ddof: usize) -> PyResult<Py<PyAny>> {
        var_std(py, self, ddof, true)
    }

    /// numpy-style `x.cumsum()` (1-D only, like the `np.cumsum` arm).
    fn cumsum(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.shape.len() > 1 || self.size == 0 {
            return Err(PyValueError::new_err("cumsum: 1-D tracer arrays only"));
        }
        let nodes = cumulative(self, BinOp::Add);
        Ok(Py::new(py, JitTracerArray::from_nodes(&self.graph, nodes))?.into_any().into())
    }

    /// numpy-style `x.cumprod()` (1-D only, like the `np.cumprod` arm).
    fn cumprod(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.shape.len() > 1 || self.size == 0 {
            return Err(PyValueError::new_err("cumprod: 1-D tracer arrays only"));
        }
        let nodes = cumulative(self, BinOp::Mul);
        Ok(Py::new(py, JitTracerArray::from_nodes(&self.graph, nodes))?.into_any().into())
    }

    /// numpy-style `x.argmax()` — running (best, index) Select fold.
    fn argmax(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        arg_reduce(py, self, true)
    }

    /// numpy-style `x.argmin()` — running (best, index) Select fold.
    fn argmin(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        arg_reduce(py, self, false)
    }

    /// `x.dot(other)` — numpy `.dot`: vector dot for 1-D operands, matrix
    /// product for 2-D, scale for scalars.
    fn dot(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(tb) = other.extract::<JitTracerArray>() {
            return symbolic_matmul(py, self, &tb);
        }
        // 2-D const operand BEFORE the flat Vec conversion (a (k,1) nested
        // list would otherwise silently flatten — see `__rmatmul__`).
        if other.extract::<Vec<Vec<f64>>>().is_ok() {
            return self.__matmul__(py, other);
        }
        if let Ok(vs) = other.extract::<Vec<f64>>() {
            let nodes: Vec<NodeId> = vs.iter().map(|&v| self.graph.constant(v)).collect();
            let tb = JitTracerArray::from_nodes(&self.graph, nodes);
            return symbolic_matmul(py, self, &tb);
        }
        // np.dot with a scalar operand multiplies.
        if let Ok(other) = other.extract::<ArrayBinArg>() {
            if matches!(other, ArrayBinArg::Tracer(_) | ArrayBinArg::Float(_)) {
                return self.array_binop(py, BinOp::Mul, other, false);
            }
        }
        Err(PyTypeError::new_err("x.dot: unsupported operand during JIT trace"))
    }

    /// Flat 1-D copy (numpy `.flatten()`). Tracer arrays own their node list,
    /// so the copy/view distinction has no observable effect during a trace.
    fn flatten(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let nodes = self.materialize();
        Ok(Py::new(py, JitTracerArray::from_nodes(&self.graph, nodes))?.into_any().into())
    }

    /// Flat 1-D view (numpy `.ravel()`); identical to `.flatten()` here.
    fn ravel(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.flatten(py)
    }

    /// Element copy (numpy `.copy()`); nodes are immutable, so a clone suffices.
    fn copy(&self) -> Self {
        self.clone()
    }

    /// `x.clip(lo, hi)` — same lowering as `np.clip(x, lo, hi)` (Max/Min
    /// chain); `None` keeps the corresponding bound open.
    #[pyo3(signature = (a_min=None, a_max=None))]
    fn clip(
        &self,
        py: Python<'_>,
        a_min: Option<TracerOrFloat>,
        a_max: Option<TracerOrFloat>,
    ) -> PyResult<Py<PyAny>> {
        let resolve = |b: Option<TracerOrFloat>, unbounded: f64| match b {
            Some(TracerOrFloat::Tracer(t)) => t.node_id,
            Some(TracerOrFloat::Float(v)) => self.graph.constant(v),
            None => self.graph.constant(unbounded),
        };
        let lo = resolve(a_min, f64::NEG_INFINITY);
        let hi = resolve(a_max, f64::INFINITY);
        let ids = self.materialize();
        let nodes: Vec<NodeId> = self.graph.with(|g| {
            ids.iter().map(|&v| {
                let m = g.binary(BinOp::Max, v, lo);
                g.binary(BinOp::Min, m, hi)
            }).collect()
        });
        Ok(Py::new(py, JitTracerArray::from_nodes_shape(
            &self.graph, nodes, self.shape.clone()))?.into_any().into())
    }

    fn __neg__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.array_unary(py, UnaryOp::Neg)
    }

    // ---- Element-wise arithmetic ----
    //
    // Semantics (numpy-compatible):
    //   array ⊕ float/int            → broadcast scalar to each element
    //   array ⊕ JitTracer            → broadcast tracer-scalar to each element
    //   array ⊕ JitTracerArray       → element-wise, sizes must match
    //   array ⊕ list/ndarray<f64>    → element-wise with constants, sizes must match
    //
    // The `r`-variants (Python dispatches them when the right operand is a
    // JitTracerArray but the left is a plain number/list/ndarray) flip operand
    // order in the binary op.

    fn __add__(&self, py: Python<'_>, other: ArrayBinArg) -> PyResult<Py<PyAny>> {
        self.array_binop(py, BinOp::Add, other, false)
    }
    fn __radd__(&self, py: Python<'_>, other: ArrayBinArg) -> PyResult<Py<PyAny>> {
        // Addition is commutative; keep swap=false for clearer node ordering
        self.array_binop(py, BinOp::Add, other, false)
    }
    fn __sub__(&self, py: Python<'_>, other: ArrayBinArg) -> PyResult<Py<PyAny>> {
        self.array_binop(py, BinOp::Sub, other, false)
    }
    fn __rsub__(&self, py: Python<'_>, other: ArrayBinArg) -> PyResult<Py<PyAny>> {
        self.array_binop(py, BinOp::Sub, other, true)
    }
    fn __mul__(&self, py: Python<'_>, other: ArrayBinArg) -> PyResult<Py<PyAny>> {
        self.array_binop(py, BinOp::Mul, other, false)
    }
    fn __rmul__(&self, py: Python<'_>, other: ArrayBinArg) -> PyResult<Py<PyAny>> {
        self.array_binop(py, BinOp::Mul, other, false)
    }
    fn __truediv__(&self, py: Python<'_>, other: ArrayBinArg) -> PyResult<Py<PyAny>> {
        self.array_binop(py, BinOp::Div, other, false)
    }
    fn __rtruediv__(&self, py: Python<'_>, other: ArrayBinArg) -> PyResult<Py<PyAny>> {
        self.array_binop(py, BinOp::Div, other, true)
    }
    fn __pow__(
        &self, py: Python<'_>, other: ArrayBinArg, _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        self.array_binop(py, BinOp::Pow, other, false)
    }
    fn __rpow__(
        &self, py: Python<'_>, other: ArrayBinArg, _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        self.array_binop(py, BinOp::Pow, other, true)
    }
    // Python `%` is floored modulo — see `super::floored_mod`.
    fn __mod__(&self, py: Python<'_>, other: ArrayBinArg) -> PyResult<Py<PyAny>> {
        self.array_elementwise(py, other, false, super::floored_mod_g)
    }
    fn __rmod__(&self, py: Python<'_>, other: ArrayBinArg) -> PyResult<Py<PyAny>> {
        self.array_elementwise(py, other, true, super::floored_mod_g)
    }

    // ---- Element-wise comparisons (return 0.0/1.0 arrays) ----
    fn __gt__(&self, py: Python<'_>, other: ArrayBinArg) -> PyResult<Py<PyAny>> {
        self.array_cmpop(py, CmpOp::Gt, other, false)
    }
    fn __ge__(&self, py: Python<'_>, other: ArrayBinArg) -> PyResult<Py<PyAny>> {
        self.array_cmpop(py, CmpOp::Ge, other, false)
    }
    fn __lt__(&self, py: Python<'_>, other: ArrayBinArg) -> PyResult<Py<PyAny>> {
        self.array_cmpop(py, CmpOp::Lt, other, false)
    }
    fn __le__(&self, py: Python<'_>, other: ArrayBinArg) -> PyResult<Py<PyAny>> {
        self.array_cmpop(py, CmpOp::Le, other, false)
    }
    fn __eq__(&self, py: Python<'_>, other: ArrayBinArg) -> PyResult<Py<PyAny>> {
        self.array_cmpop(py, CmpOp::Eq, other, false)
    }
    fn __ne__(&self, py: Python<'_>, other: ArrayBinArg) -> PyResult<Py<PyAny>> {
        self.array_cmpop(py, CmpOp::Ne, other, false)
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<JitTracerArrayIter>> {
        Py::new(slf.py(), JitTracerArrayIter {
            array: slf.clone(),
            index: 0,
        })
    }

    /// Matrix or vector @ TracerArray. Python `a @ x` lands here when `x` is
    /// the TracerArray. For 1D `a`, this is a dot product returning a scalar
    /// `JitTracer`; for 2D `a`, it is the usual matrix-vector product
    /// returning a `JitTracerArray`.
    fn __rmatmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // Both operands symbolic (e.g. `M(x) @ v`): build the products from nodes.
        if let Ok(lhs) = other.extract::<JitTracerArray>() {
            return symbolic_matmul(py, &lhs, self);
        }
        // 2-D must be probed BEFORE the flat Vec<f64> conversion: a nested
        // (k,1) or (1,1) list-of-lists would otherwise flatten into a k-vector
        // (each inner 1-element array converts to a scalar) and silently take
        // the 1-D dot path with the wrong shape semantics.
        let matrix: Option<Vec<Vec<f64>>> = other.extract().ok();

        // 1D: dot product — `sum(a[i] * x[i])`.
        if matrix.is_none() {
            if let Ok(a) = other.extract::<Vec<f64>>() {
                if a.len() != self.size {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        format!("dot product size mismatch: ({},) @ ({},)", a.len(), self.size)
                    ));
                }
                let terms: Vec<(f64, NodeId)> = (0..self.size)
                    .filter(|&i| a[i] != 0.0)
                    .map(|i| (a[i], self.get_node(i)))
                    .collect();
                let scalar_id = coeff_dot(&self.graph, terms);
                let tracer = JitTracer::new(&self.graph, scalar_id);
                return Ok(Py::new(py, tracer)?.into_any().into());
            }
        }

        // 2D: matrix-vector product.
        let a: Vec<Vec<f64>> = matrix.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "@ expected a 1D or 2D numeric array"
            )
        })?;
        let rows = a.len();
        if rows == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("empty matrix"));
        }
        let cols = a[0].len();
        if cols != self.size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("matrix columns ({}) != array size ({})", cols, self.size)
            ));
        }

        // Build result: out[i] = Σⱼ A[i][j]·x[j], one fused Dot per row.
        let mut result_nodes = Vec::with_capacity(rows);
        for row in a.iter().take(rows) {
            let terms: Vec<(f64, NodeId)> = (0..cols)
                .filter(|&j| row[j] != 0.0)
                .map(|j| (row[j], self.get_node(j)))
                .collect();
            result_nodes.push(coeff_dot(&self.graph, terms));
        }

        let result = JitTracerArray::from_nodes(&self.graph, result_nodes);
        Ok(Py::new(py, result)?.into_any().into())
    }

    /// TracerArray @ Matrix: x @ A (Python __matmul__ slot)
    fn __matmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // Both operands symbolic (e.g. `M(x) @ v`): build the products from nodes.
        if let Ok(rhs) = other.extract::<JitTracerArray>() {
            return symbolic_matmul(py, self, &rhs);
        }
        // For x @ A, treat x as row vector
        let a: Vec<Vec<f64>> = other.extract()?;
        let rows = a.len();
        if rows != self.size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("matrix rows ({}) != array size ({})", rows, self.size)
            ));
        }
        let cols = if rows > 0 { a[0].len() } else { 0 };

        let mut result_nodes = Vec::with_capacity(cols);
        for j in 0..cols {
            let terms: Vec<(f64, NodeId)> = (0..rows)
                .filter(|&i| a[i][j] != 0.0)
                .map(|i| (a[i][j], self.get_node(i)))
                .collect();
            result_nodes.push(coeff_dot(&self.graph, terms));
        }

        let result = JitTracerArray::from_nodes(&self.graph, result_nodes);
        Ok(Py::new(py, result)?.into_any().into())
    }

    /// NEP 18: intercept np.dot, np.matmul, np.sum, np.array
    #[pyo3(signature = (func, _types, args, kwargs=None))]
    fn __array_function__(
        &self,
        py: Python<'_>,
        func: &Bound<'_, PyAny>,
        _types: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let name: String = func.getattr("__name__")?.extract()?;
        // Get module for distinguishing np.linalg functions
        let module: String = func.getattr("__module__")
            .and_then(|m| m.extract())
            .unwrap_or_default();
        match name.as_str() {
            // np.diff — 1-D n-th difference, composed from elementwise
            // subtraction (traces + AD + codegen). Supports integer `n`
            // (positional or kwarg); `axis`/`prepend`/`append` are not
            // modelled → NotImplemented (fail-open, not a silent first-diff).
            "diff" => {
                if self.shape.len() > 1 || self.size == 0 { return Ok(py.NotImplemented().into()); }
                if !kwargs_subset(kwargs, &["n"]) || args.len() > 2 {
                    return Ok(py.NotImplemented().into());
                }
                let mut order: usize = 1;
                if args.len() == 2 {
                    match args.get_item(1)?.extract::<usize>() {
                        Ok(n) => order = n,
                        Err(_) => return Ok(py.NotImplemented().into()),
                    }
                }
                if let Some(k) = kwargs {
                    if let Ok(Some(v)) = k.get_item("n") {
                        match v.extract::<usize>() {
                            Ok(n) => order = n,
                            Err(_) => return Ok(py.NotImplemented().into()),
                        }
                    }
                }
                // order >= size yields an empty array, exactly like numpy
                // (the width-1 probe relies on this: np.sum(np.diff(u)) must
                // trace to the 0.0 constant there).
                let nodes = diff_nodes(self, order);
                Ok(Py::new(py, JitTracerArray::from_nodes(&self.graph, nodes))?.into_any().into())
            }
            // np.cumsum — 1-D prefix sum via an O(n) running-add chain.
            // Positional/keyword axis/dtype/out are not modelled → NotImplemented.
            "cumsum" => {
                if self.shape.len() > 1 || self.size == 0 { return Ok(py.NotImplemented().into()); }
                if args.len() > 1 || !kwargs_subset(kwargs, &[]) {
                    return Ok(py.NotImplemented().into());
                }
                let nodes = cumulative(self, BinOp::Add);
                Ok(Py::new(py, JitTracerArray::from_nodes(&self.graph, nodes))?.into_any().into())
            }
            "matmul" => {
                if args.len() != 2 { return Ok(py.NotImplemented().into()); }
                let arg0 = args.get_item(0)?;
                let arg1 = args.get_item(1)?;
                if let Ok(ta) = arg1.extract::<JitTracerArray>() {
                    ta.__rmatmul__(py, &arg0)
                } else if let Ok(ta) = arg0.extract::<JitTracerArray>() {
                    ta.__matmul__(py, &arg1)
                } else {
                    Ok(py.NotImplemented().into())
                }
            }
            // np.dot — vector-vector dot product or matrix-vector
            "dot" => {
                if args.len() != 2 { return Ok(py.NotImplemented().into()); }
                let arg0 = args.get_item(0)?;
                let arg1 = args.get_item(1)?;
                // 2-D operands mean matrix-product semantics. Probe them BEFORE
                // the flat Vec<f64> conversion below — a nested (k,1)/(1,1)
                // list-of-lists would otherwise flatten into a k-vector and
                // silently take the 1-D dot path (see `__rmatmul__`).
                if let Ok(ta) = arg1.extract::<JitTracerArray>() {
                    if arg0.extract::<Vec<Vec<f64>>>().is_ok() {
                        return ta.__rmatmul__(py, &arg0);
                    }
                }
                if let Ok(ta) = arg0.extract::<JitTracerArray>() {
                    if ta.shape.len() == 2 {
                        // Traced 2-D first operand: contract like `M @ v`.
                        if let Ok(tb) = arg1.extract::<JitTracerArray>() {
                            return symbolic_matmul(py, &ta, &tb);
                        }
                        if let Ok(vs) = arg1.extract::<Vec<f64>>() {
                            let nodes: Vec<NodeId> =
                                vs.iter().map(|&v| ta.graph.constant(v)).collect();
                            let tb = JitTracerArray::from_nodes(&ta.graph, nodes);
                            return symbolic_matmul(py, &ta, &tb);
                        }
                    }
                    if arg1.extract::<Vec<Vec<f64>>>().is_ok() {
                        return ta.__matmul__(py, &arg1);
                    }
                }
                // Convert ndarray args to TracerArray constants if needed
                let ta0 = arg0.extract::<JitTracerArray>().ok().or_else(|| {
                    let vals: Vec<f64> = arg0.extract().ok()?;
                    let nodes: Vec<NodeId> = vals.iter().map(|&v| self.graph.constant(v)).collect();
                    Some(JitTracerArray::from_nodes(&self.graph, nodes))
                });
                let ta1 = arg1.extract::<JitTracerArray>().ok().or_else(|| {
                    let vals: Vec<f64> = arg1.extract().ok()?;
                    let nodes: Vec<NodeId> = vals.iter().map(|&v| self.graph.constant(v)).collect();
                    Some(JitTracerArray::from_nodes(&self.graph, nodes))
                });
                // Both → scalar dot product (one fused Dot node). Mismatched
                // sizes RAISE, exactly like eager numpy — silently contracting
                // over min(len) would turn a user shape bug into wrong values.
                if let (Some(a), Some(b)) = (&ta0, &ta1) {
                    if a.size != b.size {
                        return Err(PyValueError::new_err(format!(
                            "np.dot: shapes ({},) and ({},) not aligned", a.size, b.size)));
                    }
                    let n = a.size;
                    let a_nodes: Vec<NodeId> = (0..n).map(|i| a.get_node(i)).collect();
                    let b_nodes: Vec<NodeId> = (0..n).map(|i| b.get_node(i)).collect();
                    let result = JitTracer::new(&a.graph, node_dot(&a.graph, a_nodes, b_nodes));
                    return Ok(Py::new(py, result)?.into_any().into());
                }
                // One is ndarray, other is TracerArray → delegate to matmul
                if let Some(ta) = ta1 {
                    ta.__rmatmul__(py, &arg0)
                } else if let Some(ta) = ta0 {
                    ta.__matmul__(py, &arg1)
                } else {
                    Ok(py.NotImplemented().into())
                }
            }
            // np.zeros_like / ones_like / empty_like — dispatched here because
            // they take the (tracer) array as their first argument. Same shape,
            // constant fill. (np.zeros/ones/empty/full with a literal size carry
            // no array arg and are intercepted by monkeypatch during the trace.)
            "zeros_like" | "empty_like" | "ones_like" => {
                if args.is_empty() { return Ok(py.NotImplemented().into()); }
                if let Ok(ta) = args.get_item(0)?.extract::<JitTracerArray>() {
                    let fill = if name == "ones_like" { 1.0 } else { 0.0 };
                    let c = self.graph.constant(fill);
                    let nodes = vec![c; ta.size];
                    return Ok(Py::new(py, JitTracerArray::from_nodes_shape(
                        &self.graph, nodes, ta.shape.clone()))?.into_any().into());
                }
                Ok(py.NotImplemented().into())
            }
            "sum" | "prod" | "amin" | "min" | "amax" | "max" => {
                if args.is_empty() { return Ok(py.NotImplemented().into()); }
                // kwargs POLICY: silently ignoring `where=` / `initial=` /
                // `dtype=` / `out=` diverges from eager numpy. Positional
                // args beyond (x, axis) (e.g. a positional dtype) likewise.
                if !kwargs_subset(kwargs, &["axis", "keepdims"]) || args.len() > 2 {
                    return Ok(py.NotImplemented().into());
                }
                if let Ok(ta) = args.get_item(0)?.extract::<JitTracerArray>() {
                    let (op, identity) = match name.as_str() {
                        "sum"           => (BinOp::Add, 0.0),
                        "prod"          => (BinOp::Mul, 1.0),
                        "amin" | "min"  => (BinOp::Min, f64::INFINITY),
                        "amax" | "max"  => (BinOp::Max, f64::NEG_INFINITY),
                        _ => unreachable!(),
                    };
                    let axis = extract_axis(args, kwargs)?;
                    let keepdims = extract_keepdims(kwargs)?;
                    ta.reduce_dispatch(py, op, identity, axis, keepdims)
                } else {
                    Ok(py.NotImplemented().into())
                }
            }
            "mean" => {
                if args.is_empty() { return Ok(py.NotImplemented().into()); }
                if !kwargs_subset(kwargs, &["axis", "keepdims"]) || args.len() > 2 {
                    return Ok(py.NotImplemented().into());
                }
                if let Ok(ta) = args.get_item(0)?.extract::<JitTracerArray>() {
                    let axis = extract_axis(args, kwargs)?;
                    let keepdims = extract_keepdims(kwargs)?;
                    return ta.mean_dispatch(py, axis, keepdims);
                }
                Ok(py.NotImplemented().into())
            }
            "var" | "std" => {
                // Variance E[(x - E[x])^2] with numpy's ddof correction:
                // divide the squared-diff sum by (n - ddof). std = sqrt(var).
                // The squared-diff sum is ONE fused `Dot(d, d)` node — the
                // structured form the tape evaluates with the 4-lane kernel —
                // instead of an O(N) Pow/Add chain. axis/where/out are not
                // modelled → NotImplemented (fail-open).
                if args.is_empty() { return Ok(py.NotImplemented().into()); }
                if !kwargs_subset(kwargs, &["ddof"]) || args.len() > 1 {
                    return Ok(py.NotImplemented().into());
                }
                let mut ddof: usize = 0;
                if let Some(k) = kwargs {
                    if let Ok(Some(v)) = k.get_item("ddof") {
                        match v.extract::<usize>() {
                            Ok(d) => ddof = d,
                            Err(_) => return Ok(py.NotImplemented().into()),
                        }
                    }
                }
                if let Ok(ta) = args.get_item(0)?.extract::<JitTracerArray>() {
                    if ddof >= ta.size { return Ok(py.NotImplemented().into()); }
                    return var_std(py, &ta, ddof, name == "std");
                }
                Ok(py.NotImplemented().into())
            }
            // np.argmax / np.argmin — running (best, index) fold over Selects.
            // axis/out are not modelled → NotImplemented (fail-open).
            "argmax" | "argmin" => {
                if args.len() != 1 || !kwargs_subset(kwargs, &[]) {
                    return Ok(py.NotImplemented().into());
                }
                if let Ok(ta) = args.get_item(0)?.extract::<JitTracerArray>() {
                    return arg_reduce(py, &ta, name == "argmax");
                }
                Ok(py.NotImplemented().into())
            }
            // np.linalg.solve(A, b) with a CONSTANT coefficient matrix: invert
            // A at trace time and emit x = A⁻¹ b as n fused dot products over
            // b's nodes. A symbolic A stays structural (no symbolic inverse).
            "solve" if module.contains("linalg") => {
                if args.len() != 2 || !kwargs_subset(kwargs, &[]) {
                    return Ok(py.NotImplemented().into());
                }
                let Ok(a_const) = args.get_item(0)?.extract::<Vec<Vec<f64>>>() else {
                    return Ok(py.NotImplemented().into());
                };
                let Ok(b) = args.get_item(1)?.extract::<JitTracerArray>() else {
                    return Ok(py.NotImplemented().into());
                };
                if a_const.len() != b.size {
                    return Err(PyValueError::new_err(format!(
                        "np.linalg.solve: matrix rows ({}) != rhs size ({})",
                        a_const.len(), b.size)));
                }
                let Some(inv) = invert_matrix(&a_const) else {
                    return Err(PyValueError::new_err(
                        "np.linalg.solve: singular (or non-square) matrix"));
                };
                let bn = b.materialize();
                let out: Vec<NodeId> = inv.iter().map(|row| {
                    let terms: Vec<(f64, NodeId)> = row.iter().zip(&bn)
                        .filter(|(c, _)| **c != 0.0)
                        .map(|(&c, &x)| (c, x))
                        .collect();
                    coeff_dot(&b.graph, terms)
                }).collect();
                Ok(Py::new(py, JitTracerArray::from_nodes(&b.graph, out))?.into_any().into())
            }
            // np.select(condlist, choicelist, default=0) — right fold of
            // per-element Selects: the first true condition wins, exactly
            // like numpy. Sizes must match or be scalar.
            "select" => {
                if args.len() < 2 || args.len() > 3 || !kwargs_subset(kwargs, &["default"]) {
                    return Ok(py.NotImplemented().into());
                }
                let to_operands = |seq: &Bound<'_, PyAny>| -> Option<Vec<ArrayBinArg>> {
                    let mut out = Vec::new();
                    for item in seq.try_iter().ok()? {
                        out.push(item.ok()?.extract::<ArrayBinArg>().ok()?);
                    }
                    Some(out)
                };
                let (Some(conds), Some(choices)) = (
                    to_operands(&args.get_item(0)?),
                    to_operands(&args.get_item(1)?),
                ) else {
                    return Ok(py.NotImplemented().into());
                };
                if conds.len() != choices.len() || conds.is_empty() {
                    return Ok(py.NotImplemented().into());
                }
                let default: ArrayBinArg = if args.len() == 3 {
                    args.get_item(2)?.extract()?
                } else if let Some(Ok(Some(d))) = kwargs.map(|k| k.get_item("default")) {
                    d.extract()?
                } else {
                    ArrayBinArg::Float(0.0)
                };
                // Common output size: max over all vector operands.
                let size_of = |a: &ArrayBinArg| -> Option<usize> {
                    match a {
                        ArrayBinArg::TracerArray(ta) => Some(ta.size),
                        ArrayBinArg::NdArray(vs) => Some(vs.len()),
                        _ => None,
                    }
                };
                let mut out_n = 1usize;
                for a in conds.iter().chain(choices.iter()).chain(std::iter::once(&default)) {
                    if let Some(s) = size_of(a) {
                        if s != 1 && out_n != 1 && s != out_n {
                            return Err(PyValueError::new_err(format!(
                                "np.select: size mismatch {} vs {}", out_n, s)));
                        }
                        out_n = out_n.max(s);
                    }
                }
                let node_at = |g: &mut Graph, a: &ArrayBinArg, nodes: &[NodeId], i: usize| -> NodeId {
                    match a {
                        ArrayBinArg::Float(v) => g.constant(*v),
                        ArrayBinArg::Tracer(t) => t.node_id,
                        ArrayBinArg::TracerArray(_) | ArrayBinArg::NdArray(_) => {
                            nodes[if nodes.len() == 1 { 0 } else { i }]
                        }
                    }
                };
                // Pre-materialize vector operands (outside the emit borrow).
                let mat = |a: &ArrayBinArg| -> Vec<NodeId> {
                    match a {
                        ArrayBinArg::TracerArray(ta) => ta.materialize(),
                        ArrayBinArg::NdArray(vs) => self.graph.with(|g| {
                            vs.iter().map(|&v| g.constant(v)).collect()
                        }),
                        _ => Vec::new(),
                    }
                };
                let cond_nodes: Vec<Vec<NodeId>> = conds.iter().map(&mat).collect();
                let choice_nodes: Vec<Vec<NodeId>> = choices.iter().map(&mat).collect();
                let default_nodes = mat(&default);
                let out: Vec<NodeId> = self.graph.with(|g| {
                    (0..out_n).map(|i| {
                        // Right fold: later conditions are the fallback.
                        let mut acc = node_at(g, &default, &default_nodes, i);
                        for k in (0..conds.len()).rev() {
                            let c = node_at(g, &conds[k], &cond_nodes[k], i);
                            let v = node_at(g, &choices[k], &choice_nodes[k], i);
                            acc = g.select(c, v, acc);
                        }
                        acc
                    }).collect()
                });
                if out_n == 1 && size_of(&default).is_none()
                    && conds.iter().chain(choices.iter()).all(|a| size_of(a).is_none())
                {
                    return Ok(Py::new(py, JitTracer::new(&self.graph, out[0]))?.into_any().into());
                }
                Ok(Py::new(py, JitTracerArray::from_nodes(&self.graph, out))?.into_any().into())
            }
            // np.searchsorted — shared with the scalar tracer's handler.
            "searchsorted" => np_searchsorted_dispatch(py, &self.graph, args, kwargs),
            "array" => {
                if args.len() == 1 {
                    let arg0 = args.get_item(0)?;
                    if let Ok(ta) = arg0.extract::<JitTracerArray>() {
                        return Ok(Py::new(py, ta)?.into_any().into());
                    }
                }
                Ok(py.NotImplemented().into())
            }
            // np.clip(x, lo, hi)
            "clip" => {
                if args.len() < 3 { return Ok(py.NotImplemented().into()); }
                let x_arg = args.get_item(0)?;
                if let Ok(tracer) = x_arg.extract::<JitTracer>() {
                    let lo = tracer.ensure_id(&args.get_item(1)?.extract::<TracerOrFloat>()?);
                    let hi = tracer.ensure_id(&args.get_item(2)?.extract::<TracerOrFloat>()?);
                    let clamped_lo = tracer.graph.binary(BinOp::Max, tracer.node_id, lo);
                    let clamped = tracer.graph.binary(BinOp::Min, clamped_lo, hi);
                    return Ok(Py::new(py, JitTracer::new(&tracer.graph, clamped))?.into_any().into());
                }
                if let Ok(ta) = x_arg.extract::<JitTracerArray>() {
                    // Accept symbolic (tracer) bounds like the scalar path; `None`
                    // keeps the unbounded ±inf default.
                    let resolve = |b: Option<TracerOrFloat>, unbounded: f64| match b {
                        Some(TracerOrFloat::Tracer(t)) => t.node_id,
                        Some(TracerOrFloat::Float(v)) => ta.graph.constant(v),
                        None => ta.graph.constant(unbounded),
                    };
                    let lo_id = resolve(args.get_item(1)?.extract()?, f64::NEG_INFINITY);
                    let hi_id = resolve(args.get_item(2)?.extract()?, f64::INFINITY);
                    let ids = ta.materialize();
                    let nodes: Vec<NodeId> = ta.graph.with(|g| {
                        ids.iter().map(|&v| {
                            let clamped_lo = g.binary(BinOp::Max, v, lo_id);
                            g.binary(BinOp::Min, clamped_lo, hi_id)
                        }).collect()
                    });
                    // Keep the input's N-D shape (the method form does too).
                    return Ok(Py::new(py, JitTracerArray::from_nodes_shape(
                        &ta.graph, nodes, ta.shape.clone()))?.into_any().into());
                }
                Ok(py.NotImplemented().into())
            }
            // np.where(cond, x, y) — shared dispatch handles traced/constant
            // conditions (scalar or vector) and broadcasting branches.
            "where" => np_where_dispatch(py, &self.graph, args),
            // np.linalg.norm(x, ord=...) — supports ord=2 (default L2), ord=1, ord=inf, ord=-inf, ord='fro'
            "norm" if module.contains("linalg") => {
                if args.is_empty() { return Ok(py.NotImplemented().into()); }
                let ta = match args.get_item(0)?.extract::<JitTracerArray>() {
                    Ok(ta) => ta,
                    Err(_) => return Ok(py.NotImplemented().into()),
                };

                // Parse ord: positional arg1 or kwarg "ord". Default 2 (Euclidean).
                let ord_obj = if args.len() >= 2 {
                    Some(args.get_item(1)?)
                } else if let Some(kw) = &kwargs {
                    kw.get_item("ord").ok().flatten()
                } else { None };

                // Map ord to a tag: 1, 2, posinf, neginf, fro
                enum Ord { L1, L2, Pinf, Ninf, Fro }
                let mut ord = Ord::L2;
                if let Some(o) = ord_obj {
                    if let Ok(s) = o.extract::<String>() {
                        match s.as_str() {
                            "fro" => ord = Ord::Fro,
                            other => return Err(PyTypeError::new_err(format!(
                                "np.linalg.norm: ord='{}' not supported in tracer", other))),
                        }
                    } else if let Ok(f) = o.extract::<f64>() {
                        if f == 1.0 { ord = Ord::L1; }
                        else if f == 2.0 { ord = Ord::L2; }
                        else if f == f64::INFINITY { ord = Ord::Pinf; }
                        else if f == f64::NEG_INFINITY { ord = Ord::Ninf; }
                        else { return Err(PyTypeError::new_err(format!(
                            "np.linalg.norm: ord={} not supported in tracer (use 1, 2, inf, -inf, or 'fro')", f))); }
                    } else {
                        return Err(PyTypeError::new_err("np.linalg.norm: ord must be a number or 'fro'"));
                    }
                }

                // axis is not supported — raise rather than silently assuming
                if let Some(kw) = &kwargs {
                    if kw.contains("axis")? && kw.get_item("axis")?.is_some_and(|v| !v.is_none()) {
                        return Err(PyTypeError::new_err("np.linalg.norm: axis argument not supported in tracer"));
                    }
                }

                // All four orders lower to ONE structured node (Dot / Reduce)
                // over the elements — the tape's fused 4-lane kernels — rather
                // than an O(N) Binary chain.
                let xs: Vec<NodeId> = (0..ta.size).map(|i| ta.get_node(i)).collect();
                let abs_of = |xs: &[NodeId]| -> Vec<NodeId> {
                    xs.iter().map(|&x| ta.graph.unary(UnaryOp::Abs, x)).collect()
                };
                let result = match ord {
                    Ord::L2 | Ord::Fro => {
                        // sqrt(x . x)
                        let sum_sq = node_dot(&ta.graph, xs.clone(), xs);
                        ta.graph.unary(UnaryOp::Sqrt, sum_sq)
                    }
                    Ord::L1 => fold_nodes(&ta.graph, BinOp::Add, 0.0, abs_of(&xs)),
                    Ord::Pinf => fold_nodes(&ta.graph, BinOp::Max, f64::NEG_INFINITY, abs_of(&xs)),
                    Ord::Ninf => fold_nodes(&ta.graph, BinOp::Min, f64::INFINITY, abs_of(&xs)),
                };
                Ok(Py::new(py, JitTracer::new(&ta.graph, result))?.into_any().into())
            }
            // np.cross(a, b) for 3D vectors → [a1*b2-a2*b1, a2*b0-a0*b2, a0*b1-a1*b0]
            "cross" => {
                if args.len() != 2 { return Ok(py.NotImplemented().into()); }
                let a = args.get_item(0)?.extract::<JitTracerArray>().ok().or_else(|| {
                    let vals: Vec<f64> = args.get_item(0).ok()?.extract().ok()?;
                    let nodes: Vec<NodeId> = vals.iter().map(|&v| self.graph.constant(v)).collect();
                    Some(JitTracerArray::from_nodes(&self.graph, nodes))
                });
                let b = args.get_item(1)?.extract::<JitTracerArray>().ok().or_else(|| {
                    let vals: Vec<f64> = args.get_item(1).ok()?.extract().ok()?;
                    let nodes: Vec<NodeId> = vals.iter().map(|&v| self.graph.constant(v)).collect();
                    Some(JitTracerArray::from_nodes(&self.graph, nodes))
                });
                if let (Some(a), Some(b)) = (&a, &b) {
                    if a.size == 3 && b.size == 3 {
                        // c[k] = a[k+1] * b[k+2] - a[k+2] * b[k+1]  (indices mod 3)
                        let mut nodes = Vec::with_capacity(3);
                        for k in 0..3usize {
                            let i = (k + 1) % 3;
                            let j = (k + 2) % 3;
                            let p1 = a.graph.binary(BinOp::Mul, a.get_node(i), b.get_node(j));
                            let p2 = a.graph.binary(BinOp::Mul, a.get_node(j), b.get_node(i));
                            nodes.push(a.graph.binary(BinOp::Sub, p1, p2));
                        }
                        return Ok(Py::new(py, JitTracerArray::from_nodes(&a.graph, nodes))?.into_any().into());
                    }
                    return Err(PyTypeError::new_err(format!(
                        "np.cross: tracer only supports 3D vectors (got sizes {} and {})",
                        a.size, b.size)));
                }
                Ok(py.NotImplemented().into())
            }
            // np.vdot(a, b) — same as dot but conjugate-free (for real: identical to dot)
            "vdot" | "inner" => {
                if args.len() != 2 { return Ok(py.NotImplemented().into()); }
                let a = args.get_item(0)?.extract::<JitTracerArray>().ok().or_else(|| {
                    let vals: Vec<f64> = args.get_item(0).ok()?.extract().ok()?;
                    let nodes: Vec<NodeId> = vals.iter().map(|&v| self.graph.constant(v)).collect();
                    Some(JitTracerArray::from_nodes(&self.graph, nodes))
                });
                let b = args.get_item(1)?.extract::<JitTracerArray>().ok().or_else(|| {
                    let vals: Vec<f64> = args.get_item(1).ok()?.extract().ok()?;
                    let nodes: Vec<NodeId> = vals.iter().map(|&v| self.graph.constant(v)).collect();
                    Some(JitTracerArray::from_nodes(&self.graph, nodes))
                });
                if let (Some(a), Some(b)) = (&a, &b) {
                    // One fused Dot node (4-lane tape kernel), not an Add chain.
                    // Mismatched sizes raise, matching eager numpy.
                    if a.size != b.size {
                        return Err(PyValueError::new_err(format!(
                            "np.{}: shapes ({},) and ({},) not aligned", name, a.size, b.size)));
                    }
                    let n = a.size;
                    let an: Vec<NodeId> = (0..n).map(|i| a.get_node(i)).collect();
                    let bn: Vec<NodeId> = (0..n).map(|i| b.get_node(i)).collect();
                    let result = JitTracer::new(&a.graph, node_dot(&a.graph, an, bn));
                    return Ok(Py::new(py, result)?.into_any().into());
                }
                Ok(py.NotImplemented().into())
            }
            // Array-building / concatenation on JitTracerArray inputs.
            //
            // Numpy semantics we implement here:
            //   np.stack       — new axis at `axis` (default 0); all inputs must share shape
            //   np.concatenate — join along `axis` (default 0); axes other than `axis` must match
            //   np.hstack/vstack/array/asarray — legacy flat-1D output (backwards compat
            //                    for existing fastsim callers that concatenate scalars/1D arrays)
            //
            // For 1-D inputs and default axis, `stack`/`concatenate` collapse to the
            // flat 1-D behavior the old code produced, so existing tests stay green.
            // ("array"/"asarray" with a single arg are handled by the earlier
            // "array" arm; only the multi-arg/stack forms reach here.)
            "stack" | "hstack" | "vstack" | "concatenate" | "asarray" => {
                if args.is_empty() { return Ok(py.NotImplemented().into()); }

                // Collect inputs into (shape, flat_nodes) pairs.
                let seq = args.get_item(0)?;
                let iter = match seq.try_iter() {
                    Ok(it) => it,
                    Err(_) => return Ok(py.NotImplemented().into()),
                };
                let mut items: Vec<(Vec<usize>, Vec<NodeId>)> = Vec::new();
                for item in iter {
                    let item = item?;
                    if let Ok(t) = item.extract::<JitTracer>() {
                        items.push((vec![], vec![t.node_id]));
                    } else if let Ok(ta) = item.extract::<JitTracerArray>() {
                        items.push((ta.shape.clone(), ta.materialize()));
                    } else if let Ok(v) = item.extract::<f64>() {
                        items.push((vec![], vec![self.graph.constant(v)]));
                    } else if let Ok(vals) = item.extract::<Vec<f64>>() {
                        let n = vals.len();
                        let nodes: Vec<NodeId> = vals.iter().map(|&v| self.graph.constant(v)).collect();
                        items.push((vec![n], nodes));
                    } else {
                        return Ok(py.NotImplemented().into());
                    }
                }
                if items.is_empty() {
                    return Ok(Py::new(py, JitTracerArray::from_nodes(&self.graph, vec![]))?.into_any().into());
                }

                // Legacy flat-1D path for array/asarray — keep shape for
                // asarray on a single array arg; otherwise flatten (the
                // scalar-assembly idiom `np.array([expr1, expr2])`).
                if matches!(name.as_str(), "array" | "asarray") {
                    if items.len() == 1 {
                        let (shape, nodes) = items.pop().unwrap();
                        let shape = if shape.is_empty() { vec![1] } else { shape };
                        return Ok(Py::new(py, JitTracerArray::from_nodes_shape(&self.graph, nodes, shape))?.into_any().into());
                    }
                    let mut flat: Vec<NodeId> = Vec::new();
                    for (_, nodes) in items { flat.extend(nodes); }
                    return Ok(Py::new(py, JitTracerArray::from_nodes(&self.graph, flat))?.into_any().into());
                }

                // stack / concatenate honor an explicit `axis` kwarg/pos.
                let axis = extract_axis(args, kwargs)?.unwrap_or(0);

                // Flat fast path: for scalar/1-D inputs, `concatenate` and
                // `hstack` are exactly the flat join (numpy: hstack of 1-D is
                // axis-0 concatenation).
                let all_1d_or_scalar = items.iter().all(|(s, _)| s.len() <= 1);
                if all_1d_or_scalar && matches!(name.as_str(), "concatenate" | "hstack") {
                    let mut flat: Vec<NodeId> = Vec::new();
                    for (_, nodes) in items { flat.extend(nodes); }
                    return Ok(Py::new(py, JitTracerArray::from_nodes(&self.graph, flat))?.into_any().into());
                }

                // N-D hstack = concatenation along axis 1; N-D vstack promotes
                // (n,) → (1,n) and scalars → (1,1), then concatenates along
                // axis 0 (numpy semantics — the former flat path produced the
                // wrong element order for 2-D hstack).
                if name == "hstack" {
                    let (out_shape, nodes) = concat_nd(&items, 1)?;
                    return Ok(Py::new(py, JitTracerArray::from_nodes_shape(
                        &self.graph, nodes, out_shape))?.into_any().into());
                }
                if name == "vstack" {
                    let items: Vec<(Vec<usize>, Vec<NodeId>)> = items.into_iter()
                        .map(|(s, n)| match s.len() {
                            0 => (vec![1, 1], n),
                            1 => { let w = s[0]; (vec![1, w], n) }
                            _ => (s, n),
                        }).collect();
                    let (out_shape, nodes) = concat_nd(&items, 0)?;
                    return Ok(Py::new(py, JitTracerArray::from_nodes_shape(
                        &self.graph, nodes, out_shape))?.into_any().into());
                }

                match name.as_str() {
                    "stack" => {
                        // All inputs must share shape; output has new axis of size n_inputs at `axis`.
                        let ref_shape = items[0].0.clone();
                        for (s, _) in &items {
                            if s != &ref_shape {
                                return Err(pyo3::exceptions::PyValueError::new_err(
                                    format!("np.stack: shape mismatch {:?} vs {:?}", ref_shape, s)
                                ));
                            }
                        }
                        let n_in = items.len();
                        let new_rank = ref_shape.len() + 1;
                        let ax = if axis < 0 { new_rank as isize + axis } else { axis } as usize;
                        if ax > ref_shape.len() {
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                format!("np.stack: axis {} out of bounds for rank {}", axis, new_rank)
                            ));
                        }
                        let mut out_shape = ref_shape.clone();
                        out_shape.insert(ax, n_in);
                        let out_size: usize = out_shape.iter().product();

                        // Build output by iterating the new axis outermost; remaining
                        // layout is preserved.  This is row-major over out_shape.
                        let ref_size: usize = ref_shape.iter().product::<usize>().max(1);
                        let out_strides = strides_row_major(&out_shape);
                        let mut nodes = vec![0u32; out_size];
                        let mut src_idx = vec![0usize; ref_shape.len()];
                        let src_strides = strides_row_major(&ref_shape);
                        for k in 0..n_in {
                            let src_nodes = &items[k].1;
                            for _ in 0..ref_size {
                                // ref multi-idx → src_flat
                                let mut src_flat = 0usize;
                                for d in 0..ref_shape.len() {
                                    src_flat += src_idx[d] * src_strides[d];
                                }
                                // Build output multi-idx: insert k at position ax
                                let mut dst_flat = 0usize;
                                let mut rd = 0usize;
                                for d in 0..new_rank {
                                    let idx_val = if d == ax { k } else { let v = src_idx[rd]; rd += 1; v };
                                    dst_flat += idx_val * out_strides[d];
                                }
                                nodes[dst_flat] = src_nodes[src_flat];
                                // row-major increment over ref_shape
                                for d in (0..ref_shape.len()).rev() {
                                    src_idx[d] += 1;
                                    if src_idx[d] < ref_shape[d] { break; }
                                    src_idx[d] = 0;
                                }
                            }
                        }
                        Ok(Py::new(py, JitTracerArray::from_nodes_shape(&self.graph, nodes, out_shape))?.into_any().into())
                    }
                    "concatenate" => {
                        let (out_shape, nodes) = concat_nd(&items, axis)?;
                        Ok(Py::new(py, JitTracerArray::from_nodes_shape(&self.graph, nodes, out_shape))?.into_any().into())
                    }
                    _ => unreachable!(),
                }
            }
            // Shape-metadata view ops (delegate to the method forms).
            "reshape" => {
                if args.is_empty() { return Ok(py.NotImplemented().into()); }
                if let Ok(ta) = args.get_item(0)?.extract::<JitTracerArray>() {
                    // np.reshape(x, newshape) — newshape is arg 1, may be int or tuple.
                    if args.len() < 2 { return Ok(py.NotImplemented().into()); }
                    let shape_arg = args.get_item(1)?;
                    let new_shape: Vec<i64> = if let Ok(v) = shape_arg.extract::<Vec<i64>>() {
                        v
                    } else if let Ok(s) = shape_arg.extract::<i64>() {
                        vec![s]
                    } else {
                        return Ok(py.NotImplemented().into());
                    };
                    return ta.reshape(py, new_shape);
                }
                Ok(py.NotImplemented().into())
            }
            "transpose" => {
                if args.is_empty() { return Ok(py.NotImplemented().into()); }
                if let Ok(ta) = args.get_item(0)?.extract::<JitTracerArray>() {
                    let axes: Option<Vec<isize>> = if args.len() >= 2 {
                        args.get_item(1)?.extract().ok()
                    } else { None };
                    return ta.transpose(py, axes);
                }
                Ok(py.NotImplemented().into())
            }
            "squeeze" => {
                if args.is_empty() { return Ok(py.NotImplemented().into()); }
                if let Ok(ta) = args.get_item(0)?.extract::<JitTracerArray>() {
                    let axis: Option<isize> = if args.len() >= 2 {
                        args.get_item(1)?.extract().ok()
                    } else if let Some(k) = kwargs {
                        match k.get_item("axis") {
                            Ok(Some(v)) => v.extract().ok(),
                            _ => None,
                        }
                    } else { None };
                    return ta.squeeze(py, axis);
                }
                Ok(py.NotImplemented().into())
            }
            "expand_dims" => {
                if args.len() < 2 { return Ok(py.NotImplemented().into()); }
                if let Ok(ta) = args.get_item(0)?.extract::<JitTracerArray>() {
                    let axis: isize = args.get_item(1)?.extract()?;
                    return ta.unsqueeze(py, axis);
                }
                Ok(py.NotImplemented().into())
            }
            // np.full_like(x, v) — x's shape, constant (or scalar-traced) fill.
            "full_like" => {
                if args.len() < 2 { return Ok(py.NotImplemented().into()); }
                let Ok(ta) = args.get_item(0)?.extract::<JitTracerArray>() else {
                    return Ok(py.NotImplemented().into());
                };
                let fv = args.get_item(1)?;
                let node = if let Ok(t) = fv.extract::<JitTracer>() { t.node_id }
                    else if let Ok(v) = fv.extract::<f64>() { ta.graph.constant(v) }
                    else { return Ok(py.NotImplemented().into()) };
                let nodes = vec![node; ta.size];
                Ok(Py::new(py, JitTracerArray::from_nodes_shape(
                    &ta.graph, nodes, ta.shape.clone()))?.into_any().into())
            }
            // np.flip — reverse along one axis, or all axes (default). Pure
            // node permutation, zero graph cost. Reversing ALL axes of a
            // row-major layout is exactly reversing the flat order.
            "flip" => {
                if args.is_empty() { return Ok(py.NotImplemented().into()); }
                let Ok(ta) = args.get_item(0)?.extract::<JitTracerArray>() else {
                    return Ok(py.NotImplemented().into());
                };
                let axis = extract_axis(args, kwargs)?;
                let nodes = ta.materialize();
                let out: Vec<NodeId> = match axis {
                    None => nodes.into_iter().rev().collect(),
                    Some(a) => {
                        let n = ta.shape.len();
                        let ax = if a < 0 { n as isize + a } else { a };
                        if ax < 0 || ax >= n as isize {
                            return Err(PyValueError::new_err(format!(
                                "np.flip: axis {} out of bounds for {}-D array", a, n)));
                        }
                        let ax = ax as usize;
                        let strides = strides_row_major(&ta.shape);
                        let (st, sh) = (strides[ax], ta.shape[ax]);
                        (0..ta.size).map(|flat| {
                            let i_ax = (flat / st) % sh;
                            nodes[flat - i_ax * st + (sh - 1 - i_ax) * st]
                        }).collect()
                    }
                };
                Ok(Py::new(py, JitTracerArray::from_nodes_shape(
                    &ta.graph, out, ta.shape.clone()))?.into_any().into())
            }
            // np.roll — cyclic shift over the flattened layout (axis=None) or
            // along one axis. Pure node permutation.
            "roll" => {
                if args.len() < 2 { return Ok(py.NotImplemented().into()); }
                let Ok(ta) = args.get_item(0)?.extract::<JitTracerArray>() else {
                    return Ok(py.NotImplemented().into());
                };
                let Ok(shift) = args.get_item(1)?.extract::<i64>() else {
                    return Ok(py.NotImplemented().into());  // tuple shifts unsupported
                };
                let axis = if args.len() >= 3 {
                    let a = args.get_item(2)?;
                    if a.is_none() { None } else { Some(a.extract::<isize>()?) }
                } else if let Some(k) = kwargs {
                    match k.get_item("axis") {
                        Ok(Some(v)) if !v.is_none() => Some(v.extract::<isize>()?),
                        _ => None,
                    }
                } else { None };
                let nodes = ta.materialize();
                let out: Vec<NodeId> = match axis {
                    None => {
                        let n = ta.size as i64;
                        (0..ta.size).map(|i| {
                            let src = (i as i64 - shift).rem_euclid(n) as usize;
                            nodes[src]
                        }).collect()
                    }
                    Some(a) => {
                        let nd = ta.shape.len();
                        let ax = if a < 0 { nd as isize + a } else { a };
                        if ax < 0 || ax >= nd as isize {
                            return Err(PyValueError::new_err(format!(
                                "np.roll: axis {} out of bounds for {}-D array", a, nd)));
                        }
                        let ax = ax as usize;
                        let strides = strides_row_major(&ta.shape);
                        let (st, sh) = (strides[ax], ta.shape[ax] as i64);
                        (0..ta.size).map(|flat| {
                            let i_ax = (flat / st) % ta.shape[ax];
                            let src_ax = (i_ax as i64 - shift).rem_euclid(sh) as usize;
                            nodes[flat - i_ax * st + src_ax * st]
                        }).collect()
                    }
                };
                Ok(Py::new(py, JitTracerArray::from_nodes_shape(
                    &ta.graph, out, ta.shape.clone()))?.into_any().into())
            }
            // np.cumprod — 1-D running product (mirrors the cumsum arm).
            "cumprod" => {
                if self.shape.len() > 1 || self.size == 0 { return Ok(py.NotImplemented().into()); }
                if args.len() > 1 || !kwargs_subset(kwargs, &[]) {
                    return Ok(py.NotImplemented().into());
                }
                let nodes = cumulative(self, BinOp::Mul);
                Ok(Py::new(py, JitTracerArray::from_nodes(&self.graph, nodes))?.into_any().into())
            }
            // np.outer(a, b) — (m, n) products; numpy flattens its inputs.
            "outer" => {
                if args.len() != 2 { return Ok(py.NotImplemented().into()); }
                let to_nodes = |arg: &Bound<'_, PyAny>| -> Option<Vec<NodeId>> {
                    if let Ok(ta) = arg.extract::<JitTracerArray>() {
                        Some(ta.materialize())
                    } else if let Ok(t) = arg.extract::<JitTracer>() {
                        Some(vec![t.node_id])
                    } else if let Ok(vs) = arg.extract::<Vec<f64>>() {
                        Some(vs.iter().map(|&v| self.graph.constant(v)).collect())
                    } else { None }
                };
                let (Some(a), Some(b)) = (to_nodes(&args.get_item(0)?), to_nodes(&args.get_item(1)?)) else {
                    return Ok(py.NotImplemented().into());
                };
                let (m, n) = (a.len(), b.len());
                let mut nodes = Vec::with_capacity(m * n);
                for &ai in &a {
                    for &bj in &b {
                        nodes.push(self.graph.binary(BinOp::Mul, ai, bj));
                    }
                }
                Ok(Py::new(py, JitTracerArray::from_nodes_shape(
                    &self.graph, nodes, vec![m, n]))?.into_any().into())
            }
            // np.atleast_1d on an array tracer is the identity.
            "atleast_1d" => {
                if args.len() != 1 { return Ok(py.NotImplemented().into()); }
                if let Ok(ta) = args.get_item(0)?.extract::<JitTracerArray>() {
                    return Ok(Py::new(py, ta)?.into_any().into());
                }
                Ok(py.NotImplemented().into())
            }
            // np.interp over a constant ascending grid — elementwise select
            // chain per node (see `super::emit_interp`). Extra kwargs
            // (left/right/period) are unsupported.
            "interp" if kwargs.is_none_or(|k| k.is_empty()) => {
                let Some((xp, fp)) = super::extract_interp_grids(args) else {
                    return Ok(py.NotImplemented().into());
                };
                let Ok(ta) = args.get_item(0)?.extract::<JitTracerArray>() else {
                    return Ok(py.NotImplemented().into());
                };
                let ids = ta.materialize();
                let nodes: Vec<NodeId> = ta.graph.with(|g| {
                    ids.iter().map(|&x| super::emit_interp_g(g, x, &xp, &fp)).collect()
                });
                Ok(Py::new(py, JitTracerArray::from_nodes_shape(
                    &ta.graph, nodes, ta.shape.clone()))?.into_any().into())
            }
            _ => Ok(py.NotImplemented().into()),
        }
    }

    fn __repr__(&self) -> String {
        if self.shape.len() == 1 {
            format!("JitTracerArray(size={})", self.size)
        } else {
            format!("JitTracerArray(shape={:?})", self.shape)
        }
    }
}

#[pyclass(unsendable)]
struct JitTracerArrayIter {
    array: JitTracerArray,
    index: usize,
}

#[pymethods]
impl JitTracerArrayIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }
    fn __next__(&mut self) -> Option<JitTracer> {
        if self.index >= self.array.size { return None; }
        let id = self.array.get_node(self.index);
        self.index += 1;
        Some(JitTracer::new(&self.array.graph, id))
    }
}
