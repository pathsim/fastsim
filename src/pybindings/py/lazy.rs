//! Lazy-rejit wrapper for JIT-compiled block callbacks.
//!
//! Block input shapes (notably `u.len()`) only become known once the Simulation
//! has resolved connections, but eager tracing at construction time locks the
//! signature to a placeholder (typically `u.size = 1`). That leads to silent
//! miscompiles when the user's callback does reductions across `u` such as
//! `betas @ u` or `sum(b*ui for b, ui in zip(betas, u))`.
//!
//! `LazyTraced` moves tracing behind a shape-keyed cache. Every call checks
//! the incoming slice lengths against the cached entry and transparently
//! re-traces through the stored Python callable if they differ. The fast path
//! stays a direct `InterpretedFn::call_into` — the check is two `usize`
//! comparisons.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::ssa::tape::InterpretedFn;
use crate::tracer::{trace_with_signature, TraceArg};


/// Describes one positional argument of the traced callable. The size of
/// `Array` args is inferred at runtime from the actual input slice.
pub(crate) enum SigArg {
    Array(&'static str),
    Scalar(&'static str),
}


struct CachedEntry {
    shape_key: Vec<usize>,
    compiled: Rc<InterpretedFn>,
    /// Jacobians compiled alongside the main graph — one per slot name
    /// listed in `LazyTraced::jacobian_slots`. Order matches the slot list.
    jacobians: Vec<Rc<InterpretedFn>>,
}


/// Wraps a Python callable with a shape-keyed cache of compiled graphs.
/// Retraces transparently on shape mismatch, falls back to direct Python
/// invocation if tracing fails.
pub(crate) struct LazyTraced {
    callable: Py<PyAny>,
    signature: Vec<SigArg>,
    /// Slot names to differentiate with respect to. For each entry the
    /// retrace derives ∂f/∂slot via AD and caches the compiled Jacobian.
    jacobian_slots: Vec<&'static str>,
    cache: RefCell<Option<CachedEntry>>,
    /// `Some(true)` once a trace has ever succeeded; `Some(false)` once a
    /// trace attempt has failed for the current shape (triggers Python
    /// fallback on subsequent evaluations at the same shape).
    last_trace_ok: Cell<Option<bool>>,
    last_shape: RefCell<Vec<usize>>,
}

impl LazyTraced {
    pub(crate) fn new(
        callable: Py<PyAny>,
        signature: Vec<SigArg>,
        jacobian_slot: Option<&'static str>,
    ) -> Rc<Self> {
        Self::new_multi_jac(
            callable,
            signature,
            jacobian_slot.map(|s| vec![s]).unwrap_or_default(),
        )
    }

    pub(crate) fn new_multi_jac(
        callable: Py<PyAny>,
        signature: Vec<SigArg>,
        jacobian_slots: Vec<&'static str>,
    ) -> Rc<Self> {
        Rc::new(Self {
            callable,
            signature,
            jacobian_slots,
            cache: RefCell::new(None),
            last_trace_ok: Cell::new(None),
            last_shape: RefCell::new(Vec::new()),
        })
    }

    /// Evaluate the traced callable with the given inputs. Retraces if the
    /// shape differs from the last cached entry. If tracing fails for the
    /// current shape, falls back to direct Python invocation.
    pub(crate) fn call_into(&self, inputs: &[&[f64]], out: &mut Vec<f64>) {
        // Fast path: cache hit, shapes match — no allocations, no Python.
        {
            let cache = self.cache.borrow();
            if let Some(entry) = cache.as_ref() {
                if shapes_match(&entry.shape_key, inputs) {
                    out.resize(entry.compiled.n_out, 0.0);
                    entry.compiled.call_into(inputs, out);
                    return;
                }
            }
        }
        self.slow_path(inputs, out);
    }

    /// Evaluate the `idx`-th cached Jacobian for the given inputs. `idx`
    /// indexes into the `jacobian_slots` list passed at construction time.
    /// No-op when the trace failed (the core block machinery then falls back
    /// to a finite-difference Jacobian).
    pub(crate) fn call_jacobian_into(&self, inputs: &[&[f64]], out: &mut Vec<f64>) {
        self.call_jacobian_idx_into(0, inputs, out);
    }

    pub(crate) fn call_jacobian_idx_into(&self, idx: usize, inputs: &[&[f64]], out: &mut Vec<f64>) {
        // Fast path: no shape-key allocation.
        {
            let cache = self.cache.borrow();
            if let Some(entry) = cache.as_ref() {
                if shapes_match(&entry.shape_key, inputs) {
                    if let Some(jac) = entry.jacobians.get(idx) {
                        out.resize(jac.n_out, 0.0);
                        jac.call_into(inputs, out);
                    }
                    return;
                }
            }
        }
        // Shape changed — retrace, then evaluate the fresh Jacobian.
        self.retrace_and_update(inputs);
        let cache = self.cache.borrow();
        if let Some(entry) = cache.as_ref() {
            if shapes_match(&entry.shape_key, inputs) {
                if let Some(jac) = entry.jacobians.get(idx) {
                    out.resize(jac.n_out, 0.0);
                    jac.call_into(inputs, out);
                }
            }
        }
    }

    /// Shape-miss path: retrace through Python if viable, then either call
    /// the freshly compiled graph or fall back to Python on failure.
    fn slow_path(&self, inputs: &[&[f64]], out: &mut Vec<f64>) {
        // Already-failed short-circuit: skip retracing if we just attempted
        // this exact shape without success.
        if self.last_trace_ok.get() == Some(false)
            && shapes_match(&self.last_shape.borrow(), inputs)
        {
            self.call_python_into(inputs, out);
            return;
        }
        self.retrace_and_update(inputs);
        let cache = self.cache.borrow();
        if let Some(entry) = cache.as_ref() {
            if shapes_match(&entry.shape_key, inputs) {
                out.resize(entry.compiled.n_out, 0.0);
                entry.compiled.call_into(inputs, out);
                return;
            }
        }
        drop(cache);
        self.call_python_into(inputs, out);
    }

    /// Allocate a shape key, retrace via Python, update the cache. Sole
    /// place in the wrapper that allocates `Vec<usize>` per call — the hot
    /// path never reaches here once a stable shape has been seen.
    fn retrace_and_update(&self, inputs: &[&[f64]]) {
        let key = self.shape_key(inputs);
        let result = Python::attach(|py| self.retrace(py, inputs, key.clone()));
        *self.last_shape.borrow_mut() = key;
        match result {
            Ok(entry) => {
                *self.cache.borrow_mut() = Some(entry);
                self.last_trace_ok.set(Some(true));
            }
            Err(_) => {
                self.last_trace_ok.set(Some(false));
            }
        }
    }

    /// Fallback path: build numpy-style args from the input slices and invoke
    /// the stored Python callable. Scalar args are passed as plain floats.
    fn call_python_into(&self, inputs: &[&[f64]], out: &mut Vec<f64>) {
        Python::attach(|py| {
            let np = py.import("numpy").ok();
            let mut py_args: Vec<Py<PyAny>> = Vec::with_capacity(inputs.len());
            for (sig, input) in self.signature.iter().zip(inputs) {
                let obj: Py<PyAny> = match sig {
                    SigArg::Scalar(_) => input.first().copied().unwrap_or(0.0)
                        .into_pyobject(py).unwrap().into_any().unbind(),
                    SigArg::Array(_) => {
                        if let Some(ref np) = np {
                            np.call_method1("array", (input.to_vec(),))
                                .unwrap().unbind()
                        } else {
                            input.to_vec().into_pyobject(py).unwrap().into_any().unbind()
                        }
                    }
                };
                py_args.push(obj);
            }
            let tuple = PyTuple::new(py, &py_args).unwrap();
            match self.callable.call1(py, tuple) {
                Ok(r) => {
                    *out = super::helpers::extract_vec_f64(py, &r);
                }
                Err(_) => {
                    // Leave out untouched — caller will propagate zeros.
                }
            }
        });
    }

    fn shape_key(&self, inputs: &[&[f64]]) -> Vec<usize> {
        inputs.iter().map(|s| s.len()).collect()
    }

    fn retrace(
        &self,
        py: Python<'_>,
        inputs: &[&[f64]],
        shape_key: Vec<usize>,
    ) -> PyResult<CachedEntry> {
        // Build the TraceArg list from the template plus the runtime shapes.
        let trace_args: Vec<TraceArg> = self.signature.iter().zip(inputs).map(|(sig, input)| {
            match sig {
                SigArg::Array(name) => TraceArg::Array { name, size: input.len() },
                SigArg::Scalar(name) => TraceArg::Scalar { name },
            }
        }).collect();

        let bound = self.callable.bind(py);
        let graph = match trace_with_signature(py, bound, &trace_args)? {
            Some(g) => g,
            None => return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "LazyTraced: re-trace produced no graph"
            )),
        };

        let compiled = Rc::new(InterpretedFn::from_graph(graph.clone()));

        let mut jacobians = Vec::with_capacity(self.jacobian_slots.len());
        for slot in &self.jacobian_slots {
            if let Some(mut g) = crate::ssa::autodiff::jacobian_wrt_slot(&graph, slot) {
                crate::ssa::optimize::optimize(&mut g);
                jacobians.push(Rc::new(InterpretedFn::from_graph(g)));
            }
        }

        Ok(CachedEntry { shape_key, compiled, jacobians })
    }
}

/// Cheap shape comparison that avoids allocating a `Vec<usize>` per call. The
/// hot path takes this branch on every step once a stable shape is cached.
#[inline]
fn shapes_match(key: &[usize], inputs: &[&[f64]]) -> bool {
    if key.len() != inputs.len() { return false; }
    for (k, slice) in key.iter().zip(inputs.iter()) {
        if *k != slice.len() { return false; }
    }
    true
}
