// Shared conversion/extraction helpers for PyO3 bindings.

use pyo3::prelude::*;
use pyo3::exceptions::{PyException, PyValueError};
use std::cell::RefCell;

// ======================================================================================
// Cooperative stop + callback-error propagation (pathsim StopSimulation parity)
// ======================================================================================

pyo3::create_exception!(
    _fastsim,
    StopSimulation,
    PyException,
    "Raised by a block or model to signal that the simulation should stop \
     immediately. The run loop catches it and terminates cleanly, as if \
     stop() had been called. Drop-in compatible with \
     pathsim.exceptions.StopSimulation."
);

thread_local! {
    /// First non-StopSimulation exception raised by a user callback during a
    /// run. Callbacks (`Box<dyn Fn>`) cannot propagate `Err` through the Rust
    /// closure boundary, so we stash it here and the run wrapper re-raises it
    /// once the loop has unwound.
    static PENDING_PYERR: RefCell<Option<PyErr>> = const { RefCell::new(None) };
}

fn store_pending_error(err: PyErr) {
    PENDING_PYERR.with(|c| {
        let mut slot = c.borrow_mut();
        // First cause is the most useful, and we stop the run immediately, so
        // later callback errors in the same step are dropped.
        if slot.is_none() {
            *slot = Some(err);
        }
    });
}

/// Take the pending user-callback error, if any. Called by the run wrappers.
pub(crate) fn take_pending_error() -> Option<PyErr> {
    PENDING_PYERR.with(|c| c.borrow_mut().take())
}

/// Clear any stale stop signal and pending error. Called at the start of a run.
pub(crate) fn reset_run_signals() {
    crate::simulation::clear_stop_requested();
    PENDING_PYERR.with(|c| *c.borrow_mut() = None);
}

/// Handle an exception raised by a user callback during a simulation step.
///
/// `StopSimulation` requests a clean cooperative stop (pathsim parity). Any
/// other exception is remembered (first one wins) for the run wrapper to
/// re-raise, and also stops the run so computation does not continue after a
/// fault. Previously these were silently swallowed.
pub(crate) fn report_callback_error(py: Python<'_>, err: PyErr) {
    if !err.is_instance_of::<StopSimulation>(py) {
        store_pending_error(err);
    }
    crate::simulation::request_stop();
}

/// Convenience for value-returning callbacks: report the error and return a
/// fallback so the current step can finish before the run stops.
pub(crate) fn on_callback_err<T>(py: Python<'_>, err: PyErr, fallback: T) -> T {
    report_callback_error(py, err);
    fallback
}

/// Emit a `UserWarning` that a constructor parameter is accepted for pathsim
/// API compatibility but not honored by the fastsim runtime.  Used where the
/// backend genuinely lacks the capability to wire the parameter through, so
/// silent divergence from pathsim is avoided (issue #30).
pub(super) fn warn_param_ignored(
    py: Python<'_>, ctor: &str, param: &str, detail: &str,
) -> PyResult<()> {
    let warnings = py.import("warnings")?;
    warnings.call_method1(
        "warn",
        (
            format!(
                "{ctor}({param}=...) is accepted for pathsim API compatibility but \
                 ignored by the fastsim runtime: {detail}"
            ),
            py.get_type::<pyo3::exceptions::PyUserWarning>(),
        ),
    )?;
    Ok(())
}

/// Emit a `fastsim.exceptions.FastSimConvergenceWarning` carrying `msg`
/// (issue #27). Falls back to a plain `UserWarning` if the Python hierarchy
/// cannot be imported, so the warning is never lost. Returns `Ok(())` even on
/// warning-machinery errors — warning is best-effort and must not fault a run.
pub(crate) fn warn_convergence(py: Python<'_>, msg: &str) -> PyResult<()> {
    let warnings = py.import("warnings")?;
    let category = py
        .import("fastsim.exceptions")
        .and_then(|m| m.getattr("FastSimConvergenceWarning"))
        .map(|c| c.unbind())
        .unwrap_or_else(|_| py.get_type::<pyo3::exceptions::PyUserWarning>().into_any().unbind());
    // stacklevel=2 points the warning at the caller's run() line, not this shim.
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("stacklevel", 2)?;
    warnings.call_method("warn", (msg, category.bind(py)), Some(&kwargs))?;
    Ok(())
}

/// Convert a Rust slice to a numpy array Py<PyAny>.
pub(super) fn to_numpy(py: Python<'_>, data: &[f64]) -> Py<PyAny> {
    match py.import("numpy") {
        Ok(np) => np.call_method1("array", (data.to_vec(),)).unwrap().unbind(),
        Err(_) => data.to_vec().into_pyobject(py).unwrap().unbind(),
    }
}

/// Extract a Vec<f64> from a Python return value (supports float, list, numpy array).
pub(crate) fn extract_vec_f64(py: Python<'_>, obj: &Py<PyAny>) -> Vec<f64> {
    if let Ok(v) = obj.extract::<Vec<f64>>(py) { return v; }
    if let Ok(f) = obj.extract::<f64>(py) { return vec![f]; }
    if let Ok(ll) = obj.extract::<Vec<Vec<f64>>>(py) {
        return ll.into_iter().flatten().collect();
    }
    if let Ok(flat) = obj.bind(py).call_method0("flatten")
        .and_then(|f| f.call_method0("tolist"))
        .and_then(|l| l.extract::<Vec<f64>>())
    {
        return flat;
    }
    if let Ok(bound) = obj.bind(py).call_method0("tolist") {
        if let Ok(v) = bound.extract::<Vec<f64>>() { return v; }
        if let Ok(f) = bound.extract::<f64>() { return vec![f]; }
        if let Ok(ll) = bound.extract::<Vec<Vec<f64>>>() {
            return ll.into_iter().flatten().collect();
        }
    }
    vec![0.0]
}

/// Extract a Vec<f64> from a user callback return and validate its length
/// against `expected` (issue #33). A wrong-length ODE/DAE return previously
/// slipped through and failed deep in the core with an illegible error; here it
/// fails at the boundary with a message naming both lengths. On mismatch the
/// error is routed through the callback-error machinery and a zero fallback of
/// the expected length is returned so the current step can unwind cleanly.
pub(crate) fn extract_vec_f64_checked(
    py: Python<'_>, obj: &Py<PyAny>, expected: usize,
) -> Vec<f64> {
    let v = extract_vec_f64(py, obj);
    if v.len() != expected {
        let err = PyValueError::new_err(format!(
            "ODE/DAE callback returned {} value(s) but the block has {} state \
             component(s); the returned derivative vector must match the state \
             dimension",
            v.len(), expected
        ));
        return on_callback_err(py, err, vec![0.0; expected]);
    }
    v
}

/// Extract an initial-value argument (scalar or list) with a sensible default.
pub fn extract_initial_value(val: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<f64>> {
    match val {
        None => Ok(vec![0.0]),
        Some(v) => {
            if let Ok(f) = v.extract::<f64>() { return Ok(vec![f]); }
            if let Ok(l) = v.extract::<Vec<f64>>() { return Ok(l); }
            Err(PyValueError::new_err("'initial_value' must be float or list of floats"))
        }
    }
}

/// Extract a scalar f64 from a Python object (handles float, int, 0-d numpy array, 1-element array).
pub(super) fn extract_scalar_f64(py: Python<'_>, obj: &Py<PyAny>) -> PyResult<f64> {
    if let Ok(v) = obj.extract::<f64>(py) { return Ok(v); }
    if let Ok(v) = obj.call_method0(py, "item").and_then(|r| r.extract::<f64>(py)) { return Ok(v); }
    if let Ok(v) = obj.call_method1(py, "__getitem__", (0,)).and_then(|r| r.extract::<f64>(py)) { return Ok(v); }
    Err(pyo3::exceptions::PyTypeError::new_err("Cannot convert to f64"))
}

/// Extract a matrix from Python (scalar, 1D list, or 2D list) as flat row-major Vec<f64>.
pub(super) fn extract_matrix(val: Option<&Bound<'_, PyAny>>, default: &[f64]) -> PyResult<Vec<f64>> {
    match val {
        None => Ok(default.to_vec()),
        Some(v) => {
            if let Ok(f) = v.extract::<f64>() { return Ok(vec![f]); }
            if let Ok(l) = v.extract::<Vec<f64>>() { return Ok(l); }
            if let Ok(ll) = v.extract::<Vec<Vec<f64>>>() {
                return Ok(ll.into_iter().flatten().collect());
            }
            if let Ok(flat) = v.call_method0("flatten").and_then(|f| f.call_method0("tolist")).and_then(|l| l.extract::<Vec<f64>>()) {
                return Ok(flat);
            }
            Err(PyValueError::new_err("matrix must be scalar, list, list of lists, or numpy array"))
        }
    }
}

// ======================================================================================
// JIT compilation helpers (analytical Jacobian attachment for traced blocks)
// ======================================================================================

/// Compile an analytical Jacobian from a Graph via symbolic AD.
/// Returns an InterpretedFn whose outputs are the n×n Jacobian entries (row-major).
pub fn compile_jacobian(graph: &crate::ssa::graph::Graph) -> Option<crate::ssa::tape::InterpretedFn> {
    let x_slot = graph.signature.slot("x")?;
    if x_slot.size == 0 { return None; }
    let mut jac_graph = crate::ssa::autodiff::jacobian_wrt_slot(graph, "x")?;
    crate::ssa::optimize::optimize(&mut jac_graph);
    Some(crate::ssa::tape::InterpretedFn::from_graph(jac_graph))
}

/// Attach a compiled Jacobian to a block as jac_dyn callback.
pub fn attach_jacobian(blk: &mut crate::blocks::block::Block, jac: crate::ssa::tape::InterpretedFn) {
    let n_jac = jac.n_out;
    let jac_rc = std::rc::Rc::new(jac);
    blk.jac_dyn = Some(Box::new(move |x, u, t, out| {
        out.resize(n_jac, 0.0);
        jac_rc.call_into(&[x, u, &[t]], out);
    }));
}
