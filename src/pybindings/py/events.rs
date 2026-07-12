// Python wrappers for event types (ZeroCrossing, Schedule, Condition) and Diagnostics.

use std::collections::HashMap;
use std::rc::Rc;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::simulation::SimEventRef;
use crate::utils::fastcell::FastCell;

use super::helpers::{extract_scalar_f64, on_callback_err, report_callback_error};

// ======================================================================================
// Diagnostics — Python wrapper matching pathsim Diagnostics dataclass
// ======================================================================================

#[pyclass(name = "Diagnostics", unsendable)]
pub struct PyDiagnostics {
    inner: crate::utils::diagnostics::Diagnostics,
    block_labels: Vec<String>,
}

/// Construct a PyDiagnostics wrapper from core diagnostics + cached block labels.
pub(super) fn py_diagnostics(
    inner: crate::utils::diagnostics::Diagnostics,
    block_labels: Vec<String>,
) -> PyDiagnostics {
    PyDiagnostics { inner, block_labels }
}

#[pymethods]
impl PyDiagnostics {
    #[getter]
    fn time(&self) -> f64 { self.inner.time }

    #[getter]
    fn loop_residuals(&self) -> HashMap<usize, f64> { self.inner.loop_residuals.clone() }

    #[getter]
    fn loop_iterations(&self) -> usize { self.inner.loop_iterations }

    #[getter]
    fn solve_residuals(&self) -> HashMap<usize, f64> { self.inner.solve_residuals.clone() }

    #[getter]
    fn solve_iterations(&self) -> usize { self.inner.solve_iterations }

    #[getter]
    fn step_errors(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = pyo3::types::PyDict::new(py);
        for (&id, &(suc, err, scl)) in &self.inner.step_errors {
            let scl_val: Py<PyAny> = scl.map_or_else(|| py.None(), |s| s.into_pyobject(py).unwrap().unbind().into());
            dict.set_item(id, (suc, err, scl_val))?;
        }
        Ok(dict.into())
    }

    /// `(label, residual)` of the block with the worst solver residual, if any.
    fn worst_block(&self) -> Option<(String, f64)> {
        self.inner.worst_block().map(|(id, err)| {
            let label = self.block_labels.get(id).cloned().unwrap_or_else(|| format!("Block_{}", id));
            (label, err)
        })
    }

    /// `(label, residual)` of the algebraic-loop booster with the worst residual, if any.
    fn worst_booster(&self) -> Option<(String, f64)> {
        self.inner.worst_booster().map(|(id, err)| (format!("Booster_{}", id), err))
    }

    /// Human-readable one-step diagnostics summary (loop/solve iterations, residuals).
    fn summary(&self) -> String {
        let labels = self.block_labels.clone();
        self.inner.summary(&move |id| labels.get(id).cloned().unwrap_or_else(|| format!("Block_{}", id)))
    }

    fn __repr__(&self) -> String { self.summary() }
    fn __str__(&self) -> String { self.summary() }
}

// ======================================================================================
// Event classes — Python wrappers for Rust event types
// ======================================================================================

/// Event that triggers when an event function crosses zero in either
/// direction. The exact crossing time is located by root-finding and the
/// attached action callback is invoked there. Drop-in compatible with
/// pathsim.events.ZeroCrossing.
#[pyclass(name = "ZeroCrossing", unsendable, subclass)]
pub struct PyZeroCrossing { pub(super) inner: SimEventRef }

#[pymethods]
impl PyZeroCrossing {
    #[new]
    #[pyo3(signature = (func_evt, func_act=None, tolerance=1e-4))]
    fn new(func_evt: Py<PyAny>, func_act: Option<Py<PyAny>>, tolerance: f64) -> Self {
        let evt_fn = move |t: f64| -> f64 {
            Python::attach(|py| {
                match func_evt.call1(py, (t,)) {
                    Ok(r) => extract_scalar_f64(py, &r).unwrap_or(0.0),
                    Err(e) => on_callback_err(py, e, 0.0),
                }
            })
        };
        let act_fn: Option<Box<dyn FnMut(f64)>> = func_act.map(|f| {
            Box::new(move |t: f64| {
                Python::attach(|py| { if let Err(e) = f.call1(py, (t,)) { report_callback_error(py, e); } });
            }) as Box<dyn FnMut(f64)>
        });
        let evt = crate::events::zerocrossing::ZeroCrossing::new(evt_fn, act_fn, tolerance);
        Self { inner: Rc::new(FastCell::new(evt)) }
    }

    fn __len__(&self) -> usize { self.inner.borrow().len() }
    fn __bool__(&self) -> bool { self.inner.borrow().is_active() }
    /// Activate the event (detected and resolved again).
    fn on(&self) { self.inner.borrow_mut().on(); }
    /// Deactivate the event (no detection until `on`).
    fn off(&self) { self.inner.borrow_mut().off(); }
    /// Reset the event to its initial scheduling/detection state.
    fn reset(&self) { self.inner.borrow_mut().reset(); }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
        let times: Vec<f64> = slf.inner.borrow().times().to_vec();
        let py = slf.py();
        let list = times.into_pyobject(py)?;
        Ok(list.call_method0("__iter__")?.unbind())
    }
}

/// Event that triggers only on an upward (negative-to-positive) zero
/// crossing of the event function. Drop-in compatible with
/// pathsim.events.ZeroCrossingUp.
#[pyclass(name = "ZeroCrossingUp", unsendable, subclass)]
pub struct PyZeroCrossingUp { pub(super) inner: SimEventRef }

#[pymethods]
impl PyZeroCrossingUp {
    #[new]
    #[pyo3(signature = (func_evt, func_act=None, tolerance=1e-4))]
    fn new(func_evt: Py<PyAny>, func_act: Option<Py<PyAny>>, tolerance: f64) -> Self {
        let evt_fn = move |t: f64| -> f64 {
            Python::attach(|py| { match func_evt.call1(py, (t,)) { Ok(r) => extract_scalar_f64(py, &r).unwrap_or(0.0), Err(e) => on_callback_err(py, e, 0.0) } })
        };
        let act_fn: Option<Box<dyn FnMut(f64)>> = func_act.map(|f| {
            Box::new(move |t: f64| { Python::attach(|py| { if let Err(e) = f.call1(py, (t,)) { report_callback_error(py, e); } }); }) as Box<dyn FnMut(f64)>
        });
        Self { inner: Rc::new(FastCell::new(crate::events::zerocrossing::ZeroCrossing::new_up(evt_fn, act_fn, tolerance))) }
    }
    fn __len__(&self) -> usize { self.inner.borrow().len() }
    fn __bool__(&self) -> bool { self.inner.borrow().is_active() }
    /// Activate the event (detected and resolved again).
    fn on(&self) { self.inner.borrow_mut().on(); }
    /// Deactivate the event (no detection until `on`).
    fn off(&self) { self.inner.borrow_mut().off(); }
    /// Reset the event to its initial scheduling/detection state.
    fn reset(&self) { self.inner.borrow_mut().reset(); }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
        let times: Vec<f64> = slf.inner.borrow().times().to_vec();
        let py = slf.py();
        let list = times.into_pyobject(py)?;
        Ok(list.call_method0("__iter__")?.unbind())
    }
}

/// Event that triggers only on a downward (positive-to-negative) zero
/// crossing of the event function. Drop-in compatible with
/// pathsim.events.ZeroCrossingDown.
#[pyclass(name = "ZeroCrossingDown", unsendable, subclass)]
pub struct PyZeroCrossingDown { pub(super) inner: SimEventRef }

#[pymethods]
impl PyZeroCrossingDown {
    #[new]
    #[pyo3(signature = (func_evt, func_act=None, tolerance=1e-4))]
    fn new(func_evt: Py<PyAny>, func_act: Option<Py<PyAny>>, tolerance: f64) -> Self {
        let evt_fn = move |t: f64| -> f64 {
            Python::attach(|py| { match func_evt.call1(py, (t,)) { Ok(r) => extract_scalar_f64(py, &r).unwrap_or(0.0), Err(e) => on_callback_err(py, e, 0.0) } })
        };
        let act_fn: Option<Box<dyn FnMut(f64)>> = func_act.map(|f| {
            Box::new(move |t: f64| { Python::attach(|py| { if let Err(e) = f.call1(py, (t,)) { report_callback_error(py, e); } }); }) as Box<dyn FnMut(f64)>
        });
        Self { inner: Rc::new(FastCell::new(crate::events::zerocrossing::ZeroCrossing::new_down(evt_fn, act_fn, tolerance))) }
    }
    fn __len__(&self) -> usize { self.inner.borrow().len() }
    fn __bool__(&self) -> bool { self.inner.borrow().is_active() }
    /// Activate the event (detected and resolved again).
    fn on(&self) { self.inner.borrow_mut().on(); }
    /// Deactivate the event (no detection until `on`).
    fn off(&self) { self.inner.borrow_mut().off(); }
    /// Reset the event to its initial scheduling/detection state.
    fn reset(&self) { self.inner.borrow_mut().reset(); }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
        let times: Vec<f64> = slf.inner.borrow().times().to_vec();
        let py = slf.py();
        let list = times.into_pyobject(py)?;
        Ok(list.call_method0("__iter__")?.unbind())
    }
}

/// Event that triggers on a periodic time schedule (a fixed start time and
/// repeating interval), invoking its action callback at each occurrence.
/// Drop-in compatible with pathsim.events.Schedule.
#[pyclass(name = "Schedule", unsendable, subclass)]
pub struct PySchedule { pub(super) inner: SimEventRef }

#[pymethods]
impl PySchedule {
    #[new]
    #[pyo3(signature = (t_start=0.0, t_end=None, t_period=1.0, func_act=None, tolerance=1e-16))]
    fn new(t_start: f64, t_end: Option<f64>, t_period: f64, func_act: Option<Py<PyAny>>, tolerance: f64) -> Self {
        let act_fn: Option<Box<dyn FnMut(f64)>> = func_act.map(|f| {
            Box::new(move |t: f64| { Python::attach(|py| { if let Err(e) = f.call1(py, (t,)) { report_callback_error(py, e); } }); }) as Box<dyn FnMut(f64)>
        });
        let evt = crate::events::schedule::Schedule::new(t_start, t_end, t_period, act_fn, tolerance);
        Self { inner: Rc::new(FastCell::new(evt)) }
    }
    fn __len__(&self) -> usize { self.inner.borrow().len() }
    fn __bool__(&self) -> bool { self.inner.borrow().is_active() }
    /// Activate the event (detected and resolved again).
    fn on(&self) { self.inner.borrow_mut().on(); }
    /// Deactivate the event (no detection until `on`).
    fn off(&self) { self.inner.borrow_mut().off(); }
    /// Reset the event to its initial scheduling/detection state.
    fn reset(&self) { self.inner.borrow_mut().reset(); }
}

/// Event that triggers at an explicit list of scheduled times, invoking its
/// action callback at each. Drop-in compatible with
/// pathsim.events.ScheduleList.
#[pyclass(name = "ScheduleList", unsendable, subclass)]
pub struct PyScheduleList { pub(super) inner: SimEventRef }

#[pymethods]
impl PyScheduleList {
    #[new]
    #[pyo3(signature = (times_evt, func_act=None, tolerance=1e-16))]
    fn new(times_evt: Vec<f64>, func_act: Option<Py<PyAny>>, tolerance: f64) -> Self {
        let act_fn: Option<Box<dyn FnMut(f64)>> = func_act.map(|f| {
            Box::new(move |t: f64| { Python::attach(|py| { if let Err(e) = f.call1(py, (t,)) { report_callback_error(py, e); } }); }) as Box<dyn FnMut(f64)>
        });
        let evt = crate::events::schedule::ScheduleList::new(times_evt, act_fn, tolerance);
        Self { inner: Rc::new(FastCell::new(evt)) }
    }
    fn __len__(&self) -> usize { self.inner.borrow().len() }
    fn __bool__(&self) -> bool { self.inner.borrow().is_active() }
}

/// Event that triggers when a user-supplied boolean condition becomes true,
/// invoking its action callback. Useful for state-dependent logic that is not
/// a simple zero crossing. Drop-in compatible with pathsim.events.Condition.
#[pyclass(name = "Condition", unsendable, subclass)]
pub struct PyCondition { pub(super) inner: SimEventRef }

#[pymethods]
impl PyCondition {
    #[new]
    #[pyo3(signature = (func_evt, func_act=None, tolerance=1e-4))]
    fn new(func_evt: Py<PyAny>, func_act: Option<Py<PyAny>>, tolerance: f64) -> Self {
        let evt_fn = move |t: f64| -> bool {
            Python::attach(|py| { match func_evt.call1(py, (t,)) { Ok(r) => r.extract::<bool>(py).unwrap_or(false), Err(e) => on_callback_err(py, e, false) } })
        };
        let act_fn: Option<Box<dyn FnMut(f64)>> = func_act.map(|f| {
            Box::new(move |t: f64| { Python::attach(|py| { if let Err(e) = f.call1(py, (t,)) { report_callback_error(py, e); } }); }) as Box<dyn FnMut(f64)>
        });
        let evt = crate::events::condition::Condition::new(evt_fn, act_fn, tolerance);
        Self { inner: Rc::new(FastCell::new(evt)) }
    }
    fn __len__(&self) -> usize { self.inner.borrow().len() }
    fn __bool__(&self) -> bool { self.inner.borrow().is_active() }
}

/// Extract the inner `SimEventRef` from any of the 6 PyEvent wrapper types.
/// Centralizes the type-dispatch used by `add_event`/`remove_event`.
pub(super) fn extract_event_ref(event: &Bound<'_, PyAny>) -> PyResult<SimEventRef> {
    if let Ok(e) = event.extract::<PyRef<'_, PyZeroCrossing>>() {
        Ok(e.inner.clone())
    } else if let Ok(e) = event.extract::<PyRef<'_, PyZeroCrossingUp>>() {
        Ok(e.inner.clone())
    } else if let Ok(e) = event.extract::<PyRef<'_, PyZeroCrossingDown>>() {
        Ok(e.inner.clone())
    } else if let Ok(e) = event.extract::<PyRef<'_, PySchedule>>() {
        Ok(e.inner.clone())
    } else if let Ok(e) = event.extract::<PyRef<'_, PyScheduleList>>() {
        Ok(e.inner.clone())
    } else if let Ok(e) = event.extract::<PyRef<'_, PyCondition>>() {
        Ok(e.inner.clone())
    } else {
        Err(PyValueError::new_err("unknown event type"))
    }
}
