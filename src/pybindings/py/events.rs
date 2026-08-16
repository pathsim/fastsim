// Python wrappers for event types (ZeroCrossing, Schedule, Condition) and Diagnostics.

use std::collections::HashMap;
use std::rc::Rc;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::events::active::ActiveFlag;
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

/// Emit the `#[pymethods]` block of one event wrapper: its constructor, plus
/// the method surface every event shares (pathsim's `Event` base class).
///
/// The shared half lives here rather than in each type because it drifted —
/// `Condition` and `ScheduleList` ended up with neither `on()` nor `off()`, and
/// only the three `ZeroCrossing`s could be iterated. A macro cannot be invoked
/// *inside* a `#[pymethods]` block (pyo3 rejects it), so this generates the
/// whole block and takes the constructor as tokens.
macro_rules! event_pymethods {
    (
        $ty:ident,
        $( #[$ctor_attr:meta] )*
        fn new( $($arg:tt)* ) -> Self $ctor:block
        $( $extra:item )*
    ) => {
        #[pymethods]
        impl $ty {
            #[new]
            $( #[$ctor_attr] )*
            fn new($($arg)*) -> Self $ctor

            $( $extra )*

            fn __len__(&self) -> usize { self.inner.borrow().len() }
            fn __bool__(&self) -> bool { self.active.get() }

            /// Iterate the times at which this event was resolved.
            fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
                let times: Vec<f64> = slf.inner.borrow().times().to_vec();
                let py = slf.py();
                Ok(times.into_pyobject(py)?.call_method0("__iter__")?.unbind())
            }

            /// Activate the event (detected and resolved again).
            ///
            /// Safe to call from inside an action function, including the action of
            /// this very event: it sets the shared activation flag and never
            /// borrows the event body.
            fn on(&self) { self.active.set(true); }

            /// Deactivate the event (no detection until `on`). Callable from inside
            /// an action function, see `on`.
            fn off(&self) { self.active.set(false); }

            /// Reset recorded event times and detection history, and reactivate.
            fn reset(&self) { self.inner.borrow_mut().reset(); }

            /// Buffer the event-function evaluation taken before a timestep.
            fn buffer(&self, t: f64) { self.inner.borrow_mut().buffer(t); }

            /// Time until the next event, for the types that can predict one.
            fn estimate(&self, t: f64) -> Option<f64> { self.inner.borrow().estimate(t) }

            /// Evaluate the event function: `(detected, close, ratio)`.
            fn detect(&self, t: f64) -> (bool, bool, f64) { self.inner.borrow_mut().detect(t) }

            /// Record an event at `t` and run the action function.
            fn resolve(&self, t: f64) { self.inner.borrow_mut().resolve(t); }

            #[getter]
            fn tolerance(&self) -> f64 { self.inner.borrow().tolerance() }

            #[setter]
            fn set_tolerance(&self, tolerance: f64) { self.inner.borrow_mut().set_tolerance(tolerance); }
        }
    };
}

// ======================================================================================
// Event classes — Python wrappers for Rust event types
// ======================================================================================

/// Event that triggers when an event function crosses zero in either
/// direction. The exact crossing time is located by root-finding and the
/// attached action callback is invoked there. Drop-in compatible with
/// pathsim.events.ZeroCrossing.
#[pyclass(name = "ZeroCrossing", unsendable, subclass)]
pub struct PyZeroCrossing { pub(super) inner: SimEventRef, active: ActiveFlag }

event_pymethods! {
    PyZeroCrossing,
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
        let active = evt.active_flag();
        Self { inner: Rc::new(FastCell::new(evt)), active }
    }
}

/// Event that triggers only on an upward (negative-to-positive) zero
/// crossing of the event function. Drop-in compatible with
/// pathsim.events.ZeroCrossingUp.
#[pyclass(name = "ZeroCrossingUp", unsendable, subclass)]
pub struct PyZeroCrossingUp { pub(super) inner: SimEventRef, active: ActiveFlag }

event_pymethods! {
    PyZeroCrossingUp,
    #[pyo3(signature = (func_evt, func_act=None, tolerance=1e-4))]
    fn new(func_evt: Py<PyAny>, func_act: Option<Py<PyAny>>, tolerance: f64) -> Self {
        let evt_fn = move |t: f64| -> f64 {
            Python::attach(|py| { match func_evt.call1(py, (t,)) { Ok(r) => extract_scalar_f64(py, &r).unwrap_or(0.0), Err(e) => on_callback_err(py, e, 0.0) } })
        };
        let act_fn: Option<Box<dyn FnMut(f64)>> = func_act.map(|f| {
            Box::new(move |t: f64| { Python::attach(|py| { if let Err(e) = f.call1(py, (t,)) { report_callback_error(py, e); } }); }) as Box<dyn FnMut(f64)>
        });
        let evt = crate::events::zerocrossing::ZeroCrossing::new_up(evt_fn, act_fn, tolerance);
        let active = evt.active_flag();
        Self { inner: Rc::new(FastCell::new(evt)), active }
    }
}

/// Event that triggers only on a downward (positive-to-negative) zero
/// crossing of the event function. Drop-in compatible with
/// pathsim.events.ZeroCrossingDown.
#[pyclass(name = "ZeroCrossingDown", unsendable, subclass)]
pub struct PyZeroCrossingDown { pub(super) inner: SimEventRef, active: ActiveFlag }

event_pymethods! {
    PyZeroCrossingDown,
    #[pyo3(signature = (func_evt, func_act=None, tolerance=1e-4))]
    fn new(func_evt: Py<PyAny>, func_act: Option<Py<PyAny>>, tolerance: f64) -> Self {
        let evt_fn = move |t: f64| -> f64 {
            Python::attach(|py| { match func_evt.call1(py, (t,)) { Ok(r) => extract_scalar_f64(py, &r).unwrap_or(0.0), Err(e) => on_callback_err(py, e, 0.0) } })
        };
        let act_fn: Option<Box<dyn FnMut(f64)>> = func_act.map(|f| {
            Box::new(move |t: f64| { Python::attach(|py| { if let Err(e) = f.call1(py, (t,)) { report_callback_error(py, e); } }); }) as Box<dyn FnMut(f64)>
        });
        let evt = crate::events::zerocrossing::ZeroCrossing::new_down(evt_fn, act_fn, tolerance);
        let active = evt.active_flag();
        Self { inner: Rc::new(FastCell::new(evt)), active }
    }
}

/// Event that triggers on a periodic time schedule (a fixed start time and
/// repeating interval), invoking its action callback at each occurrence.
/// Drop-in compatible with pathsim.events.Schedule.
#[pyclass(name = "Schedule", unsendable, subclass)]
pub struct PySchedule {
    pub(super) inner: SimEventRef,
    active: ActiveFlag,
    /// The same event, typed, so the timing attributes below reach the fields
    /// the simulation actually reads — a copy kept alongside would let a
    /// setter appear to work while changing nothing.
    sched: Rc<FastCell<crate::events::schedule::Schedule>>,
}

event_pymethods! {
    PySchedule,
    #[pyo3(signature = (t_start=0.0, t_end=None, t_period=1.0, func_act=None, tolerance=1e-16))]
    fn new(t_start: f64, t_end: Option<f64>, t_period: f64, func_act: Option<Py<PyAny>>, tolerance: f64) -> Self {
        let act_fn: Option<Box<dyn FnMut(f64)>> = func_act.map(|f| {
            Box::new(move |t: f64| { Python::attach(|py| { if let Err(e) = f.call1(py, (t,)) { report_callback_error(py, e); } }); }) as Box<dyn FnMut(f64)>
        });
        let evt = crate::events::schedule::Schedule::new(t_start, t_end, t_period, act_fn, tolerance);
        let active = evt.active_flag();
        let sched = Rc::new(FastCell::new(evt));
        Self { inner: sched.clone(), active, sched }
    }

    #[getter]
    fn t_start(&self) -> f64 { self.sched.borrow().t_start }

    #[setter]
    fn set_t_start(&self, v: f64) { self.sched.borrow_mut().t_start = v; }

    #[getter]
    fn t_period(&self) -> f64 { self.sched.borrow().t_period }

    #[setter]
    fn set_t_period(&self, v: f64) { self.sched.borrow_mut().t_period = v; }

    #[getter]
    fn t_end(&self) -> Option<f64> { self.sched.borrow().t_end }

    #[setter]
    fn set_t_end(&self, v: Option<f64>) { self.sched.borrow_mut().t_end = v; }
}

/// Event that triggers at an explicit list of scheduled times, invoking its
/// action callback at each. Drop-in compatible with
/// pathsim.events.ScheduleList.
#[pyclass(name = "ScheduleList", unsendable, subclass)]
pub struct PyScheduleList {
    pub(super) inner: SimEventRef,
    active: ActiveFlag,
    sched: Rc<FastCell<crate::events::schedule::ScheduleList>>,
}

event_pymethods! {
    PyScheduleList,
    #[pyo3(signature = (times_evt, func_act=None, tolerance=1e-16))]
    fn new(times_evt: Vec<f64>, func_act: Option<Py<PyAny>>, tolerance: f64) -> Self {
        let act_fn: Option<Box<dyn FnMut(f64)>> = func_act.map(|f| {
            Box::new(move |t: f64| { Python::attach(|py| { if let Err(e) = f.call1(py, (t,)) { report_callback_error(py, e); } }); }) as Box<dyn FnMut(f64)>
        });
        let evt = crate::events::schedule::ScheduleList::new(times_evt, act_fn, tolerance);
        let active = evt.active_flag();
        let sched = Rc::new(FastCell::new(evt));
        Self { inner: sched.clone(), active, sched }
    }

    #[getter]
    fn times_evt(&self) -> Vec<f64> { self.sched.borrow().times_evt.clone() }

    /// Reassigning the schedule sorts it, as the constructor does.
    #[setter]
    fn set_times_evt(&self, mut times: Vec<f64>) {
        times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        self.sched.borrow_mut().times_evt = times;
    }
}

/// Event that triggers when a user-supplied boolean condition becomes true,
/// invoking its action callback. Useful for state-dependent logic that is not
/// a simple zero crossing. Drop-in compatible with pathsim.events.Condition.
#[pyclass(name = "Condition", unsendable, subclass)]
pub struct PyCondition { pub(super) inner: SimEventRef, active: ActiveFlag }

event_pymethods! {
    PyCondition,
    #[pyo3(signature = (func_evt, func_act=None, tolerance=1e-4))]
    fn new(func_evt: Py<PyAny>, func_act: Option<Py<PyAny>>, tolerance: f64) -> Self {
        let evt_fn = move |t: f64| -> bool {
            Python::attach(|py| { match func_evt.call1(py, (t,)) { Ok(r) => r.extract::<bool>(py).unwrap_or(false), Err(e) => on_callback_err(py, e, false) } })
        };
        let act_fn: Option<Box<dyn FnMut(f64)>> = func_act.map(|f| {
            Box::new(move |t: f64| { Python::attach(|py| { if let Err(e) = f.call1(py, (t,)) { report_callback_error(py, e); } }); }) as Box<dyn FnMut(f64)>
        });
        let evt = crate::events::condition::Condition::new(evt_fn, act_fn, tolerance);
        let active = evt.active_flag();
        Self { inner: Rc::new(FastCell::new(evt)), active }
    }
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
