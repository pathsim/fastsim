// Core PyO3 types: Block, Solver, PortReference, Connection.

use std::rc::Rc;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::blocks::block::{Block, BlockRef};
use crate::blocks::constructors;
use crate::connection::Connection;
use crate::utils::fastcell::FastCell;
use crate::utils::portreference::{Port, PortReference};

/// Base class for all simulation blocks.
///
/// Blocks are the fundamental building elements of a simulation. Each block
/// has input and output ports connected via ``Connection`` objects. Blocks
/// can be dynamical (with internal state and an ODE solver) or algebraic
/// (pure input-output mapping).
#[pyclass(name = "Block", unsendable, subclass, from_py_object)]
#[derive(Clone)]
pub struct PyBlock {
    pub inner: BlockRef,
}

impl PyBlock {
    pub fn wrap(inner: BlockRef) -> Self { Self { inner } }
}

#[pymethods]
impl PyBlock {
    /// Create a Block. Accepts arbitrary args/kwargs so Python subclasses
    /// can call super().__init__(...) without issues.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, pyo3::types::PyTuple>, _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>) -> Self {
        Self { inner: Rc::new(FastCell::new(Block::default_block())) }
    }

    /// Initialize this Block from a Rust block (returned by factory functions).
    fn _init_from(&mut self, other: &PyBlock) {
        self.inner = other.inner.clone();
    }

    /// Adopt `other`'s behaviour **in place**, preserving this block's identity.
    ///
    /// Used when a constructor parameter is reassigned at runtime
    /// (`amp.gain = 10`): the shim rebuilds the block from its factory, but a
    /// plain `_init_from` would only repoint the Python handle at the new `Rc`
    /// while the `Simulation` and every `Connection` keep holding the old one —
    /// the change would read back correctly and have no effect on the run.
    /// Swapping the cell contents instead keeps every existing reference valid.
    ///
    /// The connection layout (register sizes resolved when the diagram was
    /// wired) and the integration engine (the live state) belong to the block's
    /// *place in the system*, not to its parameters, so they are carried over
    /// rather than taken from the freshly built block.
    fn _adopt(&self, other: &PyBlock) {
        if Rc::ptr_eq(&self.inner, &other.inner) {
            return;
        }

        let this = self.inner.borrow_mut();
        let fresh = other.inner.borrow_mut();

        std::mem::swap(this, fresh);

        // `this` now holds the freshly built behaviour, `fresh` the old block.
        // Take back what describes this block's place in the system.
        std::mem::swap(&mut this.inputs, &mut fresh.inputs);
        std::mem::swap(&mut this.outputs, &mut fresh.outputs);
        std::mem::swap(&mut this.engine, &mut fresh.engine);
    }

    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<PyPortRef> {
        if let Ok(idx) = key.extract::<usize>() {
            return Ok(PyPortRef { inner: PortReference::new(self.inner.clone(), Some(vec![Port::Index(idx)])) });
        }
        if let Ok(name) = key.extract::<String>() {
            return Ok(PyPortRef { inner: PortReference::new(self.inner.clone(), Some(vec![Port::Name(name)])) });
        }
        if let Ok(slice) = key.cast::<pyo3::types::PySlice>() {
            let stop_obj = slice.getattr("stop")?;
            if stop_obj.is_none() {
                return Err(PyValueError::new_err("port slice cannot be open-ended"));
            }
            let stop: usize = stop_obj.extract()?;
            if stop == 0 {
                return Err(PyValueError::new_err("port slice cannot end with 0"));
            }
            let start: usize = slice.getattr("start")?.extract().unwrap_or(0);
            let step: usize = slice.getattr("step")?.extract().unwrap_or(1).max(1);
            let ports: Vec<Port> = (start..stop).step_by(step).map(Port::Index).collect();
            return Ok(PyPortRef { inner: PortReference::new(self.inner.clone(), Some(ports)) });
        }
        if let Ok(list) = key.extract::<Vec<usize>>() {
            let ports: Vec<Port> = list.into_iter().map(Port::Index).collect();
            return Ok(PyPortRef { inner: PortReference::new(self.inner.clone(), Some(ports)) });
        }
        Err(PyValueError::new_err("port must be int, str, slice, or list"))
    }

    fn __len__(&self) -> usize { self.inner.borrow().len() }
    fn __bool__(&self) -> bool { self.inner.borrow().is_active() }

    /// Read scope data: returns (times, channels) where channels[i] is all samples for channel i.
    /// With `incremental=True`, returns only samples recorded since the last
    /// incremental read (advancing the cursor) — used for live streaming.
    #[pyo3(signature = (incremental=false))]
    fn read(&self, py: Python<'_>, incremental: bool) -> PyResult<(Py<PyAny>, Vec<Py<PyAny>>)> {
        if self.inner.borrow().type_name == "Spectrum" {
            let (freq, spectra) = constructors::spectrum_read(self.inner.borrow());
            let np = py.import("numpy")?;
            let freq_arr: Py<PyAny> = np.call_method1("array", (freq,))?.unbind();
            let ch_list: Vec<Py<PyAny>> = spectra.iter().map(|(re, im)| {
                let re_arr = np.call_method1("array", (re.clone(),)).unwrap();
                let im_arr = np.call_method1("array", (im.clone(),)).unwrap();
                let j = np.call_method1("complex128", (0.0, 1.0)).unwrap();
                let complex = re_arr.call_method1("__add__", (im_arr.call_method1("__mul__", (j,)).unwrap(),)).unwrap();
                complex.unbind()
            }).collect();
            return Ok((freq_arr, ch_list));
        }

        let (times, data) = if incremental {
            constructors::scope_read_incremental(self.inner.borrow_mut())
        } else {
            constructors::scope_read(self.inner.borrow())
        };
        let n_samples = data.len();
        let n_channels = if n_samples > 0 { data[0].len() } else { 0 };
        let mut channels: Vec<Vec<f64>> = vec![Vec::with_capacity(n_samples); n_channels];
        for sample in &data {
            for (ch, &val) in sample.iter().enumerate() {
                if ch < n_channels {
                    channels[ch].push(val);
                }
            }
        }
        match py.import("numpy") {
            Ok(np) => {
                let time_arr: Py<PyAny> = np.call_method1("array", (times,))?.unbind();
                let ch_list: Vec<Py<PyAny>> = channels.iter().map(|ch| {
                    np.call_method1("array", (ch.clone(),)).unwrap().unbind()
                }).collect();
                Ok((time_arr, ch_list))
            }
            Err(_) => {
                let time_obj: Py<PyAny> = times.into_pyobject(py)?.unbind();
                let ch_list: Vec<Py<PyAny>> = channels.into_iter().map(|ch| {
                    ch.into_pyobject(py).unwrap().unbind()
                }).collect();
                Ok((time_obj, ch_list))
            }
        }
    }

    /// Reset the block to its initial state (state, memory, sinks).
    fn reset(&self) { self.inner.borrow_mut().reset(); }
    /// Activate the block (participates in evaluation again).
    fn on(&self) { self.inner.borrow_mut().on(); }
    /// Deactivate the block (skipped during evaluation; outputs freeze).
    fn off(&self) { self.inner.borrow_mut().off(); }

    /// Attach an internal event to this block (used by `fastsim.port` to carry
    /// over a wrapped pathsim block's events). The simulation tracks it as a
    /// block-internal event once the block is added.
    fn add_event(&self, event: &Bound<'_, PyAny>) -> PyResult<()> {
        let evt = super::events::extract_event_ref(event)?;
        // Idempotent: the same event may reach a block from more than one side
        // — attached during porting AND published in the block's own `events`
        // list, which `Simulation` harvests. Registering it twice would detect
        // and resolve it twice per step.
        let key = std::rc::Rc::as_ptr(&evt) as *const ();
        let blk = self.inner.borrow_mut();
        if blk.events.iter().any(|e| std::rc::Rc::as_ptr(e) as *const () == key) {
            return Ok(());
        }
        blk.events.push(evt);
        Ok(())
    }

    /// Switch.select(state) — route input `state` to the output (pathsim API).
    ///
    /// `state` is the 0-based input index to select, or `None` to open the switch
    /// (route nothing). Only meaningful on a `Switch` block.
    #[pyo3(text_signature = "($self, switch_state)")]
    fn select(&self, switch_state: Option<usize>) {
        let val = switch_state.map(|s| s as f64).unwrap_or(-1.0);
        self.inner.borrow_mut().data_f64.insert("switch_state".to_string(), val);
    }

    /// Return this block's current `(inputs, outputs, states)` as three lists of
    /// floats — the raw numeric I/O snapshot (no numpy dependency, unlike
    /// `__call__`). Useful for lightweight probing of a block after a step.
    #[pyo3(text_signature = "($self)")]
    fn get_all(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        self.inner.borrow().get_all()
    }

    #[getter]
    fn type_name(&self) -> &'static str { self.inner.borrow().type_name }

    /// Stable identity for this block across Python wrappers: the address of
    /// the underlying Rc. Python's id() is NOT stable here — every Rust->Python
    /// crossing (e.g. `subsystem.blocks`) makes a fresh wrapper with a new
    /// id(), which breaks id()-keyed maps. Hosts should key on block_id.
    #[getter]
    fn block_id(&self) -> usize {
        std::rc::Rc::as_ptr(&self.inner) as *const () as usize
    }

    /// Child blocks of a composite (Subsystem); empty for leaf blocks. Lets a
    /// host recurse into a subsystem, e.g. to read Scopes nested inside it.
    #[getter]
    fn blocks(&self) -> Vec<PyBlock> {
        match self.inner.borrow().children_fn.as_ref() {
            Some(f) => f().into_iter().map(PyBlock::wrap).collect(),
            None => Vec::new(),
        }
    }

    /// Compile a **Subsystem** block into a single fused block — a
    /// ``DynamicalSystem`` with extras. The subsystem's interior is baked into
    /// one tape: ``dx/dt`` and the outputs ``y`` evaluate without per-block
    /// dispatch, the Jacobian is exact (symbolic autodiff), and the subsystem's
    /// internal events (zero-crossings, schedules, conditions) are captured as
    /// the new block's own events. The result drops into a ``Simulation`` like
    /// any other block.
    ///
    /// The subsystem must be assembled first (added to a ``Simulation`` that has
    /// run at least one step), so its interface widths are resolved. Raises
    /// ``ValueError`` on a non-subsystem block or an uncompilable interior
    /// (algebraic loop, opaque/extern block, no continuous state).
    fn compile(&self) -> PyResult<PyBlock> {
        let sub_ir = crate::ir::builder::subsystem_to_ir(&self.inner)
            .ok_or_else(|| PyValueError::new_err("compile() is only supported on Subsystem blocks"))?;
        let block = crate::compile::compile_block(&sub_ir)
            .map_err(|e| PyValueError::new_err(format!("subsystem compile failed: {e}")))?;
        Ok(PyBlock::wrap(block))
    }

    /// Export a **Subsystem** block as a source FMU (FMI 3.0, Model Exchange)
    /// written to `path`. The subsystem's interface inputs become FMI input
    /// variables an importer drives via `fmi3SetFloat64` (an open system
    /// `f(x, u, t)`); its states, outputs and parameters are exposed as usual,
    /// and (where differentiable) an analytic directional derivative ∂ẋ/∂x and
    /// the event interface are included.
    ///
    /// The subsystem must be assembled first (its interface widths resolved).
    /// Raises `ValueError` on a non-subsystem block or an uncompilable interior.
    /// See `Simulation.to_fmu` for closed systems and the option list.
    #[pyo3(signature = (
        path, name = "subsystem", *, kind = "both",
        start_time = None, stop_time = None, tolerance = None, step_size = None,
        instantiation_token = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn to_fmu(
        &self,
        path: &str,
        name: &str,
        kind: &str,
        start_time: Option<f64>,
        stop_time: Option<f64>,
        tolerance: Option<f64>,
        step_size: Option<f64>,
        instantiation_token: Option<String>,
    ) -> PyResult<()> {
        if !matches!(kind, "both" | "me" | "cs") {
            return Err(PyValueError::new_err(format!(
                "kind must be 'both', 'me' or 'cs', got '{kind}'"
            )));
        }
        let module = crate::ir::builder::module_from_subsystem(&self.inner, name)
            .ok_or_else(|| PyValueError::new_err("to_fmu() is only supported on Subsystem blocks"))?;
        let opts = crate::fmi::export::ExportOptions {
            model_name: None,
            instantiation_token,
            start_time,
            stop_time,
            tolerance,
            step_size,
            // A subsystem is a component, not a simulation: it carries no solver
            // of its own, so the Co-Simulation path keeps the codegen default.
            solver: None,
            tolerances: None,
            model_exchange: kind != "cs",
            co_simulation: kind != "me",
        };
        crate::fmi::export::export_fmu(&module, path, &opts)
            .map_err(|e| PyValueError::new_err(format!("FMU export failed: {e}")))
    }

    // inputs/outputs/state return numpy arrays for consistency with `read()`
    // and `__call__` (issue #33): a single array type avoids the list-repeat
    // footgun (`block.outputs * 2`). `state` returns an EMPTY array for a
    // stateless block rather than `None`, so `block.state[...]` / `len()` /
    // unpacking behave uniformly. This is a deliberate deviation from pathsim
    // (whose `inputs`/`outputs` are dict-like `Register` objects); fastsim
    // already exposed flat arrays here, so returning ndarrays is the smaller,
    // more useful step and keeps parity with fastsim's own array-returning APIs.
    #[getter]
    fn inputs(&self, py: Python<'_>) -> Py<PyAny> {
        super::helpers::to_numpy(py, &self.inner.borrow().inputs.to_array())
    }

    /// The output register as an ndarray whose assignments reach the block.
    ///
    /// Still a numpy array (see the note above); `fastsim.blocks._outputs`
    /// only makes `block.outputs[i] = value` land instead of writing into a
    /// snapshot that is thrown away — how a block written in Python reports
    /// its result. Falls back to the plain array if that module cannot be
    /// imported, so the getter never fails.
    #[getter]
    fn outputs<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyAny> {
        let py = slf.py();
        let data = slf.borrow().inner.borrow().outputs.to_array();
        let arr = super::helpers::to_numpy(py, &data);
        py.import("fastsim.blocks._outputs")
            .and_then(|m| m.call_method1("_view", (&arr, slf)))
            .unwrap_or_else(|_| arr.into_bound(py))
    }

    /// Replace the whole output register (`block.outputs = [...]`).
    #[setter]
    fn set_outputs(&self, values: Vec<f64>) {
        self.inner.borrow_mut().outputs.update_from_array(&values);
    }

    /// Write one output element, growing the register as pathsim's does.
    /// Used by the writable view; `block.outputs[i] = v` is the public spelling.
    fn _set_output(&self, index: usize, value: f64) {
        self.inner.borrow_mut().outputs.set_single(index, value);
    }

    /// This block's `(A, B, C, D)` if it is linear time-invariant, else `None`.
    ///
    /// Every LTI block is built from a state-space realization — transfer
    /// functions, PT1/PT2, the Butterworth and allpass filters — and this
    /// reports the matrices actually integrated, not a re-derivation.
    fn state_space(&self) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let blk = self.inner.borrow();
        let get = |k: &str| blk.data_vec2.get(k).cloned();
        Some((get("A")?, get("B")?, get("C")?, get("D")?))
    }

    #[getter]
    fn state(&self, py: Python<'_>) -> Py<PyAny> {
        let s = self.inner.borrow().state().map(|s| s.to_vec()).unwrap_or_default();
        super::helpers::to_numpy(py, &s)
    }

    #[setter]
    fn set_state(&self, val: Vec<f64>) {
        self.inner.borrow_mut().set_state(&val);
    }

    #[getter]
    fn size(&self) -> (usize, usize) { self.inner.borrow().size() }

    #[getter]
    fn shape(&self) -> (usize, usize) { self.inner.borrow().shape() }

    /// engine property — returns PySolver object or None (pathsim-compatible)
    #[getter]
    fn engine(&self) -> Option<PySolver> {
        if self.inner.borrow().has_engine() {
            Some(PySolver { block: self.inner.clone() })
        } else {
            None
        }
    }

    /// Block.__call__() — returns (inputs, outputs, states) as numpy arrays
    fn __call__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let blk = self.inner.borrow();
        let inputs = blk.inputs._data.clone();
        let outputs = blk.outputs._data.clone();
        let states: Vec<f64> = blk.engine.as_ref()
            .map(|e| e.get().to_vec())
            .unwrap_or_default();
        match py.import("numpy") {
            Ok(np) => {
                let i = np.call_method1("array", (inputs,))?.unbind();
                let o = np.call_method1("array", (outputs,))?.unbind();
                let s = np.call_method1("array", (states,))?.unbind();
                Ok((i, o, s))
            }
            Err(_) => {
                Ok((
                    inputs.into_pyobject(py)?.unbind(),
                    outputs.into_pyobject(py)?.unbind(),
                    states.into_pyobject(py)?.unbind(),
                ))
            }
        }
    }

    #[getter]
    fn _active(&self) -> bool { self.inner.borrow().is_active() }

    fn __repr__(&self) -> String {
        let blk = self.inner.borrow();
        let (n_in, n_out) = blk.size();
        format!("{}(inputs={}, outputs={})", blk.type_name, n_in, n_out)
    }

    fn __str__(&self) -> String { self.__repr__() }
}

// ======================================================================================
// PySolver — Python wrapper for Block engine with .get()/.set()
// ======================================================================================

#[pyclass(name = "Solver", unsendable)]
/// Integration engine attached to a dynamical block.
pub struct PySolver {
    block: BlockRef,
}

#[pymethods]
impl PySolver {
    fn get(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let blk = self.block.borrow();
        let state = blk.engine.as_ref()
            .map(|e| e.get().to_vec())
            .unwrap_or_default();
        match py.import("numpy") {
            Ok(np) => Ok(np.call_method1("array", (state,))?.unbind()),
            Err(_) => Ok(state.into_pyobject(py)?.unbind()),
        }
    }

    /// `engine.state` property — pathsim-compatible read access (mirrors get()).
    /// pathsim event callbacks use `block.engine.state`; expose it so they work
    /// unchanged on fastsim.
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.get(py)
    }

    fn set(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let blk = self.block.borrow_mut();
        if let Some(ref mut engine) = blk.engine {
            // Try array/list FIRST: a 1-element numpy array (the common case,
            // e.g. -engine.state) extracts cleanly as Vec. Trying f64 first
            // would force an ndim>0 -> scalar conversion (NumPy DeprecationWarning).
            if let Ok(v) = value.extract::<Vec<f64>>() {
                engine.set(&v);
            } else if let Ok(f) = value.extract::<f64>() {
                engine.set(&[f]);
            } else if let Ok(arr) = value.call_method0("tolist") {
                if let Ok(v) = arr.extract::<Vec<f64>>() {
                    engine.set(&v);
                } else if let Ok(f) = arr.extract::<f64>() {
                    engine.set(&[f]);
                }
            }
        }
        Ok(())
    }

    fn __repr__(&self) -> String {
        let blk = self.block.borrow();
        match &blk.engine {
            Some(e) => format!("Solver(type='{}', state={:?})", e.type_name, e.get()),
            None => "Solver(None)".to_string(),
        }
    }
}

// ======================================================================================
// PyPortRef
// ======================================================================================

#[pyclass(name = "PortReference", unsendable)]
/// Reference to a block with specific port indices.
pub struct PyPortRef {
    pub inner: PortReference,
}

#[pymethods]
impl PyPortRef {
    /// Current values at the referenced output ports.
    fn get_outputs(&self) -> Vec<f64> {
        self.inner.get_outputs()
    }

    /// Current values at the referenced input ports.
    fn get_inputs(&self) -> Vec<f64> {
        self.inner.get_inputs()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

// ======================================================================================
// PyConnection — mirrors pathsim Connection
// ======================================================================================

#[pyclass(name = "Connection", unsendable, subclass)]
/// Directed connection between block ports.
pub struct PyConnection {
    pub inner: Rc<Connection>,
}

#[pymethods]
impl PyConnection {
    #[new]
    #[pyo3(signature = (*args))]
    fn new(args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Self> {
        if args.len() < 2 {
            return Err(PyValueError::new_err("connection needs at least source and one target"));
        }
        let source = extract_port_ref(&args.get_item(0)?)?;
        let mut targets = Vec::new();
        for i in 1..args.len() {
            targets.push(extract_port_ref(&args.get_item(i)?)?);
        }
        let conn = Connection::new(source, targets);
        Ok(Self { inner: Rc::new(conn) })
    }

    fn __len__(&self) -> usize { self.inner.len() }
    fn __bool__(&self) -> bool { self.inner.is_active() }

    /// Activate the connection (transfers data again).
    fn on(&self) { self.inner.on(); }
    /// Deactivate the connection (no data transfer until `on`).
    fn off(&self) { self.inner.off(); }
}

fn extract_port_ref(obj: &Bound<'_, PyAny>) -> PyResult<PortReference> {
    if let Ok(block) = obj.cast::<PyBlock>() {
        let b: PyRef<PyBlock> = block.borrow();
        return Ok(PortReference::new(b.inner.clone(), None));
    }
    if let Ok(pr) = obj.cast::<PyPortRef>() {
        let p: PyRef<PyPortRef> = pr.borrow();
        return Ok(PortReference::new(p.inner.block.clone(), Some(p.inner.ports.clone())));
    }
    Err(PyValueError::new_err("expected Block or PortReference"))
}
