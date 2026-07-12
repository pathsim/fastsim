// PySimulation PyO3 wrapper and standalone Solver classes.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use pyo3::prelude::*;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};

use crate::blocks::block::BlockRef;
use crate::connection::Connection;
use crate::constants::*;
use crate::simulation::PendingOp;
use crate::utils::fastcell::FastCell;

use super::{PyBlock, PyConnection};
use super::events::{
    PyDiagnostics, extract_event_ref,
};
use super::helpers::{extract_initial_value, reset_run_signals, take_pending_error, report_callback_error};

/// Python-side handle to a simulation's mutation queue.
///
/// The Schedule callback runs while `sim.run()` is holding a `&mut self`
/// borrow on PySimulation, so calling `sim.add_block(...)` from inside
/// the callback raises `RuntimeError: Already borrowed`.  `PendingOps`
/// works around that: the callback writes into this independent
/// `Rc<RefCell<Vec<…>>>` which has nothing to do with the PySimulation
/// borrow, and the engine drains the queue at the next timestep boundary
/// (in `Simulation::timestep` → `_apply_pending_ops`).
///
/// Obtain one via `sim.pending_ops()` and capture it in your callback's
/// closure:
///
/// ```python
/// ops = sim.pending_ops()
/// def cb(t):
///     ops.add_block(my_new_block)
/// schedule = ScheduleList(times, func_act=cb)
/// sim.add_event(schedule)
/// ```
#[pyclass(name = "PendingOps", unsendable)]
pub struct PyPendingOps {
    queue: Rc<RefCell<Vec<PendingOp>>>,
}

#[pymethods]
impl PyPendingOps {
    /// Queue a block for addition at the next timestep boundary.
    fn add_block(&self, block: PyBlock) {
        self.queue.borrow_mut().push(PendingOp::AddBlock(block.inner));
    }
    /// Queue a block for removal at the next timestep boundary.
    fn remove_block(&self, block: &PyBlock) {
        self.queue.borrow_mut().push(PendingOp::RemoveBlock(block.inner.clone()));
    }
    /// Queue a connection for addition at the next timestep boundary.
    fn add_connection(&self, conn: &Bound<'_, PyConnection>) {
        let c: PyRef<PyConnection> = conn.borrow();
        self.queue.borrow_mut().push(PendingOp::AddConnection(c.inner.clone()));
    }
    /// Queue a connection for removal at the next timestep boundary.
    fn remove_connection(&self, conn: &Bound<'_, PyConnection>) {
        let c: PyRef<PyConnection> = conn.borrow();
        self.queue.borrow_mut().push(PendingOp::RemoveConnection(c.inner.clone()));
    }
    /// Queue an event for addition at the next timestep boundary.
    fn add_event(&self, event: &Bound<'_, PyAny>) -> PyResult<()> {
        let evt = extract_event_ref(event)?;
        self.queue.borrow_mut().push(PendingOp::AddEvent(evt));
        Ok(())
    }
    /// Queue an event for removal at the next timestep boundary.
    fn remove_event(&self, event: &Bound<'_, PyAny>) -> PyResult<()> {
        let evt = extract_event_ref(event)?;
        self.queue.borrow_mut().push(PendingOp::RemoveEvent(evt));
        Ok(())
    }
    fn __len__(&self) -> usize { self.queue.borrow().len() }
}

#[pyclass(name = "Simulation", unsendable)]
/// System simulation engine.
///
/// Manages blocks, connections, and events and performs transient simulation.
/// The global system equation is solved by fixed-point iteration using
/// Anderson acceleration for inter-block coupling.
///
/// `inner` lives in a `FastCell` (UnsafeCell wrapper) so all PyO3
/// methods can take `&self`.  Reads from inside a `Schedule.func_act`
/// callback (e.g. `sim.size`, `sim.time`) work because the simulation
/// is paused at the callback boundary.  Mutations route via the
/// `in_run` flag: while a `run`/`run_streaming`/`run_realtime` is on
/// the stack, `sim.add_block(...)` / `sim.add_connection(...)` etc.
/// queue onto `pending_ops_queue` (a clone of
/// `Simulation::_pending_ops`); otherwise they apply synchronously.
/// The engine drains the queue at the next timestep boundary, so
/// pathsim-style direct mutation from a callback works without an
/// extra API.
pub struct PySimulation {
    inner: FastCell<crate::simulation::Simulation>,
    in_run: Cell<bool>,
    pending_ops_queue: Rc<RefCell<Vec<PendingOp>>>,
}

/// Extract one block or a list of blocks into `Vec<BlockRef>`.  Single-
/// element call sites pass a `PyBlock` directly; bulk call sites pass any
/// iterable (typically a Python list).
fn extract_blocks(arg: &Bound<'_, PyAny>) -> PyResult<Vec<BlockRef>> {
    if let Ok(b) = arg.extract::<PyBlock>() {
        return Ok(vec![b.inner]);
    }
    let list: Vec<PyBlock> = arg.extract().map_err(|_| {
        PyValueError::new_err("expected a Block or an iterable of Blocks")
    })?;
    Ok(list.into_iter().map(|b| b.inner).collect())
}

/// Same as [`extract_blocks`] but for connections.
fn extract_connections(arg: &Bound<'_, PyAny>) -> PyResult<Vec<Rc<Connection>>> {
    if let Ok(c) = arg.cast::<PyConnection>() {
        return Ok(vec![c.borrow().inner.clone()]);
    }
    if let Ok(seq) = arg.try_iter() {
        let mut out: Vec<Rc<Connection>> = Vec::new();
        for item in seq {
            let item = item?;
            let c: PyRef<PyConnection> = item.cast::<PyConnection>()
                .map_err(|_| PyValueError::new_err("iterable must contain Connection objects"))?
                .borrow();
            out.push(c.inner.clone());
        }
        return Ok(out);
    }
    Err(PyValueError::new_err("expected a Connection or an iterable of Connections"))
}

/// RAII guard that flips `in_run` true on construction and false on drop,
/// so the flag is cleared even if the run panics or returns early.
struct InRunGuard<'a>(&'a Cell<bool>);
impl<'a> InRunGuard<'a> {
    fn new(flag: &'a Cell<bool>) -> Self {
        flag.set(true);
        Self(flag)
    }
}
impl<'a> Drop for InRunGuard<'a> {
    fn drop(&mut self) { self.0.set(false); }
}

/// Flatten a `RunStats` into the Python-facing `{name: f64}` dict. `with_runtime`
/// adds the wall-clock `runtime_ms` key (omitted by `run_until`, which reports
/// only step counts).
fn stats_to_dict(stats: &crate::simulation::RunStats, with_runtime: bool) -> HashMap<String, f64> {
    let mut result = HashMap::new();
    result.insert("total_steps".to_string(), stats.total_steps as f64);
    result.insert("successful_steps".to_string(), stats.successful_steps as f64);
    result.insert("rejected_steps".to_string(), stats.rejected_steps as f64);
    result.insert("total_evals".to_string(), stats.total_evals as f64);
    result.insert("total_solver_its".to_string(), stats.total_solver_its as f64);
    // Structured numeric outcome (issue #27). `converged` is 1.0/0.0; a missing
    // value (NaN) means "not applicable" for `truncated_at` / `worst_block`.
    result.insert("converged".to_string(), if stats.converged { 1.0 } else { 0.0 });
    result.insert("max_residual".to_string(), stats.max_residual);
    result.insert("truncated_at".to_string(), stats.truncated_at.unwrap_or(f64::NAN));
    result.insert("worst_block".to_string(), stats.worst_block.map_or(f64::NAN, |i| i as f64));
    if with_runtime {
        result.insert("runtime_ms".to_string(), stats.wall_time_secs * 1000.0);
    }
    result
}

/// Emit an unconditional `FastSimConvergenceWarning` (issue #27) when a run's
/// structured outcome reports non-convergence or truncation. Independent of the
/// simulation `log` flag so the failure is visible AND catchable in headless /
/// optimizer contexts. Never raises — the fail-open policy keeps the run's
/// (truthful) stats flowing to the caller.
fn warn_run_outcome(py: Python<'_>, stats: &crate::simulation::RunStats) {
    if stats.converged && stats.truncated_at.is_none() {
        return;
    }
    let mut msg = String::new();
    if !stats.converged {
        let where_ = stats.worst_block
            .map(|i| format!(", worst block index {i}"))
            .unwrap_or_default();
        msg.push_str(&format!(
            "simulation did not fully converge (max WRMS residual {:.3e}{}); \
             results proceed on the best-so-far state — check RunStats['converged']",
            stats.max_residual, where_,
        ));
    }
    if let Some(t) = stats.truncated_at {
        if !msg.is_empty() { msg.push_str("; "); }
        msg.push_str(&format!(
            "trajectory truncated at t={t:.6} (step size hit dt_min before the requested end)"
        ));
    }
    let _ = super::helpers::warn_convergence(py, &msg);
}

impl PySimulation {
    /// After a core run call, surface any error that could not propagate
    /// through the run loop: a graph-assembly error (bad port alias, takes
    /// precedence — the run never really started) or an exception a user
    /// callback raised mid-run.
    fn surface_run_errors(&self) -> PyResult<()> {
        if let Some(e) = self.inner.borrow_mut().take_assembly_error() {
            return Err(e.into());
        }
        // A data-dependent runtime fault recorded from inside an eval closure
        // (e.g. Divider zero denominator under zero_div='raise', issue #28).
        if let Some(e) = crate::simulation::take_runtime_fault() {
            return Err(e.into());
        }
        if let Some(err) = take_pending_error() {
            return Err(err);
        }
        Ok(())
    }
}

#[pymethods]
impl PySimulation {
    /// Mirrors pathsim: Simulation(blocks, connections, Solver=RKDP54, dt=0.01, ...)
    #[new]
    #[pyo3(signature = (
        blocks=vec![],
        connections=vec![],
        events=vec![],
        dt=SIM_TIMESTEP,
        dt_min=SIM_TIMESTEP_MIN,
        dt_max=None,
        Solver=None,
        iterations_max=SIM_ITERATIONS_MAX,
        log=true,
        diagnostics=false,
        optimizer_history=None,
        tolerance_lte_abs=None,
        tolerance_lte_rel=None,
        **kwargs
    ))]
    #[allow(non_snake_case)]
    fn new(
        py: Python<'_>,
        blocks: Vec<PyBlock>,
        connections: Vec<Bound<'_, PyConnection>>,
        events: Vec<Py<PyAny>>,
        dt: f64,
        dt_min: f64,
        dt_max: Option<f64>,
        Solver: Option<&Bound<'_, PyAny>>,
        iterations_max: usize,
        log: bool,
        diagnostics: bool,
        optimizer_history: Option<usize>,
        tolerance_lte_abs: Option<f64>,
        tolerance_lte_rel: Option<f64>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Self> {
        use crate::solvers::factories;

        let block_refs: Vec<BlockRef> = blocks.iter().map(|b| b.inner.clone()).collect();
        let conn_refs: Vec<Rc<Connection>> = connections.iter()
            .map(|c| c.borrow().inner.clone()).collect();

        // The two LTE tolerance knobs users reach for most are now explicit
        // named parameters (issue #31): visible in `inspect.signature`, no
        // longer riding invisibly through **kwargs where a typo was dropped.
        let tol_abs = tolerance_lte_abs.unwrap_or(SOL_TOLERANCE_LTE_ABS);
        let tol_rel = tolerance_lte_rel.unwrap_or(SOL_TOLERANCE_LTE_REL);
        if let Some(kw) = kwargs {
            // Validate the remaining kwargs against the known-key set: a typo'd
            // solver knob raises TypeError instead of being silently dropped.
            // `tolerance_fpi` is the sole surviving passthrough (retired, warned).
            for key in kw.keys() {
                let k: String = key.extract()?;
                if k != "tolerance_fpi" {
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "Simulation() got an unexpected keyword argument '{k}'"
                    )));
                }
            }
            // `tolerance_fpi` retired in favour of WRMS-scaled `NLS_COEF`.
            // Quietly accept and ignore it for source compatibility, but
            // emit a DeprecationWarning so users notice and migrate.
            if kw.contains("tolerance_fpi")? {
                let warnings = py.import("warnings")?;
                warnings.call_method1(
                    "warn",
                    (
                        "tolerance_fpi has been retired; the implicit-stage and \
                         algebraic-loop solvers now use a WRMS-scaled criterion \
                         (NLS_COEF = 0.1) against tolerance_lte_abs/rel.  Remove \
                         tolerance_fpi from your Simulation/Subsystem call.",
                        py.get_type::<pyo3::exceptions::PyDeprecationWarning>(),
                    ),
                )?;
            }
        }

        let factory = if let Some(solver_obj) = Solver {
            let name: String = solver_obj.getattr("__name__")
                .map_err(|_| PyValueError::new_err("solver must be a solver class"))?
                .extract::<String>()?;
            factories::factory_from_name(&name, tol_abs, tol_rel)
                .ok_or_else(|| PyValueError::new_err(factories::unknown_solver_message(&name)))?
        } else {
            factories::ssprk22_factory()
        };

        let mut sim = crate::simulation::Simulation::with_solver_and_logger(
            block_refs, conn_refs, factory, dt, log,
        );
        sim.dt_min = dt_min;
        sim.dt_max = dt_max;
        sim.iterations_max = iterations_max;
        if diagnostics {
            sim.diagnostics = Some(crate::utils::diagnostics::Diagnostics::new());
        }
        if let Some(m) = optimizer_history {
            sim.set_optimizer_history(m);
        }

        for evt_obj in &events {
            Python::attach(|py| {
                if let Ok(r) = extract_event_ref(evt_obj.bind(py)) {
                    sim.add_event(r);
                }
            });
        }

        let pending_ops_queue = Rc::clone(&sim._pending_ops);
        Ok(Self {
            inner: FastCell::new(sim),
            in_run: Cell::new(false),
            pending_ops_queue,
        })
    }

    /// Run the simulation for a given duration.
    #[pyo3(signature = (duration=10.0, reset=false, adaptive=true))]
    fn run(&self, py: Python<'_>, duration: f64, reset: bool, adaptive: bool) -> PyResult<HashMap<String, f64>> {
        let _guard = InRunGuard::new(&self.in_run);
        reset_run_signals();
        let stats = self.inner.borrow_mut().run(duration, reset, adaptive);
        // Re-raise an exception a user callback raised mid-run (StopSimulation
        // stops cleanly and leaves no pending error).
        self.surface_run_errors()?;
        // Fail-open numeric outcome: warn (visibly + catchably) but return stats.
        warn_run_outcome(py, &stats);
        Ok(stats_to_dict(&stats, true))
    }

    /// Export the assembled model as hierarchical IR (JSON). Each block is
    /// either lowered to its op-graph (for codegen / verification) or recorded
    /// as a typed `extern` call; nested subsystems recurse. See `src/ir`.
    #[pyo3(signature = (name="model"))]
    fn to_ir_json(&self, name: &str) -> PyResult<String> {
        #[allow(clippy::needless_borrow)]
        let module = crate::ir::builder::module_from_sim(&self.inner.borrow(), name);
        serde_json::to_string_pretty(&module)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("IR serialize failed: {e}")))
    }

    /// Statically compile this model into a fused `dX/dt = F(X, t)` tape (see
    /// `compile`). Discrete and event-driven blocks are supported (memory joins a
    /// global `M` vector; block-internal zero-cross/schedule/condition events drive
    /// an event-aware run loop). Raises `ValueError` with a precise reason if the
    /// model is outside the static subset (opaque/extern block whose math lowers to
    /// a call, algebraic loop, no continuous state, simulation-level events). Sinks
    /// become recorded taps; subsystems are flattened.
    ///
    /// The compiled simulation inherits this simulation's solver, adaptive
    /// tolerances (`tolerance_lte_abs`/`tolerance_lte_rel`), timestep `dt` and
    /// logging, so a compiled run integrates the same problem with the same
    /// method by default. Override any of these afterwards via `set_solver` /
    /// `dt` / `log` on the returned object.
    fn compile(&self) -> PyResult<PyCompiledSimulation> {
        #[allow(clippy::needless_borrow)]
        let module = crate::ir::builder::module_from_sim(&self.inner.borrow(), "compiled");
        match crate::compile::compile(&module) {
            Ok(mut inner) => {
                let sim = self.inner.borrow();
                // Inherit the source simulation's solver, adaptive tolerances and
                // timestep so a compiled run integrates the *same* problem the
                // interpreted run did. Without this, `compile()` would silently
                // fall back to the default explicit `RKBS32`, which on a stiff
                // model is stability-bound and takes orders of magnitude more
                // steps than the implicit solver the user selected.
                inner.set_solver(
                    sim.engine.type_name,
                    sim.engine.tolerance_lte_abs,
                    sim.engine.tolerance_lte_rel,
                );
                inner.dt = sim.dt;
                // Inherit the source simulation's logging so a compiled run prints
                // the same progress lines (and `compile()` logs a COMPILE summary).
                inner.set_logging(sim.logger.enabled);
                Ok(PyCompiledSimulation { inner })
            }
            Err(e) => Err(PyValueError::new_err(format!("cannot statically compile: {e}"))),
        }
    }

    /// Generate standalone C99 source from this model, in-process (no JSON).
    ///
    /// Builds the IR straight from the live model (the same `module_from_sim`
    /// path as `compile`) and lowers it through the `codegen` backend, returning
    /// a dict mapping each file name to its C source. Files are named after the
    /// model (`<name>.h` + `<name>.c`; the default name yields `model.h` +
    /// `model.c`), and the internal `#include`s match, so two generated models
    /// can share one build directory. See `generate_c` for the full option list
    /// (`numeric`, `reductions`, `structure`, `layout`, `solver`, `api`).
    /// `scaffold=True` additionally emits an EDITABLE build scaffold —
    /// `CMakeLists.txt` (the model as a static library plus a `<name>_demo`
    /// executable) and `<name>_main.c` (a demo driver stepping the model via
    /// `<name>_step`, printing a CSV trajectory, with marked HAL hook points).
    /// `trace=True` additionally emits `<name>_trace.json` — the model-to-code
    /// trace map (block → emitted functions with file/line, block → signals
    /// with their `SIG_*` ids, block → events) plus static metrics (packed RAM
    /// estimate, integrator stack estimate, IR op counts, per-step work).
    /// `a2l=True` additionally emits `<name>.a2l` — an ASAP2 measurement/
    /// calibration description addressed via `SYMBOL_LINK` + computed struct
    /// offsets, ready for XCP tooling (CANape, INCA).
    ///
    /// `solver` selects the integrator's Butcher tableau by name
    /// (case-insensitive): `"rk4"` (default) and `"euler"` are fixed-step;
    /// `"rkdp54"`, `"rkck54"`, `"rkf45"`, `"rkf78"`, `"rkv65"`, `"rkbs32"`,
    /// `"rkf21"`, `"rkdp87"` are adaptive (embedded-error step control);
    /// `"ssprk22"`/`"ssprk33"`/`"ssprk34"` are fixed-step. Implicit (DIRK/ESDIRK)
    /// tableaus are not yet emitted.
    ///
    /// Raises `ValueError` for an unknown option value and `RuntimeError` for a
    /// construct the backend cannot lower (e.g. an opaque `extern` block).
    #[cfg(feature = "codegen")]
    #[pyo3(signature = (
        name = "model", *,
        numeric = "double",
        reductions = "unrolled",
        structure = "hierarchical",
        layout = "compact",
        solver = "rk4",
        api = "struct",
        scaffold = false,
        trace = false,
        a2l = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn to_c<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        numeric: &str,
        reductions: &str,
        structure: &str,
        layout: &str,
        solver: &str,
        api: &str,
        scaffold: bool,
        trace: bool,
        a2l: bool,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use super::codegen::{generate_to_dict, options_from_strs};
        let sim = self.inner.borrow();
        #[allow(clippy::needless_borrow)]
        let module = crate::ir::builder::module_from_sim(&sim, name);
        let log = sim.logger.clone();
        let opts =
            options_from_strs(numeric, reductions, structure, layout, solver, api, scaffold, trace, a2l)?;
        generate_to_dict(py, &module, &opts, &log)
    }

    /// Software-in-the-loop verification of the generated C (see `to_c`).
    ///
    /// Generates the C for this model, compiles it with a local C99 compiler
    /// (``$FASTSIM_CC``, ``$CC``, then ``cc``/``clang``/``gcc``; see
    /// ``find_c_compiler``), integrates the compiled binary AND the reference
    /// engine (the statically compiled tape, ``compile()``) over the same
    /// fixed-step trajectory, and compares the state trajectories sample by
    /// sample. Both sides step identically, so sample times align exactly.
    ///
    /// Returns a report dict: ``passed`` (worst scaled error ≤ 1),
    /// ``max_scaled_error`` (``|c - ref| / (atol + rtol·|ref|)``, worst over
    /// all states and samples), ``worst_state`` / ``worst_time``, ``n_steps``,
    /// ``n_states``, ``compiler``, ``files``, and ``build_dir`` (``None``
    /// unless ``keep_build``).
    ///
    /// Scope: fixed-step explicit solvers (``rk4``, ``euler``, ``ssprk*``) and
    /// models inside the static-compile subset (adaptive step sequences
    /// diverge between backends and are rejected). For ``numeric="float"``
    /// widen ``rtol`` — a float32 target legitimately deviates from the f64
    /// reference. Raises ``RuntimeError`` when no compiler is found or the
    /// build fails, ``ValueError`` for models outside the verifiable subset.
    #[cfg(all(feature = "codegen", not(target_family = "wasm")))]
    #[pyo3(signature = (
        name = "model", *,
        duration = 1.0,
        dt = 1e-3,
        solver = "rk4",
        numeric = "double",
        reductions = "unrolled",
        structure = "hierarchical",
        layout = "compact",
        atol = 1e-9,
        rtol = 1e-6,
        keep_build = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn verify_c<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        duration: f64,
        dt: f64,
        solver: &str,
        numeric: &str,
        reductions: &str,
        structure: &str,
        layout: &str,
        atol: f64,
        rtol: f64,
        keep_build: bool,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use super::codegen::options_from_strs;
        use crate::codegen::verify::{verify_c, VerifyCOptions};
        let sim = self.inner.borrow();
        #[allow(clippy::needless_borrow)]
        let module = crate::ir::builder::module_from_sim(&sim, name);
        let log = sim.logger.clone();
        let cg = options_from_strs(numeric, reductions, structure, layout, solver, "struct", false, false, false)?;
        let mut reference = crate::compile::compile(&module).map_err(|e| {
            PyValueError::new_err(format!(
                "cannot build the verification reference (static compile): {e}"
            ))
        })?;
        let vopts = VerifyCOptions { duration, dt, atol, rtol, keep_build };
        // Pure Rust + an external compiler process — release the GIL throughout.
        let report = py
            .detach(move || verify_c(&module, &mut reference, &cg, &vopts, &log))
            .map_err(|e| match e {
                crate::codegen::CodegenError::Verify(msg) => {
                    pyo3::exceptions::PyRuntimeError::new_err(msg)
                }
                other => pyo3::exceptions::PyRuntimeError::new_err(other.to_string()),
            })?;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("passed", report.passed)?;
        d.set_item("max_scaled_error", report.max_scaled_error)?;
        d.set_item("worst_state", report.worst_state)?;
        d.set_item("worst_time", report.worst_time)?;
        d.set_item("n_steps", report.n_steps)?;
        d.set_item("n_states", report.n_states)?;
        d.set_item("atol", atol)?;
        d.set_item("rtol", rtol)?;
        d.set_item("compiler", report.compiler)?;
        d.set_item("files", report.files)?;
        d.set_item(
            "build_dir",
            report.build_dir.map(|p| p.to_string_lossy().into_owned()),
        )?;
        Ok(d)
    }

    /// Export this model as a source FMU (FMI 3.0, Model Exchange) written to
    /// `path` (conventionally `*.fmu`).
    ///
    /// Builds the IR straight from the live model (the same `module_from_sim`
    /// path as `compile`/`to_c`), lowers it through the struct-API C backend,
    /// wraps it in the FMI Model-Exchange C layer, and zips the C sources with a
    /// generated `modelDescription.xml`. The result is a *source* FMU: it ships
    /// the C plus a `buildDescription.xml` so an importer compiles it on its own
    /// platform.
    ///
    /// Phase-1 scope: closed (input-free) continuous models with state and no
    /// events. Raises `ValueError` for a model outside that subset (no
    /// continuous state, events, or an opaque block the backend cannot lower).
    /// The optional `start_time` / `stop_time` / `tolerance` / `step_size`
    /// populate `<DefaultExperiment>`; `instantiation_token` overrides the
    /// default `{fastsim-<id>}`.
    #[cfg(all(feature = "codegen", feature = "fmi"))]
    #[pyo3(signature = (
        path, name = "model", *,
        start_time = None, stop_time = None, tolerance = None, step_size = None,
        instantiation_token = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn to_fmu(
        &self,
        path: &str,
        name: &str,
        start_time: Option<f64>,
        stop_time: Option<f64>,
        tolerance: Option<f64>,
        step_size: Option<f64>,
        instantiation_token: Option<String>,
    ) -> PyResult<()> {
        #[allow(clippy::needless_borrow)]
        let module = crate::ir::builder::module_from_sim(&self.inner.borrow(), name);
        let opts = crate::fmi::export::ExportOptions {
            model_name: None,
            instantiation_token,
            start_time,
            stop_time,
            tolerance,
            step_size,
        };
        crate::fmi::export::export_fmu(&module, path, &opts)
            .map_err(|e| PyValueError::new_err(format!("FMU export failed: {e}")))
    }


    /// Begin a chunked, cooperative run (for the streaming generator).
    /// Does the one-time setup (optional reset + initial eval/sample); pair
    /// with repeated `run_until` calls. Unlike `run_streaming`, this path is
    /// sim-time driven (no wall-clock), so it is WASM-safe and lets Python
    /// own the yield/step loop, injecting mutations between chunks.
    #[pyo3(signature = (reset=false, duration=10.0))]
    fn run_begin(&self, reset: bool, duration: f64) -> PyResult<()> {
        let _guard = InRunGuard::new(&self.in_run);
        reset_run_signals();
        self.inner.borrow_mut().run_begin(reset, duration);
        self.surface_run_errors()?;
        Ok(())
    }

    /// Finalize a chunked streaming run (logs FINISHED/INTERRUPTED). Call once
    /// after the last `run_until`.
    fn run_end(&self) {
        let _guard = InRunGuard::new(&self.in_run);
        self.inner.borrow_mut().run_end();
    }

    /// Advance up to `target_time` (a chunk boundary). `end_time` is the true
    /// run end (for adaptive overshoot prevention). Returns step counts.
    #[pyo3(signature = (target_time, end_time, adaptive=true))]
    fn run_until(&self, py: Python<'_>, target_time: f64, end_time: f64, adaptive: bool) -> PyResult<HashMap<String, f64>> {
        let _guard = InRunGuard::new(&self.in_run);
        let stats = self.inner.borrow_mut().run_until(target_time, end_time, adaptive);
        self.surface_run_errors()?;
        warn_run_outcome(py, &stats);
        Ok(stats_to_dict(&stats, false))
    }

    /// Run with streaming output — calls func_callback at tickrate Hz.
    #[pyo3(signature = (duration=10.0, reset=false, adaptive=true, tickrate=10.0, func_callback=None))]
    fn run_streaming(
        &self,
        duration: f64, reset: bool, adaptive: bool,
        tickrate: f64, func_callback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let _guard = InRunGuard::new(&self.in_run);
        reset_run_signals();
        let results: Rc<FastCell<Vec<Py<PyAny>>>> = Rc::new(FastCell::new(Vec::new()));
        let results_ref = results.clone();
        let cb: Option<Py<PyAny>> = func_callback.map(|f| f.clone().unbind());
        self.inner.borrow_mut().run_streaming(duration, reset, adaptive, tickrate, move |_progress, _success, _dt| {
            if let Some(ref func) = cb {
                Python::attach(|py| {
                    if let Ok(r) = func.bind(py).call0() {
                        results_ref.borrow_mut().push(r.unbind());
                    }
                });
            }
        });
        self.surface_run_errors()?;
        // The streaming closure (the only other Rc holder) has been dropped, so
        // refcount is 1; fall back to empty rather than panicking across the FFI
        // boundary if that ever fails to hold.
        Ok(Rc::into_inner(results).map(|c| c.into_inner()).unwrap_or_default())
    }

    /// Run synchronized to wall-clock time with optional speed factor.
    #[pyo3(signature = (duration=10.0, reset=false, adaptive=true, tickrate=30.0, speed=1.0, func_callback=None))]
    fn run_realtime(
        &self,
        duration: f64, reset: bool, adaptive: bool,
        tickrate: f64, speed: f64, func_callback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let _guard = InRunGuard::new(&self.in_run);
        reset_run_signals();
        let results: Rc<FastCell<Vec<Py<PyAny>>>> = Rc::new(FastCell::new(Vec::new()));
        let results_ref = results.clone();
        let cb: Option<Py<PyAny>> = func_callback.map(|f| f.clone().unbind());
        self.inner.borrow_mut().run_realtime(duration, reset, adaptive, tickrate, speed, move |_progress, _success, _dt| {
            if let Some(ref func) = cb {
                Python::attach(|py| {
                    if let Ok(r) = func.bind(py).call0() {
                        results_ref.borrow_mut().push(r.unbind());
                    }
                });
            }
        });
        self.surface_run_errors()?;
        // The streaming closure (the only other Rc holder) has been dropped, so
        // refcount is 1; fall back to empty rather than panicking across the FFI
        // boundary if that ever fails to hold.
        Ok(Rc::into_inner(results).map(|c| c.into_inner()).unwrap_or_default())
    }

    /// Reset all blocks to their initial state and the simulation time to
    /// zero (or to `time` if given).
    #[pyo3(signature = (time=None))]
    fn reset(&self, time: Option<f64>) {
        self.inner.borrow_mut().reset(time.unwrap_or(0.0));
    }

    /// Signal a running simulation to stop cleanly at the next timestep
    /// boundary (cooperative, callable from callbacks).
    fn stop(&self) { self.inner.borrow_mut().stop(); }

    /// Linearize the system about the current operating point. Not yet
    /// implemented in fastsim — raises `NotImplementedError` (pathsim parity
    /// gap, surfaced honestly instead of a silent no-op).
    fn linearize(&self) -> PyResult<()> {
        // pathsim implements operating-point linearization; fastsim has not
        // ported it yet. A silent no-op would masquerade as success, so we
        // surface the gap explicitly (drop-in honesty).
        Err(PyNotImplementedError::new_err(
            "Simulation.linearize() is not yet implemented in fastsim",
        ))
    }
    /// Revert a previous linearization. Not yet implemented in fastsim —
    /// raises `NotImplementedError` (see `linearize`).
    fn delinearize(&self) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "Simulation.delinearize() is not yet implemented in fastsim",
        ))
    }

    /// Return a handle on the simulation's mutation queue.  Equivalent
    /// to calling `sim.add_block(...)` etc. directly from a callback,
    /// kept for explicit-batching use cases.
    fn pending_ops(&self) -> PyPendingOps {
        PyPendingOps { queue: Rc::clone(&self.pending_ops_queue) }
    }

    /// Enable per-timestep wall-clock recording.  Subsequent `run` /
    /// `timestep` calls append elapsed seconds for each step into an
    /// internal buffer.  Negligible overhead (~30 ns/step) when enabled,
    /// zero overhead when off.
    fn enable_wct_trace(&self) {
        self.inner.borrow_mut()._wct_trace = Some(Vec::new());
    }

    /// Drain the per-timestep wall-clock buffer and return the recorded
    /// times in seconds.  Leaves recording enabled with an empty buffer.
    /// Returns an empty list if tracing was never enabled.
    fn take_wct_trace(&self) -> Vec<f64> {
        let sim = self.inner.borrow_mut();
        if let Some(buf) = sim._wct_trace.as_mut() {
            std::mem::take(buf)
        } else {
            Vec::new()
        }
    }

    /// Access current diagnostics snapshot (if enabled via diagnostics=True).
    #[getter]
    fn diagnostics(&self) -> Option<PyDiagnostics> {
        let sim = self.inner.borrow();
        sim.diagnostics.as_ref().map(|d| {
            let block_labels: Vec<String> = sim.blocks.iter()
                .map(|b| b.borrow().type_name.to_string())
                .collect();
            super::events::py_diagnostics(d.clone(), block_labels)
        })
    }

    /// Find the steady-state (DC operating point) by root-finding on the
    /// system residual instead of time integration. `reset=True` restores
    /// initial conditions first.
    fn steadystate(&self, reset: Option<bool>) -> PyResult<()> {
        reset_run_signals();
        self.inner.borrow_mut().steadystate(reset.unwrap_or(false));
        self.surface_run_errors()?;
        Ok(())
    }

    /// Find the periodic steady-state limit cycle of period `period` by
    /// matrix-free Anderson-accelerated shooting on the period map
    /// `g(x_0) = x(T; x_0)`.
    ///
    /// One outer iteration:
    ///
    /// 1. Integrate the system over `[0, T]` with the inner ODE solver
    ///    (a regular transient `run(period, ...)`).
    /// 2. Per dynamic block, run one matrix-free Anderson step on
    ///    `(x_start, x_end)`, mutating `x_start` toward the limit-cycle
    ///    period-start state.
    /// 3. Check the max WRMS-scaled residual `‖x(T) − x(0)‖` across all
    ///    dynamic blocks against the simulation's `NLS_COEF` threshold
    ///    (same convergence semantics as every other implicit-stage /
    ///    steady-state residual).
    /// 4. If not converged: reset `sim.time = 0` and event schedules,
    ///    repeat from step 1.
    ///
    /// After convergence, one final transient run over `[0, T]` records
    /// the converged limit-cycle trajectory in Scope blocks.
    ///
    /// Anderson needs only function evaluations (one period integration
    /// per outer iteration) — no monodromy matrix `Φ = ∂x(T)/∂x(0)` to
    /// assemble or factorize.  Converges in roughly 5–15 iterations on
    /// smooth, mildly-coupled periodic systems.  DAE blocks pass through
    /// transparently — their `engine_postprocess` installs the
    /// appropriate `StageBuilder` on the PSS-augmented engine, exactly
    /// as with any other solver factory.
    ///
    /// PSS pays off when the natural settling time is long relative to
    /// the forcing period (high-Q resonators, weakly-damped loops, large
    /// LC filters).  On strongly-damped systems where a plain transient
    /// settles in a handful of periods, `sim.run()` is faster — the
    /// shooting iteration carries a fixed overhead (warm-up + final
    /// sample-run) that only amortizes when many periods would otherwise
    /// be needed.
    ///
    /// Parameters
    /// ----------
    /// period : float
    ///     Period length `T` (in simulation time units).  Must be positive.
    /// Solver : type, optional
    ///     Inner ODE solver class (e.g. `RKDP54`, `ESDIRK43`, `GEAR52A`).
    ///     Defaults to the simulation's current solver.
    /// tolerance_lte_abs : float, optional
    ///     Absolute WRMS weight for both the inner solver's LTE control
    ///     and the outer shooting convergence test.  Defaults to the
    ///     current solver's setting.
    /// tolerance_lte_rel : float, optional
    ///     Relative WRMS weight, same dual role as above.  Defaults to
    ///     the current solver's setting.
    /// adaptive : bool, optional
    ///     Enable adaptive timestepping during the period integration.
    ///     Only honored when the inner solver is adaptive.  Default `True`.
    /// reset : bool, optional
    ///     If `True`, restore all blocks to their `initial_value` before
    ///     shooting starts.  If `False`, seed shooting from the current
    ///     simulation state — useful after a transient warm-up that
    ///     produces a good initial guess.  Default `False`.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Aggregate run statistics summed across all shooting iterations
    ///     plus the final sample run: `total_steps`, `successful_steps`,
    ///     `rejected_steps`, `total_evals`, `total_solver_its`, `runtime_ms`.
    ///
    /// Notes
    /// -----
    /// State-flipping events (relays, hysteresis, mode-switching) make
    /// `x(T; x_0)` non-smooth at the boundaries where the event
    /// activation changes, which may stall Anderson convergence.  If
    /// shooting fails to converge within `SIM_PSS_ITERATIONS_MAX` (50)
    /// iterations, a warning is logged and the best-so-far state is
    /// retained.
    #[allow(non_snake_case)]
    #[pyo3(signature = (period, Solver=None, tolerance_lte_abs=None, tolerance_lte_rel=None, adaptive=true, reset=false))]
    fn periodic_steady_state(
        &self,
        py: Python<'_>,
        period: f64,
        Solver: Option<&Bound<'_, PyAny>>,
        tolerance_lte_abs: Option<f64>,
        tolerance_lte_rel: Option<f64>,
        adaptive: bool,
        reset: bool,
    ) -> PyResult<HashMap<String, f64>> {
        use crate::solvers::factories;
        let _guard = InRunGuard::new(&self.in_run);

        // Solver + tolerances default to whatever the Simulation is already
        // configured with.  `solver_factory` is a boxed closure (not Clone),
        // so we round-trip through the type-name registry to rebuild it.
        let (current_name, current_tol_abs, current_tol_rel) = {
            let sim = self.inner.borrow();
            (sim.engine.type_name.to_string(),
             sim.engine.tolerance_lte_abs,
             sim.engine.tolerance_lte_rel)
        };
        let tol_abs = tolerance_lte_abs.unwrap_or(current_tol_abs);
        let tol_rel = tolerance_lte_rel.unwrap_or(current_tol_rel);
        let solver_name = if let Some(solver_obj) = Solver {
            solver_obj.getattr("__name__")
                .map_err(|_| PyValueError::new_err("Solver must be a solver class"))?
                .extract::<String>()?
        } else {
            current_name
        };
        let inner_factory = factories::factory_from_name(&solver_name, tol_abs, tol_rel)
            .ok_or_else(|| PyValueError::new_err(factories::unknown_solver_message(&solver_name)))?;

        reset_run_signals();
        let stats = self.inner.borrow_mut()
            .periodic_steady_state(period, inner_factory, adaptive, reset);
        self.surface_run_errors()?;
        warn_run_outcome(py, &stats);
        Ok(stats_to_dict(&stats, true))
    }

    /// Plot the recorded results of all Scope blocks (matplotlib; one figure
    /// per scope). Convenience mirror of `scope.plot()`.
    fn plot(&self) { self.inner.borrow().plot(); }

    /// Structured numeric summary of the last run (issue #29).
    ///
    /// Unlike the `diagnostics` snapshot getter (which needs `diagnostics=True`
    /// and returns the last per-timestep record), this is ALWAYS populated and
    /// carries the run's outcome — `converged`, `max_residual`, `worst_block`,
    /// `truncated_at` — the data the convergence trackers gather every step and
    /// previously discarded into a log string. When `diagnostics=True` it also
    /// folds in the last snapshot's iteration counts and worst residual.
    #[getter]
    fn run_summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        let sim = self.inner.borrow();
        let (converged, max_residual, worst_block, truncated_at) = sim.run_outcome();
        d.set_item("converged", converged)?;
        d.set_item("max_residual", max_residual)?;
        d.set_item("worst_block", worst_block)?;
        d.set_item("truncated_at", truncated_at)?;
        if let Some(diag) = sim.last_diagnostics() {
            d.set_item("time", diag.time)?;
            d.set_item("loop_iterations", diag.loop_iterations)?;
            d.set_item("solve_iterations", diag.solve_iterations)?;
            if let Some((blk, res)) = diag.worst_block() {
                d.set_item("worst_residual_block", blk)?;
                d.set_item("worst_residual", res)?;
            }
        }
        Ok(d)
    }

    /// Change the transient solver (public name — mirrors
    /// `CompiledSimulation.set_solver`, issue #33).
    #[pyo3(signature = (solver, tolerance_lte_abs=None, tolerance_lte_rel=None))]
    fn set_solver(&self, solver: &Bound<'_, PyAny>, tolerance_lte_abs: Option<f64>, tolerance_lte_rel: Option<f64>) -> PyResult<()> {
        use crate::solvers::factories;
        let tol_abs = tolerance_lte_abs.unwrap_or(crate::constants::SOL_TOLERANCE_LTE_ABS);
        let tol_rel = tolerance_lte_rel.unwrap_or(crate::constants::SOL_TOLERANCE_LTE_REL);

        let name: String = solver.getattr("__name__")
            .map_err(|_| PyValueError::new_err("solver must be a solver class (e.g. RKDP54, ESDIRK43)"))?
            .extract::<String>()?;

        let factory = factories::factory_from_name(&name, tol_abs, tol_rel)
            .ok_or_else(|| PyValueError::new_err(factories::unknown_solver_message(&name)))?;

        self.inner.borrow_mut().set_solver(factory);
        Ok(())
    }

    /// Backwards-compatible underscore alias for [`set_solver`]. Kept because the
    /// transient path historically exposed only `_set_solver`; new code should
    /// use the public `set_solver` (same name as the compiled path).
    #[pyo3(signature = (solver, tolerance_lte_abs=None, tolerance_lte_rel=None))]
    fn _set_solver(&self, solver: &Bound<'_, PyAny>, tolerance_lte_abs: Option<f64>, tolerance_lte_rel: Option<f64>) -> PyResult<()> {
        self.set_solver(solver, tolerance_lte_abs, tolerance_lte_rel)
    }

    /// Add a block, or a list of blocks.  Accepts either a single `Block`
    /// or any iterable of `Block`.  If a `run()` is currently active
    /// (e.g. called from inside a `Schedule.func_act` callback), the
    /// operation is queued onto the simulation's pending-ops queue and
    /// applied at the next timestep boundary; otherwise it is applied
    /// synchronously.  The bulk path stays inside Rust — no per-block
    /// Python⇄Rust boundary crossing.
    fn add_block(&self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
        let blocks = extract_blocks(arg)?;
        if self.in_run.get() {
            let mut q = self.pending_ops_queue.borrow_mut();
            for b in blocks { q.push(PendingOp::AddBlock(b)); }
        } else {
            let sim = self.inner.borrow_mut();
            for b in blocks { sim.add_block(b)?; }
        }
        Ok(())
    }

    /// Remove a block (or a list/iterable of blocks). Queued to the next
    /// timestep boundary when called from inside a running `run` (e.g. an
    /// event callback), applied synchronously otherwise.
    fn remove_block(&self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
        let blocks = extract_blocks(arg)?;
        if self.in_run.get() {
            let mut q = self.pending_ops_queue.borrow_mut();
            for b in blocks { q.push(PendingOp::RemoveBlock(b)); }
        } else {
            let sim = self.inner.borrow_mut();
            for b in blocks { sim.remove_block(&b)?; }
        }
        Ok(())
    }

    /// Add a connection, or a list of connections.
    fn add_connection(&self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
        let conns = extract_connections(arg)?;
        if self.in_run.get() {
            let mut q = self.pending_ops_queue.borrow_mut();
            for c in conns { q.push(PendingOp::AddConnection(c)); }
        } else {
            let sim = self.inner.borrow_mut();
            for c in conns { sim.add_connection(c); }
        }
        Ok(())
    }

    /// Remove a connection (or a list/iterable of connections). Queued to
    /// the next timestep boundary when called from inside a running `run`,
    /// applied synchronously otherwise.
    fn remove_connection(&self, arg: &Bound<'_, PyAny>) -> PyResult<()> {
        let conns = extract_connections(arg)?;
        if self.in_run.get() {
            let mut q = self.pending_ops_queue.borrow_mut();
            for c in conns { q.push(PendingOp::RemoveConnection(c)); }
        } else {
            let sim = self.inner.borrow_mut();
            for c in conns { sim.remove_connection(&c)?; }
        }
        Ok(())
    }

    /// Add an event (ZeroCrossing / Schedule / Condition, or a list of
    /// them). Queued to the next timestep boundary when called from inside a
    /// running `run`, applied synchronously otherwise.
    fn add_event(&self, event: &Bound<'_, PyAny>) -> PyResult<()> {
        let evt = extract_event_ref(event)?;
        if self.in_run.get() {
            self.pending_ops_queue.borrow_mut().push(PendingOp::AddEvent(evt));
        } else {
            self.inner.borrow_mut().add_event(evt);
        }
        Ok(())
    }

    /// Remove an event (or a list of events). Queued to the next timestep
    /// boundary when called from inside a running `run`, applied
    /// synchronously otherwise.
    fn remove_event(&self, event: &Bound<'_, PyAny>) -> PyResult<()> {
        let evt = extract_event_ref(event)?;
        if self.in_run.get() {
            self.pending_ops_queue.borrow_mut().push(PendingOp::RemoveEvent(evt));
            Ok(())
        } else {
            self.inner.borrow_mut().remove_event(&evt).map_err(PyErr::from)
        }
    }

    /// Whether the simulation is still running. `stop()` clears this; the
    /// streaming generator checks it to exit promptly between chunks.
    #[getter]
    fn active(&self) -> bool { self.inner.borrow().is_active() }

    #[getter]
    fn time(&self) -> f64 { self.inner.borrow().time }

    #[setter]
    fn set_time(&self, t: f64) { self.inner.borrow_mut().time = t; }

    #[getter]
    fn dt(&self) -> f64 { self.inner.borrow().dt }

    #[setter]
    fn set_dt(&self, dt: f64) { self.inner.borrow_mut().dt = dt; }

    /// The active solver's class name (e.g. ``"ESDIRK43"``). Mirrors
    /// `CompiledSimulation.solver`, so `sim.compile().solver == sim.solver`.
    #[getter]
    fn solver(&self) -> String { self.inner.borrow().engine.type_name.to_string() }

    #[getter]
    fn size(&self) -> (usize, usize) { self.inner.borrow().size() }

    #[getter]
    fn blocks(&self) -> Vec<PyBlock> {
        self.inner.borrow().blocks.iter().map(|b| PyBlock::wrap(b.clone())).collect()
    }

    #[getter]
    fn connections(&self) -> usize { self.inner.borrow().connections.len() }

    #[getter]
    fn events(&self) -> usize { self.inner.borrow().events.len() }

    #[getter]
    fn has_loops(&self) -> bool { self.inner.borrow().has_loops() }

    #[getter]
    fn dag_depth(&self) -> usize { self.inner.borrow().dag_depth() }

    #[getter]
    fn loop_depth(&self) -> usize { self.inner.borrow().loop_depth() }

    #[getter]
    fn n_dynamic_blocks(&self) -> usize { self.inner.borrow().num_dynamic_blocks() }

    fn __contains__(&self, obj: &Bound<'_, PyAny>) -> bool {
        let sim = self.inner.borrow();
        if let Ok(block) = obj.cast::<PyBlock>() {
            let b: pyo3::PyRef<PyBlock> = block.borrow();
            return sim.blocks.iter().any(|bl| Rc::ptr_eq(bl, &b.inner));
        }
        if let Ok(conn) = obj.cast::<PyConnection>() {
            let c: pyo3::PyRef<PyConnection> = conn.borrow();
            return sim.connections.iter().any(|co| Rc::ptr_eq(co, &c.inner));
        }
        false
    }
}

// ======================================================================================
// Solver integrate helper — direct solver loop, no Simulation
// ======================================================================================

/// Standalone integrate: trace func, create solver, run integrate loop directly.
fn solver_integrate_impl(
    py: Python<'_>,
    solver_name: &str,
    func: &Bound<'_, PyAny>,
    initial_value: &Bound<'_, PyAny>,
    t_start: f64,
    t_stop: f64,
    dt: f64,
    dt_min: f64,
    dt_max: Option<f64>,
    adaptive: bool,
    tolerance_lte_abs: f64,
    tolerance_lte_rel: f64,
    max_iterations: usize,
    optimizer: &str,
    optimizer_history: Option<usize>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    use crate::solvers::solver::{ExplicitSolver, ImplicitSolver, Jacobian};

    // Clear any stale stop/callback-error signals so a raising user ODE callback
    // in this integrate() surfaces its own exception (issue #28).
    reset_run_signals();

    let iv = extract_initial_value(Some(initial_value))?;
    let n_x = iv.len();
    let dt_max_val = dt_max.unwrap_or(f64::MAX);

    let graph = crate::tracer::trace_with_signature(py, func, &[
        crate::tracer::TraceArg::Array { name: "x", size: n_x },
        crate::tracer::TraceArg::Scalar { name: "t" },
    ])?;

    let rhs: Box<dyn Fn(&[f64], f64) -> Vec<f64>> = if let Some(ref g) = graph {
        let interp = crate::ssa::tape::InterpretedFn::from_graph(g.clone());
        Box::new(move |x: &[f64], t: f64| interp.call(&[x, &[t]]))
    } else {
        let func_obj = func.clone().unbind();
        let n = n_x;
        Box::new(move |x: &[f64], t: f64| {
            Python::attach(|py| {
                // Route a raising user callback through the run's callback-error
                // machinery so the ORIGINAL Python exception is re-raised after
                // the loop unwinds, instead of an uncatchable PanicException on
                // an .unwrap() (issue #28).
                let f = func_obj.bind(py);
                let xl = match pyo3::types::PyList::new(py, x) {
                    Ok(l) => l,
                    Err(e) => { report_callback_error(py, e); return vec![0.0; n]; }
                };
                match f.call1((xl, t)) {
                    Ok(r) => r.extract::<Vec<f64>>().unwrap_or_else(|_| vec![0.0; n]),
                    Err(e) => { report_callback_error(py, e); vec![0.0; n] }
                }
            })
        })
    };

    let factory = crate::solvers::factories::factory_from_name(solver_name, tolerance_lte_abs, tolerance_lte_rel)
        .ok_or_else(|| PyValueError::new_err(crate::solvers::factories::unknown_solver_message(solver_name)))?;
    let mut solver = factory(&iv);
    let is_implicit = solver.is_implicit;

    let (times, states) = if is_implicit {
        let jac: Box<dyn Fn(&[f64], f64) -> Jacobian> = if let Some(ref g) = graph {
            // Only ∂f/∂x is needed for the implicit-stage Newton step; the full
            // Jacobian (which would also include ∂f/∂t) produces a rectangular
            // n×(n+1) matrix that downstream expects to be square n×n.
            let mut jac_graph = crate::ssa::autodiff::jacobian_wrt_slot(g, "x")
                .ok_or_else(|| PyValueError::new_err(
                    "internal: 'x' slot missing in traced ODE signature"))?;
            crate::ssa::optimize::optimize(&mut jac_graph);
            let jac_interp = crate::ssa::tape::InterpretedFn::from_graph(jac_graph);
            let n = n_x;
            if n == 1 {
                Box::new(move |x: &[f64], t: f64| Jacobian::Scalar(jac_interp.call(&[x, &[t]])[0]))
            } else {
                Box::new(move |x: &[f64], t: f64| Jacobian::Matrix(jac_interp.call(&[x, &[t]]), n))
            }
        } else {
            let func_obj2 = func.clone().unbind();
            let n = n_x;
            Box::new(move |x: &[f64], t: f64| {
                let f_at = |xx: &[f64]| -> Vec<f64> {
                    Python::attach(|py| {
                        let f = func_obj2.bind(py);
                        let xl = match pyo3::types::PyList::new(py, xx) {
                            Ok(l) => l,
                            Err(e) => { report_callback_error(py, e); return vec![0.0; n]; }
                        };
                        match f.call1((xl, t)) {
                            Ok(r) => r.extract::<Vec<f64>>().unwrap_or_else(|_| vec![0.0; n]),
                            Err(e) => { report_callback_error(py, e); vec![0.0; n] }
                        }
                    })
                };
                let eps = crate::constants::PY_DYNSYS_JAC_FD_STEP;
                if n == 1 {
                    let h = (eps * x[0].abs()).max(eps);
                    Jacobian::Scalar(0.5 * (f_at(&[x[0]+h])[0] - f_at(&[x[0]-h])[0]) / h)
                } else {
                    let mut j = vec![0.0; n*n];
                    let mut xb = x.to_vec();
                    for col in 0..n {
                        let h = (eps * x[col].abs()).max(eps);
                        xb[col] = x[col]+h; let fp = f_at(&xb);
                        xb[col] = x[col]-h; let fm = f_at(&xb);
                        xb[col] = x[col];
                        for row in 0..n { j[row*n+col] = 0.5*(fp[row]-fm[row])/h; }
                    }
                    Jacobian::Matrix(j, n)
                }
            })
        };

        let mut opt = match optimizer {
            "anderson" => crate::optim::anderson::Optimizer::Anderson(
                crate::optim::anderson::Anderson::with_defaults()),
            "newton_anderson" => crate::optim::anderson::Optimizer::NewtonAnderson(
                crate::optim::anderson::NewtonAnderson::with_defaults()),
            _ => crate::optim::anderson::Optimizer::Newton(
                crate::optim::anderson::Newton::new()),
        };
        if let Some(m) = optimizer_history { opt.set_m(m); }
        ImplicitSolver::integrate(
            &mut solver, &|x, t| rhs(x, t), &|x, t| jac(x, t),
            t_start, t_stop, dt, dt_min, dt_max_val, adaptive,
            max_iterations, &mut opt,
        )
    } else {
        ExplicitSolver::integrate(
            &mut solver, &|x, t| rhs(x, t),
            t_start, t_stop, dt, dt_min, dt_max_val, adaptive,
        )
    };

    // Re-raise the original exception from a raising user ODE callback (issue
    // #28): the integration loop swallowed it via a fallback so the loop could
    // unwind cleanly; surface it now instead of returning a bogus trajectory.
    if let Some(err) = take_pending_error() {
        return Err(err);
    }

    let np = py.import("numpy")?;
    let t_arr = np.call_method1("array", (times,))?;
    let n_t = states.len();
    let flat: Vec<f64> = states.into_iter().flat_map(|row| row.into_iter()).collect();
    let arr = np.call_method1("array", (flat,))?;
    let x_arr = arr.call_method1("reshape", ((n_t, n_x),))?;

    Ok((t_arr.unbind(), x_arr.unbind()))
}

// ======================================================================================
// Solver classes — with integrate() classmethod
// ======================================================================================

macro_rules! solver_class {
    ($name:ident, $pyname:expr, $doc:expr) => {
        #[pyclass(name = $pyname, subclass)]
        #[doc = $doc]
        pub struct $name;
        #[pymethods]
        impl $name {
            #[new]
            fn new() -> Self { Self }

            /// Integrate an ODE from time_start to time_end.
            #[classmethod]
            #[pyo3(signature = (func, initial_value,
                                time_start=0.0, time_end=1.0,
                                dt=0.01, dt_min=1e-16, dt_max=None, adaptive=true,
                                tolerance_lte_abs=crate::constants::SOL_TOLERANCE_LTE_ABS,
                                tolerance_lte_rel=crate::constants::SOL_TOLERANCE_LTE_REL,
                                max_iterations=200,
                                optimizer="newton", optimizer_history=None))]
            fn integrate(
                _cls: &Bound<'_, pyo3::types::PyType>,
                py: Python<'_>,
                func: &Bound<'_, PyAny>,
                initial_value: &Bound<'_, PyAny>,
                time_start: f64,
                time_end: f64,
                dt: f64,
                dt_min: f64,
                dt_max: Option<f64>,
                adaptive: bool,
                tolerance_lte_abs: f64,
                tolerance_lte_rel: f64,
                max_iterations: usize,
                optimizer: &str,
                optimizer_history: Option<usize>,
            ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
                solver_integrate_impl(
                    py, $pyname, func, initial_value,
                    time_start, time_end, dt, dt_min, dt_max, adaptive,
                    tolerance_lte_abs, tolerance_lte_rel,
                    max_iterations, optimizer, optimizer_history,
                )
            }
        }
    };
}

solver_class!(PySSPRK22, "SSPRK22",
    "Strong Stability Preserving Runge-Kutta, 2 stages, 2nd order (Heun). Explicit, fixed step; the default solver for non-stiff problems.");
solver_class!(PySSPRK33, "SSPRK33",
    "Strong Stability Preserving Runge-Kutta, 3 stages, 3rd order (Shu-Osher). Explicit, fixed step, non-stiff.");
solver_class!(PySSPRK34, "SSPRK34",
    "Strong Stability Preserving Runge-Kutta, 4 stages, 3rd order, with an enlarged stability region. Explicit, fixed step, non-stiff.");
solver_class!(PyRK4, "RK4",
    "Classic Runge-Kutta, 4 stages, 4th order. Explicit, fixed step; a robust general-purpose non-stiff integrator.");
solver_class!(PyEUF, "EUF",
    "Explicit (forward) Euler, 1st order. The simplest explicit integrator; fixed step, non-stiff.");
solver_class!(PyEUB, "EUB",
    "Implicit (backward) Euler, 1st order. L-stable, fixed step; a robust choice for stiff problems.");
solver_class!(PyRKF21, "RKF21",
    "Runge-Kutta-Fehlberg embedded pair, orders 2 and 1, for adaptive step-size control. Explicit, non-stiff.");
solver_class!(PyRKBS32, "RKBS32",
    "Bogacki-Shampine embedded pair, orders 3 and 2 (the method behind MATLAB's ode23). Explicit, adaptive, non-stiff.");
solver_class!(PyRKF45, "RKF45",
    "Runge-Kutta-Fehlberg embedded pair, orders 4 and 5, for adaptive step-size control. Explicit, non-stiff.");
solver_class!(PyRKCK54, "RKCK54",
    "Cash-Karp embedded Runge-Kutta pair, orders 5 and 4, for adaptive step-size control. Explicit, non-stiff.");
solver_class!(PyRKDP54, "RKDP54",
    "Dormand-Prince embedded Runge-Kutta pair, orders 5 and 4 (the method behind MATLAB's ode45). Explicit, adaptive; the default general-purpose non-stiff solver.");
solver_class!(PyRKV65, "RKV65",
    "Verner embedded Runge-Kutta pair, orders 6 and 5, for adaptive step-size control. Explicit, non-stiff, higher accuracy.");
solver_class!(PyRKF78, "RKF78",
    "Runge-Kutta-Fehlberg embedded pair, orders 7 and 8, for adaptive step-size control. Explicit, non-stiff, high accuracy.");
solver_class!(PyRKDP87, "RKDP87",
    "Dormand-Prince embedded Runge-Kutta pair, orders 8 and 7, for adaptive step-size control. Explicit, non-stiff, very high accuracy.");
solver_class!(PyDIRK2, "DIRK2",
    "Diagonally Implicit Runge-Kutta, 2nd order. L-stable; solves an implicit system per stage, for stiff problems.");
solver_class!(PyDIRK3, "DIRK3",
    "Diagonally Implicit Runge-Kutta, 3rd order. L-stable, for stiff problems.");
solver_class!(PyESDIRK4, "ESDIRK4",
    "Explicit-first-stage Singly Diagonally Implicit Runge-Kutta, 4th order. Stiffly accurate and L-stable, for stiff problems.");
solver_class!(PyESDIRK32, "ESDIRK32",
    "Explicit-first-stage Singly Diagonally Implicit Runge-Kutta embedded pair, orders 3 and 2, for adaptive step-size control. L-stable, for stiff problems.");
solver_class!(PyESDIRK43, "ESDIRK43",
    "Explicit-first-stage Singly Diagonally Implicit Runge-Kutta embedded pair, orders 4 and 3, for adaptive step-size control. L-stable, for stiff problems.");
solver_class!(PyESDIRK54, "ESDIRK54",
    "Explicit-first-stage Singly Diagonally Implicit Runge-Kutta embedded pair, orders 5 and 4, for adaptive step-size control. L-stable, for stiff problems.");
solver_class!(PyGEAR52A, "GEAR52A",
    "Gear-type backward differentiation formula (BDF) with adaptive order (up to 5) and adaptive step size. Multi-step and L-stable, for stiff problems.");
solver_class!(PySteadyState, "SteadyState",
    "Steady-state (DC operating point) solver. Finds the equilibrium of the system by root-finding on the residual rather than time integration.");

// ======================================================================================
// PyCompiledSimulation — Python handle to a statically compiled model
// ======================================================================================

/// A statically compiled simulation: the whole block diagram fused into one
/// ``dX/dt = F(X, t)`` tape over a single global state vector, with no per-block
/// dispatch and no Python in the inner loop. Produced by ``Simulation.compile()``.
///
/// Topology and equations are frozen at compile time (you cannot add, remove or
/// rewire blocks), but the solver and its tolerances (``set_solver``), the step
/// ``dt``, the run horizon, and every parameter (``set_param``; parameters are
/// graph inputs, not folded constants) stay adjustable between runs. Implicit
/// solvers use a compile-time native symbolic Jacobian ``∂F/∂x`` — built lazily,
/// cached, no callback and no finite differences.
///
/// ``run`` returns the accumulated trajectory directly as ``(times, states,
/// recordings)`` — ``recordings`` maps each sink label to its trace — and the
/// same data is also available via the ``times`` / ``states`` / ``recordings``
/// properties. Block-internal events (zero-cross / schedule / condition) drive an
/// event-aware run loop; simulation-level events and opaque blocks are rejected at
/// compile time. Set ``log = True`` for the same progress logging a
/// ``Simulation`` run prints.
///
/// ``run`` / ``run_batch`` release the GIL for the whole integration (the
/// tape is pure Rust, no Python in the loop), so background Python threads
/// keep running while a compiled model integrates. The object itself stays
/// bound to the thread that created it (`unsendable`): the inner scratch
/// buffers use `RefCell`, which is `Send` but not `Sync`, and pyo3 requires
/// `Sync` for thread-portable classes.
#[pyclass(name = "CompiledSimulation", unsendable)]
pub struct PyCompiledSimulation {
    inner: crate::compile::CompiledSimulation,
}

#[pymethods]
impl PyCompiledSimulation {
    /// Number of global continuous-state elements.
    #[getter]
    fn n_state(&self) -> usize { self.inner.n_state }

    /// Number of global discrete-memory elements (event/discrete blocks).
    #[getter]
    fn n_mem(&self) -> usize { self.inner.n_mem }

    /// Initial global state (block order).
    #[getter]
    fn x0(&self) -> Vec<f64> { self.inner.x0.clone() }

    /// `"<block>.<state>"` label per global state element.
    #[getter]
    fn state_labels(&self) -> Vec<String> { self.inner.state_labels.clone() }

    /// `"<block>.<param>"` labels of the live (retunable) parameters.
    #[getter]
    fn param_names(&self) -> Vec<String> { self.inner.param_names().to_vec() }

    /// Current parameter values (parallel to `param_names`).
    #[getter]
    fn params(&self) -> Vec<f64> { self.inner.params().to_vec() }

    /// Evaluate the fused derivative ``dX/dt = F(X, t)`` at an arbitrary state and
    /// time, using the current discrete memory and parameter values. ``x`` must
    /// have length ``n_state``; returns the derivative vector (same length). This
    /// is the single tape the solver calls in its inner loop — exposed so you can
    /// probe the right-hand side directly (e.g. to check a fixed point or feed an
    /// external integrator). Does not advance the simulation.
    fn deriv(&self, x: Vec<f64>, t: f64) -> Vec<f64> {
        self.inner.deriv(&x, t)
    }

    /// Retune a live (baked) parameter by its ``"<block>.<param>"`` label, in both
    /// the derivative and the signal-recording tapes. Parameters stay editable
    /// after compilation (they are graph inputs, not folded constants) — topology
    /// and equations are frozen, parameter *values* are not. Returns ``False`` if
    /// the label is unknown (see ``param_names``).
    fn set_param(&mut self, name: &str, value: f64) -> bool {
        self.inner.set_param(name, value)
    }

    /// Labels of the recorded signals (the inputs each sink/scope observes),
    /// parallel to ``eval_taps`` outputs and the keys of ``recordings``.
    #[getter]
    fn tap_labels(&self) -> Vec<String> { self.inner.tap_labels().to_vec() }

    /// Evaluate the recorded (tapped) signals at one ``(x, t)`` point under the
    /// current discrete memory — the observed-signal analogue of ``deriv``.
    /// Returns one value per ``tap_labels`` entry. Used internally to build
    /// ``recordings``; exposed for sampling a signal off the trajectory.
    fn eval_taps(&self, x: Vec<f64>, t: f64) -> Vec<f64> {
        self.inner.eval_taps(&x, t)
    }

    /// Nominal timestep (initial step for adaptive, fixed step otherwise).
    #[getter]
    fn get_dt(&self) -> f64 { self.inner.dt }
    #[setter]
    fn set_dt(&mut self, dt: f64) { self.inner.dt = dt; }

    /// Current solver method name.
    #[getter]
    fn solver(&self) -> String { self.inner.solver.clone() }

    /// LTE tolerances.
    #[getter]
    fn get_tolerance_lte_abs(&self) -> f64 { self.inner.atol }
    #[setter]
    fn set_tolerance_lte_abs(&mut self, v: f64) { self.inner.atol = v; }
    #[getter]
    fn get_tolerance_lte_rel(&self) -> f64 { self.inner.rtol }
    #[setter]
    fn set_tolerance_lte_rel(&mut self, v: f64) { self.inner.rtol = v; }

    /// Change the integration method, mirroring `Simulation.set_solver`: pass a
    /// solver class (e.g. `RKDP54`, `ESDIRK43`) and optional tolerances. Implicit
    /// methods use the compile-time native symbolic Jacobian.
    #[pyo3(signature = (solver, tolerance_lte_abs=None, tolerance_lte_rel=None))]
    fn set_solver(
        &mut self,
        solver: &Bound<'_, PyAny>,
        tolerance_lte_abs: Option<f64>,
        tolerance_lte_rel: Option<f64>,
    ) -> PyResult<()> {
        let name: String = solver
            .getattr("__name__")
            .map_err(|_| PyValueError::new_err("solver must be a solver class (e.g. RKDP54, ESDIRK43)"))?
            .extract()?;
        if crate::solvers::factories::factory_from_name(
            &name,
            crate::constants::SOL_TOLERANCE_LTE_ABS,
            crate::constants::SOL_TOLERANCE_LTE_REL,
        ).is_none() {
            return Err(PyValueError::new_err(
                crate::solvers::factories::unknown_solver_message(&name),
            ));
        }
        self.inner.set_solver(
            name,
            tolerance_lte_abs.unwrap_or(self.inner.atol),
            tolerance_lte_rel.unwrap_or(self.inner.rtol),
        );
        Ok(())
    }

    /// Run-time logging toggle (mirrors ``Simulation(log=...)``). Setting it
    /// ``True`` logs a one-line ``COMPILE`` summary and makes each subsequent
    /// ``run`` print the same ``SOLVER`` / ``TRANSIENT`` progress a ``Simulation``
    /// run does; ``False`` silences it. Inherited from the source simulation at
    /// ``compile()`` time; assign here to override.
    #[getter]
    fn get_log(&self) -> bool { self.inner.logging_enabled() }
    #[setter]
    fn set_log(&mut self, enabled: bool) { self.inner.set_logging(enabled); }

    /// Current global continuous state vector ``X`` (in block/state order; advanced
    /// by ``run``, rewound by ``reset``). Read-only snapshot.
    #[getter]
    fn state(&self) -> Vec<f64> { self.inner.state().to_vec() }

    /// Current simulation time. The getter reports where the next ``run`` resumes;
    /// the setter moves time without touching the state or the recorded trajectory
    /// (use ``reset`` to rewind state and clear recordings).
    #[getter]
    fn get_time(&self) -> f64 { self.inner.time() }
    #[setter]
    fn set_time(&mut self, t: f64) { self.inner.set_time(t); }

    /// Recorded sample times and global-state trajectory (accumulated across
    /// `run` calls, cleared by `reset`).
    #[getter]
    fn times(&self) -> Vec<f64> { self.inner.times().to_vec() }
    #[getter]
    fn states(&self) -> Vec<Vec<f64>> { self.inner.states() }

    /// Downsample the recorded trajectory: keep one of every `stride` accepted
    /// steps (the initial point and final state are always kept). `1` records
    /// every step (issue #44).
    #[setter]
    fn set_output_stride(&mut self, stride: usize) { self.inner.set_output_stride(stride); }
    #[getter]
    fn output_stride(&self) -> usize { self.inner.output_stride() }

    /// Cap the number of recorded samples for very long runs (bounded memory);
    /// `None` is unbounded (issue #44).
    #[setter]
    fn set_max_samples(&mut self, max_samples: Option<usize>) { self.inner.set_max_samples(max_samples); }
    #[getter]
    fn max_samples(&self) -> Option<usize> { self.inner.max_samples() }

    /// Recorded observed-signal traces, mapping each tap label to its time
    /// series (aligned with `times`) — the static-compile analogue of scope data.
    #[getter]
    fn recordings(&self) -> HashMap<String, Vec<f64>> {
        self.inner.tap_labels().iter().cloned().zip(self.inner.recordings()).collect()
    }

    /// Rewind to the compile-time initial condition and clear history: state ``X``
    /// and discrete memory ``M`` return to their initial values, the clock is set
    /// to ``time``, and the recorded trajectory is reset to a single point. The
    /// solver, ``dt``, tolerances and parameters are left untouched. ``run(...,
    /// reset=True)`` (the default) does this for you.
    #[pyo3(signature = (time=0.0))]
    fn reset(&mut self, time: f64) {
        self.inner.reset(time);
    }

    /// Advance the simulation by `duration`, recording the trajectory. Mirrors
    /// `Simulation.run`: stateful (continues from the current state/time) unless
    /// `reset` is set. Returns `(times, states, recordings)` of the accumulated
    /// trajectory — `states[i]` is the global state at `times[i]`, `recordings`
    /// maps each observed-signal label to its trace, ready to plot. Solver and
    /// step are taken from the `solver` / `dt` attributes. The GIL is released
    /// for the whole integration (the tape is pure Rust), so other Python
    /// threads keep running.
    #[pyo3(signature = (duration, reset=true, adaptive=true))]
    fn run(
        &mut self,
        py: Python<'_>,
        duration: f64,
        reset: bool,
        adaptive: bool,
    ) -> (Vec<f64>, Vec<Vec<f64>>, HashMap<String, Vec<f64>>) {
        // The compiled tape carries no Python callback, so the whole
        // integration is pure Rust — release the GIL for its duration so
        // other Python threads (UI, Jupyter heartbeat, loggers) keep running.
        let inner = &mut self.inner;
        py.detach(|| inner.run(duration, reset, adaptive));
        let recordings: HashMap<String, Vec<f64>> = self
            .inner
            .tap_labels()
            .iter()
            .cloned()
            .zip(self.inner.recordings())
            .collect();
        (self.inner.times().to_vec(), self.inner.states(), recordings)
    }

    /// Run a parameter sweep / Monte Carlo batch in parallel across CPU cores
    /// (issue #45). `param_sets` is a list of ``{param_name: value}`` override
    /// maps (one per run); each run starts from the compile-time initial state.
    /// Returns the final global state of each run, in input order — identical to
    /// running the sweep serially, but scaled across a thread pool.
    ///
    /// The native tape carries no Python callback, so the rayon workers run truly
    /// in parallel — and the calling thread releases the GIL for the whole
    /// sweep, so other Python threads keep running. For long trajectories, set
    /// `output_stride` / `max_samples` first to bound per-run memory.
    #[cfg(feature = "parallel")]
    #[pyo3(signature = (param_sets, duration, adaptive=true))]
    fn run_batch(
        &self,
        py: Python<'_>,
        param_sets: Vec<HashMap<String, f64>>,
        duration: f64,
        adaptive: bool,
    ) -> Vec<Vec<f64>> {
        let sets: Vec<Vec<(String, f64)>> = param_sets
            .into_iter()
            .map(|m| m.into_iter().collect())
            .collect();
        // Release the GIL for the whole sweep — the rayon workers are pure
        // Rust and the calling thread otherwise blocks every other Python
        // thread while it waits. `CompiledSimulation` is `Send` but not
        // `Sync` (interior RefCell scratch), so move an owned clone into the
        // detached closure instead of borrowing `self` across it; one clone
        // is noise next to the batch integration itself.
        let base = self.inner.clone();
        py.detach(move || base.run_batch(&sets, duration, adaptive))
    }
}
