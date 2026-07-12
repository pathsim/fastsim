// FMU block constructors: ModelExchangeFMU and CoSimulationFMU.
//
// An FMU (Functional Mock-up Unit) appears to FastSim as a regular Block with
// input/output ports. We wrap a loaded `Instance<Me>` or `Instance<Cs>` in an
// interior-mutable backend (`Rc<FastCell<Backend>>`) that is shared between:
//   - the block's `f_dyn`/`f_alg`/`update_fn`/`sample_fn` closures
//   - state-event and time-event callbacks (`ZeroCrossing.func_act`,
//     `Schedule.func_act`, `ScheduleList.func_act`)
//
// ## Error policy in hot-path closures
//
// The `Fn`/`FnMut` signatures of FastSim block and event callbacks do not
// return `Result`, so FMI calls inside them cannot propagate errors back to
// the simulation loop. We intentionally swallow their `Result` via
// `let _ = backend.instance.<call>()`: an FMU that reports an error during
// a step has logged via the logger callback (see `fmi::callbacks`) and has
// left its state inconsistent; crashing the whole simulation is worse than
// emitting an error log and letting the outer loop continue. Init-time
// calls *do* propagate via `?` because they run before the block is wired up.
//
// ## References (audited projects)
//   - PathSim    `pathsim/blocks/fmu.py`          — Python API shape
//   - fmpy       `src/fmpy/fmi3.py`, simulation.py — lifecycle sequence
//   - Reference- `fmusim/FMI3MESimulation.c`       — ME loop structure
//     FMUs      `fmusim/FMI3CSSimulation.c`       — CS loop structure

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use crate::constants::FMI_INITIALIZATION_TOL;
use crate::events::schedule::{Schedule, ScheduleList};
use crate::utils::fastcell::FastCell;
use crate::utils::register::Register;

use crate::fmi::bindings::fmi3ValueReference;
use crate::fmi::instance::{Cs, DiscreteStateUpdate, Instance, Me};
use crate::fmi::model_description::{ModelDescription, StartValue, VarType};
use crate::fmi::unzip::FmuArchive;
use crate::fmi::Result;

use super::block::{Block, BlockRef};

// =========================================================================
// Common backend helpers
// =========================================================================

fn collect_overrides(
    md: &ModelDescription,
    overrides: &HashMap<String, f64>,
) -> Result<(Vec<fmi3ValueReference>, Vec<f64>)> {
    let mut vrs = Vec::new();
    let mut vals = Vec::new();
    for (name, val) in overrides {
        let v = md
            .variable_by_name(name)
            .ok_or_else(|| crate::fmi::FmiError::UnknownVariable(name.clone()))?;
        if v.var_type == VarType::Float64 {
            vrs.push(v.value_reference);
            vals.push(*val);
        }
    }
    Ok((vrs, vals))
}

/// Baseline start values from the XML to apply during initialization.
///
/// Per FMI 3.0 §2.4.7.5, `variability="constant"` variables have fixed values
/// and cannot be set. We also skip `initial="calculated"` vars — the FMU
/// computes them itself. That leaves inputs, parameters, and `initial=exact`
/// (or `approx`) variables with explicit `start` attributes.
fn declared_float64_starts(md: &ModelDescription) -> (Vec<fmi3ValueReference>, Vec<f64>) {
    use crate::fmi::model_description::{Initial, Variability};
    let mut vrs = Vec::new();
    let mut vals = Vec::new();
    for v in &md.variables {
        if v.var_type != VarType::Float64 {
            continue;
        }
        if v.variability == Variability::Constant {
            continue;
        }
        if matches!(v.initial, Some(Initial::Calculated)) {
            continue;
        }
        let Some(sv) = &v.start else { continue };
        let x = match sv {
            StartValue::Float64(x) => *x,
            other => match other.as_f64() {
                Some(x) => x,
                None => continue,
            },
        };
        vrs.push(v.value_reference);
        vals.push(x);
    }
    (vrs, vals)
}

/// Drain the `UpdateDiscreteStates` fixed-point loop. FMI 3.0 §3.2.1 requires
/// iterating until `discreteStatesNeedUpdate=false`. Returns the last update so
/// the caller can inspect `next_event_time` and `values_changed`.
fn drain_discrete_state_updates<K>(inst: &Instance<K>) -> Result<DiscreteStateUpdate> {
    loop {
        let u = inst.update_discrete_states()?;
        if u.terminate_simulation {
            return Err(crate::fmi::FmiError::ModelDescription(
                "FMU requested termination during initialization".into(),
            ));
        }
        if !u.discrete_states_need_update {
            return Ok(u);
        }
    }
}

/// Shared FMI 3.0 initialization sequence used by both ME and CS block
/// constructors: `EnterInit → apply declared starts → apply user overrides
/// → ExitInit → drain UpdateDiscreteStates`. Returns the final discrete-state
/// update so the caller can seed time events from `next_event_time_defined`.
///
/// Ref: PathSim `blocks/fmu.py::_initialize`, Reference-FMUs
/// `fmusim/FMI3MESimulation.c:53-96` and `FMI3CSSimulation.c:70-123`.
fn run_initialization<K>(
    inst: &Instance<K>,
    md: &ModelDescription,
    start_values: &Option<HashMap<String, f64>>,
    tolerance: Option<f64>,
) -> Result<DiscreteStateUpdate> {
    inst.enter_initialization_mode(tolerance, 0.0, None)?;

    let (decl_vrs, decl_vals) = declared_float64_starts(md);
    if !decl_vrs.is_empty() {
        inst.set_float64(&decl_vrs, &decl_vals)?;
    }
    if let Some(overrides) = start_values {
        if !overrides.is_empty() {
            let (vrs, vals) = collect_overrides(md, overrides)?;
            if !vrs.is_empty() {
                inst.set_float64(&vrs, &vals)?;
            }
        }
    }

    inst.exit_initialization_mode()?;
    drain_discrete_state_updates(inst)
}

// =========================================================================
// ModelExchangeFMU
// =========================================================================

/// FMI 3.0 Model-Exchange backend — owns the instantiated FMU plus cached
/// metadata needed by block/event callbacks. Shared between closures via
/// `Rc<FastCell<MeBackend>>`; `block` and `time_events` are filled in after
/// the `Block` is wrapped in `Rc`, closing the back-reference loop used by
/// `handle_event` to mutate the block's engine and schedule new time events.
pub struct MeBackend {
    pub instance: Instance<Me>,
    pub md: ModelDescription,
    pub input_vrs: Vec<fmi3ValueReference>,
    pub output_vrs: Vec<fmi3ValueReference>,
    pub state_vrs: Vec<fmi3ValueReference>,
    pub n_event_indicators: usize,
    /// Back-reference to the owning block, set after block construction.
    /// Used by `handle_event` to call `engine.set(x_new)` when the FMU
    /// signals `values_changed` from `UpdateDiscreteStates`.
    pub block: Option<BlockRef>,
    /// ScheduleList that carries FMU-announced time events; filled in after
    /// `Block` is wrapped in `Rc`. `handle_event` appends `next_event_time`
    /// here when the FMU reports one.
    pub time_events: Option<Rc<FastCell<ScheduleList>>>,
    /// Pre-allocated scratch for `GetEventIndicators`. Sized to
    /// `n_event_indicators` at construction; re-read on every ZeroCrossing
    /// `func_evt` call without allocating.
    pub ei_buf: Vec<f64>,
    /// Pre-allocated scratch for `GetFloat64(state_vrs)` used by
    /// `handle_event` when the FMU signals `values_changed`.
    pub state_buf: Vec<f64>,
    /// Pre-allocated seed vector for `fmi3GetDirectionalDerivative` in
    /// `jac_dyn`. Sized to `n_states` at construction; the closure flips one
    /// element to 1.0 and back per column without allocating.
    pub jac_seed_buf: Vec<f64>,
    /// Pre-allocated column buffer for the sensitivity output of each
    /// directional-derivative call.
    pub jac_col_buf: Vec<f64>,
    /// Derivative VRs in the order their corresponding states appear in
    /// `state_vrs`. Captured once at construction so the `jac_dyn` closure
    /// has immediate access without hitting `ModelStructure`.
    pub state_deriv_vrs: Vec<fmi3ValueReference>,
}

impl MeBackend {
    fn apply_inputs(&self, u: &[f64]) -> Result<()> {
        if self.input_vrs.is_empty() {
            return Ok(());
        }
        let n = self.input_vrs.len().min(u.len());
        self.instance.set_float64(&self.input_vrs[..n], &u[..n])
    }

    /// Run the full event-handling sequence in response to a detected state
    /// or time event:
    /// `EnterEventMode → drain UpdateDiscreteStates → EnterContinuousTimeMode`
    /// plus engine-state reset if `values_changed` and time-event insertion
    /// if `next_event_time_defined`.
    ///
    /// Ref: `reference-fmus/fmusim/FMI3MESimulation.c:186-227` + PathSim
    /// `blocks/fmu.py::ModelExchangeFMU._handle_event`.
    fn handle_event(&mut self) {
        if self.instance.enter_event_mode().is_err() {
            return;
        }
        let u = match drain_discrete_state_updates(&self.instance) {
            Ok(u) => u,
            Err(_) => return,
        };
        let _ = self.instance.enter_continuous_time_mode();

        if u.values_changed && !self.state_vrs.is_empty() {
            // Re-use the pre-allocated scratch buffer.
            if self
                .instance
                .get_float64(&self.state_vrs, &mut self.state_buf)
                .is_ok()
            {
                if let Some(blk) = &self.block {
                    if let Some(engine) = blk.borrow_mut().engine.as_mut() {
                        engine.set(&self.state_buf);
                    }
                }
            }
        }

        if u.next_event_time_defined {
            self.insert_time_event(u.next_event_time);
        }
    }

    /// `bisect.insort` analog — append a FMU-announced time event into
    /// `time_events.times_evt` in ascending order, deduplicating within
    /// tolerance.
    fn insert_time_event(&self, t: f64) {
        let Some(tel) = &self.time_events else { return };
        let tel = tel.borrow_mut();
        let tol = crate::constants::TOLERANCE;
        let pos = tel.times_evt.partition_point(|&v| v < t - tol);
        if pos < tel.times_evt.len() && (tel.times_evt[pos] - t).abs() <= tol {
            return;
        }
        tel.times_evt.insert(pos, t);
    }
}

/// Construct a Model-Exchange FMU block.
///
/// Mirrors PathSim's `ModelExchangeFMU(fmu_path, instance_name, start_values,
/// tolerance, verbose)` signature.
pub fn model_exchange_fmu(
    fmu_path: impl AsRef<Path>,
    instance_name: &str,
    start_values: Option<HashMap<String, f64>>,
    tolerance: f64,
    verbose: bool,
) -> Result<BlockRef> {
    // --- 1. extract + parse + instantiate ---
    let archive = FmuArchive::extract(fmu_path.as_ref())?;
    let md = ModelDescription::from_file(archive.model_description())?;
    let inst = Instance::<Me>::new_model_exchange(archive, &md, instance_name, verbose)?;

    // --- 2. shared init sequence + ME-specific transition to continuous time ---
    let init_update = run_initialization(&inst, &md, &start_values, Some(tolerance))?;
    inst.enter_continuous_time_mode()?;

    // --- 3. discover port / state VRs ---
    let input_vrs: Vec<_> = md
        .inputs()
        .filter(|v| v.var_type == VarType::Float64)
        .map(|v| v.value_reference)
        .collect();
    let output_vrs: Vec<_> = md
        .outputs()
        .filter(|v| v.var_type == VarType::Float64)
        .map(|v| v.value_reference)
        .collect();
    let state_vrs: Vec<_> = md
        .continuous_states()
        .iter()
        .map(|v| v.value_reference)
        .collect();
    let state_deriv_vrs: Vec<_> = md
        .model_structure
        .continuous_state_derivatives
        .clone();
    let n_event_indicators = md.n_event_indicators();

    // --- 4. initial state from FMU (post-init) via GetFloat64 on state VRs ---
    let mut initial_state = vec![0.0; state_vrs.len()];
    if !state_vrs.is_empty() {
        inst.get_float64(&state_vrs, &mut initial_state)?;
    }

    // --- 5. assemble backend + Block ---
    let ei_buf = vec![0.0; n_event_indicators];
    let state_buf = vec![0.0; state_vrs.len()];
    let jac_seed_buf = vec![0.0; state_vrs.len()];
    let jac_col_buf = vec![0.0; state_vrs.len()];
    let backend = Rc::new(FastCell::new(MeBackend {
        instance: inst,
        md,
        input_vrs,
        output_vrs,
        state_vrs,
        n_event_indicators,
        block: None,
        time_events: None,
        ei_buf,
        state_buf,
        jac_seed_buf,
        jac_col_buf,
        state_deriv_vrs,
    }));

    let mut b = Block::default_block();
    b.type_name = "ModelExchangeFMU";
    b.role = crate::blocks::block::BlockRole {
        is_dyn: true, is_src: false, is_rec: false,
    };
    b.opaque_feedthrough = true; // opaque FMU: conservatively assume y-on-u feedthrough
    b.initial_value = Some(initial_state.clone());
    b.engine = Some(crate::solvers::solver::Solver::with_defaults(&initial_state));

    let n_in = { backend.borrow().input_vrs.len() };
    let n_out = { backend.borrow().output_vrs.len() };
    b.inputs = Register::new(Some(n_in), None);
    b.outputs = Register::new(Some(n_out), None);

    // f_dyn: set_time + set_states + set_inputs + get_derivatives
    let be = backend.clone();
    b.f_dyn = Some(Box::new(move |x, u, t, out| {
        let backend = be.borrow_mut();
        let _ = backend.instance.set_time(t);
        if !backend.state_vrs.is_empty() {
            let _ = backend.instance.set_continuous_states(x);
        }
        let _ = backend.apply_inputs(u);
        let n = backend.state_vrs.len();
        out.resize(n, 0.0);
        if n > 0 {
            let _ = backend.instance.get_continuous_state_derivatives(out);
        }
    }));

    // f_alg: set_time + set_states + set_inputs + get_outputs
    let be = backend.clone();
    b.f_alg = Some(Box::new(move |x, u, t, out| {
        let backend = be.borrow_mut();
        let _ = backend.instance.set_time(t);
        if !backend.state_vrs.is_empty() {
            let _ = backend.instance.set_continuous_states(x);
        }
        let _ = backend.apply_inputs(u);
        let n_out = backend.output_vrs.len();
        out.resize(n_out, 0.0);
        if n_out > 0 {
            let _ = backend.instance.get_float64(&backend.output_vrs, out);
        }
    }));

    // jac_dyn: if the FMU advertises `providesDirectionalDerivatives` AND
    // exports the symbol, assemble ∂ẋ/∂x column-by-column via directional
    // derivatives (seed = e_j, column j of the Jacobian).  Output layout is
    // row-major: `jac[i * n + j] = ∂ẋ_i/∂x_j` — matches the JIT AD convention
    // in `src/jit/autodiff.rs` so implicit solvers treat both paths uniformly.
    // Absent/erroring → leave `jac_dyn` None → falls back to the FD Jacobian
    // in `Block::compute_jacobian`.
    //
    // Scratch buffers (`jac_seed_buf`, `jac_col_buf`) live on `MeBackend`
    // sized at construction, so this closure allocates nothing in steady
    // state.
    let (provides_dd, has_dd_symbol, n_states, derivs_match) = {
        let be = backend.borrow();
        (
            be.md.model_exchange.as_ref()
                .map(|me| me.provides_directional_derivatives).unwrap_or(false),
            be.instance.supports_directional_derivatives(),
            be.state_vrs.len(),
            be.state_deriv_vrs.len() == be.state_vrs.len(),
        )
    };
    if provides_dd && has_dd_symbol && n_states > 0 && derivs_match {
        let be = backend.clone();
        b.jac_dyn = Some(Box::new(move |x, u, t, out| {
            let backend = be.borrow_mut();
            let _ = backend.instance.set_time(t);
            let _ = backend.instance.set_continuous_states(x);
            let _ = backend.apply_inputs(u);
            let n = backend.state_vrs.len();
            out.clear();
            out.resize(n * n, 0.0);
            if n == 0 { return; }
            // Split-borrow across disjoint fields of `MeBackend`.
            let MeBackend {
                instance, state_vrs, state_deriv_vrs, jac_seed_buf, jac_col_buf, ..
            } = &mut *backend;
            for v in jac_seed_buf.iter_mut() { *v = 0.0; }
            for j in 0..n {
                jac_seed_buf[j] = 1.0;
                if instance.get_directional_derivative(
                    state_deriv_vrs, state_vrs, jac_seed_buf, jac_col_buf,
                ).is_ok() {
                    for i in 0..n { out[i * n + j] = jac_col_buf[i]; }
                }
                // On error: column stays zero (already initialized).  Newton
                // degrades gracefully to a partial Jacobian; FD fallback
                // isn't possible here without a re-entrant handle to `f_dyn`.
                jac_seed_buf[j] = 0.0;
            }
        }));
    }

    // --- 6. wrap block + install events -------------------------------
    let blk_ref: BlockRef = Rc::new(FastCell::new(b));

    // Time-event ScheduleList: starts empty. Call `handle_event` which will
    // pull/store new times into it during event resolution.
    let time_events = Rc::new(FastCell::new(ScheduleList::new(
        Vec::new(),
        None,
        tolerance,
    )));
    {
        let be = backend.clone();
        time_events.borrow_mut().func_act =
            Some(Box::new(move |_t| be.borrow_mut().handle_event()));
    }

    // Close the back-references so `handle_event` can reach the block's
    // engine and the time-event list. This creates the (existing fastsim)
    // closure/event Rc cycle documented in `constructors::sample_hold`.
    {
        let be = backend.borrow_mut();
        be.block = Some(blk_ref.clone());
        be.time_events = Some(time_events.clone());
    }

    // Seed initial time event if the FMU announced one during init.
    if init_update.next_event_time_defined {
        backend
            .borrow_mut()
            .insert_time_event(init_update.next_event_time);
    }
    // Register the time-event list as a block event.
    blk_ref
        .borrow_mut()
        .events
        .push(time_events as crate::simulation::SimEventRef);

    // ZeroCrossing per event indicator. `func_evt(t)` fetches the i-th
    // indicator from the FMU; `func_act(t)` runs the full event handler.
    let n_ei = backend.borrow().n_event_indicators;
    for i in 0..n_ei {
        let be_evt = backend.clone();
        let func_evt = move |t: f64| -> f64 {
            let be = be_evt.borrow_mut();
            let _ = be.instance.set_time(t);
            // Split-borrow: `instance` and `ei_buf` are disjoint fields, so
            // the borrow checker allows the simultaneous &self / &mut slice.
            if be.instance.get_event_indicators(&mut be.ei_buf).is_err() {
                return 0.0;
            }
            be.ei_buf[i]
        };

        let be_act = backend.clone();
        let func_act = Box::new(move |_t: f64| be_act.borrow_mut().handle_event());

        let zc = crate::events::zerocrossing::ZeroCrossing::new(
            func_evt,
            Some(func_act),
            tolerance,
        );
        blk_ref.borrow_mut().events.push(Rc::new(FastCell::new(zc)));
    }

    // Install sample_fn: after each successful RK timestep, call
    // CompletedIntegratorStep; if the FMU signals event mode, run the event
    // handler.  Ref: FMI3MESimulation.c:179-227.
    //
    // Skip entirely when the FMU declares `needsCompletedIntegratorStep=false`
    // (FMI 3.0 §3.2.2): the spec permits omitting the call in that case,
    // which saves one FFI round-trip per successful step.
    let needs_cis = backend.borrow().md
        .model_exchange.as_ref()
        .map(|me| me.needs_completed_integrator_step)
        .unwrap_or(true);
    if needs_cis {
        let be_sample = backend.clone();
        blk_ref.borrow_mut().sample_fn = Some(Box::new(move |_blk, _t, _dt| {
            let be = be_sample.borrow_mut();
            match be.instance.completed_integrator_step(true) {
                Ok(r) if r.enter_event_mode => be.handle_event(),
                _ => {}
            }
        }));
    }

    Ok(blk_ref)
}

// =========================================================================
// CoSimulationFMU
// =========================================================================

pub struct CsBackend {
    pub instance: Instance<Cs>,
    pub md: ModelDescription,
    pub input_vrs: Vec<fmi3ValueReference>,
    pub output_vrs: Vec<fmi3ValueReference>,
    /// `maxOutputDerivativeOrder` per output (same indexing as `output_vrs`).
    /// Used to decide whether to Taylor-interpolate outputs at block times
    /// between FMU communication points.
    pub output_max_orders: Vec<u32>,
    pub dt: f64,
    /// Whether the FMU was instantiated with `eventModeUsed = true`. Drives
    /// post-init transition and in-step event handling.
    pub event_mode_used: bool,
    /// Whether the FMU was instantiated with `earlyReturnAllowed = true`. When
    /// true, DoStep may return before the requested step completes; the
    /// Schedule callback loops until the target time is reached.
    pub early_return_allowed: bool,
    /// Set to true once the FMU signals `terminateSimulation` from DoStep or
    /// UpdateDiscreteStates. Subsequent DoStep/I/O calls are skipped.
    pub terminated: bool,
    /// The last time the FMU successfully advanced to. We pass this as
    /// `currentCommunicationPoint` on the next DoStep, decoupling the Schedule
    /// event cadence from the FMU's actual time cursor.
    pub current_time: f64,

    // ----- hot-path scratch buffers (sized at construction) --------------
    /// Block inputs flattened into a Vec<f64>, handed to `SetFloat64` in
    /// `update_fn`.
    pub input_buf: Vec<f64>,
    /// FMU outputs read via `GetFloat64` and optionally Taylor-extrapolated
    /// in `update_fn`.
    pub output_buf: Vec<f64>,
    /// Indices into `output_vrs` whose declared `maxOutputDerivativeOrder`
    /// reaches the current Taylor order; rebuilt in-place each call.
    pub taylor_idx: Vec<usize>,
    /// `output_vrs` subset for the current Taylor order.
    pub taylor_vrs: Vec<fmi3ValueReference>,
    /// Order vector passed to `GetOutputDerivatives` (all entries identical).
    pub taylor_orders: Vec<i32>,
    /// Derivative values returned by `GetOutputDerivatives`.
    pub taylor_deriv: Vec<f64>,
}

impl CsBackend {
    /// Run the CS event-handling sequence: EnterEventMode → drain
    /// UpdateDiscreteStates → EnterStepMode. Invoked when `fmi3DoStep`
    /// returns `eventHandlingNeeded` and `event_mode_used=true`.
    /// Ref: `reference-fmus/fmusim/FMI3CSSimulation.c:205-233`.
    fn handle_event(&mut self) {
        if self.instance.enter_event_mode().is_err() {
            return;
        }
        loop {
            let u = match self.instance.update_discrete_states() {
                Ok(u) => u,
                Err(_) => return,
            };
            if u.terminate_simulation {
                self.terminated = true;
                return;
            }
            if !u.discrete_states_need_update {
                break;
            }
        }
        let _ = self.instance.enter_step_mode();
    }
}

/// Construct a Co-Simulation FMU block.
///
/// Mirrors PathSim's `CoSimulationFMU(fmu_path, instance_name, start_values,
/// dt)` signature, plus a `verbose` flag (symmetric to `ModelExchangeFMU`).
/// If `dt` is `None`, the FMU's `DefaultExperiment.stepSize` is used;
/// otherwise an error is raised.
pub fn cosimulation_fmu(
    fmu_path: impl AsRef<Path>,
    instance_name: &str,
    start_values: Option<HashMap<String, f64>>,
    dt: Option<f64>,
    verbose: bool,
) -> Result<BlockRef> {
    // --- 1. extract + parse ---
    let archive = FmuArchive::extract(fmu_path.as_ref())?;
    let md = ModelDescription::from_file(archive.model_description())?;

    // Communication step: explicit dt overrides DefaultExperiment.stepSize.
    let dt = dt.or(md.default_experiment.step_size).ok_or_else(|| {
        crate::fmi::FmiError::ModelDescription(
            "no communication step size: neither `dt` argument nor DefaultExperiment.stepSize"
                .into(),
        )
    })?;

    // Auto-detect FMI 3.0 CS capabilities from ModelDescription:
    //   - event_mode_used: opt in if FMU supports it (handles state/time
    //     events detected during DoStep).
    //   - early_return_allowed: opt in if FMU might return early (allows
    //     precise advance up to an internal event instead of the requested
    //     step boundary — improves bounce/event precision).
    let (event_mode_used, early_return_allowed) = md
        .co_simulation
        .as_ref()
        .map(|cs| (cs.has_event_mode, cs.might_return_early_from_do_step))
        .unwrap_or((false, false));

    // --- 2. instantiate + shared init sequence ---
    let inst = Instance::<Cs>::new_co_simulation(
        archive,
        &md,
        instance_name,
        event_mode_used,
        early_return_allowed,
        verbose,
    )?;
    let _init_update = run_initialization(&inst, &md, &start_values, Some(FMI_INITIALIZATION_TOL))?;

    // After ExitInit the FMU is in Event Mode if `event_mode_used`, else in
    // Step Mode (FMI 3.0 §4.2.5). When in Event Mode, `run_initialization`
    // already drained discrete states — we only need the explicit transition.
    // Ref: reference-fmus/fmusim/FMI3CSSimulation.c:103-122.
    if event_mode_used {
        inst.enter_step_mode()?;
    }

    // --- 4. port discovery ---
    let input_vrs: Vec<_> = md
        .inputs()
        .filter(|v| v.var_type == VarType::Float64)
        .map(|v| v.value_reference)
        .collect();
    let outputs_f64: Vec<_> = md
        .outputs()
        .filter(|v| v.var_type == VarType::Float64)
        .collect();
    let output_vrs: Vec<_> = outputs_f64.iter().map(|v| v.value_reference).collect();
    let output_max_orders: Vec<u32> =
        outputs_f64.iter().map(|v| v.max_output_derivative_order).collect();

    // --- 5. assemble backend + Block ---
    let n_in = input_vrs.len();
    let n_out = output_vrs.len();
    let backend = Rc::new(FastCell::new(CsBackend {
        instance: inst,
        md,
        input_vrs,
        output_vrs,
        output_max_orders,
        dt,
        event_mode_used,
        early_return_allowed,
        terminated: false,
        current_time: 0.0,
        input_buf: vec![0.0; n_in],
        output_buf: vec![0.0; n_out],
        // Taylor scratch: worst case every output contributes at every order.
        taylor_idx: Vec::with_capacity(n_out),
        taylor_vrs: Vec::with_capacity(n_out),
        taylor_orders: Vec::with_capacity(n_out),
        taylor_deriv: Vec::with_capacity(n_out),
    }));

    let mut b = Block::default_block();
    b.type_name = "CoSimulationFMU";
    b.role = crate::blocks::block::BlockRole {
        is_dyn: false, is_src: false, is_rec: false,
    };
    b.opaque_feedthrough = true; // opaque FMU: conservatively assume y-on-u feedthrough
    b.inputs = Register::new(Some(n_in), None);
    b.outputs = Register::new(Some(n_out), None);

    // Populate initial outputs from the post-init FMU state via the
    // backend's pre-allocated output_buf.
    {
        let backend = backend.borrow_mut();
        if !backend.output_vrs.is_empty() {
            let _ = backend
                .instance
                .get_float64(&backend.output_vrs, &mut backend.output_buf);
            for (i, v) in backend.output_buf.iter().enumerate() {
                b.outputs.set_single(i, *v);
            }
        }
    }

    // Scheduled communication step. On each tick we drive the FMU up to
    // the target time `t`, possibly via multiple DoStep calls when the FMU
    // returns early (FMI 3.0 §4.2.4).
    //
    // Sequence per iteration:
    //   1. step_size = t - current_time
    //   2. DoStep(current_time, step_size)
    //   3. If earlyReturn: advance = last_successful_time - current_time
    //      else:           advance = step_size  (with earlyReturnAllowed=false
    //                                             the spec allows FMUs to skip
    //                                             writing last_successful_time)
    //   4. Handle terminate_simulation (sticky flag) and eventHandlingNeeded
    //      (EnterEventMode → drain UpdateDiscreteStates → EnterStepMode).
    //   5. Loop until current_time >= t or FMU fails to advance.
    let be = backend.clone();
    let schedule = Schedule::new(
        0.0,
        None,
        dt,
        Some(Box::new(move |t: f64| {
            let backend = be.borrow_mut();
            loop {
                if backend.terminated {
                    return;
                }
                let current = backend.current_time;
                let remaining = t - current;
                if remaining < crate::constants::TOLERANCE {
                    return;
                }
                let r = match backend.instance.do_step(current, remaining) {
                    Ok(r) => r,
                    Err(_) => return,
                };
                let advance = if backend.early_return_allowed && r.early_return {
                    (r.last_successful_time - current).max(0.0)
                } else {
                    remaining
                };
                backend.current_time = current + advance;

                if r.terminate_simulation {
                    backend.terminated = true;
                    return;
                }
                if r.event_handling_needed && backend.event_mode_used {
                    backend.handle_event();
                }

                // Safety: if the FMU neither terminated nor advanced, bail
                // instead of spinning forever.
                if advance < crate::constants::TOLERANCE {
                    return;
                }
            }
        })),
        crate::constants::TOLERANCE,
    );
    b.events.push(Rc::new(FastCell::new(schedule)));

    // update_fn: latch current inputs into FMU, read outputs back into block.
    // Runs on every FastSim update pass. When terminated, skip the I/O.
    //
    // When the FMU declares `maxOutputDerivativeOrder>0` for any output and
    // the block's current time `t` is ahead of the FMU's last communication
    // point (`backend.current_time`), we Taylor-extrapolate outputs using
    // `fmi3GetOutputDerivatives`:
    //     y(t) = y(tc) + Σ_{k=1..max_k}  y^(k)(tc) · (t-tc)^k / k!
    // Outputs with lower `max_output_derivative_order` contribute only up to
    // their declared order.
    //
    // All scratch buffers are fields of `CsBackend` pre-allocated at
    // construction, so this closure allocates nothing on the hot path.
    let be = backend.clone();
    b.update_fn = Some(Box::new(move |blk, t| {
        let backend = be.borrow_mut();
        if backend.terminated {
            return;
        }

        // --- inputs: block register → FMU (reuses input_buf) ---
        if !backend.input_vrs.is_empty() {
            for i in 0..backend.input_vrs.len() {
                backend.input_buf[i] = blk.inputs.get_single(i);
            }
            let _ = backend
                .instance
                .set_float64(&backend.input_vrs, &backend.input_buf);
        }
        if backend.output_vrs.is_empty() {
            return;
        }

        // --- outputs: FMU → output_buf (zero-alloc) ---
        let _ = backend
            .instance
            .get_float64(&backend.output_vrs, &mut backend.output_buf);

        // --- Taylor interpolation (only if any output declares derivatives) ---
        let max_order_global = *backend.output_max_orders.iter().max().unwrap_or(&0);
        let dt_offset = t - backend.current_time;
        if max_order_global > 0 && dt_offset > crate::constants::TOLERANCE {
            let mut factorial: f64 = 1.0;
            for order in 1..=max_order_global {
                factorial *= order as f64;
                let factor = dt_offset.powi(order as i32) / factorial;

                // Refill the scratch lists in place for this order.
                backend.taylor_idx.clear();
                backend.taylor_vrs.clear();
                for (i, &m) in backend.output_max_orders.iter().enumerate() {
                    if m >= order {
                        backend.taylor_idx.push(i);
                        backend.taylor_vrs.push(backend.output_vrs[i]);
                    }
                }
                if backend.taylor_idx.is_empty() {
                    break;
                }
                let n = backend.taylor_idx.len();
                backend.taylor_orders.clear();
                backend.taylor_orders.resize(n, order as i32);
                backend.taylor_deriv.clear();
                backend.taylor_deriv.resize(n, 0.0);

                // Split-borrow: `instance` is read-only; the three taylor_*
                // slices alias disjoint fields on self.
                if backend
                    .instance
                    .get_output_derivatives(
                        &backend.taylor_vrs,
                        &backend.taylor_orders,
                        &mut backend.taylor_deriv,
                    )
                    .is_err()
                {
                    break;
                }
                for k in 0..n {
                    let i = backend.taylor_idx[k];
                    backend.output_buf[i] += backend.taylor_deriv[k] * factor;
                }
            }
        }

        for (i, v) in backend.output_buf.iter().enumerate() {
            blk.outputs.set_single(i, *v);
        }
    }));

    Ok(Rc::new(FastCell::new(b)))
}
