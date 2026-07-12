// FMU instance lifetime management.
//
// An `Instance<Kind>` owns:
//   - the extracted FMU archive (TempDir)
//   - the loaded shared library
//   - resolved FMI 3.0 function pointers
//   - the boxed `LogEnv` carried through `instanceEnvironment`
//   - the raw `fmi3Instance` pointer
//
// Type-state (`Kind` = `Me` or `Cs`) distinguishes Model-Exchange from
// Co-Simulation instances at compile time. Common lifecycle methods (init,
// terminate, get/set Float64, UpdateDiscreteStates) live on `impl<K>`;
// ME/CS-specific methods live on `impl Instance<Me>` / `impl Instance<Cs>`.
//
// Drop order: we call `fmi3FreeInstance` in `Drop::drop`, then let Rust drop
// the struct fields in declaration order. `log_env` must outlive the FMU
// because the FMU may log during `fmi3FreeInstance`. `lib` must outlive
// `log_env` only trivially (both are freed locally after). `archive` goes
// last so the FMU's resources are present while it's alive.

use std::ffi::CString;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::ptr;

use libloading::Library;

use super::bindings::*;
use super::callbacks::{intermediate_update_noop, log_message_callback, LogEnv};
use super::model_description::ModelDescription;
use super::platform::{library_extension, platform_tuple};
use super::unzip::FmuArchive;
use super::{FmiError, FmiStatus, Result};

// --- Kind markers ----------------------------------------------------------

/// Zero-sized type-state marker: the `Instance<Me>` family is a
/// Model-Exchange FMU. Enables ME-specific methods at compile time
/// (`set_time`, `get_continuous_state_derivatives`, …).
pub struct Me;

/// Zero-sized type-state marker: the `Instance<Cs>` family is a
/// Co-Simulation FMU. Enables CS-specific methods at compile time
/// (`enter_step_mode`, `do_step`, `get_output_derivatives`).
pub struct Cs;

// --- resolved function pointers -------------------------------------------

struct Fns {
    // always present
    free_instance: Fmi3FreeInstanceFn,
    enter_init: Fmi3EnterInitializationModeFn,
    exit_init: Fmi3ExitInitializationModeFn,
    enter_event: Fmi3EnterEventModeFn,
    terminate: Fmi3TerminateFn,
    update_discrete_states: Fmi3UpdateDiscreteStatesFn,
    get_float64: Fmi3GetFloat64Fn,
    set_float64: Fmi3SetFloat64Fn,

    // ME-only (None for CS instances)
    enter_continuous_time_mode: Option<Fmi3EnterContinuousTimeModeFn>,
    completed_integrator_step: Option<Fmi3CompletedIntegratorStepFn>,
    set_time: Option<Fmi3SetTimeFn>,
    set_continuous_states: Option<Fmi3SetContinuousStatesFn>,
    get_continuous_state_derivatives: Option<Fmi3GetContinuousStateDerivativesFn>,
    get_event_indicators: Option<Fmi3GetEventIndicatorsFn>,

    // CS-only (None for ME instances)
    enter_step_mode: Option<Fmi3EnterStepModeFn>,
    do_step: Option<Fmi3DoStepFn>,
    get_output_derivatives: Option<Fmi3GetOutputDerivativesFn>,

    // Optional capability, gated by `providesDirectionalDerivatives` in
    // modelDescription.xml.  Populated for ME FMUs that advertise it; absent
    // otherwise (callers must consult `Instance::supports_directional_derivatives`).
    get_directional_derivative: Option<Fmi3GetDirectionalDerivativeFn>,
}

macro_rules! load_required {
    ($lib:expr, $sym:expr, $ty:ty) => {{
        // SAFETY: we hold the Library alive as long as the returned fn ptr
        // is used (enforced by Instance's field ordering).
        unsafe {
            let s: libloading::Symbol<$ty> = $lib.get($sym)?;
            *s
        }
    }};
}

macro_rules! load_optional {
    ($lib:expr, $sym:expr, $ty:ty) => {{
        unsafe {
            let res: std::result::Result<libloading::Symbol<$ty>, _> = $lib.get($sym);
            res.ok().map(|s| *s)
        }
    }};
}

// --- Instance --------------------------------------------------------------

pub struct Instance<K> {
    // Declaration order matters — see file header comment.
    ptr: fmi3Instance,
    fns: Fns,
    // Held to keep the logging environment, loaded library and extracted FMU
    // alive for the instance's lifetime (and to drop in the right order);
    // never read back after construction.
    #[allow(dead_code)]
    log_env: Box<LogEnv>,
    #[allow(dead_code)]
    lib: Library,
    #[allow(dead_code)]
    archive: FmuArchive,
    _kind: PhantomData<K>,
}

// fmi3Instance is a raw pointer, so Instance is automatically !Send + !Sync
// via the `ptr` field. FMI 3.0 instances are single-threaded.

// --- construction: load library + resolve common symbols ------------------

/// Build a `CString`, mapping an interior-NUL failure to a descriptive error
/// instead of panicking across the FFI boundary.
fn cstr(s: impl Into<Vec<u8>>, what: &str) -> Result<CString> {
    CString::new(s).map_err(|_| {
        FmiError::ModelDescription(format!("{what} contains an interior NUL byte"))
    })
}

fn load_library(archive: &FmuArchive, model_identifier: &str) -> Result<Library> {
    let tuple = platform_tuple();
    let binaries_dir = archive.binaries_dir(tuple);
    let ext = library_extension();
    let bin_path: PathBuf = binaries_dir.join(format!("{model_identifier}.{ext}"));

    if !bin_path.exists() {
        return Err(FmiError::UnsupportedPlatform {
            tuple: tuple.to_owned(),
            available: archive.available_platforms(),
        });
    }

    // SAFETY: loading a shared library from disk; standard libloading usage.
    unsafe { Library::new(&bin_path) }.map_err(Into::into)
}

fn load_common_fns(lib: &Library) -> Result<Fns> {
    Ok(Fns {
        free_instance: load_required!(lib, sym::FREE_INSTANCE, Fmi3FreeInstanceFn),
        enter_init: load_required!(lib, sym::ENTER_INIT, Fmi3EnterInitializationModeFn),
        exit_init: load_required!(lib, sym::EXIT_INIT, Fmi3ExitInitializationModeFn),
        enter_event: load_required!(lib, sym::ENTER_EVENT, Fmi3EnterEventModeFn),
        terminate: load_required!(lib, sym::TERMINATE, Fmi3TerminateFn),
        update_discrete_states: load_required!(
            lib,
            sym::UPDATE_DISCRETE_STATES,
            Fmi3UpdateDiscreteStatesFn
        ),
        get_float64: load_required!(lib, sym::GET_FLOAT64, Fmi3GetFloat64Fn),
        set_float64: load_required!(lib, sym::SET_FLOAT64, Fmi3SetFloat64Fn),
        enter_continuous_time_mode: None,
        completed_integrator_step: None,
        set_time: None,
        set_continuous_states: None,
        get_continuous_state_derivatives: None,
        get_event_indicators: None,
        enter_step_mode: None,
        do_step: None,
        get_output_derivatives: None,
        get_directional_derivative: None,
    })
}

// --- Model Exchange construction ------------------------------------------

impl Instance<Me> {
    /// Instantiate an FMU in Model-Exchange mode.
    pub fn new_model_exchange(
        archive: FmuArchive,
        md: &ModelDescription,
        instance_name: &str,
        verbose: bool,
    ) -> Result<Self> {
        let me_info = md.model_exchange.as_ref().ok_or_else(|| {
            FmiError::ModelDescription("FMU does not support Model Exchange".into())
        })?;

        let lib = load_library(&archive, &me_info.model_identifier)?;
        let mut fns = load_common_fns(&lib)?;

        fns.enter_continuous_time_mode = Some(load_required!(
            &lib, sym::ENTER_CONTINUOUS, Fmi3EnterContinuousTimeModeFn
        ));
        fns.completed_integrator_step = Some(load_required!(
            &lib, sym::COMPLETED_INTEGRATOR_STEP, Fmi3CompletedIntegratorStepFn
        ));
        fns.set_time = Some(load_required!(&lib, sym::SET_TIME, Fmi3SetTimeFn));
        fns.set_continuous_states = Some(load_required!(
            &lib, sym::SET_CONTINUOUS_STATES, Fmi3SetContinuousStatesFn
        ));
        fns.get_continuous_state_derivatives = Some(load_required!(
            &lib,
            sym::GET_CONTINUOUS_STATE_DERIVATIVES,
            Fmi3GetContinuousStateDerivativesFn
        ));
        fns.get_event_indicators = Some(load_required!(
            &lib, sym::GET_EVENT_INDICATORS, Fmi3GetEventIndicatorsFn
        ));

        // Optional — only present if the FMU was compiled with support.
        fns.get_directional_derivative = load_optional!(
            &lib, sym::GET_DIRECTIONAL_DERIVATIVE, Fmi3GetDirectionalDerivativeFn
        );

        let instantiate: Fmi3InstantiateModelExchangeFn =
            load_required!(&lib, sym::INSTANTIATE_ME, Fmi3InstantiateModelExchangeFn);

        let log_env = Box::new(LogEnv {
            instance_name: instance_name.to_owned(),
            verbose,
        });

        let name = cstr(instance_name, "instance name")?;
        let token = cstr(md.instantiation_token.clone(), "instantiation token")?;
        let resource_uri = match archive.resource_uri() {
            Some(s) => cstr(s, "resource URI")?,
            None => CString::new("").expect("empty string has no NUL"),
        };

        let env_ptr = (&*log_env as *const LogEnv) as fmi3InstanceEnvironment;
        // SAFETY: instantiate is the FMU's canonical instantiator; pointers
        // passed in remain valid for the call duration.
        let ptr = unsafe {
            instantiate(
                name.as_ptr(),
                token.as_ptr(),
                resource_uri.as_ptr(),
                false, // visible
                verbose,
                env_ptr,
                Some(log_message_callback),
            )
        };
        if ptr.is_null() {
            return Err(FmiError::ModelDescription(
                "fmi3InstantiateModelExchange returned NULL".into(),
            ));
        }

        Ok(Self {
            ptr,
            fns,
            log_env,
            lib,
            archive,
            _kind: PhantomData,
        })
    }
}

// --- Co-Simulation construction -------------------------------------------

impl Instance<Cs> {
    /// Instantiate an FMU in Co-Simulation mode.
    ///
    /// - `event_mode_used`: requires `cs_info.has_event_mode=true` in the FMU
    /// - `early_return_allowed`: the FMU may return from `fmi3DoStep` before
    ///   the requested step completes (FMI 3.0 §4.2.4). Callers must handle
    ///   partial advances by consulting `DoStepResult.last_successful_time`.
    pub fn new_co_simulation(
        archive: FmuArchive,
        md: &ModelDescription,
        instance_name: &str,
        event_mode_used: bool,
        early_return_allowed: bool,
        verbose: bool,
    ) -> Result<Self> {
        let cs_info = md.co_simulation.as_ref().ok_or_else(|| {
            FmiError::ModelDescription("FMU does not support Co-Simulation".into())
        })?;

        let lib = load_library(&archive, &cs_info.model_identifier)?;
        let mut fns = load_common_fns(&lib)?;

        fns.enter_step_mode = Some(load_required!(
            &lib, sym::ENTER_STEP_MODE, Fmi3EnterStepModeFn
        ));
        fns.do_step = Some(load_required!(&lib, sym::DO_STEP, Fmi3DoStepFn));
        // Optional — only FMUs with `maxOutputDerivativeOrder>0` need to
        // implement this, but most export it as a stub anyway.
        fns.get_output_derivatives = load_optional!(
            &lib,
            sym::GET_OUTPUT_DERIVATIVES,
            Fmi3GetOutputDerivativesFn
        );

        // UpdateDiscreteStates & get_event_indicators are only needed if
        // hasEventMode; but loading them unconditionally is cheap.
        fns.get_event_indicators =
            load_optional!(&lib, sym::GET_EVENT_INDICATORS, Fmi3GetEventIndicatorsFn);

        let instantiate: Fmi3InstantiateCoSimulationFn =
            load_required!(&lib, sym::INSTANTIATE_CS, Fmi3InstantiateCoSimulationFn);

        let log_env = Box::new(LogEnv {
            instance_name: instance_name.to_owned(),
            verbose,
        });

        let name = cstr(instance_name, "instance name")?;
        let token = cstr(md.instantiation_token.clone(), "instantiation token")?;
        let resource_uri = match archive.resource_uri() {
            Some(s) => cstr(s, "resource URI")?,
            None => CString::new("").expect("empty string has no NUL"),
        };

        let env_ptr = (&*log_env as *const LogEnv) as fmi3InstanceEnvironment;
        let ptr = unsafe {
            instantiate(
                name.as_ptr(),
                token.as_ptr(),
                resource_uri.as_ptr(),
                false, // visible
                verbose,
                event_mode_used && cs_info.has_event_mode,
                early_return_allowed,
                ptr::null(),   // requiredIntermediateVariables — none
                0,
                env_ptr,
                Some(log_message_callback),
                // Always provide the no-op callback: if the FMU advertises
                // providesIntermediateUpdate=true it requires one, and if it
                // doesn't, it will never call it anyway.
                Some(intermediate_update_noop),
            )
        };
        if ptr.is_null() {
            return Err(FmiError::ModelDescription(
                "fmi3InstantiateCoSimulation returned NULL".into(),
            ));
        }

        Ok(Self {
            ptr,
            fns,
            log_env,
            lib,
            archive,
            _kind: PhantomData,
        })
    }
}

// --- common operations -----------------------------------------------------

fn check(status: i32, call: &'static str) -> Result<()> {
    let s = FmiStatus::from_raw(status);
    if s.is_ok_or_warning() {
        Ok(())
    } else {
        Err(FmiError::FmiStatus { call, status: s })
    }
}

impl<K> Instance<K> {
    pub fn enter_initialization_mode(
        &self,
        tolerance: Option<f64>,
        start_time: f64,
        stop_time: Option<f64>,
    ) -> Result<()> {
        let status = unsafe {
            (self.fns.enter_init)(
                self.ptr,
                tolerance.is_some(),
                tolerance.unwrap_or(0.0),
                start_time,
                stop_time.is_some(),
                stop_time.unwrap_or(0.0),
            )
        };
        check(status, "fmi3EnterInitializationMode")
    }

    pub fn exit_initialization_mode(&self) -> Result<()> {
        let status = unsafe { (self.fns.exit_init)(self.ptr) };
        check(status, "fmi3ExitInitializationMode")
    }

    pub fn enter_event_mode(&self) -> Result<()> {
        let status = unsafe { (self.fns.enter_event)(self.ptr) };
        check(status, "fmi3EnterEventMode")
    }

    pub fn terminate(&self) -> Result<()> {
        let status = unsafe { (self.fns.terminate)(self.ptr) };
        check(status, "fmi3Terminate")
    }

    pub fn set_float64(&self, vrs: &[fmi3ValueReference], values: &[f64]) -> Result<()> {
        debug_assert_eq!(vrs.len(), values.len());
        let status = unsafe {
            (self.fns.set_float64)(
                self.ptr,
                vrs.as_ptr(),
                vrs.len(),
                values.as_ptr(),
                values.len(),
            )
        };
        check(status, "fmi3SetFloat64")
    }

    pub fn get_float64(&self, vrs: &[fmi3ValueReference], out: &mut [f64]) -> Result<()> {
        debug_assert_eq!(vrs.len(), out.len());
        let status = unsafe {
            (self.fns.get_float64)(
                self.ptr,
                vrs.as_ptr(),
                vrs.len(),
                out.as_mut_ptr(),
                out.len(),
            )
        };
        check(status, "fmi3GetFloat64")
    }

    /// Result of `fmi3UpdateDiscreteStates` — returned to the caller so it
    /// can drive the fixed-point iteration.
    pub fn update_discrete_states(&self) -> Result<DiscreteStateUpdate> {
        let mut u = DiscreteStateUpdate::default();
        let status = unsafe {
            (self.fns.update_discrete_states)(
                self.ptr,
                &mut u.discrete_states_need_update,
                &mut u.terminate_simulation,
                &mut u.nominals_changed,
                &mut u.values_changed,
                &mut u.next_event_time_defined,
                &mut u.next_event_time,
            )
        };
        check(status, "fmi3UpdateDiscreteStates")?;
        Ok(u)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DiscreteStateUpdate {
    pub discrete_states_need_update: bool,
    pub terminate_simulation: bool,
    pub nominals_changed: bool,
    pub values_changed: bool,
    pub next_event_time_defined: bool,
    pub next_event_time: f64,
}

// --- Model Exchange operations --------------------------------------------

impl Instance<Me> {
    pub fn enter_continuous_time_mode(&self) -> Result<()> {
        let f = self.fns.enter_continuous_time_mode.unwrap();
        let status = unsafe { f(self.ptr) };
        check(status, "fmi3EnterContinuousTimeMode")
    }

    pub fn set_time(&self, t: f64) -> Result<()> {
        let f = self.fns.set_time.unwrap();
        let status = unsafe { f(self.ptr, t) };
        check(status, "fmi3SetTime")
    }

    pub fn set_continuous_states(&self, x: &[f64]) -> Result<()> {
        let f = self.fns.set_continuous_states.unwrap();
        let status = unsafe { f(self.ptr, x.as_ptr(), x.len()) };
        check(status, "fmi3SetContinuousStates")
    }

    pub fn get_continuous_state_derivatives(&self, out: &mut [f64]) -> Result<()> {
        let f = self.fns.get_continuous_state_derivatives.unwrap();
        let status = unsafe { f(self.ptr, out.as_mut_ptr(), out.len()) };
        check(status, "fmi3GetContinuousStateDerivatives")
    }

    pub fn get_event_indicators(&self, out: &mut [f64]) -> Result<()> {
        let f = self.fns.get_event_indicators.unwrap();
        let status = unsafe { f(self.ptr, out.as_mut_ptr(), out.len()) };
        check(status, "fmi3GetEventIndicators")
    }

    pub fn completed_integrator_step(
        &self,
        no_set_fmu_state_prior: bool,
    ) -> Result<CompletedStepResult> {
        let f = self.fns.completed_integrator_step.unwrap();
        let mut enter_event_mode = false;
        let mut terminate_simulation = false;
        let status = unsafe {
            f(
                self.ptr,
                no_set_fmu_state_prior,
                &mut enter_event_mode,
                &mut terminate_simulation,
            )
        };
        check(status, "fmi3CompletedIntegratorStep")?;
        Ok(CompletedStepResult {
            enter_event_mode,
            terminate_simulation,
        })
    }

    /// Whether the FMU exported `fmi3GetDirectionalDerivative`.  `true` implies
    /// `providesDirectionalDerivatives="true"` in modelDescription AND the
    /// symbol was resolved in the FMU's shared library.
    pub fn supports_directional_derivatives(&self) -> bool {
        self.fns.get_directional_derivative.is_some()
    }

    /// Call `fmi3GetDirectionalDerivative` with the supplied seed.  Computes
    /// `sensitivity = (∂unknowns/∂knowns) · seed`.  Lengths must satisfy
    /// `seed.len() == knowns.len()` and `sensitivity.len() == unknowns.len()`.
    /// Returns an error if the FMU did not export the symbol.
    pub fn get_directional_derivative(
        &self,
        unknowns: &[fmi3ValueReference],
        knowns: &[fmi3ValueReference],
        seed: &[f64],
        sensitivity: &mut [f64],
    ) -> Result<()> {
        let f = self.fns.get_directional_derivative.ok_or_else(|| {
            FmiError::ModelDescription(
                "FMU does not export fmi3GetDirectionalDerivative".into(),
            )
        })?;
        let status = unsafe {
            f(
                self.ptr,
                unknowns.as_ptr(),
                unknowns.len(),
                knowns.as_ptr(),
                knowns.len(),
                seed.as_ptr(),
                seed.len(),
                sensitivity.as_mut_ptr(),
                sensitivity.len(),
            )
        };
        check(status, "fmi3GetDirectionalDerivative")
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CompletedStepResult {
    pub enter_event_mode: bool,
    pub terminate_simulation: bool,
}

// --- Co-Simulation operations ---------------------------------------------

impl Instance<Cs> {
    pub fn enter_step_mode(&self) -> Result<()> {
        let f = self.fns.enter_step_mode.unwrap();
        let status = unsafe { f(self.ptr) };
        check(status, "fmi3EnterStepMode")
    }

    /// `fmi3GetOutputDerivatives` — Taylor derivatives of outputs at current
    /// time. Useful for interpolating outputs at times between communication
    /// points. Returns an error if the FMU doesn't export this function or if
    /// any of the named variables has `maxOutputDerivativeOrder` less than the
    /// requested order. `values.len()` must equal `vrs.len() == orders.len()`.
    pub fn get_output_derivatives(
        &self,
        vrs: &[fmi3ValueReference],
        orders: &[i32],
        values: &mut [f64],
    ) -> Result<()> {
        debug_assert_eq!(vrs.len(), orders.len());
        debug_assert_eq!(vrs.len(), values.len());
        let f = self.fns.get_output_derivatives.ok_or_else(|| {
            crate::fmi::FmiError::ModelDescription(
                "fmi3GetOutputDerivatives not exported by FMU".into(),
            )
        })?;
        let status = unsafe {
            f(
                self.ptr,
                vrs.as_ptr(),
                vrs.len(),
                orders.as_ptr(),
                values.as_mut_ptr(),
                values.len(),
            )
        };
        check(status, "fmi3GetOutputDerivatives")
    }

    pub fn do_step(
        &self,
        current_communication_point: f64,
        communication_step_size: f64,
    ) -> Result<DoStepResult> {
        let f = self.fns.do_step.unwrap();
        let mut event_handling_needed = false;
        let mut terminate_simulation = false;
        let mut early_return = false;
        let mut last_successful_time = 0.0;
        let status = unsafe {
            f(
                self.ptr,
                current_communication_point,
                communication_step_size,
                true, // noSetFMUStatePriorToCurrentPoint — we don't checkpoint
                &mut event_handling_needed,
                &mut terminate_simulation,
                &mut early_return,
                &mut last_successful_time,
            )
        };
        check(status, "fmi3DoStep")?;
        Ok(DoStepResult {
            event_handling_needed,
            terminate_simulation,
            early_return,
            last_successful_time,
        })
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DoStepResult {
    pub event_handling_needed: bool,
    pub terminate_simulation: bool,
    pub early_return: bool,
    pub last_successful_time: f64,
}

// --- Drop ------------------------------------------------------------------

impl<K> Drop for Instance<K> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: free_instance is resolved from this library; the FMU
            // contract is that this call terminates all FMU-internal state.
            // The log_env stays alive until after this method returns, so the
            // FMU may log during FreeInstance.
            unsafe { (self.fns.free_instance)(self.ptr) };
            self.ptr = ptr::null_mut();
        }
        // Remaining fields drop in declaration order after this returns:
        // fns (no-op), log_env (freed), lib (dlclose), archive (remove dir).
    }
}
