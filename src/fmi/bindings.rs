// FMI 3.0 C function-pointer types and primitive types.
//
// Translated directly from `reference-fmus/include/fmi3FunctionTypes.h` and
// `fmi3PlatformTypes.h` (2-clause BSD, Modelica Association Project "FMI").
//
// These are `extern "C" fn` type aliases that match the C typedefs 1:1. They
// are resolved at runtime via `libloading` from an FMU's shared library.
//
// Naming: C's `fmi3InstantiateModelExchangeTYPE` → Rust `Fmi3InstantiateModelExchangeFn`.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_void};

// --- primitive types (fmi3PlatformTypes.h) --------------------------------

pub type fmi3Instance = *mut c_void;
pub type fmi3InstanceEnvironment = *mut c_void;
pub type fmi3FMUState = *mut c_void;
pub type fmi3ValueReference = u32;

pub type fmi3Float32 = f32;
pub type fmi3Float64 = f64;
pub type fmi3Int8 = i8;
pub type fmi3UInt8 = u8;
pub type fmi3Int16 = i16;
pub type fmi3UInt16 = u16;
pub type fmi3Int32 = i32;
pub type fmi3UInt32 = u32;
pub type fmi3Int64 = i64;
pub type fmi3UInt64 = u64;
pub type fmi3Boolean = bool;
pub type fmi3Char = c_char;
pub type fmi3String = *const c_char;
pub type fmi3Byte = u8;
pub type fmi3Binary = *const fmi3Byte;
pub type fmi3Clock = bool;

// --- enums (values match C) -----------------------------------------------

pub const FMI3_OK: i32 = 0;
pub const FMI3_WARNING: i32 = 1;
pub const FMI3_DISCARD: i32 = 2;
pub const FMI3_ERROR: i32 = 3;
pub const FMI3_FATAL: i32 = 4;

// --- callbacks ------------------------------------------------------------

pub type Fmi3LogMessageCallback = unsafe extern "C" fn(
    instance_environment: fmi3InstanceEnvironment,
    status: i32,
    category: fmi3String,
    message: fmi3String,
);

pub type Fmi3IntermediateUpdateCallback = unsafe extern "C" fn(
    instance_environment: fmi3InstanceEnvironment,
    intermediate_update_time: fmi3Float64,
    intermediate_variable_set_requested: fmi3Boolean,
    intermediate_variable_get_allowed: fmi3Boolean,
    intermediate_step_finished: fmi3Boolean,
    can_return_early: fmi3Boolean,
    early_return_requested: *mut fmi3Boolean,
    early_return_time: *mut fmi3Float64,
);

// --- common lifecycle functions -------------------------------------------

pub type Fmi3GetVersionFn = unsafe extern "C" fn() -> *const c_char;

pub type Fmi3InstantiateModelExchangeFn = unsafe extern "C" fn(
    instance_name: fmi3String,
    instantiation_token: fmi3String,
    resource_path: fmi3String,
    visible: fmi3Boolean,
    logging_on: fmi3Boolean,
    instance_environment: fmi3InstanceEnvironment,
    log_message: Option<Fmi3LogMessageCallback>,
) -> fmi3Instance;

pub type Fmi3InstantiateCoSimulationFn = unsafe extern "C" fn(
    instance_name: fmi3String,
    instantiation_token: fmi3String,
    resource_path: fmi3String,
    visible: fmi3Boolean,
    logging_on: fmi3Boolean,
    event_mode_used: fmi3Boolean,
    early_return_allowed: fmi3Boolean,
    required_intermediate_variables: *const fmi3ValueReference,
    n_required_intermediate_variables: usize,
    instance_environment: fmi3InstanceEnvironment,
    log_message: Option<Fmi3LogMessageCallback>,
    intermediate_update: Option<Fmi3IntermediateUpdateCallback>,
) -> fmi3Instance;

pub type Fmi3FreeInstanceFn = unsafe extern "C" fn(instance: fmi3Instance);

pub type Fmi3EnterInitializationModeFn = unsafe extern "C" fn(
    instance: fmi3Instance,
    tolerance_defined: fmi3Boolean,
    tolerance: fmi3Float64,
    start_time: fmi3Float64,
    stop_time_defined: fmi3Boolean,
    stop_time: fmi3Float64,
) -> i32;

pub type Fmi3ExitInitializationModeFn = unsafe extern "C" fn(instance: fmi3Instance) -> i32;
pub type Fmi3EnterEventModeFn = unsafe extern "C" fn(instance: fmi3Instance) -> i32;
pub type Fmi3TerminateFn = unsafe extern "C" fn(instance: fmi3Instance) -> i32;
pub type Fmi3ResetFn = unsafe extern "C" fn(instance: fmi3Instance) -> i32;

pub type Fmi3SetDebugLoggingFn = unsafe extern "C" fn(
    instance: fmi3Instance,
    logging_on: fmi3Boolean,
    n_categories: usize,
    categories: *const fmi3String,
) -> i32;

// --- getters/setters (typed, fmi3{Get,Set}{T}) ----------------------------

macro_rules! getter_fn {
    ($name:ident, $elem:ty) => {
        pub type $name = unsafe extern "C" fn(
            instance: fmi3Instance,
            value_references: *const fmi3ValueReference,
            n_value_references: usize,
            values: *mut $elem,
            n_values: usize,
        ) -> i32;
    };
}

macro_rules! setter_fn {
    ($name:ident, $elem:ty) => {
        pub type $name = unsafe extern "C" fn(
            instance: fmi3Instance,
            value_references: *const fmi3ValueReference,
            n_value_references: usize,
            values: *const $elem,
            n_values: usize,
        ) -> i32;
    };
}

getter_fn!(Fmi3GetFloat32Fn, fmi3Float32);
getter_fn!(Fmi3GetFloat64Fn, fmi3Float64);
getter_fn!(Fmi3GetInt8Fn, fmi3Int8);
getter_fn!(Fmi3GetUInt8Fn, fmi3UInt8);
getter_fn!(Fmi3GetInt16Fn, fmi3Int16);
getter_fn!(Fmi3GetUInt16Fn, fmi3UInt16);
getter_fn!(Fmi3GetInt32Fn, fmi3Int32);
getter_fn!(Fmi3GetUInt32Fn, fmi3UInt32);
getter_fn!(Fmi3GetInt64Fn, fmi3Int64);
getter_fn!(Fmi3GetUInt64Fn, fmi3UInt64);
getter_fn!(Fmi3GetBooleanFn, fmi3Boolean);
getter_fn!(Fmi3GetStringFn, fmi3String);

setter_fn!(Fmi3SetFloat32Fn, fmi3Float32);
setter_fn!(Fmi3SetFloat64Fn, fmi3Float64);
setter_fn!(Fmi3SetInt8Fn, fmi3Int8);
setter_fn!(Fmi3SetUInt8Fn, fmi3UInt8);
setter_fn!(Fmi3SetInt16Fn, fmi3Int16);
setter_fn!(Fmi3SetUInt16Fn, fmi3UInt16);
setter_fn!(Fmi3SetInt32Fn, fmi3Int32);
setter_fn!(Fmi3SetUInt32Fn, fmi3UInt32);
setter_fn!(Fmi3SetInt64Fn, fmi3Int64);
setter_fn!(Fmi3SetUInt64Fn, fmi3UInt64);
setter_fn!(Fmi3SetBooleanFn, fmi3Boolean);
setter_fn!(Fmi3SetStringFn, fmi3String);

// --- FMU state (checkpointing — not used in phase 1 but cheap to expose) --

pub type Fmi3GetFMUStateFn =
    unsafe extern "C" fn(instance: fmi3Instance, state: *mut fmi3FMUState) -> i32;
pub type Fmi3SetFMUStateFn =
    unsafe extern "C" fn(instance: fmi3Instance, state: fmi3FMUState) -> i32;
pub type Fmi3FreeFMUStateFn =
    unsafe extern "C" fn(instance: fmi3Instance, state: *mut fmi3FMUState) -> i32;

// --- UpdateDiscreteStates (used by both ME and CS-with-event-mode) --------

pub type Fmi3UpdateDiscreteStatesFn = unsafe extern "C" fn(
    instance: fmi3Instance,
    discrete_states_need_update: *mut fmi3Boolean,
    terminate_simulation: *mut fmi3Boolean,
    nominals_of_continuous_states_changed: *mut fmi3Boolean,
    values_of_continuous_states_changed: *mut fmi3Boolean,
    next_event_time_defined: *mut fmi3Boolean,
    next_event_time: *mut fmi3Float64,
) -> i32;

pub type Fmi3EvaluateDiscreteStatesFn = unsafe extern "C" fn(instance: fmi3Instance) -> i32;

// --- Model Exchange ------------------------------------------------------

pub type Fmi3EnterContinuousTimeModeFn = unsafe extern "C" fn(instance: fmi3Instance) -> i32;

pub type Fmi3CompletedIntegratorStepFn = unsafe extern "C" fn(
    instance: fmi3Instance,
    no_set_fmu_state_prior_to_current_point: fmi3Boolean,
    enter_event_mode: *mut fmi3Boolean,
    terminate_simulation: *mut fmi3Boolean,
) -> i32;

pub type Fmi3SetTimeFn = unsafe extern "C" fn(instance: fmi3Instance, time: fmi3Float64) -> i32;

pub type Fmi3SetContinuousStatesFn = unsafe extern "C" fn(
    instance: fmi3Instance,
    continuous_states: *const fmi3Float64,
    n_continuous_states: usize,
) -> i32;

pub type Fmi3GetContinuousStateDerivativesFn = unsafe extern "C" fn(
    instance: fmi3Instance,
    derivatives: *mut fmi3Float64,
    n_continuous_states: usize,
) -> i32;

pub type Fmi3GetEventIndicatorsFn = unsafe extern "C" fn(
    instance: fmi3Instance,
    event_indicators: *mut fmi3Float64,
    n_event_indicators: usize,
) -> i32;

pub type Fmi3GetContinuousStatesFn = unsafe extern "C" fn(
    instance: fmi3Instance,
    continuous_states: *mut fmi3Float64,
    n_continuous_states: usize,
) -> i32;

pub type Fmi3GetNominalsOfContinuousStatesFn = unsafe extern "C" fn(
    instance: fmi3Instance,
    nominals: *mut fmi3Float64,
    n_continuous_states: usize,
) -> i32;

pub type Fmi3GetNumberOfEventIndicatorsFn =
    unsafe extern "C" fn(instance: fmi3Instance, n_event_indicators: *mut usize) -> i32;

pub type Fmi3GetNumberOfContinuousStatesFn =
    unsafe extern "C" fn(instance: fmi3Instance, n_continuous_states: *mut usize) -> i32;

// --- Directional derivatives (optional capability; FMI 3.0 §2.3.6) --------
//
// Computes the Jacobian-vector product `sensitivity = (∂unknowns/∂knowns) · seed`.
// Typical ME usage: `unknowns = state_deriv_vrs`, `knowns = state_vrs`, and
// iterating `seed = e_j` over unit vectors yields the `∂ẋ/∂x` Jacobian
// column-by-column.  `n_seed` must equal `n_knowns`; `n_sensitivity` must
// equal `n_unknowns`.

pub type Fmi3GetDirectionalDerivativeFn = unsafe extern "C" fn(
    instance: fmi3Instance,
    unknowns: *const fmi3ValueReference,
    n_unknowns: usize,
    knowns: *const fmi3ValueReference,
    n_knowns: usize,
    seed: *const fmi3Float64,
    n_seed: usize,
    sensitivity: *mut fmi3Float64,
    n_sensitivity: usize,
) -> i32;

// --- Co-Simulation --------------------------------------------------------

pub type Fmi3EnterStepModeFn = unsafe extern "C" fn(instance: fmi3Instance) -> i32;

pub type Fmi3GetOutputDerivativesFn = unsafe extern "C" fn(
    instance: fmi3Instance,
    value_references: *const fmi3ValueReference,
    n_value_references: usize,
    orders: *const fmi3Int32,
    values: *mut fmi3Float64,
    n_values: usize,
) -> i32;

pub type Fmi3DoStepFn = unsafe extern "C" fn(
    instance: fmi3Instance,
    current_communication_point: fmi3Float64,
    communication_step_size: fmi3Float64,
    no_set_fmu_state_prior_to_current_point: fmi3Boolean,
    event_handling_needed: *mut fmi3Boolean,
    terminate_simulation: *mut fmi3Boolean,
    early_return: *mut fmi3Boolean,
    last_successful_time: *mut fmi3Float64,
) -> i32;

// --- symbol names (for dlsym / GetProcAddress lookup) --------------------

/// Null-terminated FMI 3.0 function names in the order we resolve them.
/// These match the exported symbols in every compliant FMU binary.
pub mod sym {
    pub const GET_VERSION: &[u8] = b"fmi3GetVersion\0";

    pub const INSTANTIATE_ME: &[u8] = b"fmi3InstantiateModelExchange\0";
    pub const INSTANTIATE_CS: &[u8] = b"fmi3InstantiateCoSimulation\0";
    pub const FREE_INSTANCE: &[u8] = b"fmi3FreeInstance\0";

    pub const ENTER_INIT: &[u8] = b"fmi3EnterInitializationMode\0";
    pub const EXIT_INIT: &[u8] = b"fmi3ExitInitializationMode\0";
    pub const ENTER_EVENT: &[u8] = b"fmi3EnterEventMode\0";
    pub const TERMINATE: &[u8] = b"fmi3Terminate\0";
    pub const RESET: &[u8] = b"fmi3Reset\0";
    pub const SET_DEBUG_LOGGING: &[u8] = b"fmi3SetDebugLogging\0";

    pub const GET_FLOAT64: &[u8] = b"fmi3GetFloat64\0";
    pub const SET_FLOAT64: &[u8] = b"fmi3SetFloat64\0";
    pub const GET_INT32: &[u8] = b"fmi3GetInt32\0";
    pub const SET_INT32: &[u8] = b"fmi3SetInt32\0";
    pub const GET_BOOLEAN: &[u8] = b"fmi3GetBoolean\0";
    pub const SET_BOOLEAN: &[u8] = b"fmi3SetBoolean\0";

    pub const UPDATE_DISCRETE_STATES: &[u8] = b"fmi3UpdateDiscreteStates\0";

    // Model Exchange
    pub const ENTER_CONTINUOUS: &[u8] = b"fmi3EnterContinuousTimeMode\0";
    pub const COMPLETED_INTEGRATOR_STEP: &[u8] = b"fmi3CompletedIntegratorStep\0";
    pub const SET_TIME: &[u8] = b"fmi3SetTime\0";
    pub const SET_CONTINUOUS_STATES: &[u8] = b"fmi3SetContinuousStates\0";
    pub const GET_CONTINUOUS_STATE_DERIVATIVES: &[u8] = b"fmi3GetContinuousStateDerivatives\0";
    pub const GET_EVENT_INDICATORS: &[u8] = b"fmi3GetEventIndicators\0";
    pub const GET_CONTINUOUS_STATES: &[u8] = b"fmi3GetContinuousStates\0";
    pub const GET_NOMINALS: &[u8] = b"fmi3GetNominalsOfContinuousStates\0";
    pub const GET_N_EVENT_INDICATORS: &[u8] = b"fmi3GetNumberOfEventIndicators\0";
    pub const GET_N_CONTINUOUS_STATES: &[u8] = b"fmi3GetNumberOfContinuousStates\0";

    // Co-Simulation
    pub const ENTER_STEP_MODE: &[u8] = b"fmi3EnterStepMode\0";
    pub const DO_STEP: &[u8] = b"fmi3DoStep\0";
    pub const GET_OUTPUT_DERIVATIVES: &[u8] = b"fmi3GetOutputDerivatives\0";

    // Optional (capability-gated by `providesDirectionalDerivatives`)
    pub const GET_DIRECTIONAL_DERIVATIVE: &[u8] = b"fmi3GetDirectionalDerivative\0";
}
