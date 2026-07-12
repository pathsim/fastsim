// Callback bridging between the FMU's C API and Rust-side state.
//
// FMI 3.0 drops the FMI 2.0 `CallbackFunctions` struct in favour of direct
// function pointers plus an opaque `instanceEnvironment` that the FMU hands
// back on every callback. We put a boxed `LogEnv` there so the static
// `extern "C"` shim can recover per-instance context.
//
// Ref: reference-fmus/src/FMI3.c:15-27 (`cb_logMessage3`).

use std::ffi::CStr;

use super::bindings::{
    fmi3Boolean, fmi3Float64, fmi3InstanceEnvironment, fmi3String, FMI3_ERROR, FMI3_FATAL,
    FMI3_WARNING,
};

/// Per-instance context carried through `instanceEnvironment`. Pointed to by a
/// boxed pointer owned by the `Instance`; must outlive the FMU instance ptr.
pub struct LogEnv {
    pub instance_name: String,
    pub verbose: bool,
}

/// `extern "C"` shim matching `fmi3LogMessageCallback`. The FMU passes our
/// boxed `LogEnv` pointer back here unchanged.
///
/// # Safety
///
/// `instance_environment` must be a valid pointer to a `LogEnv`
/// produced by `Box::into_raw`, and must outlive this call. `category` and
/// `message` must be null or NUL-terminated UTF-8.
pub unsafe extern "C" fn log_message_callback(
    instance_environment: fmi3InstanceEnvironment,
    status: i32,
    category: fmi3String,
    message: fmi3String,
) {
    if instance_environment.is_null() {
        return;
    }
    // SAFETY: caller guarantees the pointer originates from our Box<LogEnv>.
    let env: &LogEnv = unsafe { &*(instance_environment as *const LogEnv) };

    let is_noise = status < FMI3_WARNING && !env.verbose;
    if is_noise {
        return;
    }

    let cat = c_str_or_empty(category);
    let msg = c_str_or_empty(message);
    let (level, log_level) = match status {
        FMI3_ERROR | FMI3_FATAL => ("ERROR", crate::utils::logger::LogLevel::Error),
        FMI3_WARNING => ("WARN", crate::utils::logger::LogLevel::Warning),
        _ => ("INFO", crate::utils::logger::LogLevel::Info),
    };
    // Route through the single sink (issue #29) instead of a raw eprintln.
    crate::utils::sink::emit(
        log_level,
        &format!("[FMU {name}] {level} [{cat}] {msg}", name = env.instance_name),
    );
}

/// Minimal `fmi3IntermediateUpdateCallback`. The FMU calls this from
/// inside `fmi3DoStep` when `providesIntermediateUpdate=true`. We decline
/// early return and ignore the intermediate-step hooks.
///
/// # Safety
///
/// Matches the FMI 3.0 C signature exactly. Pointer writes guard for
/// null. Called synchronously from the FMU thread inside DoStep.
pub unsafe extern "C" fn intermediate_update_noop(
    _instance_environment: fmi3InstanceEnvironment,
    _intermediate_update_time: fmi3Float64,
    _intermediate_variable_set_requested: fmi3Boolean,
    _intermediate_variable_get_allowed: fmi3Boolean,
    _intermediate_step_finished: fmi3Boolean,
    _can_return_early: fmi3Boolean,
    early_return_requested: *mut fmi3Boolean,
    early_return_time: *mut fmi3Float64,
) {
    if !early_return_requested.is_null() {
        unsafe { *early_return_requested = false };
    }
    if !early_return_time.is_null() {
        unsafe { *early_return_time = 0.0 };
    }
}

fn c_str_or_empty<'a>(p: fmi3String) -> &'a str {
    if p.is_null() {
        return "";
    }
    // SAFETY: caller guarantees NUL-terminated, UTF-8 string with lifetime
    // outlasting this call. We borrow for the duration of the callback.
    unsafe { CStr::from_ptr(p) }.to_str().unwrap_or("")
}
