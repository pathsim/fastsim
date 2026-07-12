//! Single logging sink (issue #29).
//!
//! Every warning the engine emits — the `Logger` (progress/convergence) and the
//! formerly-stray `eprintln!` production sites (port-alias mismatches, port
//! count mismatches, dt_min truncation, …) — routes through `emit` here instead
//! of writing to stdout/stderr directly. A host (Python, a UI, a test harness)
//! can install its own `LogSink` via [`set_sink`] to capture or redirect all of
//! it from one place.
//!
//! This is deliberately pragmatic: the default sink preserves the historical
//! print behaviour (INFO/WARNING to stdout, ERROR to stderr). A full
//! `log`/`tracing` subscriber integration is left as future work — it can drop
//! in as an alternative `LogSink` without touching any call site.

use std::sync::RwLock;

use super::logger::LogLevel;

/// A destination for engine log records. Implement this and install it with
/// [`set_sink`] to capture warnings/errors instead of printing them.
pub trait LogSink: Send + Sync {
    fn emit(&self, level: LogLevel, msg: &str);
}

/// Default sink: INFO/WARNING to stdout (progress output, not errors — a host
/// capturing stderr separately must not render them as errors), ERROR to
/// stderr. Matches the historical `Logger` behaviour.
pub struct DefaultSink;

impl LogSink for DefaultSink {
    fn emit(&self, level: LogLevel, msg: &str) {
        if level >= LogLevel::Error {
            eprintln!("{msg}");
        } else {
            println!("{msg}");
        }
    }
}

// `None` means "use the DefaultSink". `RwLock::new` / `None` are const, so no
// lazy-init machinery is needed.
static SINK: RwLock<Option<Box<dyn LogSink>>> = RwLock::new(None);

/// Route a log record through the installed sink (or the default one).
pub fn emit(level: LogLevel, msg: &str) {
    let guard = SINK.read().unwrap_or_else(|e| e.into_inner());
    match guard.as_ref() {
        Some(sink) => sink.emit(level, msg),
        None => DefaultSink.emit(level, msg),
    }
}

/// Install a custom sink for all subsequent engine log records.
pub fn set_sink(sink: Box<dyn LogSink>) {
    *SINK.write().unwrap_or_else(|e| e.into_inner()) = Some(sink);
}

/// Restore the default (print) sink.
pub fn reset_sink() {
    *SINK.write().unwrap_or_else(|e| e.into_inner()) = None;
}

/// Emit a warning through the sink. Convenience for the formerly-stray
/// `eprintln!` warning sites.
pub fn warn(msg: &str) {
    emit(LogLevel::Warning, msg);
}

/// Emit an error through the sink.
pub fn error(msg: &str) {
    emit(LogLevel::Error, msg);
}
