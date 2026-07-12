// Crate-wide error type for the simulation engine.
//
// Mirrors the well-modeled `fmi::FmiError` pattern: a single `thiserror` enum
// for everything the engine can reject, so the pybindings layer can map each
// cause to a Python exception instead of funneling stringly-typed errors (or
// panicking) on user input.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SimError {
    #[error("block already part of simulation")]
    DuplicateBlock,

    #[error("block not part of simulation")]
    BlockNotFound,

    #[error("connection not part of simulation")]
    ConnectionNotFound,

    #[error("event not part of simulation")]
    EventNotFound,

    #[error("input port alias '{0}' not found on block")]
    InputPortAlias(String),

    #[error("output port alias '{0}' not found on block")]
    OutputPortAlias(String),

    #[error("a Subsystem may contain only one 'Interface' block")]
    MultipleInterfaces,

    #[error("a Subsystem 'blocks' list must contain an 'Interface' block")]
    MissingInterface,

    #[error("unknown operation '{op}' (expected one of: {expected})")]
    UnknownOp { op: String, expected: &'static str },

    // ---- Numeric-failure variant group (issue #27) --------------------------
    // These give convergence/step failures a home in the type system so run
    // results can be reported truthfully instead of silently swallowed. Per the
    // documented policy, the non-convergence / truncation variants normally
    // surface as `FastSimConvergenceWarning` (populated into `RunStats`); they
    // become hard errors only where a caller opts into strict handling.
    #[error("implicit solver did not converge at t={time:.6} (max residual {residual:.3e})")]
    SolverNonConvergence { time: f64, residual: f64 },

    #[error("algebraic loop did not converge (max residual {residual:.3e})")]
    AlgebraicLoopNonConvergence { residual: f64 },

    #[error("required step size fell below dt_min ({dt_min:.3e}) at t={time:.6}; trajectory truncated")]
    StepBelowMinimum { time: f64, dt_min: f64 },

    #[error("singular Jacobian encountered at t={time:.6}")]
    SingularJacobian { time: f64 },

    #[error("invalid block parameter: {0}")]
    InvalidBlockParam(String),

    #[error("trajectory truncated at t={time:.6} before the requested end")]
    TruncatedTrajectory { time: f64 },
}

pub type Result<T> = std::result::Result<T, SimError>;

// Cross the PyO3 boundary carrying the full message. Each variant maps to a
// class from the `fastsim.exceptions` hierarchy (issue #33) whose bases include
// the builtin the engine historically raised (ValueError / RuntimeError), so
// `except ValueError` / `except RuntimeError` keep working while callers can
// also catch `FastSimError` distinctly. If the Python hierarchy cannot be
// imported (e.g. the extension used outside the `fastsim` package), we fall
// back to the plain builtin so behaviour never regresses to a panic.
#[cfg(feature = "python")]
impl SimError {
    /// The `fastsim.exceptions` class name this error maps to.
    fn py_class_name(&self) -> &'static str {
        match self {
            SimError::InputPortAlias(_) | SimError::OutputPortAlias(_) => "PortConnectionError",
            SimError::InvalidBlockParam(_) => "InvalidBlockParameterError",
            SimError::SolverNonConvergence { .. } => "ConvergenceError",
            SimError::AlgebraicLoopNonConvergence { .. } => "AlgebraicLoopError",
            SimError::StepBelowMinimum { .. } => "StepSizeError",
            SimError::TruncatedTrajectory { .. } => "TruncatedTrajectoryError",
            SimError::SingularJacobian { .. } => "SingularJacobianError",
            // Topology / lookup / op errors keep the historical ValueError shape.
            _ => "FastSimValueError",
        }
    }

    /// Whether the historical builtin base for this error is `RuntimeError`
    /// (solver failures) rather than `ValueError` (configuration errors).
    fn is_runtime_shaped(&self) -> bool {
        matches!(
            self,
            SimError::SolverNonConvergence { .. }
                | SimError::AlgebraicLoopNonConvergence { .. }
                | SimError::StepBelowMinimum { .. }
                | SimError::TruncatedTrajectory { .. }
                | SimError::SingularJacobian { .. }
        )
    }

    pub fn to_pyerr(&self, py: pyo3::Python<'_>) -> pyo3::PyErr {
        use pyo3::prelude::*;
        let msg = self.to_string();
        let cls_name = self.py_class_name();
        if let Ok(cls) = py
            .import("fastsim.exceptions")
            .and_then(|m| m.getattr(cls_name))
        {
            if let Ok(inst) = cls.call1((msg.clone(),)) {
                return PyErr::from_value(inst);
            }
        }
        // Fallback to the historical builtin so `except ValueError`/`except
        // RuntimeError` still catches these even without the Python package.
        if self.is_runtime_shaped() {
            pyo3::exceptions::PyRuntimeError::new_err(msg)
        } else {
            pyo3::exceptions::PyValueError::new_err(msg)
        }
    }
}

#[cfg(feature = "python")]
impl From<SimError> for pyo3::PyErr {
    fn from(e: SimError) -> Self {
        // Reentrant GIL acquisition: every `?`/`.into()` site is already inside
        // a pymethod holding the GIL, so this hands back the existing token and
        // routes through the `fastsim.exceptions` hierarchy transparently.
        pyo3::Python::attach(|py| e.to_pyerr(py))
    }
}
