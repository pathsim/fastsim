# fastsim exception + warning hierarchy.
#
# Drop-in mirror of pathsim.exceptions (``StopSimulation``) plus a fastsim-native
# hierarchy (issue #33) so callers can catch engine failures distinctly from
# builtin exceptions while ``except ValueError`` / ``except RuntimeError`` keep
# working unchanged.
#
# Every concrete error uses DUAL bases: ``FastSimError`` (the fastsim marker)
# and the builtin the engine historically raised (``ValueError`` for
# configuration/parameter errors, ``RuntimeError`` for solver failures). The
# Rust core maps its ``SimError`` variants onto these classes by name (see
# ``src/error.rs``), so a bad LUT still raises something you can catch with
# ``except ValueError`` — and now also with ``except FastSimError``.

from fastsim._fastsim import StopSimulation as _StopSimulation


class StopSimulation(_StopSimulation):
    """Raised by a block or model to signal that the simulation should stop
    immediately. The run loop catches it and terminates cleanly, as if `stop()`
    had been called. Drop-in compatible with pathsim.exceptions.StopSimulation.
    """


# -- error hierarchy -------------------------------------------------------------------

class FastSimError(Exception):
    """Base class for every fastsim-specific error.

    Catch this to handle any failure originating in the fastsim engine while
    letting unrelated exceptions propagate.
    """


class FastSimValueError(FastSimError, ValueError):
    """A configuration/lookup error (bad block, connection, event, or operation).

    Subclasses ``ValueError`` so existing ``except ValueError`` handlers keep
    catching it — this is the class the engine's topology/lookup errors map to.
    """


class InvalidBlockParameterError(FastSimError, ValueError):
    """A block constructor was given an invalid parameter (bad LUT table, a
    non-positive size, an unknown option, ...).

    Subclasses ``ValueError`` for backwards compatibility.
    """


class PortConnectionError(FastSimError, ValueError):
    """A connection references a port alias that does not exist on the block.

    Subclasses ``ValueError`` for backwards compatibility.
    """


class SolverError(FastSimError, RuntimeError):
    """Base class for numerical solver failures.

    Subclasses ``RuntimeError`` so existing ``except RuntimeError`` handlers
    keep catching solver faults.
    """


class ConvergenceError(SolverError):
    """The implicit solver's inner Newton/fixed-point iteration did not converge."""


class AlgebraicLoopError(SolverError):
    """An algebraic (feedthrough) loop did not converge."""


class StepSizeError(SolverError):
    """The adaptive step controller required a step below ``dt_min``."""


class SingularJacobianError(SolverError):
    """A singular Jacobian was encountered during an implicit solve."""


class TruncatedTrajectoryError(SolverError):
    """Integration stopped before the requested end time (e.g. step floor hit)."""


# -- warning hierarchy -----------------------------------------------------------------

class FastSimWarning(UserWarning):
    """Base class for every fastsim-specific warning.

    Subclasses ``UserWarning`` so it is visible by default and catchable with
    ``warnings.catch_warnings`` / ``pytest.warns`` in headless and optimizer
    contexts (issue #27).
    """


class FastSimConvergenceWarning(FastSimWarning):
    """A run completed but a convergence subsystem reported a problem: an
    implicit solve or algebraic loop did not converge, or the trajectory was
    truncated at the ``dt_min`` floor.

    Emitted unconditionally (independent of the simulation ``log`` flag) so the
    failure is both visible AND catchable, while — per the documented fail-open
    policy — the run still returns its (truthful) ``RunStats`` instead of
    hard-failing existing workflows.
    """


__all__ = [
    "StopSimulation",
    "FastSimError",
    "FastSimValueError",
    "InvalidBlockParameterError",
    "PortConnectionError",
    "SolverError",
    "ConvergenceError",
    "AlgebraicLoopError",
    "StepSizeError",
    "SingularJacobianError",
    "TruncatedTrajectoryError",
    "FastSimWarning",
    "FastSimConvergenceWarning",
]
