//! Python surface for the `codegen` module (feature `codegen`).
//!
//! Two entry points, both returning a dict `{file name: C source}` and sharing
//! one string-option parser that mirrors [`crate::codegen::CodegenOptions`]:
//!
//! - [`PySimulation::to_c`](super::simulation) — the in-process path: builds the
//!   IR straight from the live model (`module_from_sim`, exactly like
//!   `compile()`) and lowers it, with no JSON round-trip.
//! - [`generate_c`] — the `fastsim.ir.Module` dataclass path: that object only
//!   holds JSON, so it deserializes the IR back into a `Module` first.
//!
//! Folding codegen into the `fastsim` extension (rather than a separate
//! `fastsim_codegen` module) is what lets `to_c` skip the JSON detour: codegen
//! is just another in-tree IR backend alongside `compile`.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::codegen::{
    CodegenOptions, GeneratedFile, Layout, ModelApi, Numeric, Reductions,
    SolverChoice, Structure,
};

/// Resolve a string option against its allowed values, or raise `ValueError`
/// listing them (so a typo in `layout="libary"` gives an actionable message).
fn parse_opt<T: Copy>(field: &str, value: &str, table: &[(&str, T)]) -> PyResult<T> {
    table
        .iter()
        .find(|(k, _)| *k == value)
        .map(|(_, v)| *v)
        .ok_or_else(|| {
            let allowed: Vec<&str> = table.iter().map(|(k, _)| *k).collect();
            PyValueError::new_err(format!(
                "{field}: unknown value {value:?}; expected one of {allowed:?}"
            ))
        })
}

/// Resolve the `solver=` kwarg to a [`SolverChoice`]. Accepts `"euler"`
/// (forward Euler) and every explicit tableau name in the runtime registry,
/// case-insensitively (`"rk4"`, `"rkdp54"`, `"rkck54"`, `"rkf45"`, ...). One
/// uniform path: there is no special-cased solver, just the tableau registry.
fn resolve_solver(solver: &str) -> PyResult<SolverChoice> {
    let name = if solver.eq_ignore_ascii_case("euler") || solver.eq_ignore_ascii_case("forward_euler") {
        "EUF".to_string()
    } else {
        solver.to_ascii_uppercase()
    };
    SolverChoice::by_name(&name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "solver: unknown or non-explicit value {solver:?}; expected \"euler\" or an explicit \
             tableau (rk4, ssprk22, ssprk33, ssprk34, rkf21, rkbs32, rkf45, rkck54, rkdp54, \
             rkv65, rkf78, rkdp87)"
        ))
    })
}

/// Resolve the `numeric=` kwarg: `"double"`, `"float"`, `"fixed"` (Q16.16),
/// or an explicit Q format `"qM.N"` with `M + N == 32` signed bits (the sign
/// bit counts toward `M`), e.g. `"q16.16"`, `"q4.28"`.
fn resolve_numeric(numeric: &str) -> PyResult<Numeric> {
    match numeric {
        "double" => return Ok(Numeric::Double),
        "float" => return Ok(Numeric::Float),
        "fixed" => return Ok(Numeric::Fixed { frac: crate::codegen::DEFAULT_FIXED_FRAC }),
        _ => {}
    }
    if let Some(rest) = numeric.strip_prefix('q') {
        if let Some((m, n)) = rest.split_once('.') {
            if let (Ok(m), Ok(n)) = (m.parse::<u32>(), n.parse::<u32>()) {
                if m + n == 32 && (1..=30).contains(&n) {
                    return Ok(Numeric::Fixed { frac: n as u8 });
                }
                return Err(PyValueError::new_err(format!(
                    "numeric: Q format q{m}.{n} must satisfy M + N == 32 with 1 <= N <= 30                      (int32 storage; the sign bit counts toward M)"
                )));
            }
        }
    }
    Err(PyValueError::new_err(format!(
        "numeric: unknown value {numeric:?}; expected \"double\", \"float\",          \"fixed\" (= q16.16) or an explicit \"qM.N\" with M + N == 32"
    )))
}

/// Build [`CodegenOptions`] from the string kwargs shared by `to_c` and
/// `generate_c`. Each unknown value raises `ValueError` via [`parse_opt`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn options_from_strs(
    numeric: &str,
    reductions: &str,
    structure: &str,
    layout: &str,
    solver: &str,
    api: &str,
    scaffold: bool,
    trace: bool,
    a2l: bool,
) -> PyResult<CodegenOptions> {
    Ok(CodegenOptions {
        scaffold,
        trace,
        a2l,
        numeric: resolve_numeric(numeric)?,
        reductions: parse_opt(
            "reductions",
            reductions,
            &[("unrolled", Reductions::Unrolled), ("vectorized", Reductions::Vectorized)],
        )?,
        structure: parse_opt(
            "structure",
            structure,
            &[("hierarchical", Structure::Hierarchical), ("flat", Structure::Flat)],
        )?,
        layout: parse_opt(
            "layout",
            layout,
            &[("compact", Layout::Compact), ("library", Layout::Library)],
        )?,
        solver: resolve_solver(solver)?,
        api: parse_opt("api", api, &[("struct", ModelApi::Struct)])?,
    })
}

/// Lower `module` with `opts` and box the result into a Python dict
/// `{file name: source}`, preserving emission order. `Unsupported` constructs
/// surface as `RuntimeError`. Logs the configuration and the emitted files on
/// `log` (`Simulation.to_c` passes the simulation's logger).
pub(crate) fn generate_to_dict<'py>(
    py: Python<'py>,
    module: &crate::ir::schema::Module,
    opts: &CodegenOptions,
    log: &crate::utils::logger::Logger,
) -> PyResult<Bound<'py, PyDict>> {
    let files: Vec<GeneratedFile> = crate::codegen::generate_logged(module, opts, log)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out = PyDict::new(py);
    for f in files {
        out.set_item(f.name, f.contents)?;
    }
    Ok(out)
}

/// Generate C source from a fastsim IR model in JSON form.
///
/// `ir_json` is the hierarchical IR as emitted by
/// ``fastsim.Simulation.to_ir_json()`` (or ``Module.to_json()``). Prefer
/// ``Simulation.to_c(...)`` when you have a live simulation — it skips the JSON
/// round-trip; this entry exists for the ``fastsim.ir.Module`` dataclass, which
/// only carries JSON. Returns a dict mapping each file name to its C source.
/// ``layout="compact"`` yields ``model.h`` + ``model.c``; ``layout="library"``
/// additionally splits out ``solver.{c,h}`` (the integrator) and, under the
/// hierarchical structure, ``blocks.{c,h}`` (the per-block functions). Compile
/// the ``.c`` files together.
///
/// Each option is a string; the choices mirror the Rust ``CodegenOptions``:
///
/// - ``numeric``: ``"double"`` (default), ``"float"``, or fixed point on
///   ``int32``: ``"fixed"`` (= q16.16) / ``"qM.N"`` with ``M + N == 32`` —
///   integer Q arithmetic with int64 intermediates; transcendentals are
///   rejected (model them with LUT1D), fixed-step tableaus only. The header
///   emits ``<NAME>_Q_*`` conversion macros.
/// - ``reductions``: ``"unrolled"`` (default) or ``"vectorized"`` (Reduce/Dot as
///   a counted loop over a coefficient array).
/// - ``structure``: ``"hierarchical"`` (default; one function per block, readable)
///   or ``"flat"`` (one fused ``dx/dt``).
/// - ``layout``: ``"compact"`` (default; ``.c`` + ``.h``) or ``"library"``
///   (multi-file split by concern).
/// - ``solver``: the integrator's Butcher tableau, by name (case-insensitive).
///   ``"rk4"`` (default) and ``"euler"`` are fixed-step; the adaptive methods
///   (``"rkdp54"``, ``"rkck54"``, ``"rkf45"``, ``"rkf78"``, ``"rkv65"``,
///   ``"rkbs32"``, ``"rkf21"``, ``"rkdp87"``) emit the embedded-error step
///   controller; fixed-step ``"ssprk22"``/``"ssprk33"``/``"ssprk34"`` are also
///   available. Implicit (DIRK/ESDIRK) tableaus are not yet emitted.
/// - ``scaffold``: ``False`` (default) or ``True`` — additionally emit an
///   EDITABLE build scaffold: ``CMakeLists.txt`` (static model library + a
///   ``<name>_demo`` executable) and ``<name>_main.c`` (a demo driver stepping
///   the model via ``<name>_step`` with marked HAL hook points).
/// - ``trace``: ``False`` (default) or ``True`` — additionally emit
///   ``<name>_trace.json``, the model-to-code trace map (block → emitted
///   functions with file/line, block → states/outputs/params with their
///   ``SIG_*`` ids, block → events) plus static metrics (packed RAM estimate,
///   integrator stack estimate, IR op counts, per-step work).
/// - ``a2l``: ``False`` (default) or ``True`` — additionally emit
///   ``<name>.a2l``, an ASAP2 measurement/calibration description (MEASUREMENT
///   for time/states/outputs/inputs/memory, CHARACTERISTIC for parameters)
///   addressed via ``SYMBOL_LINK`` against one global model instance plus
///   computed struct offsets — ready for XCP tooling (CANape, INCA).
/// - ``api``: ``"struct"`` (the only API): a single ``model_t`` holding time /
///   states / signals / parameters / memory, with ``get_signal`` / ``set_signal``
///   accessors by id. Reentrant by construction (each instance owns its state)
///   and embeddable (inputs are set through ``set_signal``).
///
/// The emitted API (``<name>_t``, ``<NAME>_SIG_*`` ids, ``<name>_run`` semantics,
/// event handling, ``jvp``), the C99 + libm requirement, the model-name symbol
/// prefixing, and the ABI stability policy are specified in ``doc/codegen.md``.
///
/// Raises ``ValueError`` for malformed IR JSON or an unknown option value, and
/// ``RuntimeError`` if the model uses a construct the backend does not lower
/// (e.g. an opaque ``extern`` block, or an unsupported option combination).
#[pyfunction]
#[pyo3(signature = (
    ir_json, *,
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
pub fn generate_c<'py>(
    py: Python<'py>,
    ir_json: &str,
    numeric: &str,
    reductions: &str,
    structure: &str,
    layout: &str,
    solver: &str,
    api: &str,
    scaffold: bool,
    trace: bool,
    a2l: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let module: crate::ir::schema::Module = serde_json::from_str(ir_json)
        .map_err(|e| PyValueError::new_err(format!("invalid IR JSON: {e}")))?;
    let opts = options_from_strs(numeric, reductions, structure, layout, solver, api, scaffold, trace, a2l)?;
    // No simulation (and thus no per-sim log flag) here — log like the
    // Simulation default (log=True).
    let log = crate::utils::logger::Logger::new(true, "");
    generate_to_dict(py, &module, &opts, &log)
}

/// Locate the C compiler ``Simulation.verify_c`` would use, or ``None``.
///
/// Resolution order: ``$FASTSIM_CC``, ``$CC``, then ``cc`` / ``clang`` /
/// ``gcc`` on PATH. A candidate must compile AND run a floating-point + libm
/// probe. Lets callers (and the test suite) check tool availability without
/// running a verification.
#[cfg(not(target_family = "wasm"))]
#[pyfunction]
pub fn find_c_compiler(py: Python<'_>) -> Option<String> {
    py.detach(crate::codegen::verify::find_compiler)
}
