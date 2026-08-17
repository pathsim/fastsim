// FMI 3.0 Model-Exchange and Co-Simulation PyO3 bindings.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::blocks::fmu::{cosimulation_fmu, model_exchange_fmu};
use crate::fmi::FmiError;

use super::PyBlock;

/// A `start_values` entry: either one value for the whole variable, or one per
/// array element. PyO3 tries the variants in order, so a Python float takes the
/// scalar path and any sequence of floats the vector path.
///
/// A scalar given for an array variable is broadcast across its elements, the
/// same rule FMI 3.0 §2.4.7.5 applies to the XML's own `start` attribute.
#[derive(FromPyObject)]
pub(super) enum StartOverride {
    Scalar(f64),
    Vector(Vec<f64>),
}

impl From<StartOverride> for Vec<f64> {
    fn from(o: StartOverride) -> Self {
        match o {
            StartOverride::Scalar(x) => vec![x],
            StartOverride::Vector(v) => v,
        }
    }
}

/// Normalize the Python-facing map to the core's `name -> values` form.
fn to_start_values(
    m: Option<HashMap<String, StartOverride>>,
) -> Option<HashMap<String, Vec<f64>>> {
    m.map(|m| m.into_iter().map(|(k, v)| (k, v.into())).collect())
}

fn fmi_err_to_py(e: FmiError) -> PyErr {
    match e {
        FmiError::UnknownVariable(_) => PyValueError::new_err(e.to_string()),
        FmiError::UnsupportedFmiVersion(_) => PyValueError::new_err(e.to_string()),
        FmiError::UnsupportedPlatform { .. } => {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        }
        _ => pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
    }
}

/// FMI 3.0 Model-Exchange FMU block.
///
/// Wraps a Model-Exchange FMU so its ODE right-hand-side (`GetDerivatives`)
/// is integrated by FastSim's solver. Event indicators become `ZeroCrossing`
/// block events; FMU-announced time events populate a `ScheduleList`.
///
/// Parameters
/// ----------
/// fmu_path : str
///     Path to the `.fmu` archive.
/// instance_name : str, optional
///     Name passed to `fmi3InstantiateModelExchange` (default: "fmu_instance").
/// start_values : dict[str, float | Sequence[float]], optional
///     Override start values for variables declared in `modelDescription.xml`.
///     Keys are variable names (Float64-typed variables only). A float sets the
///     variable, and for an array variable it is broadcast across every element.
///     A sequence sets the elements individually and must have exactly as many
///     entries as the array has elements.
/// tolerance : float, optional
///     Event-detection tolerance and `toleranceDefined` argument to
///     `fmi3EnterInitializationMode` (default: 1e-10).
/// verbose : bool, optional
///     Forward INFO/WARNING log messages from the FMU's logger callback to
///     stderr. Errors are always shown (default: False).
#[pyfunction]
#[pyo3(signature = (
    fmu_path,
    instance_name = "fmu_instance",
    start_values = None,
    tolerance = 1e-10,
    verbose = false,
))]
#[allow(non_snake_case)]
pub(super) fn ModelExchangeFMU(
    fmu_path: &str,
    instance_name: &str,
    start_values: Option<HashMap<String, StartOverride>>,
    tolerance: f64,
    verbose: bool,
) -> PyResult<PyBlock> {
    let blk = model_exchange_fmu(
        fmu_path,
        instance_name,
        to_start_values(start_values),
        tolerance,
        verbose,
    )
        .map_err(fmi_err_to_py)?;
    Ok(PyBlock::wrap(blk))
}

/// FMI 3.0 Co-Simulation FMU block.
///
/// Wraps a Co-Simulation FMU so its `DoStep` is invoked at fixed communication
/// points scheduled via a block-internal `Schedule` event. FMU-signaled
/// `eventEncountered` triggers the full Event-Mode handshake
/// (`EnterEventMode → drain UpdateDiscreteStates → EnterStepMode`).
///
/// Parameters
/// ----------
/// fmu_path : str
///     Path to the `.fmu` archive.
/// instance_name : str, optional
///     Name passed to `fmi3InstantiateCoSimulation` (default: "fmu_instance").
/// start_values : dict[str, float | Sequence[float]], optional
///     Override start values for variables declared in `modelDescription.xml`.
///     A float is broadcast across an array variable's elements; a sequence sets
///     them individually.
/// dt : float, optional
///     Communication step size. If `None`, `DefaultExperiment.stepSize` from
///     the FMU is used; an error is raised if neither is available.
/// tolerance : float, optional
///     Relative tolerance passed to `fmi3EnterInitializationMode`, guiding the
///     FMU's internal solver. Defaults to `DefaultExperiment.tolerance` from
///     the FMU, falling back to 1e-6.
/// verbose : bool, optional
///     Forward INFO/WARNING log messages from the FMU's logger callback to
///     stderr (default: False).
#[pyfunction]
#[pyo3(signature = (
    fmu_path,
    instance_name = "fmu_instance",
    start_values = None,
    dt = None,
    tolerance = None,
    verbose = false,
))]
#[allow(non_snake_case)]
pub(super) fn CoSimulationFMU(
    fmu_path: &str,
    instance_name: &str,
    start_values: Option<HashMap<String, StartOverride>>,
    dt: Option<f64>,
    tolerance: Option<f64>,
    verbose: bool,
) -> PyResult<PyBlock> {
    let blk = cosimulation_fmu(
        fmu_path,
        instance_name,
        to_start_values(start_values),
        dt,
        tolerance,
        verbose,
    )
        .map_err(fmi_err_to_py)?;
    Ok(PyBlock::wrap(blk))
}
