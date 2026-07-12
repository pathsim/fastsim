// FMI 3.0 Model-Exchange and Co-Simulation PyO3 bindings.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::blocks::fmu::{cosimulation_fmu, model_exchange_fmu};
use crate::fmi::FmiError;

use super::PyBlock;

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
/// start_values : dict[str, float], optional
///     Override start values for variables declared in `modelDescription.xml`.
///     Keys are variable names; values are floats (Float64-typed variables only).
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
    start_values: Option<HashMap<String, f64>>,
    tolerance: f64,
    verbose: bool,
) -> PyResult<PyBlock> {
    let blk = model_exchange_fmu(fmu_path, instance_name, start_values, tolerance, verbose)
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
/// start_values : dict[str, float], optional
///     Override start values for variables declared in `modelDescription.xml`.
/// dt : float, optional
///     Communication step size. If `None`, `DefaultExperiment.stepSize` from
///     the FMU is used; an error is raised if neither is available.
/// verbose : bool, optional
///     Forward INFO/WARNING log messages from the FMU's logger callback to
///     stderr (default: False).
#[pyfunction]
#[pyo3(signature = (
    fmu_path,
    instance_name = "fmu_instance",
    start_values = None,
    dt = None,
    verbose = false,
))]
#[allow(non_snake_case)]
pub(super) fn CoSimulationFMU(
    fmu_path: &str,
    instance_name: &str,
    start_values: Option<HashMap<String, f64>>,
    dt: Option<f64>,
    verbose: bool,
) -> PyResult<PyBlock> {
    let blk = cosimulation_fmu(fmu_path, instance_name, start_values, dt, verbose)
        .map_err(fmi_err_to_py)?;
    Ok(PyBlock::wrap(blk))
}
