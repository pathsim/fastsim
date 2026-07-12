// PyO3 Python bindings for fastsim
// Drop-in replacement for pathsim's core API (Simulation, Block, Connection, Events).
//
// This module is split into focused submodules:
//   - `core`       — PyBlock, PySolver, PyPortRef, PyConnection.
//   - `simulation` — PySimulation + standalone Solver classes.
//   - `blocks`     — ~90 `#[pyfunction]` block constructor wrappers.
//   - `jit`        — JitFunction/JitJacobian + traced-block constructors.
//   - `events`     — ZeroCrossing*, Schedule*, Condition, Diagnostics.
//   - `fmi`        — FMI 3.0 Model-Exchange and Co-Simulation bindings.
//   - `helpers`    — PyO3 extraction/conversion helpers.

use pyo3::prelude::*;

pub(crate) mod lazy;
mod helpers;
pub use helpers::{extract_initial_value, compile_jacobian, attach_jacobian};
use helpers::StopSimulation;

mod core;
pub use core::{PyBlock, PyConnection, PyPortRef, PySolver};

mod events;
use events::{
    PyDiagnostics,
    PyZeroCrossing, PyZeroCrossingUp, PyZeroCrossingDown,
    PySchedule, PyScheduleList, PyCondition,
};

mod jit;
use jit::{
    PyJitFunction, PyJitJacobian, jit_compile, jit_jacobian,
    _trace_dynamical_system, _trace_dynamical_function, _trace_wrapper,
    _trace_mass_matrix_dae, _trace_fully_implicit_dae, _trace_semi_explicit_dae,
    _trace_bvp1d, _trace_algebraic_constraint,
};

mod simulation;
use simulation::{
    PySimulation, PyPendingOps, PyCompiledSimulation,
    PySSPRK22, PySSPRK33, PySSPRK34, PyRK4, PyEUF, PyEUB,
    PyRKF21, PyRKBS32, PyRKF45, PyRKCK54, PyRKDP54, PyRKV65, PyRKF78, PyRKDP87,
    PyDIRK2, PyDIRK3, PyESDIRK4, PyESDIRK32, PyESDIRK43, PyESDIRK54, PyGEAR52A, PySteadyState,
};

mod blocks;
use blocks::{
    Integrator, Amplifier, Adder, Multiplier, Constant, Source, Scope, Spectrum,
    Function, ODE, MassMatrixDAE, SemiExplicitDAE, FullyImplicitDAE, StateSpace,
    PT1, PT2, LeadLag, PID, StepSource, Step, SinusoidalSource, TriangleWaveSource,
    SquareWaveSource, Divider, Sin, Cos, Exp, Abs, Sqrt, Log, Tanh, Tan, Atan, Sinh,
    Cosh, Log10, Pow, Clip, Rescale, Atan2, Mod, Norm, PowProd, Polynomial, Matrix, Alias,
    GreaterThan, LessThan, Equal, LogicAnd, LogicOr, LogicNot, Pulse, Clock,
    ClockSource, PulseSource, ChirpSource, ChirpPhaseNoiseSource,
    SinusoidalPhaseNoiseSource, GaussianPulseSource, WhiteNoise, PinkNoise,
    RandomNumberGenerator, DynamicalSystem, DynamicalFunction, Differentiator,
    Delay, SampleHold, ZeroOrderHold, FirstOrderHold, Wrapper, FIR, ADC, DAC,
    DiscreteIntegrator, DiscreteDerivative, DiscreteStateSpace,
    DiscreteTransferFunction, TappedDelay,
    Comparator, Switch, Relay, Counter,
    CounterUp, CounterDown, AntiWindupPID, RateLimiter, Backlash, Deadband,
    TransferFunctionNumDen, TransferFunctionPRC, TransferFunction,
    TransferFunctionZPG, ButterworthLowpassFilter, ButterworthHighpassFilter,
    ButterworthBandpassFilter, ButterworthBandstopFilter, AllpassFilter, LUT1D,
    Interface, Subsystem,
};

#[cfg(feature = "fmi")]
mod fmi;
#[cfg(feature = "fmi")]
use fmi::{ModelExchangeFMU, CoSimulationFMU};

#[cfg(feature = "codegen")]
mod codegen;
#[cfg(feature = "codegen")]
use codegen::generate_c;

#[pymodule]
fn _fastsim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBlock>()?;
    m.add_class::<PyPortRef>()?;
    m.add_class::<PyConnection>()?;
    m.add_class::<PySimulation>()?;
    m.add_class::<PyPendingOps>()?;
    m.add_class::<PyCompiledSimulation>()?;

    // pathsim-compatible stop exception (raise from a block/event to stop a run)
    m.add("StopSimulation", m.py().get_type::<StopSimulation>())?;

    // Block constructors — basic
    m.add_function(wrap_pyfunction!(Integrator, m)?)?;
    m.add_function(wrap_pyfunction!(Amplifier, m)?)?;
    m.add_function(wrap_pyfunction!(Adder, m)?)?;
    m.add_function(wrap_pyfunction!(Multiplier, m)?)?;
    m.add_function(wrap_pyfunction!(Divider, m)?)?;
    m.add_function(wrap_pyfunction!(Constant, m)?)?;
    m.add_function(wrap_pyfunction!(Source, m)?)?;
    m.add_function(wrap_pyfunction!(Scope, m)?)?;
    m.add_function(wrap_pyfunction!(Spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(Function, m)?)?;
    m.add_function(wrap_pyfunction!(ODE, m)?)?;
    m.add_function(wrap_pyfunction!(MassMatrixDAE, m)?)?;
    m.add_function(wrap_pyfunction!(SemiExplicitDAE, m)?)?;
    m.add_function(wrap_pyfunction!(_trace_semi_explicit_dae, m)?)?;
    m.add_function(wrap_pyfunction!(_trace_mass_matrix_dae, m)?)?;
    m.add_function(wrap_pyfunction!(_trace_fully_implicit_dae, m)?)?;
    m.add_function(wrap_pyfunction!(_trace_dynamical_system, m)?)?;
    m.add_function(wrap_pyfunction!(_trace_dynamical_function, m)?)?;
    m.add_function(wrap_pyfunction!(_trace_wrapper, m)?)?;
    m.add_function(wrap_pyfunction!(_trace_bvp1d, m)?)?;
    m.add_function(wrap_pyfunction!(_trace_algebraic_constraint, m)?)?;
    m.add_function(wrap_pyfunction!(FullyImplicitDAE, m)?)?;

    // Block constructors — LTI / control
    m.add_function(wrap_pyfunction!(StateSpace, m)?)?;
    m.add_function(wrap_pyfunction!(PT1, m)?)?;
    m.add_function(wrap_pyfunction!(PT2, m)?)?;
    m.add_function(wrap_pyfunction!(LeadLag, m)?)?;
    m.add_function(wrap_pyfunction!(PID, m)?)?;

    // Block constructors — sources
    m.add_function(wrap_pyfunction!(StepSource, m)?)?;
    m.add_function(wrap_pyfunction!(Step, m)?)?;
    m.add_function(wrap_pyfunction!(SinusoidalSource, m)?)?;
    m.add_function(wrap_pyfunction!(TriangleWaveSource, m)?)?;
    m.add_function(wrap_pyfunction!(SquareWaveSource, m)?)?;
    m.add_function(wrap_pyfunction!(Pulse, m)?)?;
    m.add_function(wrap_pyfunction!(PulseSource, m)?)?;
    m.add_function(wrap_pyfunction!(Clock, m)?)?;
    m.add_function(wrap_pyfunction!(ClockSource, m)?)?;
    m.add_function(wrap_pyfunction!(ChirpSource, m)?)?;
    m.add_function(wrap_pyfunction!(ChirpPhaseNoiseSource, m)?)?;
    m.add_function(wrap_pyfunction!(SinusoidalPhaseNoiseSource, m)?)?;
    m.add_function(wrap_pyfunction!(GaussianPulseSource, m)?)?;
    m.add_function(wrap_pyfunction!(WhiteNoise, m)?)?;
    m.add_function(wrap_pyfunction!(PinkNoise, m)?)?;
    m.add_function(wrap_pyfunction!(RandomNumberGenerator, m)?)?;

    // Block constructors — math
    m.add_function(wrap_pyfunction!(Sin, m)?)?;
    m.add_function(wrap_pyfunction!(Cos, m)?)?;
    m.add_function(wrap_pyfunction!(Exp, m)?)?;
    m.add_function(wrap_pyfunction!(Abs, m)?)?;
    m.add_function(wrap_pyfunction!(Sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(Log, m)?)?;
    m.add_function(wrap_pyfunction!(Tanh, m)?)?;
    m.add_function(wrap_pyfunction!(Tan, m)?)?;
    m.add_function(wrap_pyfunction!(Atan, m)?)?;
    m.add_function(wrap_pyfunction!(Sinh, m)?)?;
    m.add_function(wrap_pyfunction!(Cosh, m)?)?;
    m.add_function(wrap_pyfunction!(Log10, m)?)?;
    m.add_function(wrap_pyfunction!(Pow, m)?)?;
    m.add_function(wrap_pyfunction!(Clip, m)?)?;
    m.add_function(wrap_pyfunction!(Rescale, m)?)?;
    m.add_function(wrap_pyfunction!(Atan2, m)?)?;
    m.add_function(wrap_pyfunction!(Mod, m)?)?;
    m.add_function(wrap_pyfunction!(Norm, m)?)?;
    m.add_function(wrap_pyfunction!(PowProd, m)?)?;
    m.add_function(wrap_pyfunction!(Polynomial, m)?)?;
    m.add_function(wrap_pyfunction!(Matrix, m)?)?;
    m.add_function(wrap_pyfunction!(Alias, m)?)?;

    // Block constructors — logic
    m.add_function(wrap_pyfunction!(GreaterThan, m)?)?;
    m.add_function(wrap_pyfunction!(LessThan, m)?)?;
    m.add_function(wrap_pyfunction!(Equal, m)?)?;
    m.add_function(wrap_pyfunction!(LogicAnd, m)?)?;
    m.add_function(wrap_pyfunction!(LogicOr, m)?)?;
    m.add_function(wrap_pyfunction!(LogicNot, m)?)?;

    // Block constructors — medium-effort
    m.add_function(wrap_pyfunction!(Differentiator, m)?)?;
    m.add_function(wrap_pyfunction!(Delay, m)?)?;
    m.add_function(wrap_pyfunction!(SampleHold, m)?)?;
    m.add_function(wrap_pyfunction!(ZeroOrderHold, m)?)?;
    m.add_function(wrap_pyfunction!(FirstOrderHold, m)?)?;
    m.add_function(wrap_pyfunction!(Wrapper, m)?)?;
    m.add_function(wrap_pyfunction!(FIR, m)?)?;
    m.add_function(wrap_pyfunction!(DiscreteIntegrator, m)?)?;
    m.add_function(wrap_pyfunction!(DiscreteDerivative, m)?)?;
    m.add_function(wrap_pyfunction!(DiscreteStateSpace, m)?)?;
    m.add_function(wrap_pyfunction!(DiscreteTransferFunction, m)?)?;
    m.add_function(wrap_pyfunction!(TappedDelay, m)?)?;
    m.add_function(wrap_pyfunction!(ADC, m)?)?;
    m.add_function(wrap_pyfunction!(DAC, m)?)?;
    m.add_function(wrap_pyfunction!(Comparator, m)?)?;
    m.add_function(wrap_pyfunction!(Switch, m)?)?;
    m.add_function(wrap_pyfunction!(Relay, m)?)?;
    m.add_function(wrap_pyfunction!(Counter, m)?)?;
    m.add_function(wrap_pyfunction!(CounterUp, m)?)?;
    m.add_function(wrap_pyfunction!(CounterDown, m)?)?;
    m.add_function(wrap_pyfunction!(AntiWindupPID, m)?)?;
    m.add_function(wrap_pyfunction!(RateLimiter, m)?)?;
    m.add_function(wrap_pyfunction!(Backlash, m)?)?;
    m.add_function(wrap_pyfunction!(Deadband, m)?)?;
    m.add_function(wrap_pyfunction!(TransferFunctionNumDen, m)?)?;
    m.add_function(wrap_pyfunction!(TransferFunction, m)?)?;
    m.add_function(wrap_pyfunction!(TransferFunctionPRC, m)?)?;
    m.add_function(wrap_pyfunction!(TransferFunctionZPG, m)?)?;
    m.add_function(wrap_pyfunction!(ButterworthLowpassFilter, m)?)?;
    m.add_function(wrap_pyfunction!(ButterworthHighpassFilter, m)?)?;
    m.add_function(wrap_pyfunction!(ButterworthBandpassFilter, m)?)?;
    m.add_function(wrap_pyfunction!(ButterworthBandstopFilter, m)?)?;
    m.add_function(wrap_pyfunction!(AllpassFilter, m)?)?;
    m.add_function(wrap_pyfunction!(LUT1D, m)?)?;
    m.add_function(wrap_pyfunction!(Interface, m)?)?;
    m.add_function(wrap_pyfunction!(Subsystem, m)?)?;
    m.add_function(wrap_pyfunction!(DynamicalSystem, m)?)?;
    m.add_function(wrap_pyfunction!(DynamicalFunction, m)?)?;

    // Solver classes
    m.add_class::<PySSPRK22>()?;
    m.add_class::<PySSPRK33>()?;
    m.add_class::<PySSPRK34>()?;
    m.add_class::<PyRK4>()?;
    m.add_class::<PyEUF>()?;
    m.add_class::<PyEUB>()?;
    m.add_class::<PyRKF21>()?;
    m.add_class::<PyRKBS32>()?;
    m.add_class::<PyRKF45>()?;
    m.add_class::<PyRKCK54>()?;
    m.add_class::<PyRKDP54>()?;
    m.add_class::<PyRKV65>()?;
    m.add_class::<PyRKF78>()?;
    m.add_class::<PyRKDP87>()?;
    m.add_class::<PyDIRK2>()?;
    m.add_class::<PyDIRK3>()?;
    m.add_class::<PyESDIRK4>()?;
    m.add_class::<PyESDIRK32>()?;
    m.add_class::<PyESDIRK43>()?;
    m.add_class::<PyESDIRK54>()?;
    m.add_class::<PyGEAR52A>()?;
    m.add_class::<PySteadyState>()?;

    // Event classes + Diagnostics
    m.add_class::<PyDiagnostics>()?;
    m.add_class::<PyZeroCrossing>()?;
    m.add_class::<PyZeroCrossingUp>()?;
    m.add_class::<PyZeroCrossingDown>()?;
    m.add_class::<PySchedule>()?;
    m.add_class::<PyScheduleList>()?;
    m.add_class::<PyCondition>()?;

    // Tracer
    crate::tracer::register(m)?;

    // Standalone JIT API
    m.add_class::<PyJitFunction>()?;
    m.add_class::<PyJitJacobian>()?;
    m.add_function(wrap_pyfunction!(jit_compile, m)?)?;
    m.add_function(wrap_pyfunction!(jit_jacobian, m)?)?;

    // Block constructors — FMU (FMI 3.0 import)
    #[cfg(feature = "fmi")]
    {
        m.add_function(wrap_pyfunction!(ModelExchangeFMU, m)?)?;
        m.add_function(wrap_pyfunction!(CoSimulationFMU, m)?)?;
    }

    // Code generation (IR JSON -> C). The in-process `Simulation.to_c` lives on
    // PySimulation; this free function serves the `fastsim.ir.Module` dataclass.
    #[cfg(feature = "codegen")]
    m.add_function(wrap_pyfunction!(generate_c, m)?)?;
    // SIL verification support (native only: needs a local compiler + process).
    #[cfg(all(feature = "codegen", not(target_family = "wasm")))]
    m.add_function(wrap_pyfunction!(codegen::find_c_compiler, m)?)?;

    Ok(())
}
