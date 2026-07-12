// fastsim — A fast, Rust-based block-diagram system simulation framework
// Drop-in replacement for pathsim with identical Python API via PyO3
//
// Architecture: Monolithic crate, bottom-up implementation
// Port strategy: 1:1 line-by-line from pathsim (Python)

// Crate-wide clippy allows for patterns that are deliberate here, so that
// `cargo clippy -- -D warnings` gates on real issues only:
// - non_snake_case: every block/solver `#[pyfunction]` mirrors a pathsim
//   PascalCase class name (Integrator, RK4, ...). Renaming them would break
//   the drop-in Python API, which is the whole point.
// - too_many_arguments / type_complexity: the solver factories and boxed
//   stage/closure types are inherently wide; splitting them hurts clarity.
// - needless_range_loop: index-based loops mirror the numerical math (and
//   often index several parallel arrays); the "idiomatic" rewrite is less
//   readable and error-prone in the hot path.
#![allow(non_snake_case)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]

pub mod constants;
pub mod error;
pub mod utils;
pub mod blocks;
pub mod connection;
pub mod optim;
pub mod solvers;
pub mod events;
pub mod simulation;
pub mod subsystem;
pub mod ssa;
#[cfg(any(feature = "python", test))]
pub mod tracer;
pub mod ir;
pub mod compile;
#[cfg(feature = "codegen")]
pub mod codegen;
#[cfg(feature = "fmi")]
pub mod fmi;
pub mod pybindings;
