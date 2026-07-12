//! Python tracing frontend: one of several producers of an `ssa::graph::Graph`.
//!
//! A user's Python function is called with symbolic `JitTracer` arguments so
//! its operations record into an SSA graph; that graph is the deliverable and
//! everything downstream (tape lowering, autodiff, codegen) operates on it.
//! The block constructors are a second, native producer (via `ssa::build`);
//! this module is the Python one.
//!
//! The `ndshape` / `ufunc_table` helpers are pure (no PyO3) and also compile
//! under `test` for their own unit tests; the `frontend` itself is PyO3-gated.

#[cfg(any(feature = "python", test))]
pub mod ndshape;
#[cfg(any(feature = "python", test))]
pub mod ufunc_table;

#[cfg(feature = "python")]
mod frontend;
#[cfg(feature = "python")]
pub use frontend::*;
