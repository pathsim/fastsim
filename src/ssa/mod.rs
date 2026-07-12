//! The SSA graph: fastsim's single source of truth for compute.
//!
//! A hash-consed DAG of scalar ops with canonical f64 semantics (`graph`),
//! optimised in place (`optimize`), differentiated symbolically (`autodiff`),
//! and constructed either natively or symbolically through one `Builder`
//! abstraction (`build`). Every downstream consumer attaches here: the block
//! runtime derives its closures and Jacobians from it, `compile` fuses block
//! graphs into one and lowers it to a tape, `codegen` emits C from the IR
//! mirror of it, and the Python `tracer` produces it. PyO3-free and always
//! compiled; the tracer and codegen are clients.

pub mod op;
pub mod graph;
pub mod build;
pub mod optimize;
pub mod autodiff;
pub mod tape;
