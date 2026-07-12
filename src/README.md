# `src/` — fastsim Rust core

The simulation engine. A block-diagram system is described as a set of
`Block`s connected by `Connection`s; `Simulation` drives them forward through
time, resolving algebraic loops and discrete events along the way.

## Theory

A fastsim model is a directed graph. Each `Block` owns internal state and
outputs that depend on its inputs (and optionally on its state's time
derivative). A `Connection` transports one block's outputs into another's
inputs. `Simulation::timestep` sequences the per-step work: buffer → solve
(implicit) or update (explicit) → step (advance integrator state) → event
resolution → sample. Adaptive stepping, fixed-point iteration on algebraic
loops and discrete event detection all live in this loop.

## Implementation

- `lib.rs` — crate entrypoint, re-exports public API
- `simulation.rs` — `Simulation` struct, `timestep`, `run`, event dispatch,
  adaptive step control, loop resolution (the hot path)
- `connection.rs` — `Connection` (directed edge between block ports),
  port-reference indirection
- `subsystem.rs` — hierarchical composition: a subsystem is a block that
  wraps its own block-graph and exposes an `Interface`
- `blocks/` — block definition, FMU wrapper, and every built-in block kind
  (see its own README)
- `solvers/`, `optim/`, `tracer/`, `ssa/`, `events/`, `fmi/`, `utils/` — see their own
  READMEs
- `constants.rs` — every numerical tolerance the engine uses; tuning happens
  here, not in the call sites
- `pybindings.rs` — PyO3 layer exposing the engine to Python

## How it fits in

The data flow is: Python API (`pybindings.rs`) constructs `Simulation`,
`Block`s, `Connection`s. `Simulation` drives `Block::step` / `Block::update`
which in turn delegate to `solvers/` for integration and `optim/` for
implicit-equation solving. User-provided RHS functions are routed through
`tracer/`+`ssa/` for automatic Rust-IR compilation when possible (plain-Python
callback fallback otherwise). Events from `events/` are hooked into
`timestep`. FMU blocks in `blocks/fmu.rs` use the `fmi/` bindings.

## Optimizations

- Flat DAG evaluation order precomputed once after topology change
  (`utils/graph.rs`) so the hot path avoids graph traversal.
- JIT compilation eliminates Python overhead on user-provided RHS.
- Block-output buffers are reused across steps (no per-step allocations).
- Reverted steps (adaptive rejection) are zero-cost: the pre-step state is
  kept in `history` and restored in-place.
