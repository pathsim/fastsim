# `src/pybindings/` — PyO3 bindings

Exposes the Rust core to Python as a drop-in replacement for pathsim's
public API (`Simulation`, `Block`, `Connection`, events, solvers). One
compiled `_fastsim` shared library, built via `maturin`.

## Theory

PyO3 builds a Python extension module from Rust code annotated with
`#[pyclass]` / `#[pymethods]` / `#[pyfunction]`. Each public pathsim type
or factory gets a Rust wrapper that holds an `Rc<FastCell<T>>` pointing
into the engine. Attribute access and method calls are thin forwards
into the core — **the hot path never re-enters Python**; it runs the
tape interpreter (`jit::InterpretedFn::call_into`) or the FMU FFI.

## Layout

`mod.rs` declares `pub mod pybindings;` and delegates to `py/mod.rs`,
which is the real entry point. The split into focused submodules comes
from the original single-file `pybindings.rs` (~3100 LOC) being broken
up for navigability. The `#[pymodule]` registration happens in
`py/mod.rs` and ties the submodules together.

## Implementation

- `py/mod.rs` — `#[pymodule] fn _fastsim` registration. Imports and
  re-exports every Python-visible type from the submodules and plugs
  them into the module object.
- `py/core.rs` — `PyBlock`, `PySolver`, `PyPortRef`, `PyConnection`.
  The foundation types that every other wrapper references (block
  indexing `blk[i]`, `Connection(src, dst)`, etc.).
- `py/simulation.rs` — `PySimulation` plus a classy class per solver
  (`RKCK54`, `RKDP54`, `ESDIRK43`, …) that the user instantiates and
  passes as `Solver=` to `Simulation`. Roughly one `#[pyclass]` per
  Butcher tableau.
- `py/blocks.rs` — ~90 factory wrappers (`ODE`, `Integrator`,
  `Amplifier`, DAE variants, signal sources, scope/spectrum, nonlinear
  blocks, control blocks). Defines the `block!` macro (~46 use-sites)
  that produces a `#[pyfunction]` wrapper in one line for the common
  1:1 passthrough-to-core-constructor shape.
- `py/jit.rs` — user-facing `JitFunction` / `JitJacobian` classes for
  direct symbolic compilation (`jit_compile`, `jit_jacobian`) plus the
  `_trace_*` factories invoked by block constructors that need lazy
  tracing (`_trace_dynamical_system`, `_trace_wrapper`, the three DAE
  tracers). The tracing goes through `jit::tracer::trace_with_signature`
  — this module is just the PyO3 surface.
- `py/events.rs` — `ZeroCrossing*`, `Schedule*`, `Condition`,
  `Diagnostics`. Each maps to a core event type in `src/events/`.
- `py/fmi.rs` — `ModelExchangeFMU` / `CoSimulationFMU` constructors,
  thin wrappers around `blocks::fmu::{model_exchange_fmu,
  cosimulation_fmu}`.
- `py/helpers.rs` — extraction utilities shared across submodules
  (`extract_initial_value`, `extract_vec_f64`, `compile_jacobian`,
  `attach_jacobian`).
- `py/lazy.rs` — `LazyTraced` wrapper: shape-keyed cache of compiled
  JIT graphs. Handles the "block input shapes only resolve after
  Connection resolution" case by retracing transparently when
  `shape_key` misses. The fast path is two `usize` comparisons plus
  `InterpretedFn::call_into`; the slow path re-enters Python to rebuild
  the graph.

## How it fits in

- `maturin develop --release` → builds the cdylib, links it as
  `fastsim._fastsim`, and exposes the `#[pymodule]` registration.
- `fastsim/__init__.py` (Python side) re-exports the names from the
  compiled module into the `fastsim` namespace, matching pathsim's
  top-level layout.
- `src/blocks/constructors/*` is the pure-Rust counterpart to
  `py/blocks.rs` — each pathsim block has a core factory there plus a
  thin PyO3 wrapper here.

## Optimizations

- **`Rc<FastCell<T>>` everywhere**: PyO3 classes hold shared references
  to the engine-owned block instances. `FastCell` is an `UnsafeCell`
  wrapper (no runtime borrow checking) — the single-threaded Python
  GIL enforces exclusive access.
- **Lazy JIT retrace** (`lazy.rs`): compiled graphs are cached keyed on
  input shapes. Shape-stable calls hit the fast path with zero
  allocations; shape changes trigger a transparent retrace + cache
  update. The common case is one trace and then thousands of fast
  evaluations per simulation.
- **`block!` macro** (`py/blocks.rs`): eliminates boilerplate for the
  many 1:1 constructor wrappers. One macro invocation per block type
  gives a `#[pyfunction]` with the right signature, correct
  `extract_initial_value`, and the right default return type.
