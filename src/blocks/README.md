# `src/blocks/` — Block primitives & FMU integration

Defines the core `Block` type and the FMU wrapper. All concrete block kinds
(ODE, StateSpace, Filters, Logic, ...) live one level deeper in `constructors/`.

## Theory

A block is the atomic unit of a fastsim model. It has:
- **Inputs** (`inputs`) — flat f64 slice receiving upstream outputs
- **Outputs** (`outputs`) — flat f64 slice feeding downstream inputs
- **State** (via `engine: Solver`) — internal continuous/discrete state
- **RHS closures** — `f_dyn(x, u, t) -> ẋ`, `f_alg(x, u, t) -> y`,
  plus optional Jacobians
- **Type tag** — a string identifying the block kind, used for polymorphism
  in `match`-style dispatch

Blocks with an engine contribute to the ODE/DAE system. Algebraic-only
blocks are updated in topological order during algebraic-loop resolution.

## Implementation

- `block.rs` — the `Block` struct, its `BlockRole` (feedthrough / dynamic
  flags) and the `BlockRef = Rc<FastCell<Block>>` alias. `Block` owns its
  optional `engine: Solver`, its RHS closures (`f_dyn`, `f_alg`, `jac_dyn`,
  …), its `inputs`/`outputs` port buffers, and a whole family of
  `Box<dyn FnMut>` callback slots — `LenFn`, `UpdateFn`, `SolveFn`, `StepFn`,
  `ResetFn`, `SampleFn`, `BufferFn`, `RevertFn`, `SetSolverFn`, `OnOffFn`,
  `EnginePostprocessFn`, `BlockFn` — which `Simulation` invokes polymorphically
  per lifecycle phase. The specific closure set installed at construction
  time is what distinguishes one block kind from another.
- `fmu.rs` — adapts an external FMI 3.0 FMU into a fastsim block. Supports
  both Co-Simulation (`Instance<Cs>`) and Model Exchange (`Instance<Me>`).
  Delegates to `crate::fmi` for the low-level bindings and uses
  `utils::register::Register` to map FMI value-references to flat port
  indices. Schedules FMU event iterations via `events::schedule::ScheduleList`
  when the FMU declares internal events.
- `mod.rs` — re-exports public block types; pulls in `constructors/`.
- `constructors/` — factory functions that build specific block kinds.
  See its own README.

## How it fits in

- `Simulation` in the parent `src/` calls `Block::solve` / `step` / `update`
  and reads `Block::outputs` via `Connection`s.
- The `engine: Option<Solver>` field links to `solvers/` — presence of an
  engine makes the block "dynamic" (contributes to state vector).
- `solve_fn`/`step_fn` are installed by `solvers/factories.rs` based on the
  solver kind (explicit RK vs. implicit DIRK). This keeps integration
  strategy separate from block definition.
- JIT-traced RHS closures come from `jit/` via the `constructors/` factories.

## Optimizations

- Closures are `Box<dyn Fn>` chosen to match the block's arity — no runtime
  argument dispatch.
- `len_fn` lets a block report a dynamic output length (e.g. `Scope` depends
  on how many channels are connected), avoiding heap churn.
- FMU block holds the loaded library + archive for its lifetime — no repeat
  unzip on reset.
