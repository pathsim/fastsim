# `src/solvers/` — ODE integrators

All per-block integrators: explicit Runge-Kutta (fixed-step and adaptive),
DIRK/ESDIRK (implicit), Euler forward (EUF, one-stage explicit), Euler
backward (EUB, one-stage implicit), and SteadyState (iterative root-finder
for `f(x) = 0`). Each solver advances a block's continuous state
`x(t) → x(t+dt)` given its RHS `ẋ = f(x, u, t)`.

## Theory

A solver is a Butcher tableau (coefficients `A`, `b`, `c`, plus
error-weights `b̂`) + a state machine. For each stage `i = 1..s`:

- **Explicit RK**: `k_i = f(x_0 + dt · Σ_{j<i} a_{ij} k_j)` — evaluate and go.
- **Implicit DIRK**: `k_i = f(x_0 + dt · Σ_{j<i} a_{ij} k_j + dt · a_{ii} k_i)`
  — solve for `k_i` with an implicit-equation solver (`optim::Anderson` or
  `NewtonAnderson`, picked by `factories.rs`).

After all stages, the solver builds `x_{new} = x_0 + dt · Σ b_i k_i` and,
for adaptive variants, estimates the error via `Σ (b_i - b̂_i) k_i`.

## Implementation

- `solver.rs` — the single `Solver` struct (same pattern as `Block`: one
  struct with overridable callback fields). Holds the numerical state
  (`x`, `initial_value`, `history`), tolerances (`tolerance_lte_{abs,rel}`),
  the current stage index plus the RK working data (`ks`, `bt`, `tr`,
  `a_final`, `eval_stages`, embedded order `m`, safety factor `beta`), and
  the three overridable closures `step_fn` / `solve_fn` / `buffer_fn` plus
  an optional `stage_builder` for the implicit-stage Newton solve.
  Exposes an `Option<Optimizer>` (set by implicit factories). `Jacobian`
  enum in the same file unifies scalar and flat-matrix Jacobians passed
  into `solve_fn`. `ExplicitSolver` / `ImplicitSolver` are thin newtype
  wrappers used only by the standalone `integrate()` Python API.
- `stage.rs` — the `StageBuilder` trait and its three implementations:
  `OdeStageBuilder` (`r = x − x_0 − dt·Σa·k`, Newton matrix `I − dt·a_ii·J`),
  `MassMatrixStageBuilder` (`r = M·(x − x_0) − dt·Σa·M·k`, matrix
  `M − dt·a_ii·J`), `FullyImplicitStageBuilder` (`r = F(x, ẋ, t)`).
  SemiExplicit DAE blocks reduce to plain ODE form via inner `z`-elimination
  inside their constructor, so they reuse `OdeStageBuilder` rather than
  carrying their own builder.  Every builder returns a *WRMS-scaled*
  residual norm (via `Solver::wrms_norm` / `wrms_norm_diff`), which is
  what `Simulation::_solve` checks against `NLS_COEF`.  Also hosts the
  `Mass` struct, the `FiContext`/`FiFn` types for fully-implicit DAE
  blocks, and the utility helpers `num_jac_wrt_z` (central-difference
  ∂g/∂z) and `solve_xdot_for_f` (Newton on F(x,ẋ)=0).
- `tableaus.rs` — a `TableauKind` enum (ExplicitRK / DIRK / ESDIRK) and
  `Tableau` struct, plus `pub const` tableau definitions for every
  supported method (SSPRK22/33/34, RK4, RKBS32, RKF21/45/78, RKCK54,
  RKDP54/87, RKV65, DIRK2/3, ESDIRK4/32/43/54, EUF). Exposes a flat `ALL`
  slice and a `by_name(&str)` lookup.
- `factories.rs` — one generic `build_from_tableau` that populates a
  `Solver` from a `Tableau` and installs the right step/solve/buffer
  closures based on `TableauKind`; plus thin per-tableau wrappers
  (`rk4_factory`, `rkdp54_factory`, `esdirk43_factory`, ...). Special
  factories `euf_factory`, `eub_factory`, `steadystate_factory` exist
  for solvers that don't fit the RK schema. Implicit factories install
  `Optimizer::default_newton_anderson()` and attach an `OdeStageBuilder`.
  The Anderson history depth (`OPT_HISTORY = 4` default) is reconfigurable
  per Simulation via `Simulation::set_optimizer_history(m)` / the Python
  `optimizer_history=` kwarg.

## How it fits in

- `blocks/block.rs` holds `engine: Option<Solver>`. `Simulation::timestep`
  calls `block.buffer / solve / step`, which delegate to the `Solver`'s
  installed closures.
- Implicit solvers drive `optim::Anderson` / `NewtonAnderson` for their
  stage equation (inner loop), while the outer `Simulation::_solve` loop
  couples multiple blocks through the DAG.
- Adaptive solvers read `SOL_TOLERANCE_LTE_*`, `SOL_SCALE_*`, `SOL_BETA`
  from `constants.rs`.  Inner-Newton convergence is governed by `NLS_COEF`
  (also in `constants.rs`) applied to the WRMS-scaled residual the
  StageBuilders return — `tolerance_lte_abs` and `tolerance_lte_rel`
  double as the WRMS weights, so there is no separate `tolerance_fpi`
  knob.

## Convergence semantics

Every implicit-stage solve, the algebraic-loop boosters
(`optim::ConnectionBooster`), and the steady-state operating-point loop
all share a single convergence test:

```
‖r_i / (atol + rtol·|x_i|)‖_RMS  <  NLS_COEF      // = 0.1
```

`r` is whatever residual the caller has (different per StageBuilder);
the WRMS scaling is computed by `Solver::wrms_norm` against the solver's
`tolerance_lte_abs/rel`.  The unitless threshold `NLS_COEF` matches
ARKODE/CVODES practice and keeps the inner Newton accuracy proportional
to the outer step's local-truncation tolerance.  This replaces an older
absolute `tolerance_fpi` floor that over-converged well-scaled state
components on multi-scale stiff problems (e.g. Robertson).

## Periodic steady state (shooting)

`pss.rs` adds matrix-free Anderson-accelerated shooting on top of any
inner ODE solver.  A PSS engine is a normal solver (RK / DIRK / GEAR /
DAE-extended) augmented with `Solver::pss_ext = Some(PssExt)`; during
the transient integration over one period `[0, T]` it delegates every
`step_fn` / `solve_fn` / `buffer_fn` / `stage_builder` / `opt` to the
inner solver, so all existing solver and DAE machinery passes through
unchanged.  Zero overhead on the hot path: `pss_ext` is `None` for
ordinary integrators.

The shooting fixed-point map `g(x_0) = x(T; x_0)` is closed by the
outer `Simulation::periodic_steady_state(period, inner_factory, ...)`
loop:

1. `sim.run(period, false, adaptive)` — integrate one period from the
   current `x_0` using the inner solver.
2. Per dynamic block, call `engine.pss_close_period()`, which runs one
   matrix-free Anderson step on `(x_start, x_end)`, mutates `x_start`
   toward the limit-cycle fixed point, copies it into `engine.x` as
   the IC for the next period, and clears the inner solver's transient
   bookkeeping (`history`, `err_prev`, `opt`).
3. Convergence: max WRMS-scaled residual `‖x(T) − x(0)‖` across all
   dynamic blocks compared against `NLS_COEF` — same semantics as
   every other implicit-stage / steady-state residual.
4. After convergence, one final transient run records the converged
   limit-cycle trajectory in Scope blocks.

Why matrix-free Anderson rather than full Newton shooting?  Anderson
needs only function evaluations (one period integration per outer
iteration) — no monodromy matrix `Φ = ∂x(T)/∂x(0)` to assemble or
factorize.  Converges in 5–15 iterations on smooth, mildly-coupled
periodic systems; degrades gracefully on stiff problems where true
Newton-shooting would be the next step up.

DAE blocks pass through without special handling: their
`engine_postprocess` (Block-level callback) installs the appropriate
`StageBuilder` on the PSS-augmented engine after the factory returns,
exactly the same path as with any other solver factory.

Discrete events fire inside the period integration as usual.
State-flipping events (e.g. a relay toggling mid-period) can make
`x(T; x_0)` non-smooth at the boundaries where the event activation
changes, which may stall Anderson convergence — diagnose by inspecting
the PSS warning if the iteration count hits `SIM_PSS_ITERATIONS_MAX`.

## Optimizations

- Butcher-tableau rows are stored as `Vec<f64>` (not a matrix) — zeros in
  strict-lower triangle are simply absent. Saves both memory and
  multiplications.
- `history` is a ring buffer (`VecDeque`) — adaptive rejected steps revert
  to the pre-step state without allocation.
- The `solve_fn` closure is picked once at block construction (via
  `factories.rs`), so the inner solve dispatches without any `match` in
  the hot path.
- For ESDIRK with first-stage `a₁₁ = 0`, the first stage is explicit —
  `OdeStageBuilder` detects this and skips the implicit solve.
