# `src/optim/` — Fixed-point accelerators

Solvers for nonlinear equations in fixed-point form `x = g(x)`, used by the
implicit ODE path and for algebraic-loop resolution. Core method is
Anderson acceleration (Type-II, default history depth `OPT_HISTORY = 4` —
runtime-configurable per Simulation via `set_optimizer_history(m)`).

## Theory

Fastsim's implicit integrators produce a per-stage fixed-point equation
`x = x_0 + dt · Σ a · f(x)`.  The caller iterates:

```
for k = 0, 1, ..., iterations_max:
    g_k = x_0 + dt · Σ a · f(x_k)                          # RHS evaluation
    x_{k+1} = step(x_k, g_k)                                # accelerated update
    if ‖(g_k − x_k) / (atol + rtol·|x_k|)‖_RMS  <  NLS_COEF: converged
```

The convergence check is a WRMS-scaled max-norm against the unitless
`NLS_COEF = 0.1` (CVODES/ARKODE convention), not an absolute residual
floor — see `solvers/README.md` for the rationale.

`step` is one of:

- **Plain Picard**: `x_{k+1} = g_k`. Linearly convergent if `‖g'‖ < 1`.
- **Anderson (Type-II)**: minimise `‖ Σ γ_j (g_{k−m+j} − x_{k−m+j}) ‖²`
  over the last `m` iterates, producing a superlinearly convergent
  update.  The thin LS solve uses faer's truncated SVD with the numpy
  `rcond=max(n,m)·ε_mach` cutoff — rank-deficient `dR` (typical near
  convergence when residual differences become near-collinear) is
  absorbed by singular-value truncation, no extra Tikhonov dampening
  needed.
- **NewtonAnderson** (hybrid): a block-local Newton step (via the
  Jacobian provided by the caller) followed by an Anderson correction
  on the resulting iterate.  This is the default for implicit ODE/DAE
  blocks: Newton handles the local stiffness via the analytical
  Jacobian, Anderson handles the **cross-block coupling** that no
  per-block Jacobian sees.

## Implementation

- `anderson.rs` — the whole optimizer core. Four public types:
  - `Anderson` — vector Anderson with rotating-scratch dx/dr ring
    buffers (no per-iteration allocations after warm-up).  History depth
    `m` is mutable via `set_m(m)`.
  - `NewtonAnderson` — scalar or flat-matrix Newton preconditioner
    wrapped around an inner `Anderson`.  Handles `Option<f64>` Jacobian
    for scalar blocks and `Option<&[f64]>` flat Jacobian for vector
    blocks.
  - `Newton` — plain Newton (used by the standalone `integrate()` path
    when the full Jacobian is known).
  - `Optimizer` enum — dispatches between the three, plus `default_*()`
    factory helpers the solver factories call.
  Internal helpers live in the same file: `lstsq_solve` uses a thin
  faer SVD with truncated pseudo-inverse, `newton_step_scalar_inplace`
  does the scalar Newton update. The matrix Newton path lives in
  `linsolve.rs` (the three optimizer types each hold a cached
  `LinearSolver` and route matrix solves through it).
- `linsolve.rs` — `LinearSolver`: the single dense/sparse faer-LU path for
  the Newton matrix solve, with content-based factorization caching (a
  cache hit on an unchanged Jacobian skips reassembling `A = jac − I` and
  the refactorization). `newton_step_matrix` (stage residual form) and
  `newton_solve` (raw residual form) are the methods; `newton_step_matrix_
  inplace` / `newton_solve_inplace` are one-shot convenience wrappers over a
  transient solver (no caching).
- `booster.rs` — `ConnectionBooster`: wraps a `Connection` and applies
  an Anderson step to the transferred values on every `update()`.
  Per-edge accelerator for algebraic-loop resolution: reads the source
  outputs, accelerates them against its stored history, writes the
  accelerated values to the target inputs, and returns the **WRMS-
  scaled** residual so the outer loop can check convergence consistently
  with the implicit-stage solves.  Stores its own `(atol, rtol)`
  weights, set by `Simulation` from the engine-level outer tolerances
  at booster construction.
- `mod.rs` — three lines: `pub mod anderson; pub mod booster; pub mod linsolve;`.

## How it fits in

- `solvers/stage.rs` builds the stage residual `g(x) = x_0 + dt·Σa·k` and
  calls `Optimizer::step` / `step_matrix`.  The StageBuilder returns the
  WRMS norm to its caller.
- `simulation.rs::_solve` drives the outer FPI loop over all dynamic
  blocks; each block's `solve()` invokes its own `Optimizer` and reports
  back its WRMS norm.  Convergence is the max-vs-`NLS_COEF` check.
- `simulation.rs::_loops` runs the algebraic-loop solve via the
  `ConnectionBooster`s — same WRMS / `NLS_COEF` semantics.
- On step rejection (`Solver::revert`), the optimizer's `dx/dr` buffers
  are cleared: each implicit RK stage has its own fixed-point equation
  `g_i(x)`, and on `dt` change every `g_i` changes, so the prior
  attempt's history would point in the wrong direction for the retry.

## Optimizations

- **Truncated SVD pseudo-inverse** for the LS solve absorbs rank-deficient
  `dR` cleanly (no Tikhonov, no step-magnitude safeguarding) — the
  earlier normal-equations Cholesky path squared the conditioning number
  and required Tikhonov + Pollock/Rebholz dampening to stay usable.
- **Scratch buffers reused**: `dx`, `dr`, `x_prev`, `r_prev`, `_step`,
  `_c` are `SmallVec<[f64; 8]>` fields; zero allocation after warm-up.
- **Scalar fast-path** for `n == 1` (one Integrator block) bypasses the
  LS solve entirely — it's the secant method in closed form, no matrix
  factorisation.
- **WRMS norm computation is fused** into a single pass over the
  residual (`Solver::wrms_norm` / `wrms_norm_diff`), allocation-free.
- **Configurable history depth**: small `m` (1–2) is often optimal for
  stiff problems with good analytical Jacobians (Newton dominates,
  Anderson contributes a single secant correction), while larger `m` is
  the right choice for purely algebraic loops.
