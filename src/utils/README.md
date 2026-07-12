# `src/utils/` — Shared infrastructure

Grab-bag of support types that don't belong to any single subsystem:
interior-mutability cell, shape-aware register, diagnostics,
deterministic RNG, logging, portreference, DAG analysis, Gilbert
realization helper, numerical-Jacobian fallback.

## Implementation

- `fastcell.rs` — `FastCell<T>`: drop-in replacement for `RefCell<T>` built
  on `UnsafeCell<T>`. No runtime borrow check at all — the simulation loop
  guarantees exclusive access by construction (blocks borrowed in DAG
  order, Python callbacks synchronous, events don't re-enter their own
  block). Marked `Send` (safe because fastsim is single-threaded), never
  `Sync`.
- `register.rs` — `Register`: a named-port value container backed by a
  flat `Vec<f64>`. `PortKey` enum supports duck-typed access patterns
  (`int`, `str`, `slice`, `list`) that mirror pathsim's Python API. Used
  by every block for input/output storage and by FMU blocks to map FMI
  value-references to flat port indices.
- `portreference.rs` — `PortReference`: a typed handle to one or more ports
  of a `BlockRef`. `Port` enum selects between "all ports" (implicit),
  a single index, a slice range, or an explicit index list. Used to assemble
  `Connection`s without copying block references.
- `graph.rs` — directed-graph analysis: DFS, Tarjan SCC (for algebraic-loop
  detection), topological sort, precomputation of flat evaluation order
  for `Simulation`. All look-ups use `Vec`-based indexing (NodeIds are a
  contiguous 0..n range), no HashMap in the hot path. Rebuilt only when
  topology changes (flagged by `_graph_dirty`).
- `gilbert.rs` — Gilbert realization: lowers a pole-residue transfer
  function to minimal state-space form. Preserves complex-conjugate pole
  pairs as real 2×2 Jordan blocks (modal form) so `A`, `B`, `C` stay real.
  Exposes `gilbert_realization_siso`, `gilbert_realization` (MIMO via
  `MimoResidues = Vec<Vec<Vec<Complex64>>>`), and the `GilbertSS` result
  struct. Called by `blocks/constructors/lti.rs` for TransferFunction-PRC
  lowering.
- `numerical.rs` — one public function `num_jac` that computes `∂f/∂x` by
  central finite differences and returns a `solvers::solver::Jacobian`
  (scalar for 1-D, flat row-major matrix otherwise). Used by implicit
  blocks that have neither an analytic nor a JIT-derived Jacobian. Uses
  `NUM_JAC_REL` (relative step size) and `NUM_JAC_TOL` (absolute floor).
- `rng.rs` — `Rng` using **Xoshiro256+** (Blackman & Vigna 2018) for
  uniform samples and Box-Muller for Gaussians. Self-contained, no
  external crate. Deterministic and reproducible under explicit seeding.
- `logger.rs` — `LogLevel` enum + `Logger` (level-gated `eprintln!`-based
  output) + `ProgressTracker` / `ProgressStats` for the
  `successful_steps` / `runtime_ms` / ... numbers returned by
  `Simulation::run`.
- `diagnostics.rs` — three trackers:
  - `ConvergenceTracker` for the algebraic-loop and implicit-stage solves
    (per-block / per-booster residual history; convergence check is a
    max-vs-threshold against the unitless `NLS_COEF` since residuals come
    in WRMS-scaled — see `solvers::Solver::wrms_norm`),
  - `StepTracker` for adaptive-step error tracking and accept/reject
    scaling,
  - `Diagnostics` top-level aggregator surfaced through the Python API.
- `mod.rs` — re-exports.

## How it fits in

- `FastCell` is used pervasively — every `BlockRef`, `ConnectionRef`,
  `SimEventRef` wraps its contents in `Rc<FastCell<_>>`.
- `graph.rs` is the piece that makes `Simulation::timestep` fast: it
  precomputes `dag_flat_blocks` / `dag_flat_conns` once per topology
  change and the hot path just iterates these flat arrays.
- `numerical.rs` is called from implicit solvers when neither an analytic
  nor a JIT-derived Jacobian is available. Uses `NUM_JAC_REL` / `NUM_JAC_TOL`
  from `constants.rs`.
- `logger.rs` respects verbosity controlled by `Simulation.log`.

## Optimizations

- **`FastCell` has no borrow check at all** (uses `UnsafeCell` directly):
  the engine never aliases a block's interior during a step — DAG order
  makes that invariant structural, not something that needs runtime
  enforcement.
- **DAG precomputation** in `graph.rs`: algebraic-loop Tarjan + topo sort
  runs O(V+E) once on topology change. The hot path costs nothing for
  graph traversal.
- **Register** holds values in a single `Vec<f64>` with name→index maps on
  the side — O(1) indexed access in the hot path, name lookups only at
  construction.
- **RNG** carries 32 bytes of state (`[u64; 4]`) — a single arithmetic
  step per sample, zero allocation, trivial reseeding.
