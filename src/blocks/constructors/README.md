# `src/blocks/constructors/` — Concrete block kinds

Factory functions that build a configured `Block` for each block type fastsim
supports. Each file groups blocks of a related domain; `core.rs` holds
fundamental primitives (`source`, `ode`, `function`, `dynamical_system`,
`statespace`, ...) that the JIT traces into Rust-native RHS closures.

## Theory

A block factory is a pure constructor: given user-specified parameters
(time constants, matrices, gains, ...), it:

1. Builds the RHS closures `f_dyn` / `f_alg` and their Jacobians.
2. Installs a `Solver` engine with the correct initial state and dimensions.
3. Wires the right `solve_fn` / `step_fn` for the solver family (via
   `solvers/factories.rs`).
4. Returns the assembled `BlockRef`.

Most factories have a Python-callback form (slow) and a JIT-traced form
(fast). The tracer in `jit/tracer.rs` attempts to trace the user function
first; if any operation is unsupported, `_trace_or_none` falls back to the
Python callback.

## Implementation (files)

- `core.rs` — primitive factories: `source`, `ode`, `function`,
  `dynamical_system`, `statespace`, ..., plus shared helpers
  `flat_to_mat`, `matvec_into`, `vec_add_into`
- `ctrl.rs` — control blocks: PID, AntiWindupPID, LeadLag, RateLimiter,
  Deadband, Backlash, ...
- `dae.rs` — DAE blocks: MassMatrix, SemiExplicit, FullyImplicit (inner
  Newton for algebraic constraints — see `DAE_*` in `constants.rs`)
- `discrete.rs` — Delay, SampleHold, FIR, ADC, DAC, Spectrum, Wrapper
- `dynsys.rs` — DynamicalSystem / DynamicalFunction factories
- `filters.rs` — analog Butterworth lowpass/highpass/bandpass/bandstop,
  Allpass. Poles computed analytically, lowered to state-space via
  `transfer_function_num_den`.
- `lti.rs` — StateSpace + narrower PT1/PT2 + TransferFunction variants.
  Lowers pole-residue transfer functions into state-space via
  `utils::gilbert::gilbert_realization(_siso)`.
- `math_logic.rs` — arithmetic (Adder, Multiplier, Divider, ...),
  trig/exp (Sin, Cos, ...), comparison (Equal, GreaterThan), logic
  (LogicAnd, LogicOr, LogicNot)
- `noise.rs` — WhiteNoise, PinkNoise (1/f), RandomNumberGenerator
- `nonlinear.rs` — Abs, Clip, Comparator, Relay, Switch, Rescale, ...
- `scope.rs` — Scope, Spectrum (recording)
- `sources.rs` — Step, Sinusoidal, Triangle, Square, Pulse, Clock,
  Chirp, GaussianPulse
- `table.rs` — LUT1D (1D table interpolation)
- `mod.rs` — re-exports, shared types (`DaeJacFn`, `out_port_map`, ...)

## How it fits in

- Every factory returns a `BlockRef`, consumed by `Simulation::new`.
- For JIT-eligible blocks, `constructors` pass the user RHS through
  `jit::tracer::_trace_*` helpers, which build an SSA graph. If tracing
  succeeds, the block's `f_dyn` is a Rust closure over the compiled graph.
- LTI factories (`lti.rs`, `filters.rs`) share `FILTER_POLY_REAL_TOL` from
  `constants.rs` for polynomial-projection tolerance.
- DAE factories use `DAE_*` constants for inner-Newton convergence.

## Optimizations

- JIT compilation lifts 2–100× on typical block RHS (see JIT README).
- Gilbert realization reuses a single shared `utils::gilbert` path for
  both SISO and MIMO transfer functions.
- Butterworth filters build poles analytically rather than calling scipy
  — no external dependency at runtime, coefficients materialized once.
- Factory-level constants moved to `constants.rs` so filter/source/DAE
  tolerances are tunable from one file.
