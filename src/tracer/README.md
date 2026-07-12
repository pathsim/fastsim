# `src/tracer/` (+ `src/ssa/`) — User-function tracing, SSA graph, optimization, AD

Turns a user-provided Python RHS (lambda / def) into a flat Rust-native
tape that evaluates in the engine hot path without Python overhead.
Delivers the ~45–175× speedups we see versus pathsim on the comprehensive
benchmark.

## Theory

Classical **operator-overloading trace**: the user function is called once
with symbolic `JitTracer` / `JitTracerArray` values in place of floats.
Every arithmetic operation, numpy ufunc call and numpy array-function call
on the symbolic values records a node into an SSA graph. After tracing:

1. A sequence of **optimization passes** (constant folding, strength
   reduction, CSE, algebraic identities, FMA fusion, DCE) simplifies the
   graph.
2. The graph is lowered to a **flat tape** of `TapeOp { opcode, arg0,
   arg1, arg2 }` — a single byte per op + three `u32` indices.
3. **Symbolic differentiation** walks the graph to produce derivative
   nodes, which are optimised and compiled the same way.

The **graph IR is scalar-SSA and always flat row-major**. N-D tensor
shape lives exclusively on the Python-side `JitTracerArray` wrapper as a
metadata `shape: Vec<usize>` — `product(shape) == size`. View ops
(reshape, squeeze, unsqueeze) are zero-cost because they only rearrange
the flat `Vec<NodeId>` owned by the wrapper; the underlying graph is
untouched. Broadcasting, slicing, and axis-aware reductions expand
during tracing into scalar-SSA nodes. The tape never knows about shape.

## Branching

Data-dependent selection **is** traceable and lowers to SSA. A comparison
on a tracer (`x > 0`, `np.greater`, …) records a `Cmp` node (result
`1.0`/`0.0`); `np.where(cond, a, b)` and the `fastsim.where` helper record
a `Select(cond, a, b)` node (`graph.rs` `Node::Select`, built by
`SharedGraph::select` in `tracer/mod.rs`); `np.clip` / `fastsim.clip`
lower to `Max`/`Min`. Both the scalar `JitTracer` and the N-D
`JitTracerArray` handle `where`/`clip` (via `__array_function__`), so the
same numpy idioms trace either way.

What is **not** traceable is a bare Python `if`/`else` on a tracer value:
choosing a branch needs the concrete runtime value, which does not exist
during a symbolic trace. `JitTracer.__bool__` therefore raises a
`TypeError` ("…use `np.where(cond, then_val, else_val)`") rather than
silently picking one side. That error is structural: the `_trace_*`
probes treat it as a fatal trace failure (only `ValueError` / `IndexError`,
i.e. shape mismatches, are swallowed for a later retrace) so the block
falls back to the opaque Python callback. The rule is: rewrite
control-flow as `np.where` and it compiles; leave it as `if` and the block
stays interpreted. Boolean MASK indexing (`x[x > 0]`) is equally
structural: the output shape depends on runtime values, so it is rejected
with a pointer to `np.where`. Constant integer fancy indexing
(`x[[0, 2]]`), negative slice steps (`x[::-1]`), `Ellipsis` and
`None`/newaxis all trace (pure node permutations).

### Randomness (`fastsim.random_uniform` / `random_normal`)

`np.random.*` is untraceable (hidden global state) and irreproducible, so it
forces the opaque fallback. fastsim instead offers **stateless, counter-based**
draws keyed by an explicit value, in the style of JAX's PRNG: `random_uniform(k)`
is a *pure function* of `k` and lowers to a single `UnaryOp::RandUniform` node (a
splitmix64 finalizer over the key's bits → `[0, 1)`). `random_normal(k)` composes
Box-Muller from two `RandUniform` draws plus ordinary math ops, so it traces,
optimizes, and (eventually) codegens through the same path. The derivative is 0
(autodiff treats the hash like `floor`/`sign`).

Because the kernel is identical in the interpreter, the tape, the generated C
(`codegen::RNG_HELPER_C`, a `fastsim_rand_uniform` emitted in the model header)
and the pure-Python twin (`fastsim/random.py`), every backend agrees: the traced
and eager paths match bit-for-bit, and a compiled or code-generated noise source
replays identically. The canonical idiom is a time-derived stepwise key,
`random_normal(t // dt)` (`//` traces via `floor(a/b)`). Keys are scalar for now;
build vector noise by stacking scalar draws with offset keys.

This is what makes the stochastic source blocks first-class. In **discrete**
(zero-order-hold) mode — `sampling_period` set — `WhiteNoise`,
`RandomNumberGenerator`, `SinusoidalPhaseNoiseSource` and `ChirpPhaseNoiseSource`
lower their sample-and-hold noise to `RandUniform` keyed by `floor(t/sp)`, so
they `compile()` into the fused tape and code-generate with the real noise (not
a zero-noise nominal). **Continuous** mode draws fresh every solver step (not a
pure function of t) and **PinkNoise** (recursive Voss-McCartney) stay on the
stateful-RNG runtime and remain opaque to static compile.

### Modulo semantics (`%` / `np.remainder` / `np.fmod`)

Python and numpy `%` are FLOORED modulo (the result's sign follows the
divisor); the raw `Mod` op is C `fmod` (sign follows the dividend). The
tracer therefore lowers `%` and `np.remainder`/`np.mod` as a composite over
existing ops — numpy's own fixup `m = fmod(a, b); if m != 0 and sign(m) !=
sign(b): m += b` — so traced and eager paths agree on negative operands.
The `m != 0` guard is exact (`Gt(|m|, 0)`, NOT the tolerance-banded `Ne`):
near-exact multiples leave `m` within a few ULP of zero, where a banded
test would wrongly suppress the fixup (a finite jump of `b`). `np.fmod`
keeps the raw `Mod` lowering.

### Equality semantics (`==` / `!=`)

`CmpOp::Eq`/`Ne` are **not** bit-exact IEEE equality. They test whether
`|a - b| < JIT_FLOAT_EQ_TOL` (`constants.rs`, `1e-15`). This is a deliberate
choice: exact float equality in a numeric RHS is almost always a footgun, and
the band makes a traced `np.where(x == c, …)` robust to last-bit rounding.

The semantics are applied **consistently** across every fastsim evaluator:
`Graph::interpret`, the `Tape` hot loop, the IR reference evaluator, and the
generated C (`codegen` emits the same `eq_tol` literal). So a model behaves
identically whether it runs interpreted, compiled, or code-generated.

The one place this diverges is a block that **falls back to the opaque Python
callback** (trace failure): there `==` is plain numpy, i.e. exact. In practice
this only bites if the same `==` expression is reachable both traced and
untraced; transcendental-light equality tests on floats are the rare case
where the two paths could disagree at the 1e-15 level. The differential fuzzer
(`graph.rs` `fuzz_tape_matches_interpret_bit_exact`) pins interpret↔tape
parity for the band; it does not (and cannot) pin the opaque-fallback path.

## Implementation

- `tracer/frontend/` — the operator-overloaded types and Python-visible
  entry points. `frontend/mod.rs` holds the scalar `JitTracer`, the
  `where`/`clip` helpers, the composite-ufunc emitters (`unary_composite` /
  `binary_composite` / `floored_mod` / `emit_interp`), the constant-factory
  monkeypatches (`np.zeros/ones/empty/full` plus
  `np.arange/linspace/eye/diag`, so factory results work as assignment
  targets), and the trace driver; the N-D `JitTracerArray` and its numpy-
  protocol surface live in `frontend/array.rs`:
  - `JitTracer` (scalar) and `JitTracerArray` (N-D) with full Python-
    operator slots (`__add__`, `__gt__`, `__getitem__`, …) and numpy
    `__array_ufunc__` / `__array_function__` protocol handlers.
  - `JitTracerArray` surface:
    - Metadata: `.shape` / `.ndim` / `.size` / `.T`
    - View ops (zero-cost): `reshape`, `transpose`, `squeeze`,
      `unsqueeze` — methods *and* via NEP-18 (`np.reshape`,
      `np.transpose`, `np.squeeze`, `np.expand_dims`)
    - N-D slicing: `__getitem__` accepts `int`, `slice`, and
      `tuple of (int | slice)` with numpy semantics (int-indices drop
      the axis, slices keep it, missing axes default to `:`)
    - N-D broadcasting on all elementwise binary / unary / cmp ops
      (right-aligned, size-1 axes broadcast)
    - Axis-aware reductions: `np.sum / mean / min / max / prod` accept
      `axis=int` (positive or negative) and `keepdims=bool`
    - Array assembly: `np.concatenate / stack` honour `axis=N`; legacy
      1-D flat behavior is preserved when all inputs are scalar/1-D
  - `trace_with_signature` — drives one trace pass (build the scratch
    graph, invoke the user function, collect outputs, optimise).
  - `_trace_ode`, `_trace_function_block`, `_trace_source` — block-
    specific Python-facing factories that wrap `trace_with_signature`
    with the right `TraceArg` signature and output handling.
  Further `_trace_*` for `dynamical_system`, `dynamical_function`,
  `wrapper`, `mass_matrix_dae`, `semi_explicit_dae`, `fully_implicit_dae`
  live alongside their PyO3 bindings in `src/pybindings/py/*.rs` — they
  all funnel through `tracer::trace_with_signature`.
- `ssa/graph.rs` — the SSA `Graph`, `Node` enum (`Const`, `Input`, `Param`,
  `Unary`, `Binary`, `Cmp`, `Select`, `Fma`, `Reduce`, `Dot`), hash-consing
  dedupe, and the recursive reference `interpret`. The op vocabulary and
  its canonical f64 semantics live in `ssa/op.rs` (the op manifest, also
  the IR's op kinds); `ufunc_table.rs` (this module) maps numpy ufunc names
  onto it.
- `ssa/optimize.rs` — two tiers. `optimize()` (trace-time, seen by every
  consumer including codegen): constant fold, strength reduce and algebraic
  identities to a fixed point, then FMA detection. `lower_for_tape()`
  (tape-lowering only, applied inside `InterpretedFn::from_graph` — codegen
  consumes the IR from the un-transformed graphs): `canonicalize`, a
  VALUE-EXACT value-numbering rebuild (merges the duplicates in-place
  rewrites and AD leave behind, sorts commutative operands, folds
  Select/Cmp/Fma/Reduce/Dot constants, subsumes DCE), then
  `reassociate_chains`, which bundles Add/Fma accumulation chains into
  fused `Reduce`/`Dot` kernels — reassociating (ULP-class, like FMA
  fusion), and gated away from any value that reaches a DISCRETIZING
  consumer (comparison operand, `Select` condition, floor/ceil/round/
  trunc/sign/`RandUniform`, `Mod`), where a ULP shift would be a finite
  jump instead of a rounding change.
- `ssa/autodiff.rs` — symbolic differentiation with memoised subexpression
  deduplication. Emits new nodes into the same graph so the AD result
  benefits from the same optimisations.
- `ssa/tape.rs` — the flat tape (`InterpretedFn`), liveness-driven work-slot
  reuse, and the differential fuzzers pinning interpret == tape bit-exact
  plus raw ≈ lowered value preservation.

## How it fits in

- Every `constructors/` factory that has a callable RHS routes through
  `_trace_or_none` → `jit::tracer::_trace_*`. Trace success produces a
  Rust closure that evaluates the flat tape; trace failure falls back to
  Python callback (visible with `FASTSIM_JIT_DEBUG=1`).
- The `LazyTraced` wrapper (`src/pybindings/py/lazy.rs`) sits between
  block construction and the hot path: shapes that are only known after
  Connection resolution trigger a transparent retrace under a shape-
  keyed cache. The fast path is two `usize` comparisons plus a
  `Tape::eval`.
- `autodiff` is invoked by implicit factories to obtain an analytical
  Jacobian — avoids the numerical-Jacobian fallback in `utils/numerical.rs`.
- `constants.rs` holds `JIT_FLOAT_EQ_TOL` (for `CmpOp::Eq`/`Ne`) and
  `JIT_FINITE_DIFF_REL` (for numeric-derivative fallbacks).

## Optimizations

- **Operator-overload tracing** handles every Python callable (lambdas,
  closures over instance attributes, decorated, interactive) — no
  `inspect.getsource` needed.
- **Hash-consing** in `Graph::add`: identical nodes share a `NodeId`, so
  CSE is automatic at insertion time.
- **Flat tape representation** keeps the evaluator in L1: `Tape::eval`
  is a tight `for` loop over `TapeOp[]` with a `match` on a single `u8`
  opcode (the `Graph::interpret` recursive evaluator is a reference path
  for tests). N-D shape metadata does not survive lowering — it lives on
  the Tracer wrapper at trace time only.
- **Zero-cost N-D views**: `reshape` / `squeeze` / `unsqueeze` mutate
  shape metadata only; `transpose` permutes `Vec<NodeId>` entries
  without emitting new graph nodes. No runtime cost after the trace.
- **Stack-only Cholesky / LU** and **reused scratch** mean a traced ODE
  block has **zero allocations** in its hot path.
- **Symmetric coverage** between `JitTracer` and `JitTracerArray`: the
  unary/binary/comparison ufunc tables (`ufunc_table.rs`), the composite
  ufuncs (radians/degrees, exp2, copysign, logaddexp, heaviside,
  float_power, scipy's expit, …), numpy-compatible N-D broadcasting, the
  array METHOD forms (`x.sum()/mean()/min()/max()/prod()/dot()/flatten()/
  ravel()/clip()/copy()`, lowered through the same dispatch as the `np.*`
  function forms), `np.interp` over constant grids (select chain),
  structural ops (full_like, flip, roll, cumprod, outer, atleast_1d), and
  mixed scalar-tracer/array operands in either position — a user writing
  pure numpy code on the RHS gets JIT coverage without rewriting.
  Data-dependent branching is covered too: `np.where` and comparisons
  lower to a `Select` (ternary) SSA node, and `np.clip` to `Max`/`Min` —
  only a bare Python `if` on a tracer is unsupportable (see *Branching*).
  The measuring stick is `tests/python/test_tracer_corpus.py` (every idiom
  classified TRACED / GAP / STRUCTURAL, gap closures flip entries
  explicitly) plus the differential fuzzer `tests/python/test_tracer_fuzz.py`
  (random expressions, traced vs eager numpy).
