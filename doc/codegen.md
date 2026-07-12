# fastsim C99 Code Generation — Generated-Code Contract

`Simulation.to_c(name, **opts)` (or `fastsim.ir.Module.to_c`) lowers a model to
self-contained C. This document is the **customer-facing specification** of the
emitted API and its stability guarantees. It describes the corrected, symbol-
prefixed contract (see the "Symbol namespacing" section).

Every generated file carries a banner stamping the generator version
(`fastsim <version>`) and the license notice; see "Versioning & ABI stability"
and "License (Output)" below.

---

## Compiler requirements

- **Language:** C99 (uses `restrict`, `//` comments, `<stdint.h>`, C99 declarations).
- **Standard library:** `<math.h>` + `libm` only (`-lm`). Also `<stddef.h>`,
  `<stdint.h>`, and `<stdio.h>` in *your* `main` — the model itself never does I/O.
- **No other dependencies:** no allocation, no threads, no OS calls, no globals.
  The code is freestanding apart from libm and is safe on embedded targets.

```bash
cc -std=c99 model.c main.c -lm -o model      # Compact layout
cc -std=c99 *.c -lm -o model                 # Library layout (multi-file)
```

`float` numeric models call the `f`-suffixed libm functions (`sinf`, `powf`, …).

---

## File set (by `layout`)

Files are named after the model (`<name>.h`, `<name>.c`, ...; a missing or
empty name falls back to `model`), and every internal `#include` uses these
names -- so two generated models coexist in one build directory, matching the
model-name prefix on all extern symbols and include guards.

| Layout               | Files                                                        |
|----------------------|-------------------------------------------------------------|
| `compact` (default)  | `<name>.h`, `<name>.c`                                       |
| `library`            | `<name>.{h,c}`, `<name>_solver.{h,c}`, and `<name>_blocks.{h,c}` (hierarchical structure) |

Compile every `.c` together. In the Library layout, `<name>_solver.{c,h}` holds
the integrator and, for `structure="hierarchical"`, `<name>_blocks.{c,h}` holds
the per-block region functions.

---

## The emitted API (struct / "rtModel")

`api="struct"` is the only API. One instance struct holds all state, so the code
is **reentrant by construction** — allocate as many instances as you like; they
never share storage.

Let `<name>` be the identifier you pass to `to_c(...)` (default `"model"`), and
`<NAME>` its uppercase form.

### Types & dimensions

```c
#define <NAME>_N_STATE  <n>          /* continuous-state count */

typedef struct {
    T time;                          /* simulation time                    */
    T x[<NAME>_N_STATE];             /* continuous states (if any)         */
    T sig[...];                      /* block output signals (if any)      */
    T p[...];                        /* tunable parameters (if any)        */
    T u[...];                        /* external inputs, open systems      */
    T mem[...];                      /* discrete memory (if any)           */
    /* event counters + carried adaptive step, when applicable */
} <name>_t;
```

`T` is `double` (or `float` for `numeric="float"`).

### Addressable signals

```c
enum {
    <NAME>_SIG_<blockname> = <id>u,  /* one per state, output, and parameter */
    ...
    <NAME>_SIG_COUNT = <count>u
};
```

Ids are assigned in a fixed order: **states first**, then block **outputs**, then
**parameters**. States and parameters are settable; outputs are read-only.
External inputs are addressed through `m->u[]`, not the signal-id space.

### Entry points

```c
void <name>_init(<name>_t *restrict m);
void <name>_run (<name>_t *restrict m, T t_end, T dt);
void <name>_step(<name>_t *restrict m, T dt);                   /* ONE step: the RT entry point */
void <name>_handle_events(<name>_t *restrict m, T dt);          /* if the model has events */

T   <name>_get_signal(const <name>_t *restrict m, uint16_t id); /* 0 if id out of range   */
int <name>_set_signal(<name>_t *restrict m, uint16_t id, T v);  /* 0 ok, -1 not settable  */

/* Exact forward-mode directional derivative, when the model is differentiable: */
void <name>_jvp(<name>_t *restrict m,
                const T *x_seed, const T *u_seed, const T *p_seed,
                T *d_sig, T *d_dxdt);
```

Under the **Library** layout these are additionally external and callable:

```c
void <name>_outputs(<name>_t *restrict m);                      /* recompute m->sig       */
void <name>_deriv  (<name>_t *restrict m, T *restrict dxdt);    /* state derivative       */
void <name>_blk_<i>_alg  (<name>_t *restrict m);                /* per-block, hierarchical */
void <name>_blk_<i>_deriv(<name>_t *restrict m, T *restrict dxdt);
```

### Semantics

- **`<name>_init`** seeds `time`, states, parameters and memory to their initial
  values and refreshes `m->sig`. Call once before stepping.
- **`<name>_run(m, t_end, dt)`** integrates from the current `m->time` to `t_end`.
  For a **fixed-step** solver, `dt` is the step; for an **adaptive** solver, `dt`
  seeds the first step and the accepted step size is carried in the struct
  (`m->fs_h`) across calls, so chunked runs keep their step history. Events fire
  at step boundaries via `<name>_handle_events`; `m->sig` is refreshed on return.
  It is safe to call repeatedly to advance in chunks.
- **`<name>_step(m, dt)`** advances by exactly ONE step: events due now (first
  call only -- `run` and `step` share the guard), one RK step over `dt`, event
  handling at the new time, output refresh. Fixed work per call -- bounded
  stage count, no loops over time, no allocation -- so it is the entry point
  for a periodic real-time task or timer ISR at rate `1/dt`. N calls compose
  exactly (bit-for-bit) to `run(t0 + N*dt, dt)`. With an adaptive tableau the
  embedded error estimate is ignored (`step` has no step-size control; use
  `run` for adaptive integration).
- **`get_signal` / `set_signal`** read/write by id (see above). Setting a state or
  parameter and then calling `run` is how you drive an embedded/HIL model.
- **`<name>_jvp`** computes `∂{sig, dxdt}/∂{x, u, p} · seed` exactly (forward-mode
  AD, not finite differences). Emitted only when every op is differentiable.

### Usage sketch

```c
<name>_t m;
<name>_init(&m);
<name>_set_signal(&m, <NAME>_SIG_x0, 2.0);   /* optional: override an initial state */
<name>_run(&m, 10.0, 1e-3);
double y = <name>_get_signal(&m, <NAME>_SIG_out);
```

---

## Symbol namespacing (HIL: linking two models)

Every extern symbol and every include guard is derived from `<name>`
(`<name>_init`, `<name>_t`, `FASTSIM_<NAME>_MODEL_H`, …), and the shared runtime
helpers (`FASTSIM_EQ_TOL`, the RNG/digamma helpers) sit behind one
`FASTSIM_RT_HELPERS` guard. Two models generated with **distinct names** therefore
compile and link into a single binary — the standard plant + controller HIL
setup — with no duplicate-symbol or colliding-guard errors. Give each model a
unique `name`.

---

## Options (`to_c(name, **opts)`)

| Option       | Values                                                                 |
|--------------|------------------------------------------------------------------------|
| `numeric`    | `"double"` (default), `"float"`, `"fixed"` (= q16.16), `"qM.N"` (M+N=32) |
| `reductions` | `"unrolled"` (default), `"vectorized"` (Reduce/Dot as a counted loop)  |
| `structure`  | `"hierarchical"` (default; one function per block), `"flat"` (fused)   |
| `layout`     | `"compact"` (default), `"library"`                                     |
| `solver`     | fixed: `"rk4"`, `"euler"`, `"ssprk22/33/34"`; adaptive (embedded error): `"rkdp54"`, `"rkck54"`, `"rkf45"`, `"rkf78"`, `"rkv65"`, `"rkbs32"`, `"rkf21"`, `"rkdp87"` |
| `api`        | `"struct"` (only)                                                      |
| `scaffold`   | `False` (default), `True`: additionally emit `CMakeLists.txt` + `<name>_main.c` (see below) |
| `trace`      | `False` (default), `True`: additionally emit `<name>_trace.json` (see below) |
| `a2l`        | `False` (default), `True`: additionally emit `<name>.a2l` (see below) |

Implicit (DIRK/ESDIRK) tableaus are not yet emitted. See `help(Simulation.to_c)`.

---

## Build scaffolding (`scaffold=True`)

```python
files = sim.to_c("decay", scaffold=True)   # + CMakeLists.txt, decay_main.c
```

Two extra files turn the sources into a running binary with zero hand-written
glue -- and unlike the model sources they are EDITABLE starting points (their
banners say so; a rerun with `scaffold=True` overwrites them):

- **`CMakeLists.txt`** -- the model as a static library (link this into your
  firmware/app; plain CMake, cross-compile via your usual toolchain file) plus
  a `<name>_demo` executable.
- **`<name>_main.c`** -- a demo driver stepping the model via `<name>_step`
  and printing a CSV trajectory (time + states + block outputs) to stdout,
  with marked HAL hook points (`read_inputs` / `write_outputs`) where real
  sensor/actuator I/O goes. `FASTSIM_DURATION` / `FASTSIM_DT` are overridable
  at compile time.

```bash
cmake -B build && cmake --build build
./build/decay_demo > trajectory.csv
```

---

## Trace map & static metrics (`trace=True`)

```python
files = sim.to_c("decay", trace=True)      # + decay_trace.json
report = json.loads(files["decay_trace.json"])
```

`<name>_trace.json` is the machine-readable model-to-code map, derived from
the same plan the emitter used (map and code agree by construction):

- **`blocks[]`** -- per block: the emitted functions (with `file`/`line`
  pointing at the definition in the generated sources), its states / outputs /
  parameters with their `SIG_*` signal ids (the substrate for calibration
  maps), its events (guard/effect functions) and IR op counts.
- **`signals[]`** -- the full addressable-variable inventory (name, id, kind,
  settable, initial value), exactly the `get_signal`/`set_signal` id space.
- **`entry_points[]`** -- `init`/`step`/`run`/... resolved to file/line.
- **`metrics`** -- static estimates: `model_struct_bytes_packed` (RAM lower
  bound, padding-free), `integrator_stack_bytes` (the RK kernel's locals --
  the dominant stack user), `tableau_const_bytes` (ROM constants), IR op
  counts and `per_step_ops_estimate` (stages x (alg + deriv) -- a proxy for
  FLOPs/code size, not a cycle count). Suitable for CI size gates.

---

## Fixed point (`numeric="fixed"` / `"qM.N"`)

```python
files = sim.to_c("decay", numeric="q16.16")   # int32 Q16.16, int64 intermediates
```

For targets without an FPU the whole model lowers to integer arithmetic in one
global Q format on `int32_t` (`"fixed"` = q16.16; any `qM.N` with `M + N == 32`
and 1 <= N <= 30 — the sign bit counts toward `M`). Everything in `<name>_t`
(signals, parameters, `time`, `dt`) is a raw Q value; the header emits
`<NAME>_Q_FRAC` / `_Q_ONE` / `_Q_FROM_DOUBLE` / `_Q_TO_DOUBLE` for the
host/tooling boundary, and `a2l=True` describes the signals as `SLONG` with a
LINEAR `2^-N` conversion, so calibration tools display physical values.

Semantics and scope:

- Sums/products widen to `int64_t` and truncate back on store — a DEFINED
  wrap, never C signed-overflow UB. Multiplication/division carry the `2^N`
  scale explicitly (`(a*b) >> N`, `(a << N) / b`).
- Arithmetic, comparisons, `min`/`max`, `abs`, `floor`/`ceil`/`round`/`trunc`,
  `select`, reductions, dot products and **LUT1D** (interpolation in Q) lower
  fully. Transcendentals (`sin`, `exp`, ...), `pow`, `atan2`, `hypot` have no
  integer lowering and are rejected with a pointer at LUT1D — the classic
  embedded pattern.
- Fixed-step tableaus only (`rk4`, `euler`, `ssprk*`); the adaptive error
  controller needs `pow`. The scaffold is rejected (its demo driver prints a
  floating CSV) — drive `<name>_step` from your own loop instead.
- Pick `dt` representable in the format (for q16.16 e.g. `1/1024`); `time`
  advances in Q, so the format bounds the horizon (q16.16: ±32768 time units).
- Accuracy budget: resolution `2^-N` per op, accumulating over stages and
  steps. The test suite pins a Q16.16 RK4 decay to `e^-1` within 5e-3 over
  1024 steps; validate your own model against a `double` build of the same C.

---

## Calibration map (`a2l=True`)

```python
files = sim.to_c("decay", a2l=True)        # + decay.a2l
```

`<name>.a2l` is an ASAP2 (ASAM MCD-2 MC) description of the generated model
for XCP calibration/measurement tooling (CANape, INCA, XCPsim): a MEASUREMENT
entry for time, every state, block output, external input and discrete-memory
element, and a CHARACTERISTIC entry for every tunable parameter. Names match
the `SIG_*` enum inventory, so A2L and generated C agree by construction.

Addressing is `SYMBOL_LINK "<name>" <offset>`: the calibration tool resolves
one global model-instance symbol from your ELF and adds the byte offset within
`<name>_t`. Define the instance as a non-static global named after the model
(`decay_t decay;`), or remap the symbol name in the tool. Offsets follow the
natural-alignment layout shared by the mainstream ABIs (x86-64 SysV, MSVC x64,
AAPCS64) with 64-bit `size_t`; the same offsets are exposed in the trace map's
`struct_layout` section (`trace=True`) for custom tooling, and are pinned
against the emitted header by the test suite.

---

## Real-time stepping (`<name>_step`)

The generated model needs no host loop: wire `<name>_step` into a periodic
task or timer ISR and exchange I/O around it.

```c
/* 1 kHz control task (RTOS tick or hardware timer ISR) */
void control_tick(void) {
    plant_set_signal(&m, PLANT_SIG_u, read_adc());   /* inputs  */
    plant_step(&m, 1e-3);                            /* advance one period */
    write_dac(plant_get_signal(&m, PLANT_SIG_y));    /* outputs */
}
```

Per call the cost is one RK step (fixed stage count), the event pass and the
output refresh -- statically bounded, allocation-free, libm only. Discrete
events (periodic samplers, zero crossings, conditions) fire inside `step` at
step-boundary resolution, exactly as they do inside `run`.

---

## SIL verification (`verify_c`)

```python
report = sim.verify_c("decay", duration=2.0, dt=1e-3)
assert report["passed"], report
```

`Simulation.verify_c(name, **opts)` closes the loop on this contract per model,
per machine: it compiles the emitted C with a local C99 compiler
(`$FASTSIM_CC`, `$CC`, then `cc`/`clang`/`gcc` -- `fastsim._fastsim.find_c_compiler()`
shows which), integrates the binary and the reference engine (the statically
compiled tape) over the same fixed-step trajectory, and compares the state
trajectories sample by sample. Both sides step identically, so sample times
align exactly; the report carries the worst scaled error
`|c - ref| / (atol + rtol*|ref|)` (`passed` = <= 1), the offending state and
time, plus step/compiler metadata. `keep_build=True` keeps the temp build
directory for inspection.

Scope: fixed-step explicit solvers and models inside the static-compile subset
(adaptive tableaus adapt their step sequence independently per backend and are
rejected up front). For `numeric="float"` widen `rtol` -- a float32 target
legitimately deviates from the f64 reference.

---

## Versioning & ABI stability

- **Generator version.** Every file's banner carries `fastsim <version>` (the
  package version). This identifies the generator that produced the artifact.
- **IR version.** The intermediate representation the backend consumes is **IR v1**
  and is golden-pinned (see `src/ir/README.md`); a change forces a deliberate bump.
- **ABI surface.** The generated-code ABI is: the `<name>_t` field layout, the
  `<NAME>_SIG_*` names and id assignment, the entry-point names and signatures, and
  the `<NAME>_N_STATE` macro. Within a patch/minor release these are stable for a
  fixed model and option set. A change to any of them is reflected by the generator
  version in the banner and noted in the changelog — pin the generator version if
  you compile the emitted C into a long-lived binary.

---

## License (Output)

The generated C is **"Output"** under fastsim's
[PolyForm Noncommercial License 1.0.0](../LICENSE): free for noncommercial use,
but using or distributing it in a commercial product requires a commercial
license. Each generated file is stamped with this notice so the term travels with
the code. Commercial licensing: **info@pathsim.org** (see
[COMMERCIAL.md](../COMMERCIAL.md)).
