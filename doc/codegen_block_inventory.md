# Codegen block inventory (issue #48)

Verified classification of every public block against the C code generator. The
issue asks for two things:

1. **All eventful blocks _without an internal solver_ must be handled by codegen.**
2. **All codegen settings must produce SiL-identical results** to the Rust path.

This document covers point 1 (the acceptance gap). Point 2 (numeric parity across
the setting permutations) is the job of the fuzzing harness and is tracked
separately — a block appearing as `OK` below means codegen *emits* C for it, not
yet that the emitted C is bit-for-bit faithful.

Reproduce with:

```
maturin develop            # build the extension with the codegen feature
python scripts/codegen_block_inventory.py
```

The tool constructs a representative instance of each block, assembles a minimal
closed sim, snapshots the IR (`sim.to_ir`) to count events / opaque events /
extern blocks / continuous state, and then tries `sim.to_c()`.

## Summary

| Category | Count | Blocks |
|---|---|---|
| Codegen-accepted | 94 | all the leaf math/source/discrete/filter/LTI blocks, incl. every op-expressible eventful block |
| **Eventful, opaque → rejected (the #48 gap)** | **2** | **`Wrapper`, `PinkNoise`** |
| Internal-solver / extern → rejected (out of scope) | 5 | `AlgebraicConstraint`, `BVP1D`, `FullyImplicitDAE`, `MassMatrixDAE`, `SemiExplicitDAE` |
| Needs external resource (out of scope) | 2 | `CoSimulationFMU`, `ModelExchangeFMU` (need an FMU file; opaque co-sim) |

## The gap: eventful blocks codegen cannot yet lower

Both have an op-expressible representation available — they are lowerable, they
just still emit their sampling event as an opaque host closure.

### `Wrapper`
`_trace_wrapper` (`src/pybindings/py/jit.rs`) already JIT-traces `func(u)` into a
`LazyTraced` op-graph. But the scheduled re-evaluation is pushed as a runtime
`Schedule` event whose action is a host closure (`t_evt.call_into`), so the IR
event is `opaque = true` and `collect_events` rejects it
(`src/codegen/system.rs:1319`).

**Fix direction:** express the sampling as an op-expressible discrete event — a
memory (ZOH) slot written by the traced graph in a `Schedule` effect region,
exactly the pattern `SampleHold` / `ZeroOrderHold` already use (both `OK`). The
traced op-graph is in hand, so this is a lowering change, not new math.

### `PinkNoise`
Voss-McCartney: `num_octaves` independent random values plus a sample counter;
each sample updates one octave selected by the counter's bit pattern
(`src/blocks/constructors/noise.rs`). The update runs in an opaque event.

**Fix direction:** carry the octaves + counter as memory slots and express the
per-sample octave update as an op-expressible periodic event, reusing the
`fastsim_rand_*` header helpers (the same helpers that already let `WhiteNoise`
and `RandomNumberGenerator` lower). Heavier than `Wrapper` because of the
counter-bit octave selection.

## Note: the noise blocks that _are_ already handled

`WhiteNoise` (sampling-period mode), `RandomNumberGenerator`,
`SinusoidalPhaseNoiseSource` and `Chirp*` lower with **zero events**: their sample
is a deterministic pure function of `t` (`out = scale · normal(key(t))`,
`noise.rs`), so there is no event to drop and they are SiL-reproducible by
construction. The `ev=0 / codegen=OK` rows for them are correct, not a silently
dropped event.

## Out of scope for #48

`AlgebraicConstraint`, `BVP1D`, `FullyImplicitDAE`, `MassMatrixDAE`,
`SemiExplicitDAE` carry an **internal solver** (Newton / collocation) and are
represented as opaque extern blocks (`extern=1`, `ev=0`); the issue explicitly
excludes internal-solver blocks. The two FMU blocks wrap external co-simulation
binaries and are opaque by nature.

## Full table

```
block                        ev opq ext  st  codegen
----------------------------------------------------
ADC                           1   0   0   1  OK
Abs                           0   0   0   1  OK
Adder                         0   0   0   1  OK
AlgebraicConstraint           0   0   1   1  REJECT (internal solver)
Alias                         0   0   0   1  OK
AllpassFilter                 0   0   0   2  OK
Amplifier                     0   0   0   1  OK
AntiWindupPID                 0   0   0   3  OK
Atan / Atan2                  0   0   0   1  OK
Backlash                      0   0   0   2  OK
Butterworth{LP,HP,BP,BS}      0   0   0  3-5 OK
Chirp[PhaseNoise]Source       0   0   0   2  OK
Clip                          0   0   0   1  OK
Clock / ClockSource           0   0   0   1  OK
CoSimulationFMU               -   -   -   -  SKIP (needs FMU file)
Comparator                    0   0   0   1  OK
Constant                      0   0   0   1  OK
Cos / Cosh                    0   0   0   1  OK
Counter[Up,Down]              1   0   0   1  OK
DAC                           1   0   0   1  OK
Deadband                      0   0   0   1  OK
Delay                         1   0   0   1  OK
Differentiator                0   0   0   2  OK
DiscreteDerivative            1   0   0   1  OK
DiscreteIntegrator            1   0   0   1  OK
DiscreteStateSpace            1   0   0   1  OK
DiscreteTransferFunction      1   0   0   1  OK
Divider                       0   0   0   1  OK
DynamicalFunction             0   0   0   1  OK
DynamicalSystem               0   0   0   2  OK
Equal                         0   0   0   1  OK
Exp                           0   0   0   1  OK
FIR                           1   0   0   1  OK
FirstOrderHold                1   0   0   1  OK
FullyImplicitDAE              0   0   1   1  REJECT (internal solver)
Function                      0   0   0   1  OK
GaussianPulseSource           0   0   0   1  OK
GreaterThan / LessThan        0   0   0   1  OK
Integrator                    0   0   0   2  OK
LUT1D                         0   0   0   1  OK
LeadLag                       0   0   0   2  OK
Log / Log10                   0   0   0   1  OK
Logic{And,Or,Not}             0   0   0   1  OK
MassMatrixDAE                 0   0   1   1  REJECT (internal solver)
Matrix                        0   0   0   1  OK
Mod                           0   0   0   1  OK
ModelExchangeFMU              -   -   -   -  SKIP (needs FMU file)
Multiplier / Norm             0   0   0   1  OK
ODE                           0   0   0   2  OK
PID                           0   0   0   3  OK
PT1 / PT2                     0   0   0  2-3 OK
PinkNoise                     1   1   1   1  REJECT (opaque event)  <-- #48 gap
Polynomial                    0   0   0   1  OK
Pow / PowProd                 0   0   0   1  OK
Pulse / PulseSource           4   0   0   1  OK
RandomNumberGenerator         0   0   0   1  OK
RateLimiter                   0   0   0   2  OK
Relay                         2   0   0   1  OK
Rescale                       0   0   0   1  OK
SampleHold                    1   0   0   1  OK
Scope                         0   0   1   0  OK (sink, dropped)
SemiExplicitDAE               0   0   1   1  REJECT (internal solver)
Sin / Sinh                    0   0   0   1  OK
SinusoidalPhaseNoiseSource    0   0   0   2  OK
SinusoidalSource              0   0   0   1  OK
Source                        0   0   0   1  OK
Spectrum                      0   0   1   0  OK (sink, dropped)
Sqrt                          0   0   0   1  OK
SquareWaveSource              0   0   0   1  OK
StateSpace                    0   0   0   3  OK
Step / StepSource             1   0   0   1  OK
Switch                        0   0   0   1  OK
Tan / Tanh                    0   0   0   1  OK
TappedDelay                   1   0   0   1  OK
TransferFunction[NumDen,PRC,ZPG] 0 0 0  2  OK
TriangleWaveSource            0   0   0   1  OK
WhiteNoise                    0   0   0   1  OK (deterministic-of-t)
Wrapper                       1   1   1   1  REJECT (opaque event)  <-- #48 gap
ZeroOrderHold                 1   0   0   1  OK
```
