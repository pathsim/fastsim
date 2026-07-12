<p align="center">
  <img src="doc/logo.png" width="350" alt="FastSim Logo" />
</p>

<p align="center">
  <strong>A Rust block-diagram simulation engine</strong>
</p>

<p align="center">
  Drop-in replacement for <a href="https://github.com/pathsim/pathsim">PathSim</a>
</p>

---

FastSim is a Rust reimplementation of [PathSim](https://github.com/pathsim/pathsim) with an identical Python API via PyO3. Python callbacks are automatically traced into an optimized SSA graph, differentiated symbolically, and evaluated in Rust.

## Features

- **Drop-in compatible**: same API as PathSim; swap the import
- **Rust engine**: zero-copy data paths, flat DAG evaluation, dynamic block sizing
- **21 ODE solvers**: explicit and implicit, adaptive and fixed-step
- **Standalone solvers**: `RKDP54.integrate(func, x0, time_end=50)` with automatic JIT compilation
- **JIT compiler**: Python functions traced into flat-tape IR with CSE, constant folding, strength reduction, and FMA detection
- **SIL verification**: `sim.verify_c()` compiles the generated C locally and pins it against the reference engine, sample by sample
- **Automatic differentiation**: symbolic forward-mode AD for analytical Jacobians
- **Standalone JIT + AD API**: `jit(func)` and `jacobian(func)` as JAX-style transformations
- **Event handling**: zero-crossing detection, scheduled events, conditions for hybrid systems
- **Hierarchical**: nest subsystems for modular designs
- **Mutable parameters**: change block parameters at runtime with automatic reconstruction
- **Dynamic ports**: blocks adapt their state dimension to connected inputs

## Quick Example

```python
from fastsim import Simulation, Connection
from fastsim.blocks import Integrator, Amplifier, Adder, Scope

# Damped harmonic oscillator: x'' + 0.5x' + 2x = 0
int_v = Integrator(5)       # velocity, v0=5
int_x = Integrator(2)       # position, x0=2
amp_c = Amplifier(-0.5)     # damping
amp_k = Amplifier(-2)       # spring
add = Adder()
scp = Scope()

sim = Simulation(
    blocks=[int_v, int_x, amp_c, amp_k, add, scp],
    connections=[
        Connection(int_v, int_x, amp_c),
        Connection(int_x, amp_k, scp),
        Connection(amp_c, add),
        Connection(amp_k, add[1]),
        Connection(add, int_v),
    ],
)

sim.run(30)
time, [x] = scp.read()
```

## Standalone ODE Solvers

All 21 solvers are available as standalone integrators with automatic JIT compilation of the right-hand side and symbolic Jacobian generation for implicit solvers.

```python
from fastsim.solvers import RKDP54, ESDIRK43

def lorenz(x, t):
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    return [sigma*(x[1]-x[0]), x[0]*(rho-x[2])-x[1], x[0]*x[1]-beta*x[2]]

# Explicit adaptive solver
t, x = RKDP54.integrate(lorenz, [1, 1, 1], time_end=50.0)

# Implicit solver for stiff systems (Jacobian generated automatically via AD)
t, x = ESDIRK43.integrate(robertson, [1, 0, 0], time_end=1.0, tolerance_lte_abs=1e-8)
```

## JIT Compilation and Automatic Differentiation

Python functions are automatically traced into an optimized SSA computation graph and evaluated in Rust. Available as standalone transformations (JAX-style):

```python
from fastsim.jit import jit, jacobian

# JIT compile (lazy tracing on first call)
f = jit(lorenz)
result = f([1.0, 1.0, 1.0], 0.0)

# Eager compilation with known input size
f = jit(lorenz, n_x=3)

# Automatic Jacobian via symbolic AD
jac_fn = jacobian(lorenz)
J = jac_fn([1.0, 1.0, 1.0], 0.0)  # 3x3 numpy array
```

Supported operations: arithmetic, `np.sin/cos/tan/exp/log/tanh/...`, `np.dot`, `np.clip`, `np.where`, `np.linalg.norm`, `np.cross`, matrix multiply (`@`), `np.sum`, branching via `fastsim.where()`, and more. Falls back to Python for unsupported patterns.

## Automatic Compilation in Blocks

Python functions in ODE, Function, and DynamicalSystem blocks are automatically traced and compiled. No configuration needed.

```python
from fastsim.blocks import ODE
import numpy as np

a, b, c = 0.04, 1e4, 3e7

def robertson(x, u, t):
    return np.array([
        -a*x[0] + b*x[1]*x[2],
         a*x[0] - b*x[1]*x[2] - c*x[1]**2,
         c*x[1]**2
    ])

ode = ODE(robertson, initial_value=[1.0, 0.0, 0.0])
print(ode.jit_compiled)  # True
```

## Custom Blocks

```python
from fastsim.blocks import StateSpace

class ButterworthLowpass(StateSpace):
    def __init__(self, cutoff, order=2):
        from scipy.signal import butter, tf2ss
        b, a = butter(order, cutoff, analog=True)
        A, B, C, D = tf2ss(b, a)
        super().__init__(A=A.tolist(), B=B.tolist(), C=C.tolist(), D=D.tolist())
```

Custom blocks run at full Rust speed; only the constructor runs in Python.

## Mutable Parameters

All block parameters can be modified at runtime. Setting a parameter reconstructs the Rust block automatically while preserving engine state.

```python
from fastsim.blocks import Amplifier, PT1

amp = Amplifier(gain=5.0)
amp.gain = 10.0  # instant, no performance cost

pt1 = PT1(K=1.0, T=0.5)
pt1.set(K=5.0, T=1.0)  # batched update, single reinit
```

## Native-CPU builds (performance)

The distributed wheels are compiled for a portable baseline (SSE2) so they run on
any x86-64 CPU. On FMA/AVX2-heavy models the tape's scalar op loop leaves single-
digit-percent performance on the table because `mul_add` lowers to a libm `fma()`
call without hardware FMA. If you build from source for one specific machine, opt
into the local instruction set:

```bash
# build the extension tuned for the current CPU (FMA/AVX2 where available)
RUSTFLAGS="-C target-cpu=native" maturin develop --release
# or for a plain cargo build
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

This is intentionally not baked into `.cargo/config.toml` — a `target-cpu=native`
binary may crash with `SIGILL` on an older CPU, so it must stay opt-in for
source builds only.

## WebAssembly / Pyodide build

fastsim compiles to `wasm32-unknown-emscripten` and runs in the browser via
[Pyodide](https://pyodide.org). The FMI feature (FMU import) relies on runtime
dynamic-library loading and is excluded from WASM builds; everything else
(solvers, JIT tape interpreter, all blocks) runs unchanged.

```bash
# one-time toolchain setup
rustup toolchain install nightly
rustup component add rust-src --toolchain nightly
rustup target add wasm32-unknown-emscripten --toolchain nightly
pip install pyodide-build                 # into your venv
# emscripten matching the pinned Pyodide version (0.29.4 -> emcc 4.0.9), e.g. via emsdk

# build the wheel (writes dist/fastsim-*-wasm32.whl)
EMSDK_DIR=/path/to/emsdk ./scripts/build_pyodide.sh
```

The wheel installs in Pyodide with `micropip.install` and exposes the full
Python API (`import fastsim`). Override the target release with
`PYODIDE_VERSION=...` (must match your installed emscripten).

## Code Generation (C99)

Any simulation compiles to self-contained, dependency-free **C99** (libm only) —
one struct-based model you can drop into an embedded target, a HIL rig, or an FMU.
The generated code is reentrant by construction (all state lives in the instance
struct) and every extern symbol is prefixed with the model name, so two models
link into one binary.

```python
from fastsim import Simulation, Connection
from fastsim.blocks import Integrator, Amplifier

integ, amp = Integrator(1.0), Amplifier(-1.0)
sim = Simulation([integ, amp], [Connection(integ, amp), Connection(amp, integ)])
sim.run(1.0)                       # assemble the model

files = sim.to_c("decay")          # {"decay.h": "...", "decay.c": "..."}
for name, src in files.items():
    open(name, "w").write(src)
```

```c
/* main.c */
#include <stdio.h>
#include "decay.h"
int main(void) {
    decay_t m;                     /* one instance struct */
    decay_init(&m);
    decay_run(&m, 1.0, 1e-3);      /* integrate to t = 1 */
    printf("%.6f\n", m.x[0]);      /* -> 0.367879 (e^-1) */
}
```

```bash
cc decay.c main.c -lm -o decay && ./decay
```

The emitted API, compiler requirements, and the ABI stability policy are
specified in **[doc/codegen.md](doc/codegen.md)**. Passing options to `to_c(...)`
(numeric type, `layout="library"`, solver, `structure="flat"`) is documented
there and in `help(Simulation.to_c)`.

## License

fastsim is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE):
**free for noncommercial use** (research, teaching, academia, personal and hobby
projects). Commercial use, **including shipping fastsim-generated C code in a
commercial product**, requires a commercial license.

The generated C code is "Output" under the license and carries the same
noncommercial limitation; each generated file is stamped with this notice.

For commercial licensing — including shipping fastsim-generated C in a product —
see **[COMMERCIAL.md](COMMERCIAL.md)** or contact **info@pathsim.org**.

Need a fully open-source option? The pure-Python implementation,
[pathsim](https://github.com/pathsim), is available separately under the MIT
License with no field-of-use restriction.
