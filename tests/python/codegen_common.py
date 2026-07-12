"""Shared fixtures + helpers for the codegen permutation tests.

Used by:
  - ``test_codegen_clang_matrix.py`` - generation + import-contract guard + native
    gcc compile/run/trajectory (always-local).
  - ``test_codegen_wasm_matrix.py`` - faithful clang->wasm compile + Node run with
    the production import table (gated on Docker + Node).

The single source for the representative systems, the codegen axes, the verify
tolerances, the import contract and the C harnesses, so the two layers cannot
drift apart.
"""
import itertools
import re

import numpy as np

import fastsim as fs
from fastsim import blocks as B
from fastsim.solvers import RK4, EUF

# --------------------------------------------------------------------------------------
# Verify tolerances (mirror pathview-fastsim/src/lib/constants/codegen.ts). float codegen
# is compared against the double reference, so it gets a looser margin.
# --------------------------------------------------------------------------------------

ATOL = {"double": 1e-6, "float": 1e-4}
RTOL = {"double": 1e-4, "float": 1e-2}

# --------------------------------------------------------------------------------------
# Import contract - MUST stay in sync with WASM_IMPORT_NAMES in
# pathview-fastsim/src/lib/codegen/clang/wasmRunner.ts. The math names are exactly
# fastsim's op manifest (src/ssa/op.rs: unary_c_fn / binary_c_fn) plus fma and copysign;
# the mem names are the libc builtins clang lowers non-trivial copies to.
# --------------------------------------------------------------------------------------

_MATH = {
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh",
    "asinh", "acosh", "atanh", "exp", "expm1", "log", "log10", "log2", "log1p",
    "sqrt", "cbrt", "hypot", "pow", "fabs", "floor", "ceil", "trunc", "round",
    "fmod", "fmin", "fmax", "fma", "copysign", "erf", "erfc", "lgamma", "tgamma",
}
PROVIDED_IMPORTS = _MATH | {n + "f" for n in _MATH} | {"memcpy", "memmove", "memset"}

# C control-flow keywords that are followed by "(" but are not function calls.
_C_KEYWORDS = {"if", "for", "while", "switch", "sizeof", "return", "do", "else"}


def external_calls(files):
    """Functions the generated C calls but does not define - the link-time imports
    (every undefined symbol clang turns into an ``env.*`` import under
    ``--allow-undefined``). Strips comments, collects ``name(`` call sites, then
    subtracts locally-defined functions (``name(...) {``) and C keywords."""
    called, defined = set(), set()
    for src in files.values():
        code = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
        code = re.sub(r"//.*", "", code)
        called.update(m.group(1) for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", code))
        defined.update(m.group(1) for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{", code))
    return called - defined - _C_KEYWORDS


# --------------------------------------------------------------------------------------
# Representative systems: name -> (factory, duration). Each factory returns a fresh
# Simulation (built twice: once for codegen, once for the reference run).
# --------------------------------------------------------------------------------------

def _ode():
    s = B.SinusoidalSource(frequency=1.0, amplitude=1.0)
    i = B.Integrator(0.0); o = B.Scope()
    return fs.Simulation([s, i, o], [fs.Connection(s, i), fs.Connection(i, o)], dt=0.05, log=False)


def _mathchain():
    s = B.SinusoidalSource(frequency=1.0, amplitude=0.5)
    f = B.Function(lambda u: np.exp(np.sin(u)) + np.sqrt(np.abs(u) + 1.0) + u ** 3
                   + np.log(np.abs(u) + 1.0) + np.tanh(u) + np.arctan2(u, 1.0) + np.hypot(u, 0.5))
    i = B.Integrator(0.0); o = B.Scope()
    return fs.Simulation([s, f, i, o],
        [fs.Connection(s, f), fs.Connection(f, i), fs.Connection(i, o)], dt=0.02, log=False)


def _harmonic():
    m, c, k = 0.8, 0.2, 1.5
    I1 = B.Integrator(5.0); I2 = B.Integrator(2.0)
    A1 = B.Amplifier(c); A2 = B.Amplifier(k); A3 = B.Amplifier(-1 / m)
    P1 = B.Adder(); S = B.Scope()
    return fs.Simulation([I1, I2, A1, A2, A3, P1, S], [
        fs.Connection(I1, I2, A1, S), fs.Connection(I2, A2, S[1]),
        fs.Connection(A1, P1), fs.Connection(A2, P1[1]), fs.Connection(P1, A3),
        fs.Connection(A3, I1)], dt=0.02, log=False)


def _lorenz():
    sigma, rho, beta = 10, 28, 8 / 3
    i1 = B.Integrator(1.0); i2 = B.Integrator(1.0); i3 = B.Integrator(1.0)
    a1 = B.Amplifier(sigma); ax = B.Adder("+-"); cr = B.Constant(rho); ar = B.Adder("+-")
    mr = B.Multiplier(); ay = B.Adder("-+"); mxy = B.Multiplier()
    ab = B.Amplifier(beta); az = B.Adder("+-"); s = B.Scope()
    return fs.Simulation([i1, i2, i3, a1, ax, cr, ar, mr, ay, mxy, ab, az, s], [
        fs.Connection(i1, ax[1], mr[0], mxy[0], s[0]), fs.Connection(i2, ax[0], ay[0], mxy[1], s[1]),
        fs.Connection(i3, ar[1], ab, s[2]), fs.Connection(ax, a1), fs.Connection(a1, i1),
        fs.Connection(cr, ar[0]), fs.Connection(ar, mr[1]), fs.Connection(mr, ay[1]), fs.Connection(ay, i2),
        fs.Connection(mxy, az[0]), fs.Connection(ab, az[1]), fs.Connection(az, i3)], dt=0.01, log=False)


def _statespace():
    A = [[0.0, 1.0], [-1.0, -0.3]]; Bm = [[0.0], [1.0]]; Cm = [[1.0, 0.0]]; D = [[0.0]]
    s = B.SinusoidalSource(frequency=0.5, amplitude=1.0)
    ss = B.StateSpace(A, Bm, Cm, D); o = B.Scope()
    return fs.Simulation([s, ss, o], [fs.Connection(s, ss), fs.Connection(ss, o)], dt=0.02, log=False)


def _pid():
    s = B.SinusoidalSource(frequency=0.5, amplitude=1.0)
    p = B.PID(Kp=2.0, Ki=0.5, Kd=0.1); i = B.Integrator(0.0); o = B.Scope()
    return fs.Simulation([s, p, i, o],
        [fs.Connection(s, p), fs.Connection(p, i), fs.Connection(i, o)], dt=0.01, log=False)


def _event():
    s = B.StepSource(amplitude=[1.0, 2.0, 0.5], tau=[0.1, 0.5, 1.2])
    i = B.Integrator(0.0); o = B.Scope()
    return fs.Simulation([s, i, o], [fs.Connection(s, i), fs.Connection(i, o)], dt=0.02, log=False)


def _rng():
    r = B.RandomNumberGenerator(sampling_period=0.05, seed=7)
    i = B.Integrator(0.0); o = B.Scope()
    return fs.Simulation([r, i, o], [fs.Connection(r, i), fs.Connection(i, o)], dt=0.05, log=False)


SYSTEMS = {
    "ode": (_ode, 2.0),
    "mathchain": (_mathchain, 2.0),
    "harmonic": (_harmonic, 10.0),
    "lorenz": (_lorenz, 5.0),
    "statespace": (_statespace, 4.0),
    "pid": (_pid, 3.0),
    "event": (_event, 2.0),
    "rng": (_rng, 2.0),
}

# Systems too precision-sensitive to compare float codegen against the (always-double)
# reference - noise integration accumulates the float<->double sampling difference.
TRAJ_DOUBLE_ONLY = {"rng"}

AXES = {
    "numeric": ["double", "float"],
    "reductions": ["unrolled", "vectorized"],
    "structure": ["hierarchical", "flat"],
    "layout": ["compact", "library"],
    "solver": ["rk4", "euler"],
    "api": ["struct"],
}
KEYS = list(AXES)
COMBOS = list(itertools.product(*AXES.values()))
SOLVER_CLS = {"rk4": RK4, "euler": EUF}

# Subset used for the (heavier) trajectory checks: the axes that move the numerics or
# the emitted symbols. layout/reductions/api are emission-shape variants already
# compiled+linked by the import-contract test.
TRAJ_COMBOS = [dict(zip(["numeric", "structure", "solver"], c))
               | {"reductions": "unrolled", "layout": "compact", "api": "struct"}
               for c in itertools.product(["double", "float"], ["hierarchical", "flat"], ["rk4", "euler"])]


def combo_id(combo):
    return "_".join(combo)


# --------------------------------------------------------------------------------------
# C harnesses (struct API). Both integrate one dt step at a time (REC=1) so the rows
# align with the reference index-for-index. `gen_main_c_print` writes rows to stdout
# (native gcc run); `gen_main_c_buffer` fills a `buffer[]` and exports `run`/`buffer`,
# matching the in-app harness.ts so the Node/WASM runner reads it like wasmRunner.ts.
# --------------------------------------------------------------------------------------

def _harness_dims(header, duration, dt):
    macro = re.search(r"#define\s+(\w*N_STATE)\b", header).group(1)
    # Entry points are prefixed with the model name (`<name>_init`/`<name>_run`);
    # capture that prefix + the instance struct type so the harness works for any
    # model name, not just the default "model".
    m = re.search(r"void\s+(\w+)_init\(\s*(?:const\s+)?(\w+)\s*\*", header)
    prefix, init_type = m.group(1), m.group(2)
    nsteps = max(0, round(duration / dt))
    return macro, init_type, prefix, nsteps


def gen_main_c_print(header, duration, dt, header_include="model.h"):
    macro, init_type, prefix, nsteps = _harness_dims(header, duration, dt)
    return "\n".join([
        '#include "%s"' % header_include, "#include <stdio.h>", "#define NSTEPS %dL" % nsteps,
        "int main(void) {", "    %s m; %s_init(&m);" % (init_type, prefix),
        '    printf("%.17g", (double)m.time);',
        '    for (int s = 0; s < %s; s++) printf(" %%.17g", (double)m.x[s]);' % macro, '    printf("\\n");',
        "    for (long i = 0; i < NSTEPS; i++) {",
        "        %s_run(&m, (double)(i + 1) * %r, %r);" % (prefix, dt, dt),
        '        printf("%.17g", (double)m.time);',
        '        for (int s = 0; s < %s; s++) printf(" %%.17g", (double)m.x[s]);' % macro, '        printf("\\n");',
        "    }", "    return 0;", "}",
    ])


def gen_main_c_buffer(header, duration, dt, header_include="model.h"):
    macro, init_type, prefix, nsteps = _harness_dims(header, duration, dt)
    max_rows = nsteps + 2
    return "\n".join([
        '#include "%s"' % header_include,
        "/* Stub required by the wasm-ld toolchain (mirrors harness.ts). */",
        "void __wasm_signal(int sig) { (void)sig; }",
        "#define STRIDE (1 + %s)" % macro, "#define NSTEPS %dL" % nsteps,
        "double buffer[%d * STRIDE];" % max_rows,
        "int run(void) {",
        "    %s m; %s_init(&m);" % (init_type, prefix),
        "    long row = 0;",
        "    buffer[row * STRIDE] = (double)m.time;",
        "    for (int s = 0; s < %s; s++) buffer[row * STRIDE + 1 + s] = (double)m.x[s];" % macro,
        "    row++;",
        "    for (long i = 0; i < NSTEPS; i++) {",
        "        %s_run(&m, (double)(i + 1) * %r, %r);" % (prefix, dt, dt),
        "        buffer[row * STRIDE] = (double)m.time;",
        "        for (int s = 0; s < %s; s++) buffer[row * STRIDE + 1 + s] = (double)m.x[s];" % macro,
        "        row++;",
        "    }",
        "    return (int)row;",
        "}",
    ])


def state_count(header):
    return int(re.search(r"#define\s+\w*N_STATE\s+(\d+)", header).group(1))


# --------------------------------------------------------------------------------------
# Reference + comparison
# --------------------------------------------------------------------------------------

def reference(sim, duration, dt, solver_cls):
    """fastsim's own fixed-step state trajectory (same solver + dt as the C)."""
    sim._set_solver(solver_cls)
    sim.dt = dt
    c = sim.compile()
    t, states, _ = c.run(duration, True, False)
    return np.asarray(t, dtype=float), np.asarray(states, dtype=float)


def match_worst(xc, xr, numeric):
    """Worst-case error ratio with permutation-invariant state pairing.

    ``to_c`` and ``compile()`` can order the state vector differently (PID: same
    values, columns rotated), so pair C states to reference states by the
    assignment minimising total error, then take the worst paired ratio. A
    genuinely wrong state still fails (no reference series matches it)."""
    from scipy.optimize import linear_sum_assignment
    n = min(len(xc), len(xr))
    assert n > 0, "no overlapping samples"
    assert xc.shape[1] == xr.shape[1], f"state count mismatch C={xc.shape[1]} ref={xr.shape[1]}"
    nc = xc.shape[1]
    cost = np.empty((nc, nc))
    abs_cost = np.empty((nc, nc))
    for i in range(nc):
        for j in range(nc):
            abs_err = np.abs(xc[:n, i] - xr[:n, j])
            denom = ATOL[numeric] + RTOL[numeric] * np.abs(xr[:n, j])
            cost[i, j] = (abs_err / denom).max()
            abs_cost[i, j] = abs_err.max()
    ri, ci = linear_sum_assignment(cost)
    return float(cost[ri, ci].max()), float(abs_cost[ri, ci].max())
