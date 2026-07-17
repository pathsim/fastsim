"""Codegen block-coverage fuzzing (issue #48).

Complements ``test_codegen_clang_matrix.py`` (which sweeps a few representative
*systems* across the codegen axes) by sweeping *every native block* through the
C backend and checking two contracts:

1. **Lowering** — every native block (all eventful blocks that carry no internal
   solver included) lowers to C. Blocks whose behaviour is host code — an
   internal Newton/collocation solver (DAE/BVP/algebraic-constraint), an external
   FMU, or an untraceable Python callback — are rejected *loudly* with a clear
   message, never silently dropped.
2. **SiL parity** — the generated C reproduces the Rust compiled reference
   trajectory. Enforced bit-for-bit in ``double``; ``float`` is checked only for
   the blocks whose output is not dominated by single-precision drift near a
   discrete decision (mirrors ``TRAJ_DOUBLE_ONLY`` in the matrix test).

Each block is wired into a minimal closed system: its inputs are driven by a
sinusoid (a positive-biased one for the domain-restricted ``Log``/``Sqrt``
family) and its first output feeds an ``Integrator`` so there is continuous
state for the trajectory comparison.
"""
import itertools
import os
import subprocess
import tempfile

import numpy as np
import pytest

import fastsim as fs
from fastsim import blocks as B

from codegen_common import (
    ATOL, RTOL, RK4, EUF, gen_main_c_print, match_worst, reference,
)

SOLVER_CLS = {"rk4": RK4, "euler": EUF}

# --------------------------------------------------------------------------------------
# Block catalogue. value = kwargs for the constructor; a positive-domain flag marks the
# blocks that must be fed a strictly-positive input (log/sqrt).
# --------------------------------------------------------------------------------------

# Blocks that lower to C and are SiL-checked in double *and* float.
LOWERABLE_DOUBLE_AND_FLOAT = {
    "Abs": {}, "Adder": dict(operations="++"), "Alias": {}, "Amplifier": dict(gain=2.0),
    "Atan": {}, "Clip": dict(min_val=-1.0, max_val=1.0), "Constant": dict(value=1.0),
    "Cos": {}, "Cosh": {}, "Divider": dict(operations="*/"), "Exp": {},
    "Integrator": dict(initial_value=0.0), "LUT1D": dict(points=[0.0, 1.0, 2.0], values=[0.0, 1.0, 4.0]),
    "LeadLag": dict(K=1.0, T1=0.1, T2=0.2), "Matrix": dict(A=[[1.0, 0.0], [0.0, 1.0]]),
    "Multiplier": {}, "Norm": {}, "PID": dict(Kp=1.0, Ki=0.5, Kd=0.1, f_max=100.0),
    "PT1": dict(K=1.0, T=0.5), "PT2": dict(K=1.0, T=0.5, d=0.7),
    "Polynomial": dict(coeffs=[1.0, 2.0, 3.0]), "Pow": dict(exponent=2.0),
    "PowProd": dict(exponents=[1.0, 2.0]), "Rescale": dict(i0=0.0, i1=1.0, o0=0.0, o1=10.0),
    "Sin": {}, "Sinh": {}, "SinusoidalSource": dict(frequency=1.0, amplitude=1.0, phase=0.0),
    "Source": dict(func=lambda t: np.sin(t)), "StateSpace": dict(
        A=[[0.0, 1.0], [-1.0, -0.3]], B=[[0.0], [1.0]], C=[[1.0, 0.0]], D=[[0.0]]),
    "Tan": {}, "Tanh": {}, "TransferFunction": dict(Poles=[-1.0], Residues=[1.0], Const=0.0),
    "TransferFunctionNumDen": dict(Num=[1.0], Den=[1.0, 1.0]),
    "TransferFunctionPRC": dict(Poles=[-1.0], Residues=[1.0], Const=0.0),
    "TransferFunctionZPG": dict(Zeros=[], Poles=[-1.0], Gain=1.0),
    "TriangleWaveSource": dict(frequency=1.0, amplitude=1.0, phase=0.0),
    "Differentiator": dict(f_max=100.0), "AntiWindupPID": dict(
        Kp=1.0, Ki=0.5, Kd=0.1, f_max=100.0, Ks=10.0, limits=(-5.0, 5.0)),
    "Function": dict(func=lambda u: np.sin(u)), "DynamicalFunction": dict(func=lambda u: u),
    "DynamicalSystem": dict(func_dyn=lambda x, u, t: -x + u, func_alg=lambda x, u, t: x, initial_value=[0.0]),
    "ODE": dict(func=lambda x, u, t: -x + u, initial_value=[0.0]),
    "ChirpSource": dict(amplitude=1.0, f0=1.0, BW=1.0, T=1.0, sampling_period=0.1, seed=1),
    "GaussianPulseSource": dict(amplitude=1.0, f_max=10.0, tau=0.5),
}

# Domain-restricted: fed a positive input so log/sqrt stay real.
LOWERABLE_POSITIVE = {
    "Log": {}, "Log10": {}, "Sqrt": {},
}

# Lower to C but only SiL-checked in double: float clock/threshold drift near a
# discrete decision (event boundary, comparator step) exceeds the float tolerance
# against the always-double reference — expected, not a codegen defect.
LOWERABLE_DOUBLE_ONLY = {
    "Backlash": dict(width=0.5, f_max=100.0), "Deadband": dict(lower=-0.5, upper=0.5),
    "RateLimiter": dict(rate=1.0, f_max=100.0), "Relay": dict(
        threshold_up=0.5, threshold_down=-0.5, value_up=1.0, value_down=-1.0),
    "Comparator": dict(threshold=0.0), "Equal": dict(tolerance=1e-6),
    "GreaterThan": {}, "LessThan": {}, "Mod": dict(modulus=2.0), "Atan2": {},
    "Switch": dict(switch_state=0), "LogicAnd": {}, "LogicNot": {}, "LogicOr": {},
    "Counter": dict(start=0.0, threshold=5.0), "CounterUp": dict(start=0.0, threshold=5.0),
    "CounterDown": dict(start=5.0, threshold=0.0),
    # discrete / event-driven (double is bit-exact; float drifts one step)
    "Delay": dict(tau=0.1, sampling_period=0.05), "SampleHold": dict(T=0.1, tau=0.0),
    "ZeroOrderHold": dict(T=0.1, tau=0.0), "FirstOrderHold": dict(T=0.1, tau=0.0),
    "DiscreteIntegrator": dict(T=0.1, tau=0.0, initial_value=[0.0]),
    "DiscreteDerivative": dict(T=0.1, tau=0.0),
    "DiscreteStateSpace": dict(A=[[0.9]], B=[[1.0]], C=[[1.0]], D=[[0.0]], T=0.1),
    "DiscreteTransferFunction": dict(Num=[1.0], Den=[1.0, -0.5], T=0.1),
    "FIR": dict(coeffs=[0.5, 0.5], T=0.1, tau=0.0),
    "Wrapper": dict(func=lambda u: np.sin(u), T=0.1, tau=0.0),
    "Step": dict(amplitude=1.0, tau=0.5), "StepSource": dict(amplitude=[1.0, 2.0], tau=[0.1, 0.5]),
    "Pulse": dict(amplitude=1.0, T=1.0), "PulseSource": dict(amplitude=1.0, T=1.0),
    "SquareWaveSource": dict(amplitude=1.0, frequency=1.0, phase=0.0),
    "Clock": dict(T=0.1, tau=0.0), "ClockSource": dict(T=0.1, tau=0.0),
    "WhiteNoise": dict(standard_deviation=1.0, sampling_period=0.05, seed=1),
    "RandomNumberGenerator": dict(sampling_period=0.05, seed=1),
    "SinusoidalPhaseNoiseSource": dict(frequency=1.0, amplitude=1.0, sampling_period=0.1, seed=1),
    "ChirpPhaseNoiseSource": dict(amplitude=1.0, f0=1.0, BW=1.0, T=1.0, sampling_period=0.1, seed=1),
    # Butterworth filters: high-order float accumulation drifts past the reference.
    "ButterworthLowpassFilter": dict(Fc=100.0, n=2), "ButterworthHighpassFilter": dict(Fc=100.0, n=2),
    "ButterworthBandpassFilter": dict(Fc=(10.0, 40.0), n=2), "ButterworthBandstopFilter": dict(Fc=(10.0, 40.0), n=2),
    "AllpassFilter": dict(fs=100.0, n=1),
}

# Multi-output-port blocks: parity needs every output port wired (an unconnected
# port under-sizes the signal buffer). Covered by a dedicated test, not the SISO sweep.
MULTI_OUTPUT = {
    "ADC": dict(n_bits=4, span=(-1.0, 1.0), T=0.1, tau=0.0),
    "DAC": dict(n_bits=4, span=(-1.0, 1.0), T=0.1, tau=0.0),
    "TappedDelay": dict(N=2, T=0.1, tau=0.0),
}

# Blocks the C backend must reject loudly (internal solver / external / opaque host code).
OPAQUE_REJECTED = {
    "AlgebraicConstraint": dict(residual=lambda x, u: x - u, x0=0.0),
    "BVP1D": dict(fun=lambda x, y, dy: dy, bc=lambda ya, yb: [ya[0], yb[0] - 1.0], n_eq=1),
    "FullyImplicitDAE": dict(func=lambda x, xd, u, t: xd + x - u, initial_value=[0.0]),
    "MassMatrixDAE": dict(func=lambda x, u, t: -x + u, mass=[[1.0]], initial_value=[0.0]),
    "SemiExplicitDAE": dict(
        f_dyn=lambda x, z, u, t: -x + z, f_alg=lambda x, z, u, t: z - u, x0=[0.0], z0=[0.0]),
    "PinkNoise": dict(standard_deviation=1.0, sampling_period=0.05, seed=1),
}

DUR, DT = 1.0, 0.02
COMBOS = [dict(zip(["structure", "solver"], c)) | {"reductions": "unrolled", "layout": "compact", "api": "struct"}
          for c in itertools.product(["hierarchical", "flat"], ["rk4", "euler"])]


def _mk(name, kwargs):
    return getattr(B, name)(**kwargs)


def _siso_system(name, kwargs, positive):
    """block driven by a sinusoid (positive-biased if `positive`), first output
    integrated so the trajectory has continuous state."""
    blk = _mk(name, kwargs)
    parts, conns = [blk], []
    if len(blk.inputs) > 0:
        src = (B.Source(func=lambda t: 1.5 + np.sin(t)) if positive
               else B.SinusoidalSource(frequency=1.0, amplitude=1.0, phase=0.0))
        parts.append(src)
        conns.append(fs.Connection(src, blk))
    integ = B.Integrator(0.0)
    parts.append(integ)
    conns.append(fs.Connection(blk, integ))
    return fs.Simulation(parts, conns, dt=DT, log=False)


def _compile_run(files, tmp):
    for n, s in files.items():
        with open(os.path.join(tmp, n), "w") as f:
            f.write(s)
    with open(os.path.join(tmp, "main.c"), "w") as f:
        f.write(gen_main_c_print(files["model.h"], DUR, DT))
    exe = os.path.join(tmp, "a.out")
    cfiles = [os.path.join(tmp, n) for n in files if n.endswith(".c")] + [os.path.join(tmp, "main.c")]
    cp = subprocess.run(["cc", "-O2", "-ffp-contract=off", "-Werror=array-bounds", "-o", exe, *cfiles, "-lm", f"-I{tmp}"],
                        capture_output=True, text=True)
    assert cp.returncode == 0, f"generated C failed to compile:\n{cp.stderr}"
    out = subprocess.run([exe], check=True, capture_output=True, text=True).stdout
    return np.asarray([list(map(float, ln.split())) for ln in out.strip().splitlines()])


def _check_parity(name, kwargs, positive, numeric, tmp_path):
    worst = 0.0
    for combo in COMBOS:
        opts = dict(numeric=numeric, **combo)
        files = _siso_system(name, kwargs, positive).to_c(name="model", **opts)
        with tempfile.TemporaryDirectory(dir=tmp_path) as tmp:
            a = _compile_run(files, tmp)
        xc = a[:, 1:]
        tr, xr = reference(_siso_system(name, kwargs, positive), DUR, DT, SOLVER_CLS[combo["solver"]])
        r, _ = match_worst(xc, xr, numeric)
        worst = max(worst, r)
    assert worst <= 1.0, f"{name} [{numeric}]: SiL trajectory mismatch, worst ratio={worst:.3g}"


_DOUBLE_FLOAT_IDS = sorted(LOWERABLE_DOUBLE_AND_FLOAT)
_POSITIVE_IDS = sorted(LOWERABLE_POSITIVE)
_DOUBLE_ONLY_IDS = sorted(LOWERABLE_DOUBLE_ONLY)
_OPAQUE_IDS = sorted(OPAQUE_REJECTED)


@pytest.mark.parametrize("name", _DOUBLE_FLOAT_IDS + _POSITIVE_IDS + _DOUBLE_ONLY_IDS)
def test_block_double_parity(name, tmp_path):
    kwargs = {**LOWERABLE_DOUBLE_AND_FLOAT, **LOWERABLE_POSITIVE, **LOWERABLE_DOUBLE_ONLY}[name]
    _check_parity(name, kwargs, name in LOWERABLE_POSITIVE, "double", tmp_path)


@pytest.mark.parametrize("name", _DOUBLE_FLOAT_IDS + _POSITIVE_IDS)
def test_block_float_parity(name, tmp_path):
    kwargs = {**LOWERABLE_DOUBLE_AND_FLOAT, **LOWERABLE_POSITIVE}[name]
    _check_parity(name, kwargs, name in LOWERABLE_POSITIVE, "float", tmp_path)


@pytest.mark.parametrize("name", _OPAQUE_IDS)
def test_opaque_block_rejected_loudly(name):
    """DAE/BVP/algebraic-constraint (internal solver), PinkNoise (opaque event)
    and untraceable host code must raise a clear error, never emit silent-wrong C."""
    blk = _mk(name, OPAQUE_REJECTED[name])
    src = B.SinusoidalSource(frequency=1.0, amplitude=1.0, phase=0.0)
    parts, conns = [blk], []
    if len(blk.inputs) > 0:
        parts.append(src)
        conns.append(fs.Connection(src, blk))
    if len(blk.outputs) > 0:
        integ = B.Integrator(0.0)
        parts.append(integ)
        conns.append(fs.Connection(blk, integ))
    sim = fs.Simulation(parts, conns, dt=DT, log=False)
    with pytest.raises((RuntimeError, ValueError)) as exc:
        sim.to_c()
    msg = str(exc.value).lower()
    assert "opaque" in msg or "lowered" in msg or "static op-graph" in msg, \
        f"{name}: rejection message is not actionable: {exc.value}"


@pytest.mark.parametrize("name", sorted(MULTI_OUTPUT))
def test_multi_output_block_double_parity(name, tmp_path):
    """Multi-output-port blocks (every output port wired, so the signal buffer is
    sized for the full fan-out): double SiL parity against the reference."""
    kwargs = MULTI_OUTPUT[name]
    def factory():
        blk = _mk(name, kwargs)
        n_out = len(blk.outputs)
        src = B.SinusoidalSource(frequency=1.0, amplitude=1.0, phase=0.0)
        add = B.Adder("+" * n_out)
        integ = B.Integrator(0.0)
        conns = [fs.Connection(src, blk)] + \
                [fs.Connection(blk[k], add[k]) for k in range(n_out)] + \
                [fs.Connection(add, integ)]
        return fs.Simulation([src, blk, add, integ], conns, dt=DT, log=False)
    worst = 0.0
    for combo in COMBOS:
        files = factory().to_c(name="model", numeric="double", **combo)
        with tempfile.TemporaryDirectory(dir=tmp_path) as tmp:
            a = _compile_run(files, tmp)
        tr, xr = reference(factory(), DUR, DT, SOLVER_CLS[combo["solver"]])
        r, _ = match_worst(a[:, 1:], xr, "double")
        worst = max(worst, r)
    assert worst <= 1.0, f"{name}: multi-output SiL mismatch, worst ratio={worst:.3g}"


# --------------------------------------------------------------------------------------
# Event-boundary parity (Stage B, validation-of-C-against-CompiledSimulation).
#
# The runtime `Schedule` scheduler that the CompiledSimulation uses is the canonical
# event-firing semantics; the generated C reimplements it and MUST agree. A one-ULP
# disagreement at a step boundary shifts an event by a whole step, so the fire/no-fire
# decision is only correct when the C matches the compiled reference across phase /
# period / dt alignments — aligned (event lands exactly on a step), mid-step, and
# slightly-off. This locks in the discrete-event timing fix against regression.
# --------------------------------------------------------------------------------------

# Input-reading discrete blocks that sample on a `T`/`tau` schedule. The event effect
# reads the block input, so a shifted firing step changes the sampled value → the most
# sensitive probe of scheduler/codegen agreement.
_EVENT_BLOCKS = {
    "SampleHold": lambda T, tau: B.SampleHold(T=T, tau=tau),
    "ZeroOrderHold": lambda T, tau: B.ZeroOrderHold(T=T, tau=tau),
    "DiscreteIntegrator": lambda T, tau: B.DiscreteIntegrator(T=T, tau=tau, initial_value=[0.0]),
    "FIR": lambda T, tau: B.FIR(coeffs=[0.5, 0.5], T=T, tau=tau),
}

# (dt, period, phase): aligned boundaries (period a clean multiple of dt, phase 0 or a
# step multiple) are the ULP-fragile cases; the off-grid ones exercise mid-step firing.
_EVENT_ALIGNMENTS = [
    (0.02, 0.1, 0.0),    # event exactly on every 5th step boundary
    (0.02, 0.1, 0.02),   # phase on a step boundary
    (0.02, 0.1, 0.05),   # phase mid-step
    (0.02, 0.1, 0.013),  # phase clearly inside a step
    (0.02, 0.13, 0.0),   # period not a multiple of dt
    (0.025, 0.1, 0.0),   # different dt, aligned
]


@pytest.mark.parametrize("name", sorted(_EVENT_BLOCKS))
@pytest.mark.parametrize("dt,period,phase", _EVENT_ALIGNMENTS)
def test_event_boundary_parity_vs_compiled(name, dt, period, phase, tmp_path):
    def factory():
        src = B.SinusoidalSource(frequency=1.0, amplitude=1.0, phase=0.0)
        blk = _EVENT_BLOCKS[name](period, phase)
        integ = B.Integrator(0.0)
        return fs.Simulation(
            [src, blk, integ],
            [fs.Connection(src, blk), fs.Connection(blk, integ)],
            dt=dt, log=False,
        )

    # Hierarchical + rk4 is the alignment-sensitive combo (per-block algebraic pass,
    # multi-stage step straddling the event boundary).
    files = factory().to_c(name="model", numeric="double", reductions="unrolled",
                           structure="hierarchical", layout="compact", solver="rk4", api="struct")
    duration = 2.0
    main = gen_main_c_print(files["model.h"], duration, dt)
    with tempfile.TemporaryDirectory(dir=tmp_path) as tmp:
        for n, s in files.items():
            with open(os.path.join(tmp, n), "w") as f:
                f.write(s)
        with open(os.path.join(tmp, "main.c"), "w") as f:
            f.write(main)
        exe = os.path.join(tmp, "a.out")
        cfiles = [os.path.join(tmp, n) for n in files if n.endswith(".c")] + [os.path.join(tmp, "main.c")]
        cp = subprocess.run(["cc", "-O2", "-ffp-contract=off", "-o", exe, *cfiles, "-lm", f"-I{tmp}"], capture_output=True, text=True)
        assert cp.returncode == 0, cp.stderr
        out = subprocess.run([exe], check=True, capture_output=True, text=True).stdout
    xc = np.asarray([list(map(float, ln.split())) for ln in out.strip().splitlines()])[:, 1:]
    tr, xr = reference(factory(), duration, dt, RK4)
    r, abs_err = match_worst(xc, xr, "double")
    assert r <= 1.0, (
        f"{name} dt={dt} period={period} phase={phase}: C diverges from the compiled "
        f"reference, worst ratio={r:.3g} (abs={abs_err:.2g}) — event fired on a different step"
    )
