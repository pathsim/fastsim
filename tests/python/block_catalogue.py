"""The single block catalogue: every native block with constructor kwargs that
build a valid, numerically well-behaved instance.

One source, three consumers:

  * ``test_codegen_block_coverage.py`` — sweeps every block through the C backend.
  * ``test_block_parity.py`` — compares every block against pathsim.
  * anything else that needs "one of each block" without re-deriving arguments.

Blocks are grouped by the property that decides how a test may treat them, not
by taxonomy — the groups ARE the test parametrisation:

``LOWERABLE_DOUBLE_AND_FLOAT``
    Lower to C and hold parity in both double and float.
``LOWERABLE_POSITIVE``
    Same, but the input must stay strictly positive (log/sqrt domain).
``LOWERABLE_DOUBLE_ONLY``
    Lower to C, but only double is compared: these sit near a discrete decision
    (event boundary, comparator threshold, quantiser) where float drift moves
    the decision by a step. Expected, not a defect.
``MULTI_OUTPUT``
    Several output ports; parity needs every port wired or the signal buffer is
    under-sized.
``OPAQUE_REJECTED``
    Must be rejected loudly by the C backend (internal solver, external FMU, or
    untraceable host callback) — never silently mis-emitted.

Keeping the kwargs here (rather than inline in one test) is what lets a new
block be covered everywhere by adding a single line.
"""
import numpy as np

# --------------------------------------------------------------------------------------
# Lowerable blocks
# --------------------------------------------------------------------------------------

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
    "Function": dict(func=lambda u: np.sin(u)),
    # DynamicalFunction's callback takes (u, t) in both engines — a one-argument
    # lambda constructs fine and only fails once it is called.
    "DynamicalFunction": dict(func=lambda u, t: u),
    "DynamicalSystem": dict(func_dyn=lambda x, u, t: -x + u, func_alg=lambda x, u, t: x, initial_value=[0.0]),
    "ODE": dict(func=lambda x, u, t: -x + u, initial_value=[0.0]),
    "ChirpSource": dict(amplitude=1.0, f0=1.0, BW=1.0, T=1.0, sampling_period=0.1, seed=1),
    "GaussianPulseSource": dict(amplitude=1.0, f_max=10.0, tau=0.5),
}

# Domain-restricted: fed a positive input so log/sqrt stay real.
LOWERABLE_POSITIVE = {
    "Log": {}, "Log10": {}, "Sqrt": {},
}

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

MULTI_OUTPUT = {
    "ADC": dict(n_bits=4, span=(-1.0, 1.0), T=0.1, tau=0.0),
    "DAC": dict(n_bits=4, span=(-1.0, 1.0), T=0.1, tau=0.0),
    "TappedDelay": dict(N=2, T=0.1, tau=0.0),
}

OPAQUE_REJECTED = {
    # `func` is pathsim's name; fastsim accepts it as an alias for `residual`,
    # so one catalogue entry builds the block in both engines.
    "AlgebraicConstraint": dict(func=lambda x, u: x - u, x0=0.0),
    "BVP1D": dict(fun=lambda x, y, dy: dy, bc=lambda ya, yb: [ya[0], yb[0] - 1.0], n_eq=1),
    "FullyImplicitDAE": dict(func=lambda x, xd, u, t: xd + x - u, initial_value=[0.0]),
    "MassMatrixDAE": dict(func=lambda x, u, t: -x + u, mass=[[1.0]], initial_value=[0.0]),
    "SemiExplicitDAE": dict(
        f_dyn=lambda x, z, u, t: -x + z, f_alg=lambda x, z, u, t: z - u, x0=[0.0], z0=[0.0]),
    "PinkNoise": dict(standard_deviation=1.0, sampling_period=0.05, seed=1),
}

# Every catalogued block, name -> kwargs.
ALL_BLOCKS = {
    **LOWERABLE_DOUBLE_AND_FLOAT, **LOWERABLE_POSITIVE, **LOWERABLE_DOUBLE_ONLY,
    **MULTI_OUTPUT, **OPAQUE_REJECTED,
}

# Blocks whose input must stay strictly positive.
POSITIVE_DOMAIN = set(LOWERABLE_POSITIVE)


def catalogue_gaps(block_module):
    """Names exported by `block_module` that the catalogue does not cover.

    Used by a guard test so a newly added block cannot slip past every sweep:
    the only acceptable gaps are the non-simulation entries listed in
    ``UNCATALOGUED``.
    """
    exported = {n for n in dir(block_module) if n[:1].isupper()}
    return sorted(exported - set(ALL_BLOCKS) - UNCATALOGUED)


# Deliberately outside the catalogue: `Block` is the base class, `Scope`/`Spectrum`
# are recorders with no dynamics to compare, and the FMU blocks need an external
# .fmu file (covered by the dedicated FMI tests).
UNCATALOGUED = {"Block", "Scope", "Spectrum", "CoSimulationFMU", "ModelExchangeFMU"}
