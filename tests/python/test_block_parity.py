"""Block-level parity: every catalogued block, fastsim vs pathsim.

Complements ``test_trajectory_match.py``, which hand-writes one test per block
and covers 40 of them. This sweeps the whole catalogue
(``block_catalogue.py``) so a block cannot be added without being compared.

**Fixed grid, not interpolation.** Both engines run the same fixed-step solver
at the same ``dt`` with ``adaptive=False``, so the recorded time grids are
identical and the comparison is index-for-index. The older helper interpolates
both trajectories onto 200 points, which manufactures differences wherever the
step points differ — on an adaptive run that artefact reached 5e-7 on a solver
that is accurate to 1e-16. Any deviation this module reports is a real one.

**Where the block output is read.** The block feeds a `Scope` directly (its
instantaneous output) AND an `Integrator` into a second channel, so both the
algebraic output and its accumulated effect are compared: a sign error in a
derivative shows up in the integral even when the output looks plausible.

When a block does deviate, the failure is not automatically fastsim's: the
engines are independent implementations of the same block. Confirm against an
independent reference (analytic or scipy) before deciding which side to fix.
"""
import numpy as np
import pytest

import fastsim as fs
from fastsim import blocks as FB
from fastsim.solvers import RK4 as F_RK4

from block_catalogue import (
    ALL_BLOCKS, LOWERABLE_DOUBLE_AND_FLOAT, LOWERABLE_DOUBLE_ONLY,
    LOWERABLE_POSITIVE, MULTI_OUTPUT, POSITIVE_DOMAIN, catalogue_gaps,
)

pytestmark = pytest.mark.pathsim

try:
    import pathsim as ps
    from pathsim import blocks as PB
    from pathsim.solvers import RK4 as P_RK4
    HAS_PATHSIM = True
except ImportError:  # pragma: no cover - conftest skips the module
    HAS_PATHSIM = False
    P_RK4 = None

DT = 0.01
DURATION = 1.0

# Absolute/relative budget for "the same block, same solver, same steps". Two
# independent implementations of the same recurrence accumulate different
# rounding, so this is a rounding budget, not a modelling one.
ATOL = 1e-9
RTOL = 1e-7

# --------------------------------------------------------------------------------------
# Entries that depend on which pathsim is installed.
#
# A fix is merged upstream long before it reaches PyPI, so an entry pinned to a
# version number is wrong against one of the two — a released wheel or a source
# checkout — whichever it was not written for. Probing the installed pathsim for
# the behaviour itself is right against both, and drops the entry by itself on
# the day the fix arrives instead of leaving a permanent excuse behind.
#
# Each probe answers one question: is the OLD behaviour still there?
# --------------------------------------------------------------------------------------

def _relay_starts_at_zero():
    """pathsim <=0.24.0 left a `Relay`'s output at the register default 0.0
    until the first threshold crossing, so a two-state hysteresis relay reported
    a third value. Fixed upstream (pathsim#246)."""
    relay = PB.Relay(threshold_up=0.5, threshold_down=-0.5,
                     value_up=1.0, value_down=-1.0)
    return float(np.atleast_1d(relay.outputs[0])[0]) != -1.0


def _matrix_rejects_nested_list():
    """pathsim <=0.24.0 required an ndarray for `Matrix(A=...)`. Fixed upstream
    (pathsim#247)."""
    try:
        PB.Matrix(A=[[1.0, 0.0], [0.0, 1.0]])
        return False
    except Exception:
        return True


def _schedule_fires_a_step_early():
    """pathsim <=0.24.0 resolved a scheduled event at the START of the step it
    was found in, so a schedule off the step grid fired up to a full `dt` early.
    Fixed upstream (pathsim#248) and ported here, which is why the two engines
    disagree until the fix is released."""
    import pathsim.events as pe

    fired = []
    evt = pe.Schedule(t_start=0.05, t_period=0.1, func_act=fired.append)
    src, sco = PB.Constant(1.0), PB.Scope()
    sim = ps.Simulation([src, sco], [ps.Connection(src, sco)], events=[evt],
                        dt=0.01, log=False)
    sim.run(0.2, reset=True)
    return bool(fired) and abs(fired[0] - 0.05) > 1e-9


def _schedule_stamps_the_drifted_time():
    """pathsim <=0.24.0 stamped `resolve(t)` with the numerically drifted time
    it was handed, and dropped a tick that landed a few ulp behind a step
    boundary (clock drift pushed it into the next step, and the final tick of a
    run off the end). Fixed upstream (pathsim#249) and ported here: `Schedule`
    absorbs the drift in `detect` and resolves at the exact scheduled time —
    which is why every Schedule-clocked block disagrees with an installed
    pathsim that predates the fix."""
    import pathsim.events as pe

    evt = pe.Schedule(t_start=0.0, t_period=0.01)
    evt.resolve(0.005)  # off-schedule resolve time
    return abs(evt._times[0] - 0.0) > 1e-12


def _pending(name, probe, reason):
    """``{name: reason}`` while the installed pathsim still shows the old
    behaviour, ``{}`` once it no longer does."""
    if not HAS_PATHSIM:
        return {}
    try:
        still_broken = probe()
    except Exception:
        still_broken = True
    if not still_broken:
        return {}
    version = getattr(ps, "__version__", "unknown")
    return {name: f"{reason} (installed pathsim {version})"}


# Blocks that draw random numbers. Their streams are engine-specific (different
# PRNG), so their VALUES cannot be compared — only their statistics, which is
# what `test_trajectory_match.py` already does. Excluded here by design.
STOCHASTIC = {
    "WhiteNoise", "RandomNumberGenerator", "PinkNoise",
    "SinusoidalPhaseNoiseSource", "ChirpPhaseNoiseSource", "ChirpSource",
}

# Blocks that carry an internal solver (DAE/BVP/algebraic constraint). Their
# inner Newton/collocation iterates to its own tolerance in each engine, so the
# results agree only to that tolerance — compared with a widened budget.
INNER_SOLVER = {
    "AlgebraicConstraint", "BVP1D", "FullyImplicitDAE", "MassMatrixDAE", "SemiExplicitDAE",
}

# Blocks whose inputs are digital BITS, one port per bit. Driving them with an
# analogue sweep compares two different kinds of nonsense; they get a real
# digital code instead (see `_build`).
BIT_INPUTS = {"DAC"}

# Blocks that divide by, or take a fractional power of, an input. A driving
# signal that crosses zero makes them singular, which would show up as a
# "parity failure" that is really a modelling choice in the test. They get a
# strictly-positive drive bounded away from zero.
NONZERO_INPUTS = {"Divider", "PowProd", "Pow"}

COMPARABLE = sorted(
    set(ALL_BLOCKS) - STOCHASTIC
    - {"CoSimulationFMU", "ModelExchangeFMU"}
)

# Blocks whose pathsim constructor takes different arguments than fastsim's, so
# the shared catalogue cannot build both. A drop-in gap in its own right —
# tracked here rather than silently skipped, and separate from a VALUE
# disagreement, which is what this module is really about.
SIGNATURE_MISMATCH = {
    # Real drop-in gaps: the same block takes different argument NAMES in each
    # engine, so pathsim source does not run on fastsim unchanged. fastsim is
    # the drop-in replacement, so fastsim is the side that should also accept
    # pathsim's names.
    "SemiExplicitDAE": "fastsim takes `f_dyn`/`f_alg`/`x0`, "
                       "pathsim takes `func_dyn`/`func_alg`/`initial_value`",
    "BVP1D": "fastsim takes `n_eq`/`n_mesh`/`initial`/`x_out`, "
             "pathsim takes `n`/`n_nodes`/`y0`/`x_eval`",
    # Not a drop-in gap in fastsim: the shared catalogue simply cannot build the
    # pathsim side while the installed pathsim rejects a nested list for `A`.
    **_pending("Matrix", _matrix_rejects_nested_list,
               "pathsim rejects a nested list for `A`; fixed upstream"),
}

_SCHEDULE_DRIFT_REASON = (
    "pathsim resolves a drift-boundary tick one step later and stamps the "
    "drifted time; fixed upstream (pathsim#249)"
)

# Confirmed VALUE disagreements, each with the side that is wrong and why.
# `strict=True`: when a fix lands, the unexpected pass fails the suite so the
# entry has to be removed — the list cannot rot into a permanent excuse.
KNOWN_DIVERGENCE = {
    # Every Schedule-clocked block, in one stroke: the tick-timing fix
    # (pathsim#249) moves which step a drift-boundary tick fires on, so all of
    # them disagree with a pre-fix pathsim and agree again once it releases.
    **{k: v for name in (
        "ADC", "Delay", "DiscreteDerivative", "DiscreteIntegrator",
        "DiscreteStateSpace", "DiscreteTransferFunction", "FIR", "SampleHold",
        "StepSource", "TappedDelay", "Wrapper", "ZeroOrderHold",
    ) for k, v in _pending(name, _schedule_stamps_the_drifted_time,
                           _SCHEDULE_DRIFT_REASON).items()},
    # Verified to agree exactly once pathsim carries the fix, so this entry is
    # present only while the installed one does not.
    **_pending("Relay", _relay_starts_at_zero,
               "pathsim starts the output at 0.0 instead of `value_down`; "
               "fixed upstream"),
    # FirstOrderHold interpolates BETWEEN samples, so a one-step shift in when
    # its schedule fires changes the slope over the whole run, not just at the
    # sample instants — which is why it is the block that shows this and the
    # other sampled ones do not.
    **_pending("FirstOrderHold", _schedule_fires_a_step_early,
               "pathsim resolves a scheduled event up to one `dt` early, so the "
               "hold samples at the wrong instants; fixed upstream"),
    # Investigated: NOT a block defect. With clean 0/1 bits both engines rebuild
    # the same code; what differs is whether the DAC's sampling event sees the
    # bit values from before or after the driving edges in the same step. Same
    # scheduling class as Delay.
    "DAC": "sampling event and the driving bit edges are ordered differently "
           "within a step, so the held code lags by one sample in one engine. "
           "Scheduling difference, not a block defect.",
    # fastsim's side is fixed (see test_pulse_edges.py): with zero rise/fall it
    # now matches the ideal pulse exactly. What remains is the FIRST sample:
    # with tau=0 the pulse is high on [0, T*duty), so t=0 belongs to the high
    # phase — fastsim reports `amplitude`, pathsim still reports its initial
    # 'low' because the phase events have not fired yet at t=0.
    "Pulse": "t=0 only: pathsim reports its initial 'low' phase although a "
             "tau=0 pulse is already high at t=0. fastsim matches the ideal "
             "pulse exactly over the whole run.",
    "PulseSource": "same first-sample difference as Pulse.",
    # Investigated: the block itself agrees. Wired directly (sources -> PowProd
    # -> scope) both engines match the definition prod(u_i**e_i) EXACTLY, with
    # and without a downstream integrator. The disagreement only appears in this
    # harness's wiring, and adding one more sink connection makes it vanish — so
    # it tracks the algebraic evaluation ORDER, not the block. Same class as the
    # Delay/DAC scheduling findings.
    "PowProd": "agrees exactly when wired directly; the deviation depends on "
               "graph shape (an extra sink connection removes it), so it is "
               "evaluation-order dependent, not a block defect.",
}

# Blocks whose OUTPUT matches but whose INTEGRAL does not: the engines carry a
# discontinuity through the integrator differently. Not a block defect — listed
# apart so it is never "fixed" in the block.
INTEGRAL_ONLY_DIVERGENCE = {
    # Outputs are identical sample for sample; the integrals differ because the
    # RK stages inside a step see different values across the discontinuity.
    # Measured against the ANALYTIC integral of the square wave, fastsim is the
    # closer of the two (3.3e-3 vs 1.0e-2 over 0.5s at dt=0.01), so this is
    # approximation quality, not a defect — deliberately not aligned downwards.
    "Clock": "output matches; integrating across the edge differs, fastsim is "
             "closer to the analytic integral (3.3e-3 vs 1.0e-2)",
    "ClockSource": "as Clock",
    "SquareWaveSource": "as Clock; the two differ only at the t=0.5 edge",
    # Output agrees to ~3e-12 — each engine's inner Newton converges to its own
    # tolerance, and integrating that residual accumulates it. Widening the
    # budget until this passed would only hide how much drift is tolerated.
    "AlgebraicConstraint": "output agrees to ~3e-12; the inner Newton's "
                           "residual accumulates once integrated",
}


def _build(engine, blocks_mod, solver_cls, name, kwargs):
    """`source -> block -> (scope, integrator -> scope)` in the given engine.

    Sources (no inputs) skip the driving source; sinks would have no output to
    record and are not in the catalogue.
    """
    Conn = engine.Connection
    blk = getattr(blocks_mod, name)(**kwargs)
    sco = blocks_mod.Scope()
    parts, conns = [blk, sco], []

    n_in = len(blk.inputs)
    if n_in > 0:
        if name in BIT_INPUTS:
            # A DAC's inputs are BITS, not an analogue level: pathsim rebuilds
            # the code as sum(inputs[i] * 2**i), so anything other than a clean
            # 0/1 per port is meaningless input, and comparing the two engines
            # on it compares two kinds of nonsense. Drive each port with a
            # pulse train that is exactly 0 or 1, halving in rate per bit, so
            # the code counts through its range.
            for bit in range(n_in):
                pulse = blocks_mod.PulseSource(
                    amplitude=1.0, T=0.1 * (2 ** (bit + 1)), duty=0.5)
                parts.append(pulse)
                conns.append(Conn(pulse, blk[bit]))
        else:
            # EVERY input port gets its own phase-shifted drive. Driving only
            # port 0 leaves the rest at zero, which silently turns a two-input
            # block into a one-input one — `AND(u, 0)` is 0 under any truth
            # convention, so such a block would "pass" without being compared.
            for port in range(n_in):
                phase = 0.13 * port
                if name in POSITIVE_DOMAIN or name in NONZERO_INPUTS:
                    src = blocks_mod.Source(
                        func=(lambda ph: lambda t: 1.5 + 0.5 * np.sin(2 * np.pi * (t + ph)))(phase))
                else:
                    src = blocks_mod.SinusoidalSource(
                        frequency=1.0, amplitude=1.0, phase=2 * np.pi * phase)
                parts.append(src)
                conns.append(Conn(src, blk[port]))

    # Channel 0: the block's instantaneous output. Channel 1: its integral.
    integ = blocks_mod.Integrator(0.0)
    parts.append(integ)
    conns.append(Conn(blk, integ, sco[0]))
    conns.append(Conn(integ, sco[1]))

    return engine.Simulation(parts, conns, Solver=solver_cls, dt=DT, log=False), sco


def _run(engine, blocks_mod, solver_cls, name, kwargs):
    sim, sco = _build(engine, blocks_mod, solver_cls, name, kwargs)
    sim.run(duration=DURATION, reset=True, adaptive=False)
    t, data = sco.read()
    return np.asarray(t), [np.asarray(ch) for ch in data]


# Channel meaning, so a failure says WHAT disagrees. Keeping them apart matters:
# a block whose output matches but whose integral does not is not a block defect
# — it is the engines integrating across a discontinuity differently, which is a
# solver/event question and must not be "fixed" in the block.
CH_OUTPUT, CH_INTEGRAL = 0, 1
CH_NAME = {CH_OUTPUT: "block output", CH_INTEGRAL: "integral of the output"}


def _deviations(name, kwargs):
    """Worst scaled deviation PER CHANNEL, index-for-index on the shared grid."""
    tf, df = _run(fs, FB, F_RK4, name, kwargs)
    tp, dp = _run(ps, PB, P_RK4, name, kwargs)

    assert len(tf) == len(tp), (
        f"{name}: step counts differ ({len(tf)} vs {len(tp)}) — the fixed-step "
        f"grids must match before values can be compared")
    assert np.allclose(tf, tp, rtol=0, atol=1e-12), f"{name}: time grids differ"
    assert len(df) == len(dp), f"{name}: channel counts differ ({len(df)} vs {len(dp)})"

    atol = ATOL * (1e6 if name in INNER_SOLVER else 1.0)
    rtol = RTOL * (1e4 if name in INNER_SOLVER else 1.0)
    out = []
    for a, b in zip(df, dp):
        scaled = np.abs(a - b) / (atol + rtol * np.abs(b))
        i = int(np.argmax(scaled))
        out.append((float(scaled[i]), i))
    return out, tf


def _param(name, known=None):
    marks = []
    if name in SIGNATURE_MISMATCH:
        marks.append(pytest.mark.skip(reason=f"constructor parity: {SIGNATURE_MISMATCH[name]}"))
    elif name in (KNOWN_DIVERGENCE if known is None else known):
        reason = (KNOWN_DIVERGENCE if known is None else known)[name]
        marks.append(pytest.mark.xfail(strict=True, reason=reason))
    return pytest.param(name, marks=marks)


def _param_integral(name):
    """Integral-channel parametrisation: a block whose OUTPUT diverges will also
    diverge here, plus the blocks that only differ once integrated."""
    return _param(name, known={**KNOWN_DIVERGENCE, **INTEGRAL_ONLY_DIVERGENCE})


@pytest.mark.parametrize("name", [_param(n) for n in COMPARABLE])
def test_block_matches_pathsim(name):
    """The block's own output agrees between the engines."""
    devs, t = _deviations(name, ALL_BLOCKS[name])
    worst, where = devs[CH_OUTPUT]
    assert worst <= 1.0, (
        f"{name}: block OUTPUT differs between fastsim and pathsim — worst scaled "
        f"deviation {worst:.3g} at t={t[where]:.4f} (budget atol={ATOL}, rtol={RTOL}). "
        f"Check against an independent reference before assuming which side is wrong.")


@pytest.mark.parametrize("name", [_param_integral(n) for n in COMPARABLE])
def test_block_integral_matches_pathsim(name):
    """Integrating the block's output agrees too.

    Separate from the output check on purpose: when the outputs match but the
    integrals do not, the block is fine and the engines differ in how they carry
    a discontinuity through the integrator — a solver/event question, not a
    block one, and fixing it in the block would be wrong.
    """
    devs, t = _deviations(name, ALL_BLOCKS[name])
    if len(devs) <= CH_INTEGRAL:
        pytest.skip("no integral channel (block has no usable output)")
    worst, where = devs[CH_INTEGRAL]
    out_worst = devs[CH_OUTPUT][0]
    assert worst <= 1.0, (
        f"{name}: INTEGRAL of the output differs — worst scaled deviation "
        f"{worst:.3g} at t={t[where]:.4f}, while the output itself deviates by "
        f"{out_worst:.3g}. "
        + ("The output matches, so this is integration across a discontinuity, "
           "not a block defect." if out_worst <= 1.0 else
           "The output differs too — fix that first."))


def test_catalogue_covers_every_block():
    """A newly added block must be catalogued, so every sweep picks it up."""
    gaps = catalogue_gaps(FB)
    assert not gaps, (
        f"blocks missing from block_catalogue.py: {gaps}. Add them to the group "
        f"matching how they must be tested, or to UNCATALOGUED with a reason.")


def test_exclusion_lists_stay_honest():
    """Guard every exclusion list against silent growth: each name must still
    exist, so a renamed block cannot quietly drop out of the sweep."""
    names = STOCHASTIC | INNER_SOLVER | set(SIGNATURE_MISMATCH) | set(KNOWN_DIVERGENCE)
    unknown = sorted(n for n in names if n not in ALL_BLOCKS)
    assert not unknown, f"exclusion lists name blocks that no longer exist: {unknown}"
