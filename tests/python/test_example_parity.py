"""Whole-model parity: every pathsim example, run under both engines.

``test_block_parity.py`` compares blocks one at a time. This compares complete
models — the level where wiring, scheduling, event ordering and solver
interaction actually meet. A block can be individually correct and still be
wired or scheduled differently in a real model, which is exactly what single
block tests cannot see.

The corpus is pathsim's own ``examples/`` directory, located via the installed
package (or ``$PATHSIM_EXAMPLES``). Nothing is vendored: the examples stay owned
by pathsim and this picks up whatever the installed version ships.

Two independent gates, deliberately not merged into one verdict:

``test_example_runs``
    the model builds and integrates under fastsim at all. A missing block or a
    renamed argument fails here.
``test_example_matches_pathsim``
    the recorded trajectories agree. Only reached when both engines ran.

Keeping them apart matters: a missing *plot* method is an API gap, not a
numerical one, and letting it mask a real trajectory difference would make the
suite dishonest. The runner therefore stubs the presentation layer (``plot``,
``plot2D``, ``plot3D``) — a validation harness compares recorded data, it never
renders.

Examples are executed in a subprocess: they are third-party scripts that call
``sys.exit``, mutate globals and open figures, and one crash must not take the
session with it.

Cost: ~30s for the run gate and ~2min for the trajectory gate, dominated by
process startup (each example is launched twice). Deselect with
``-k "not test_example_matches_pathsim"`` if the numeric layer needs to move to
a nightly job.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.pathsim

try:
    import pathsim
    HAS_PATHSIM = True
except ImportError:  # pragma: no cover - conftest skips the module
    HAS_PATHSIM = False

# Per-example wall-clock ceiling. Examples are demos, not benchmarks; anything
# slower is almost certainly stuck rather than merely heavy.
TIMEOUT = 120

# Numerical budget for "the same model in two engines". Wider than the block
# harness: an example may legitimately use an adaptive solver, where the two
# engines take different step sequences and only the trajectory itself is
# comparable.
RTOL = 1e-6

# Examples that cannot be compared, with the reason. Not a dumping ground:
# every entry names what makes the example unsuitable, and the guard test below
# fails if one of them disappears from the corpus.
UNCOMPARABLE = {
    "example_radar.py": "needs RealtimeScope (interactive plotting block)",
    "example_spectrum_rf_oneport.py": "needs the optional scikit-rf dependency",
}

# Examples pathsim runs but fastsim does not, each with the API gap that stops
# it. `xfail(strict=True)`: closing a gap makes the suite fail until the entry
# is removed, so this list cannot quietly outlive the problem.
KNOWN_RUN_GAPS = {
    "example_filters.py": "solver BDF2 not implemented",
    "example_solver_hotswap.py": "solver GEAR21 not implemented",
    "examples_odes/example_vanderpol.py": "solver not implemented",
    "example_kalman_filter.py": "block KalmanFilter not implemented",
}


def _pathsim_predates_schedule_drift_fix():
    """True while the installed pathsim still resolves a scheduled tick at the
    drifted time (pre pathsim#249). fastsim carries the fix, so
    Schedule-clocked examples disagree until pathsim releases it."""
    import pathsim.events as pe

    evt = pe.Schedule(t_start=0.0, t_period=0.01)
    evt.resolve(0.005)
    return abs(evt._times[0] - 0.0) > 1e-12


# Examples that run under both engines but disagree in VALUE while the
# installed pathsim predates a fix fastsim already carries. Same discipline as
# KNOWN_RUN_GAPS: xfail(strict=True), so a released fix forces the entry out.
KNOWN_VALUE_GAPS = (
    {
        "example_sar.py": "SAR ADC is Schedule-clocked; installed pathsim "
                          "predates the tick-timing fix (pathsim#249)",
    }
    if HAS_PATHSIM and _pathsim_predates_schedule_drift_fix()
    else {}
)


def _examples_dir():
    """Locate pathsim's examples/.

    The published wheel ships no examples, so this looks in three places, in
    order: an explicit ``$PATHSIM_EXAMPLES``, the source tree around an editable
    install, and a sibling ``pathsim`` checkout next to this repository (the
    usual local layout). CI sets the variable explicitly.
    """
    env = os.environ.get("PATHSIM_EXAMPLES")
    if env:
        return Path(env)
    pkg = Path(pathsim.__file__).resolve().parent
    here = Path(__file__).resolve()
    candidates = [pkg / "examples", pkg.parent / "examples",
                  pkg.parent.parent / "examples"]
    # sibling checkout: <repos>/fastsim/tests/python/... -> <repos>/pathsim
    for parent in here.parents:
        candidates.append(parent.parent / "pathsim" / "examples")
    for cand in candidates:
        if cand.is_dir():
            return cand
    return None


EXAMPLES_DIR = _examples_dir() if HAS_PATHSIM else None
_RUNNER = Path(__file__).with_name("_example_runner.py")


def _collect():
    if not EXAMPLES_DIR or not EXAMPLES_DIR.is_dir():
        return []
    return sorted(p for p in EXAMPLES_DIR.rglob("*.py") if not p.name.startswith("_"))


EXAMPLES = _collect()


def _rel(path):
    return path.relative_to(EXAMPLES_DIR).as_posix()


def _param(path, gaps=None):
    name, rel = path.name, _rel(path)
    marks = []
    if name in UNCOMPARABLE:
        marks.append(pytest.mark.skip(reason=UNCOMPARABLE[name]))
    else:
        gaps = KNOWN_RUN_GAPS if gaps is None else gaps
        reason = gaps.get(rel) or gaps.get(name)
        if reason:
            marks.append(pytest.mark.xfail(strict=True, reason=reason))
    return pytest.param(path, marks=marks, id=rel)


def _run(path, engine):
    """Execute one example under `engine`; return its recorded scope data."""
    proc = subprocess.run(
        [sys.executable, str(_RUNNER), engine, str(path)],
        capture_output=True, text=True, timeout=TIMEOUT,
        cwd=str(path.parent),
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        raise RuntimeError(tail[-1] if tail else "example failed with no message")
    for line in proc.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    raise RuntimeError("runner produced no result")


def _worst_relative(a, b):
    """Worst deviation between two scope recordings, scaled per channel."""
    if len(a) != len(b):
        return float("inf"), f"scope count {len(a)} vs {len(b)}"
    worst, where = 0.0, ""
    for si, (sa, sb) in enumerate(zip(a, b)):
        ta, tb = np.asarray(sa["t"]), np.asarray(sb["t"])
        if len(ta) < 2 or len(tb) < 2:
            continue
        # Compare on the coarser engine's own sample times, interpolating the
        # other only where the grids genuinely differ (adaptive runs).
        grid = ta if len(ta) <= len(tb) else tb
        for ci, (ca, cb) in enumerate(zip(sa["d"], sb["d"])):
            va = np.interp(grid, ta, np.asarray(ca))
            vb = np.interp(grid, tb, np.asarray(cb))
            scale = max(1e-9, float(np.max(np.abs(vb))))
            d = float(np.max(np.abs(va - vb))) / scale
            if d > worst:
                worst, where = d, f"scope {si} channel {ci}"
    return worst, where


@pytest.mark.skipif(not EXAMPLES, reason="pathsim examples/ not found")
@pytest.mark.parametrize("path", [_param(p) for p in EXAMPLES])
def test_example_runs(path):
    """The example builds and integrates under fastsim."""
    try:
        _run(path, "fastsim")
    except subprocess.TimeoutExpired:
        pytest.fail(f"{path.name}: exceeded {TIMEOUT}s under fastsim")
    except RuntimeError as e:
        pytest.fail(f"{path.name}: does not run under fastsim — {e}")


# Examples that run in both engines but whose trajectories differ, each with the
# measured worst relative deviation and the attributed cause.
#
# Every entry was investigated; none is an unexplained defect. The causes are
# systematic, and each was established by isolating the suspected factor rather
# than by assuming it:
#
# GEAR52A
#     fastsim's GEAR52A is deliberately not pathsim's (LSODA-style order ramp-up
#     vs an ESDIRK32 startup integrator), so the two take different step
#     sequences. Demonstrated on `example_vanderpol_subsystem`: the same
#     subsystem model under a fixed-step solver agrees to 0.0e+00 between the
#     engines and to 6.1e-08 against scipy, while GEAR52A on the stiff
#     configuration is where the trajectories part. Five of the six examples
#     that select GEAR52A are in this list.
# events + adaptive step
#     an event instant that differs by one ULP moves a discontinuity by a whole
#     step, and the trajectories separate from there. Physical sensitivity of the
#     model, not an implementation difference.
# stochastic sources
#     the engines draw from different PRNG streams by construction; only the
#     statistics are comparable (covered in `test_trajectory_match.py`).
# algebraic loop
#     each engine iterates its own solver to its own tolerance. `example_diode`
#     runs fixed-step on an identical time grid and differs by 2.4e-07 — the
#     size of a convergence tolerance, not of a modelling error.
KNOWN_TRAJECTORY_GAPS = {
    "examples_event/example_stickslip_event.py": "7.6e+00 — RKBS32 + friction events",
    "example_phasenoise.py": "1.7e+00 — stochastic source, engine-specific PRNG stream",
    "examples_event/example_billards_sphere.py": "1.6e+00 — RKBS32 + collision events",
    "example_noise.py": "1.4e+00 — PinkNoise/WhiteNoise, engine-specific PRNG stream",
    "example_vanderpol_subsystem.py": "1.5e+00 — GEAR52A on stiff VDP (mu=1000) over "
                                      "3000 time units; the subsystem itself is "
                                      "bit-identical under a fixed-step solver",
    "example_nested_subsystems.py": "1.5e+00 — GEAR52A, same cause as vanderpol_subsystem",
    "examples_event/example_bouncing_pendulum.py": "1.4e+00 — RKCK54 + contact events",
    "example_dualslope.py": "6.3e-01 — RKBS32 + comparator events",
    "example_cascade.py": "3.6e-01 — RKCK54 + stochastic source",
    "example_abs_braking.py": "2.4e-01 — RKCK54 + switching events",
    "examples_odes/example_flame.py": "1.1e-01 — ESDIRK43 on a stiff flame model",
    "example_deltasigma.py": "7.2e-02 — RKBS32 + quantiser feedback (decisions flip)",
    "examples_odes/example_robertson.py": "3.5e-02 — GEAR52A on stiff Robertson",
    "example_stickslip.py": "1.9e-02 — GEAR52A + friction",
    "examples_event/example_bouncingball_friction.py": "1.6e-02 — RKBS32 + contact events",
    "example_reactor.py": "5.1e-03 — GEAR52A",
    # AntiWindupPID is identical between the engines: the same controller under a
    # fixed-step solver agrees to 2.5e-16. Both examples drive it with RKCK54
    # around its saturating anti-windup branch, and the two step controllers take
    # different sequences across that non-smooth nonlinearity (143 vs 157 steps).
    "example_dcmotor.py": "6.7e-01 — RKCK54 across the AntiWindupPID saturation; "
                          "the controller itself agrees to 2.5e-16 fixed-step",
    "example_pid_antiwindup.py": "1.3e-02 — same cause as example_dcmotor",
    "examples_event/example_bouncingball.py": "4.2e-03 — RKF21 + contact events",
    # The Condition that stops the run after 10 bounces is not implicated: both
    # engines record the same 15 floor contacts and both end at t=15, while the
    # traces already part company at t=1.65, long before it can fire.
    "examples_event/example_bouncingball_switched.py": "2.1e-03 — RKBS32 + contact "
                                                      "events, as example_bouncingball",
    "examples_event/example_volterralotka_event.py": "2.0e-03 — RKBS32 + events",
    "example_diode.py": "8.5e-06 — algebraic loop, each engine to its own tolerance",
    "examples_event/example_thermostat.py": "4.9e-06 — RKBS32 + relay events",
}


@pytest.mark.skipif(not EXAMPLES, reason="pathsim examples/ not found")
@pytest.mark.parametrize(
    "path", [_param(p, gaps={**KNOWN_RUN_GAPS, **KNOWN_TRAJECTORY_GAPS,
                             **KNOWN_VALUE_GAPS}) for p in EXAMPLES])
def test_example_matches_pathsim(path):
    """The recorded trajectories agree between the engines."""
    try:
        ref = _run(path, "pathsim")
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        pytest.skip(f"example does not run under pathsim either: {e}")
    try:
        got = _run(path, "fastsim")
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        pytest.fail(f"{path.name}: does not run under fastsim — {e}")

    if not ref:
        pytest.skip("example records no scope data")

    worst, where = _worst_relative(got, ref)
    assert worst <= RTOL, (
        f"{path.name}: trajectories differ by {worst:.3e} relative ({where}); "
        f"budget {RTOL}. Check against an independent reference before "
        f"assuming which engine is wrong.")


def test_corpus_is_present():
    """A silently empty corpus would make every test above vacuous.

    Skipped when the corpus is simply unavailable (pathsim installed from a
    wheel), but a hard failure under ``PATHSIM_REQUIRED`` — the same switch
    ``conftest.py`` uses to turn the pathsim skips into failures on CI, so the
    layer cannot green-pass by finding nothing.
    """
    if not EXAMPLES and not os.environ.get("PATHSIM_REQUIRED"):
        pytest.skip(
            "pathsim examples/ not found (wheel install). Set PATHSIM_EXAMPLES "
            "to compare whole models.")
    assert EXAMPLES, (
        "PATHSIM_REQUIRED is set but no pathsim examples were found — set "
        "PATHSIM_EXAMPLES to the examples/ directory")
    assert len(EXAMPLES) >= 20, f"only {len(EXAMPLES)} examples found, expected the full corpus"


def test_uncomparable_entries_still_exist():
    """Guard every exclusion list: a renamed example must not silently drop out
    of the corpus while its excuse stays behind."""
    if not EXAMPLES:
        pytest.skip("no corpus to check the exclusion lists against")
    names = {p.name for p in EXAMPLES} | {_rel(p) for p in EXAMPLES}
    for label, table in (("UNCOMPARABLE", UNCOMPARABLE),
                         ("KNOWN_RUN_GAPS", KNOWN_RUN_GAPS),
                         ("KNOWN_TRAJECTORY_GAPS", KNOWN_TRAJECTORY_GAPS)):
        stale = sorted(n for n in table if n not in names)
        assert not stale, f"{label} names examples that no longer exist: {stale}"
