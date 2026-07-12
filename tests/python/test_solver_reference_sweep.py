"""Independent (scipy / analytic) cross-validation over the FULL solver set.

The pathsim trajectory-match layer (``test_trajectory_match.py``) pins RKCK54 /
ESDIRK43, so 16 of 18 solvers — including RKF45 and the ground-up NDF/Nordsieck
GEAR52A, which has no external oracle at all — are never cross-checked there.
Worse, an inherited-parity defect is invisible to pathsim-parity by design (both
engines would share it), so an *independent* reference is required.

This module sweeps every solver's standalone ``integrate`` against a reference
that does not come from either engine:

  * a closed-form analytic solution (exponential decay), for every solver;
  * scipy ``Radau`` on the canonical stiff Robertson system, for the implicit /
    BDF solvers;
  * scipy ``Radau`` (dense) on non-stiff Van der Pol, for the adaptive explicit
    solvers.

Run: python -m pytest tests/python/test_solver_reference_sweep.py -v
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import fastsim.solvers as S

# Every solver exposes the same standalone `integrate(...)` surface. SteadyState
# is a boundary/fixed-point solver, not an IVP integrator, so it is out of scope.
ALL_SOLVERS = [
    "DIRK2", "DIRK3", "ESDIRK32", "ESDIRK4", "ESDIRK43", "ESDIRK54",
    "EUB", "EUF", "GEAR52A", "RK4", "RKBS32", "RKCK54", "RKDP54", "RKDP87",
    "RKF21", "RKF45", "RKF78", "RKV65", "SSPRK22", "SSPRK33", "SSPRK34",
]

# Implicit / BDF solvers — the only ones that can take the stiff Robertson system
# with a sane step budget.
IMPLICIT_SOLVERS = [
    "DIRK2", "DIRK3", "ESDIRK32", "ESDIRK4", "ESDIRK43", "ESDIRK54", "EUB", "GEAR52A",
]

# Adaptive explicit embedded-pair solvers — swept on a non-stiff oscillator.
EXPLICIT_ADAPTIVE_SOLVERS = [
    "RKBS32", "RKCK54", "RKDP54", "RKDP87", "RKF21", "RKF45", "RKF78", "RKV65",
]


def _integrate(name, f, x0, t_end, **kw):
    opts = dict(
        time_start=0.0, time_end=t_end, dt=kw.get("dt", 0.01),
        dt_min=1e-15, dt_max=kw.get("dt_max", 0.25), adaptive=True,
        tolerance_lte_abs=kw.get("atol", 1e-9), tolerance_lte_rel=kw.get("rtol", 1e-7),
        max_iterations=100, optimizer="newton_anderson",
    )
    return getattr(S, name).integrate(f, list(x0), **opts)


@pytest.mark.parametrize("name", ALL_SOLVERS)
def test_analytic_decay(name):
    """ẋ = -x, x(0) = 1 -> x(1) = e^-1. Independent (closed-form) reference for
    EVERY solver, so even a solver with no engine oracle is validated."""
    t, x = _integrate(name, lambda x, t: [-x[0]], [1.0], 1.0, dt=0.01, dt_max=0.25)
    assert abs(t[-1] - 1.0) < 1e-9
    # Loose bound covers first-order forward/backward Euler (~2e-4); higher-order
    # methods land many orders tighter.
    assert abs(x[-1, 0] - np.exp(-1.0)) < 1e-3, f"{name}: x(1)={x[-1, 0]}"


@pytest.mark.parametrize("name", IMPLICIT_SOLVERS)
def test_robertson_vs_scipy_radau(name):
    """Robertson (canonical stiff) against scipy Radau — an independent stiff
    oracle, plus the analytic mass-conservation invariant."""
    a, b, c = 0.04, 1e4, 3e7

    def f(x, t):
        return [-a * x[0] + b * x[1] * x[2],
                a * x[0] - b * x[1] * x[2] - c * x[1] ** 2,
                c * x[1] ** 2]

    t, x = _integrate(name, f, [1.0, 0.0, 0.0], 1.0, dt=1e-3, dt_max=0.1, atol=1e-8, rtol=1e-6)
    sol = solve_ivp(lambda tt, xx: f(xx, tt), [0, 1], [1.0, 0.0, 0.0],
                    method="Radau", rtol=1e-10, atol=1e-12)
    ref = sol.y[:, -1]
    assert abs(x[-1].sum() - 1.0) < 1e-6, f"{name}: mass={x[-1].sum()}"
    assert np.max(np.abs(x[-1] - ref)) < 1e-5, f"{name} vs Radau: {np.max(np.abs(x[-1] - ref)):.2e}"


@pytest.mark.parametrize("name", EXPLICIT_ADAPTIVE_SOLVERS)
def test_vanderpol_vs_scipy_radau(name):
    """Non-stiff Van der Pol (mu=1) over a full period against scipy Radau dense
    — an independent reference for the adaptive explicit set, incl. RKF45."""
    mu = 1.0

    def f(x, t):
        return [x[1], mu * (1 - x[0] ** 2) * x[1] - x[0]]

    t, x = _integrate(name, f, [2.0, 0.0], 10.0, dt=0.05, dt_max=0.5, atol=1e-8, rtol=1e-6)
    sol = solve_ivp(lambda tt, xx: f(xx, tt), [0, 10], [2.0, 0.0],
                    method="Radau", rtol=1e-10, atol=1e-12, dense_output=True)
    ref = sol.sol(t).T
    assert np.max(np.abs(x - ref)) < 5e-3, f"{name} vs Radau: {np.max(np.abs(x - ref)):.2e}"
