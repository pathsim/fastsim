"""What a bare ``import fastsim`` puts within reach.

A drop-in replacement is only one where the *spelling* carries over, and code
written against pathsim reaches its submodules as attributes of the package:

.. code-block:: python

    def build(engine):
        ...
        return engine.Simulation(..., Solver=engine.solvers.RK4)

    build(pathsim)   # worked
    build(fastsim)   # AttributeError: module 'fastsim' has no attribute 'solvers'

`fastsim.solvers` and `fastsim.events` were importable by name but not imported
by the package, so that factory — the natural way to write a model that runs
under either engine — worked for one and not the other.
"""
import importlib

import pytest

import fastsim as fs

try:
    import pathsim as ps
    HAS_PATHSIM = True
except ImportError:                                       # pragma: no cover
    HAS_PATHSIM = False


# Submodules a bare `import fastsim` must expose, without importing them first.
REQUIRED_SUBMODULES = ["blocks", "events", "solvers"]

# pathsim internals with no fastsim counterpart. Listed so this test fails if
# one ever gains one and nobody notices, rather than silently staying absent.
KNOWN_ABSENT = {"Duplex", "LoggerManager", "metadata", "optim", "utils"}


@pytest.mark.parametrize("name", REQUIRED_SUBMODULES)
def test_submodule_is_an_attribute_of_the_package(name):
    """`fastsim.solvers` — not just `import fastsim.solvers`."""
    assert hasattr(fs, name), (
        f"`import fastsim` does not expose `fastsim.{name}`; a model factory "
        f"taking the engine module as an argument cannot reach it")
    assert getattr(fs, name) is importlib.import_module(f"fastsim.{name}")


def test_the_usual_entry_points_resolve_through_the_package():
    """The names a ported script actually types."""
    assert fs.solvers.RK4 is not None
    assert fs.events.ZeroCrossing is not None
    assert fs.blocks.Integrator is not None


def test_engine_module_factory_runs_under_fastsim():
    """The pattern from the failure above, end to end."""
    def build(engine):
        src = engine.blocks.Constant(2.0)
        integ = engine.blocks.Integrator()
        sco = engine.blocks.Scope()
        return engine.Simulation(
            blocks=[src, integ, sco],
            connections=[engine.Connection(src, integ), engine.Connection(integ, sco)],
            Solver=engine.solvers.RK4, dt=1e-3, log=False), sco

    sim, sco = build(fs)
    sim.run(1.0, reset=True, adaptive=False)
    _, [x] = sco.read()
    assert abs(x[-1] - 2.0) < 1e-9, f"expected 2*1s = 2, got {x[-1]}"


@pytest.mark.skipif(not HAS_PATHSIM, reason="pathsim not installed")
def test_top_level_surface_matches_pathsim():
    """Every public name pathsim exposes, fastsim exposes too.

    Except the ones listed as absent — pathsim internals with no counterpart.
    Asserting the list exactly means a newly added one fails here instead of
    silently widening the gap.
    """
    public = lambda m: {n for n in dir(m) if not n.startswith("_")}
    missing = public(ps) - public(fs)
    assert missing == KNOWN_ABSENT, (
        f"top-level surface drifted: unexpectedly missing {sorted(missing - KNOWN_ABSENT)}, "
        f"no longer missing {sorted(KNOWN_ABSENT - missing)}")


@pytest.mark.skipif(not HAS_PATHSIM, reason="pathsim not installed")
@pytest.mark.parametrize("name", REQUIRED_SUBMODULES)
def test_pathsim_exposes_the_same_submodules(name):
    """The requirement above is not invented — it is what pathsim does."""
    assert hasattr(ps, name)
