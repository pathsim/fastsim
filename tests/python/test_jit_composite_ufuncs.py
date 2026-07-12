"""Coverage for the composite numpy surface added on top of the op manifest.

`np.deg2rad`/`rad2deg`/`square`/`reciprocal` and `np.diff`/`np.cumsum` are not
dedicated SSA ops: the tracer lowers them to compositions of existing ops, so
they trace (no Python-callback fallback), evaluate bit-for-bit like numpy, and
get autodiff / codegen for free. Each test asserts the block JIT-compiled and
the traced value matches the numpy reference.
"""

import numpy as np
import pytest

from fastsim import Simulation, Connection
from fastsim.blocks import Function, Constant, Scope


def _jit(block):
    return block.__dict__.get("_jit_compiled", False)


def _alg(fn, c, nout):
    """Wire scalar Constant(s) -> Function(fn) -> Scope, run briefly, read y."""
    c = np.atleast_1d(np.asarray(c, dtype=float))
    srcs = [Constant(float(v)) for v in c]
    f = Function(fn)
    sco = Scope()
    conns = [Connection(srcs[i], f[i]) for i in range(len(c))]
    conns += [Connection(f[j], sco[j]) for j in range(nout)]
    sim = Simulation(srcs + [f, sco], conns, log=False)
    sim.run(0.05)
    _, ch = sco.read()
    y = np.array([cc[-1] for cc in ch[:nout]])
    return y, _jit(f)


# ---- scalar composite ufuncs ----

SCALAR_CASES = [
    ("deg2rad", lambda u: np.deg2rad(u), 90.0, np.deg2rad(90.0)),
    ("rad2deg", lambda u: np.rad2deg(u), np.pi, 180.0),
    ("square", lambda u: np.square(u), 3.0, 9.0),
    ("reciprocal", lambda u: np.reciprocal(u), 4.0, 0.25),
]


@pytest.mark.parametrize("name,fn,inp,ref", SCALAR_CASES)
def test_scalar_composite_ufunc(name, fn, inp, ref):
    y, jit = _alg(fn, inp, 1)
    assert jit, f"{name}: expected JIT compile, got Python fallback"
    assert np.isclose(y[0], ref, atol=1e-12), f"{name}: {y[0]} != {ref}"


# ---- array composite ufuncs + structural ops ----

def test_array_square():
    x = np.array([1.0, 2.0, 4.0, 7.0])
    y, jit = _alg(lambda u: np.square(u), x, len(x))
    assert jit
    assert np.allclose(y, np.square(x))


def test_array_deg2rad():
    x = np.array([0.0, 30.0, 90.0, 180.0])
    y, jit = _alg(lambda u: np.deg2rad(u), x, len(x))
    assert jit
    assert np.allclose(y, np.deg2rad(x))


def test_cumsum():
    x = np.array([1.0, 2.0, 4.0, 7.0])
    y, jit = _alg(lambda u: np.cumsum(u), x, len(x))
    assert jit
    assert np.allclose(y, np.cumsum(x))


def test_diff_via_reduction():
    # np.diff returns length n-1; consumed by a sum it telescopes to x[-1]-x[0].
    x = np.array([1.0, 2.0, 4.0, 7.0])
    y, jit = _alg(lambda u: np.sum(np.diff(u)), x, 1)
    assert jit
    assert np.isclose(y[0], x[-1] - x[0], atol=1e-12)
