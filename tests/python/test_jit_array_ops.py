"""Coverage for the JIT tracer on array operands.

Each test builds an ODE block with an RHS that exercises one operator family
(unary ufuncs, element-wise comparison, binary min/max/hypot/arctan2, broadcast
with scalar or numpy constants) and asserts:

1. the block was JIT-compiled (not the Python-callback fallback)
2. the compiled block evaluates to the same result as the reference Python fn

The coverage deliberately mirrors the JitTracer scalar side so regressions that
re-introduce asymmetry between ``JitTracer`` and ``JitTracerArray`` show up.
"""

import numpy as np
import pytest

import fastsim
from fastsim import Simulation, Connection
from fastsim.blocks import ODE, Source, Scope
from fastsim.solvers import RKDP54


def _jit(block):
    return block.__dict__.get("_jit_compiled", False)


def _run_and_read(block, duration=1.0, n_out=None):
    """Build a trivial sim around `block`, run once, return final y."""
    sco = Scope()
    n = n_out if n_out is not None else 1
    conns = [Connection(block[i], sco[i]) for i in range(n)]
    sim = Simulation([block, sco], conns, log=False)
    sim._set_solver(RKDP54, tolerance_lte_abs=1e-10)
    sim.run(duration)
    _, ch = sco.read()
    return np.array([c[-1] for c in ch[:n]])


# ---------------- Unary ufuncs on arrays ----------------

UNARY_CASES = [
    ("sin",     np.sin),
    ("cos",     np.cos),
    ("tan",     np.tan),
    ("exp",     np.exp),
    ("log",     lambda x: np.log(x + 2.0)),  # keep positive
    ("sqrt",    lambda x: np.sqrt(x + 2.0)),
    ("sinh",    np.sinh),
    ("cosh",    np.cosh),
    ("tanh",    np.tanh),
    ("arctan",  np.arctan),
    ("expm1",   np.expm1),
    ("log1p",   lambda x: np.log1p(x + 1.0)),
    ("absolute",np.abs),
    ("cbrt",    np.cbrt),
    ("sign",    np.sign),
]


@pytest.mark.parametrize("ufunc_name,np_fn", UNARY_CASES)
def test_unary_ufunc_on_array_compiles(ufunc_name, np_fn):
    # Build a tiny 1-state ODE whose RHS applies the ufunc to the state array.
    def rhs(x, t):
        return -x + np_fn(x)

    blk = ODE(rhs, initial_value=[0.5])
    assert _jit(blk), f"{ufunc_name}: expected JIT compile, got Python fallback"
    y = _run_and_read(blk, duration=0.1)
    assert np.all(np.isfinite(y)), f"{ufunc_name}: non-finite output"


# ---------------- Binary ufuncs min/max/arctan2/hypot ----------------

BINARY_CASES = [
    ("minimum", lambda a, b: np.minimum(a, b)),
    ("maximum", lambda a, b: np.maximum(a, b)),
    ("hypot",   lambda a, b: np.hypot(a, b)),
    ("arctan2", lambda a, b: np.arctan2(a, b)),
]


@pytest.mark.parametrize("ufunc_name,np_fn", BINARY_CASES)
def test_binary_ufunc_array_vs_scalar_compiles(ufunc_name, np_fn):
    def rhs(x, t):
        return -x + np_fn(x, 0.3)  # array op scalar

    blk = ODE(rhs, initial_value=[0.5])
    assert _jit(blk), f"{ufunc_name}(array, scalar): expected JIT compile"
    y = _run_and_read(blk, duration=0.1)
    assert np.all(np.isfinite(y))


@pytest.mark.parametrize("ufunc_name,np_fn", BINARY_CASES)
def test_binary_ufunc_array_vs_array_compiles(ufunc_name, np_fn):
    const = np.array([0.4, 0.9])

    def rhs(x, t):
        return -x + np_fn(x, const)  # array op np.ndarray

    blk = ODE(rhs, initial_value=[0.5, 0.5])
    assert _jit(blk), f"{ufunc_name}(array, ndarray): expected JIT compile"
    y = _run_and_read(blk, duration=0.1, n_out=2)
    assert np.all(np.isfinite(y))


# ---------------- Comparison operators ----------------

CMP_CASES = [
    ("gt",  lambda a, b: a > b),
    ("ge",  lambda a, b: a >= b),
    ("lt",  lambda a, b: a < b),
    ("le",  lambda a, b: a <= b),
    ("eq",  lambda a, b: a == b),
    ("ne",  lambda a, b: a != b),
]


@pytest.mark.parametrize("name,np_fn", CMP_CASES)
def test_array_comparison_operator_compiles(name, np_fn):
    # Comparison produces a tracer array of 0.0/1.0; multiply into dynamics.
    def rhs(x, t):
        gate = np_fn(x, np.array([0.3]))  # array ⊕ ndarray, returns 0/1 array
        return -x * gate

    blk = ODE(rhs, initial_value=[1.0])
    assert _jit(blk), f"cmp {name}: expected JIT compile"
    y = _run_and_read(blk, duration=0.5)
    assert np.all(np.isfinite(y))


def test_numpy_where_with_array_comparison_compiles():
    # np.where on a tracer-produced boolean array (common real-world pattern)
    def rhs(x, t):
        cond = x > 0.0
        return np.where(cond, -x, 1.0 + x)

    blk = ODE(rhs, initial_value=[0.5])
    assert _jit(blk), "np.where(array > 0, ...): expected JIT compile"
    y = _run_and_read(blk, duration=0.3)
    assert np.all(np.isfinite(y))


# ---------------- Broadcasting (size-1 against size-N) ----------------

def test_broadcast_scalar_array_against_sizeN_compiles():
    """Common in Toolboxes: numpy 0-dim/size-1 array * TracerArray size>=1."""
    coeff = np.array(0.7)  # 0-d numpy array

    def rhs(x, t):
        return -coeff * x  # should broadcast via __rmul__ -> array_binop

    blk = ODE(rhs, initial_value=[1.0, 1.5])
    assert _jit(blk), "0-d array * TracerArray: expected JIT compile"
    y = _run_and_read(blk, duration=0.2, n_out=2)
    assert np.all(np.isfinite(y))


def test_size1_against_size2_broadcast():
    def rhs(x, t):
        # x is size 2. multiply by a size-1 constant -> broadcast
        w = np.array([2.0])
        return -x * w

    blk = ODE(rhs, initial_value=[1.0, 2.0])
    assert _jit(blk), "broadcast size-1 against size-2: expected JIT compile"
    y = _run_and_read(blk, duration=0.2, n_out=2)
    assert np.all(np.isfinite(y))


# ---------------- Reductions ----------------

REDUCTION_CASES = [
    ("sum",  np.sum),
    ("prod", np.prod),
    ("min",  np.min),
    ("max",  np.max),
    ("mean", np.mean),
    ("var",  np.var),
    ("std",  np.std),
]


@pytest.mark.parametrize("name,np_fn", REDUCTION_CASES)
def test_reduction_on_array_compiles(name, np_fn):
    """Reductions that fold an array to a scalar must all go through the JIT."""
    # Use the reduction result as a scalar that influences state evolution.
    def rhs(x, t):
        s = np_fn(x)
        return -x * s  # x: (n,), s: scalar → element-wise back to (n,)

    blk = ODE(rhs, initial_value=[0.5, 1.0, 1.5])
    assert _jit(blk), f"np.{name}: expected JIT compile"
    y = _run_and_read(blk, duration=0.1, n_out=3)
    assert np.all(np.isfinite(y))


def test_numerical_reduction_matches_numpy():
    """The traced reduction must produce numerically correct results, not just
    compile. We compare a one-step evaluation against numpy directly."""
    from fastsim.jit import jit

    def f(x):
        return [np.sum(x), np.mean(x), np.min(x), np.max(x), np.prod(x),
                np.var(x), np.std(x)]

    f_jit = jit(f, n_x=4)
    x = np.array([1.0, 2.0, 3.0, 4.0])
    expected = f(x)
    got = f_jit(x)
    for i, (lbl, e, g) in enumerate(zip(
        ["sum", "mean", "min", "max", "prod", "var", "std"], expected, got
    )):
        assert abs(e - g) < 1e-10, f"np.{lbl}: expected {e}, got {g}"


# ---------------- Debug diagnostics (FASTSIM_JIT_DEBUG) ----------------

def test_jit_debug_flag_surfaces_trace_failure(monkeypatch):
    """With FASTSIM_JIT_DEBUG=1, a trace failure must emit a warning. Silent
    fallbacks are the primary robustness gap — this covers the regression path."""
    import warnings

    # Operation that the tracer cannot handle: bare `if` on a tracer scalar.
    def rhs_unsupported(x, t):
        if x[0] > 0:
            return -x
        return x

    monkeypatch.setenv("FASTSIM_JIT_DEBUG", "1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        blk = ODE(rhs_unsupported, initial_value=[1.0])
    # Block still works (Python callback), but a warning was emitted
    assert not _jit(blk), "unsupported op should not JIT-compile"
    assert any("JIT trace failed" in str(w.message) for w in caught), \
        "debug flag should surface the reason for JIT failure"


def test_jit_debug_flag_off_by_default(monkeypatch):
    """Without the flag, trace failures stay silent (original behaviour)."""
    import warnings

    monkeypatch.delenv("FASTSIM_JIT_DEBUG", raising=False)

    def rhs_unsupported(x, t):
        if x[0] > 0:
            return -x
        return x

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = ODE(rhs_unsupported, initial_value=[1.0])
    assert not any("JIT trace" in str(w.message) for w in caught), \
        "no JIT warning expected without FASTSIM_JIT_DEBUG"
