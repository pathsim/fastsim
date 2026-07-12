"""Regression coverage for the standalone `jit()` / `jacobian()` call surface.

The compiled tape is specialized to the traced `x` length. Historically a
later call with a *different* length evaluated the cached tape anyway, which
read out of bounds (undefined behavior — observed as garbage values like
3.3e+257). The contract now is:

- `JitFunction` / `JitJacobian` re-trace transparently on a shape change, so
  every call returns the value of the traced function for *that* shape.
- The Rust-side `InterpretedFn::call` validates slice lengths and panics with
  a precise message rather than reading out of bounds (covered by Rust unit
  tests in `src/ssa/tape.rs`).
"""

import numpy as np

from fastsim.jit import jit, jacobian


def test_jit_retraces_on_shorter_input():
    f = jit(lambda x, t: x.sum())
    assert f(np.arange(8.0)) == 28.0          # traced with n=8
    assert f(np.array([1.0])) == 1.0          # shorter → re-trace, not OOB
    assert f(np.array([1.0, 2.0])) == 3.0     # different again
    assert f(np.arange(8.0)) == 28.0          # and back


def test_jit_retraces_on_longer_input():
    f = jit(lambda x, t: x.sum(), n_x=2)      # eager trace with n=2
    assert f(np.array([1.0, 2.0])) == 3.0
    assert f(np.arange(5.0)) == 10.0          # longer → re-trace


def test_jacobian_retraces_on_shape_change():
    def f(x, t):
        return np.array([x[i] ** 2 for i in range(len(x))])

    J = jacobian(f)
    x3 = np.array([1.0, 2.0, 3.0])
    assert np.allclose(J(x3), np.diag(2.0 * x3))
    x2 = np.array([4.0, 5.0])
    assert np.allclose(J(x2), np.diag(2.0 * x2))  # re-trace, shape (2, 2)


def test_jit_scalar_and_array_alternation():
    f = jit(lambda x, t: 2.0 * x[0])
    assert f(np.array([3.0])) == 6.0
    assert f(7.0) == 14.0                     # scalar call → x = [7.0]
    assert f(np.array([1.0, 1.0])) == 2.0     # wider → re-trace


def test_jit_rejects_non_numeric_t():
    import pytest

    f = jit(lambda x, t: x[0] * t)
    assert f(np.array([2.0]), 3.0) == 6.0
    with pytest.raises(TypeError, match="t must be a number"):
        f(np.array([2.0]), "not a time")      # was silently t=0.0


def test_sign_matches_numpy_at_zero():
    # numpy `sign` semantics end to end through the tape: 0 at 0 (Rust's
    # `signum` returns 1 there), ±1 elsewhere, NaN passthrough.
    f = jit(lambda x, t: np.sign(x))
    got = f(np.array([-2.0, 0.0, 3.0]))
    assert np.array_equal(got, np.sign(np.array([-2.0, 0.0, 3.0])))
    assert np.isnan(f(np.array([np.nan, 1.0, 1.0]))[0])
