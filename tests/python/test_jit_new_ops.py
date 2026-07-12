"""Tests for CasADi-parity operations added to the JIT tracer.

Covers:
- New unary ops: asin, acos, asinh, acosh, atanh, ceil, round, trunc,
  log2, log1p, expm1, cbrt, erf, erfc, gamma, lgamma
- New binary op: hypot
- Tier-2 robustness: np.linalg.norm with ord=1/2/inf/fro, np.cross non-3D error

Each numeric op is verified against its numpy/math/scipy reference. Derivatives
are checked against finite differences (where defined) or the closed form.
"""
import math
import numpy as np
import pytest
from scipy import special

from scipy import special as sp

from fastsim.jit import jit, jacobian


# ---------------------------------------------------------------------------
# Value tests: trace, evaluate, compare to reference
# ---------------------------------------------------------------------------

UNARY_CASES = [
    # (fastsim_traceable, reference, x)  — x must lie in the function domain
    (lambda x: np.arcsin(x[0]),   lambda v: math.asin(v),    0.3),
    (lambda x: np.arccos(x[0]),   lambda v: math.acos(v),    0.3),
    (lambda x: np.arcsinh(x[0]),  lambda v: math.asinh(v),   0.7),
    (lambda x: np.arccosh(x[0]),  lambda v: math.acosh(v),   2.0),
    (lambda x: np.arctanh(x[0]),  lambda v: math.atanh(v),   0.5),
    (lambda x: np.ceil(x[0]),     lambda v: math.ceil(v),    2.3),
    (lambda x: np.rint(x[0]),     lambda v: round(v),        2.7),
    (lambda x: np.trunc(x[0]),    lambda v: math.trunc(v),  -2.7),
    (lambda x: np.log2(x[0]),     lambda v: math.log2(v),    8.0),
    (lambda x: np.log1p(x[0]),    lambda v: math.log1p(v),   0.5),
    (lambda x: np.expm1(x[0]),    lambda v: math.expm1(v),   0.5),
    (lambda x: np.cbrt(x[0]),     lambda v: v ** (1/3),      27.0),
    (lambda x: sp.erf(x[0]),      lambda v: math.erf(v),     0.7),
    (lambda x: sp.erfc(x[0]),     lambda v: math.erfc(v),    0.7),
    (lambda x: sp.gamma(x[0]),    lambda v: math.gamma(v),   3.5),
    (lambda x: sp.gammaln(x[0]),  lambda v: math.lgamma(v),  3.5),
]


@pytest.mark.parametrize("fastsim_fn,ref_fn,x_val", UNARY_CASES)
def test_unary_op_value(fastsim_fn, ref_fn, x_val):
    """Each new unary op produces the same value as its numpy/math reference."""
    def wrapped(x):
        return [fastsim_fn(x)]
    f = jit(wrapped, n_x=1)
    got = float(f([x_val]))
    expected = ref_fn(x_val)
    assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_hypot_value():
    f = jit(lambda x: [np.hypot(x[0], x[1])], n_x=2)
    got = float(f([3.0, 4.0]))
    assert math.isclose(got, 5.0, rel_tol=1e-12)
    # Stability: hypot should not overflow for large values
    got = float(f([1e200, 1e200]))
    assert math.isclose(got, math.hypot(1e200, 1e200), rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Derivative tests: compare symbolic AD against closed-form / finite differences
# ---------------------------------------------------------------------------

AD_CASES = [
    # (fn, expected_derivative_at_x, x)
    (lambda x: np.arcsin(x[0]),  lambda v: 1.0 / math.sqrt(1 - v*v),          0.3),
    (lambda x: np.arccos(x[0]),  lambda v: -1.0 / math.sqrt(1 - v*v),         0.3),
    (lambda x: np.arcsinh(x[0]), lambda v: 1.0 / math.sqrt(1 + v*v),          0.7),
    (lambda x: np.arccosh(x[0]), lambda v: 1.0 / math.sqrt(v*v - 1),          2.0),
    (lambda x: np.arctanh(x[0]), lambda v: 1.0 / (1 - v*v),                   0.5),
    (lambda x: np.log2(x[0]),    lambda v: 1.0 / (v * math.log(2)),           8.0),
    (lambda x: np.log1p(x[0]),   lambda v: 1.0 / (1 + v),                     0.5),
    (lambda x: np.expm1(x[0]),   lambda v: math.exp(v),                       0.5),
    (lambda x: np.cbrt(x[0]),    lambda v: 1.0 / (3 * v**(2/3)),              27.0),
    (lambda x: sp.erf(x[0]),     lambda v: 2/math.sqrt(math.pi)*math.exp(-v*v), 0.7),
    (lambda x: sp.erfc(x[0]),    lambda v: -2/math.sqrt(math.pi)*math.exp(-v*v), 0.7),
]


@pytest.mark.parametrize("fn,ref_deriv,x_val", AD_CASES)
def test_derivative_closed_form(fn, ref_deriv, x_val):
    """Symbolic AD matches the analytical derivative."""
    def wrapped(x):
        return [fn(x)]
    j = jacobian(wrapped, n_x=1)
    got = float(j([x_val]))
    expected = ref_deriv(x_val)
    assert math.isclose(got, expected, rel_tol=1e-10, abs_tol=1e-10)


def test_hypot_derivative():
    """d/da hypot(a,b) = a / hypot(a,b), d/db = b / hypot(a,b)"""
    j = jacobian(lambda x: [np.hypot(x[0], x[1])], n_x=2)
    J = np.asarray(j([3.0, 4.0]))
    # expected gradient: [3/5, 4/5]
    assert np.allclose(J.ravel(), [0.6, 0.8], rtol=1e-12)


def test_ceil_round_trunc_zero_derivative():
    """Rounding functions have zero derivative (piecewise constant)."""
    for fn in (np.ceil, np.rint, np.trunc):
        def wrapped(x, _fn=fn):
            return [_fn(x[0]) + 0.5 * x[0]]
        # signature-based wrapper with default doesn't work (trace_to_graph
        # counts defaults as params); instead bind via closure in a helper.
        f_bound = (lambda _f=fn: (lambda x: [_f(x[0]) + 0.5 * x[0]]))()
        j = jacobian(f_bound, n_x=1)
        got = float(j([2.3]))
        assert math.isclose(got, 0.5, rel_tol=1e-12)


def test_gamma_derivative_uses_digamma():
    """d/dx gammaln(x) == digamma(x); d/dx gamma(x) == gamma(x)·digamma(x).
    Our in-house digamma is an asymptotic + recurrence expansion accurate to
    ~1e-9 (scipy uses a more sophisticated polynomial). 1e-8 tolerance is well
    inside what AD needs for Jacobians."""
    from scipy.special import digamma
    for xv in (0.5, 1.0, 2.3, 4.7):
        j = jacobian(lambda x: [sp.gammaln(x[0])], n_x=1)
        got = float(j([xv]))
        expected = digamma(xv)
        assert math.isclose(got, expected, rel_tol=1e-8, abs_tol=1e-8), \
            f"d/dx lgamma({xv}) = {got}, expected {expected}"

        j = jacobian(lambda x: [sp.gamma(x[0])], n_x=1)
        got = float(j([xv]))
        expected = sp.gamma(xv) * digamma(xv)
        assert math.isclose(got, expected, rel_tol=1e-8, abs_tol=1e-8), \
            f"d/dx gamma({xv}) = {got}, expected {expected}"


# ---------------------------------------------------------------------------
# Robustness: np.linalg.norm with ord, np.cross non-3D
# ---------------------------------------------------------------------------

class TestNormVariants:
    def test_norm_default_l2(self):
        f = jit(lambda x: [np.linalg.norm(x)], n_x=3)
        got = float(f([3.0, 4.0, 0.0]))
        assert math.isclose(got, 5.0, rel_tol=1e-12)

    def test_norm_l1(self):
        f = jit(lambda x: [np.linalg.norm(x, ord=1)], n_x=3)
        got = float(f([1.0, -2.0, 3.0]))
        assert math.isclose(got, 6.0, rel_tol=1e-12)

    def test_norm_inf(self):
        f = jit(lambda x: [np.linalg.norm(x, ord=np.inf)], n_x=3)
        got = float(f([1.0, -5.0, 3.0]))
        assert math.isclose(got, 5.0, rel_tol=1e-12)

    def test_norm_neg_inf(self):
        f = jit(lambda x: [np.linalg.norm(x, ord=-np.inf)], n_x=3)
        got = float(f([1.0, -5.0, 3.0]))
        assert math.isclose(got, 1.0, rel_tol=1e-12)

    def test_norm_fro(self):
        # 'fro' is the same as L2 for 1-D
        f = jit(lambda x: [np.linalg.norm(x, ord='fro')], n_x=3)
        got = float(f([3.0, 4.0, 0.0]))
        assert math.isclose(got, 5.0, rel_tol=1e-12)

    def test_norm_unsupported_ord_raises(self):
        with pytest.raises(TypeError, match="ord=3"):
            jit(lambda x: [np.linalg.norm(x, ord=3)], n_x=2)([1.0, 2.0])

    def test_norm_unsupported_string_raises(self):
        with pytest.raises(TypeError, match="not supported"):
            jit(lambda x: [np.linalg.norm(x, ord='nuc')], n_x=2)([1.0, 2.0])


class TestCrossRobustness:
    """Verify np.cross requires 3D inputs. Non-3D must raise a clear error
    rather than silently fall back to numpy (which was the pre-fix behavior)."""

    def test_cross_3d_with_constants(self):
        # x is 3D → cross with a constant vector
        f = jit(lambda x: np.cross(x, [0.0, 1.0, 0.0]), n_x=3)
        got = np.asarray(f([1.0, 0.0, 0.0]))
        assert np.allclose(got, [0, 0, 1], rtol=1e-12)

    def test_cross_non_3d_raises(self):
        # n_x=2 makes the tracer array 2D — cross must reject
        with pytest.raises((TypeError, ValueError)):
            jit(lambda x: np.cross(x, [1.0, 2.0]), n_x=2)([1.0, 2.0])


# ---------------------------------------------------------------------------
# Compound / real-world smoke tests
# ---------------------------------------------------------------------------

def test_compound_expression():
    """Mix new ops: exp(-x^2) * erf(x) + log1p(x^2)"""
    def f(x):
        return [np.exp(-x[0]**2) * sp.erf(x[0]) + np.log1p(x[0]**2)]
    fn = jit(f, n_x=1)
    got = float(fn([1.3]))
    expected = math.exp(-1.3**2) * math.erf(1.3) + math.log1p(1.3**2)
    assert math.isclose(got, expected, rel_tol=1e-12)


def test_compound_jacobian():
    """AD through a chain of new ops against scipy/finite-difference."""
    def f(x):
        return [np.arcsinh(x[0]) + np.hypot(x[0], x[1])]
    fn = jacobian(f, n_x=2)
    J = np.asarray(fn([0.7, 1.2])).ravel()
    # d/dx0 = 1/sqrt(1+x0^2) + x0/hypot(x0,x1)
    # d/dx1 =                    x1/hypot(x0,x1)
    h = math.hypot(0.7, 1.2)
    expected = np.array([1.0/math.sqrt(1 + 0.7**2) + 0.7/h, 1.2/h])
    assert np.allclose(J, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Array / multi-output / more permutations
# ---------------------------------------------------------------------------

class TestArrayBehavior:
    """Tests that exercise numpy-array interactions inside traced functions."""

    def test_builtin_sum_over_list(self):
        """Python sum() over a list comprehension of tracer values."""
        fn = jit(lambda x: [sum(x[i] * (i + 1) for i in range(3))], n_x=3)
        got = float(fn([2.0, 3.0, 4.0]))
        # 2*1 + 3*2 + 4*3 = 2 + 6 + 12 = 20
        assert math.isclose(got, 20.0)

    def test_python_list_of_tracers_as_output(self):
        """Return a Python list — common pattern for multi-output ODEs."""
        def f(x):
            return [x[0] + x[1], x[0] * x[1], x[0] - x[1]]
        fn = jit(f, n_x=2)
        got = np.asarray(fn([3.0, 5.0]))
        assert np.allclose(got, [8.0, 15.0, -2.0])

    def test_np_array_constant_multiply(self):
        """Multiplication tracer * np.array(constant) broadcasts element-wise."""
        def f(x):
            c = np.array([2.0, 3.0, 4.0])
            # multiplying each element manually (broadcast not traced)
            return [x[0] * c[0], x[0] * c[1], x[0] * c[2]]
        fn = jit(f, n_x=1)
        got = np.asarray(fn([5.0]))
        assert np.allclose(got, [10.0, 15.0, 20.0])

    def test_np_dot_tracer_and_const_vector(self):
        """np.dot(tracer_array, constant_vector)"""
        def f(x):
            c = np.array([1.0, 2.0, 3.0])
            return [np.dot(x, c)]
        fn = jit(f, n_x=3)
        got = float(fn([4.0, 5.0, 6.0]))
        # 4*1 + 5*2 + 6*3 = 32
        assert math.isclose(got, 32.0, rel_tol=1e-12)

    def test_np_sum_on_tracer_array(self):
        """np.sum(tracer_array) → scalar tracer."""
        def f(x):
            return [np.sum(x)]
        fn = jit(f, n_x=4)
        got = float(fn([1.0, 2.0, 3.0, 4.0]))
        assert math.isclose(got, 10.0)

    def test_matmul_with_constant_matrix(self):
        """@ operator: constant 2x3 matrix times tracer vector (3D)."""
        def f(x):
            M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            y = M @ x
            return [y[0], y[1]]
        fn = jit(f, n_x=3)
        got = np.asarray(fn([1.0, 1.0, 1.0]))
        assert np.allclose(got, [6.0, 15.0])

    def test_np_where_elementwise(self):
        """np.where routed through __array_function__ on scalar tracer."""
        def f(x):
            return [np.where(x[0] > 0.0, np.exp(x[0]), np.log(-x[0] + 1.0))]
        fn = jit(f, n_x=1)
        got_pos = float(fn([0.5]))
        got_neg = float(fn([-0.5]))
        assert math.isclose(got_pos, math.exp(0.5), rel_tol=1e-12)
        assert math.isclose(got_neg, math.log(0.5 + 1.0), rel_tol=1e-12)


class TestPermutations:
    """More permutations: nested ops, mixed paths, edge-ish domains."""

    def test_deeply_nested(self):
        """Chain many ops: sqrt(erf(exp(-tanh(x^2))))"""
        def f(x):
            return [np.sqrt(sp.erf(np.exp(-np.tanh(x[0]**2))))]
        fn = jit(f, n_x=1)
        got = float(fn([0.5]))
        expected = math.sqrt(math.erf(math.exp(-math.tanh(0.25))))
        assert math.isclose(got, expected, rel_tol=1e-12)

    def test_mixed_trig_and_hyperbolic(self):
        """asin(sin(x)) and atanh(tanh(x)) should recover x for safe inputs."""
        def f(x):
            return [np.arcsin(np.sin(x[0])), np.arctanh(np.tanh(x[0]))]
        fn = jit(f, n_x=1)
        got = np.asarray(fn([0.4]))
        assert np.allclose(got, [0.4, 0.4], rtol=1e-12)

    def test_polynomial_gamma_interaction(self):
        """Gamma function composed with polynomial expressions."""
        def f(x):
            return [sp.gamma(x[0]**2 + 1.0)]
        fn = jit(f, n_x=1)
        got = float(fn([1.5]))
        expected = math.gamma(1.5**2 + 1.0)
        assert math.isclose(got, expected, rel_tol=1e-12)

    def test_erf_symmetry(self):
        """erf(-x) = -erf(x)"""
        def f(x):
            return [sp.erf(x[0]) + sp.erf(-x[0])]
        fn = jit(f, n_x=1)
        got = float(fn([0.8]))
        assert abs(got) < 1e-15

    def test_multi_output_derivatives(self):
        """Jacobian of a vector function with new ops."""
        def f(x):
            return [np.log1p(x[0]), np.sqrt(x[0] + x[1]), np.hypot(x[0], x[1])]
        j = jacobian(f, n_x=2)
        J = np.asarray(j([3.0, 4.0]))
        # Expected:
        # d/dx0 log1p(x0) = 1/(1+x0) = 1/4
        # d/dx0 sqrt(x0+x1) = 1/(2*sqrt(7))
        # d/dx0 hypot(x0,x1) = x0/5 = 3/5
        # d/dx1 log1p(x0) = 0
        # d/dx1 sqrt(x0+x1) = 1/(2*sqrt(7))
        # d/dx1 hypot(x0,x1) = x1/5 = 4/5
        expected = np.array([
            [0.25,              0.0],
            [1/(2*math.sqrt(7)), 1/(2*math.sqrt(7))],
            [0.6,               0.8],
        ])
        assert np.allclose(J, expected, rtol=1e-12)

    def test_round_trip_cbrt_pow(self):
        """cbrt(x^3) == x for positive x."""
        def f(x):
            return [np.cbrt(x[0]**3)]
        fn = jit(f, n_x=1)
        got = float(fn([2.7]))
        assert math.isclose(got, 2.7, rel_tol=1e-12)


class TestNumpyArrayInsideTrace:
    """Numpy arrays constructed *inside* the traced function.

    When ``np.array([tracer, ...])`` is called, numpy picks object dtype
    (because JitTracer is not a float) and subsequent arithmetic goes through
    each tracer's ``__add__``/``__mul__`` etc. This works for many numpy
    patterns even though the array dtype is ``object``.
    """

    def test_const_np_array_is_fine(self):
        """Constant np.array inside trace → values become compile-time constants."""
        def f(x):
            c = np.array([1.0, 2.0, 3.0])
            return [x[0] * c[0] + x[1] * c[1] + x[2] * c[2]]
        fn = jit(f, n_x=3)
        got = float(fn([1.0, 1.0, 1.0]))
        assert math.isclose(got, 6.0, rel_tol=1e-12)

    def test_np_array_of_tracers_indexing(self):
        """np.array([tracer, tracer]) → object-array that supports indexing."""
        def f(x):
            arr = np.array([x[0], x[1]])
            return [arr[0] + arr[1]]
        fn = jit(f, n_x=2)
        assert math.isclose(float(fn([3.0, 5.0])), 8.0)

    def test_np_array_of_tracers_elementwise(self):
        """arr + arr, arr * const — via Python operators on object dtype."""
        def f(x):
            a = np.array([x[0], x[1]])
            b = np.array([x[0] * 2, x[1] * 2])
            c = a + b
            return [c[0], c[1]]
        fn = jit(f, n_x=2)
        got = np.asarray(fn([3.0, 5.0]))
        assert np.allclose(got, [9.0, 15.0])

    def test_np_array_tracer_times_const_vector(self):
        """tracer_array * const_vector (element-wise)."""
        def f(x):
            a = np.array([x[0], x[1]])
            c = np.array([2.0, 3.0])
            r = a * c
            return [r[0], r[1]]
        fn = jit(f, n_x=2)
        got = np.asarray(fn([3.0, 5.0]))
        assert np.allclose(got, [6.0, 15.0])

    def test_np_sum_on_object_array(self):
        """np.sum over a np.array of tracers reduces via Python +."""
        def f(x):
            arr = np.array([x[0], x[1], x[2]])
            return [np.sum(arr)]
        fn = jit(f, n_x=3)
        assert math.isclose(float(fn([1.0, 2.0, 3.0])), 6.0)

    def test_np_dot_object_array_with_const(self):
        """np.dot(tracer_object_array, const_vector)."""
        def f(x):
            a = np.array([x[0], x[1]])
            c = np.array([2.0, 3.0])
            return [np.dot(a, c)]
        fn = jit(f, n_x=2)
        assert math.isclose(float(fn([3.0, 5.0])), 21.0)

    def test_np_array_slicing(self):
        """Slicing works on object arrays; reductions over slices too."""
        def f(x):
            a = np.array([x[0], x[1], x[2], x[3]])
            return [a[:2].sum() + a[2:].sum()]
        fn = jit(f, n_x=4)
        assert math.isclose(float(fn([1.0, 2.0, 3.0, 4.0])), 10.0)

    def test_list_comprehension_is_fine(self):
        """Preferred pattern: build a Python list comprehension, not np.array."""
        def f(x):
            return [x[i] ** 2 for i in range(3)]
        fn = jit(f, n_x=3)
        got = np.asarray(fn([1.0, 2.0, 3.0]))
        assert np.allclose(got, [1.0, 4.0, 9.0])

    def test_np_zeros_assignment_now_traces(self):
        """The imperative idiom ``buf = np.zeros(n); buf[i] = tracer`` now traces:
        numpy's array constructors are monkeypatched during tracing to produce
        graph-backed arrays, and __setitem__ records the assignment."""
        def f(x):
            buf = np.zeros(2)
            buf[0] = x[0]
            buf[1] = x[0] * 2
            return [buf[0] + buf[1]]
        got = float(jit(f, n_x=1)([3.0]))
        assert math.isclose(got, 9.0, rel_tol=1e-12)

    def test_np_zeros_like_and_full_trace(self):
        """zeros_like (via __array_function__) and np.full (monkeypatched)."""
        def f(x):
            a = np.zeros_like(x)
            a[0] = x[0] * 3
            b = np.full(1, 2.0)
            return [a[0] + b[0]]
        got = float(jit(f, n_x=1)([4.0]))
        assert math.isclose(got, 14.0, rel_tol=1e-12)  # 12 + 2


class TestNumpyArrayFunctions:
    """np.stack / np.concatenate / np.hstack / np.vstack / np.asarray / np.array
    on sequences of tracers, via the __array_function__ protocol."""

    def test_np_stack_scalars(self):
        def f(x):
            return np.stack([x[0], x[1], x[0] + x[1]])
        fn = jit(f, n_x=2)
        got = np.asarray(fn([2.0, 3.0]))
        assert np.allclose(got, [2.0, 3.0, 5.0])

    def test_np_concatenate_arrays(self):
        def f(x):
            return np.concatenate([np.stack([x[0], x[1]]),
                                   np.stack([x[1] * 2, x[0] - x[1]])])
        fn = jit(f, n_x=2)
        got = np.asarray(fn([2.0, 3.0]))
        assert np.allclose(got, [2.0, 3.0, 6.0, -1.0])

    def test_np_hstack_mixed(self):
        """hstack of a scalar tracer, tracer-array, and constant list."""
        def f(x):
            return np.hstack([np.stack([x[0]]), np.stack([x[1], x[2]])])
        fn = jit(f, n_x=3)
        got = np.asarray(fn([1.0, 2.0, 3.0]))
        assert np.allclose(got, [1.0, 2.0, 3.0])

    def test_np_asarray_of_tracers(self):
        def f(x):
            return np.asarray([x[0], x[1] ** 2, np.sin(x[2])])
        fn = jit(f, n_x=3)
        got = np.asarray(fn([1.0, 2.0, 0.5]))
        assert np.allclose(got, [1.0, 4.0, math.sin(0.5)])

    def test_scipy_special_erf_through_ufunc(self):
        """scipy.special.erf is a numpy ufunc — goes through __array_ufunc__."""
        def f(x):
            return [sp.erf(x[0]) + sp.erfc(x[0])]
        fn = jit(f, n_x=1)
        # erf(x) + erfc(x) = 1
        assert math.isclose(float(fn([0.7])), 1.0, rel_tol=1e-12)

    def test_not_equal_scalar_ufunc(self):
        """np.not_equal on a scalar tracer goes through __array_ufunc__ (it was
        previously missing from the scalar dispatch table and returned
        NotImplemented while working on tracer arrays)."""
        def f(x):
            return [np.not_equal(x[0], 1.0)]
        fn = jit(f, n_x=1)
        assert float(fn([2.0])) == 1.0  # 2 != 1 -> True
        assert float(fn([1.0])) == 0.0  # 1 != 1 -> False
