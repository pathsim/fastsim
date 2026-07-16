"""Tracer-coverage corpus: the measuring stick for JIT lowering of user numpy code.

Every entry is a real-world RHS idiom ``f(x, t)`` traced eagerly through
``fastsim.jit.jit(fn, n_x=...)`` and classified:

- ``TRACED``     must trace AND match the eager numpy evaluation at sample
                 points (value parity, tight tolerance).
- ``GAP``        known coverage gap: the trace MUST fail today. When the gap
                 is closed, the test fails with "gap closed" and the entry is
                 flipped to ``TRACED`` — progress is explicit in the diff.
- ``STRUCTURAL`` can never trace (Python semantics force a concrete value:
                 bare ``if``, ``float()``, builtin ``min``, ``math.*``).
                 Documented so the boundary is pinned by a test.
- ``MISCOMPILE`` traces but produces values that DIVERGE from eager Python
                 (known bug, documented until fixed; flipping to TRACED is
                 the fix's acceptance criterion).

If a GAP entry suddenly traces but produces wrong values, the test fails
loudly with "SILENT MISCOMPILE" — that is the worst outcome and must never
slip through.
"""

import math

import numpy as np
import pytest

from fastsim.jit import jit

try:
    import scipy.special as _sps
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

TRACED = "traced"
GAP = "gap"
STRUCTURAL = "structural"
MISCOMPILE = "miscompile"

# Deterministic sample points, chosen away from discontinuities (floor/sign/
# round boundaries) so value parity is well-defined for every entry.
SAMPLE_T = [0.7, 2.3]


def _sample_x(n, shift=0.0):
    base = np.array([0.6, -1.3, 2.1, 0.45, -0.8, 1.7, 0.9, -2.2])
    return base[:n] + shift


def _case(name, n_x, status, fn):
    return pytest.param(fn, n_x, status, id=name)


# =============================================================================
# Corpus
# =============================================================================

CORPUS = [
    # ---------------- arithmetic / scalar ops (baseline) ----------------
    _case("arith_mix", 3, TRACED,
          lambda x, t: 2.0 * x[0] + x[1] / 3.0 - x[0] * x[2] + x[1] ** 2),
    _case("pow_int", 2, TRACED, lambda x, t: x[0] ** 3 + x[1] ** 4),
    _case("pow_frac", 1, TRACED, lambda x, t: abs(x[0]) ** 0.5),
    _case("floordiv", 1, TRACED, lambda x, t: x[0] // 0.45),
    _case("fmod_positive", 1, TRACED, lambda x, t: np.fmod(abs(x[0]) + 1.0, 0.7)),
    _case("neg_pos_abs", 2, TRACED, lambda x, t: -x[0] + (+x[1]) + abs(x[1])),
    _case("time_use", 1, TRACED, lambda x, t: x[0] * np.sin(t) + t ** 2),
    # Python `%` / np.remainder are FLOORED modulo (sign of divisor); lowered
    # as a composite over fmod (was a documented MISCOMPILE before that).
    _case("pymod_negative", 1, TRACED, lambda x, t: (x[0] - 5.0) % 0.7),
    _case("pymod_negative_divisor", 1, TRACED, lambda x, t: (x[0] + 5.0) % -0.7),
    _case("rmod_scalar", 1, TRACED, lambda x, t: 5.3 % (x[0] + 3.0)),
    _case("np_remainder_negative", 1, TRACED,
          lambda x, t: np.remainder(x[0] - 5.0, 0.7)),
    _case("np_fmod_negative", 1, TRACED, lambda x, t: np.fmod(x[0] - 5.0, 0.7)),
    _case("mod_array", 3, TRACED, lambda x, t: (x - 5.0) % 0.7),

    # ---------------- comparisons / branching ----------------
    _case("where_scalar", 2, TRACED,
          lambda x, t: np.where(x[0] > x[1], x[0], x[1])),
    _case("where_array", 3, TRACED,
          lambda x, t: np.where(x > 0.0, x, 0.1 * x)),
    _case("clip_scalar", 1, TRACED, lambda x, t: np.clip(x[0], -1.0, 1.0)),
    _case("clip_array", 3, TRACED, lambda x, t: np.clip(x, -1.0, 1.0)),
    _case("cmp_arith", 2, TRACED,
          lambda x, t: (x[0] > 0.0) * x[1] + (x[0] <= 0.0) * 2.0),
    _case("nested_where", 2, TRACED,
          lambda x, t: np.where(x[0] > 1.0, 1.0, np.where(x[0] < -1.0, -1.0, x[0]))),

    # ---------------- unary ufuncs ----------------
    _case("trig", 3, TRACED, lambda x, t: np.sin(x) + np.cos(x) * np.tan(x[0])),
    _case("hyperbolic", 3, TRACED, lambda x, t: np.sinh(x) + np.cosh(x) - np.tanh(x)),
    _case("exp_log", 2, TRACED,
          lambda x, t: np.exp(x[0]) + np.log(np.abs(x[1]) + 1.0)),
    _case("log_variants", 1, TRACED,
          lambda x, t: np.log10(abs(x[0]) + 1.0) + np.log2(abs(x[0]) + 1.0)
                       + np.log1p(abs(x[0])) + np.expm1(x[0])),
    _case("roots", 2, TRACED, lambda x, t: np.sqrt(np.abs(x[0])) + np.cbrt(x[1])),
    _case("inverse_trig", 1, TRACED,
          lambda x, t: np.arcsin(np.tanh(x[0])) + np.arccos(np.tanh(x[0]))
                       + np.arctan(x[0])),
    _case("rounding_safe", 1, TRACED,
          lambda x, t: np.floor(x[0] + 0.001) + np.ceil(x[0] + 0.001)
                       + np.trunc(x[0] + 0.001) + np.sign(x[0])),
    _case("composite_ufuncs", 2, TRACED,
          lambda x, t: np.square(x[0]) + np.deg2rad(x[1]) + np.rad2deg(x[0])
                       + np.reciprocal(x[1] + 3.0)),

    # ---------------- binary ufuncs ----------------
    _case("min_max", 2, TRACED,
          lambda x, t: np.minimum(x[0], x[1]) + np.maximum(x[0], 0.0)),
    _case("atan2_hypot", 2, TRACED,
          lambda x, t: np.arctan2(x[0], x[1] + 3.0) + np.hypot(x[0], x[1])),

    # ---------------- reductions ----------------
    _case("np_sum", 4, TRACED, lambda x, t: np.sum(x)),
    _case("np_prod", 3, TRACED, lambda x, t: np.prod(x)),
    _case("np_min_max_fn", 4, TRACED, lambda x, t: np.min(x) + np.max(x)),
    _case("np_mean", 4, TRACED, lambda x, t: np.mean(x)),
    _case("np_var_std", 4, TRACED, lambda x, t: np.var(x) + np.std(x)),
    _case("sum_axis", 4, TRACED,
          lambda x, t: np.sum(x.reshape(2, 2), axis=0)),
    _case("sum_keepdims", 4, TRACED,
          lambda x, t: np.sum(x.reshape(2, 2), axis=1, keepdims=True) * 2.0),
    _case("builtin_sum_iter", 4, TRACED, lambda x, t: sum(xi for xi in x)),
    _case("builtin_sum_zip", 3, TRACED,
          lambda x, t: sum(a * xi for a, xi in zip([1.0, 2.0, 3.0], x))),

    # ---------------- linear algebra ----------------
    _case("matvec_const", 3, TRACED,
          lambda x, t: np.array([[1.0, 2.0, 0.0],
                                 [0.0, 1.0, -1.0],
                                 [0.5, 0.0, 1.0]]) @ x),
    _case("vecmat_const", 2, TRACED,
          lambda x, t: x @ np.array([[1.0, 2.0], [3.0, 4.0]])),
    _case("dot_const", 3, TRACED, lambda x, t: np.dot(x, [1.0, -2.0, 0.5])),
    _case("dot_symbolic", 4, TRACED, lambda x, t: np.dot(x[:2], x[2:])),
    _case("matmul_symbolic", 4, TRACED,
          lambda x, t: x.reshape(2, 2) @ x[:2]),
    _case("norm_l2", 3, TRACED, lambda x, t: np.linalg.norm(x)),
    _case("norm_l1_inf", 3, TRACED,
          lambda x, t: np.linalg.norm(x, 1) + np.linalg.norm(x, np.inf)),
    _case("cross_3d", 3, TRACED, lambda x, t: np.cross(x, [0.0, 1.0, -1.0])),
    _case("vdot_inner", 4, TRACED,
          lambda x, t: np.vdot(x[:2], x[2:]) + np.inner(x[:2], x[2:])),
    _case("eye_matvec", 2, TRACED, lambda x, t: np.eye(2) @ x),

    # ---------------- array construction / assembly ----------------
    _case("zeros_imperative", 3, TRACED,
          lambda x, t: _zeros_imperative(x)),
    _case("zeros_slice_assign", 4, TRACED,
          lambda x, t: _zeros_slice_assign(x)),
    _case("ones_full", 2, TRACED,
          lambda x, t: np.ones(2) * x[0] + np.full(2, 3.0) * x[1]),
    _case("concatenate_1d", 2, TRACED, lambda x, t: np.concatenate([x, x * 2.0])),
    _case("stack_default", 2, TRACED, lambda x, t: np.stack([x, x * 2.0])),
    _case("hstack", 2, TRACED, lambda x, t: np.hstack([x, [1.0, 2.0]])),
    _case("np_array_of_tracers", 2, TRACED,
          lambda x, t: np.array([x[0] * 2.0, x[1] + 1.0])),
    _case("list_output", 3, TRACED, lambda x, t: [xi * 2.0 for xi in x]),
    _case("zeros_like_fill", 3, TRACED,
          lambda x, t: np.zeros_like(x) + x[0]),

    # ---------------- indexing / views ----------------
    _case("index_negative", 3, TRACED, lambda x, t: x[-1] + x[0]),
    _case("slice_basic", 4, TRACED, lambda x, t: x[1:3] * 2.0),
    _case("slice_step", 4, TRACED, lambda x, t: x[::2] + x[1::2]),
    _case("index_2d", 4, TRACED, lambda x, t: x.reshape(2, 2)[0, 1]),
    _case("slice_2d_col", 4, TRACED, lambda x, t: x.reshape(2, 2)[:, 1]),
    _case("reshape_T", 4, TRACED, lambda x, t: (x.reshape(2, 2).T @ x[:2])),
    _case("np_view_fns", 4, TRACED,
          lambda x, t: np.transpose(np.reshape(x, (2, 2))) @ x[2:]),
    _case("shape_attrs", 3, TRACED,
          lambda x, t: x[0] * x.shape[0] + x[1] * len(x) + x[2] * x.size),
    _case("iter_unpack", 2, TRACED, lambda x, t: _iter_unpack(x)),

    # ---------------- sequence ops ----------------
    _case("np_diff", 4, TRACED, lambda x, t: np.diff(x)),
    _case("np_cumsum", 4, TRACED, lambda x, t: np.cumsum(x)),

    # ---------------- closures / helper functions ----------------
    _case("closure_const", 2, TRACED, lambda x, t: _CLOSURE_GAIN * x[0] + x[1]),
    _case("helper_fn", 2, TRACED, lambda x, t: _helper(x) * 2.0),

    # ---------------- array METHOD forms (x.sum(), x.dot(v), ...) ----------------
    _case("method_sum", 3, TRACED, lambda x, t: x.sum()),
    _case("method_mean", 3, TRACED, lambda x, t: x.mean()),
    _case("method_min_max", 3, TRACED, lambda x, t: x.min() + x.max()),
    _case("method_prod", 3, TRACED, lambda x, t: x.prod()),
    _case("method_dot", 4, TRACED, lambda x, t: x[:2].dot(x[2:])),
    _case("method_dot_const", 3, TRACED, lambda x, t: x.dot([1.0, -2.0, 0.5])),
    _case("method_flatten", 4, TRACED, lambda x, t: x.reshape(2, 2).flatten()),
    _case("method_ravel", 4, TRACED, lambda x, t: x.reshape(2, 2).ravel()),
    _case("method_clip", 3, TRACED, lambda x, t: x.clip(-1.0, 1.0)),
    _case("method_copy", 3, TRACED, lambda x, t: x.copy() * 2.0),
    _case("method_sum_axis", 4, TRACED, lambda x, t: x.reshape(2, 2).sum(axis=0)),
    _case("method_mean_keepdims", 4, TRACED,
          lambda x, t: x.reshape(2, 2).mean(axis=1, keepdims=True) * 2.0),

    # ---------------- extended ufuncs ----------------
    _case("np_radians_degrees", 2, TRACED,
          lambda x, t: np.radians(x[0]) + np.degrees(x[1])),
    _case("np_fmin_fmax", 2, TRACED,
          lambda x, t: np.fmin(x[0], x[1]) + np.fmax(x[0], 0.0)),
    _case("np_exp2", 1, TRACED, lambda x, t: np.exp2(x[0])),
    _case("np_copysign", 2, TRACED, lambda x, t: np.copysign(x[0], x[1])),
    _case("np_logaddexp", 2, TRACED, lambda x, t: np.logaddexp(x[0], x[1])),
    _case("np_heaviside", 1, TRACED, lambda x, t: np.heaviside(x[0], 0.5)),
    _case("np_float_power", 1, TRACED,
          lambda x, t: np.float_power(np.abs(x[0]) + 1.0, 2.5)),
    _case("np_copysign_array", 3, TRACED, lambda x, t: np.copysign(x, -1.0)),
    _case("np_exp2_array", 3, TRACED, lambda x, t: np.exp2(x)),

    # ---------------- extended array functions ----------------
    _case("np_interp", 1, TRACED,
          lambda x, t: np.interp(x[0], [-2.0, 0.0, 2.0], [0.0, 1.0, 0.0])),
    _case("np_interp_array", 3, TRACED,
          lambda x, t: np.interp(x, [-2.0, 0.0, 2.0], [0.0, 1.0, 0.0])),
    _case("np_interp_clamp", 1, TRACED,
          lambda x, t: np.interp(x[0] + 10.0, [-2.0, 0.0, 2.0], [0.0, 1.0, 0.5])),
    _case("np_full_like", 3, TRACED, lambda x, t: np.full_like(x, 2.0) * x[0]),
    _case("np_flip", 3, TRACED, lambda x, t: np.flip(x) + x),
    _case("np_flip_axis", 4, TRACED,
          lambda x, t: np.flip(x.reshape(2, 2), 0) @ x[:2]),
    _case("np_roll", 3, TRACED, lambda x, t: np.roll(x, 1) + x),
    _case("np_roll_negative", 3, TRACED, lambda x, t: np.roll(x, -2) + x),
    _case("np_cumprod", 3, TRACED, lambda x, t: np.cumprod(x)),
    _case("np_outer", 2, TRACED, lambda x, t: np.outer(x, x).reshape(-1)),
    _case("np_atleast_1d", 1, TRACED, lambda x, t: np.atleast_1d(x[0]) * 2.0),

    # ---------------- constant factories as assignment targets ----------------
    _case("arange_assign", 1, TRACED, lambda x, t: _arange_assign(x)),
    _case("linspace_assign", 1, TRACED, lambda x, t: _linspace_assign(x)),
    _case("arange_arith", 3, TRACED, lambda x, t: np.arange(3.0) * x),
    _case("diag_of_tracer", 2, TRACED, lambda x, t: np.diag(x) @ x),
    _case("diag_const_matvec", 2, TRACED, lambda x, t: np.diag([2.0, 3.0]) @ x),

    # ---------------- mixed scalar-tracer dispatch (fuzzer finding, fixed) ----
    _case("mixed_ufunc_scalar_array", 3, TRACED, lambda x, t: np.minimum(x[0], x)),
    _case("mixed_ufunc_scalar_ndarray", 3, TRACED,
          lambda x, t: np.minimum(x[0], np.array([1.0, 2.0, 3.0]))),
    _case("where_scalar_cond_array", 3, TRACED,
          lambda x, t: np.where(x[0] > 0.0, x, -x)),
    _case("where_const_cond_tracer_branches", 2, TRACED,
          lambda x, t: np.where(np.pi > 3.0, x[0], x[1])),

    # ---------------- column matrices (fuzzer finding, fixed) ----------------
    _case("matmul_col_matrix", 1, TRACED,
          lambda x, t: np.array([[1.0], [2.0]]) @ x),
    _case("matmul_1x1_nested", 1, TRACED,
          lambda x, t: np.array([[3.0]]) @ (np.array([[2.0]]) @ x)),

    # ---------------- extended indexing ----------------
    _case("fancy_index_list", 3, TRACED, lambda x, t: x[[0, 2]]),
    _case("fancy_index_negative", 3, TRACED, lambda x, t: x[[-1, 0]]),
    _case("fancy_index_nparray", 3, TRACED, lambda x, t: x[np.array([2, 1])]),
    _case("negative_step", 3, TRACED, lambda x, t: x[::-1]),
    _case("negative_step_partial", 4, TRACED, lambda x, t: x[3:0:-2]),
    _case("ellipsis_index", 4, TRACED, lambda x, t: x.reshape(2, 2)[..., 0]),
    _case("ellipsis_leading", 4, TRACED, lambda x, t: x.reshape(2, 2)[0, ...]),
    _case("newaxis_index", 2, TRACED,
          lambda x, t: (x[:, None] * x[None, :]).reshape(-1)),
    _case("setitem_negative_step", 3, TRACED, lambda x, t: _setitem_reverse(x)),

    # ---------------- scipy.special composites ----------------
    _case("scipy_expit", 1, TRACED if HAS_SCIPY else STRUCTURAL,
          (lambda x, t: _sps.expit(x[0])) if HAS_SCIPY else (lambda x, t: x[0])),

    # ---------------- former GAPs, closed by the tracer-review fixes --------
    _case("np_argmax", 3, TRACED, lambda x, t: np.argmax(x) * 1.0),
    _case("np_argmin", 3, TRACED, lambda x, t: np.argmin(x) * 1.0),
    _case("method_argmax_argmin", 4, TRACED,
          lambda x, t: x.argmax() * 10.0 + x.argmin()),
    _case("np_linalg_solve_2x2", 2, TRACED,
          lambda x, t: np.linalg.solve(np.array([[2.0, 1.0], [1.0, 3.0]]), x)),
    _case("np_linalg_solve_3x3", 3, TRACED,
          lambda x, t: np.linalg.solve(
              np.array([[4.0, 1.0, 0.0], [1.0, 3.0, -1.0], [0.0, -1.0, 2.0]]), x)),
    _case("np_select", 3, TRACED,
          lambda x, t: np.select([x > 1.0, x > 0.0], [x, 0.5 * x], default=-1.0)),
    _case("np_searchsorted_left", 1, TRACED,
          lambda x, t: np.searchsorted([-2.0, 0.0, 1.5, 3.0], x[0]) * 1.0),
    _case("np_searchsorted_right", 1, TRACED,
          lambda x, t: np.searchsorted([-2.0, 0.0, 1.5, 3.0], x[0], side="right") * 1.0),
    _case("np_searchsorted_array", 3, TRACED,
          lambda x, t: np.searchsorted([-2.0, 0.0, 1.5, 3.0], x) * 1.0),
    _case("method_var_std_ddof", 4, TRACED,
          lambda x, t: x.var(ddof=1) + x.std(ddof=1)),
    _case("method_cumsum_cumprod", 3, TRACED,
          lambda x, t: x.cumsum() + x.cumprod()),

    # -- bool masks stay structural: the output SHAPE depends on runtime values --
    _case("bool_mask_getitem", 3, STRUCTURAL, lambda x, t: x[np.array([True, False, True])]),

    # =========================================================================
    # Fixed miscompiles (doc/tracer-review.md P0) — pinned as TRACED so a
    # regression reintroduces a loud failure, not a silent divergence.
    # =========================================================================
    # P0-1 (fixed): var/std support ddof; other kwargs reject → fallback.
    _case("var_ddof", 4, TRACED, lambda x, t: np.var(x, ddof=1)),
    _case("std_ddof", 4, TRACED, lambda x, t: np.std(x, ddof=1)),
    # P0-2 (fixed): np.diff supports integer n (positional and kwarg).
    _case("diff_order2", 4, TRACED, lambda x, t: np.diff(x, 2)),
    _case("diff_order_kwarg", 4, TRACED, lambda x, t: np.diff(x, n=2)),
    # P0-3 (fixed): unknown reduction kwargs now REJECT the trace (fail-open
    # fallback) instead of being silently ignored.
    _case("sum_where_rejected", 4, GAP,
          lambda x, t: np.sum(x, where=np.array([True, True, False, False]))),
    _case("sum_initial_rejected", 4, GAP,
          lambda x, t: np.sum(x, initial=10.0)),
    # P0-5 (fixed): hstack/vstack of 2-D inputs concatenate along the correct
    # axis with the correct output shape.
    _case("hstack_2d", 4, TRACED,
          lambda x, t: np.hstack([x.reshape(2, 2), x.reshape(2, 2)]).ravel()),
    _case("vstack_2d", 4, TRACED,
          lambda x, t: (np.vstack([x.reshape(2, 2), x.reshape(2, 2)]) @ x[:2])),
    _case("vstack_1d_shape", 3, TRACED,
          lambda x, t: (np.vstack([x, 2.0 * x]) @ x)),
    # P1-7/8 (fixed): np.where and np.clip keep N-D shapes through the trace.
    _case("where_2d_shape", 4, TRACED,
          lambda x, t: np.where(x.reshape(2, 2) > 0, x.reshape(2, 2), 0.0) @ x[:2]),
    _case("clip_2d_shape", 4, TRACED,
          lambda x, t: np.clip(x.reshape(2, 2), -1.0, 1.0) @ x[:2]),

    # =========================================================================
    # STRUCTURAL — never traceable (Python forces a concrete value).
    # =========================================================================
    _case("bare_if", 1, STRUCTURAL, lambda x, t: _bare_if(x)),
    _case("builtin_min", 2, STRUCTURAL, lambda x, t: min(x[0], x[1])),
    _case("math_module", 1, STRUCTURAL, lambda x, t: math.sin(x[0])),
    _case("float_conversion", 1, STRUCTURAL, lambda x, t: float(x[0]) * 2.0),
    _case("bool_mask_index", 3, STRUCTURAL, lambda x, t: np.sum(x[x > 0.0])),
]


_CLOSURE_GAIN = 2.5


def _helper(x):
    return x[0] * x[1] + np.sin(x[0])


def _zeros_imperative(x):
    dx = np.zeros(3)
    dx[0] = x[1] * 2.0
    dx[1] = -x[0] + x[2]
    dx[2] = x[0] * x[1]
    return dx


def _zeros_slice_assign(x):
    dx = np.zeros(4)
    dx[1:] = x[:3] * 2.0
    dx[0] = x[3]
    return dx


def _iter_unpack(x):
    a, b = x
    return a * b


def _arange_assign(x):
    a = np.arange(3.0)
    a[0] = x[0]
    return a


def _linspace_assign(x):
    a = np.linspace(0.0, 1.0, 3)
    a[1] = x[0] * 2.0
    return a


def _setitem_reverse(x):
    dx = np.zeros(3)
    dx[::-1] = x * 2.0
    return dx


def _bare_if(x):
    if x[0] > 0:
        return [x[0]]
    return [-x[0]]


# =============================================================================
# Harness
# =============================================================================


def _eval_points(fn, n_x):
    """Reference (eager numpy) outputs at the sample points, flattened."""
    outs = []
    for i, t in enumerate(SAMPLE_T):
        x = _sample_x(n_x, shift=0.1 * i)
        outs.append(np.asarray(fn(x, t), dtype=float).ravel())
    return outs


def _trace(fn, n_x):
    """Eager-trace `fn`; returns the compiled JitFunction or raises."""
    return jit(fn, n_x=n_x)


def _parity(compiled, fn, n_x, rtol=1e-12, atol=1e-12):
    """Compare compiled vs eager numpy at the sample points."""
    for i, t in enumerate(SAMPLE_T):
        x = _sample_x(n_x, shift=0.1 * i)
        ref = np.asarray(fn(x.copy(), t), dtype=float).ravel()
        got = np.atleast_1d(np.asarray(compiled(x, t), dtype=float)).ravel()
        if got.shape != ref.shape:
            # An arity mismatch IS a divergence (e.g. np.diff's ignored `n`):
            # report it like a value mismatch so MISCOMPILE entries can pin it.
            return False, ref, got
        ok = np.allclose(got, ref, rtol=rtol, atol=atol)
        if not ok:
            return False, ref, got
    return True, None, None


def test_dot_size_mismatch_raises():
    """Fixed (doc/tracer-review.md P0-4): np.dot/np.inner with mismatched
    operand sizes raise during the trace, exactly like eager numpy — instead
    of silently contracting over min(len)."""
    for fn in (
        lambda x, t: np.dot(x[:3], [1.0, 2.0]),
        lambda x, t: np.inner(x[:3], [1.0, 2.0]),
    ):
        with pytest.raises(ValueError):
            fn(_sample_x(4), 0.7)  # eager numpy rejects the shapes
        with pytest.raises(ValueError):
            jit(fn, n_x=4)  # and so does the trace


@pytest.mark.parametrize("fn,n_x,status", CORPUS)
def test_corpus(fn, n_x, status):
    if status == TRACED:
        compiled = _trace(fn, n_x)  # must not raise
        ok, ref, got = _parity(compiled, fn, n_x)
        assert ok, f"value mismatch: eager {ref} vs traced {got}"
    elif status == MISCOMPILE:
        # Documents a known traced-vs-eager divergence. Fixing it makes this
        # entry fail with "miscompile fixed" — then flip it to TRACED.
        compiled = _trace(fn, n_x)
        ok, _, _ = _parity(compiled, fn, n_x)
        assert not ok, "miscompile fixed — flip this corpus entry to TRACED"
    else:  # GAP / STRUCTURAL: the trace must fail (loudly, not silently).
        try:
            compiled = _trace(fn, n_x)
        except Exception:
            return  # expected: no trace
        # It traced — either the gap was closed (flip the entry) or we have
        # a silent miscompile (worst case: wrong values, no error).
        ok, ref, got = _parity(compiled, fn, n_x)
        if ok:
            pytest.fail(
                f"gap closed — flip this corpus entry to TRACED ({status})")
        pytest.fail(
            f"SILENT MISCOMPILE: traces but diverges (eager {ref} vs traced {got})")
