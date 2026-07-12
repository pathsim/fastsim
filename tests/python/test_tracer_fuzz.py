"""Differential fuzzer for the JIT tracer: random numpy expressions, traced
vs eagerly evaluated, compared at sample points.

This complements the Rust-side graph fuzzers (`fuzz_tape_matches_interpret_
bit_exact`, `fuzz_optimize_preserves_semantics`, `test_diff_fuzz_vs_finite_
diff`): those start from random *graphs*, so they never exercise the Python
operator-overload surface (dunder dispatch, `__array_ufunc__` /
`__array_function__`, broadcasting, reductions, slicing). This fuzzer starts
from random *Python expressions* over the supported op vocabulary, so a
regression anywhere in the tracer's lowering shows up as a value divergence.

Generator design notes:
- Fully deterministic (`random.Random(seed)`), no time- or hash-order
  dependence.
- Only continuous, domain-safe compositions: `exp` args are bounded through
  `tanh`, `log`/`sqrt` args are shifted positive, divisors are bounded away
  from zero. Discontinuous ops (floor/ceil/round/sign/trunc, `==`/`!=`) are
  deliberately excluded here — a last-ULP difference between Rust libm and
  numpy could flip them and produce spurious mismatches. The corpus
  (`test_tracer_corpus.py`) covers those at hand-picked safe points instead.
- `where(a > b, c, d)` IS included: `>` on continuous random operands is
  ULP-stable enough in practice, and select lowering is exactly the kind of
  thing this fuzzer must cover. If a seed ever lands on a knife-edge, the
  seed is deterministic — adjust the generator, don't loosen the tolerance.
"""

import random

import numpy as np
import pytest

from fastsim.jit import jit

N_SEEDS = 300
RTOL = 1e-9
ATOL = 1e-11


# =============================================================================
# Expression tree: closures over (x, t) built recursively
# =============================================================================
# Each node is a tuple (kind, fn) where fn(x, t) evaluates the subtree with
# plain numpy semantics — the SAME callable is traced and eagerly evaluated,
# so any divergence is the tracer's fault, not the generator's.
# `kind` is "scalar" or "array" (1-D of size n).


def _gen(rng, n, depth, kind):
    """Generate a subtree of the requested kind ("scalar"/"array")."""
    if depth <= 0:
        return _leaf(rng, n, kind)
    roll = rng.random()
    if kind == "scalar":
        if roll < 0.15:
            return _leaf(rng, n, kind)
        if roll < 0.30:
            # Reduction: array subtree -> scalar. Function and METHOD forms
            # both lower through the same dispatch — fuzz both spellings.
            sub = _gen(rng, n, depth - 1, "array")
            red = rng.choice([
                np.sum, np.mean, np.prod, np.min, np.max,
                lambda a: a.sum(), lambda a: a.mean(), lambda a: a.prod(),
                lambda a: a.min(), lambda a: a.max(),
            ])
            return lambda x, t: red(sub(x, t))
        if roll < 0.40:
            # Dot with a constant vector (function or method form).
            sub = _gen(rng, n, depth - 1, "array")
            c = _const_vec(rng, n)
            if rng.random() < 0.5:
                return lambda x, t: np.dot(sub(x, t), c)
            return lambda x, t: sub(x, t).dot(c)
        if roll < 0.50:
            # Indexing an array subtree.
            sub = _gen(rng, n, depth - 1, "array")
            i = rng.randrange(n)
            return lambda x, t: sub(x, t)[i]
        return _compose(rng, n, depth, kind)
    else:
        if roll < 0.15:
            return _leaf(rng, n, kind)
        if roll < 0.25:
            # Constant matrix @ array subtree (n x n, bounded entries) —
            # includes the once-broken (1,1) shape (corpus: matmul_col_matrix).
            sub = _gen(rng, n, depth - 1, "array")
            m = np.array([[_const(rng) for _ in range(n)] for _ in range(n)])
            return lambda x, t: m @ sub(x, t)
        return _compose(rng, n, depth, kind)


def _compose(rng, n, depth, kind):
    """Unary / binary / where composition, kind-preserving."""
    roll = rng.random()
    if roll < 0.40:
        sub = _gen(rng, n, depth - 1, kind)
        return _unary(rng, sub)
    if roll < 0.85:
        a = _gen(rng, n, depth - 1, kind)
        # Mixing scalar into array ops exercises broadcasting.
        b_kind = kind if rng.random() < 0.7 else "scalar"
        b = _gen(rng, n, depth - 1, b_kind)
        swap = rng.random() < 0.5 and kind == "array" and b_kind == "scalar"
        return _binary(rng, a, b, swap)
    # where(a > b, c, d)
    a = _gen(rng, n, depth - 1, kind)
    b = _gen(rng, n, depth - 1, kind)
    c = _gen(rng, n, depth - 1, kind)
    d = _gen(rng, n, depth - 1, kind)
    return lambda x, t: np.where(a(x, t) > b(x, t), c(x, t), d(x, t))


def _leaf(rng, n, kind):
    if kind == "array":
        if rng.random() < 0.6:
            return lambda x, t: x
        c = _const_vec(rng, n)
        return lambda x, t: x * 0.0 + c  # constant array via broadcast
    roll = rng.random()
    if roll < 0.4:
        i = rng.randrange(n)
        return lambda x, t: x[i]
    if roll < 0.6:
        return lambda x, t: t
    c = _const(rng)
    return lambda x, t: x[0] * 0.0 + c  # scalar constant (keeps tracer in play)


def _const(rng):
    return round(rng.uniform(-2.0, 2.0), 3)


def _const_vec(rng, n):
    return np.array([_const(rng) for _ in range(n)])


def _unary(rng, sub):
    ops = [
        lambda a: np.sin(a),
        lambda a: np.cos(a),
        lambda a: np.tanh(a),
        lambda a: np.arctan(a),
        lambda a: -a,
        lambda a: np.abs(a),
        lambda a: np.exp(np.tanh(a) * 2.0),       # bounded exp
        lambda a: np.exp2(np.tanh(a) * 3.0),      # bounded exp2
        lambda a: np.log(np.abs(a) + 1.0),        # positive log
        lambda a: np.sqrt(np.abs(a)),             # safe sqrt
        lambda a: np.clip(a, -1.5, 1.5),
        lambda a: np.square(a),
        lambda a: np.radians(a),
        lambda a: np.degrees(a) * 0.01,           # keep magnitudes bounded
        # piecewise-linear, continuous: exercises the interp select chain
        lambda a: np.interp(a, [-3.0, -1.0, 0.0, 1.0, 3.0],
                            [0.0, 0.5, 1.0, 0.5, 0.0]),
    ]
    op = rng.choice(ops)
    return lambda x, t: op(sub(x, t))


def _binary(rng, a, b, swap=False):
    dunder_ops = [
        lambda u, v: u + v,
        lambda u, v: u - v,
        lambda u, v: u * v,
        lambda u, v: u / (v * v + 1.0),           # guarded division
    ]
    ufunc_ops = [
        lambda u, v: np.minimum(u, v),
        lambda u, v: np.maximum(u, v),
        lambda u, v: np.fmin(u, v),
        lambda u, v: np.fmax(u, v),
        lambda u, v: np.arctan2(u, v * v + 1.0),  # guarded atan2
        lambda u, v: np.hypot(u, v),
        lambda u, v: np.logaddexp(np.tanh(u), np.tanh(v)),  # bounded
    ]
    # Scalar-first mixed ufuncs (`swap`) included since the mixed-dispatch
    # promotion landed (corpus: mixed_ufunc_scalar_array). Discontinuous
    # binary composites (copysign, heaviside) stay out per the continuity
    # policy — the corpus pins them at safe points.
    op = rng.choice(dunder_ops + ufunc_ops)
    if swap:
        a, b = b, a
    return lambda x, t: op(a(x, t), b(x, t))


def _gen_func(rng, n):
    """A complete f(x, t): 1-3 scalar outputs or one array output."""
    if rng.random() < 0.5:
        k = rng.randrange(1, 4)
        subs = [_gen(rng, n, depth=rng.randrange(2, 5), kind="scalar")
                for _ in range(k)]
        return lambda x, t: [s(x, t) for s in subs]
    sub = _gen(rng, n, depth=rng.randrange(2, 5), kind="array")
    return lambda x, t: sub(x, t)


# =============================================================================
# Differential check
# =============================================================================


def _sample_points(rng, n, k=3):
    pts = []
    for _ in range(k):
        x = np.array([round(rng.uniform(-2.0, 2.0), 3) for _ in range(n)])
        t = round(rng.uniform(0.0, 3.0), 3)
        pts.append((x, t))
    return pts


@pytest.mark.parametrize("seed", range(N_SEEDS))
def test_fuzz_traced_matches_eager(seed):
    rng = random.Random(seed)
    n = rng.choice([1, 2, 3, 4])
    fn = _gen_func(rng, n)
    pts = _sample_points(rng, n)

    # The generator only emits supported ops — the trace itself must succeed.
    # A trace failure here is a coverage REGRESSION, not an acceptable fallback.
    compiled = jit(fn, n_x=n)

    for x, t in pts:
        ref = np.asarray(fn(x.copy(), float(t)), dtype=float).ravel()
        got = np.atleast_1d(
            np.asarray(compiled(x, float(t)), dtype=float)).ravel()
        assert got.shape == ref.shape, (
            f"seed {seed}: arity traced {got.shape} vs eager {ref.shape}")
        assert np.allclose(got, ref, rtol=RTOL, atol=ATOL, equal_nan=True), (
            f"seed {seed} at x={x}, t={t}: eager {ref} vs traced {got}")
