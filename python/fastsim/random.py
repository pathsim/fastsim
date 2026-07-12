"""Stateless, traceable random number generation.

Unlike ``np.random.*`` (hidden global state, untraceable, irreproducible across
runs), these draws are *pure functions of an explicit key*: ``random_normal(k)``
returns the same value every time for the same ``k``. That makes noise sources
JIT-compilable (they lower to a single SSA hash node) and bit-for-bit
reproducible. The design mirrors JAX's counter-based PRNG.

The functions are polymorphic so the *same* model code works whether it is being
traced (key is a ``JitTracer``) or run eagerly (key is a Python/NumPy scalar);
both paths use the identical splitmix64 hash, so results agree bit-for-bit.

Typical use: derive a key from time so each step draws once, e.g. for a
white-noise source sampled at step ``dt``::

    import fastsim as fs

    def noise_source(t):
        return sigma * fs.random_normal(t // dt)   # stepwise, reproducible

For independent streams, offset the key (``random_normal(t // dt + channel)``).
"""

from __future__ import annotations

import math
import struct

import numpy as np

try:
    from fastsim._fastsim import (
        JitTracer as _JitTracer,
        JitTracerArray as _JitTracerArray,
        random_uniform as _native_uniform,
        random_normal as _native_normal,
    )
except ImportError:  # native ext not built with these symbols
    _JitTracer = _JitTracerArray = None
    _native_uniform = _native_normal = None

# f64::MIN_POSITIVE — matches the Rust guard against log(0) in random_normal.
_TINY = 2.2250738585072014e-308
_M64 = (1 << 64) - 1


def _hash_uniform_scalar(key: float) -> float:
    """Pure-Python twin of the Rust ``rand_uniform`` (splitmix64 finalizer over
    the key's IEEE bits → uniform [0, 1)). Bit-identical to the traced path."""
    bits = struct.unpack("<Q", struct.pack("<d", float(key)))[0]
    z = (bits + 0x9E3779B97F4A7C15) & _M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    z ^= z >> 31
    return (z >> 11) * (1.0 / (1 << 53))


def _hash_normal_scalar(key: float) -> float:
    u1 = max(_hash_uniform_scalar(key), _TINY)
    u2 = _hash_uniform_scalar(float(key) + 0.5)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _is_tracer(x) -> bool:
    return _JitTracer is not None and isinstance(x, _JitTracer)


def _reject_tracer_array(x, fn: str) -> None:
    if _JitTracerArray is not None and isinstance(x, _JitTracerArray):
        raise TypeError(
            f"fastsim.{fn}: a traced *array* key is not supported; pass a scalar "
            f"key (e.g. derived from t) and assemble a vector by stacking draws."
        )


def random_uniform(key):
    """Stateless uniform draw in ``[0, 1)``, keyed by ``key``.

    - ``key`` a ``JitTracer`` (inside a trace): records one SSA hash node.
    - ``key`` a Python/NumPy scalar: pure-Python hash (bit-identical to the trace).
    - ``key`` an ``ndarray``: element-wise draws (eager fallback path).
    """
    if _is_tracer(key):
        return _native_uniform(key)
    _reject_tracer_array(key, "random_uniform")
    if np.isscalar(key):
        return _hash_uniform_scalar(float(key))
    arr = np.asarray(key, dtype=float)
    return np.vectorize(_hash_uniform_scalar)(arr)


def random_normal(key):
    """Stateless standard-normal draw ``N(0, 1)``, keyed by ``key``.

    Box-Muller over two decorrelated uniform draws. Polymorphic like
    :func:`random_uniform`; the traced and eager paths agree bit-for-bit.
    """
    if _is_tracer(key):
        return _native_normal(key)
    _reject_tracer_array(key, "random_normal")
    if np.isscalar(key):
        return _hash_normal_scalar(float(key))
    arr = np.asarray(key, dtype=float)
    return np.vectorize(_hash_normal_scalar)(arr)
