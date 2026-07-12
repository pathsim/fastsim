"""Traceable stateless RNG: fastsim.random_uniform / random_normal.

The defining property is that the *same* call is a pure function of its key, so
the traced (JIT) path and the eager (pure-Python) path agree bit-for-bit, and a
compiled noise source replays identically across runs.
"""
import numpy as np
import pytest

import fastsim as fs
from fastsim.jit import jit


def test_uniform_scalar_range_and_determinism():
    for k in (0.0, 1.0, 3.0, -7.5, 1000.25):
        u = fs.random_uniform(k)
        assert 0.0 <= u < 1.0
        assert u == fs.random_uniform(k)  # deterministic


def test_traced_equals_eager_bitexact():
    def f(x):
        return [fs.random_uniform(x[0]), fs.random_normal(x[0])]
    g = jit(f)
    for k in (0.0, 1.0, 3.0, 42.0, -7.5, 1000.25):
        tu, tn = g([k])
        assert tu == fs.random_uniform(k)
        assert tn == fs.random_normal(k)


def test_normal_is_standard():
    xs = np.array([fs.random_normal(i) for i in range(20000)])
    assert abs(xs.mean()) < 0.05
    assert abs(xs.std() - 1.0) < 0.05


def test_array_fallback_eager():
    arr = fs.random_uniform(np.arange(5.0))
    assert arr.shape == (5,)
    assert np.all((arr >= 0.0) & (arr < 1.0))
    # element-wise parity with the scalar path
    for i in range(5):
        assert arr[i] == fs.random_uniform(float(i))


def test_floordiv_traces():
    # `t // dt` is the canonical stepwise-key idiom and must trace.
    g = jit(lambda x: [x[0] // 0.01])
    got = float(np.ravel(g([0.035]))[0])
    assert got == np.floor(0.035 / 0.01)


def test_noise_source_compiles_and_is_reproducible():
    from fastsim.blocks import Source, Integrator, Scope
    dt = 0.01
    src = Source(lambda t: fs.random_normal(t // dt))
    itg = Integrator(0.0)          # random walk: dx = noise
    sco = Scope()
    sim = fs.Simulation(
        blocks=[src, itg, sco],
        connections=[fs.Connection(src, itg), fs.Connection(itg, sco)],
        dt=dt,
    )
    compiled = sim.compile()       # noise source must lower into the fused tape
    compiled.reset(); _, s1, _ = compiled.run(0.5)
    compiled.reset(); _, s2, _ = compiled.run(0.5)
    assert np.allclose(np.ravel(s1), np.ravel(s2))


def test_traced_array_key_rejected_clearly():
    # A traced *array* key is explicitly unsupported (scalar keys only for now).
    def f(u):
        return [fs.random_uniform(u)]  # u is a JitTracerArray during trace
    with pytest.raises(Exception):
        jit(f, n_x=2)([1.0, 2.0])
