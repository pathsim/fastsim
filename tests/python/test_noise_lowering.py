"""Discrete (zero-order-hold) noise sources lower into the SSA like any other
block: they compile() into the fused tape and code-generate to C with the real
noise included (not the old zero-noise nominal). Runs are reproducible by seed.

Continuous-sampling mode draws fresh every solver step (not a pure function of t)
and stays on the stateful RNG path — not asserted here.
"""
import numpy as np
import pytest

import fastsim as fs
from fastsim.blocks import (
    RandomNumberGenerator, WhiteNoise, Integrator, Scope,
    SinusoidalPhaseNoiseSource, ChirpPhaseNoiseSource,
)


def _walk(make_src, dt=0.05, t_end=1.0):
    src = make_src()
    itg = Integrator(0.0)            # random walk so the model has continuous state
    sco = Scope()
    sim = fs.Simulation(
        blocks=[src, itg, sco],
        connections=[fs.Connection(src, itg), fs.Connection(itg, sco)],
        dt=dt,
    )
    return sim


@pytest.mark.parametrize("make_src", [
    lambda: RandomNumberGenerator(sampling_period=0.05, seed=42),
    lambda: WhiteNoise(sampling_period=0.05, seed=7),
    lambda: WhiteNoise(sampling_period=0.05, spectral_density=2.0, seed=7),
])
def test_discrete_noise_compiles_and_reproduces(make_src):
    sim = _walk(make_src)
    compiled = sim.compile()                       # must lower noise into the tape
    compiled.reset(); _, s1, _ = compiled.run(1.0)
    compiled.reset(); _, s2, _ = compiled.run(1.0)
    assert np.allclose(np.ravel(s1), np.ravel(s2))  # reproducible by seed
    # the random walk must actually move (noise is non-trivially present)
    assert abs(float(np.ravel(s1)[-1])) > 0.0


@pytest.mark.parametrize("make_src", [
    lambda: RandomNumberGenerator(sampling_period=0.05, seed=42),
    lambda: WhiteNoise(sampling_period=0.05, seed=7),
])
def test_discrete_noise_codegens_with_helper(make_src):
    sim = _walk(make_src)
    files = sim.to_c()
    assert isinstance(files, dict)
    blob = "\n".join(files.values())
    # the PRNG helper is emitted and actually called in the model body
    assert "fastsim_rand_uniform" in blob
    body = files.get("model.c", "")
    assert "rand_uniform(" in body, "noise must appear in the generated model body"


@pytest.mark.parametrize("make_src", [
    lambda: SinusoidalPhaseNoiseSource(frequency=2.0, amplitude=1.0,
                                       sig_white=0.1, sig_cum=0.05,
                                       sampling_period=0.01, seed=5),
    lambda: ChirpPhaseNoiseSource(amplitude=1.0, f0=1.0, BW=5.0, T=1.0,
                                  sig_white=0.1, sig_cum=0.05,
                                  sampling_period=0.01, seed=9),
])
def test_phase_noise_lowers_real_noise(make_src):
    # These used to export the zero-noise *nominal* for compile/codegen; now the
    # actual white + cumulative phase noise lowers into the SSA. The block has a
    # random-walk state, so it compiles on its own.
    src = make_src()
    sco = Scope()
    sim = fs.Simulation(blocks=[src, sco], connections=[fs.Connection(src, sco)], dt=0.01)
    compiled = sim.compile()
    compiled.reset(); _, s1, _ = compiled.run(0.5)
    compiled.reset(); _, s2, _ = compiled.run(0.5)
    assert np.allclose(np.ravel(s1), np.ravel(s2))     # reproducible by seed
    files = sim.to_c()
    # the noise is in the generated model body, not nominalised away
    assert "rand_uniform(" in files.get("model.c", "")


def test_source_output_is_the_documented_keyed_draw():
    # The block output is exactly `random_uniform(floor(t/sp) + phase)`, the same
    # public, stateless kernel the op-graph and codegen use — so the interpreted,
    # compiled and code-generated runs share one definition of the noise. With an
    # integer seed the phase is `seed % 1_000_000_007` (== seed for small seeds).
    sp, seed = 0.05, 123
    inv_sp = 1.0 / sp                       # the block's exact key arithmetic
    phase = seed % 1_000_000_007
    src = RandomNumberGenerator(sampling_period=sp, seed=seed)
    sco = Scope()
    sim = fs.Simulation(blocks=[src, sco], connections=[fs.Connection(src, sco)], dt=sp)
    sim.run(0.3)
    t, d = sco.read()
    for ti, di in zip(np.ravel(t), np.ravel(d)):
        key = float(np.floor(ti * inv_sp)) + phase
        assert abs(di - fs.random_uniform(key)) < 1e-12
