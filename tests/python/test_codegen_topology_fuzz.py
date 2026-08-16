"""Differential fuzzer over model TOPOLOGY: generated C vs the compiled engine.

The existing codegen tests sweep a fixed set of hand-written systems across the
option axes. Hand-written systems encode what their author thought to try — the
subsystem defects that reached users lived in a shape nobody had written down.
This generates random but *valid* models instead, so the shape itself is the
variable.

Complements the other fuzzers: `test_tracer_fuzz.py` randomises expressions
inside one block, the Rust graph fuzzers randomise op-graphs. This randomises
how blocks are WIRED, including subsystem nesting, which neither of those
reaches.

Generator design (same discipline as `test_tracer_fuzz.py`):

- Fully deterministic per seed (`random.Random(seed)`); a failure is replayable
  by its seed alone.
- Only shapes the backend is required to handle: no algebraic loops (rejected
  by design), no opaque blocks, no stochastic sources (their streams are engine
  specific).
- Numerically tame by construction — gains bounded away from zero and from
  large values, states driven by bounded sources, so a mismatch means a wiring
  or lowering defect rather than an amplified rounding difference.
- Every generated model ends in an `Integrator` feeding a `Scope`, so there is
  continuous state to compare rather than a purely algebraic snapshot.

The reference is fastsim's own compiled tape (`Simulation.compile()`), i.e. the
same IR through a different backend — so a mismatch localises to the C emission
rather than to the model.
"""
import os
import random
import subprocess
import tempfile

import numpy as np
import pytest

import fastsim as fs
from fastsim import blocks as B
from fastsim.solvers import RK4

from codegen_common import CC, gen_main_c_print, match_worst, needs_cc, reference

# Generation and the shape invariants need no compiler, so they sweep wide.
# The compile-and-run comparison is the expensive one and takes a subset.
N_SEEDS = 120
N_SEEDS_COMPILED = 24
DT, DURATION = 0.02, 1.0


# Compiler resolution and the skip marker come from `codegen_common`.
_CC = CC

# Algebraic blocks that are safe to chain arbitrarily: total, bounded, and
# defined for every real input. Excludes anything with a domain restriction
# (log/sqrt), a singularity (divide), or a discrete decision whose output can
# flip on a rounding difference.
SAFE_UNARY = [
    lambda r: B.Amplifier(gain=r.uniform(0.2, 2.0)),
    lambda r: B.Amplifier(gain=-r.uniform(0.2, 2.0)),
    lambda r: B.Tanh(),
    lambda r: B.Sin(),
    lambda r: B.Cos(),
    lambda r: B.Atan(),
    lambda r: B.Abs(),
    lambda r: B.Clip(min_val=-2.0, max_val=2.0),
    lambda r: B.Rescale(i0=-1.0, i1=1.0, o0=-0.5, o1=0.5),
    lambda r: B.PT1(K=r.uniform(0.5, 1.5), T=r.uniform(0.2, 1.0)),
]

SAFE_SOURCE = [
    lambda r: B.SinusoidalSource(frequency=r.uniform(0.5, 2.0), amplitude=r.uniform(0.5, 1.5)),
    lambda r: B.Constant(value=r.uniform(-1.0, 1.0)),
    lambda r: B.TriangleWaveSource(frequency=r.uniform(0.5, 1.5), amplitude=1.0),
]


def _chain(rng, depth):
    """A linear chain of algebraic blocks; returns (blocks, connections, last)."""
    blocks, conns = [], []
    prev = None
    for _ in range(depth):
        blk = rng.choice(SAFE_UNARY)(rng)
        blocks.append(blk)
        if prev is not None:
            conns.append(fs.Connection(prev, blk))
        prev = blk
    return blocks, conns, prev


def _subsystem(rng, depth, n_in):
    """A Subsystem with `n_in` inputs and one output, sometimes nested deeper.

    `n_in > 1` is the shape that matters most: a multi-input interface is one
    port carrying several channels, and resolving that splice per port instead
    of per element is exactly the defect that reached users. A generator that
    only ever built single-input subsystems would be blind to it.
    """
    iface = fs.Interface()
    inner_blocks, inner_conns, last = _chain(rng, depth)
    conns = list(inner_conns)

    if n_in == 1:
        conns.append(fs.Connection(iface[0], inner_blocks[0]))
        members = [iface] + inner_blocks
    else:
        # Each interface channel drives its own chain; an adder recombines them,
        # so a collapsed splice changes the result instead of cancelling out.
        add = B.Adder("+" * n_in)
        members = [iface, add] + inner_blocks
        conns.append(fs.Connection(iface[0], inner_blocks[0]))
        conns.append(fs.Connection(last, add[0]))
        for ch in range(1, n_in):
            side, side_conns, side_last = _chain(rng, rng.randint(1, 2))
            members.extend(side)
            conns.extend(side_conns)
            conns.append(fs.Connection(iface[ch], side[0]))
            conns.append(fs.Connection(side_last, add[ch]))
        last = add

    conns.append(fs.Connection(last, iface[0]))
    sub = fs.Subsystem(members, conns)

    if rng.random() < 0.35:
        # one more level of nesting around it, preserving the input count
        outer_if = fs.Interface()
        outer_conns = [fs.Connection(outer_if[ch], sub[ch]) for ch in range(n_in)]
        outer_conns.append(fs.Connection(sub, outer_if[0]))
        return fs.Subsystem([outer_if, sub], outer_conns)
    return sub


def build_model(seed):
    """A random, valid model: sources -> (chains / subsystems) -> adder ->
    integrator -> scope."""
    rng = random.Random(seed)
    n_paths = rng.randint(1, 3)
    blocks, conns, tails = [], [], []

    for _ in range(n_paths):
        src = rng.choice(SAFE_SOURCE)(rng)
        blocks.append(src)
        if rng.random() < 0.5:
            n_in = rng.choice([1, 2, 2, 3])   # weighted towards multi-input
            sub = _subsystem(rng, rng.randint(1, 3), n_in)
            blocks.append(sub)
            conns.append(fs.Connection(src, sub[0]))
            # Give the remaining interface channels their own drive, so every
            # channel carries a distinct signal and a collapsed splice shows up.
            for ch in range(1, n_in):
                extra = rng.choice(SAFE_SOURCE)(rng)
                blocks.append(extra)
                conns.append(fs.Connection(extra, sub[ch]))
            tails.append(sub)
        else:
            chain, chain_conns, last = _chain(rng, rng.randint(1, 3))
            blocks.extend(chain)
            conns.extend(chain_conns)
            conns.append(fs.Connection(src, chain[0]))
            tails.append(last)

    if len(tails) == 1:
        head = tails[0]
    else:
        add = B.Adder("+" * len(tails))
        blocks.append(add)
        for i, tail in enumerate(tails):
            conns.append(fs.Connection(tail, add[i]))
        head = add

    integ, sco = B.Integrator(0.0), B.Scope()
    blocks += [integ, sco]
    conns += [fs.Connection(head, integ), fs.Connection(integ, sco)]
    return fs.Simulation(blocks, conns, dt=DT, log=False)


def _compile_and_run(files, tmp):
    """Build the emitted C with a harness and return its trajectory rows."""
    for name, src in files.items():
        with open(os.path.join(tmp, name), "w") as fh:
            fh.write(src)
    with open(os.path.join(tmp, "main.c"), "w") as fh:
        fh.write(gen_main_c_print(files["model.h"], DURATION, DT))
    exe = os.path.join(tmp, "a.out")
    cfiles = [os.path.join(tmp, n) for n in files if n.endswith(".c")]
    cfiles.append(os.path.join(tmp, "main.c"))
    cc = (_CC or "cc").split()
    proc = subprocess.run([*cc, "-O2", "-ffp-contract=off", "-o", exe, *cfiles,
                           "-lm", f"-I{tmp}"], capture_output=True, text=True)
    assert proc.returncode == 0, f"generated C did not compile:\n{proc.stderr}"
    out = subprocess.run([exe], check=True, capture_output=True, text=True).stdout
    return np.asarray([[float(v) for v in ln.split()] for ln in out.strip().splitlines()])


@needs_cc
@pytest.mark.parametrize("seed", range(N_SEEDS_COMPILED))
def test_generated_c_matches_the_compiled_reference(seed, tmp_path):
    """The emitted C, compiled and run, reproduces the compiled tape.

    The reference is the same IR through fastsim's other backend, so a mismatch
    localises to the C emission rather than to the model.
    """
    files = build_model(seed).to_c(
        name="model", numeric="double", reductions="unrolled",
        structure="hierarchical", layout="compact", api="struct", solver="rk4")
    assert files, f"seed {seed}: produced no files"

    with tempfile.TemporaryDirectory(dir=tmp_path) as tmp:
        rows = _compile_and_run(files, tmp)

    _, x_ref = reference(build_model(seed), DURATION, DT, RK4)
    worst, _ = match_worst(rows[:, 1:], x_ref, "double")
    assert worst <= 1.0, (
        f"seed {seed}: generated C diverges from the compiled reference, "
        f"worst scaled error {worst:.3g}. Replay with build_model({seed}).")


def test_corpus_actually_contains_subsystems():
    """The generator must produce the shape it exists for.

    A fuzzer that never builds a multi-input subsystem would be blind to the
    splice defect that motivated it, and nothing else in the suite would say so.
    Checked on the emitted C, where the backend qualifies every block with its
    subsystem path (``root/sub0/Amplifier``) — that name only appears if a
    subsystem was actually flattened.
    """
    with_sub = 0
    depths = set()
    for seed in range(N_SEEDS):
        src = build_model(seed).to_c(
            name="model", numeric="double", reductions="unrolled",
            structure="hierarchical", layout="compact", api="struct",
            solver="rk4")["model.c"]
        paths = [ln for ln in src.splitlines() if "root/sub" in ln]
        if paths:
            with_sub += 1
            depths.update(ln.count("/sub") for ln in paths)

    assert with_sub >= N_SEEDS // 4, (
        f"only {with_sub}/{N_SEEDS} generated models contain a subsystem — the "
        f"generator has degenerated away from the shape it exists to cover")
    assert max(depths) >= 2, (
        f"no nested subsystems generated (max depth {max(depths)}); splices must "
        f"be exercised through more than one level")


@pytest.mark.parametrize("seed", range(N_SEEDS))
def test_generation_never_fails(seed):
    """Every generated topology emits C at all — the always-on guard, no
    compiler needed, so it sweeps the full seed range."""
    files = build_model(seed).to_c(
        name="model", numeric="double", reductions="unrolled",
        structure="hierarchical", layout="compact", api="struct", solver="rk4")
    assert files, f"seed {seed}: produced no files"


@pytest.mark.parametrize("seed", range(N_SEEDS))
def test_generated_c_is_self_consistent(seed):
    """Structure and layout must not change the emitted semantics.

    Cheap invariant that needs no compiler: the same model emitted flat and
    hierarchically, compact and as a library, must declare the same state count
    and the same addressable signals. A wiring defect that depends on emission
    shape — which is exactly what the subsystem splice bug was — shows up here.
    """
    def emit(**opts):
        return build_model(seed).to_c(
            name="model", numeric="double", reductions="unrolled",
            api="struct", solver="rk4", **opts)

    base = emit(structure="hierarchical", layout="compact")["model.h"]
    for opts in (dict(structure="flat", layout="compact"),
                 dict(structure="hierarchical", layout="library")):
        other = emit(**opts)["model.h"]
        for macro in ("_N_STATE",):
            a = [ln for ln in base.splitlines() if macro in ln]
            b = [ln for ln in other.splitlines() if macro in ln]
            assert a == b, f"seed {seed}: {opts} changed {macro}: {a} vs {b}"
        # The addressable signal enum is the block's public identity; it must
        # not depend on how the code was split across files.
        def sig_names(src):
            return sorted(ln.strip() for ln in src.splitlines() if "_SIG_" in ln)
        assert sig_names(base) == sig_names(other), (
            f"seed {seed}: {opts} changed the addressable signal set")
