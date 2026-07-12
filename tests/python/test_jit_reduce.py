"""Vector reductions lower to the structured `Reduce` op, not an Add-chain.

The tracer collapses ``np.sum`` / ``np.prod`` / ``np.min`` / ``np.max`` /
``np.mean`` over a traced array into a single ``Reduce`` graph node instead of
an N-deep ``Binary`` chain. This keeps the trace graph (and the exported IR)
O(1) per reduction rather than O(N) and gives codegen a loop to emit. These
tests guard both that the reductions still JIT-compile and run, and that the
structured op actually survives into the IR.
"""

import numpy as np
import pytest

from fastsim import Connection, Simulation
from fastsim.blocks import ODE, Scope
from fastsim.solvers import RKDP54


def _jit(block):
    return block.__dict__.get("_jit_compiled", False)


def _run_and_read(block, n_out, duration=0.1):
    """Build a trivial sim around `block`, run once, return final y vector."""
    sco = Scope()
    conns = [Connection(block[i], sco[i]) for i in range(n_out)]
    sim = Simulation([block, sco], conns, log=False)
    sim._set_solver(RKDP54, tolerance_lte_abs=1e-10)
    sim.run(duration)
    _, ch = sco.read()
    return np.array([c[-1] for c in ch[:n_out]])


REDUCTIONS = [
    ("sum", np.sum),
    ("prod", np.prod),
    ("amin", np.amin),
    ("amax", np.amax),
    ("mean", np.mean),
]


@pytest.mark.parametrize("name,np_fn", REDUCTIONS)
def test_reduction_compiles_and_runs(name, np_fn):
    # dx = -x + reduce(x): a scalar reduction broadcast back over the state.
    def rhs(x, t):
        return -x + np_fn(x)

    blk = ODE(rhs, initial_value=[0.5, 0.3, 0.7])
    assert _jit(blk), f"{name}: expected JIT compile, got Python fallback"
    y = _run_and_read(blk, n_out=3)
    assert np.all(np.isfinite(y)), f"{name}: non-finite output"


def test_sum_lowers_to_single_reduce_in_ir():
    # dx_i = sum(x) - x_i over a 6-state ODE. The sum is one shared Reduce node
    # (CSE across all six outputs); no scalar Add chain remains.
    n = 6

    def rhs(x, t):
        s = np.sum(x)
        return [s - x[i] for i in range(n)]

    blk = ODE(rhs, initial_value=[float(k + 1) for k in range(n)])
    sim = Simulation([blk], log=False)
    js = sim.to_ir("reduce_demo").to_json()
    assert '"Reduce"' in js, "np.sum should lower to a Reduce op in the IR"
    assert '"Add"' not in js, "the addition tree should be gone (one Reduce, no Add)"
