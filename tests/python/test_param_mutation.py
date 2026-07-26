"""Reassigning a constructor parameter must affect the running simulation.

Regression tests for a bug where `amp.gain = 10` read back as 10 but the
simulation kept computing with the old value: the shim rebuilt the Rust block
from its factory and repointed the Python handle at the new one, while the
`Simulation` and every `Connection` still held the old block. The value looked
updated and had no effect.

The block is now adopted in place, so existing references stay valid.
"""

import numpy as np

from fastsim import Simulation, Connection
from fastsim.blocks import Adder, Amplifier, Constant, Integrator, Scope


def test_parameter_change_affects_the_run():
    src, amp, sco = Constant(1.0), Amplifier(2.0), Scope()
    sim = Simulation(
        blocks=[src, amp, sco],
        connections=[Connection(src, amp), Connection(amp, sco)],
        dt=0.1, log=False,
    )

    sim.run(0.5)
    assert np.isclose(sco.read()[1][0][-1], 2.0)

    amp.gain = 10.0
    sim.reset()
    sim.run(0.5)
    assert np.isclose(sco.read()[1][0][-1], 10.0), "gain change did not reach the run"


def test_parameter_change_preserves_state():
    """The engine state belongs to the block's place in the system, not to its
    parameters, so it survives the rebuild."""
    src, integ, sco = Constant(1.0), Integrator(0.0), Scope()
    sim = Simulation(
        blocks=[src, integ, sco],
        connections=[Connection(src, integ), Connection(integ, sco)],
        dt=0.01, log=False,
    )

    sim.run(1.0)
    before = np.atleast_1d(integ.state).copy()

    src.value = 2.0

    assert np.allclose(np.atleast_1d(integ.state), before), "state lost on rebuild"

    # x(1) = 1, then two seconds at the new rate 2 -> x(3) = 5. Had the change
    # not reached the run it would be 3; the tolerance only covers the
    # fixed-step discretization rest, not that difference.
    sim.run(2.0)
    assert np.isclose(sco.read()[1][0][-1], 5.0, atol=5e-2)


def test_parameter_change_preserves_multi_port_wiring():
    """Register sizes come from the connection layout, which the freshly built
    block does not know about."""
    a, b, add, sco = Constant(1.0), Constant(2.0), Adder("++"), Scope()
    sim = Simulation(
        blocks=[a, b, add, sco],
        connections=[
            Connection(a, add[0]), Connection(b, add[1]), Connection(add, sco),
        ],
        dt=0.1, log=False,
    )

    sim.run(0.2)
    assert np.isclose(sco.read()[1][0][-1], 3.0)

    b.value = 10.0
    sim.reset()
    sim.run(0.2)
    assert np.isclose(sco.read()[1][0][-1], 11.0), "wiring broke on rebuild"


def test_set_batches_the_same_way():
    """`set(**kwargs)` takes the same in-place path as attribute assignment."""
    src, amp, sco = Constant(1.0), Amplifier(2.0), Scope()
    sim = Simulation(
        blocks=[src, amp, sco],
        connections=[Connection(src, amp), Connection(amp, sco)],
        dt=0.1, log=False,
    )

    amp.set(gain=4.0)
    sim.run(0.5)
    assert np.isclose(sco.read()[1][0][-1], 4.0)
