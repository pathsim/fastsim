"""Operating-point linearization: `Simulation.to_statespace`.

Checks the assembled global model against analytically known systems, and the
loud failure for blocks that have no linear model. The API mirrors pathsim's
`Simulation.to_statespace`, so models port between the two projects.
"""

import numpy as np
import pytest

from fastsim import Simulation, Connection
from fastsim.blocks import (
    Adder, Amplifier, Comparator, Constant, Integrator, SampleHold, Scope,
)
from fastsim.exceptions import LinearizationError


# -- analytic reference systems --------------------------------------------------------

def test_cascade_matches_analytic_model():
    """u -> gain(3) -> int -> gain(2) -> int, so dx1 = 3u, dx2 = 2*x1, y = x2."""
    src, g1, i1, g2, i2 = (
        Constant(0.0), Amplifier(3.0), Integrator(0.0), Amplifier(2.0), Integrator(0.0)
    )
    sim = Simulation(
        blocks=[src, g1, i1, g2, i2],
        connections=[
            Connection(src, g1), Connection(g1, i1),
            Connection(i1, g2), Connection(g2, i2),
        ],
        log=False,
    )

    ss = sim.to_statespace(inputs=[g1[0]], outputs=[i2[0]])

    assert np.allclose(ss.A, [[0, 0], [2, 0]])
    assert np.allclose(ss.B, [[3], [0]])
    assert np.allclose(ss.C, [[0, 1]])
    assert np.allclose(ss.D, [[0]])


def test_negative_feedback_gives_minus_k():
    """Closing a loop with gain k around an integrator gives A = -k."""
    k = 5.0
    ref, add, integ, gain = Constant(0.0), Adder("+-"), Integrator(0.0), Amplifier(k)
    sim = Simulation(
        blocks=[ref, add, integ, gain],
        connections=[
            Connection(ref, add[0]), Connection(integ, gain),
            Connection(gain, add[1]), Connection(add, integ),
        ],
        log=False,
    )

    ss = sim.to_statespace(inputs=[add[0]], outputs=[integ[0]])

    assert np.allclose(ss.A, [[-k]])
    assert np.allclose(ss.B, [[1.0]])
    assert np.allclose(ss.C, [[1.0]])
    assert np.allclose(ss.D, [[0.0]])


def test_algebraic_loop_is_resolved():
    """A loop surviving the break is eliminated, not rejected.

    Two gains of 0.5 fed through an adder give a closed-loop gain of
    0.5 / (1 - 0.25).
    """
    src, add, a, b = Constant(0.0), Adder("++"), Amplifier(0.5), Amplifier(0.5)
    sim = Simulation(
        blocks=[src, add, a, b],
        connections=[
            Connection(src, add[0]), Connection(add, a),
            Connection(a, b), Connection(b, add[1]),
        ],
        log=False,
    )

    ss = sim.to_statespace(inputs=[add[0]], outputs=[a[0]])

    assert np.allclose(ss.D, [[0.5 / (1 - 0.25)]])


def test_ill_posed_loop_is_rejected():
    """A unity-gain algebraic loop has no linear model."""
    a, b = Amplifier(1.0), Amplifier(1.0)
    sim = Simulation(
        blocks=[a, b],
        connections=[Connection(a, b), Connection(b, a)],
        log=False,
    )

    with pytest.raises(LinearizationError):
        sim.to_statespace(inputs=[], outputs=[a[0]])


# -- labels ----------------------------------------------------------------------------

def test_labels_are_consistent_across_categories():
    """The same block is named the same way in every label list."""
    src, g1, i1, g2, i2 = (
        Constant(0.0), Amplifier(3.0), Integrator(0.0), Amplifier(2.0), Integrator(0.0)
    )
    sim = Simulation(
        blocks=[src, g1, i1, g2, i2],
        connections=[
            Connection(src, g1), Connection(g1, i1),
            Connection(i1, g2), Connection(g2, i2),
        ],
        log=False,
    )

    # break at the SECOND amplifier, tap the SECOND integrator
    ss = sim.to_statespace(inputs=[g2[0]], outputs=[i2[0]])

    assert ss.state_labels == ["Integrator_0", "Integrator_1"]
    assert ss.output_labels == ["Integrator_1"]
    assert ss.input_labels == ["Amplifier_1"]


# -- blocks without a linear model -----------------------------------------------------

def test_switching_block_is_rejected():
    """A comparator has no linear model at its switching point."""
    c = Comparator()
    sim = Simulation(blocks=[c], connections=[], log=False)

    with pytest.raises(LinearizationError, match="Comparator"):
        sim.to_statespace(inputs=[c[0]], outputs=[c[0]])


def test_discrete_block_is_rejected():
    """Discrete-time blocks have no continuous-time linear model."""
    sh = SampleHold(T=1.0)
    sim = Simulation(blocks=[sh], connections=[], log=False)

    with pytest.raises(LinearizationError):
        sim.to_statespace(inputs=[sh[0]], outputs=[sh[0]])


# -- query semantics -------------------------------------------------------------------

def test_to_statespace_is_a_pure_query():
    """Assembling the model must leave the simulation untouched."""

    def build():
        src, integ, sco = Constant(1.0), Integrator(0.0), Scope()
        sim = Simulation(
            blocks=[src, integ, sco],
            connections=[Connection(src, integ), Connection(integ, sco)],
            dt=0.01, log=False,
        )
        return sim, integ, sco

    sim_ref, _, sco_ref = build()
    sim_ref.run(2.0)
    _, (y_ref,) = sco_ref.read()

    sim, integ, sco = build()
    sim.to_statespace(inputs=[integ[0]], outputs=[integ[0]])
    sim.run(2.0)
    _, (y,) = sco.read()

    assert np.allclose(y, y_ref)


def test_linearize_is_not_a_mode_switch():
    """`linearize()` has no surrogate mode to switch into and says so."""
    sim = Simulation(blocks=[Integrator(0.0)], connections=[], log=False)

    with pytest.raises(NotImplementedError, match="to_statespace"):
        sim.linearize()
