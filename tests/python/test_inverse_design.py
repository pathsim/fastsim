"""Parameter sensitivity and inverse design.

`sensitivity()` is the primitive: `dy/dp` at an operating point, assembled from
the same interconnection elimination as the linearization, with the per-block
derivatives coming from AD over the SSA graph. `solve_inverse()` is a thin
Newton layer on top — it answers "what settings put the system here?" where
`steadystate()` answers "given these settings, where does it settle?".
"""

import numpy as np
import pytest

from fastsim import Simulation, Connection
from fastsim.blocks import Adder, Amplifier, Comparator, Constant, Integrator


def _loop(c=2.0, k=3.0):
    """`dx/dt = k*c - x`, so the steady state is `x = k*c` and `y = x`."""
    src = Constant(c)
    gain = Amplifier(k)
    add = Adder("+-")
    integ = Integrator(0.0)
    feedback = Amplifier(1.0)

    sim = Simulation(
        blocks=[src, gain, add, integ, feedback],
        connections=[
            Connection(src, gain),
            Connection(gain, add[0]),
            Connection(integ, feedback),
            Connection(feedback, add[1]),
            Connection(add, integ),
        ],
        dt=0.01, log=False,
    )
    return sim, src, gain, integ


# -- sensitivity -----------------------------------------------------------------------

def test_sensitivity_matches_analytic():
    """`x_ss = k*c`, so `dy/dk = c`."""
    c = 2.0
    sim, _, gain, integ = _loop(c=c)

    S = sim.sensitivity(outputs=[integ[0]], wrt=[gain.param("gain")])

    assert S.shape == (1, 1)
    assert np.isclose(S[0, 0], c)


def test_sensitivity_matches_finite_differences():
    """The exact AD result agrees with a differenced reference."""
    sim, _, gain, integ = _loop()
    S = sim.sensitivity(outputs=[integ[0]], wrt=[gain.param("gain")])

    h = 1e-4
    ys = []
    for k in (3.0 - h, 3.0 + h):
        sim_h, _, gain_h, integ_h = _loop(k=k)
        sim_h.steadystate(reset=False)
        ys.append(integ_h[0].get_outputs()[0])

    assert np.isclose(S[0, 0], (ys[1] - ys[0]) / (2 * h), atol=1e-5)


def test_sensitivity_shape_for_multiple_parameters():
    sim, src, gain, integ = _loop()

    S = sim.sensitivity(
        outputs=[integ[0]],
        wrt=[gain.param("gain"), src.param("value")],
    )

    assert S.shape == (1, 2)


def test_unimplemented_mode_is_explicit():
    sim, _, gain, integ = _loop()

    with pytest.raises(NotImplementedError, match="steadystate"):
        sim.sensitivity(outputs=[integ[0]], wrt=[gain.param("gain")], mode="transient")


def test_sensitivity_rejects_non_linearizable_blocks():
    from fastsim.exceptions import LinearizationError

    c = Comparator()
    sim = Simulation(blocks=[c], connections=[], log=False)

    with pytest.raises(LinearizationError):
        sim.sensitivity(outputs=[c[0]], wrt=[])


# -- inverse solve ---------------------------------------------------------------------

def test_solve_inverse_finds_the_analytic_value():
    """`x_ss = gain * c`, so hitting `y = 10` with `c = 2` needs `gain = 5`."""
    sim, _, gain, integ = _loop()

    result = sim.solve_inverse(
        targets=[(integ[0], 10.0)],
        free=[gain.param("gain")],
    )

    assert result["success"]
    assert np.isclose(result["values"]["gain"], 5.0)
    assert np.isclose(gain.gain, 5.0), "the solved value must be left applied"


def test_solve_inverse_rejects_non_square_problems():
    """One free parameter per scalar target; anything else is rejected rather
    than silently least-squares fitted."""
    sim, _, gain, integ = _loop()

    with pytest.raises(ValueError, match="not square"):
        sim.solve_inverse(targets=[(integ[0], 1.0)], free=[])

    with pytest.raises(ValueError, match="not square"):
        sim.solve_inverse(
            targets=[(integ[0], 1.0)],
            free=[gain.param("gain"), Constant(0.0).param("value")],
        )


def test_solve_inverse_rejects_a_foreign_block():
    """A parameter on a block outside the diagram is caught by name, before any
    numerics — a more specific failure than a singular Jacobian."""
    sim, _, _, integ = _loop()
    foreign = Amplifier(1.0)

    with pytest.raises(ValueError, match="not part of simulation"):
        sim.solve_inverse(targets=[(integ[0], 10.0)], free=[foreign.param("gain")])


def test_solve_inverse_reports_a_dead_parameter():
    """A parameter that is in the diagram but cannot move the tapped output
    gives a singular sensitivity, and the failure says so."""
    from fastsim.blocks import Scope

    src, gain, add, integ, feedback = (
        Constant(2.0), Amplifier(3.0), Adder("+-"), Integrator(0.0), Amplifier(1.0)
    )
    # a side branch that observes the source but does not feed the loop
    idle, sco = Amplifier(5.0), Scope()

    sim = Simulation(
        blocks=[src, gain, add, integ, feedback, idle, sco],
        connections=[
            Connection(src, gain),
            Connection(gain, add[0]),
            Connection(integ, feedback),
            Connection(feedback, add[1]),
            Connection(add, integ),
            Connection(src, idle),
            Connection(idle, sco),
        ],
        dt=0.01, log=False,
    )

    with pytest.raises(RuntimeError, match="singular"):
        sim.solve_inverse(targets=[(integ[0], 10.0)], free=[idle.param("gain")])


# -- parameter handles -----------------------------------------------------------------

def test_parameter_handle_round_trips():
    amp = Amplifier(2.0)
    p = amp.param("gain")

    assert p.value == 2.0
    p.value = 7.0
    assert amp.gain == 7.0


def test_unknown_parameter_name_is_rejected():
    amp = Amplifier(2.0)

    with pytest.raises(AttributeError, match="gain"):
        amp.param("not_a_parameter")
