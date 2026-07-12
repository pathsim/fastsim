"""The `Simulation` facade tolerates user-defined attributes (pathsim parity).

pathsim simulations are plain Python objects users freely tag with extra
attributes; the facade's `__setattr__` used to delegate everything to the Rust
engine and raise `AttributeError` for anything it didn't know.
"""

from fastsim import Simulation, Connection
from fastsim.blocks import Integrator, Amplifier


def _sim():
    i, a = Integrator(1.0), Amplifier(-1.0)
    return Simulation(
        blocks=[i, a],
        connections=[Connection(i, a), Connection(a, i)],
        log=False,
    )


def test_user_attribute_round_trips():
    sim = _sim()
    sim.my_tag = {"campaign": 42}
    assert sim.my_tag == {"campaign": 42}
    sim.my_tag = None
    assert sim.my_tag is None


def test_engine_attributes_still_delegate():
    sim = _sim()
    sim.dt = 0.25  # a real engine attribute must reach the engine, not shadow it
    assert sim.__dict__["_sim"].dt == 0.25
    assert "dt" not in {k for k in sim.__dict__ if k != "_sim"}


def test_unknown_attribute_read_still_raises():
    sim = _sim()
    try:
        _ = sim.definitely_not_an_attribute
        raise AssertionError("expected AttributeError")
    except AttributeError:
        pass
