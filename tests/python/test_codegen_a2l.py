"""`to_c(a2l=True)` emits the ASAP2 calibration description `<name>.a2l`.

Contract checks from the Python side — offset agreement with the emitted
struct is pinned by the Rust tests `struct_layout_matches_emitted_header` and
`a2l_offsets_agree_with_struct_layout`.
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


def test_a2l_is_opt_in():
    assert not any(n.endswith(".a2l") for n in _sim().to_c("decay"))


def test_a2l_structure():
    files = _sim().to_c("decay", a2l=True)
    assert "decay.a2l" in files
    a2l = files["decay.a2l"]
    assert "ASAP2_VERSION 1 71" in a2l
    assert "/begin PROJECT decay" in a2l
    assert "/begin MEASUREMENT time" in a2l
    assert 'SYMBOL_LINK "decay" 0' in a2l  # time at offset 0
    assert a2l.count("/begin MEASUREMENT") >= 2  # time + state (+ signals)
    # Every /begin has its /end.
    for kw in ("PROJECT", "MODULE", "MEASUREMENT", "CHARACTERISTIC", "COMPU_METHOD", "RECORD_LAYOUT"):
        assert a2l.count(f"/begin {kw}") == a2l.count(f"/end {kw}"), kw


def test_a2l_float_numeric_uses_float32():
    files = _sim().to_c("decay", a2l=True, numeric="float")
    a2l = files["decay.a2l"]
    assert "FLOAT32_IEEE" in a2l and "FLOAT64_IEEE" not in a2l
