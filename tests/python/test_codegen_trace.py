"""`to_c(trace=True)` emits the model-to-code trace map `<name>_trace.json`.

Contract checks from the Python side — deep validation (line references point
at real definitions) lives in the Rust test `trace_map_points_at_real_definitions`.
"""

import json

from fastsim import Simulation, Connection
from fastsim.blocks import Integrator, Amplifier


def _sim():
    i, a = Integrator(1.0), Amplifier(-1.0)
    return Simulation(
        blocks=[i, a],
        connections=[Connection(i, a), Connection(a, i)],
        log=False,
    )


def test_trace_is_opt_in():
    assert not any(n.endswith("_trace.json") for n in _sim().to_c("decay"))


def test_trace_map_contents():
    files = _sim().to_c("decay", trace=True)
    assert "decay_trace.json" in files
    doc = json.loads(files["decay_trace.json"])

    assert doc["model"] == "decay"
    assert doc["metrics"]["n_state"] == 1
    assert doc["metrics"]["model_struct_bytes_packed"] > 0
    assert doc["metrics"]["integrator_stack_bytes"] > 0

    # Block -> code: every listed function reference names an emitted file.
    for b in doc["blocks"]:
        for f in b["functions"]:
            assert f["file"] in files and f["line"] >= 1

    # Signal map: ids usable with <name>_get_signal / set_signal.
    kinds = {s["kind"] for s in doc["signals"]}
    assert "State" in kinds
    names = [s["name"] for s in doc["signals"]]
    assert len(names) == len(set(names)), "signal names disambiguated"


def test_trace_combines_with_scaffold():
    files = _sim().to_c("decay", trace=True, scaffold=True)
    assert "decay_trace.json" in files and "CMakeLists.txt" in files
    doc = json.loads(files["decay_trace.json"])
    # The trace's file inventory includes the scaffold (emitted before it).
    assert "decay_main.c" in doc["files"]
