"""`to_c(scaffold=True)` emits the build scaffold (CMakeLists + demo main).

Pure-Python contract checks — the compile-and-run leg lives in the Rust test
`scaffold_emits_buildable_demo` (tests/codegen_verify_system.rs).
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


def test_scaffold_is_opt_in():
    files = _sim().to_c("decay")
    assert "CMakeLists.txt" not in files
    assert not any(n.endswith("_main.c") for n in files)


def test_scaffold_files_present_and_marked_editable():
    files = _sim().to_c("decay", scaffold=True)
    assert "CMakeLists.txt" in files and "decay_main.c" in files
    assert "decay.c" in files  # model sources unchanged

    cmake = files["CMakeLists.txt"]
    assert "project(decay C)" in cmake
    assert "decay.c" in cmake
    assert "add_executable(decay_demo decay_main.c)" in cmake

    main = files["decay_main.c"]
    assert "decay_step(&m, FASTSIM_DT)" in main
    assert "EDITABLE" in main            # scaffold marks itself editable
    assert "decay_set_signal" in main    # HAL hook guidance present


def test_scaffold_library_layout_lists_all_sources():
    files = _sim().to_c("decay", scaffold=True, layout="library")
    cmake = files["CMakeLists.txt"]
    for src in ("decay.c", "decay_solver.c"):
        assert src in cmake, cmake
    # Library entry header is the solver header.
    assert '#include "decay_solver.h"' in files["decay_main.c"]
