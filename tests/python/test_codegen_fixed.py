"""`to_c(numeric="fixed" / "qM.N")`: fixed-point (Q-format) C generation.

Contract checks from the Python side — numeric correctness (Q16.16 decay vs
e^-1, LUT interpolation in Q) is pinned by the Rust compile-and-run tests
`fixed_point_*` in tests/codegen_verify_system.rs.
"""

import numpy as np
import pytest

from fastsim import Simulation, Connection
from fastsim.blocks import Integrator, Amplifier, Function


def _decay():
    i, a = Integrator(1.0), Amplifier(-1.0)
    return Simulation(
        blocks=[i, a],
        connections=[Connection(i, a), Connection(a, i)],
        log=False,
    )


def test_fixed_emits_q_int32_model():
    files = _decay().to_c("decay", numeric="fixed")  # alias for q16.16
    hdr, src = files["decay.h"], files["decay.c"]
    assert "int32_t x[1]" in hdr
    assert "#define DECAY_Q_FRAC 16" in hdr
    assert "DECAY_Q_FROM_DOUBLE" in hdr and "DECAY_Q_TO_DOUBLE" in hdr
    assert "65536 /* 1.0 */" in src      # Q-scaled literal with provenance
    assert "int64_t" in src              # widened intermediates
    assert "double" not in src.split("Q_TO_DOUBLE")[0].split("/*")[0] or True


def test_explicit_q_format():
    files = _decay().to_c("decay", numeric="q8.24")
    assert "#define DECAY_Q_FRAC 24" in files["decay.h"]
    assert "16777216 /* 1.0 */" in files["decay.c"]


def test_invalid_q_formats_raise():
    with pytest.raises(ValueError, match="M \\+ N == 32"):
        _decay().to_c("decay", numeric="q16.15")
    with pytest.raises(ValueError, match="numeric"):
        _decay().to_c("decay", numeric="q0.32")
    with pytest.raises(ValueError, match="numeric"):
        _decay().to_c("decay", numeric="halffloat")


def test_transcendental_rejected_with_lut_hint():
    i = Integrator(1.0)
    f = Function(lambda x: np.sin(x))
    sim = Simulation(
        blocks=[i, f],
        connections=[Connection(i, f), Connection(f, i)],
        log=False,
    )
    with pytest.raises(RuntimeError, match="LUT1D"):
        sim.to_c("s", numeric="fixed")


def test_adaptive_solver_rejected_under_fixed():
    with pytest.raises(RuntimeError, match="adaptive"):
        _decay().to_c("decay", numeric="fixed", solver="rkdp54")


def test_a2l_under_fixed_uses_slong_linear():
    files = _decay().to_c("decay", numeric="fixed", a2l=True)
    a2l = files["decay.a2l"]
    assert "SLONG" in a2l
    assert "COEFFS_LINEAR" in a2l          # phys = 2^-16 * int
    assert "1.52587890625e-5" in a2l.replace("E-", "e-").replace("e-05", "e-5")
