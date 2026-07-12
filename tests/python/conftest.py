"""Shared fixtures and markers for fastsim tests."""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "pathsim: requires pathsim package")
    # PATHSIM_REQUIRED=1 (CI) turns the soft pathsim skip into a hard failure so
    # the drop-in-parity claim is actually verified rather than silently skipped.
    if os.environ.get("PATHSIM_REQUIRED") and not HAS_PATHSIM:
        raise pytest.UsageError(
            "PATHSIM_REQUIRED is set but the `pathsim` package is not importable; "
            "the fastsim<->pathsim trajectory-parity layer cannot be verified."
        )


try:
    import pathsim  # noqa: F401
    HAS_PATHSIM = True
except ImportError:
    HAS_PATHSIM = False


def pytest_collection_modifyitems(config, items):
    """Single, unified skip mechanism: auto-skip anything carrying the ``pathsim``
    marker when pathsim is not installed (the test modules mark instead of each
    rolling its own ``unittest.skipUnless``)."""
    if HAS_PATHSIM:
        return
    skip = pytest.mark.skip(reason="pathsim not installed")
    for item in items:
        if "pathsim" in item.keywords:
            item.add_marker(skip)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Session-finish hook: report how many codegen (C/WASM) combinations actually
    RAN versus skipped, so a run that quietly skipped the commercial differentiator
    (no compiler, no ziglang/Node) is visible instead of green-passing silently."""
    stats = terminalreporter.stats
    ran = skipped = 0
    for outcome in ("passed", "failed"):
        for rep in stats.get(outcome, []):
            if rep.when == "call" and "codegen" in rep.nodeid:
                ran += 1
    for rep in stats.get("skipped", []):
        if "codegen" in rep.nodeid:
            skipped += 1
    if ran or skipped:
        terminalreporter.write_sep(
            "-", f"codegen verification: {ran} ran, {skipped} skipped"
        )
        if ran == 0 and skipped:
            terminalreporter.write_line(
                "WARNING: every codegen combination was SKIPPED — the C/WASM "
                "differentiator was not exercised (set FASTSIM_CC / install ziglang+Node).",
                yellow=True,
            )
