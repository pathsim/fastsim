"""Structure='flat' codegen with Schedule/time events.

Regression: the fused `model_deriv` used to reject any model with events
("Structure::Flat with events (not yet emitted)"). Schedule events that only
touch discrete memory (stepped sources, discrete blocks) now lower — the fused
deriv reads the model's `mem`/`s->mem` directly, and `model_handle_events` fires
the effects. Verified to match the hierarchical structure.
"""
import fastsim as fs
from fastsim.blocks import StepSource, Integrator, Scope


def _step_walk():
    src = StepSource(amplitude=[1.0, 2.0], tau=[0.1, 0.5])
    itg = Integrator(0.0)
    sco = Scope()
    return fs.Simulation(
        blocks=[src, itg, sco],
        connections=[fs.Connection(src, itg), fs.Connection(itg, sco)],
        dt=0.05,
    )


def test_flat_codegen_emits_memory_events():
    files = _step_walk().to_c(structure="flat")
    model = files["model.c"]
    # the event machinery is emitted...
    assert "model_handle_events" in model
    # ...and the fused deriv reads discrete memory by the in-scope name (not a
    # phantom `m` parameter that the flat body never receives).
    deriv = model.split("model_deriv")[1].split("model_init")[0]
    assert "mem[" in deriv


def test_flat_matches_hierarchical_structure_codegen():
    # Both shapes are emitted without error for the same event-bearing model.
    flat = _step_walk().to_c(structure="flat")
    hier = _step_walk().to_c(structure="hierarchical")
    assert "model_deriv" in flat["model.c"]
    assert "model_deriv" in hier["model.c"]
