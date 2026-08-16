"""Subprocess runner for `test_example_parity.py`.

Executes one pathsim example under the requested engine and prints the recorded
scope data as JSON on a marker line. Kept as a separate file (not an inline
string) so it can be run by hand when an example misbehaves:

    python tests/python/_example_runner.py fastsim path/to/example.py

Under ``fastsim`` the example's ``import pathsim`` is redirected to fastsim —
that redirection IS the drop-in claim under test.
"""
import json
import os
import sys


def _headless():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: None
    plt.pause = lambda *a, **k: None


def _stub_presentation(engine):
    """Give every block the plotting methods a demo might call.

    A validation harness compares recorded data and never renders, so a missing
    `plot2D` must not be reported as a simulation failure. Only ADDS missing
    names; an engine that has them keeps its own.
    """
    for name in dir(engine.blocks):
        cls = getattr(engine.blocks, name, None)
        if not isinstance(cls, type):
            continue
        for meth in ("plot", "plot2D", "plot3D"):
            if not hasattr(cls, meth):
                try:
                    setattr(cls, meth, lambda self, *a, **k: None)
                except (AttributeError, TypeError):
                    pass


def _load(engine_name):
    if engine_name == "fastsim":
        import fastsim as engine
        sys.modules["pathsim"] = engine
        for sub in ("blocks", "solvers", "events", "utils", "optim"):
            try:
                sys.modules[f"pathsim.{sub}"] = __import__(
                    f"fastsim.{sub}", fromlist=["*"])
            except ImportError:
                pass
    else:
        import pathsim as engine
    return engine


def main():
    engine_name, path = sys.argv[1], sys.argv[2]
    _headless()
    engine = _load(engine_name)
    _stub_presentation(engine)

    ns = {"__name__": "__main__", "__file__": path}
    os.chdir(os.path.dirname(os.path.abspath(path)))
    with open(path, encoding="utf-8", errors="replace") as fh:
        source = fh.read()
    exec(compile(source, path, "exec"), ns)

    # Harvest every Scope in declaration order — that ordering is what makes the
    # two runs comparable without needing names.
    Scope = engine.blocks.Scope
    out = []
    for value in ns.values():
        if isinstance(value, Scope):
            try:
                t, data = value.read()
                out.append({
                    "t": [float(x) for x in t],
                    "d": [[float(x) for x in ch] for ch in data],
                })
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                out.append({"t": [], "d": [], "error": str(exc)})
    print("@@RESULT@@" + json.dumps(out))


if __name__ == "__main__":
    main()
