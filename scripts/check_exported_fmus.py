#!/usr/bin/env python
"""Check the published FastSim FMUs the way an importing tool would.

Runs over every model directory under ``fmus/``:

1. ``fmpy.validate_fmu`` — the structural checks (modelDescription against the
   schema, value references, model structure).
2. Compile the source FMU for this platform. FastSim exports *source* FMUs, so
   the importer builds the binary; FMPy does that from the FMU's own
   ``sources/buildDescription.xml``.
3. Simulate in Model Exchange and in Co-Simulation, replaying ``{name}_in.csv``
   for models that have inputs, over the interval ``{name}_ref.opt`` declares.
4. Compare against ``{name}_ref.csv`` — FastSim's own answer — on the reference
   time grid.

This is the reproducible form of the export compatibility claim: run it and read
the errors, rather than taking the claim on trust. It needs FMPy and a C
compiler::

    python scripts/check_exported_fmus.py

Exits non-zero if any FMU fails to validate, build, simulate, or match.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FMUS = ROOT / "fmus"

# Comparison bands, not claims of agreement: the printed number is the result.
# Model Exchange is integrated by the importer's own adaptive solver at the
# `RelTol` the `_ref.opt` declares, so it drifts from the fixed-step reference by
# roughly that tolerance times the number of steps; Co-Simulation replays the
# same step size FastSim used and tracks far more closely. Both bands sit well
# under the signal amplitude, so a real defect still fails them.
TOLERANCES = {"ModelExchange": 1e-4, "CoSimulation": 1e-6}


def read_opt(path: Path) -> dict[str, float]:
    """Parse the `key,value` lines of a `_ref.opt` file."""
    opt: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("//")[0].strip()
        if not line or "," not in line:
            continue
        key, _, value = line.partition(",")
        try:
            opt[key.strip()] = float(value.strip())
        except ValueError:
            pass  # non-numeric entries such as SolverType
    return opt


def read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=float, deletechars="")
    return list(rows.dtype.names), rows


def build_binary(fmu_path: Path, workdir: Path) -> Path:
    """Unpack the source FMU, compile it, and repack — what an importer does."""
    from fmpy.build import build_platform_binary

    unpacked = workdir / f"{fmu_path.stem}_unpacked"
    with zipfile.ZipFile(fmu_path) as z:
        z.extractall(unpacked)

    cmake_options = {}
    try:
        from fastsim._fastsim import find_c_compiler

        cc = find_c_compiler()
        if cc:
            cmake_options["CMAKE_C_COMPILER"] = cc
    except Exception:
        pass  # let CMake find a compiler on its own

    build_platform_binary(unpacked, cmake_options=cmake_options)

    built = workdir / f"{fmu_path.stem}_built.fmu"
    shutil.make_archive(str(built.with_suffix("")), "zip", unpacked)
    shutil.move(str(built.with_suffix(".zip")), str(built))
    return built


def check_model(model_dir: Path, workdir: Path) -> bool:
    from fmpy import simulate_fmu
    from fmpy.validation import validate_fmu

    name = model_dir.name
    fmu_path = model_dir / f"{name}.fmu"
    if not fmu_path.is_file():
        print(f"{name}: no {name}.fmu")
        return False

    print(f"\n{name}")

    problems = validate_fmu(str(fmu_path))
    if problems:
        for p in problems:
            print(f"  validate_fmu: {p}")
        return False
    print("  validate_fmu           no problems")

    opt = read_opt(model_dir / f"{name}_ref.opt")
    ref_names, ref = read_csv(model_dir / f"{name}_ref.csv")
    t_ref = ref["time"]
    outputs = [n for n in ref_names if n != "time"]

    # Replay the declared input trajectory, if the model has one. FMPy takes a
    # structured array whose first field is the time base.
    in_path = model_dir / f"{name}_in.csv"
    signals = None
    if in_path.is_file():
        in_names, in_rows = read_csv(in_path)
        signals = in_rows
        print(f"  input trajectory       {', '.join(n for n in in_names if n != 'time')}"
              f" ({len(in_rows)} points)")

    built = build_binary(fmu_path, workdir)
    print(f"  built binary           {built.stat().st_size / 1024:.0f} KiB")

    ok = True
    for fmi_type, tol in TOLERANCES.items():
        kwargs = dict(
            start_time=opt["StartTime"],
            stop_time=opt["StopTime"],
            output_interval=opt.get("OutputIntervalLength", opt["StepSize"]),
            fmi_type=fmi_type,
            output=outputs,
        )
        if signals is not None:
            kwargs["input"] = signals
        if fmi_type == "ModelExchange":
            kwargs.update(solver="CVode", relative_tolerance=opt.get("RelTol", 1e-8))
        else:
            kwargs["step_size"] = opt["StepSize"]

        res = simulate_fmu(str(built), **kwargs)
        worst = 0.0
        for var in outputs:
            got = np.interp(t_ref, np.asarray(res["time"]), np.asarray(res[var]))
            worst = max(worst, float(np.max(np.abs(got - ref[var]))))
        verdict = "ok" if worst <= tol else f"FAIL (> {tol:g})"
        print(f"  {fmi_type:<22} worst |FMU - reference| = {worst:.3e}   {verdict}")
        ok &= worst <= tol

    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fmus", type=Path, default=DEFAULT_FMUS, help="directory of model dirs")
    args = ap.parse_args()

    model_dirs = sorted(p for p in args.fmus.iterdir() if p.is_dir())
    if not model_dirs:
        print(f"no model directories under {args.fmus}")
        return 1

    workdir = Path(tempfile.mkdtemp(prefix="fastsim_fmu_check_"))
    try:
        results = {d.name: check_model(d, workdir) for d in model_dirs}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    failed = [n for n, ok in results.items() if not ok]
    print()
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print(f"all {len(results)} exported FMUs validate, build and match the reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
