#!/usr/bin/env python
"""Cross-check FastSim's FMI 3.0 import against FMUs from commercial tools.

Complements ``verify_third_party_fmus.py`` (PMSF's hand-written test FMUs) with
FMUs exported by Dymola, Altair Twin Activate and MapleSim, downloaded from the
vendors' public compatibility repositories:

- https://github.com/CATIA-Systems/dymola-fmi-compatibility
- https://github.com/altairengineering/fmus
- https://github.com/Maplesoft-fmigroup/MapleSim_FMI

The FMUs are fetched on demand into ``--workdir`` rather than redistributed
here. Each is imported as a ``CoSimulationFMU`` and stepped on the same
communication grid as FMPy driving the same embedded binary; outputs are
compared index-aligned (interpolating across a discontinuity would charge the
model's jumps to the importer). ``matrix_inverse`` is additionally driven with
a full-rank matrix and checked against ``numpy.linalg.inv`` — a functional
element-order check for array ports.

Notes on the two known non-1e-15 cases:

- Dymola's ``CoupledClutches`` is event-rich and its embedded CVode runs at the
  ``DefaultExperiment`` tolerance; even FMPy differs from itself by ~1.3e-1
  between its two CS modes. FastSim and FMPy agree to ~3e-3 when both use
  event mode, which is the band such a model leaves two importers.
- Altair's clocked FMUs use FMI 3.0 Clocks, which FastSim does not schedule
  (they load; the clocked counters simply never advance). FMPy 0.3.26 fails to
  instantiate them at all.

Usage::

    python scripts/verify_vendor_fmus.py [--workdir DIR]
        [--fmpy-python <interpreter with fmpy installed>]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

import fastsim
from fastsim.blocks import Constant, CoSimulationFMU, Scope

RAW = "https://raw.githubusercontent.com"
DYMOLA_DIR = "CATIA-Systems/dymola-fmi-compatibility/main/2026x%20Refresh%201%2C%202026-04-08"
ALTAIR_DIR = "altairengineering/fmus/master/Altair-Activate/3.0/export"
MAPLE_DIR = "Maplesoft-fmigroup/MapleSim_FMI/main/MapleSim_2025/FMI_Export/FMI_3"

# (local name, url, dt, stop)
CASES = [
    ("sinewave_array.fmu", f"{RAW}/{ALTAIR_DIR}/cs/x86_64-windows/sinewave_array/sinewave_array.fmu", 1e-3, 1.0),
    ("CoupledClutches_fmi3_Cvode.fmu", f"{RAW}/{DYMOLA_DIR}/CoupledClutches_fmi3_Cvode.fmu", 1e-3, 1.5),
    ("CoupledClutches3.fmu", f"{RAW}/{MAPLE_DIR}/CoupledClutches/CoupledClutches3.fmu", 1e-3, 1.5),
    ("CT3_dirderiv.fmu", f"{RAW}/{MAPLE_DIR}/ControlledTemperature/CT3_dirderiv.fmu", 1e-2, 5.0),
]
MATRIX_INVERSE = (
    "matrix_inverse.fmu",
    f"{RAW}/{ALTAIR_DIR}/me/x86_64-windows/matrix_inverse/matrix_inverse.fmu",
)

# CoupledClutches is event-rich (see module docstring); the rest must agree to
# machine precision.
BANDS = {"CoupledClutches_fmi3_Cvode.fmu": 1e-2}
DEFAULT_BAND = 1e-12


def fetch(url: str, dest: Path) -> Path:
    if not dest.is_file():
        urllib.request.urlretrieve(url, dest)
    return dest


def fastsim_channels(path: Path, dt: float, stop: float) -> np.ndarray:
    fmu = CoSimulationFMU(str(path), instance_name="vendor", dt=dt)
    n = len(fmu.outputs)
    sco = Scope(labels=[f"y{i}" for i in range(n)], sampling_period=dt)
    sim = fastsim.Simulation(
        [fmu, sco],
        [fastsim.Connection(fmu[i], sco[i]) for i in range(n)],
        dt=dt,
        log=False,
    )
    sim.run(stop)
    _, ys = sco.read()
    return np.vstack(ys)


def has_event_mode(path: Path) -> bool:
    import re
    import zipfile

    x = zipfile.ZipFile(path).read("modelDescription.xml").decode("utf-8", "replace")
    m = re.search(r'hasEventMode\s*=\s*"(\w+)"', x)
    return bool(m) and m.group(1) == "true"


def fmpy_channels(fmpy_python: str, path: Path, dt: float, stop: float) -> np.ndarray:
    """Same grid under FMPy; array outputs flattened row-major, which is
    exactly FastSim's flat port order. Event mode matches what the FMU
    declares — the same decision FastSim makes, so both importers run the
    same CS mode (FMPy errors out if event mode is forced on an FMU that
    does not offer it)."""
    script = f"""
import numpy as np, json
from fmpy import simulate_fmu
r = simulate_fmu(r"{path}", stop_time={stop}, output_interval={dt},
                 fmi_type="CoSimulation", step_size={dt},
                 use_event_mode={has_event_mode(path)})
names = [n for n in r.dtype.names if n != "time"]
chans = []
for n in names:
    a = np.asarray(r[n], float).reshape(len(r), -1)
    chans.extend(a[:, k].tolist() for k in range(a.shape[1]))
print(json.dumps(chans))
"""
    out = subprocess.run([fmpy_python, "-c", script], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[-400:])
    return [np.asarray(c) for c in json.loads(out.stdout.strip().splitlines()[-1])]


def check_matrix_inverse(path: Path) -> bool:
    """Drive the 3x3 input with distinct entries; the output must be the
    inverse, element by element."""
    A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    fmu = CoSimulationFMU(str(path), instance_name="mi", dt=1e-2)
    srcs = [Constant(v) for v in A.ravel()]
    sco = Scope(labels=[f"y{i}" for i in range(9)], sampling_period=1e-2)
    conns = [fastsim.Connection(s, fmu[i]) for i, s in enumerate(srcs)]
    conns += [fastsim.Connection(fmu[i], sco[i]) for i in range(9)]
    sim = fastsim.Simulation([fmu, sco, *srcs], conns, dt=1e-2, log=False)
    sim.run(0.5)
    _, ys = sco.read()
    got = np.array([ys[i][-1] for i in range(9)]).reshape(3, 3)
    err = float(np.max(np.abs(got - np.linalg.inv(A))))
    ok = err < 1e-12
    print(f"{path.name:35s} vs numpy.linalg.inv          max |err| = {err:.3e}   "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", type=Path, default=Path("vendor-fmus"),
                    help="download/cache directory (default: ./vendor-fmus)")
    ap.add_argument("--fmpy-python", default=sys.executable,
                    help="interpreter with fmpy installed (default: this one)")
    args = ap.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    ok = True
    for name, url, dt, stop in CASES:
        path = fetch(url, args.workdir / name)
        try:
            y_fs = fastsim_channels(path, dt, stop)
            y_ref = fmpy_channels(args.fmpy_python, path, dt, stop)
        except Exception as e:
            print(f"{name:35s} ERROR {type(e).__name__}: {str(e)[:100]}")
            ok = False
            continue
        worst = 0.0
        for i in range(min(len(y_fs), len(y_ref))):
            m = min(len(y_fs[i]), len(y_ref[i]))
            scale = max(1.0, float(np.max(np.abs(y_ref[i]))))
            worst = max(worst, float(np.max(np.abs(y_fs[i][:m] - y_ref[i][:m]))) / scale)
        band = BANDS.get(name, DEFAULT_BAND)
        good = worst <= band
        ok &= good
        print(f"{name:35s} vs fmpy (same grid & mode)   worst rel = {worst:.3e}   "
              f"{'ok' if good else f'FAIL (> {band:g})'}")

    name, url = MATRIX_INVERSE
    ok &= check_matrix_inverse(fetch(url, args.workdir / name))

    print("all vendor FMU checks passed" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
