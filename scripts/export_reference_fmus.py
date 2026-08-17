#!/usr/bin/env python
"""Generate the published FastSim FMUs and their reference solutions.

These are the artifacts under ``fmus/`` that other tools import to check
FastSim's FMI 3.0 export against. Regenerate them with::

    python scripts/export_reference_fmus.py

Each model directory follows the FMI Cross-Check file convention
(FMI-CROSS-CHECK-RULES.md §9.1.4), which is the shape importing tools already
know how to consume:

    {name}.fmu       the FMU as `to_fmu()` writes it
    {name}_ref.csv   reference solution computed by FastSim itself
    {name}_ref.opt   the experiment settings that solution was produced with
    {name}_in.csv    input trajectory, for models that have inputs

The reference solution is FastSim's own answer, not the FMU's — that is the
point of it. `docs/source/examples/fmi-export` checks the FMU against this same
reference and against the closed form, so the numbers here are the ones the
example page reports.

Input trajectories are constant on purpose. §9.1.4 says intermediate values are
obtained by linear interpolation, but it does not say how finely an importer
resamples the trajectory — per communication step or per solver stage — and on a
shaped signal those choices disagree at the 1e-4 level. That would blunt the
comparison: a constant is identical under every scheme, so whatever deviation is
left belongs to the FMU.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import fastsim
from fastsim import Connection, Interface, Simulation, Subsystem
from fastsim.blocks import ODE, Scope, Source
from fastsim.solvers import RK4

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fmus"

# --- oscillator: damped linear oscillator, no inputs -----------------------
OMEGA_0 = 2 * np.pi   # natural frequency [rad/s]
ZETA = 0.15           # damping ratio
OSC_STOP, OSC_DT = 3.0, 1e-3

# --- suspension: forced mass-spring-damper exported as a Subsystem ---------
MASS, DAMPING, STIFFNESS = 1.0, 0.4, 25.0
SUS_STOP, SUS_DT = 5.0, 1e-3
# A constant 1 N step from t = 0. Deliberately constant rather than shaped: an
# importer replaying `_in.csv` decides for itself how finely to resample the
# trajectory — per communication step, or per solver stage — and on a ramp those
# choices disagree at the 1e-4 level, which would blunt the comparison. A
# constant is identical under every scheme, so any deviation left is the FMU's.
FORCE_KNOTS = [(0.0, 1.0), (SUS_STOP, 1.0)]


def force(t: float) -> float:
    ts, vs = zip(*FORCE_KNOTS)
    return float(np.interp(t, ts, vs))


def build_oscillator():
    """x'' + 2 zeta w0 x' + w0^2 x = 0, released from x = 1."""
    osc = ODE(
        lambda x, u, t: np.array(
            [x[1], -2 * ZETA * OMEGA_0 * x[1] - OMEGA_0**2 * x[0]]
        ),
        initial_value=[1.0, 0.0],
    )
    return osc


def build_plant():
    """m x'' + c x' + k x = F(t), as a bare block. Two outputs: x and x'."""
    return ODE(
        lambda x, u, t: np.array(
            [x[1], (u[0] - DAMPING * x[1] - STIFFNESS * x[0]) / MASS]
        ),
        initial_value=[0.0, 0.0],
    )


def build_suspension():
    """The same plant wrapped in a Subsystem, which is what gets exported.

    The Interface routes the force in and the position out, but the FMU exposes
    both of the plant's outputs (`ODE_y0` = x, `ODE_y1` = x'), so the reference
    below is recorded from the bare plant — same dynamics, both signals
    reachable. Wiring a Scope to the subsystem instead would silently leave the
    velocity column at zero.
    """
    iface = Interface()
    plant = build_plant()
    return Subsystem(
        [iface, plant],
        [Connection(iface, plant), Connection(plant[0], iface)],
    )


def write_ref_opt(path: Path, start: float, stop: float, step: float, rtol: float) -> None:
    """The `_ref.opt` block, in the key/value CSV form the rules prescribe.

    `StepSize` is non-zero because the reference is produced with a fixed-step
    solver, and `SolverType` names it so a comparison run can match.
    """
    path.write_text(
        "\n".join(
            [
                f"StartTime,{start}",
                f"StopTime,{stop}",
                f"StepSize,{step}",
                f"RelTol,{rtol}",
                "SolverType,FixedStep",
                f"OutputIntervalLength,{step}",
            ]
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def clip_to(stop: float, t: np.ndarray, *columns: np.ndarray):
    """Trim a recording to [0, stop].

    A fixed-step run overshoots by up to one step, and `_ref.opt` promises the
    solution covers exactly StartTime..StopTime.
    """
    keep = t <= stop + 1e-12
    return t[keep], [c[keep] for c in columns]


def write_csv(path: Path, header: list[str], columns: list[np.ndarray]) -> None:
    # The csv dialect terminates rows with CRLF of its own accord, so pinning
    # the file's newline alone is not enough — `lineterminator` is what makes
    # the published file byte-identical whatever platform wrote it.
    with path.open("w", newline="\n", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for row in zip(*columns):
            w.writerow([f"{v:.12g}" for v in row])


def export_oscillator(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Reference solution from FastSim itself.
    # Recording every step of a fixed-step run, rather than through the Scope's
    # `sampling_period`, gives an exactly uniform grid: that scheduled sampling
    # path collapses two ticks that land in one step into a single sample, which
    # leaves a gap a published reference should not have.
    osc = build_oscillator()
    sco = Scope(labels=["x", "v"])
    sim = Simulation(
        blocks=[osc, sco],
        connections=[Connection(osc[0], sco[0]), Connection(osc[1], sco[1])],
        Solver=RK4,
        dt=OSC_DT,
        log=False,
    )
    sim.run(OSC_STOP, reset=True, adaptive=False)
    t_all, (x_all, v_all) = sco.read()
    t, (x, v) = clip_to(OSC_STOP, t_all, x_all, v_all)

    # The FMU, exported from a freshly built copy so the reference run's final
    # state cannot leak into it.
    osc_fmu = build_oscillator()
    Simulation(blocks=[osc_fmu], connections=[], Solver=RK4, dt=OSC_DT, log=False).to_fmu(
        str(outdir / "oscillator.fmu"),
        name="oscillator",
        start_time=0.0,
        stop_time=OSC_STOP,
        step_size=OSC_DT,
        tolerance=1e-8,
    )

    # Column names must be the FMU's output variable names.
    write_csv(outdir / "oscillator_ref.csv", ["time", "ODE_y0", "ODE_y1"], [t, x, v])
    write_ref_opt(outdir / "oscillator_ref.opt", 0.0, OSC_STOP, OSC_DT, 1e-8)

    # Closed form of the reference, as a sanity check on what we publish.
    omega_d = OMEGA_0 * np.sqrt(1 - ZETA**2)
    exact = np.exp(-ZETA * OMEGA_0 * t) * (
        np.cos(omega_d * t) + ZETA * OMEGA_0 / omega_d * np.sin(omega_d * t)
    )
    print(f"oscillator  worst |x - analytic| = {np.max(np.abs(x - exact)):.3e}")


def export_suspension(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Reference from the bare plant, so both of the FMU's outputs are reachable
    # (see `build_suspension`). Same equations, same force, same solver.
    plant = build_plant()
    src = Source(force)
    sco = Scope(labels=["x", "v"])
    sim = Simulation(
        blocks=[src, plant, sco],
        connections=[
            Connection(src, plant),
            Connection(plant[0], sco[0]),
            Connection(plant[1], sco[1]),
        ],
        Solver=RK4,
        dt=SUS_DT,
        log=False,
    )
    sim.run(SUS_STOP, reset=True, adaptive=False)
    t_all, (x_all, v_all) = sco.read()
    t, (x, v) = clip_to(SUS_STOP, t_all, x_all, v_all)

    build_suspension().to_fmu(
        str(outdir / "suspension.fmu"),
        name="suspension",
        start_time=0.0,
        stop_time=SUS_STOP,
        step_size=SUS_DT,
    )

    write_csv(outdir / "suspension_ref.csv", ["time", "ODE_y0", "ODE_y1"], [t, x, v])
    write_ref_opt(outdir / "suspension_ref.opt", 0.0, SUS_STOP, SUS_DT, 1e-8)
    # Only the knots: the signal is piecewise linear, so an importer applying
    # linear interpolation reconstructs it exactly.
    write_csv(
        outdir / "suspension_in.csv",
        ["time", "in"],
        [np.array([k[0] for k in FORCE_KNOTS]), np.array([k[1] for k in FORCE_KNOTS])],
    )

    # Step response of a damped second-order system, released from rest:
    #   x(t) = (F/k) [ 1 - e^{-z w t} ( cos w_d t + z w / w_d sin w_d t ) ]
    # The system is lightly damped (zeta = c / (2 sqrt(km)) = 0.04, tau = 5 s),
    # so it is still ringing at t = 5 s and this checks the whole trajectory
    # rather than just where it ends up.
    w_n = np.sqrt(STIFFNESS / MASS)
    zeta = DAMPING / (2 * np.sqrt(STIFFNESS * MASS))
    w_d = w_n * np.sqrt(1 - zeta**2)
    exact = (1.0 / STIFFNESS) * (
        1 - np.exp(-zeta * w_n * t) * (np.cos(w_d * t) + zeta * w_n / w_d * np.sin(w_d * t))
    )
    print(f"suspension  worst |x - analytic| = {np.max(np.abs(x - exact)):.3e}")


def main() -> int:
    print(f"fastsim {fastsim.__version__} -> {OUT}")
    export_oscillator(OUT / "oscillator")
    export_suspension(OUT / "suspension")
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(ROOT).as_posix():44s} {p.stat().st_size / 1024:8.1f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
