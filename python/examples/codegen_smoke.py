"""End-to-end smoke test for the codegen Python surface.

Builds a decay model (x' = -x, x(0) = 1), runs it in fastsim, generates C via
`sim.to_c(...)`, compiles + runs that C, and checks both agree with e^-1.
Run against a fastsim built with the `codegen` feature (the default wheel /
`maturin develop --features python,codegen`).
"""
import os
import subprocess
import sys
import tempfile

from fastsim import Simulation, Connection
from fastsim.blocks import Integrator, Amplifier

CC = os.environ.get("CC", "gcc")


def build_run(files, main, t_end):
    with tempfile.TemporaryDirectory() as d:
        for name, src in files.items():
            with open(os.path.join(d, name), "w") as f:
                f.write(src)
        with open(os.path.join(d, "main.c"), "w") as f:
            f.write('#include <stdio.h>\n#include "model.h"\n' + main)
        cs = [n for n in files if n.endswith(".c")] + ["main.c"]
        subprocess.run([CC, *cs, "-O0", "-o", "m.exe", "-lm"], cwd=d, check=True)
        out = subprocess.run([os.path.join(d, "m.exe")], capture_output=True, text=True, check=True)
        return [float(x) for x in out.stdout.split()]


def main():
    integ = Integrator(1.0)
    amp = Amplifier(-1.0)
    sim = Simulation([integ, amp], [Connection(integ, amp), Connection(amp, integ)])
    sim.run(1.0)  # assemble + reference run

    # Default (Compact) layout.
    files = sim.to_c("decay")
    assert set(files) == {"model.h", "model.c"}, sorted(files)
    print("compact files:", sorted(files))

    # Struct ("rtModel") API: entry points are prefixed with the model name so two
    # generated models can be linked into one binary without symbol collisions.
    c_main = (
        "int main(void){decay_t m;decay_init(&m);"
        "decay_run(&m,1.0,1e-3);printf(\"%.17g\", m.x[0]);return 0;}\n"
    )
    (x,) = build_run(files, c_main, 1.0)
    import math
    assert abs(x - math.exp(-1.0)) < 1e-4, f"compact C x={x} e^-1={math.exp(-1.0)}"
    print(f"compact decay: C x(1)={x:.6f}  e^-1={math.exp(-1.0):.6f}  OK")

    # Library layout: 6 files (model/blocks/solver, .h + .c).
    lib = sim.to_c("decay", layout="library")
    assert set(lib) == {"model.h", "model.c", "blocks.h", "blocks.c", "solver.h", "solver.c"}, sorted(lib)
    print("library files:", sorted(lib))

    # Bad option -> ValueError.
    try:
        sim.to_c("decay", layout="libary")
        raise SystemExit("expected ValueError for bad layout")
    except ValueError as e:
        print("bad option raised ValueError:", str(e)[:60])

    print("ALL OK")


if __name__ == "__main__":
    sys.exit(main())
