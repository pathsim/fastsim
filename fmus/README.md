# FMUs exported by FastSim

FMI 3.0 FMUs produced by FastSim's `to_fmu()`, published so other tools can
import them and check FastSim's export for themselves.

| Model | Interfaces | Inputs | Outputs | States |
|---|---|---|---|---|
| `oscillator` | Model Exchange + Co-Simulation | — | `ODE_y0`, `ODE_y1` | 2 |
| `suspension` | Model Exchange + Co-Simulation | `in` | `ODE_y0`, `ODE_y1` | 2 |

`oscillator` is a damped linear oscillator released from rest at `x = 1`.
`suspension` is a forced mass–spring–damper exported from a `Subsystem`, so it
also covers the component case where a port enters through an `Interface`.

## Files

Each directory follows the FMI Cross-Check file convention
([FMI-CROSS-CHECK-RULES.md](https://github.com/modelica/fmi-cross-check/blob/master/FMI-CROSS-CHECK-RULES.md)
§9.1.4), which is the shape importing tools already know how to consume:

- `{name}.fmu` — the FMU
- `{name}_ref.csv` — reference solution, computed by FastSim itself
- `{name}_ref.opt` — the experiment settings that solution was produced with
- `{name}_in.csv` — input trajectory, for models that have inputs

The cross-check repository itself covers FMI 1.0 and 2.0 only, so these cannot
be submitted there; the layout is flat (`fmus/{model}/`) rather than the nested
`{version}/{type}/{platform}/{tool}/{version}/` tree that repository uses.

## Sources and binaries

FastSim exports **source FMUs**: the archive carries the generated C together
with a `buildDescription.xml`, so any importing tool can compile it for its own
platform. The published archives here additionally embed pre-built binaries
for:

- `x86_64-windows`
- `x86_64-linux`

so tools without a C toolchain can use them directly. On other platforms the
sources still compile locally — FMPy does it in one call:

```python
from fmpy.build import build_platform_binary
build_platform_binary("path/to/unpacked/fmu")
```

Both interfaces come from one archive — the same generated C backs Model
Exchange and Co-Simulation.

## Reproducing

Regenerate the artifacts, then embed the platform binaries (the second step
needs a local C compiler for the Windows build and Docker for the Linux one):

```
python scripts/export_reference_fmus.py
python scripts/add_fmu_binaries.py
```

Check them the way an importing tool would — validate, compile, simulate in both
interfaces, replay `_in.csv`, and compare against `_ref.csv`:

```
python scripts/check_exported_fmus.py
```

Last run, with FMPy 0.3.26 on `x86_64-windows`:

```
oscillator
  validate_fmu           no problems
  ModelExchange          worst |FMU - reference| = 1.529e-06   ok
  CoSimulation           worst |FMU - reference| = 1.847e-10   ok
suspension
  validate_fmu           no problems
  input trajectory       in (2 points)
  ModelExchange          worst |FMU - reference| = 1.461e-06   ok
  CoSimulation           worst |FMU - reference| = 4.749e-12   ok
```

Model Exchange is integrated by the importer's own adaptive solver at the
`RelTol` the `_ref.opt` declares, so it sits further from the fixed-step
reference than Co-Simulation, which replays the same step size FastSim used.

The references themselves are checked against closed forms rather than against
the FMU: `oscillator` to 3.206e-11 and `suspension` to 1.906e-12, both reported
by the generator.

## Input trajectories

`suspension_in.csv` holds a constant force. §9.1.4 fixes linear interpolation
between the points but not how finely an importer resamples the trajectory —
per communication step, or per solver stage — and on a shaped signal those
choices disagree at the 1e-4 level, which would blunt the comparison. A constant
is identical under every scheme, so whatever deviation is left belongs to the
FMU.
