# Commercial Licensing

fastsim is distributed under the [PolyForm Noncommercial License 1.0.0](LICENSE).
That license permits **any noncommercial use** — research, teaching, academic
work, evaluation, personal and hobby projects — at no cost. **Commercial use
requires a commercial license.** This page explains, in plain language, where
that line falls and what a commercial license covers.

For a quote or to discuss terms, contact **info@pathsim.org**.

---

## When you need a commercial license

You need a commercial license if you use fastsim in the course of running a
business or delivering a commercial product or service. Two distinct scopes
matter, because they are priced and governed separately:

### 1. Engine use (running the simulator)

Using the fastsim engine — the Python package, the Rust crate, the solvers, the
JIT, FMU import — to do work for a commercial purpose. Examples:

- running simulations as part of a commercial product, service, or paid consulting;
- embedding the Python/native library in a commercial application or internal
  tooling at a for-profit company;
- offering fastsim-powered analysis to customers.

### 2. Shipping generated code (the "Output" terms)

This is the heart of the code-generation business model and deserves its own
attention. The C code emitted by `Simulation.to_c(...)` (and `to_fmu(...)`) is
**"Output"** under the PolyForm Noncommercial License. **The noncommercial
limitation travels with the generated code** — every generated file is stamped
with the notice. So:

- Generating C from a fastsim model and **shipping that C** (or an FMU, a
  firmware image, a HIL binary, a library, or any artifact built from it) **in a
  commercial product requires a commercial license**, *even if* the generation
  was done during noncommercial evaluation.
- The generated code is the deliverable a commercial customer most often wants
  (an embedded controller, a plant model for a HIL rig, an FMU for a toolchain).
  Distributing it commercially is exactly what the Output terms govern.

A commercial license lifts the noncommercial limitation on the generated Output
so you can compile it into, and distribute it as part of, a commercial product.

---

## What noncommercial use always allows (no license needed)

- Research, coursework, teaching, and academic publications.
- Personal and hobby projects.
- Evaluating fastsim — including generating and compiling C — to decide whether
  to adopt it commercially.
- Contributing to fastsim itself.

---

## The fully open-source alternative

If you need an option with **no field-of-use restriction**, the pure-Python
engine [pathsim](https://github.com/pathsim) is available separately under the
**MIT License**. fastsim is a drop-in replacement for pathsim's Python API, so
prototyping against pathsim and moving to fastsim (or vice versa) is
straightforward. Note that pathsim does not provide the C code generator.

---

## Summary

| Activity                                             | Noncommercial license | Commercial license |
|------------------------------------------------------|:---------------------:|:------------------:|
| Research / teaching / personal use                   | ✅                    | ✅                 |
| Evaluating the engine and codegen commercially       | ✅                    | ✅                 |
| Running the engine for a commercial purpose          | ❌                    | ✅                 |
| Shipping generated C / FMUs in a commercial product  | ❌                    | ✅                 |

Contact **info@pathsim.org** for commercial terms.
