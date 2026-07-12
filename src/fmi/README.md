# `src/fmi/` — FMI 3.0 bindings

Raw FFI layer and safe wrappers for loading and driving FMI 3.0 FMUs
(Functional Mock-up Units) as fastsim blocks. Supports both Co-Simulation
(CS) and Model Exchange (ME) mode.

## Theory

An FMU is a zip archive containing a shared library (the model compiled
to C) and a `modelDescription.xml` declaring its variables, units, and
capability flags. The FMI 3.0 C API is a set of `fmi3*` functions
(instantiation, getters/setters by value-reference, step/derivatives, event
iteration, discrete-state update). This module provides:

1. **Raw bindings** to the C API (`bindings.rs`, matching `fmi3Functions.h`).
2. **Safe wrappers** (`instance.rs`) that RAII-manage the FMU instance,
   validate state transitions, and route errors through Rust's `Result`.
3. **Model-description parsing** (`model_description.rs`) extracts variable
   metadata, value-references, initial-values, and solver capabilities from
   the XML.
4. **Cross-platform loading** (`platform.rs`, `unzip.rs`) handles
   OS-specific shared-library discovery and archive extraction.

The resulting `Instance<Cs>` or `Instance<Me>` is consumed by
`blocks/fmu.rs` to build a fastsim block.

## Implementation

- `mod.rs` — module entry. Also defines `FmiError` (thiserror-based error
  enum covering I/O, ZIP, XML, dlopen, platform-not-supported, FMI status
  failures) and `FmiStatus` (FMI 3.0 return-status enum with `from_raw` /
  `is_ok_or_warning` helpers).
- `bindings.rs` — `extern "C"` declarations for the FMI 3.0 symbols,
  opaque types (`fmi3Instance`, `fmi3ValueReference`, ...), FFI structs.
  No allocations, no logic — pure FFI.
- `instance.rs` — `Instance<K>` typed on unit-struct phantom tags `Me`
  (Model Exchange) and `Cs` (Co-Simulation). Owns the loaded library, the
  FMU instance pointer, and the resolved function-pointer table. Shared
  methods live in `impl<K> Instance<K>`; mode-specific methods
  (`do_step` for Cs, `get_continuous_state_derivatives` /
  `update_discrete_states` / `get_directional_derivative` for Me) are on
  the specialised impls. `DiscreteStateUpdate` captures the
  event-iteration result flags (`discrete_states_need_update`,
  `terminate_simulation`, …).
  The optional FMI 3.0 capability `providesDirectionalDerivatives` is
  exposed via `supports_directional_derivatives()` + `get_directional_
  derivative(unknowns, knowns, seed, sensitivity)` — callers in
  `blocks/fmu.rs` use this to build an analytical `∂ẋ/∂x` Jacobian
  column-by-column, bypassing the numerical FD fallback.
- `callbacks.rs` — the logger + allocator callbacks passed to FMU
  instantiation.
- `model_description.rs` — XML parser (via `roxmltree`) yielding a typed
  `ModelDescription` with variables, scalars, initial values.
- `platform.rs` — platform-specific binary prefix (`linux64`, `darwin64`,
  `win64`) and library suffix.
- `unzip.rs` — extraction to a temp directory via `zip` crate.
- `mod.rs` — module glue.

## How it fits in

- `blocks/fmu.rs` is the only caller: it constructs an `Instance`,
  registers its outputs/inputs as fastsim ports, and installs `f_dyn`
  / `f_alg` closures that forward through the instance.
- `FMI_INITIALIZATION_TOL` in `constants.rs` is the tolerance passed to
  `fmiEnterInitializationMode`.
- `utils/register.rs` provides the value-reference ↔ port-index mapping.

## Optimizations

- **Zero-copy setters**: the FFI takes `*const fmi3ValueReference` and
  `*const f64` directly from Rust slices — no conversion in the hot path.
- **Import-table caching**: function pointers resolved once at load time,
  not per call.
- **Archive kept live**: the extracted directory and the `libloading`
  `Library` handle live on the instance so subsequent calls don't re-unzip
  or re-dlopen.
- Typed mode (`Instance<Cs>` vs `Instance<Me>`) via phantom-type tag —
  mode-specific methods only appear on the correct type, no runtime
  dispatch cost.
