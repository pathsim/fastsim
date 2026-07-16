# `src/events/` — Discrete event handling

Hybrid-system support: zero-crossing detection, scheduled discrete
events, arbitrary user conditions, and the per-timestep event resolution
loop that slots into `Simulation::timestep`.

## Theory

A continuous-time simulation occasionally needs discrete state changes:
a relay flipping, a bouncing ball reversing velocity, a clock pulse
firing at a given time. Two flavours:

- **Zero-crossing events**: fire when a user-supplied `g(x, t)` changes
  sign. Requires bracketing by `_buffer`, precise localisation via
  bisection during `timestep`.
- **Scheduled events**: fire at pre-computed times (fixed schedule, or
  generated from block state like Clock).

On each event: the engine locks the step to land exactly on the event
instant, invokes the user action, then continues from the post-event
state.

## Implementation

- `event.rs` — the concrete `Event` struct that every flavour builds on:
  holds the fire-time list, active flag, buffered previous value, and the
  lifecycle methods `buffer(t)` / `estimate(t)` / `detect(t)` / `resolve(t)`.
- `eventtype.rs` — the `SimEvent` trait implemented by every event kind;
  defines the dispatch surface `Simulation` sees (`Rc<FastCell<dyn SimEvent>>`).
- `zerocrossing.rs` — `ZeroCrossing`: stores a function `f(x, u, t) → f64`
  and fires when it changes sign. `detect` returns a bracket flag, `resolve`
  runs a bisection down to `EVT_TOLERANCE`.
- `schedule.rs` — `Schedule` (single-shot at a fixed time) and `ScheduleList`
  (an iterator of firing times, typical for Clock / pulse trains).
- `condition.rs` — `Condition`: any user-supplied boolean predicate
  wrapped as an event.
- `impls.rs` — `SimEvent` implementations for each concrete type.
- `mod.rs` — module re-exports.

## How it fits in

- `Simulation` owns `events: Vec<SimEventRef>`. `_buffer` / `_check_events`
  / `_handle_events` in `simulation.rs` drive the event lifecycle each
  step.
- FMU blocks with internal events (e.g. state resets in Model Exchange)
  register `ScheduleList` events via `fmu.rs`.
- `EVT_TOLERANCE` in `constants.rs` is the bisection precision for
  zero-crossing localisation.

## Optimizations

- Events share the step buffer (`history`) with the integrator — no
  separate state copy for the pre/post-event values.
- Zero-crossing uses a two-phase algorithm: coarse bracket (single step),
  then bisection; events that don't bracket skip the bisection entirely.
- The event loop processes events in firing-time order so a single
  step can handle multiple co-firing events without restart.
- **Dense-output localisation** (adaptive runs): before the retry, the
  crossing is root-found on the solvers' step interpolant
  (`Simulation::_localize_events_theta` → `Solver::dense_seek`): each probe
  repositions the dynamic states on the interpolant and refreshes the DAG —
  one system update instead of one full re-integration per secant retry, so
  the single retry lands within tolerance. Default on; opt out via
  `sim.dense_events = false` or `FASTSIM_DENSE_EVENTS=0`.
  The localizer is a *pure accelerator*: it only overrides the legacy secant
  ratio when the probes converge onto a θ every bracketed event is already
  close at (so the retry provably resolves); otherwise — `Condition` events
  (time-based close test), crossings the interpolant cannot reproduce, or
  models whose engines expose no interpolant (e.g. Subsystem wrappers) — it
  returns `None` and the legacy retry cascade with its termination guarantees
  runs unchanged. Probe updates run on throw-away interpolated states:
  `Event::detect` implementations must stay effectively pure (they are today;
  irreversible work belongs in `resolve`), and algebraic-loop nonconvergence
  during a probe is not recorded into the run outcome (`Simulation::_probing`).
