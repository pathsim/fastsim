"""Auto-porting ``Simulation`` facade (issue #17).

Thin Python wrapper around the Rust ``_fastsim.Simulation``. Its only added
responsibility is to make block provenance irrelevant at the call site: any
block in the ``blocks`` list — fastsim-native or an arbitrary pathsim block —
is run through :func:`fastsim.port.port` before reaching the Rust core. fastsim
blocks pass through unchanged (Tier 0); pathsim blocks are accelerated where
possible and shimmed otherwise. Each decision is logged through
``logging.getLogger("fastsim.port")`` and surfaced on stdout when the
simulation's ``log`` flag is set, alongside the Rust run log.

Connections follow the agreed **port-first** convention: fastsim's
``Connection`` only accepts fastsim blocks, so a connected block is already a
fastsim block by the time it reaches here (Tier 0 passthrough). Auto-porting
therefore never has to rewire connection endpoints — it only ports blocks that
arrive un-ported (e.g. a lone pathsim block in the ``blocks`` list).

The Rust ``Simulation`` is not subclassable from Python, so this is a
composition wrapper: unknown attributes/methods delegate to the inner instance,
keeping the full API (``run``, ``reset``, ``_set_solver``, ``steadystate``,
``add_block``, ``time``, ``blocks``, ``__contains__``, ...) transparent.
"""

from __future__ import annotations

import logging
import sys

from fastsim import _fastsim
from fastsim.port import port as _port, log as _port_log


def _ensure_port_logging(enabled):
    """Gate the visibility of port() decisions on the simulation's `log` flag.

    Library-friendly: we never touch ``propagate`` and only attach a fallback
    handler when *nothing* up the chain would otherwise emit (so an app that has
    configured logging keeps full control and we don't double-log). The level is
    set on every call so a later ``log=False`` simulation goes quiet again.
    """
    _port_log.setLevel(logging.INFO if enabled else logging.WARNING)
    if enabled and not _port_log.hasHandlers():
        # stdout, not the StreamHandler default (stderr): these are progress
        # messages, not errors. A host capturing stderr separately (pathview's
        # Pyodide worker) would otherwise render them as errors.
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        _port_log.addHandler(handler)


def _autoport(blocks):
    """Port every block (memoized by identity), preserving order."""
    memo = {}
    out = []
    for b in blocks:
        key = id(b)
        if key not in memo:
            memo[key] = _port(b)
        out.append(memo[key])
    return out


class Simulation:
    """Transient block-diagram simulation.

    Assembles `blocks` and `connections` into a system and integrates it
    through time: fixed or adaptive stepping, explicit and implicit solvers,
    discrete events (zero-crossings, schedules, conditions), algebraic loops
    and hierarchical subsystems. Sink blocks (``Scope``, ``Spectrum``, ...)
    record the trajectory during ``run`` and are read back afterwards.

    Example
    -------

    A damped harmonic oscillator ``x'' + 0.5*x' + 2*x = 0``:

    .. code-block:: python

        from fastsim import Simulation, Connection
        from fastsim.blocks import Integrator, Amplifier, Adder, Scope

        int_v = Integrator(5)       # velocity,  v(0) = 5
        int_x = Integrator(2)       # position,  x(0) = 2
        amp_c = Amplifier(-0.5)     # damping
        amp_k = Amplifier(-2)       # spring
        add = Adder()
        scp = Scope()

        sim = Simulation(
            blocks=[int_v, int_x, amp_c, amp_k, add, scp],
            connections=[
                Connection(int_v, int_x, amp_c),
                Connection(int_x, amp_k, scp),
                Connection(amp_c, add),
                Connection(amp_k, add[1]),
                Connection(add, int_v),
            ],
        )
        sim.run(30)
        time, [x] = scp.read()

    Beyond ``run`` the same model drives ``steadystate`` (DC operating point),
    ``periodic_steady_state`` (limit cycles), ``linearize``, ``compile()``
    (a fused native tape), ``to_c()`` (standalone C99) and ``to_fmu()``.

    pathsim interop
    ---------------
    pathsim blocks may be mixed in freely: every entry in `blocks` is run
    through :func:`fastsim.port.port` before reaching the Rust engine —
    fastsim blocks pass through untouched, pathsim blocks are accelerated
    where possible and shimmed otherwise (decisions are logged via the
    ``fastsim.port`` logger when `log` is set). fastsim ``Connection`` objects
    only accept fastsim blocks, so a *connected* block must be ported before
    wiring (``integ = port(Integrator())`` — the port-first convention);
    auto-porting covers blocks supplied un-ported in the `blocks` list.

    Parameters
    ----------
    blocks : list[Block], optional
        Blocks in the system. fastsim and pathsim blocks may be mixed; pathsim
        blocks are ported automatically.
    connections : list[Connection], optional
        Connections between block ports (built against fastsim / ported blocks).
    Solver : Solver, optional
        Numerical integrator class (default ``SSPRK22``); forwarded to the core
        engine along with any further keyword arguments (``dt``, ``log``,
        ``tolerance_lte_abs``, ...).
    """

    def __init__(self, blocks=None, connections=None, events=None, dt=0.01,
                 dt_min=1e-16, dt_max=None, Solver=None, tolerance_fpi=None,
                 iterations_max=200, log=True, diagnostics=False,
                 tolerance_lte_abs=None, tolerance_lte_rel=None, **solver_kwargs):
        # Explicit, pathsim-compatible signature (not *args/**kwargs) so the
        # constructor is introspectable 1:1 — tooling reads the real parameter
        # set. The two LTE tolerance knobs (`tolerance_lte_abs`/`_rel`) users
        # reach for most are now explicit parameters (issue #31), visible in
        # `inspect.signature`; unknown `**solver_kwargs` are validated by the
        # Rust engine and raise TypeError instead of being silently dropped.
        _ensure_port_logging(log)
        # port() attaches any translated internal events to the block itself, so
        # the Rust simulation tracks them automatically — no harvesting here.
        ported = _autoport(blocks or [])
        conns = connections if connections is not None else []
        # `tolerance_fpi` is retired (kept in the signature for pathsim parity);
        # only forward it when the user set it explicitly, so a default call does
        # not spam a DeprecationWarning on every construction.
        extra = dict(solver_kwargs)
        if tolerance_fpi is not None:
            extra["tolerance_fpi"] = tolerance_fpi
        # Bypass __setattr__ (which delegates to the inner instance).
        self.__dict__["_sim"] = _fastsim.Simulation(
            ported,
            conns,
            events=events if events is not None else [],
            dt=dt,
            dt_min=dt_min,
            dt_max=dt_max,
            Solver=Solver,
            iterations_max=iterations_max,
            log=log,
            diagnostics=diagnostics,
            tolerance_lte_abs=tolerance_lte_abs,
            tolerance_lte_rel=tolerance_lte_rel,
            **extra,
        )

    # -- IR export ---------------------------------------------------------------------

    def to_ir(self, name="model"):
        """Export the assembled model as hierarchical IR.

        Returns a :class:`fastsim.ir.Module`: a typed, inspectable snapshot of
        the model where each block is lowered to its scalar op-graph (for code
        generation / verification) or recorded as a typed ``extern`` call, with
        nested subsystems recursed.
        """
        from fastsim import ir

        return ir.Module.from_json(self.__dict__["_sim"].to_ir_json(name))

    def to_c(self, name="model", **options):
        """Generate standalone C99 source from the assembled model.

        Lowers the model straight to C in-process (no IR JSON round-trip):
        returns a dict mapping each file name to its C source. Files are named
        after the model (``<name>.h`` + ``<name>.c``; the default name yields
        ``model.h`` + ``model.c``) with matching internal ``#include``\\s, so two
        generated models can share one build directory. See
        :meth:`fastsim.ir.Module.to_c` for the full option list (``numeric``,
        ``reductions``, ``structure``, ``layout``, ``solver``, ``api``,
        ``scaffold`` -- the latter additionally emits ``CMakeLists.txt`` and an
        editable ``<name>_main.c`` demo driver). The
        code generator is built into
        the ``fastsim`` extension under the ``codegen`` feature; if the wheel was
        built without it, ``to_c`` raises ``AttributeError``.
        """
        return self.__dict__["_sim"].to_c(name, **options)

    def verify_c(self, name="model", **options):
        """Software-in-the-loop verification of the generated C.

        Compiles the C emitted for this model with a local C99 compiler
        (``$FASTSIM_CC``, ``$CC``, then ``cc``/``clang``/``gcc``), integrates
        the binary and the reference engine (the statically compiled tape)
        over the same fixed-step trajectory, and compares the state
        trajectories sample by sample. Returns a report dict — ``passed``,
        ``max_scaled_error`` (``|c - ref| / (atol + rtol*|ref|)``),
        ``worst_state``, ``worst_time``, ``n_steps``, ``compiler``, ... — see
        the engine docstring for the full key list and scope
        (fixed-step explicit solvers, static-compile subset).

        Keyword options: ``duration`` (default 1.0), ``dt`` (1e-3), ``solver``
        (``"rk4"``), ``numeric`` / ``reductions`` / ``structure`` / ``layout``
        (as in :meth:`to_c`), ``atol`` (1e-9), ``rtol`` (1e-6), ``keep_build``
        (keep the temp build directory for inspection).
        """
        return self.__dict__["_sim"].verify_c(name, **options)

    # -- cooperative streaming generator -----------------------------------------------

    def run_streaming(
        self, duration=10.0, reset=False, adaptive=True, tickrate=10.0, func_callback=None
    ):
        """Run the simulation as a generator, yielding at a fixed rate.

        Drop-in for pathsim's ``run_streaming``: a Python generator that
        advances the engine and yields at a fixed WALL-CLOCK rate (`tickrate`
        yields per real second). Each yield is the return value of
        `func_callback` (or ``None``), so a caller can extract intermediate
        results, inject mutations, or stop between ticks. Unlike the core
        engine's blocking ``run_streaming`` (which runs to completion and
        returns all frames at once), this cedes control between chunks, making
        it suitable for live UIs.

        Parameters
        ----------
        duration : float
            simulation time to advance (in time units)
        reset : bool
            reset the simulation before running (default False)
        adaptive : bool
            use adaptive timesteps if the solver is adaptive (default True)
        tickrate : float
            number of yields per time unit, i.e. ``duration * tickrate`` ticks
            total (default 10)
        func_callback : callable, optional
            called with no arguments at each tick; its return value is yielded

        Yields
        ------
        object
            ``func_callback()`` if callable, else ``None``; a final value is
            always yielded once the run completes.
        """
        import time

        sim = self.__dict__["_sim"]
        sim.run_begin(reset, duration)

        t0 = sim.time
        t_end = t0 + duration
        # `tickrate` is a WALL-CLOCK rate (yields per real second), like pathsim,
        # not a sim-time divisor. We advance the engine in sim-time slices and
        # grow/shrink the slice so roughly one wall-clock tick elapses per chunk.
        # A fast sim then yields a handful of times instead of duration*tickrate,
        # keeping the costly per-tick extract (func_callback: read all scopes +
        # serialize) rare. Time is measured in Python (perf_counter) — robust in
        # Pyodide/WASM, no reliance on Rust std::time::Instant.
        tick = 1.0 / tickrate if tickrate and tickrate > 0 else 0.0
        chunk = duration / max(1.0, tickrate)
        t = t0

        while sim.active and sim.time < t_end:
            target = min(t + chunk, t_end)
            w0 = time.perf_counter()
            sim.run_until(target, t_end, adaptive)
            elapsed = time.perf_counter() - w0
            t = sim.time
            yield func_callback() if callable(func_callback) else None
            # Converge the slice toward `tick` wall-seconds (bounded growth).
            if tick > 0.0 and elapsed > 1e-6:
                chunk *= max(0.2, min(8.0, tick / elapsed))

        # Finalize the progress tracker (logs FINISHED) before the last frame.
        sim.run_end()

        # Final yield with the complete results (mirrors pathsim).
        yield func_callback() if callable(func_callback) else None

    def add_block(self, arg):
        """Add a block, or a list of blocks.  Accepts either a single `Block`
        or any iterable of `Block`.  If a `run()` is currently active
        (e.g. called from inside a `Schedule.func_act` callback), the
        operation is queued onto the simulation's pending-ops queue and
        applied at the next timestep boundary; otherwise it is applied
        synchronously.  The bulk path stays inside Rust — no per-block
        Python⇄Rust boundary crossing.
        """
        return self.__dict__["_sim"].add_block(arg)

    def add_connection(self, arg):
        """Add a connection, or a list of connections."""
        return self.__dict__["_sim"].add_connection(arg)

    def add_event(self, event):
        """Add an event (or list of events) to the simulation."""
        return self.__dict__["_sim"].add_event(event)

    def compile(self):
        """Statically compile this model into a fused `dX/dt = F(X, t)` tape (see
        `compile`). Discrete and event-driven blocks are supported (memory joins a
        global `M` vector; block-internal zero-cross/schedule/condition events drive
        an event-aware run loop). Raises `ValueError` with a precise reason if the
        model is outside the static subset (opaque/extern block whose math lowers to
        a call, algebraic loop, no continuous state, simulation-level events). Sinks
        become recorded taps; subsystems are flattened.

        The compiled simulation inherits this simulation's solver, adaptive
        tolerances (`tolerance_lte_abs`/`tolerance_lte_rel`), timestep `dt` and
        logging, so a compiled run integrates the same problem with the same
        method by default. Override any of these afterwards via `set_solver` /
        `dt` / `log` on the returned object.
        """
        return self.__dict__["_sim"].compile()

    def delinearize(self):
        """Revert a previous linearization, restoring the original nonlinear blocks."""
        return self.__dict__["_sim"].delinearize()

    def enable_wct_trace(self):
        """Enable per-timestep wall-clock recording.  Subsequent `run` /
        `timestep` calls append elapsed seconds for each step into an
        internal buffer.  Negligible overhead (~30 ns/step) when enabled,
        zero overhead when off.
        """
        return self.__dict__["_sim"].enable_wct_trace()

    def linearize(self):
        """Linearize the system about the current operating point (state and inputs), replacing nonlinear blocks with their local linear approximation."""
        return self.__dict__["_sim"].linearize()

    def pending_ops(self):
        """Return a handle on the simulation's mutation queue.  Equivalent
        to calling `sim.add_block(...)` etc. directly from a callback,
        kept for explicit-batching use cases.
        """
        return self.__dict__["_sim"].pending_ops()

    def periodic_steady_state(self, period, Solver=None, tolerance_lte_abs=None, tolerance_lte_rel=None, adaptive=True, reset=False):
        """Find the periodic steady-state limit cycle of period `period` by
        matrix-free Anderson-accelerated shooting on the period map
        `g(x_0) = x(T; x_0)`.

        One outer iteration:

        1. Integrate the system over `[0, T]` with the inner ODE solver
           (a regular transient `run(period, ...)`).
        2. Per dynamic block, run one matrix-free Anderson step on
           `(x_start, x_end)`, mutating `x_start` toward the limit-cycle
           period-start state.
        3. Check the max WRMS-scaled residual `‖x(T) − x(0)‖` across all
           dynamic blocks against the simulation's `NLS_COEF` threshold
           (same convergence semantics as every other implicit-stage /
           steady-state residual).
        4. If not converged: reset `sim.time = 0` and event schedules,
           repeat from step 1.

        After convergence, one final transient run over `[0, T]` records
        the converged limit-cycle trajectory in Scope blocks.

        Anderson needs only function evaluations (one period integration
        per outer iteration) — no monodromy matrix `Φ = ∂x(T)/∂x(0)` to
        assemble or factorize.  Converges in roughly 5–15 iterations on
        smooth, mildly-coupled periodic systems.  DAE blocks pass through
        transparently — their `engine_postprocess` installs the
        appropriate `StageBuilder` on the PSS-augmented engine, exactly
        as with any other solver factory.

        PSS pays off when the natural settling time is long relative to
        the forcing period (high-Q resonators, weakly-damped loops, large
        LC filters).  On strongly-damped systems where a plain transient
        settles in a handful of periods, `sim.run()` is faster — the
        shooting iteration carries a fixed overhead (warm-up + final
        sample-run) that only amortizes when many periods would otherwise
        be needed.

        Parameters
        ----------
        period : float
            Period length `T` (in simulation time units).  Must be positive.
        Solver : type, optional
            Inner ODE solver class (e.g. `RKDP54`, `ESDIRK43`, `GEAR52A`).
            Defaults to the simulation's current solver.
        tolerance_lte_abs : float, optional
            Absolute WRMS weight for both the inner solver's LTE control
            and the outer shooting convergence test.  Defaults to the
            current solver's setting.
        tolerance_lte_rel : float, optional
            Relative WRMS weight, same dual role as above.  Defaults to
            the current solver's setting.
        adaptive : bool, optional
            Enable adaptive timestepping during the period integration.
            Only honored when the inner solver is adaptive.  Default `True`.
        reset : bool, optional
            If `True`, restore all blocks to their `initial_value` before
            shooting starts.  If `False`, seed shooting from the current
            simulation state — useful after a transient warm-up that
            produces a good initial guess.  Default `False`.

        Returns
        -------
        dict
            Aggregate run statistics summed across all shooting iterations
            plus the final sample run: `total_steps`, `successful_steps`,
            `rejected_steps`, `total_evals`, `total_solver_its`, `runtime_ms`.

        Notes
        -----
        State-flipping events (relays, hysteresis, mode-switching) make
        `x(T; x_0)` non-smooth at the boundaries where the event
        activation changes, which may stall Anderson convergence.  If
        shooting fails to converge within `SIM_PSS_ITERATIONS_MAX` (50)
        iterations, a warning is logged and the best-so-far state is
        retained.
        """
        return self.__dict__["_sim"].periodic_steady_state(period, Solver, tolerance_lte_abs, tolerance_lte_rel, adaptive, reset)

    def plot(self):
        """Plot the recorded results from all Scope blocks in the simulation."""
        return self.__dict__["_sim"].plot()

    def remove_block(self, arg):
        """Remove a block (or list of blocks) from the simulation."""
        return self.__dict__["_sim"].remove_block(arg)

    def remove_connection(self, arg):
        """Remove a connection (or list of connections) from the simulation."""
        return self.__dict__["_sim"].remove_connection(arg)

    def remove_event(self, event):
        """Remove an event (or list of events) from the simulation."""
        return self.__dict__["_sim"].remove_event(event)

    def reset(self, time=None):
        """Reset all blocks to their initial state and the simulation time to zero (or to ``time`` if given)."""
        return self.__dict__["_sim"].reset(time)

    def run(self, duration=10.0, reset=False, adaptive=True):
        """Run the simulation for a given duration."""
        return self.__dict__["_sim"].run(duration, reset, adaptive)

    def run_begin(self, reset=False, duration=10.0):
        """Begin a chunked, cooperative run (for the streaming generator).
        Does the one-time setup (optional reset + initial eval/sample); pair
        with repeated `run_until` calls. Unlike `run_streaming`, this path is
        sim-time driven (no wall-clock), so it is WASM-safe and lets Python
        own the yield/step loop, injecting mutations between chunks.
        """
        return self.__dict__["_sim"].run_begin(reset, duration)

    def run_end(self):
        """Finalize a chunked streaming run (logs FINISHED/INTERRUPTED). Call once
        after the last `run_until`.
        """
        return self.__dict__["_sim"].run_end()

    def run_realtime(self, duration=10.0, reset=False, adaptive=True, tickrate=30.0, speed=1.0, func_callback=None):
        """Run synchronized to wall-clock time with optional speed factor."""
        return self.__dict__["_sim"].run_realtime(duration, reset, adaptive, tickrate, speed, func_callback)

    def run_until(self, target_time, end_time, adaptive=True):
        """Advance up to `target_time` (a chunk boundary). `end_time` is the true
        run end (for adaptive overshoot prevention). Returns step counts.
        """
        return self.__dict__["_sim"].run_until(target_time, end_time, adaptive)

    def steadystate(self, reset):
        """Find the steady-state (DC operating point) of the system by root-finding on the residual rather than time integration."""
        return self.__dict__["_sim"].steadystate(reset)

    def stop(self):
        """Signal a running simulation to stop cleanly at the next timestep boundary."""
        return self.__dict__["_sim"].stop()

    def take_wct_trace(self):
        """Drain the per-timestep wall-clock buffer and return the recorded
        times in seconds.  Leaves recording enabled with an empty buffer.
        Returns an empty list if tracing was never enabled.
        """
        return self.__dict__["_sim"].take_wct_trace()

    def to_fmu(self, path, name='model', *, start_time=None, stop_time=None, tolerance=None, step_size=None, instantiation_token=None):
        """Export this model as a source FMU (FMI 3.0, Model Exchange) written to
        `path` (conventionally `*.fmu`).

        Builds the IR straight from the live model (the same `module_from_sim`
        path as `compile`/`to_c`), lowers it through the struct-API C backend,
        wraps it in the FMI Model-Exchange C layer, and zips the C sources with a
        generated `modelDescription.xml`. The result is a *source* FMU: it ships
        the C plus a `buildDescription.xml` so an importer compiles it on its own
        platform.

        Phase-1 scope: closed (input-free) continuous models with state and no
        events. Raises `ValueError` for a model outside that subset (no
        continuous state, events, or an opaque block the backend cannot lower).
        The optional `start_time` / `stop_time` / `tolerance` / `step_size`
        populate `<DefaultExperiment>`; `instantiation_token` overrides the
        default `{fastsim-<id>}`.
        """
        return self.__dict__["_sim"].to_fmu(path, name, start_time=start_time, stop_time=stop_time, tolerance=tolerance, step_size=step_size, instantiation_token=instantiation_token)

    def to_ir_json(self, name='model'):
        """Export the assembled model as hierarchical IR (JSON). Each block is
        either lowered to its op-graph (for codegen / verification) or recorded
        as a typed `extern` call; nested subsystems recurse. See `src/ir`.
        """
        return self.__dict__["_sim"].to_ir_json(name)

    # -- transparent delegation to the Rust simulation ---------------------------------

    def __getattr__(self, name):
        # Only invoked when normal lookup fails; never for "_sim" itself once set.
        if name == "_sim":
            raise AttributeError(name)
        return getattr(self.__dict__["_sim"], name)

    def __setattr__(self, name, value):
        # Engine attributes (dt, log, solver knobs, ...) delegate to the Rust
        # simulation. Anything the engine does not know falls back to the
        # wrapper's own __dict__ — pathsim simulations are plain Python objects
        # users freely tag with extra attributes, and a drop-in replacement
        # must not turn that into an AttributeError.
        try:
            setattr(self.__dict__["_sim"], name, value)
        except AttributeError:
            self.__dict__[name] = value

    def __contains__(self, item):
        # Python looks up dunders on the type, so __getattr__ can't forward this.
        return item in self.__dict__["_sim"]

    def __repr__(self):
        return repr(self.__dict__["_sim"])
