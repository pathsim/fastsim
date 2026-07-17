"""Port arbitrary pathsim block *instances* into a fastsim Simulation.

Issue #17 — tiered, instance-level porting with logging.

A block handed to a fastsim ``Simulation`` may be a fastsim block already, a
thin pathsim subclass that fastsim can fully accelerate, or an arbitrary
pathsim block with custom integration logic. :func:`port` classifies the block
and applies the highest-acceleration strategy that is still faithful:

* **Tier 0 — passthrough.** Already a fastsim block → returned unchanged.
* **Tier 1 — accelerate.** The block exposes clean ``op_dyn`` (and optionally
  ``op_alg``) operators and does not override the engine hooks
  (``step`` / ``solve`` / ``update``) beyond the generic pathsim bases
  (``ODE`` / ``DynamicalSystem``). Its raw RHS callables are extracted and
  rebuilt as a fastsim-native ``DynamicalSystem``, which JIT-traces them into
  the Rust core (full speed; transparent Python-callback fallback if a trace
  fails).
* **Tier 3 — shim fallback.** Anything else (custom ``step`` / ``solve``,
  no operators) → the **engine-shim / RHS-capture** bridge: the block keeps its
  own methods, but ``block.engine`` is replaced by a :class:`CaptureShim` that
  only captures the RHS while fastsim's own (Rust) engine does the integration.
  fastsim owns the single stage loop, so there is no second engine and no stage
  synchronization to get wrong. Blocks without ``initial_value`` (algebraic /
  internal-solver blocks, e.g. a ``BVP1D`` subclass with a custom ``update``)
  take the **algebraic shim**: their own ``update(t)`` drives the outputs
  through a Python-callback ``DynamicalFunction``.

(Tier 2 — partial acceleration, e.g. a JIT-able ``op_alg`` with custom
dynamics — is a future refinement; such blocks currently take Tier 3.)

Every decision is logged through ``logging.getLogger("fastsim.port")`` so the
porting is observable. The eventual ``Simulation`` facade applies :func:`port`
automatically; connections are built against the **returned** (ported) block.
"""

from __future__ import annotations

import logging

import numpy as np

from fastsim import _fastsim

log = logging.getLogger("fastsim.port")

# Engine/lifecycle hooks whose overriding means acceleration would silently drop
# custom behaviour. fastsim's block does not expose these as Python methods, so
# the adapter cannot detect such overrides — we must check them pathsim-side
# (against the defining class in the MRO) before accelerating.
_ENGINE_HOOKS = ("step", "solve", "update", "buffer", "revert", "sample", "reset")


class CaptureShim:
    """Stand-in for a pathsim ``Solver`` that captures the RHS, never integrates.

    Installed on a wrapped pathsim block as ``block.engine`` for Tier-3 ports.
    The block's own ``step`` / ``solve`` call ``engine.step(f, dt)`` /
    ``engine.solve(f, J, dt)`` — we store ``f`` so the fastsim wrapper can
    forward it to the real engine. ``state`` is written by the wrapper before
    each evaluation so the wrapped block sees the current integrator state.
    """

    def __init__(self, initial_value):
        self._x = np.atleast_1d(np.asarray(initial_value, dtype=float)).copy()
        self._f = None

    # -- state access used by the wrapped block's update / op_dyn / op_alg --

    @property
    def state(self):
        return self._x

    @state.setter
    def state(self, value):
        self._x = np.atleast_1d(np.asarray(value, dtype=float))

    def get(self):
        return self._x

    def set(self, x):
        self._x = np.atleast_1d(np.asarray(x, dtype=float))

    def __len__(self):
        return len(self._x)

    def __bool__(self):
        # pathsim blocks gate dynamic behaviour on ``if self.engine:`` — a
        # shim is always a live engine.
        return True

    # -- capture points: store f, do NOT integrate --

    def step(self, f, dt):
        self._f = np.atleast_1d(np.asarray(f, dtype=float)).copy()
        return True, 0.0, None

    def solve(self, f, J, dt):
        self._f = np.atleast_1d(np.asarray(f, dtype=float)).copy()
        return 0.0

    # -- no-ops: fastsim's real engine owns buffering / reverting / history --

    def buffer(self, dt):
        pass

    def revert(self):
        pass

    def reset(self, initial_value=None):
        if initial_value is not None:
            self._x = np.atleast_1d(np.asarray(initial_value, dtype=float)).copy()
        self._f = None


# -- classification helpers ------------------------------------------------------------

_TIER1_BASES_CACHE = None


def _tier1_bases():
    """Lazily resolve (generic pathsim dynamic bases, pathsim Block).

    Built lazily so importing fastsim does not require pathsim. Returns
    ``((), None)`` when pathsim is unavailable, which forces every non-fastsim
    block down the Tier-3 shim path.
    """
    global _TIER1_BASES_CACHE
    if _TIER1_BASES_CACHE is not None:
        return _TIER1_BASES_CACHE
    try:
        from pathsim.blocks import ODE, DynamicalSystem
        from pathsim.blocks._block import Block as PsBlock
        _TIER1_BASES_CACHE = ((ODE, DynamicalSystem), PsBlock)
    except Exception:
        _TIER1_BASES_CACHE = ((), None)
    return _TIER1_BASES_CACHE


def _is_fastsim_block(obj) -> bool:
    return isinstance(obj, _fastsim.Block)


def _defining_class(cls, name):
    """Class in ``cls.__mro__`` that actually defines attribute ``name``."""
    for c in cls.__mro__:
        if name in c.__dict__:
            return c
    return None


def _allowed_hook_classes():
    """Set of classes a thin subclass may inherit its engine hooks from (the
    generic pathsim dynamic bases + pathsim ``Block``), or ``None`` if pathsim
    is unavailable."""
    bases, ps_block = _tier1_bases()
    if not bases:
        return None
    allowed = set(bases)
    if ps_block is not None:
        allowed.add(ps_block)
    return allowed


def _overridden_hook(cls, allowed):
    """Name of the first engine/lifecycle hook ``cls`` overrides relative to the
    generic bases, or ``None`` if it overrides none. Shared by the instance- and
    class-level acceleration checks."""
    for hook in _ENGINE_HOOKS:
        if _defining_class(cls, hook) not in allowed:
            return hook
    return None


def _tier1_reason(block):
    """Return ``None`` if the block qualifies for Tier-1 acceleration, else a
    short human-readable reason why it does not (used for the fallback log).
    """
    # Mixed-signal blocks keep their internal events; the shim path keeps the
    # pathsim block alive so its event callbacks (which close over it) stay valid
    # and can be forwarded — operator extraction would discard them.
    if getattr(block, "events", None):
        return "has internal events (mixed-signal); shim forwards them"
    allowed = _allowed_hook_classes()
    if allowed is None:
        return "pathsim not importable"
    if getattr(block, "op_dyn", None) is None:
        return "no op_dyn operator (custom dynamics)"
    if getattr(block, "initial_value", None) is None:
        return "no initial_value"
    hook = _overridden_hook(type(block), allowed)
    if hook is not None:
        return f"overrides {hook}() in {_defining_class(type(block), hook).__name__}"
    return None


def _has_passthrough(block, iv):
    """Whether the block has algebraic feedthrough, without leaking engine state.

    ``DynamicalSystem.__len__`` probes ``op_alg``'s u-dependence via the engine
    state, so a stateless (engine-less) block needs one to evaluate it. We lend a
    temporary :class:`CaptureShim` only when none is present and restore the
    original engine afterwards — so port() never mutates a block it merely
    inspects (Tier-1 accelerate discards the wrapped block).
    """
    prev = block.engine
    if prev is None:
        block.engine = CaptureShim(iv)
    try:
        return len(block) > 0
    finally:
        block.engine = prev


# -- porting strategies ----------------------------------------------------------------

def _accelerate(block):
    """Tier 1: extract op_dyn / op_alg and rebuild as a JIT-able fastsim block."""
    from fastsim import blocks as fs_blocks

    # The DynamicOperator is itself callable (op(x, u, t)); fall back to it if
    # the private raw-function attribute is ever renamed in pathsim.
    op_dyn = block.op_dyn
    func_dyn = getattr(op_dyn, "_func", op_dyn)
    jac_dyn = getattr(op_dyn, "_jac_x", None)

    op_alg = getattr(block, "op_alg", None)
    if op_alg is not None:
        func_alg = getattr(op_alg, "_func", op_alg)
    else:
        # ODE-style block: y = x (no algebraic output operator).
        func_alg = lambda x, u, t: x

    iv = block.initial_value
    return fs_blocks.DynamicalSystem(
        func_dyn, func_alg, iv,
        has_passthrough=_has_passthrough(block, iv),
        jac_dyn=jac_dyn,
    )


# Internal event types fastsim can reconstruct. pathsim and fastsim share the
# event API (func_evt/func_act/tolerance, schedule timing fields) and identical
# detection logic (1:1 port), so a translated event detects bit-identically.
_FUNC_EVT_EVENTS = ("ZeroCrossing", "ZeroCrossingUp", "ZeroCrossingDown", "Condition")


def _translate_one(ev, wrapped_act):
    """Reconstruct a single pathsim event as the equivalent fastsim event, or
    ``None`` if its type has no translation yet."""
    from fastsim import events as fs_events

    name = type(ev).__name__
    if name in _FUNC_EVT_EVENTS:
        # ZeroCrossing family + Condition: func_evt + func_act + tolerance.
        return getattr(fs_events, name)(ev.func_evt, wrapped_act, getattr(ev, "tolerance", 1e-4))
    if name == "ScheduleList":  # subclass of Schedule — check first
        return fs_events.ScheduleList(list(ev.times_evt), wrapped_act, getattr(ev, "tolerance", 1e-16))
    if name == "Schedule":
        return fs_events.Schedule(
            getattr(ev, "t_start", 0.0), getattr(ev, "t_end", None),
            getattr(ev, "t_period", 1.0), wrapped_act, getattr(ev, "tolerance", 1e-16),
        )
    return None


def _make_wrapped_act(func_act, fs_block, shim):
    """Wrap a pathsim event action so its state mutation reaches fastsim's engine.

    The original ``func_act`` mutates the wrapped block (i.e. the shim's state);
    fastsim's real engine must adopt that, otherwise the next ``update`` syncs
    the engine state back onto the shim and clobbers the action. Algebraic
    shims (``shim is None``) carry no engine state — the action's mutations
    live on the wrapped block itself and are picked up by the next ``update``.
    """
    if shim is None:
        return func_act

    def wrapped(t):
        func_act(t)
        fs_block.state = np.atleast_1d(shim.get()).tolist()
    return wrapped


def _translate_events(ps_block, fs_block, shim):
    """Translate a wrapped block's internal events to fastsim events.

    Returns ``(translated, unsupported_type_names)``. Event functions close over
    the (still-live) pathsim block, whose state/inputs/outputs the shim keeps
    synced each update, so detection reads the current simulation state.
    """
    translated, unsupported = [], []
    for ev in getattr(ps_block, "events", None) or []:
        act = getattr(ev, "func_act", None)
        wrapped_act = _make_wrapped_act(act, fs_block, shim) if act is not None else None
        fs_ev = _translate_one(ev, wrapped_act)
        if fs_ev is None:
            unsupported.append(type(ev).__name__)
        else:
            translated.append(fs_ev)
    return translated, unsupported


def _port_algebraic_shim(block):
    """Tier 3b: algebraic shim — faithful Python fallback for stateless blocks.

    A pathsim block without ``initial_value`` computes its outputs purely from
    its inputs in ``update(t)`` (directly, or through an internal solver like
    ``BVP1D``'s collocation — including any custom ``update`` override, e.g.
    pathsim-chem's ``GLC`` post-processing). The wrapper feeds fastsim's stage
    inputs into the still-live pathsim block, runs its own ``update``, and
    returns its outputs — hosted on a raw ``_fastsim.DynamicalFunction``
    (Python callback; never traced, so internal scipy/solver calls are fine).
    Internal state the block keeps across evaluations (warm-started meshes,
    caches) lives on the block instance the closure holds, exactly as it would
    under pathsim's own loop.
    """
    type_name = type(block).__name__

    def func_alg(u, t):
        block.inputs.update_from_array(np.atleast_1d(u))
        block.update(t)
        return block.outputs.to_array()

    fs_block = _fastsim.DynamicalFunction(func_alg)

    translated, unsupported = _translate_events(block, fs_block, None)
    for ev in translated:
        fs_block.add_event(ev)
    if translated:
        log.info("PORT events: %s -> %d fastsim event(s) forwarded", type_name, len(translated))
    if unsupported:
        log.warning(
            "PORT: %s has unsupported internal event type(s) %s — NOT carried over",
            type_name, unsupported,
        )
    log.info("PORT algebraic shim: %s (stateless; update() drives the outputs)", type_name)
    return fs_block


def _port_via_shim(block):
    """Tier 3: engine-shim / RHS-capture fallback (faithful, ~10-100x slower).

    Internal events are translated to fastsim events and attached to the returned
    block as block-internal events (via ``add_event``), so any Simulation tracks
    them automatically — no facade harvesting needed.
    """
    type_name = type(block).__name__
    iv = getattr(block, "initial_value", None)
    if iv is None:
        # No continuous state: an algebraic (or internal-solver) block whose
        # outputs come entirely from `update(t)` — e.g. a `BVP1D` subclass with
        # a custom update. It gets the algebraic shim.
        return _port_algebraic_shim(block)

    shim = CaptureShim(iv)
    block.engine = shim
    n = len(shim)
    warned = [False]

    def func_dyn(x, u, t):
        # Sync fastsim's stage state + inputs into the wrapped block, then run
        # its own step() — which routes its RHS into the shim — and read it back.
        shim.state = x
        block.inputs.update_from_array(np.atleast_1d(u))
        block.step(t, 0.0)  # dt is irrelevant: the shim captures f, doesn't integrate
        f = shim._f
        if f is None:
            # Contract violation: the block's step() did not route its RHS
            # through engine.step/solve, so the shim captured nothing. Warn once
            # rather than silently integrating zeros.
            if not warned[0]:
                log.warning(
                    "PORT: %s.step() did not call engine.step/solve; the shim "
                    "captured no RHS — its dynamics are treated as zero.", type_name,
                )
                warned[0] = True
            return np.zeros(n)
        return f

    def func_alg(x, u, t):
        shim.state = x
        block.inputs.update_from_array(np.atleast_1d(u))
        block.update(t)
        return block.outputs.to_array()

    # Raw `_fastsim.DynamicalSystem` (NOT the JIT block class): these GIL closures
    # must never be traced — a partial trace builds a degenerate Jacobian and
    # panics the implicit solver. The Python-callback path is what the shim needs.
    fs_block = _fastsim.DynamicalSystem(func_dyn, func_alg, iv, _has_passthrough(block, iv))

    translated, unsupported = _translate_events(block, fs_block, shim)
    for ev in translated:
        fs_block.add_event(ev)  # attach as block-internal events
    if translated:
        log.info("PORT events: %s -> %d fastsim event(s) forwarded", type_name, len(translated))
    if unsupported:
        log.warning(
            "PORT: %s has unsupported internal event type(s) %s — NOT carried over",
            type_name, unsupported,
        )
    return fs_block


# -- class-level porting (toolbox classes) ---------------------------------------------

def _shim_factory(original_cls):
    """Build a class whose instances are fastsim blocks shim-wrapping ``original_cls``.

    ``__new__`` constructs the original (pathsim) instance and returns its
    per-instance shim port — so ``Ported(*args)`` yields a fastsim block, never
    an instance of the factory, and the wrapped block's custom hooks run intact.
    """

    def __new__(factory_cls, *args, **kwargs):
        return _port_instance(original_cls(*args, **kwargs))

    ported = type(
        original_cls.__name__,
        (object,),
        {"__new__": __new__, "__doc__": original_cls.__doc__,
         "_pathsim_origin": original_cls},
    )
    ported.__qualname__ = getattr(original_cls, "__qualname__", original_cls.__name__)
    ported.__module__ = getattr(original_cls, "__module__", ported.__module__)
    return ported


def _class_accelerable(cls):
    """True if ``cls`` is a thin subclass of *any* pathsim base fastsim can
    rebase (the adapt registry: ``Function``, ``ODE``, ``DynamicalSystem``,
    ``StateSpace``, the DAE bases, ``TransferFunction`` ...) that defines no
    engine/lifecycle hook of its own — so rebasing onto the fastsim namesake
    preserves behaviour for dynamic *and* algebraic blocks alike.

    The hook check is done pathsim-side: adapt's own override check cannot see
    ``step`` / ``solve`` / ``update`` because fastsim's Rust block does not
    expose them as Python methods, so a custom-hook subclass would be silently
    mis-rebased. We require every engine hook to come from a rebasable base (a
    registry key) or pathsim's generic ``Block``; a hook defined in the subclass
    (or a non-rebasable intermediate) is custom logic that must take the shim.
    """
    from fastsim.adapter import _build_registry

    registry = _build_registry()
    _, ps_block = _tier1_bases()
    if not registry or ps_block is None:
        return False
    if not any(base in registry for base in cls.__mro__):
        return False
    allowed = set(registry.keys())
    allowed.add(ps_block)
    return _overridden_hook(cls, allowed) is None


def _port_class(cls):
    """Port a block *class* so its instances are fastsim blocks: rebase onto
    fastsim via :func:`fastsim.adapter.adapt` when adaptable, else a shim-wrapper.
    """
    name = cls.__name__

    if issubclass(cls, _fastsim.Block):
        # Passthrough is a no-op (nothing is ported), so log at DEBUG to avoid
        # noise — only real porting decisions warrant INFO.
        log.debug("PORT class passthrough: %s is already a fastsim block class", name)
        return cls

    if _class_accelerable(cls):
        from fastsim.adapter import adapt
        adapted = adapt(cls, strict=False)
        if adapted is not cls:
            log.info("PORT class accelerate: %s via adapt (class rebasing)", name)
            return adapted

    log.info("PORT class shim-wrap: %s (instances ported via engine shim)", name)
    return _shim_factory(cls)


# -- instance-level porting ------------------------------------------------------------

def _port_instance(block):
    type_name = type(block).__name__

    if _is_fastsim_block(block):
        # No-op passthrough: log at DEBUG so a system of native fastsim blocks
        # stays quiet; only actual porting (accelerate/shim) logs at INFO.
        log.debug("PORT passthrough: %s is already a fastsim block", type_name)
        return block

    reason = _tier1_reason(block)
    if reason is None:
        ported = _accelerate(block)
        jit = bool(getattr(ported, "jit_compiled", False))
        log.info(
            "PORT accelerate: %s via operator extraction (JIT: %s)",
            type_name, "yes" if jit else "python-fallback",
        )
        return ported

    ported = _port_via_shim(block)
    log.info("PORT shim fallback: %s (reason: %s)", type_name, reason)
    return ported


# -- public API ------------------------------------------------------------------------

def port(obj):
    """Port a pathsim block onto the fastsim engine, accelerating where possible.

    Accepts a pathsim block *class* or *instance* and returns a fastsim-native
    equivalent. The block is accelerated as far as it can be and falls back to a
    faithful Python shim otherwise. fastsim blocks and classes pass through
    unchanged. Each decision is logged to the ``fastsim.port`` logger.

    The strategy is chosen automatically:

    - *passthrough* -- the input is already a fastsim block (or block class).
    - *accelerate* -- the block exposes clean ``op_dyn`` / ``op_alg`` operators
      and overrides no engine hooks; its right-hand-side callables are extracted
      and JIT-traced into the Rust core, running at full native speed.
    - *shim* -- a block with custom ``step`` / ``solve`` or internal events keeps
      its own methods while fastsim's engine drives the integration through a
      capture shim; internal events are translated to fastsim events.

    Passing a class is the intended path for pathsim *toolbox* blocks: port the
    class once, then construct instances and wire them with fastsim
    ``Connection`` objects as usual.

    Example
    -------

    Use a pathsim toolbox block class inside a fastsim simulation:

    .. code-block:: python

        from fastsim import Simulation, Connection, port
        from chem_toolbox import Reactor          # a pathsim block class

        Reactor = port(Reactor)                   # port once
        r = Reactor(rate=2.0)                     # instances are fastsim blocks
        sim = Simulation(
            blocks=[r, scope],
            connections=[Connection(r, scope)],   # wired like any fastsim block
            )

    Parameters
    ----------
    obj : type | pathsim.blocks.Block
        A pathsim block class or instance, or an already-fastsim block (class).

    Returns
    -------
    type | fastsim.blocks.Block
        A fastsim-native block class when `obj` is a class, otherwise a
        fastsim-native block instance. Build connections against the returned
        object.
    """
    if isinstance(obj, type):
        return _port_class(obj)
    return _port_instance(obj)
