"""Adapt pathsim-derived block subclasses so they run on the fastsim engine.

Many pathsim ecosystem toolboxes (pathsim-chem, ...) define domain-specific
blocks by *inheriting* from generic pathsim base blocks (Function, ODE,
StateSpace, DynamicalSystem, ...) and only overriding ``__init__`` to compute
parameters they then forward to ``super().__init__(...)``. Such classes don't
touch the simulation-loop hooks (``step``, ``update``, ``solve``, ``buffer``,
...) — all of that stays in the base.

Because fastsim's base blocks are API-compatible with pathsim's, we can clone
such a subclass with the pathsim base *swapped* for the fastsim base. The
resulting class keeps all custom attributes and ``__init__`` logic but routes
the hot-path methods through fastsim's Rust core.

Public entry point:

- :func:`adapt(cls)` — clone a single block class. Refuses (or warns) when the
  subclass would lose correctness/performance by overriding a base method —
  see :func:`_check_overrides`.
"""

from __future__ import annotations

import types
import warnings
from typing import Optional


# Class attributes that a Toolbox subclass may legitimately define without
# breaking the adapter — ``__init__`` computing parameters for ``super()`` is
# the whole point, and the other dunders are Python infrastructure.
_SAFE_OVERRIDES: frozenset[str] = frozenset({
    "__init__", "__new__", "__repr__", "__str__",
    "__init_subclass__", "__class_getitem__",
    "__doc__", "__dict__", "__weakref__", "__module__", "__qualname__",
    "__annotations__", "__hash__", "__eq__",
    # Block-level class attributes used for port labeling — data, not methods
    "input_port_labels", "output_port_labels",
})


_registry_cache: Optional[dict[type, type]] = None


def _build_registry() -> dict[type, type]:
    """Map pathsim base classes to their fastsim equivalents.

    Built lazily so ``import fastsim`` doesn't require pathsim to be installed.
    """
    global _registry_cache
    if _registry_cache is not None:
        return _registry_cache

    mapping: dict[type, type] = {}
    try:
        import pathsim.blocks as ps_blocks  # type: ignore[import-not-found]
    except ImportError:
        _registry_cache = mapping
        return mapping

    import fastsim.blocks as fs_blocks

    # Only generic "composition base" blocks — the ones a domain toolbox would
    # realistically inherit from to expose a domain API while delegating the
    # heavy lifting to a standard block. End-user blocks (Integrator, Amplifier,
    # Source, Scope, ...) are not intended as subclassing targets and stay out.
    _BASE_NAMES = (
        # Generic ODE / DAE composition targets
        "Function",
        "ODE",
        "StateSpace",
        "DynamicalSystem",
        "DynamicalFunction",
        "MassMatrixDAE",
        "SemiExplicitDAE",
        "FullyImplicitDAE",
        "Wrapper",
        # Linear transfer-function bases
        "TransferFunction",
        "TransferFunctionNumDen",
        "TransferFunctionPRC",
        "TransferFunctionZPG",
    )
    for name in _BASE_NAMES:
        ps_cls = getattr(ps_blocks, name, None)
        fs_cls = getattr(fs_blocks, name, None)
        if isinstance(ps_cls, type) and isinstance(fs_cls, type):
            mapping[ps_cls] = fs_cls

    _registry_cache = mapping
    return mapping


def _fs_base_methods(fs_cls: type) -> set[str]:
    """Names of callable members that `fs_cls` (or its parents) expose.

    Used to decide whether a subclass override shadows real engine behaviour.
    """
    names: set[str] = set()
    for c in fs_cls.__mro__:
        if c is object:
            continue
        for name in dir(c):
            if callable(getattr(c, name, None)):
                names.add(name)
    return names


def _check_overrides(cls: type, fs_base: type) -> list[str]:
    """Return names of methods/properties in `cls.__dict__` that would shadow
    members of `fs_base`. Empty list == safe to clone.
    """
    base_api = _fs_base_methods(fs_base)
    conflicts: list[str] = []
    for name, val in cls.__dict__.items():
        if name in _SAFE_OVERRIDES:
            continue
        if name not in base_api:
            continue  # new attribute, not a shadow
        if callable(val) or isinstance(val, (property, staticmethod, classmethod)):
            conflicts.append(name)
    return conflicts


def _is_pathsim_derived(cls: type, registry: dict[type, type]) -> bool:
    """True iff some ancestor of `cls` is a known pathsim base."""
    return any(base in registry for base in cls.__mro__)


def _make_cell(value):
    """Build a fresh Python closure cell containing `value`."""
    # Lambda-over-capture pattern works uniformly from Python 3.8+.
    x = value
    return (lambda: x).__closure__[0]


def _clone_method_rebinding_class(func, new_cls):
    """Return a copy of `func` with its ``__class__`` closure cell pointing at
    `new_cls`. Required because a bare ``super()`` compiles to
    ``super(__class__, first_arg)``, and the ``__class__`` cell is captured at
    class-definition time from the enclosing class. A clone with a new identity
    must rebuild that cell, otherwise ``super()`` raises
    ``TypeError: obj is not an instance or subtype of type``.
    """
    if not isinstance(func, types.FunctionType):
        return func
    closure = func.__closure__
    if not closure:
        return func
    freevars = func.__code__.co_freevars
    if "__class__" not in freevars:
        return func
    idx = freevars.index("__class__")
    new_closure = list(closure)
    new_closure[idx] = _make_cell(new_cls)
    return types.FunctionType(
        func.__code__,
        func.__globals__,
        name=func.__name__,
        argdefs=func.__defaults__,
        closure=tuple(new_closure),
    )


def adapt(
    cls: type,
    *,
    strict: bool = True,
    _memo: Optional[dict[type, type]] = None,
) -> type:
    """Clone `cls` onto the fastsim engine.

    The clone keeps `cls`'s name, module, qualname, attributes and custom
    methods; only the pathsim bases in its MRO are substituted for their
    fastsim equivalents from the registry. Subclasses further up the chain are
    adapted recursively, so a hierarchy like
    ``Process -> ResidenceTime -> DynamicalSystem`` is fully rebased.

    Methods that carry an implicit ``__class__`` cell (anything using bare
    ``super()``) are copied with a fresh cell pointing at the new clone — so
    ``super()`` keeps finding the right MRO.

    This is a clone, not in-place mutation: the original `cls` is untouched
    and ``isinstance(obj, cls)`` on an adapted instance returns ``False``. Use
    the returned class when constructing blocks for a fastsim Simulation.

    Parameters
    ----------
    cls : type
        Toolbox block class to adapt.
    strict : bool, default True
        Raise ``TypeError`` if any ancestor in the cloned chain overrides a
        method/property of the fastsim base. With ``strict=False`` the override
        is kept and a warning is emitted.

    Returns
    -------
    type
        New class, rebased on fastsim.

    See Also
    --------
    port : Adapt *and* accelerate a pathsim block (class or instance),
        JIT-tracing its right-hand side into the Rust core where possible.
        Prefer `port` for plain simulation speed-ups; use `adapt` when you need
        the rebased *class* itself (e.g. to subclass it further or inspect its
        MRO).

    Examples
    --------
    Take a pathsim toolbox block class and run it on the fast Rust engine. The
    adapted class behaves exactly like the original — same constructor, same
    attributes — but its `pathsim` bases are swapped for their fastsim
    equivalents, so it plugs straight into a fastsim `Simulation`:

    .. code-block:: python

        from fastsim import Simulation, Connection, adapt
        from fastsim.blocks import Scope
        from chem_toolbox import Reactor          # a pathsim-based block class

        FastReactor = adapt(Reactor)              # rebased onto the fastsim engine
        r = FastReactor(rate=2.0, volume=1.5)     # constructed like the original
        sco = Scope()

        sim = Simulation(
            blocks=[r, sco],
            connections=[Connection(r, sco)],
        )
        sim.run(10.0)

    The original class is untouched, so both engines can be used side by side:

    .. code-block:: python

        r_slow = Reactor(rate=2.0, volume=1.5)    # still the pathsim block
        isinstance(r, Reactor)                    # False — `adapt` returns a clone

    A whole inheritance chain is rebased in one call. Given
    ``Process -> ResidenceTime -> pathsim.blocks.DynamicalSystem``, ``adapt``
    walks the MRO and substitutes the pathsim base, keeping `Process`'s and
    `ResidenceTime`'s own methods intact.
    """
    if _memo is None:
        _memo = {}
    if cls in _memo:
        return _memo[cls]

    registry = _build_registry()

    if not _is_pathsim_derived(cls, registry):
        _memo[cls] = cls
        return cls

    # Build new bases: direct registry hit, or recurse into toolbox class.
    new_bases: list[type] = []
    for base in cls.__bases__:
        if base in registry:
            new_bases.append(registry[base])
        elif _is_pathsim_derived(base, registry):
            new_bases.append(adapt(base, strict=strict, _memo=_memo))
        else:
            new_bases.append(base)
    new_bases_tup = tuple(new_bases)

    if new_bases_tup == cls.__bases__:
        _memo[cls] = cls
        return cls

    # Override safety against any fastsim-block base we're introducing.
    for fs_base in new_bases_tup:
        if fs_base in registry.values():
            conflicts = _check_overrides(cls, fs_base)
            if conflicts:
                msg = (
                    f"{cls.__module__}.{cls.__name__} overrides "
                    f"{sorted(conflicts)} from {fs_base.__name__}. Adapting "
                    f"would route these through Python and bypass the Rust "
                    f"core — behaviour and performance will differ. Remove "
                    f"the overrides or pass strict=False to opt in."
                )
                if strict:
                    raise TypeError(msg)
                warnings.warn(msg, stacklevel=2)

    # Build shell class first so we can rebind __class__ cells to it, then
    # attach the (rebound) methods.
    cloned = types.new_class(cls.__name__, bases=new_bases_tup)
    cloned.__module__ = cls.__module__
    cloned.__qualname__ = getattr(cls, "__qualname__", cls.__name__)

    for name, val in cls.__dict__.items():
        if name in ("__dict__", "__weakref__", "__module__", "__qualname__"):
            continue
        if isinstance(val, types.FunctionType):
            val = _clone_method_rebinding_class(val, cloned)
        try:
            setattr(cloned, name, val)
        except (AttributeError, TypeError):
            # Slot-backed or read-only attributes on PyO3 bases: skip silently.
            pass

    _memo[cls] = cloned
    return cloned
