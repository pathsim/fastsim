# fastsim.blocks — Drop-in replacement for pathsim.blocks
#
# Blocks are thin Python shim classes over Rust factories: the constructor
# computes parameters in Python and delegates to the Rust core; the simulation
# hot path stays in Rust. All block parameters are mutable at runtime — setting
# one reconstructs the Rust block automatically (mirrors pathsim's @mutable).
#
# The concrete factory-backed + JIT block classes are generated into
# `_generated.py` (see scripts/gen_blocks.py) with explicit `__init__`
# signatures and docstrings; shared behaviour lives in `_shim.py`. Scope,
# Spectrum, BVP1D and AlgebraicConstraint are hand-written below.

import inspect
from functools import lru_cache

from fastsim._fastsim import Block
from fastsim import _fastsim

from ._docstrings_pathsim import DOCS as _DOCS_PATHSIM
from ._docstrings_fastsim import DOCS_FASTSIM as _DOCS_FASTSIM
from ._shim import _build_info, _port_getitem

# fastsim-specific docstrings override pathsim mirrors when names collide.
_DOCS = {**_DOCS_PATHSIM, **_DOCS_FASTSIM}

# All factory-backed and JIT block classes (explicit shims).
from . import _generated as _generated_mod
from ._generated import *  # noqa: F401,F403

# Apply the fastsim-specific docstring overrides to the GENERATED classes too.
# Previously only the four hand-written classes got them (via
# `_finalize_block_class`), so a generated block with a fastsim override — e.g.
# CoSimulationFMU, whose extracted stub is empty — kept the empty inline
# docstring and `info()`/`help()` disagreed with the registry. Applying the
# override here makes `_DOCS_FASTSIM` the single authoritative store for every
# block, generated or hand-written.
for _name, _doc in _DOCS_FASTSIM.items():
    _cls = getattr(_generated_mod, _name, None)
    if isinstance(_cls, type) and issubclass(_cls, Block) and _doc:
        _cls.__doc__ = _doc


def _params_from_signature(sig):
    """Extract `{name: {default: value}}` from an inspect.Signature.

    PyO3 represents Rust-side default values opaquely as ``Ellipsis``, so we map
    ``Ellipsis`` and missing defaults to ``None``, matching pathsim's convention
    for parameters without a default.
    """
    if sig is None:
        return {}
    out = {}
    for pname, param in sig.parameters.items():
        if pname in ("self", "args", "kwargs"):
            continue
        d = param.default
        if d is inspect.Parameter.empty or d is Ellipsis:
            d = None
        out[pname] = {"default": d}
    return out


def _finalize_block_class(cls):
    """Give a hand-written ``Block`` subclass the same docstring + introspection
    that the generated blocks get: pull the detailed docstring from the central
    registry ``_DOCS`` and attach the ``info()`` classmethod."""
    doc = _DOCS.get(cls.__name__)
    if doc:
        cls.__doc__ = doc
    if not hasattr(cls, "input_port_labels"):
        cls.input_port_labels = None
    if not hasattr(cls, "output_port_labels"):
        cls.output_port_labels = None
    if "info" not in cls.__dict__:
        try:
            _params = _params_from_signature(inspect.signature(cls.__init__))
        except (ValueError, TypeError):
            _params = {}
        cls.info = classmethod(
            lru_cache(maxsize=None)(lambda c, _p=_params: _build_info(c, _p))
        )
    return cls


# ======================================================================================
# Recording blocks (hand-written: they carry a plot() method)
# ======================================================================================

_Scope_factory = getattr(_fastsim, "Scope")
_Spectrum_factory = getattr(_fastsim, "Spectrum")

# Color palette matching pathsim
_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']


class Scope(Block):
    # Docstring + info() are attached from the central registry by
    # _finalize_block_class(Scope) below (uniform with all other blocks).

    def __init__(self, labels=None, sampling_period=None, t_wait=0.0):
        super().__init__()
        self.__dict__['_init_params'] = {
            'labels': labels, 'sampling_period': sampling_period, 't_wait': t_wait,
        }
        self.__dict__['_labels'] = labels or []
        kwargs = {'t_wait': t_wait}
        if labels is not None:
            kwargs['labels'] = labels
        if sampling_period is not None:
            kwargs['sampling_period'] = sampling_period
        self._init_from(_Scope_factory(**kwargs))

    def plot(self, *args, **kwargs):
        """Plot recorded data with interactive legend picking."""
        import matplotlib.pyplot as plt

        time, data = self.read()
        if time is None:
            return None, None

        fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True, dpi=120)
        ax.set_prop_cycle(color=_COLORS)

        labels = self.__dict__.get('_labels', [])
        for p, d in enumerate(data):
            lb = labels[p] if p < len(labels) else f"port {p}"
            ax.plot(time, d, *args, **kwargs, label=lb)

        ax.legend(fancybox=False)
        ax.set_xlabel("time [s]")
        ax.grid()

        # Legend picking
        lines = ax.get_lines()
        leg = ax.get_legend()
        lined = {}
        for legline, origline in zip(leg.get_lines(), lines):
            legline.set_picker(5)
            lined[legline] = origline

        def on_pick(event):
            legline = event.artist
            origline = lined[legline]
            visible = not origline.get_visible()
            origline.set_visible(visible)
            legline.set_alpha(1.0 if visible else 0.2)
            fig.canvas.draw()

        fig.canvas.mpl_connect("pick_event", on_pick)
        plt.show(block=False)
        return fig, ax


class Spectrum(Block):
    # Docstring + info() are attached from the central registry by
    # _finalize_block_class(Spectrum) below (uniform with all other blocks).

    def __init__(self, freq=None, t_wait=0.0, alpha=0.0, labels=None):
        super().__init__()
        self.__dict__['_init_params'] = {
            'freq': freq, 't_wait': t_wait, 'alpha': alpha, 'labels': labels,
        }
        self.__dict__['_labels'] = labels or []
        kwargs = {'t_wait': t_wait, 'alpha': alpha}
        if freq is not None:
            kwargs['freq'] = list(freq)
        if labels is not None:
            kwargs['labels'] = labels
        self._init_from(_Spectrum_factory(**kwargs))

    def plot(self, *args, **kwargs):
        """Plot frequency spectrum with interactive legend picking."""
        import matplotlib.pyplot as plt

        freq, data = self.read()
        if freq is None:
            return None, None

        fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True, dpi=120)
        ax.set_prop_cycle(color=_COLORS)

        labels = self.__dict__.get('_labels', [])
        for p, d in enumerate(data):
            lb = labels[p] if p < len(labels) else f"port {p}"
            ax.plot(freq, abs(d), *args, **kwargs, label=lb)

        ax.legend(fancybox=False)
        ax.set_xlabel("freq [Hz]")
        ax.set_ylabel("magnitude")
        ax.grid()

        # Legend picking
        lines = ax.get_lines()
        leg = ax.get_legend()
        lined = {}
        for legline, origline in zip(leg.get_lines(), lines):
            legline.set_picker(5)
            lined[legline] = origline

        def on_pick(event):
            legline = event.artist
            origline = lined[legline]
            visible = not origline.get_visible()
            origline.set_visible(visible)
            legline.set_alpha(1.0 if visible else 0.2)
            fig.canvas.draw()

        fig.canvas.mpl_connect("pick_event", on_pick)
        plt.show(block=False)
        return fig, ax


# ======================================================================================
# Boundary-value & algebraic-constraint blocks (native collocation / Newton)
# ======================================================================================

from .bvp import BVP1D  # noqa: E402
from .algebraic import AlgebraicConstraint  # noqa: E402

# Unify the hand-written block classes with the generated ones: detailed
# docstrings come from the central registry, and each gets the standard info().
for _cls in (Scope, Spectrum, BVP1D, AlgebraicConstraint):
    _finalize_block_class(_cls)
del _cls
