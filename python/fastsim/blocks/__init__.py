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
import warnings
from functools import lru_cache

from fastsim._fastsim import Block
from fastsim import _fastsim

from ._docstrings import DOCS as _DOCS
from ._shim import _build_info, _port_getitem

# All factory-backed and JIT block classes (explicit shims).
from . import _generated as _generated_mod
from ._generated import *  # noqa: F401,F403

# Apply the registry docstrings to the GENERATED classes at import time, so
# `_docstrings.py` stays the single runtime authority even when it was edited
# without re-running scripts/gen_blocks.py (the inline docstrings in
# `_generated.py` are then just a stale cache for static tooling).
for _name, _doc in _DOCS.items():
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
        # Drop pathsim-compatibility aliases: they are accepted by the
        # constructor so pathsim source runs unchanged, but they are not
        # parameters of the block. `info()` drives UIs and introspection, where
        # an alias would show up as a second, duplicate field.
        for _alias in getattr(cls, "_compat_aliases", ()):
            _params.pop(_alias, None)
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


def _pickable_legend(fig, ax):
    """Clicking a legend entry toggles its trace, as in pathsim."""
    lines = ax.get_lines()
    leg = ax.get_legend()
    if leg is None:
        return
    lined = {}
    for legline, origline in zip(leg.get_lines(), lines):
        legline.set_picker(5)
        lined[legline] = origline

    def on_pick(event):
        origline = lined[event.artist]
        visible = not origline.get_visible()
        origline.set_visible(visible)
        event.artist.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw()

    fig.canvas.mpl_connect("pick_event", on_pick)


def _new_axes(figsize):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize,
                           tight_layout=True, dpi=120)
    ax.set_prop_cycle(color=_COLORS)
    return fig, ax


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

    def read(self, *args, **kwargs):
        """Recorded times and per-port data.

        Returns ``(time, data)`` with `data` an array of shape
        ``(n_ports, n_samples)``, and ``(None, None)`` when nothing was
        recorded — both as in pathsim, whose examples do array arithmetic on
        the result (``data_moon - data_earth``) and test the empty case with
        ``if time is None``.
        """
        import numpy as np

        time, data = super().read(*args, **kwargs)
        if time is None or len(time) == 0:
            return None, None
        return np.asarray(time), np.asarray(data)

    def _label(self, port):
        labels = self.__dict__.get('_labels', [])
        return labels[port] if port < len(labels) else f"port {port}"

    def plot(self, *args, **kwargs):
        """Plot every recorded port against time."""
        import matplotlib.pyplot as plt

        time, data = self.read()
        if time is None:
            warnings.warn("no recording available for plotting in 'Scope.plot'")
            return None, None

        fig, ax = _new_axes((8, 4))
        for p, d in enumerate(data):
            ax.plot(time, d, *args, **kwargs, label=self._label(p))

        ax.legend(fancybox=False)
        ax.set_xlabel("time [s]")
        ax.grid()

        _pickable_legend(fig, ax)
        plt.show(block=False)
        self.__dict__['fig'], self.__dict__['ax'] = fig, ax
        return fig, ax

    def plot2D(self, *args, axes=(0, 1), **kwargs):
        """Plot one recorded port against another (a phase portrait)."""
        import matplotlib.pyplot as plt

        time, data = self.read()
        if time is None:
            warnings.warn("no recording available for plotting in 'Scope.plot2D'")
            return None, None
        if len(data) < 2 or len(axes) != 2:
            warnings.warn("not enough channels for plotting in 'Scope.plot2D'")
            return None, None
        if not all(0 <= i < data.shape[0] for i in axes):
            warnings.warn(f"selected axes {axes} out of bounds for data shape {data.shape}")
            return None, None

        i, j = axes
        fig, ax = _new_axes((4, 4))
        ax.plot(data[i], data[j], *args, **kwargs)
        ax.set_xlabel(self._label(i))
        ax.set_ylabel(self._label(j))
        ax.grid()

        plt.show(block=False)
        self.__dict__['fig'], self.__dict__['ax'] = fig, ax
        return fig, ax

    def plot3D(self, *args, axes=(0, 1, 2), **kwargs):
        """Plot three recorded ports against each other."""
        import matplotlib.pyplot as plt

        time, data = self.read()
        if time is None:
            warnings.warn("no recording available for plotting in 'Scope.plot3D'")
            return None, None
        if len(data) < 3 or len(axes) != 3:
            warnings.warn("not enough channels for plotting in 'Scope.plot3D'")
            return None, None
        if not all(0 <= i < data.shape[0] for i in axes):
            warnings.warn(f"selected axes {axes} out of bounds for data shape {data.shape}")
            return None, None

        i, j, k = axes
        fig = plt.figure(figsize=(4, 4), tight_layout=True, dpi=120)
        ax = fig.add_subplot(projection="3d")
        ax.set_prop_cycle(color=_COLORS)
        ax.plot(data[i], data[j], data[k], *args, **kwargs)
        ax.set_xlabel(self._label(i))
        ax.set_ylabel(self._label(j))
        ax.set_zlabel(self._label(k))
        ax.grid()

        plt.show(block=False)
        self.__dict__['fig'], self.__dict__['ax'] = fig, ax
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
        """Plot the magnitude of every recorded spectrum.

        Keeps the figure and axis on the block (``self.fig`` / ``self.ax``) as
        pathsim does, so a caller can adjust the plot afterwards —
        ``Spc.ax.set_yscale("log")`` in pathsim's ``example_noise.py``.
        """
        import matplotlib.pyplot as plt

        freq, data = self.read()
        if freq is None:
            warnings.warn("no recording available for plotting in 'Spectrum.plot'")
            return None, None

        fig, ax = _new_axes((8, 4))
        labels = self.__dict__.get('_labels', [])
        for p, d in enumerate(data):
            lb = labels[p] if p < len(labels) else f"port {p}"
            ax.plot(freq, abs(d), *args, **kwargs, label=lb)

        ax.legend(fancybox=False)
        ax.set_xlabel("freq [Hz]")
        ax.set_ylabel("magnitude")
        ax.grid()

        _pickable_legend(fig, ax)
        plt.show(block=False)
        self.__dict__['fig'], self.__dict__['ax'] = fig, ax
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
