# fastsim.blocks._shim — shared base classes for the block library.
#
# Every block is a thin Python shim over a Rust factory: the constructor
# computes parameters in Python and delegates to the Rust core via
# `_init_from()`; the simulation hot path stays in Rust. Constructor
# parameters are mutable at runtime — setting one (e.g. `amp.gain = 10`)
# reconstructs the Rust block automatically, preserving engine state when the
# dimensions match (mirrors pathsim's @mutable behaviour).
#
# The concrete block classes live in `_generated.py` (materialized by
# `scripts/gen_blocks.py`) and in `__init__.py` (hand-written Scope/Spectrum,
# BVP1D, AlgebraicConstraint). They are thin: name, ports, docstring and an
# explicit `__init__` signature — all shared behaviour lives here.

import inspect
from functools import lru_cache

from fastsim._fastsim import Block
from fastsim import _fastsim


def _build_info(cls, params):
    """Build the same info dict shape as pathsim's Block.info()."""
    return {
        "type": cls.__name__,
        "description": cls.__doc__,
        "input_port_labels": cls.input_port_labels,
        "output_port_labels": cls.output_port_labels,
        "parameters": params,
    }


def _port_getitem(self, key):
    """Shared __getitem__ for blocks with port labels."""
    if isinstance(key, str):
        all_labels = {}
        ipl = getattr(self, 'input_port_labels', None)
        opl = getattr(self, 'output_port_labels', None)
        if ipl:
            all_labels.update(ipl)
        if opl:
            all_labels.update(opl)
        if key not in all_labels:
            raise ValueError(
                f"Port alias '{key}' not defined for {type(self).__name__}. "
                f"Available: {list(all_labels.keys())}")
        return Block.__getitem__(self, all_labels[key])
    return Block.__getitem__(self, key)


def _adapt_function_arity(func):
    """Normalize a Function-block callable so it works under BOTH calling
    conventions fastsim uses for the input vector `u`: the per-block runtime
    splats it (``func(*u)``) while the JIT tracer passes a single array
    (``func(u)``). A multi-positional-arg lambda only matches the runtime form,
    so the tracer raises ``TypeError`` and silently falls back to the slow
    opaque path. Wrapping such a callable to accept either shape lets it
    JIT-compile like every other block. Single-arg callables are returned
    unchanged."""
    if getattr(func, "_fastsim_arity_adapted", False):
        return func
    try:
        ps = inspect.signature(func).parameters.values()
        n = sum(1 for p in ps if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.default is p.empty)
        variadic = any(p.kind is p.VAR_POSITIONAL for p in ps)
    except (TypeError, ValueError):
        n, variadic = 1, False
    if n <= 1 or variadic:
        return func

    def adapted(*args):
        u = args[0] if len(args) == 1 else args
        return func(*(u[i] for i in range(n)))
    adapted._fastsim_arity_adapted = True
    return adapted


def _jit_debug_enabled():
    """Opt-in JIT trace diagnostics, enabled via ``FASTSIM_JIT_DEBUG=1``."""
    import os
    return os.environ.get("FASTSIM_JIT_DEBUG", "").lower() in ("1", "true", "yes", "on")


def _trace_or_none(rust_fn, *args):
    """Attempt a Rust tracer call; swallow trace failures and return None so
    the caller falls back to the Python callback path. With FASTSIM_JIT_DEBUG=1
    the failure reason is emitted as a warning."""
    import warnings
    try:
        block = rust_fn(*args)
        if block is not None:
            return block
        if _jit_debug_enabled():
            warnings.warn(
                f"JIT trace returned None (likely empty graph or unsupported "
                f"output shape) for {rust_fn.__name__}; using Python callback.",
                stacklevel=3,
            )
    except Exception as e:
        if _jit_debug_enabled():
            warnings.warn(
                f"JIT trace failed in {rust_fn.__name__}: {type(e).__name__}: {e}; "
                f"using Python callback.",
                stacklevel=3,
            )
    return None


def _factory_kwargs(params):
    """Only pass explicitly-set parameters to the Rust factory. Ellipsis marks a
    parameter the caller did not supply, so the factory applies its own default
    (the PyO3 ``param=...`` convention)."""
    return {k: v for k, v in params.items() if v is not Ellipsis}


class _ShimBlock(Block):
    """Base for factory-backed blocks. Concrete subclasses set ``_factory_name``,
    ``_param_defaults`` and the port-label class attributes, and provide an
    explicit ``__init__`` that forwards its parameters to :meth:`_shim_init`."""

    _factory_name = None
    _param_defaults = {}
    input_port_labels = None
    output_port_labels = None

    __getitem__ = _port_getitem

    @classmethod
    @lru_cache(maxsize=None)
    def info(cls):
        """Block metadata for introspection and UI integration.

        Mirrors `pathsim.blocks.Block.info()`.
        """
        params = {k: {"default": v} for k, v in cls._param_defaults.items()}
        return _build_info(cls, params)

    def _shim_init(self, params):
        """Store the constructor parameters and build the Rust block."""
        factory = getattr(_fastsim, self._factory_name)
        self.__dict__['_init_params'] = params
        self._init_from(factory(**_factory_kwargs(params)))

    def __setattr__(self, attr, value):
        params = self.__dict__.get('_init_params')
        if params is not None and attr in params:
            old_state = self.state
            params[attr] = value
            factory = getattr(_fastsim, self._factory_name)
            self._init_from(factory(**_factory_kwargs(params)))
            if old_state is not None and self.state is not None:
                if len(old_state) == len(self.state):
                    self.state = old_state
        else:
            super().__setattr__(attr, value)

    def __getattr__(self, attr):
        params = self.__dict__.get('_init_params')
        if params is not None and attr in params:
            v = params[attr]
            return None if v is Ellipsis else v
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{attr}'")

    def set(self, **kwargs):
        """Update multiple parameters at once with a single reinit."""
        params = self.__dict__.get('_init_params', {})
        valid = {k: v for k, v in kwargs.items() if k in params}
        if not valid:
            return
        old_state = self.state
        params.update(valid)
        factory = getattr(_fastsim, self._factory_name)
        self._init_from(factory(**_factory_kwargs(params)))
        if old_state is not None and self.state is not None:
            if len(old_state) == len(self.state):
                self.state = old_state


class _JitShimBlock(Block):
    """Base for blocks that try JIT compilation with a Python fallback
    (Source, ODE, Function, the DAE family, DynamicalSystem, ...). Concrete
    subclasses set ``_factory_name``, ``_jit_fn`` (a callable ``params -> block
    or None``), ``_param_defaults`` and ``_adapt_func``, and provide an explicit
    ``__init__`` that forwards its parameters to :meth:`_jit_init`."""

    _factory_name = None
    _jit_fn = None
    _param_defaults = {}
    _adapt_func = False
    input_port_labels = None
    output_port_labels = None

    __getitem__ = _port_getitem

    @classmethod
    @lru_cache(maxsize=None)
    def info(cls):
        """Block metadata for introspection and UI integration.

        Mirrors `pathsim.blocks.Block.info()`.
        """
        params = {k: {"default": v} for k, v in cls._param_defaults.items()}
        return _build_info(cls, params)

    def _jit_init(self, params):
        """Store parameters, then build via the JIT path with a factory fallback."""
        if self._adapt_func and params.get('func') is not None:
            params['func'] = _adapt_function_arity(params['func'])
        self.__dict__['_init_params'] = params
        self.__dict__['_jit_compiled'] = False
        jit_block = self._jit_fn(params)
        if jit_block is not None:
            self._init_from(jit_block)
            self.__dict__['_jit_compiled'] = True
        else:
            factory = getattr(_fastsim, self._factory_name)
            self._init_from(factory(**params))

    def __setattr__(self, attr, value):
        params = self.__dict__.get('_init_params')
        if params is not None and attr in params:
            old_state = self.state
            params[attr] = value
            if self._adapt_func and attr == 'func' and value is not None:
                params['func'] = _adapt_function_arity(value)
            jit_block = self._jit_fn(params)
            if jit_block is not None:
                self._init_from(jit_block)
                self.__dict__['_jit_compiled'] = True
            else:
                factory = getattr(_fastsim, self._factory_name)
                self._init_from(factory(**params))
                self.__dict__['_jit_compiled'] = False
            if old_state is not None and self.state is not None:
                if len(old_state) == len(self.state):
                    self.state = old_state
        else:
            super().__setattr__(attr, value)

    def __getattr__(self, attr):
        params = self.__dict__.get('_init_params')
        if params is not None and attr in params:
            return params[attr]
        if attr == 'jit_compiled':
            return self.__dict__.get('_jit_compiled', False)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{attr}'")

    def set(self, **kwargs):
        """Update multiple parameters at once with a single reinit."""
        params = self.__dict__.get('_init_params', {})
        valid = {k: v for k, v in kwargs.items() if k in params}
        if not valid:
            return
        old_state = self.state
        params.update(valid)
        jit_block = self._jit_fn(params)
        if jit_block is not None:
            self._init_from(jit_block)
            self.__dict__['_jit_compiled'] = True
        else:
            factory = getattr(_fastsim, self._factory_name)
            self._init_from(factory(**params))
            self.__dict__['_jit_compiled'] = False
        if old_state is not None and self.state is not None:
            if len(old_state) == len(self.state):
                self.state = old_state


# JIT tracer wrappers (params -> traced block or None). Referenced by the
# generated JIT block classes via `_jit_fn`.
def _jit_source(params):
    return _trace_or_none(_fastsim._trace_source, params['func'])

def _jit_ode(params):
    return _trace_or_none(_fastsim._trace_ode, params['func'], params['initial_value'])

def _jit_function(params):
    return _trace_or_none(_fastsim._trace_function_block, params['func'])

def _jit_mass_matrix(params):
    return _trace_or_none(
        _fastsim._trace_mass_matrix_dae,
        params['func'], params['mass'], params['initial_value'],
    )

def _jit_semi_explicit(params):
    return _trace_or_none(
        _fastsim._trace_semi_explicit_dae,
        params['f_dyn'], params['f_alg'], params['x0'], params['z0'],
    )

def _jit_fully_implicit(params):
    return _trace_or_none(
        _fastsim._trace_fully_implicit_dae,
        params['func'], params['initial_value'],
    )

def _jit_wrapper(params):
    return _trace_or_none(
        _fastsim._trace_wrapper,
        params['func'], params.get('T', 1.0), params.get('tau', 0.0),
    )

def _jit_dynamical_system(params):
    return _trace_or_none(
        _fastsim._trace_dynamical_system,
        params['func_dyn'], params['func_alg'],
        params['initial_value'], params['has_passthrough'],
    )

def _jit_dynamical_function(params):
    return _trace_or_none(_fastsim._trace_dynamical_function, params['func'])
