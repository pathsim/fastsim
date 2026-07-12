"""JIT compilation and automatic differentiation for fastsim.

Provides JAX-style function transformations that trace Python functions
into optimised Rust IR for fast evaluation and symbolic differentiation.

Example
-------

.. code-block:: python

    from fastsim.jit import jit, jacobian

    def lorenz(x, t):
        sigma, rho, beta = 10.0, 28.0, 8.0/3.0
        return [sigma*(x[1]-x[0]), x[0]*(rho-x[2])-x[1], x[0]*x[1]-beta*x[2]]

    # JIT compile — lazy tracing on first call
    f_fast = jit(lorenz)
    result = f_fast([1.0, 1.0, 1.0], 0.0)

    # Automatic Jacobian via symbolic AD
    jac_fn = jacobian(lorenz)
    J = jac_fn([1.0, 1.0, 1.0], 0.0)  # 3x3 numpy array
"""

try:
    from fastsim._fastsim import jit_compile as _jit_compile, jit_jacobian as _jit_jacobian
except ImportError:
    _jit_compile = None
    _jit_jacobian = None


def jit(func, n_x=None):
    """Trace and compile a Python function to optimised Rust IR.

    The returned callable evaluates the function in Rust with zero Python
    overhead per call.  Tracing is lazy by default (on first call) or
    eager if ``n_x`` is provided.  The compiled tape is specialized to the
    traced input length; calling with a different-length ``x`` re-traces
    transparently.

    Supported operations: arithmetic (incl. ``//`` floor-division),
    ``np.sin/cos/tan/exp/log/tanh/...``, ``np.dot``, ``np.clip``, ``np.where``,
    ``np.linalg.norm``, ``np.cross``, matrix multiply (``@``, incl. two traced
    operands like a state-dependent ``M(x) @ v``), ``np.sum``,
    ``fastsim.random_uniform`` / ``random_normal`` (stateless traceable noise),
    and more. Unsupported patterns fall back to the original Python function.

    Parameters
    ----------
    func : callable
        Python function with signature ``f(x)`` or ``f(x, t)``.
    n_x : int, optional
        Input dimension for eager tracing.  If omitted, tracing is
        deferred to the first call.
    """
    return _jit_compile(func, n_x=n_x)


def jacobian(func, n_x=None):
    """Compute the Jacobian of a Python function via symbolic AD.

    The returned callable evaluates the full Jacobian matrix in Rust.
    Differentiation is performed symbolically on the traced computation
    graph — no finite differences, no tape replay.  The Jacobian graph
    is optimised (constant folding, CSE) before evaluation.  Like ``jit``,
    the result is shape-specialized and re-traces on a different-length
    ``x``.

    Parameters
    ----------
    func : callable
        Python function with signature ``f(x)`` or ``f(x, t)``.
        Must return a list/array of floats.
    n_x : int, optional
        Input dimension for eager tracing.
    """
    return _jit_jacobian(func, n_x=n_x)


# Tracer primitives (internal, used by block constructors for JIT traces).
# We deliberately do NOT re-export fastsim-specific helpers like `where_` /
# `clip` — user code should be drop-in-compatible with pathsim and use
# `np.where` / `np.clip`, both of which are transparently intercepted by the
# tracer. The Rust-side functions remain importable from `fastsim._fastsim`
# for internal/test use.
try:
    from fastsim._fastsim import JitTracer, JitTracerArray
except ImportError:
    JitTracer = None
    JitTracerArray = None
