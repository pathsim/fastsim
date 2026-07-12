"""Native ``scipy.solve_bvp`` as a fastsim block — with free parameters and
interior (multipoint) conditions.

`BVP1D` rebuilds the Kierzenka–Shampine collocation BVP solver (4th-order
Lobatto-IIIa / Simpson collocation + residual-based mesh refinement) natively in
Rust, with the Newton Jacobian from **AD on the traced ``fun``/``bc``/``icond``**
(exact). The hot path is allocation-free. It solves

    y'(x) = fun(x, y, p, inputs),   bc(y(a), y(b), p, inputs) = 0,

optionally with **free parameters** ``p`` (extra unknowns, determined by extra
conditions — eigenvalues, unknown lengths/fluxes, …) and **interior conditions**
``icond(y@ports, p, inputs) = 0`` imposed at arbitrary ``x_ports`` (multipoint
BVP, beyond scipy). Boundary/parameter data flows in through the block inputs and
is re-read each evaluation; the adapted mesh + parameters are warmstarted.
"""

import numpy as np

from fastsim._fastsim import _trace_bvp1d, Block


class BVP1D(Block):
    # Detailed docstring + info() attached from the central registry by
    # _finalize_block_class(BVP1D) in blocks/__init__.py.

    def __init__(self, fun, bc, n_eq, domain=(0.0, 1.0), n_mesh=11, initial=None,
                 x_out=None, tol=1e-6, n_params=0, p0=None,
                 x_ports=None, interior_conditions=None):
        super().__init__()
        a, b = domain
        x0 = np.linspace(a, b, n_mesh)
        if initial is None:
            Y0 = np.zeros((n_eq, n_mesh))
        elif callable(initial):
            Y0 = np.asarray(initial(x0), dtype=float).reshape(n_eq, n_mesh)
        else:
            Y0 = np.asarray(initial, dtype=float).reshape(n_eq, n_mesh)
        y0 = Y0.T.reshape(-1)                       # node-major
        p0v = (np.zeros(n_params) if (n_params > 0 and p0 is None)
               else np.asarray([] if p0 is None else p0, dtype=float))
        xq = x0 if x_out is None else np.asarray(x_out, dtype=float)
        xp = np.array([]) if x_ports is None else np.asarray(x_ports, dtype=float)

        # n_bc (number of boundary conditions) is inferred natively from the bc
        # trace; inputs flow in dynamically through the block ports.
        blk = _trace_bvp1d(fun, bc, interior_conditions, n_eq, n_params,
                           x0, y0, p0v, np.asarray(xq, float), xp, tol)
        if blk is None:
            raise ValueError(
                "BVP1D: fun/bc/icond are not JIT-traceable "
                "(set FASTSIM_JIT_DEBUG=1 to see why)."
            )
        self._init_from(blk)
        self.__dict__["_bvp"] = {
            "n_eq": n_eq, "x_out": np.asarray(xq, float), "n_params": n_params,
        }

    @property
    def x(self):
        """Output sample points."""
        return self.__dict__["_bvp"]["x_out"]

    def solution(self):
        """Solution at the output points, shape ``(n_eq, n_out)`` (like scipy)."""
        m = self.__dict__["_bvp"]
        n, nq = m["n_eq"], len(m["x_out"])
        return np.asarray(self.outputs, dtype=float).reshape(-1)[:n * nq].reshape(nq, n).T

    def parameters(self):
        """Converged free parameters ``p``, shape ``(n_params,)``."""
        m = self.__dict__["_bvp"]
        base = m["n_eq"] * len(m["x_out"])
        return np.asarray(self.outputs, dtype=float).reshape(-1)[base:base + m["n_params"]]
