"""Native algebraic constraint block: solve ``F(x, u) = 0`` for ``x``.

``AlgebraicConstraint`` exposes the same warmstarted Newton + AD-Jacobian core
that ``SemiExplicitDAE`` uses internally to eliminate its algebraic variable, but
as a standalone block: at every evaluation it solves

    F(x, inputs) = 0

for ``x`` (Newton, ``∂F/∂x`` from AD on the traced ``residual``, warmstarted from
the previous solve, factored with the persistent sparse linear solver) and emits
the converged ``x``. Inputs flow in dynamically through the block's ports — there
is no input-count declaration, identical to every other traced block.

It is the base primitive for instantaneous algebraic relations: chemical
equilibrium, flash / vapor-liquid equilibrium, steady-state operating points,
implicit constitutive laws. Feeding it a zeroed rate ``F := f(x, u)`` recovers the
quasi-steady-state (pseudo-steady-state) approximation.
"""

import numpy as np

from fastsim._fastsim import _trace_algebraic_constraint, Block


class AlgebraicConstraint(Block):
    # Detailed docstring + info() attached from the central registry by
    # _finalize_block_class(AlgebraicConstraint) in blocks/__init__.py.

    def __init__(self, residual, x0):
        super().__init__()
        x0v = np.asarray(x0, dtype=float).reshape(-1)
        blk = _trace_algebraic_constraint(residual, x0v)
        if blk is None:
            raise ValueError(
                "AlgebraicConstraint: residual is not JIT-traceable "
                "(set FASTSIM_JIT_DEBUG=1 to see why)."
            )
        self._init_from(blk)
