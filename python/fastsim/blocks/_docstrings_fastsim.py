"""Fastsim-specific block docstrings.

These cover blocks that don't exist in pathsim (DAE family, …) and follow the
pathsim docstring style — sections for math, tradeoffs, parameters and
attributes — so that ``Block.info()`` returns the same shape of metadata for
all blocks regardless of origin.

This file is **hand-written** and merged with ``_docstrings_pathsim.py``
(auto-generated) by the public ``DOCS`` mapping in ``blocks/__init__.py``.
Entries here override entries from the generated file when a name collides.
"""

DOCS_FASTSIM = {

    "MassMatrixDAE": """Mass-matrix DAE block.

Solves an implicit ODE with a (possibly singular) constant mass matrix:

.. math::

    \\mathbf{M} \\, \\dot{x} = f(x, u, t), \\quad y = x

`f` has the same signature as in :class:`ODE` — only the way the solver
integrates it differs. The mass matrix `M` is stored on the block and
installed into the solver's stage builder when an implicit solver is
attached. Explicit solvers see a pure ODE and will silently produce wrong
results for singular `M` — use one of the ESDIRK/DIRK/EUB families for any
non-trivial mass.

The JIT path traces `func` and derives :math:`\\partial f / \\partial x`
analytically via auto-differentiation when no analytical Jacobian is
supplied.

Parameters
----------
func : callable
    right-hand side ``f(x, u, t) -> ndarray``
mass : Mass
    mass matrix descriptor (dense, banded or sparse)
initial_value : array_like
    initial state, must have the same length as ``mass.n``
jac : callable, optional
    analytical :math:`\\partial f / \\partial x` as a flat row-major
    ``n × n`` array. If omitted, numerical or AD-derived Jacobians are
    used downstream.
""",

    "SemiExplicitDAE": """Semi-explicit Index-1 DAE block.

Solves an Index-1 system with split differential and algebraic states

.. math::

    \\dot{x} &= f_\\mathrm{dyn}(x, z, u, t) \\\\
    0       &= f_\\mathrm{alg}(x, z, u, t)

The algebraic state :math:`z` is eliminated by an inner Newton on
:math:`f_\\mathrm{alg}(x, z, u, t) = 0` at every RHS evaluation
(warmstarted from the previous call). The outer solver sees a plain ODE
in :math:`x`, so any of the explicit or implicit solvers in fastsim can
be attached.

The block output is :math:`[x; z]` (with `z` taken from the converged
inner Newton), so downstream blocks see both differential and algebraic
states.

Trade-offs vs formulating the same system as a :class:`MassMatrixDAE`
with a block-diagonal singular mass:

- explicit solvers (RKDP54, RKF78, RKV65, …) work
- smaller Newton problem per stage (size :math:`n_z` instead of
  :math:`n_x + n_z`)
- inner Newton cost per RHS call (typically 1–3 iterations once
  warmstarted)
- adaptive error control watches only :math:`x`, not :math:`z`

Parameters
----------
f_dyn : callable
    differential RHS ``f_dyn(x, z, u, t) -> ndarray`` of length ``n_x``
f_alg : callable
    algebraic constraint ``f_alg(x, z, u, t) -> ndarray`` of length ``n_z``
x0 : array_like
    initial differential state (length ``n_x``)
z0 : array_like
    initial algebraic state (length ``n_z``), used as Newton warmstart
jac_z : callable, optional
    analytical :math:`\\partial f_\\mathrm{alg} / \\partial z` as a flat
    row-major ``n_z × n_z`` array. Falls back to central differences if
    omitted.
""",

    "FullyImplicitDAE": """Fully-implicit DAE block.

For systems that can't be cast into semi-explicit or mass-matrix form —
implicit constitutive relations, mixed differential/algebraic with
non-trivial coupling — the residual form

.. math::

    F(x, \\dot{x}, u, t) = 0, \\quad y = x

is solved directly. Only implicit solvers (ESDIRK/DIRK family) work; the
block installs a fully-implicit stage builder into the engine via the
post-processing hook.

The JIT path traces `func` and derives both
:math:`\\partial F / \\partial x` and :math:`\\partial F / \\partial \\dot{x}`
via auto-differentiation when no analytical Jacobians are supplied.
Index-1 systems (singular :math:`\\partial F / \\partial \\dot{x}`):
prefer DIRK over ESDIRK for stability.

Parameters
----------
func : callable
    residual ``F(x, xdot, u, t) -> ndarray``
initial_value : array_like
    consistent :math:`x_0`. The caller is responsible for choosing it such
    that there exists an :math:`\\dot{x}_0` with
    :math:`F(x_0, \\dot{x}_0, u_0, 0) \\approx 0`.
jac_x : callable, optional
    analytical :math:`\\partial F / \\partial x` as a flat row-major
    ``n × n`` array. Falls back to numerical (central differences) if
    omitted.
jac_xdot : callable, optional
    analytical :math:`\\partial F / \\partial \\dot{x}` as a flat row-major
    ``n × n`` array. Falls back to numerical (central differences) if
    omitted.
""",

    "BVP1D": """Boundary-value problem block (native ``scipy.solve_bvp``).

Solves a first-order two-point boundary-value problem

.. math::

    y'(x) = f(x, y, p, u), \\quad \\mathrm{bc}(y(a), y(b), p, u) = 0

natively at every evaluation with a Kierzenka–Shampine collocation solver
(4th-order Lobatto-IIIa / Simpson collocation with residual-based mesh
refinement), the Newton Jacobian assembled from auto-differentiation of the
traced ``fun``/``bc``/``icond``.

Optionally with free parameters :math:`p` (extra unknowns fixed by extra
conditions — eigenvalues, unknown fluxes or lengths) and interior conditions
``icond(y@ports, p, u) = 0`` imposed at arbitrary ``x_ports`` (multipoint BVP,
beyond scipy). Boundary and parameter data flow in through the block inputs
``u`` and are re-read every evaluation; the adapted mesh and parameters are
warmstarted across evaluations. Inputs flow in dynamically through the block
ports — no input count is declared.

The block output is the solution sampled at the fixed query points ``x_out``
(4th-order Hermite interpolation), followed by the converged free parameters
``p``; use :meth:`solution` and :meth:`parameters` to read them back in shape.

Parameters
----------
fun : callable
    first-order RHS ``fun(x, y, p, u) -> y'`` for a single point (``x``
    scalar, ``y`` shape ``(n_eq,)``, ``p`` shape ``(n_params,)``)
bc : callable
    two-point boundary conditions ``bc(ya, yb, p, u) -> residual``
n_eq : int
    number of first-order equations
domain : tuple, optional
    spatial interval ``(a, b)`` (default ``(0, 1)``)
n_mesh : int, optional
    initial number of mesh nodes (default 11)
initial : callable or array_like, optional
    initial guess ``initial(x) -> (n_eq, n_mesh)`` or an array; zeros if omitted
x_out : array_like, optional
    output sample points (defaults to the initial mesh)
tol : float, optional
    collocation residual tolerance (default 1e-6)
n_params : int, optional
    number of unknown free parameters ``p`` (default 0)
p0 : array_like, optional
    initial parameter guess (defaults to zeros)
x_ports : array_like, optional
    interior-condition locations (multipoint BVP)
interior_conditions : callable, optional
    ``icond(y_ports, p, u) -> residual`` at ``x_ports``; required iff
    ``x_ports`` is given. Well-posed when
    ``len(bc) + len(icond) == n_eq + n_params``.

Attributes
----------
x : ndarray
    output sample points (``x_out``)
""",

    "AlgebraicConstraint": """Algebraic constraint block — solve ``F(x, u) = 0`` for ``x``.

Solves a square nonlinear algebraic system

.. math::

    F(x, u) = 0

for the unknown :math:`x` at every evaluation: a warmstarted Newton with
:math:`\\partial F / \\partial x` from auto-differentiation of the traced
``residual``, factored with the persistent sparse linear solver, emitting the
converged :math:`x`. It is the standalone counterpart of the inner
``z``-elimination in :class:`SemiExplicitDAE` — the same Newton core exposed as
its own block.

Use it for instantaneous algebraic relations: chemical equilibrium, flash /
vapour–liquid equilibrium, steady-state operating points, implicit constitutive
laws. Feeding it a zeroed rate ``F := f(x, u)`` recovers the quasi-steady-state
(pseudo-steady-state) approximation, without the name prescribing it.

Inputs ``u`` flow in dynamically through the block ports; no input count is
declared, identical to every other traced block.

Parameters
----------
residual : callable
    square residual ``residual(x, u) -> F`` for the unknown ``x`` (shape
    ``(n,)``) and the dynamically-wired inputs ``u``; requires
    ``len(F) == len(x0)``
x0 : array_like
    initial guess / warmstart seed; its length fixes ``n``
""",

    "CoSimulationFMU": """Co-Simulation FMU block (FMI 3.0).

Wraps an imported Functional Mock-up Unit (FMU) exported for **Co-Simulation**
as a native fastsim block. The FMU carries its own solver and is advanced one
communication step at a time: at each master step the block writes its inputs to
the FMU's input variables, calls the FMU's ``doStep`` over the communication
interval, and reads the FMU's outputs back onto its output ports. This lets a
third-party model (Modelica, Simulink, etc.) participate in a fastsim diagram
without re-implementing it.

The block is discrete in time (it exchanges data on the communication grid set
by ``dt``); between exchanges the FMU integrates internally with its own step
size. Input/output ports are derived from the FMU's model description.

Parameters
----------
fmu_path : str
    filesystem path to the ``.fmu`` archive to load (a Co-Simulation FMU)
instance_name : str, optional
    instance name passed to the FMU at instantiation (default
    ``"fmu_instance"``); used in FMU log messages
start_values : dict, optional
    mapping of FMU variable name to initial value, applied during
    initialization before the first step (default: the FMU's own defaults)
dt : float, optional
    communication step size (seconds) between master and FMU. ``None`` (default)
    uses the simulation's step; otherwise the FMU is advanced on this fixed grid
verbose : bool, optional
    forward the FMU's internal log messages to stdout (default ``False``)
""",

}
