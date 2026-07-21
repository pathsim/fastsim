//! Global constants and tolerances for fastsim.
//!
//! All numerical tolerances and hardcoded thresholds used across the engine
//! live here, so tuning happens in one file and every subsystem's assumptions
//! are visible in one place. Originally ported 1:1 from pathsim/_constants.py;
//! extended with fastsim-specific thresholds (Anderson, DAE, Filter, Source,
//! JIT, FMI).

// ======================================================================
// Global
// ======================================================================

/// Generic "basically zero" threshold, at the order of f64 epsilon. Used as a
/// division-guard and as a fallback floor where context does not suggest a
/// more specific value.
pub const TOLERANCE: f64 = 1e-16;

// ======================================================================
// Simulation (top-level)
// ======================================================================

pub const SIM_TIMESTEP: f64 = 0.01;
pub const SIM_TIMESTEP_MIN: f64 = 1e-16;
pub const SIM_ITERATIONS_MAX: usize = 200;

/// Max number of shooting iterations in `Simulation::periodic_steady_state`.
/// Anderson-accelerated shooting typically converges in 5–15 iterations on
/// smooth periodic problems; 50 leaves headroom for stiff or weakly-damped
/// limit cycles while still failing fast on pathological cases.
pub const SIM_PSS_ITERATIONS_MAX: usize = 50;

// ======================================================================
// Solver (per-block integrator)
// ======================================================================

pub const SOL_TOLERANCE_LTE_ABS: f64 = 1e-8;
pub const SOL_TOLERANCE_LTE_REL: f64 = 1e-4;
pub const SOL_ITERATIONS_MAX: usize = 200;
/// Default minimum step-size rescale factor for RK / DIRK / ESDIRK.
/// Kept at PathSim's historical 0.1 to preserve trajectory-match parity
/// (drop-in replacement promise).  Solvers with reasons to deviate
/// override `Solver::scale_min` per-instance — GEAR52A bumps it to 0.2
/// to avoid the dt-cascade on stiff bootstrap.
pub const SOL_SCALE_MIN: f64 = 0.1;
pub const SOL_SCALE_MAX: f64 = 10.0;
pub const SOL_BETA: f64 = 0.9;

/// Gustafsson PI step-size controller numerator for the I term:
/// `factor ∝ err^(-PI_ALPHA / p)` where `p` is the truncation order.
/// Combined with `PI_BETA / p` on the previous error this gives the
/// PI controller of Hairer-Wanner II §IV.2 (DOPRI5 default tuning).
/// Falls back to pure I-term (`err_prev = 0` sentinel) on the first
/// step and after every step rejection.
pub const PI_ALPHA: f64 = 0.7;
pub const PI_BETA: f64 = 0.4;

// ======================================================================
// Optimizer (Anderson / NewtonAnderson)
// ======================================================================

/// Buffer depth for Anderson acceleration (history length). 4 matches pathsim.
pub const OPT_HISTORY: usize = 4;

/// ARKODE/CVODES-style nonlinear-solve safety coefficient.  Used as the
/// unitless threshold against the WRMS-scaled residual reported by every
/// implicit-stage builder, the algebraic-loop ConnectionBoosters, and the
/// steady-state operating-point loop.  0.1 is the canonical default that
/// keeps inner-Newton accuracy proportional to the outer step's
/// `(atol + rtol·|x|)` weights.
pub const NLS_COEF: f64 = 0.1;

// ======================================================================
// Events
// ======================================================================

pub const EVT_TOLERANCE: f64 = 1e-4;

/// Cap on dense-output bisection probes per event localisation (each probe is
/// one DAG update on interpolated states — cheap next to the full
/// re-integration a secant retry costs, but still bounded).
pub const DENSE_EVT_MAX_PROBES: usize = 32;

/// Smallest θ the localisation probes (a retry ratio of ~0 would stall dt).
pub const DENSE_EVT_THETA_MIN: f64 = 1e-6;

/// Offset keeping the first probe strictly inside the step (θ < 1).
pub const DENSE_EVT_THETA_EDGE: f64 = 1e-9;

/// Bracket width at which the θ-bisection stops refining.
pub const DENSE_EVT_THETA_WIDTH: f64 = 1e-12;

// ======================================================================
// Numerical Jacobian (fallback when no symbolic Jacobian exists)
// ======================================================================

pub const NUM_JAC_REL: f64 = 1e-3;
pub const NUM_JAC_TOL: f64 = 1e-12;

/// Perturbation base for central-difference numerical Jacobians in implicit
/// stage solves (`∂F/∂x`, `∂F/∂ẋ`, `∂g/∂z`). The per-column step is
/// `sqrt(base) * max(|x_j|, 1)`, the standard sqrt(machine-eps) heuristic.
pub const STAGE_NUM_JAC_PERTURB: f64 = 1e-7;

/// Entries in the computed Jacobian smaller than this are treated as zero,
/// letting strength-reduction passes prune them.
pub const JACOBIAN_ZERO_THRESHOLD: f64 = 1e-10;

/// Gating for the cached Newton linear solver's sparse-LU path. A sparse
/// factorization only pays off for large *and* genuinely sparse Newton matrices
/// (e.g. a block-diagonal Jacobian from many independent state blocks). Below
/// `LINSOLVE_SPARSE_MIN_DIM` the dense partial-pivot LU is always faster, and a
/// matrix denser than `LINSOLVE_SPARSE_MAX_DENSITY` (nonzeros / n²) is solved
/// dense regardless of size, since sparse bookkeeping then costs more than it
/// saves.
pub const LINSOLVE_SPARSE_MIN_DIM: usize = 48;
pub const LINSOLVE_SPARSE_MAX_DENSITY: f64 = 0.25;

/// Central-difference step base for the numerical Jacobian of a Python
/// `DynamicalSystem` callback (no symbolic derivative available). The per-
/// element step is `max(step * |x_j|, step)`, i.e. ~sqrt(machine-eps).
pub const PY_DYNSYS_JAC_FD_STEP: f64 = 1e-8;

/// Mass-matrix entries below this magnitude count as a structurally zero row
/// when classifying a `MassMatrixStageBuilder`'s algebraic (singular-`M`) rows.
pub const MASS_ZERO_THRESHOLD: f64 = 1e-14;

// ======================================================================
// DAE (algebraic-constraint solving in semi-explicit / fully-implicit blocks)
// ======================================================================

/// Newton convergence target for algebraic constraint `g(x,z,u,t)=0`.
pub const DAE_ALGEBRAIC_TOLERANCE: f64 = 1e-10;
/// Singularity detection threshold on `∂g/∂z`. Below this the inner Newton
/// bails out early instead of dividing by a near-zero pivot.
pub const DAE_JACOBIAN_SINGULAR_TOL: f64 = 1e-14;
/// Max iterations for consistent-ẋ Newton in fully-implicit DAE blocks.
pub const DAE_FULLIMPLICIT_MAX_ITER: usize = 20;

// ======================================================================
// BVP 1D (collocation BVP solver — scipy.solve_bvp rebuild)
// ======================================================================

/// Node cap and max solve→refine rounds per evaluation of the collocation
/// BVP solver (`src/optim/colloc_bvp.rs`).
pub const PDE_BVP_MAX_NODES: usize = 2000;
pub const PDE_BVP_MAX_ROUNDS: usize = 16;

// ======================================================================
// LTI / filters (polynomial projection)
// ======================================================================

/// Relative threshold at which residual imaginary parts from conjugate-pair
/// expansion are collapsed to zero. Used by Butterworth / transfer-function
/// constructors and by Gilbert pole-residue realization.
pub const FILTER_POLY_REAL_TOL: f64 = 1e-9;

// ======================================================================
// Sources / interpolation
// ======================================================================

/// Minimum rise/fall time for Trapezoidal-pulse sources. Guards against the
/// `dt / t_rise` singularity when the user sets rise-time = 0.
pub const SOURCE_RISE_FALL_TIME_MIN: f64 = 1e-12;

/// Threshold for the sample-time gap `|t1 − t0|` in linear interpolation.
/// Below this the interpolator treats the two samples as coincident.
pub const INTERPOLATION_TIME_TOL: f64 = 1e-15;

/// Extra time margin kept when pruning the continuous-time `Delay` ring buffer,
/// so the oldest sample needed for interpolation at `t - tau` is never dropped.
pub const DELAY_PRUNE_MARGIN: f64 = 0.01;

// ======================================================================
// JIT
// ======================================================================

/// Floating-point equality tolerance inside compiled tapes (CmpOp::Eq / Ne).
/// At the noise floor so only truly identical values compare equal.
pub const JIT_FLOAT_EQ_TOL: f64 = 1e-15;

/// Relative central-difference step size for numeric-derivative fallbacks.
pub const JIT_FINITE_DIFF_REL: f64 = 1e-6;

/// Minimum term count for bundling an Add/Fma accumulation chain into one
/// fused `Reduce`/`Dot` node at tape lowering (`ssa::optimize::
/// reassociate_chains`). Below this, the unrolled binary ops beat the fused
/// kernel's gather-into-scratch overhead.
pub const JIT_REASSOC_MIN_TERMS: usize = 4;

/// Time tolerance for firing scheduled (time) events in generated FMU code:
/// an event whose scheduled time is within this of the current time fires now.
/// Emitted into the FMU wrapper as `EVENT_TIME_TOL` (centralised, not inline).
pub const FMU_EVENT_TIME_TOL: f64 = 1e-9;

/// Relative tolerance on the sample step `dt` before the Spectrum block's
/// twiddle-factor rotator cache is considered stale and rebuilt.
pub const SPECTRUM_DT_REL_TOL: f64 = 1e-12;

// ======================================================================
// Static compile (tape optimization)
// ======================================================================

/// Max fixpoint passes of the op-graph optimizer (CSE / fold / DCE / strength
/// reduction) when lowering a spliced graph to a tape. The optimizer reports
/// convergence and the loop breaks early; this is just the safety bound.
pub const COMPILE_OPTIMIZE_MAX_PASSES: usize = 8;

// ======================================================================
// FMI / FMU
// ======================================================================

/// Initialization tolerance passed to `fmiEnterInitializationMode` for FMI 3.0.
pub const FMI_INITIALIZATION_TOL: f64 = 1e-6;
