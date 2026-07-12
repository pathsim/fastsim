// Solver factory functions — configure a `Solver` instance from a `Tableau`.
//
// Butcher tableau data lives in `tableaus.rs` as `pub const` definitions.
// Each historical `*_factory()` function is now a thin wrapper around
// `build_from_tableau()`, which sets up the step/solve/buffer callbacks
// according to the tableau's `TableauKind`.
//
// Three special solvers (EUF, EUB, SteadyState) don't fit the RK tableau
// schema and keep their bespoke factories below.

use std::cell::RefCell;
use std::rc::Rc;

use crate::constants::*;
use crate::solvers::solver::{Solver, Jacobian};
use crate::solvers::tableaus::{self as tbl, Tableau, TableauKind};
use crate::solvers::stage::OdeStageBuilder;
use crate::solvers::bdf::{
    compute_bdf_alphas, compute_bdf_coefficients, compute_bdf_l_vector, Nordsieck, NDF_KAPPA,
};
use crate::optim::anderson::Optimizer;

/// Type alias for solver factory closures.
pub type SolverFactory = Box<dyn Fn(&[f64]) -> Solver>;

// ======================================================================================
// Tableau → Solver population
// ======================================================================================

/// Copy tableau data into solver fields.  Shape-preserving conversion:
/// empty bt[i] slices map back to `Option::None` (ESDIRK explicit first stage).
fn populate_from_tableau(s: &mut Solver, t: &Tableau) {
    s.type_name = t.name;
    s.n = t.n;
    s.m = t.m;
    s.s = t.s;
    s.is_adaptive = t.is_adaptive();
    s.is_explicit = t.is_explicit();
    s.is_implicit = t.is_implicit();
    s.eval_stages = t.eval_stages.to_vec();
    s.bt = t.bt.iter()
        .map(|row| if row.is_empty() { None } else { Some(row.to_vec()) })
        .collect();
    if !t.tr.is_empty() { s.tr = Some(t.tr.to_vec()); }
    if !t.a_final.is_empty() { s.a_final = Some(t.a_final.to_vec()); }
}

/// Build a `SolverFactory` that assembles a `Solver` from the given tableau.
/// Callbacks (step/solve/buffer) are selected by the tableau's `kind`.
pub fn build_from_tableau(
    t: &'static Tableau,
    tol_abs: f64,
    tol_rel: f64,
) -> SolverFactory {
    match t.kind {
        TableauKind::ExplicitRK => Box::new(move |iv: &[f64]| {
            let mut s = Solver::new(iv, tol_abs, tol_rel);
            populate_from_tableau(&mut s, t);
            s.step_fn = Some(make_explicit_rk_step());
            s
        }),
        TableauKind::DIRK | TableauKind::ESDIRK => Box::new(move |iv: &[f64]| {
            let mut s = Solver::new(iv, tol_abs, tol_rel);
            populate_from_tableau(&mut s, t);
            s.opt = Some(Optimizer::default_newton_anderson());
            s.buffer_fn = Some(make_implicit_buffer());
            s.solve_fn = Some(make_dirk_solve());
            s.step_fn = Some(make_dirk_step());
            s.stage_builder = Some(Box::new(OdeStageBuilder::new()));
            s
        }),
    }
}

// ======================================================================================
// Explicit RK step callback
// ======================================================================================

/// Generic explicit Runge-Kutta step callback.
fn make_explicit_rk_step() -> Box<dyn FnMut(&mut Solver, &[f64], f64) -> (bool, f64, Option<f64>)> {
    Box::new(|solver: &mut Solver, f: &[f64], dt: f64| {
        let stage = solver._stage;
        let n = f.len();

        // Dynamic resize: adapt solver state to match f dimension
        if n != solver.x.len() {
            solver.x.resize(n, 0.0);
            solver.initial_value.resize(n, 0.0);
            for h in solver.history.iter_mut() { h.resize(n, 0.0); }
        }

        // Ensure ks has enough slots, then store slope in-place
        while solver.ks.len() <= stage { solver.ks.push(Vec::new()); }
        let ks_stage = &mut solver.ks[stage];
        ks_stage.resize(n, 0.0);
        ks_stage.copy_from_slice(f);

        if solver.history.is_empty() { return (true, 0.0, None); }
        if stage >= solver.bt.len() { return (true, 0.0, None); }

        let x_0 = &solver.history[0];
        let bt_stage = match &solver.bt[stage] {
            Some(coeffs) => coeffs,
            None => return (true, 0.0, None),
        };

        // x_new = x_0 + dt * sum(b_i * k_i)
        for j in 0..n {
            let mut s = 0.0;
            for (i, &b) in bt_stage.iter().enumerate() {
                if i < solver.ks.len() && !solver.ks[i].is_empty() {
                    s += solver.ks[i][j] * b;
                }
            }
            solver.x[j] = x_0[j] + dt * s;
        }

        if solver.tr.is_none() || !solver.is_last_stage() {
            return (true, 0.0, None);
        }

        solver.error_controller(dt)
    })
}

// ======================================================================================
// Explicit RK factories (tableau-backed)
// ======================================================================================

// --- Fixed-step ---

pub fn ssprk22_factory() -> SolverFactory {
    build_from_tableau(&tbl::SSPRK22, SOL_TOLERANCE_LTE_ABS, SOL_TOLERANCE_LTE_REL)
}

pub fn rk4_factory() -> SolverFactory {
    build_from_tableau(&tbl::RK4, SOL_TOLERANCE_LTE_ABS, SOL_TOLERANCE_LTE_REL)
}

pub fn ssprk33_factory() -> SolverFactory {
    build_from_tableau(&tbl::SSPRK33, SOL_TOLERANCE_LTE_ABS, SOL_TOLERANCE_LTE_REL)
}

pub fn ssprk34_factory() -> SolverFactory {
    build_from_tableau(&tbl::SSPRK34, SOL_TOLERANCE_LTE_ABS, SOL_TOLERANCE_LTE_REL)
}

/// EUF: Explicit forward Euler — custom step, not a standard RK tableau.
pub fn euf_factory() -> SolverFactory {
    Box::new(|iv: &[f64]| {
        let mut s = Solver::with_defaults(iv);
        s.type_name = "EUF";
        s.is_explicit = true;
        s.n = 1; s.s = 1;
        s.eval_stages = vec![0.0];
        s.step_fn = Some(Box::new(|solver: &mut Solver, f: &[f64], dt: f64| {
            if solver.history.is_empty() { return (true, 0.0, None); }
            let x_0 = &solver.history[0];
            solver.x = x_0.iter().zip(f.iter()).map(|(&x0, &fi)| x0 + dt * fi).collect();
            (true, 0.0, None)
        }));
        s
    })
}

// --- Adaptive ---

pub fn rkf21_factory(tol_abs: f64, tol_rel: f64) -> SolverFactory {
    build_from_tableau(&tbl::RKF21, tol_abs, tol_rel)
}

pub fn rkbs32_factory(tol_abs: f64, tol_rel: f64) -> SolverFactory {
    build_from_tableau(&tbl::RKBS32, tol_abs, tol_rel)
}

pub fn rkf45_factory(tol_abs: f64, tol_rel: f64) -> SolverFactory {
    build_from_tableau(&tbl::RKF45, tol_abs, tol_rel)
}

pub fn rkck54_factory(tol_abs: f64, tol_rel: f64) -> SolverFactory {
    build_from_tableau(&tbl::RKCK54, tol_abs, tol_rel)
}

pub fn rkdp54_factory(tol_abs: f64, tol_rel: f64) -> SolverFactory {
    build_from_tableau(&tbl::RKDP54, tol_abs, tol_rel)
}

pub fn rkv65_factory(tol_abs: f64, tol_rel: f64) -> SolverFactory {
    build_from_tableau(&tbl::RKV65, tol_abs, tol_rel)
}

pub fn rkf78_factory(tol_abs: f64, tol_rel: f64) -> SolverFactory {
    build_from_tableau(&tbl::RKF78, tol_abs, tol_rel)
}

pub fn rkdp87_factory(tol_abs: f64, tol_rel: f64) -> SolverFactory {
    build_from_tableau(&tbl::RKDP87, tol_abs, tol_rel)
}

// ======================================================================================
// Implicit DIRK / ESDIRK callbacks
// ======================================================================================

/// Generic DIRK solve callback — dispatches the implicit update to the
/// installed `StageBuilder`.
///
/// Responsibilities kept here: state resizing, slope storage, ESDIRK explicit-
/// stage filtering.  The residual / Newton-matrix assembly (which depends on
/// the problem form — ODE, mass-matrix DAE, semi-explicit, fully implicit)
/// lives inside the builder.
fn make_dirk_solve() -> Box<dyn FnMut(&mut Solver, &[f64], Option<Jacobian>, f64) -> f64> {
    Box::new(move |solver: &mut Solver, f: &[f64], j: Option<Jacobian>, dt: f64| {
        let stage = solver._stage;
        let n = f.len();

        // Dynamic resize
        if n != solver.x.len() {
            solver.x.resize(n, 0.0);
            solver.initial_value.resize(n, 0.0);
            for h in solver.history.iter_mut() { h.resize(n, 0.0); }
        }

        // ESDIRK: first stage is explicit -> early exit
        if solver.is_first_stage()
            && stage < solver.bt.len() && solver.bt[stage].is_none() { return 0.0; }

        // Store slope in-place
        while solver.ks.len() <= stage { solver.ks.push(Vec::new()); }
        solver.ks[stage].resize(n, 0.0);
        solver.ks[stage].copy_from_slice(f);

        if solver.history.is_empty() { return 0.0; }
        if stage >= solver.bt.len() || solver.bt[stage].is_none() { return 0.0; }

        // Take the builder out to call it with &mut solver, put it back afterwards.
        let mut builder = solver.stage_builder.take()
            .expect("DIRK solve_fn requires stage_builder");
        let residual = builder.solve_stage(solver, f, j, dt);
        solver.stage_builder = Some(builder);
        residual
    })
}

/// Generic DIRK step callback — handles explicit first stage (ESDIRK) and error control.
fn make_dirk_step() -> Box<dyn FnMut(&mut Solver, &[f64], f64) -> (bool, f64, Option<f64>)> {
    Box::new(|solver: &mut Solver, f: &[f64], dt: f64| {
        let stage = solver._stage;
        let n = f.len();

        // ESDIRK: first stage is explicit — store slope
        if solver.is_first_stage()
            && stage < solver.bt.len() && solver.bt[stage].is_none() {
                while solver.ks.len() <= stage { solver.ks.push(Vec::new()); }
                solver.ks[stage].resize(n, 0.0);
                solver.ks[stage].copy_from_slice(f);
            }

        // Last stage: error control
        if solver.is_last_stage() {
            // Non-stiffly accurate: compute final output from a_final
            if let Some(ref a_coeffs) = solver.a_final {
                if !solver.history.is_empty() {
                    let x_0 = &solver.history[0];
                    for j in 0..n {
                        let mut s = 0.0;
                        for (i, &a) in a_coeffs.iter().enumerate() {
                            if i < solver.ks.len() && !solver.ks[i].is_empty() {
                                s += solver.ks[i][j] * a;
                            }
                        }
                        solver.x[j] = x_0[j] + dt * s;
                    }
                }
            }

            if solver.tr.is_none() {
                return (true, 0.0, None);
            }
            return solver.error_controller(dt);
        }

        (true, 0.0, None)
    })
}

/// Generic DIRK buffer callback — resets optimizer.
fn make_implicit_buffer() -> Box<dyn FnMut(&mut Solver, f64)> {
    Box::new(|solver: &mut Solver, _dt: f64| {
        solver.push_state_history();
        solver._stage = 0;
        if let Some(ref mut opt) = solver.opt {
            opt.reset();
        }
    })
}

// ======================================================================================
// Implicit factories (tableau-backed)
// ======================================================================================

pub fn dirk2_factory() -> SolverFactory {
    build_from_tableau(&tbl::DIRK2, SOL_TOLERANCE_LTE_ABS, SOL_TOLERANCE_LTE_REL)
}

pub fn dirk3_factory() -> SolverFactory {
    build_from_tableau(&tbl::DIRK3, SOL_TOLERANCE_LTE_ABS, SOL_TOLERANCE_LTE_REL)
}

pub fn esdirk4_factory() -> SolverFactory {
    build_from_tableau(&tbl::ESDIRK4, SOL_TOLERANCE_LTE_ABS, SOL_TOLERANCE_LTE_REL)
}

pub fn esdirk32_factory(tol_abs: f64, tol_rel: f64) -> SolverFactory {
    build_from_tableau(&tbl::ESDIRK32, tol_abs, tol_rel)
}

pub fn esdirk43_factory(tol_abs: f64, tol_rel: f64) -> SolverFactory {
    build_from_tableau(&tbl::ESDIRK43, tol_abs, tol_rel)
}

pub fn esdirk54_factory(tol_abs: f64, tol_rel: f64) -> SolverFactory {
    build_from_tableau(&tbl::ESDIRK54, tol_abs, tol_rel)
}

// ======================================================================================
// Special solvers (custom solve_fn)
// ======================================================================================

/// EUB: Implicit backward Euler.
pub fn eub_factory() -> SolverFactory {
    Box::new(|iv: &[f64]| {
        let mut s = Solver::with_defaults(iv);
        s.type_name = "EUB";
        s.is_explicit = false;
        s.is_implicit = true;
        s.n = 1; s.s = 1;
        s.eval_stages = vec![1.0];
        s.opt = Some(Optimizer::default_newton_anderson());
        s.buffer_fn = Some(make_implicit_buffer());
        {
            let mut g_buf: Vec<f64> = Vec::new();
            let mut jac_buf: Vec<f64> = Vec::new();
            s.solve_fn = Some(Box::new(move |solver: &mut Solver, f: &[f64], j: Option<Jacobian>, dt: f64| {
                if solver.history.is_empty() { return 0.0; }
                let n = f.len();
                g_buf.resize(n, 0.0);
                let x_0 = &solver.history[0];
                for i in 0..n { g_buf[i] = x_0[i] + dt * f[i]; }
                // WRMS norm of (g − x) BEFORE the Newton step.  Same scale
                // convention as the StageBuilders so the simulation-level
                // NLS_COEF threshold is consistent.
                let wrms = solver.wrms_norm_diff(&g_buf, &solver.x);
                let opt = solver.opt.as_mut().unwrap();
                let _l2 = match j {
                    Some(Jacobian::Scalar(jac)) => {
                        opt.step(&mut solver.x, &g_buf, Some(dt * jac))
                    }
                    Some(Jacobian::Matrix(ref jac_flat, nr)) => {
                        let nn = nr * nr;
                        jac_buf.resize(nn, 0.0);
                        for i in 0..nn { jac_buf[i] = jac_flat[i] * dt; }
                        opt.step_matrix(&mut solver.x, &g_buf, Some(&jac_buf), nr)
                    }
                    Some(Jacobian::Sparse(ref sj)) => {
                        // Dense fallback (SAJ-1).
                        let nr = sj.n;
                        jac_buf.clear();
                        jac_buf.resize(nr * nr, 0.0);
                        for k in 0..sj.values.len() {
                            jac_buf[sj.rows[k] as usize * nr + sj.cols[k] as usize] = sj.values[k] * dt;
                        }
                        opt.step_matrix(&mut solver.x, &g_buf, Some(&jac_buf), nr)
                    }
                    None => {
                        opt.step(&mut solver.x, &g_buf, None)
                    }
                };
                wrms
            }));
        }
        s
    })
}

/// SteadyState pseudo-solver for DC operating point.
pub fn steadystate_factory() -> SolverFactory {
    Box::new(|iv: &[f64]| {
        let mut s = Solver::with_defaults(iv);
        s.type_name = "SteadyState"; s.is_explicit = false; s.is_implicit = true; s.n = 1; s.s = 1;
        s.eval_stages = vec![1.0];
        s.opt = Some(Optimizer::default_newton_anderson());
        s.buffer_fn = Some(make_implicit_buffer());
        {
            let mut g_buf: Vec<f64> = Vec::new();
            let mut jac_buf: Vec<f64> = Vec::new();
            s.solve_fn = Some(Box::new(move |solver: &mut Solver, f: &[f64], j: Option<Jacobian>, _dt: f64| {
                let n = f.len();
                g_buf.resize(n, 0.0);
                for i in 0..n { g_buf[i] = solver.x[i] + f[i]; }
                // WRMS norm of the steady-state residual `f` (since
                // g − x = f at the operating point) BEFORE the Newton step.
                // Matches the scaling used by all StageBuilders so the
                // simulation-level NLS_COEF threshold is consistent.
                let wrms = solver.wrms_norm(f);
                let opt = solver.opt.as_mut().unwrap();
                let _l2 = match j {
                    Some(Jacobian::Scalar(jac)) => {
                        opt.step(&mut solver.x, &g_buf, Some(1.0 + jac))
                    }
                    Some(Jacobian::Matrix(ref jac_flat, nr)) => {
                        jac_buf.resize(nr * nr, 0.0);
                        for r in 0..nr {
                            for c in 0..nr {
                                let ident = if r == c { 1.0 } else { 0.0 };
                                jac_buf[r*nr+c] = ident + jac_flat[r*nr+c];
                            }
                        }
                        opt.step_matrix(&mut solver.x, &g_buf, Some(&jac_buf), nr)
                    }
                    Some(Jacobian::Sparse(ref sj)) => {
                        // Dense fallback (SAJ-1): A = I + J.
                        let nr = sj.n;
                        jac_buf.clear();
                        jac_buf.resize(nr * nr, 0.0);
                        for i in 0..nr { jac_buf[i * nr + i] = 1.0; }
                        for k in 0..sj.values.len() {
                            jac_buf[sj.rows[k] as usize * nr + sj.cols[k] as usize] += sj.values[k];
                        }
                        opt.step_matrix(&mut solver.x, &g_buf, Some(&jac_buf), nr)
                    }
                    None => {
                        opt.step(&mut solver.x, &g_buf, None)
                    }
                };
                wrms
            }));
        }
        s
    })
}

// ======================================================================================
// GEAR52A — variable-step, variable-order BDF (orders 1..5), Nordsieck form.
// ======================================================================================
//
// Implementation notes
// --------------------
// State that has to persist across the three closures (`buffer_fn`,
// `solve_fn`, `step_fn`) lives in `GearState` behind `Rc<RefCell<…>>`.
// `Solver`'s public surface is kept untouched on purpose so RK solvers see
// no new fields they don't use.
//
// Per timestep the integrator works in three coordinated stages:
//
//   1. **Predictor (Pascal-advance on Nordsieck z).**  The polynomial that
//      stored z represents (degree `q`, fitted through past `q+1` history
//      points) is advanced from `t_{n-1}` to `t_n` in-place.  z_pred[0] is
//      the predicted x_n; z_pred[1] = h·y'_pred.  This is mathematically
//      equivalent to a Lagrange extrapolation but uses constant Pascal
//      additions instead of explicit basis evaluation.
//
//   2. **Corrector (Newton on the BDF stage equation).**  The implicit
//      equation `x_n = z_pred[0] − l[0]·z_pred[1] + l[0]·h·f(x_n)` is
//      driven to convergence via the existing Anderson optimiser.  `l[0]`
//      is the BDF beta coefficient; `(z_pred[0] − l[0]·z_pred[1])` is the
//      Nordsieck reformulation of `Σ K_j·x_{n-1-j}` (provably equivalent —
//      the polynomial Pascal-advances to give the same linear functional).
//
//   3. **Error control (history-residual based, unchanged).**  After
//      Newton converges, build truncation residuals at orders `n_active±1`
//      from the BDF coefficient table and `solver.history`.  PI controller
//      decides accept/reject; order is nudged based on `(err_m, err_p)`.
//
// **Re-init policy.**  After every buffer push we re-fit z from history via
// `Nordsieck::init_from_history` instead of carrying state incrementally.
// Trade-off: one extra `(q+1)×(q+1)` LU per step, but trivial step-rejection
// and order-change handling — z is always consistent with the current
// history, so reverts and order jumps cost nothing extra.  The full
// SUNDIALS-style incremental Nordsieck would be a second-pass optimisation
// once benchmarks show the per-step LU cost matters; in Rust on small
// systems it's currently dominated by the f-evaluation.
//
// Order ramp-up (LSODA-style; no embedded ESDIRK startup): the first
// `n_min` steps run at the lowest order that's compatible with the
// available history depth.  As soon as enough history has been buffered,
// the variable-order machinery kicks in.

struct GearState {
    /// Target order for the *next* step.  Adapted by the error controller
    /// in `step_fn`.  Clamped against `history.len()` in `buffer_fn` to
    /// give `n_active`.
    n_target: usize,
    n_min: usize,
    n_max: usize,
    /// Order actually used to advance the current step.
    n_active: usize,

    // Nordsieck representation of the underlying polynomial.
    nordsieck: Nordsieck,
    /// BDF `l`-vector at `(n_active, h_n)` — drives Pascal predictor +
    /// l-vector corrector and supplies `β = l[0]`.
    l: Vec<f64>,
    /// Snapshot of `z_pred[0]` taken in `buffer_fn` (right after `predict`)
    /// so `solve_fn` can reuse it across Newton iterations without holding
    /// a mutable borrow on `Nordsieck`.
    z_pred_0: Vec<f64>,
    /// Snapshot of `z_pred[1]` — same rationale.
    z_pred_1: Vec<f64>,

    // Coefficients at `n_active − 1` for the lower-order error estimate.
    have_down: bool,
    beta_down: f64,
    k_down: Vec<f64>,
    // Coefficients at `n_active + 1` for the higher-order error estimate.
    have_up: bool,
    beta_up: f64,
    k_up: Vec<f64>,

    // First-step LTE estimation.  At order 1 the standard `tr_m` (residual
    // at order n_active − 1) doesn't exist, so the controller used to
    // accept blindly — leaving an O(h_0²) error floor on conservative or
    // oscillatory systems where it doesn't dissipate.  Workaround: compare
    // the BDF1 result against an explicit forward-Euler reference,
    // `LTE ≈ (x_n − y_0 − h·f(y_0)) / 2`, and feed it through the same PI
    // controller as the rest of the order-adapt machinery.  Captures
    // `f(y_0)` from the first solve_fn call of the step (where solver.x is
    // still the predictor = y_0 for the first step).
    f_y0_buf: Vec<f64>,
    f_y0_captured: bool,

    /// Order-change throttle: # of consecutive successful steps at the
    /// current order.  ode15s requires `q+1` such steps before the order
    /// can change again — prevents the controller from thrashing the
    /// order at phase transitions.  Reset to 0 whenever `n_target` moves.
    steps_at_order: usize,
}

impl GearState {
    fn new(n_state: usize) -> Self {
        // Nordsieck poly degree = BDF order − 1.  At BDF max order 5 the
        // polynomial is degree 4, so `Nordsieck::q_max = 4`.  z has 5 entries
        // (z[0..=4]) — exactly enough to express x_{n-1}, x_{n-2}, …, x_{n-5}
        // via Pascal predictor.  For the equivalence between the Nordsieck
        // residual `g = z_pred[0] − β·z_pred[1] + β·h·f` and the BDF residual
        // `g = β·h·f + Σ K_j · x_{n-1-j}` to hold, the Nordsieck polynomial
        // must have degree (q−1) when BDF is at order q.
        Self {
            n_target: 2, n_min: 2, n_max: 5, n_active: 1,
            nordsieck: Nordsieck::new(n_state, 4),
            l: vec![1.0, 1.0],
            z_pred_0: vec![0.0; n_state],
            z_pred_1: vec![0.0; n_state],
            have_down: false, beta_down: 0.0, k_down: Vec::new(),
            have_up: false, beta_up: 0.0, k_up: Vec::new(),
            f_y0_buf: vec![0.0; n_state],
            f_y0_captured: false,
            steps_at_order: 0,
        }
    }
}

/// GEAR52A: variable-step, variable-order BDF stiff solver.
///
/// Adapts both timestep and order (1..5) per step.  Uses an LSODA-style order
/// ramp-up (BDF1 → … → BDF5 as history fills) instead of an embedded
/// single-step startup integrator like PathSim's GEAR (ESDIRK32-startup).
/// Enables the Gustafsson PI step-size controller via
/// `Solver::use_pi_controller` because BDF rejection cost (order reset) is
/// high and damping the step sequence pays off.
pub fn gear52a_factory(tol_abs: f64, tol_rel: f64) -> SolverFactory {
    Box::new(move |iv: &[f64]| {
        let mut s = Solver::new(iv, tol_abs, tol_rel);
        s.type_name = "GEAR52A";
        s.is_explicit = false;
        s.is_implicit = true;
        s.is_adaptive = true;
        // Single implicit stage: g = β·dt·f + Σ α_i · x_{n-i}.
        s.n = 2;
        s.m = 1;
        s.s = 1;
        s.eval_stages = vec![1.0];
        // History depth `n_max + 1` so we can also build the order-`n_max+1`
        // truncation residual `tr_p` once enough steps have run.
        s.history_maxlen = 6;
        s.history_dt_maxlen = 6;
        s.use_pi_controller = true;
        // Less-aggressive shrink on rejected steps (5x per hit instead of
        // 10x) keeps GEAR's stiff-bootstrap reject-cascade from underflowing
        // `dt_min` on systems with rapid initial transients (HIRES at
        // dt0 = t_end/100 was the canonical failure).
        s.scale_min = 0.2;
        s.opt = Some(Optimizer::default_newton_anderson());

        let state = Rc::new(RefCell::new(GearState::new(iv.len())));

        // -- buffer_fn --
        // Pushes (x, dt) onto histories, then re-fits Nordsieck z from
        // history (always — keeps z consistent with history through
        // step rejections and order changes for free).  Pascal-predict
        // advances z to t_n, leaving z_pred[0] as the Newton initial guess.
        let st = state.clone();
        s.buffer_fn = Some(Box::new(move |solver: &mut Solver, dt: f64| {
            solver.push_state_history();
            // Push dt in lockstep.
            while solver.history_dt.len() >= solver.history_dt_maxlen {
                solver.history_dt.pop_back();
            }
            solver.history_dt.push_front(dt);

            solver._stage = 0;
            if let Some(opt) = solver.opt.as_mut() { opt.reset(); }

            let mut g = st.borrow_mut();
            let h_len = solver.history.len();
            // Allow first-step LTE capture for the upcoming Newton solve.
            g.f_y0_captured = false;

            // Resize Nordsieck if state dimension changed (FMI blocks etc.).
            if g.nordsieck.z[0].len() != solver.x.len() {
                g.nordsieck = Nordsieck::new(solver.x.len(), g.n_max - 1);
                g.z_pred_0 = vec![0.0; solver.x.len()];
                g.z_pred_1 = vec![0.0; solver.x.len()];
            }

            // Decide order: clamped by history depth available for the
            // polynomial fit. The Nordsieck polynomial has degree
            // (n_active − 1) and is fitted through `n_active` past history
            // points, so we need h_len >= n_active.
            g.n_active = g.n_target.min(h_len).max(1);

            // Sync solver.n / m so Solver::step_factor (PI) sees the right
            // accuracy order: order = m.min(n) + 1 = n_active.
            solver.n = g.n_active;
            solver.m = g.n_active.saturating_sub(1);

            // Re-fit z from history.  Nordsieck poly degree = n_active − 1
            // (so the polynomial is fitted through `n_active` past history
            // points — matching what BDF order n_active actually uses).
            // For n_active == 1 the polynomial is degree 0 (constant
            // x_{n-1}) which `init_first_step` already produces.
            let poly_q = g.n_active.saturating_sub(1);
            if poly_q == 0 {
                g.nordsieck.init_first_step(&solver.x, dt);
                // init_first_step sets q = 1 by convention (degree-1 poly
                // with zero slope).  Override to degree 0 — z[1] is zero
                // anyway, so the residual is g = x_{n-1} + h·f (BDF1).
                g.nordsieck.q = 0;
                g.nordsieck.h = dt;
            } else {
                g.nordsieck.init_from_history(
                    &solver.history, &solver.history_dt, poly_q,
                );
            }

            // Pascal-predict: advances z by one step to t_n.  z_pred[0] is
            // now the polynomial extrapolation of x to t_n (the Newton
            // initial guess); z_pred[1] = h·y'_pred is the derivative used
            // by the corrector residual.
            g.nordsieck.predict();

            // Snapshot z_pred for solve_fn / step_fn consumers (avoids
            // re-borrowing Nordsieck during Newton iterations).
            // Disjoint borrow: take `nordsieck` and `z_pred_*` from `g` via
            // split fields — Rust can't see through `RefMut` so we work
            // with raw mutable references.
            {
                let GearState { nordsieck, z_pred_0, z_pred_1, .. } = &mut *g;
                z_pred_0.copy_from_slice(&nordsieck.z[0]);
                z_pred_1.copy_from_slice(&nordsieck.z[1]);
            }

            // NDF (Numerical Differentiation Formula) coefficients at the
            // active order.  Modifies plain BDF by `β_NDF = 1/(α_0 − κ_q·α_q)`
            // where κ_q comes from the Shampine-Reichelt 1997 NDF table.
            // For order 5 (κ_5 = 0) this collapses to plain BDF; orders 1-4
            // get a small κ-correction that shrinks the LTE estimate's
            // prefactor and slightly enlarges the stability region.
            let alphas = compute_bdf_alphas(g.n_active, &solver.history_dt);
            let kappa_q = NDF_KAPPA[g.n_active.min(5)];
            let beta_ndf = 1.0 / (alphas[0] - kappa_q * alphas[g.n_active]);
            g.l = compute_bdf_l_vector(g.n_active, &solver.history_dt, beta_ndf);

            // Lower-order estimate (`tr_m`).  Used for accept/reject and
            // step-size rescale.  Computed from the BDF coefficient table
            // at order n_active − 1 (history-residual, unchanged).
            g.have_down = g.n_active > 1;
            if g.have_down {
                let (bd, kd) = compute_bdf_coefficients(g.n_active - 1, &solver.history_dt);
                g.beta_down = bd;
                g.k_down = kd;
            }

            // Higher-order estimate (`tr_p`).  Needs at least `n_active + 1`
            // buffered states; otherwise we keep growing the order via the
            // ramp-up branch in `step_fn`.
            g.have_up = g.n_active < g.n_max && h_len > g.n_active;
            if g.have_up {
                let (bu, ku) = compute_bdf_coefficients(g.n_active + 1, &solver.history_dt);
                g.beta_up = bu;
                g.k_up = ku;
            }

            // Set Newton initial guess from the predictor.
            solver.x.copy_from_slice(&g.nordsieck.z[0]);
        }));

        // -- solve_fn --
        // One Newton/Anderson step on the Nordsieck-form BDF residual:
        //
        //     g(x) = z_pred[0] − β · z_pred[1] + β · h · f(x)
        //
        // where `β = l[0]`.  Provably equivalent to the history-residual
        // form `g(x) = β·h·f(x) + Σ K_j · x_{n-1-j}`; the Pascal predictor
        // bakes the past x sum into `(z_pred[0] − β·z_pred[1])`.
        // Newton matrix scaling: `β·h`.  The outer Simulation::_solve loop
        // re-enters this until the WRMS-scaled residual drops below NLS_COEF.
        let st_solve = state.clone();
        let mut g_buf: Vec<f64> = Vec::new();
        let mut jac_buf: Vec<f64> = Vec::new();
        s.solve_fn = Some(Box::new(move |solver: &mut Solver, f: &[f64], jac: Option<Jacobian>, dt: f64| {
            let n = f.len();
            if n != solver.x.len() {
                solver.x.resize(n, 0.0);
                solver.initial_value.resize(n, 0.0);
                for h in solver.history.iter_mut() { h.resize(n, 0.0); }
            }
            if solver.history.is_empty() { return 0.0; }

            // Capture f(y_0) on the first Newton iteration of every step.
            // For order-1 (first step), the predictor is identity so
            // solver.x = y_0 here and `f` is exactly `f(y_0)` — used by
            // step_fn to estimate BDF1 LTE.  Harmless capture on later
            // steps (different `f`) since step_fn only consults this when
            // n_active == 1, which only occurs at h_len == 1.
            {
                let mut g_mut = st_solve.borrow_mut();
                if !g_mut.f_y0_captured {
                    g_mut.f_y0_buf.resize(n, 0.0);
                    g_mut.f_y0_buf.copy_from_slice(f);
                    g_mut.f_y0_captured = true;
                }
            }

            let g = st_solve.borrow();
            let beta = g.l[0];
            let beta_dt = beta * dt;

            g_buf.resize(n, 0.0);
            for i in 0..n {
                g_buf[i] = g.z_pred_0[i] - beta * g.z_pred_1[i] + beta_dt * f[i];
            }

            let wrms = solver.wrms_norm_diff(&g_buf, &solver.x);

            let opt = solver.opt.as_mut().expect("GEAR52A needs an optimizer");
            let _l2 = match jac {
                Some(Jacobian::Scalar(j)) => {
                    opt.step(&mut solver.x, &g_buf, Some(beta_dt * j))
                }
                Some(Jacobian::Matrix(ref jflat, nr)) => {
                    let nn = nr * nr;
                    jac_buf.resize(nn, 0.0);
                    for i in 0..nn { jac_buf[i] = jflat[i] * beta_dt; }
                    opt.step_matrix(&mut solver.x, &g_buf, Some(&jac_buf), nr)
                }
                Some(Jacobian::Sparse(ref sj)) => {
                    // Dense fallback (SAJ-1).
                    let nr = sj.n;
                    jac_buf.clear();
                    jac_buf.resize(nr * nr, 0.0);
                    for k in 0..sj.values.len() {
                        jac_buf[sj.rows[k] as usize * nr + sj.cols[k] as usize] = sj.values[k] * beta_dt;
                    }
                    opt.step_matrix(&mut solver.x, &g_buf, Some(&jac_buf), nr)
                }
                None => opt.step(&mut solver.x, &g_buf, None),
            };
            wrms
        }));

        // -- step_fn --
        // After Newton convergence, build truncation residuals at orders
        // (n_active − 1) and (n_active + 1) and feed them through the PI
        // step-size controller.  Order is then nudged up or down depending
        // on which estimate is smaller.
        let st_step = state.clone();
        let mut tr_buf: Vec<f64> = Vec::new();
        s.step_fn = Some(Box::new(move |solver: &mut Solver, f: &[f64], dt: f64| {
            let n = f.len();
            let mut g = st_step.borrow_mut();

            // First-step (n_active == 1) error control via BDF1 vs forward
            // Euler comparison.  Both are first-order; their LTE differs
            // by a sign on `h²·y''/2`, so
            //
            //     LTE_BDF1 ≈ (x_n^BDF1 − x_n^FE) / 2
            //              = (x_n − y_0 − h · f(y_0)) / 2.
            //
            // `g.f_y0_buf` was captured by `solve_fn` on its first
            // iteration (where solver.x = y_0 because the order-1
            // predictor is identity).  Feeds into the same PI controller
            // as the rest of the order-adapt machinery — replaces the old
            // "blindly accept the first step" behaviour that left a hard
            // O(h_0²) accuracy floor on conservative systems.
            if !g.have_down {
                if !g.f_y0_captured || solver.history.is_empty() {
                    // No reference available — fall back to blind accept.
                    if g.n_target < g.n_max { g.n_target += 1; }
                    return (true, 0.0, Some(1.0));
                }
                tr_buf.resize(n, 0.0);
                let y0 = &solver.history[0];
                for i in 0..n {
                    tr_buf[i] = (solver.x[i] - y0[i] - dt * g.f_y0_buf[i]) * 0.5;
                }
                let (success, err_m, scale) = solver.error_controller_with_tr(&tr_buf, dt);
                if g.n_target < g.n_max {
                    g.n_target += 1;
                    // Order changed → reset PI history.  err_prev was
                    // computed against a different order's accuracy, mixing
                    // it into the next step's P-term would mis-direct the
                    // controller.
                    solver.err_prev = 0.0;
                }
                return (success, err_m, scale);
            }

            // tr_m: residual of order n_active − 1.
            tr_buf.resize(n, 0.0);
            let beta_dt_d = g.beta_down * dt;
            for i in 0..n {
                let mut s = beta_dt_d * f[i];
                for (j, &alpha) in g.k_down.iter().enumerate() {
                    if j < solver.history.len() {
                        s += alpha * solver.history[j][i];
                    }
                }
                tr_buf[i] = solver.x[i] - s;
            }

            // err_m: max-norm of the order-(n_active − 1) residual.
            // Conceptually estimates LTE_{n_active − 1}.  Used in #2 for
            // h_new comparison at the lower-order candidate.
            let err_m = solver.scaled_max_norm(&tr_buf[..n]);

            // err_q: direct LTE_{n_active} estimate via the corrector
            // discrepancy ξ = h·f(x_n) − z_pred[1].  ode15s/Shampine-
            // Reichelt formula:  LTE_q ≈ (κ_q + 1/(q+1)) · ξ.  For BDF5
            // (κ_5 = 0) this reduces to ξ/(q+1).  Lower orders get
            // a smaller prefactor due to negative κ values, shrinking
            // err_q and effectively letting the controller choose larger
            // steps in low-order regimes (ramp-up, problems where the
            // optimal order is < 5).
            let q = g.n_active as f64;
            let kappa_q = NDF_KAPPA[g.n_active.min(5)];
            let lte_prefactor = kappa_q + 1.0 / (q + 1.0);
            let mut max_q_norm: f64 = TOLERANCE;
            for i in 0..n {
                let xi_i = dt * f[i] - g.z_pred_1[i];
                let scale_i = solver.tolerance_lte_abs
                    + solver.tolerance_lte_rel * solver.x[i].abs();
                let se = (lte_prefactor * xi_i).abs() / scale_i;
                if se > max_q_norm { max_q_norm = se; }
            }
            let err_q = max_q_norm.max(TOLERANCE);

            // Accept/reject + rescale driven by err_q (current-order LTE).
            // Order argument to step_factor is `q+1` so the formula's
            // exponent matches LTE ∝ h^{q+1}.
            let success = err_q <= 1.0;
            let scale = Some(solver.step_factor(err_q, g.n_active + 1, success));

            if g.have_up {
                // err_p: standalone scaled max-norm of the order-(n_active+1)
                // residual.  Used only for the order decision; not fed back
                // into the rescale (Hairer-Wanner / ode15s convention).
                let beta_dt_u = g.beta_up * dt;
                let mut max_p: f64 = TOLERANCE;
                for i in 0..n {
                    let mut s = beta_dt_u * f[i];
                    for (j, &alpha) in g.k_up.iter().enumerate() {
                        if j < solver.history.len() {
                            s += alpha * solver.history[j][i];
                        }
                    }
                    let res = solver.x[i] - s;
                    let scale_i = solver.tolerance_lte_abs
                        + solver.tolerance_lte_rel * solver.x[i].abs();
                    let se = res.abs() / scale_i;
                    if se > max_p { max_p = se; }
                }
                let err_p = max_p.max(TOLERANCE);

                // ode15s-style order decision: for each candidate order q'
                // ∈ {q-1, q, q+1}, compute the implied next-step h_new at
                // that order and pick the q* maximizing it.  Honours the
                // exponent `1/(q'+1)` for each order's LTE scaling, so a
                // lower err at a higher order doesn't automatically win
                // when the exponent shifts.
                let beta_safe = solver.beta;
                let h_new_m = beta_safe * dt * err_m.powf(-1.0 / q);
                let h_new_q = beta_safe * dt * err_q.powf(-1.0 / (q + 1.0));
                let h_new_p = beta_safe * dt * err_p.powf(-1.0 / (q + 2.0));

                // Pick the order with the largest h_new.  Throttle still
                // applies — order can only change after q+1 consecutive
                // accepts, regardless of which order maximizes h_new.
                let can_change = g.steps_at_order > g.n_active;
                if can_change {
                    if h_new_m > h_new_q && h_new_m > h_new_p {
                        if g.n_target > g.n_min {
                            g.n_target -= 1;
                            g.steps_at_order = 0;
                            solver.err_prev = 0.0;
                        }
                    } else if h_new_p > h_new_q && h_new_p > h_new_m
                        && g.n_target < g.n_max {
                            g.n_target += 1;
                            g.steps_at_order = 0;
                            solver.err_prev = 0.0;
                        }
                    // Otherwise current order wins — leave n_target alone.
                }
            } else if success && g.n_target < g.n_max {
                // No higher-order estimate yet — keep climbing the ramp-up.
                // Ramp-up is exempt from throttling; order grows monotonically
                // until the variable-order machinery kicks in.
                g.n_target += 1;
                g.steps_at_order = 0;
                solver.err_prev = 0.0;
            }
            if success {
                g.steps_at_order += 1;
            }
            // NOTE: at n_active == n_max == 5, no order-down logic activates
            // because we have no err_p estimate (BDF6 isn't zero-stable).
            // Order is simply held at n_max while the PI controller adapts
            // dt.  Adding a "drop if err_m << 1" heuristic causes oscillation
            // (drop → err_p < err_m → grow → drop → …) so we leave it alone.

            (success, err_m, scale)
        }));

        s
    })
}

// ======================================================================================
// Factory lookup
// ======================================================================================

/// Helper: create a factory from a solver name string.
pub fn factory_from_name(name: &str, tol_abs: f64, tol_rel: f64) -> Option<SolverFactory> {
    // Tableau-backed solvers dispatch via the registry.
    if let Some(t) = tbl::by_name(name) {
        return Some(build_from_tableau(t, tol_abs, tol_rel));
    }
    // Non-tableau specials.
    match name {
        "EUF" => Some(euf_factory()),
        "EUB" => Some(eub_factory()),
        "SteadyState" => Some(steadystate_factory()),
        "GEAR52A" => Some(gear52a_factory(tol_abs, tol_rel)),
        _ => None,
    }
}

/// Solver names that exist in pathsim but are not yet implemented in fastsim.
/// Listed so the lookup error can diagnose the gap precisely instead of just
/// reporting a generic "unknown solver".
const PATHSIM_UNIMPLEMENTED: &[&str] = &[
    "BDF", "BDF2", "BDF3", "BDF4", "BDF5", "BDF6", "GEAR", "GEAR21", "GEAR32", "GEAR43", "GEAR54",
    "ESDIRK85",
];

/// All solver names fastsim can construct (tableau-backed plus the specials).
pub fn available_solver_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = tbl::all_names().collect();
    names.extend_from_slice(&["EUF", "EUB", "SteadyState", "GEAR52A"]);
    names
}

/// Build a helpful diagnostic for a solver name that `factory_from_name`
/// could not resolve.
pub fn unknown_solver_message(name: &str) -> String {
    let available = available_solver_names().join(", ");
    if PATHSIM_UNIMPLEMENTED.contains(&name) {
        format!(
            "solver '{name}' exists in pathsim but is not yet implemented in fastsim. \
             Available solvers: {available}"
        )
    } else {
        format!("unknown solver: '{name}'. Available solvers: {available}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssprk22_integrates() {
        let factory = ssprk22_factory();
        let mut solver = factory(&[0.0]);
        let dt = 0.01;
        for _ in 0..100 {
            solver.buffer(dt);
            for si in 0..solver.n_stages() {
                solver._stage = si;
                solver.step(&[1.0], dt);
            }
        }
        assert!((solver.x[0] - 1.0).abs() < 1e-10, "Got {}", solver.x[0]);
    }

    #[test]
    fn test_rk4_exponential() {
        let factory = rk4_factory();
        let mut solver = factory(&[1.0]);
        let dt = 0.01;
        for _ in 0..100 {
            solver.buffer(dt);
            for si in 0..solver.n_stages() {
                solver._stage = si;
                let f = vec![solver.x[0]];
                solver.step(&f, dt);
            }
        }
        let error = (solver.x[0] - std::f64::consts::E).abs();
        assert!(error < 1e-8, "RK4 error: {}", error);
    }

    #[test]
    fn test_rkdp54_adaptive() {
        let factory = rkdp54_factory(1e-8, 0.0);
        let solver = factory(&[1.0]);
        assert!(solver.is_adaptive);
        assert_eq!(solver.s, 7);
    }

    #[test]
    fn factory_from_name_covers_all_tableaus() {
        for t in tbl::ALL {
            assert!(factory_from_name(t.name, 1e-8, 0.0).is_some(), "missing: {}", t.name);
        }
        assert!(factory_from_name("EUF", 0.0, 0.0).is_some());
        assert!(factory_from_name("EUB", 0.0, 0.0).is_some());
        assert!(factory_from_name("SteadyState", 0.0, 0.0).is_some());
        assert!(factory_from_name("GEAR52A", 1e-8, 1e-6).is_some());
        assert!(factory_from_name("Nonsense", 0.0, 0.0).is_none());
    }

    #[test]
    fn gear52a_metadata() {
        let factory = gear52a_factory(1e-8, 1e-6);
        let s = factory(&[1.0]);
        assert_eq!(s.type_name, "GEAR52A");
        assert!(s.is_implicit);
        assert!(s.is_adaptive);
        assert_eq!(s.history_maxlen, 6);
        assert_eq!(s.history_dt_maxlen, 6);
        assert!(s.use_pi_controller);
        assert!(s.opt.is_some());
        assert_eq!(s.eval_stages, vec![1.0]);
    }

    /// Drive `Solver::buffer` / `solve_fn` / `step_fn` directly (bypassing
    /// `ImplicitSolver::integrate` which is RK-tableau-specific).  Mirrors how
    /// `Simulation::_solve` calls these closures during a real block step:
    /// repeated `solve` calls per step until the WRMS-scaled residual drops
    /// under `NLS_COEF`, then `step` for the error/order controller.
    fn drive_gear<F, J>(
        solver: &mut Solver,
        rhs: F,
        jac: J,
        t_end: f64,
        mut dt: f64,
        max_iter: usize,
    ) -> (f64, Vec<f64>)
    where
        F: Fn(&[f64], f64) -> Vec<f64>,
        J: Fn(&[f64], f64) -> Jacobian,
    {
        let mut t = 0.0;
        let mut accept_count = 0;
        let mut total_count = 0;
        while t < t_end {
            let dt_step = dt.min(t_end - t);
            solver.buffer(dt_step);

            // Newton iteration on the BDF residual.
            for _ in 0..max_iter {
                let f = rhs(&solver.x, t + dt_step);
                let j = jac(&solver.x, t + dt_step);
                let residual = solver.solve(&f, Some(j), dt_step);
                if residual < NLS_COEF { break; }
            }
            let f = rhs(&solver.x, t + dt_step);
            let (success, _err, scale) = solver.step(&f, dt_step);
            total_count += 1;

            if success {
                t += dt_step;
                accept_count += 1;
            } else {
                solver.revert();
            }
            if let Some(s) = scale { dt = (dt * s).clamp(1e-12, 1.0); }
            if total_count > 10_000 { panic!("drive_gear runaway: t={}", t); }
        }
        let _ = accept_count;
        (t, solver.x.clone())
    }

    #[test]
    fn gear52a_integrates_exponential() {
        // ẋ = x is a sanity check for the full control loop.  GTE on a
        // rapidly-growing solution accumulates: BDF tracks LTE, not GTE, so
        // we only assert that GEAR52A is in the correct ballpark — actual
        // accuracy verification happens against PathSim/SciPy in the Python
        // trajectory tests.
        let factory = gear52a_factory(1e-10, 1e-8);
        let mut solver = factory(&[1.0]);
        let (_t, x) = drive_gear(
            &mut solver,
            |x, _t| vec![x[0]],
            |_x, _t| Jacobian::Scalar(1.0),
            1.0, 0.01, 50,
        );
        let err = (x[0] - std::f64::consts::E).abs();
        assert!(err < 1e-2, "GEAR52A ẋ=x: x(1) = {}, err = {:.2e}", x[0], err);
        // Ensure it actually advanced the state — early bug had the solver
        // stuck at x(0) = 1.
        assert!(x[0] > 2.7 && x[0] < 2.74, "value out of range: {}", x[0]);
    }

    /// Robertson stiff test — GEAR52A must integrate without divergence and
    /// preserve mass near 1.  Tight tolerance + fairly long horizon to make
    /// sure the order-ramp-up + variable-order machinery actually engages.
    #[test]
    fn gear52a_robertson_stays_bounded() {
        let factory = gear52a_factory(1e-8, 1e-6);
        let mut solver = factory(&[1.0, 0.0, 0.0]);
        let (_t, x) = drive_gear(
            &mut solver,
            |x, _t| {
                let (a, b, c) = (0.04, 1e4, 3e7);
                vec![
                    -a * x[0] + b * x[1] * x[2],
                     a * x[0] - b * x[1] * x[2] - c * x[1] * x[1],
                     c * x[1] * x[1],
                ]
            },
            |x, _t| {
                let (a, b, c) = (0.04, 1e4, 3e7);
                Jacobian::Matrix(vec![
                    -a,                   b * x[2],            b * x[1],
                     a,          -b * x[2] - 2.0*c*x[1],      -b * x[1],
                     0.0,                 2.0 * c * x[1],       0.0,
                ], 3)
            },
            10.0, 1e-4, 100,
        );
        let mass = x[0] + x[1] + x[2];
        assert!((mass - 1.0).abs() < 1e-3, "Robertson mass: {} vs 1.0", mass);
        for &c in &x { assert!((-1e-6..=1.0+1e-6).contains(&c), "out-of-range: {}", c); }
    }
}
