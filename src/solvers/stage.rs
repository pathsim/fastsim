// Stage-builder abstraction for implicit RK stages.
//
// Rationale: the Newton problem solved at each DIRK/ESDIRK stage takes the
// form
//
//     A · Δ = -r(x_i)
//
// where residual `r` and Newton matrix `A` depend on the *problem form*:
//
//     ODE:        r = x - x_0 - dt·Σa·k_j,        A = I - dt·a_ii·J
//     MassMatrix: r = M(x - x_0) - dt·Σa·(M·k_j), A = M - dt·a_ii·J
//     FullyImpl:  r = F(X, K, t),                  A = ∂F/∂ẋ + dt·a_ii·∂F/∂x
//
// (SemiExplicit DAE blocks reduce to plain ODE form via inner `z`-elimination
// inside the block constructor, so no dedicated builder is needed.)
//
// Extracting these into builder structs gives `make_dirk_solve` a single
// integration point: the ESDIRK step loop calls `solver.stage_builder.solve_stage`
// without knowing the problem form.  Each builder owns its own scratch buffers
// (residual, Newton matrix, etc.) so the hot path stays allocation-free.
//
// Convergence: every builder returns a *WRMS-scaled* residual norm
// (`Solver::wrms_norm`/`wrms_norm_diff`), not an absolute L2 norm.  The
// downstream `Simulation::_solve` checks this against the unitless ARKODE-
// style coefficient `NLS_COEF`.  This couples inner-Newton accuracy to the
// outer step's `(atol + rtol·|x|)` weights so multi-scale stiff problems
// don't get over-converged on the well-scaled state components.

use std::cell::RefCell;
use std::rc::Rc;

use crate::solvers::solver::{Solver, Jacobian};

use crate::constants::MASS_ZERO_THRESHOLD;

/// Implements the Newton-solve step for a single RK stage of a specific
/// problem form.  Wraps any per-builder scratch buffers so solver state
/// stays form-agnostic.
pub trait StageBuilder {
    /// Run the Newton update for the current stage.  Returns the **WRMS-
    /// scaled residual norm** (computed via `Solver::wrms_norm` /
    /// `wrms_norm_diff`) — *not* the optimiser's internal L2 norm.  The
    /// caller (`Simulation::_solve`) checks this against the unitless
    /// `NLS_COEF` to decide convergence.
    ///
    /// Preconditions set by `make_dirk_solve`:
    /// - `solver._stage` indexes the current stage.
    /// - `solver.history` contains the buffered `x_0`.
    /// - `solver.ks[solver._stage]` has been populated with the current slope `f`.
    /// - `solver.bt[solver._stage]` is `Some(...)` (i.e. not an ESDIRK explicit stage).
    fn solve_stage(
        &mut self,
        solver: &mut Solver,
        f: &[f64],
        jac: Option<Jacobian>,
        dt: f64,
    ) -> f64;

    /// Number of algebraic state components.  Non-zero only for DAE builders.
    /// The adaptive error controller will skip these components once
    /// implemented.
    fn n_alg(&self) -> usize { 0 }
}

// ======================================================================================
// Mass matrix — constant coefficient, row-major n×n
// ======================================================================================

/// Constant mass matrix `M` for the DAE form `M·ẋ = f(x, u, t)`.
///
/// Stored row-major in a flat `Vec<f64>`.  `n_alg` counts trailing rows that
/// are entirely zero — those rows denote algebraic constraints `0 = f_i`.
/// `is_singular` is currently just `n_alg > 0`; richer rank detection (e.g.
/// LU with pivot threshold) can be added later without changing the API.
#[derive(Clone, Debug)]
pub struct Mass {
    pub data: Vec<f64>,
    pub n: usize,
    pub is_singular: bool,
    pub n_alg: usize,
}

impl Mass {
    /// Build from a row-major flat buffer of length `n*n`.
    pub fn from_flat(data: Vec<f64>, n: usize) -> Self {
        assert_eq!(data.len(), n * n, "mass matrix size mismatch");
        // Count contiguous trailing all-zero rows as algebraic components.
        let mut n_alg = 0;
        for i in (0..n).rev() {
            let row_is_zero = (0..n)
                .all(|j| data[i * n + j].abs() < MASS_ZERO_THRESHOLD);
            if row_is_zero { n_alg += 1; } else { break; }
        }
        Self { data, n, is_singular: n_alg > 0, n_alg }
    }

    /// Identity mass matrix of size n×n (used mainly for tests / default).
    pub fn identity(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n { data[i * n + i] = 1.0; }
        Self { data, n, is_singular: false, n_alg: 0 }
    }

    /// `out = M · x`.  Caller owns the output slice's length == n.
    #[inline]
    pub fn matvec(&self, x: &[f64], out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.n);
        debug_assert_eq!(x.len(), self.n);
        for i in 0..self.n {
            let mut s = 0.0;
            for j in 0..self.n { s += self.data[i * self.n + j] * x[j]; }
            out[i] = s;
        }
    }
}

// ======================================================================================
// OdeStageBuilder — pure ODE (no mass matrix, no algebraic variables)
// ======================================================================================

/// Stage builder for plain ODEs: `dx/dt = f(x, u, t)`.
///
/// Residual:  g = x_0 + dt · Σ a_ij · k_j  (implicit fixed-point form x = g)
/// Newton matrix scaling: `dt · a_ii` on the supplied Jacobian.
pub struct OdeStageBuilder {
    g_buf: Vec<f64>,
    jac_buf: Vec<f64>,
}

impl OdeStageBuilder {
    pub fn new() -> Self {
        Self { g_buf: Vec::new(), jac_buf: Vec::new() }
    }
}

impl Default for OdeStageBuilder {
    fn default() -> Self { Self::new() }
}

impl StageBuilder for OdeStageBuilder {
    fn solve_stage(
        &mut self,
        solver: &mut Solver,
        f: &[f64],
        j: Option<Jacobian>,
        dt: f64,
    ) -> f64 {
        let stage = solver._stage;
        let n = f.len();
        debug_assert!(!solver.history.is_empty(),
            "OdeStageBuilder::solve_stage called without buffered x_0");

        let bt_stage = match &solver.bt[stage] {
            Some(coeffs) => coeffs,
            None => return 0.0,  // caller should have filtered explicit stages
        };

        // g = x_0 + dt · Σ a_ij · k_j   (fixed-point RHS for x_i = g)
        self.g_buf.resize(n, 0.0);
        let x_0 = &solver.history[0];
        for j_idx in 0..n {
            let mut s = 0.0;
            for (i, &a) in bt_stage.iter().enumerate() {
                if i < solver.ks.len() && !solver.ks[i].is_empty() {
                    s += solver.ks[i][j_idx] * a;
                }
            }
            self.g_buf[j_idx] = x_0[j_idx] + dt * s;
        }

        // ARKODE-style WRMS residual norm `||(g − x) / (atol + rtol·|x|)||_RMS`
        // on the *current* iterate, before the Newton step.  Each component is
        // weighted individually so multi-scale problems (e.g. Robertson with
        // x[0]≈0.7, x[1]≈1e-5) are not over-converged on the well-scaled
        // components.  Downstream `Simulation::_solve` checks this against the
        // unitless `NLS_COEF`, not the absolute `tolerance_fpi`.
        let wrms_norm = solver.wrms_norm_diff(&self.g_buf, &solver.x);

        let a_ii = bt_stage[stage];
        let opt = solver.opt.as_mut().expect("Implicit solver needs optimizer");
        // Newton step: opt.step* returns its own unscaled L2 norm; we discard
        // it here in favour of `wrms_norm` for the convergence test.
        let _l2 = match j {
            Some(Jacobian::Scalar(jac)) => {
                opt.step(&mut solver.x, &self.g_buf, Some(dt * a_ii * jac))
            }
            Some(Jacobian::Matrix(ref jac_flat, n_rows)) => {
                let scale = dt * a_ii;
                let nn = n_rows * n_rows;
                self.jac_buf.resize(nn, 0.0);
                for i in 0..nn { self.jac_buf[i] = jac_flat[i] * scale; }
                opt.step_matrix(&mut solver.x, &self.g_buf, Some(&self.jac_buf), n_rows)
            }
            // Sparse Newton step: assemble A = scale·J − I straight from the
            // coordinate pattern, no dense materialisation or nonzero scan.
            Some(Jacobian::Sparse(ref sj)) => {
                opt.step_matrix_sparse(
                    &mut solver.x, &self.g_buf,
                    &sj.rows[..], &sj.cols[..], &sj.values, dt * a_ii, sj.n,
                )
            }
            None => {
                opt.step(&mut solver.x, &self.g_buf, None)
            }
        };
        wrms_norm
    }
}

// ======================================================================================
// MassMatrixStageBuilder — constant mass matrix DAE `M·ẋ = f(x, u, t)`
// ======================================================================================

/// Stage builder for DAEs with constant mass matrix `M`.
///
/// Residual:   r = M·(x - x_0) - dt·Σ a_ij·K_j   where K_j is the stored slope.
/// Newton system: (M - dt·a_ii·J) · Δ = r     →   x -= Δ
///
/// For the explicit first stage of an ESDIRK this builder is not called
/// (filtered upstream in `make_dirk_solve`).  Singular `M` (`mass.is_singular`)
/// is valid and exercises the DAE path; a regular `M` degenerates to the same
/// Newton system as the plain ODE after rescaling.
pub struct MassMatrixStageBuilder {
    mass: Mass,
    res_buf: Vec<f64>,
    mat_buf: Vec<f64>,
    dx_buf: Vec<f64>,
    bt_buf: Vec<f64>,
}

impl MassMatrixStageBuilder {
    pub fn new(mass: Mass) -> Self {
        Self {
            mass,
            res_buf: Vec::new(),
            mat_buf: Vec::new(),
            dx_buf: Vec::new(),
            bt_buf: Vec::new(),
        }
    }
}

impl StageBuilder for MassMatrixStageBuilder {
    fn solve_stage(
        &mut self,
        solver: &mut Solver,
        _f: &[f64],
        j: Option<Jacobian>,
        dt: f64,
    ) -> f64 {
        let stage = solver._stage;
        let n = self.mass.n;
        debug_assert_eq!(solver.x.len(), n);

        // Copy the (small) Butcher row into scratch instead of cloning it anew
        // each Newton iteration.
        match &solver.bt[stage] {
            Some(coeffs) => { self.bt_buf.clear(); self.bt_buf.extend_from_slice(coeffs); }
            None => return 0.0,
        }
        let a_ii = self.bt_buf[stage];

        // r = M·(x - x_0) - dt · Σ a_k · ks[k]
        self.res_buf.resize(n, 0.0);
        self.dx_buf.resize(n, 0.0);
        let x_0 = &solver.history[0];
        for i in 0..n { self.dx_buf[i] = solver.x[i] - x_0[i]; }
        self.mass.matvec(&self.dx_buf, &mut self.res_buf);
        for k_idx in 0..self.bt_buf.len() {
            let a_k = self.bt_buf[k_idx];
            if k_idx < solver.ks.len() && !solver.ks[k_idx].is_empty() {
                let ks_k = &solver.ks[k_idx];
                for i in 0..n {
                    self.res_buf[i] -= dt * a_k * ks_k[i];
                }
            }
        }

        // Newton matrix A = M - dt·a_ii·J
        let scale = dt * a_ii;
        self.mat_buf.resize(n * n, 0.0);
        self.mat_buf.copy_from_slice(&self.mass.data);
        match j {
            Some(Jacobian::Matrix(ref jac_flat, nr)) => {
                debug_assert_eq!(nr, n);
                for i in 0..n * n { self.mat_buf[i] -= scale * jac_flat[i]; }
            }
            Some(Jacobian::Sparse(ref sj)) => {
                // Subtract only the structural nonzeros from M (dense fallback
                // for SAJ-1; the sparse path lands in SAJ-3/4).
                debug_assert_eq!(sj.n, n);
                for k in 0..sj.values.len() {
                    let idx = sj.rows[k] as usize * n + sj.cols[k] as usize;
                    self.mat_buf[idx] -= scale * sj.values[k];
                }
            }
            Some(Jacobian::Scalar(jac)) => {
                // Scalar Jacobian is interpreted as jac·I.
                for i in 0..n { self.mat_buf[i * n + i] -= scale * jac; }
            }
            None => {
                // No Jacobian → fall back to A = M (functional iteration).
            }
        }

        // WRMS norm on the implicit-equation residual `r = M·(x − x_0) − dt·Σa·k`,
        // before Newton step.  See `OdeStageBuilder` for rationale.
        let wrms_norm = solver.wrms_norm(&self.res_buf);

        // Unified optimiser path: Newton step (+ Anderson acceleration if
        // the solver was configured with Anderson/NewtonAnderson).
        let opt = solver.opt.as_mut().expect("Mass-matrix DAE needs optimizer");
        let _l2 = crate::optim::anderson::optimizer_step_residual(
            opt, &mut solver.x, &self.res_buf, &self.mat_buf, n,
        );
        wrms_norm
    }

    fn n_alg(&self) -> usize { self.mass.n_alg }
}

// ======================================================================================
// FullyImplicitStageBuilder — `F(x, ẋ, u, t) = 0`
// ======================================================================================

/// Callback signature for `F(x, xdot, u, t) -> residual`.
pub type FiFn = Rc<dyn Fn(&[f64], &[f64], &[f64], f64) -> Vec<f64>>;

/// Shared scratch populated by the block's `f_dyn`, read by the builder.
/// Lets the builder access block inputs and stage time without widening the
/// `StageBuilder::solve_stage` signature.
#[derive(Clone, Default)]
pub struct FiContext {
    pub t: Rc<RefCell<f64>>,
    pub u: Rc<RefCell<Vec<f64>>>,
}

impl FiContext {
    pub fn new() -> Self {
        Self { t: Rc::new(RefCell::new(0.0)), u: Rc::new(RefCell::new(Vec::new())) }
    }
}

/// Stage builder for fully-implicit DAEs `F(x, ẋ, u, t) = 0`.
///
/// Stage variable is the Runge-Kutta derivative `K_i = ẋ_i`.  Given
/// `solver.x = X_i` from the enclosing Newton/FPI loop, the builder recovers
/// `K_i = (X_i - x_0 - dt·Σ_{j<i} a_ij·K_j) / (dt·a_ii)`, evaluates the
/// residual `R = F(X_i, K_i, u, t)`, assembles the Newton matrix
/// `A = dt·a_ii·∂F/∂x + ∂F/∂ẋ`, and applies `X_i -= A⁻¹·R`.
/// `K_i` is stored in `solver.ks[stage]` so the standard DIRK step finalises
/// `x_new = x_0 + dt·Σ b_j · K_j` as usual.
pub struct FullyImplicitStageBuilder {
    f: FiFn,
    jac_x: Option<FiFn>,
    jac_xdot: Option<FiFn>,
    ctx: FiContext,
    k_buf: Vec<f64>,
    mat_buf: Vec<f64>,
    jx_buf: Vec<f64>,
    jxd_buf: Vec<f64>,
    // Per-call scratch, reused across Newton iterations like `MassMatrixStageBuilder`'s
    // buffers (the previous code reallocated/cloned these every iteration).
    sum_k_buf: Vec<f64>,
    xi_buf: Vec<f64>,
    x0_buf: Vec<f64>,
    u_buf: Vec<f64>,
    bt_buf: Vec<f64>,
}

impl FullyImplicitStageBuilder {
    pub fn new(f: FiFn, jac_x: Option<FiFn>, jac_xdot: Option<FiFn>, ctx: FiContext) -> Self {
        Self {
            f, jac_x, jac_xdot, ctx,
            k_buf: Vec::new(), mat_buf: Vec::new(),
            jx_buf: Vec::new(), jxd_buf: Vec::new(),
            sum_k_buf: Vec::new(), xi_buf: Vec::new(),
            x0_buf: Vec::new(), u_buf: Vec::new(), bt_buf: Vec::new(),
        }
    }

}

/// Newton-solve `F(x, xdot_guess, u, t) = 0` for `xdot_guess` (in-place),
/// holding `x`, `u`, `t` fixed.  Used to recover the explicit-stage slope
/// `K_0 = ẋ` for ESDIRK when the problem is fully implicit.
///
/// Uses at most `max_iter` Newton iterations with a numerical `∂F/∂ẋ`.
/// Detects singular `∂F/∂ẋ` (e.g. algebraic rows in Index-1 DAEs) and
/// bails out cleanly without propagating NaN — callers should treat the
/// returned `xdot_guess` as a best-effort warmstart in that case.
pub fn solve_xdot_for_f(
    f: &FiFn,
    x: &[f64],
    xdot_guess: &mut Vec<f64>,
    u: &[f64],
    t: f64,
    tol: f64,
    max_iter: usize,
    lin: &mut crate::optim::linsolve::LinearSolver,
) -> f64 {
    let n = x.len();
    if xdot_guess.len() != n { xdot_guess.resize(n, 0.0); }
    let mut jxd = Vec::with_capacity(n * n);
    let mut last_norm = f64::INFINITY;
    for _ in 0..max_iter {
        let r = (f)(x, xdot_guess, u, t);
        num_jac_wrt_xdot(f, x, xdot_guess, u, t, &mut jxd);

        // Cheap singularity check: if any diagonal dominance in ∂F/∂ẋ is
        // below threshold (all-zero rows, typical for algebraic rows),
        // don't try to invert — stop with the current guess.
        let singular = (0..n).any(|i| {
            (0..n).all(|j| jxd[i * n + j].abs() < crate::constants::NUM_JAC_TOL)
        });
        if singular {
            break;
        }

        let prev = xdot_guess.clone();
        last_norm = lin.newton_solve(xdot_guess, &r, &jxd, n);
        if !xdot_guess.iter().all(|v| v.is_finite()) {
            *xdot_guess = prev;
            break;
        }
        if last_norm < tol { break; }
    }
    last_norm
}

/// Central-difference numerical Jacobian of `eval` w.r.t. the `base` vector,
/// perturbing one element at a time. Output is flat row-major `n × n` (assumes a
/// square system: `eval` returns `n` residuals for an `n`-vector input).
fn num_jac_central(base: &[f64], out: &mut Vec<f64>, mut eval: impl FnMut(&[f64]) -> Vec<f64>) {
    let n = base.len();
    out.resize(n * n, 0.0);
    let mut p = base.to_vec();
    let h_base = crate::constants::STAGE_NUM_JAC_PERTURB.sqrt();
    for j in 0..n {
        let h = h_base * base[j].abs().max(1.0);
        p[j] = base[j] + h;
        let fp = eval(&p);
        p[j] = base[j] - h;
        let fm = eval(&p);
        p[j] = base[j];
        for i in 0..n {
            out[i * n + j] = (fp[i] - fm[i]) / (2.0 * h);
        }
    }
}

/// Central-difference numerical Jacobian `∂g/∂z` for the reduced
/// semi-explicit DAE: `g(x, z, u, t)` differentiated w.r.t. `z`.
pub fn num_jac_wrt_z(
    g: &FiFn, x: &[f64], z: &[f64], u: &[f64], t: f64, out: &mut Vec<f64>,
) {
    num_jac_central(z, out, |zp| (g)(x, zp, u, t));
}

/// Central-difference numerical Jacobian `∂F/∂x` (with `xdot`, `u`, `t` held fixed).
fn num_jac_wrt_x(f: &FiFn, x: &[f64], xdot: &[f64], u: &[f64], t: f64, out: &mut Vec<f64>) {
    num_jac_central(x, out, |xp| (f)(xp, xdot, u, t));
}

/// Central-difference numerical Jacobian `∂F/∂ẋ` (with `x`, `u`, `t` held fixed).
fn num_jac_wrt_xdot(f: &FiFn, x: &[f64], xdot: &[f64], u: &[f64], t: f64, out: &mut Vec<f64>) {
    num_jac_central(xdot, out, |xdp| (f)(x, xdp, u, t));
}

impl StageBuilder for FullyImplicitStageBuilder {
    fn solve_stage(
        &mut self,
        solver: &mut Solver,
        _f: &[f64],
        _j: Option<Jacobian>,
        dt: f64,
    ) -> f64 {
        let stage = solver._stage;
        let n = solver.x.len();
        // Copy the (small) Butcher row into scratch instead of cloning it anew
        // each Newton iteration.
        match &solver.bt[stage] {
            Some(c) => { self.bt_buf.clear(); self.bt_buf.extend_from_slice(c); }
            None => return 0.0,
        }
        let a_ii = self.bt_buf[stage];
        let dt_aii = dt * a_ii;
        // Base state x_0 = history[0] into scratch (read-only for this stage).
        self.x0_buf.clear();
        self.x0_buf.extend_from_slice(&solver.history[0]);

        // Sum over already-completed stages: Σ_{j<i} a_ij · K_j.
        self.sum_k_buf.clear();
        self.sum_k_buf.resize(n, 0.0);
        for j in 0..stage {
            if j < solver.ks.len() && !solver.ks[j].is_empty() {
                let a = self.bt_buf[j];
                for i in 0..n {
                    self.sum_k_buf[i] += a * solver.ks[j][i];
                }
            }
        }

        // Initialise stage derivative K_i.  Warmstart: reuse previous value
        // if present, otherwise derive from the current X_i guess.
        self.k_buf.resize(n, 0.0);
        let have_prev = stage < solver.ks.len() && !solver.ks[stage].is_empty();
        if have_prev {
            self.k_buf.copy_from_slice(&solver.ks[stage]);
        } else {
            for i in 0..n {
                self.k_buf[i] = (solver.x[i] - self.x0_buf[i] - dt * self.sum_k_buf[i]) / dt_aii;
            }
        }

        // X_i(K_i) = x_0 + dt·(Σ_{j<i} a·K_j + a_ii·K_i) — recompute from K_i.
        self.xi_buf.resize(n, 0.0);
        for i in 0..n {
            self.xi_buf[i] = self.x0_buf[i] + dt * self.sum_k_buf[i] + dt_aii * self.k_buf[i];
        }

        let t = *self.ctx.t.borrow();
        // Input vector into scratch (avoids a per-iteration clone).
        {
            let ub = self.ctx.u.borrow();
            self.u_buf.clear();
            self.u_buf.extend_from_slice(&ub);
        }

        // Residual R_K(K_i) = F(X_i(K_i), K_i, u, t).
        let r = (self.f)(&self.xi_buf, &self.k_buf, &self.u_buf, t);

        // Jacobians at (X_i, K_i).
        match &self.jac_x {
            Some(jfn) => self.jx_buf = (jfn)(&self.xi_buf, &self.k_buf, &self.u_buf, t),
            None      => num_jac_wrt_x(&self.f, &self.xi_buf, &self.k_buf, &self.u_buf, t, &mut self.jx_buf),
        }
        match &self.jac_xdot {
            Some(jfn) => self.jxd_buf = (jfn)(&self.xi_buf, &self.k_buf, &self.u_buf, t),
            None      => num_jac_wrt_xdot(&self.f, &self.xi_buf, &self.k_buf, &self.u_buf, t, &mut self.jxd_buf),
        }

        // Newton matrix w.r.t. K_i:  A = dt·a_ii·∂F/∂x + ∂F/∂ẋ.
        self.mat_buf.resize(n * n, 0.0);
        for i in 0..n * n {
            self.mat_buf[i] = dt_aii * self.jx_buf[i] + self.jxd_buf[i];
        }

        // WRMS norm on the implicit-equation residual `R = F(X_i, K_i, u, t)`,
        // before the Newton step.  See `OdeStageBuilder` for rationale.
        let wrms_norm = solver.wrms_norm(&r);

        // Unified optimiser step on K_i: Newton (+ Anderson if configured).
        // The iterate here is K_i, not solver.x — that's fine because the
        // Anderson history is keyed on the input buffer the caller passes.
        let opt = solver.opt.as_mut().expect("Fully-implicit DAE needs optimizer");
        let _l2 = crate::optim::anderson::optimizer_step_residual(
            opt, &mut self.k_buf, &r, &self.mat_buf, n,
        );

        // Propagate updated K_i → X_i and into solver state.
        while solver.ks.len() <= stage { solver.ks.push(Vec::new()); }
        solver.ks[stage].resize(n, 0.0);
        solver.ks[stage].copy_from_slice(&self.k_buf);
        for i in 0..n {
            solver.x[i] = self.x0_buf[i] + dt * self.sum_k_buf[i] + dt_aii * self.k_buf[i];
        }

        wrms_norm
    }
}

// `newton_solve_inplace` moved to `crate::optim::linsolve` (shared with the
// optimiser's matrix-Newton step).

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mass_identity_has_no_algebraic_rows() {
        let m = Mass::identity(3);
        assert_eq!(m.n_alg, 0);
        assert!(!m.is_singular);
    }

    #[test]
    fn mass_detects_trailing_zero_rows() {
        // [[1 0 0]
        //  [0 1 0]
        //  [0 0 0]]  -- last row algebraic
        let data = vec![1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 0.0];
        let m = Mass::from_flat(data, 3);
        assert_eq!(m.n_alg, 1);
        assert!(m.is_singular);
    }

    #[test]
    fn mass_matvec_identity() {
        let m = Mass::identity(3);
        let x = [1.0, 2.0, 3.0];
        let mut out = [0.0; 3];
        m.matvec(&x, &mut out);
        assert_eq!(out, x);
    }

    #[test]
    fn mass_matvec_diagonal() {
        let data = vec![2.0, 0.0, 0.0,  0.0, 3.0, 0.0,  0.0, 0.0, 4.0];
        let m = Mass::from_flat(data, 3);
        let x = [1.0, 1.0, 1.0];
        let mut out = [0.0; 3];
        m.matvec(&x, &mut out);
        assert_eq!(out, [2.0, 3.0, 4.0]);
    }
}
