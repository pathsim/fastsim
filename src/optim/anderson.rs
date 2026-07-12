// Anderson and NewtonAnderson fixed-point accelerators
// Ported from pathsim/optim/anderson.py

use std::collections::VecDeque;
use smallvec::SmallVec;

use crate::constants::{TOLERANCE, OPT_HISTORY};

/// Stack-allocated small vec for Anderson internal buffers.
type AVec = SmallVec<[f64; 8]>;

// Implementation notes:
//   - Vector path (n > 1): solve `min_c ||dR·c − F||` via faer's thin SVD
//     with truncated pseudo-inverse (threshold `max(shape)·ε_mach·σ_max`).
//     This matches pathsim's `np.linalg.lstsq(…, rcond=None)` — rank-deficient
//     dR (near-identical residuals as Anderson converges) is absorbed by
//     singular-value truncation, not safeguards.  The earlier normal-equations
//     + Cholesky path squared the conditioning (κ²) and required Tikhonov +
//     step-magnitude rejection to stay usable; thin QR fixed κ but blew up on
//     exact rank loss; truncated SVD handles both cleanly.
//   - Scalar path (n == 1): closed-form min-norm solution.  An underdetermined
//     1×m system has a unique min-norm solution that does not need a matrix
//     factorization; we keep the direct formula for speed.

/// Anderson acceleration for fixed-point iteration.
///
/// Solves nonlinear equations in fixed-point form x = g(x) by computing
/// the next iterate as a linear combination of previous iterates whose
/// coefficients minimise the least-squares residual.
pub struct Anderson {
    /// Buffer depth (number of stored iterates)
    m: usize,
    /// Rolling difference buffer for x
    dx_buffer: VecDeque<AVec>,
    /// Rolling difference buffer for residuals
    dr_buffer: VecDeque<AVec>,
    /// Previous iterate (valid iff !first_call)
    x_prev: AVec,
    /// Previous residual (valid iff !first_call)
    r_prev: AVec,
    /// True before the first difference has been stored
    first_call: bool,
    /// Scratch buffer for residual (reused across calls)
    _result: AVec,
    /// Scratch buffer for the tentative Anderson step Δx_aa
    _step: AVec,
    /// Scratch slot that rotates into dx_buffer (re-used)
    _dx_tmp: AVec,
    /// Scratch slot that rotates into dr_buffer (re-used)
    _dr_tmp: AVec,
    /// LS coefficient vector scratch (re-used)
    _c: AVec,
    /// Cached dense/sparse Newton linear solver for the matrix-residual path
    /// (`optimizer_step_residual`), so a constant mass-matrix DAE `A = M − dt·a·J`
    /// is factored once and reused — matching `Newton`/`NewtonAnderson`.
    lin: crate::optim::linsolve::LinearSolver,
}

impl Anderson {
    pub fn new(m: usize) -> Self {
        Self {
            m,
            dx_buffer: VecDeque::with_capacity(m),
            dr_buffer: VecDeque::with_capacity(m),
            x_prev: AVec::new(),
            r_prev: AVec::new(),
            first_call: true,
            _result: AVec::new(),
            _step: AVec::new(),
            _dx_tmp: AVec::new(),
            _dr_tmp: AVec::new(),
            _c: AVec::new(),
            lin: crate::optim::linsolve::LinearSolver::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(OPT_HISTORY)
    }

    /// Reset the accelerator (keeps scratch buffers allocated).
    pub fn reset(&mut self) {
        self.dx_buffer.clear();
        self.dr_buffer.clear();
        self.first_call = true;
    }

    /// Reconfigure the rolling-buffer depth.  Truncates the dx/dr buffers if
    /// the new depth is smaller and resets `first_call` so the next iteration
    /// re-establishes a clean history.  Cheap when called between solves.
    pub fn set_m(&mut self, m: usize) {
        self.m = m;
        while self.dx_buffer.len() > m { self.dx_buffer.pop_front(); }
        while self.dr_buffer.len() > m { self.dr_buffer.pop_front(); }
        self.first_call = true;
    }

    /// Perform one iteration on the fixed-point solution.
    ///
    /// # Arguments
    /// * `x` - current solution (modified in-place with result)
    /// * `g` - current evaluation of g(x)
    ///
    /// # Returns
    /// residual_norm
    pub fn step(&mut self, x: &mut [f64], g: &[f64]) -> f64 {
        let n = x.len();

        // Compute residual in-place using _result as scratch
        self._result.resize(n, 0.0);
        for i in 0..n { self._result[i] = g[i] - x[i]; }
        let res_norm = vec_norm(&self._result);

        // fallback to regular FPI if m == 0
        if self.m == 0 {
            x.copy_from_slice(g);
            return res_norm;
        }

        // if no buffer yet, regular fixed-point update
        if self.first_call {
            self.x_prev.resize(n, 0.0);
            self.r_prev.resize(n, 0.0);
            self.x_prev.copy_from_slice(x);
            self.r_prev.copy_from_slice(&self._result);
            self.first_call = false;
            x.copy_from_slice(g);
            return res_norm;
        }

        // Compute differences into scratch (no alloc after warm-up)
        self._dx_tmp.resize(n, 0.0);
        self._dr_tmp.resize(n, 0.0);
        for i in 0..n {
            self._dx_tmp[i] = x[i] - self.x_prev[i];
            self._dr_tmp[i] = self._result[i] - self.r_prev[i];
        }

        // Rotate scratch into the buffer: recycle the oldest slot once full,
        // otherwise push a fresh clone (only runs `m` times total per reset).
        if self.dx_buffer.len() >= self.m {
            let mut oldest_dx = self.dx_buffer.pop_front().unwrap();
            let mut oldest_dr = self.dr_buffer.pop_front().unwrap();
            std::mem::swap(&mut oldest_dx, &mut self._dx_tmp);
            std::mem::swap(&mut oldest_dr, &mut self._dr_tmp);
            self.dx_buffer.push_back(oldest_dx);
            self.dr_buffer.push_back(oldest_dr);
        } else {
            self.dx_buffer.push_back(self._dx_tmp.clone());
            self.dr_buffer.push_back(self._dr_tmp.clone());
        }

        // Save current iterate as previous for next call (no alloc)
        self.x_prev.copy_from_slice(x);
        self.r_prev.copy_from_slice(&self._result);

        let buf_len = self.dx_buffer.len();

        // Scalar case (n == 1): underdetermined 1×m LS.  Min-norm solution is
        //   c_k = dR_k · res / ||dR||² → step = res · (dR · dX) / ||dR||²
        // No safeguarding: rejecting large steps breaks stiff regimes where
        // ||dX|| ≫ ||dR|| is the correct secant extrapolation.
        if n == 1 {
            let mut dr2: f64 = 0.0;
            let mut dr_dx: f64 = 0.0;
            for i in 0..buf_len {
                dr2 += self.dr_buffer[i][0] * self.dr_buffer[i][0];
                dr_dx += self.dr_buffer[i][0] * self.dx_buffer[i][0];
            }
            if dr2 <= TOLERANCE {
                x.copy_from_slice(g);
                return res_norm;
            }
            let step = self._result[0] * dr_dx / dr2;
            x[0] -= step;
            return res_norm;
        }

        // Vector case: QR-based LS solve, then x ← x − Σ c_j · dx_j.
        self._c.resize(buf_len, 0.0);
        lstsq_solve(&self.dr_buffer, &self._result, buf_len, n, &mut self._c);

        // Defensive NaN/Inf guard: the SVD pseudo-inverse is intrinsically
        // well-defined (truncation kills zero singular values) but if the
        // input dR/res themselves carry NaN from an upstream overflow, the
        // result will too.  Fall back to pure Picard then.  This is *not* a
        // step-magnitude safeguard — large coefficients from a well-conditioned
        // solve are genuine secant extrapolation in stiff regimes and must
        // pass through unchanged.
        if !self._c.iter().all(|c| c.is_finite()) {
            x.copy_from_slice(g);
            return res_norm;
        }

        for (j, dx_j) in self.dx_buffer.iter().enumerate() {
            let c_j = self._c[j];
            for i in 0..n { x[i] -= c_j * dx_j[i]; }
        }

        res_norm
    }

    /// Solve for fixed point: find x such that f(x) = 0 where g(x) = f(x) + x.
    /// For testing purposes.
    pub fn solve(
        &mut self,
        func: &dyn Fn(&[f64]) -> Vec<f64>,
        x0: &[f64],
        iterations_max: usize,
        tolerance: f64,
    ) -> Result<(Vec<f64>, f64, usize), String> {
        let mut x = x0.to_vec();
        for i in 0..iterations_max {
            let fx = func(&x);
            let g: Vec<f64> = fx.iter().zip(x.iter()).map(|(&fi, &xi)| fi + xi).collect();
            let res = self.step(&mut x, &g);
            if res < tolerance {
                return Ok((x, res, i));
            }
        }
        Err(format!("did not converge in {} steps", iterations_max))
    }
}

/// Newton-Anderson hybrid solver.
///
/// Extends Anderson by prepending a Newton step when a Jacobian is available.
pub struct NewtonAnderson {
    /// Inner Anderson accelerator
    anderson: Anderson,
    /// Cached dense Newton linear solver (reuses the LU when the Newton matrix
    /// is unchanged, i.e. constant Jacobian at fixed dt/a_ii). Survives `reset`:
    /// the content compare invalidates it when the matrix actually changes.
    lin: crate::optim::linsolve::LinearSolver,
}

impl NewtonAnderson {
    pub fn new(m: usize) -> Self {
        Self {
            anderson: Anderson::new(m),
            lin: crate::optim::linsolve::LinearSolver::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(OPT_HISTORY)
    }

    pub fn reset(&mut self) {
        self.anderson.reset();
    }

    pub fn set_m(&mut self, m: usize) {
        self.anderson.set_m(m);
    }

    /// Perform one iteration in-place.
    /// With Jacobian: Newton step as preconditioner, then Anderson on the
    /// Newton-updated x. Anderson tracks the convergence of the outer
    /// fixed-point iteration (inter-block coupling).
    pub fn step(&mut self, x: &mut [f64], g: &[f64], jac: Option<f64>) -> f64 {
        match jac {
            None => self.anderson.step(x, g),
            Some(j) => {
                // Newton step: x = x - (J-I)^{-1} * (g - x)
                newton_step_scalar_inplace(x, g, j);
                // Anderson acceleration on the preconditioned iterate
                self.anderson.step(x, g)
            }
        }
    }

    /// Perform one iteration in-place with flat matrix Jacobian.
    pub fn step_matrix(
        &mut self,
        x: &mut [f64],
        g: &[f64],
        jac: Option<&[f64]>,
        n: usize,
    ) -> f64 {
        match jac {
            None => self.anderson.step(x, g),
            Some(j) => {
                self.lin.newton_step_matrix(x, g, j, n);
                self.anderson.step(x, g)
            }
        }
    }

    /// Sparse-Jacobian variant of [`Self::step_matrix`]: Newton precondition via
    /// the sparse `A = scale·J − I` (built from the coordinate pattern), then
    /// Anderson on the preconditioned iterate.
    pub fn step_matrix_sparse(
        &mut self,
        x: &mut [f64],
        g: &[f64],
        rows: &[u32],
        cols: &[u32],
        values: &[f64],
        scale: f64,
        n: usize,
    ) -> f64 {
        self.lin.newton_step_matrix_sparse(x, g, rows, cols, values, scale, n);
        self.anderson.step(x, g)
    }
}

/// Pure Newton solver. Same API as NewtonAnderson but without Anderson acceleration.
/// Converges quadratically when Jacobian is available — typically 2-3 iterations.
/// Intended for standalone integrate() where the full Jacobian is known.
pub struct Newton {
    /// Cached dense Newton linear solver (same caching as `NewtonAnderson`).
    lin: crate::optim::linsolve::LinearSolver,
}

impl Default for Newton {
    fn default() -> Self {
        Self::new()
    }
}

impl Newton {
    pub fn new() -> Self {
        Self { lin: crate::optim::linsolve::LinearSolver::new() }
    }
    pub fn with_defaults() -> Self { Self::new() }
    pub fn reset(&mut self) {}

    pub fn step(&mut self, x: &mut [f64], g: &[f64], jac: Option<f64>) -> f64 {
        match jac {
            None => {
                // No Jacobian — pure fixed-point iteration
                let mut res_sq = 0.0;
                for i in 0..x.len() {
                    let r = g[i] - x[i];
                    res_sq += r * r;
                    x[i] = g[i];
                }
                res_sq.sqrt()
            }
            Some(j) => newton_step_scalar_inplace(x, g, j),
        }
    }

    pub fn step_matrix(
        &mut self,
        x: &mut [f64],
        g: &[f64],
        jac: Option<&[f64]>,
        n: usize,
    ) -> f64 {
        match jac {
            None => {
                let mut res_sq = 0.0;
                for i in 0..x.len() {
                    let r = g[i] - x[i];
                    res_sq += r * r;
                    x[i] = g[i];
                }
                res_sq.sqrt()
            }
            Some(j) => self.lin.newton_step_matrix(x, g, j, n),
        }
    }

    /// Sparse-Jacobian variant of [`Self::step_matrix`] (pure Newton, no Anderson).
    pub fn step_matrix_sparse(
        &mut self,
        x: &mut [f64],
        g: &[f64],
        rows: &[u32],
        cols: &[u32],
        values: &[f64],
        scale: f64,
        n: usize,
    ) -> f64 {
        self.lin.newton_step_matrix_sparse(x, g, rows, cols, values, scale, n)
    }
}

/// Unified optimizer enum — dispatch to Anderson, NewtonAnderson, or Newton.
// The variants differ in size (NewtonAnderson carries the Anderson history plus
// the cached LinearSolver), but an Optimizer is a single owned per-solver field,
// never stored in bulk, so boxing would only add an indirection on the Newton
// hot path for no memory benefit.
#[allow(clippy::large_enum_variant)]
pub enum Optimizer {
    Anderson(Anderson),
    NewtonAnderson(NewtonAnderson),
    Newton(Newton),
}

/// One iteration of a Newton-on-residual solve, unified across optimiser
/// variants.  The caller provides the current iterate `x`, the residual
/// `r(x)` of the implicit equation, and the flat row-major Newton matrix
/// `A = ∂r/∂x`.  Internally we take a Newton step `g = x − A⁻¹·r`, then
/// either commit it (pure Newton) or hand `(x, g)` to Anderson acceleration
/// (Anderson / NewtonAnderson variants).
///
/// Returns `‖r‖₂`.
pub fn optimizer_step_residual(
    opt: &mut Optimizer,
    x: &mut [f64],
    r: &[f64],
    mat: &[f64],
    n: usize,
) -> f64 {
    // g = x − A⁻¹·r   (Newton update applied to a scratch copy of x). Every
    // variant routes through a cached `LinearSolver`, so a constant Newton matrix
    // `A = M − dt·a_ii·J` (linear DAE at fixed dt) is factored once and reused.
    let mut g: AVec = smallvec::SmallVec::from_slice(x);
    match opt {
        Optimizer::Newton(ne) => {
            let res_norm = ne.lin.newton_solve(&mut g, r, mat, n);
            x.copy_from_slice(&g);
            res_norm
        }
        Optimizer::NewtonAnderson(na) => {
            na.lin.newton_solve(&mut g, r, mat, n);
            na.anderson.step(x, &g)
        }
        Optimizer::Anderson(ander) => {
            ander.lin.newton_solve(&mut g, r, mat, n);
            ander.step(x, &g)
        }
    }
}

impl Optimizer {
    pub fn default_newton_anderson() -> Self {
        Self::NewtonAnderson(NewtonAnderson::with_defaults())
    }
}

impl Optimizer {
    pub fn reset(&mut self) {
        match self {
            Self::Anderson(a) => a.reset(),
            Self::NewtonAnderson(na) => na.reset(),
            Self::Newton(n) => n.reset(),
        }
    }

    /// Reconfigure the Anderson rolling-buffer depth.  No-op for pure Newton
    /// (which has no history).
    pub fn set_m(&mut self, m: usize) {
        match self {
            Self::Anderson(a) => a.set_m(m),
            Self::NewtonAnderson(na) => na.set_m(m),
            Self::Newton(_) => {}
        }
    }

    pub fn step(&mut self, x: &mut [f64], g: &[f64], jac: Option<f64>) -> f64 {
        match self {
            Self::Anderson(a) => a.step(x, g),
            Self::NewtonAnderson(na) => na.step(x, g, jac),
            Self::Newton(n) => n.step(x, g, jac),
        }
    }

    pub fn step_matrix(&mut self, x: &mut [f64], g: &[f64], jac: Option<&[f64]>, n: usize) -> f64 {
        match self {
            Self::Anderson(a) => a.step(x, g),
            Self::NewtonAnderson(na) => na.step_matrix(x, g, jac, n),
            Self::Newton(ne) => ne.step_matrix(x, g, jac, n),
        }
    }

    /// Sparse-Jacobian Newton step: `A = scale·J − I` from the coordinate pattern
    /// (`rows`, `cols`, `values`). Anderson-only optimizers ignore the Jacobian
    /// and fall back to a plain fixed-point step, matching `step_matrix`.
    pub fn step_matrix_sparse(
        &mut self,
        x: &mut [f64],
        g: &[f64],
        rows: &[u32],
        cols: &[u32],
        values: &[f64],
        scale: f64,
        n: usize,
    ) -> f64 {
        match self {
            Self::Anderson(a) => a.step(x, g),
            Self::NewtonAnderson(na) => na.step_matrix_sparse(x, g, rows, cols, values, scale, n),
            Self::Newton(ne) => ne.step_matrix_sparse(x, g, rows, cols, values, scale, n),
        }
    }
}

// ====================== Helper functions ======================

/// Euclidean norm of a vector.
fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Newton step for scalar/1D case in-place: x = x - res / (jac - 1)
fn newton_step_scalar_inplace(x: &mut [f64], g: &[f64], jac: f64) -> f64 {
    let n = x.len();
    let inv_jm1 = 1.0 / (jac - 1.0);
    let mut res_sq = 0.0;
    for i in 0..n {
        let res_i = g[i] - x[i];
        res_sq += res_i * res_i;
        x[i] -= res_i * inv_jm1;
    }
    res_sq.sqrt()
}

// `newton_step_matrix_inplace` moved to `crate::optim::linsolve` (shared with the
// fully-implicit DAE stage).

/// Solve the thin least-squares problem
///   minimize  ||dR · c − res||₂
/// via faer's thin SVD with Moore-Penrose pseudo-inverse: small singular
/// values are truncated relative to σ_max using numpy's default threshold
/// `rcond = max(n, m_eff) · ε_mach`.  This matches pathsim's
/// `np.linalg.lstsq(…, rcond=None)` behaviour bit-for-bit in semantics:
/// rank-deficient dR (e.g., two near-identical residuals near convergence)
/// is handled by dropping the degenerate directions instead of blowing up.
///
/// Beyond n residual-difference columns the system is underdetermined and
/// the oldest entries span no new directions in the n-dim residual space,
/// so we drop them and zero out their coefficients.
fn lstsq_solve(
    dr_buffer: &VecDeque<AVec>,
    res: &[f64],
    buf_len: usize,
    n: usize,
    c_out: &mut [f64],
) {
    use faer::Mat;
    use faer::linalg::solvers::Svd;

    // Keep only the m_eff newest columns; zero out the rest of c_out so the
    // caller's `Σ c_j dx_j` loop over buf_len entries stays correct.
    let m_eff = buf_len.min(n);
    // The Σ⁺·Uᵀ·res scratch (`tmp`) is a stack array of LSTSQ_MMAX; a history
    // depth set past it (via the public `set_m`) would index out of bounds.
    debug_assert!(
        m_eff <= LSTSQ_MMAX,
        "Anderson history {m_eff} exceeds LSTSQ_MMAX {LSTSQ_MMAX}"
    );
    let start = buf_len - m_eff;
    for j in 0..start { c_out[j] = 0.0; }

    let dr = Mat::<f64>::from_fn(n, m_eff, |i, j| dr_buffer[start + j][i]);

    // Thin SVD: U is n×m_eff, V is m_eff×m_eff, S is m_eff-diagonal.
    let svd = match Svd::new_thin(dr.as_ref()) {
        Ok(s) => s,
        Err(_) => {
            // Decomposition failed — skip Anderson update for this iteration.
            for j in 0..m_eff { c_out[start + j] = 0.0; }
            return;
        }
    };
    let u = svd.U();
    let v = svd.V();
    let s_col = svd.S().column_vector();

    // numpy-compatible rank threshold: max(shape) · ε_mach · σ_max.
    let s_max = (0..m_eff).fold(0.0_f64, |m, i| m.max(s_col[i]));
    let rcond = (n.max(m_eff) as f64) * f64::EPSILON * s_max;

    // tmp = Σ⁺ · Uᵀ · res    (size ≤ OPT_HISTORY ≤ LSTSQ_MMAX)
    let mut tmp = [0.0_f64; LSTSQ_MMAX];
    for i in 0..m_eff {
        let mut dot = 0.0;
        for k in 0..n { dot += u[(k, i)] * res[k]; }
        let si = s_col[i];
        tmp[i] = if si > rcond { dot / si } else { 0.0 };
    }

    // c = V · tmp
    for j in 0..m_eff {
        let mut sum = 0.0;
        for i in 0..m_eff { sum += v[(j, i)] * tmp[i]; }
        c_out[start + j] = sum;
    }
}

/// Upper bound on Anderson history size (OPT_HISTORY ≤ 4 in practice; 8 is
/// safety margin for the small stack-allocated scratch buffer above).
const LSTSQ_MMAX: usize = 8;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anderson_scalar_cos() {
        // Fixed point: x = cos(x), solution ≈ 0.7390851332
        let mut acc = Anderson::with_defaults();
        let result = acc.solve(
            &|x: &[f64]| vec![x[0].cos() - x[0]],
            &[0.5],
            100,
            1e-10,
        );
        assert!(result.is_ok());
        let (x, _res, _iters) = result.unwrap();
        assert!((x[0] - 0.7390851332).abs() < 1e-6);
    }

    #[test]
    fn test_anderson_vector() {
        // f(x,y) = (cos(y) - x, sin(x) - y)
        // Fixed point: x = cos(y), y = sin(x)
        let mut acc = Anderson::new(4);
        let result = acc.solve(
            &|x: &[f64]| vec![x[1].cos() - x[0], x[0].sin() - x[1]],
            &[0.5, 0.5],
            200,
            1e-8,
        );
        assert!(result.is_ok());
        let (x, _res, _iters) = result.unwrap();
        // Verify fixed point: x ≈ cos(y) and y ≈ sin(x)
        assert!((x[0] - x[1].cos()).abs() < 1e-6);
        assert!((x[1] - x[0].sin()).abs() < 1e-6);
    }

    #[test]
    fn test_anderson_step_basic() {
        let mut acc = Anderson::with_defaults();
        let mut x = vec![0.5];
        let _res1 = acc.step(&mut x, &[0.5_f64.cos()]);
        assert!(!x.is_empty());
        let g = vec![x[0].cos()];
        let _res2 = acc.step(&mut x, &g);
        assert!(!x.is_empty());
    }

    #[test]
    fn test_newton_anderson_scalar() {
        let mut na = NewtonAnderson::with_defaults();
        let mut x = vec![0.5];
        for _ in 0..20 {
            let g = vec![f64::cos(x[0])];
            let jac = -f64::sin(x[0]);
            let res = na.step(&mut x, &g, Some(jac));
            if res < 1e-10 {
                break;
            }
        }
        assert!((x[0] - 0.7390851332).abs() < 1e-6);
    }

    #[test]
    fn test_anderson_m_zero_fallback() {
        let mut acc = Anderson::new(0);
        let mut x = vec![1.0];
        let res = acc.step(&mut x, &[2.0]);
        assert_eq!(x, vec![2.0]); // pure FPI: returns g
        assert!((res - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_newton_step_matrix() {
        let mut x = vec![0.0, 0.0];
        let g = vec![1.6, 1.8];
        let jac_flat = vec![2.0, 1.0, 1.0, 3.0];
        crate::optim::linsolve::newton_step_matrix_inplace(&mut x, &g, &jac_flat, 2);
        assert!(x[0].abs() < 10.0);
        assert!(x[1].abs() < 10.0);
    }

    #[test]
    fn test_anderson_reset() {
        let mut acc = Anderson::with_defaults();
        let mut x1 = vec![1.0];
        acc.step(&mut x1, &[2.0]);
        let mut x2 = vec![1.5];
        acc.step(&mut x2, &[1.8]);
        acc.reset();
        assert!(acc.first_call);
        assert!(acc.dx_buffer.is_empty());
    }
}
