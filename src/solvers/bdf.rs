// Variable-step BDF (Backward Differentiation Formula) coefficients.
//
// Used by `gear52a_factory` (and any future fixed-order GEAR variants).  The
// standard tabulated BDF coefficients (see e.g. Hairer-Wanner II Table III.1)
// are defined for *constant* step size; under variable steps they have to be
// recomputed each step from the dt history.  Ported 1:1 from
// pathsim/solvers/gear.py:compute_bdf_coefficients.
//
// For BDF of order m at step n (with dt-history `[h_n, h_{n-1}, ...,
// h_{n-m+1}]`) the implicit update reads
//
//     Σ_{j=0..m} α_j · x_{n-j} = h_n · f(x_n, t_n)
//
// We rephrase as a fixed-point equation suitable for the existing
// Newton-Anderson optimiser:
//
//     x_n = (1/α_0) · h_n · f_n − Σ_{j=1..m} (α_j/α_0) · x_{n-j}
//         = β · h_n · f_n + Σ_{j=1..m} K_{j-1} · x_{n-j}
//
// where `β = 1/α_0` and `K_{j-1} = -α_j/α_0`.  Newton matrix is
// `I − β · h_n · ∂f/∂x`.
//
// Coefficient construction uses the Lagrange-interpolation identity
// `Σ α_j · θ_j^k = δ_{k,1}` (k = 0..m) on the *normalised* node positions
//
//     θ_0 = 0,  θ_j = -1 - Σ_{i=1..j-1} ρ_i   for j ≥ 1
//
// with `ρ_i = h_{n-i} / h_n`.  The (m+1)×(m+1) Vandermonde system is solved
// once per step and shared across the order-(n-1)/n/(n+1) candidates the
// variable-order error controller compares — keeping the linalg cost
// negligible vs the RHS / Jacobian evaluations.

use std::collections::VecDeque;
use faer::Mat;
use faer::prelude::*;

/// Variable-step BDF coefficients for the given order.
///
/// `dt_history[0]` is the *current* step `h_n`, `dt_history[1]` is `h_{n-1}`,
/// etc.  Caller guarantees `dt_history.len() >= order`.
///
/// Returns `(beta, alpha)` such that the BDF update reads
/// `x_n = beta · h_n · f_n + Σ alpha[j] · x_{n-1-j}` (j = 0..order-1, with
/// `x_{n-1}` paired with `alpha[0]`).
///
/// Special case `order == 1` (or only a single buffered dt available) is
/// backward Euler: `beta = 1`, `alpha = [1]`.
pub fn compute_bdf_coefficients(order: usize, dt_history: &VecDeque<f64>) -> (f64, Vec<f64>) {
    // `(β, K)` is just the post-processing of the raw `α` array: β = 1/α_0 and
    // K_j = -α_j/α_0. The backward-Euler special case (α = [1, -1]) maps to the
    // same `(1, [1])` the old direct path returned.
    let alphas = compute_bdf_alphas(order, dt_history);
    let alpha0 = alphas[0];
    let beta = 1.0 / alpha0;
    let k_coeffs: Vec<f64> = alphas[1..].iter().map(|&a| -a / alpha0).collect();
    (beta, k_coeffs)
}

/// Variable-step BDF `α`-coefficients for the NDF corrector.
///
/// Returns the full `α[0..=q]` array satisfying `Σ α_j · θ_j^k = δ_{k,1}`
/// for `k = 0..q` (same Vandermonde system as `compute_bdf_coefficients`,
/// just exposed without the `(β, K)` post-processing).  Used by the NDF
/// (Numerical Differentiation Formula) corrector — Shampine-Reichelt 1997 —
/// which combines `α_0` and `α_q` with an order-dependent `κ` parameter:
/// `β_NDF = 1 / (α_0 − κ_q · α_q)`.
///
/// (For `κ = 0` this collapses to plain BDF: `β_BDF = 1/α_0`.)
///
/// This is the single source of truth for the Vandermonde solve;
/// [`compute_bdf_coefficients`] derives its `(β, K)` from this `α` array. The
/// solve is a `(q+1) × (q+1)` LU, O(q^3) ≤ ~200 flops, negligible per step.
pub fn compute_bdf_alphas(order: usize, dt_history: &VecDeque<f64>) -> Vec<f64> {
    assert!(order >= 1, "BDF order must be >= 1");
    if dt_history.len() < 2 {
        // Backward Euler: α_0 y_n + α_1 y_{n-1} = h·f → α = [1, -1].
        return vec![1.0, -1.0];
    }
    let h_n = dt_history[0];
    let m = order;
    let mut theta = vec![-1.0; m + 1];
    theta[0] = 0.0;
    for j in 2..=m {
        let mut s = 0.0;
        for i in 1..j { s += dt_history[i] / h_n; }
        theta[j] -= s;
    }
    let n = m + 1;
    let mut a = Mat::<f64>::zeros(n, n);
    let mut rhs = Mat::<f64>::zeros(n, 1);
    rhs[(1, 0)] = 1.0;
    for k in 0..n {
        for j in 0..n {
            a[(k, j)] = theta[j].powi(k as i32);
        }
    }
    let lu = a.partial_piv_lu();
    let alpha_col = lu.solve(&rhs);
    (0..n).map(|i| alpha_col[(i, 0)]).collect()
}

/// NDF safety κ-coefficients from Shampine & Reichelt 1997, "The MATLAB
/// ODE Suite" §3.  These are negative (the formula `α·y − κ·α_q·(y − y_pred)`
/// reads with a sign convention that makes negative κ effectively additive
/// when working out the modified `β`).  Order 5 has κ = 0 — NDF5 ≡ BDF5
/// because no κ value gave a useful improvement for that order.
///
/// For BDF (κ = 0 across the board) leave this off and use plain
/// `compute_bdf_coefficients` instead.
pub const NDF_KAPPA: [f64; 6] = [
    0.0,        // unused (no order 0)
    -0.1850,    // NDF1
    -1.0 / 9.0, // NDF2
    -0.0823,    // NDF3
    -0.0415,    // NDF4
    0.0,        // NDF5 = BDF5
];

/// Variable-step BDF `l`-vector for the Nordsieck corrector.
///
/// `l[0..=q]` defines how the Newton discrepancy `xi = h_n · f(y_n) -
/// z_pred[1]` is distributed across the Nordsieck columns to produce
/// `z_n = z_pred + l ⊗ xi`.
///
/// Definitionally, `λ(τ) = Σ l[i] · τ^i` is the unique polynomial of degree
/// `q` satisfying:
///   * `λ(0) = γ = β`  (the BDF beta coefficient — `1 / α_0`),
///   * `λ'(0) = 1`,
///   * `λ(τ_j) = 0` for `j = 1..=q-1` where `τ_j = -(t_n − t_{n-j}) / h_n`.
///
/// Conditions 1 and 2 fix `l[0]` and `l[1]`; conditions 3 give a
/// `(q-1) × (q-1)` Vandermonde-type system for `l[2..=q]`.  Cost: one small
/// LU per step (≤ 4×4 for q ≤ 5), trivial in Rust.
///
/// `beta` is the precomputed `compute_bdf_coefficients(q, dt_history).0` —
/// passed in to avoid a redundant linalg solve.
pub fn compute_bdf_l_vector(q: usize, dt_history: &VecDeque<f64>, beta: f64) -> Vec<f64> {
    let mut l = vec![0.0; q + 1];
    if q == 0 { return l; }
    l[0] = beta;
    l[1] = 1.0;
    if q == 1 { return l; }

    // tau_j = -(t_n - t_{n-j}) / h_n for j = 1..=q-1.
    // Same `theta` as `compute_bdf_coefficients`, just renamed for the
    // Lagrange context.
    let h_n = dt_history[0];
    let mut tau = vec![0.0_f64; q];
    tau[1] = -1.0;
    for j in 2..q {
        let mut s = 0.0;
        for i in 1..j { s += dt_history[i] / h_n; }
        tau[j] = -1.0 - s;
    }

    // (q-1) × (q-1) system in l[2..=q]:
    //     Σ_{i=2..=q} l[i] · tau[j]^i = -beta - tau[j]   for j = 1..=q-1
    let m = q - 1;
    let mut a = Mat::<f64>::zeros(m, m);
    let mut b = Mat::<f64>::zeros(m, 1);
    for j_idx in 0..m {
        let j = j_idx + 1;
        let tau_j = tau[j];
        b[(j_idx, 0)] = -beta - tau_j;
        for i_idx in 0..m {
            let i = i_idx + 2;
            a[(j_idx, i_idx)] = tau_j.powi(i as i32);
        }
    }
    let lu = a.partial_piv_lu();
    let sol = lu.solve(&b);
    for i_idx in 0..m {
        l[i_idx + 2] = sol[(i_idx, 0)];
    }
    l
}

// ======================================================================================
// Nordsieck representation for variable-order BDF.
// ======================================================================================
//
// Stores the Taylor expansion of the state at `t_n` directly:
//
//     z[0] = y(t_n)
//     z[i] = h_n^i / i! · y^(i)(t_n)        for i = 1..q
//
// where `q` is the current integration order.  This is the form used by
// SUNDIALS / CVODE / LSODA and has three operational advantages over the
// raw-history representation:
//
//   1. **Predictor is a constant Pascal-matrix multiplication.**  Advancing
//      the polynomial from `t_n` to `t_n + h_n` (i.e. predicting `t_{n+1}`)
//      is `z[i] ← Σ_{j=i..q} C(j, i) · z[j]`, computable in-place via `q`
//      passes of `z[j-1] += z[j]`.  No per-step linalg solve.
//
//   2. **Step-size rescaling is a single multiplication per coefficient.**
//      `z[i] ← (h_new / h_old)^i · z[i]`.  In raw-history form, a step
//      change forces re-fitting the BDF coefficients against the new dt
//      ratios; in Nordsieck form the polynomial just gets reparametrised.
//
//   3. **Order changes are column add/drop with simple rescaling.**  Up by
//      one: extend `z` with one more derivative, computed from the most
//      recent corrector residual.  Down by one: drop `z[q]` (its
//      contribution is bounded by the LTE estimate).
//
// The corrector still requires the BDF `l`-vector (length `q + 1`) which is
// step-size-dependent under variable steps.  We compute it from the buffered
// step ratios; same numerical content as `compute_bdf_coefficients` but
// expressed in a form that fits the Nordsieck update.
//
// References:
//   * Hindmarsh & Petzold, "Algorithms for Numerical Solution of ODEs",
//     Sec. 5 (Adams + BDF Nordsieck framework).
//   * SUNDIALS / CVODE source `cvode/src/cvode_impl.h` and `cvode.c`.
//   * Hairer & Wanner II, §III.6 (variable-step BDF).

/// Mutable Nordsieck z-array for a variable-order BDF integrator.
///
/// `z[i]` has length `n_state` and represents the i-th scaled derivative.
/// `q` is the current order (active indices are `0..=q`).  `q_max` is the
/// physical capacity (= 5 for GEAR52A); `z[q+1..=q_max]` are kept zeroed
/// when inactive so order-up only needs to fill the new column.
#[derive(Debug, Clone)]
pub struct Nordsieck {
    pub z: Vec<Vec<f64>>,
    pub q: usize,
    pub q_max: usize,
    pub h: f64,
}

impl Nordsieck {
    /// Allocate a Nordsieck array of capacity `q_max + 1` for state size
    /// `n_state`.  All entries zeroed; `q` initialised to 1 and `h` to 0
    /// (caller seeds via `init_first_step`).
    pub fn new(n_state: usize, q_max: usize) -> Self {
        let z = (0..=q_max).map(|_| vec![0.0; n_state]).collect();
        Self { z, q: 1, q_max, h: 0.0 }
    }

    /// Seed `z` from the initial value `x_0` and the first dt.  After this,
    /// `z[0] = x_0`, all higher derivatives are zero (they will be filled in
    /// as soon as the integrator runs its first step at order 1).
    pub fn init_first_step(&mut self, x_0: &[f64], h_0: f64) {
        debug_assert_eq!(self.z[0].len(), x_0.len());
        self.z[0].copy_from_slice(x_0);
        for i in 1..=self.q_max {
            for v in self.z[i].iter_mut() { *v = 0.0; }
        }
        self.h = h_0;
        self.q = 1;
    }

    /// Pascal-matrix predictor — advance `z` to the next time level
    /// in-place.
    ///
    /// `z[i] ← Σ_{j=i..q} C(j, i) · z[j]` is the Taylor extrapolation of the
    /// polynomial currently stored in `z` to time `t + h_n`.  The
    /// equivalent in-place algorithm is `q` passes of `z[j-1] += z[j]` for
    /// `j` running down from `q` — cheaper than building the binomial
    /// coefficient matrix explicitly.
    ///
    /// Cost: `q · q · n_state / 2` adds.  No multiplications.
    pub fn predict(&mut self) {
        let q = self.q;
        let n = self.z[0].len();
        for i in 0..q {
            for j in (i + 1..=q).rev() {
                let (lo, hi) = self.z.split_at_mut(j);
                let dst = &mut lo[j - 1];
                let src = &hi[0];
                for k in 0..n {
                    dst[k] += src[k];
                }
            }
        }
    }

    /// Apply the Newton correction `xi` (the discrepancy
    /// `h · f(y_n) − z_pred[1]`) using the BDF `l`-vector:
    /// `z[i] ← z[i] + l[i] · xi` for `i = 0..=q`.
    ///
    /// Caller's responsibility: `l.len() >= q + 1` and the same `xi` was
    /// produced by the converged Newton solve at this order.
    pub fn correct(&mut self, l: &[f64], xi: &[f64]) {
        let q = self.q;
        let n = self.z[0].len();
        debug_assert!(l.len() > q);
        debug_assert_eq!(xi.len(), n);
        for i in 0..=q {
            let li = l[i];
            let zi = &mut self.z[i];
            for k in 0..n {
                zi[k] += li * xi[k];
            }
        }
    }

    /// Rescale `z` for a step-size change `h_new / h_old = ratio`.
    ///
    /// `z[i] ← ratio^i · z[i]` and `self.h ← ratio · self.h`.  Required
    /// before predict/correct on every step where the controller chose a
    /// new dt.  Cost: `q · n_state` mults; no allocations.
    pub fn rescale(&mut self, ratio: f64) {
        let mut factor = 1.0;
        for i in 0..=self.q {
            if i > 0 {
                factor *= ratio;
                let zi = &mut self.z[i];
                for v in zi.iter_mut() { *v *= factor; }
            }
        }
        self.h *= ratio;
    }

    /// Drop the highest derivative — order down by one.  Just shrinks `q`;
    /// `z[q]` stays in memory but is logically inactive.
    pub fn order_down(&mut self) {
        if self.q > 1 { self.q -= 1; }
    }

    /// Fit a polynomial of degree `q` through `history[0..=q]` and seed `z`
    /// from its Taylor expansion at `t_n`.
    ///
    /// Used after every order change to keep `z` consistent with the
    /// history-based BDF residual evaluator (which the error controller
    /// still uses).  Also doubles as a clean re-bootstrap mechanism: any
    /// time the integrator is unsure whether `z` reflects current state
    /// (e.g. after a step rejection that the controller didn't fully
    /// account for), calling this with the current `q` resyncs.
    ///
    /// Caller invariants: `history.len() > q`, `dt_history.len() > q`,
    /// `q <= q_max`.  Sets `self.q = q` and `self.h = dt_history[0]`.
    ///
    /// **dt_history convention** (matches `compute_bdf_coefficients`):
    ///   * `dt_history[0]` is the *upcoming* step `h_n` (used as the
    ///     scaling for `z[i]` so a subsequent `predict()` advances by one
    ///     step).
    ///   * `dt_history[i]` for `i >= 1` is the gap from `history[i-1]`
    ///     backward to `history[i]` (the dt of the step that produced
    ///     `history[i-1]`).
    ///
    /// Cost: one `(q+1) × (q+1)` LU factor plus `n_state` triangular
    /// solves.  Negligible for `q <= 5`.
    pub fn init_from_history(
        &mut self,
        history: &std::collections::VecDeque<Vec<f64>>,
        dt_history: &VecDeque<f64>,
        q: usize,
    ) {
        debug_assert!(q <= self.q_max);
        debug_assert!(history.len() > q);
        debug_assert!(dt_history.len() > q);
        let n_state = history[0].len();
        debug_assert_eq!(self.z[0].len(), n_state);

        // Normalised node positions σ_j = (t_{n-1-j} − t_{n-1}) / h_n.
        // history[0] sits at the center (σ_0 = 0); history[j] for j >= 1
        // sits one or more backward steps away.  The backward gaps live in
        // dt_history[1..=q], scaled by h_n = dt_history[0].
        let h_n = dt_history[0];
        let mut sigma = vec![0.0_f64; q + 1];
        for j in 1..=q {
            let mut s = 0.0;
            for i in 1..=j {
                s += dt_history[i] / h_n;
            }
            sigma[j] = -s;
        }

        // Build Vandermonde V[j, i] = σ_j^i and RHS containing history[j]
        // as its j-th row.  Solving V · Z = X gives z (all state components
        // at once via the matrix RHS — single LU + n_state back-subs).
        let m = q + 1;
        let mut v = Mat::<f64>::zeros(m, m);
        for j in 0..m {
            for i in 0..m {
                v[(j, i)] = sigma[j].powi(i as i32);
            }
        }
        let mut rhs = Mat::<f64>::zeros(m, n_state);
        for j in 0..m {
            for s in 0..n_state {
                rhs[(j, s)] = history[j][s];
            }
        }
        let lu = v.partial_piv_lu();
        let sol = lu.solve(&rhs);
        for i in 0..m {
            for s in 0..n_state {
                self.z[i][s] = sol[(i, s)];
            }
        }
        // Zero out unused rows so order_up later starts from a clean slate.
        for i in (m)..=self.q_max {
            for v in self.z[i].iter_mut() { *v = 0.0; }
        }
        self.q = q;
        self.h = h_n;
    }

    /// Increase order by one, filling `z[q+1]` from the new corrector
    /// magnitude.  The standard BDF order-up rule (Hairer-Wanner II.5.13):
    /// `z_new[q+1] = (h^{q+1} / (q+1)!) · y^(q+1)(t_n)`, approximated by the
    /// finite difference of the latest corrector update with the previous
    /// one.  Caller passes `delta_q1 = ξ_n − ξ_{n-1}` (already scaled).
    ///
    /// `xi_factor` is the constant `1 / ((q+1) · (1 + xi_1) · … )`
    /// (Hairer-Wanner Eq. III.5.7).  In practice we just store `delta_q1`
    /// directly; the controller decides whether the magnitude is small
    /// enough to justify the order increase.
    pub fn order_up(&mut self, delta_q_plus_1: &[f64]) {
        if self.q >= self.q_max { return; }
        let new_q = self.q + 1;
        debug_assert_eq!(delta_q_plus_1.len(), self.z[0].len());
        self.z[new_q].copy_from_slice(delta_q_plus_1);
        self.q = new_q;
    }
}

/// Polynomial extrapolation of `x_n` from past states.
///
/// Builds the unique polynomial of degree `k - 1` through the points
/// `(t_{n-1}, history[0])`, `(t_{n-2}, history[1])`, …,
/// `(t_{n-k}, history[k-1])` and evaluates it at `t_n`.  Used as the initial
/// guess for the Newton iteration in `gear52a_factory`'s `solve_fn` so the
/// implicit solve starts inside the quadratic-convergence basin instead of
/// from `x_{n-1}`.
///
/// Caller invariants: `history.len() >= k`, `dt_history.len() >= k`,
/// `k >= 1`.  All vectors in `history` must have length `n_state`.
///
/// Returns `x_n_pred[i] = Σ_{j=0..k-1} L_j(t_n) · history[j][i]`.  Uses the
/// same `theta` normalisation as `compute_bdf_coefficients`:
/// `theta[j] = -(t_n - t_{n-j}) / h_n`, so `L_j(t_n) = ∏_{i ≠ j} theta[i+1] /
/// (theta[i+1] - theta[j+1])`.
///
/// Cost: `O(k² + k · n_state)`.  For `k <= 5` and small `n_state` this is a
/// few hundred flops — cheap relative to a Newton iteration.
pub fn lagrange_predict_from_history(
    history: &std::collections::VecDeque<Vec<f64>>,
    dt_history: &VecDeque<f64>,
    k: usize,
    n_state: usize,
) -> Vec<f64> {
    debug_assert!(k >= 1);
    debug_assert!(history.len() >= k);
    debug_assert!(dt_history.len() >= k);

    // Special case: order 1 → predictor is just x_{n-1}.
    if k == 1 {
        return history[0].clone();
    }

    // Build theta[1..=k] the same way `compute_bdf_coefficients` does.
    let h_n = dt_history[0];
    let mut theta = vec![0.0_f64; k + 1];
    theta[1] = -1.0;
    for j in 2..=k {
        let mut s = 0.0;
        for i in 1..j {
            s += dt_history[i] / h_n;
        }
        theta[j] = -1.0 - s;
    }

    // Lagrange weight for history index j: ∏_{i ≠ j} theta[i+1] / (theta[i+1] - theta[j+1])
    let mut result = vec![0.0_f64; n_state];
    for j in 0..k {
        let theta_j = theta[j + 1];
        let mut weight = 1.0;
        for i in 0..k {
            if i == j { continue; }
            let theta_i = theta[i + 1];
            weight *= theta_i / (theta_i - theta_j);
        }
        let hist_j = &history[j];
        for idx in 0..n_state {
            result[idx] += weight * hist_j[idx];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dt_hist(values: &[f64]) -> VecDeque<f64> {
        values.iter().copied().collect()
    }

    #[test]
    fn bdf1_constant_step_is_backward_euler() {
        let h = dt_hist(&[0.1]);
        let (beta, k) = compute_bdf_coefficients(1, &h);
        assert!((beta - 1.0).abs() < 1e-12);
        assert_eq!(k.len(), 1);
        assert!((k[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bdf2_constant_step_matches_table() {
        // BDF2 constant: x_n = (4/3)x_{n-1} - (1/3)x_{n-2} + (2/3)·h·f_n
        let h = dt_hist(&[0.1, 0.1]);
        let (beta, k) = compute_bdf_coefficients(2, &h);
        assert!((beta - 2.0 / 3.0).abs() < 1e-10, "beta = {}", beta);
        assert!((k[0] - 4.0 / 3.0).abs() < 1e-10, "K[0] = {}", k[0]);
        assert!((k[1] + 1.0 / 3.0).abs() < 1e-10, "K[1] = {}", k[1]);
    }

    #[test]
    fn bdf3_constant_step_matches_table() {
        // BDF3 constant: β=6/11, α=[18/11, -9/11, 2/11]
        let h = dt_hist(&[0.05, 0.05, 0.05]);
        let (beta, k) = compute_bdf_coefficients(3, &h);
        assert!((beta - 6.0 / 11.0).abs() < 1e-10, "beta = {}", beta);
        assert!((k[0] - 18.0 / 11.0).abs() < 1e-10);
        assert!((k[1] + 9.0 / 11.0).abs() < 1e-10);
        assert!((k[2] - 2.0 / 11.0).abs() < 1e-10);
    }

    #[test]
    fn bdf4_constant_step_matches_table() {
        // BDF4 constant: β=12/25, α=[48/25, -36/25, 16/25, -3/25]
        let h = dt_hist(&[0.01, 0.01, 0.01, 0.01]);
        let (beta, k) = compute_bdf_coefficients(4, &h);
        assert!((beta - 12.0 / 25.0).abs() < 1e-10);
        assert!((k[0] - 48.0 / 25.0).abs() < 1e-10);
        assert!((k[1] + 36.0 / 25.0).abs() < 1e-10);
        assert!((k[2] - 16.0 / 25.0).abs() < 1e-10);
        assert!((k[3] + 3.0 / 25.0).abs() < 1e-10);
    }

    #[test]
    fn bdf5_constant_step_matches_table() {
        // BDF5 constant: β=60/137, α=[300/137, -300/137, 200/137, -75/137, 12/137]
        let h = dt_hist(&[0.02, 0.02, 0.02, 0.02, 0.02]);
        let (beta, k) = compute_bdf_coefficients(5, &h);
        assert!((beta - 60.0 / 137.0).abs() < 1e-10);
        assert!((k[0] - 300.0 / 137.0).abs() < 1e-10);
        assert!((k[1] + 300.0 / 137.0).abs() < 1e-10);
        assert!((k[2] - 200.0 / 137.0).abs() < 1e-10);
        assert!((k[3] + 75.0 / 137.0).abs() < 1e-10);
        assert!((k[4] - 12.0 / 137.0).abs() < 1e-10);
    }

    #[test]
    fn bdf2_variable_step_consistency() {
        // Variable step: at the boundary where ρ → 1 the variable-step
        // coefficients must collapse onto the constant-step ones.
        let h = dt_hist(&[0.1, 0.1000001]);
        let (beta, k) = compute_bdf_coefficients(2, &h);
        assert!((beta - 2.0 / 3.0).abs() < 1e-4);
        assert!((k[0] - 4.0 / 3.0).abs() < 1e-4);
        assert!((k[1] + 1.0 / 3.0).abs() < 1e-4);
    }

    #[test]
    fn bdf2_variable_step_exact_on_quadratic() {
        // BDF2 has order 2: must integrate x(t) = t² exactly under any
        // (positive) variable-step history.  Taking h_n=1, h_{n-1}=2 puts
        // t_{n-2}=0, t_{n-1}=2, t_n=3 with x_n=9, x_{n-1}=4, x_{n-2}=0,
        // and f_n = 2·t_n = 6.  The reconstructed x_n must match exactly.
        let h = dt_hist(&[1.0, 2.0]);
        let (beta, k) = compute_bdf_coefficients(2, &h);
        let h_n = 1.0;
        let t_n = 3.0;
        let f_n = 2.0 * t_n;
        let x_n_predicted = beta * h_n * f_n + k[0] * 4.0 + k[1] * 0.0;
        assert!((x_n_predicted - 9.0).abs() < 1e-10,
            "BDF2 must be exact on a quadratic, got {} (β={}, K={:?})",
            x_n_predicted, beta, k);
    }

    fn hist(rows: &[&[f64]]) -> std::collections::VecDeque<Vec<f64>> {
        rows.iter().map(|r| r.to_vec()).collect()
    }

    // -- alpha tests --

    #[test]
    fn alpha_q1_is_backward_euler() {
        let h = dt_hist(&[0.1]);
        let alphas = compute_bdf_alphas(1, &h);
        assert_eq!(alphas, vec![1.0, -1.0]);
    }

    #[test]
    fn alpha_q2_constant_step() {
        // BDF2 const-step: α = [3/2, -2, 1/2]
        let h = dt_hist(&[0.1, 0.1]);
        let alphas = compute_bdf_alphas(2, &h);
        assert!((alphas[0] - 1.5).abs() < 1e-12);
        assert!((alphas[1] + 2.0).abs() < 1e-12);
        assert!((alphas[2] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn alpha_consistent_with_beta() {
        // β = 1/α_0 from compute_bdf_coefficients should match.
        let h = dt_hist(&[0.05, 0.07, 0.1]);
        let alphas = compute_bdf_alphas(3, &h);
        let (beta, _) = compute_bdf_coefficients(3, &h);
        assert!((1.0 / alphas[0] - beta).abs() < 1e-10);
    }

    // -- l-vector tests --

    #[test]
    fn l_vector_q1_is_trivial() {
        let dt = dt_hist(&[0.1]);
        let l = compute_bdf_l_vector(1, &dt, 1.0);
        assert_eq!(l, vec![1.0, 1.0]);
    }

    #[test]
    fn l_vector_q2_constant_step() {
        let dt = dt_hist(&[0.1, 0.1]);
        let beta = 2.0 / 3.0;
        let l = compute_bdf_l_vector(2, &dt, beta);
        assert!((l[0] - beta).abs() < 1e-12);
        assert!((l[1] - 1.0).abs() < 1e-12);
        assert!((l[2] - 1.0 / 3.0).abs() < 1e-12, "got {}", l[2]);
    }

    #[test]
    fn l_vector_q3_constant_step() {
        let dt = dt_hist(&[0.1, 0.1, 0.1]);
        let beta = 6.0 / 11.0;
        let l = compute_bdf_l_vector(3, &dt, beta);
        // CVODE constant: l = [6/11, 1, 6/11, 1/11]
        assert!((l[0] - beta).abs() < 1e-12);
        assert!((l[1] - 1.0).abs() < 1e-12);
        assert!((l[2] - 6.0 / 11.0).abs() < 1e-12, "l[2] = {}", l[2]);
        assert!((l[3] - 1.0 / 11.0).abs() < 1e-12, "l[3] = {}", l[3]);
    }

    #[test]
    fn l_vector_q4_constant_step() {
        let dt = dt_hist(&[0.1, 0.1, 0.1, 0.1]);
        let beta = 12.0 / 25.0;
        let l = compute_bdf_l_vector(4, &dt, beta);
        // CVODE constant: l = [12/25, 1, 7/10, 1/5, 1/50]
        assert!((l[0] - beta).abs() < 1e-12);
        assert!((l[1] - 1.0).abs() < 1e-12);
        assert!((l[2] - 7.0 / 10.0).abs() < 1e-12, "l[2] = {}", l[2]);
        assert!((l[3] - 1.0 / 5.0).abs() < 1e-12, "l[3] = {}", l[3]);
        assert!((l[4] - 1.0 / 50.0).abs() < 1e-12, "l[4] = {}", l[4]);
    }

    #[test]
    fn l_vector_q5_constant_step() {
        let dt = dt_hist(&[0.1, 0.1, 0.1, 0.1, 0.1]);
        let beta = 60.0 / 137.0;
        let l = compute_bdf_l_vector(5, &dt, beta);
        // CVODE constant: l = [60/137, 1, 225/274, 85/274, 15/274, 1/274]
        assert!((l[0] - beta).abs() < 1e-12);
        assert!((l[1] - 1.0).abs() < 1e-12);
        assert!((l[2] - 225.0 / 274.0).abs() < 1e-12, "l[2] = {}", l[2]);
        assert!((l[3] - 85.0 / 274.0).abs() < 1e-12, "l[3] = {}", l[3]);
        assert!((l[4] - 15.0 / 274.0).abs() < 1e-12, "l[4] = {}", l[4]);
        assert!((l[5] - 1.0 / 274.0).abs() < 1e-12, "l[5] = {}", l[5]);
    }

    // -- Nordsieck tests --

    #[test]
    fn nordsieck_init_first_step_seeds_z0() {
        let mut ns = Nordsieck::new(3, 5);
        ns.init_first_step(&[1.0, 2.0, 3.0], 0.1);
        assert_eq!(ns.z[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(ns.q, 1);
        assert!((ns.h - 0.1).abs() < 1e-15);
        for i in 1..=ns.q_max {
            assert!(ns.z[i].iter().all(|&v| v == 0.0), "z[{}] not zero", i);
        }
    }

    #[test]
    fn nordsieck_predict_pascal_matches_taylor_q2() {
        // y(t) = 1 + 2t + 3t² → y(0)=1, hy'(0)=2h, h²y''(0)/2 = 3h².  Predictor at t+h: y(h) = 1+2h+3h².
        let h = 0.5;
        let mut ns = Nordsieck::new(1, 5);
        ns.h = h;
        ns.q = 2;
        ns.z[0] = vec![1.0];
        ns.z[1] = vec![2.0 * h];
        ns.z[2] = vec![3.0 * h * h];
        ns.predict();
        // After predict: z[0] = y(h), z[1] = h·y'(h) = h·(2 + 6h), z[2] unchanged
        let expected_y = 1.0 + 2.0 * h + 3.0 * h * h;
        let expected_dy_h = h * (2.0 + 6.0 * h);
        let expected_d2y_h2_2 = 3.0 * h * h;
        assert!((ns.z[0][0] - expected_y).abs() < 1e-12, "z[0] = {} vs {}", ns.z[0][0], expected_y);
        assert!((ns.z[1][0] - expected_dy_h).abs() < 1e-12, "z[1] = {} vs {}", ns.z[1][0], expected_dy_h);
        assert!((ns.z[2][0] - expected_d2y_h2_2).abs() < 1e-12, "z[2] = {} vs {}", ns.z[2][0], expected_d2y_h2_2);
    }

    #[test]
    fn nordsieck_predict_q3_cubic_exact() {
        // y(t) = t³ → at t=0: y=0, y'=0, y''=0, y'''=6.  Scaled: z[0]=0, z[1]=0, z[2]=0, z[3]=h³.
        let h = 0.4;
        let mut ns = Nordsieck::new(1, 5);
        ns.h = h;
        ns.q = 3;
        ns.z[0] = vec![0.0];
        ns.z[1] = vec![0.0];
        ns.z[2] = vec![0.0];
        ns.z[3] = vec![h * h * h];
        ns.predict();
        // y(h) = h³.
        assert!((ns.z[0][0] - h * h * h).abs() < 1e-12);
    }

    #[test]
    fn nordsieck_correct_applies_l_vector() {
        let mut ns = Nordsieck::new(2, 3);
        ns.q = 2;
        ns.z[0] = vec![1.0, 1.0];
        ns.z[1] = vec![0.0, 0.0];
        ns.z[2] = vec![0.0, 0.0];
        // BDF2 constant l = [2/3, 1, 1/3].  ξ = (10, -10).
        let l = [2.0 / 3.0, 1.0, 1.0 / 3.0];
        let xi = [10.0, -10.0];
        ns.correct(&l, &xi);
        assert!((ns.z[0][0] - (1.0 + 2.0/3.0 * 10.0)).abs() < 1e-12);
        assert!((ns.z[0][1] - (1.0 - 2.0/3.0 * 10.0)).abs() < 1e-12);
        assert!((ns.z[1][0] - 10.0).abs() < 1e-12);
        assert!((ns.z[1][1] + 10.0).abs() < 1e-12);
        assert!((ns.z[2][0] - 10.0/3.0).abs() < 1e-12);
        assert!((ns.z[2][1] + 10.0/3.0).abs() < 1e-12);
    }

    #[test]
    fn nordsieck_rescale_geometric() {
        let mut ns = Nordsieck::new(1, 5);
        ns.h = 0.1;
        ns.q = 3;
        ns.z[0] = vec![1.0];
        ns.z[1] = vec![1.0];
        ns.z[2] = vec![1.0];
        ns.z[3] = vec![1.0];
        // Halve the step.
        ns.rescale(0.5);
        assert!((ns.z[0][0] - 1.0).abs() < 1e-15);
        assert!((ns.z[1][0] - 0.5).abs() < 1e-15);
        assert!((ns.z[2][0] - 0.25).abs() < 1e-15);
        assert!((ns.z[3][0] - 0.125).abs() < 1e-15);
        assert!((ns.h - 0.05).abs() < 1e-15);
    }

    #[test]
    fn nordsieck_order_down_shrinks_q() {
        let mut ns = Nordsieck::new(1, 5);
        ns.q = 4;
        ns.order_down();
        assert_eq!(ns.q, 3);
    }

    #[test]
    fn nordsieck_order_up_fills_new_column() {
        let mut ns = Nordsieck::new(2, 5);
        ns.q = 2;
        ns.order_up(&[7.0, -7.0]);
        assert_eq!(ns.q, 3);
        assert_eq!(ns.z[3], vec![7.0, -7.0]);
    }

    #[test]
    fn nordsieck_order_up_capped_at_q_max() {
        let mut ns = Nordsieck::new(1, 3);
        ns.q = 3;
        ns.order_up(&[42.0]);
        assert_eq!(ns.q, 3, "must not exceed q_max");
    }

    #[test]
    fn nordsieck_init_from_history_q1() {
        // history[0] = x at center t_c, history[1] = x at t_c - gap_01.
        // dt_history[0] = h_n (upcoming, scales z), dt_history[1] = gap_01.
        let mut ns = Nordsieck::new(1, 5);
        let history = hist(&[&[3.0], &[1.0]]);
        let dt = dt_hist(&[1.0, 1.0]); // h_n = 1, gap_01 = 1
        ns.init_from_history(&history, &dt, 1);
        assert!((ns.z[0][0] - 3.0).abs() < 1e-12);
        // Linear polynomial through (σ=0, x=3) and (σ=-1, x=1):
        // p(σ) = 3 + 2σ → z[0] = 3, z[1] = 2.
        assert!((ns.z[1][0] - 2.0).abs() < 1e-12, "got {}", ns.z[1][0]);
        for i in 2..=5 { assert!(ns.z[i][0].abs() < 1e-15); }
        assert_eq!(ns.q, 1);
        assert!((ns.h - 1.0).abs() < 1e-12);
    }

    #[test]
    fn nordsieck_init_from_history_q2_recovers_quadratic() {
        // y(t) = 1 + 2t + 3t².  Center t_c = 2, past: t = 1, 0.  Values:
        // history[0] = y(2) = 17, history[1] = y(1) = 6, history[2] = y(0) = 1.
        // h_n = 1 (scaling).  gap_01 = 2 - 1 = 1.  gap_12 = 1 - 0 = 1.
        // At t_c: z[0]=17, z[1]=h·y'(2)=14, z[2]=h²·y''(2)/2=3.
        let mut ns = Nordsieck::new(1, 5);
        let history = hist(&[&[17.0], &[6.0], &[1.0]]);
        let dt = dt_hist(&[1.0, 1.0, 1.0]);
        ns.init_from_history(&history, &dt, 2);
        assert!((ns.z[0][0] - 17.0).abs() < 1e-10);
        assert!((ns.z[1][0] - 14.0).abs() < 1e-10, "z[1] = {}", ns.z[1][0]);
        assert!((ns.z[2][0] - 3.0).abs() < 1e-10, "z[2] = {}", ns.z[2][0]);
        assert!(ns.z[3][0].abs() < 1e-15);
    }

    #[test]
    fn nordsieck_init_from_history_q3_variable_step() {
        // y(t) = t³.  Center t_c = 3.  Past at t = 2, 1, 0.  History = [27, 8, 1, 0].
        // gap_01 = 1 (from 3 to 2), gap_12 = 1 (2 to 1), gap_23 = 1 (1 to 0).
        // h_n = 1 (scaling).
        let mut ns = Nordsieck::new(1, 5);
        let history = hist(&[&[27.0], &[8.0], &[1.0], &[0.0]]);
        let dt = dt_hist(&[1.0, 1.0, 1.0, 1.0]);
        ns.init_from_history(&history, &dt, 3);
        // y(t)=t³ → at t_c=3: z[0]=27, z[1]=h·27=27, z[2]=h²·9=9, z[3]=h³·1=1.
        assert!((ns.z[0][0] - 27.0).abs() < 1e-9);
        assert!((ns.z[1][0] - 27.0).abs() < 1e-9, "z[1] = {}", ns.z[1][0]);
        assert!((ns.z[2][0] - 9.0).abs() < 1e-9, "z[2] = {}", ns.z[2][0]);
        assert!((ns.z[3][0] - 1.0).abs() < 1e-9, "z[3] = {}", ns.z[3][0]);
    }

    #[test]
    fn nordsieck_predict_then_correct_round_trip() {
        // y(t) = 1 + 2t.  Past at t = 0, 0.1, history=[1.2, 1.0].  Center t_c = 0.1.
        // h_n = 0.1 (scaling), gap_01 = 0.1.  z[0]=1.2, z[1]=h·2=0.2.
        // Predict (advance by h_n): center moves to 0.2, z[0]=1.4, z[1]=0.2.
        let mut ns = Nordsieck::new(1, 5);
        let history = hist(&[&[1.2], &[1.0]]);
        let dt = dt_hist(&[0.1, 0.1]);
        ns.init_from_history(&history, &dt, 1);
        ns.predict();
        assert!((ns.z[0][0] - 1.4).abs() < 1e-12, "after predict: {}", ns.z[0][0]);
        assert!((ns.z[1][0] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn nordsieck_init_variable_step_quadratic() {
        // y(t) = t².  Center t_c = 3.  Past at t = 1.5 and 0.
        // gap_01 = 1.5, gap_12 = 1.5.  h_n = 0.5 (different from gaps!).
        // history = [9, 2.25, 0].
        // z scaled by h_n = 0.5: z[0]=9, z[1]=0.5·2t at t=3 = 0.5·6 = 3,
        // z[2] = 0.25 · y''(3)/2 = 0.25 · 1 = 0.25.
        let mut ns = Nordsieck::new(1, 5);
        let history = hist(&[&[9.0], &[2.25], &[0.0]]);
        let dt = dt_hist(&[0.5, 1.5, 1.5]);
        ns.init_from_history(&history, &dt, 2);
        assert!((ns.z[0][0] - 9.0).abs() < 1e-10);
        assert!((ns.z[1][0] - 3.0).abs() < 1e-10, "z[1] = {}", ns.z[1][0]);
        assert!((ns.z[2][0] - 0.25).abs() < 1e-10, "z[2] = {}", ns.z[2][0]);
    }

    #[test]
    fn lagrange_predict_order1_returns_last_state() {
        let h = hist(&[&[3.2, -2.7]]);
        let dt = dt_hist(&[0.1]);
        let pred = lagrange_predict_from_history(&h, &dt, 1, 2);
        assert_eq!(pred, vec![3.2, -2.7]);
    }

    #[test]
    fn lagrange_predict_order2_extrapolates_linear() {
        // Linear sequence: y(t) = 1 + 2*t.  Past values at t = 0.0, 0.1.
        // Predictor at t = 0.2 must give 1.4 exactly.
        let h = hist(&[&[1.2], &[1.0]]); // history[0] = y(0.1), history[1] = y(0.0)
        let dt = dt_hist(&[0.1, 0.1]);    // h_n (current) = 0.1, h_{n-1} = 0.1
        let pred = lagrange_predict_from_history(&h, &dt, 2, 1);
        assert!((pred[0] - 1.4).abs() < 1e-12, "got {}", pred[0]);
    }

    #[test]
    fn lagrange_predict_order3_exact_on_quadratic() {
        // y(t) = t².  Past values at t = 0, 1, 2.  At t = 3 → y = 9.
        let h = hist(&[&[4.0], &[1.0], &[0.0]]);
        let dt = dt_hist(&[1.0, 1.0, 1.0]);
        let pred = lagrange_predict_from_history(&h, &dt, 3, 1);
        assert!((pred[0] - 9.0).abs() < 1e-10, "got {}", pred[0]);
    }

    #[test]
    fn lagrange_predict_order4_variable_step_exact_on_cubic() {
        // y(t) = t³.  Use variable steps: t = {0, 1.5, 2.5, 3.0}, predict at 3.5.
        // Past values: history[0]=y(3.0)=27, [1]=y(2.5)=15.625, [2]=y(1.5)=3.375, [3]=y(0)=0.
        // dt_history[0] = h_n = 0.5 (gap from 3.0 to 3.5), [1]=0.5, [2]=1.0, [3]=1.5.
        let h = hist(&[&[27.0], &[15.625], &[3.375], &[0.0]]);
        let dt = dt_hist(&[0.5, 0.5, 1.0, 1.5]);
        let pred = lagrange_predict_from_history(&h, &dt, 4, 1);
        assert!((pred[0] - 42.875).abs() < 1e-10, "got {} expected 3.5³ = 42.875", pred[0]);
    }

    #[test]
    fn lagrange_predict_multidim() {
        // Both components quadratic: y₀(t) = t², y₁(t) = -t² + 1.
        // Past: t = 0, 1, 2 → (0,1), (1,0), (4,-3).  At t=3 → (9, -8).
        let h = hist(&[&[4.0, -3.0], &[1.0, 0.0], &[0.0, 1.0]]);
        let dt = dt_hist(&[1.0, 1.0, 1.0]);
        let pred = lagrange_predict_from_history(&h, &dt, 3, 2);
        assert!((pred[0] - 9.0).abs() < 1e-10);
        assert!((pred[1] + 8.0).abs() < 1e-10);
    }

    #[test]
    fn bdf3_exact_on_cubic_variable_step() {
        // BDF3 has order 3: must integrate cubic x(t) = t³ exactly.
        // dt_history: h_n, h_{n-1}, h_{n-2} = 0.5, 1.0, 1.5
        let h = dt_hist(&[0.5, 1.0, 1.5]);
        let (beta, k) = compute_bdf_coefficients(3, &h);
        // t_{n-3} = 0, t_{n-2} = 1.5, t_{n-1} = 2.5, t_n = 3.0
        let t_n3 = 0.0;
        let t_n2 = t_n3 + h[2];
        let t_n1 = t_n2 + h[1];
        let t_n = t_n1 + h[0];
        let f_n = 3.0 * t_n.powi(2);
        let x_n_predicted = beta * h[0] * f_n
            + k[0] * t_n1.powi(3)
            + k[1] * t_n2.powi(3)
            + k[2] * t_n3.powi(3);
        let x_n = t_n.powi(3);
        assert!((x_n_predicted - x_n).abs() < 1e-10,
            "BDF3 cubic test: pred={}, exact={}", x_n_predicted, x_n);
    }
}
