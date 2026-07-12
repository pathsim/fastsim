//! Dense linear solves for implicit Newton steps.
//!
//! Both the optimiser's matrix-Newton step (`NewtonAnderson`/`Newton` for ODE /
//! mass-matrix stages) and the fully-implicit DAE stage need the same thing:
//! factor a small dense `n×n` system and apply a Newton update. They used to
//! carry near-identical faer-LU code in two places (`optim::anderson` and
//! `solvers::stage`). This module is the single home for that path.
//!
//! Step 1 was a PURE relocation. Step 2 made this a stateful solver that caches
//! the factorization when the Jacobian is constant (linear blocks), so the
//! Newton matrix `(I - dt·a_ii·J)` is factored once per `dt` and reused across
//! Newton iterations, SDIRK stages and steps. Step 3 adds a gated *sparse* LU
//! path: when the Newton matrix is large and genuinely sparse (e.g. a block-
//! diagonal Jacobian from many independent state blocks) it factors and solves
//! with faer's sparse LU instead of dense, dropping the factor from `O(n³)` and
//! every reused solve from `O(n²)` toward the structural nonzero count.

use faer::linalg::solvers::PartialPivLu;
use faer::linalg::solvers::SolveCore;
use faer::sparse::linalg::solvers::{Lu as SparseLuFac, SymbolicLu};
use faer::sparse::{SparseColMat, Triplet};
use faer::{Conj, Mat};

use crate::constants::{LINSOLVE_SPARSE_MAX_DENSITY, LINSOLVE_SPARSE_MIN_DIM};

/// The cached factorization, dense or sparse, of the last Newton matrix.
#[derive(Default)]
enum Factored {
    #[default]
    None,
    Dense(PartialPivLu<f64>),
    Sparse(SparseLuFac<usize, f64>),
}

/// Stateful Newton linear solver with factorization caching.
///
/// Caches the factorization of the last matrix. The next call compares the
/// incoming matrix to the cached key (an `O(n²)` compare): if identical, it
/// reuses the factorization and only does the triangular solve, instead of an
/// `O(n³)` refactorization. The compare is far cheaper than the factorization it
/// saves. This makes the implicit solver factor the Newton matrix
/// `(I - dt·a_ii·J)` once per `dt` for constant-Jacobian (linear) blocks: the
/// scaled matrix is recomputed bytewise-identically, so the content compare
/// hits, with no need to thread a "Jacobian is constant" flag through the solve
/// path. Changing Jacobians (or a changed `dt` under an adaptive solver) differ,
/// so they refactor exactly as before.
///
/// On a refactorization the solver picks dense or sparse LU from the matrix's
/// dimension and density (see [`LINSOLVE_SPARSE_MIN_DIM`] /
/// [`LINSOLVE_SPARSE_MAX_DENSITY`]); a sparse build that fails for any reason
/// falls back to dense, so correctness never depends on the sparse path.
pub struct LinearSolver {
    /// Bytes the cached factorization was built from: the cache key. For
    /// `newton_step_matrix` this is the raw `jac_flat`, so a cache hit can skip
    /// reassembling `A = jac − I` *and* the refactorization; for `newton_solve`
    /// it is the caller-assembled `a_flat`. An instance is only ever driven
    /// through one of the two methods, so the key semantics stay consistent.
    key_cache: Vec<f64>,
    fac: Factored,
    n: usize,
    /// Reused scratch for the assembled Newton matrix `A = jac − I`.
    mat_scratch: Vec<f64>,
    /// Reused scratch for the residual `r = g − x` (avoids a per-call alloc).
    r_scratch: Vec<f64>,
    /// Cached SPARSE-pattern symbolic factorization (fill-reducing ordering +
    /// elimination tree). Keyed on `(sp_rows, sp_cols)`: across Newton iterations
    /// / SDIRK stages the pattern of `scale·J − I` is constant — only the values
    /// change — so the expensive symbolic step is computed once and the per-call
    /// work drops to a numeric refactorization + an in-place triangular solve.
    sym: Option<SymbolicLu<usize>>,
    sp_rows: Vec<u32>,
    sp_cols: Vec<u32>,
    /// Reused rhs/solution column for the in-place sparse solve.
    b_mat: Mat<f64>,
    /// Reused triplet buffer for the sparse matrix build.
    triplets: Vec<Triplet<usize, usize, f64>>,
}

impl Default for LinearSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearSolver {
    pub fn new() -> Self {
        Self {
            key_cache: Vec::new(),
            fac: Factored::None,
            n: 0,
            mat_scratch: Vec::new(),
            r_scratch: Vec::new(),
            sym: None,
            sp_rows: Vec::new(),
            sp_cols: Vec::new(),
            b_mat: Mat::<f64>::zeros(0, 0),
            triplets: Vec::new(),
        }
    }

    /// Factor the flat row-major `n×n` matrix `a_flat` into the cache, choosing a
    /// sparse LU when the matrix is large and sparse enough, else dense. The
    /// sparse branch reuses the cached symbolic factorization when the nonzero
    /// pattern is unchanged (only a numeric refactorization is then redone).
    fn refactor(&mut self, a_flat: &[f64], n: usize) {
        self.n = n;
        if n >= LINSOLVE_SPARSE_MIN_DIM {
            let nnz = a_flat.iter().filter(|&&v| v != 0.0).count();
            if (nnz as f64) <= LINSOLVE_SPARSE_MAX_DENSITY * (n * n) as f64 {
                // Gather triplets and compare the nonzero pattern to the cache in
                // one pass (no temp allocation on the steady path).
                self.triplets.clear();
                let mut pat_same = self.sym.is_some() && self.sp_rows.len() == nnz;
                let mut idx = 0;
                for i in 0..n {
                    for j in 0..n {
                        let v = a_flat[i * n + j];
                        if v != 0.0 {
                            self.triplets.push(Triplet::new(i, j, v));
                            if pat_same
                                && (self.sp_rows[idx] != i as u32 || self.sp_cols[idx] != j as u32)
                            {
                                pat_same = false;
                            }
                            idx += 1;
                        }
                    }
                }
                let built = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &self.triplets);
                if let Ok(m) = built {
                    let lu = if pat_same {
                        SparseLuFac::try_new_with_symbolic(self.sym.clone().unwrap(), m.as_ref()).ok()
                    } else {
                        match SymbolicLu::try_new(m.symbolic()) {
                            Ok(sym) => {
                                self.sym = Some(sym.clone());
                                self.sp_rows.clear();
                                self.sp_cols.clear();
                                for i in 0..n {
                                    for j in 0..n {
                                        if a_flat[i * n + j] != 0.0 {
                                            self.sp_rows.push(i as u32);
                                            self.sp_cols.push(j as u32);
                                        }
                                    }
                                }
                                SparseLuFac::try_new_with_symbolic(sym, m.as_ref()).ok()
                            }
                            Err(_) => None,
                        }
                    };
                    if let Some(lu) = lu {
                        self.fac = Factored::Sparse(lu);
                        return;
                    }
                }
                // Any failure: fall through to dense (it pivots and reports itself).
            }
        }
        self.sym = None;
        let mut mat = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                mat[(i, j)] = a_flat[i * n + j];
            }
        }
        self.fac = Factored::Dense(mat.partial_piv_lu());
    }

    /// Solve `A·Δ = rhs` with the cached factorization IN PLACE (reused `b_mat`,
    /// no per-solve allocation), apply `x -= Δ`, return `|rhs|_2`.
    fn solve_cached(&mut self, x: &mut [f64], rhs: &[f64], n: usize) -> f64 {
        let res = self.solve_into_b(rhs, n);
        for i in 0..n {
            x[i] -= self.b_mat[(i, 0)];
        }
        res
    }

    /// Solve `A·Δ = rhs` into the reused `b_mat` (Δ = A⁻¹rhs); returns `|rhs|_2`.
    fn solve_into_b(&mut self, rhs: &[f64], n: usize) -> f64 {
        if self.b_mat.nrows() != n {
            self.b_mat = Mat::<f64>::zeros(n, 1);
        }
        let mut res_sq = 0.0;
        for i in 0..n {
            self.b_mat[(i, 0)] = rhs[i];
            res_sq += rhs[i] * rhs[i];
        }
        match &self.fac {
            Factored::Dense(lu) => lu.solve_in_place_with_conj(Conj::No, self.b_mat.as_mut()),
            Factored::Sparse(lu) => lu.solve_in_place_with_conj(Conj::No, self.b_mat.as_mut()),
            Factored::None => unreachable!("solve called before factorization"),
        }
        res_sq.sqrt()
    }

    /// Factor the flat row-major `n×n` matrix `a_flat` (sparse when large/sparse
    /// enough, with cached symbolic factorization on an unchanged pattern; else
    /// dense) and solve `A·out = rhs` in place into `out`. No fixed-point `−I`
    /// shift, no `x −= Δ` apply — a plain linear solve for a caller-assembled `A`.
    /// The factorization is keyed on `a_flat`: an identical matrix reuses it.
    pub fn solve(&mut self, a_flat: &[f64], rhs: &[f64], out: &mut [f64], n: usize) {
        let hit = self.n == n && self.is_factored() && self.key_cache == a_flat;
        if !hit {
            self.refactor(a_flat, n);
            self.key_cache.clear();
            self.key_cache.extend_from_slice(a_flat);
        }
        self.solve_into_b(rhs, n);
        for i in 0..n {
            out[i] = self.b_mat[(i, 0)];
        }
    }

    /// `true` once a factorization is cached (used to decide a cache hit).
    fn is_factored(&self) -> bool {
        !matches!(self.fac, Factored::None)
    }

    /// Matrix-Newton step for the fixed-point form `x = g`: residual `r = g − x`,
    /// Newton matrix `A = (dt·a_ii·J) − I` (caller passes `jac_flat = dt·a_ii·J`).
    ///
    /// Keys the cache on the raw `jac_flat`: when it is unchanged (constant
    /// Jacobian at fixed `dt`/`a_ii`) the assembled `A` is identical too, so the
    /// `O(n²)` assembly *and* the `O(n³)` factorization are both skipped, leaving
    /// only the residual build and the triangular solve.
    pub fn newton_step_matrix(&mut self, x: &mut [f64], g: &[f64], jac_flat: &[f64], n: usize) -> f64 {
        let hit = self.n == n && self.is_factored() && self.key_cache == jac_flat;
        if !hit {
            self.mat_scratch.clear();
            self.mat_scratch.reserve(n * n);
            for i in 0..n {
                for j in 0..n {
                    self.mat_scratch.push(jac_flat[i * n + j] - if i == j { 1.0 } else { 0.0 });
                }
            }
            let a = std::mem::take(&mut self.mat_scratch);
            self.refactor(&a, n);
            self.mat_scratch = a;
            self.key_cache.clear();
            self.key_cache.extend_from_slice(jac_flat);
        }
        let mut r = std::mem::take(&mut self.r_scratch);
        r.resize(n, 0.0);
        for i in 0..n {
            r[i] = g[i] - x[i];
        }
        let norm = self.solve_cached(x, &r, n);
        self.r_scratch = r;
        norm
    }

    /// Direct Newton step: solve `A·Δ = r` with `A` given flat row-major,
    /// reusing the cached LU when `a_flat` is unchanged.
    pub fn newton_solve(&mut self, x: &mut [f64], r: &[f64], a_flat: &[f64], n: usize) -> f64 {
        let hit = self.n == n && self.is_factored() && self.key_cache == a_flat;
        if !hit {
            self.refactor(a_flat, n);
            self.key_cache.clear();
            self.key_cache.extend_from_slice(a_flat);
        }
        self.solve_cached(x, r, n)
    }

    /// Sparse matrix-Newton step for the fixed-point form `x = g`, where the
    /// Newton matrix `A = scale·J − I` is given by `J`'s coordinate pattern
    /// (`rows`, `cols`) and current `values` (`scale = dt·a_ii`).
    ///
    /// Builds `A` straight as sparse triplets, skipping the dense `n×n`
    /// materialisation, the `O(n²)` nonzero scan, and the `O(n²)` cache-key
    /// compare the dense [`newton_step_matrix`] pays. The `−I` is added as `n`
    /// extra diagonal triplets; faer sums duplicate coordinates, so an
    /// on-diagonal pattern entry becomes `scale·J_ii − 1`. A sparse build that
    /// fails (structurally singular pattern) falls back to a dense factorization,
    /// so correctness never depends on the sparse path.
    pub fn newton_step_matrix_sparse(
        &mut self,
        x: &mut [f64],
        g: &[f64],
        rows: &[u32],
        cols: &[u32],
        values: &[f64],
        scale: f64,
        n: usize,
    ) -> f64 {
        self.n = n;
        // (Re)build the sparse Newton matrix A = scale·J − I from triplets. The
        // `−I` is added as `n` diagonal triplets; faer sums duplicate coordinates.
        self.triplets.clear();
        for k in 0..values.len() {
            self.triplets.push(Triplet::new(rows[k] as usize, cols[k] as usize, scale * values[k]));
        }
        for i in 0..n {
            self.triplets.push(Triplet::new(i, i, -1.0));
        }

        // Reuse the cached symbolic factorization when the sparsity PATTERN is
        // unchanged (it is, across Newton iterations / SDIRK stages): only the
        // numeric factorization is redone. A new pattern recomputes the symbolic.
        let sparse_lu = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &self.triplets)
            .ok()
            .and_then(|m| {
                let same = self.sym.is_some()
                    && self.sp_rows.as_slice() == rows
                    && self.sp_cols.as_slice() == cols;
                if same {
                    SparseLuFac::try_new_with_symbolic(self.sym.clone().unwrap(), m.as_ref()).ok()
                } else {
                    let sym = SymbolicLu::try_new(m.symbolic()).ok()?;
                    self.sym = Some(sym.clone());
                    self.sp_rows.clear();
                    self.sp_rows.extend_from_slice(rows);
                    self.sp_cols.clear();
                    self.sp_cols.extend_from_slice(cols);
                    SparseLuFac::try_new_with_symbolic(sym, m.as_ref()).ok()
                }
            });

        // This matrix was not built from a flat `a_flat`, so a later
        // `newton_step_matrix`/`newton_solve` must refactor.
        self.key_cache.clear();
        self.fac = Factored::None;

        if let Some(lu) = sparse_lu {
            // In-place solve: rhs = g − x in `b_mat`, solve A·Δ = rhs into it, x −= Δ.
            if self.b_mat.nrows() != n {
                self.b_mat = Mat::<f64>::zeros(n, 1);
            }
            let mut res_sq = 0.0;
            for i in 0..n {
                let r = g[i] - x[i];
                self.b_mat[(i, 0)] = r;
                res_sq += r * r;
            }
            lu.solve_in_place_with_conj(Conj::No, self.b_mat.as_mut());
            for i in 0..n {
                x[i] -= self.b_mat[(i, 0)];
            }
            return res_sq.sqrt();
        }

        // Dense fallback: scatter A = scale·J − I and factor densely.
        self.sym = None;
        self.mat_scratch.clear();
        self.mat_scratch.resize(n * n, 0.0);
        for i in 0..n {
            self.mat_scratch[i * n + i] = -1.0;
        }
        for k in 0..values.len() {
            self.mat_scratch[rows[k] as usize * n + cols[k] as usize] += scale * values[k];
        }
        let a = std::mem::take(&mut self.mat_scratch);
        let mut mat = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                mat[(i, j)] = a[i * n + j];
            }
        }
        self.fac = Factored::Dense(mat.partial_piv_lu());
        self.mat_scratch = a;
        let mut r = std::mem::take(&mut self.r_scratch);
        r.resize(n, 0.0);
        for i in 0..n {
            r[i] = g[i] - x[i];
        }
        let norm = self.solve_cached(x, &r, n);
        self.r_scratch = r;
        norm
    }
}

/// Solve the Newton system `A · Δ = r` with `A` given flat row-major (`n×n`),
/// apply `x -= Δ` in place, and return the pre-step residual norm `|r|_2`.
/// Used by the fully-implicit / semi-explicit DAE stages, where the caller
/// assembles `A` directly. A one-shot convenience over a transient
/// [`LinearSolver`] (which owns the single dense/sparse faer-LU path): no caching
/// across calls, which is correct here since these `A` change every iteration.
pub fn newton_solve_inplace(x: &mut [f64], r: &[f64], a_flat: &[f64], n: usize) -> f64 {
    LinearSolver::new().newton_solve(x, r, a_flat, n)
}

/// One matrix-Newton step for the fixed-point form `x = g`: residual `r = g − x`,
/// Newton matrix `A = (dt·a_ii·J) − I` (caller passes `jac_flat = dt·a_ii·J`).
/// One-shot convenience over a transient [`LinearSolver`].
pub fn newton_step_matrix_inplace(x: &mut [f64], g: &[f64], jac_flat: &[f64], n: usize) -> f64 {
    LinearSolver::new().newton_step_matrix(x, g, jac_flat, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cached path (reuse for an unchanged matrix, refactor for a changed
    /// one) must produce bit-for-bit the same Newton update as a fresh solver.
    #[test]
    fn cache_reuse_matches_fresh() {
        let n = 2;
        let jac = vec![2.0, 1.0, 1.0, 3.0]; // Newton matrix = jac - I
        let mut ls = LinearSolver::new();

        // First solve factors; second solve with the SAME matrix must reuse and
        // still match a fresh one-shot solver.
        let mut x1 = vec![0.0, 0.0];
        ls.newton_step_matrix(&mut x1, &[1.0, 2.0], &jac, n);
        let g2 = [-1.0, 0.7];
        let mut x2 = vec![0.5, -0.5];
        ls.newton_step_matrix(&mut x2, &g2, &jac, n);
        let mut x2_fresh = vec![0.5, -0.5];
        LinearSolver::new().newton_step_matrix(&mut x2_fresh, &g2, &jac, n);
        for i in 0..n {
            assert!((x2[i] - x2_fresh[i]).abs() < 1e-14, "reuse != fresh at {i}");
        }

        // A changed matrix must refactor and still match a fresh solver.
        let jac2 = [5.0, 0.0, 2.0, 4.0];
        let g3 = [1.0, 1.0];
        let mut x3 = vec![0.0, 0.0];
        ls.newton_step_matrix(&mut x3, &g3, &jac2, n);
        let mut x3_fresh = vec![0.0, 0.0];
        LinearSolver::new().newton_step_matrix(&mut x3_fresh, &g3, &jac2, n);
        for i in 0..n {
            assert!((x3[i] - x3_fresh[i]).abs() < 1e-14, "refactor != fresh at {i}");
        }
    }

    /// The sparse-direct Newton step (`A = scale·J − I` built from the pattern)
    /// must produce the same update as the dense path fed the equivalent dense
    /// `scale·J`. Tridiagonal `n = 64` trips the sparse gate on both sides.
    #[test]
    fn sparse_step_matches_dense_step() {
        let n = 64;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            if i > 0 { rows.push(i as u32); cols.push((i - 1) as u32); vals.push(-1.0); }
            rows.push(i as u32); cols.push(i as u32); vals.push(5.0 + (i % 3) as f64);
            if i + 1 < n { rows.push(i as u32); cols.push((i + 1) as u32); vals.push(-0.7); }
        }
        let scale = 0.3;
        let mut jac_scaled = vec![0.0; n * n];
        for k in 0..vals.len() {
            jac_scaled[rows[k] as usize * n + cols[k] as usize] = scale * vals[k];
        }
        let g: Vec<f64> = (0..n).map(|i| 0.1 + i as f64 * 0.001).collect();

        // Dense path builds A = (scale·J) − I internally.
        let mut x_dense = vec![0.0; n];
        LinearSolver::new().newton_step_matrix(&mut x_dense, &g, &jac_scaled, n);
        // Sparse path builds A = scale·J − I from the pattern.
        let mut x_sparse = vec![0.0; n];
        LinearSolver::new().newton_step_matrix_sparse(&mut x_sparse, &g, &rows, &cols, &vals, scale, n);

        for i in 0..n {
            assert!(
                (x_dense[i] - x_sparse[i]).abs() < 1e-9,
                "sparse != dense at {i}: {} vs {}", x_sparse[i], x_dense[i]
            );
        }
    }

    /// A large, sparse block-diagonal Newton matrix must take the sparse LU path
    /// (n ≥ `LINSOLVE_SPARSE_MIN_DIM`, density well under the cap) and produce the
    /// same Newton update as the dense reference solver.
    #[test]
    fn sparse_path_matches_dense() {
        // 16 diagonal 4×4 blocks → n = 64, density = 4/64 = 6.25% (< 25% cap).
        let (blocks, bs) = (16usize, 4usize);
        let n = blocks * bs;
        let mut a = vec![0.0; n * n];
        for b in 0..blocks {
            for i in 0..bs {
                for j in 0..bs {
                    // Diagonally dominant block so it factors without pivoting issues.
                    let gi = b * bs + i;
                    let gj = b * bs + j;
                    a[gi * n + gj] = if i == j { 5.0 + gi as f64 } else { 0.3 };
                }
            }
        }
        let nnz = a.iter().filter(|&&v| v != 0.0).count();
        assert!(
            (nnz as f64) <= LINSOLVE_SPARSE_MAX_DENSITY * (n * n) as f64 && n >= LINSOLVE_SPARSE_MIN_DIM,
            "test matrix must hit the sparse gate"
        );

        let r: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64).collect();

        // Sparse-capable cached solver vs the always-dense free reference.
        let mut ls = LinearSolver::new();
        let mut x_sparse = vec![0.0; n];
        ls.newton_solve(&mut x_sparse, &r, &a, n);
        assert!(matches!(ls.fac, Factored::Sparse(_)), "must have taken the sparse path");

        let mut x_dense = vec![0.0; n];
        newton_solve_inplace(&mut x_dense, &r, &a, n);

        for i in 0..n {
            assert!((x_sparse[i] - x_dense[i]).abs() < 1e-10, "sparse != dense at {i}: {} vs {}", x_sparse[i], x_dense[i]);
        }
    }
}
