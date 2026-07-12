// Factorization-cache benchmark for the implicit-solver Newton linear solve
// (J2/J3). Compares the cached `LinearSolver` (constant Jacobian at fixed
// dt/a_ii -> matrix unchanged -> reuse the LU, triangular solve only) against
// refactoring every call (a fresh solver each iteration = the pre-cache
// behaviour). Both arms allocate the same per-call scratch, so the delta is
// exactly the saved O(n^3) factorization.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use fastsim::optim::linsolve::LinearSolver;

/// Deterministic, diagonally-dominant Jacobian (LCG, no rand). The Newton matrix
/// `A = jac - I` is then well-conditioned and dense.
fn dense_jac(n: usize) -> Vec<f64> {
    let mut s: u64 = 0x1234_5678_9abc_def1;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
    };
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = if i == j { 5.0 } else { 0.1 * next() };
        }
    }
    a
}

/// Tridiagonal Jacobian of dim `n`: diagonal plus sub/super-diagonal. Density
/// `3/n`, so for `n >= LINSOLVE_SPARSE_MIN_DIM` it trips the sparse-LU gate. This
/// is the shape a sparse AD Jacobian has for a 1-D coupled chain (reaction
/// diffusion / nearest-neighbour stencil).
fn tridiag_jac(n: usize) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        a[i * n + i] = 5.0 + (i % 3) as f64;
        if i > 0 { a[i * n + (i - 1)] = -1.0; }
        if i + 1 < n { a[i * n + (i + 1)] = -0.7; }
    }
    a
}

/// The same tridiagonal matrix as [`tridiag_jac`] in coordinate (pattern) form:
/// `(rows, cols, values)` over the structural nonzeros only.
fn tridiag_pattern(n: usize) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        if i > 0 { rows.push(i as u32); cols.push((i - 1) as u32); vals.push(-1.0); }
        rows.push(i as u32); cols.push(i as u32); vals.push(5.0 + (i % 3) as f64);
        if i + 1 < n { rows.push(i as u32); cols.push((i + 1) as u32); vals.push(-0.7); }
    }
    (rows, cols, vals)
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("newton_linsolve");
    for &n in &[16usize, 32, 64] {
        let jac = dense_jac(n);
        let g: Vec<f64> = (0..n).map(|i| 0.1 + i as f64 * 0.01).collect();

        // Cached: one solver reused across iterations -> the matrix is unchanged,
        // so every call after the first reuses the factorization.
        group.bench_function(format!("cached_n{n}"), |b| {
            let mut ls = LinearSolver::new();
            let mut warm = vec![0.0; n];
            ls.newton_step_matrix(&mut warm, &g, &jac, n); // prime the cache
            b.iter(|| {
                let mut x = vec![0.0; n];
                black_box(ls.newton_step_matrix(&mut x, black_box(&g), black_box(&jac), n));
            });
        });

        // Refactor: a fresh solver each iteration -> always factors (pre-cache).
        group.bench_function(format!("refactor_n{n}"), |b| {
            b.iter(|| {
                let mut ls = LinearSolver::new();
                let mut x = vec![0.0; n];
                black_box(ls.newton_step_matrix(&mut x, black_box(&g), black_box(&jac), n));
            });
        });
    }
    group.finish();

    // Sparse (tridiagonal) Newton matrix, refactored every call: the per-step
    // cost a *nonlinear* stiff system pays (Jacobian changes every step -> cache
    // miss -> full refactor). Today this still materialises the dense `A =
    // jac - I` and scans all n^2 entries for nonzeros before the sparse factor;
    // that O(n^2) overhead is what the sparse-AD-Jacobian path (SAJ-4) removes.
    let mut sgroup = c.benchmark_group("newton_linsolve_sparse");
    for &n in &[64usize, 128, 256] {
        let jac = tridiag_jac(n);
        let g: Vec<f64> = (0..n).map(|i| 0.1 + i as f64 * 0.001).collect();
        sgroup.bench_function(format!("refactor_tridiag_n{n}"), |b| {
            b.iter(|| {
                let mut ls = LinearSolver::new();
                let mut x = vec![0.0; n];
                black_box(ls.newton_step_matrix(&mut x, black_box(&g), black_box(&jac), n));
            });
        });

        // Sparse-direct path (SAJ-4): same matrix `A = J - I`, but `A` is built
        // straight from the coordinate pattern with no dense materialisation or
        // nonzero scan. Compared against `refactor_tridiag` (identical A, identical
        // refactor-every-call setup) this isolates the saved O(n^2) overhead.
        let (rows, cols, vals) = tridiag_pattern(n);
        sgroup.bench_function(format!("sparse_direct_tridiag_n{n}"), |b| {
            b.iter(|| {
                let mut ls = LinearSolver::new();
                let mut x = vec![0.0; n];
                black_box(ls.newton_step_matrix_sparse(
                    &mut x, black_box(&g), black_box(&rows), black_box(&cols),
                    black_box(&vals), 1.0, n,
                ));
            });
        });
    }
    sgroup.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
