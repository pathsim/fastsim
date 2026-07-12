// Index-based loops mirror the linear-algebra math (and index parallel
// buffers), matching the crate's convention.
#![allow(clippy::needless_range_loop)]

// Micro-benchmark: Anderson's LS solve (faer LU) vs a custom mini-Cholesky.
//
// The LS problem in the Anderson step is small and SPD after Tikhonov reg:
//   (dR * dRᵀ + reg·I) · c = dR · res    with dim buf_len ≤ 4 typically.
//
// faer::Mat allocates on construction; for a 4×4 solve that overhead may
// dominate. This bench measures whether a stack-only Cholesky pays off.

use std::collections::VecDeque;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use smallvec::SmallVec;

type AVec = SmallVec<[f64; 8]>;

// === CURRENT IMPLEMENTATION (faer-based, copied from src/optim/anderson.rs) ===

fn lstsq_faer(
    dr_buffer: &VecDeque<AVec>,
    res: &[f64],
    buf_len: usize,
    n: usize,
    c_out: &mut [f64],
) {
    use faer::Mat;
    use faer::prelude::*;

    let mut ata = Mat::<f64>::zeros(buf_len, buf_len);
    for i in 0..buf_len {
        for j in i..buf_len {
            let dot: f64 = (0..n).map(|k| dr_buffer[i][k] * dr_buffer[j][k]).sum();
            ata[(i, j)] = dot;
            ata[(j, i)] = dot;
        }
    }
    let trace: f64 = (0..buf_len).map(|i| ata[(i, i)]).sum();
    let reg = (1e-10 * trace / buf_len as f64).max(1e-12);
    for i in 0..buf_len {
        ata[(i, i)] += reg;
    }
    let mut atb = Mat::<f64>::zeros(buf_len, 1);
    for i in 0..buf_len {
        atb[(i, 0)] = (0..n).map(|k| dr_buffer[i][k] * res[k]).sum();
    }
    let lu = ata.partial_piv_lu();
    let c_mat = lu.solve(&atb);
    for i in 0..buf_len {
        c_out[i] = c_mat[(i, 0)];
    }
}

// === CANDIDATE: stack-only Cholesky on a flat [f64; M*M] buffer ===
// AᵀA + reg·I is SPD, so Cholesky (no pivoting) is exact and fastest for small m.
//   A = L Lᵀ, solve L y = b, then Lᵀ x = y.

fn lstsq_chol(
    dr_buffer: &VecDeque<AVec>,
    res: &[f64],
    buf_len: usize,
    n: usize,
    c_out: &mut [f64],
) {
    const MMAX: usize = 8;
    debug_assert!(buf_len <= MMAX);

    // Build AᵀA (symmetric, packed into flat row-major)
    let mut a = [0.0_f64; MMAX * MMAX];
    let mut b = [0.0_f64; MMAX];

    for i in 0..buf_len {
        for j in i..buf_len {
            let mut dot = 0.0;
            for k in 0..n {
                dot += dr_buffer[i][k] * dr_buffer[j][k];
            }
            a[i * MMAX + j] = dot;
            a[j * MMAX + i] = dot;
        }
        let mut dot = 0.0;
        for k in 0..n {
            dot += dr_buffer[i][k] * res[k];
        }
        b[i] = dot;
    }
    let mut trace = 0.0;
    for i in 0..buf_len {
        trace += a[i * MMAX + i];
    }
    let reg = (1e-10 * trace / buf_len as f64).max(1e-12);
    for i in 0..buf_len {
        a[i * MMAX + i] += reg;
    }

    // In-place Cholesky: overwrite lower triangle with L
    for i in 0..buf_len {
        let mut sum = a[i * MMAX + i];
        for k in 0..i {
            let v = a[i * MMAX + k];
            sum -= v * v;
        }
        let diag = sum.sqrt();
        a[i * MMAX + i] = diag;
        let inv_diag = 1.0 / diag;
        for j in (i + 1)..buf_len {
            let mut s = a[j * MMAX + i];
            for k in 0..i {
                s -= a[j * MMAX + k] * a[i * MMAX + k];
            }
            a[j * MMAX + i] = s * inv_diag;
        }
    }

    // Forward solve: L y = b, store y in b
    for i in 0..buf_len {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * MMAX + k] * b[k];
        }
        b[i] = s / a[i * MMAX + i];
    }
    // Back solve: Lᵀ x = y
    for i in (0..buf_len).rev() {
        let mut s = b[i];
        for k in (i + 1)..buf_len {
            s -= a[k * MMAX + i] * c_out[k];
        }
        c_out[i] = s / a[i * MMAX + i];
    }
}

// === test harness ===

fn make_problem(buf_len: usize, n: usize) -> (VecDeque<AVec>, AVec) {
    // Deterministic pseudo-data; not orthogonal so Tikhonov is exercised
    let mut dr = VecDeque::with_capacity(buf_len);
    for i in 0..buf_len {
        let mut v = AVec::with_capacity(n);
        for k in 0..n {
            let t = (i + 1) as f64 * 0.7 + (k + 1) as f64 * 0.3;
            v.push((t.sin() * 0.5 + t.cos() * 0.1) * 1e-3);
        }
        dr.push_back(v);
    }
    let mut res = AVec::with_capacity(n);
    for k in 0..n {
        res.push(((k as f64 + 1.0) * 0.11).sin() * 1e-5);
    }
    (dr, res)
}

fn bench_lstsq(c: &mut Criterion) {
    let mut group = c.benchmark_group("lstsq");
    for &(buf_len, n) in &[(2, 2), (4, 2), (4, 8), (4, 32), (8, 2), (8, 32)] {
        let (dr, res) = make_problem(buf_len, n);
        let tag = format!("m{}_n{}", buf_len, n);

        group.bench_function(BenchmarkId::new("faer", &tag), |b| {
            let mut out = vec![0.0; buf_len];
            b.iter(|| {
                lstsq_faer(black_box(&dr), black_box(&res), buf_len, n, &mut out);
                black_box(&out);
            });
        });
        group.bench_function(BenchmarkId::new("chol", &tag), |b| {
            let mut out = vec![0.0; buf_len];
            b.iter(|| {
                lstsq_chol(black_box(&dr), black_box(&res), buf_len, n, &mut out);
                black_box(&out);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_lstsq);
criterion_main!(benches);
