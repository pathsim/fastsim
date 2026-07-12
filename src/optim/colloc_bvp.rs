//! Collocation boundary-value-problem solver — a native rebuild of
//! `scipy.solve_bvp` (Kierzenka–Shampine) with three extensions: free
//! **parameters** `p`, **interior point conditions** (multipoint BVP), and an
//! **allocation-free hot path** (all scratch lives in a reused [`BvpWorkspace`]).
//!
//! 4th-order Lobatto-IIIa (Simpson) collocation: on each interval `[x_i,x_{i+1}]`
//! the solution is the C¹ Hermite cubic through `(y_i,f_i)`/`(y_{i+1},f_{i+1})`
//! and the collocation residual is `y_{i+1}−y_i − h/6 (f_i+4 f_mid+f_{i+1})`. The
//! coupled nonlinear system
//!
//! ```text
//!   collocation(y, p) = 0          (m·n equations)
//!   bc(y_a, y_b, p)   = 0          (n_bc equations)
//!   icond(y@ports, p) = 0          (n_ic equations)   n_bc + n_ic = n + k
//! ```
//!
//! is solved by a damped Newton with the EXACT Jacobian assembled from AD blocks
//! (`∂f/∂y`, `∂f/∂p`, `∂bc/∂·`, `∂icond/∂·`). The unknown vector is
//! `z = [y (node-major, (m+1)·n) ; p (k)]`. Interior conditions evaluate `y` at
//! arbitrary `x_ports` through the Hermite interpolant, so each such row couples
//! to its bracketing nodes (via the node Jacobians) and to `p`.
//!
//! The linear solve uses the shared persistent [`LinearSolver`]: it auto-selects
//! a sparse LU for the large block-banded collocation Jacobian, reuses the cached
//! symbolic factorization across Newton iterations / refinement rounds, and
//! solves in place over reused buffers — no per-iteration allocation.

use crate::optim::linsolve::LinearSolver;

/// Closures the solver needs; all WRITE into a caller-provided buffer (cleared
/// first) so the hot path never allocates. `(x, y, p, out)` / `(ya, yb, p, out)`
/// / `(y_ports, p, out)`.
type FFn<'a> = dyn Fn(f64, &[f64], &[f64], &mut Vec<f64>) + 'a;
type BFn<'a> = dyn Fn(&[f64], &[f64], &[f64], &mut Vec<f64>) + 'a;
type IFn<'a> = dyn Fn(&[f64], &[f64], &mut Vec<f64>) + 'a;

pub struct BvpFns<'a> {
    pub n_eq: usize,
    pub n_params: usize,
    pub n_bc: usize,
    /// Interior-condition port locations (empty for a plain two-point BVP).
    pub x_ports: &'a [f64],
    pub n_ic: usize,
    pub fun: &'a FFn<'a>,
    pub jac_fy: &'a FFn<'a>, // ∂f/∂y, n×n
    pub jac_fp: &'a FFn<'a>, // ∂f/∂p, n×k
    pub bc: &'a BFn<'a>,
    pub jac_bc_ya: &'a BFn<'a>, // n_bc×n
    pub jac_bc_yb: &'a BFn<'a>, // n_bc×n
    pub jac_bc_p: &'a BFn<'a>,  // n_bc×k
    pub icond: &'a IFn<'a>,
    pub jac_ic_y: &'a IFn<'a>, // n_ic×(n_ports·n)
    pub jac_ic_p: &'a IFn<'a>, // n_ic×k
}

/// A converged (or best-effort) BVP solution on its adapted mesh.
pub struct BvpSolution {
    pub x: Vec<f64>,
    /// Solution, node-major: `y[i*n_eq + k]`.
    pub y: Vec<f64>,
    /// RHS values at the nodes (Hermite slopes), node-major.
    pub f: Vec<f64>,
    /// Converged parameter values.
    pub p: Vec<f64>,
    /// Whether the residual-based refinement reached the tolerance.
    pub converged: bool,
}

/// Reused scratch for an allocation-free hot path (held by the block across
/// evaluations; buffers only grow on mesh refinement).
#[derive(Default)]
pub struct BvpWorkspace {
    fval: Vec<f64>,
    residual: Vec<f64>,
    jac: Vec<f64>, // dense ndof×ndof, row-major
    rhs: Vec<f64>,
    prev: Vec<f64>,
    yt: Vec<f64>,
    // small per-call buffers
    b_n: Vec<f64>,
    b_nn: Vec<f64>,
    b_nk: Vec<f64>,
    b_misc: Vec<f64>,
    y_mid: Vec<f64>,
    f_mid: Vec<f64>,
    ji: Vec<f64>,
    jj: Vec<f64>,
    jm: Vec<f64>,
    jpi: Vec<f64>,
    jpj: Vec<f64>,
    jmp: Vec<f64>,
    y_ports: Vec<f64>,
    b_nn2: Vec<f64>,
    ic_buf: Vec<f64>,
    hres: Vec<f64>,
    hres_y: Vec<f64>,
    hres_s: Vec<f64>,
    // Shared persistent linear solver: the collocation Jacobian is large and
    // block-banded, so `solve` auto-selects sparse LU and reuses the cached
    // symbolic factorization across Newton iterations / mesh-refinement rounds.
    lin: LinearSolver,
}

#[inline]
fn nd(v: &[f64], i: usize, n: usize) -> &[f64] {
    &v[i * n..i * n + n]
}


/// Locate the interval containing `xq` and its Hermite weights.
fn hermite_weights(x: &[f64], xq: f64) -> (usize, f64, f64, f64, f64, f64) {
    let mut j = 0;
    while j + 1 < x.len() - 1 && x[j + 1] < xq {
        j += 1;
    }
    let h = x[j + 1] - x[j];
    let t = if h > 0.0 { (xq - x[j]) / h } else { 0.0 };
    let (t2, t3) = (t * t, t * t * t);
    (j, h, 1.0 - 3.0 * t2 + 2.0 * t3, 3.0 * t2 - 2.0 * t3, t - 2.0 * t2 + t3, -t2 + t3)
}

impl BvpWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

}

/// Evaluate all node RHS values into `fval` (length `(m+1)·n`). Free function so
/// the line search can pass disjoint `ws` field buffers (no aliasing on `&mut
/// self`), keeping the hot path allocation-free.
fn eval_f(f: &BvpFns, x: &[f64], y: &[f64], p: &[f64], n: usize, fval: &mut Vec<f64>, scratch: &mut Vec<f64>) {
    fval.clear();
    fval.resize(x.len() * n, 0.0);
    for i in 0..x.len() {
        (f.fun)(x[i], nd(y, i, n), p, scratch);
        fval[i * n..i * n + n].copy_from_slice(scratch);
    }
}

/// Full residual into `out` (length ndof). `fval` must already hold the node RHS.
#[allow(clippy::too_many_arguments)]
fn residual(
    f: &BvpFns, x: &[f64], y: &[f64], p: &[f64], fval: &[f64],
    out: &mut Vec<f64>, y_mid: &mut Vec<f64>, f_mid: &mut Vec<f64>,
    b_misc: &mut Vec<f64>, y_ports: &mut Vec<f64>,
) {
    let n = f.n_eq;
    let m = x.len() - 1;
    out.clear();
    for i in 0..m {
        let h = x[i + 1] - x[i];
        let yi = nd(y, i, n);
        let yj = nd(y, i + 1, n);
        let fi = nd(fval, i, n);
        let fj = nd(fval, i + 1, n);
        y_mid.clear();
        for k in 0..n {
            y_mid.push(0.5 * (yi[k] + yj[k]) - (h / 8.0) * (fj[k] - fi[k]));
        }
        (f.fun)(x[i] + 0.5 * h, y_mid, p, f_mid);
        for k in 0..n {
            out.push(yj[k] - yi[k] - (h / 6.0) * (fi[k] + 4.0 * f_mid[k] + fj[k]));
        }
    }
    (f.bc)(nd(y, 0, n), nd(y, m, n), p, b_misc);
    out.extend_from_slice(b_misc);
    if f.n_ic > 0 {
        y_ports.clear();
        for &xq in f.x_ports {
            let (j, h, h00, h01, h10, h11) = hermite_weights(x, xq);
            let yj = nd(y, j, n);
            let yj1 = nd(y, j + 1, n);
            let fj = nd(fval, j, n);
            let fj1 = nd(fval, j + 1, n);
            for k in 0..n {
                y_ports.push(h00 * yj[k] + h01 * yj1[k] + h * (h10 * fj[k] + h11 * fj1[k]));
            }
        }
        (f.icond)(y_ports, p, b_misc);
        out.extend_from_slice(b_misc);
    }
}

/// One damped-Newton solve on the current mesh (alloc-free via `ws`).
fn newton(
    x: &[f64],
    y: &mut [f64],
    p: &mut [f64],
    f: &BvpFns,
    ws: &mut BvpWorkspace,
    tol: f64,
    maxit: usize,
) -> bool {
    let n = f.n_eq;
    let k = f.n_params;
    let m = x.len() - 1;
    let ny = (m + 1) * n;
    let ndof = ny + k;

    for _ in 0..maxit {
        eval_f(f, x, y, p, n, &mut ws.fval, &mut ws.b_n);
        residual(f, x, y, p, &ws.fval, &mut ws.residual, &mut ws.y_mid, &mut ws.f_mid, &mut ws.b_misc, &mut ws.y_ports);
        let rnorm = ws.residual.iter().map(|v| v * v).sum::<f64>().sqrt();
        if rnorm < tol {
            return true;
        }

        // assemble dense Jacobian (row-major ndof×ndof) into ws.jac
        ws.jac.clear();
        ws.jac.resize(ndof * ndof, 0.0);
        let put = |jac: &mut [f64], r: usize, c: usize, v: f64| jac[r * ndof + c] = v;

        // --- collocation rows ---
        for i in 0..m {
            let h = x[i + 1] - x[i];
            let yi = nd(y, i, n);
            let yj = nd(y, i + 1, n);
            // y_mid + node/mid jacobians (read fval directly — disjoint from y_mid)
            ws.y_mid.clear();
            for q in 0..n {
                let fiq = ws.fval[i * n + q];
                let fjq = ws.fval[(i + 1) * n + q];
                ws.y_mid.push(0.5 * (yi[q] + yj[q]) - (h / 8.0) * (fjq - fiq));
            }
            (f.jac_fy)(x[i], yi, p, &mut ws.ji);
            (f.jac_fy)(x[i + 1], yj, p, &mut ws.jj);
            (f.jac_fy)(x[i] + 0.5 * h, &ws.y_mid, p, &mut ws.jm);
            // dy_mid/dy_i = I/2 + h/8 Ji ; dy_mid/dy_{i+1} = I/2 − h/8 Jj
            // dcol/dy_i = −I − h/6 (Ji + 4 Jm·dymid_dyi)
            for pr in 0..n {
                for c in 0..n {
                    let e = pr * n + c;
                    let id = if pr == c { 1.0 } else { 0.0 };
                    // Jm·dymid_dyi  and  Jm·dymid_dyj  (n×n each)
                    let mut jm_dl = 0.0;
                    let mut jm_dr = 0.0;
                    for t in 0..n {
                        let dl = 0.5 * if t == c { 1.0 } else { 0.0 } + (h / 8.0) * ws.ji[t * n + c];
                        let dr = 0.5 * if t == c { 1.0 } else { 0.0 } - (h / 8.0) * ws.jj[t * n + c];
                        jm_dl += ws.jm[pr * n + t] * dl;
                        jm_dr += ws.jm[pr * n + t] * dr;
                    }
                    let dl = -id - (h / 6.0) * (ws.ji[e] + 4.0 * jm_dl);
                    let dr = id - (h / 6.0) * (ws.jj[e] + 4.0 * jm_dr);
                    put(&mut ws.jac, i * n + pr, i * n + c, dl);
                    put(&mut ws.jac, i * n + pr, (i + 1) * n + c, dr);
                }
            }
            // parameter columns: dcol/dp = −h/6 (Jpi + 4 dfmid_dp + Jpj)
            if k > 0 {
                (f.jac_fp)(x[i], yi, p, &mut ws.jpi);
                (f.jac_fp)(x[i + 1], yj, p, &mut ws.jpj);
                (f.jac_fp)(x[i] + 0.5 * h, &ws.y_mid, p, &mut ws.jmp);
                for pr in 0..n {
                    for c in 0..k {
                        // dfmid_dp = Jmp + Jm·(−h/8 (Jpj − Jpi))
                        let mut jm_dp = 0.0;
                        for t in 0..n {
                            jm_dp += ws.jm[pr * n + t] * (-(h / 8.0) * (ws.jpj[t * k + c] - ws.jpi[t * k + c]));
                        }
                        let dfmid = ws.jmp[pr * k + c] + jm_dp;
                        let dp = -(h / 6.0) * (ws.jpi[pr * k + c] + 4.0 * dfmid + ws.jpj[pr * k + c]);
                        put(&mut ws.jac, i * n + pr, ny + c, dp);
                    }
                }
            }
        }

        // --- boundary-condition rows ---
        let base = m * n;
        (f.jac_bc_ya)(nd(y, 0, n), nd(y, m, n), p, &mut ws.b_nn);
        (f.jac_bc_yb)(nd(y, 0, n), nd(y, m, n), p, &mut ws.b_nn2);
        for pr in 0..f.n_bc {
            for c in 0..n {
                put(&mut ws.jac, base + pr, c, ws.b_nn[pr * n + c]);
                put(&mut ws.jac, base + pr, m * n + c, ws.b_nn2[pr * n + c]);
            }
        }
        if k > 0 {
            (f.jac_bc_p)(nd(y, 0, n), nd(y, m, n), p, &mut ws.b_nk);
            for pr in 0..f.n_bc {
                for c in 0..k {
                    put(&mut ws.jac, base + pr, ny + c, ws.b_nk[pr * k + c]);
                }
            }
        }

        // --- interior-condition rows ---
        if f.n_ic > 0 {
            // recompute y_ports + ∂icond/∂y_ports, ∂icond/∂p
            ws.y_ports.clear();
            for &xq in f.x_ports {
                let (j, h, h00, h01, h10, h11) = hermite_weights(x, xq);
                let yj = nd(y, j, n);
                let yj1 = nd(y, j + 1, n);
                let fj = nd(&ws.fval, j, n);
                let fj1 = nd(&ws.fval, j + 1, n);
                for q in 0..n {
                    ws.y_ports
                        .push(h00 * yj[q] + h01 * yj1[q] + h * (h10 * fj[q] + h11 * fj1[q]));
                }
            }
            (f.jac_ic_y)(&ws.y_ports, p, &mut ws.ic_buf); // n_ic × (n_ports·n)
            let np = f.x_ports.len();
            let ic_base = base + f.n_bc;
            for (pidx, &xq) in f.x_ports.iter().enumerate() {
                let (j, h, h00, h01, h10, h11) = hermite_weights(x, xq);
                // ∂y_port/∂y_j = h00 I + h h10 Jfy_j ; ∂/∂y_{j+1} = h01 I + h h11 Jfy_{j+1}
                (f.jac_fy)(x[j], nd(y, j, n), p, &mut ws.ji);
                (f.jac_fy)(x[j + 1], nd(y, j + 1, n), p, &mut ws.jj);
                for ridx in 0..f.n_ic {
                    for c in 0..n {
                        // sum over port components q: ic_y[ridx, pidx*n+q] * ∂y_port_q/∂y_node
                        let mut dj = 0.0;
                        let mut dj1 = 0.0;
                        for q in 0..n {
                            let w = ws.ic_buf[ridx * (np * n) + pidx * n + q];
                            let id = if q == c { 1.0 } else { 0.0 };
                            dj += w * (h00 * id + h * h10 * ws.ji[q * n + c]);
                            dj1 += w * (h01 * id + h * h11 * ws.jj[q * n + c]);
                        }
                        ws.jac[(ic_base + ridx) * ndof + j * n + c] += dj;
                        ws.jac[(ic_base + ridx) * ndof + (j + 1) * n + c] += dj1;
                    }
                }
            }
            if k > 0 {
                (f.jac_ic_p)(&ws.y_ports, p, &mut ws.b_nk);
                for ridx in 0..f.n_ic {
                    for c in 0..k {
                        put(&mut ws.jac, ic_base + ridx, ny + c, ws.b_nk[ridx * k + c]);
                    }
                }
            }
        }

        // solve J·dz = residual via the shared persistent LU (auto-sparse for the
        // block-banded collocation Jacobian, cached symbolic factorization). dz
        // lands in ws.rhs for the line search below; no apply here.
        ws.rhs.clear();
        ws.rhs.resize(ndof, 0.0);
        ws.lin.solve(&ws.jac, &ws.residual, &mut ws.rhs, ndof);
        if ws.rhs[..ndof].iter().any(|v| !v.is_finite()) {
            return false;
        }

        // backtracking line search on ‖residual‖ (alloc-free: trial point in ws.yt,
        // residual evaluated from disjoint ws field buffers)
        ws.prev.clear();
        ws.prev.extend_from_slice(y);
        ws.prev.extend_from_slice(p);
        let mut step = 1.0;
        loop {
            ws.yt.clear();
            for kk in 0..ndof {
                ws.yt.push(ws.prev[kk] - step * ws.rhs[kk]);
            }
            let (yt_y, yt_p) = ws.yt.split_at(ny);
            eval_f(f, x, yt_y, yt_p, n, &mut ws.fval, &mut ws.b_n);
            residual(f, x, yt_y, yt_p, &ws.fval, &mut ws.residual, &mut ws.y_mid, &mut ws.f_mid, &mut ws.b_misc, &mut ws.y_ports);
            let rt = ws.residual.iter().map(|v| v * v).sum::<f64>().sqrt();
            if rt < rnorm || step < 1.0 / 64.0 {
                y.copy_from_slice(yt_y);
                p.copy_from_slice(yt_p);
                break;
            }
            step *= 0.5;
        }
    }
    false
}

/// Per-interval Hermite-interpolant residual estimate (drives refinement).
#[allow(clippy::too_many_arguments)]
fn hermite_residual(
    x: &[f64], y: &[f64], fval: &[f64], n: usize, f: &BvpFns, p: &[f64],
    out: &mut Vec<f64>, yt: &mut Vec<f64>, scratch: &mut Vec<f64>,
) {
    let m = x.len() - 1;
    let s = 0.5 * (3.0_f64 / 7.0).sqrt();
    out.clear();
    out.resize(m, 0.0);
    let mut dyt = [0.0; 64]; // interior systems are small (n ≤ 64)
    for i in 0..m {
        let h = x[i + 1] - x[i];
        let yi = nd(y, i, n);
        let yj = nd(y, i + 1, n);
        let fi = nd(fval, i, n);
        let fj = nd(fval, i + 1, n);
        let mut acc = 0.0;
        let mut fscale = 0.0_f64;
        for &t in &[0.5 + s, 0.5 - s] {
            let (t2, t3) = (t * t, t * t * t);
            yt.clear();
            for k in 0..n {
                yt.push(yi[k] * (1.0 - 3.0 * t2 + 2.0 * t3) + yj[k] * (3.0 * t2 - 2.0 * t3)
                    + h * fi[k] * (t - 2.0 * t2 + t3) + h * fj[k] * (-t2 + t3));
                dyt[k] = (yi[k] * (-6.0 * t + 6.0 * t2) + yj[k] * (6.0 * t - 6.0 * t2)) / h
                    + fi[k] * (1.0 - 4.0 * t + 3.0 * t2) + fj[k] * (-2.0 * t + 3.0 * t2);
            }
            (f.fun)(x[i] + t * h, yt, p, scratch);
            for k in 0..n {
                let r = dyt[k] - scratch[k];
                acc += r * r;
            }
        }
        for k in 0..n {
            fscale = fscale.max(fi[k].abs()).max(fj[k].abs());
        }
        out[i] = (0.5 * acc).sqrt() / (1.0 + fscale) * h;
    }
}

fn refine(x: &[f64], res: &[f64], tol: f64) -> Vec<f64> {
    let mut xn = vec![x[0]];
    for i in 0..x.len() - 1 {
        if res[i] > tol {
            let k = if res[i] > 100.0 * tol { 2 } else { 1 };
            for j in 1..=k {
                xn.push(x[i] + (x[i + 1] - x[i]) * j as f64 / (k + 1) as f64);
            }
        }
        xn.push(x[i + 1]);
    }
    xn
}

fn interp(x_old: &[f64], y_old: &[f64], x_new: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0; x_new.len() * n];
    for (inew, &xv) in x_new.iter().enumerate() {
        let mut j = 0;
        while j + 1 < x_old.len() - 1 && x_old[j + 1] < xv {
            j += 1;
        }
        let denom = x_old[j + 1] - x_old[j];
        let frac = if denom > 0.0 { (xv - x_old[j]) / denom } else { 0.0 };
        for k in 0..n {
            let a = y_old[j * n + k];
            let b = y_old[(j + 1) * n + k];
            y[inew * n + k] = a + frac * (b - a);
        }
    }
    y
}

/// Solve the BVP: collocation + Newton(AD Jacobian) + residual-based refinement.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_bvp(
    f: &BvpFns,
    ws: &mut BvpWorkspace,
    x0: &[f64],
    y0: &[f64],
    p0: &[f64],
    tol: f64,
    max_nodes: usize,
    max_rounds: usize,
) -> BvpSolution {
    let n = f.n_eq;
    let mut x = x0.to_vec();
    let mut y = y0.to_vec();
    let mut p = p0.to_vec();
    let mut converged = false;
    for _ in 0..max_rounds {
        let ok = newton(&x, &mut y, &mut p, f, ws, tol.min(1e-8), 40);
        eval_f(f, &x, &y, &p, n, &mut ws.fval, &mut ws.b_n);
        hermite_residual(&x, &y, &ws.fval, n, f, &p, &mut ws.hres, &mut ws.hres_y, &mut ws.hres_s);
        let max_res = ws.hres.iter().cloned().fold(0.0_f64, f64::max);
        converged = ok && max_res < tol;
        if converged || x.len() >= max_nodes {
            break;
        }
        let xn = refine(&x, &ws.hres, tol);
        if xn.len() == x.len() {
            break;
        }
        y = interp(&x, &y, &xn, n);
        x = xn;
    }
    eval_f(f, &x, &y, &p, n, &mut ws.fval, &mut ws.b_n);
    let fvals = ws.fval.clone();
    BvpSolution { x, y, f: fvals, p, converged }
}

#[cfg(test)]
mod tests {
    use super::*;

    // helper to box closures
    fn run(
        n: usize, k: usize, n_bc: usize, x_ports: &[f64], n_ic: usize,
        fun: &FFn, jac_fy: &FFn, jac_fp: &FFn,
        bc: &BFn, jba: &BFn, jbb: &BFn, jbp: &BFn,
        icond: &IFn, jicy: &IFn, jicp: &IFn,
        x0: &[f64], y0: &[f64], p0: &[f64],
    ) -> BvpSolution {
        let fns = BvpFns {
            n_eq: n, n_params: k, n_bc, x_ports, n_ic,
            fun, jac_fy, jac_fp, bc, jac_bc_ya: jba, jac_bc_yb: jbb, jac_bc_p: jbp,
            icond, jac_ic_y: jicy, jac_ic_p: jicp,
        };
        let mut ws = BvpWorkspace::new();
        solve_bvp(&fns, &mut ws, x0, y0, p0, 1e-6, 2000, 20)
    }

    #[test]
    fn boundary_layer() {
        let eps = 1e-2;
        let fun = move |_x: f64, y: &[f64], _p: &[f64], o: &mut Vec<f64>| { o.clear(); o.extend([y[1], y[1] / eps]); };
        let jfy = move |_x: f64, _y: &[f64], _p: &[f64], o: &mut Vec<f64>| { o.clear(); o.extend([0.0, 1.0, 0.0, 1.0 / eps]); };
        let nop = |_x: f64, _y: &[f64], _p: &[f64], o: &mut Vec<f64>| o.clear();
        let bc = |ya: &[f64], yb: &[f64], _p: &[f64], o: &mut Vec<f64>| { o.clear(); o.extend([ya[0], yb[0] - 1.0]); };
        let jba = |_a: &[f64], _b: &[f64], _p: &[f64], o: &mut Vec<f64>| { o.clear(); o.extend([1.0, 0.0, 0.0, 0.0]); };
        let jbb = |_a: &[f64], _b: &[f64], _p: &[f64], o: &mut Vec<f64>| { o.clear(); o.extend([0.0, 0.0, 1.0, 0.0]); };
        let nbp = |_a: &[f64], _b: &[f64], _p: &[f64], o: &mut Vec<f64>| o.clear();
        let nic = |_y: &[f64], _p: &[f64], o: &mut Vec<f64>| o.clear();
        let m = 10;
        let x0: Vec<f64> = (0..=m).map(|i| i as f64 / m as f64).collect();
        let mut y0 = vec![0.0; (m + 1) * 2];
        for i in 0..=m { y0[i * 2] = x0[i]; y0[i * 2 + 1] = 1.0; }
        let s = run(2, 0, 2, &[], 0, &fun, &jfy, &nop, &bc, &jba, &jbb, &nbp, &nic, &nic, &nic, &x0, &y0, &[]);
        assert!(s.converged);
        let exact = |xx: f64| (f64::exp(xx / eps) - 1.0) / (f64::exp(1.0 / eps) - 1.0);
        let err = (0..s.x.len()).map(|i| (s.y[i * 2] - exact(s.x[i])).abs()).fold(0.0, f64::max);
        assert!(err < 1e-4, "err {err}");
    }

    #[test]
    fn eigenvalue_free_param() {
        // u'' + λ u = 0, u(0)=u(1)=0, u'(0)=1 → λ = π².  y=[u,u'], p=[λ].
        let fun = |_x: f64, y: &[f64], p: &[f64], o: &mut Vec<f64>| { o.clear(); o.extend([y[1], -p[0] * y[0]]); };
        let jfy = |_x: f64, _y: &[f64], p: &[f64], o: &mut Vec<f64>| { o.clear(); o.extend([0.0, 1.0, -p[0], 0.0]); };
        let jfp = |_x: f64, y: &[f64], _p: &[f64], o: &mut Vec<f64>| { o.clear(); o.extend([0.0, -y[0]]); };
        let bc = |ya: &[f64], yb: &[f64], _p: &[f64], o: &mut Vec<f64>| { o.clear(); o.extend([ya[0], yb[0], ya[1] - 1.0]); };
        let jba = |_a: &[f64], _b: &[f64], _p: &[f64], o: &mut Vec<f64>| { o.clear(); o.extend([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]); };
        let jbb = |_a: &[f64], _b: &[f64], _p: &[f64], o: &mut Vec<f64>| { o.clear(); o.extend([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]); };
        let jbp = |_a: &[f64], _b: &[f64], _p: &[f64], o: &mut Vec<f64>| { o.clear(); o.extend([0.0, 0.0, 0.0]); };
        let nic = |_y: &[f64], _p: &[f64], o: &mut Vec<f64>| o.clear();
        let m = 40;
        let x0: Vec<f64> = (0..=m).map(|i| i as f64 / m as f64).collect();
        let mut y0 = vec![0.0; (m + 1) * 2];
        for i in 0..=m { let xv = x0[i]; y0[i * 2] = (std::f64::consts::PI * xv).sin() / std::f64::consts::PI; y0[i * 2 + 1] = (std::f64::consts::PI * xv).cos(); }
        let s = run(2, 1, 3, &[], 0, &fun, &jfy, &jfp, &bc, &jba, &jbb, &jbp, &nic, &nic, &nic, &x0, &y0, &[8.0]);
        assert!(s.converged, "not converged");
        assert!((s.p[0] - std::f64::consts::PI.powi(2)).abs() < 1e-3, "λ = {}", s.p[0]);
    }

    #[test]
    fn interior_condition() {
        // -u'' = 1, u(0)=0, u(0.5)=0.1 → u = -x²/2 + 0.45 x.  n=2, k=0.
        let fun = |_x: f64, y: &[f64], _p: &[f64], o: &mut Vec<f64>| { o.clear(); o.extend([y[1], -1.0]); };
        let jfy = |_x: f64, _y: &[f64], _p: &[f64], o: &mut Vec<f64>| { o.clear(); o.extend([0.0, 1.0, 0.0, 0.0]); };
        let nop = |_x: f64, _y: &[f64], _p: &[f64], o: &mut Vec<f64>| o.clear();
        let bc = |ya: &[f64], _yb: &[f64], _p: &[f64], o: &mut Vec<f64>| { o.clear(); o.push(ya[0]); };
        let jba = |_a: &[f64], _b: &[f64], _p: &[f64], o: &mut Vec<f64>| { o.clear(); o.extend([1.0, 0.0]); };
        let jbb = |_a: &[f64], _b: &[f64], _p: &[f64], o: &mut Vec<f64>| { o.clear(); o.extend([0.0, 0.0]); };
        let nbp = |_a: &[f64], _b: &[f64], _p: &[f64], o: &mut Vec<f64>| o.clear();
        let icond = |yp: &[f64], _p: &[f64], o: &mut Vec<f64>| { o.clear(); o.push(yp[0] - 0.1); };
        let jicy = |_yp: &[f64], _p: &[f64], o: &mut Vec<f64>| { o.clear(); o.extend([1.0, 0.0]); }; // ∂ic/∂[u,u'] at the port
        let nicp = |_yp: &[f64], _p: &[f64], o: &mut Vec<f64>| o.clear();
        let m = 40;
        let x0: Vec<f64> = (0..=m).map(|i| i as f64 / m as f64).collect();
        let y0 = vec![0.0; (m + 1) * 2];
        let s = run(2, 0, 1, &[0.5], 1, &fun, &jfy, &nop, &bc, &jba, &jbb, &nbp, &icond, &jicy, &nicp, &x0, &y0, &[]);
        assert!(s.converged, "not converged");
        let exact = |xx: f64| -xx * xx / 2.0 + 0.45 * xx;
        let err = (0..s.x.len()).map(|i| (s.y[i * 2] - exact(s.x[i])).abs()).fold(0.0, f64::max);
        assert!(err < 1e-4, "err {err}");
    }
}
