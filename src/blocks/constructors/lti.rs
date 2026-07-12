// Linear time-invariant (LTI) block constructors: StateSpace and its
// narrower specialisations PT1 / PT2, plus TransferFunctionNumDen which
// lowers a numerator/denominator polynomial pair into controllable
// canonical state-space form.
//
// Helpers `flat_to_mat`, `matvec_into`, `vec_add_into` live here because
// they are specific to matrix-based LTI blocks; `math_logic::matrix_block`
// re-imports them via the parent module.

use std::rc::Rc;

use num_complex::Complex64;
use smallvec::SmallVec;

use std::cell::RefCell;

use crate::blocks::block::{Block, BlockRef, BlockRole};
use crate::constants::FILTER_POLY_REAL_TOL;
use crate::ssa::build::{dot_f64, Builder, GraphBuilder};
use crate::ssa::graph::{Graph, InputSignature};

use super::amplifier;
use crate::solvers::solver::Solver;
use crate::utils::fastcell::FastCell;
use crate::utils::gilbert::{gilbert_realization, gilbert_realization_siso, MimoResidues};
use crate::utils::register::Register;

/// Affine matrix form `out[r] = sum_j m1[r,j]*x[j] + sum_k m2[r,k]*u[k]`, the
/// shared kernel of every state-space region: `y = Cx + Du` (alg) and
/// `dx/dt = Ax + Bu` (dyn). Zero coefficients are skipped, so companion /
/// sparse realisations (transfer functions, filters) yield lean op-graphs.
/// `m1`/`m2` are row-major flat (`rows x n1`, `rows x n2`).
pub(crate) fn affine_mv_eval<B: Builder>(
    b: &B,
    m1: &[f64], n1: usize,
    m2: &[f64], n2: usize,
    rows: usize,
    x: &[B::N], u: &[B::N],
    out: &mut Vec<B::N>,
) {
    out.clear();
    // GRAPH path (built once per block): each output row is a fused dot product
    // `Σ coeff·value` recorded as one `Dot` node. Zero coefficients are skipped
    // so companion / sparse realisations stay lean. The NATIVE runtime hot path
    // does NOT come through here (it would pay per-call collection); it uses the
    // streaming `affine_mv_into` below.
    let mut coeffs: SmallVec<[B::N; 8]> = SmallVec::new();
    let mut vals: SmallVec<[B::N; 8]> = SmallVec::new();
    for r in 0..rows {
        coeffs.clear();
        vals.clear();
        for j in 0..n1.min(x.len()) {
            let c = m1[r * n1 + j];
            if c != 0.0 {
                coeffs.push(b.cst(c));
                vals.push(x[j]);
            }
        }
        for k in 0..n2.min(u.len()) {
            let c = m2[r * n2 + k];
            if c != 0.0 {
                coeffs.push(b.cst(c));
                vals.push(u[k]);
            }
        }
        out.push(b.dot(&coeffs, &vals));
    }
}

/// Allocation-free native affine matvec for the runtime hot path:
/// `out[r] = m1[r,:]·x + m2[r,:]·u`, folded with the 4-lane `dot_f64` over the
/// contiguous matrix rows. Unlike the generic `affine_mv_eval` (which collects
/// operand lists for the graph `Dot`), this never allocates and vectorizes for
/// dense `A`, so a StateSpace-heavy simulation stays fast. The op-graph form
/// still goes through `affine_mv_eval`; the eval-parity test guards that native
/// and graph agree to 1e-12. Dense rows are folded whole (no zero-skip branch,
/// which would defeat vectorization); sparse companion forms pay a few extra
/// multiply-by-zero, negligible at the low orders where they occur.
pub(crate) fn affine_mv_into(
    out: &mut Vec<f64>,
    m1: &[f64], n1: usize,
    m2: &[f64], n2: usize,
    rows: usize,
    x: &[f64], u: &[f64],
) {
    out.clear();
    let kx = n1.min(x.len());
    let ku = n2.min(u.len());
    for r in 0..rows {
        // `A` row through the vectorized dot; the `B` row (usually few inputs)
        // appended serially so a tiny `ku` does not pay the 4-lane combine cost.
        let mut s = dot_f64(&m1[r * n1..r * n1 + kx], &x[..kx]);
        let b_row = &m2[r * n2..r * n2 + ku];
        for k in 0..ku {
            s += b_row[k] * u[k];
        }
        out.push(s);
    }
}

/// Build a state-space region graph with `("x", ns)` and `("u", ni)` slots.
fn statespace_region_graph(m1: &[f64], n1: usize, m2: &[f64], n2: usize, rows: usize, ns: usize, ni: usize) -> Graph {
    let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("x", ns), ("u", ni)])));
    let out_nodes = {
        let gb = GraphBuilder::new(&cell);
        let x: Vec<_> = (0..ns as u32).map(|i| gb.input(i)).collect();
        let u: Vec<_> = (0..ni as u32).map(|i| gb.input(ns as u32 + i)).collect();
        let mut out = Vec::new();
        affine_mv_eval(&gb, m1, n1, m2, n2, rows, &x, &u, &mut out);
        out
    };
    let mut g = cell.into_inner();
    g.outputs = out_nodes;
    g
}

// ======================================================================================
// StateSpace: dx/dt = Ax+Bu, y = Cx+Du — overrides len, update, solve, step
// ======================================================================================

/// Helper: faer Mat from row-major flat storage (allocates once).
pub(crate) fn flat_to_mat(data: &[f64], rows: usize, cols: usize) -> faer::Mat<f64> {
    faer::Mat::from_fn(rows, cols, |i, j| data[i * cols + j])
}

/// Helper: matvec in-place: out = mat * x. No allocation.
/// Uses manual loop for small matrices, faer SIMD for larger ones.
#[inline]
pub(crate) fn matvec_into(out: &mut [f64], mat: &faer::Mat<f64>, x: &[f64]) {
    let rows = out.len();
    let cols = mat.ncols().min(x.len());
    if rows * cols <= 64 {
        for i in 0..rows {
            let mut s = 0.0;
            for j in 0..cols { s += mat[(i, j)] * x[j]; }
            out[i] = s;
        }
    } else {
        use faer::col::ColRef;
        let v = ColRef::from_slice(&x[..cols]);
        let r = mat * v;
        for i in 0..rows { out[i] = r[i]; }
    }
}

/// StateSpace: dx/dt = Ax + Bu, y = Cx + Du
pub fn statespace(
    a: Vec<f64>, b_mat: Vec<f64>, c: Vec<f64>, d: Vec<f64>,
    ns: usize, ni: usize, no: usize,
    initial_value: Option<Vec<f64>>,
) -> BlockRef {
    let mut blk = Block::new(None, None);
    blk.type_name = "StateSpace";
    blk.inputs = Register::new(Some(ni), None);
    blk.outputs = Register::new(Some(no), None);

    let iv = initial_value.unwrap_or_else(|| vec![0.0; ns]);
    blk.initial_value = Some(iv.clone());
    blk.engine = Some(Solver::with_defaults(&iv));

    let has_passthrough = d.iter().any(|&v| v != 0.0);
    blk.role = BlockRole { is_dyn: true, is_src: false, is_rec: false };
    blk.len_fn = Some(Box::new(move |b| {
        if !b._active { 0 } else if has_passthrough { 1 } else { 0 }
    }));

    // Native closures and op-graphs both derive from `affine_mv_eval`.
    // dx/dt = Ax + Bu
    let (a_dyn, b_dyn) = (a.clone(), b_mat.clone());
    blk.f_dyn = Some(Box::new(move |x, u, _t, out| {
        affine_mv_into(out, &a_dyn, ns, &b_dyn, ni, ns, x, u);
    }));

    // J = A is derived by AD from the `dyn_` op-graph (∂(Ax+Bu)/∂x = A) via the
    // block's Operator, so no hand-written `jac_dyn` is needed. A is fixed, so AD
    // folds it to constants and the linear solver's factorization cache reuses
    // the factor across dt (the `jacobian_const` flag stays as the hint).
    blk.jacobian_const = true;

    // y = Cx + Du
    let (c_alg, d_alg) = (c.clone(), d.clone());
    blk.f_alg = Some(Box::new(move |x, u, _t, out| {
        affine_mv_into(out, &c_alg, ns, &d_alg, ni, no, x, u);
    }));

    let alg_graph = statespace_region_graph(&c, ns, &d, ni, no, ns, ni);
    let dyn_graph = statespace_region_graph(&a, ns, &b_mat, ni, ns, ns, ni);
    blk.set_dynamic("StateSpace", alg_graph, dyn_graph);

    Rc::new(FastCell::new(blk))
}



// ======================================================================================
// PT1: H(s) = K/(1+Ts) — StateSpace with specific matrices
// ======================================================================================

/// PT1 first-order lag: H(s) = K / (1 + Ts)
pub fn pt1(k: f64, t: f64) -> BlockRef {
    statespace(vec![-1.0/t], vec![k/t], vec![1.0], vec![0.0], 1, 1, 1, None)
}

/// PT2 second-order lag: H(s) = K / (T^2 s^2 + 2dTs + 1)
pub fn pt2(k: f64, t: f64, d: f64) -> BlockRef {
    statespace(
        vec![0.0, 1.0, -1.0/(t*t), -2.0*d/t],
        vec![0.0, 1.0], vec![k/(t*t), 0.0], vec![0.0],
        2, 1, 1, None,
    )
}

// ======================================================================================
// TransferFunctionNumDen: H(s) = Num(s)/Den(s) — scipy.signal.tf2ss-compatible form
// ======================================================================================

/// Transfer function in descending-power coefficient form, lowered to
/// state-space exactly as `scipy.signal.tf2ss` does. Matching scipy's
/// canonical ordering means analog filter blocks (Butterworth &c.) produce
/// bit-compatible trajectories against pathsim under identical adaptive
/// solver settings.
///
/// Formulation (strictly proper or same-degree num/den, both in descending
/// powers of `s`, `den[0]` is the leading coefficient):
///
/// * pad numerator on the left with zeros to match `den.len()`
/// * normalise by `den[0]`
/// * `A[0,:] = -den_norm[1..]`  (first row)
///   `A[i, i-1] = 1 for i ≥ 1`  (sub-diagonal identity)
/// * `B = [1, 0, 0, …]ᵀ`
/// * `D = num_norm[0]`  (direct feedthrough)
/// * `C = num_norm[1..] − D · den_norm[1..]`
pub fn transfer_function_num_den(num: &[f64], den: &[f64]) -> BlockRef {
    assert!(!den.is_empty(), "TransferFunction: denominator must not be empty");
    assert!(!num.is_empty(), "TransferFunction: numerator must not be empty");
    assert!(num.len() <= den.len(),
        "TransferFunction: numerator degree ({}) > denominator degree ({})",
        num.len() - 1, den.len() - 1);

    let n = den.len() - 1;
    if n == 0 {
        let gain = if den[0] != 0.0 { num[0] / den[0] } else { 0.0 };
        return amplifier(gain);
    }

    let (num_norm, den_norm) = normalize_tf(num, den);
    let (a, b_mat, c, d) = tf2ss_scipy_raw(&num_norm, &den_norm, n);
    statespace(a, b_mat, c, d, n, 1, 1, None)
}

/// Normalise a transfer function to monic denominator: divide both polynomials
/// by `den[0]` and left-pad the numerator to the denominator's length. Shared by
/// the continuous / discrete transfer-function and proto-then-scale filter paths.
pub(crate) fn normalize_tf(num: &[f64], den: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let a0 = den[0];
    let den_norm: Vec<f64> = den.iter().map(|&c| c / a0).collect();
    let mut num_norm = vec![0.0; den.len()];
    let offset = den.len() - num.len();
    for (i, &c) in num.iter().enumerate() {
        num_norm[offset + i] = c / a0;
    }
    (num_norm, den_norm)
}

/// Build scipy-`tf2ss`-compatible `(A, B, C, D)` from a normalised `num` /
/// `den` pair (same length, `den[0] = 1`). Pulled out so that Butterworth /
/// Allpass filters can post-scale `A`, `B` (pathsim's "proto-then-scale"
/// pattern) before handing off to `statespace()`.
///
/// State ordering matches scipy so that trajectories are bit-close under
/// identical adaptive solver settings.
pub(crate) fn tf2ss_scipy_raw(
    num_norm: &[f64],
    den_norm: &[f64],
    n: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    assert_eq!(num_norm.len(), den_norm.len());
    assert_eq!(num_norm.len(), n + 1);

    // A: first row = −den_norm[1..]; sub-diagonal = 1.
    let mut a = vec![0.0_f64; n * n];
    for j in 0..n {
        a[j] = -den_norm[j + 1];
    }
    for i in 1..n {
        a[i * n + (i - 1)] = 1.0;
    }

    // B = [1, 0, …, 0]ᵀ.
    let mut b_mat = vec![0.0_f64; n];
    b_mat[0] = 1.0;

    // D = num_norm[0]; C_i = num_norm[i+1] − D · den_norm[i+1].
    let d_val = num_norm[0];
    let d = vec![d_val];

    let mut c = vec![0.0_f64; n];
    for i in 0..n {
        c[i] = num_norm[i + 1] - d_val * den_norm[i + 1];
    }

    (a, b_mat, c, d)
}

// ======================================================================================
// Polynomial expansion helper — supports complex roots (including repeated)
// and returns real coefficients when roots come in conjugate pairs.
// ======================================================================================

// Tolerance for collapsing tiny imaginary artefacts from polynomial expansion
// over conjugate pairs — lives in `constants::FILTER_POLY_REAL_TOL`.

/// Expand `∏ᵢ (s − rootᵢ)` into descending-degree coefficients. Roots may be
/// real, complex, or repeated — the product is computed in complex arithmetic
/// and then projected onto the real axis, asserting the imaginary parts are
/// negligible. (Conjugate pairs produce real coefficients exactly; isolated
/// complex roots without their partner produce a non-real polynomial, which
/// we flag.)
pub(crate) fn poly_from_complex_roots(roots: &[Complex64]) -> Vec<f64> {
    let mut coeffs: Vec<Complex64> = vec![Complex64::new(1.0, 0.0)];
    for &r in roots {
        // Multiply existing polynomial by (s − r): shift up + subtract r · prev.
        let mut next = vec![Complex64::new(0.0, 0.0); coeffs.len() + 1];
        for (i, &c) in coeffs.iter().enumerate() {
            next[i] += c;
            next[i + 1] -= r * c;
        }
        coeffs = next;
    }

    // Assert imaginary parts vanish (conjugate-pair roots guarantee this).
    let max_mag = coeffs.iter().map(|c| c.norm()).fold(0.0_f64, f64::max).max(1.0);
    let tol = FILTER_POLY_REAL_TOL * max_mag;
    for (i, c) in coeffs.iter().enumerate() {
        assert!(c.im.abs() < tol,
            "poly_from_complex_roots: coefficient {} has non-negligible imaginary \
             part {:.3e} — did you forget a conjugate pole/zero?", i, c.im);
    }

    coeffs.into_iter().map(|c| c.re).collect()
}

// ======================================================================================
// TransferFunctionZPG: H(s) = k · ∏(s − zᵢ) / ∏(s − pⱼ) — real + complex roots
// ======================================================================================

/// Transfer function in zero-pole-gain form:
/// `H(s) = gain · ∏ᵢ (s − zᵢ) / ∏ⱼ (s − pⱼ)`.
///
/// SISO. Zeros and poles may be real or complex, and may have arbitrary
/// multiplicity (repeated roots). Complex roots must appear in conjugate
/// pairs so that the resulting polynomial has real coefficients; missing
/// conjugates are detected and reported via the polynomial-expansion sanity
/// check in `poly_from_complex_roots`.
pub fn transfer_function_zpg(
    zeros: &[Complex64],
    poles: &[Complex64],
    gain: f64,
) -> BlockRef {
    assert!(!poles.is_empty(), "TransferFunctionZPG: at least one pole required");
    assert!(zeros.len() <= poles.len(),
        "TransferFunctionZPG: improper TF (more zeros than poles) not supported");

    let num_poly = poly_from_complex_roots(zeros);
    let den_poly = poly_from_complex_roots(poles);
    let num_scaled: Vec<f64> = num_poly.iter().map(|&c| c * gain).collect();

    transfer_function_num_den(&num_scaled, &den_poly)
}

// ======================================================================================
// TransferFunctionPRC: H(s) = D + Σ Rₙ / (s − pₙ)
//   — SISO and MIMO via Gilbert realisation (native Rust, mirrors pathsim).
// ======================================================================================

/// SISO transfer function in pole-residue-constant form:
/// `H(s) = constant + Σ rᵢ / (s − pᵢ)`.
///
/// Accepts real or complex poles/residues. Complex-conjugate pairs are
/// realised as real 2×2 Jordan blocks via
/// [`gilbert_realization_siso`][crate::utils::gilbert::gilbert_realization_siso],
/// so the resulting state-space matrices stay real. Missing conjugates are
/// added automatically for numerical robustness.
pub fn transfer_function_prc(
    poles: &[Complex64],
    residues: &[Complex64],
    constant: f64,
) -> BlockRef {
    let ss = gilbert_realization_siso(poles, residues, constant, FILTER_POLY_REAL_TOL);
    statespace(ss.a, ss.b, ss.c, ss.d, ss.n_state, ss.n_in, ss.m_out, None)
}

/// MIMO transfer function in pole-residue-constant form:
/// `H(s) = D + Σ Rₙ / (s − pₙ)` with matrix residues `Rₙ ∈ ℂ^{m × n}`.
///
/// `residues[k][row][col]` is the `(row, col)` entry of the residue matrix at
/// pole `poles[k]`. `constant` is either a single scalar (broadcast to an
/// `m × n` matrix) or a flat row-major `m · n` vector.
pub fn transfer_function_prc_mimo(
    poles: &[Complex64],
    residues: &MimoResidues,
    constant: &[f64],
    m_out: usize,
    n_in: usize,
) -> BlockRef {
    let ss = gilbert_realization(poles, residues, constant, m_out, n_in, FILTER_POLY_REAL_TOL);
    statespace(ss.a, ss.b, ss.c, ss.d, ss.n_state, ss.n_in, ss.m_out, None)
}

/// Alias for `transfer_function_prc` — matches pathsim's deprecated
/// `TransferFunction` class. Kept for drop-in API parity.
pub fn transfer_function(
    poles: &[Complex64],
    residues: &[Complex64],
    constant: f64,
) -> BlockRef {
    transfer_function_prc(poles, residues, constant)
}

#[cfg(test)]
mod tf_tests {
    use super::*;
    use num_complex::Complex64 as C;

    fn real(x: f64) -> Complex64 { Complex64::new(x, 0.0) }

    #[test]
    fn test_poly_from_complex_roots_real_only() {
        // (s − 2)(s − 3) = s² − 5s + 6
        let p = poly_from_complex_roots(&[real(2.0), real(3.0)]);
        assert_eq!(p, vec![1.0, -5.0, 6.0]);
        assert_eq!(poly_from_complex_roots(&[]), vec![1.0]);
        assert_eq!(poly_from_complex_roots(&[real(5.0)]), vec![1.0, -5.0]);
    }

    #[test]
    fn test_poly_from_conjugate_pair() {
        // (s − (1+2j))(s − (1−2j)) = s² − 2s + 5 (expected real coefficients)
        let p = poly_from_complex_roots(&[C::new(1.0, 2.0), C::new(1.0, -2.0)]);
        assert!((p[0] - 1.0).abs() < 1e-12);
        assert!((p[1] - (-2.0)).abs() < 1e-12);
        assert!((p[2] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_poly_repeated_roots() {
        // (s − 1)³ = s³ − 3s² + 3s − 1
        let p = poly_from_complex_roots(&[real(1.0), real(1.0), real(1.0)]);
        assert_eq!(p, vec![1.0, -3.0, 3.0, -1.0]);
    }

    #[test]
    #[should_panic(expected = "non-negligible imaginary part")]
    fn test_poly_rejects_isolated_complex_root() {
        // Complex root without its conjugate → non-real polynomial.
        let _ = poly_from_complex_roots(&[C::new(1.0, 2.0)]);
    }

    #[test]
    fn test_zpg_siso_real() {
        // H(s) = 2(s − 1) / ((s + 3)(s + 5)) — 2nd-order proper TF.
        let tf = transfer_function_zpg(&[real(1.0)], &[real(-3.0), real(-5.0)], 2.0);
        let b = tf.borrow();
        assert_eq!(b.type_name, "StateSpace");
        assert_eq!(b.size().1, 2);
    }

    #[test]
    fn test_zpg_complex_conjugate_poles() {
        // H(s) = 1 / ((s + 1 + 2j)(s + 1 − 2j)) = 1 / (s² + 2s + 5).
        let tf = transfer_function_zpg(&[], &[C::new(-1.0, 2.0), C::new(-1.0, -2.0)], 1.0);
        let b = tf.borrow();
        assert_eq!(b.size().1, 2);
    }

    #[test]
    fn test_zpg_repeated_poles() {
        // Double pole at s = -1 → (s + 1)² = s² + 2s + 1.
        let tf = transfer_function_zpg(&[], &[real(-1.0), real(-1.0)], 1.0);
        let b = tf.borrow();
        assert_eq!(b.size().1, 2);
    }

    #[test]
    fn test_prc_real_poles() {
        // H(s) = 1 + 2/(s + 1) + 3/(s + 4) — two real poles, scalar residues.
        let tf = transfer_function_prc(
            &[real(-1.0), real(-4.0)],
            &[real(2.0), real(3.0)],
            1.0,
        );
        let b = tf.borrow();
        assert_eq!(b.size().1, 2);
        assert_eq!(b.type_name, "StateSpace");
    }

    #[test]
    fn test_prc_complex_conjugate_residues() {
        // p = −1 ± 2j, r = 0.5 ∓ 0.25j ⇒ real 2×2 Jordan block.
        let tf = transfer_function_prc(
            &[C::new(-1.0, 2.0), C::new(-1.0, -2.0)],
            &[C::new(0.5, -0.25), C::new(0.5, 0.25)],
            0.0,
        );
        let b = tf.borrow();
        assert_eq!(b.size().1, 2);
    }

    #[test]
    fn test_prc_mimo_2x2() {
        // 2-input 2-output system with 2 real poles.
        let residues: MimoResidues = vec![
            vec![
                vec![real(1.0), real(2.0)],
                vec![real(3.0), real(4.0)],
            ],
            vec![
                vec![real(5.0), real(6.0)],
                vec![real(7.0), real(8.0)],
            ],
        ];
        let tf = transfer_function_prc_mimo(
            &[real(-1.0), real(-3.0)],
            &residues, &[0.0], 2, 2,
        );
        let b = tf.borrow();
        let (ni, nx) = (b.inputs.len(), b.size().1);
        assert_eq!(ni, 2);
        assert_eq!(nx, 4);  // N_poles · n_in = 2 · 2
        assert_eq!(b.outputs.len(), 2);
    }

    #[test]
    #[should_panic(expected = "at least one pole required")]
    fn test_zpg_requires_poles() {
        let _ = transfer_function_zpg(&[real(1.0)], &[], 1.0);
    }

    #[test]
    #[should_panic(expected = "improper TF")]
    fn test_zpg_rejects_improper_tf() {
        let _ = transfer_function_zpg(&[real(1.0), real(2.0)], &[real(3.0)], 1.0);
    }
}
