// Gilbert realization: pole-residue transfer function → minimal state-space.
//
// Mirrors pathsim's `gilbert_realization` (utils/gilbert.py) bit-for-bit:
//   * complex-conjugate pole pairs are preserved as real-valued 2×2 Jordan
//     blocks (the "modal form" — a similarity transform of diag(p, p̄) into
//     real coordinates)
//   * missing conjugate pairs are auto-completed to keep A, B, C real
//   * single-output scalar residues are lifted to (N, 1, 1) shape internally
//   * MIMO residues of shape (N, m, n) are supported with block-diagonal A
//     and B built via Kronecker products
//
// The algorithm is cheap (O(N²) worst case for the companion matrix) and runs
// once at construction time, so we stay in pure Rust with no linalg deps —
// just direct Vec<f64> arithmetic.

use num_complex::Complex64;

/// MIMO residues indexed as `residues[pole_index][output_row][input_col]`.
/// Single-input single-output (SISO) callers typically pass a vector of
/// scalar residues; see [`gilbert_realization_siso`] for that convenience.
pub type MimoResidues = Vec<Vec<Vec<Complex64>>>;

/// Output of Gilbert's method: `(A, B, C, D)` as flat row-major matrices
/// with the given dimensions.
///
/// * `A` — `n_state × n_state`
/// * `B` — `n_state × n_in`
/// * `C` — `m_out × n_state`
/// * `D` — `m_out × n_in`
pub(crate) struct GilbertSS {
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    pub d: Vec<f64>,
    pub n_state: usize,
    pub n_in: usize,
    pub m_out: usize,
}

/// SISO convenience wrapper: scalar residues, `constant` is a single scalar.
pub(crate) fn gilbert_realization_siso(
    poles: &[Complex64],
    residues: &[Complex64],
    constant: f64,
    tolerance: f64,
) -> GilbertSS {
    assert_eq!(poles.len(), residues.len(),
        "gilbert: poles ({}) and residues ({}) length mismatch",
        poles.len(), residues.len());

    let mimo: MimoResidues = residues.iter().map(|&r| vec![vec![r]]).collect();
    gilbert_realization(poles, &mimo, &[constant], 1, 1, tolerance)
}

/// Full MIMO Gilbert realization of `H(s) = D + Σ Rₙ / (s − pₙ)`.
///
/// `residues` must have length `poles.len()`; each inner matrix is `m_out × n_in`.
/// `constant` is the flat row-major `m_out × n_in` direct-feedthrough matrix.
///
/// Real poles produce a 1×1 scalar block on the diagonal of `A`. Complex poles
/// are expected in conjugate pairs (`p`, `p̄`): missing conjugates are added
/// automatically, and the pair is realised as a real 2×2 Jordan block so that
/// `A`, `B`, `C`, `D` are all real-valued.
pub(crate) fn gilbert_realization(
    poles: &[Complex64],
    residues: &MimoResidues,
    constant: &[f64],
    m_out: usize,
    n_in: usize,
    tolerance: f64,
) -> GilbertSS {
    assert!(!poles.is_empty(), "gilbert: at least one pole required");
    assert_eq!(poles.len(), residues.len(),
        "gilbert: poles ({}) and residues ({}) length mismatch",
        poles.len(), residues.len());
    for (k, r) in residues.iter().enumerate() {
        assert_eq!(r.len(), m_out,
            "gilbert: residue {} has {} rows, expected m_out={}", k, r.len(), m_out);
        for (i, row) in r.iter().enumerate() {
            assert_eq!(row.len(), n_in,
                "gilbert: residue {} row {} has {} cols, expected n_in={}",
                k, i, row.len(), n_in);
        }
    }

    // --- 1. Normalise inputs: add missing conjugate pairs, drop tiny imaginary
    //        parts from near-real poles. ---
    //
    // Membership is checked with a tolerance-based distance rather than `==`,
    // because conjugate-pair generators that reach us through frequency
    // transformations (e.g. highpass `ω_c / p`) accumulate ULP-level drift.
    // Strict equality spuriously treats drifted duplicates as new poles and
    // would double the state-space dimension.
    let close = |a: Complex64, b: Complex64| -> bool {
        let scale = (a.norm() + b.norm()).max(1.0);
        (a - b).norm() < 1e-9 * scale
    };

    let mut norm_poles: Vec<Complex64> = Vec::with_capacity(poles.len());
    let mut norm_res: MimoResidues = Vec::with_capacity(residues.len());

    for (p, r) in poles.iter().zip(residues.iter()) {
        let nearly_real = p.im == 0.0
            || (p.re != 0.0 && (p.im / p.re).abs() < tolerance);
        if nearly_real {
            norm_poles.push(Complex64::new(p.re, 0.0));
            // Drop imaginary part element-wise (matches pathsim's `R.real`).
            let r_real: Vec<Vec<Complex64>> = r.iter()
                .map(|row| row.iter().map(|c| Complex64::new(c.re, 0.0)).collect())
                .collect();
            norm_res.push(r_real);
        } else {
            // Truly complex pole. Add as-is if new; also add conjugate if missing.
            if !norm_poles.iter().any(|q| close(*q, *p)) {
                norm_poles.push(*p);
                norm_res.push(r.clone());
            }
            let conj_p = p.conj();
            if !norm_poles.iter().any(|q| close(*q, conj_p)) {
                norm_poles.push(conj_p);
                let r_conj: Vec<Vec<Complex64>> = r.iter()
                    .map(|row| row.iter().map(|c| c.conj()).collect())
                    .collect();
                norm_res.push(r_conj);
            }
        }
    }

    let big_n = norm_poles.len();

    // --- 2. Build the small `a` (N×N) companion matrix and `b` (N,) vector
    //        directly. We'll kron-expand later. ---
    let mut a_small = vec![0.0_f64; big_n * big_n];
    let mut b_small = vec![0.0_f64; big_n];

    // `c_small` is (m, n·N) — matches pathsim's flat layout. Initialised to
    // ones per pathsim, overwritten below for every pole column. We keep the
    // `= 1` default so columns we never touch (should be none) stay sane.
    let mut c_small = vec![1.0_f64; m_out * (n_in * big_n)];

    // Track the previous pole to detect the second member of a CC pair.
    let mut p_old = Complex64::new(0.0, 0.0);

    for (k, (p, r)) in norm_poles.iter().zip(norm_res.iter()).enumerate() {
        // `is_cc`: this pole is the conjugate of the immediately previous one.
        // Pathsim's exact condition: `(p.imag != 0.0 and p == np.conj(p_old))`.
        // Use `close` for the equality check so that conjugate pairs that came
        // in via frequency transforms still register as a pair.
        let is_cc = p.im != 0.0 && close(*p, p_old.conj());
        p_old = *p;

        a_small[k * big_n + k] = p.re;
        b_small[k] = 1.0;
        if is_cc {
            a_small[k * big_n + (k - 1)] = -p.im;
            a_small[(k - 1) * big_n + k] = p.im;
            b_small[k] = 0.0;
            b_small[k - 1] = 2.0;
        }

        // Fill C columns: for input i, column index = k + N·i.
        // Real pole → real part of residue; second of a CC pair → imag part.
        for i in 0..n_in {
            for row in 0..m_out {
                let col = k + big_n * i;
                let r_entry = r[row][i];
                let val = if is_cc { r_entry.im } else { r_entry.re };
                c_small[row * (n_in * big_n) + col] = val;
            }
        }
    }

    // --- 3. Kron expand: A = I_n ⊗ a_small,  B = (I_n ⊗ b_small)ᵀ. ---
    // A block-diagonal with `n_in` copies of `a_small`.
    let n_state = n_in * big_n;
    let mut a = vec![0.0_f64; n_state * n_state];
    for block in 0..n_in {
        let off = block * big_n;
        for r in 0..big_n {
            for c in 0..big_n {
                a[(off + r) * n_state + (off + c)] = a_small[r * big_n + c];
            }
        }
    }

    // B[state, input]: column `input` has `b_small` in rows `[input*N, (input+1)*N)`.
    let mut b = vec![0.0_f64; n_state * n_in];
    for input in 0..n_in {
        for r in 0..big_n {
            b[(input * big_n + r) * n_in + input] = b_small[r];
        }
    }

    // C is already the right shape (m × n_state). Re-use the buffer.
    let c = c_small;

    // D: accept either a scalar (broadcast) or a pre-built m×n matrix.
    let d: Vec<f64> = if constant.len() == 1 {
        vec![constant[0]; m_out * n_in]
    } else {
        assert_eq!(constant.len(), m_out * n_in,
            "gilbert: constant must be a scalar or a flat m_out·n_in = {} matrix",
            m_out * n_in);
        constant.to_vec()
    };

    GilbertSS { a, b, c, d, n_state, n_in, m_out }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64 as C;

    fn approx_vec_eq(actual: &[f64], expected: &[f64], eps: f64) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < eps,
                "index {}: got {}, expected {} (|diff| = {:.3e})",
                i, a, e, (a - e).abs());
        }
    }

    #[test]
    fn test_siso_two_real_poles() {
        // H(s) = 0 + 2/(s + 1) + 3/(s + 4)
        let ss = gilbert_realization_siso(
            &[C::new(-1.0, 0.0), C::new(-4.0, 0.0)],
            &[C::new( 2.0, 0.0), C::new( 3.0, 0.0)],
            0.0,
            1e-9,
        );
        assert_eq!(ss.n_state, 2);
        assert_eq!(ss.n_in, 1);
        assert_eq!(ss.m_out, 1);
        // A = diag(-1, -4) (real pole ⇒ scalar block)
        approx_vec_eq(&ss.a, &[-1.0, 0.0,  0.0, -4.0], 1e-12);
        // B = [1; 1]
        approx_vec_eq(&ss.b, &[1.0, 1.0], 1e-12);
        // C = [2, 3]
        approx_vec_eq(&ss.c, &[2.0, 3.0], 1e-12);
        // D = [0]
        approx_vec_eq(&ss.d, &[0.0], 1e-12);
    }

    #[test]
    fn test_siso_complex_conjugate_pair() {
        // H(s) = (s + 2) / ((s + 1)² + 4)
        // Poles: p = -1 ± 2j. Residues: r = (1 ∓ j·(1/4)) / 2 (from PFD).
        // Actual residue computation:
        //   r1 = lim (s - p) · H(s) at s = -1 + 2j
        //      = (s + 2) / (s - p̄) at s = p
        //      = (1 + 2j) / (4j)
        //      = (1 + 2j) · (-j) / 4 = (2 - j) / 4
        let ss = gilbert_realization_siso(
            &[C::new(-1.0, 2.0), C::new(-1.0, -2.0)],
            &[C::new(0.5, -0.25), C::new(0.5, 0.25)],
            0.0,
            1e-9,
        );
        assert_eq!(ss.n_state, 2);
        // A = [[-1, -2], [2, -1]] — real 2×2 Jordan block for p = -1 ± 2j.
        // Eigenvalues: -1 ± 2j as required.
        approx_vec_eq(&ss.a, &[-1.0, -2.0,  2.0, -1.0], 1e-12);
        // B = [2; 0] (pathsim convention for the first of a CC pair)
        approx_vec_eq(&ss.b, &[2.0, 0.0], 1e-12);
        // C = [Re(r_1), Im(r_2)] = [0.5, 0.25]
        approx_vec_eq(&ss.c, &[0.5, 0.25], 1e-12);
    }

    #[test]
    fn test_auto_adds_missing_conjugate() {
        // User passes only one pole of a CC pair — Gilbert must add the other.
        let ss = gilbert_realization_siso(
            &[C::new(-1.0, 2.0)],           // only p, no p̄
            &[C::new(0.5, -0.25)],
            0.0,
            1e-9,
        );
        assert_eq!(ss.n_state, 2, "conjugate must be auto-added");
    }

    #[test]
    fn test_near_real_pole_treated_as_real() {
        // Imag part is below tolerance → treated as real, single-block.
        let ss = gilbert_realization_siso(
            &[C::new(-1.0, 1e-15)],
            &[C::new( 2.0, 0.0)],
            0.0,
            1e-9,
        );
        assert_eq!(ss.n_state, 1);
        approx_vec_eq(&ss.a, &[-1.0], 1e-12);
    }

    #[test]
    fn test_mimo_2x2_real_poles() {
        // 2 outputs, 2 inputs, 2 real poles. Each residue is a 2×2 real matrix.
        //   p_1 = -1, R_1 = [[1, 2], [3, 4]]
        //   p_2 = -3, R_2 = [[5, 6], [7, 8]]
        let residues: MimoResidues = vec![
            vec![
                vec![C::new(1.0, 0.0), C::new(2.0, 0.0)],
                vec![C::new(3.0, 0.0), C::new(4.0, 0.0)],
            ],
            vec![
                vec![C::new(5.0, 0.0), C::new(6.0, 0.0)],
                vec![C::new(7.0, 0.0), C::new(8.0, 0.0)],
            ],
        ];
        let ss = gilbert_realization(
            &[C::new(-1.0, 0.0), C::new(-3.0, 0.0)],
            &residues,
            &[0.0],
            2, 2, 1e-9,
        );
        assert_eq!(ss.n_state, 2 * 2);  // N = 2 poles, n_in = 2
        assert_eq!(ss.n_in, 2);
        assert_eq!(ss.m_out, 2);
        // A = block-diag(a, a) with a = diag(-1, -3) ⇒ 4×4 diag(-1, -3, -1, -3)
        approx_vec_eq(&ss.a, &[
            -1.0,  0.0,  0.0,  0.0,
             0.0, -3.0,  0.0,  0.0,
             0.0,  0.0, -1.0,  0.0,
             0.0,  0.0,  0.0, -3.0,
        ], 1e-12);
        // B is 4×2: each input column picks its own copy of b = [1; 1].
        //   B[0, 0] = 1, B[1, 0] = 1 (input 0 → states 0, 1)
        //   B[2, 1] = 1, B[3, 1] = 1 (input 1 → states 2, 3)
        approx_vec_eq(&ss.b, &[
            1.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            0.0, 1.0,
        ], 1e-12);
        // C is 2×4. Column k + N·i pulls R[:,:,i][row, col=(pole index k)].
        // Actually: col index = k + N*i, with k ∈ [0, N), i ∈ [0, n_in).
        //   col 0 (k=0, i=0): R[row, 0] of R_1 and R_2 → [1, 5] down rows, wait
        // Pathsim's loop: `C[:, k + N*i] = real(R[:, i])` for residue at pole k.
        // So C[row, k + N*i] = R_k[row, i].
        //   col 0 (k=0, i=0): R_0[:,0] = [1, 3]
        //   col 1 (k=1, i=0): R_1[:,0] = [5, 7]
        //   col 2 (k=0, i=1): R_0[:,1] = [2, 4]
        //   col 3 (k=1, i=1): R_1[:,1] = [6, 8]
        approx_vec_eq(&ss.c, &[
            1.0, 5.0, 2.0, 6.0,
            3.0, 7.0, 4.0, 8.0,
        ], 1e-12);
        // D: scalar 0.0 broadcast to 2×2 zeros.
        approx_vec_eq(&ss.d, &[0.0, 0.0, 0.0, 0.0], 1e-12);
    }

    #[test]
    fn test_constant_scalar_vs_matrix() {
        // Const = 2.0 (scalar) for 1×1 → D = [2.0]
        let ss = gilbert_realization_siso(
            &[C::new(-1.0, 0.0)], &[C::new(1.0, 0.0)], 2.0, 1e-9);
        approx_vec_eq(&ss.d, &[2.0], 1e-12);

        // Const as full 1×1 matrix → also works
        let ss = gilbert_realization(
            &[C::new(-1.0, 0.0)],
            &vec![vec![vec![C::new(1.0, 0.0)]]],
            &[3.0], 1, 1, 1e-9);
        approx_vec_eq(&ss.d, &[3.0], 1e-12);
    }
}
