// Analog Butterworth + Allpass filters, built natively in Rust from
// prototype-pole formulas (no scipy round-trip).
//
// Reference: same transformations as `scipy.signal.butter` / `lp2lp_zpk` /
// `lp2hp_zpk` / `lp2bp_zpk` / `lp2bs_zpk` — analytically well-defined for
// every filter order `n`. All final coefficients end up as real polynomials
// and are lowered to state-space via `transfer_function_num_den`.

use std::f64::consts::PI;

use num_complex::Complex64;

use crate::blocks::block::BlockRef;
use crate::error::SimError;
use super::lti::{normalize_tf, poly_from_complex_roots, statespace, tf2ss_scipy_raw, transfer_function_num_den};

// Shared user-parameter validation for the filter constructors (issue #28):
// reject a non-positive order or frequency with a typed `SimError` instead of
// panicking uncatchably across the PyO3 boundary.
fn check_order(ctor: &str, n: usize) -> Result<(), SimError> {
    if n < 1 {
        return Err(SimError::InvalidBlockParam(format!(
            "{ctor}: filter order must be >= 1")));
    }
    Ok(())
}

fn check_freq(ctor: &str, what: &str, f: f64) -> Result<(), SimError> {
    // NaN must be rejected too (`!(f > 0.0)` did that implicitly; clippy's
    // suggested `f <= 0.0` would let NaN through).
    if f.is_nan() || f <= 0.0 {
        return Err(SimError::InvalidBlockParam(format!(
            "{ctor}: {what} must be > 0 (got {f})")));
    }
    Ok(())
}

fn check_band(ctor: &str, f_lo: f64, f_hi: f64) -> Result<(), SimError> {
    if !(0.0 < f_lo && f_lo < f_hi) {
        return Err(SimError::InvalidBlockParam(format!(
            "{ctor}: need 0 < f_lo < f_hi (got f_lo={f_lo}, f_hi={f_hi})")));
    }
    Ok(())
}

// ======================================================================================
// Prototype poles (Butterworth LP, cutoff ω = 1)
// ======================================================================================

/// Analog Butterworth lowpass prototype poles (cutoff ω = 1), on the unit
/// circle in the open left half-plane.
///
/// Layout: `[p, p̄, p', p̄', …]` with conjugates listed adjacently and
/// produced via `.conj()` on the upper-half pole (not an independent
/// trig call). This guarantees *bit-exact* conjugate equality so that
/// Gilbert's `is_cc` detection fires correctly downstream.
///
/// For odd `n`, the real pole at `-1` is placed first.
fn butter_lp_proto_poles(n: usize) -> Vec<Complex64> {
    let mut poles = Vec::with_capacity(n);

    if n % 2 == 1 {
        poles.push(Complex64::new(-1.0, 0.0));
    }

    // Half-count of conjugate pairs. Angles for the upper-half poles:
    // θ_k = π/2 + π·(2k − 1)/(2n) for k = 1..=n/2.
    let pairs = n / 2;
    for k in 1..=pairs {
        let theta = PI / 2.0 + PI * (2 * k - 1) as f64 / (2 * n) as f64;
        let p = Complex64::new(theta.cos(), theta.sin());
        poles.push(p);
        poles.push(p.conj());  // bit-exact conjugate
    }

    poles
}

/// Scale a polynomial (given in descending-degree flat form) by a scalar.
fn scale_poly(p: &[f64], k: f64) -> Vec<f64> {
    p.iter().map(|&c| c * k).collect()
}

// ======================================================================================
// Butterworth filters
// ======================================================================================

/// Build a proto-at-ω=1 state space (via scipy-compatible companion form)
/// and frequency-scale by multiplying `A` and `B` by `ω`. This is pathsim's
/// exact pattern (`scipy.signal.butter(..., Wn=1.0)` + `tf2ss` + `ω · A`,
/// `ω · B`) and it preserves bit-close trajectory parity with pathsim under
/// identical adaptive solver settings.
fn proto_then_scale(
    num_proto: &[f64],
    den_proto: &[f64],
    omega: f64,
) -> BlockRef {
    let n = den_proto.len() - 1;
    if n == 0 {
        // 0-th order proto is just a static gain; frequency scaling is moot.
        return transfer_function_num_den(num_proto, den_proto);
    }

    let (num_norm, den_norm) = normalize_tf(num_proto, den_proto);
    let (mut a, mut b, c, d) = tf2ss_scipy_raw(&num_norm, &den_norm, n);

    // Frequency scaling: substituting `s → s/ω` in `H_proto(s)` is equivalent
    // to scaling the state-space matrices as `(ω · A, ω · B, C, D)`.
    for entry in a.iter_mut() { *entry *= omega; }
    for entry in b.iter_mut() { *entry *= omega; }

    statespace(a, b, c, d, n, 1, 1, None)
}

/// Butterworth analog lowpass filter. `fc` is the corner frequency in Hz.
///
/// Built as `butter(n, Wn=1)` prototype in companion form, then frequency-
/// scaled by `ω_c = 2π·fc` on `A` and `B`. Matches pathsim's trajectory to
/// solver tolerance.
pub fn butter_lowpass(fc: f64, n: usize) -> Result<BlockRef, SimError> {
    check_order("butter_lowpass", n)?;
    check_freq("butter_lowpass", "cutoff frequency", fc)?;

    // Proto (ω = 1): H(s) = 1 / ∏ₖ(s − p_k_proto).
    let num_proto = vec![1.0];
    let den_proto = poly_from_complex_roots(&butter_lp_proto_poles(n));

    Ok(proto_then_scale(&num_proto, &den_proto, 2.0 * PI * fc))
}

/// Butterworth analog highpass filter. `fc` is the corner frequency in Hz.
///
/// Built as `butter(n, Wn=1, btype='high')` prototype (zeros at origin,
/// poles at `1/p_k_LP`), then frequency-scaled on `A` and `B`.
pub fn butter_highpass(fc: f64, n: usize) -> Result<BlockRef, SimError> {
    check_order("butter_highpass", n)?;
    check_freq("butter_highpass", "cutoff frequency", fc)?;

    // HP proto poles = 1 / (LP proto poles); keep conjugate adjacency so the
    // resulting polynomial expansion stays real.
    let hp_proto_poles: Vec<Complex64> = butter_lp_proto_poles(n)
        .into_iter()
        .map(|p| Complex64::new(1.0, 0.0) / p)
        .collect();

    // n zeros at origin → num(s) = s^n.
    let mut num_proto = vec![0.0; n + 1];
    num_proto[0] = 1.0;
    let den_proto = poly_from_complex_roots(&hp_proto_poles);

    Ok(proto_then_scale(&num_proto, &den_proto, 2.0 * PI * fc))
}

/// Butterworth analog bandpass filter with passband `[f_lo, f_hi]` in Hz.
///
/// Derived via `s → (s² + ω₀²) / (BW·s)` from the LP prototype, where
/// `ω₀ = √(ω_lo · ω_hi)` and `BW = ω_hi − ω_lo`. Each LP pole maps to two
/// BP poles; `n` zeros appear at the origin. Final filter order is `2n`.
pub fn butter_bandpass(f_lo: f64, f_hi: f64, n: usize) -> Result<BlockRef, SimError> {
    check_order("butter_bandpass", n)?;
    check_band("butter_bandpass", f_lo, f_hi)?;

    let w_lo = 2.0 * PI * f_lo;
    let w_hi = 2.0 * PI * f_hi;
    let w0 = (w_lo * w_hi).sqrt();
    let bw = w_hi - w_lo;

    // Scale LP poles by BW/2, then apply quadratic transform:
    //   p_BP = p_scaled ± √(p_scaled² − ω₀²)
    let mut bp_poles = Vec::with_capacity(2 * n);
    for p in butter_lp_proto_poles(n) {
        let ps = p * (bw / 2.0);
        let disc = (ps * ps - Complex64::new(w0 * w0, 0.0)).sqrt();
        bp_poles.push(ps + disc);
        bp_poles.push(ps - disc);
    }

    // `n` zeros at origin → num(s) = (BW)^n · s^n.
    let mut num = vec![0.0; 2 * n + 1];
    num[n] = bw.powi(n as i32);

    let den = poly_from_complex_roots(&bp_poles);
    Ok(transfer_function_num_den(&num, &den))
}

/// Butterworth analog bandstop (notch) filter with stopband `[f_lo, f_hi]` in Hz.
///
/// Derived via `s → BW·s / (s² + ω₀²)`. Each LP pole `p_k_LP` maps to two BS
/// poles via `p_hp = (BW/2) / p_k_LP` and the same quadratic transform as BP.
/// Zeros: `n` conjugate pairs at `±j·ω₀`.
pub fn butter_bandstop(f_lo: f64, f_hi: f64, n: usize) -> Result<BlockRef, SimError> {
    check_order("butter_bandstop", n)?;
    check_band("butter_bandstop", f_lo, f_hi)?;

    let w_lo = 2.0 * PI * f_lo;
    let w_hi = 2.0 * PI * f_hi;
    let w0 = (w_lo * w_hi).sqrt();
    let bw = w_hi - w_lo;

    // BS pole transformation: invert then quadratic.
    let mut bs_poles = Vec::with_capacity(2 * n);
    for p in butter_lp_proto_poles(n) {
        let ph = Complex64::new(bw / 2.0, 0.0) / p;
        let disc = (ph * ph - Complex64::new(w0 * w0, 0.0)).sqrt();
        bs_poles.push(ph + disc);
        bs_poles.push(ph - disc);
    }

    // Zeros: n conjugate pairs at ±j·ω₀.
    let mut zeros = Vec::with_capacity(2 * n);
    for _ in 0..n {
        zeros.push(Complex64::new(0.0, w0));
        zeros.push(Complex64::new(0.0, -w0));
    }

    // Gain so that H(0) = 1 (stopband rejects, passband passes).
    // For LP prototype with prod(-p_LP) = 1 (all Butter LPs), the BS gain
    // reduces to prod(-z_BS) / prod(-p_BS).
    let prod_neg_z: Complex64 = zeros.iter().map(|&z| -z).product();
    let prod_neg_p: Complex64 = bs_poles.iter().map(|&p| -p).product();
    let k_bs = (prod_neg_z / prod_neg_p).re;

    let num = scale_poly(&poly_from_complex_roots(&zeros), k_bs);
    let den = poly_from_complex_roots(&bs_poles);
    Ok(transfer_function_num_den(&num, &den))
}

// ======================================================================================
// Allpass filter
// ======================================================================================

/// Analog allpass filter with characteristic frequency `fs` (Hz) and order `n`.
///
/// Follows pathsim's sign convention: `H(s) = [(ω_s − s) / (ω_s + s)]^n` so
/// `H(0) = 1` for all `n` and magnitude stays unity across all frequencies
/// while phase rolls from 0 to −n·π.
///
/// Built as pathsim: proto polynomials `num = convolve([-1, 1], …, n times)`,
/// `den = convolve([1, 1], …, n times)`, then frequency-scaled on `A` and `B`.
pub fn allpass_filter(fs: f64, n: usize) -> Result<BlockRef, SimError> {
    check_order("allpass_filter", n)?;
    check_freq("allpass_filter", "frequency", fs)?;

    // Proto factors: `(1 − s)` numerator, `(1 + s)` denominator, both in
    // descending-s form as `[-1, 1]` and `[1, 1]`.
    let mut num_proto = vec![-1.0_f64, 1.0_f64];
    let mut den_proto = vec![ 1.0_f64, 1.0_f64];
    for _ in 1..n {
        num_proto = convolve(&num_proto, &[-1.0, 1.0]);
        den_proto = convolve(&den_proto, &[ 1.0, 1.0]);
    }

    Ok(proto_then_scale(&num_proto, &den_proto, 2.0 * PI * fs))
}

/// 1-D polynomial convolution (same convention as `np.convolve`).
///
/// Used by `allpass_filter` to build the n-fold proto numerator and
/// denominator — same operation pathsim reaches for via `np.convolve`.
fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(actual: f64, expected: f64, tol: f64, ctx: &str) {
        assert!((actual - expected).abs() < tol,
            "{}: got {}, expected {} (|diff| = {:.3e})",
            ctx, actual, expected, (actual - expected).abs());
    }

    #[test]
    fn test_butter_lp_proto_poles_n1() {
        let p = butter_lp_proto_poles(1);
        assert_eq!(p.len(), 1);
        // Single pole at -1.
        approx_eq(p[0].re, -1.0, 1e-12, "p[0].re");
        approx_eq(p[0].im,  0.0, 1e-12, "p[0].im");
    }

    #[test]
    fn test_butter_lp_proto_poles_n2() {
        let p = butter_lp_proto_poles(2);
        assert_eq!(p.len(), 2);
        // -√2/2 ± j·√2/2
        let sqrt2_over_2 = (2.0_f64).sqrt() / 2.0;
        approx_eq(p[0].re, -sqrt2_over_2, 1e-12, "p[0].re");
        approx_eq(p[0].im,  sqrt2_over_2, 1e-12, "p[0].im");
        approx_eq(p[1].re, -sqrt2_over_2, 1e-12, "p[1].re");
        approx_eq(p[1].im, -sqrt2_over_2, 1e-12, "p[1].im");
    }

    #[test]
    fn test_butter_lp_proto_poles_n3_has_real_pole_at_minus_1() {
        // Odd orders always include a real pole at s = -1.
        let p = butter_lp_proto_poles(3);
        assert_eq!(p.len(), 3);
        // One of the three poles must be very close to -1 + 0j.
        let has_real = p.iter().any(|c| (c.re + 1.0).abs() < 1e-12 && c.im.abs() < 1e-12);
        assert!(has_real, "n=3 Butter LP proto should contain a real pole at -1");
    }

    #[test]
    fn test_butter_lp_poles_all_in_lhp() {
        for n in 1..=8 {
            for p in butter_lp_proto_poles(n) {
                assert!(p.re < 0.0,
                    "order {}: pole {:?} is not in LHP", n, p);
                assert!((p.norm() - 1.0).abs() < 1e-12,
                    "order {}: pole {:?} not on unit circle", n, p);
            }
        }
    }

    #[test]
    fn test_butter_lowpass_constructs() {
        // Construction should succeed for common orders + corner frequencies.
        for &fc in &[1.0, 100.0, 1_000.0] {
            for n in 1..=5 {
                let blk = butter_lowpass(fc, n).unwrap();
                assert_eq!(blk.borrow().type_name, "StateSpace");
                assert_eq!(blk.borrow().size().1, n);
            }
        }
    }

    #[test]
    fn test_butter_highpass_has_correct_order() {
        let blk = butter_highpass(50.0, 4).unwrap();
        assert_eq!(blk.borrow().size().1, 4);
    }

    #[test]
    fn test_butter_bandpass_doubles_order() {
        // BP of order n has final filter order 2n.
        for n in 1..=4 {
            let blk = butter_bandpass(50.0, 100.0, n).unwrap();
            assert_eq!(blk.borrow().size().1, 2 * n,
                "bandpass order n={} should yield 2n states", n);
        }
    }

    #[test]
    fn test_butter_bandstop_doubles_order() {
        for n in 1..=3 {
            let blk = butter_bandstop(50.0, 100.0, n).unwrap();
            assert_eq!(blk.borrow().size().1, 2 * n);
        }
    }

    #[test]
    fn test_allpass_filter_order() {
        for n in 1..=5 {
            let blk = allpass_filter(100.0, n).unwrap();
            assert_eq!(blk.borrow().size().1, n);
        }
    }

    #[test]
    fn test_butter_zero_order_rejected() {
        let err = butter_lowpass(10.0, 0).err().unwrap();
        assert!(err.to_string().contains("order must be >= 1"));
    }

    #[test]
    fn test_butter_bandpass_requires_ordered_freqs() {
        let err = butter_bandpass(100.0, 50.0, 2).err().unwrap();
        assert!(err.to_string().contains("f_lo < f_hi"));
    }

    #[test]
    fn test_butter_lowpass_dc_gain_is_one() {
        // Construct, run a long constant input, check steady-state ≈ input.
        let blk = butter_lowpass(1.0, 2).unwrap();
        let b = blk.borrow_mut();
        b.inputs.set_single(0, 1.0);
        // Let the filter's state evolve into steady-state via manual Euler
        // steps. Far enough below cutoff (DC), steady-state output = input.
        // We'll use the f_dyn / f_alg directly.
        let dt = 0.01;
        let n_steps = 5000;
        let mut x = vec![0.0_f64; b.size().1];
        let mut out_buf = vec![0.0; x.len()];
        for _ in 0..n_steps {
            out_buf.clear();
            (b.f_dyn.as_ref().unwrap())(&x, &b.inputs._data, 0.0, &mut out_buf);
            for i in 0..x.len() { x[i] += dt * out_buf[i]; }
        }
        let mut y = Vec::new();
        (b.f_alg.as_ref().unwrap())(&x, &b.inputs._data, 0.0, &mut y);
        // DC gain should be 1 — steady-state output approximately equals input.
        approx_eq(y[0], 1.0, 5e-2, "LP DC gain steady-state");
    }
}
