// ======================================================================================
// Spectrum: Running Fourier Transform (RFT) analyzer — sample-accumulator variant
// ======================================================================================
//
// The RFT is defined by the ODE  dx/dt = u(t)·exp(-j·ω·t) - α·x.
// Instead of letting the RK solver integrate this (which suffers from phase-error
// accumulation at high ω·dt — simulations with ω·T ≫ 1 like radar produce NaN),
// we bypass the solver and update the accumulator directly in `sample_fn`, once
// per accepted step. Per-step exponential integrator with trapezoidal quadrature
// of the source term (second-order accurate, O(dt²) global error):
//
//   x(t+dt) = decay · x(t)
//           + (dt/2) · [decay · phase(t) · u(t) + phase(t+dt) · u(t+dt)]
//
//   phase(t+dt) = phase(t) · rotator
//
// where rotator = exp(-j·ω·dt) and decay = exp(-α·dt) are cached.  The
// oscillation is solved exactly (multiplicative rotation introduces only ~1 ULP
// per step; RK would accrue O((ω·dt)^{p+1}) phase error per step).
//
// State layout (all Vec<f64> inside data_vec, SoA for SIMD-friendliness):
//   x_re[i*nf + k], x_im[i*nf + k]       — Fourier accumulator, one per (input, freq)
//   phase_re[k],   phase_im[k]           — exp(-j·ω_k·t_prev), one per freq
//   phase_new_re[k], phase_new_im[k]     — scratch, exp(-j·ω_k·t_curr) during a call
//   rot_re[k],     rot_im[k]             — exp(-j·ω_k·dt), recomputed on dt change
//   u_prev[i]                            — previous-step input values
//   omega[k]                             — angular frequencies

use std::collections::HashMap;
use std::rc::Rc;

use crate::blocks::block::{Block, BlockRef, BlockRole};
use crate::utils::fastcell::FastCell;

/// Create a Spectrum block (Running Fourier Transform).
///
/// # Arguments
/// * `freq` - evaluation frequencies in Hz
/// * `t_wait` - wait time before starting RFT
/// * `alpha` - exponential forgetting factor (0 = standard RFT)
/// * `n_inputs` - number of input channels (unused — derived from connections)
/// * `labels` - optional channel labels
pub fn spectrum(freq: Vec<f64>, t_wait: f64, alpha: f64, _n_inputs: usize, labels: Vec<String>) -> BlockRef {
    let n_freqs = freq.len();
    let omega: Vec<f64> = freq.iter().map(|&f| 2.0 * std::f64::consts::PI * f).collect();

    let mut b = Block::new(None, Some(HashMap::new()));
    b.type_name = "Spectrum";
    // Pure recorder: no ODE, no solver engine, no RK stages.
    b.role = BlockRole { is_dyn: false, is_src: false, is_rec: true };

    // Store metadata
    b.data_f64.insert("t_wait".to_string(), t_wait);
    b.data_f64.insert("alpha".to_string(), alpha);
    b.data_f64.insert("n_freqs".to_string(), n_freqs as f64);
    b.data_f64.insert("time".to_string(), 0.0);
    b.data_f64.insert("t_sample".to_string(), 0.0);
    b.data_f64.insert("dt_cached".to_string(), 0.0);
    b.data_f64.insert("decay".to_string(), 1.0);
    b.data_vec.insert("freq".to_string(), freq);
    b.data_vec.insert("omega".to_string(), omega);
    // Accumulator and phase arrays start empty; sized lazily on first sample.
    b.data_vec.insert("x_re".to_string(), Vec::new());
    b.data_vec.insert("x_im".to_string(), Vec::new());
    b.data_vec.insert("phase_re".to_string(), Vec::new());
    b.data_vec.insert("phase_im".to_string(), Vec::new());
    b.data_vec.insert("phase_new_re".to_string(), Vec::new());
    b.data_vec.insert("phase_new_im".to_string(), Vec::new());
    b.data_vec.insert("rot_re".to_string(), Vec::new());
    b.data_vec.insert("rot_im".to_string(), Vec::new());
    b.data_vec.insert("u_prev".to_string(), Vec::new());
    if !labels.is_empty() {
        b.data_strings.insert("labels".to_string(), labels);
    }

    // len = 0 (recording block, no passthrough)
    b.len_fn = Some(Box::new(|_| 0));

    // reset: clear accumulator + phase + cached dt
    b.reset_fn = Some(Box::new(|blk| {
        blk.inputs.reset();
        blk.outputs.reset();
        blk.data_f64.insert("time".to_string(), 0.0);
        blk.data_f64.insert("t_sample".to_string(), 0.0);
        blk.data_f64.insert("dt_cached".to_string(), 0.0);
        blk.data_f64.insert("decay".to_string(), 1.0);
        for key in ["x_re", "x_im", "phase_re", "phase_im", "phase_new_re",
                    "phase_new_im", "rot_re", "rot_im", "u_prev"] {
            if let Some(v) = blk.data_vec.get_mut(key) { v.clear(); }
        }
    }));

    // sample: exponential-integrator accumulator update, called per accepted step
    b.sample_fn = Some(Box::new(move |blk, t, dt| {
        let t_wait = blk.data_f64.get("t_wait").copied().unwrap_or(0.0);
        blk.data_f64.insert("t_sample".to_string(), t);
        if t < t_wait || dt <= 0.0 { return; }

        let alpha = blk.data_f64.get("alpha").copied().unwrap_or(0.0);
        let nf = blk.data_f64.get("n_freqs").copied().unwrap_or(0.0) as usize;
        if nf == 0 { return; }

        let ni = blk.inputs._data.len();
        if ni == 0 { return; }

        // Lazy sizing: a mismatch on `x_re` or `u_prev` means the input width
        // just changed (fresh block, reset, or connection change). In that case
        // we also clear the phase so the uninitialized-phase branch below
        // re-anchors it to the current t.
        let x_mismatch     = blk.data_vec.get("x_re").map(|v| v.len()).unwrap_or(0) != ni * nf;
        let u_prev_mismatch = blk.data_vec.get("u_prev").map(|v| v.len()).unwrap_or(0) != ni;
        if x_mismatch || u_prev_mismatch {
            for key in ["x_re", "x_im"] {
                if let Some(v) = blk.data_vec.get_mut(key) { v.clear(); v.resize(ni * nf, 0.0); }
            }
            if let Some(v) = blk.data_vec.get_mut("u_prev") { v.clear(); v.resize(ni, 0.0); }
            if let Some(v) = blk.data_vec.get_mut("phase_re") { v.clear(); }
            if let Some(v) = blk.data_vec.get_mut("phase_im") { v.clear(); }
        }

        // (Re)build rotator cache if dt changed or uninitialized.
        let dt_cached = blk.data_f64.get("dt_cached").copied().unwrap_or(0.0);
        let rotator_stale = (dt - dt_cached).abs() > dt_cached.abs() * crate::constants::SPECTRUM_DT_REL_TOL;
        if rotator_stale || blk.data_vec.get("rot_re").map(|v| v.len()).unwrap_or(0) != nf {
            let omega = blk.data_vec.get("omega").cloned().unwrap_or_default();
            let mut rot_re = Vec::with_capacity(nf);
            let mut rot_im = Vec::with_capacity(nf);
            for &w in &omega {
                let (s, c) = (-w * dt).sin_cos();
                rot_re.push(c);
                rot_im.push(s);
            }
            blk.data_vec.insert("rot_re".to_string(), rot_re);
            blk.data_vec.insert("rot_im".to_string(), rot_im);
            blk.data_vec.insert("phase_new_re".to_string(), vec![0.0; nf]);
            blk.data_vec.insert("phase_new_im".to_string(), vec![0.0; nf]);
            blk.data_f64.insert("dt_cached".to_string(), dt);
            blk.data_f64.insert("decay".to_string(), (-alpha * dt).exp());
        }

        // An empty phase array is the signal that integration has not yet
        // started (fresh block, post-reset, or input width change). Anchor the
        // phase to the current t, snapshot u as u_prev for the next step's
        // trapezoidal rule, and return — no accumulation until an actual dt of
        // simulation time has elapsed.
        if blk.data_vec.get("phase_re").is_none_or(|v| v.is_empty()) {
            let omega = blk.data_vec.get("omega").cloned().unwrap_or_default();
            let mut phase_re = vec![0.0; nf];
            let mut phase_im = vec![0.0; nf];
            for (k, &w) in omega.iter().enumerate() {
                let (s, c) = (-w * t).sin_cos();
                phase_re[k] = c;
                phase_im[k] = s;
            }
            blk.data_vec.insert("phase_re".to_string(), phase_re);
            blk.data_vec.insert("phase_im".to_string(), phase_im);
            if let Some(up) = blk.data_vec.get_mut("u_prev") {
                up.copy_from_slice(&blk.inputs._data);
            }
            // `time` stays 0 — integration has not happened yet.
            return;
        }
        let decay = blk.data_f64.get("decay").copied().unwrap_or(1.0);
        let half_dt = 0.5 * dt;

        // Move-out slices (can't borrow multiple &mut from one HashMap).
        let mut phase_re     = std::mem::take(blk.data_vec.get_mut("phase_re").unwrap());
        let mut phase_im     = std::mem::take(blk.data_vec.get_mut("phase_im").unwrap());
        let mut phase_new_re = std::mem::take(blk.data_vec.get_mut("phase_new_re").unwrap());
        let mut phase_new_im = std::mem::take(blk.data_vec.get_mut("phase_new_im").unwrap());
        let rot_re           = std::mem::take(blk.data_vec.get_mut("rot_re").unwrap());
        let rot_im           = std::mem::take(blk.data_vec.get_mut("rot_im").unwrap());
        let mut x_re         = std::mem::take(blk.data_vec.get_mut("x_re").unwrap());
        let mut x_im         = std::mem::take(blk.data_vec.get_mut("x_im").unwrap());
        let mut u_prev       = std::mem::take(blk.data_vec.get_mut("u_prev").unwrap());

        // Compute phase_new[k] = phase[k] · rotator[k] — the rotated phase at t_curr.
        for k in 0..nf {
            phase_new_re[k] = phase_re[k] * rot_re[k] - phase_im[k] * rot_im[k];
            phase_new_im[k] = phase_re[k] * rot_im[k] + phase_im[k] * rot_re[k];
        }

        // Trapezoidal accumulator for each input:
        //   x[i,k] ← decay·x[i,k]
        //          + (dt/2)·(decay·phase_prev[k]·u_prev[i] + phase_new[k]·u_curr[i])
        let u = &blk.inputs._data;
        for i in 0..ni {
            let upi = u_prev[i];
            let uci = u[i];
            let base = i * nf;
            for k in 0..nf {
                let src_re = decay * phase_re[k] * upi + phase_new_re[k] * uci;
                let src_im = decay * phase_im[k] * upi + phase_new_im[k] * uci;
                x_re[base + k] = decay * x_re[base + k] + half_dt * src_re;
                x_im[base + k] = decay * x_im[base + k] + half_dt * src_im;
            }
        }

        // Commit: phase_new becomes the stored phase for next step; cache u.
        std::mem::swap(&mut phase_re, &mut phase_new_re);
        std::mem::swap(&mut phase_im, &mut phase_new_im);
        u_prev.copy_from_slice(u);

        // Put slices back.
        blk.data_vec.insert("phase_re".to_string(), phase_re);
        blk.data_vec.insert("phase_im".to_string(), phase_im);
        blk.data_vec.insert("phase_new_re".to_string(), phase_new_re);
        blk.data_vec.insert("phase_new_im".to_string(), phase_new_im);
        blk.data_vec.insert("rot_re".to_string(), rot_re);
        blk.data_vec.insert("rot_im".to_string(), rot_im);
        blk.data_vec.insert("x_re".to_string(), x_re);
        blk.data_vec.insert("x_im".to_string(), x_im);
        blk.data_vec.insert("u_prev".to_string(), u_prev);

        blk.data_f64.insert("time".to_string(), t - t_wait);
    }));

    Rc::new(FastCell::new(b))
}

/// Read spectrum data from a Spectrum block.
///
/// Returns (freq, spectra) where spectra is a list of complex spectra per input.
/// Each spectrum element is (real, imag) pairs — returned as Vec<(Vec<f64>, Vec<f64>)>.
pub fn spectrum_read(block: &Block) -> (Vec<f64>, Vec<(Vec<f64>, Vec<f64>)>) {
    let freq = block.data_vec.get("freq").cloned().unwrap_or_default();
    let n_freqs = block.data_f64.get("n_freqs").copied().unwrap_or(0.0) as usize;
    let alpha = block.data_f64.get("alpha").copied().unwrap_or(0.0);
    let time = block.data_f64.get("time").copied().unwrap_or(0.0);

    let x_re = block.data_vec.get("x_re").cloned().unwrap_or_default();
    let x_im = block.data_vec.get("x_im").cloned().unwrap_or_default();
    let n_inputs = x_re.len().checked_div(n_freqs).unwrap_or(0);

    if n_freqs == 0 || time == 0.0 || n_inputs == 0 {
        let zeros = vec![(vec![0.0; n_freqs], vec![0.0; n_freqs]); n_inputs];
        return (freq, zeros);
    }

    let scale = if alpha != 0.0 {
        alpha / (1.0 - (-alpha * time).exp())
    } else {
        1.0 / time
    };

    let mut spectra = Vec::with_capacity(n_inputs);
    for i in 0..n_inputs {
        let base = i * n_freqs;
        let mut re = Vec::with_capacity(n_freqs);
        let mut im = Vec::with_capacity(n_freqs);
        for k in 0..n_freqs {
            re.push(x_re[base + k] * scale);
            im.push(x_im[base + k] * scale);
        }
        spectra.push((re, im));
    }

    (freq, spectra)
}
