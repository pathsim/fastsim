// Discrete-time / event-driven block constructors:
// Delay, SampleHold, FirstOrderHold, Wrapper, FIR, ADC, DAC, Spectrum,
// DiscreteIntegrator, DiscreteDerivative, DiscreteStateSpace,
// DiscreteTransferFunction, TappedDelay.

use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::rc::Rc;

use crate::blocks::block::{Block, BlockRef, BlockRole, UpdateFn};
use crate::error::SimError;
use crate::blocks::blockops::{
    mem_read_alg_graph, EventKindSpec, EventSpec, MemSpec, MemTarget,
};
use crate::constants::INTERPOLATION_TIME_TOL;
use crate::events::schedule::Schedule;
use crate::ssa::build::{Builder, GraphBuilder};
use crate::ssa::graph::{Graph, InputSignature};
use crate::utils::fastcell::FastCell;
use crate::utils::register::Register;

use super::out_port_map;

// ======================================================================================
// Shared scaffold for periodic "held-output" discrete blocks
// ======================================================================================

/// Canonical held-output update: copy the held buffer element-wise into the
/// block's output register. Shared by SampleHold / Wrapper / FIR / ADC, which
/// each compute their held buffer in a scheduled event and hold it between
/// samples.
fn held_vec_update(held: Rc<FastCell<Vec<f64>>>) -> UpdateFn {
    Box::new(move |blk, _t| {
        let h = held.borrow();
        for (i, &v) in h.iter().enumerate() {
            blk.outputs.set_single(i, v);
        }
    })
}

/// Attach a periodic `Schedule` event (first fire at `tau`, then every
/// `period`) whose action is `action`. Centralizes the `Schedule::new` + push
/// that every periodic discrete block otherwise repeats verbatim.
fn add_periodic_schedule(blk: &BlockRef, tau: f64, period: f64, action: Box<dyn FnMut(f64)>) {
    let evt = Schedule::new(tau, None, period, Some(action), crate::constants::TOLERANCE);
    blk.borrow_mut().events.push(Rc::new(FastCell::new(evt)));
}

// ======================================================================================
// Delay: y(t) = u(t - tau)
//
// Two modes (mirrors pathsim):
// - Continuous mode (sampling_period=0): adaptive buffer with linear interpolation
// - Discrete mode (sampling_period>0): ring buffer with scheduled sampling events
// ======================================================================================

/// Delay: y(t) = u(t - tau), with continuous interpolation or discrete ring buffer
pub fn delay(tau: f64, sampling_period: f64) -> BlockRef {
    let mut b = Block::default_block();
    b.type_name = "Delay";
    b.role = BlockRole { is_dyn: false, is_src: false, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));

    if sampling_period <= 0.0 {
        // Continuous mode: adaptive rolling buffer with interpolation
        // Buffer stores (time, value) pairs, interpolates at t - tau
        let buffer: Rc<FastCell<Vec<(f64, f64)>>> = Rc::new(FastCell::new(Vec::new()));
        let buf_upd = buffer.clone();
        let buf_samp = buffer.clone();
        let buf_reset = buffer.clone();

        // update: interpolate buffer at t - tau
        b.update_fn = Some(Box::new(move |blk, t| {
            let target = t - tau;
            let buf = buf_upd.borrow();
            if buf.is_empty() || target < 0.0 {
                blk.outputs._data[0] = 0.0;
                return;
            }
            // Binary search for interpolation point
            let n = buf.len();
            if n == 1 || target <= buf[0].0 {
                blk.outputs._data[0] = buf[0].1;
                return;
            }
            if target >= buf[n - 1].0 {
                blk.outputs._data[0] = buf[n - 1].1;
                return;
            }
            // Find interval [i, i+1] containing target
            let mut lo = 0;
            let mut hi = n - 1;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if buf[mid].0 <= target { lo = mid; } else { hi = mid; }
            }
            // Linear interpolation
            let (t0, v0) = buf[lo];
            let (t1, v1) = buf[hi];
            let frac = if (t1 - t0).abs() > INTERPOLATION_TIME_TOL { (target - t0) / (t1 - t0) } else { 0.0 };
            blk.outputs._data[0] = v0 + frac * (v1 - v0);
        }));

        // sample: add current (t, u) to buffer, prune old entries
        b.sample_fn = Some(Box::new(move |blk, t, _dt| {
            let u = blk.inputs._data[0];
            let buf = buf_samp.borrow_mut();
            buf.push((t, u));
            // Prune entries older than t - tau (keep one extra for interpolation)
            let cutoff = t - tau - crate::constants::DELAY_PRUNE_MARGIN;
            while buf.len() > 2 && buf[0].0 < cutoff {
                buf.remove(0);
            }
        }));

        // reset: clear buffer
        b.reset_fn = Some(Box::new(move |blk| {
            blk.inputs.reset();
            blk.outputs.reset();
            buf_reset.borrow_mut().clear();
        }));

    } else {
        // Discrete mode: ring buffer with scheduled sampling
        use std::collections::VecDeque;

        let n_samples = (tau / sampling_period).round().max(1.0) as usize;
        let mut ring: VecDeque<f64> = VecDeque::with_capacity(n_samples + 1);
        for _ in 0..n_samples { ring.push_back(0.0); }

        let ring = Rc::new(FastCell::new(ring));
        let ring_upd = ring.clone();
        let ring_samp = ring.clone();

        let sample_flag = Rc::new(FastCell::new(false));
        let flag_samp = sample_flag.clone();

        b.update_fn = Some(Box::new(move |blk, _t| {
            blk.outputs._data[0] = ring_upd.borrow().front().copied().unwrap_or(0.0);
        }));

        b.sample_fn = Some(Box::new(move |blk, _t, _dt| {
            if *flag_samp.borrow() {
                *flag_samp.borrow_mut() = false;
                let u = blk.inputs._data[0];
                let r = ring_samp.borrow_mut();
                r.push_back(u);
                while r.len() > n_samples { r.pop_front(); }
            }
        }));

        use crate::events::schedule::Schedule;
        let flag_evt = sample_flag.clone();
        let evt = Schedule::new(
            0.0, None, sampling_period,
            Some(Box::new(move |_t| { *flag_evt.borrow_mut() = true; })),
            crate::constants::TOLERANCE,
        );
        b.events.push(Rc::new(FastCell::new(evt)));

        // IR (Memory + Event): a `ring` slot of n_samples; each sampling period
        // shifts (drop oldest, push u at the back). Output y = ring[0] (oldest).
        {
            let alg = {
                let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([(format!("mem{}", 0u32), n_samples)])));
                let y = {
                    let gb = GraphBuilder::new(&cell);
                    gb.input(0) // ring front (oldest)
                };
                let mut g = cell.into_inner();
                g.outputs.push(y);
                g
            };
            let effect = {
                let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([
                    (format!("mem{}", 0u32), n_samples),
                    ("u".to_string(), 1),
                ])));
                let outs = {
                    let gb = GraphBuilder::new(&cell);
                    let ring: Vec<u32> = (0..n_samples as u32).map(|i| gb.input(i)).collect();
                    let u = gb.input(n_samples as u32);
                    let mut outs: Vec<u32> = ring[1..].to_vec(); // ring[1..n]
                    outs.push(u); // new back = u
                    outs
                };
                let mut g = cell.into_inner();
                g.outputs = outs;
                g
            };
            let targets: Vec<MemTarget> = (0..n_samples as u32).map(|i| MemTarget { slot: 0, offset: i }).collect();
            let memory = vec![MemSpec { name: "ring".into(), init: vec![0.0; n_samples] }];
            let events = vec![EventSpec {
                kind: EventKindSpec::SchedulePeriodic { period: sampling_period, phase: 0.0 },
                effect,
                targets,
            }];
            b.set_discrete("Delay", alg, memory, events);
        }

        // Reset: clear ring and refill with n zeros (pathsim parity with
        // Delay.reset: self._ring.clear(); self._ring.extend([0.0] * self._n)).
        let ring_reset = ring.clone();
        b.reset_fn = Some(Box::new(move |blk| {
            blk.inputs.reset();
            blk.outputs.reset();
            let r = ring_reset.borrow_mut();
            r.clear();
            for _ in 0..n_samples { r.push_back(0.0); }
        }));
    }

    Rc::new(FastCell::new(b))
}

// ======================================================================================
// SampleHold: samples input at fixed intervals, holds between samples
// ======================================================================================

/// SampleHold (alias: ZeroOrderHold) — samples input(s) at fixed intervals
/// and holds them at the output until the next sample. Each input channel
/// is sampled and held independently.
pub fn sample_hold(period: f64, tau: f64) -> BlockRef {
    let held: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(Vec::new()));
    let held_evt = held.clone();
    let held_reset = held.clone();

    let mut b = Block::default_block();
    b.type_name = "SampleHold";
    b.role = BlockRole { is_dyn: false, is_src: false, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));
    b.update_fn = Some(held_vec_update(held.clone()));

    // IR (Memory + Event), shape-poly in the input width `ni`: one memory slot
    // `held`; each period held' = u; alg output y = held.
    b.set_discrete_lazy("SampleHold", move |ni| {
        let alg = mem_read_alg_graph(0, ni);
        let effect = {
            let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", ni)])));
            let outs: Vec<u32> = {
                let gb = GraphBuilder::new(&cell);
                (0..ni as u32).map(|i| gb.input(i)).collect()
            };
            let mut g = cell.into_inner();
            g.outputs = outs;
            g
        };
        let targets: Vec<MemTarget> = (0..ni as u32).map(|i| MemTarget { slot: 0, offset: i }).collect();
        let memory = vec![MemSpec { name: "held".into(), init: vec![0.0; ni] }];
        let events = vec![EventSpec {
            kind: EventKindSpec::SchedulePeriodic { period, phase: tau },
            effect,
            targets,
        }];
        (alg, memory, events)
    });

    let blk_ref: BlockRef = Rc::new(FastCell::new(b));
    let blk_evt = blk_ref.clone();

    add_periodic_schedule(&blk_ref, tau, period, Box::new(move |_t| {
        let inp = blk_evt.borrow().inputs.to_array();
        let h = held_evt.borrow_mut();
        h.clear();
        h.extend_from_slice(&inp);
    }));

    blk_ref.borrow_mut().reset_fn = Some(Box::new(move |blk| {
        blk.inputs.reset();
        blk.outputs.reset();
        held_reset.borrow_mut().clear();
    }));

    blk_ref
}

// Spectrum (Running Fourier Transform) lives in its own module — see spectrum.rs.


// ======================================================================================
// Wrapper: discrete-time block that calls func periodically via Schedule event
// ======================================================================================

/// Wrapper: calls `func` at fixed intervals T, holds output between samples.
/// Unlike Function (evaluated continuously), Wrapper evaluates only at discrete times.
pub fn wrapper(
    func: impl Fn(&[f64]) -> Vec<f64> + 'static,
    period: f64,
    tau: f64,
) -> BlockRef {
    let func = Rc::new(func);
    let func_evt = func.clone();

    // Shared output buffer — written by event, read by update
    let output: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(Vec::new()));
    let output_evt = output.clone();
    let output_reset = output.clone();

    let mut b = Block::default_block();
    b.type_name = "Wrapper";
    b.role = BlockRole { is_dyn: false, is_src: false, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));
    b.update_fn = Some(held_vec_update(output.clone()));

    let blk_ref: BlockRef = Rc::new(FastCell::new(b));
    let blk_evt = blk_ref.clone();

    add_periodic_schedule(&blk_ref, tau, period, Box::new(move |_t| {
        let u_data = &blk_evt.borrow().inputs._data;
        let y = func_evt(u_data);
        let out = output_evt.borrow_mut();
        out.resize(y.len(), 0.0);
        out.copy_from_slice(&y);
    }));

    blk_ref.borrow_mut().reset_fn = Some(Box::new(move |blk| {
        blk.inputs.reset();
        blk.outputs.reset();
        output_reset.borrow_mut().clear();
    }));

    blk_ref
}

// ======================================================================================
// FIR: Finite Impulse Response filter (discrete-time via Schedule event)
// ======================================================================================

/// FIR filter: y[n] = sum(coeffs[k] * x[n-k]) sampled at period T.
///
/// Supports vector inputs — the same coefficients are applied to each
/// channel in parallel.
pub fn fir(coeffs: Vec<f64>, period: f64, tau: f64) -> BlockRef {
    let n = coeffs.len();
    let output: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(Vec::new()));
    let out_upd = output.clone();
    let out_evt = output.clone();
    let out_reset = output.clone();

    // Buffer of past sample vectors. Each entry has the same width once an
    // input has been seen. Initialised to `n` empty vectors and grown lazily
    // on first sample.
    let buffer: Rc<FastCell<VecDeque<Vec<f64>>>> =
        Rc::new(FastCell::new(VecDeque::from(vec![Vec::new(); n])));
    let buf_evt = buffer.clone();
    let buf_reset = buffer.clone();

    let mut b = Block::new(
        Some(HashMap::from([("in".to_string(), 0)])),
        Some(out_port_map()),
    );
    b.type_name = "FIR";
    b.role = BlockRole { is_dyn: false, is_src: false, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));

    b.update_fn = Some(held_vec_update(out_upd));

    // IR (Memory + Event), shape-poly: slots [0]=buffer(N*ni newest-first),
    // [1]=held(ni). Each period: shift buffer (insert u at front) and
    // held[i] = sum_k coeffs[k] * new_buffer[k][i]. alg output y = held.
    {
        let coeffs_ir = coeffs.clone();
        b.set_discrete_lazy("FIR", move |ni| {
            let nn = coeffs_ir.len();
            let alg = mem_read_alg_graph(1, ni);
            let effect = {
                let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([
                    (format!("mem{}", 0u32), nn * ni),
                    ("u".to_string(), ni),
                ])));
                let outs = {
                    let gb = GraphBuilder::new(&cell);
                    let old: Vec<u32> = (0..(nn * ni) as u32).map(|i| gb.input(i)).collect();
                    let uu: Vec<u32> = (0..ni as u32).map(|i| gb.input((nn * ni) as u32 + i)).collect();
                    // held[i] = coeffs[0]*u[i] + sum_{k>=1} coeffs[k]*old[(k-1)*ni+i]
                    let mut held = Vec::with_capacity(ni);
                    for i in 0..ni {
                        let mut acc = gb.mul(gb.cst(coeffs_ir[0]), uu[i]);
                        for k in 1..nn {
                            acc = gb.add(acc, gb.mul(gb.cst(coeffs_ir[k]), old[(k - 1) * ni + i]));
                        }
                        held.push(acc);
                    }
                    // new buffer: [u, old[0..(N-1)*ni])
                    let mut newbuf: Vec<u32> = uu.clone();
                    for k in 1..nn {
                        for i in 0..ni {
                            newbuf.push(old[(k - 1) * ni + i]);
                        }
                    }
                    held.extend_from_slice(&newbuf);
                    held
                };
                let mut g = cell.into_inner();
                g.outputs = outs;
                g
            };
            let targets: Vec<MemTarget> = (0..ni as u32)
                .map(|i| MemTarget { slot: 1, offset: i })
                .chain((0..(nn * ni) as u32).map(|i| MemTarget { slot: 0, offset: i }))
                .collect();
            let memory = vec![
                MemSpec { name: "buffer".into(), init: vec![0.0; nn * ni] },
                MemSpec { name: "held".into(), init: vec![0.0; ni] },
            ];
            let events = vec![EventSpec {
                kind: EventKindSpec::SchedulePeriodic { period, phase: tau },
                effect,
                targets,
            }];
            (alg, memory, events)
        });
    }

    let blk_ref: BlockRef = Rc::new(FastCell::new(b));
    let blk_evt = blk_ref.clone();

    add_periodic_schedule(&blk_ref, tau, period, Box::new(move |_t| {
            let input = blk_evt.borrow().inputs.to_array();
            let width = input.len();

            let buf = buf_evt.borrow_mut();
            //ensure all buffer entries match the current input width
            for entry in buf.iter_mut() {
                if entry.len() != width {
                    entry.clear();
                    entry.resize(width, 0.0);
                }
            }
            buf.push_front(input);
            while buf.len() > n { buf.pop_back(); }

            //y[i] = sum_k coeffs[k] * buf[k][i]
            let mut y = vec![0.0_f64; width];
            for (c, sample) in coeffs.iter().zip(buf.iter()) {
                for i in 0..width {
                    y[i] += c * sample[i];
                }
            }
            let out = out_evt.borrow_mut();
            out.clear();
            out.extend_from_slice(&y);
    }));

    let n_reset = n;
    blk_ref.borrow_mut().reset_fn = Some(Box::new(move |blk| {
        blk.inputs.reset();
        blk.outputs.reset();
        out_reset.borrow_mut().clear();
        let buf = buf_reset.borrow_mut();
        buf.clear();
        for _ in 0..n_reset { buf.push_back(Vec::new()); }
    }));

    blk_ref
}

// ======================================================================================
// ADC: Analog-to-Digital Converter (discrete-time via Schedule event)
// ======================================================================================

/// ADC: quantizes analog input to n_bits binary outputs, sampled at period T.
/// Output port i = bit i (port 0 = LSB, port n_bits-1 = MSB).
pub fn adc(n_bits: usize, span_lo: f64, span_hi: f64, period: f64, tau: f64) -> BlockRef {
    let outputs_held: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(vec![0.0; n_bits]));
    let out_upd = outputs_held.clone();
    let out_evt = outputs_held.clone();
    let out_reset = outputs_held.clone();

    let mut b = Block::new(
        Some(HashMap::from([("in".to_string(), 0)])),
        None,
    );
    b.type_name = "ADC";
    b.role = BlockRole { is_dyn: false, is_src: false, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));

    b.update_fn = Some(held_vec_update(out_upd));

    // IR (Memory + Event): one `bits` slot of size n_bits; each period the
    // sample is quantised to a code and split into bits via float arithmetic
    // (`bit_i = floor(code/2^i) - 2*floor(code/2^(i+1))`, exact for integer code).
    {
        let alg = mem_read_alg_graph(0, n_bits);
        let effect = {
            let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", 1usize)])));
            let outs = {
                let gb = GraphBuilder::new(&cell);
                let u0 = gb.input(0);
                let lo = gb.cst(span_lo);
                let hi = gb.cst(span_hi);
                let clamped = gb.min(gb.max(u0, lo), hi);
                let scaled = gb.div(gb.sub(clamped, lo), gb.sub(hi, lo));
                let levels = (1u64 << n_bits) as f64;
                let code = gb.min(gb.floor(gb.mul(scaled, gb.cst(levels))), gb.cst(levels - 1.0));
                (0..n_bits)
                    .map(|i| {
                        let lo_i = gb.floor(gb.div(code, gb.cst((1u64 << i) as f64)));
                        let lo_i1 = gb.floor(gb.div(code, gb.cst((1u64 << (i + 1)) as f64)));
                        gb.sub(lo_i, gb.mul(gb.cst(2.0), lo_i1))
                    })
                    .collect::<Vec<u32>>()
            };
            let mut g = cell.into_inner();
            g.outputs = outs;
            g
        };
        let targets: Vec<MemTarget> = (0..n_bits as u32).map(|i| MemTarget { slot: 0, offset: i }).collect();
        let memory = vec![MemSpec { name: "bits".into(), init: vec![0.0; n_bits] }];
        let events = vec![EventSpec {
            kind: EventKindSpec::SchedulePeriodic { period, phase: tau },
            effect,
            targets,
        }];
        b.set_discrete("ADC", alg, memory, events);
    }

    let blk_ref: BlockRef = Rc::new(FastCell::new(b));
    let blk_evt = blk_ref.clone();

    add_periodic_schedule(&blk_ref, tau, period, Box::new(move |_t| {
            let analog_in = blk_evt.borrow().inputs.get_single(0);
            let clamped = analog_in.clamp(span_lo, span_hi);
            let scaled = (clamped - span_lo) / (span_hi - span_lo);
            let levels = (1u64 << n_bits) as f64;
            let code = (scaled * levels).floor().min(levels - 1.0) as u64;
            let held = out_evt.borrow_mut();
            held.resize(n_bits, 0.0);
            for i in 0..n_bits {
                held[i] = ((code >> i) & 1) as f64;
            }
    }));

    let nb = n_bits;
    blk_ref.borrow_mut().reset_fn = Some(Box::new(move |blk| {
        blk.inputs.reset();
        blk.outputs.reset();
        let held = out_reset.borrow_mut();
        held.clear();
        held.resize(nb, 0.0);
    }));

    blk_ref
}

// ======================================================================================
// DAC: Digital-to-Analog Converter (discrete-time via Schedule event)
// ======================================================================================

/// DAC: converts n_bits binary inputs to analog output, sampled at period T.
/// Input port i = bit i (port 0 = LSB, port n_bits-1 = MSB).
pub fn dac(n_bits: usize, span_lo: f64, span_hi: f64, period: f64, tau: f64) -> BlockRef {
    let output_val = Rc::new(FastCell::new(0.0_f64));
    let out_upd = output_val.clone();
    let out_evt = output_val.clone();
    let out_reset = output_val.clone();

    let mut b = Block::new(
        None,
        Some(out_port_map()),
    );
    b.type_name = "DAC";
    b.role = BlockRole { is_dyn: false, is_src: false, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));

    b.update_fn = Some(Box::new(move |blk, _t| {
        blk.outputs.set_single(0, *out_upd.borrow());
    }));

    // IR (Memory + Event): one `held` output slot; each period reads the bit
    // inputs, forms a code, and maps it onto [span_lo, span_hi].
    {
        let alg = mem_read_alg_graph(0, 1);
        let effect = {
            let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", n_bits)])));
            let y = {
                let gb = GraphBuilder::new(&cell);
                let half = gb.cst(0.5);
                let mut code: Option<u32> = None;
                for i in 0..n_bits {
                    let bit = gb.gt(gb.input(i as u32), half);
                    let term = gb.mul(bit, gb.cst((1u64 << i) as f64));
                    code = Some(match code {
                        None => term,
                        Some(a) => gb.add(a, term),
                    });
                }
                let code = code.unwrap_or_else(|| gb.cst(0.0));
                let levels = (1u64 << n_bits) as f64;
                let scaled = gb.div(code, gb.cst(levels - 1.0));
                let lo = gb.cst(span_lo);
                gb.add(lo, gb.mul(gb.cst(span_hi - span_lo), scaled))
            };
            let mut g = cell.into_inner();
            g.outputs.push(y);
            g
        };
        let memory = vec![MemSpec { name: "held".into(), init: vec![0.0] }];
        let events = vec![EventSpec {
            kind: EventKindSpec::SchedulePeriodic { period, phase: tau },
            effect,
            targets: vec![MemTarget { slot: 0, offset: 0 }],
        }];
        b.set_discrete("DAC", alg, memory, events);
    }

    let blk_ref: BlockRef = Rc::new(FastCell::new(b));
    let blk_evt = blk_ref.clone();

    add_periodic_schedule(&blk_ref, tau, period, Box::new(move |_t| {
        let blk = blk_evt.borrow();
        let mut code: u64 = 0;
        for i in 0..n_bits {
            if blk.inputs.get_single(i) > 0.5 {
                code |= 1u64 << i;
            }
        }
        let levels = (1u64 << n_bits) as f64;
        let scaled = if levels > 1.0 { code as f64 / (levels - 1.0) } else { 0.0 };
        *out_evt.borrow_mut() = span_lo + (span_hi - span_lo) * scaled;
    }));

    blk_ref.borrow_mut().reset_fn = Some(Box::new(move |blk| {
        blk.inputs.reset();
        blk.outputs.reset();
        *out_reset.borrow_mut() = 0.0;
    }));

    blk_ref
}

// ======================================================================================
// FirstOrderHold: causal linear extrapolation between two consecutive samples
// ======================================================================================

/// FirstOrderHold — reconstruct a continuous signal from periodic samples
/// by linear extrapolation across one sampling interval. Causal
/// (one-sample-lag) variant matching the Simulink ``First-Order Hold`` block.
///
/// Between two consecutive sample times t_{k-1} and t_k:
///   y(t) = u_{k-1} + (u_{k-1} - u_{k-2}) / T · (t - t_{k-1})
///
/// During the very first interval (only one sample captured) the output
/// is held at the most recent sample.
///
/// Supports vector inputs.
pub fn first_order_hold(period: f64, tau: f64) -> BlockRef {
    // Shared state: previous sample, current sample, time of latest sample,
    // number of samples captured so far.
    let u_prev: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(Vec::new()));
    let u_curr: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(Vec::new()));
    let t_curr: Rc<FastCell<f64>> = Rc::new(FastCell::new(tau));
    let n_samples: Rc<FastCell<usize>> = Rc::new(FastCell::new(0));

    let up_upd = u_prev.clone();
    let uc_upd = u_curr.clone();
    let tc_upd = t_curr.clone();
    let ns_upd = n_samples.clone();

    let mut b = Block::default_block();
    b.type_name = "FirstOrderHold";
    b.role = BlockRole { is_dyn: false, is_src: false, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));

    b.update_fn = Some(Box::new(move |blk, t| {
        let n = *ns_upd.borrow();
        let uc = uc_upd.borrow();
        if n < 2 {
            for (i, &v) in uc.iter().enumerate() {
                blk.outputs.set_single(i, v);
            }
            return;
        }
        let up = up_upd.borrow();
        let tc = *tc_upd.borrow();
        let dt = t - tc;
        let inv_t = 1.0 / period;
        for i in 0..uc.len() {
            let slope = (uc[i] - up[i]) * inv_t;
            blk.outputs.set_single(i, uc[i] + slope * dt);
        }
    }));

    // IR (Memory + Event), shape-poly: slots [0]=u_prev, [1]=u_curr, [2]=t_curr,
    // [3]=n. Each period: u_prev<-u_curr, u_curr<-u, t_curr<-t, n<-n+1. alg
    // linearly extrapolates y = u_curr + slope*(t - t_curr) once n>=2, else u_curr.
    {
        b.set_discrete_lazy("FirstOrderHold", move |ni| {
            let inv_t = 1.0 / period;
            let alg = {
                let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([
                    (format!("mem{}", 0u32), ni),
                    (format!("mem{}", 1u32), ni),
                    (format!("mem{}", 2u32), 1usize),
                    (format!("mem{}", 3u32), 1usize),
                    ("t".to_string(), 1usize),
                ])));
                let outs = {
                    let gb = GraphBuilder::new(&cell);
                    let up: Vec<u32> = (0..ni as u32).map(|i| gb.input(i)).collect();
                    let uc: Vec<u32> = (0..ni as u32).map(|i| gb.input(ni as u32 + i)).collect();
                    let tc = gb.input(2 * ni as u32);
                    let n = gb.input(2 * ni as u32 + 1);
                    let t = gb.input(2 * ni as u32 + 2);
                    let ge_n = gb.ge(n, gb.cst(2.0));
                    let dt = gb.sub(t, tc);
                    let inv = gb.cst(inv_t);
                    (0..ni)
                        .map(|i| {
                            let slope = gb.mul(gb.sub(uc[i], up[i]), inv);
                            let interp = gb.add(uc[i], gb.mul(slope, dt));
                            gb.select(ge_n, interp, uc[i])
                        })
                        .collect::<Vec<u32>>()
                };
                let mut g = cell.into_inner();
                g.outputs = outs;
                g
            };
            let effect = {
                let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([
                    (format!("mem{}", 1u32), ni),
                    ("u".to_string(), ni),
                    (format!("mem{}", 3u32), 1usize),
                    ("t".to_string(), 1usize),
                ])));
                let outs = {
                    let gb = GraphBuilder::new(&cell);
                    let uc: Vec<u32> = (0..ni as u32).map(|i| gb.input(i)).collect();
                    let uu: Vec<u32> = (0..ni as u32).map(|i| gb.input(ni as u32 + i)).collect();
                    let n = gb.input(2 * ni as u32);
                    let t = gb.input(2 * ni as u32 + 1);
                    let mut outs = uc.clone(); // u_prev' = u_curr
                    outs.extend_from_slice(&uu); // u_curr' = u
                    outs.push(t); // t_curr' = t
                    outs.push(gb.add(n, gb.cst(1.0))); // n' = n + 1
                    outs
                };
                let mut g = cell.into_inner();
                g.outputs = outs;
                g
            };
            let mut targets: Vec<MemTarget> = (0..ni as u32).map(|i| MemTarget { slot: 0, offset: i }).collect();
            targets.extend((0..ni as u32).map(|i| MemTarget { slot: 1, offset: i }));
            targets.push(MemTarget { slot: 2, offset: 0 });
            targets.push(MemTarget { slot: 3, offset: 0 });
            let memory = vec![
                MemSpec { name: "u_prev".into(), init: vec![0.0; ni] },
                MemSpec { name: "u_curr".into(), init: vec![0.0; ni] },
                MemSpec { name: "t_curr".into(), init: vec![tau] },
                MemSpec { name: "n".into(), init: vec![0.0] },
            ];
            let events = vec![EventSpec {
                kind: EventKindSpec::SchedulePeriodic { period, phase: tau },
                effect,
                targets,
            }];
            (alg, memory, events)
        });
    }

    let blk_ref: BlockRef = Rc::new(FastCell::new(b));
    let blk_evt = blk_ref.clone();

    let up_evt = u_prev.clone();
    let uc_evt = u_curr.clone();
    let tc_evt = t_curr.clone();
    let ns_evt = n_samples.clone();

    add_periodic_schedule(&blk_ref, tau, period, Box::new(move |t| {
        let inp = blk_evt.borrow().inputs.to_array();
        let up = up_evt.borrow_mut();
        let uc = uc_evt.borrow_mut();
        //prev <- curr, curr <- new
        up.clear();
        up.extend_from_slice(uc);
        uc.clear();
        uc.extend_from_slice(&inp);
        *tc_evt.borrow_mut() = t;
        *ns_evt.borrow_mut() += 1;
    }));

    let up_rst = u_prev.clone();
    let uc_rst = u_curr.clone();
    let tc_rst = t_curr.clone();
    let ns_rst = n_samples.clone();
    blk_ref.borrow_mut().reset_fn = Some(Box::new(move |blk| {
        blk.inputs.reset();
        blk.outputs.reset();
        up_rst.borrow_mut().clear();
        uc_rst.borrow_mut().clear();
        *tc_rst.borrow_mut() = tau;
        *ns_rst.borrow_mut() = 0;
    }));

    blk_ref
}

// ======================================================================================
// DiscreteIntegrator: y[k+1] = y[k] + T · u[k] (forward Euler)
// ======================================================================================

/// DiscreteIntegrator — forward-Euler discrete-time integrator.
///
///   y[k+1] = y[k] + T · u[k]
///
/// The output at sample `k` is the accumulated sum of past inputs; the
/// current input `u[k]` only enters at the next sample.
///
/// Supports vector inputs (per-channel state).
pub fn discrete_integrator(period: f64, tau: f64, initial_value: Vec<f64>) -> BlockRef {
    let iv = if initial_value.is_empty() { vec![0.0] } else { initial_value };

    let state: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(iv.clone()));
    let held: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(iv.clone()));

    let st_evt = state.clone();
    let st_reset = state.clone();
    let held_upd = held.clone();
    let held_evt = held.clone();
    let held_reset = held.clone();
    let iv_reset = iv.clone();
    let iv_evt = iv.clone();

    let mut b = Block::default_block();
    b.type_name = "DiscreteIntegrator";
    b.role = BlockRole { is_dyn: false, is_src: false, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));

    //prime output to initial value so it is visible before the first event
    for (i, &v) in iv.iter().enumerate() {
        b.outputs.set_single(i, v);
    }

    b.update_fn = Some(held_vec_update(held_upd));

    // IR (Memory + Event): memory slots [0]=state, [1]=held. Each period:
    // held' = state; state' = state + period*u. alg output: y = held.
    {
        let n = iv.len();
        let alg = mem_read_alg_graph(1, n);
        let effect = {
            let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([
                (format!("mem{}", 0u32), n),
                ("u".to_string(), n),
            ])));
            let outs = {
                let gb = GraphBuilder::new(&cell);
                let st: Vec<u32> = (0..n as u32).map(|i| gb.input(i)).collect();
                let uu: Vec<u32> = (0..n as u32).map(|i| gb.input(n as u32 + i)).collect();
                let p = gb.cst(period);
                let mut outs = Vec::with_capacity(2 * n);
                for &s in &st {
                    outs.push(s); // held' = state
                }
                for i in 0..n {
                    outs.push(gb.add(st[i], gb.mul(p, uu[i]))); // state' = state + period*u
                }
                outs
            };
            let mut g = cell.into_inner();
            g.outputs = outs;
            g
        };
        let targets: Vec<MemTarget> = (0..n as u32)
            .map(|i| MemTarget { slot: 1, offset: i })
            .chain((0..n as u32).map(|i| MemTarget { slot: 0, offset: i }))
            .collect();
        let memory = vec![
            MemSpec { name: "state".into(), init: iv.clone() },
            MemSpec { name: "held".into(), init: iv.clone() },
        ];
        let events = vec![EventSpec {
            kind: EventKindSpec::SchedulePeriodic { period, phase: tau },
            effect,
            targets,
        }];
        b.set_discrete("DiscreteIntegrator", alg, memory, events);
    }

    let blk_ref: BlockRef = Rc::new(FastCell::new(b));
    let blk_evt = blk_ref.clone();

    add_periodic_schedule(&blk_ref, tau, period, Box::new(move |_t| {
        let inp = blk_evt.borrow().inputs.to_array();
        let s = st_evt.borrow_mut();
        //grow state to match input width if needed
        if s.len() < inp.len() {
            let pad = inp.len() - s.len();
            for k in 0..pad {
                let idx = s.len() + k;
                s.push(if idx < iv_evt.len() { iv_evt[idx] } else { 0.0 });
            }
        }
        //y[k] is the state before advancing
        let h = held_evt.borrow_mut();
        h.clear();
        h.extend_from_slice(s);
        //advance: x[k+1] = x[k] + T · u[k]
        for i in 0..s.len().min(inp.len()) {
            s[i] += period * inp[i];
        }
    }));

    blk_ref.borrow_mut().reset_fn = Some(Box::new(move |blk| {
        blk.inputs.reset();
        blk.outputs.reset();
        let s = st_reset.borrow_mut();
        s.clear();
        s.extend_from_slice(&iv_reset);
        let h = held_reset.borrow_mut();
        h.clear();
        h.extend_from_slice(&iv_reset);
    }));

    blk_ref
}

// ======================================================================================
// DiscreteDerivative: y[k] = (u[k] - u[k-1]) / T
// ======================================================================================

/// DiscreteDerivative — backward-difference derivative of a periodically
/// sampled signal:
///
///   y[k] = (u[k] - u[k-1]) / T
///
/// Supports vector inputs (per-channel state).
pub fn discrete_derivative(period: f64, tau: f64) -> BlockRef {
    let prev: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(Vec::new()));
    let out: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(Vec::new()));
    let prev_evt = prev.clone();
    let prev_reset = prev.clone();
    let out_upd = out.clone();
    let out_evt = out.clone();
    let out_reset = out.clone();

    let mut b = Block::default_block();
    b.type_name = "DiscreteDerivative";
    b.role = BlockRole { is_dyn: false, is_src: false, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));

    b.update_fn = Some(held_vec_update(out_upd));

    // IR (Memory + Event), shape-poly: slots [0]=out, [1]=prev. Each period:
    // out' = (u - prev)/period; prev' = u. alg output y = out.
    b.set_discrete_lazy("DiscreteDerivative", move |ni| {
        let inv_t = 1.0 / period;
        let alg = mem_read_alg_graph(0, ni);
        let effect = {
            let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([
                (format!("mem{}", 1u32), ni),
                ("u".to_string(), ni),
            ])));
            let outs = {
                let gb = GraphBuilder::new(&cell);
                let prev: Vec<u32> = (0..ni as u32).map(|i| gb.input(i)).collect();
                let uu: Vec<u32> = (0..ni as u32).map(|i| gb.input(ni as u32 + i)).collect();
                let inv = gb.cst(inv_t);
                let mut outs = Vec::with_capacity(2 * ni);
                for i in 0..ni {
                    outs.push(gb.mul(gb.sub(uu[i], prev[i]), inv)); // out' = (u-prev)/T
                }
                outs.extend_from_slice(&uu); // prev' = u
                outs
            };
            let mut g = cell.into_inner();
            g.outputs = outs;
            g
        };
        let targets: Vec<MemTarget> = (0..ni as u32)
            .map(|i| MemTarget { slot: 0, offset: i })
            .chain((0..ni as u32).map(|i| MemTarget { slot: 1, offset: i }))
            .collect();
        let memory = vec![
            MemSpec { name: "out".into(), init: vec![0.0; ni] },
            MemSpec { name: "prev".into(), init: vec![0.0; ni] },
        ];
        let events = vec![EventSpec {
            kind: EventKindSpec::SchedulePeriodic { period, phase: tau },
            effect,
            targets,
        }];
        (alg, memory, events)
    });

    let blk_ref: BlockRef = Rc::new(FastCell::new(b));
    let blk_evt = blk_ref.clone();

    let inv_t = 1.0 / period;
    add_periodic_schedule(&blk_ref, tau, period, Box::new(move |_t| {
        let inp = blk_evt.borrow().inputs.to_array();
        let p = prev_evt.borrow_mut();
        //pad prev with zeros if input width changed
        if p.len() < inp.len() { p.resize(inp.len(), 0.0); }
        let o = out_evt.borrow_mut();
        o.clear();
        for i in 0..inp.len() {
            o.push((inp[i] - p[i]) * inv_t);
        }
        //store current as previous
        p.clear();
        p.extend_from_slice(&inp);
    }));

    blk_ref.borrow_mut().reset_fn = Some(Box::new(move |blk| {
        blk.inputs.reset();
        blk.outputs.reset();
        prev_reset.borrow_mut().clear();
        out_reset.borrow_mut().clear();
    }));

    blk_ref
}

// ======================================================================================
// DiscreteStateSpace: x[k+1] = A x[k] + B u[k], y[k] = C x[k] + D u[k]
// ======================================================================================

/// DiscreteStateSpace — discrete-time MIMO state-space block:
///
///   x[k+1] = A x[k] + B u[k]
///   y[k]   = C x[k] + D u[k]
///
/// `a, b_mat, c, d` are flat row-major matrices of shape (ns,ns), (ns,ni),
/// (no,ns), (no,ni) respectively. The output is held between sample events
/// (no algebraic passthrough).
pub fn discrete_state_space(
    a: Vec<f64>, b_mat: Vec<f64>, c: Vec<f64>, d: Vec<f64>,
    ns: usize, ni: usize, no: usize,
    period: f64, tau: f64,
    initial_value: Option<Vec<f64>>,
) -> Result<BlockRef, SimError> {
    use super::lti::{flat_to_mat, matvec_into};

    let iv = initial_value.unwrap_or_else(|| vec![0.0; ns]);
    if iv.len() != ns {
        return Err(SimError::InvalidBlockParam(format!(
            "DiscreteStateSpace: initial_value must have length ns={ns} (got {})", iv.len())));
    }

    let mat_a = flat_to_mat(&a, ns, ns);
    let mat_b = flat_to_mat(&b_mat, ns, ni);
    let mat_c = flat_to_mat(&c, no, ns);
    let mat_d = flat_to_mat(&d, no, ni);

    let state: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(iv.clone()));
    let out: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(vec![0.0; no]));

    let st_evt = state.clone();
    let st_reset = state.clone();
    let out_upd = out.clone();
    let out_evt = out.clone();
    let out_reset = out.clone();

    let mut b = Block::new(None, None);
    b.type_name = "DiscreteStateSpace";
    b.inputs = Register::new(Some(ni), None);
    b.outputs = Register::new(Some(no), None);
    b.role = BlockRole { is_dyn: false, is_src: false, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));

    b.update_fn = Some(held_vec_update(out_upd));

    // IR (Memory + Event): slots [0]=state(ns), [1]=held_out(no). Each period:
    // held_out' = C*state + D*u; state' = A*state + B*u (both use the old state).
    {
        use super::lti::affine_mv_eval;
        let alg = mem_read_alg_graph(1, no);
        let effect = {
            let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([
                (format!("mem{}", 0u32), ns),
                ("u".to_string(), ni),
            ])));
            let outs = {
                let gb = GraphBuilder::new(&cell);
                let st: Vec<u32> = (0..ns as u32).map(|i| gb.input(i)).collect();
                let uu: Vec<u32> = (0..ni as u32).map(|i| gb.input(ns as u32 + i)).collect();
                let mut held_out = Vec::new();
                affine_mv_eval(&gb, &c, ns, &d, ni, no, &st, &uu, &mut held_out);
                let mut state_out = Vec::new();
                affine_mv_eval(&gb, &a, ns, &b_mat, ni, ns, &st, &uu, &mut state_out);
                held_out.extend_from_slice(&state_out);
                held_out
            };
            let mut g = cell.into_inner();
            g.outputs = outs;
            g
        };
        let targets: Vec<MemTarget> = (0..no as u32)
            .map(|i| MemTarget { slot: 1, offset: i })
            .chain((0..ns as u32).map(|i| MemTarget { slot: 0, offset: i }))
            .collect();
        let memory = vec![
            MemSpec { name: "state".into(), init: iv.clone() },
            MemSpec { name: "held_out".into(), init: vec![0.0; no] },
        ];
        let events = vec![EventSpec {
            kind: EventKindSpec::SchedulePeriodic { period, phase: tau },
            effect,
            targets,
        }];
        b.set_discrete("DiscreteStateSpace", alg, memory, events);
    }

    let blk_ref: BlockRef = Rc::new(FastCell::new(b));
    let blk_evt = blk_ref.clone();

    let iv_reset = iv.clone();
    let mut scratch_cu = vec![0.0; no];
    let mut scratch_du = vec![0.0; no];
    let mut scratch_ax = vec![0.0; ns];
    let mut scratch_bu = vec![0.0; ns];
    add_periodic_schedule(&blk_ref, tau, period, Box::new(move |_t| {
        let inp = blk_evt.borrow().inputs.to_array();
        //pad u to ni in case fewer connections than declared
        let mut u = inp;
        if u.len() < ni { u.resize(ni, 0.0); }

        let s = st_evt.borrow_mut();
        //y[k] = C x[k] + D u[k]
        matvec_into(&mut scratch_cu, &mat_c, s);
        matvec_into(&mut scratch_du, &mat_d, &u);
        let o = out_evt.borrow_mut();
        o.clear();
        for i in 0..no { o.push(scratch_cu[i] + scratch_du[i]); }

        //x[k+1] = A x[k] + B u[k]
        matvec_into(&mut scratch_ax, &mat_a, s);
        matvec_into(&mut scratch_bu, &mat_b, &u);
        for i in 0..ns { s[i] = scratch_ax[i] + scratch_bu[i]; }
    }));

    let no_reset = no;
    blk_ref.borrow_mut().reset_fn = Some(Box::new(move |blk| {
        blk.inputs.reset();
        blk.outputs.reset();
        let s = st_reset.borrow_mut();
        s.clear();
        s.extend_from_slice(&iv_reset);
        let o = out_reset.borrow_mut();
        o.clear();
        o.resize(no_reset, 0.0);
    }));

    Ok(blk_ref)
}

// ======================================================================================
// DiscreteTransferFunction: H(z) = Num(z)/Den(z) realised as DiscreteStateSpace
// ======================================================================================

/// DiscreteTransferFunction — SISO discrete transfer function in
/// num/den form (descending powers of z), realised as a
/// `DiscreteStateSpace` via the controllable canonical form returned by
/// the same `tf2ss` routine used for analog transfer functions.
pub fn discrete_transfer_function(num: &[f64], den: &[f64], period: f64, tau: f64) -> Result<BlockRef, SimError> {
    use super::lti::tf2ss_scipy_raw;

    if den.is_empty() {
        return Err(SimError::InvalidBlockParam(
            "DiscreteTransferFunction: denominator must not be empty".to_string()));
    }
    if num.is_empty() {
        return Err(SimError::InvalidBlockParam(
            "DiscreteTransferFunction: numerator must not be empty".to_string()));
    }
    if num.len() > den.len() {
        return Err(SimError::InvalidBlockParam(format!(
            "DiscreteTransferFunction: numerator degree ({}) > denominator degree ({})",
            num.len() - 1, den.len() - 1)));
    }

    let n = den.len() - 1;
    let blk = if n == 0 {
        //pure gain: emit as one-state pass-through so the discrete-time
        //sample/hold semantics still apply uniformly.
        let gain = if den[0] != 0.0 { num[0] / den[0] } else { 0.0 };
        discrete_state_space(
            vec![0.0], vec![0.0], vec![0.0], vec![gain],
            1, 1, 1, period, tau, Some(vec![0.0]),
        )?
    } else {
        let (num_norm, den_norm) = crate::blocks::constructors::lti::normalize_tf(num, den);
        let (a, b_mat, c, d) = tf2ss_scipy_raw(&num_norm, &den_norm, n);
        discrete_state_space(a, b_mat, c, d, n, 1, 1, period, tau, None)?
    };

    blk.borrow_mut().type_name = "DiscreteTransferFunction";
    Ok(blk)
}

// ======================================================================================
// TappedDelay: N parallel outputs holding the current and N-1 past samples
// ======================================================================================

/// TappedDelay — single-input, N-output delay line.  Outputs the current
/// and N-1 past samples in parallel:
///
///   y_i[k] = u[k - i],   i = 0, 1, …, N-1
///
/// SISO only — vector input would need a 2-D output layout that does not
/// fit the flat output register.
pub fn tapped_delay(n: usize, period: f64, tau: f64) -> Result<BlockRef, SimError> {
    if n < 1 {
        return Err(SimError::InvalidBlockParam(
            "TappedDelay: n must be >= 1".to_string()));
    }

    let buffer: Rc<FastCell<VecDeque<f64>>> =
        Rc::new(FastCell::new(VecDeque::from(vec![0.0; n])));
    let buf_evt = buffer.clone();
    let buf_reset = buffer.clone();

    let outputs_held: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(vec![0.0; n]));
    let out_upd = outputs_held.clone();
    let out_evt = outputs_held.clone();
    let out_reset = outputs_held.clone();

    let mut b = Block::new(
        Some(HashMap::from([("in".to_string(), 0)])),
        None,
    );
    b.type_name = "TappedDelay";
    b.outputs = Register::new(Some(n), None);
    b.role = BlockRole { is_dyn: false, is_src: false, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));

    b.update_fn = Some(held_vec_update(out_upd));

    // IR (Memory + Event): one `buf` slot (n taps). Each period shifts right
    // and inserts u at the front: buf' = [u, buf[0], ..., buf[n-2]]. y = buf.
    {
        let alg = mem_read_alg_graph(0, n);
        let effect = {
            let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([
                (format!("mem{}", 0u32), n),
                ("u".to_string(), 1),
            ])));
            let outs = {
                let gb = GraphBuilder::new(&cell);
                let buf: Vec<u32> = (0..n as u32).map(|i| gb.input(i)).collect();
                let u = gb.input(n as u32);
                let mut outs = Vec::with_capacity(n);
                outs.push(u); // buf'[0] = u
                for i in 1..n {
                    outs.push(buf[i - 1]); // buf'[i] = buf[i-1]
                }
                outs
            };
            let mut g = cell.into_inner();
            g.outputs = outs;
            g
        };
        let targets: Vec<MemTarget> = (0..n as u32).map(|i| MemTarget { slot: 0, offset: i }).collect();
        let memory = vec![MemSpec { name: "buf".into(), init: vec![0.0; n] }];
        let events = vec![EventSpec {
            kind: EventKindSpec::SchedulePeriodic { period, phase: tau },
            effect,
            targets,
        }];
        b.set_discrete("TappedDelay", alg, memory, events);
    }

    let blk_ref: BlockRef = Rc::new(FastCell::new(b));
    let blk_evt = blk_ref.clone();

    let n_taps = n;
    add_periodic_schedule(&blk_ref, tau, period, Box::new(move |_t| {
        let u = blk_evt.borrow().inputs.get_single(0);
        let buf = buf_evt.borrow_mut();
        buf.push_front(u);
        while buf.len() > n_taps { buf.pop_back(); }
        let held = out_evt.borrow_mut();
        held.clear();
        for i in 0..n_taps { held.push(buf[i]); }
    }));

    let n_reset = n;
    blk_ref.borrow_mut().reset_fn = Some(Box::new(move |blk| {
        blk.inputs.reset();
        blk.outputs.reset();
        let buf = buf_reset.borrow_mut();
        buf.clear();
        for _ in 0..n_reset { buf.push_back(0.0); }
        let held = out_reset.borrow_mut();
        held.clear();
        held.resize(n_reset, 0.0);
    }));

    Ok(blk_ref)
}
