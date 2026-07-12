// Deterministic source constructors: step, sinusoidal, triangle, square,
// pulse, clock, chirp, gaussian-pulse.  All delegate to the primitive
// `source(f(t))` from the parent module.

use std::collections::HashMap;
use std::rc::Rc;

use std::cell::RefCell;

use crate::blocks::block::{Block, BlockRef, BlockRole};
use crate::error::SimError;
use crate::blocks::blockops::{
    mem_read_alg_graph, EventKindSpec, EventSpec, MemSpec, MemTarget,
};
use crate::constants::SOURCE_RISE_FALL_TIME_MIN;
use crate::ssa::build::{Builder, F64Builder, GraphBuilder};
use crate::ssa::graph::{Graph, InputSignature};
use crate::utils::fastcell::FastCell;
use crate::utils::rng::Rng;

use super::noise::{noise_normal_eval, noise_seed_phase, noise_step_key};
use super::out_port_map;

/// SinusoidalSource math (single source): y = amplitude*sin(2*pi*freq*t + phase).
pub(crate) fn sinusoidal_eval<B: Builder>(
    b: &B,
    freq: B::N,
    amp: B::N,
    phase: B::N,
    t: B::N,
    out: &mut Vec<B::N>,
) {
    out.clear();
    let two_pi = b.cst(std::f64::consts::TAU);
    let w = b.mul(two_pi, freq); // 2*pi*f
    let wt = b.mul(w, t); // 2*pi*f*t
    let arg = b.add(wt, phase); // + phase
    let s = b.sin(arg);
    out.push(b.mul(amp, s));
}

/// SinusoidalSource op-graph: frequency / amplitude / phase as runtime-mutable
/// Params, time on slot "t".
pub(crate) fn sinusoidal_source_graph(frequency: f64, amplitude: f64, phase: f64) -> Graph {
    let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("t", 1usize)])));
    {
        let mut g = cell.borrow_mut();
        g.n_params = 3;
        g.param_defaults = vec![frequency, amplitude, phase];
        g.param_names = vec!["frequency".into(), "amplitude".into(), "phase".into()];
    }
    let gb = GraphBuilder::new(&cell);
    let t = gb.input(0);
    let freq = gb.param(0);
    let amp = gb.param(1);
    let ph = gb.param(2);
    let mut out = Vec::new();
    sinusoidal_eval(&gb, freq, amp, ph, t, &mut out);
    let mut g = cell.into_inner();
    g.outputs = out;
    g
}

// ======================================================================================

/// StepSource: piecewise-constant output that switches amplitudes at scheduled times
pub fn step_source(amplitudes: Vec<f64>, times: Vec<f64>) -> Result<BlockRef, SimError> {
    if amplitudes.len() != times.len() {
        return Err(SimError::InvalidBlockParam(format!(
            "StepSource: amplitude ({}) and tau ({}) must have the same length",
            amplitudes.len(), times.len())));
    }

    let output_val: Rc<FastCell<f64>> = Rc::new(FastCell::new(0.0));
    let out_upd = output_val.clone();
    let out_evt = output_val.clone();

    let mut b = Block::new(
        Some(HashMap::new()),
        Some(out_port_map()),
    );
    b.type_name = "StepSource";
    b.role = BlockRole { is_dyn: false, is_src: true, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));

    // update: output held value
    b.update_fn = Some(Box::new(move |blk, _t| {
        blk.outputs._data[0] = *out_upd.borrow();
    }));

    // IR (Memory + Event): slots [0]=out, [1]=cnt. At each scheduled time the
    // output advances to amplitudes[cnt] (held otherwise) and cnt increments.
    {
        let amps = amplitudes.clone();
        let alg = mem_read_alg_graph(0, 1);
        let effect = {
            let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([
                (format!("mem{}", 0u32), 1usize),
                (format!("mem{}", 1u32), 1usize),
            ])));
            let outs = {
                let gb = GraphBuilder::new(&cell);
                let old_out = gb.input(0);
                let cnt = gb.input(1);
                let mut acc = old_out;
                for k in (0..amps.len()).rev() {
                    acc = gb.select(gb.eq(cnt, gb.cst(k as f64)), gb.cst(amps[k]), acc);
                }
                let cnt1 = gb.add(cnt, gb.cst(1.0));
                vec![acc, cnt1]
            };
            let mut g = cell.into_inner();
            g.outputs = outs;
            g
        };
        let memory = vec![
            MemSpec { name: "out".into(), init: vec![0.0] },
            MemSpec { name: "cnt".into(), init: vec![0.0] },
        ];
        let events = vec![EventSpec {
            kind: EventKindSpec::ScheduleFixed(times.clone()),
            effect,
            targets: vec![MemTarget { slot: 0, offset: 0 }, MemTarget { slot: 1, offset: 0 }],
        }];
        b.set_discrete("StepSource", alg, memory, events);
    }

    // Internal ScheduleList event: fires at each time, counter tracks index
    use crate::events::schedule::ScheduleList;
    let counter: Rc<FastCell<usize>> = Rc::new(FastCell::new(0));
    let cnt = counter.clone();
    let amps = amplitudes.clone();

    let evt = ScheduleList::new(
        times,
        Some(Box::new(move |_t| {
            let idx = *cnt.borrow();
            if idx < amps.len() {
                *out_evt.borrow_mut() = amps[idx];
                *cnt.borrow_mut() = idx + 1;
            }
        })),
        crate::constants::TOLERANCE,
    );
    b.events.push(Rc::new(FastCell::new(evt)));

    let out_reset = output_val.clone();
    let cnt_reset = counter.clone();
    b.reset_fn = Some(Box::new(move |blk| {
        blk.inputs.reset();
        blk.outputs.reset();
        *out_reset.borrow_mut() = 0.0;
        *cnt_reset.borrow_mut() = 0;
    }));

    Ok(Rc::new(FastCell::new(b)))
}

/// TriangleWave math: y = A*(2*|tt - floor(tt+0.5)| - 1), tt = (t+tau)*freq.
pub(crate) fn triangle_eval<B: Builder>(b: &B, freq: B::N, amp: B::N, tau: B::N, t: B::N) -> B::N {
    let tt = b.mul(b.add(t, tau), freq);
    let fl = b.floor(b.add(tt, b.cst(0.5)));
    let inner = b.abs(b.sub(tt, fl));
    b.mul(amp, b.sub(b.mul(b.cst(2.0), inner), b.cst(1.0)))
}

/// SquareWave math: y = A if frac((t+tau)*freq) < 0.5 else -A.
pub(crate) fn square_eval<B: Builder>(b: &B, amp: B::N, freq: B::N, tau: B::N, t: B::N) -> B::N {
    let y = b.mul(b.add(t, tau), freq);
    let frac = b.sub(y, b.floor(y));
    b.select(b.lt(frac, b.cst(0.5)), amp, b.neg(amp))
}

/// Clock math: y = 1 if frac(t/period) < 0.5 else 0.
pub(crate) fn clock_eval<B: Builder>(b: &B, period: B::N, t: B::N) -> B::N {
    let y = b.div(t, period);
    let frac = b.sub(y, b.floor(y));
    b.select(b.lt(frac, b.cst(0.5)), b.cst(1.0), b.cst(0.0))
}

/// GaussianPulse math: y = A*exp(-((t-tau)/sigma)^2).
pub(crate) fn gaussian_pulse_eval<B: Builder>(b: &B, amp: B::N, tau: B::N, sigma: B::N, t: B::N) -> B::N {
    let dt = b.sub(t, tau);
    let r = b.div(dt, sigma);
    b.mul(amp, b.exp(b.neg(b.mul(r, r))))
}

/// Build a single-output, time-driven source graph: one `"t"` input slot and
/// the given scalar params, with `f` wiring the body over a `GraphBuilder`.
pub(super) fn t_source_graph(
    param_defaults: Vec<f64>,
    param_names: &[&str],
    f: impl FnOnce(&GraphBuilder, &[u32], u32) -> u32,
) -> Graph {
    let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("t", 1usize)])));
    {
        let mut g = cell.borrow_mut();
        g.n_params = param_defaults.len();
        g.param_defaults = param_defaults;
        g.param_names = param_names.iter().map(|s| s.to_string()).collect();
    }
    let y = {
        let gb = GraphBuilder::new(&cell);
        let params: Vec<u32> = (0..param_names.len() as u32).map(|i| gb.param(i)).collect();
        let t = gb.input(0);
        f(&gb, &params, t)
    };
    let mut g = cell.into_inner();
    g.outputs.push(y);
    g
}

pub(super) fn t_source_block(type_name: &'static str, f_alg: crate::blocks::block::BlockFn, graph: Graph) -> BlockRef {
    let mut b = Block::new(Some(HashMap::new()), Some(out_port_map()));
    b.type_name = type_name;
    b.role = BlockRole { is_dyn: false, is_src: true, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));
    b.f_alg = Some(f_alg);
    b.set_alg(type_name, graph);
    Rc::new(FastCell::new(b))
}

/// SinusoidalSource: y = A * sin(2*pi*f*t + phi)
pub fn sinusoidal_source(frequency: f64, amplitude: f64, phase: f64) -> BlockRef {
    let mut b = Block::new(Some(HashMap::new()), Some(out_port_map()));
    b.type_name = "Source";
    b.role = BlockRole { is_dyn: false, is_src: true, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));
    b.f_alg = Some(Box::new(move |_x, _u, t, out| {
        sinusoidal_eval(&F64Builder, frequency, amplitude, phase, t, out)
    }));
    b.set_alg(
        "SinusoidalSource",
        sinusoidal_source_graph(frequency, amplitude, phase),
    );
    Rc::new(FastCell::new(b))
}

/// TriangleWaveSource: periodic triangle wave with given frequency, amplitude, and phase
pub fn triangle_wave_source(frequency: f64, amplitude: f64, phase: f64) -> BlockRef {
    let tau = phase / (std::f64::consts::TAU * frequency);
    let f_alg: crate::blocks::block::BlockFn = Box::new(move |_x, _u, t, out| {
        out.clear();
        out.push(triangle_eval(&F64Builder, frequency, amplitude, tau, t));
    });
    let graph = t_source_graph(
        vec![frequency, amplitude, tau],
        &["frequency", "amplitude", "tau"],
        |gb, p, t| triangle_eval(gb, p[0], p[1], p[2], t),
    );
    t_source_block("TriangleWave", f_alg, graph)
}

/// SquareWaveSource: periodic square wave toggling between +A and -A
pub fn square_wave_source(amplitude: f64, frequency: f64, phase: f64) -> BlockRef {
    let tau = phase / (std::f64::consts::TAU * frequency);
    let f_alg: crate::blocks::block::BlockFn = Box::new(move |_x, _u, t, out| {
        out.clear();
        out.push(square_eval(&F64Builder, amplitude, frequency, tau, t));
    });
    let graph = t_source_graph(
        vec![amplitude, frequency, tau],
        &["amplitude", "frequency", "tau"],
        |gb, p, t| square_eval(gb, p[0], p[1], p[2], t),
    );
    t_source_block("SquareWave", f_alg, graph)
}

/// PulseSource: trapezoidal pulse with rise/fall times and duty cycle
pub fn pulse_source(amplitude: f64, t_period: f64, t_rise: f64, t_fall: f64, tau: f64, duty: f64) -> BlockRef {
    // Trapezoidal pulse with rise/fall times, duty cycle, and phase offset
    // 4 Schedule events: rising, high, falling, low — mirrors pathsim PulseSource
    let t_rise = t_rise.max(SOURCE_RISE_FALL_TIME_MIN);
    let t_fall = t_fall.max(SOURCE_RISE_FALL_TIME_MIN);
    let t_plateau = t_period * duty;

    let phase_state: Rc<FastCell<&'static str>> = Rc::new(FastCell::new("low"));
    let phase_start: Rc<FastCell<f64>> = Rc::new(FastCell::new(tau));

    let ps_upd = phase_state.clone();
    let pst_upd = phase_start.clone();

    let mut b = Block::new(
        Some(HashMap::new()),
        Some(out_port_map()),
    );
    b.type_name = "PulseSource";
    b.role = BlockRole { is_dyn: false, is_src: true, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));

    // update: interpolate value based on current phase
    b.update_fn = Some(Box::new(move |blk, t| {
        let dt = t - *pst_upd.borrow();
        let val = match *ps_upd.borrow() {
            "rising" => amplitude * (dt / t_rise).min(1.0),
            "high" => amplitude,
            "falling" => amplitude * (1.0 - (dt / t_fall).min(1.0)),
            _ => 0.0,
        };
        blk.outputs._data[0] = val;
    }));

    // 4 Schedule events
    use crate::events::schedule::Schedule;
    let t_start_rise = tau;
    let t_start_high = t_start_rise + t_rise;
    let t_start_fall = t_start_high + t_plateau;
    let t_start_low = t_start_fall + t_fall;

    for (t_start, phase_name) in [
        (t_start_rise, "rising"),
        (t_start_high, "high"),
        (t_start_fall, "falling"),
        (t_start_low, "low"),
    ] {
        let ps = phase_state.clone();
        let pst = phase_start.clone();
        let evt = Schedule::new(
            t_start.max(0.0), None, t_period,
            Some(Box::new(move |t| {
                *ps.borrow_mut() = phase_name;
                *pst.borrow_mut() = t;
                // Output is computed by update_fn based on phase — no need to set here
            })),
            crate::constants::TOLERANCE,
        );
        b.events.push(Rc::new(FastCell::new(evt)));
    }

    // Reset: pathsim parity — restore initial 'low' phase and tau-anchored start time.
    // Note: pathsim's PulseSource.reset(t) with an explicit time (to phase-shift the
    // pulse at runtime) is not exposed here, because Block::reset takes no arguments.
    let ps_reset = phase_state.clone();
    let pst_reset = phase_start.clone();
    b.reset_fn = Some(Box::new(move |blk| {
        blk.inputs.reset();
        blk.outputs.reset();
        *ps_reset.borrow_mut() = "low";
        *pst_reset.borrow_mut() = tau;
    }));

    // IR (Memory + Event): slots [0]=phase (0=low,1=rising,2=high,3=falling),
    // [1]=phase_start. Four periodic events set the phase + capture the time;
    // alg interpolates the trapezoid value from (phase, t - phase_start).
    {
        let alg = {
            let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([
                (format!("mem{}", 0u32), 1usize),
                (format!("mem{}", 1u32), 1usize),
                ("t".to_string(), 1usize),
            ])));
            let y = {
                let gb = GraphBuilder::new(&cell);
                let phase = gb.input(0);
                let pstart = gb.input(1);
                let t = gb.input(2);
                let dt = gb.sub(t, pstart);
                let amp = gb.cst(amplitude);
                let one = gb.cst(1.0);
                let rising = gb.mul(amp, gb.min(gb.div(dt, gb.cst(t_rise)), one));
                let falling = gb.mul(amp, gb.sub(one, gb.min(gb.div(dt, gb.cst(t_fall)), one)));
                let zero = gb.cst(0.0);
                let s3 = gb.select(gb.eq(phase, gb.cst(3.0)), falling, zero);
                let s2 = gb.select(gb.eq(phase, gb.cst(2.0)), amp, s3);
                gb.select(gb.eq(phase, gb.cst(1.0)), rising, s2)
            };
            let mut g = cell.into_inner();
            g.outputs.push(y);
            g
        };
        let phase_event = |idx: f64, t_start: f64| {
            let effect = {
                let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("t", 1usize)])));
                let outs = {
                    let gb = GraphBuilder::new(&cell);
                    vec![gb.cst(idx), gb.input(0)]
                };
                let mut g = cell.into_inner();
                g.outputs = outs;
                g
            };
            EventSpec {
                kind: EventKindSpec::SchedulePeriodic { period: t_period, phase: t_start.max(0.0) },
                effect,
                targets: vec![MemTarget { slot: 0, offset: 0 }, MemTarget { slot: 1, offset: 0 }],
            }
        };
        let memory = vec![
            MemSpec { name: "phase".into(), init: vec![0.0] },
            MemSpec { name: "phase_start".into(), init: vec![tau] },
        ];
        let events = vec![
            phase_event(1.0, t_start_rise),
            phase_event(2.0, t_start_high),
            phase_event(3.0, t_start_fall),
            phase_event(0.0, t_start_low),
        ];
        b.set_discrete("PulseSource", alg, memory, events);
    }

    Rc::new(FastCell::new(b))
}

/// ClockSource: square wave between 0 and 1 with given period
pub fn clock_source(period: f64) -> BlockRef {
    let f_alg: crate::blocks::block::BlockFn = Box::new(move |_x, _u, t, out| {
        out.clear();
        out.push(clock_eval(&F64Builder, period, t));
    });
    let graph = t_source_graph(vec![period], &["period"], |gb, p, t| clock_eval(gb, p[0], t));
    t_source_block("ClockSource", f_alg, graph)
}

/// ChirpSource: swept-frequency sinusoid with triangular frequency modulation
/// Chirp output: y = amplitude * sin(2*pi*x + phase). Reads the phase state x.
pub(crate) fn chirp_alg_eval<B: Builder>(b: &B, amp: B::N, phase: B::N, x0: B::N) -> B::N {
    let arg = b.add(b.mul(b.cst(std::f64::consts::TAU), x0), phase);
    b.mul(amp, b.sin(arg))
}

/// Chirp phase rate: dx/dt = f0 + bw*0.5*(1 + tri(t/T)), triangular FM sweep.
pub(crate) fn chirp_dyn_eval<B: Builder>(b: &B, f0: B::N, bw: B::N, t_period: B::N, t: B::N) -> B::N {
    let tt = b.div(t, t_period);
    let tri = b.sub(
        b.mul(b.cst(2.0), b.abs(b.sub(tt, b.floor(b.add(tt, b.cst(0.5)))))),
        b.cst(1.0),
    );
    b.add(f0, b.mul(b.mul(bw, b.cst(0.5)), b.add(b.cst(1.0), tri)))
}

/// Chirp op-graph (shared by `chirp_source` and the nominal lowering of
/// `chirp_phase_noise_source`). Unified param list `[amplitude, phase, f0, bw,
/// t_period]` shared by both regions; alg uses 0,1 (reads state x), dyn uses
/// 2,3,4 (reads time t). The phase-noise terms are NOT part of this graph (they
/// are a stochastic runtime input — see `chirp_phase_noise_source`).
fn chirp_ops(amplitude: f64, f0: f64, bw: f64, t_period: f64, phase: f64) -> (crate::ssa::graph::Graph, crate::ssa::graph::Graph) {
    let names: &[&str] = &["amplitude", "phase", "f0", "bw", "t_period"];
    let defaults = vec![amplitude, phase, f0, bw, t_period];
    let alg = {
        let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("x", 1usize)])));
        {
            let mut g = cell.borrow_mut();
            g.n_params = 5;
            g.param_defaults = defaults.clone();
            g.param_names = names.iter().map(|s| s.to_string()).collect();
        }
        let y = {
            let gb = GraphBuilder::new(&cell);
            chirp_alg_eval(&gb, gb.param(0), gb.param(1), gb.input(0))
        };
        let mut g = cell.into_inner();
        g.outputs.push(y);
        g
    };
    let dyn_ = {
        let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("t", 1usize)])));
        {
            let mut g = cell.borrow_mut();
            g.n_params = 5;
            g.param_defaults = defaults;
            g.param_names = names.iter().map(|s| s.to_string()).collect();
        }
        let y = {
            let gb = GraphBuilder::new(&cell);
            chirp_dyn_eval(&gb, gb.param(2), gb.param(3), gb.param(4), gb.input(0))
        };
        let mut g = cell.into_inner();
        g.outputs.push(y);
        g
    };
    (alg, dyn_)
}

pub fn chirp_source(amplitude: f64, f0: f64, bw: f64, t_period: f64, phase: f64) -> BlockRef {
    use crate::solvers::solver::Solver;

    let mut b = Block::new(Some(HashMap::new()), Some(out_port_map()));
    b.type_name = "ChirpSource";
    b.role = BlockRole { is_dyn: true, is_src: true, is_rec: false };
    b.initial_value = Some(vec![0.0]);
    b.engine = Some(Solver::with_defaults(&[0.0]));
    b.len_fn = Some(Box::new(|_| 0));

    b.f_dyn = Some(Box::new(move |_x, _u, t, out| {
        out.clear();
        out.push(chirp_dyn_eval(&F64Builder, f0, bw, t_period, t));
    }));
    b.f_alg = Some(Box::new(move |x, _u, _t, out| {
        out.clear();
        out.push(chirp_alg_eval(&F64Builder, amplitude, phase, x[0]));
    }));

    let (alg, dyn_) = chirp_ops(amplitude, f0, bw, t_period, phase);
    b.set_dynamic("ChirpSource", alg, dyn_);

    Rc::new(FastCell::new(b))
}

/// GaussianPulseSource: y = A * exp(-((t - tau) / sigma)^2)
pub fn gaussian_pulse_source(amplitude: f64, f_max: f64, tau: f64) -> BlockRef {
    let sigma = 0.5 / f_max;
    let f_alg: crate::blocks::block::BlockFn = Box::new(move |_x, _u, t, out| {
        out.clear();
        out.push(gaussian_pulse_eval(&F64Builder, amplitude, tau, sigma, t));
    });
    let graph = t_source_graph(
        vec![amplitude, tau, sigma],
        &["amplitude", "tau", "sigma"],
        |gb, p, t| gaussian_pulse_eval(gb, p[0], p[1], p[2], t),
    );
    t_source_block("GaussianPulse", f_alg, graph)
}

// ======================================================================================
// Phase-noise variants — port of pathsim SinusoidalPhaseNoiseSource /
// ChirpPhaseNoiseSource. State is the cumulative-noise integral (engine.state),
// and `noise_1` / `noise_2` are the white / cumulative samples refreshed either
// every timestep (sampling_period=None) or on a periodic Schedule event.
// ======================================================================================

// --- Discrete (zero-order-hold) phase-noise lowering ---
//
// In discrete mode the white and cumulative samples are refreshed on a periodic
// `sp` Schedule, i.e. they are a zero-order hold over `floor(t/sp)`. That makes
// them a pure function of t — `n_k(t) = normal(stepkey(t) + offset)` — so the
// whole source (alg + the `dx/dt = n_2` random walk) lowers to SSA and is shared
// by the runtime closures and the compile/codegen op-graphs. Continuous mode
// keeps the live stateful-RNG path and the nominal (zero-noise) export.

/// Two decorrelated standard normals for one discrete step: white `n_1` at the
/// step key, cumulative `n_2` at an offset key (independent stream).
fn phase_noise_pair<B: Builder>(b: &B, inv_sp: B::N, key_phase: B::N, t: B::N) -> (B::N, B::N) {
    let base = noise_step_key(b, t, inv_sp, key_phase);
    let n1 = noise_normal_eval(b, base);
    let n2 = noise_normal_eval(b, b.add(base, b.cst(0.25)));
    (n1, n2)
}

/// SinusoidalPhaseNoiseSource alg: `y = A·sin(ωt + φ + σ_c·x + σ_w·n_1(t))`.
#[allow(clippy::too_many_arguments)]
fn sin_pn_alg<B: Builder>(
    b: &B, freq: B::N, amp: B::N, phase: B::N, sig_cum: B::N, sig_white: B::N,
    inv_sp: B::N, key_phase: B::N, x: B::N, t: B::N,
) -> B::N {
    let (n1, _) = phase_noise_pair(b, inv_sp, key_phase, t);
    let wt = b.mul(b.mul(b.cst(std::f64::consts::TAU), freq), t);
    let arg = b.add(b.add(b.add(wt, phase), b.mul(sig_cum, x)), b.mul(sig_white, n1));
    b.mul(amp, b.sin(arg))
}

/// Phase-noise random walk derivative `dx/dt = n_2(t)` (shared by both sources).
fn pn_walk_dyn<B: Builder>(b: &B, inv_sp: B::N, key_phase: B::N, t: B::N) -> B::N {
    let (_, n2) = phase_noise_pair(b, inv_sp, key_phase, t);
    n2
}

/// ChirpPhaseNoiseSource alg: `y = A·sin(2π(x + σ_w·n_1(t)) + φ)`.
#[allow(clippy::too_many_arguments)]
fn chirp_pn_alg<B: Builder>(
    b: &B, amp: B::N, phase: B::N, sig_white: B::N,
    inv_sp: B::N, key_phase: B::N, x: B::N, t: B::N,
) -> B::N {
    let (n1, _) = phase_noise_pair(b, inv_sp, key_phase, t);
    chirp_alg_eval(b, amp, phase, b.add(x, b.mul(sig_white, n1)))
}

/// ChirpPhaseNoiseSource dyn: deterministic sweep `+ σ_c·n_2(t)`.
#[allow(clippy::too_many_arguments)]
fn chirp_pn_dyn<B: Builder>(
    b: &B, f0: B::N, bw: B::N, t_period: B::N, sig_cum: B::N,
    inv_sp: B::N, key_phase: B::N, t: B::N,
) -> B::N {
    let (_, n2) = phase_noise_pair(b, inv_sp, key_phase, t);
    b.add(chirp_dyn_eval(b, f0, bw, t_period, t), b.mul(sig_cum, n2))
}

/// Assemble a discrete dynamic phase-noise source from generic alg/dyn closures.
/// `alg(B, x, t)` and `dyn(B, t)` carry the per-block parameters captured by the
/// caller; both run on `F64Builder` (runtime) and `GraphBuilder` (op-graph).
fn discrete_phase_noise_block(
    type_name: &'static str,
    params: Vec<f64>,
    names: &'static [&'static str],
    alg_eval: impl Fn(&F64Builder, f64, f64) -> f64 + 'static,
    dyn_eval: impl Fn(&F64Builder, f64) -> f64 + 'static,
    alg_graph: impl Fn(&GraphBuilder, &[u32], u32, u32) -> u32,
    dyn_graph: impl Fn(&GraphBuilder, &[u32], u32) -> u32,
) -> BlockRef {
    use crate::solvers::solver::Solver;
    let mut b = Block::new(Some(HashMap::new()), Some(out_port_map()));
    b.type_name = type_name;
    b.role = BlockRole { is_dyn: true, is_src: true, is_rec: false };
    b.initial_value = Some(vec![0.0]);
    b.engine = Some(Solver::with_defaults(&[0.0]));
    b.len_fn = Some(Box::new(|_| 0));
    b.f_dyn = Some(Box::new(move |_x, _u, t, out| out.push(dyn_eval(&F64Builder, t))));
    b.f_alg = Some(Box::new(move |x, _u, t, out| out.push(alg_eval(&F64Builder, x[0], t))));
    b.reset_fn = Some(Box::new(|blk| {
        blk.inputs.reset();
        blk.outputs.reset();
        if let Some(ref mut engine) = blk.engine { engine.reset(Some(&[0.0])); }
    }));

    let make_graph = |sig: InputSignature, build: &dyn Fn(&GraphBuilder, &[u32]) -> u32| -> Graph {
        let cell = RefCell::new(Graph::new(sig));
        {
            let mut g = cell.borrow_mut();
            g.n_params = params.len();
            g.param_defaults = params.clone();
            g.param_names = names.iter().map(|s| s.to_string()).collect();
        }
        let y = {
            let gb = GraphBuilder::new(&cell);
            let p: Vec<u32> = (0..names.len() as u32).map(|i| gb.param(i)).collect();
            build(&gb, &p)
        };
        let mut g = cell.into_inner();
        g.outputs.push(y);
        g
    };
    let alg = make_graph(
        InputSignature::from_named_sizes([("x", 1usize), ("t", 1usize)]),
        &|gb, p| { let x = gb.input(0); let t = gb.input(1); alg_graph(gb, p, x, t) },
    );
    let dyn_ = make_graph(
        InputSignature::from_named_sizes([("t", 1usize)]),
        &|gb, p| { let t = gb.input(0); dyn_graph(gb, p, t) },
    );
    b.set_dynamic(type_name, alg, dyn_);
    Rc::new(FastCell::new(b))
}

/// SinusoidalPhaseNoiseSource: y(t) = A sin(ωt + φ + σ_w·n_w(t) + σ_c·∫n_c(τ)dτ).
///
/// Output is evaluated algebraically from the engine-integrated cumulative
/// noise plus the current white-noise sample. The engine integrates `n_2`
/// (i.e. `dx/dt = n_2`, a random-walk process).
pub fn sinusoidal_phase_noise_source(
    frequency: f64,
    amplitude: f64,
    phase: f64,
    sig_cum: f64,
    sig_white: f64,
    sampling_period: Option<f64>,
    seed: Option<u64>,
) -> BlockRef {
    // Discrete (zero-order-hold) mode lowers the real noise into SSA.
    if let Some(sp) = sampling_period {
        let inv_sp = 1.0 / sp;
        let key_phase = noise_seed_phase(seed);
        let params = vec![frequency, amplitude, phase, sig_cum, sig_white, inv_sp, key_phase];
        const NAMES: &[&str] = &[
            "frequency", "amplitude", "phase", "sig_cum", "sig_white", "inv_sp", "key_phase",
        ];
        return discrete_phase_noise_block(
            "SinusoidalPhaseNoiseSource", params, NAMES,
            move |b, x, t| sin_pn_alg(b, frequency, amplitude, phase, sig_cum, sig_white, inv_sp, key_phase, x, t),
            move |b, t| pn_walk_dyn(b, inv_sp, key_phase, t),
            |gb, p, x, t| sin_pn_alg(gb, p[0], p[1], p[2], p[3], p[4], p[5], p[6], x, t),
            |gb, p, t| pn_walk_dyn(gb, p[5], p[6], t),
        );
    }

    use crate::solvers::solver::Solver;
    let omega = std::f64::consts::TAU * frequency;

    let rng = Rc::new(FastCell::new(match seed {
        Some(s) => Rng::new(s),
        None => Rng::from_entropy(),
    }));

    // Initial samples
    let noise_1 = Rc::new(FastCell::new(rng.borrow_mut().normal()));
    let noise_2 = Rc::new(FastCell::new(rng.borrow_mut().normal()));

    let mut b = Block::new(
        Some(HashMap::new()),
        Some(out_port_map()),
    );
    b.type_name = "SinusoidalPhaseNoiseSource";
    b.role = BlockRole { is_dyn: true, is_src: true, is_rec: false };
    b.initial_value = Some(vec![0.0]);
    b.engine = Some(Solver::with_defaults(&[0.0]));
    b.len_fn = Some(Box::new(|_| 0));

    // dx/dt = n_2 (random-walk integrator for cumulative phase noise)
    let n2_fdyn = noise_2.clone();
    b.f_dyn = Some(Box::new(move |_x, _u, _t, out| {
        out.push(*n2_fdyn.borrow());
    }));

    // y = A sin(ωt + φ + σ_w·n_1 + σ_c·x)
    let n1_alg = noise_1.clone();
    b.f_alg = Some(Box::new(move |x, _u, t, out| {
        let phase_err = sig_white * *n1_alg.borrow() + sig_cum * x[0];
        out.push(amplitude * (omega * t + phase + phase_err).sin());
    }));

    // Noise refresh — either periodic Schedule (discrete) or every-step sample_fn (continuous)
    if let Some(sp) = sampling_period {
        let n1_evt = noise_1.clone();
        let n2_evt = noise_2.clone();
        let rng_evt = rng.clone();
        use crate::events::schedule::Schedule;
        let evt = Schedule::new(
            0.0, None, sp,
            Some(Box::new(move |_t| {
                *n1_evt.borrow_mut() = rng_evt.borrow_mut().normal();
                *n2_evt.borrow_mut() = rng_evt.borrow_mut().normal();
            })),
            crate::constants::TOLERANCE,
        );
        b.events.push(Rc::new(FastCell::new(evt)));
    } else {
        let n1_samp = noise_1.clone();
        let n2_samp = noise_2.clone();
        let rng_samp = rng.clone();
        b.sample_fn = Some(Box::new(move |_blk, _t, _dt| {
            *n1_samp.borrow_mut() = rng_samp.borrow_mut().normal();
            *n2_samp.borrow_mut() = rng_samp.borrow_mut().normal();
        }));
    }

    // Reset: clear state, regenerate noise samples (pathsim parity)
    let n1_reset = noise_1.clone();
    let n2_reset = noise_2.clone();
    let rng_reset = rng.clone();
    b.reset_fn = Some(Box::new(move |blk| {
        blk.inputs.reset();
        blk.outputs.reset();
        if let Some(ref mut engine) = blk.engine {
            engine.reset(Some(&[0.0]));
        }
        *n1_reset.borrow_mut() = rng_reset.borrow_mut().normal();
        *n2_reset.borrow_mut() = rng_reset.borrow_mut().normal();
    }));

    // Op-graph for compile()/codegen: the deterministic sinusoid at its NOMINAL
    // (zero-noise) operating point. The white sample n_1 and the random-walk
    // input n_2 (`dx/dt = n_2`) are a stochastic RNG runtime input not
    // expressible in static SSA, taken at their zero-mean nominal (0). With
    // n_2 = 0 the cumulative-noise state stays at its initial 0, so the export
    // reduces to y = A·sin(ωt + φ). The interpreted runtime keeps the live noise.
    //
    // alg reads state x (slot 0) and time t (slot 1); dyn integrates the nominal
    // zero random walk. Params [frequency, amplitude, phase, sig_cum] shared.
    let names: &[&str] = &["frequency", "amplitude", "phase", "sig_cum"];
    let defaults = vec![frequency, amplitude, phase, sig_cum];
    let alg = {
        let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([
            ("x", 1usize),
            ("t", 1usize),
        ])));
        {
            let mut g = cell.borrow_mut();
            g.n_params = 4;
            g.param_defaults = defaults.clone();
            g.param_names = names.iter().map(|s| s.to_string()).collect();
        }
        let y = {
            let gb = GraphBuilder::new(&cell);
            let x = gb.input(0);
            let t = gb.input(1);
            let freq = gb.param(0);
            let amp = gb.param(1);
            let ph = gb.param(2);
            let sc = gb.param(3);
            // arg = 2π·freq·t + phase + sig_cum·x
            let wt = gb.mul(gb.mul(gb.cst(std::f64::consts::TAU), freq), t);
            let arg = gb.add(gb.add(wt, ph), gb.mul(sc, x));
            gb.mul(amp, gb.sin(arg))
        };
        let mut g = cell.into_inner();
        g.outputs.push(y);
        g
    };
    let dyn_ = {
        let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("t", 1usize)])));
        {
            let mut g = cell.borrow_mut();
            g.n_params = 4;
            g.param_defaults = defaults;
            g.param_names = names.iter().map(|s| s.to_string()).collect();
        }
        // dx/dt = sig_cum·n_2, nominal n_2 = 0 → constant 0 derivative.
        let y = {
            let gb = GraphBuilder::new(&cell);
            gb.cst(0.0)
        };
        let mut g = cell.into_inner();
        g.outputs.push(y);
        g
    };
    b.set_dynamic("SinusoidalPhaseNoiseSource", alg, dyn_);

    Rc::new(FastCell::new(b))
}

/// ChirpPhaseNoiseSource: frequency-swept sinusoid with white and cumulative
/// phase noise. Pathsim's `ChirpSource` is an alias for this class.
///
/// - `dx/dt = f0 + BW·(1 + tri(t, 1/T))/2 + σ_c·n_2`
/// - `y = A sin(2π(x + σ_w·n_1) + φ)`
///
/// where `tri` is a unit-amplitude triangle wave of period `T`.
pub fn chirp_phase_noise_source(
    amplitude: f64,
    f0: f64,
    bw: f64,
    t_period: f64,
    phase: f64,
    sig_cum: f64,
    sig_white: f64,
    sampling_period: Option<f64>,
    seed: Option<u64>,
) -> BlockRef {
    // Discrete (zero-order-hold) mode lowers the real noise into SSA.
    if let Some(sp) = sampling_period {
        let inv_sp = 1.0 / sp;
        let key_phase = noise_seed_phase(seed);
        let params = vec![amplitude, phase, f0, bw, t_period, sig_cum, sig_white, inv_sp, key_phase];
        const NAMES: &[&str] = &[
            "amplitude", "phase", "f0", "bw", "t_period", "sig_cum", "sig_white", "inv_sp", "key_phase",
        ];
        return discrete_phase_noise_block(
            "ChirpPhaseNoiseSource", params, NAMES,
            move |b, x, t| chirp_pn_alg(b, amplitude, phase, sig_white, inv_sp, key_phase, x, t),
            move |b, t| chirp_pn_dyn(b, f0, bw, t_period, sig_cum, inv_sp, key_phase, t),
            |gb, p, x, t| chirp_pn_alg(gb, p[0], p[1], p[6], p[7], p[8], x, t),
            |gb, p, t| chirp_pn_dyn(gb, p[2], p[3], p[4], p[5], p[7], p[8], t),
        );
    }

    use crate::solvers::solver::Solver;
    let pi2 = std::f64::consts::TAU;

    let rng = Rc::new(FastCell::new(match seed {
        Some(s) => Rng::new(s),
        None => Rng::from_entropy(),
    }));

    let noise_1 = Rc::new(FastCell::new(rng.borrow_mut().normal()));
    let noise_2 = Rc::new(FastCell::new(rng.borrow_mut().normal()));

    let mut b = Block::new(
        Some(HashMap::new()),
        Some(out_port_map()),
    );
    b.type_name = "ChirpPhaseNoiseSource";
    b.role = BlockRole { is_dyn: true, is_src: true, is_rec: false };
    b.initial_value = Some(vec![0.0]);
    b.engine = Some(Solver::with_defaults(&[0.0]));
    b.len_fn = Some(Box::new(|_| 0));

    // dx/dt = f0 + BW·(1 + tri(t, 1/T))/2 + σ_c·n_2
    let n2_fdyn = noise_2.clone();
    b.f_dyn = Some(Box::new(move |_x, _u, t, out| {
        let tt = t * (1.0 / t_period);
        let tri = 2.0 * (tt - (tt + 0.5).floor()).abs() - 1.0; // [-1, 1]
        let freq = f0 + bw * 0.5 * (1.0 + tri) + sig_cum * *n2_fdyn.borrow();
        out.push(freq);
    }));

    // y = A sin(2π(x + σ_w·n_1) + φ)
    let n1_alg = noise_1.clone();
    b.f_alg = Some(Box::new(move |x, _u, _t, out| {
        let phi = pi2 * (x[0] + sig_white * *n1_alg.borrow()) + phase;
        out.push(amplitude * phi.sin());
    }));

    if let Some(sp) = sampling_period {
        let n1_evt = noise_1.clone();
        let n2_evt = noise_2.clone();
        let rng_evt = rng.clone();
        use crate::events::schedule::Schedule;
        let evt = Schedule::new(
            0.0, None, sp,
            Some(Box::new(move |_t| {
                *n1_evt.borrow_mut() = rng_evt.borrow_mut().normal();
                *n2_evt.borrow_mut() = rng_evt.borrow_mut().normal();
            })),
            crate::constants::TOLERANCE,
        );
        b.events.push(Rc::new(FastCell::new(evt)));
    } else {
        let n1_samp = noise_1.clone();
        let n2_samp = noise_2.clone();
        let rng_samp = rng.clone();
        b.sample_fn = Some(Box::new(move |_blk, _t, _dt| {
            *n1_samp.borrow_mut() = rng_samp.borrow_mut().normal();
            *n2_samp.borrow_mut() = rng_samp.borrow_mut().normal();
        }));
    }

    let n1_reset = noise_1.clone();
    let n2_reset = noise_2.clone();
    let rng_reset = rng.clone();
    b.reset_fn = Some(Box::new(move |blk| {
        blk.inputs.reset();
        blk.outputs.reset();
        if let Some(ref mut engine) = blk.engine {
            engine.reset(Some(&[0.0]));
        }
        *n1_reset.borrow_mut() = rng_reset.borrow_mut().normal();
        *n2_reset.borrow_mut() = rng_reset.borrow_mut().normal();
    }));

    // Op-graph for compile()/codegen: the deterministic chirp at its NOMINAL
    // (zero-noise) operating point — identical to `chirp_source`. The white /
    // cumulative phase-noise samples (σ_w·n_1, σ_c·n_2) are a stochastic RNG
    // runtime input not expressible in static SSA, taken at their zero-mean
    // nominal (0). The interpreted runtime (`f_alg`/`f_dyn`) keeps the live noise.
    let (alg, dyn_) = chirp_ops(amplitude, f0, bw, t_period, phase);
    b.set_dynamic("ChirpPhaseNoiseSource", alg, dyn_);

    Rc::new(FastCell::new(b))
}
