// Stochastic source block constructors: white noise, pink noise, and
// a stand-alone random number generator.  Each maintains its own RNG
// state so that different instances produce independent streams, and
// deterministic seeding is supported for reproducibility.

use std::collections::HashMap;
use std::rc::Rc;

use crate::blocks::block::{Block, BlockFn, BlockRef, BlockRole};
use crate::ssa::build::{Builder, F64Builder};
use crate::utils::fastcell::FastCell;
use crate::utils::rng::Rng;

use super::out_port_map;
use super::sources::{t_source_block, t_source_graph};

// ======================================================================================
// Shared keyed-noise kernel (stateless, traceable)
//
// A discrete (zero-order-hold) noise source is a *pure function of t*: the sample
// held over the interval [k·sp, (k+1)·sp) is drawn from `key = floor(t/sp) +
// phase`, where `phase` is a per-instance offset so independent instances
// decorrelate. Evaluating the same expression with `F64Builder` (the runtime
// closure) and `GraphBuilder` (the op-graph for compile/codegen) gives identical
// numbers, so the interpreted, compiled and code-generated runs all agree — and
// the draws are reproducible by seed, unlike a stateful RNG. Continuous-sampling
// mode (no `sampling_period`) draws fresh every solver step and so is not a pure
// function of t; it keeps the stateful RNG path and stays opaque to compile.
// ======================================================================================

/// Resolve an optional seed to a per-instance stream phase. `None` draws one
/// entropy sample at construction (independent instances, fixed thereafter).
pub(crate) fn noise_seed_phase(seed: Option<u64>) -> f64 {
    let raw = match seed {
        Some(s) => s,
        None => (Rng::from_entropy().uniform() * 9_007_199_254_740_992.0) as u64,
    };
    (raw % 1_000_000_007) as f64
}

/// Discrete step key: `floor(t · inv_sp) + phase`.
pub(crate) fn noise_step_key<B: Builder>(b: &B, t: B::N, inv_sp: B::N, phase: B::N) -> B::N {
    b.add(b.floor(b.mul(t, inv_sp)), phase)
}

/// Uniform `[0, 1)` draw for the keyed step.
pub(crate) fn noise_uniform_eval<B: Builder>(b: &B, key: B::N) -> B::N {
    b.rand_uniform(key)
}

/// Standard-normal draw via Box-Muller — bit-parity with `fastsim.random_normal`
/// and the `RandUniform` tape op (same two decorrelated uniforms, same order).
pub(crate) fn noise_normal_eval<B: Builder>(b: &B, key: B::N) -> B::N {
    let u1 = b.rand_uniform(key);
    let u2 = b.rand_uniform(b.add(key, b.cst(0.5)));
    let u1c = b.max(u1, b.cst(f64::MIN_POSITIVE));
    let r = b.sqrt(b.mul(b.cst(-2.0), b.ln(u1c)));
    let ang = b.mul(b.cst(std::f64::consts::TAU), u2);
    b.mul(r, b.cos(ang))
}

// ======================================================================================
// Noise sources
// ======================================================================================

/// White noise source with Gaussian distribution.
///
/// Continuous mode (sampling_period=None): generates new sample every timestep.
/// Discrete mode: generates sample at fixed intervals (zero-order hold).
///
/// If spectral_density is Some, output is scaled as sqrt(S0/dt) for correct
/// stochastic integration. Otherwise uses constant standard_deviation.
pub fn white_noise(
    standard_deviation: f64,
    spectral_density: Option<f64>,
    sampling_period: Option<f64>,
    seed: Option<u64>,
) -> BlockRef {
    // Discrete (zero-order-hold) mode lowers to `out = scale · normal(key(t))`.
    // The sample period IS the integration step here (`dt = sp`, a known
    // constant), so even the spectral-density scaling `sqrt(S0/sp)` is a constant
    // and the whole source is a pure function of t → first-class for compile/codegen.
    if let Some(sp) = sampling_period {
        let phase = noise_seed_phase(seed);
        let inv_sp = 1.0 / sp;
        let scale = match spectral_density {
            Some(sd) => (sd / sp).sqrt(),
            None => standard_deviation,
        };
        let f_alg: BlockFn = Box::new(move |_x, _u, t, out| {
            out.clear();
            let key = noise_step_key(&F64Builder, t, inv_sp, phase);
            out.push(F64Builder.mul(scale, noise_normal_eval(&F64Builder, key)));
        });
        let graph = t_source_graph(
            vec![inv_sp, phase, scale],
            &["inv_sp", "phase", "scale"],
            |gb, p, t| {
                let key = noise_step_key(gb, t, p[0], p[1]);
                gb.mul(p[2], noise_normal_eval(gb, key))
            },
        );
        return t_source_block("WhiteNoise", f_alg, graph);
    }

    // Continuous mode draws fresh every solver step (`dt` is the live step, the
    // sample is not a pure function of t) → stateful RNG path, opaque to compile.
    let mut b = Block::new(
        Some(HashMap::new()),
        Some(out_port_map()),
    );
    b.type_name = "WhiteNoise";
    b.role = BlockRole { is_dyn: false, is_src: true, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));

    let rng = Rc::new(FastCell::new(match seed {
        Some(s) => Rng::new(s),
        None => Rng::from_entropy(),
    }));
    let rng_samp = rng.clone();
    let sd_val = spectral_density;
    let std_dev = standard_deviation;
    b.sample_fn = Some(Box::new(move |blk, _t, dt| {
        blk.outputs._data[0] = generate_white_sample(rng_samp.borrow_mut(), sd_val, std_dev, dt);
    }));

    Rc::new(FastCell::new(b))
}

#[inline]
fn generate_white_sample(rng: &mut Rng, spectral_density: Option<f64>, std_dev: f64, dt: f64) -> f64 {
    if let Some(sd) = spectral_density {
        rng.normal() * (sd / dt).sqrt()
    } else {
        rng.normal_scaled(0.0, std_dev)
    }
}

/// Pink noise (1/f noise) source using the Voss-McCartney algorithm.
///
/// Maintains num_octaves independent random values representing different
/// frequency bands. At each sample, one octave is updated based on the binary
/// representation of the sample counter.
pub fn pink_noise(
    standard_deviation: f64,
    spectral_density: Option<f64>,
    num_octaves: usize,
    sampling_period: Option<f64>,
    seed: Option<u64>,
) -> BlockRef {
    let mut b = Block::new(
        Some(HashMap::new()),
        Some(out_port_map()),
    );
    b.type_name = "PinkNoise";
    b.role = BlockRole { is_dyn: false, is_src: true, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));

    let rng = Rc::new(FastCell::new(match seed {
        Some(s) => Rng::new(s),
        None => Rng::from_entropy(),
    }));

    // Initialize octave values
    let mut init_octaves = vec![0.0f64; num_octaves];
    for v in init_octaves.iter_mut() {
        *v = rng.borrow_mut().normal();
    }
    let octaves = Rc::new(FastCell::new(init_octaves));
    let n_samples = Rc::new(FastCell::new(0u64));

    // Holds the latest discrete sample (None in continuous mode → reset is a no-op
    // for this cell). Declared outside the branch so `reset_fn` can capture it.
    let current: Option<Rc<FastCell<f64>>> = if let Some(sp) = sampling_period {
        // Discrete mode: Schedule event generates sample directly into shared
        // cell; update_fn copies it into outputs each step (pathsim parity).
        let sd_val = spectral_density;
        let std_dev = standard_deviation;

        // Initial sample
        let initial = generate_pink_sample(
            rng.borrow_mut(), octaves.borrow_mut(),
            n_samples.borrow_mut(), num_octaves, sd_val, std_dev, sp,
        );
        b.outputs._data[0] = initial;

        let cur = Rc::new(FastCell::new(initial));
        let cur_evt = cur.clone();
        let cur_upd = cur.clone();
        let rng_evt = rng.clone();
        let oct_evt = octaves.clone();
        let ns_evt = n_samples.clone();

        b.update_fn = Some(Box::new(move |blk, _t| {
            blk.outputs._data[0] = *cur_upd.borrow();
        }));

        use crate::events::schedule::Schedule;
        let evt = Schedule::new(
            0.0, None, sp,
            Some(Box::new(move |_t| {
                let sample = generate_pink_sample(
                    rng_evt.borrow_mut(), oct_evt.borrow_mut(),
                    ns_evt.borrow_mut(), num_octaves, sd_val, std_dev, sp,
                );
                *cur_evt.borrow_mut() = sample;
            })),
            crate::constants::TOLERANCE,
        );
        b.events.push(Rc::new(FastCell::new(evt)));
        Some(cur)
    } else {
        // Continuous mode
        let rng_samp = rng.clone();
        let oct_samp = octaves.clone();
        let ns_samp = n_samples.clone();
        let sd_val = spectral_density;
        let std_dev = standard_deviation;

        b.sample_fn = Some(Box::new(move |blk, _t, dt| {
            let sample = generate_pink_sample(
                rng_samp.borrow_mut(), oct_samp.borrow_mut(),
                ns_samp.borrow_mut(), num_octaves, sd_val, std_dev, dt,
            );
            blk.outputs._data[0] = sample;
        }));
        None
    };

    // Reset: pathsim parity — regenerate octave_values from fresh RNG draws,
    // zero sample counter, clear current discrete sample (mirrors PinkNoise.reset).
    let rng_reset = rng.clone();
    let oct_reset = octaves.clone();
    let ns_reset = n_samples.clone();
    let cur_reset = current;
    b.reset_fn = Some(Box::new(move |blk| {
        blk.inputs.reset();
        blk.outputs.reset();
        *ns_reset.borrow_mut() = 0;
        let oct = oct_reset.borrow_mut();
        let r = rng_reset.borrow_mut();
        for v in oct.iter_mut() {
            *v = r.normal();
        }
        if let Some(ref c) = cur_reset {
            *c.borrow_mut() = 0.0;
        }
    }));

    Rc::new(FastCell::new(b))
}

#[inline]
fn generate_pink_sample(
    rng: &mut Rng, octaves: &mut [f64], n_samples: &mut u64,
    num_octaves: usize, spectral_density: Option<f64>, std_dev: f64, dt: f64,
) -> f64 {
    *n_samples += 1;

    // Find trailing zeros to pick which octave to update (Voss-McCartney)
    let mut n = *n_samples;
    let mut octave_idx = 0;
    while (n & 1) == 0 && octave_idx < num_octaves - 1 {
        n >>= 1;
        octave_idx += 1;
    }

    // Update selected octave
    octaves[octave_idx] = rng.normal();

    // Sum all octaves
    let pink_sample: f64 = octaves.iter().sum();

    // Scale output
    if let Some(sd) = spectral_density {
        pink_sample * (sd / num_octaves as f64 / dt).sqrt()
    } else {
        pink_sample * std_dev / (num_octaves as f64).sqrt()
    }
}

/// Random number generator source (uniform [0, 1)).
///
/// Continuous mode: new sample every timestep.
/// Discrete mode: new sample at fixed intervals (zero-order hold).
pub fn random_number_generator(sampling_period: Option<f64>, seed: Option<u64>) -> BlockRef {
    // Discrete (zero-order-hold) mode lowers to an op-graph `out = uniform(key(t))`
    // — first-class for compile()/codegen, runtime == compiled by construction.
    if let Some(sp) = sampling_period {
        let phase = noise_seed_phase(seed);
        let inv_sp = 1.0 / sp;
        let f_alg: BlockFn = Box::new(move |_x, _u, t, out| {
            out.clear();
            let key = noise_step_key(&F64Builder, t, inv_sp, phase);
            out.push(noise_uniform_eval(&F64Builder, key));
        });
        let graph = t_source_graph(
            vec![inv_sp, phase],
            &["inv_sp", "phase"],
            |gb, p, t| {
                let key = noise_step_key(gb, t, p[0], p[1]);
                noise_uniform_eval(gb, key)
            },
        );
        return t_source_block("RandomNumberGenerator", f_alg, graph);
    }

    // Continuous mode draws fresh every solver step → not a pure function of t,
    // stays on the stateful RNG path (opaque to static compile).
    let mut b = Block::new(
        Some(HashMap::new()),
        Some(out_port_map()),
    );
    b.type_name = "RandomNumberGenerator";
    b.role = BlockRole { is_dyn: false, is_src: true, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));

    let rng = Rc::new(FastCell::new(match seed {
        Some(s) => Rng::new(s),
        None => Rng::from_entropy(),
    }));
    let rng_samp = rng.clone();
    b.sample_fn = Some(Box::new(move |blk, _t, _dt| {
        blk.outputs._data[0] = rng_samp.borrow_mut().uniform();
    }));

    Rc::new(FastCell::new(b))
}
