// Block constructor functions — each creates a Block with specific callbacks
// Ported 1:1 from pathsim/blocks/*.py
//
// In Python: class Integrator(Block) with method overrides
// In Rust: fn integrator() -> BlockRef with callback fields set

use std::collections::HashMap;

// Used only by the test modules below (the constructor bodies live in the
// sibling files, so the parent module itself does not reference these).
#[cfg(test)]
use std::rc::Rc;
#[cfg(test)]
use crate::blocks::block::BlockRef;

/// Optional scalar Jacobian `∂f/∂x` for 1-D ODE blocks (legacy form).
type ScalarJacFn = Box<dyn Fn(&[f64], &[f64], f64) -> f64>;
/// Optional vector Jacobian `∂f/∂x` for multi-D ODE / MassMatrix blocks
/// (flat row-major `n_x × n_x`).
type VecJacFn = Box<dyn Fn(&[f64], &[f64], f64) -> Vec<f64>>;
/// Optional Jacobian for DAE blocks — depends on a second state array
/// (`z` for semi-explicit, `xdot` for fully-implicit), signature
/// `(x, z_or_xdot, u, t) -> Vec<f64>`, flat row-major.
type DaeJacFn = Box<dyn Fn(&[f64], &[f64], &[f64], f64) -> Vec<f64>>;

/// Single labeled output port named "out" — convenience for sources and MISO blocks.
fn out_port_map() -> HashMap<String, usize> {
    HashMap::from([("out".to_string(), 0)])
}

// ======================================================================================
// Sub-module wiring — each `mod X; pub use X::*;` re-exports a thematic group.
// ======================================================================================

pub(crate) mod core;
pub use core::{
    adder, amplifier, constant, divider, function, integrator, integrator_vec,
    multiplier, source, ZeroDiv,
};

mod scope;
pub use scope::{scope, scope_read, scope_read_incremental};

mod dynsys;
pub use dynsys::{dynamical_function, dynamical_system, ode};

mod dae;
pub use dae::{fully_implicit_dae, mass_matrix_dae, semi_explicit_dae};

mod bvp;
pub use bvp::{bvp1d, Bvp1dClosures};

mod algebraic;
pub use algebraic::algebraic_constraint;

mod lti;
pub use lti::{
    pt1, pt2, statespace,
    transfer_function, transfer_function_num_den, transfer_function_prc,
    transfer_function_prc_mimo, transfer_function_zpg,
};

mod ctrl;
pub use ctrl::{anti_windup_pid, differentiator, lead_lag, pid};

mod nonlinear;
pub use nonlinear::{
    backlash, comparator, counter, counter_down, counter_up,
    deadband, rate_limiter, relay, switch,
};

mod spectrum;
pub use spectrum::{spectrum, spectrum_read};

mod discrete;
pub use discrete::{
    adc, dac, delay, fir, first_order_hold, sample_hold,
    wrapper, discrete_integrator, discrete_derivative, discrete_state_space,
    discrete_transfer_function, tapped_delay,
};

mod math_logic;
pub use math_logic::{
    abs_block, alias_block, atan2_block, atan_block, clip_block, cos_block, cosh_block,
    equal_block, exp_block, greater_than, less_than, log10_block, log_block,
    logic_and, logic_block, logic_not, logic_or, math_block, matrix_block,
    mod_block, mod_single, norm_block, polynomial, pow_block, pow_prod,
    rescale_block, sin_block, sinh_block, sqrt_block, tan_block, tanh_block,
};

pub(crate) mod sources;
pub use sources::{
    chirp_phase_noise_source, chirp_source, clock_source, gaussian_pulse_source,
    pulse_source, sinusoidal_phase_noise_source, sinusoidal_source,
    square_wave_source, step_source, triangle_wave_source,
};

mod noise;
pub use noise::{pink_noise, random_number_generator, white_noise};

pub(crate) mod table;
pub use table::{lut1d, ExtrapMode};

mod filters;
pub use filters::{
    allpass_filter, butter_bandpass, butter_bandstop, butter_highpass, butter_lowpass,
};

















#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrator() {
        let blk = integrator(5.0);
        let b = blk.borrow_mut();
        b.update(0.0);
        assert_eq!(b.outputs.get_single(0), 5.0);
        assert_eq!(b.len(), 0);
    }

    #[test]
    fn test_amplifier() {
        let blk = amplifier(3.0);
        blk.borrow_mut().inputs.set_single(0, 4.0);
        blk.borrow_mut().update(0.0);
        assert_eq!(blk.borrow().outputs.get_single(0), 12.0);
    }

    #[test]
    fn test_adder() {
        let blk = adder(None);
        blk.borrow_mut().inputs.update_from_array(&[1.0, 2.0, 3.0]);
        blk.borrow_mut().update(0.0);
        assert_eq!(blk.borrow().outputs.get_single(0), 6.0);
    }

    #[test]
    fn test_adder_operations() {
        let blk = adder(Some("+-"));
        blk.borrow_mut().inputs.update_from_array(&[10.0, 3.0]);
        blk.borrow_mut().update(0.0);
        assert_eq!(blk.borrow().outputs.get_single(0), 7.0);
    }

    #[test]
    fn test_divider_default_equals_multiplier() {
        // operations=None → identical to Multiplier (all inputs multiplied).
        let blk = divider(None, ZeroDiv::Warn).unwrap();
        blk.borrow_mut().inputs.update_from_array(&[2.0, 3.0, 4.0]);
        blk.borrow_mut().update(0.0);
        assert_eq!(blk.borrow().outputs.get_single(0), 24.0);
    }

    #[test]
    fn test_divider_star_slash() {
        // Pathsim default ops "*/": num / den = 10 / 4 = 2.5.
        let blk = divider(Some("*/"), ZeroDiv::Warn).unwrap();
        blk.borrow_mut().inputs.update_from_array(&[10.0, 4.0]);
        blk.borrow_mut().update(0.0);
        assert_eq!(blk.borrow().outputs.get_single(0), 2.5);
    }

    #[test]
    fn test_divider_mixed_ops() {
        // "**/" → (u0 * u1) / u2.
        let blk = divider(Some("**/"), ZeroDiv::Warn).unwrap();
        blk.borrow_mut().inputs.update_from_array(&[6.0, 4.0, 3.0]);
        blk.borrow_mut().update(0.0);
        assert_eq!(blk.borrow().outputs.get_single(0), 8.0);
    }

    #[test]
    fn test_divider_zero_div_clamp() {
        let blk = divider(Some("*/"), ZeroDiv::Clamp).unwrap();
        blk.borrow_mut().inputs.update_from_array(&[1.0, 0.0]);
        blk.borrow_mut().update(0.0);
        let y = blk.borrow().outputs.get_single(0);
        assert!(y.is_finite() && y > 1e15, "clamp should keep output finite-but-large, got {}", y);
    }

    #[test]
    fn test_divider_zero_div_raise() {
        // A zero denominator under ZeroDiv::Raise no longer panics (issue #28):
        // it records a catchable runtime fault and requests a cooperative stop,
        // which the run wrapper re-raises as a Python exception.
        crate::simulation::clear_stop_requested();
        let blk = divider(Some("*/"), ZeroDiv::Raise).unwrap();
        blk.borrow_mut().inputs.update_from_array(&[1.0, 0.0]);
        blk.borrow_mut().update(0.0);
        assert!(crate::simulation::take_stop_requested(), "zero denom must request stop");
        let fault = crate::simulation::take_runtime_fault();
        assert!(fault.is_some(), "zero denom must record a runtime fault");
        assert!(fault.unwrap().to_string().contains("denominator evaluated to zero"));
    }

    #[test]
    fn test_constant() {
        let blk = constant(42.0);
        blk.borrow_mut().update(0.0);
        assert_eq!(blk.borrow().outputs.get_single(0), 42.0);
        assert_eq!(blk.borrow().len(), 0);
    }

    #[test]
    fn test_source() {
        let blk = source(|t| t * 2.0);
        blk.borrow_mut().update(5.0);
        assert_eq!(blk.borrow().outputs.get_single(0), 10.0);
    }

    #[test]
    fn test_scope() {
        let blk = scope(None, 0.0, Vec::new());
        blk.borrow_mut().inputs.set_single(0, 1.0);
        blk.borrow_mut().sample(0.0, 0.01);
        blk.borrow_mut().inputs.set_single(0, 2.0);
        blk.borrow_mut().sample(0.1, 0.01);

        let (times, data) = scope_read(blk.borrow());
        assert_eq!(times.len(), 2);
        assert_eq!(data[0], vec![1.0]);
        assert_eq!(data[1], vec![2.0]);
    }

    #[test]
    fn test_function_block() {
        let blk = function(|u: &[f64]| vec![u[0] * u[0]]);
        blk.borrow_mut().inputs.set_single(0, 3.0);
        blk.borrow_mut().update(0.0);
        assert_eq!(blk.borrow().outputs.get_single(0), 9.0);
    }

    #[test]
    fn test_ode_block() {
        let blk = ode(|x: &[f64], _u: &[f64], _t: f64| vec![-x[0]], &[1.0], None);
        blk.borrow_mut().update(0.0);
        assert_eq!(blk.borrow().outputs.get_single(0), 1.0);
        assert_eq!(blk.borrow().len(), 0);
    }

    #[test]
    fn test_pt1_block() {
        let blk = pt1(2.0, 0.5);
        let b = blk.borrow();
        assert_eq!(b.size(), (1, 1)); // 1 state
        assert_eq!(b.len(), 0); // no passthrough (D=0)
    }

    #[test]
    fn test_math_sin() {
        let blk = sin_block();
        blk.borrow_mut().inputs.set_single(0, std::f64::consts::FRAC_PI_2);
        blk.borrow_mut().update(0.0);
        assert!((blk.borrow().outputs.get_single(0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_all_blocks_are_blockref() {
        // All constructor functions return BlockRef — no type conflicts
        let blocks: Vec<BlockRef> = vec![
            integrator(0.0),
            amplifier(2.0),
            adder(None),
            multiplier(),
            constant(5.0),
            source(|t| t),
            scope(None, 0.0, Vec::new()),
            function(|u: &[f64]| vec![u[0]]),
            ode(|x: &[f64], _u: &[f64], _t: f64| vec![-x[0]], &[1.0], None),
            pt1(1.0, 1.0),
            sin_block(),
            greater_than(),
        ];

        // All are the same type — can be stored in one Vec
        assert_eq!(blocks.len(), 12);

        // Can all be used uniformly
        for blk in &blocks {
            let _ = blk.borrow().len();
            let _ = blk.borrow().is_active();
        }
    }
}





#[cfg(test)]
mod wrapper_tests {
    use super::*;

    #[test]
    fn test_relay_event_detection() {
        let rly = relay(5.0, -5.0, 100.0, 0.0);

        // Set input below threshold, run update
        rly.borrow_mut().inputs._data[0] = 3.0;
        rly.borrow_mut().update(0.0);
        eprintln!("relay output after input=3: {}", rly.borrow().outputs._data[0]);

        // Buffer events at current state (input=3, func_evt = 3-5 = -2)
        for evt in &rly.borrow().events {
            evt.borrow_mut().buffer(0.0);
        }

        // Now set input above threshold
        rly.borrow_mut().inputs._data[0] = 7.0;
        rly.borrow_mut().update(1.0);
        eprintln!("relay output after input=7: {}", rly.borrow().outputs._data[0]);

        // Detect events (func_evt = 7-5 = +2, prev was -2 → crossing!)
        for (i, evt) in rly.borrow().events.iter().enumerate() {
            let (det, close, ratio) = evt.borrow_mut().detect(1.0);
            eprintln!("  event[{}]: detected={}, close={}, ratio={}", i, det, close, ratio);
        }
    }

    #[test]
    fn test_sample_hold_event_fires() {
        let cst = constant(3.0);
        let sh = sample_hold(0.1, 0.0);
        let sco = scope(None, 0.0, vec![]);

        let conn1 = crate::connection::Connection::new(
            crate::utils::portreference::PortReference::new(cst.clone(), None),
            vec![crate::utils::portreference::PortReference::new(sh.clone(), None)],
        );
        let conn2 = crate::connection::Connection::new(
            crate::utils::portreference::PortReference::new(sh.clone(), None),
            vec![crate::utils::portreference::PortReference::new(sco.clone(), None)],
        );

        let mut sim = crate::simulation::Simulation::with_defaults(
            vec![cst.clone(), sh.clone(), sco.clone()],
            vec![Rc::new(conn1), Rc::new(conn2)],
        );
        sim.run(0.5, false, false);

        let sh_out = sh.borrow().outputs.get_single(0);
        assert!(sh_out == 3.0, "SH output should be 3.0 after run, got {}", sh_out);

        let (_times, data) = scope_read(sco.borrow());
        // t=0 is 0.0 (event fires but input hasn't propagated yet) — correct behavior
        // From t=period onwards, scope should see 3.0
        let later_values: Vec<f64> = data.iter().skip(1).map(|d| d[0]).collect();
        assert!(later_values.contains(&3.0),
            "Scope should see 3.0 after first event period, got {:?}", &later_values[..5.min(later_values.len())]);
    }
}
