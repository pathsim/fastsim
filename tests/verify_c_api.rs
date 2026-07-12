//! End-to-end coverage of the SIL verification API (`codegen::verify`).
//!
//! Positive leg: a compiled oscillator's C trajectory matches the tape
//! reference within the default tolerances. Negative leg: a DIFFERENT model as
//! reference must fail — a verifier that cannot fail proves nothing.

#![cfg(feature = "codegen")]

mod common;

use std::rc::Rc;

use fastsim::blocks::constructors::{amplifier, integrator};
use fastsim::codegen::verify::{verify_c, VerifyCOptions};
use fastsim::codegen::{CodegenError, CodegenOptions};
use fastsim::connection::Connection;
use fastsim::ir::builder::module_from_sim;
use fastsim::simulation::Simulation;
use fastsim::utils::logger::Logger;
use fastsim::utils::portreference::PortReference;

fn conn(
    a: &fastsim::blocks::block::BlockRef,
    b: &fastsim::blocks::block::BlockRef,
) -> Rc<Connection> {
    Rc::new(Connection::new(
        PortReference::new(a.clone(), None),
        vec![PortReference::new(b.clone(), None)],
    ))
}

/// x'' = -x as two integrators + gain: x(0)=1, v(0)=0 -> x = cos t.
fn oscillator_sim() -> Simulation {
    let int_v = integrator(0.0);
    let int_x = integrator(1.0);
    let amp = amplifier(-1.0);
    Simulation::with_defaults(
        vec![int_v.clone(), int_x.clone(), amp.clone()],
        vec![conn(&int_v, &int_x), conn(&int_x, &amp), conn(&amp, &int_v)],
    )
}

/// Decay x' = -2x (deliberately different dynamics for the negative leg).
fn decay_sim() -> Simulation {
    let i = integrator(1.0);
    let k = amplifier(-2.0);
    Simulation::with_defaults(vec![i.clone(), k.clone()], vec![conn(&i, &k), conn(&k, &i)])
}

#[test]
fn verify_c_passes_on_matching_model() {
    if common::find_cc().is_none() {
        eprintln!("no working C compiler found — skipping verify_c positive leg");
        return;
    }
    let sim = oscillator_sim();
    let module = module_from_sim(&sim, "osc");
    let mut reference = fastsim::compile::compile(&module).expect("oscillator compiles");
    let report = verify_c(
        &module,
        &mut reference,
        &CodegenOptions::default(),
        &VerifyCOptions { duration: 2.0, dt: 1e-3, ..Default::default() },
        &Logger::disabled(),
    )
    .expect("verification runs");
    assert!(
        report.passed,
        "C vs tape deviates: max scaled error {} at {} ({:?})",
        report.max_scaled_error, report.worst_time, report.worst_state
    );
    assert_eq!(report.n_steps, 2000);
    assert_eq!(report.n_states, 2);
    assert!(report.build_dir.is_none(), "build dir cleaned up by default");
}

#[test]
fn verify_c_fails_on_mismatched_reference() {
    if common::find_cc().is_none() {
        eprintln!("no working C compiler found — skipping verify_c negative leg");
        return;
    }
    // C from the oscillator, reference from the decay: must NOT pass.
    let module = module_from_sim(&oscillator_sim(), "osc");
    let wrong = module_from_sim(&decay_sim(), "osc");
    let mut reference = fastsim::compile::compile(&wrong).expect("decay compiles");
    // Same n_state? oscillator has 2, decay has 1 -> state-count mismatch error.
    let err = verify_c(
        &module,
        &mut reference,
        &CodegenOptions::default(),
        &VerifyCOptions::default(),
        &Logger::disabled(),
    )
    .expect_err("state-count mismatch must be rejected");
    assert!(matches!(err, CodegenError::Verify(_)), "got {err:?}");

    // Same shape, different dynamics: gain -1 vs -0.5 -> trajectories diverge.
    let a = oscillator_sim();
    let module = module_from_sim(&a, "osc");
    let b_int_v = integrator(0.0);
    let b_int_x = integrator(1.0);
    let b_amp = amplifier(-0.5);
    let b = Simulation::with_defaults(
        vec![b_int_v.clone(), b_int_x.clone(), b_amp.clone()],
        vec![conn(&b_int_v, &b_int_x), conn(&b_int_x, &b_amp), conn(&b_amp, &b_int_v)],
    );
    let mut reference = fastsim::compile::compile(&module_from_sim(&b, "osc")).unwrap();
    let report = verify_c(
        &module,
        &mut reference,
        &CodegenOptions::default(),
        &VerifyCOptions { duration: 1.0, dt: 1e-2, ..Default::default() },
        &Logger::disabled(),
    )
    .expect("verification runs");
    assert!(!report.passed, "diverging dynamics must fail verification");
    assert!(report.max_scaled_error > 1.0);
}

#[test]
fn verify_c_rejects_adaptive_solver() {
    // No compiler needed: the adaptive rejection happens before tool lookup.
    let sim = oscillator_sim();
    let module = module_from_sim(&sim, "osc");
    let mut reference = fastsim::compile::compile(&module).unwrap();
    let cg = CodegenOptions {
        solver: fastsim::codegen::SolverChoice::by_name("RKBS32").expect("adaptive tableau"),
        ..Default::default()
    };
    let err = verify_c(&module, &mut reference, &cg, &VerifyCOptions::default(), &Logger::disabled())
        .expect_err("adaptive tableaus are out of scope");
    assert!(matches!(err, CodegenError::Verify(ref m) if m.contains("adaptive")), "got {err:?}");
}
