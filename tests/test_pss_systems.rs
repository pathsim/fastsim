// Periodic-steady-state (PSS) shooting integration tests.
//
// Verifies the end-to-end PSS workflow:
//   - `Simulation::periodic_steady_state(period, inner_factory, ...)` converges
//   - Converged period-start state matches analytical / long-transient reference
//   - Works with both explicit (RKDP54) and implicit (ESDIRK43) inner solvers

use std::f64::consts::PI;
use std::rc::Rc;

use fastsim::blocks::constructors::*;
use fastsim::connection::Connection;
use fastsim::simulation::Simulation;
use fastsim::solvers::factories::{esdirk43_factory, rkdp54_factory};
use fastsim::utils::portreference::PortReference;

// ==================================================================================
// Test 1: Linear lowpass, sinusoidal forcing.
//
// ODE:    dx/dt = -x + sin(ωt)
// Steady state (analytical):
//   x(t) = A·sin(ωt) + B·cos(ωt), with A = 1/(1+ω²), B = -ω/(1+ω²)
//   ⇒ x(0) = B = -ω/(1+ω²)
//
// With ω = 1: x(0) = -1/2.
// ==================================================================================

#[test]
fn pss_linear_lowpass_matches_analytical() {
    const OMEGA: f64 = 1.0;
    const PERIOD: f64 = 2.0 * PI / OMEGA;
    const X0_EXPECTED: f64 = -OMEGA / (1.0 + OMEGA * OMEGA);  // = -0.5

    let src = source(move |t| (OMEGA * t).sin());
    let plant = dynamical_system(
        |x, u, _t| vec![-x[0] + u[0]],
        |x, _u, _t| vec![x[0]],
        &[0.0],
        false,
        None,
    );
    let sco = scope(None, 0.0, Vec::new());

    let conn_src_plant = Rc::new(Connection::new(
        PortReference::new(src.clone(), None),
        vec![PortReference::new(plant.clone(), None)],
    ));
    let conn_plant_sco = Rc::new(Connection::new(
        PortReference::new(plant.clone(), None),
        vec![PortReference::new(sco.clone(), None)],
    ));

    let mut sim = Simulation::with_solver(
        vec![src.clone(), plant.clone(), sco.clone()],
        vec![conn_src_plant, conn_plant_sco],
        rkdp54_factory(1e-10, 1e-8),
        0.01,
    );

    sim.periodic_steady_state(
        PERIOD,
        rkdp54_factory(1e-10, 1e-8),
        true,   // adaptive
        true,   // reset
    );

    // After PSS + final-transient run, sample at the end of the period
    // should match the period start (limit-cycle closure) and both should
    // match the analytical x(0).
    let (times, data) = scope_read(sco.borrow());
    assert!(times.len() > 10, "scope should have many samples");

    let x_first = data[0][0];
    let x_last = data[data.len() - 1][0];

    assert!((x_first - X0_EXPECTED).abs() < 1e-3,
        "PSS x(0) = {:.6}, analytical = {:.6}, diff = {:.2e}",
        x_first, X0_EXPECTED, (x_first - X0_EXPECTED).abs());

    assert!((x_last - x_first).abs() < 1e-3,
        "Limit-cycle closure: x(T) = {:.6}, x(0) = {:.6}, diff = {:.2e}",
        x_last, x_first, (x_last - x_first).abs());
}

// ==================================================================================
// Test 2: Same system, implicit inner solver (ESDIRK43).
//
// Verifies that PSS-augmented implicit solvers also work — the inner
// `stage_builder` / `opt` pass through transparently to the period
// integration.
// ==================================================================================

#[test]
fn pss_linear_lowpass_with_esdirk43() {
    const OMEGA: f64 = 1.0;
    const PERIOD: f64 = 2.0 * PI / OMEGA;
    const X0_EXPECTED: f64 = -0.5;

    let src = source(move |t| (OMEGA * t).sin());
    let plant = dynamical_system(
        |x, u, _t| vec![-x[0] + u[0]],
        |x, _u, _t| vec![x[0]],
        &[0.0],
        false,
        None,
    );
    let sco = scope(None, 0.0, Vec::new());

    let conn_src_plant = Rc::new(Connection::new(
        PortReference::new(src.clone(), None),
        vec![PortReference::new(plant.clone(), None)],
    ));
    let conn_plant_sco = Rc::new(Connection::new(
        PortReference::new(plant.clone(), None),
        vec![PortReference::new(sco.clone(), None)],
    ));

    let mut sim = Simulation::with_solver(
        vec![src.clone(), plant.clone(), sco.clone()],
        vec![conn_src_plant, conn_plant_sco],
        esdirk43_factory(1e-10, 1e-8),
        0.01,
    );

    sim.periodic_steady_state(
        PERIOD,
        esdirk43_factory(1e-10, 1e-8),
        true,
        true,
    );

    let (_, data) = scope_read(sco.borrow());
    let x_first = data[0][0];
    let x_last = data[data.len() - 1][0];

    assert!((x_first - X0_EXPECTED).abs() < 1e-3,
        "PSS (ESDIRK43) x(0) = {:.6}, analytical = {:.6}", x_first, X0_EXPECTED);
    assert!((x_last - x_first).abs() < 1e-3,
        "Limit-cycle closure: x(T) - x(0) = {:.2e}", (x_last - x_first).abs());
}

// ==================================================================================
// Test 3: Driven Van der Pol oscillator (2D, nonlinear) vs long-transient run.
//
// System:  ẍ - μ(1 - x²)ẋ + x = A·sin(ωt)
// State:   x₁ = x, x₂ = ẋ
//          dx₁/dt = x₂
//          dx₂/dt = μ(1 - x₁²)·x₂ - x₁ + A·sin(ωt)
//
// Run a long transient (40 periods) to reach the limit cycle, capture
// x(40·T).  Then run PSS from zero and verify the same period-start
// state (modulo phase, since the limit cycle is unique mod period).
// ==================================================================================

#[test]
fn pss_driven_van_der_pol_matches_long_transient() {
    const MU: f64 = 1.0;
    const A_FORCE: f64 = 1.2;
    const OMEGA: f64 = 1.0;                       // close to natural frequency → lock
    const PERIOD: f64 = 2.0 * PI / OMEGA;
    const N_WARMUP_PERIODS: f64 = 40.0;

    // --- Reference: long transient to reach steady state ---
    let ref_x_at_end = {
        let src = source(move |t| A_FORCE * (OMEGA * t).sin());
        let vdp = dynamical_system(
            move |x, u, _t| vec![x[1], MU * (1.0 - x[0] * x[0]) * x[1] - x[0] + u[0]],
            |x, _u, _t| vec![x[0]],
            &[0.5, 0.0],
            false,
            None,
        );
        let conn = Rc::new(Connection::new(
            PortReference::new(src.clone(), None),
            vec![PortReference::new(vdp.clone(), None)],
        ));
        let mut sim = Simulation::with_solver(
            vec![src.clone(), vdp.clone()],
            vec![conn],
            rkdp54_factory(1e-10, 1e-8),
            0.01,
        );
        // Long warm-up: 40 periods of transient.  Use multiple short runs to
        // keep stats reasonable; what matters is the state at the end.
        sim.run(N_WARMUP_PERIODS * PERIOD, false, true);
        // Read the state at the *end* of period N (sim.time = N·T).
        // To compare with PSS's period-start, we run one more period and
        // take the state at sim.time = (N+1)·T which equals x(0) of period N+1.
        vdp.borrow().engine.as_ref().unwrap().x.clone()
    };

    // --- PSS from zero ---
    let pss_x_at_start = {
        let src = source(move |t| A_FORCE * (OMEGA * t).sin());
        let vdp = dynamical_system(
            move |x, u, _t| vec![x[1], MU * (1.0 - x[0] * x[0]) * x[1] - x[0] + u[0]],
            |x, _u, _t| vec![x[0]],
            &[0.5, 0.0],
            false,
            None,
        );
        let conn = Rc::new(Connection::new(
            PortReference::new(src.clone(), None),
            vec![PortReference::new(vdp.clone(), None)],
        ));
        let mut sim = Simulation::with_solver(
            vec![src.clone(), vdp.clone()],
            vec![conn],
            rkdp54_factory(1e-10, 1e-8),
            0.01,
        );
        sim.periodic_steady_state(
            PERIOD,
            rkdp54_factory(1e-10, 1e-8),
            true,
            true,
        );
        // After PSS + final transient, sim.time = period, so engine.x = x(T)
        // which equals x(0) of the converged limit cycle.
        vdp.borrow().engine.as_ref().unwrap().x.clone()
    };

    // Both states should lie on the same limit cycle and at the same phase
    // (period start, t mod T = 0).  Tolerance accounts for finite transient
    // decay and integrator accuracy.
    let diff: f64 = pss_x_at_start.iter()
        .zip(ref_x_at_end.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(diff < 5e-2,
        "PSS state {:?} vs long-transient {:?}, diff = {:.4e}",
        pss_x_at_start, ref_x_at_end, diff);
}
