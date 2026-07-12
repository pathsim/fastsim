// Full-simulation tests driving FMU blocks through fastsim::simulation::Simulation.
//
// These are the "real" validation tests: we wire the FMU block into a block
// graph with a Scope for tracing, run the Simulation forward, and check the
// recorded trajectory against analytical expectations.

use std::rc::Rc;

use fastsim::blocks::block::BlockRef;
use fastsim::blocks::constructors::{scope, scope_read};
use fastsim::blocks::fmu::model_exchange_fmu;
use fastsim::connection::Connection;
use fastsim::simulation::Simulation;
use fastsim::utils::portreference::{Port, PortReference};

const DAHLQUIST_FMU: &str = "tests/fixtures/fmi/Dahlquist.fmu";
const BOUNCING_BALL_FMU: &str = "tests/fixtures/fmi/BouncingBall.fmu";
const VAN_DER_POL_FMU: &str = "tests/fixtures/fmi/VanDerPol.fmu";

fn connect_port(src: &BlockRef, src_port: usize, dst: &BlockRef, dst_port: usize) -> Rc<Connection> {
    Rc::new(Connection::new(
        PortReference::new(src.clone(), Some(vec![Port::Index(src_port)])),
        vec![PortReference::new(
            dst.clone(),
            Some(vec![Port::Index(dst_port)]),
        )],
    ))
}

fn connect_all(src: &BlockRef, dst: &BlockRef) -> Rc<Connection> {
    Rc::new(Connection::new(
        PortReference::new(src.clone(), None),
        vec![PortReference::new(dst.clone(), None)],
    ))
}

// -------------------------------------------------------------------------
// Dahlquist — no events, analytical reference: x(t) = exp(-t)
// -------------------------------------------------------------------------

#[test]
fn dahlquist_me_full_simulation() {
    let fmu = model_exchange_fmu(DAHLQUIST_FMU, "dq", None, 1e-8, false).expect("ctor");
    let scp = scope(None, 0.0, vec!["x".into()]);

    // Dahlquist has 1 output (x); wire it to scope port 0.
    let conn = connect_all(&fmu, &scp);

    let mut sim = Simulation::with_defaults(vec![fmu, scp.clone()], vec![conn]);
    sim.dt = 1e-3;
    sim.run(1.0, true, false);

    let (times, data) = scope_read(scp.borrow());
    assert!(!times.is_empty(), "scope recorded no samples");

    // Find x at t≈1.0 (last sample).
    let x_final = data.last().unwrap()[0];
    let expected = (-1.0_f64).exp();
    assert!(
        (x_final - expected).abs() < 1e-3,
        "x(1)={} vs exp(-1)={}",
        x_final,
        expected
    );
}

// -------------------------------------------------------------------------
// BouncingBall — the hero test. State event on h=0; velocity flips × -e.
// -------------------------------------------------------------------------

#[test]
fn bouncing_ball_me_full_simulation() {
    let fmu = model_exchange_fmu(BOUNCING_BALL_FMU, "bb", None, 1e-10, false).expect("ctor");
    let scp = scope(None, 0.0, vec!["h".into(), "v".into()]);

    // Wire port 0 (h) and port 1 (v) of the FMU into scope ports 0/1.
    let conn_h = connect_port(&fmu, 0, &scp, 0);
    let conn_v = connect_port(&fmu, 1, &scp, 1);

    let mut sim =
        Simulation::with_defaults(vec![fmu.clone(), scp.clone()], vec![conn_h, conn_v]);
    sim.dt = 1e-3;
    sim.run(1.5, true, false);

    let (times, data) = scope_read(scp.borrow());
    assert!(!times.is_empty());

    // The FMU's ZeroCrossing fires when `h` crosses zero. We should see at
    // least one recorded event on the block's ZeroCrossing event.
    let bb_events = &fmu.borrow().events;
    // events[0] = ScheduleList, events[1] = ZeroCrossing.
    let zc = bb_events[1].clone();
    let n_events = zc.borrow().len();
    assert!(
        n_events >= 1,
        "expected at least one bounce event, got {}",
        n_events
    );

    // h should never go significantly below 0 — the bounce keeps it on/above.
    let min_h = data.iter().map(|d| d[0]).fold(f64::INFINITY, f64::min);
    assert!(min_h > -5e-3, "ball went through floor: min h = {}", min_h);

    // After the first bounce (t ≈ 0.45s, v ≈ -4.43 pre, ~+3.10 post),
    // velocity should at some point be positive (ball going up).
    let v_max = data.iter().map(|d| d[1]).fold(f64::NEG_INFINITY, f64::max);
    assert!(
        v_max > 1.0,
        "expected positive velocity after bounce, max v = {}",
        v_max
    );
}

// -------------------------------------------------------------------------
// VanDerPol — no events. Oscillator: x' = y, y' = mu*(1-x²)*y - x, mu=1.
// Verify the solution oscillates (amplitude ≥ ~1.5, sign flips at least once).
// -------------------------------------------------------------------------

#[test]
fn van_der_pol_me_full_simulation() {
    let fmu = model_exchange_fmu(VAN_DER_POL_FMU, "vdp", None, 1e-8, false).expect("ctor");
    let scp = scope(None, 0.0, vec!["x0".into(), "x1".into()]);

    let conn_x0 = connect_port(&fmu, 0, &scp, 0);
    let conn_x1 = connect_port(&fmu, 1, &scp, 1);

    let mut sim = Simulation::with_defaults(
        vec![fmu.clone(), scp.clone()],
        vec![conn_x0, conn_x1],
    );
    sim.dt = 1e-2;
    sim.run(10.0, true, false);

    let (_times, data) = scope_read(scp.borrow());
    assert!(data.len() > 100);

    let x0_max = data.iter().map(|d| d[0]).fold(f64::NEG_INFINITY, f64::max);
    let x0_min = data.iter().map(|d| d[0]).fold(f64::INFINITY, f64::min);
    // VanDerPol's limit cycle amplitude for x is ~2 for mu=1.
    assert!(
        x0_max > 1.5 && x0_min < -1.5,
        "expected oscillation amplitude, got [{}, {}]",
        x0_min,
        x0_max
    );
}

// -------------------------------------------------------------------------
// BouncingBall in Co-Simulation — the FMU declares hasEventMode, so our
// cosimulation_fmu opts into event mode and translates `eventEncountered`
// flags from DoStep into full event-handling sequences.
// -------------------------------------------------------------------------

#[test]
fn bouncing_ball_cs_event_mode() {
    use fastsim::blocks::fmu::cosimulation_fmu;

    // Small dt so the FMU detects the bounce precisely.
    let fmu = cosimulation_fmu(BOUNCING_BALL_FMU, "bb_cs", None, Some(1e-2), false).expect("ctor");
    let scp = scope(None, 0.0, vec!["h".into(), "v".into()]);
    let conn_h = connect_port(&fmu, 0, &scp, 0);
    let conn_v = connect_port(&fmu, 1, &scp, 1);

    let mut sim = Simulation::with_defaults(
        vec![fmu.clone(), scp.clone()],
        vec![conn_h, conn_v],
    );
    sim.dt = 1e-2;
    sim.run(1.5, true, false);

    let (_t, data) = scope_read(scp.borrow());
    assert!(!data.is_empty());

    // Ball must stay above floor (bounce event mode triggered correctly).
    let min_h = data.iter().map(|d| d[0]).fold(f64::INFINITY, f64::min);
    assert!(min_h > -1e-2, "ball fell through floor: min h = {}", min_h);

    // Velocity flips positive after bounce.
    let v_max = data.iter().map(|d| d[1]).fold(f64::NEG_INFINITY, f64::max);
    assert!(v_max > 1.0, "expected upward velocity after bounce, got {}", v_max);
}

// -------------------------------------------------------------------------
// Error handling
// -------------------------------------------------------------------------

#[test]
fn invalid_start_value_key_returns_error() {
    use std::collections::HashMap;

    let mut starts = HashMap::new();
    starts.insert("does_not_exist".to_owned(), 42.0);
    let res = model_exchange_fmu(DAHLQUIST_FMU, "err", Some(starts), 1e-6, false);
    assert!(matches!(
        res,
        Err(fastsim::fmi::FmiError::UnknownVariable(_))
    ));
}
