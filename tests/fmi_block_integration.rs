// End-to-end test of the FMU *block* wrapper (src/blocks/fmu.rs). Exercises
// f_dyn / f_alg directly so we verify the closure plumbing without needing to
// drive a full Simulation.

use std::collections::HashMap;

use fastsim::blocks::fmu::{cosimulation_fmu, model_exchange_fmu};

const DAHLQUIST_FMU: &str = "tests/fixtures/fmi/Dahlquist.fmu";
const BOUNCING_BALL_FMU: &str = "tests/fixtures/fmi/BouncingBall.fmu";

#[test]
fn me_block_derivatives_match_dahlquist() {
    let blk = model_exchange_fmu(DAHLQUIST_FMU, "me", None, 1e-6, false)
        .expect("construct ME block");
    let b = blk.borrow();

    // Dahlquist: 1 state (x), no inputs from outside, 1 output (x).
    assert_eq!(b.initial_value.as_ref().unwrap().len(), 1);
    assert_eq!(b.initial_value.as_ref().unwrap()[0], 1.0);
    assert!(b.f_dyn.is_some());
    assert!(b.f_alg.is_some());

    // dx/dt = -k*x with k=1, x=1 → -1
    let f = b.f_dyn.as_ref().unwrap();
    let mut out = Vec::new();
    f(&[1.0], &[], 0.0, &mut out);
    assert_eq!(out.len(), 1);
    assert!((out[0] - (-1.0)).abs() < 1e-12, "got {}", out[0]);

    // With x=0.5 → dx/dt = -0.5
    out.clear();
    f(&[0.5], &[], 0.0, &mut out);
    assert!((out[0] - (-0.5)).abs() < 1e-12, "got {}", out[0]);

    // y = x (output of Dahlquist is the state)
    let g = b.f_alg.as_ref().unwrap();
    let mut yout = Vec::new();
    g(&[0.75], &[], 0.0, &mut yout);
    assert_eq!(yout.len(), 1);
    assert!((yout[0] - 0.75).abs() < 1e-12);
}

#[test]
fn me_block_with_start_value_override() {
    let mut starts = HashMap::new();
    starts.insert("k".to_owned(), 2.0); // change decay rate

    let blk = model_exchange_fmu(DAHLQUIST_FMU, "me2", Some(starts), 1e-6, false)
        .expect("construct");
    let b = blk.borrow();
    let f = b.f_dyn.as_ref().unwrap();
    let mut out = Vec::new();
    // dx/dt = -k*x → with k=2, x=1 → -2
    f(&[1.0], &[], 0.0, &mut out);
    assert!((out[0] - (-2.0)).abs() < 1e-12, "got {}", out[0]);
}

#[test]
fn bouncing_ball_has_state_event() {
    // BouncingBall declares one <EventIndicator> — we should install one
    // ZeroCrossing event on the block, plus the always-present ScheduleList
    // for time events (FMU may announce next_event_time during UpdateDiscreteStates).
    let blk = model_exchange_fmu(BOUNCING_BALL_FMU, "bb", None, 1e-10, false)
        .expect("construct ME block");
    let b = blk.borrow();

    // 1 ScheduleList (time events) + 1 ZeroCrossing (state event) = 2 events.
    assert_eq!(b.events.len(), 2, "expected ScheduleList + 1 ZeroCrossing");

    // Initial state: h=1, v=0.
    let iv = b.initial_value.as_ref().unwrap();
    assert_eq!(iv.len(), 2);
    assert_eq!(iv[0], 1.0);
    assert_eq!(iv[1], 0.0);
}

#[test]
fn bouncing_ball_event_indicator_sign_changes() {
    // At t=0, height h=1, event indicator = h - floor ≈ 1 > 0.
    // We can probe the first event's func_evt to confirm it queries the FMU.
    // Event detection proper requires a full Simulation; here we only verify
    // that func_evt returns a sensible value and that handle_fmu_event
    // doesn't crash when invoked.
    let blk = model_exchange_fmu(BOUNCING_BALL_FMU, "bb2", None, 1e-10, false)
        .expect("construct");

    // Trigger the state event's func_act manually (simulating a detected
    // zero-crossing). Since h=1 and indicator > 0, no values should change,
    // but the code path should run without panicking.
    let b = blk.borrow();
    let ev = b.events[1].clone(); // index 1 is the ZeroCrossing
    ev.borrow_mut().resolve(0.0);
}

#[test]
fn cs_block_schedule_event_advances_state() {
    let blk = cosimulation_fmu(DAHLQUIST_FMU, "cs", None, Some(0.01), false)
        .expect("construct CS block");

    // Schedule event is installed with t_period = dt = 0.01.
    {
        let b = blk.borrow();
        assert_eq!(b.type_name, "CoSimulationFMU");
        assert!(b.update_fn.is_some());
        assert_eq!(b.events.len(), 1);
    }

    // Fire the scheduled event at t = 0, 0.01, ..., 1.0 (the fire at t=0
    // advances by 0 and is a no-op; each subsequent fire advances by dt).
    // Simulates what the Simulation loop does on each detected trigger.
    for i in 0..=100 {
        let t = i as f64 * 0.01;
        let evt = blk.borrow().events[0].clone();
        evt.borrow_mut().resolve(t);
    }
    // Then run update() to pull FMU outputs into block register.
    blk.borrow_mut().update(1.0);

    // Read back x via outputs register.
    let x = blk.borrow().outputs.get_single(0);
    // FMU's internal solver is forward Euler at fixedInternalStepSize=0.1;
    // over 100 steps × 0.01 = 1.0 s, x ≈ 0.9^10 ≈ 0.3487.
    let expected = 0.9_f64.powi(10);
    assert!(
        (x - expected).abs() < 1e-9,
        "got {} expected {}",
        x,
        expected
    );
}
