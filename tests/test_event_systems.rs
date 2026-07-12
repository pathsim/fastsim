// End-to-end integration tests for the event system
// Tests event detection, resolution, and interaction with simulation loop

use fastsim::utils::fastcell::FastCell;
use std::rc::Rc;

use fastsim::simulation::{Simulation, SimEventRef};
use fastsim::blocks::constructors::*;
use fastsim::connection::Connection;
use fastsim::events::zerocrossing::ZeroCrossing;
use fastsim::events::schedule::{Schedule, ScheduleList};
use fastsim::events::condition::Condition;


use fastsim::utils::portreference::PortReference;

// Helper: connect block output port 0 to another block input port 0
fn connect(src: &fastsim::blocks::block::BlockRef, dst: &fastsim::blocks::block::BlockRef) -> Rc<Connection> {
    Rc::new(Connection::new(
        PortReference::new(src.clone(), None),
        vec![PortReference::new(dst.clone(), None)],
    ))
}

// ======================================================================================
// Schedule events in simulation
// ======================================================================================

#[test]
fn test_schedule_event_fires_periodically() {
    // Schedule event at t=0.5, period=0.5 -> fires at 0.5, 1.0, 1.5, ...
    let times_log: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(Vec::new()));
    let times_clone = times_log.clone();

    let evt = Schedule::new(
        0.5, None, 0.5,
        Some(Box::new(move |t| { times_clone.borrow_mut().push(t); })),
        1e-10,
    );
    let evt_ref: SimEventRef = Rc::new(FastCell::new(evt));

    // Simple system: constant -> scope
    let c = constant(1.0);
    let s = scope(None, 0.0, vec![]);
    let conn = connect(&c, &s);

    let mut sim = Simulation::with_defaults(vec![c, s], vec![conn]);
    sim.add_event(evt_ref.clone());
    sim.dt = 0.01;

    sim.run(2.0, true, false);

    let fired = times_log.borrow();
    // Should fire at ~0.5, ~1.0, ~1.5, ~2.0
    assert!(fired.len() >= 3, "Schedule should fire at least 3 times, got {}", fired.len());

    // First event should be near t=0.5
    assert!((fired[0] - 0.5).abs() < 0.02, "First event at {}, expected ~0.5", fired[0]);
}

#[test]
fn test_cooperative_stop_halts_run_early() {
    // An event action that requests a cooperative stop at t=0.5 must terminate
    // the run well before the requested duration. This is the core mechanism
    // the pybindings StopSimulation handler drives (request_stop()).
    let evt = Schedule::new(
        0.5, None, 0.5,
        Some(Box::new(|_t| fastsim::simulation::request_stop())),
        1e-10,
    );
    let evt_ref: SimEventRef = Rc::new(FastCell::new(evt));

    let c = constant(1.0);
    let s = scope(None, 0.0, vec![]);
    let conn = connect(&c, &s);

    let mut sim = Simulation::with_defaults(vec![c, s], vec![conn]);
    sim.add_event(evt_ref.clone());
    sim.dt = 0.01;

    sim.run(10.0, true, false);

    assert!(!sim.is_active(), "sim should be inactive after a stop request");
    assert!(sim.time < 1.0, "sim should have stopped near t=0.5, got t={}", sim.time);
}

#[test]
fn test_schedule_list_fires_at_specific_times() {
    let times_log: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(Vec::new()));
    let times_clone = times_log.clone();

    let evt = ScheduleList::new(
        vec![0.25, 0.75, 1.5],
        Some(Box::new(move |t| { times_clone.borrow_mut().push(t); })),
        1e-10,
    );
    let evt_ref: SimEventRef = Rc::new(FastCell::new(evt));

    let c = constant(1.0);
    let s = scope(None, 0.0, vec![]);
    let conn = connect(&c, &s);

    let mut sim = Simulation::with_defaults(vec![c, s], vec![conn]);
    sim.add_event(evt_ref.clone());
    sim.dt = 0.01;

    sim.run(2.0, true, false);

    let fired = times_log.borrow();
    assert_eq!(fired.len(), 3, "ScheduleList should fire exactly 3 times, got {}", fired.len());

    // Verify approximate event times
    assert!((fired[0] - 0.25).abs() < 0.02, "Event 0 at {}, expected ~0.25", fired[0]);
    assert!((fired[1] - 0.75).abs() < 0.02, "Event 1 at {}, expected ~0.75", fired[1]);
    assert!((fired[2] - 1.5).abs() < 0.02, "Event 2 at {}, expected ~1.5", fired[2]);

    // Should deactivate after all events fired
    assert!(!evt_ref.borrow().is_active());
}

// ======================================================================================
// ZeroCrossing events in simulation
// ======================================================================================

#[test]
fn test_zerocrossing_detects_sign_change() {
    // Integrator from -1.0, rate=1.0 -> crosses zero at t=1.0
    let cross_times: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(Vec::new()));
    let cross_clone = cross_times.clone();

    let evt = ZeroCrossing::new(
        |t| t - 1.0,  // zero at t=1.0
        Some(Box::new(move |t| { cross_clone.borrow_mut().push(t); })),
        1e-6,
    );
    let evt_ref: SimEventRef = Rc::new(FastCell::new(evt));

    let c = constant(1.0);
    let s = scope(None, 0.0, vec![]);
    let conn = connect(&c, &s);

    let mut sim = Simulation::with_defaults(vec![c, s], vec![conn]);
    sim.add_event(evt_ref.clone());
    sim.dt = 0.01;

    sim.run(2.0, true, false);

    let fired = cross_times.borrow();
    assert!(!fired.is_empty(), "ZeroCrossing should detect at least one event");
    // Event should be near t=1.0
    assert!((fired[0] - 1.0).abs() < 0.02, "ZeroCrossing at {}, expected ~1.0", fired[0]);
}

#[test]
fn test_zerocrossing_up_only_positive_direction() {
    let cross_times: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(Vec::new()));
    let cross_clone = cross_times.clone();

    // sin(2*pi*t) crosses zero upward at t=0, 1, 2, ...
    // and downward at t=0.5, 1.5, ...
    let evt = ZeroCrossing::new_up(
        |t| (2.0 * std::f64::consts::PI * t).sin(),
        Some(Box::new(move |t| { cross_clone.borrow_mut().push(t); })),
        1e-6,
    );
    let evt_ref: SimEventRef = Rc::new(FastCell::new(evt));

    let c = constant(1.0);
    let s = scope(None, 0.0, vec![]);
    let conn = connect(&c, &s);

    let mut sim = Simulation::with_defaults(vec![c, s], vec![conn]);
    sim.add_event(evt_ref);
    sim.dt = 0.01;

    sim.run(2.5, true, false);

    let fired = cross_times.borrow();
    // Should only fire on upward crossings (near t=1.0, t=2.0 — t=0 is initial)
    // Should NOT fire at t=0.5, t=1.5 (downward)
    for &t in fired.iter() {
        // Each crossing should be near an integer (upward crossing)
        let nearest_int = t.round();
        assert!(
            (t - nearest_int).abs() < 0.05,
            "ZeroCrossingUp fired at t={}, not near upward crossing", t
        );
    }
}

#[test]
fn test_zerocrossing_down_only_negative_direction() {
    let cross_times: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(Vec::new()));
    let cross_clone = cross_times.clone();

    // sin(2*pi*t) crosses zero downward at t=0.5, 1.5, ...
    let evt = ZeroCrossing::new_down(
        |t| (2.0 * std::f64::consts::PI * t).sin(),
        Some(Box::new(move |t| { cross_clone.borrow_mut().push(t); })),
        1e-6,
    );
    let evt_ref: SimEventRef = Rc::new(FastCell::new(evt));

    let c = constant(1.0);
    let s = scope(None, 0.0, vec![]);
    let conn = connect(&c, &s);

    let mut sim = Simulation::with_defaults(vec![c, s], vec![conn]);
    sim.add_event(evt_ref);
    sim.dt = 0.01;

    sim.run(2.5, true, false);

    let fired = cross_times.borrow();
    // Should only fire on downward crossings (near t=0.5, t=1.5)
    for &t in fired.iter() {
        let nearest_half = (t * 2.0).round() / 2.0;
        // Downward crossings are at 0.5, 1.5, 2.5, ...
        let frac = nearest_half % 1.0;
        assert!(
            (frac - 0.5).abs() < 0.1,
            "ZeroCrossingDown fired at t={}, not near downward crossing", t
        );
    }
}

// ======================================================================================
// Condition events in simulation
// ======================================================================================

#[test]
fn test_condition_one_shot_deactivates() {
    let fired_at: Rc<FastCell<Option<f64>>> = Rc::new(FastCell::new(None));
    let fired_clone = fired_at.clone();

    // Condition: t > 0.5 -> fires once, then deactivates
    let evt = Condition::new(
        |t| t > 0.5,
        Some(Box::new(move |t| { *fired_clone.borrow_mut() = Some(t); })),
        1e-6,
    );
    let evt_ref: SimEventRef = Rc::new(FastCell::new(evt));

    let c = constant(1.0);
    let s = scope(None, 0.0, vec![]);
    let conn = connect(&c, &s);

    let mut sim = Simulation::with_defaults(vec![c, s], vec![conn]);
    sim.add_event(evt_ref.clone());
    sim.dt = 0.01;

    sim.run(2.0, true, false);

    let result = fired_at.borrow();
    assert!(result.is_some(), "Condition event should have fired");
    let t = result.unwrap();
    assert!((t - 0.5).abs() < 0.02, "Condition fired at {}, expected ~0.5", t);

    // Should be deactivated after firing
    assert!(!evt_ref.borrow().is_active());

    // Should have fired exactly once
    assert_eq!(evt_ref.borrow().len(), 1);
}

// ======================================================================================
// Multiple events in same simulation
// ======================================================================================

#[test]
fn test_multiple_events_coexist() {
    let schedule_count: Rc<FastCell<usize>> = Rc::new(FastCell::new(0));
    let zc_count: Rc<FastCell<usize>> = Rc::new(FastCell::new(0));
    let cond_count: Rc<FastCell<usize>> = Rc::new(FastCell::new(0));

    let sc = schedule_count.clone();
    let zc = zc_count.clone();
    let cc = cond_count.clone();

    let evt_schedule = Schedule::new(
        0.0, None, 0.5,
        Some(Box::new(move |_t| { *sc.borrow_mut() += 1; })),
        1e-10,
    );

    let evt_zc = ZeroCrossing::new(
        |t| t - 0.75,
        Some(Box::new(move |_t| { *zc.borrow_mut() += 1; })),
        1e-6,
    );

    let evt_cond = Condition::new(
        |t| t > 1.2,
        Some(Box::new(move |_t| { *cc.borrow_mut() += 1; })),
        1e-6,
    );

    let c = constant(1.0);
    let s = scope(None, 0.0, vec![]);
    let conn = connect(&c, &s);

    let mut sim = Simulation::with_defaults(vec![c, s], vec![conn]);
    sim.add_event(Rc::new(FastCell::new(evt_schedule)));
    sim.add_event(Rc::new(FastCell::new(evt_zc)));
    sim.add_event(Rc::new(FastCell::new(evt_cond)));
    sim.dt = 0.01;

    sim.run(2.0, true, false);

    assert!(*schedule_count.borrow() >= 3, "Schedule should fire multiple times");
    assert!(*zc_count.borrow() >= 1, "ZeroCrossing should fire at least once");
    assert_eq!(*cond_count.borrow(), 1, "Condition should fire exactly once");
}

// ======================================================================================
// Event add/remove
// ======================================================================================

#[test]
fn test_add_and_remove_event() {
    let fired: Rc<FastCell<bool>> = Rc::new(FastCell::new(false));
    let fired_clone = fired.clone();

    let evt = ZeroCrossing::new(
        |t| t - 0.5,
        Some(Box::new(move |_t| { *fired_clone.borrow_mut() = true; })),
        1e-6,
    );
    let evt_ref: SimEventRef = Rc::new(FastCell::new(evt));

    let c = constant(1.0);
    let s = scope(None, 0.0, vec![]);
    let conn = connect(&c, &s);

    let mut sim = Simulation::with_defaults(vec![c, s], vec![conn]);
    sim.add_event(evt_ref.clone());
    let _ = sim.remove_event(&evt_ref);
    sim.dt = 0.01;

    sim.run(2.0, true, false);

    assert!(!*fired.borrow(), "Removed event should not fire");
}

#[test]
fn test_remove_nonexistent_event_returns_error() {
    let evt = ZeroCrossing::from_evt(|t| t - 1.0);
    let evt_ref: SimEventRef = Rc::new(FastCell::new(evt));

    let c = constant(1.0);
    let s = scope(None, 0.0, vec![]);
    let conn = connect(&c, &s);

    let mut sim = Simulation::with_defaults(vec![c, s], vec![conn]);
    let result = sim.remove_event(&evt_ref);
    assert!(result.is_err());
}

// ======================================================================================
// Event reset and reuse
// ======================================================================================

#[test]
fn test_event_reset_allows_refire() {
    let fire_count: Rc<FastCell<usize>> = Rc::new(FastCell::new(0));
    let fc = fire_count.clone();

    let evt = Condition::new(
        |t| t > 0.3,
        Some(Box::new(move |_t| { *fc.borrow_mut() += 1; })),
        1e-6,
    );
    let evt_ref: SimEventRef = Rc::new(FastCell::new(evt));

    let c = constant(1.0);
    let s = scope(None, 0.0, vec![]);
    let conn = connect(&c, &s);

    let mut sim = Simulation::with_defaults(vec![c, s], vec![conn]);
    sim.add_event(evt_ref.clone());
    sim.dt = 0.01;

    // First run
    sim.run(1.0, true, false);
    assert_eq!(*fire_count.borrow(), 1);
    assert!(!evt_ref.borrow().is_active());

    // Reset event and run again
    evt_ref.borrow_mut().reset();
    assert!(evt_ref.borrow().is_active());

    sim.run(1.0, true, false);
    assert_eq!(*fire_count.borrow(), 2);
}

// ======================================================================================
// Schedule with end time
// ======================================================================================

#[test]
fn test_schedule_deactivates_at_end_time() {
    let fire_count: Rc<FastCell<usize>> = Rc::new(FastCell::new(0));
    let fc = fire_count.clone();

    // Schedule: start=0, end=1.0, period=0.25
    // Should fire at: 0, 0.25, 0.5, 0.75, 1.0 then stop
    let evt = Schedule::new(
        0.0, Some(1.0), 0.25,
        Some(Box::new(move |_t| { *fc.borrow_mut() += 1; })),
        1e-10,
    );
    let evt_ref: SimEventRef = Rc::new(FastCell::new(evt));

    let c = constant(1.0);
    let s = scope(None, 0.0, vec![]);
    let conn = connect(&c, &s);

    let mut sim = Simulation::with_defaults(vec![c, s], vec![conn]);
    sim.add_event(evt_ref.clone());
    sim.dt = 0.01;

    sim.run(3.0, true, false);

    let count = *fire_count.borrow();
    assert!(count <= 6, "Schedule with end_time should stop, got {} fires", count);
    assert!(count >= 3, "Schedule should fire at least 3 times before end, got {}", count);
}

// ======================================================================================
// Event on/off control
// ======================================================================================

#[test]
fn test_event_on_off_control() {
    let fire_count: Rc<FastCell<usize>> = Rc::new(FastCell::new(0));
    let fc = fire_count.clone();

    let evt = Schedule::new(
        0.5, None, 0.25,
        Some(Box::new(move |_t| { *fc.borrow_mut() += 1; })),
        1e-10,
    );
    let evt_ref: SimEventRef = Rc::new(FastCell::new(evt));

    let c = constant(1.0);
    let s = scope(None, 0.0, vec![]);
    let conn = connect(&c, &s);

    let mut sim = Simulation::with_defaults(vec![c, s], vec![conn]);
    sim.add_event(evt_ref.clone());
    sim.dt = 0.01;

    // Deactivate before running (use reset=false so event stays off)
    evt_ref.borrow_mut().off();

    sim.run(2.0, false, false);
    assert_eq!(*fire_count.borrow(), 0, "Deactivated event should not fire");

    // Re-activate and continue
    evt_ref.borrow_mut().on();
    sim.run(2.0, false, false);
    assert!(*fire_count.borrow() > 0, "Re-activated event should fire");
}

// ======================================================================================
// Multi-event timestep: two-phase resolution (regression for double-resolve)
// ======================================================================================

#[test]
fn test_adaptive_multi_event_timestep_resolves_each_once() {
    // Mixed Schedule + Condition inside the same adaptive timestep. With a large
    // initial dt the solver will see a "close" Schedule and a not-yet-close
    // Condition in the same detected-events list. The two-phase resolve logic
    // must revert first and resolve only in a timestep where all events are close.
    let sched_fires: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(Vec::new()));
    let cond_fires: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(Vec::new()));
    let sf = sched_fires.clone();
    let cf = cond_fires.clone();

    let sched = Schedule::new(
        0.5, None, 10.0,
        Some(Box::new(move |t| { sf.borrow_mut().push(t); })),
        1e-8,
    );
    let cond = Condition::new(
        |t| t > 0.3,
        Some(Box::new(move |t| { cf.borrow_mut().push(t); })),
        0.05,
    );

    let c = constant(1.0);
    let s = scope(None, 0.0, vec![]);
    let conn = connect(&c, &s);

    let mut sim = Simulation::with_defaults(vec![c, s], vec![conn]);
    sim.add_event(Rc::new(FastCell::new(sched)));
    sim.add_event(Rc::new(FastCell::new(cond)));
    sim.dt = 1.0;

    sim.run(2.0, true, false);

    let sched_log = sched_fires.borrow();
    let cond_log = cond_fires.borrow();
    assert_eq!(sched_log.len(), 1, "Schedule should fire exactly once, got {:?}", *sched_log);
    assert_eq!(cond_log.len(), 1, "Condition should fire exactly once, got {:?}", *cond_log);
    assert!((sched_log[0] - 0.5).abs() < 0.02, "Schedule fired at {}, expected ~0.5", sched_log[0]);
    assert!(cond_log[0] >= 0.3 && cond_log[0] <= 0.5, "Condition fired at {}, expected in [0.3, 0.5]", cond_log[0]);
}

#[test]
fn test_adaptive_simultaneous_zerocrossings_resolve_once_each() {
    // Two ZeroCrossings that cross exactly at the same time. Both land "close"
    // in the same timestep. Each func_act must be called exactly once.
    let c1: Rc<FastCell<usize>> = Rc::new(FastCell::new(0));
    let c2: Rc<FastCell<usize>> = Rc::new(FastCell::new(0));
    let c1c = c1.clone();
    let c2c = c2.clone();

    let zc1 = ZeroCrossing::new(
        |t| t - 0.5,
        Some(Box::new(move |_| { *c1c.borrow_mut() += 1; })),
        1e-6,
    );
    let zc2 = ZeroCrossing::new(
        |t| t - 0.5,
        Some(Box::new(move |_| { *c2c.borrow_mut() += 1; })),
        1e-6,
    );

    let c = constant(1.0);
    let s = scope(None, 0.0, vec![]);
    let conn = connect(&c, &s);

    let mut sim = Simulation::with_defaults(vec![c, s], vec![conn]);
    sim.add_event(Rc::new(FastCell::new(zc1)));
    sim.add_event(Rc::new(FastCell::new(zc2)));
    sim.dt = 0.1;

    sim.run(1.0, true, false);

    assert_eq!(*c1.borrow(), 1, "zc1 should fire exactly once");
    assert_eq!(*c2.borrow(), 1, "zc2 should fire exactly once");
}
