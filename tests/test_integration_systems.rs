// End-to-end integration tests
// Tests block construction, simulation lifecycle, and data flow

use std::rc::Rc;

use fastsim::simulation::Simulation;
use fastsim::blocks::block::BlockRef;
use fastsim::blocks::constructors::*;
use fastsim::connection::Connection;
use fastsim::utils::portreference::{PortReference, Port};

// ==================================================================================
// Block constructor tests
// ==================================================================================

#[test]
fn test_integrator_block() {
    let blk = integrator(5.0);
    blk.borrow_mut().update(0.0);
    assert_eq!(blk.borrow().outputs.get_single(0), 5.0);
    assert_eq!(blk.borrow().len(), 0);
}

#[test]
fn test_amplifier_block() {
    let blk = amplifier(3.0);
    blk.borrow_mut().inputs.set_single(0, 4.0);
    blk.borrow_mut().update(0.0);
    assert_eq!(blk.borrow().outputs.get_single(0), 12.0);
}

#[test]
fn test_adder_block() {
    let blk = adder(None);
    blk.borrow_mut().inputs.update_from_array(&[1.0, 2.0, 3.0]);
    blk.borrow_mut().update(0.0);
    assert_eq!(blk.borrow().outputs.get_single(0), 6.0);
}

#[test]
fn test_constant_block() {
    let blk = constant(42.0);
    blk.borrow_mut().update(0.0);
    assert_eq!(blk.borrow().outputs.get_single(0), 42.0);
}

#[test]
fn test_source_block() {
    let blk = source(|t| t * 2.0);
    blk.borrow_mut().update(5.0);
    assert_eq!(blk.borrow().outputs.get_single(0), 10.0);
}

#[test]
fn test_scope_block() {
    let blk = scope(None, 0.0, Vec::new());
    blk.borrow_mut().inputs.set_single(0, 1.0);
    blk.borrow_mut().sample(0.0, 0.01);
    blk.borrow_mut().inputs.set_single(0, 2.0);
    blk.borrow_mut().sample(0.1, 0.01);

    let (times, data) = scope_read(blk.borrow());
    assert_eq!(times.len(), 2);
    assert_eq!(data[0], vec![1.0]);
}

// ==================================================================================
// Connection tests with BlockRef
// ==================================================================================

#[test]
fn test_connection_between_blocks() {
    let src = constant(7.0);
    let dst = amplifier(2.0);

    // Create connection: src[0] -> dst[0]
    let conn = Rc::new(Connection::new(
        PortReference::new(src.clone(), None),
        vec![PortReference::new(dst.clone(), None)],
    ));

    // Update source
    src.borrow_mut().update(0.0);
    assert_eq!(src.borrow().outputs.get_single(0), 7.0);

    // Transfer data via connection
    conn.update();
    assert_eq!(dst.borrow().inputs.get_single(0), 7.0);

    // Update destination
    dst.borrow_mut().update(0.0);
    assert_eq!(dst.borrow().outputs.get_single(0), 14.0);
}

#[test]
fn test_connection_multi_target() {
    let src = constant(5.0);
    let dst1 = amplifier(1.0);
    let dst2 = amplifier(2.0);

    let conn = Rc::new(Connection::new(
        PortReference::new(src.clone(), None),
        vec![
            PortReference::new(dst1.clone(), None),
            PortReference::new(dst2.clone(), None),
        ],
    ));

    src.borrow_mut().update(0.0);
    conn.update();

    assert_eq!(dst1.borrow().inputs.get_single(0), 5.0);
    assert_eq!(dst2.borrow().inputs.get_single(0), 5.0);
}

// ==================================================================================
// Simulation with blocks and connections
// ==================================================================================

#[test]
fn test_simulation_constant_to_scope() {
    let src = constant(42.0);
    let sco = scope(None, 0.0, Vec::new());

    let conn = Rc::new(Connection::new(
        PortReference::new(src.clone(), None),
        vec![PortReference::new(sco.clone(), None)],
    ));

    let mut sim = Simulation::with_defaults(
        vec![src.clone(), sco.clone()],
        vec![conn],
    );

    sim.run(0.1, false, false);

    let (times, data) = scope_read(sco.borrow());
    assert!(!times.is_empty(), "Scope should have recorded data");
    // All recorded values should be 42.0
    for d in &data {
        assert_eq!(d[0], 42.0);
    }
}

#[test]
fn test_simulation_source_to_scope() {
    let src = source(|t| t.sin());
    let sco = scope(None, 0.0, Vec::new());

    let conn = Rc::new(Connection::new(
        PortReference::new(src.clone(), None),
        vec![PortReference::new(sco.clone(), None)],
    ));

    let mut sim = Simulation::with_defaults(
        vec![src.clone(), sco.clone()],
        vec![conn],
    );

    sim.run(1.0, false, false);

    let (times, data) = scope_read(sco.borrow());
    assert!(times.len() > 10, "Should have many samples");

    // Verify data matches sin(t)
    for (i, &t) in times.iter().enumerate() {
        let expected = t.sin();
        let actual = data[i][0];
        assert!((actual - expected).abs() < 1e-10,
            "At t={}, expected sin(t)={}, got {}", t, expected, actual);
    }
}

#[test]
fn test_simulation_amplifier_chain() {
    // Constant(3) -> Amp(2) -> Amp(5) -> Scope
    // Expected: 3 * 2 * 5 = 30
    let src = constant(3.0);
    let amp1 = amplifier(2.0);
    let amp2 = amplifier(5.0);
    let sco = scope(None, 0.0, Vec::new());

    let c1 = Rc::new(Connection::new(
        PortReference::new(src.clone(), None),
        vec![PortReference::new(amp1.clone(), None)],
    ));
    let c2 = Rc::new(Connection::new(
        PortReference::new(amp1.clone(), None),
        vec![PortReference::new(amp2.clone(), None)],
    ));
    let c3 = Rc::new(Connection::new(
        PortReference::new(amp2.clone(), None),
        vec![PortReference::new(sco.clone(), None)],
    ));

    let mut sim = Simulation::with_defaults(
        vec![src, amp1, amp2, sco.clone()],
        vec![c1, c2, c3],
    );

    sim.run(0.1, false, false);

    let (_, data) = scope_read(sco.borrow());
    assert!(!data.is_empty());
    // Should be 3 * 2 * 5 = 30
    assert_eq!(data.last().unwrap()[0], 30.0);
}

#[test]
fn test_simulation_adder() {
    // Const(3) -> Adder[0]
    // Const(7) -> Adder[1]
    // Adder -> Scope
    // Expected: 3 + 7 = 10
    let c1 = constant(3.0);
    let c2 = constant(7.0);
    let add = adder(None);
    let sco = scope(None, 0.0, Vec::new());

    let conn1 = Rc::new(Connection::new(
        PortReference::new(c1.clone(), None),
        vec![PortReference::new(add.clone(), Some(vec![Port::Index(0)]))],
    ));
    let conn2 = Rc::new(Connection::new(
        PortReference::new(c2.clone(), None),
        vec![PortReference::new(add.clone(), Some(vec![Port::Index(1)]))],
    ));
    let conn3 = Rc::new(Connection::new(
        PortReference::new(add.clone(), None),
        vec![PortReference::new(sco.clone(), None)],
    ));

    let mut sim = Simulation::with_defaults(
        vec![c1, c2, add, sco.clone()],
        vec![conn1, conn2, conn3],
    );

    sim.run(0.05, false, false);

    let (_, data) = scope_read(sco.borrow());
    assert!(!data.is_empty());
    assert_eq!(data.last().unwrap()[0], 10.0);
}

#[test]
fn test_simulation_reset() {
    let mut sim = Simulation::with_defaults(Vec::new(), Vec::new());
    sim.run(1.0, false, false);
    assert!(sim.time > 0.0);
    sim.reset(0.0);
    assert_eq!(sim.time, 0.0);
}

#[test]
fn test_simulation_diagnostics() {
    let mut sim = Simulation::new(
        Vec::new(), Vec::new(), Vec::new(),
        0.01, 1e-16, None, 200, true,
    );
    sim.run(0.1, false, false);
    assert!(sim.diagnostics.is_some());
}

#[test]
fn test_all_blocks_are_same_type() {
    // All constructors return BlockRef — can be stored in one Vec
    let blocks: Vec<BlockRef> = vec![
        integrator(0.0),
        amplifier(2.0),
        adder(None),
        multiplier(),
        constant(5.0),
        source(|t| t),
        scope(None, 0.0, Vec::new()),
        function(|u: &[f64]| u.to_vec()),
        ode(|x: &[f64], _u: &[f64], _t: f64| vec![-x[0]], &[1.0], None),
        pt1(1.0, 1.0),
        sin_block(),
        greater_than(),
    ];

    assert_eq!(blocks.len(), 12);

    // All can be passed to Simulation
    let mut sim = Simulation::with_defaults(blocks, Vec::new());
    sim.run(0.01, false, false);
}
