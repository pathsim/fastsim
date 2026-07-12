// Regression tests for algebraic-loop port resolution.
//
// In an algebraic loop the iteration order in `Simulation::_loops` is
// `block.update()` -> `conn.update()`. Without eager port resolution
// during graph assembly the first iteration would see input registers
// still at their `Block::default_block()` size, and any block accessing
// inputs by position would misread (panic on out-of-bounds for a
// `function` closure indexing `u[1]`, or silently sum fewer terms in
// `adder` etc.).
//
// Mirrors pathsim PR #214.

use std::rc::Rc;

use fastsim::blocks::constructors::{adder, constant, function};
use fastsim::connection::Connection;
use fastsim::simulation::Simulation;
use fastsim::utils::portreference::{Port, PortReference};


#[test]
fn function_in_algebraic_loop_reaches_correct_fixed_point() {
    // c -> A[0]
    // B -> A[1]   (loop closer)
    // A -> B
    // A indexes u[1] directly. Without eager resize the register starts
    // at len=1 and the first loop iteration panics on out-of-bounds.
    let c = constant(1.0);
    let a = function(|u| vec![u[0] + 0.1 * u[1]]);
    let b = function(|u| vec![0.5 * u[0]]);

    let conn1 = Rc::new(Connection::new(
        PortReference::new(c.clone(), Some(vec![Port::Index(0)])),
        vec![PortReference::new(a.clone(), Some(vec![Port::Index(0)]))],
    ));
    let conn2 = Rc::new(Connection::new(
        PortReference::new(a.clone(), Some(vec![Port::Index(0)])),
        vec![PortReference::new(b.clone(), Some(vec![Port::Index(0)]))],
    ));
    let conn3 = Rc::new(Connection::new(
        PortReference::new(b.clone(), Some(vec![Port::Index(0)])),
        vec![PortReference::new(a.clone(), Some(vec![Port::Index(1)]))],
    ));

    let mut sim = Simulation::with_defaults(
        vec![c, a.clone(), b],
        vec![conn1, conn2, conn3],
    );

    sim.run(0.05, false, false);

    // Fixed point: A = 1 + 0.1 * 0.5 * A  ->  A = 1 / 0.95
    let a_out = a.borrow().outputs.get_single(0);
    assert!(
        (a_out - 1.0 / 0.95).abs() < 1e-6,
        "A out = {} (expected ~1.0526)", a_out
    );
}


#[test]
fn adder_in_algebraic_loop_converges_correctly() {
    // c1 -> Add[0]
    // B  -> Add[1]   (loop closer)
    // Add -> B
    // Without eager resize the first iteration's adder sees u = [c1]
    // only and yields a transient false value before later iterations
    // resize. Eager resolution removes that transient so the first
    // iteration already sees both inputs.
    let c1 = constant(2.0);
    let add = adder(Some("++"));
    let b = function(|u| vec![0.5 * u[0]]);

    let conn1 = Rc::new(Connection::new(
        PortReference::new(c1.clone(), Some(vec![Port::Index(0)])),
        vec![PortReference::new(add.clone(), Some(vec![Port::Index(0)]))],
    ));
    let conn2 = Rc::new(Connection::new(
        PortReference::new(add.clone(), Some(vec![Port::Index(0)])),
        vec![PortReference::new(b.clone(), Some(vec![Port::Index(0)]))],
    ));
    let conn3 = Rc::new(Connection::new(
        PortReference::new(b.clone(), Some(vec![Port::Index(0)])),
        vec![PortReference::new(add.clone(), Some(vec![Port::Index(1)]))],
    ));

    let mut sim = Simulation::with_defaults(
        vec![c1, add.clone(), b],
        vec![conn1, conn2, conn3],
    );

    sim.run(0.05, false, false);

    // Fixed point: y = 2 + 0.5 * y  ->  y = 4
    let y = add.borrow().outputs.get_single(0);
    assert!(
        (y - 4.0).abs() < 1e-6,
        "adder out = {} (expected 4.0)", y
    );
}


#[test]
fn assemble_resizes_all_input_registers() {
    // After Simulation::with_defaults all registers must reach their
    // final size during _assemble_graph, before run() is called.
    let c = constant(1.0);
    let a = function(|u| vec![u[0] + u[1]]);
    let b = function(|u| vec![u[0]]);

    let conn1 = Rc::new(Connection::new(
        PortReference::new(c.clone(), Some(vec![Port::Index(0)])),
        vec![PortReference::new(a.clone(), Some(vec![Port::Index(0)]))],
    ));
    let conn2 = Rc::new(Connection::new(
        PortReference::new(a.clone(), Some(vec![Port::Index(0)])),
        vec![PortReference::new(b.clone(), Some(vec![Port::Index(0)]))],
    ));
    let conn3 = Rc::new(Connection::new(
        PortReference::new(b.clone(), Some(vec![Port::Index(0)])),
        vec![PortReference::new(a.clone(), Some(vec![Port::Index(1)]))],
    ));

    let _sim = Simulation::with_defaults(
        vec![c.clone(), a.clone(), b.clone()],
        vec![conn1, conn2, conn3],
    );

    assert!(a.borrow().inputs.len() >= 2);
    assert!(!b.borrow().inputs.is_empty());
    assert!(!c.borrow().outputs.is_empty());
    assert!(!a.borrow().outputs.is_empty());
    assert!(!b.borrow().outputs.is_empty());
}
