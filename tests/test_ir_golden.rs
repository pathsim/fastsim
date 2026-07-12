// IR golden / versioning test.
//
// Pins the exact serialized JSON of representative `Module`s so any schema
// change is caught deliberately. On an intended change, bump `IR_VERSION` in
// `src/ir/schema.rs` and refresh the goldens:
//
//     UPDATE_IR_GOLDEN=1 cargo test --test test_ir_golden
//
// The committed goldens double as human-readable documentation of the wire
// format (see src/ir/README.md).

use std::rc::Rc;

use fastsim::blocks::constructors::*;
use fastsim::connection::Connection;
use fastsim::events::schedule::ScheduleList;
use fastsim::ir::builder::module_from_sim;
use fastsim::ir::schema::{Module, IR_VERSION};
use fastsim::simulation::Simulation;
use fastsim::subsystem::{interface, subsystem};
use fastsim::utils::fastcell::FastCell;
use fastsim::utils::portreference::PortReference;

fn conn(a: &fastsim::blocks::block::BlockRef, b: &fastsim::blocks::block::BlockRef) -> Rc<Connection> {
    Rc::new(Connection::new(
        PortReference::new(a.clone(), None),
        vec![PortReference::new(b.clone(), None)],
    ))
}

/// Compare a freshly built module's pretty JSON against the committed golden.
/// EOL is normalized so the check is stable across git's LF/CRLF conversion.
fn check_golden(name: &str, m: &Module) {
    assert_eq!(m.ir_version, IR_VERSION, "module ir_version must match IR_VERSION");
    let actual = serde_json::to_string_pretty(m).expect("serialize");

    // re-serializing a parsed copy must be byte-identical (lossless roundtrip).
    let parsed: Module = serde_json::from_str(&actual).expect("deserialize");
    assert_eq!(actual, serde_json::to_string_pretty(&parsed).unwrap(), "roundtrip not stable for {name}");

    let path = format!("{}/tests/golden/{}.json", env!("CARGO_MANIFEST_DIR"), name);
    if std::env::var("UPDATE_IR_GOLDEN").is_ok() {
        std::fs::create_dir_all(format!("{}/tests/golden", env!("CARGO_MANIFEST_DIR"))).unwrap();
        std::fs::write(&path, &actual).unwrap();
        eprintln!("updated golden: {path}");
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!("missing golden {path}; regenerate with UPDATE_IR_GOLDEN=1 cargo test --test test_ir_golden")
    });
    assert_eq!(
        actual.replace("\r\n", "\n"),
        expected.replace("\r\n", "\n"),
        "IR golden drift for '{name}'. If intended, bump IR_VERSION and run \
         UPDATE_IR_GOLDEN=1 cargo test --test test_ir_golden"
    );
}

/// Flat, feature-rich model: every block role + ops/params/state/memory, an
/// op-bearing event, an opaque block with an opaque event, and a global event.
#[test]
fn golden_full() {
    let src = sinusoidal_source(2.0, 1.5, 0.3); // Source: ops + params
    let amp = amplifier(2.5); // Algebraic: param op-graph
    let integ = integrator(0.0); // Dynamic: state + alg/dyn regions
    let sh = sample_hold(0.1, 0.0); // discrete: memory + op-bearing Schedule event
    let sco = scope(Some(0.25), 0.1, vec![]); // opaque extern + opaque Schedule event

    let mut sim = Simulation::with_defaults(
        vec![src.clone(), amp.clone(), integ.clone(), sh.clone(), sco.clone()],
        vec![conn(&src, &amp), conn(&amp, &integ), conn(&integ, &sh), conn(&sh, &sco)],
    );
    // a standalone simulation-level event (opaque host action)
    sim.add_event(Rc::new(FastCell::new(ScheduleList::from_times(vec![0.2, 0.5]))));
    sim.run(0.6, false, false);

    check_golden("full", &module_from_sim(&sim, "full"));
}

/// Nested subsystem: locks the recursive Subsystem / Interface shape and the
/// `BlockId::INTERFACE`-referencing connections.
#[test]
fn golden_subsystem() {
    let iface = interface();
    let amp = amplifier(2.0);
    let sub = subsystem(
        vec![iface.clone(), amp.clone()],
        vec![conn(&iface, &amp), conn(&amp, &iface)],
        10,
    )
    .unwrap();
    let src = sinusoidal_source(1.0, 1.0, 0.0);
    let sco = scope(None, 0.0, vec![]);
    let mut sim = Simulation::with_defaults(
        vec![src.clone(), sub.clone(), sco.clone()],
        vec![conn(&src, &sub), conn(&sub, &sco)],
    );
    sim.run(0.03, false, false);

    check_golden("subsystem", &module_from_sim(&sim, "subsystem"));
}

/// Purely algebraic feedback loop: locks the `Schedule.sccs` / `back_edges`
/// shape (err and kp form one SCC with a cut connection).
#[test]
fn golden_algebraic_loop() {
    let src = sinusoidal_source(1.0, 1.0, 0.0);
    let err = adder(Some("+-"));
    let kp = amplifier(0.5);
    let sco = scope(None, 0.0, vec![]);
    let mut sim = Simulation::with_defaults(
        vec![src.clone(), err.clone(), kp.clone(), sco.clone()],
        vec![conn(&src, &err), conn(&err, &kp), conn(&kp, &err), conn(&kp, &sco)],
    );
    sim.run(0.03, false, false);

    check_golden("loop", &module_from_sim(&sim, "loop"));
}

/// Parity / regression guard for the unified port-granular schedule: a model
/// with a feedback into an algebraically-*unread* input (here a MIMO StateSpace
/// `y = x + 1*u0`, `dx/dt = -x + u1`, with the output fed back into `u1`) is a
/// FALSE algebraic loop at block granularity. The runtime resolves it, and the
/// IR schedule (built by the same port-granular assembly) must too, so codegen
/// can statically order it. This used to fail with "cannot order the algebraic
/// pass"; it must now generate.
#[cfg(feature = "codegen")]
#[test]
fn codegen_orders_false_loop_through_unread_input() {
    use fastsim::blocks::block::BlockRef;
    use fastsim::codegen::{generate, CodegenOptions};
    use fastsim::utils::portreference::{Port, PortReference};

    let port_conn = |a: &BlockRef, ao: Option<usize>, b: &BlockRef, bi: Option<usize>| {
        Rc::new(Connection::new(
            PortReference::new(a.clone(), ao.map(|i| vec![Port::Index(i)])),
            vec![PortReference::new(b.clone(), bi.map(|i| vec![Port::Index(i)]))],
        ))
    };

    // ns=1, ni=2, no=1: y = x + 1*u0 (D=[1,0]); dx/dt = -x + u1.
    let ss = statespace(vec![-1.0], vec![0.0, 1.0], vec![1.0], vec![1.0, 0.0], 1, 2, 1, None);
    let src = constant(1.0);
    let amp = amplifier(0.5);
    let mut sim = Simulation::with_defaults(
        vec![src.clone(), ss.clone(), amp.clone()],
        vec![
            port_conn(&src, None, &ss, Some(0)),
            conn(&ss, &amp),
            port_conn(&amp, None, &ss, Some(1)), // feedback into the alg-unread input
        ],
    );
    sim.run(0.1, false, false); // runtime resolves the (false) loop

    let module = module_from_sim(&sim, "mimo");
    let files = generate(&module, &CodegenOptions::default())
        .expect("a false loop through an unread input must order and codegen");
    assert!(files.iter().any(|f| f.name == "mimo.c"));
}
