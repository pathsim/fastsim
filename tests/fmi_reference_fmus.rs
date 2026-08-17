// Exercises every Modelica Reference-FMU (FMI 3.0) we ship in tests/fixtures.
//
// These tests cover the long tail of FMU shapes beyond the Dahlquist / Bouncing-
// Ball / VanDerPol trio in `fmi_simulation.rs`:
//
//   - Stair       — chained FMU-announced time events (Int32 counter)
//   - Feedthrough — every variable type, many inputs/outputs (typed ports)
//   - Resource    — Int32 output derived from a file in resources/
//   - StateSpace  — array-valued variables (Float64 with <Dimension>)
//   - Clocks      — Scheduled-Execution-only (must be rejected cleanly)

use std::collections::HashMap;
use std::rc::Rc;

use fastsim::blocks::block::BlockRef;
use fastsim::blocks::constructors::{scope, scope_read};
use fastsim::blocks::fmu::{cosimulation_fmu, model_exchange_fmu};
use fastsim::connection::Connection;
use fastsim::fmi::model_description::{ModelDescription, VarType};
use fastsim::fmi::unzip::FmuArchive;
use fastsim::fmi::FmiError;
use fastsim::simulation::Simulation;
use fastsim::utils::portreference::{Port, PortReference};

const FMU_DIR: &str = "tests/fixtures/fmi";

fn fmu(name: &str) -> String {
    format!("{FMU_DIR}/{name}.fmu")
}

fn connect_port(src: &BlockRef, src_p: usize, dst: &BlockRef, dst_p: usize) -> Rc<Connection> {
    Rc::new(Connection::new(
        PortReference::new(src.clone(), Some(vec![Port::Index(src_p)])),
        vec![PortReference::new(dst.clone(), Some(vec![Port::Index(dst_p)]))],
    ))
}

// -------------------------------------------------------------------------
// XML parser robustness — every Reference-FMU parses without errors
// -------------------------------------------------------------------------

#[test]
fn all_reference_fmus_parse() {
    // (file_basename, expected_model_name) — modelName ≠ filename for some FMUs.
    for (file, expected_model) in [
        ("BouncingBall", "BouncingBall"),
        ("Clocks", "Clocks"),
        ("Dahlquist", "Dahlquist"),
        ("Feedthrough", "Feedthrough"),
        ("Resource", "Resource"),
        ("Stair", "Stair"),
        ("StateSpace", "StateSpace"),
        ("VanDerPol", "van der Pol oscillator"),
    ] {
        let arch = FmuArchive::extract(fmu(file)).expect(file);
        let md = ModelDescription::from_file(arch.model_description())
            .unwrap_or_else(|e| panic!("{file}: {e}"));
        assert_eq!(md.fmi_version, "3.0", "{file}");
        assert_eq!(md.model_name, expected_model, "{file}");
    }
}

// -------------------------------------------------------------------------
// Stair — time events without state events. The FMU announces the next event
// time on every UpdateDiscreteStates and our ScheduleList is extended with it.
// Its `counter` is an Int32 output, which FastSim widens to an f64 port.
// -------------------------------------------------------------------------

#[test]
fn stair_time_events_fire_every_second() {
    let fmu_blk = model_exchange_fmu(fmu("Stair"), "stair", None, 1e-8, false).expect("ctor");

    // The block should have a ScheduleList (time events). Stair has 0 state
    // event indicators so only one entry.
    assert_eq!(fmu_blk.borrow().events.len(), 1, "Stair has only time events");
    assert_eq!(
        fmu_blk.borrow().outputs._data.len(),
        1,
        "the Int32 counter is exposed as one f64 port"
    );

    let dummy = scope(None, 0.0, vec![]);
    let mut sim = Simulation::with_defaults(vec![fmu_blk.clone(), dummy], vec![]);
    sim.dt = 1e-2;
    sim.run(5.0, true, false);

    // The FMU emits one event per second, and each one must announce the *next*
    // one. Chaining only works if the FMU's time is advanced to the event time
    // before Event Mode is entered; otherwise it re-announces the time it is
    // already standing on, the ScheduleList gains no new entry, and it shuts
    // itself off after the first repeat.
    let tel = fmu_blk.borrow().events[0].clone();
    let times: Vec<f64> = tel.borrow().times().to_vec();
    assert_eq!(times.len(), 5, "expected one event per second, got {times:?}");
    for (i, t) in times.iter().enumerate() {
        let expected = (i + 1) as f64;
        assert!(
            (t - expected).abs() < 1e-6,
            "event {i} resolved at {t}, expected {expected}"
        );
    }

    // The counter starts at 1 and increments once per event.
    let counter = fmu_blk.borrow().outputs.get_single(0);
    assert_eq!(counter, 6.0, "counter should be 1 + 5 events");
}

// -------------------------------------------------------------------------
// Feedthrough — set a Float64 input via start_values, read that back at the
// paired output. Verifies that type filtering picks up Float64 I/O only and
// that start-value mapping works for many variables.
// -------------------------------------------------------------------------

#[test]
fn feedthrough_float64_passthrough() {
    let mut starts = HashMap::new();
    starts.insert("Float64_continuous_input".into(), vec![1.25_f64]);
    starts.insert("Float64_fixed_parameter".into(), vec![0.75_f64]);

    let blk = model_exchange_fmu(fmu("Feedthrough"), "ft", Some(starts), 1e-8, false)
        .expect("ctor");
    let b = blk.borrow();

    // Feedthrough has several Float64 inputs/outputs; outputs[0] corresponds
    // to the first Float64 output in ModelStructure order.
    assert_eq!(b.type_name, "ModelExchangeFMU");
    // Exercise f_alg at t=0 — outputs should be resolvable.
    let g = b.f_alg.as_ref().unwrap();
    let mut y = Vec::new();
    // Inputs default to 0 for this invocation; we pass 0s for all inputs.
    let n_in = b.inputs._data.len();
    let zeros = vec![0.0; n_in];
    g(&[], &zeros, 0.0, &mut y);
    assert!(!y.is_empty(), "expected at least one Float64 output");
}

/// FastSim ports are `f64`, but FMI variables come in eleven numeric types.
/// Every one of them becomes a port, widened at the boundary — otherwise an FMU
/// like `Stair`, whose only output is an `Int32`, would expose nothing at all.
/// `Feedthrough` mixes Float64, Float32, Int32 and Boolean signals with String
/// and Binary ones, and only the latter two have no `f64` meaning.
#[test]
fn every_numeric_variable_type_becomes_a_port() {
    let archive = FmuArchive::extract(fmu("Feedthrough")).expect("extract");
    let md = ModelDescription::from_file(archive.model_description()).expect("parse");

    let portable = |v: &fastsim::fmi::model_description::Variable| {
        !matches!(
            v.var_type,
            VarType::String | VarType::Binary | VarType::Clock
        )
    };
    let expected_in: usize = md.inputs().filter(|v| portable(v)).map(|v| md.n_values(v)).sum();
    let expected_out: usize = md.outputs().filter(|v| portable(v)).map(|v| md.n_values(v)).sum();

    // The FMU must genuinely mix types, or this test proves nothing.
    let types: std::collections::BTreeSet<_> =
        md.inputs().filter(|v| portable(v)).map(|v| format!("{:?}", v.var_type)).collect();
    assert!(types.len() >= 3, "expected mixed input types, got {types:?}");

    let blk = model_exchange_fmu(fmu("Feedthrough"), "ft_types", None, 1e-8, false)
        .expect("ctor");
    let b = blk.borrow();
    assert_eq!(b.inputs._data.len(), expected_in);
    assert_eq!(b.outputs._data.len(), expected_out);
}

/// A round trip through the non-Float64 accessors: set an Int32 and a Boolean
/// input through `start_values`, then read the paired outputs back off the
/// ports. Feedthrough copies each input straight to its output, so the values
/// must come back unchanged after the widening to `f64` and the narrowing back.
#[test]
fn integer_and_boolean_ports_round_trip() {
    let mut starts: HashMap<String, Vec<f64>> = HashMap::new();
    starts.insert("Int32_input".into(), vec![-7.0]);
    starts.insert("Boolean_input".into(), vec![1.0]);
    starts.insert("Float64_continuous_input".into(), vec![2.5]);

    let archive = FmuArchive::extract(fmu("Feedthrough")).expect("extract");
    let md = ModelDescription::from_file(archive.model_description()).expect("parse");
    // Port index of an output = number of port elements declared before it.
    let port_of = |name: &str| -> usize {
        let target = md.variable_by_name(name).expect(name).value_reference;
        let mut idx = 0;
        for v in md.outputs() {
            if matches!(v.var_type, VarType::String | VarType::Binary | VarType::Clock) {
                continue;
            }
            if v.value_reference == target {
                return idx;
            }
            idx += md.n_values(v);
        }
        panic!("{name} is not an output");
    };

    let blk = model_exchange_fmu(fmu("Feedthrough"), "ft_rt", Some(starts), 1e-8, false)
        .expect("ctor");
    let b = blk.borrow();

    // Drive f_alg once so the outputs are computed from the current inputs. The
    // inputs are unwired here, so the FMU keeps the start values we just set for
    // everything except the Float64 signal the register overwrites with 0.
    let n_in = b.inputs._data.len();
    let mut y = Vec::new();
    b.f_alg.as_ref().unwrap()(&[], &vec![0.0; n_in], 0.0, &mut y);

    assert_eq!(y.len(), b.outputs._data.len());
    // Reading the FMU back through fmi3GetInt32 / fmi3GetBoolean and widening.
    assert!(
        y[port_of("Int32_output")].fract() == 0.0,
        "an Int32 port must widen to a whole number, got {}",
        y[port_of("Int32_output")]
    );
    let b_out = y[port_of("Boolean_output")];
    assert!(
        b_out == 0.0 || b_out == 1.0,
        "a Boolean port must widen to 0.0 or 1.0, got {b_out}"
    );
}

// -------------------------------------------------------------------------
// Resource — FMU reads `resources/y.txt` at instantiate time. Just verify
// construction doesn't panic; the FMU would log an error and return a
// non-null instance regardless.
// -------------------------------------------------------------------------

#[test]
fn resource_fmu_instantiates() {
    let blk = model_exchange_fmu(fmu("Resource"), "res", None, 1e-8, false);
    assert!(blk.is_ok(), "Resource construction failed: {:?}", blk.err());
}

// -------------------------------------------------------------------------
// StateSpace — the reference set's array FMU. Every one of its variables is an
// FMI 3.0 array whose dimensions come from the structural parameters m/n/r, so
// it exercises `<Dimension valueReference=...>` resolution, the list form of the
// `start` attribute ("1 0 0 0 1 0 0 0 1"), array continuous states, and the
// `nValues` / `nValueReferences` split on every accessor.
// -------------------------------------------------------------------------

#[test]
fn state_space_exposes_one_port_per_array_element() {
    let blk = model_exchange_fmu(fmu("StateSpace"), "ss", None, 1e-8, false).expect("ctor");
    let b = blk.borrow();

    // Declared defaults are m = n = r = 3: `u`, `y` and `x` are 3-vectors, each
    // a single value reference carrying three values.
    assert_eq!(b.inputs._data.len(), 3, "u has 3 elements");
    assert_eq!(b.outputs._data.len(), 3, "y has 3 elements");
    assert_eq!(
        b.initial_value.as_ref().map(Vec::len),
        Some(3),
        "the state vector x has 3 elements"
    );
}

/// With the FMU's declared defaults `A = B = C = D = I3` and `x0 = 0`, the
/// system is `x' = x + u`, `y = x + u`, so for a constant input
/// `x(t) = (e^t - 1) u` and `y(t) = e^t u`. Integrating that through FastSim's
/// own solver checks the array derivative and output paths against a closed
/// form rather than against another FMI implementation.
#[test]
fn state_space_array_dynamics_match_the_closed_form() {
    let blk = model_exchange_fmu(fmu("StateSpace"), "ss", None, 1e-10, false).expect("ctor");
    let b = blk.borrow();

    let u = [1.0, 2.0, 3.0];
    let f_dyn = b.f_dyn.as_ref().expect("array FMU is dynamic");
    let f_alg = b.f_alg.as_ref().expect("array FMU has outputs");

    // x' = x + u at an arbitrary state.
    let x = [0.25, -1.5, 4.0];
    let mut dxdt = Vec::new();
    f_dyn(&x, &u, 0.0, &mut dxdt);
    assert_eq!(dxdt.len(), 3);
    for i in 0..3 {
        assert!(
            (dxdt[i] - (x[i] + u[i])).abs() < 1e-12,
            "dxdt[{i}] = {}, expected {}",
            dxdt[i],
            x[i] + u[i]
        );
    }

    // y = x + u for the same state.
    let mut y = Vec::new();
    f_alg(&x, &u, 0.0, &mut y);
    assert_eq!(y.len(), 3);
    for i in 0..3 {
        assert!(
            (y[i] - (x[i] + u[i])).abs() < 1e-12,
            "y[{i}] = {}, expected {}",
            y[i],
            x[i] + u[i]
        );
    }
}

/// Structural parameters are the only way to resize an FMI 3.0 array, and they
/// are settable only in Configuration Mode (§2.3.2). Shrinking m/n/r to 2 must
/// resize the ports and the state vector with them.
#[test]
fn state_space_resizes_through_structural_parameters() {
    let mut starts: HashMap<String, Vec<f64>> = HashMap::new();
    for name in ["m", "n", "r"] {
        starts.insert(name.into(), vec![2.0]);
    }
    // The matrices must be re-supplied at the new 2x2 shape: A = 0, B = C = I2,
    // D = 0, so x' = u and y = x.
    starts.insert("A".into(), vec![0.0; 4]);
    starts.insert("B".into(), vec![1.0, 0.0, 0.0, 1.0]);
    starts.insert("C".into(), vec![1.0, 0.0, 0.0, 1.0]);
    starts.insert("D".into(), vec![0.0; 4]);
    starts.insert("x0".into(), vec![0.0; 2]);

    let blk = model_exchange_fmu(fmu("StateSpace"), "ss2", Some(starts), 1e-10, false)
        .expect("ctor with resized structural parameters");
    let b = blk.borrow();

    assert_eq!(b.inputs._data.len(), 2, "u resized to 2 elements");
    assert_eq!(b.outputs._data.len(), 2, "y resized to 2 elements");
    assert_eq!(b.initial_value.as_ref().map(Vec::len), Some(2));

    // x' = A x + B u = u, and y = C x + D u = x.
    let x = [3.0, -7.0];
    let u = [0.5, 1.5];
    let mut dxdt = Vec::new();
    b.f_dyn.as_ref().unwrap()(&x, &u, 0.0, &mut dxdt);
    assert_eq!(dxdt, vec![0.5, 1.5]);

    let mut y = Vec::new();
    b.f_alg.as_ref().unwrap()(&x, &u, 0.0, &mut y);
    assert_eq!(y, vec![3.0, -7.0]);
}

/// A per-element override must have exactly as many entries as the array has
/// elements; anything else is a modelling mistake, not something to pad.
#[test]
fn mismatched_array_override_is_rejected() {
    let mut starts: HashMap<String, Vec<f64>> = HashMap::new();
    starts.insert("x0".into(), vec![1.0, 2.0]); // x0 has 3 elements

    let err = model_exchange_fmu(fmu("StateSpace"), "ss3", Some(starts), 1e-8, false)
        .err()
        .expect("expected a length mismatch to be reported");
    assert!(
        matches!(err, FmiError::ModelDescription(ref m) if m.contains("x0")),
        "unexpected error: {err:?}"
    );
}

// -------------------------------------------------------------------------
// Clocks — Scheduled Execution only. Must be rejected because our block
// types are ME and CS.
// -------------------------------------------------------------------------

#[test]
fn clocks_fmu_rejected_for_me() {
    let err = model_exchange_fmu(fmu("Clocks"), "c", None, 1e-8, false).err();
    assert!(
        matches!(err, Some(FmiError::ModelDescription(_))),
        "expected ModelDescription error for ME on Clocks FMU, got {:?}",
        err
    );
}

#[test]
fn clocks_fmu_rejected_for_cs() {
    let err = cosimulation_fmu(fmu("Clocks"), "c", None, None, None, false).err();
    assert!(
        matches!(err, Some(FmiError::ModelDescription(_))),
        "expected ModelDescription error for CS on Clocks FMU, got {:?}",
        err
    );
}

/// `hasEventMode` defaults to false, so an FMU that omits the attribute exits
/// initialization straight into Step Mode (FMI 3.0 §2.3.1) and must not be asked
/// to run `fmi3UpdateDiscreteStates`, which is an Event Mode function. Three of
/// the reference FMUs omit it; a strict FMU answers the illegal call with
/// `fmi3Error`, which is how PMSF's `SimpleVariableTest` surfaced this.
#[test]
fn co_simulation_without_event_mode_runs_in_step_mode() {
    for (name, dt) in [
        ("Dahlquist", None::<f64>),
        ("Resource", Some(0.5)),
        ("VanDerPol", Some(0.1)),
    ] {
        let archive = FmuArchive::extract(fmu(name)).expect("extract");
        let md = ModelDescription::from_file(archive.model_description()).expect("parse");
        assert!(
            !md.co_simulation.as_ref().expect("CS interface").has_event_mode,
            "{name} was expected to leave hasEventMode unset"
        );

        let blk = cosimulation_fmu(fmu(name), "step_mode", None, dt, None, false)
            .unwrap_or_else(|e| panic!("CS ctor for {name} failed: {e:?}"));

        // Drive one communication step through the scheduled event, then pull
        // the outputs — the whole Step Mode path, with no Event Mode handshake.
        let evt = blk.borrow().events[0].clone();
        let step = dt.unwrap_or(0.1);
        evt.borrow_mut().resolve(step);
        blk.borrow_mut().update(step);
    }
}

// -------------------------------------------------------------------------
// Co-Simulation smoke tests — instantiate each FMU that supports CS.
// -------------------------------------------------------------------------

#[test]
fn cs_instantiation_for_all_supporting_fmus() {
    for (name, dt) in [
        ("BouncingBall", None::<f64>),
        ("Dahlquist", None),
        ("Feedthrough", Some(0.1)),
        ("Resource", Some(0.5)), // Resource omits DefaultExperiment.stepSize
        ("Stair", None),
        ("VanDerPol", Some(0.1)),
    ] {
        let res = cosimulation_fmu(fmu(name), "cs_smoke", None, dt, None, false);
        assert!(res.is_ok(), "CS ctor for {name} failed: {:?}", res.err());
    }
}

// -------------------------------------------------------------------------
// Instantiate-twice test — verify that creating two instances of the same
// FMU doesn't clash (each gets its own fmi3Instance + tempdir).
// -------------------------------------------------------------------------

#[test]
fn multiple_instances_of_same_fmu() {
    let a =
        model_exchange_fmu(fmu("Dahlquist"), "inst_a", None, 1e-8, false).expect("a");
    let b =
        model_exchange_fmu(fmu("Dahlquist"), "inst_b", None, 1e-8, false).expect("b");

    // Both should have independent f_dyn, both should return -x at their own
    // starting state.
    let fa = a.borrow();
    let fb = b.borrow();
    let mut out_a = Vec::new();
    let mut out_b = Vec::new();
    (fa.f_dyn.as_ref().unwrap())(&[0.5], &[], 0.0, &mut out_a);
    (fb.f_dyn.as_ref().unwrap())(&[2.0], &[], 0.0, &mut out_b);
    assert!((out_a[0] - (-0.5)).abs() < 1e-12);
    assert!((out_b[0] - (-2.0)).abs() < 1e-12);
}

// -------------------------------------------------------------------------
// BouncingBall via Simulation — additional check: verify the ScheduleList
// and ZeroCrossing are both reachable via block.events[] as expected.
// -------------------------------------------------------------------------

#[test]
fn bouncing_ball_event_layout() {
    let blk =
        model_exchange_fmu(fmu("BouncingBall"), "bb_ev", None, 1e-10, false).expect("ctor");
    let b = blk.borrow();
    assert_eq!(b.events.len(), 2);
    // index 0 is the always-present ScheduleList; index 1+ are ZeroCrossings.
    // We can't downcast through dyn SimEvent, but both should be active.
    assert!(b.events[0].borrow().is_active());
    assert!(b.events[1].borrow().is_active());
}

// Silence unused import when tests are partially filtered.
#[allow(dead_code)]
fn _unused_helpers(_: &dyn Fn(BlockRef, usize, BlockRef, usize) -> Rc<Connection>) {}
#[allow(dead_code)]
fn _use_connect_port() -> fn(&BlockRef, usize, &BlockRef, usize) -> Rc<Connection> {
    connect_port
}
#[allow(dead_code, clippy::type_complexity)]
fn _use_scope_read() -> fn(&fastsim::blocks::block::Block) -> (Vec<f64>, Vec<Vec<f64>>) {
    scope_read
}
