//! FMU export verification (FMI 3.0, Model Exchange, source FMU).
//!
//! Three layers, mirroring the codegen verification philosophy:
//!  1. structure — the exported file set and `modelDescription.xml` are present
//!     and parse back through the importer's own parser (always runs);
//!  2. packaging — the `.fmu` zip extracts via the importer's `FmuArchive`
//!     (always runs);
//!  3. behaviour — the generated FMI C compiles, and driving the real
//!     `fmi3*` entry points through a Model-Exchange RK4 host loop reproduces
//!     the native fastsim trajectory (skips when no C compiler is available).
#![cfg(all(feature = "codegen", feature = "fmi"))]

use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use fastsim::fmi::export::{export_fmu_bytes, fmu_files, ExportOptions};
use fastsim::fmi::model_description::{Causality, ModelDescription};
use fastsim::fmi::unzip::FmuArchive;
use fastsim::ir::eval::{eval_region, EvalCtx};
use fastsim::ir::schema::{
    BinOpKind, BlockId, Block, BlockRole, Child, Connection, ConnectionId, Direction, Event,
    EventId, EventKind, Interface, MemorySlot, MemorySlotId, Module, NodeId, Op, Param, ParamId,
    ParamValue, Port, PortRef, Ports, Region, Regions, Schedule, ScheduleTimes, StateId, StateVar,
    Subsystem, SubsystemId, UnaryOpKind, Write,
};

mod common;
use common::find_cc;

static UNIQ: AtomicU64 = AtomicU64::new(0);

/// A single self-contained dynamic block: dx/dt = -x, x(0) = 1, y = x. Closed
/// (no inputs), continuous, no events — the phase-1 export target.
fn decay_module() -> Module {
    let block = Block {
        id: BlockId(0),
        name: "decay".into(),
        type_name: "ODE".into(),
        role: BlockRole::Dynamic,
        ports: Ports { inputs: vec![], outputs: vec![Port { name: "out".into(), size: 1 }] },
        params: vec![],
        state: vec![StateVar { id: StateId(0), name: "x0".into(), init: 1.0 }],
        memory: vec![],
        regions: Regions {
            alg: Region {
                ops: vec![Op::State { id: StateId(0) }],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
            },
            dyn_: Region {
                ops: vec![
                    Op::State { id: StateId(0) },
                    Op::Unary { op: UnaryOpKind::Neg, a: NodeId(0) },
                ],
                writes: vec![Write::StateDeriv { id: StateId(0), src: NodeId(1) }],
            },
        },
        events: vec![],
    };
    let root = Subsystem {
        id: SubsystemId(0),
        name: "root".into(),
        interface: Interface::default(),
        children: vec![Child::Block(block)],
        connections: vec![],
        schedule: Schedule { topo: vec![BlockId(0)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    Module::new("decay", root)
}

/// A nonlinear closed model: dx/dt = -sin(x), x(0) = 0.5, y = x. Its state
/// Jacobian is ∂ẋ/∂x = -cos(x), so directional derivatives exercise the
/// forward-AD of a unary op (sin) exactly.
fn nonlinear_module() -> Module {
    let block = Block {
        id: BlockId(0),
        name: "nl".into(),
        type_name: "ODE".into(),
        role: BlockRole::Dynamic,
        ports: Ports { inputs: vec![], outputs: vec![Port { name: "out".into(), size: 1 }] },
        params: vec![],
        state: vec![StateVar { id: StateId(0), name: "x0".into(), init: 0.5 }],
        memory: vec![],
        regions: Regions {
            alg: Region {
                ops: vec![Op::State { id: StateId(0) }],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
            },
            dyn_: Region {
                ops: vec![
                    Op::State { id: StateId(0) },
                    Op::Unary { op: UnaryOpKind::Sin, a: NodeId(0) },
                    Op::Unary { op: UnaryOpKind::Neg, a: NodeId(1) },
                ],
                writes: vec![Write::StateDeriv { id: StateId(0), src: NodeId(2) }],
            },
        },
        events: vec![],
    };
    let root = Subsystem {
        id: SubsystemId(0),
        name: "root".into(),
        interface: Interface::default(),
        children: vec![Child::Block(block)],
        connections: vec![],
        schedule: Schedule { topo: vec![BlockId(0)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    Module::new("nl", root)
}

/// Reference RK4 over the IR `dyn` region through `fastsim::ir::eval` — the same
/// method (and `dt`) the FMI host loop below uses.
fn ir_rk4(dyn_region: &Region, x0: f64, t_end: f64, dt: f64) -> f64 {
    let f = |x: f64, t: f64| {
        let ctx = EvalCtx { inputs: &[], state: &[x], memory: &[], params: &[], t };
        eval_region(dyn_region, &ctx).unwrap()[0]
    };
    let (mut x, mut t) = (x0, 0.0);
    while t < t_end - 0.5 * dt {
        let k1 = f(x, t);
        let k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt);
        let k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt);
        let k4 = f(x + dt * k3, t + dt);
        x += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        t += dt;
    }
    x
}

#[test]
fn exports_expected_file_set_and_parsable_description() {
    let m = decay_module();
    let files = fmu_files(&m, &ExportOptions::default()).unwrap();
    let names: Vec<&str> = files.iter().map(|f| f.name.as_str()).collect();
    for want in [
        "modelDescription.xml",
        "sources/buildDescription.xml",
        "sources/fmi3.h",
        "sources/decay.h",
        "sources/decay.c",
        "sources/fmu.c",
    ] {
        assert!(names.contains(&want), "missing FMU entry {want}; got {names:?}");
    }

    // The modelDescription.xml parses back through the importer's parser.
    let md_src = &files.iter().find(|f| f.name == "modelDescription.xml").unwrap().contents;
    let md = ModelDescription::from_str(md_src).unwrap();
    assert_eq!(md.fmi_version, "3.0");
    assert_eq!(md.model_name, "decay");
    assert!(md.model_exchange.is_some());
    assert_eq!(md.model_exchange.as_ref().unwrap().model_identifier, "decay");
    assert_eq!(md.n_continuous_states(), 1);
    assert_eq!(md.n_event_indicators(), 0);

    // The continuous state has an exact start of 1.0; its derivative points back.
    let states = md.continuous_states();
    assert_eq!(states.len(), 1);
    assert_eq!(states[0].start.as_ref().and_then(|s| s.as_f64()), Some(1.0));
    let ders: Vec<_> = md.continuous_state_derivatives().collect();
    assert_eq!(ders.len(), 1);
    assert_eq!(ders[0].derivative_of, Some(states[0].value_reference));

    // The block output is observable.
    assert!(md.outputs().any(|v| v.causality == Causality::Output));

    // The build description names the model identifier and the one TU to build.
    let bd = &files.iter().find(|f| f.name == "sources/buildDescription.xml").unwrap().contents;
    assert!(bd.contains("modelIdentifier=\"decay\""), "{bd}");
    assert!(bd.contains("fmu.c"), "{bd}");

    // The wrapper includes the generated model TU (named after the model) and
    // checks the token.
    let fmu_c = &files.iter().find(|f| f.name == "sources/fmu.c").unwrap().contents;
    assert!(fmu_c.contains("#include \"decay.c\""), "{fmu_c}");
    assert!(fmu_c.contains(&format!("\"{}\"", md.instantiation_token)), "{fmu_c}");
}

#[test]
fn fmu_zip_extracts_through_importer_archive() {
    let m = decay_module();
    let bytes = export_fmu_bytes(&m, &ExportOptions::default()).unwrap();
    assert_eq!(&bytes[..2], b"PK", "not a zip");

    let uniq = UNIQ.fetch_add(1, Ordering::Relaxed);
    let fmu_path = std::env::temp_dir().join(format!("fastsim_decay_{uniq}.fmu"));
    std::fs::write(&fmu_path, &bytes).unwrap();

    // The importer's own unzip must accept the package and find a parsable
    // modelDescription.xml at the root.
    let archive = FmuArchive::extract(&fmu_path).unwrap();
    let md = ModelDescription::from_file(archive.model_description()).unwrap();
    assert_eq!(md.model_name, "decay");
    let _ = std::fs::remove_file(&fmu_path);
}

/// Drive the exported FMI ME C through a host RK4 loop and compare the final
/// state to the native IR reference (and the analytic `e^{-t}`).
#[test]
fn me_c_drives_to_native_trajectory() {
    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping FMU ME behaviour check");
        return;
    };

    let m = decay_module();
    let files = fmu_files(&m, &ExportOptions::default()).unwrap();
    let token = ModelDescription::from_str(
        &files.iter().find(|f| f.name == "modelDescription.xml").unwrap().contents,
    )
    .unwrap()
    .instantiation_token;

    // Write the FMU sources flat into a temp dir (they reference each other by
    // bare name: fmu.c -> "decay.c" -> "decay.h", fmu.c -> "fmi3.h").
    let uniq = UNIQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("fastsim_fmu_me_{uniq}"));
    std::fs::create_dir_all(&dir).unwrap();
    for f in &files {
        if let Some(base) = f.name.strip_prefix("sources/") {
            std::fs::write(dir.join(base), &f.contents).unwrap();
        }
    }

    // A Model-Exchange host: instantiate, initialise, then fixed-step RK4 calling
    // the real fmi3 entry points. Prints final state x[0] and the same value read
    // back through fmi3GetFloat64(valueReference=0).
    let main_c = format!(
        r#"#include <stdio.h>
#include "fmi3.h"
#define T_END 5.0
#define DT 0.01
int main(void) {{
    fmi3Instance m = fmi3InstantiateModelExchange("decay", "{token}", "", fmi3False, fmi3False, NULL, NULL);
    if (!m) {{ fprintf(stderr, "instantiate failed\n"); return 1; }}
    if (fmi3EnterInitializationMode(m, fmi3False, 0.0, 0.0, fmi3False, 0.0) != fmi3OK) return 2;
    if (fmi3ExitInitializationMode(m) != fmi3OK) return 3;
    if (fmi3EnterContinuousTimeMode(m) != fmi3OK) return 4;
    size_t ns = 0;
    fmi3GetNumberOfContinuousStates(m, &ns);
    double x[8], k1[8], k2[8], k3[8], k4[8], xt[8];
    fmi3GetContinuousStates(m, x, ns);
    double t = 0.0;
    while (t < T_END - 0.5 * DT) {{
        fmi3SetTime(m, t); fmi3SetContinuousStates(m, x, ns);
        fmi3GetContinuousStateDerivatives(m, k1, ns);
        for (size_t i = 0; i < ns; i++) xt[i] = x[i] + 0.5 * DT * k1[i];
        fmi3SetTime(m, t + 0.5 * DT); fmi3SetContinuousStates(m, xt, ns);
        fmi3GetContinuousStateDerivatives(m, k2, ns);
        for (size_t i = 0; i < ns; i++) xt[i] = x[i] + 0.5 * DT * k2[i];
        fmi3SetContinuousStates(m, xt, ns);
        fmi3GetContinuousStateDerivatives(m, k3, ns);
        for (size_t i = 0; i < ns; i++) xt[i] = x[i] + DT * k3[i];
        fmi3SetTime(m, t + DT); fmi3SetContinuousStates(m, xt, ns);
        fmi3GetContinuousStateDerivatives(m, k4, ns);
        for (size_t i = 0; i < ns; i++) x[i] += DT / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        t += DT;
        fmi3CompletedIntegratorStep(m, fmi3True, NULL, NULL);
    }}
    fmi3SetTime(m, t); fmi3SetContinuousStates(m, x, ns);
    fmi3ValueReference vr0 = 0; double v0 = 0.0;
    fmi3GetFloat64(m, &vr0, 1, &v0, 1);
    printf("%.17g %.17g\n", x[0], v0);
    fmi3FreeInstance(m);
    return 0;
}}
"#
    );
    std::fs::write(dir.join("main.c"), main_c).unwrap();

    // Compile fmu.c (which #includes model.c) + main.c into one exe. model.c is
    // NOT a separate unit — it is included — so we must not list it.
    let exe = dir.join("me.exe");
    let out = fastsim::codegen::verify::cc_command_in(&cc, &dir)
        .args(["fmu.c", "main.c", "-O0", "-o", "me.exe", "-lm"])
        .output()
        .expect("spawn cc");
    assert!(
        out.status.success(),
        "FMU C did not compile:\n{}\n--- fmu.c ---\n{}",
        String::from_utf8_lossy(&out.stderr),
        files.iter().find(|f| f.name == "sources/fmu.c").unwrap().contents,
    );

    let run = match Command::new(&exe).output() {
        Ok(r) if r.status.success() => r,
        _ => {
            eprintln!("exe would not launch — skipping numeric comparison");
            return;
        }
    };
    let text = String::from_utf8_lossy(&run.stdout);
    let got: Vec<f64> = text.split_whitespace().filter_map(|t| t.parse().ok()).collect();
    assert_eq!(got.len(), 2, "unexpected exe output: {text:?}");
    let (x_final, v0) = (got[0], got[1]);

    // The reference: the same RK4 over the IR, and the analytic decay.
    let dyn_region = match &decay_module().root.children[0] {
        Child::Block(b) => b.regions.dyn_.clone(),
        _ => unreachable!(),
    };
    let reference = ir_rk4(&dyn_region, 1.0, 5.0, 0.01);
    assert!(
        (x_final - reference).abs() < 1e-9,
        "FMU ME trajectory {x_final} != IR RK4 reference {reference}"
    );
    assert!(
        (x_final - (-5.0f64).exp()).abs() < 1e-6,
        "FMU ME trajectory {x_final} != analytic e^-5 {}",
        (-5.0f64).exp()
    );
    // GetFloat64(vr=0) reads the state signal — same value as x[0].
    assert!((v0 - x_final).abs() < 1e-12, "get_signal(0)={v0} != state {x_final}");
}

/// Export a single-state model, then compile a host that reads the exact
/// directional derivative ∂ẋ/∂x·1 (analytic JVP) and a central finite difference
/// of the same via `fmi3GetContinuousStateDerivatives`. Returns `(sens, fd, x)`
/// or `None` (no C compiler). Asserts the FMU advertises + compiles.
fn jvp_state_jacobian_vs_fd(m: &Module, tag: &str) -> Option<(f64, f64, f64)> {
    let files = fmu_files(m, &ExportOptions::default()).unwrap();
    let md = ModelDescription::from_str(
        &files.iter().find(|f| f.name == "modelDescription.xml").unwrap().contents,
    )
    .unwrap();
    assert!(
        md.model_exchange.as_ref().unwrap().provides_directional_derivatives,
        "ME info should advertise providesDirectionalDerivatives"
    );
    let state_vr = md.continuous_states()[0].value_reference;
    let der_vr = md.continuous_state_derivatives().next().unwrap().value_reference;
    let model_id = md.model_exchange.as_ref().unwrap().model_identifier.clone();
    let token = md.instantiation_token.clone();

    let cc = find_cc()?;
    let dir = write_sources(&files, tag);
    let main_c = format!(
        r#"#include <stdio.h>
#include "fmi3.h"
int main(void) {{
    fmi3Instance m = fmi3InstantiateModelExchange("{model_id}", "{token}", "", fmi3False, fmi3False, NULL, NULL);
    if (!m) return 1;
    if (fmi3EnterInitializationMode(m, fmi3False, 0.0, 0.0, fmi3False, 0.0) != fmi3OK) return 2;
    if (fmi3ExitInitializationMode(m) != fmi3OK) return 3;
    if (fmi3EnterContinuousTimeMode(m) != fmi3OK) return 4;
    fmi3ValueReference unk = {der_vr}u, kn = {state_vr}u;
    double seed = 1.0, sens = 0.0;
    if (fmi3GetDirectionalDerivative(m, &unk, 1, &kn, 1, &seed, 1, &sens, 1) != fmi3OK) return 5;
    size_t ns = 0; fmi3GetNumberOfContinuousStates(m, &ns);
    double x[1], fm[1], fp[1], h = 1e-6;
    fmi3GetContinuousStates(m, x, ns);
    double xm[1] = {{ x[0] - h }}, xp[1] = {{ x[0] + h }};
    fmi3SetContinuousStates(m, xm, ns); fmi3GetContinuousStateDerivatives(m, fm, ns);
    fmi3SetContinuousStates(m, xp, ns); fmi3GetContinuousStateDerivatives(m, fp, ns);
    double fd = (fp[0] - fm[0]) / (2.0 * h);
    printf("%.17g %.17g %.17g\n", sens, fd, x[0]);
    fmi3FreeInstance(m);
    return 0;
}}
"#
    );
    std::fs::write(dir.join("main.c"), main_c).unwrap();
    let out = fastsim::codegen::verify::cc_command_in(&cc, &dir)
        .args(["fmu.c", "main.c", "-O0", "-o", "dd.exe", "-lm"])
        .output()
        .expect("spawn cc");
    assert!(
        out.status.success(),
        "FMU C did not compile:\n{}\n--- fmu.c ---\n{}",
        String::from_utf8_lossy(&out.stderr),
        files.iter().find(|f| f.name == "sources/fmu.c").unwrap().contents,
    );
    let run = match Command::new(dir.join("dd.exe")).output() {
        Ok(r) if r.status.success() => r,
        _ => return None,
    };
    let got: Vec<f64> = String::from_utf8_lossy(&run.stdout)
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect();
    assert_eq!(got.len(), 3, "unexpected output");
    Some((got[0], got[1], got[2]))
}

/// A single-state model `dx/dt = <dyn ops>, y = x` for JVP tests.
fn single_state_module(name: &str, x0: f64, dyn_ops: Vec<Op>, dyn_src: NodeId) -> Module {
    let block = Block {
        id: BlockId(0),
        name: name.into(),
        type_name: "ODE".into(),
        role: BlockRole::Dynamic,
        ports: Ports { inputs: vec![], outputs: vec![Port { name: "out".into(), size: 1 }] },
        params: vec![],
        state: vec![StateVar { id: StateId(0), name: "x".into(), init: x0 }],
        memory: vec![],
        regions: Regions {
            alg: Region {
                ops: vec![Op::State { id: StateId(0) }],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
            },
            dyn_: Region { ops: dyn_ops, writes: vec![Write::StateDeriv { id: StateId(0), src: dyn_src }] },
        },
        events: vec![],
    };
    let root = Subsystem {
        id: SubsystemId(0),
        name: "root".into(),
        interface: Interface::default(),
        children: vec![Child::Block(block)],
        connections: vec![],
        schedule: Schedule { topo: vec![BlockId(0)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    Module::new(name, root)
}

#[test]
fn directional_derivative_is_analytic_and_matches_finite_difference() {
    match jvp_state_jacobian_vs_fd(&nonlinear_module(), "dd_sin") {
        None => eprintln!("no C compiler — skipping directional-derivative behaviour check"),
        Some((sens, fd, x)) => {
            let analytic = -(x.cos()); // ∂(-sin x)/∂x
            assert!((sens - analytic).abs() < 1e-12, "analytic JVP {sens} != -cos(x) {analytic}");
            assert!((sens - fd).abs() < 1e-6, "analytic JVP {sens} != finite diff {fd}");
        }
    }
}

#[test]
fn directional_derivative_fmod() {
    // dx/dt = fmod(x, 2); ∂/∂x = 1 a.e. (here x0 = 3.5).
    use fastsim::ir::schema::BinOpKind::Mod;
    let m = single_state_module(
        "fmod",
        3.5,
        vec![Op::State { id: StateId(0) }, Op::Const(2.0), Op::Binary { op: Mod, a: NodeId(0), b: NodeId(1) }],
        NodeId(2),
    );
    if let Some((sens, fd, _)) = jvp_state_jacobian_vs_fd(&m, "dd_fmod") {
        assert!((sens - fd).abs() < 1e-6, "fmod JVP {sens} != FD {fd}");
        assert!((sens - 1.0).abs() < 1e-9, "fmod ∂/∂x should be 1, got {sens}");
    }
}

#[test]
fn directional_derivative_min_reduction() {
    use fastsim::ir::schema::ReduceKind;
    // dx/dt = min(x, 2); at x0 = 1 the active operand is x, so ∂/∂x = 1.
    let m = single_state_module(
        "minr",
        1.0,
        vec![
            Op::State { id: StateId(0) },
            Op::Const(2.0),
            Op::Reduce { op: ReduceKind::Min, args: vec![NodeId(0), NodeId(1)] },
        ],
        NodeId(2),
    );
    if let Some((sens, fd, _)) = jvp_state_jacobian_vs_fd(&m, "dd_min") {
        assert!((sens - fd).abs() < 1e-6, "min-reduce JVP {sens} != FD {fd}");
        assert!((sens - 1.0).abs() < 1e-9, "min(x,2) ∂/∂x at x=1 should be 1, got {sens}");
    }
}

/// Run a batch of single directional-derivative queries `(known_vr, unknown_vr)`
/// (each with seed 1) against an exported FMU and return the sensitivities, or
/// `None` (no C compiler). Exercises arbitrary known/unknown combinations.
fn directional_queries(m: &Module, tag: &str, queries: &[(u32, u32)]) -> Option<Vec<f64>> {
    let files = fmu_files(m, &ExportOptions::default()).unwrap();
    let md = ModelDescription::from_str(
        &files.iter().find(|f| f.name == "modelDescription.xml").unwrap().contents,
    )
    .unwrap();
    let model_id = md.model_exchange.as_ref().unwrap().model_identifier.clone();
    let token = md.instantiation_token.clone();
    let cc = find_cc()?;
    let dir = write_sources(&files, tag);
    let kn = queries.iter().map(|(k, _)| format!("{k}u")).collect::<Vec<_>>().join(", ");
    let un = queries.iter().map(|(_, u)| format!("{u}u")).collect::<Vec<_>>().join(", ");
    let n = queries.len();
    let main_c = format!(
        r#"#include <stdio.h>
#include "fmi3.h"
int main(void) {{
    fmi3Instance m = fmi3InstantiateModelExchange("{model_id}", "{token}", "", fmi3False, fmi3False, NULL, NULL);
    if (!m) return 1;
    fmi3EnterInitializationMode(m, fmi3False, 0.0, 0.0, fmi3False, 0.0);
    fmi3ExitInitializationMode(m);
    fmi3EnterContinuousTimeMode(m);
    fmi3ValueReference kn[] = {{ {kn} }};
    fmi3ValueReference un[] = {{ {un} }};
    double seed = 1.0;
    for (size_t i = 0; i < {n}; i++) {{
        double sens = -123.0;
        if (fmi3GetDirectionalDerivative(m, &un[i], 1, &kn[i], 1, &seed, 1, &sens, 1) != fmi3OK) return 7;
        printf("%.17g ", sens);
    }}
    fmi3FreeInstance(m);
    return 0;
}}
"#
    );
    std::fs::write(dir.join("main.c"), main_c).unwrap();
    let out = fastsim::codegen::verify::cc_command_in(&cc, &dir)
        .args(["fmu.c", "main.c", "-O0", "-o", "dq.exe", "-lm"])
        .output()
        .expect("spawn cc");
    assert!(out.status.success(), "FMU C did not compile:\n{}", String::from_utf8_lossy(&out.stderr));
    let run = match Command::new(dir.join("dq.exe")).output() {
        Ok(r) if r.status.success() => r,
        _ => return None,
    };
    let v: Vec<f64> = String::from_utf8_lossy(&run.stdout)
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect();
    assert_eq!(v.len(), n, "expected {n} sensitivities");
    Some(v)
}

#[test]
fn directional_derivative_inputs_and_outputs() {
    // Open subsystem x' = u, y = x. Exercise the non-state-Jacobian axes:
    //   ∂ẋ/∂u = 1, ∂y/∂x = 1, ∂y/∂u = 0 (y reads the state, not the input).
    let m = subsystem_input_module();
    let files = fmu_files(&m, &ExportOptions::default()).unwrap();
    let md = ModelDescription::from_str(
        &files.iter().find(|f| f.name == "modelDescription.xml").unwrap().contents,
    )
    .unwrap();
    let x = md.continuous_states()[0].value_reference;
    let der = md.continuous_state_derivatives().next().unwrap().value_reference;
    let u = md.inputs().find(|v| v.name == "u").unwrap().value_reference;
    let y = md.outputs().next().unwrap().value_reference;

    if let Some(v) = directional_queries(&m, "dq_io", &[(u, der), (x, y), (u, y)]) {
        assert!((v[0] - 1.0).abs() < 1e-12, "∂ẋ/∂u = {} expected 1", v[0]);
        assert!((v[1] - 1.0).abs() < 1e-12, "∂y/∂x = {} expected 1", v[1]);
        assert!((v[2] - 0.0).abs() < 1e-12, "∂y/∂u = {} expected 0", v[2]);
    }
}

/// Single-state model `dx/dt = p` with a scalar parameter `p = 2`, `y = x`.
fn param_module() -> Module {
    let block = Block {
        id: BlockId(0),
        name: "pblk".into(),
        type_name: "ODE".into(),
        role: BlockRole::Dynamic,
        ports: Ports { inputs: vec![], outputs: vec![Port { name: "out".into(), size: 1 }] },
        params: vec![Param { id: ParamId(0), name: "p".into(), value: ParamValue::Scalar(2.0) }],
        state: vec![StateVar { id: StateId(0), name: "x".into(), init: 0.0 }],
        memory: vec![],
        regions: Regions {
            alg: Region {
                ops: vec![Op::State { id: StateId(0) }],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
            },
            dyn_: Region {
                ops: vec![Op::Param { id: ParamId(0) }],
                writes: vec![Write::StateDeriv { id: StateId(0), src: NodeId(0) }],
            },
        },
        events: vec![],
    };
    let root = Subsystem {
        id: SubsystemId(0),
        name: "root".into(),
        interface: Interface::default(),
        children: vec![Child::Block(block)],
        connections: vec![],
        schedule: Schedule { topo: vec![BlockId(0)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    Module::new("param", root)
}

#[test]
fn directional_derivative_parameter() {
    // dx/dt = p, so ∂ẋ/∂p = 1.
    let m = param_module();
    let files = fmu_files(&m, &ExportOptions::default()).unwrap();
    let md = ModelDescription::from_str(
        &files.iter().find(|f| f.name == "modelDescription.xml").unwrap().contents,
    )
    .unwrap();
    let p = md.variables.iter().find(|v| v.causality == Causality::Parameter).unwrap().value_reference;
    let der = md.continuous_state_derivatives().next().unwrap().value_reference;
    if let Some(v) = directional_queries(&m, "dq_param", &[(p, der)]) {
        assert!((v[0] - 1.0).abs() < 1e-12, "∂ẋ/∂p = {} expected 1", v[0]);
    }
}

#[test]
fn directional_derivative_lut1d() {
    // dx/dt = lut1d(x); on segment [1,2] of {0:0, 1:10, 2:40} the slope is 30.
    let m = single_state_module(
        "lut",
        1.5,
        vec![
            Op::State { id: StateId(0) },
            Op::Lut1d {
                input: NodeId(0),
                points: vec![0.0, 1.0, 2.0],
                values: vec![0.0, 10.0, 40.0],
                clamp: false,
            },
        ],
        NodeId(1),
    );
    if let Some((sens, fd, _)) = jvp_state_jacobian_vs_fd(&m, "dd_lut") {
        assert!((sens - fd).abs() < 1e-6, "lut1d JVP {sens} != FD {fd}");
        assert!((sens - 30.0).abs() < 1e-9, "lut1d slope on [1,2] should be 30, got {sens}");
    }
}

/// A zero-cross model with two states: a clock `x' = -1, x(0) = 1` (the guard
/// `z = x` falls through zero at t = 1) and a counter `c' = 0, c(0) = 0`. On the
/// falling crossing the event adds 1 to `c`. So after t = 1 the counter is 1.
fn zerocross_module() -> Module {
    let block = Block {
        id: BlockId(0),
        name: "zc".into(),
        type_name: "ZeroCross".into(),
        role: BlockRole::Dynamic,
        ports: Ports {
            inputs: vec![],
            outputs: vec![Port { name: "x".into(), size: 1 }, Port { name: "c".into(), size: 1 }],
        },
        params: vec![],
        state: vec![
            StateVar { id: StateId(0), name: "x".into(), init: 1.0 },
            StateVar { id: StateId(1), name: "c".into(), init: 0.0 },
        ],
        memory: vec![],
        regions: Regions {
            alg: Region {
                ops: vec![Op::State { id: StateId(0) }, Op::State { id: StateId(1) }],
                writes: vec![
                    Write::Output { port: 0, elem: 0, src: NodeId(0) },
                    Write::Output { port: 1, elem: 0, src: NodeId(1) },
                ],
            },
            dyn_: Region {
                ops: vec![Op::Const(-1.0), Op::Const(0.0)],
                writes: vec![
                    Write::StateDeriv { id: StateId(0), src: NodeId(0) },
                    Write::StateDeriv { id: StateId(1), src: NodeId(1) },
                ],
            },
        },
        events: vec![Event {
            id: EventId(0),
            kind: EventKind::ZeroCross {
                guard: Region { ops: vec![Op::State { id: StateId(0) }], writes: vec![] },
                direction: Direction::Falling,
            },
            effect: Region {
                ops: vec![
                    Op::State { id: StateId(1) },
                    Op::Const(1.0),
                    Op::Binary { op: BinOpKind::Add, a: NodeId(0), b: NodeId(1) },
                ],
                writes: vec![Write::StateWrite { id: StateId(1), src: NodeId(2) }],
            },
            opaque: false,
        }],
    };
    let root = Subsystem {
        id: SubsystemId(0),
        name: "root".into(),
        interface: Interface::default(),
        children: vec![Child::Block(block)],
        connections: vec![],
        schedule: Schedule { topo: vec![BlockId(0)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    Module::new("zc", root)
}

/// A periodic-event model: state `x' = 0, x(0) = 0`, an event every 0.5 (first at
/// 0.5) adds 1 to `x`. So `x` steps 1 -> 2 -> 3 at t = 0.5, 1.0, 1.5.
fn periodic_module() -> Module {
    let block = Block {
        id: BlockId(0),
        name: "per".into(),
        type_name: "Periodic".into(),
        role: BlockRole::Dynamic,
        ports: Ports { inputs: vec![], outputs: vec![Port { name: "x".into(), size: 1 }] },
        params: vec![],
        state: vec![StateVar { id: StateId(0), name: "x".into(), init: 0.0 }],
        memory: vec![],
        regions: Regions {
            alg: Region {
                ops: vec![Op::State { id: StateId(0) }],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
            },
            dyn_: Region {
                ops: vec![Op::Const(0.0)],
                writes: vec![Write::StateDeriv { id: StateId(0), src: NodeId(0) }],
            },
        },
        events: vec![Event {
            id: EventId(0),
            kind: EventKind::Schedule { times: ScheduleTimes::Periodic { period: 0.5, phase: 0.5 } },
            effect: Region {
                ops: vec![
                    Op::State { id: StateId(0) },
                    Op::Const(1.0),
                    Op::Binary { op: BinOpKind::Add, a: NodeId(0), b: NodeId(1) },
                ],
                writes: vec![Write::StateWrite { id: StateId(0), src: NodeId(2) }],
            },
            opaque: false,
        }],
    };
    let root = Subsystem {
        id: SubsystemId(0),
        name: "root".into(),
        interface: Interface::default(),
        children: vec![Child::Block(block)],
        connections: vec![],
        schedule: Schedule { topo: vec![BlockId(0)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    Module::new("per", root)
}

/// A fixed-schedule (`ScheduleList`) model: state `x' = 0, x(0) = 0`, events at
/// the *non-uniform* absolute times 0.3, 0.7, 1.2 each add 1 to `x`. So `x` steps
/// 1 -> 2 -> 3 at exactly those times. The uneven spacing distinguishes the Fixed
/// (time-list) path from the Periodic (constant-stride) path.
fn fixed_schedule_module() -> Module {
    let block = Block {
        id: BlockId(0),
        name: "fix".into(),
        type_name: "Fixed".into(),
        role: BlockRole::Dynamic,
        ports: Ports { inputs: vec![], outputs: vec![Port { name: "x".into(), size: 1 }] },
        params: vec![],
        state: vec![StateVar { id: StateId(0), name: "x".into(), init: 0.0 }],
        memory: vec![],
        regions: Regions {
            alg: Region {
                ops: vec![Op::State { id: StateId(0) }],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
            },
            dyn_: Region {
                ops: vec![Op::Const(0.0)],
                writes: vec![Write::StateDeriv { id: StateId(0), src: NodeId(0) }],
            },
        },
        events: vec![Event {
            id: EventId(0),
            kind: EventKind::Schedule { times: ScheduleTimes::Fixed(vec![0.3, 0.7, 1.2]) },
            effect: Region {
                ops: vec![
                    Op::State { id: StateId(0) },
                    Op::Const(1.0),
                    Op::Binary { op: BinOpKind::Add, a: NodeId(0), b: NodeId(1) },
                ],
                writes: vec![Write::StateWrite { id: StateId(0), src: NodeId(2) }],
            },
            opaque: false,
        }],
    };
    let root = Subsystem {
        id: SubsystemId(0),
        name: "root".into(),
        interface: Interface::default(),
        children: vec![Child::Block(block)],
        connections: vec![],
        schedule: Schedule { topo: vec![BlockId(0)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    Module::new("fix", root)
}

/// An open subsystem with an external input: `x' = u`, `x(0) = 0`, `y = x`. The
/// root interface input `u` feeds the integrator; the output port exposes `y`.
/// An importer drives `u` via fmi3SetFloat64.
fn subsystem_input_module() -> Module {
    let integ = Block {
        id: BlockId(0),
        name: "integ".into(),
        type_name: "Integrator".into(),
        role: BlockRole::Dynamic,
        ports: Ports {
            inputs: vec![Port { name: "in".into(), size: 1 }],
            outputs: vec![Port { name: "out".into(), size: 1 }],
        },
        params: vec![],
        state: vec![StateVar { id: StateId(0), name: "x".into(), init: 0.0 }],
        memory: vec![],
        regions: Regions {
            alg: Region {
                ops: vec![Op::State { id: StateId(0) }],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
            },
            dyn_: Region {
                // dx/dt = u (the external input)
                ops: vec![Op::Input { port: 0, elem: 0 }],
                writes: vec![Write::StateDeriv { id: StateId(0), src: NodeId(0) }],
            },
        },
        events: vec![],
    };
    let pr = |block: BlockId, port: u32| PortRef { block, port, elems: None };
    let root = Subsystem {
        id: SubsystemId(0),
        name: "sub".into(),
        interface: Interface {
            inputs: vec![Port { name: "u".into(), size: 1 }],
            outputs: vec![Port { name: "y".into(), size: 1 }],
        },
        children: vec![Child::Block(integ)],
        connections: vec![
            // interface input u -> integ.in
            Connection {
                id: ConnectionId(0),
                src: pr(BlockId::INTERFACE, 0),
                targets: vec![pr(BlockId(0), 0)],
            },
            // integ.out -> interface output y
            Connection {
                id: ConnectionId(1),
                src: pr(BlockId(0), 0),
                targets: vec![pr(BlockId::INTERFACE, 0)],
            },
        ],
        schedule: Schedule { topo: vec![BlockId(0)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    Module::new("sub", root)
}

/// An integrator block `x' = in, y = x` with one input and one output port.
fn integrator_block(id: u32, name: &str) -> Block {
    Block {
        id: BlockId(id),
        name: name.into(),
        type_name: "Integrator".into(),
        role: BlockRole::Dynamic,
        ports: Ports {
            inputs: vec![Port { name: "in".into(), size: 1 }],
            outputs: vec![Port { name: "out".into(), size: 1 }],
        },
        params: vec![],
        state: vec![StateVar { id: StateId(0), name: "x".into(), init: 0.0 }],
        memory: vec![],
        regions: Regions {
            alg: Region {
                ops: vec![Op::State { id: StateId(0) }],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
            },
            dyn_: Region {
                ops: vec![Op::Input { port: 0, elem: 0 }],
                writes: vec![Write::StateDeriv { id: StateId(0), src: NodeId(0) }],
            },
        },
        events: vec![],
    }
}

/// `x' = u` like [`subsystem_input_module`], but the root interface input feeds a
/// *nested* subsystem (root iface -> inner subsystem iface -> integrator),
/// exercising the recursive resolution of external inputs through nesting.
fn nested_subsystem_input_module() -> Module {
    let pr = |block: BlockId, port: u32| PortRef { block, port, elems: None };
    let iface_io = || Interface {
        inputs: vec![Port { name: "u".into(), size: 1 }],
        outputs: vec![Port { name: "y".into(), size: 1 }],
    };
    // Inner subsystem: interface u -> integ -> interface y.
    let inner = Subsystem {
        id: SubsystemId(1),
        name: "inner".into(),
        interface: iface_io(),
        children: vec![Child::Block(integrator_block(0, "integ"))],
        connections: vec![
            Connection { id: ConnectionId(0), src: pr(BlockId::INTERFACE, 0), targets: vec![pr(BlockId(0), 0)] },
            Connection { id: ConnectionId(1), src: pr(BlockId(0), 0), targets: vec![pr(BlockId::INTERFACE, 0)] },
        ],
        schedule: Schedule { topo: vec![BlockId(0)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    // Root: interface u -> inner subsystem -> interface y.
    let root = Subsystem {
        id: SubsystemId(0),
        name: "outer".into(),
        interface: iface_io(),
        children: vec![Child::Subsystem(inner)],
        connections: vec![
            Connection { id: ConnectionId(0), src: pr(BlockId::INTERFACE, 0), targets: vec![pr(BlockId(0), 0)] },
            Connection { id: ConnectionId(1), src: pr(BlockId(0), 0), targets: vec![pr(BlockId::INTERFACE, 0)] },
        ],
        schedule: Schedule { topo: vec![BlockId(0)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    Module::new("nested", root)
}

/// A subsystem with an *internal* block output that is not on the interface:
/// interface u -> integ (x'=u, y=x) -> interface y, plus an internal amplifier
/// reading the integrator (its output drives nothing on the interface). So the
/// integrator output is an FMI `output`, the amplifier output an FMI `local`.
fn subsystem_with_local_module() -> Module {
    let pr = |block: BlockId, port: u32| PortRef { block, port, elems: None };
    let amp = Block {
        id: BlockId(1),
        name: "amp".into(),
        type_name: "Amplifier".into(),
        role: BlockRole::Algebraic,
        ports: Ports {
            inputs: vec![Port { name: "in".into(), size: 1 }],
            outputs: vec![Port { name: "out".into(), size: 1 }],
        },
        params: vec![],
        state: vec![],
        memory: vec![],
        regions: Regions {
            alg: Region {
                ops: vec![
                    Op::Input { port: 0, elem: 0 },
                    Op::Const(-1.0),
                    Op::Binary { op: BinOpKind::Mul, a: NodeId(0), b: NodeId(1) },
                ],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(2) }],
            },
            dyn_: Region::default(),
        },
        events: vec![],
    };
    let root = Subsystem {
        id: SubsystemId(0),
        name: "withlocal".into(),
        interface: Interface {
            inputs: vec![Port { name: "u".into(), size: 1 }],
            outputs: vec![Port { name: "y".into(), size: 1 }],
        },
        children: vec![Child::Block(integrator_block(0, "integ")), Child::Block(amp)],
        connections: vec![
            Connection { id: ConnectionId(0), src: pr(BlockId::INTERFACE, 0), targets: vec![pr(BlockId(0), 0)] },
            Connection { id: ConnectionId(1), src: pr(BlockId(0), 0), targets: vec![pr(BlockId::INTERFACE, 0)] },
            Connection { id: ConnectionId(2), src: pr(BlockId(0), 0), targets: vec![pr(BlockId(1), 0)] },
        ],
        schedule: Schedule { topo: vec![BlockId(0), BlockId(1)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    Module::new("withlocal", root)
}

#[test]
fn subsystem_output_causality_distinguishes_interface_from_internal() {
    let files = fmu_files(&subsystem_with_local_module(), &ExportOptions::default()).unwrap();
    let md = ModelDescription::from_str(
        &files.iter().find(|f| f.name == "modelDescription.xml").unwrap().contents,
    )
    .unwrap();
    // Exactly one FMI output (the interface output `y`), driven by the integrator.
    let outputs: Vec<_> = md.outputs().collect();
    assert_eq!(outputs.len(), 1, "exactly one interface output");
    // The amplifier output is observable but `local`, not `output`.
    let locals = md.variables.iter().filter(|v| v.causality == Causality::Local).count();
    assert!(locals >= 1, "internal amplifier output should be a local variable");
    // A closed model keeps all outputs as outputs (regression guard).
    let closed = fmu_files(&decay_module(), &ExportOptions::default()).unwrap();
    let cmd = ModelDescription::from_str(
        &closed.iter().find(|f| f.name == "modelDescription.xml").unwrap().contents,
    )
    .unwrap();
    assert!(cmd.outputs().count() >= 1, "closed model still exposes outputs");
}

/// Write the FMU's `sources/` files flat into a fresh temp dir and return it.
fn write_sources(files: &[fastsim::fmi::export::FmuFile], tag: &str) -> std::path::PathBuf {
    let uniq = UNIQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("fastsim_fmu_{tag}_{uniq}"));
    std::fs::create_dir_all(&dir).unwrap();
    for f in files {
        if let Some(base) = f.name.strip_prefix("sources/") {
            std::fs::write(dir.join(base), &f.contents).unwrap();
        }
    }
    dir
}

#[test]
fn zero_cross_event_fires_effect_once() {
    let m = zerocross_module();
    let files = fmu_files(&m, &ExportOptions::default()).unwrap();
    let md = ModelDescription::from_str(
        &files.iter().find(|f| f.name == "modelDescription.xml").unwrap().contents,
    )
    .unwrap();
    // The model declares exactly one event indicator and requires
    // CompletedIntegratorStep (where the FMU records the previous indicator).
    assert_eq!(md.n_event_indicators(), 1);
    assert!(md.model_exchange.as_ref().unwrap().needs_completed_integrator_step);
    let token = md.instantiation_token.clone();

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping zero-cross behaviour check");
        return;
    };
    let dir = write_sources(&files, "zc");

    // A Model-Exchange host with host-side event detection (compare indicators
    // across steps) and the FMI event lifecycle. The counter state x[1] should
    // be exactly 1 after the single falling crossing.
    let main_c = format!(
        r#"#include <stdio.h>
#include "fmi3.h"
#define DT 0.01
#define T_END 1.5
int main(void) {{
    fmi3Instance m = fmi3InstantiateModelExchange("zc", "{token}", "", fmi3False, fmi3False, NULL, NULL);
    if (!m) return 1;
    fmi3EnterInitializationMode(m, fmi3False, 0.0, 0.0, fmi3False, 0.0);
    fmi3ExitInitializationMode(m);
    fmi3EnterContinuousTimeMode(m);
    size_t ns = 0, nz = 0;
    fmi3GetNumberOfContinuousStates(m, &ns);
    fmi3GetNumberOfEventIndicators(m, &nz);
    double x[8], k1[8], k2[8], k3[8], k4[8], xt[8], zpre[4], zpost[4];
    fmi3GetContinuousStates(m, x, ns);
    fmi3CompletedIntegratorStep(m, fmi3True, NULL, NULL);
    fmi3GetEventIndicators(m, zpre, nz);
    double t = 0.0;
    while (t < T_END - 0.5 * DT) {{
        fmi3SetTime(m, t); fmi3SetContinuousStates(m, x, ns); fmi3GetContinuousStateDerivatives(m, k1, ns);
        for (size_t i = 0; i < ns; i++) xt[i] = x[i] + 0.5 * DT * k1[i];
        fmi3SetTime(m, t + 0.5 * DT); fmi3SetContinuousStates(m, xt, ns); fmi3GetContinuousStateDerivatives(m, k2, ns);
        for (size_t i = 0; i < ns; i++) xt[i] = x[i] + 0.5 * DT * k2[i];
        fmi3SetContinuousStates(m, xt, ns); fmi3GetContinuousStateDerivatives(m, k3, ns);
        for (size_t i = 0; i < ns; i++) xt[i] = x[i] + DT * k3[i];
        fmi3SetTime(m, t + DT); fmi3SetContinuousStates(m, xt, ns); fmi3GetContinuousStateDerivatives(m, k4, ns);
        for (size_t i = 0; i < ns; i++) x[i] += DT / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        t += DT;
        fmi3SetTime(m, t); fmi3SetContinuousStates(m, x, ns);
        fmi3GetEventIndicators(m, zpost, nz);
        int ev = 0;
        for (size_t i = 0; i < nz; i++) if ((zpre[i] > 0.0) != (zpost[i] > 0.0)) ev = 1;
        if (ev) {{
            fmi3EnterEventMode(m);
            fmi3Boolean a, b, c, d, e2; double nt;
            fmi3UpdateDiscreteStates(m, &a, &b, &c, &d, &e2, &nt);
            fmi3EnterContinuousTimeMode(m);
            fmi3GetContinuousStates(m, x, ns);   /* effect changed the state */
        }}
        fmi3CompletedIntegratorStep(m, fmi3True, NULL, NULL);
        for (size_t i = 0; i < nz; i++) zpre[i] = zpost[i];
    }}
    fmi3GetContinuousStates(m, x, ns);
    printf("%.17g\n", x[1]);
    fmi3FreeInstance(m);
    return 0;
}}
"#
    );
    std::fs::write(dir.join("main.c"), main_c).unwrap();
    let out = fastsim::codegen::verify::cc_command_in(&cc, &dir)
        .args(["fmu.c", "main.c", "-O0", "-o", "zc.exe", "-lm"])
        .output()
        .expect("spawn cc");
    assert!(
        out.status.success(),
        "FMU C did not compile:\n{}\n--- fmu.c ---\n{}",
        String::from_utf8_lossy(&out.stderr),
        files.iter().find(|f| f.name == "sources/fmu.c").unwrap().contents,
    );
    let run = match Command::new(dir.join("zc.exe")).output() {
        Ok(r) if r.status.success() => r,
        _ => {
            eprintln!("exe would not launch — skipping numeric comparison");
            return;
        }
    };
    let got: f64 = String::from_utf8_lossy(&run.stdout).trim().parse().unwrap_or(f64::NAN);
    assert!((got - 1.0).abs() < 1e-9, "counter after one crossing = {got}, expected 1");
}

/// A periodic event that increments a discrete *memory* counter (not a state):
/// `x' = 0`, memory `c += 1` every 0.5, output `y = c`. The effect is a
/// MemoryWrite, so it must NOT flag `valuesOfContinuousStatesChanged`.
fn mem_event_module() -> Module {
    let block = Block {
        id: BlockId(0),
        name: "memev".into(),
        type_name: "Counter".into(),
        role: BlockRole::Dynamic,
        ports: Ports { inputs: vec![], outputs: vec![Port { name: "y".into(), size: 1 }] },
        params: vec![],
        state: vec![StateVar { id: StateId(0), name: "x".into(), init: 0.0 }],
        memory: vec![MemorySlot { id: MemorySlotId(0), name: "c".into(), size: 1, init: vec![0.0] }],
        regions: Regions {
            alg: Region {
                ops: vec![Op::Memory { slot: MemorySlotId(0), offset: 0 }],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
            },
            dyn_: Region {
                ops: vec![Op::Const(0.0)],
                writes: vec![Write::StateDeriv { id: StateId(0), src: NodeId(0) }],
            },
        },
        events: vec![Event {
            id: EventId(0),
            kind: EventKind::Schedule { times: ScheduleTimes::Periodic { period: 0.5, phase: 0.5 } },
            effect: Region {
                ops: vec![
                    Op::Memory { slot: MemorySlotId(0), offset: 0 },
                    Op::Const(1.0),
                    Op::Binary { op: BinOpKind::Add, a: NodeId(0), b: NodeId(1) },
                ],
                writes: vec![Write::MemoryWrite { slot: MemorySlotId(0), offset: 0, src: NodeId(2) }],
            },
            opaque: false,
        }],
    };
    let root = Subsystem {
        id: SubsystemId(0),
        name: "root".into(),
        interface: Interface::default(),
        children: vec![Child::Block(block)],
        connections: vec![],
        schedule: Schedule { topo: vec![BlockId(0)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    Module::new("memev", root)
}

#[test]
fn values_changed_flag_reflects_state_writes_only() {
    // A state-modifying event (periodic_module: x += 1) sets `changed`; a
    // memory-only event (mem_event_module: c += 1) must not.
    let state_c = fmu_files(&periodic_module(), &ExportOptions::default())
        .unwrap()
        .into_iter()
        .find(|f| f.name == "sources/fmu.c")
        .unwrap()
        .contents;
    assert!(state_c.contains("changed = 1"), "state event should flag changed:\n{state_c}");

    let mem_c = fmu_files(&mem_event_module(), &ExportOptions::default())
        .unwrap()
        .into_iter()
        .find(|f| f.name == "sources/fmu.c")
        .unwrap()
        .contents;
    assert!(mem_c.contains("effect_0_0(m);"), "mem event effect should still fire:\n{mem_c}");
    assert!(!mem_c.contains("changed = 1"), "mem-only event must NOT flag changed:\n{mem_c}");
}

#[test]
fn periodic_event_advances_state() {
    let m = periodic_module();
    let files = fmu_files(&m, &ExportOptions::default()).unwrap();
    let md = ModelDescription::from_str(
        &files.iter().find(|f| f.name == "modelDescription.xml").unwrap().contents,
    )
    .unwrap();
    // A time event has no indicator.
    assert_eq!(md.n_event_indicators(), 0);
    let token = md.instantiation_token.clone();

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping periodic-event behaviour check");
        return;
    };
    let dir = write_sources(&files, "per");

    // A host that integrates and fires the time event whenever simulation time
    // reaches the FMU-reported next event time. Events at 0.5, 1.0, 1.5 → x = 3.
    let main_c = format!(
        r#"#include <stdio.h>
#include <math.h>
#include "fmi3.h"
#define DT 0.01
#define T_END 1.6
int main(void) {{
    fmi3Instance m = fmi3InstantiateModelExchange("per", "{token}", "", fmi3False, fmi3False, NULL, NULL);
    if (!m) return 1;
    fmi3EnterInitializationMode(m, fmi3False, 0.0, 0.0, fmi3False, 0.0);
    fmi3ExitInitializationMode(m);
    fmi3EnterContinuousTimeMode(m);
    size_t ns = 0;
    fmi3GetNumberOfContinuousStates(m, &ns);
    double x[4], k1[4];
    fmi3GetContinuousStates(m, x, ns);
    fmi3Boolean a, b, c, d, e2; double nt = INFINITY;
    fmi3EnterEventMode(m);
    fmi3UpdateDiscreteStates(m, &a, &b, &c, &d, &e2, &nt);
    fmi3EnterContinuousTimeMode(m);
    fmi3GetContinuousStates(m, x, ns);
    double next = e2 ? nt : INFINITY;
    double t = 0.0;
    while (t < T_END - 0.5 * DT) {{
        fmi3SetTime(m, t); fmi3SetContinuousStates(m, x, ns); fmi3GetContinuousStateDerivatives(m, k1, ns);
        for (size_t i = 0; i < ns; i++) x[i] += DT * k1[i];
        t += DT;
        fmi3SetTime(m, t); fmi3SetContinuousStates(m, x, ns);
        fmi3CompletedIntegratorStep(m, fmi3True, NULL, NULL);
        if (t + 1e-9 >= next) {{
            fmi3EnterEventMode(m);
            fmi3UpdateDiscreteStates(m, &a, &b, &c, &d, &e2, &nt);
            fmi3EnterContinuousTimeMode(m);
            fmi3GetContinuousStates(m, x, ns);
            next = e2 ? nt : INFINITY;
        }}
    }}
    fmi3GetContinuousStates(m, x, ns);
    printf("%.17g\n", x[0]);
    fmi3FreeInstance(m);
    return 0;
}}
"#
    );
    std::fs::write(dir.join("main.c"), main_c).unwrap();
    let out = fastsim::codegen::verify::cc_command_in(&cc, &dir)
        .args(["fmu.c", "main.c", "-O0", "-o", "per.exe", "-lm"])
        .output()
        .expect("spawn cc");
    assert!(
        out.status.success(),
        "FMU C did not compile:\n{}\n--- fmu.c ---\n{}",
        String::from_utf8_lossy(&out.stderr),
        files.iter().find(|f| f.name == "sources/fmu.c").unwrap().contents,
    );
    let run = match Command::new(dir.join("per.exe")).output() {
        Ok(r) if r.status.success() => r,
        _ => {
            eprintln!("exe would not launch — skipping numeric comparison");
            return;
        }
    };
    let got: f64 = String::from_utf8_lossy(&run.stdout).trim().parse().unwrap_or(f64::NAN);
    assert!((got - 3.0).abs() < 1e-9, "state after 3 periodic events = {got}, expected 3");
}

#[test]
fn fixed_schedule_event_fires_at_listed_times() {
    let m = fixed_schedule_module();
    let files = fmu_files(&m, &ExportOptions::default()).unwrap();
    let fmu_c = files.iter().find(|f| f.name == "sources/fmu.c").unwrap().contents.clone();
    // The Fixed path emits the explicit time list and indexes it with `fi_`; it
    // must NOT fall back to the periodic constant-stride update.
    assert!(fmu_c.contains("times_0_0"), "fixed schedule should emit a time table:\n{fmu_c}");
    assert!(fmu_c.contains("m->fi_0_0"), "fixed schedule should index with fi_:\n{fmu_c}");

    let md = ModelDescription::from_str(
        &files.iter().find(|f| f.name == "modelDescription.xml").unwrap().contents,
    )
    .unwrap();
    // A time event has no indicator.
    assert_eq!(md.n_event_indicators(), 0);
    let token = md.instantiation_token.clone();

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping fixed-schedule behaviour check");
        return;
    };
    let dir = write_sources(&files, "fix");

    // A host that integrates and fires the time event whenever simulation time
    // reaches the FMU-reported next event time. Events at the uneven times
    // 0.3, 0.7, 1.2 → x = 3. The FMU drives the schedule entirely via the
    // nextEventTime it reports back; the host never hard-codes the times.
    let main_c = format!(
        r#"#include <stdio.h>
#include <math.h>
#include "fmi3.h"
#define DT 0.01
#define T_END 1.5
int main(void) {{
    fmi3Instance m = fmi3InstantiateModelExchange("fix", "{token}", "", fmi3False, fmi3False, NULL, NULL);
    if (!m) return 1;
    fmi3EnterInitializationMode(m, fmi3False, 0.0, 0.0, fmi3False, 0.0);
    fmi3ExitInitializationMode(m);
    fmi3EnterContinuousTimeMode(m);
    size_t ns = 0;
    fmi3GetNumberOfContinuousStates(m, &ns);
    double x[4], k1[4];
    fmi3GetContinuousStates(m, x, ns);
    fmi3Boolean a, b, c, d, e2; double nt = INFINITY;
    fmi3EnterEventMode(m);
    fmi3UpdateDiscreteStates(m, &a, &b, &c, &d, &e2, &nt);
    fmi3EnterContinuousTimeMode(m);
    fmi3GetContinuousStates(m, x, ns);
    double next = e2 ? nt : INFINITY;
    double t = 0.0;
    while (t < T_END - 0.5 * DT) {{
        fmi3SetTime(m, t); fmi3SetContinuousStates(m, x, ns); fmi3GetContinuousStateDerivatives(m, k1, ns);
        for (size_t i = 0; i < ns; i++) x[i] += DT * k1[i];
        t += DT;
        fmi3SetTime(m, t); fmi3SetContinuousStates(m, x, ns);
        fmi3CompletedIntegratorStep(m, fmi3True, NULL, NULL);
        if (t + 1e-9 >= next) {{
            fmi3EnterEventMode(m);
            fmi3UpdateDiscreteStates(m, &a, &b, &c, &d, &e2, &nt);
            fmi3EnterContinuousTimeMode(m);
            fmi3GetContinuousStates(m, x, ns);
            next = e2 ? nt : INFINITY;
        }}
    }}
    fmi3GetContinuousStates(m, x, ns);
    printf("%.17g\n", x[0]);
    fmi3FreeInstance(m);
    return 0;
}}
"#
    );
    std::fs::write(dir.join("main.c"), main_c).unwrap();
    let out = fastsim::codegen::verify::cc_command_in(&cc, &dir)
        .args(["fmu.c", "main.c", "-O0", "-o", "fix.exe", "-lm"])
        .output()
        .expect("spawn cc");
    assert!(
        out.status.success(),
        "FMU C did not compile:\n{}\n--- fmu.c ---\n{}",
        String::from_utf8_lossy(&out.stderr),
        fmu_c,
    );
    let run = match Command::new(dir.join("fix.exe")).output() {
        Ok(r) if r.status.success() => r,
        _ => {
            eprintln!("exe would not launch — skipping numeric comparison");
            return;
        }
    };
    let got: f64 = String::from_utf8_lossy(&run.stdout).trim().parse().unwrap_or(f64::NAN);
    assert!((got - 3.0).abs() < 1e-9, "state after 3 fixed-schedule events = {got}, expected 3");
}

/// Compile an open `x' = u` FMU, drive its single input `u = 2` via
/// fmi3SetFloat64, integrate to t = 3 (RK4), and return `(x_final, y_output)` —
/// or `None` when there is no C compiler (skip). Asserts the FMU compiled.
fn drive_unit_input_integrator(files: &[fastsim::fmi::export::FmuFile], tag: &str) -> Option<(f64, f64)> {
    let md = ModelDescription::from_str(
        &files.iter().find(|f| f.name == "modelDescription.xml").unwrap().contents,
    )
    .unwrap();
    let input_vr = md.inputs().find(|v| v.name == "u").expect("input variable u").value_reference;
    let output_vr = md.outputs().next().expect("an output").value_reference;
    let model_id = md.model_exchange.as_ref().unwrap().model_identifier.clone();
    let token = md.instantiation_token.clone();

    let cc = find_cc()?;
    let dir = write_sources(files, tag);
    // Drive u = 2, integrate x' = u to t = 3 (x(3) = 6); read state and output y.
    let main_c = format!(
        r#"#include <stdio.h>
#include "fmi3.h"
#define DT 0.01
#define T_END 3.0
int main(void) {{
    fmi3Instance m = fmi3InstantiateModelExchange("{model_id}", "{token}", "", fmi3False, fmi3False, NULL, NULL);
    if (!m) return 1;
    fmi3EnterInitializationMode(m, fmi3False, 0.0, 0.0, fmi3False, 0.0);
    fmi3ValueReference uvr = {input_vr}u; double uval = 2.0;
    if (fmi3SetFloat64(m, &uvr, 1, &uval, 1) != fmi3OK) return 5;
    fmi3ExitInitializationMode(m);
    fmi3EnterContinuousTimeMode(m);
    size_t ns = 0; fmi3GetNumberOfContinuousStates(m, &ns);
    double x[4], k1[4], k2[4], k3[4], k4[4], xt[4];
    fmi3GetContinuousStates(m, x, ns);
    double t = 0.0;
    while (t < T_END - 0.5 * DT) {{
        fmi3SetTime(m, t); fmi3SetContinuousStates(m, x, ns); fmi3GetContinuousStateDerivatives(m, k1, ns);
        for (size_t i = 0; i < ns; i++) xt[i] = x[i] + 0.5 * DT * k1[i];
        fmi3SetTime(m, t + 0.5 * DT); fmi3SetContinuousStates(m, xt, ns); fmi3GetContinuousStateDerivatives(m, k2, ns);
        for (size_t i = 0; i < ns; i++) xt[i] = x[i] + 0.5 * DT * k2[i];
        fmi3SetContinuousStates(m, xt, ns); fmi3GetContinuousStateDerivatives(m, k3, ns);
        for (size_t i = 0; i < ns; i++) xt[i] = x[i] + DT * k3[i];
        fmi3SetTime(m, t + DT); fmi3SetContinuousStates(m, xt, ns); fmi3GetContinuousStateDerivatives(m, k4, ns);
        for (size_t i = 0; i < ns; i++) x[i] += DT / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        t += DT;
    }}
    fmi3SetTime(m, t); fmi3SetContinuousStates(m, x, ns);
    fmi3GetContinuousStateDerivatives(m, k1, ns);   /* refresh signals for the output read */
    fmi3ValueReference yvr = {output_vr}u; double yval = 0.0;
    fmi3GetFloat64(m, &yvr, 1, &yval, 1);
    printf("%.17g %.17g\n", x[0], yval);
    fmi3FreeInstance(m);
    return 0;
}}
"#
    );
    std::fs::write(dir.join("main.c"), main_c).unwrap();
    let out = fastsim::codegen::verify::cc_command_in(&cc, &dir)
        .args(["fmu.c", "main.c", "-O0", "-o", "run.exe", "-lm"])
        .output()
        .expect("spawn cc");
    assert!(
        out.status.success(),
        "FMU C did not compile:\n{}\n--- fmu.c ---\n{}",
        String::from_utf8_lossy(&out.stderr),
        files.iter().find(|f| f.name == "sources/fmu.c").unwrap().contents,
    );
    let run = match Command::new(dir.join("run.exe")).output() {
        Ok(r) if r.status.success() => r,
        _ => return None,
    };
    let v: Vec<f64> = String::from_utf8_lossy(&run.stdout)
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect();
    assert_eq!(v.len(), 2, "unexpected output");
    Some((v[0], v[1]))
}

#[test]
fn subsystem_with_input_exports_open_fmu() {
    let files = fmu_files(&subsystem_input_module(), &ExportOptions::default()).unwrap();
    let md = ModelDescription::from_str(
        &files.iter().find(|f| f.name == "modelDescription.xml").unwrap().contents,
    )
    .unwrap();
    // The interface input `u` is an FMI input variable; `y` is an output.
    assert_eq!(md.inputs().find(|v| v.name == "u").expect("input u").causality, Causality::Input);
    // The generated struct carries an external-input vector the deriv reads.
    let model_h = &files.iter()
        .find(|f| f.name.starts_with("sources/") && f.name.ends_with(".h") && !f.name.ends_with("fmi3.h"))
        .unwrap().contents;
    let model_c = &files.iter()
        .find(|f| f.name.ends_with(".c") && !f.name.ends_with("fmu.c"))
        .unwrap().contents;
    assert!(model_h.contains("u[1]"), "struct should have an input vector:\n{model_h}");
    assert!(model_c.contains("m->u[0]"), "deriv should read the external input:\n{model_c}");

    match drive_unit_input_integrator(&files, "sub") {
        None => eprintln!("no C compiler / exe — skipping subsystem-input behaviour check"),
        Some((x, y)) => {
            assert!((x - 6.0).abs() < 1e-9, "x(3) with u=2 = {x}, expected 6");
            assert!((y - x).abs() < 1e-12, "output y={y} != state x={x}");
        }
    }
}

#[test]
fn nested_subsystem_input_resolves_through_nesting() {
    // The root-interface input feeds a *nested* subsystem before reaching the
    // integrator, exercising the recursive resolution of external inputs.
    let files = fmu_files(&nested_subsystem_input_module(), &ExportOptions::default()).unwrap();
    let md = ModelDescription::from_str(
        &files.iter().find(|f| f.name == "modelDescription.xml").unwrap().contents,
    )
    .unwrap();
    assert_eq!(md.inputs().count(), 1, "one external input through the nesting");
    assert_eq!(md.inputs().next().unwrap().causality, Causality::Input);
    let model_c = &files.iter()
        .find(|f| f.name.ends_with(".c") && !f.name.ends_with("fmu.c"))
        .unwrap().contents;
    assert!(model_c.contains("m->u[0]"), "nested deriv should read the external input:\n{model_c}");

    match drive_unit_input_integrator(&files, "nested") {
        None => eprintln!("no C compiler / exe — skipping nested-subsystem behaviour check"),
        Some((x, y)) => {
            assert!((x - 6.0).abs() < 1e-9, "nested x(3) with u=2 = {x}, expected 6");
            assert!((y - x).abs() < 1e-12, "nested output y={y} != state x={x}");
        }
    }
}
