//! Whole-system verification: emit a complete `model.c` from an IR `Module`,
//! integrate it in compiled C, and compare the trajectory to (a) an IR-level
//! RK4 reference driven by `fastsim::ir::eval` and (b) the analytic solution.
//! Skips when no working C compiler is available (see `common`).
#![cfg(feature = "codegen")]

use fastsim::ir::eval::{eval_region, EvalCtx};
use fastsim::ir::schema::{
    BinOpKind, Block, BlockId, BlockRole, Child, CmpKind, Connection, ConnectionId, Direction,
    Event, EventId, EventKind, Interface, MemorySlot, MemorySlotId, Module, NodeId, Op, Port,
    Param, ParamId, ParamValue, PortRef, Ports, Region, Regions, Schedule, ScheduleTimes,
    StateId, StateVar, Subsystem, SubsystemId, UnaryOpKind, Write,
};
use fastsim::codegen::{
    generate, CodegenOptions, Layout, ModelApi, SolverChoice, Structure,
};

mod common;
use common::{compile_and_run_files, compile_and_run_named, concat_sources, find_cc};

/// A single self-contained dynamic block: dx/dt = -x, x(0) = 1, y = x.
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

/// Reference RK4 over the IR `dyn` region — the same method the generated C
/// loop uses, evaluated through `fastsim::ir::eval`.
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

/// SISO port set with `nin` inputs and `nout` outputs.
fn io(nin: usize, nout: usize) -> Ports {
    Ports {
        inputs: (0..nin).map(|i| Port { name: format!("in{i}"), size: 1 }).collect(),
        outputs: (0..nout).map(|i| Port { name: format!("out{i}"), size: 1 }).collect(),
    }
}

/// A single input port of `nin` elements and a single output port of `nout`
/// elements (the builder's MIMO convention: one wide "in"/"out" port).
fn io1(nin: u32, nout: u32) -> Ports {
    Ports {
        inputs: if nin > 0 { vec![Port { name: "in".into(), size: nin }] } else { vec![] },
        outputs: if nout > 0 { vec![Port { name: "out".into(), size: nout }] } else { vec![] },
    }
}

/// A simple dynamic integrator block: y = x, dx/dt = u.
fn integrator(id: u32, name: &str, x0: f64) -> Block {
    Block {
        id: BlockId(id),
        name: name.into(),
        type_name: "Integrator".into(),
        role: BlockRole::Dynamic,
        ports: io(1, 1),
        params: vec![],
        state: vec![StateVar { id: StateId(0), name: "x0".into(), init: x0 }],
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

/// Harmonic oscillator wired from three blocks: int_x (dx/dt = v), int_v
/// (dv/dt = -x), and an inverting amplifier (-x) closing the loop. With
/// x(0)=1, v(0)=0 the exact solution is x(t)=cos t, v(t)=-sin t.
fn oscillator_module() -> Module {
    let int_x = integrator(0, "int_x", 1.0); // state x, dx/dt = v_in
    let int_v = integrator(1, "int_v", 0.0); // state v, dv/dt = (-x)_in
    let amp = Block {
        id: BlockId(2),
        name: "amp".into(),
        type_name: "Amplifier".into(),
        role: BlockRole::Algebraic,
        ports: io(1, 1),
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

    let conn = |id: u32, sb: u32, tb: u32| Connection {
        id: ConnectionId(id),
        src: PortRef { block: BlockId(sb), port: 0, elems: None },
        targets: vec![PortRef { block: BlockId(tb), port: 0, elems: None }],
    };
    let root = Subsystem {
        id: SubsystemId(0),
        name: "root".into(),
        interface: Interface::default(),
        children: vec![Child::Block(int_x), Child::Block(int_v), Child::Block(amp)],
        connections: vec![
            conn(0, 0, 2), // x      -> amp.in
            conn(1, 2, 1), // -x     -> int_v.in
            conn(2, 1, 0), // v      -> int_x.in
        ],
        schedule: Schedule {
            topo: vec![BlockId(0), BlockId(1), BlockId(2)],
            groups: vec![],
            sccs: vec![],
            back_edges: vec![],
        },
    };
    Module::new("oscillator", root)
}

#[test]
fn generated_oscillator_matches_analytic() {
    let module = oscillator_module();
    let files = generate(&module, &CodegenOptions::default()).expect("generate");
    let model = concat_sources(&files);
    // Hierarchical provenance + wiring are visible in the generated C.
    assert!(model.contains("/* alg of int_x */"), "{model}");
    assert!(model.contains("/* d/dt int_v */"), "{model}");
    assert!(model.contains("#define OSCILLATOR_N_STATE 2"), "{model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping oscillator integration check");
        return;
    };

    let (t_end, dt) = (2.0_f64, 1e-3_f64);
    let main = format!(
        "#include <stdio.h>\nint main(void) {{\n\
         \x20   oscillator_t m;\n\
         \x20   oscillator_init(&m);\n\
         \x20   oscillator_run(&m, {t_end:?}, {dt:?});\n\
         \x20   printf(\"%.17g %.17g\", m.x[0], m.x[1]);\n\
         \x20   return 0;\n\
         }}\n"
    );
    match compile_and_run_files(&cc, 11, &main, &files).expect("compile oscillator model.c") {
        None => eprintln!("oscillator exe would not launch — skipping numeric check"),
        Some(got) => {
            assert_eq!(got.len(), 2);
            // x(t) = cos t, v(t) = -sin t.
            assert!((got[0] - t_end.cos()).abs() < 1e-5, "x: C={} cos={}", got[0], t_end.cos());
            assert!((got[1] - (-t_end.sin())).abs() < 1e-5, "v: C={} -sin={}", got[1], -t_end.sin());
        }
    }
}

/// Every generated file carries the license notice in its banner. The generated
/// C is "Output" under PolyForm Noncommercial: commercial use needs a license, and
/// the notice must travel on the artifact itself (not just in the repo LICENSE) so
/// it reaches downstream recipients. Guards both layouts against silent regression.
#[test]
fn generated_files_carry_license_notice() {
    for layout in [Layout::Compact, Layout::Library] {
        let opts = CodegenOptions { layout, ..Default::default() };
        let files = generate(&decay_module(), &opts).expect("generate");
        assert!(!files.is_empty());
        for f in &files {
            assert!(
                f.contents.contains("PolyForm Noncommercial"),
                "{} ({layout:?}) is missing the license banner",
                f.name
            );
            assert!(
                f.contents.contains("commercial license") && f.contents.contains("info@pathsim.org"),
                "{} ({layout:?}) is missing the commercial-license notice",
                f.name
            );
        }
    }
}

/// Adaptive integrator: the `decay` model (dx/dt = -x, x(0) = 1) emitted with an
/// adaptive tableau (RKDP54) gets the embedded-error step controller + the carried
/// `fs_h` field, and integrates to the analytic solution x(t) = e^-t. The seed `dt`
/// is deliberately coarse (0.5): a fixed-step method would be wildly inaccurate, so
/// matching e^-5 proves the step controller is actually adapting.
#[test]
fn generated_adaptive_decay_matches_analytic() {
    let module = decay_module();
    let opts = CodegenOptions { solver: SolverChoice::by_name("RKDP54").unwrap(), ..Default::default() };
    let files = generate(&module, &opts).expect("generate adaptive decay");
    let model = concat_sources(&files);
    // Adaptive emission markers: the error-estimating substep, the carried step,
    // and the I-controller.
    assert!(model.contains("fs_trial_step"), "expected adaptive trial-step kernel, got: {model}");
    assert!(model.contains("m->fs_h"), "expected carried adaptive step fs_h: {model}");
    assert!(model.contains("static const double fs_tr"), "expected embedded-error coeffs: {model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping adaptive integration check");
        return;
    };
    let (t_end, dt) = (5.0_f64, 0.5_f64); // coarse seed step on purpose
    let main = format!(
        "#include <stdio.h>\nint main(void) {{\n\
         \x20   decay_t m;\n\
         \x20   decay_init(&m);\n\
         \x20   decay_run(&m, {t_end:?}, {dt:?});\n\
         \x20   printf(\"%.17g\", m.x[0]);\n\
         \x20   return 0;\n}}\n"
    );
    match compile_and_run_files(&cc, 31, &main, &files).expect("compile adaptive decay") {
        None => eprintln!("adaptive decay exe would not launch — skipping numeric check"),
        Some(got) => {
            let exact = (-t_end).exp();
            // rtol-level accuracy (SOL_TOLERANCE_LTE_REL = 1e-4) against the analytic value.
            assert!((got[0] - exact).abs() < 1e-4, "x(5): C={} exact={}", got[0], exact);
        }
    }
}

/// End-to-end LUT1D: a real `lut1d` block (built via the runtime constructor)
/// goes through `module_from_sim` -> codegen. Its alg region is lowered to a
/// structured `Op::Lut1d`, so the generated C carries a `static const` table
/// (not an unrolled select chain). `constant(0.5)` -> lut([0,1,2]->[0,10,40])
/// gives 5, fed into an integrator, so `x(1) = 5`.
#[test]
fn generated_lut1d_block_emits_table() {
    use fastsim::blocks::constructors::{constant, integrator, lut1d, ExtrapMode};
    use fastsim::connection::Connection as RtConn;
    use fastsim::ir::builder::module_from_sim;
    use fastsim::simulation::Simulation;
    use fastsim::utils::portreference::PortReference;
    use std::rc::Rc;

    let conn = |a: &fastsim::blocks::block::BlockRef, b: &fastsim::blocks::block::BlockRef| {
        Rc::new(RtConn::new(
            PortReference::new(a.clone(), None),
            vec![PortReference::new(b.clone(), None)],
        ))
    };
    let src = constant(0.5);
    let lut = lut1d(vec![0.0, 1.0, 2.0], vec![0.0, 10.0, 40.0], ExtrapMode::Extrapolate).unwrap();
    let integ = integrator(0.0);
    let mut sim = Simulation::with_defaults(
        vec![src.clone(), lut.clone(), integ.clone()],
        vec![conn(&src, &lut), conn(&lut, &integ)],
    );
    sim.run(0.01, false, false);

    let module = module_from_sim(&sim, "lut");
    let files = generate(&module, &CodegenOptions::default()).expect("generate lut model");
    let model = concat_sources(&files);
    assert!(model.contains("static const double _lx"), "expected a LUT table, got: {model}");
    assert!(model.contains("static const double _ly"), "{model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping LUT-block check");
        return;
    };
    let (t_end, dt) = (1.0_f64, 1e-3_f64);
    let main = format!(
        "#include <stdio.h>\nint main(void) {{\n\
         \x20   lut_t m;\n\
         \x20   lut_init(&m);\n\
         \x20   lut_run(&m, {t_end:?}, {dt:?});\n\
         \x20   printf(\"%.17g\", m.x[0]);\n\
         \x20   return 0;\n}}\n"
    );
    match compile_and_run_files(&cc, 24, &main, &files).expect("compile lut model") {
        None => eprintln!("lut exe would not launch — skipping numeric check"),
        Some(got) => assert!((got[0] - 5.0).abs() < 1e-6, "lut integral: C={} expected=5", got[0]),
    }
}

/// Reentrant: instance state (the counter's memory + event index) lives in a
/// its own `model_t` instance, so two instances run independently. Instance
/// `a` runs to t=0.95 (10 firings), `b` to t=0.45 (5) — a shared file-static
/// counter would make `b` continue from `a`'s value, proving non-interference.
#[test]
fn generated_reentrant_independent_instances() {
    let counter = Block {
        id: BlockId(0),
        name: "counter".into(),
        type_name: "Counter".into(),
        role: BlockRole::Algebraic,
        ports: io(0, 1),
        params: vec![],
        state: vec![],
        memory: vec![MemorySlot { id: MemorySlotId(0), name: "c".into(), size: 1, init: vec![0.0] }],
        regions: Regions {
            alg: Region {
                ops: vec![Op::Memory { slot: MemorySlotId(0), offset: 0 }],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
            },
            dyn_: Region::default(),
        },
        events: vec![Event {
            id: EventId(0),
            kind: EventKind::Schedule { times: ScheduleTimes::Periodic { period: 0.1, phase: 0.0 } },
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
        children: vec![Child::Block(counter)],
        connections: vec![],
        schedule: Schedule { topo: vec![BlockId(0)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    let module = Module::new("reentrant_counter", root);

    // A block with discrete memory (the counter) is reentrant by derivation:
    // its per-instance state lives in its own model_t instance, no opt-in needed.
    let opts = CodegenOptions::default();
    let files = generate(&module, &opts).expect("generate reentrant");
    let model = concat_sources(&files);
    assert!(model.contains("typedef struct {"), "{model}");
    assert!(model.contains("} reentrant_counter_t;"), "{model}");
    assert!(model.contains("void reentrant_counter_run(reentrant_counter_t * restrict m,"), "{model}");
    assert!(!model.contains("static double mem"), "mem must live in the struct, not file-static: {model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping reentrant check");
        return;
    };
    // The struct API is reentrant by construction: each model_t instance carries
    // its own state, so two instances advanced to different times stay independent.
    let main = "#include <stdio.h>\nint main(void) {\n\
         \x20   reentrant_counter_t a, b;\n\
         \x20   reentrant_counter_init(&a);\n\
         \x20   reentrant_counter_init(&b);\n\
         \x20   reentrant_counter_run(&a, 0.95, 0.01);\n\
         \x20   reentrant_counter_run(&b, 0.45, 0.01);\n\
         \x20   printf(\"%.17g %.17g\", a.sig[0], b.sig[0]);\n\
         \x20   return 0;\n}\n";
    match compile_and_run_files(&cc, 23, main, &files).expect("compile reentrant model") {
        None => eprintln!("reentrant exe would not launch — skipping numeric check"),
        Some(got) => {
            assert_eq!(got.len(), 2);
            assert!((got[0] - 10.0).abs() < 1e-9, "instance a: C={} expected=10", got[0]);
            assert!((got[1] - 5.0).abs() < 1e-9, "instance b: C={} expected=5", got[1]);
        }
    }
}

/// Struct API: the oscillator emitted as a single `oscillator_t` with
/// `get_signal`/`set_signal` accessors. Drives it through the struct entry points
/// and reads the states back by name: `x(2) = cos 2`, `v(2) = -sin 2`.
#[test]
fn generated_struct_api_oscillator() {
    let module = oscillator_module();
    let opts = CodegenOptions { api: ModelApi::Struct, ..Default::default() };
    let files = generate(&module, &opts).expect("generate struct api");
    let model = concat_sources(&files);
    assert!(model.contains("} oscillator_t;"), "{model}");
    assert!(model.contains("oscillator_get_signal"), "{model}");
    assert!(model.contains("int oscillator_set_signal"), "{model}");
    assert!(model.contains("OSCILLATOR_SIG_int_x"), "{model}");
    assert!(model.contains("OSCILLATOR_SIG_int_v"), "{model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping struct-api check");
        return;
    };
    // set_signal raises the initial state to x(0) = 2, so the linear system
    // gives x(t) = 2 cos t, v(t) = -2 sin t — verifying both set and get.
    let main = "#include <stdio.h>\nint main(void) {\n\
         \x20   oscillator_t m;\n\
         \x20   oscillator_init(&m);\n\
         \x20   oscillator_set_signal(&m, OSCILLATOR_SIG_int_x, 2.0);\n\
         \x20   oscillator_run(&m, 2.0, 1e-3);\n\
         \x20   printf(\"%.17g %.17g\",\n\
         \x20          oscillator_get_signal(&m, OSCILLATOR_SIG_int_x),\n\
         \x20          oscillator_get_signal(&m, OSCILLATOR_SIG_int_v));\n\
         \x20   return 0;\n}\n";
    match compile_and_run_files(&cc, 25, main, &files).expect("compile struct-api model") {
        None => eprintln!("struct-api exe would not launch — skipping numeric check"),
        Some(got) => {
            assert_eq!(got.len(), 2);
            assert!((got[0] - 2.0 * 2.0_f64.cos()).abs() < 1e-5, "x: C={} 2cos2={}", got[0], 2.0 * 2.0_f64.cos());
            assert!((got[1] - (-2.0 * 2.0_f64.sin())).abs() < 1e-5, "v: C={} -2sin2={}", got[1], -2.0 * 2.0_f64.sin());
        }
    }
}

/// Struct API + Library layout: the model splits into model.{h,c}, blocks.{h,c}
/// (the per-block `blk_i_alg`/`blk_i_deriv`) and solver.{h,c} (the integrator),
/// and the multi-file build links + runs identically to the compact struct model.
#[test]
fn generated_struct_api_library_oscillator() {
    let module = oscillator_module();
    let opts = CodegenOptions {
        api: ModelApi::Struct,
        layout: Layout::Library,
        structure: Structure::Hierarchical,
        ..Default::default()
    };
    let files = generate(&module, &opts).expect("generate struct library");
    let names: Vec<&str> = files.iter().map(|f| f.name.as_str()).collect();
    // Files are named after the model so two generated models can share a
    // build directory (file-level twin of the issue-#34 symbol prefixing).
    for want in [
        "oscillator.h", "oscillator.c",
        "oscillator_blocks.h", "oscillator_blocks.c",
        "oscillator_solver.h", "oscillator_solver.c",
    ] {
        assert!(names.contains(&want), "missing {want}; got {names:?}");
    }
    let src_of = |n: &str| files.iter().find(|f| f.name == n).map(|f| f.contents.clone()).unwrap_or_default();
    // The integrator lives in the solver TU; the per-block functions in blocks.
    assert!(src_of("oscillator_solver.c").contains("void oscillator_run("), "solver TU missing model_run");
    assert!(src_of("oscillator_blocks.c").contains("oscillator_blk_0_alg"), "blocks TU missing per-block fns");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping struct-library check");
        return;
    };
    // Same trajectory as the compact struct oscillator: set x(0)=2 -> x=2cos t, v=-2sin t.
    let main = "#include <stdio.h>\nint main(void) {\n\
         \x20   oscillator_t m;\n\
         \x20   oscillator_init(&m);\n\
         \x20   oscillator_set_signal(&m, OSCILLATOR_SIG_int_x, 2.0);\n\
         \x20   oscillator_run(&m, 2.0, 1e-3);\n\
         \x20   printf(\"%.17g %.17g\",\n\
         \x20          oscillator_get_signal(&m, OSCILLATOR_SIG_int_x),\n\
         \x20          oscillator_get_signal(&m, OSCILLATOR_SIG_int_v));\n\
         \x20   return 0;\n}\n";
    match compile_and_run_files(&cc, 25, main, &files).expect("compile struct-library model") {
        None => eprintln!("struct-library exe would not launch — skipping numeric check"),
        Some(got) => {
            assert_eq!(got.len(), 2);
            assert!((got[0] - 2.0 * 2.0_f64.cos()).abs() < 1e-5, "x: C={} 2cos2={}", got[0], 2.0 * 2.0_f64.cos());
            assert!((got[1] - (-2.0 * 2.0_f64.sin())).abs() < 1e-5, "v: C={} -2sin2={}", got[1], -2.0 * 2.0_f64.sin());
        }
    }
}

/// HIL link scenario (issue #34): two independently generated models — a `plant`
/// (the oscillator) and a `controller` (the decay) — are compiled + LINKED into
/// one binary, with both headers `#include`d into `main.c`, straight from the
/// generator output. Symbols, include guards AND file names all derive from the
/// model name, so the two file sets coexist in one directory without any manual
/// renaming (which the pre-fix `model.{h,c}` naming used to require).
#[test]
fn generated_two_models_link_without_collision() {
    let opts = CodegenOptions { api: ModelApi::Struct, ..Default::default() };
    let plant = generate(&oscillator_module(), &opts).expect("generate plant"); // oscillator.*
    let ctrl = generate(&decay_module(), &opts).expect("generate controller"); // decay.*

    let mut sources: Vec<(String, String)> = plant
        .iter()
        .chain(ctrl.iter())
        .map(|f| (f.name.clone(), f.contents.clone()))
        .collect();
    // Guard against silent regression to shared names: every file unique.
    let mut names: Vec<&str> = sources.iter().map(|(n, _)| n.as_str()).collect();
    names.sort_unstable();
    let n_before = names.len();
    names.dedup();
    assert_eq!(n_before, names.len(), "colliding generated file names: {names:?}");
    sources.sort_by(|a, b| a.0.cmp(&b.0));

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping two-model link check");
        return;
    };
    // Distinct struct types, entry points and guards let both headers land in one
    // TU and both models advance independently in the same process.
    let main = "#include <stdio.h>\n#include \"oscillator.h\"\n#include \"decay.h\"\n\
         int main(void) {\n\
         \x20   oscillator_t a; decay_t b;\n\
         \x20   oscillator_init(&a); decay_init(&b);\n\
         \x20   oscillator_run(&a, 1.0, 1e-3); decay_run(&b, 1.0, 1e-3);\n\
         \x20   printf(\"%.17g %.17g\", a.x[0], b.x[0]);\n\
         \x20   return 0;\n}\n";
    match compile_and_run_named(&cc, 50, &sources, main).expect("compile linked plant+controller") {
        None => eprintln!("linked exe would not launch — skipping numeric check"),
        Some(got) => {
            assert_eq!(got.len(), 2, "got {got:?}");
            // plant: oscillator x(1) = cos 1; controller: decay x(1) = e^-1.
            assert!((got[0] - 1.0_f64.cos()).abs() < 1e-4, "plant x(1): C={} cos1={}", got[0], 1.0_f64.cos());
            assert!((got[1] - (-1.0_f64).exp()).abs() < 1e-4, "controller x(1): C={} e^-1={}", got[1], (-1.0_f64).exp());
        }
    }
}

/// Struct API directional derivative: the analytic `oscillator_jvp` computes the
/// Jacobian-vector product ∂ẋ/∂x · seed by forward-mode AD. The oscillator's
/// state Jacobian is the constant `[[0, 1], [-1, 0]]` (ẋ = v, v̇ = -x), so unit
/// seeds give its columns. The derivative w.r.t. `x` flows through the amplifier
/// signal (`-x`) into `v̇`, exercising the tangent `d_sig` propagation.
#[test]
fn generated_struct_api_jvp_oscillator() {
    let module = oscillator_module();
    let opts = CodegenOptions { api: ModelApi::Struct, ..Default::default() };
    let files = generate(&module, &opts).expect("generate struct api");
    let model = concat_sources(&files);
    assert!(model.contains("void oscillator_jvp("), "{model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping JVP check");
        return;
    };
    // J·e0 = column 0 = [0, -1]; J·e1 = column 1 = [1, 0].
    // model_jvp(m, x_seed, u_seed, p_seed, d_sig, d_dxdt); u/p empty here.
    let main = "#include <stdio.h>\nint main(void) {\n\
         \x20   oscillator_t m;\n\
         \x20   oscillator_init(&m);\n\
         \x20   double e0[2] = {1.0, 0.0}, e1[2] = {0.0, 1.0}, c0[2], c1[2];\n\
         \x20   double dummy[1], dsig[8];\n\
         \x20   oscillator_jvp(&m, e0, dummy, dummy, dsig, c0);\n\
         \x20   oscillator_jvp(&m, e1, dummy, dummy, dsig, c1);\n\
         \x20   printf(\"%.17g %.17g %.17g %.17g\", c0[0], c0[1], c1[0], c1[1]);\n\
         \x20   return 0;\n}\n";
    match compile_and_run_files(&cc, 27, main, &files).expect("compile struct-api jvp model") {
        None => eprintln!("jvp exe would not launch — skipping numeric check"),
        Some(got) => {
            assert_eq!(got.len(), 4, "got {got:?}");
            // [c0[0], c0[1], c1[0], c1[1]] == [0, -1, 1, 0].
            let expect = [0.0, -1.0, 1.0, 0.0];
            for (i, (g, e)) in got.iter().zip(expect).enumerate() {
                assert!((g - e).abs() < 1e-12, "J entry {i}: C={g} expected={e}");
            }
        }
    }
}

/// Flat structure with a *signal-reading* zero-cross event: a ramp `x' = -1,
/// x(0) = 1.5, out = x` feeds a detector whose event guard reads that signal and
/// fires (counter `c += 1`) on the falling zero crossing at t = 1.5. Flat fuses
/// `dx/dt`, but must still emit the algebraic `sig[]` pass the event needs.
#[test]
fn generated_flat_signal_reading_event() {
    let ramp = Block {
        id: BlockId(0),
        name: "ramp".into(),
        type_name: "Ramp".into(),
        role: BlockRole::Dynamic,
        ports: io(0, 1),
        params: vec![],
        state: vec![StateVar { id: StateId(0), name: "x".into(), init: 1.5 }],
        memory: vec![],
        regions: Regions {
            alg: Region {
                ops: vec![Op::State { id: StateId(0) }],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
            },
            dyn_: Region {
                ops: vec![Op::Const(-1.0)],
                writes: vec![Write::StateDeriv { id: StateId(0), src: NodeId(0) }],
            },
        },
        events: vec![],
    };
    let detect = Block {
        id: BlockId(1),
        name: "detect".into(),
        type_name: "ZeroCross".into(),
        role: BlockRole::Algebraic,
        ports: io(1, 1),
        params: vec![],
        memory: vec![MemorySlot { id: MemorySlotId(0), name: "c".into(), size: 1, init: vec![0.0] }],
        state: vec![],
        regions: Regions {
            alg: Region {
                ops: vec![Op::Memory { slot: MemorySlotId(0), offset: 0 }],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
            },
            dyn_: Region::default(),
        },
        events: vec![Event {
            id: EventId(0),
            // Guard reads the block *input* (a signal), so Flat must build sig[].
            kind: EventKind::ZeroCross {
                guard: Region { ops: vec![Op::Input { port: 0, elem: 0 }], writes: vec![] },
                direction: Direction::Falling,
            },
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
    let conn = Connection {
        id: ConnectionId(0),
        src: PortRef { block: BlockId(0), port: 0, elems: None },
        targets: vec![PortRef { block: BlockId(1), port: 0, elems: None }],
    };
    let root = Subsystem {
        id: SubsystemId(0),
        name: "root".into(),
        interface: Interface::default(),
        children: vec![Child::Block(ramp), Child::Block(detect)],
        connections: vec![conn],
        schedule: Schedule { topo: vec![BlockId(0), BlockId(1)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    let module = Module::new("flatzc", root);
    let opts = CodegenOptions { structure: Structure::Flat, ..Default::default() };
    let files = generate(&module, &opts).expect("generate flat signal-reading event");
    let model = concat_sources(&files);
    // Flat fuses the deriv, yet the algebraic pass and the guard are present.
    assert!(model.contains("void flatzc_outputs("), "flat should still emit model_outputs:\n{model}");
    assert!(model.contains("guard_1_0("), "{model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping flat signal-event check");
        return;
    };
    let main = "#include <stdio.h>\nint main(void) {\n\
         \x20   flatzc_t m;\n\
         \x20   flatzc_init(&m);\n\
         \x20   flatzc_run(&m, 2.0, 0.001);\n\
         \x20   printf(\"%.17g\", m.sig[1]);\n\
         \x20   return 0;\n}\n";
    match compile_and_run_files(&cc, 29, main, &files).expect("compile flat signal-event model") {
        None => eprintln!("flat signal-event exe would not launch — skipping numeric check"),
        Some(got) => assert!((got[0] - 1.0).abs() < 1e-9, "flat zero-cross counter: C={} expected=1", got[0]),
    }
}

/// Struct API directional derivative through the digamma helper: a single state
/// with `dx/dt = lgamma(x)` has Jacobian `∂ẋ/∂x = digamma(x)`. The emitted
/// `model_jvp` must call the generated `fastsim_digamma` helper and match the
/// reference `ssa::op::digamma` exactly (this is the op that previously made the
/// whole export fail because C has no stdlib digamma).
#[test]
fn generated_struct_api_jvp_digamma() {
    let block = Block {
        id: BlockId(0),
        name: "g".into(),
        type_name: "ODE".into(),
        role: BlockRole::Dynamic,
        ports: io(0, 1),
        params: vec![],
        state: vec![StateVar { id: StateId(0), name: "x".into(), init: 3.0 }],
        memory: vec![],
        regions: Regions {
            alg: Region {
                ops: vec![Op::State { id: StateId(0) }],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
            },
            dyn_: Region {
                ops: vec![
                    Op::State { id: StateId(0) },
                    Op::Unary { op: UnaryOpKind::Lgamma, a: NodeId(0) },
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
    let module = Module::new("gamma", root);
    let opts = CodegenOptions { api: ModelApi::Struct, ..Default::default() };
    let files = generate(&module, &opts).expect("generate struct api");
    let model = concat_sources(&files);
    assert!(model.contains("fastsim_digamma("), "JVP should call the digamma helper:\n{model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping digamma JVP check");
        return;
    };
    let main = "#include <stdio.h>\nint main(void) {\n\
         \x20   gamma_t m;\n\
         \x20   gamma_init(&m);\n\
         \x20   double seed[1] = {1.0}, col[1], dummy[1], dsig[2];\n\
         \x20   gamma_jvp(&m, seed, dummy, dummy, dsig, col);\n\
         \x20   printf(\"%.17g\", col[0]);\n\
         \x20   return 0;\n}\n";
    match compile_and_run_files(&cc, 28, main, &files).expect("compile digamma jvp model") {
        None => eprintln!("digamma JVP exe would not launch — skipping numeric check"),
        Some(got) => {
            let expected = fastsim::ssa::op::digamma(3.0); // ∂ lgamma/∂x at x=3
            assert_eq!(got.len(), 1, "got {got:?}");
            assert!((got[0] - expected).abs() < 1e-9, "digamma JVP: C={} expected={}", got[0], expected);
        }
    }
}

/// Struct API with events: a periodic counter (`c += 1` every 0.1) feeds an
/// integrator, so `x` integrates the staircase to 5.5 over [0,1). The event's
/// `next_*` counter lives in the struct, `gamma_handle_events(m, dt)` fires it,
/// and the integrator state is read back by name.
#[test]
fn generated_struct_api_with_events() {
    let counter = Block {
        id: BlockId(0),
        name: "counter".into(),
        type_name: "Counter".into(),
        role: BlockRole::Algebraic,
        ports: io(0, 1),
        params: vec![],
        state: vec![],
        memory: vec![MemorySlot { id: MemorySlotId(0), name: "c".into(), size: 1, init: vec![0.0] }],
        regions: Regions {
            alg: Region {
                ops: vec![Op::Memory { slot: MemorySlotId(0), offset: 0 }],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
            },
            dyn_: Region::default(),
        },
        events: vec![Event {
            id: EventId(0),
            kind: EventKind::Schedule { times: ScheduleTimes::Periodic { period: 0.1, phase: 0.0 } },
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
    let integ = Block {
        id: BlockId(1),
        name: "integ".into(),
        type_name: "Integrator".into(),
        role: BlockRole::Dynamic,
        ports: io(1, 1),
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
    };
    let root = Subsystem {
        id: SubsystemId(0),
        name: "root".into(),
        interface: Interface::default(),
        children: vec![Child::Block(counter), Child::Block(integ)],
        connections: vec![Connection {
            id: ConnectionId(0),
            src: PortRef { block: BlockId(0), port: 0, elems: None },
            targets: vec![PortRef { block: BlockId(1), port: 0, elems: None }],
        }],
        schedule: Schedule { topo: vec![BlockId(0), BlockId(1)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    let module = Module::new("evsys", root);

    let opts = CodegenOptions { api: ModelApi::Struct, ..Default::default() };
    let files = generate(&module, &opts).expect("generate struct events");
    let model = concat_sources(&files);
    assert!(model.contains("} evsys_t;"), "{model}");
    assert!(model.contains("void evsys_handle_events(evsys_t"), "{model}");
    assert!(model.contains("k_0_0"), "{model}");
    assert!(model.contains("static void effect_0_0(evsys_t"), "{model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping struct-events check");
        return;
    };
    let main = "#include <stdio.h>\nint main(void) {\n\
         \x20   evsys_t m;\n\
         \x20   evsys_init(&m);\n\
         \x20   evsys_run(&m, 1.0, 0.01);\n\
         \x20   printf(\"%.17g\", evsys_get_signal(&m, EVSYS_SIG_integ));\n\
         \x20   return 0;\n}\n";
    match compile_and_run_files(&cc, 26, main, &files).expect("compile struct-events model") {
        None => eprintln!("struct-events exe would not launch — skipping numeric check"),
        // 5.49 (not the ideal 5.5): the k-th tick lands one step late because
        // `0.01` summed ten times is one ULP below `0.1`, matching the runtime
        // scheduler bit-for-bit (see `generated_periodic_event_counter`).
        Some(got) => assert!((got[0] - 5.49).abs() < 1e-6, "struct events: C={} expected=5.49", got[0]),
    }
}

/// Flat structure: the same three-block oscillator fused into one `model_deriv`.
/// The block boundaries dissolve (no `sig[]`, no per-block functions, the
/// inputs inlined along the connections), but the dynamics are identical, so it
/// still integrates to `x(t) = cos t`, `v(t) = -sin t`.
#[test]
fn generated_flat_oscillator_matches_analytic() {
    let module = oscillator_module();
    let opts = CodegenOptions { structure: Structure::Flat, ..Default::default() };
    let files = generate(&module, &opts).expect("generate flat");
    let model = concat_sources(&files);

    // Fused: derivatives written straight into dxdt, no per-block functions
    // (the block boundaries dissolve into the two driver functions).
    assert!(model.contains("dxdt[0] ="), "{model}");
    assert!(model.contains("dxdt[1] ="), "{model}");
    assert!(!model.contains("oscillator_blk_0_alg"), "flat should have no per-block fns: {model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping flat oscillator check");
        return;
    };

    let (t_end, dt) = (2.0_f64, 1e-3_f64);
    let main = format!(
        "#include <stdio.h>\nint main(void) {{\n\
         \x20   oscillator_t m;\n\
         \x20   oscillator_init(&m);\n\
         \x20   oscillator_run(&m, {t_end:?}, {dt:?});\n\
         \x20   printf(\"%.17g %.17g\", m.x[0], m.x[1]);\n\
         \x20   return 0;\n\
         }}\n"
    );
    match compile_and_run_files(&cc, 21, &main, &files).expect("compile flat oscillator model.c") {
        None => eprintln!("flat oscillator exe would not launch — skipping numeric check"),
        Some(got) => {
            assert_eq!(got.len(), 2);
            assert!((got[0] - t_end.cos()).abs() < 1e-5, "flat x: C={} cos={}", got[0], t_end.cos());
            assert!((got[1] - (-t_end.sin())).abs() < 1e-5, "flat v: C={} -sin={}", got[1], -t_end.sin());
        }
    }
}

/// End-to-end against *real* builder IR: a `constant -> integrator` ramp built
/// with fastsim's own constructors, exported via `module_from_sim`, code-genned
/// and integrated. dx/dt = c with x(0)=0 gives x(t) = c·t. Validates that
/// `build_plan` consumes the real connection/schedule/BlockId structure (and a
/// real Source block), not just hand-built fixtures.
#[test]
fn generated_ramp_from_real_builder_ir() {
    use fastsim::blocks::constructors::{constant, integrator};
    use fastsim::connection::Connection as RtConn;
    use fastsim::ir::builder::module_from_sim;
    use fastsim::simulation::Simulation;
    use fastsim::utils::portreference::PortReference;
    use std::rc::Rc;

    let c = constant(2.0);
    let integ = integrator(0.0);
    let conn = Rc::new(RtConn::new(
        PortReference::new(c.clone(), None),
        vec![PortReference::new(integ.clone(), None)],
    ));
    let mut sim = Simulation::with_defaults(vec![c.clone(), integ.clone()], vec![conn]);
    sim.run(0.01, false, false); // assemble (resolves ports/schedule)

    let module = module_from_sim(&sim, "ramp");
    let files = generate(&module, &CodegenOptions::default()).expect("generate real-IR system");
    let model = concat_sources(&files);
    assert!(model.contains("void ramp_run("), "{model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping real-IR integration check");
        return;
    };

    let (t_end, dt) = (1.5_f64, 1e-3_f64);
    let main = format!(
        "#include <stdio.h>\nint main(void) {{\n\
         \x20   ramp_t m;\n\
         \x20   ramp_init(&m);\n\
         \x20   ramp_run(&m, {t_end:?}, {dt:?});\n\
         \x20   printf(\"%.17g\", m.x[0]);\n\
         \x20   return 0;\n\
         }}\n"
    );
    match compile_and_run_files(&cc, 12, &main, &files).expect("compile ramp model.c") {
        None => eprintln!("ramp exe would not launch — skipping numeric check"),
        Some(got) => {
            assert_eq!(got.len(), 1);
            // x(t) = 2 * t.
            assert!((got[0] - 2.0 * t_end).abs() < 1e-9, "ramp: C={} expected={}", got[0], 2.0 * t_end);
        }
    }
}

/// A 2-element vector decay with genuine size-2 ports: a vector integrator
/// (dx/dt = u) and an inverting amplifier (-1 elementwise), cross-wired whole
/// port, so dx_i/dt = -x_i and x_i(t) = x0_i·e^-t. Exercises element-level MIMO
/// wiring.
fn vec_decay_module() -> Module {
    let int = Block {
        id: BlockId(0),
        name: "int".into(),
        type_name: "Integrator".into(),
        role: BlockRole::Dynamic,
        ports: io1(2, 2),
        params: vec![],
        state: vec![
            StateVar { id: StateId(0), name: "x0".into(), init: 1.0 },
            StateVar { id: StateId(1), name: "x1".into(), init: 0.5 },
        ],
        memory: vec![],
        regions: Regions {
            alg: Region {
                ops: vec![Op::State { id: StateId(0) }, Op::State { id: StateId(1) }],
                writes: vec![
                    Write::Output { port: 0, elem: 0, src: NodeId(0) },
                    Write::Output { port: 0, elem: 1, src: NodeId(1) },
                ],
            },
            dyn_: Region {
                ops: vec![Op::Input { port: 0, elem: 0 }, Op::Input { port: 0, elem: 1 }],
                writes: vec![
                    Write::StateDeriv { id: StateId(0), src: NodeId(0) },
                    Write::StateDeriv { id: StateId(1), src: NodeId(1) },
                ],
            },
        },
        events: vec![],
    };
    let amp = Block {
        id: BlockId(1),
        name: "amp".into(),
        type_name: "Amplifier".into(),
        role: BlockRole::Algebraic,
        ports: io1(2, 2),
        params: vec![],
        state: vec![],
        memory: vec![],
        regions: Regions {
            alg: Region {
                ops: vec![
                    Op::Input { port: 0, elem: 0 },              // n0
                    Op::Const(-1.0),                             // n1
                    Op::Binary { op: BinOpKind::Mul, a: NodeId(0), b: NodeId(1) }, // n2
                    Op::Input { port: 0, elem: 1 },              // n3
                    Op::Binary { op: BinOpKind::Mul, a: NodeId(3), b: NodeId(1) }, // n4
                ],
                writes: vec![
                    Write::Output { port: 0, elem: 0, src: NodeId(2) },
                    Write::Output { port: 0, elem: 1, src: NodeId(4) },
                ],
            },
            dyn_: Region::default(),
        },
        events: vec![],
    };
    let conn = |id: u32, sb: u32, tb: u32| Connection {
        id: ConnectionId(id),
        src: PortRef { block: BlockId(sb), port: 0, elems: None },
        targets: vec![PortRef { block: BlockId(tb), port: 0, elems: None }],
    };
    let root = Subsystem {
        id: SubsystemId(0),
        name: "root".into(),
        interface: Interface::default(),
        children: vec![Child::Block(int), Child::Block(amp)],
        connections: vec![conn(0, 0, 1), conn(1, 1, 0)],
        schedule: Schedule {
            topo: vec![BlockId(0), BlockId(1)],
            groups: vec![],
            sccs: vec![],
            back_edges: vec![],
        },
    };
    Module::new("vecdecay", root)
}

#[test]
fn generated_vector_decay_mimo() {
    let module = vec_decay_module();
    let files = generate(&module, &CodegenOptions::default()).expect("generate MIMO system");
    let model = concat_sources(&files);
    assert!(model.contains("#define VECDECAY_N_STATE 2"), "{model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping MIMO integration check");
        return;
    };

    let (t_end, dt) = (1.5_f64, 1e-3_f64);
    let main = format!(
        "#include <stdio.h>\nint main(void) {{\n\
         \x20   vecdecay_t m;\n\
         \x20   vecdecay_init(&m);\n\
         \x20   vecdecay_run(&m, {t_end:?}, {dt:?});\n\
         \x20   printf(\"%.17g %.17g\", m.x[0], m.x[1]);\n\
         \x20   return 0;\n\
         }}\n"
    );
    match compile_and_run_files(&cc, 13, &main, &files).expect("compile MIMO model.c") {
        None => eprintln!("MIMO exe would not launch — skipping numeric check"),
        Some(got) => {
            assert_eq!(got.len(), 2);
            let e = (-t_end).exp();
            assert!((got[0] - 1.0 * e).abs() < 1e-6, "x0: C={} expected={}", got[0], e);
            assert!((got[1] - 0.5 * e).abs() < 1e-6, "x1: C={} expected={}", got[1], 0.5 * e);
        }
    }
}

/// Nested subsystem from real builder IR: a `constant(2)` drives a subsystem
/// that wraps an integrator (interface → integrator → interface). After
/// interface flattening the constant feeds the inner integrator directly, so
/// dx/dt = 2 and x(t) = 2·t. Validates one level of subsystem inlining.
#[test]
fn generated_nested_subsystem_from_real_ir() {
    use fastsim::blocks::constructors::{constant, integrator};
    use fastsim::connection::Connection as RtConn;
    use fastsim::ir::builder::module_from_sim;
    use fastsim::simulation::Simulation;
    use fastsim::subsystem::{interface, subsystem};
    use fastsim::utils::portreference::PortReference;
    use std::rc::Rc;

    let conn = |a: &fastsim::blocks::block::BlockRef, b: &fastsim::blocks::block::BlockRef| {
        Rc::new(RtConn::new(
            PortReference::new(a.clone(), None),
            vec![PortReference::new(b.clone(), None)],
        ))
    };

    let iface = interface();
    let integ = integrator(0.0);
    let sub = subsystem(
        vec![iface.clone(), integ.clone()],
        vec![conn(&iface, &integ), conn(&integ, &iface)],
        10,
    )
    .unwrap();
    let cst = constant(2.0);
    let mut sim = Simulation::with_defaults(vec![cst.clone(), sub.clone()], vec![conn(&cst, &sub)]);
    sim.run(0.01, false, false);

    let module = module_from_sim(&sim, "nested");
    let files = generate(&module, &CodegenOptions::default()).expect("generate nested system");
    let model = concat_sources(&files);
    assert!(model.contains("_N_STATE 1"), "{model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping nested-subsystem check");
        return;
    };

    let (t_end, dt) = (1.5_f64, 1e-3_f64);
    let main = format!(
        "#include <stdio.h>\nint main(void) {{\n\
         \x20   nested_t m;\n\
         \x20   nested_init(&m);\n\
         \x20   nested_run(&m, {t_end:?}, {dt:?});\n\
         \x20   printf(\"%.17g\", m.x[0]);\n\
         \x20   return 0;\n\
         }}\n"
    );
    match compile_and_run_files(&cc, 14, &main, &files).expect("compile nested model.c") {
        None => eprintln!("nested exe would not launch — skipping numeric check"),
        Some(got) => {
            assert_eq!(got.len(), 1);
            assert!((got[0] - 2.0 * t_end).abs() < 1e-9, "nested: C={} expected={}", got[0], 2.0 * t_end);
        }
    }
}

/// Deeply nested subsystems from real builder IR: a `constant(2)` drives an
/// outer subsystem that wraps an inner subsystem that wraps an integrator
/// (two interface hops). After recursive flattening the constant feeds the
/// innermost integrator directly, so dx/dt = 2 and x(t) = 2·t. Validates that
/// `flatten` inlines subsystems at depth > 1.
#[test]
fn generated_deeply_nested_subsystem() {
    use fastsim::blocks::constructors::{constant, integrator};
    use fastsim::connection::Connection as RtConn;
    use fastsim::ir::builder::module_from_sim;
    use fastsim::ir::schema::Child;
    use fastsim::simulation::Simulation;
    use fastsim::subsystem::{interface, subsystem};
    use fastsim::utils::portreference::PortReference;
    use std::rc::Rc;

    let conn = |a: &fastsim::blocks::block::BlockRef, b: &fastsim::blocks::block::BlockRef| {
        Rc::new(RtConn::new(
            PortReference::new(a.clone(), None),
            vec![PortReference::new(b.clone(), None)],
        ))
    };

    // Inner subsystem: interface -> integrator -> interface.
    let iface_inner = interface();
    let integ = integrator(0.0);
    let sub_inner = subsystem(
        vec![iface_inner.clone(), integ.clone()],
        vec![conn(&iface_inner, &integ), conn(&integ, &iface_inner)],
        10,
    )
    .unwrap();
    // Outer subsystem: interface -> inner subsystem -> interface.
    let iface_outer = interface();
    let sub_outer = subsystem(
        vec![iface_outer.clone(), sub_inner.clone()],
        vec![conn(&iface_outer, &sub_inner), conn(&sub_inner, &iface_outer)],
        20,
    )
    .unwrap();
    let cst = constant(2.0);
    let mut sim =
        Simulation::with_defaults(vec![cst.clone(), sub_outer.clone()], vec![conn(&cst, &sub_outer)]);
    sim.run(0.01, false, false);

    let module = module_from_sim(&sim, "deep_nested");

    // Confirm the IR genuinely nests two levels deep (else the recursion is not
    // exercised): root has a Subsystem child that itself has a Subsystem child.
    let outer = module
        .root
        .children
        .iter()
        .find_map(|c| match c {
            Child::Subsystem(s) => Some(s),
            _ => None,
        })
        .expect("root has a subsystem child");
    assert!(
        outer.children.iter().any(|c| matches!(c, Child::Subsystem(_))),
        "outer subsystem should itself contain a subsystem (depth 2)"
    );

    let files = generate(&module, &CodegenOptions::default()).expect("generate deeply nested system");
    let model = concat_sources(&files);
    assert!(model.contains("_N_STATE 1"), "{model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping deep-nesting check");
        return;
    };

    let (t_end, dt) = (1.5_f64, 1e-3_f64);
    let main = format!(
        "#include <stdio.h>\nint main(void) {{\n\
         \x20   deep_nested_t m;\n\
         \x20   deep_nested_init(&m);\n\
         \x20   deep_nested_run(&m, {t_end:?}, {dt:?});\n\
         \x20   printf(\"%.17g\", m.x[0]);\n\
         \x20   return 0;\n\
         }}\n"
    );
    match compile_and_run_files(&cc, 15, &main, &files).expect("compile deeply nested model.c") {
        None => eprintln!("deep-nested exe would not launch — skipping numeric check"),
        Some(got) => {
            assert_eq!(got.len(), 1);
            assert!((got[0] - 2.0 * t_end).abs() < 1e-9, "deep-nested: C={} expected={}", got[0], 2.0 * t_end);
        }
    }
}

/// Discrete memory read: a block carries a memory slot (init 3.0) and
/// integrates it, dx/dt = mem[0] = 3, so x(t) = 3·t. Verifies the memory array,
/// its initialization, and `Op::Memory` addressing (no events yet).
#[test]
fn generated_memory_read() {
    let blk = Block {
        id: BlockId(0),
        name: "memconst".into(),
        type_name: "MemConst".into(),
        role: BlockRole::Dynamic,
        ports: io(0, 1),
        params: vec![],
        state: vec![StateVar { id: StateId(0), name: "x".into(), init: 0.0 }],
        memory: vec![MemorySlot { id: MemorySlotId(0), name: "c".into(), size: 1, init: vec![3.0] }],
        regions: Regions {
            alg: Region {
                ops: vec![Op::State { id: StateId(0) }],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
            },
            dyn_: Region {
                ops: vec![Op::Memory { slot: MemorySlotId(0), offset: 0 }],
                writes: vec![Write::StateDeriv { id: StateId(0), src: NodeId(0) }],
            },
        },
        events: vec![],
    };
    let root = Subsystem {
        id: SubsystemId(0),
        name: "root".into(),
        interface: Interface::default(),
        children: vec![Child::Block(blk)],
        connections: vec![],
        schedule: Schedule { topo: vec![BlockId(0)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    let module = Module::new("memconst", root);

    let files = generate(&module, &CodegenOptions::default()).expect("generate memory system");
    let model = concat_sources(&files);
    assert!(model.contains("mem[1];"), "{model}");
    // Memory lives in the per-instance model_t struct, not file-static storage.
    assert!(!model.contains("static double mem"), "mem must be in the struct: {model}");
    assert!(model.contains("} memconst_t;"), "{model}");
    assert!(model.contains("m->mem[0] = 3.0;"), "{model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping memory-read check");
        return;
    };

    let (t_end, dt) = (1.5_f64, 1e-3_f64);
    let main = format!(
        "#include <stdio.h>\nint main(void) {{\n\
         \x20   memconst_t m;\n\
         \x20   memconst_init(&m);\n\
         \x20   memconst_run(&m, {t_end:?}, {dt:?});\n\
         \x20   printf(\"%.17g\", m.x[0]);\n\
         \x20   return 0;\n\
         }}\n"
    );
    match compile_and_run_files(&cc, 15, &main, &files).expect("compile memory model.c") {
        None => eprintln!("memory exe would not launch — skipping numeric check"),
        Some(got) => {
            assert_eq!(got.len(), 1);
            assert!((got[0] - 3.0 * t_end).abs() < 1e-9, "mem read: C={} expected={}", got[0], 3.0 * t_end);
        }
    }
}

/// Discrete periodic event: a counter holds `c` in memory and a periodic event
/// (period 0.1) does `c' = c + 1`; its output feeds an integrator, so
/// dx/dt = c(t) is a staircase. The event fires at the first step whose
/// accumulated time reaches `k·0.1` — bit-identical to the runtime scheduler.
/// At dt = 0.01, `0.01` summed ten times is `0.09999999999999999`, one ULP below
/// `0.1`, so the k-th tick lands one step late (t ≈ k·0.1 + 0.01) exactly as the
/// reference runtime does. That shifts the staircase by one step versus the ideal
/// 5.5, giving 5.49 — the point being SiL parity with the runtime, not the ideal.
#[test]
fn generated_periodic_event_counter() {
    let counter = Block {
        id: BlockId(0),
        name: "counter".into(),
        type_name: "Counter".into(),
        role: BlockRole::Algebraic,
        ports: io(0, 1),
        params: vec![],
        state: vec![],
        memory: vec![MemorySlot { id: MemorySlotId(0), name: "c".into(), size: 1, init: vec![0.0] }],
        regions: Regions {
            alg: Region {
                ops: vec![Op::Memory { slot: MemorySlotId(0), offset: 0 }],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
            },
            dyn_: Region::default(),
        },
        events: vec![Event {
            id: EventId(0),
            kind: EventKind::Schedule { times: ScheduleTimes::Periodic { period: 0.1, phase: 0.0 } },
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
    let integ = Block {
        id: BlockId(1),
        name: "integ".into(),
        type_name: "Integrator".into(),
        role: BlockRole::Dynamic,
        ports: io(1, 1),
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
    };
    let root = Subsystem {
        id: SubsystemId(0),
        name: "root".into(),
        interface: Interface::default(),
        children: vec![Child::Block(counter), Child::Block(integ)],
        connections: vec![Connection {
            id: ConnectionId(0),
            src: PortRef { block: BlockId(0), port: 0, elems: None },
            targets: vec![PortRef { block: BlockId(1), port: 0, elems: None }],
        }],
        schedule: Schedule { topo: vec![BlockId(0), BlockId(1)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    let module = Module::new("counter", root);

    let files = generate(&module, &CodegenOptions::default()).expect("generate event system");
    let model = concat_sources(&files);
    assert!(model.contains("void counter_handle_events("), "{model}");
    assert!(!model.contains("static double mem"), "mem must be in the struct: {model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping event check");
        return;
    };

    let (t_end, dt) = (1.0_f64, 0.01_f64);
    let main = format!(
        "#include <stdio.h>\nint main(void) {{\n\
         \x20   counter_t m;\n\
         \x20   counter_init(&m);\n\
         \x20   counter_run(&m, {t_end:?}, {dt:?});\n\
         \x20   printf(\"%.17g\", m.x[0]);\n\
         \x20   return 0;\n\
         }}\n"
    );
    match compile_and_run_files(&cc, 16, &main, &files).expect("compile event model.c") {
        None => eprintln!("event exe would not launch — skipping numeric check"),
        Some(got) => {
            assert_eq!(got.len(), 1);
            assert!((got[0] - 5.49).abs() < 1e-6, "event counter: C={} expected=5.49", got[0]);
        }
    }
}

/// Pure-discrete model (no continuous state): a standalone counter holds `c` in
/// memory and a periodic event (period 0.1) does `c' = c + 1`; its alg output
/// reads `c`. With `n_state == 0` the emitter omits `model_deriv` and the stage kernel
/// and emits a discrete `model_run` that just advances time and fires the event
/// at each boundary. Over [0, 0.95] the event fires at t = 0, 0.1, ..., 0.9
/// (10 times), so `c = 10`.
#[test]
fn generated_pure_discrete_counter() {
    let counter = Block {
        id: BlockId(0),
        name: "counter".into(),
        type_name: "Counter".into(),
        role: BlockRole::Algebraic,
        ports: io(0, 1),
        params: vec![],
        state: vec![],
        memory: vec![MemorySlot { id: MemorySlotId(0), name: "c".into(), size: 1, init: vec![0.0] }],
        regions: Regions {
            alg: Region {
                ops: vec![Op::Memory { slot: MemorySlotId(0), offset: 0 }],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
            },
            dyn_: Region::default(),
        },
        events: vec![Event {
            id: EventId(0),
            kind: EventKind::Schedule { times: ScheduleTimes::Periodic { period: 0.1, phase: 0.0 } },
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
        children: vec![Child::Block(counter)],
        connections: vec![],
        schedule: Schedule { topo: vec![BlockId(0)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    let module = Module::new("discrete_counter", root);

    let files = generate(&module, &CodegenOptions::default()).expect("generate pure-discrete system");
    let model = concat_sources(&files);
    assert!(model.contains("_N_STATE 0"), "{model}");
    // Pure-discrete: the model_t carries memory + event counters, no states.
    assert!(model.contains("void discrete_counter_init(discrete_counter_t * restrict m)"), "{model}");
    // No continuous integration is emitted.
    assert!(!model.contains("model_deriv"), "{model}");
    assert!(!model.contains("fs_stages_step"), "{model}");
    assert!(model.contains("void discrete_counter_run(discrete_counter_t * restrict m, double t_end,"), "{model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping pure-discrete check");
        return;
    };

    let (t_end, dt) = (0.95_f64, 0.01_f64);
    let main = format!(
        "#include <stdio.h>\nint main(void) {{\n\
         \x20   discrete_counter_t m;\n\
         \x20   discrete_counter_init(&m);\n\
         \x20   discrete_counter_run(&m, {t_end:?}, {dt:?});\n\
         \x20   printf(\"%.17g\", m.sig[0]);\n\
         \x20   return 0;\n\
         }}\n"
    );
    match compile_and_run_files(&cc, 17, &main, &files).expect("compile pure-discrete model.c") {
        None => eprintln!("pure-discrete exe would not launch — skipping numeric check"),
        Some(got) => {
            assert_eq!(got.len(), 1);
            assert!((got[0] - 10.0).abs() < 1e-9, "pure-discrete counter: C={} expected=10", got[0]);
        }
    }
}

/// Fixed-schedule event: a counter increments on a fixed list of times
/// (0.2, 0.5, 0.8). All three fall in [0, 1.0], so `c = 3`.
#[test]
fn generated_fixed_schedule_counter() {
    let counter = Block {
        id: BlockId(0),
        name: "counter".into(),
        type_name: "Counter".into(),
        role: BlockRole::Algebraic,
        ports: io(0, 1),
        params: vec![],
        state: vec![],
        memory: vec![MemorySlot { id: MemorySlotId(0), name: "c".into(), size: 1, init: vec![0.0] }],
        regions: Regions {
            alg: Region {
                ops: vec![Op::Memory { slot: MemorySlotId(0), offset: 0 }],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
            },
            dyn_: Region::default(),
        },
        events: vec![Event {
            id: EventId(0),
            kind: EventKind::Schedule { times: ScheduleTimes::Fixed(vec![0.2, 0.5, 0.8]) },
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
        children: vec![Child::Block(counter)],
        connections: vec![],
        schedule: Schedule { topo: vec![BlockId(0)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    let module = Module::new("fixed_counter", root);
    let files = generate(&module, &CodegenOptions::default()).expect("generate fixed-schedule system");
    let model = concat_sources(&files);
    assert!(model.contains("static const double times_0_0[] = { 0.2, 0.5, 0.8 };"), "{model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping fixed-schedule check");
        return;
    };
    let main = "#include <stdio.h>\nint main(void) {\n\
         \x20   fixed_counter_t m;\n\
         \x20   fixed_counter_init(&m);\n\
         \x20   fixed_counter_run(&m, 1.0, 0.01);\n\
         \x20   printf(\"%.17g\", m.sig[0]);\n\
         \x20   return 0;\n}\n";
    match compile_and_run_files(&cc, 18, main, &files).expect("compile fixed-schedule model.c") {
        None => eprintln!("fixed-schedule exe would not launch — skipping numeric check"),
        Some(got) => assert!((got[0] - 3.0).abs() < 1e-9, "fixed schedule: C={} expected=3", got[0]),
    }
}

/// Zero-cross event: `x' = -1`, `x(0) = 1` (so `x = 1 - t`), guard `= x`. On the
/// falling crossing at `t = 1`, a memory counter increments once, so `c = 1`.
#[test]
fn generated_zerocross_event() {
    let blk = Block {
        id: BlockId(0),
        name: "zc".into(),
        type_name: "ZeroCross".into(),
        role: BlockRole::Dynamic,
        ports: io(0, 1),
        params: vec![],
        state: vec![StateVar { id: StateId(0), name: "x".into(), init: 1.0 }],
        memory: vec![MemorySlot { id: MemorySlotId(0), name: "c".into(), size: 1, init: vec![0.0] }],
        regions: Regions {
            alg: Region {
                ops: vec![Op::Memory { slot: MemorySlotId(0), offset: 0 }],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
            },
            dyn_: Region {
                ops: vec![Op::Const(-1.0)],
                writes: vec![Write::StateDeriv { id: StateId(0), src: NodeId(0) }],
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
        children: vec![Child::Block(blk)],
        connections: vec![],
        schedule: Schedule { topo: vec![BlockId(0)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    let module = Module::new("zerocross", root);
    let files = generate(&module, &CodegenOptions::default()).expect("generate zero-cross system");
    let model = concat_sources(&files);
    assert!(model.contains("guard_0_0("), "{model}");
    assert!(model.contains("cur_0_0 < 0 && m->prev_0_0 >= 0"), "{model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping zero-cross check");
        return;
    };
    let main = "#include <stdio.h>\nint main(void) {\n\
         \x20   zerocross_t m;\n\
         \x20   zerocross_init(&m);\n\
         \x20   zerocross_run(&m, 1.5, 0.001);\n\
         \x20   printf(\"%.17g\", m.sig[0]);\n\
         \x20   return 0;\n}\n";
    match compile_and_run_files(&cc, 19, main, &files).expect("compile zero-cross model.c") {
        None => eprintln!("zero-cross exe would not launch — skipping numeric check"),
        Some(got) => assert!((got[0] - 1.0).abs() < 1e-9, "zero-cross: C={} expected=1", got[0]),
    }
}

/// Condition event: `x' = 1`, `x(0) = 0` (so `x = t`), guard `= (x > 0.5)`. The
/// guard rises from 0 to 1 once (at `t = 0.5`); the event fires on that rising
/// edge, so the memory counter `c = 1`.
#[test]
fn generated_condition_event() {
    let blk = Block {
        id: BlockId(0),
        name: "cond".into(),
        type_name: "Condition".into(),
        role: BlockRole::Dynamic,
        ports: io(0, 1),
        params: vec![],
        state: vec![StateVar { id: StateId(0), name: "x".into(), init: 0.0 }],
        memory: vec![MemorySlot { id: MemorySlotId(0), name: "c".into(), size: 1, init: vec![0.0] }],
        regions: Regions {
            alg: Region {
                ops: vec![Op::Memory { slot: MemorySlotId(0), offset: 0 }],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
            },
            dyn_: Region {
                ops: vec![Op::Const(1.0)],
                writes: vec![Write::StateDeriv { id: StateId(0), src: NodeId(0) }],
            },
        },
        events: vec![Event {
            id: EventId(0),
            kind: EventKind::Condition {
                guard: Region {
                    ops: vec![
                        Op::State { id: StateId(0) },
                        Op::Const(0.5),
                        Op::Cmp { op: CmpKind::Gt, a: NodeId(0), b: NodeId(1) },
                    ],
                    writes: vec![],
                },
            },
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
        children: vec![Child::Block(blk)],
        connections: vec![],
        schedule: Schedule { topo: vec![BlockId(0)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    let module = Module::new("condition", root);
    let files = generate(&module, &CodegenOptions::default()).expect("generate condition system");
    let model = concat_sources(&files);
    assert!(model.contains("guard_0_0("), "{model}");
    assert!(model.contains("cur_0_0 != 0 && (!m->init_0_0 || m->prev_0_0 == 0)"), "{model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping condition check");
        return;
    };
    let main = "#include <stdio.h>\nint main(void) {\n\
         \x20   condition_t m;\n\
         \x20   condition_init(&m);\n\
         \x20   condition_run(&m, 1.0, 0.001);\n\
         \x20   printf(\"%.17g\", m.sig[0]);\n\
         \x20   return 0;\n}\n";
    match compile_and_run_files(&cc, 20, main, &files).expect("compile condition model.c") {
        None => eprintln!("condition exe would not launch — skipping numeric check"),
        Some(got) => assert!((got[0] - 1.0).abs() < 1e-9, "condition: C={} expected=1", got[0]),
    }
}

#[test]
fn generated_model_c_integrates_correctly() {
    let module = decay_module();
    let dyn_region = match &module.root.children[0] {
        Child::Block(b) => b.regions.dyn_.clone(),
        _ => unreachable!(),
    };

    // The model.c itself emits regardless of a compiler being present.
    let files = generate(&module, &CodegenOptions::default()).expect("generate");
    let model = concat_sources(&files);
    assert!(model.contains("void decay_run("), "{model}");
    assert!(model.contains("_N_STATE 1"), "{model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping model.c integration check");
        return;
    };

    let (t_end, dt) = (2.0, 1e-3);
    let main = format!(
        "#include <stdio.h>\nint main(void) {{\n\
         \x20   decay_t m;\n\
         \x20   decay_init(&m);\n\
         \x20   decay_run(&m, {t_end:?}, {dt:?});\n\
         \x20   printf(\"%.17g\", m.x[0]);\n\
         \x20   return 0;\n\
         }}\n"
    );
    match compile_and_run_files(&cc, 10, &main, &files).expect("compile model.c") {
        None => eprintln!("model.c exe would not launch — skipping numeric check"),
        Some(got) => {
            assert_eq!(got.len(), 1, "expected one state output");
            let c_final = got[0];
            // Matches the IR-RK4 reference (same method) tightly...
            let ref_final = ir_rk4(&dyn_region, 1.0, t_end, dt);
            assert!((c_final - ref_final).abs() < 1e-9, "C={c_final} ir_rk4={ref_final}");
            // ...and the analytic solution e^-t within RK4 truncation error.
            let analytic = (-t_end).exp();
            assert!((c_final - analytic).abs() < 1e-6, "C={c_final} e^-t={analytic}");
        }
    }
}

/// The public `<name>_step` (the RTOS/ISR entry point) composes EXACTLY to
/// `<name>_run`: N step calls of `dt` produce bit-identical state, memory and
/// time to one `run(t0 + N·dt, dt)` — including the initial-event pass
/// (`fs_started`) and per-step event handling. Covered for a continuous model
/// with a phase-0 periodic event and for a pure-discrete model; an adaptive
/// tableau driven at a fixed rate through `step` is checked against the
/// analytic solution (no `run` parity there — `run` adapts its steps).
#[test]
fn step_composes_to_run() {
    // -- continuous + periodic event (counter feeding an integrator) --
    let counter = Block {
        id: BlockId(0),
        name: "cnt".into(),
        type_name: "Counter".into(),
        role: BlockRole::Algebraic,
        ports: io(0, 1),
        params: vec![],
        state: vec![],
        memory: vec![MemorySlot { id: MemorySlotId(0), name: "c".into(), size: 1, init: vec![0.0] }],
        regions: Regions {
            alg: Region {
                ops: vec![Op::Memory { slot: MemorySlotId(0), offset: 0 }],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
            },
            dyn_: Region::default(),
        },
        events: vec![Event {
            id: EventId(0),
            kind: EventKind::Schedule { times: ScheduleTimes::Periodic { period: 0.1, phase: 0.0 } },
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
    let integ = integrator(1, "integ", 0.0);
    let root = Subsystem {
        id: SubsystemId(0),
        name: "root".into(),
        interface: Interface::default(),
        children: vec![Child::Block(counter), Child::Block(integ)],
        connections: vec![Connection {
            id: ConnectionId(0),
            src: PortRef { block: BlockId(0), port: 0, elems: None },
            targets: vec![PortRef { block: BlockId(1), port: 0, elems: None }],
        }],
        schedule: Schedule { topo: vec![BlockId(0), BlockId(1)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    let module = Module::new("stepper", root);
    let files = generate(&module, &CodegenOptions::default()).expect("generate stepper");
    let model = concat_sources(&files);
    assert!(model.contains("void stepper_step(stepper_t"), "public step missing: {model}");
    assert!(model.contains("fs_started"), "initial-event guard missing: {model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping step-parity check");
        return;
    };

    // 100 steps of 0.01 vs run(1.0): states, memory and time must match
    // BIT-EXACTLY (%.17g round-trips f64).
    let main = "#include <stdio.h>\nint main(void) {\n\
         \x20   stepper_t a, b;\n\
         \x20   stepper_init(&a);\n\
         \x20   stepper_init(&b);\n\
         \x20   stepper_run(&a, 1.0, 0.01);\n\
         \x20   for (int k = 0; k < 100; k++) stepper_step(&b, 0.01);\n\
         \x20   printf(\"%.17g %.17g\\n\", a.x[0], b.x[0]);\n\
         \x20   printf(\"%.17g %.17g\\n\", a.mem[0], b.mem[0]);\n\
         \x20   printf(\"%.17g %.17g\\n\", a.time, b.time);\n\
         \x20   printf(\"%.17g %.17g\\n\", a.sig[0], b.sig[0]);\n\
         \x20   return 0;\n\
         }\n";
    match compile_and_run_files(&cc, 60, main, &files).expect("compile step-parity model") {
        None => eprintln!("step-parity exe would not launch — skipping numeric check"),
        Some(got) => {
            assert_eq!(got.len(), 8, "got {got:?}");
            for pair in got.chunks(2) {
                assert!(
                    pair[0].to_bits() == pair[1].to_bits(),
                    "step composition diverged from run: run={} step={} (all: {got:?})",
                    pair[0], pair[1]
                );
            }
            // Sanity: the phase-0 event fired 11 times over [0, 1].
            assert!((got[2] - 11.0).abs() < 1e-12, "counter after 1s: {}", got[2]);
        }
    }

    // -- adaptive tableau, fixed-rate stepping: x'' = -x, x(0)=1 -> cos t --
    let module = oscillator_module();
    let opts = CodegenOptions {
        solver: fastsim::codegen::SolverChoice::by_name("RKBS32").expect("adaptive tableau"),
        ..Default::default()
    };
    let files = generate(&module, &opts).expect("generate adaptive stepper");
    let model = concat_sources(&files);
    assert!(model.contains("void oscillator_step(oscillator_t"), "{model}");
    let main = "#include <stdio.h>\nint main(void) {\n\
         \x20   oscillator_t m;\n\
         \x20   oscillator_init(&m);\n\
         \x20   for (int k = 0; k < 2000; k++) oscillator_step(&m, 1e-3);\n\
         \x20   printf(\"%.17g %.17g\", m.x[0], m.time);\n\
         \x20   return 0;\n\
         }\n";
    match compile_and_run_files(&cc, 61, main, &files).expect("compile adaptive stepper") {
        None => eprintln!("adaptive-step exe would not launch — skipping numeric check"),
        Some(got) => {
            assert_eq!(got.len(), 2);
            assert!((got[0] - 2.0_f64.cos()).abs() < 1e-5, "x(2) = {} vs cos 2 = {}", got[0], 2.0_f64.cos());
            assert!((got[1] - 2.0).abs() < 1e-9, "time after 2000 steps: {}", got[1]);
        }
    }
}

/// `scaffold=true` additionally emits `CMakeLists.txt` + an editable
/// `<base>_main.c` demo driver (CSV over `<name>_step`); the default file set
/// stays untouched. The demo driver must actually compile and produce a
/// plausible CSV trajectory.
#[test]
fn scaffold_emits_buildable_demo() {
    let module = oscillator_module();
    let plain = generate(&module, &CodegenOptions::default()).expect("generate plain");
    assert!(
        !plain.iter().any(|f| f.name == "CMakeLists.txt" || f.name.ends_with("_main.c")),
        "scaffold files must be opt-in"
    );

    let opts = CodegenOptions { scaffold: true, ..Default::default() };
    let files = generate(&module, &opts).expect("generate scaffold");
    let names: Vec<&str> = files.iter().map(|f| f.name.as_str()).collect();
    assert!(names.contains(&"oscillator_main.c"), "missing demo driver: {names:?}");
    assert!(names.contains(&"CMakeLists.txt"), "missing CMakeLists: {names:?}");

    let src_of = |n: &str| files.iter().find(|f| f.name == n).map(|f| f.contents.as_str()).unwrap();
    let cmake = src_of("CMakeLists.txt");
    assert!(cmake.contains("project(oscillator C)"), "{cmake}");
    assert!(cmake.contains("oscillator.c"), "library sources listed: {cmake}");
    assert!(cmake.contains("add_executable(oscillator_demo oscillator_main.c)"), "{cmake}");
    let main_c = src_of("oscillator_main.c");
    assert!(main_c.contains("oscillator_step(&m, FASTSIM_DT)"), "{main_c}");
    assert!(main_c.contains("EDITABLE"), "scaffold must mark itself editable: {main_c}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping scaffold build check");
        return;
    };
    // Compile the demo exactly as CMake would (model sources + demo main) and
    // check the CSV: header + one row per step + the initial row, time
    // reaching the (shortened) duration.
    let dir = std::env::temp_dir().join(format!("fastsim_scaffold_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for f in &files {
        std::fs::write(dir.join(&f.name), &f.contents).unwrap();
    }
    let out = fastsim::codegen::verify::cc_command_in(&cc, &dir)
        .args([
            "oscillator.c", "oscillator_main.c", "-DFASTSIM_DURATION=0.5",
            "-DFASTSIM_DT=1e-2", "-std=c99", "-O0", "-o", "demo.exe", "-lm",
        ])
        .output()
        .expect("spawn cc");
    assert!(out.status.success(), "scaffold compile failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let run = match std::process::Command::new(dir.join("demo.exe")).output() {
        Ok(r) if r.status.success() => r,
        _ => {
            eprintln!("scaffold demo would not launch — skipping CSV check");
            return;
        }
    };
    let stdout = String::from_utf8_lossy(&run.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert!(lines[0].starts_with("time,"), "CSV header: {}", lines[0]);
    assert_eq!(lines.len(), 1 + 1 + 50, "header + initial row + 50 steps: {}", lines.len());
    let last_t: f64 = lines.last().unwrap().split(',').next().unwrap().parse().unwrap();
    assert!((last_t - 0.5).abs() < 1e-12, "final time {last_t}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// `trace=true` emits `<base>_trace.json`: block → function map whose
/// file/line references point at REAL definitions in the emitted text,
/// signal ids matching the layout, and plausible static metrics.
#[test]
fn trace_map_points_at_real_definitions() {
    let module = oscillator_module();
    let opts = CodegenOptions {
        trace: true,
        structure: Structure::Hierarchical,
        ..Default::default()
    };
    let files = generate(&module, &opts).expect("generate with trace");
    let trace_src = &files
        .iter()
        .find(|f| f.name == "oscillator_trace.json")
        .expect("trace file emitted")
        .contents;
    let doc: serde_json::Value = serde_json::from_str(trace_src).expect("valid JSON");

    assert_eq!(doc["model"], "oscillator");
    assert_eq!(doc["metrics"]["n_state"], 2);
    assert_eq!(doc["metrics"]["solver_stages"], 4); // RK4
    // RK4 kernel: x0[2] + k[4][2] doubles = 10 * 8 bytes.
    assert_eq!(doc["metrics"]["integrator_stack_bytes"], 80);
    assert!(doc["metrics"]["per_step_ops_estimate"].as_u64().unwrap() > 0);

    // Every function reference resolves to a line that REALLY defines it.
    let line_of = |file: &str, line: usize| -> &str {
        files
            .iter()
            .find(|f| f.name == file)
            .map(|f| f.contents.lines().nth(line - 1).unwrap())
            .unwrap()
    };
    let mut checked = 0;
    let mut check_def = |d: &serde_json::Value| {
        let (sym, file, line) = (
            d["symbol"].as_str().unwrap(),
            d["file"].as_str().unwrap(),
            d["line"].as_u64().unwrap() as usize,
        );
        let text = line_of(file, line);
        assert!(
            text.contains(&format!("{sym}(")) && !text.starts_with(' '),
            "{sym}: {file}:{line} is not a definition: {text}"
        );
        checked += 1;
    };
    for ep in doc["entry_points"].as_array().unwrap() {
        check_def(ep);
    }
    for b in doc["blocks"].as_array().unwrap() {
        for f in b["functions"].as_array().unwrap() {
            check_def(f);
        }
    }
    assert!(checked >= 6, "too few resolved definitions: {checked}");

    // Signal map covers states + outputs + params with unique ids.
    let signals = doc["signals"].as_array().unwrap();
    assert!(signals.iter().any(|s| s["kind"] == "State"));
    assert!(signals.iter().any(|s| s["kind"] == "Output" || s["kind"] == "Local"));
    let mut ids: Vec<u64> = signals.iter().map(|s| s["signal_id"].as_u64().unwrap()).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), signals.len(), "signal ids must be unique");

    // Default emission stays trace-free.
    let plain = generate(&module, &CodegenOptions::default()).expect("plain");
    assert!(!plain.iter().any(|f| f.name.ends_with("_trace.json")));
}

/// The struct-layout calculator behind the A2L/trace offsets MUST mirror the
/// emitted `<name>_t` field order and types exactly. Parse the emitted header
/// and pin the two against each other, on a model that exercises every field
/// class: states, signals, memory, a periodic event (real + int bookkeeping)
/// and an adaptive tableau (`fs_h`).
#[test]
fn struct_layout_matches_emitted_header() {
    let counter = Block {
        id: BlockId(0),
        name: "cnt".into(),
        type_name: "Counter".into(),
        role: BlockRole::Algebraic,
        ports: io(0, 1),
        params: vec![],
        state: vec![],
        memory: vec![MemorySlot { id: MemorySlotId(0), name: "c".into(), size: 1, init: vec![0.0] }],
        regions: Regions {
            alg: Region {
                ops: vec![Op::Memory { slot: MemorySlotId(0), offset: 0 }],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
            },
            dyn_: Region::default(),
        },
        events: vec![Event {
            id: EventId(0),
            kind: EventKind::Schedule { times: ScheduleTimes::Periodic { period: 0.1, phase: 0.0 } },
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
    let integ = integrator(1, "integ", 0.0);
    let root = Subsystem {
        id: SubsystemId(0),
        name: "root".into(),
        interface: Interface::default(),
        children: vec![Child::Block(counter), Child::Block(integ)],
        connections: vec![Connection {
            id: ConnectionId(0),
            src: PortRef { block: BlockId(0), port: 0, elems: None },
            targets: vec![PortRef { block: BlockId(1), port: 0, elems: None }],
        }],
        schedule: Schedule { topo: vec![BlockId(0), BlockId(1)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    let module = Module::new("laid", root);
    let opts = CodegenOptions {
        trace: true,
        solver: fastsim::codegen::SolverChoice::by_name("RKBS32").unwrap(), // adaptive -> fs_h
        ..Default::default()
    };
    let files = generate(&module, &opts).expect("generate");

    // Parse the emitted struct body: `type name;` / `type name[count];`.
    let header = &files.iter().find(|f| f.name == "laid.h").unwrap().contents;
    let body = header
        .split("typedef struct {").nth(1).expect("struct start")
        .split("} laid_t;").next().expect("struct end");
    let mut parsed: Vec<(String, String, usize)> = Vec::new();
    for line in body.lines() {
        let decl = line.trim().split(';').next().unwrap_or("").trim();
        if decl.is_empty() { continue; }
        let mut it = decl.split_whitespace();
        let (Some(ty), Some(rest)) = (it.next(), it.next()) else { continue };
        let (nm, count) = match rest.split_once('[') {
            Some((n, c)) => (n, c.trim_end_matches(']').parse::<usize>().unwrap()),
            None => (rest, 1),
        };
        parsed.push((ty.to_string(), nm.to_string(), count));
    }

    // The trace's struct_layout is the calculator's output.
    let trace: serde_json::Value = serde_json::from_str(
        &files.iter().find(|f| f.name == "laid_trace.json").unwrap().contents,
    ).unwrap();
    let calc: Vec<(String, String, usize)> = trace["struct_layout"]
        .as_array().unwrap()
        .iter()
        .map(|f| (
            f["ctype"].as_str().unwrap().to_string(),
            f["name"].as_str().unwrap().to_string(),
            f["count"].as_u64().unwrap() as usize,
        ))
        .collect();
    assert_eq!(parsed, calc, "layout calculator diverged from the emitted struct");

    // fs_h must land 8-aligned after the int bookkeeping (padding accounted).
    let fs_h = trace["struct_layout"].as_array().unwrap().iter()
        .find(|f| f["name"] == "fs_h").expect("fs_h present");
    assert_eq!(fs_h["offset"].as_u64().unwrap() % 8, 0);
    assert_eq!(trace["metrics"]["model_struct_bytes_aligned64"].as_u64().unwrap() % 8, 0);
}

/// `a2l=true` emits `<base>.a2l` whose SYMBOL_LINK offsets agree with the
/// trace map's struct layout — measurement and calibration tooling see the
/// same addresses the C compiler will produce.
#[test]
fn a2l_offsets_agree_with_struct_layout() {
    // Decay with a LIVE gain parameter: dx/dt = -k*x, k as IR Param (the test
    // helpers bake gains as constants, so build the param-carrying amp here).
    let integ = integrator(0, "integ", 1.0);
    let amp = Block {
        id: BlockId(1),
        name: "amp".into(),
        type_name: "Amplifier".into(),
        role: BlockRole::Algebraic,
        ports: io(1, 1),
        params: vec![Param {
            id: ParamId(0),
            name: "gain".into(),
            value: ParamValue::Scalar(-1.0),
        }],
        state: vec![],
        memory: vec![],
        regions: Regions {
            alg: Region {
                ops: vec![
                    Op::Input { port: 0, elem: 0 },
                    Op::Param { id: ParamId(0) },
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
        name: "root".into(),
        interface: Interface::default(),
        children: vec![Child::Block(integ), Child::Block(amp)],
        connections: vec![
            Connection {
                id: ConnectionId(0),
                src: PortRef { block: BlockId(0), port: 0, elems: None },
                targets: vec![PortRef { block: BlockId(1), port: 0, elems: None }],
            },
            Connection {
                id: ConnectionId(1),
                src: PortRef { block: BlockId(1), port: 0, elems: None },
                targets: vec![PortRef { block: BlockId(0), port: 0, elems: None }],
            },
        ],
        schedule: Schedule { topo: vec![BlockId(0), BlockId(1)], groups: vec![], sccs: vec![], back_edges: vec![] },
    };
    let module = Module::new("oscillator", root);
    let opts = CodegenOptions { a2l: true, trace: true, ..Default::default() };
    let files = generate(&module, &opts).expect("generate");
    let a2l = &files.iter().find(|f| f.name == "oscillator.a2l").expect("a2l emitted").contents;
    assert!(a2l.contains("ASAP2_VERSION 1 71"), "{a2l}");
    assert!(a2l.contains("/begin PROJECT oscillator"), "{a2l}");

    let trace: serde_json::Value = serde_json::from_str(
        &files.iter().find(|f| f.name == "oscillator_trace.json").unwrap().contents,
    ).unwrap();
    let layout = trace["struct_layout"].as_array().unwrap();
    let off_of = |nm: &str| layout.iter().find(|f| f["name"] == nm).unwrap()["offset"].as_u64().unwrap();
    let real = 8u64;

    // time at its struct offset; first state at x[]'s offset; first param at p[]'s.
    let has_link = |sym_off: u64| a2l.contains(&format!("SYMBOL_LINK \"oscillator\" {sym_off}"));
    assert!(has_link(off_of("time")), "time offset missing:\n{a2l}");
    assert!(has_link(off_of("x")), "x[0] offset missing:\n{a2l}");
    let _ = real;
    assert!(has_link(off_of("p")), "p[0] offset missing:\n{a2l}");
    assert!(a2l.contains("/begin CHARACTERISTIC amp_p0"), "param name:\n{a2l}");

    // Structure: one CHARACTERISTIC per param, MEASUREMENTs for time+states+sigs.
    let n_char = a2l.matches("/begin CHARACTERISTIC").count();
    let n_meas = a2l.matches("/begin MEASUREMENT").count();
    let n_state = trace["metrics"]["n_state"].as_u64().unwrap() as usize;
    let n_sig = trace["metrics"]["n_sig"].as_u64().unwrap() as usize;
    let n_param = trace["metrics"]["n_param"].as_u64().unwrap() as usize;
    assert_eq!(n_char, n_param, "{a2l}");
    assert_eq!(n_meas, 1 + n_state + n_sig, "{a2l}");

    // Default emission stays a2l-free.
    let plain = generate(&module, &CodegenOptions::default()).expect("plain");
    assert!(!plain.iter().any(|f| f.name.ends_with(".a2l")));
}

/// End-to-end fixed point: decay dx/dt = -x generated as Q16.16 integer C
/// (RK4, dt = 1/1024 — exactly representable in Q16.16), compiled and run;
/// x(1) must land on e^-1 within the quantization budget. Also pins the
/// emitted shapes: int32 storage, Q-scaled literals, int64 arithmetic, the
/// boundary conversion macros.
#[test]
fn fixed_point_decay_matches_reference() {
    use fastsim::codegen::Numeric;
    use fastsim::blocks::constructors::{amplifier as rt_amp, integrator as rt_integ};
    use fastsim::connection::Connection as RtConn;
    use fastsim::ir::builder::module_from_sim;
    use fastsim::simulation::Simulation as RtSim;
    use fastsim::utils::portreference::PortReference;
    use std::rc::Rc;

    let conn = |a: &fastsim::blocks::block::BlockRef, b: &fastsim::blocks::block::BlockRef| {
        Rc::new(RtConn::new(
            PortReference::new(a.clone(), None),
            vec![PortReference::new(b.clone(), None)],
        ))
    };
    let i = rt_integ(1.0);
    let a = rt_amp(-1.0);
    let sim = RtSim::with_defaults(vec![i.clone(), a.clone()], vec![conn(&i, &a), conn(&a, &i)]);
    let module = module_from_sim(&sim, "fxdecay");

    let opts = CodegenOptions { numeric: Numeric::Fixed { frac: 16 }, ..Default::default() };
    let files = generate(&module, &opts).expect("generate fixed decay");
    let model = concat_sources(&files);
    assert!(model.contains("int32_t x[1]"), "int32 storage: {model}");
    assert!(model.contains("65536 /* 1.0 */"), "Q-scaled literal: {model}");
    assert!(model.contains("int64_t"), "widened arithmetic: {model}");
    assert!(model.contains("FXDECAY_Q_FROM_DOUBLE"), "conversion macros: {model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping fixed-point decay check");
        return;
    };
    let main = "#include <stdio.h>\nint main(void) {\n\
         \x20   fxdecay_t m;\n\
         \x20   fxdecay_init(&m);\n\
         \x20   fxdecay_run(&m, FXDECAY_Q_FROM_DOUBLE(1.0), FXDECAY_Q_FROM_DOUBLE(1.0 / 1024.0));\n\
         \x20   printf(\"%.17g %.17g\", FXDECAY_Q_TO_DOUBLE(m.x[0]), FXDECAY_Q_TO_DOUBLE(m.time));\n\
         \x20   return 0;\n}\n";
    match compile_and_run_files(&cc, 70, main, &files).expect("compile fixed decay") {
        None => eprintln!("fixed decay exe would not launch — skipping numeric check"),
        Some(got) => {
            assert_eq!(got.len(), 2);
            let x_ref = (-1.0_f64).exp();
            assert!(
                (got[0] - x_ref).abs() < 5e-3,
                "Q16.16 x(1) = {} vs e^-1 = {x_ref} (outside quantization budget)",
                got[0]
            );
            assert!((got[1] - 1.0).abs() < 1e-3, "final Q time: {}", got[1]);
        }
    }
}

/// LUT1D is the documented escape hatch for transcendentals under fixed
/// point — the interpolation itself must run in Q. constant(0.5) ->
/// lut([0,1,2] -> [0,10,40]) = 5 into an integrator: x(1) = 5.
#[test]
fn fixed_point_lut_interpolates_in_q() {
    use fastsim::codegen::Numeric;
    use fastsim::blocks::constructors::{constant, integrator as rt_integ, lut1d, ExtrapMode};
    use fastsim::connection::Connection as RtConn;
    use fastsim::ir::builder::module_from_sim;
    use fastsim::simulation::Simulation as RtSim;
    use fastsim::utils::portreference::PortReference;
    use std::rc::Rc;

    let conn = |a: &fastsim::blocks::block::BlockRef, b: &fastsim::blocks::block::BlockRef| {
        Rc::new(RtConn::new(
            PortReference::new(a.clone(), None),
            vec![PortReference::new(b.clone(), None)],
        ))
    };
    let src = constant(0.5);
    let lut = lut1d(vec![0.0, 1.0, 2.0], vec![0.0, 10.0, 40.0], ExtrapMode::Extrapolate).unwrap();
    let integ = rt_integ(0.0);
    let mut sim = RtSim::with_defaults(
        vec![src.clone(), lut.clone(), integ.clone()],
        vec![conn(&src, &lut), conn(&lut, &integ)],
    );
    sim.run(0.01, false, false);
    let module = module_from_sim(&sim, "fxlut");

    let opts = CodegenOptions { numeric: Numeric::Fixed { frac: 16 }, ..Default::default() };
    let files = generate(&module, &opts).expect("generate fixed lut");
    let model = concat_sources(&files);
    assert!(model.contains("static const int32_t _lx"), "Q table: {model}");

    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping fixed LUT check");
        return;
    };
    let main = "#include <stdio.h>\nint main(void) {\n\
         \x20   fxlut_t m;\n\
         \x20   fxlut_init(&m);\n\
         \x20   fxlut_run(&m, FXLUT_Q_FROM_DOUBLE(1.0), FXLUT_Q_FROM_DOUBLE(1.0 / 1024.0));\n\
         \x20   printf(\"%.17g\", FXLUT_Q_TO_DOUBLE(m.x[0]));\n\
         \x20   return 0;\n}\n";
    match compile_and_run_files(&cc, 71, main, &files).expect("compile fixed lut") {
        None => eprintln!("fixed lut exe would not launch — skipping numeric check"),
        Some(got) => {
            assert!((got[0] - 5.0).abs() < 1e-2, "Q lut integral: {} expected 5", got[0]);
        }
    }
}

/// Fixed point rejects what it cannot lower, with actionable messages: a
/// transcendental op points at LUT1D, an adaptive tableau at the fixed-step
/// set, the scaffold at the floating demo driver.
#[test]
fn fixed_point_rejects_unsupported_cleanly() {
    use fastsim::codegen::Numeric;
    use fastsim::blocks::constructors::{integrator as rt_integ, sin_block};
    use fastsim::connection::Connection as RtConn;
    use fastsim::ir::builder::module_from_sim;
    use fastsim::simulation::Simulation as RtSim;
    use fastsim::utils::portreference::PortReference;
    use std::rc::Rc;

    let conn = |a: &fastsim::blocks::block::BlockRef, b: &fastsim::blocks::block::BlockRef| {
        Rc::new(RtConn::new(
            PortReference::new(a.clone(), None),
            vec![PortReference::new(b.clone(), None)],
        ))
    };
    let i = rt_integ(1.0);
    let sn = sin_block();
    let sim = RtSim::with_defaults(vec![i.clone(), sn.clone()], vec![conn(&i, &sn), conn(&sn, &i)]);
    let module = module_from_sim(&sim, "fxsin");
    let fixed = CodegenOptions { numeric: Numeric::Fixed { frac: 16 }, ..Default::default() };
    let err = generate(&module, &fixed).expect_err("sin must be rejected under fixed");
    let msg = err.to_string();
    assert!(msg.contains("LUT1D"), "actionable message: {msg}");

    let i2 = rt_integ(1.0);
    let a2 = fastsim::blocks::constructors::amplifier(-1.0);
    let sim2 = RtSim::with_defaults(vec![i2.clone(), a2.clone()], vec![conn(&i2, &a2), conn(&a2, &i2)]);
    let module2 = module_from_sim(&sim2, "fxad");
    let adaptive = CodegenOptions {
        numeric: Numeric::Fixed { frac: 16 },
        solver: fastsim::codegen::SolverChoice::by_name("RKBS32").unwrap(),
        ..Default::default()
    };
    let msg = generate(&module2, &adaptive).expect_err("adaptive under fixed").to_string();
    assert!(msg.contains("adaptive"), "{msg}");

    let scaffold = CodegenOptions {
        numeric: Numeric::Fixed { frac: 16 },
        scaffold: true,
        ..Default::default()
    };
    let msg = generate(&module2, &scaffold).expect_err("scaffold under fixed").to_string();
    assert!(msg.contains("scaffold"), "{msg}");
}
