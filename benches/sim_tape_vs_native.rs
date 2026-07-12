// WP2 system-level benchmark gate: a full `run()` of a control loop built
// only from the five converted blocks, comparing the 2a op-graph (tape)
// runtime against hand-written native closures.
//
// Both arms build the identical model via the real constructors (so solver
// engines, registers, roles, schedule are byte-identical); the "native" arm
// then OVERWRITES just the `f_alg`/`f_dyn` closures with the pre-refactor
// native versions. The only thing differing between arms is how those five
// blocks evaluate, so the wall-clock delta is exactly the per-call tape
// overhead seen at the system level (does it survive solver/bookkeeping cost?).

use std::rc::Rc;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

use fastsim::blocks::block::{BlockFn, BlockRef};
use fastsim::blocks::constructors::{adder, amplifier, integrator, scope, sinusoidal_source};
use fastsim::connection::Connection;
use fastsim::simulation::Simulation;
use fastsim::utils::portreference::PortReference;

fn connect(src: &BlockRef, dst: &BlockRef) -> Rc<Connection> {
    Rc::new(Connection::new(
        PortReference::new(src.clone(), None),
        vec![PortReference::new(dst.clone(), None)],
    ))
}

/// setpoint(sin) -> err(+ -) -> Kp -> plant(integrator) -> err (feedback),
/// plant -> scope. Exercises sinusoidal_source, adder (2-input shape-poly),
/// amplifier, integrator (alg + dyn) in a fixed-point loop.
fn make_loop(native: bool) -> Simulation {
    let src = sinusoidal_source(1.0, 1.0, 0.0);
    let err = adder(Some("+-"));
    let kp = amplifier(4.0);
    let plant = integrator(0.0);
    let sco = scope(None, 0.0, vec![]);

    if native {
        overwrite_native(&src, NativeKind::Sin { freq: 1.0, amp: 1.0, phase: 0.0 });
        overwrite_native(&err, NativeKind::Adder { ops: vec![1.0, -1.0] });
        overwrite_native(&kp, NativeKind::Amplifier { gain: 4.0 });
        overwrite_native(&plant, NativeKind::Integrator);
    }

    let conns = vec![
        connect(&src, &err),
        connect(&err, &kp),
        connect(&kp, &plant),
        connect(&plant, &err),
        connect(&plant, &sco),
    ];
    Simulation::with_defaults(vec![src, err, kp, plant, sco], conns)
}

enum NativeKind {
    Sin { freq: f64, amp: f64, phase: f64 },
    Adder { ops: Vec<f64> },
    Amplifier { gain: f64 },
    Integrator,
}

/// Replace a block's op-graph-derived closures with the original native ones,
/// keeping all other scaffolding (engine, registers, role) intact.
fn overwrite_native(blk: &BlockRef, kind: NativeKind) {
    let b = blk.borrow_mut();
    match kind {
        NativeKind::Sin { freq, amp, phase } => {
            let f: BlockFn = Box::new(move |_x, _u, t, out| {
                out.push(amp * (2.0 * std::f64::consts::PI * freq * t + phase).sin());
            });
            b.f_alg = Some(f);
        }
        NativeKind::Adder { ops } => {
            let f: BlockFn = Box::new(move |_x, u, _t, out| {
                let y = if ops.is_empty() {
                    u.iter().sum::<f64>()
                } else {
                    u.iter()
                        .zip(ops.iter().chain(std::iter::repeat(&0.0)))
                        .map(|(&ui, &oi)| ui * oi)
                        .sum::<f64>()
                };
                out.push(y);
            });
            b.f_alg = Some(f);
        }
        NativeKind::Amplifier { gain } => {
            let f: BlockFn = Box::new(move |_x, u, _t, out| {
                out.extend(u.iter().map(|&v| v * gain));
            });
            b.f_alg = Some(f);
        }
        NativeKind::Integrator => {
            b.f_alg = Some(Box::new(|x, _u, _t, out| out.extend_from_slice(x)));
            b.f_dyn = Some(Box::new(|_x, u, _t, out| out.extend_from_slice(u)));
        }
    }
}

fn bench_loop(c: &mut Criterion) {
    let dur = 1.0_f64;
    let mut group = c.benchmark_group("sim/control_loop");
    group.bench_function("native", |b| {
        b.iter_batched(
            || make_loop(true),
            |mut sim| {
                sim.run(dur, false, false);
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("tape", |b| {
        b.iter_batched(
            || make_loop(false),
            |mut sim| {
                sim.run(dur, false, false);
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(sim, bench_loop);
criterion_main!(sim);
