// Micro-benchmarks for the event-loop overhead in Simulation::timestep.
// Measures how Vec allocations + Rc::clone() per step scale with event count.

use std::rc::Rc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use fastsim::blocks::constructors::*;
use fastsim::connection::Connection;
use fastsim::events::schedule::Schedule;
use fastsim::events::zerocrossing::ZeroCrossing;
use fastsim::simulation::{SimEventRef, Simulation};
use fastsim::utils::fastcell::FastCell;
use fastsim::utils::portreference::PortReference;

fn connect(
    src: &fastsim::blocks::block::BlockRef,
    dst: &fastsim::blocks::block::BlockRef,
) -> Rc<Connection> {
    Rc::new(Connection::new(
        PortReference::new(src.clone(), None),
        vec![PortReference::new(dst.clone(), None)],
    ))
}

// Trivial system: constant -> scope. Keeps the timestep dominated by event-loop
// overhead, not solver/block work.
fn build_sim(n_events: usize, event_kind: EventKind) -> Simulation {
    let c = constant(1.0);
    let s = scope(None, 0.0, vec![]);
    let conn = connect(&c, &s);

    let mut sim = Simulation::with_defaults(vec![c, s], vec![conn]);
    sim.dt = 0.001;

    for i in 0..n_events {
        let evt: SimEventRef = match event_kind {
            EventKind::Schedule => {
                // Periods far beyond sim duration so nothing fires during the run
                Rc::new(FastCell::new(Schedule::new(
                    1e6 + i as f64,
                    None,
                    1e6,
                    None,
                    1e-8,
                )))
            }
            EventKind::ZeroCrossing => {
                // Static positive function, never triggers
                let offset = -(1.0 + i as f64);
                Rc::new(FastCell::new(ZeroCrossing::new(
                    move |_t| offset,
                    None,
                    1e-6,
                )))
            }
        };
        sim.add_event(evt);
    }
    sim
}

#[derive(Clone, Copy)]
enum EventKind {
    Schedule,
    ZeroCrossing,
}

fn bench_fixed_step(c: &mut Criterion) {
    // Non-adaptive: exercises _buffer + _detected_events every step
    let mut group = c.benchmark_group("event_loop/fixed_step");
    // 1000 steps per run (dt=0.001, duration=1.0)
    group.throughput(Throughput::Elements(1000));
    for &n in &[0usize, 1, 10, 100] {
        group.bench_with_input(
            BenchmarkId::new("schedule", n),
            &n,
            |b, &n| {
                let mut sim = build_sim(n, EventKind::Schedule);
                b.iter(|| {
                    sim.run(1.0, true, false);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("zerocrossing", n),
            &n,
            |b, &n| {
                let mut sim = build_sim(n, EventKind::ZeroCrossing);
                b.iter(|| {
                    sim.run(1.0, true, false);
                });
            },
        );
    }
    group.finish();
}

fn bench_adaptive(c: &mut Criterion) {
    // Adaptive: also exercises _estimate_events each iteration
    let mut group = c.benchmark_group("event_loop/adaptive");
    group.throughput(Throughput::Elements(1000));
    for &n in &[0usize, 1, 10, 100] {
        group.bench_with_input(
            BenchmarkId::new("schedule", n),
            &n,
            |b, &n| {
                let mut sim = build_sim(n, EventKind::Schedule);
                b.iter(|| {
                    sim.run(1.0, true, true);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("zerocrossing", n),
            &n,
            |b, &n| {
                let mut sim = build_sim(n, EventKind::ZeroCrossing);
                b.iter(|| {
                    sim.run(1.0, true, true);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_fixed_step, bench_adaptive);
criterion_main!(benches);
