// System-level benchmark for the implicit-solver factorization cache (J2/J3):
// a chain of stiff dense StateSpace blocks integrated with a fixed-step implicit
// solver (DIRK3). Each block's stage solve runs Newton with its constant
// Jacobian J = A; at fixed dt the Newton matrix `(I - dt·a_ii·A)` is unchanged,
// so the cached LinearSolver factors once and reuses it across iterations,
// stages and steps.
//
// A/B against the pre-cache state: `git checkout c4a16b1 -- src/optim/linsolve.rs
// src/optim/anderson.rs`, rebuild, re-run, then restore.

use std::rc::Rc;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};

use fastsim::blocks::block::BlockRef;
use fastsim::blocks::constructors::{sinusoidal_source, statespace};
use fastsim::connection::Connection;
use fastsim::simulation::Simulation;
use fastsim::solvers::factories::dirk3_factory;
use fastsim::utils::portreference::PortReference;

/// Dense, diagonally-dominant, stiff `A`: diagonal spans several decades of
/// decay rates (stiffness), off-diagonal coupling keeps it dense (full matvec /
/// factorization). Deterministic LCG, no `rand`.
fn stiff_a(n: usize) -> Vec<f64> {
    let mut s: u64 = 0xC0FF_EE12_3456_789A;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
    };
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                // decay rate from ~-1 to ~-200 across the diagonal -> stiff.
                a[i * n + j] = -(1.0 + (i as f64 / n as f64) * 200.0);
            } else {
                a[i * n + j] = 0.05 * next();
            }
        }
    }
    a
}

fn ss_block(n: usize) -> BlockRef {
    statespace(stiff_a(n), vec![1.0; n], vec![1.0; n], vec![0.0], n, 1, 1, None)
}

fn connect(src: &BlockRef, dst: &BlockRef) -> Rc<Connection> {
    Rc::new(Connection::new(
        PortReference::new(src.clone(), None),
        vec![PortReference::new(dst.clone(), None)],
    ))
}

fn make_chain(m: usize, n: usize) -> Simulation {
    let src = sinusoidal_source(1.0, 1.0, 0.0);
    let blocks: Vec<BlockRef> = (0..m).map(|_| ss_block(n)).collect();
    let mut conns = vec![connect(&src, &blocks[0])];
    for i in 1..m {
        conns.push(connect(&blocks[i - 1], &blocks[i]));
    }
    let mut all = vec![src];
    all.extend(blocks);
    let mut sim = Simulation::with_defaults(all, conns);
    sim.set_solver(dirk3_factory()); // fixed-step implicit -> constant dt
    sim
}

fn bench_chain(c: &mut Criterion) {
    let dur = 1.0;
    let mut group = c.benchmark_group("stiff_statespace_dirk3");
    for &(m, n) in &[(6usize, 16usize), (4, 32)] {
        group.bench_function(format!("m{m}_n{n}"), |b| {
            b.iter_batched(
                || make_chain(m, n),
                |mut sim| {
                    sim.run(dur, false, false); // fixed-step
                    black_box(&sim);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_chain);
criterion_main!(benches);
