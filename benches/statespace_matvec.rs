// StateSpace matvec benchmark: validates the L2 (alloc-free SmallVec operands)
// and L3 (4-lane multi-accumulator `dot`) runtime optimization. The dominant
// cost in a StateSpace-heavy model is each block's `dx/dt = A*x + B*u`, a dense
// n*n matvec evaluated every solver stage.
//
// `statespace_fdyn` calls the block's native `f_dyn` closure directly, isolating
// the matvec primitive. `statespace_chain` runs a fixed-step simulation of many
// chained dense StateSpace blocks, so the system-level WCT reflects the matvec.
//
// A/B: run on the optimized branch, then `git checkout master -- <the four
// changed source files>` and re-run to get the pre-optimization baseline.

use std::rc::Rc;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};

use fastsim::blocks::block::BlockRef;
use fastsim::blocks::constructors::{sinusoidal_source, statespace};
use fastsim::connection::Connection;
use fastsim::simulation::Simulation;
use fastsim::utils::portreference::PortReference;

/// Deterministic dense matrix (LCG, no `rand` dependency, no `Math::random`):
/// diagonally-dominant-negative so the chained sim stays stable, every entry
/// nonzero so the matvec is genuinely dense (zeros would be skipped).
fn dense_a(n: usize) -> Vec<f64> {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f64) / ((1u64 << 31) as f64) - 1.0 // in [-1, 1)
    };
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = if i == j { -1.0 } else { 0.08 * next() };
        }
    }
    a
}

/// SISO-interfaced dense StateSpace: `n` states, 1 input, 1 output, no feedthrough.
fn ss_block(n: usize) -> BlockRef {
    statespace(dense_a(n), vec![1.0; n], vec![1.0; n], vec![0.0], n, 1, 1, None)
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
    Simulation::with_defaults(all, conns)
}

fn bench_fdyn(c: &mut Criterion) {
    let mut group = c.benchmark_group("statespace_fdyn");
    for &n in &[8usize, 16, 32, 64] {
        let blk = ss_block(n);
        let b = blk.borrow();
        let f = b.f_dyn.as_ref().expect("statespace has f_dyn");
        let x: Vec<f64> = (0..n).map(|i| 0.1 + i as f64 * 0.01).collect();
        let u = vec![1.0];
        let mut out: Vec<f64> = Vec::with_capacity(n);
        group.bench_function(format!("dense_n{n}"), |bn| {
            bn.iter(|| {
                out.clear();
                f(black_box(&x), black_box(&u), 0.0, &mut out);
                black_box(&out[0]);
            });
        });
    }
    group.finish();
}

fn bench_chain(c: &mut Criterion) {
    let dur = 1.0;
    let mut group = c.benchmark_group("statespace_chain");
    for &(m, n) in &[(12usize, 8usize), (8, 24)] {
        group.bench_function(format!("m{m}_n{n}"), |b| {
            b.iter_batched(
                || make_chain(m, n),
                |mut sim| {
                    sim.run(dur, false, false);
                    black_box(&sim);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_fdyn, bench_chain);
criterion_main!(benches);
