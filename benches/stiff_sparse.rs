// System-level benchmark for the gated sparse-LU path (Step 3 / S3): a long
// chain of small stiff StateSpace blocks, statically compiled. The fused system
// Jacobian `∂F/∂x` is block-bidiagonal (each block couples to itself and the
// previous one through B·C), hence large and sparse: density ≈ 2/m for m blocks,
// so it trips the sparse gate (n ≥ LINSOLVE_SPARSE_MIN_DIM, density ≤ cap) and
// the cached LinearSolver factors/solves with sparse LU instead of dense.
//
// A/B: this bench is run twice — once as-is (sparse path active), once with the
// sparse gate raised so every solve is dense, by temporarily setting
// `LINSOLVE_SPARSE_MIN_DIM` to `usize::MAX` in src/constants.rs and rebuilding.
// The dense build measures the baseline the sparse path is compared against.

use std::rc::Rc;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};

use fastsim::blocks::block::BlockRef;
use fastsim::blocks::constructors::{sinusoidal_source, statespace};
use fastsim::compile::compile;
use fastsim::connection::Connection;
use fastsim::ir::builder::module_from_sim;
use fastsim::simulation::Simulation;
use fastsim::utils::portreference::PortReference;

/// Small dense stiff `A` (per block). Deterministic LCG, no `rand`.
fn stiff_a(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
    };
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                a[i * n + j] = -(1.0 + (i as f64 / n as f64) * 200.0);
            } else {
                a[i * n + j] = 0.05 * next();
            }
        }
    }
    a
}

fn ss_block(n: usize, seed: u64) -> BlockRef {
    statespace(stiff_a(n, seed), vec![1.0; n], vec![1.0; n], vec![0.0], n, 1, 1, None)
}

fn connect(src: &BlockRef, dst: &BlockRef) -> Rc<Connection> {
    Rc::new(Connection::new(
        PortReference::new(src.clone(), None),
        vec![PortReference::new(dst.clone(), None)],
    ))
}

/// Compile a long chain of small stiff StateSpace blocks (sparse system Jacobian).
fn compile_chain(m: usize, n: usize) -> fastsim::compile::CompiledSimulation {
    let src = sinusoidal_source(1.0, 1.0, 0.0);
    let blocks: Vec<BlockRef> = (0..m).map(|k| ss_block(n, 0xC0FFEE ^ k as u64)).collect();
    let mut conns = vec![connect(&src, &blocks[0])];
    for i in 1..m {
        conns.push(connect(&blocks[i - 1], &blocks[i]));
    }
    let mut all = vec![src];
    all.extend(blocks);
    let mut sim = Simulation::with_defaults(all, conns);
    sim.run(0.01, false, false); // assemble (resolves shape-poly widths)
    let mut c = compile(&module_from_sim(&sim, "sparse_chain")).expect("chain compiles");
    c.set_solver("DIRK3", 1e-9, 1e-7);
    c.dt = 0.01;
    c
}

fn bench_chain(c: &mut Criterion) {
    let dur = 1.0;
    let mut group = c.benchmark_group("stiff_sparse_dirk3");
    // (m blocks, n per block) → N = m·n, density ≈ 2/m. Both trip the sparse gate.
    for &(m, n) in &[(16usize, 4usize), (24, 4)] {
        let nstate = compile_chain(m, n).n_state;
        group.bench_function(format!("N{nstate}_m{m}_n{n}"), |b| {
            b.iter_batched(
                || compile_chain(m, n),
                |mut c| {
                    c.run(dur, true, false);
                    black_box(&c);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_chain);
criterion_main!(benches);
