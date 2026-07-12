// System-level benchmark for the AD constant-Jacobian eval-once optimization
// (Step 2 / S2): a chain of stiff dense StateSpace blocks statically compiled
// into a single fused `dX/dt`, integrated with a fixed-step implicit solver
// (DIRK3). The AD-derived Jacobian `∂F/∂x = A` is globally constant (linear
// time-invariant), so the compiled run loop evaluates the Jacobian tape once and
// reuses the buffer across every Newton iteration, stage and step.
//
// A/B is done in-process on the same compiled model: the "const" group keeps the
// optimization, the "recompute" group calls `force_recompute_jacobian()` to take
// the general per-iteration path. Both produce identical results — this measures
// only the saved tape evaluations and matrix assembly (the linear solver's
// factorization cache already covers the `O(n³)` factor in both).

use std::rc::Rc;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};

use fastsim::blocks::block::BlockRef;
use fastsim::blocks::constructors::{sinusoidal_source, statespace};
use fastsim::compile::compile;
use fastsim::connection::Connection;
use fastsim::ir::builder::module_from_sim;
use fastsim::simulation::Simulation;
use fastsim::utils::portreference::PortReference;

/// Dense, diagonally-dominant, stiff `A`: the diagonal spans several decades of
/// decay rates, off-diagonal coupling keeps it dense. Deterministic LCG.
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

/// Build, assemble and statically compile a stiff StateSpace chain.
fn compile_chain(m: usize, n: usize) -> fastsim::compile::CompiledSimulation {
    let src = sinusoidal_source(1.0, 1.0, 0.0);
    let blocks: Vec<BlockRef> = (0..m).map(|_| ss_block(n)).collect();
    let mut conns = vec![connect(&src, &blocks[0])];
    for i in 1..m {
        conns.push(connect(&blocks[i - 1], &blocks[i]));
    }
    let mut all = vec![src];
    all.extend(blocks);
    let mut sim = Simulation::with_defaults(all, conns);
    sim.run(0.01, false, false); // assemble (resolves shape-poly widths)
    let mut c = compile(&module_from_sim(&sim, "stiff_chain")).expect("chain compiles");
    c.set_solver("DIRK3", 1e-9, 1e-7);
    c.dt = 0.01;
    c
}

fn bench_chain(c: &mut Criterion) {
    let dur = 1.0;
    let mut group = c.benchmark_group("stiff_compiled_dirk3");
    for &(m, n) in &[(6usize, 16usize), (4, 32)] {
        // Sanity: the compiled Jacobian must actually be detected constant,
        // otherwise this bench measures nothing.
        assert!(compile_chain(m, n).jacobian_is_constant(),
            "stiff StateSpace chain m{m}_n{n} must have a constant Jacobian");

        group.bench_function(format!("const_m{m}_n{n}"), |b| {
            b.iter_batched(
                || compile_chain(m, n),
                |mut c| {
                    c.run(dur, true, false);
                    black_box(&c);
                },
                BatchSize::SmallInput,
            );
        });
        group.bench_function(format!("recompute_m{m}_n{n}"), |b| {
            b.iter_batched(
                || {
                    let mut c = compile_chain(m, n);
                    c.force_recompute_jacobian();
                    c
                },
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
