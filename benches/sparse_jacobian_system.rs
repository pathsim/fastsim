// End-to-end system benchmark for the sparse AD Jacobian (SAJ-5). A long chain of
// stiff, NONLINEAR scalar ODE stages, statically compiled and integrated with an
// implicit solver (DIRK3). Stage i is a stiff logistic ODE with nearest-neighbour
// coupling:
//
//   dx_i/dt = -k_i * x_i  -  b * x_i^2  +  c * x_{i-1}     (stage 0 takes a drive)
//
// built from primitive op-lowering blocks (integrator / amplifier / pow / adder)
// so the fused system traces to SSA and compile derives an analytic Jacobian.
// That Jacobian is bidiagonal (self + previous neighbour) and state-dependent
// (the x_i^2 term), hence large + sparse + NONLINEAR: it changes every step (no
// factorization cache), so the implicit solve hits the sparse-AD Newton path.
//
// A/B: run as-is for the sparse number; for the dense baseline raise
// `LINSOLVE_SPARSE_MIN_DIM` to `usize::MAX` in src/constants.rs and rebuild (the
// compile-time gate then keeps the Jacobian dense), exactly like `stiff_sparse`.

use std::rc::Rc;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};

use fastsim::blocks::block::BlockRef;
use fastsim::blocks::constructors::{adder, amplifier, integrator, pow_block, sinusoidal_source};
use fastsim::compile::compile;
use fastsim::connection::Connection;
use fastsim::ir::builder::module_from_sim;
use fastsim::simulation::Simulation;
use fastsim::utils::portreference::PortReference;

fn wire(src: &BlockRef, dst: &BlockRef) -> Rc<Connection> {
    Rc::new(Connection::new(
        PortReference::new(src.clone(), None),
        vec![PortReference::new(dst.clone(), None)],
    ))
}

/// Build a chain of `n` stiff nonlinear logistic stages with nearest-neighbour
/// coupling. Every stage sums three scalar terms (self, nonlinear, coupling/drive)
/// through one Adder, so the Adder's scalar `sum` semantics are exact.
fn build_chain(n: usize) -> (Vec<BlockRef>, Vec<Rc<Connection>>) {
    let drive = sinusoidal_source(1.0, 1.0, 0.0);
    let integs: Vec<BlockRef> = (0..n).map(|_| integrator(0.1)).collect();
    let mut blocks: Vec<BlockRef> = vec![drive.clone()];
    let mut conns: Vec<Rc<Connection>> = Vec::new();

    for i in 0..n {
        // Stiffness grows along the chain so the implicit solver is exercised.
        let k = 1.0 + (i as f64 / n as f64) * 200.0;
        let self_gain = amplifier(-k); //  -k_i * x_i
        let sq = pow_block(2.0); //          x_i^2
        let sq_gain = amplifier(-0.5); //   -b * x_i^2
        let coupling = if i == 0 { drive.clone() } else { amplifier(0.3) }; // c * x_{i-1}
        let sum = adder(Some("+++"));

        conns.push(wire(&integs[i], &self_gain));
        conns.push(wire(&integs[i], &sq));
        conns.push(wire(&sq, &sq_gain));
        if i > 0 {
            conns.push(wire(&integs[i - 1], &coupling));
        }
        conns.push(wire(&self_gain, &sum));
        conns.push(wire(&sq_gain, &sum));
        conns.push(wire(&coupling, &sum));
        conns.push(wire(&sum, &integs[i]));

        blocks.push(self_gain);
        blocks.push(sq);
        blocks.push(sq_gain);
        if i > 0 {
            blocks.push(coupling);
        }
        blocks.push(sum);
    }
    blocks.extend(integs);
    (blocks, conns)
}

fn compile_chain(n: usize) -> fastsim::compile::CompiledSimulation {
    let (blocks, conns) = build_chain(n);
    let mut sim = Simulation::with_defaults(blocks, conns);
    sim.run(0.01, false, false); // assemble (resolve shape-poly widths)
    let mut c = compile(&module_from_sim(&sim, "nonlinear_sparse_chain")).expect("chain compiles");
    c.set_solver("DIRK3", 1e-9, 1e-7);
    c.dt = 0.01;
    c
}

fn bench_chain(c: &mut Criterion) {
    let dur = 0.5;
    let mut group = c.benchmark_group("sparse_jac_nonlinear_chain");
    for &n in &[64usize, 128] {
        let nstate = compile_chain(n).n_state;
        group.bench_function(format!("N{nstate}"), |b| {
            b.iter_batched(
                || compile_chain(n),
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
