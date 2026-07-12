// Microbenchmarks for the `jit::graph::Tape` hot path.
//
// Measures `InterpretedFn::call_into` directly — the tape interpreter
// loop is what every JIT-traced ODE / Function / Source / DAE block
// pays per evaluation.  Workloads are hand-built `Graph`s (no tracer)
// so the noise from PyO3 / tracing is excluded.
//
// Profiles cover the realistic mix:
//   - polynomial: lots of `Const + Mul + Add` (typical ODE RHS like
//     `dx/dt = -a·x + b·u`, Lorenz, Van der Pol)
//   - trig_chain: a stack of unary transcendentals (radar / signal-processing
//     style derivatives)
//   - mixed: a small DynamicalSystem-shaped block with const params,
//     state reads, and a transcendental for non-trivial tape size

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};

use fastsim::ssa::graph::{BinOp, Graph, InputSignature, UnaryOp};
use fastsim::ssa::tape::InterpretedFn;
use fastsim::ssa::optimize::optimize;

/// Run the same graph optimization pipeline the tracer applies before
/// lowering — keeps the bench representative of real ODE / Function /
/// DynamicalSystem block evaluation (constant fold, strength reduce,
/// algebraic simplify, FMA fusion, then DCE inside `from_graph`).
fn lower(mut g: Graph) -> InterpretedFn {
    optimize(&mut g);
    InterpretedFn::from_graph(g)
}

// --------------------------------------------------------------------------
// Workload builders
// --------------------------------------------------------------------------

/// A 4th-degree polynomial: `y = c4·x^4 + c3·x^3 + c2·x^2 + c1·x + c0`,
/// expressed as the Horner form `((((c4·x + c3)·x + c2)·x + c1)·x + c0)`
/// to maximise the realistic Const/Mul/Add density.
fn build_polynomial_4th() -> InterpretedFn {
    let sig = InputSignature::from_named_sizes([("x", 1)]);
    let mut g = Graph::new(sig);
    let x = g.input(0);

    let c4 = g.constant(0.5);
    let c3 = g.constant(-1.2);
    let c2 = g.constant(3.0);
    let c1 = g.constant(-2.5);
    let c0 = g.constant(0.7);

    let t1 = g.binary(BinOp::Mul, c4, x);
    let t2 = g.binary(BinOp::Add, t1, c3);
    let t3 = g.binary(BinOp::Mul, t2, x);
    let t4 = g.binary(BinOp::Add, t3, c2);
    let t5 = g.binary(BinOp::Mul, t4, x);
    let t6 = g.binary(BinOp::Add, t5, c1);
    let t7 = g.binary(BinOp::Mul, t6, x);
    let t8 = g.binary(BinOp::Add, t7, c0);
    g.outputs.push(t8);

    lower(g)
}

/// `y = sin(cos(tan(exp(x))))` — five transcendental ops in series.
fn build_trig_chain() -> InterpretedFn {
    let sig = InputSignature::from_named_sizes([("x", 1)]);
    let mut g = Graph::new(sig);
    let x = g.input(0);
    let a = g.unary(UnaryOp::Exp, x);
    let b = g.unary(UnaryOp::Tan, a);
    let c = g.unary(UnaryOp::Cos, b);
    let d = g.unary(UnaryOp::Sin, c);
    g.outputs.push(d);
    lower(g)
}

/// Realistic mixed workload, modelled on a DynamicalSystem block:
/// `dx/dt = -a·x + b·sin(u) + c`,  with state `x`, input `u`, and 3
/// const params folded inline.
fn build_mixed_dx() -> InterpretedFn {
    let sig = InputSignature::from_named_sizes([("x", 1), ("u", 1)]);
    let mut g = Graph::new(sig);
    let x = g.input(0);
    let u = g.input(1);

    let a = g.constant(0.8);
    let b = g.constant(1.5);
    let c = g.constant(0.3);

    let neg_a = g.unary(UnaryOp::Neg, a);
    let t1 = g.binary(BinOp::Mul, neg_a, x); // -a·x
    let s = g.unary(UnaryOp::Sin, u);
    let t2 = g.binary(BinOp::Mul, b, s);     // b·sin(u)
    let t3 = g.binary(BinOp::Add, t1, t2);
    let dx = g.binary(BinOp::Add, t3, c);    // + c
    g.outputs.push(dx);
    lower(g)
}

// --------------------------------------------------------------------------
// Benches
// --------------------------------------------------------------------------

fn bench_polynomial(c: &mut Criterion) {
    let f = build_polynomial_4th();
    let mut group = c.benchmark_group("jit_tape/polynomial4");
    group.bench_function("call_into", |b| {
        let x = [0.7_f64];
        let inputs: [&[f64]; 1] = [&x];
        let mut out = [0.0_f64; 1];
        b.iter(|| {
            f.call_into(black_box(&inputs), black_box(&mut out));
            black_box(&out);
        })
    });
    group.finish();
}

fn bench_trig_chain(c: &mut Criterion) {
    let f = build_trig_chain();
    let mut group = c.benchmark_group("jit_tape/trig_chain");
    group.bench_function("call_into", |b| {
        let x = [0.5_f64];
        let inputs: [&[f64]; 1] = [&x];
        let mut out = [0.0_f64; 1];
        b.iter(|| {
            f.call_into(black_box(&inputs), black_box(&mut out));
            black_box(&out);
        })
    });
    group.finish();
}

fn bench_mixed_dx(c: &mut Criterion) {
    let f = build_mixed_dx();
    let mut group = c.benchmark_group("jit_tape/mixed_dx");
    group.bench_function("call_into", |b| {
        b.iter_batched(
            || (0.4_f64, 0.7_f64),
            |(xv, uv)| {
                let x = [xv];
                let u = [uv];
                let inputs: [&[f64]; 2] = [&x, &u];
                let mut out = [0.0_f64; 1];
                f.call_into(&inputs, &mut out);
                black_box(out);
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

/// Dense matrix-vector product `y = A·x` (n outputs, each a fused `Dot` of an
/// n-element const row with the n-element input). Exercises the tape's `Dot`
/// arm (gather + canonical 4-lane reduction) — the matvec at the heart of a
/// Newton step on a dense system.
fn build_matvec(n: usize) -> InterpretedFn {
    let sig = InputSignature::from_named_sizes([("x", n)]);
    let mut g = Graph::new(sig);
    let x: Vec<_> = (0..n as u32).map(|i| g.input(i)).collect();
    for r in 0..n {
        let row: Vec<_> = (0..n)
            .map(|c| g.constant(0.5 + ((r * n + c) as f64) * 1e-3))
            .collect();
        let d = g.dot(row, x.clone());
        g.outputs.push(d);
    }
    lower(g)
}

fn bench_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_tape/matvec_dot");
    for n in [8usize, 32, 64] {
        let f = build_matvec(n);
        let x: Vec<f64> = (0..n).map(|i| 0.1 + i as f64 * 0.01).collect();
        let mut out = vec![0.0_f64; n];
        group.bench_function(format!("n{n}"), |b| {
            let inputs: [&[f64]; 1] = [&x];
            b.iter(|| {
                f.call_into(black_box(&inputs), black_box(&mut out));
                black_box(&out);
            })
        });
    }
    group.finish();
}

criterion_group!(jit_tape, bench_polynomial, bench_trig_chain, bench_mixed_dx, bench_matvec);
criterion_main!(jit_tape);
