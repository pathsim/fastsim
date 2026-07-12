// WP2 benchmark gate: op-graph-derived block closures (2a) vs the original
// hand-written native closures.
//
// The 2a refactor makes each block's runtime `f_alg` / `f_dyn` a lowering of
// its op-graph (`InterpretedFn` tape) instead of a bespoke native closure.
// This bench quantifies the per-call overhead that buys us the single-source
// op-graph (and thus the IR / codegen / verification). The "native" arm
// replicates the pre-refactor closures verbatim; the "tape" arm pulls the
// derived `f_alg` straight off the constructed block.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use fastsim::blocks::constructors::{amplifier, constant, integrator, sinusoidal_source};

/// Pull a block's `f_alg` and invoke it once into `out` (caller clears, per
/// the BlockFn contract).
macro_rules! call_alg {
    ($blk:expr, $x:expr, $u:expr, $t:expr, $out:expr) => {{
        let b = $blk.borrow();
        let f = b.f_alg.as_ref().unwrap();
        $out.clear();
        f($x, $u, $t, &mut $out);
    }};
}

fn bench_amplifier_scalar(c: &mut Criterion) {
    let mut g = c.benchmark_group("amplifier_scalar");
    let gain = 2.0_f64;
    let u = [1.3_f64];
    let mut out = Vec::with_capacity(1);

    g.bench_function("native", |b| {
        let f = move |_x: &[f64], u: &[f64], _t: f64, out: &mut Vec<f64>| {
            out.extend(u.iter().map(|&v| v * gain));
        };
        b.iter(|| {
            out.clear();
            f(black_box(&[]), black_box(&u), 0.0, &mut out);
            black_box(&out);
        })
    });

    let blk = amplifier(gain);
    g.bench_function("tape", |b| {
        b.iter(|| {
            call_alg!(blk, black_box(&[]), black_box(&u), 0.0, out);
            black_box(&out);
        })
    });
    g.finish();
}

fn bench_amplifier_vec8(c: &mut Criterion) {
    let mut g = c.benchmark_group("amplifier_vec8");
    let gain = 2.0_f64;
    let u = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8_f64];
    let mut out = Vec::with_capacity(8);

    g.bench_function("native", |b| {
        let f = move |_x: &[f64], u: &[f64], _t: f64, out: &mut Vec<f64>| {
            out.extend(u.iter().map(|&v| v * gain));
        };
        b.iter(|| {
            out.clear();
            f(black_box(&[]), black_box(&u), 0.0, &mut out);
            black_box(&out);
        })
    });

    let blk = amplifier(gain);
    // Prime the shape-lazy cache at width 8.
    call_alg!(blk, &[], &u, 0.0, out);
    g.bench_function("tape", |b| {
        b.iter(|| {
            call_alg!(blk, black_box(&[]), black_box(&u), 0.0, out);
            black_box(&out);
        })
    });
    g.finish();
}

fn bench_integrator_alg(c: &mut Criterion) {
    let mut g = c.benchmark_group("integrator_alg");
    let x = [0.7_f64];
    let mut out = Vec::with_capacity(1);

    g.bench_function("native", |b| {
        let f = |x: &[f64], _u: &[f64], _t: f64, out: &mut Vec<f64>| {
            out.extend_from_slice(x);
        };
        b.iter(|| {
            out.clear();
            f(black_box(&x), black_box(&[]), 0.0, &mut out);
            black_box(&out);
        })
    });

    let blk = integrator(0.0);
    g.bench_function("tape", |b| {
        b.iter(|| {
            call_alg!(blk, black_box(&x), black_box(&[]), 0.0, out);
            black_box(&out);
        })
    });
    g.finish();
}

fn bench_sinusoidal_source(c: &mut Criterion) {
    let mut g = c.benchmark_group("sinusoidal_source");
    let (freq, amp, phase) = (1.0_f64, 2.0_f64, 0.3_f64);
    let mut out = Vec::with_capacity(1);

    g.bench_function("native", |b| {
        let f = move |_x: &[f64], _u: &[f64], t: f64, out: &mut Vec<f64>| {
            out.push(amp * (2.0 * std::f64::consts::PI * freq * t + phase).sin());
        };
        b.iter(|| {
            out.clear();
            f(black_box(&[]), black_box(&[]), black_box(0.3), &mut out);
            black_box(&out);
        })
    });

    let blk = sinusoidal_source(freq, amp, phase);
    g.bench_function("tape", |b| {
        b.iter(|| {
            call_alg!(blk, black_box(&[]), black_box(&[]), black_box(0.3), out);
            black_box(&out);
        })
    });
    g.finish();
}

fn bench_constant(c: &mut Criterion) {
    let mut g = c.benchmark_group("constant");
    let mut out = Vec::with_capacity(1);

    g.bench_function("native", |b| {
        let value = 3.5_f64;
        let f = move |_x: &[f64], _u: &[f64], _t: f64, out: &mut Vec<f64>| out.push(value);
        b.iter(|| {
            out.clear();
            f(black_box(&[]), black_box(&[]), 0.0, &mut out);
            black_box(&out);
        })
    });

    let blk = constant(3.5);
    g.bench_function("tape", |b| {
        b.iter(|| {
            call_alg!(blk, black_box(&[]), black_box(&[]), 0.0, out);
            black_box(&out);
        })
    });
    g.finish();
}

criterion_group!(
    ops,
    bench_amplifier_scalar,
    bench_amplifier_vec8,
    bench_integrator_alg,
    bench_sinusoidal_source,
    bench_constant
);
criterion_main!(ops);
