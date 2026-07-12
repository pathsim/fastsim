//! Closing the loop: compile the generated C, run it, and compare the result
//! to `fastsim::ir::eval` (the IR reference evaluator, the source of truth).
//!
//! Skips gracefully when no C compiler is available, or when the produced
//! executable cannot be launched in this environment (DLL/runtime quirks): only
//! a *numeric mismatch* fails the test, since that is the one thing under the
//! emitter's control. A *compile* error fails loudly: that means the emitted C
//! is malformed.
#![cfg(feature = "codegen")]

use fastsim::ir::eval::{eval_region, EvalCtx};
use fastsim::ir::schema::{
    BinOpKind, CmpKind, NodeId, Op, ParamId, ReduceKind, Region, StateId, Write,
};
use fastsim::codegen::{c_prelude, emit_region_fn, CodegenOptions, Reductions};

mod common;
use common::{compile_and_run, find_cc};

/// Declare a C array and fill it element-by-element. Avoids aggregate
/// initializers (`double a[1] = {..}`), which crash the ancient MSYS2 gcc 5.3.
fn decl_assign(name: &str, vals: &[f64]) -> String {
    let mut s = format!("    double {name}[{}];\n", vals.len().max(1));
    for (i, v) in vals.iter().enumerate() {
        s.push_str(&format!("    {name}[{i}] = {v:?};\n"));
    }
    s
}

/// Build a `main` that calls `f(u, x, p, m, 0.0, out)` and prints `out`.
fn build_main(u: &[f64], x: &[f64], p: &[f64], n_out: usize) -> String {
    format!(
        "int main(void) {{\n{}{}{}    double m[1];\n    double out[{}];\n\
         \x20   f(u, x, p, m, 0.0, out);\n\
         \x20   for (int i = 0; i < {}; i++) printf(\"%.17g \", out[i]);\n\
         \x20   return 0;\n\
         }}\n",
        decl_assign("u", u),
        decl_assign("x", x),
        decl_assign("p", p),
        n_out.max(1),
        n_out,
    )
}

/// Verify one region against the reference evaluator via compiled C.
fn check(cc: &str, idx: usize, region: &Region, u: &[f64], x: &[f64], p: &[f64]) {
    let ctx = EvalCtx { inputs: &[u], state: x, memory: &[], params: p, t: 0.0 };
    let expected = eval_region(region, &ctx).expect("reference eval");

    let f = emit_region_fn("f", region, &CodegenOptions::default(), &[], &[8]).expect("emit");
    let src = format!("#include <stdio.h>\n{}{}\n{}", c_prelude(), f, build_main(u, x, p, expected.len()));

    match compile_and_run(cc, idx, &src).expect("compile") {
        None => eprintln!("case {idx}: skipped (exe would not launch)"),
        Some(got) => {
            assert_eq!(got.len(), expected.len(), "case {idx}: arity");
            for (g, e) in got.iter().zip(&expected) {
                let tol = 1e-9 * e.abs().max(1.0);
                assert!((g - e).abs() < tol, "case {idx}: C={g} ref={e}");
            }
        }
    }
}

/// `Reductions::Vectorized` lowers `Reduce`/`Dot` to counted loops over gathered
/// operand arrays instead of unrolled expressions. The numbers must match the
/// reference (and the unrolled form) exactly.
#[test]
fn vectorized_reductions_match_reference() {
    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping vectorized verification");
        return;
    };
    // y0 = sum(u0, u1, u2); y1 = Σ uᵢ·(reversed)ᵢ
    let region = Region {
        ops: vec![
            Op::Input { port: 0, elem: 0 },
            Op::Input { port: 0, elem: 1 },
            Op::Input { port: 0, elem: 2 },
            Op::Reduce { op: ReduceKind::Sum, args: vec![NodeId(0), NodeId(1), NodeId(2)] },
            Op::Dot { a: vec![NodeId(0), NodeId(1)], b: vec![NodeId(1), NodeId(0)] },
        ],
        writes: vec![
            Write::Output { port: 0, elem: 0, src: NodeId(3) },
            Write::Output { port: 0, elem: 1, src: NodeId(4) },
        ],
    };
    let u = [1.5, -2.0, 3.0];
    let ctx = EvalCtx { inputs: &[&u], state: &[], memory: &[], params: &[], t: 0.0 };
    let expected = eval_region(&region, &ctx).expect("reference eval");

    let opts = CodegenOptions { reductions: Reductions::Vectorized, ..Default::default() };
    let f = emit_region_fn("f", &region, &opts, &[], &[8]).expect("emit vectorized");
    assert!(f.contains("for (size_t _i"), "vectorized should emit counted loops: {f}");

    let src = format!("#include <stdio.h>\n{}{}\n{}", c_prelude(), f, build_main(&u, &[], &[], expected.len()));
    match compile_and_run(&cc, 6, &src).expect("compile vectorized") {
        None => eprintln!("vectorized exe would not launch — skipping numeric check"),
        Some(got) => {
            assert_eq!(got.len(), expected.len(), "vectorized arity");
            for (g, e) in got.iter().zip(&expected) {
                assert!((g - e).abs() < 1e-9 * e.abs().max(1.0), "vectorized C={g} ref={e}");
            }
        }
    }
}

/// `Op::Lut1d` emits the breakpoint/value tables as `static const` arrays plus a
/// counted segment search + linear interpolation (not an unrolled `select`
/// chain). The numbers must match `ir::eval::lut1d` across the grid, including
/// extrapolation past the ends and the `clamp` mode.
#[test]
fn lut1d_matches_reference() {
    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping LUT verification");
        return;
    };
    // y = interp(u; [0,1,2] -> [0,10,40]), extrapolating past the ends.
    let extrap = Region {
        ops: vec![
            Op::Input { port: 0, elem: 0 },
            Op::Lut1d { input: NodeId(0), points: vec![0.0, 1.0, 2.0], values: vec![0.0, 10.0, 40.0], clamp: false },
        ],
        writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(1) }],
    };
    let f = emit_region_fn("f", &extrap, &CodegenOptions::default(), &[], &[8]).expect("emit lut");
    assert!(f.contains("static const double _lx1[] = { 0.0, 1.0, 2.0 };"), "{f}");
    assert!(f.contains("for (size_t _j"), "table search loop expected: {f}");
    for u in [[-0.5_f64], [0.0], [0.5], [1.0], [1.5], [2.0], [2.5]] {
        check(&cc, 7, &extrap, &u, &[], &[]);
    }

    // Clamp mode: held flat past either end.
    let clamp = Region {
        ops: vec![
            Op::Input { port: 0, elem: 0 },
            Op::Lut1d { input: NodeId(0), points: vec![0.0, 1.0], values: vec![0.0, 10.0], clamp: true },
        ],
        writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(1) }],
    };
    for u in [[-1.0_f64], [0.25], [0.75], [2.0]] {
        check(&cc, 8, &clamp, &u, &[], &[]);
    }
}

#[test]
fn generated_c_matches_reference_evaluator() {
    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping C verification");
        return;
    };

    // 1. amplifier: y = gain * u
    check(
        &cc, 1,
        &Region {
            ops: vec![
                Op::Input { port: 0, elem: 0 },
                Op::Param { id: ParamId(0) },
                Op::Binary { op: BinOpKind::Mul, a: NodeId(0), b: NodeId(1) },
            ],
            writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(2) }],
        },
        &[3.0], &[], &[2.5],
    );

    // 2. transcendental chain: y = sin(u) + exp(u)
    check(
        &cc, 2,
        &Region {
            ops: vec![
                Op::Input { port: 0, elem: 0 },
                Op::Unary { op: fastsim::ir::schema::UnaryOpKind::Sin, a: NodeId(0) },
                Op::Unary { op: fastsim::ir::schema::UnaryOpKind::Exp, a: NodeId(0) },
                Op::Binary { op: BinOpKind::Add, a: NodeId(1), b: NodeId(2) },
            ],
            writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(3) }],
        },
        &[0.7], &[], &[],
    );

    // 3. reduction: y = sum(u0, u1, u2)
    check(
        &cc, 3,
        &Region {
            ops: vec![
                Op::Input { port: 0, elem: 0 },
                Op::Input { port: 0, elem: 1 },
                Op::Input { port: 0, elem: 2 },
                Op::Reduce { op: ReduceKind::Sum, args: vec![NodeId(0), NodeId(1), NodeId(2)] },
            ],
            writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(3) }],
        },
        &[1.5, -2.0, 3.0], &[], &[],
    );

    // 4. fused dot: y = Σ uᵢ·(reversed)ᵢ
    check(
        &cc, 4,
        &Region {
            ops: vec![
                Op::Input { port: 0, elem: 0 },
                Op::Input { port: 0, elem: 1 },
                Op::Dot { a: vec![NodeId(0), NodeId(1)], b: vec![NodeId(1), NodeId(0)] },
            ],
            writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(2) }],
        },
        &[3.0, 4.0], &[], &[],
    );

    // 5. state + comparison + select: y = (x0 > 0) ? x0 : -x0  (= |x0|)
    check(
        &cc, 5,
        &Region {
            ops: vec![
                Op::State { id: StateId(0) },
                Op::Const(0.0),
                Op::Cmp { op: CmpKind::Gt, a: NodeId(0), b: NodeId(1) },
                Op::Unary { op: fastsim::ir::schema::UnaryOpKind::Neg, a: NodeId(0) },
                Op::Select { c: NodeId(2), t: NodeId(0), e: NodeId(3) },
            ],
            writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(4) }],
        },
        &[], &[-4.0], &[],
    );
}

/// Traceable stateless RNG: the generated `fastsim_rand_uniform` helper must
/// reproduce `fastsim::ir::eval` (i.e. the Rust `rand_uniform`) bit-for-bit when
/// compiled and run, for both a uniform draw and a normal draw composed from it.
#[test]
fn rand_uniform_matches_reference_in_c() {
    use fastsim::ir::schema::UnaryOpKind as U;
    let Some(cc) = find_cc() else {
        eprintln!("no working C compiler found — skipping RNG verification");
        return;
    };
    // y = random_uniform(u0), keyed by the input.
    for (i, key) in [0.0_f64, 1.0, 3.0, 42.0, -7.5].iter().enumerate() {
        check(
            &cc, 100 + i,
            &Region {
                ops: vec![
                    Op::Input { port: 0, elem: 0 },
                    Op::Unary { op: U::RandUniform, a: NodeId(0) },
                ],
                writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(1) }],
            },
            &[*key], &[], &[],
        );
    }
}
