//! Coverage half of the op-vocabulary parity guard (the jit/IR halves live in
//! `fastsim::ir::builder` tests). Every sub-enum variant must lower to a C
//! expression through the real `emit_region_fn` path; `RandUniform` and
//! `Digamma` lower through the `fastsim_rand_uniform` / `fastsim_digamma`
//! helpers emitted in the header. Each `all_*()` lists its enum's variants
//! exactly once; the inner `_guard` match is exhaustive, so adding a variant to
//! fastsim fails to compile here until it is listed too.
#![cfg(feature = "codegen")]

use fastsim::ir::schema::{
    BinOpKind, CmpKind, NodeId, Op, Region, ReduceKind, UnaryOpKind, Write,
};
use fastsim::codegen::{emit_region_fn, CodegenOptions};

macro_rules! variant_list {
    ($name:ident, $ty:ty, $($v:ident),+ $(,)?) => {
        fn $name() -> Vec<$ty> {
            fn _guard(o: $ty) { match o { $(<$ty>::$v => {}),+ } }
            vec![$(<$ty>::$v),+]
        }
    };
}
variant_list!(all_bin, BinOpKind, Add, Sub, Mul, Div, Pow, Mod, Min, Max, Atan2, Hypot);
variant_list!(
    all_unary, UnaryOpKind,
    Neg, Sin, Cos, Tan, Atan, Sinh, Cosh, Tanh, Exp, Log, Log10, Abs, Sqrt, Sign, Floor,
    Asin, Acos, Asinh, Acosh, Atanh, Ceil, Round, Trunc, Log2, Log1p, Expm1, Cbrt,
    Erf, Erfc, Lgamma, Tgamma, Digamma, RandUniform,
);
variant_list!(all_cmp, CmpKind, Gt, Ge, Lt, Le, Eq, Ne);
variant_list!(all_reduce, ReduceKind, Sum, Product, Min, Max);

fn emit_ok(region: &Region) -> bool {
    emit_region_fn("f", region, &CodegenOptions::default(), &[], &[8]).is_ok()
}

#[test]
fn codegen_lowers_multi_input_ports() {
    // Two input ports of size 1; the hierarchical driver must resolve the second
    // port's element to the flat `u[1]` (parity with the struct API), not error.
    let region = Region {
        ops: vec![Op::Input { port: 1, elem: 0 }],
        writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(0) }],
    };
    let c = emit_region_fn("mp", &region, &CodegenOptions::default(), &[], &[1, 1]).expect("multi-port lowers");
    assert!(c.contains("v0 = u[1]"), "second input port should read u[1]:\n{c}");
}

#[test]
fn codegen_covers_every_binary_op() {
    for op in all_bin() {
        let region = Region {
            ops: vec![
                Op::Input { port: 0, elem: 0 },
                Op::Input { port: 0, elem: 1 },
                Op::Binary { op, a: NodeId(0), b: NodeId(1) },
            ],
            writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(2) }],
        };
        assert!(emit_ok(&region), "binary {op:?} did not lower to C");
    }
}

#[test]
fn codegen_covers_every_unary_op() {
    for op in all_unary() {
        let region = Region {
            ops: vec![
                Op::Input { port: 0, elem: 0 },
                Op::Unary { op, a: NodeId(0) },
            ],
            writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(1) }],
        };
        // Every unary op lowers to C. RandUniform and Digamma go through the
        // `fastsim_rand_uniform` / `fastsim_digamma` helpers emitted in the header.
        assert!(emit_ok(&region), "unary {op:?} did not lower to C");
    }
}

#[test]
fn codegen_covers_every_cmp_op() {
    for op in all_cmp() {
        let region = Region {
            ops: vec![
                Op::Input { port: 0, elem: 0 },
                Op::Input { port: 0, elem: 1 },
                Op::Cmp { op, a: NodeId(0), b: NodeId(1) },
            ],
            writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(2) }],
        };
        assert!(emit_ok(&region), "cmp {op:?} did not lower to C");
    }
}

#[test]
fn codegen_covers_every_reduce_op() {
    for op in all_reduce() {
        let region = Region {
            ops: vec![
                Op::Input { port: 0, elem: 0 },
                Op::Input { port: 0, elem: 1 },
                Op::Input { port: 0, elem: 2 },
                Op::Reduce { op, args: vec![NodeId(0), NodeId(1), NodeId(2)] },
            ],
            writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(3) }],
        };
        assert!(emit_ok(&region), "reduce {op:?} did not lower to C");
    }
}
