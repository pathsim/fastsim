// Numpy ufunc-name → graph-op dispatch tables.
//
// Pure `&str → Option<Op>` maps, factored out of the tracer so the dispatch
// coverage is one cohesive, directly unit-testable concern (no PyO3 / no
// Python feature needed). Coverage mirrors the operator set in `JitTracer`.

use crate::ssa::graph::{BinOp, CmpOp, UnaryOp};

/// Ufunc-name to `BinOp` map.
pub(crate) fn binop_for_ufunc(n: &str) -> Option<BinOp> {
    match n {
        "add"                    => Some(BinOp::Add),
        "subtract"               => Some(BinOp::Sub),
        "multiply"               => Some(BinOp::Mul),
        "true_divide" | "divide" => Some(BinOp::Div),
        "power"                  => Some(BinOp::Pow),
        // Raw C-style fmod (sign of the dividend). `remainder`/`mod` are
        // Python-style FLOORED modulo and lower as a composite instead
        // (`tracer::frontend::floored_mod`).
        "fmod"                   => Some(BinOp::Mod),
        // `float_power` is plain f64 pow for our all-f64 graphs.
        "float_power"            => Some(BinOp::Pow),
        // Rust's `f64::min`/`max` ignore a NaN operand, which is exactly
        // numpy's `fmin`/`fmax` semantics. (`minimum`/`maximum` propagate NaN
        // in numpy; mapping them to the same op diverges only on NaN inputs.)
        "minimum" | "fmin"       => Some(BinOp::Min),
        "maximum" | "fmax"       => Some(BinOp::Max),
        "arctan2"                => Some(BinOp::Atan2),
        "hypot"                  => Some(BinOp::Hypot),
        _ => None,
    }
}

/// Ufunc-name to `UnaryOp` map.
pub(crate) fn unary_op_for_ufunc(n: &str) -> Option<UnaryOp> {
    match n {
        "sin"       => Some(UnaryOp::Sin),
        "cos"       => Some(UnaryOp::Cos),
        "tan"       => Some(UnaryOp::Tan),
        "arctan"    => Some(UnaryOp::Atan),
        "arcsin"    => Some(UnaryOp::Asin),
        "arccos"    => Some(UnaryOp::Acos),
        "sinh"      => Some(UnaryOp::Sinh),
        "cosh"      => Some(UnaryOp::Cosh),
        "tanh"      => Some(UnaryOp::Tanh),
        "arcsinh"   => Some(UnaryOp::Asinh),
        "arccosh"   => Some(UnaryOp::Acosh),
        "arctanh"   => Some(UnaryOp::Atanh),
        "exp"       => Some(UnaryOp::Exp),
        "expm1"     => Some(UnaryOp::Expm1),
        "log"       => Some(UnaryOp::Log),
        "log10"     => Some(UnaryOp::Log10),
        "log2"      => Some(UnaryOp::Log2),
        "log1p"     => Some(UnaryOp::Log1p),
        "absolute" | "fabs" => Some(UnaryOp::Abs),
        "sqrt"      => Some(UnaryOp::Sqrt),
        "cbrt"      => Some(UnaryOp::Cbrt),
        "sign"      => Some(UnaryOp::Sign),
        "floor"     => Some(UnaryOp::Floor),
        "ceil"      => Some(UnaryOp::Ceil),
        "rint"      => Some(UnaryOp::Round),
        "trunc" | "fix" => Some(UnaryOp::Trunc),
        "negative"  => Some(UnaryOp::Neg),
        "erf"       => Some(UnaryOp::Erf),
        "erfc"      => Some(UnaryOp::Erfc),
        "gamma"     => Some(UnaryOp::Tgamma),
        "gammaln"   => Some(UnaryOp::Lgamma),
        _ => None,
    }
}

/// Ufunc-name to `CmpOp` map (mirrors `JitTracer` plus `not_equal`).
pub(crate) fn cmpop_for_ufunc(n: &str) -> Option<CmpOp> {
    match n {
        "greater"       => Some(CmpOp::Gt),
        "greater_equal" => Some(CmpOp::Ge),
        "less"          => Some(CmpOp::Lt),
        "less_equal"    => Some(CmpOp::Le),
        "equal"         => Some(CmpOp::Eq),
        "not_equal"     => Some(CmpOp::Ne),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_names_map_unknown_dont() {
        assert_eq!(binop_for_ufunc("add"), Some(BinOp::Add));
        assert_eq!(binop_for_ufunc("true_divide"), Some(BinOp::Div));
        assert_eq!(binop_for_ufunc("nope"), None);

        assert_eq!(unary_op_for_ufunc("sqrt"), Some(UnaryOp::Sqrt));
        assert_eq!(unary_op_for_ufunc("absolute"), Some(UnaryOp::Abs));
        assert_eq!(unary_op_for_ufunc("nope"), None);

        assert_eq!(cmpop_for_ufunc("not_equal"), Some(CmpOp::Ne));
        assert_eq!(cmpop_for_ufunc("nope"), None);
    }
}
