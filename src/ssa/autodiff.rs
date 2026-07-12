// Symbolic automatic differentiation on SSA Graph.
//
// Computes ∂output/∂input by applying chain rule to graph nodes,
// producing new nodes in the same graph. Hash-consing automatically
// deduplicates shared derivative subexpressions.
//
// After differentiation, the graph can be optimized (constant folding
// removes 0*x and 1*x terms) and lowered to the flat-tape interpreter
// (`InterpretedFn::from_graph`) the same way as the primal graph.

use std::collections::HashMap;
use super::graph::*;

/// Differentiate a node with respect to a target node (typically `Input(i)`).
/// Returns the NodeId of the derivative in the same graph.
/// Uses memoization to avoid recomputing derivatives of shared subexpressions.
pub fn differentiate(
    graph: &mut Graph,
    node: NodeId,
    wrt: NodeId,
    memo: &mut HashMap<NodeId, NodeId>,
) -> NodeId {
    if let Some(&cached) = memo.get(&node) {
        return cached;
    }

    let result = diff_node(graph, node, wrt, memo);
    memo.insert(node, result);
    result
}

fn diff_node(
    graph: &mut Graph,
    node: NodeId,
    wrt: NodeId,
    memo: &mut HashMap<NodeId, NodeId>,
) -> NodeId {
    // Base case: differentiating the target w.r.t. itself = 1
    if node == wrt {
        return graph.constant(1.0);
    }

    // Clone the node to avoid borrow issues
    let n = graph.nodes[node as usize].clone();

    match n {
        // Constants, params and unrelated inputs → 0
        Node::Const(_) | Node::Param(_) => graph.constant(0.0),
        Node::Input(i) => {
            // d(input_i)/d(wrt) = 1 iff this is the differentiation target. The
            // `node == wrt` (NodeId) check above misses optimizer-introduced
            // DUPLICATES of the target input: identity rewrites like `x - 0 → x`
            // / `x * 1 → x` (ssa/optimize.rs) clone the input node's CONTENT into
            // a fresh NodeId, so an input element can appear under several ids
            // before `canonicalize` (tape-lowering only) merges them. Two
            // `Input(i)` nodes are the SAME input, so compare by flat index here —
            // otherwise the gradient through a `… - 0.0` boundary term is silently
            // dropped (rank-deficient Jacobian).
            let same_input = matches!(graph.nodes[wrt as usize], Node::Input(j) if j == i);
            graph.constant(if same_input { 1.0 } else { 0.0 })
        }

        // d/dx(a + b) = da + db
        Node::Binary(BinOp::Add, a, b) => {
            let da = differentiate(graph, a, wrt, memo);
            let db = differentiate(graph, b, wrt, memo);
            graph.binary(BinOp::Add, da, db)
        }

        // d/dx(a - b) = da - db
        Node::Binary(BinOp::Sub, a, b) => {
            let da = differentiate(graph, a, wrt, memo);
            let db = differentiate(graph, b, wrt, memo);
            graph.binary(BinOp::Sub, da, db)
        }

        // d/dx(a * b) = a*db + b*da  (product rule)
        Node::Binary(BinOp::Mul, a, b) => {
            let da = differentiate(graph, a, wrt, memo);
            let db = differentiate(graph, b, wrt, memo);
            let t1 = graph.binary(BinOp::Mul, a, db);
            let t2 = graph.binary(BinOp::Mul, b, da);
            graph.binary(BinOp::Add, t1, t2)
        }

        // d/dx(a / b) = (b*da - a*db) / b^2  (quotient rule)
        Node::Binary(BinOp::Div, a, b) => {
            let da = differentiate(graph, a, wrt, memo);
            let db = differentiate(graph, b, wrt, memo);
            let bda = graph.binary(BinOp::Mul, b, da);
            let adb = graph.binary(BinOp::Mul, a, db);
            let num = graph.binary(BinOp::Sub, bda, adb);
            let b2 = graph.binary(BinOp::Mul, b, b);
            graph.binary(BinOp::Div, num, b2)
        }

        // d/dx(a^b):
        // If b is constant c: d/dx(a^c) = c * a^(c-1) * da  (power rule)
        // General: d/dx(a^b) = a^b * (b' * ln(a) + b * a'/a)
        Node::Binary(BinOp::Pow, a, b) => {
            if let Some(c) = graph.const_value(b) {
                // Power rule: c * a^(c-1) * da
                let da = differentiate(graph, a, wrt, memo);
                let c_node = graph.constant(c);
                let cm1 = graph.constant(c - 1.0);
                let a_cm1 = graph.binary(BinOp::Pow, a, cm1);
                let c_a_cm1 = graph.binary(BinOp::Mul, c_node, a_cm1);
                graph.binary(BinOp::Mul, c_a_cm1, da)
            } else {
                // General case: a^b * (db*ln(a) + b*da/a)
                let da = differentiate(graph, a, wrt, memo);
                let db = differentiate(graph, b, wrt, memo);
                let a_b = graph.binary(BinOp::Pow, a, b);
                let ln_a = graph.unary(UnaryOp::Log, a);
                let db_ln_a = graph.binary(BinOp::Mul, db, ln_a);
                let da_over_a = graph.binary(BinOp::Div, da, a);
                let b_da_a = graph.binary(BinOp::Mul, b, da_over_a);
                let inner = graph.binary(BinOp::Add, db_ln_a, b_da_a);
                graph.binary(BinOp::Mul, a_b, inner)
            }
        }

        // d/dx(a % b): only valid for constant b, where d/dx(a%b) = da.
        // For variable b, the derivative is discontinuous and not well-defined.
        // We check and return 0 if b depends on wrt (safe fallback).
        Node::Binary(BinOp::Mod, a, b) => {
            let db = differentiate(graph, b, wrt, memo);
            let da = differentiate(graph, a, wrt, memo);
            if graph.const_value(db) == Some(0.0) {
                // b is constant w.r.t. wrt → d/dx(a % b) = da
                da
            } else {
                // b varies w.r.t. wrt → not differentiable, return 0
                graph.constant(0.0)
            }
        }

        // d/dx(min(a,b)) ≈ select(a<b, da, db)
        Node::Binary(BinOp::Min, a, b) => {
            let da = differentiate(graph, a, wrt, memo);
            let db = differentiate(graph, b, wrt, memo);
            let cond = graph.cmp(CmpOp::Lt, a, b);
            graph.select(cond, da, db)
        }

        // d/dx(max(a,b)) ≈ select(a>b, da, db)
        Node::Binary(BinOp::Max, a, b) => {
            let da = differentiate(graph, a, wrt, memo);
            let db = differentiate(graph, b, wrt, memo);
            let cond = graph.cmp(CmpOp::Gt, a, b);
            graph.select(cond, da, db)
        }

        // d/dx(atan2(a,b)) = (b*da - a*db) / (a^2 + b^2)
        Node::Binary(BinOp::Atan2, a, b) => {
            let da = differentiate(graph, a, wrt, memo);
            let db = differentiate(graph, b, wrt, memo);
            let bda = graph.binary(BinOp::Mul, b, da);
            let adb = graph.binary(BinOp::Mul, a, db);
            let num = graph.binary(BinOp::Sub, bda, adb);
            let a2 = graph.binary(BinOp::Mul, a, a);
            let b2 = graph.binary(BinOp::Mul, b, b);
            let den = graph.binary(BinOp::Add, a2, b2);
            graph.binary(BinOp::Div, num, den)
        }

        // --- Unary ops: chain rule d/dx(f(a)) = f'(a) * da ---

        // d/dx(-a) = -da
        Node::Unary(UnaryOp::Neg, a) => {
            let da = differentiate(graph, a, wrt, memo);
            graph.unary(UnaryOp::Neg, da)
        }

        // d/dx(sin(a)) = cos(a) * da
        Node::Unary(UnaryOp::Sin, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let cos_a = graph.unary(UnaryOp::Cos, a);
            graph.binary(BinOp::Mul, cos_a, da)
        }

        // d/dx(cos(a)) = -sin(a) * da
        Node::Unary(UnaryOp::Cos, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let sin_a = graph.unary(UnaryOp::Sin, a);
            let neg_sin = graph.unary(UnaryOp::Neg, sin_a);
            graph.binary(BinOp::Mul, neg_sin, da)
        }

        // d/dx(tan(a)) = (1 + tan(a)^2) * da = sec(a)^2 * da
        Node::Unary(UnaryOp::Tan, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let tan_a = graph.unary(UnaryOp::Tan, a);
            let tan2 = graph.binary(BinOp::Mul, tan_a, tan_a);
            let one = graph.constant(1.0);
            let sec2 = graph.binary(BinOp::Add, one, tan2);
            graph.binary(BinOp::Mul, sec2, da)
        }

        // d/dx(atan(a)) = da / (1 + a^2)
        Node::Unary(UnaryOp::Atan, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let a2 = graph.binary(BinOp::Mul, a, a);
            let one = graph.constant(1.0);
            let den = graph.binary(BinOp::Add, one, a2);
            graph.binary(BinOp::Div, da, den)
        }

        // d/dx(sinh(a)) = cosh(a) * da
        Node::Unary(UnaryOp::Sinh, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let cosh_a = graph.unary(UnaryOp::Cosh, a);
            graph.binary(BinOp::Mul, cosh_a, da)
        }

        // d/dx(cosh(a)) = sinh(a) * da
        Node::Unary(UnaryOp::Cosh, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let sinh_a = graph.unary(UnaryOp::Sinh, a);
            graph.binary(BinOp::Mul, sinh_a, da)
        }

        // d/dx(tanh(a)) = (1 - tanh(a)^2) * da
        Node::Unary(UnaryOp::Tanh, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let tanh_a = graph.unary(UnaryOp::Tanh, a);
            let tanh2 = graph.binary(BinOp::Mul, tanh_a, tanh_a);
            let one = graph.constant(1.0);
            let sech2 = graph.binary(BinOp::Sub, one, tanh2);
            graph.binary(BinOp::Mul, sech2, da)
        }

        // d/dx(exp(a)) = exp(a) * da
        Node::Unary(UnaryOp::Exp, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let exp_a = graph.unary(UnaryOp::Exp, a);
            graph.binary(BinOp::Mul, exp_a, da)
        }

        // d/dx(ln(a)) = da / a
        Node::Unary(UnaryOp::Log, a) => {
            let da = differentiate(graph, a, wrt, memo);
            graph.binary(BinOp::Div, da, a)
        }

        // d/dx(log10(a)) = da / (a * ln(10))
        Node::Unary(UnaryOp::Log10, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let ln10 = graph.constant(std::f64::consts::LN_10);
            let a_ln10 = graph.binary(BinOp::Mul, a, ln10);
            graph.binary(BinOp::Div, da, a_ln10)
        }

        // d/dx(|a|) = sign(a) * da
        Node::Unary(UnaryOp::Abs, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let sign_a = graph.unary(UnaryOp::Sign, a);
            graph.binary(BinOp::Mul, sign_a, da)
        }

        // d/dx(sqrt(a)) = da / (2 * sqrt(a))
        Node::Unary(UnaryOp::Sqrt, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let sqrt_a = graph.unary(UnaryOp::Sqrt, a);
            let two = graph.constant(2.0);
            let den = graph.binary(BinOp::Mul, two, sqrt_a);
            graph.binary(BinOp::Div, da, den)
        }

        // sign, floor, ceil, round, trunc have zero derivative (piecewise constant).
        // Comparisons through these are undefined at discontinuities; we return 0
        // which is correct almost everywhere and matches CasADi's convention.
        // RandUniform is a hash of its argument: not meaningfully differentiable,
        // and noise sources never want a gradient flowing back through the key.
        Node::Unary(UnaryOp::Sign, _)
        | Node::Unary(UnaryOp::Floor, _)
        | Node::Unary(UnaryOp::Ceil, _)
        | Node::Unary(UnaryOp::Round, _)
        | Node::Unary(UnaryOp::Trunc, _)
        | Node::Unary(UnaryOp::RandUniform, _) => graph.constant(0.0),

        // d/dx(asin(a)) = da / sqrt(1 - a^2)
        Node::Unary(UnaryOp::Asin, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let a2 = graph.binary(BinOp::Mul, a, a);
            let one = graph.constant(1.0);
            let one_minus_a2 = graph.binary(BinOp::Sub, one, a2);
            let den = graph.unary(UnaryOp::Sqrt, one_minus_a2);
            graph.binary(BinOp::Div, da, den)
        }

        // d/dx(acos(a)) = -da / sqrt(1 - a^2)
        Node::Unary(UnaryOp::Acos, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let a2 = graph.binary(BinOp::Mul, a, a);
            let one = graph.constant(1.0);
            let one_minus_a2 = graph.binary(BinOp::Sub, one, a2);
            let den = graph.unary(UnaryOp::Sqrt, one_minus_a2);
            let quot = graph.binary(BinOp::Div, da, den);
            graph.unary(UnaryOp::Neg, quot)
        }

        // d/dx(asinh(a)) = da / sqrt(1 + a^2)
        Node::Unary(UnaryOp::Asinh, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let a2 = graph.binary(BinOp::Mul, a, a);
            let one = graph.constant(1.0);
            let one_plus_a2 = graph.binary(BinOp::Add, one, a2);
            let den = graph.unary(UnaryOp::Sqrt, one_plus_a2);
            graph.binary(BinOp::Div, da, den)
        }

        // d/dx(acosh(a)) = da / sqrt(a^2 - 1)
        Node::Unary(UnaryOp::Acosh, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let a2 = graph.binary(BinOp::Mul, a, a);
            let one = graph.constant(1.0);
            let a2_minus_1 = graph.binary(BinOp::Sub, a2, one);
            let den = graph.unary(UnaryOp::Sqrt, a2_minus_1);
            graph.binary(BinOp::Div, da, den)
        }

        // d/dx(atanh(a)) = da / (1 - a^2)
        Node::Unary(UnaryOp::Atanh, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let a2 = graph.binary(BinOp::Mul, a, a);
            let one = graph.constant(1.0);
            let den = graph.binary(BinOp::Sub, one, a2);
            graph.binary(BinOp::Div, da, den)
        }

        // d/dx(log2(a)) = da / (a * ln(2))
        Node::Unary(UnaryOp::Log2, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let ln2 = graph.constant(std::f64::consts::LN_2);
            let a_ln2 = graph.binary(BinOp::Mul, a, ln2);
            graph.binary(BinOp::Div, da, a_ln2)
        }

        // d/dx(log1p(a)) = da / (1 + a)
        Node::Unary(UnaryOp::Log1p, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let one = graph.constant(1.0);
            let den = graph.binary(BinOp::Add, one, a);
            graph.binary(BinOp::Div, da, den)
        }

        // d/dx(expm1(a)) = exp(a) * da = (expm1(a) + 1) * da
        Node::Unary(UnaryOp::Expm1, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let exp_a = graph.unary(UnaryOp::Exp, a);
            graph.binary(BinOp::Mul, exp_a, da)
        }

        // d/dx(cbrt(a)) = da / (3 * cbrt(a)^2)
        Node::Unary(UnaryOp::Cbrt, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let cbrt_a = graph.unary(UnaryOp::Cbrt, a);
            let cbrt2 = graph.binary(BinOp::Mul, cbrt_a, cbrt_a);
            let three = graph.constant(3.0);
            let den = graph.binary(BinOp::Mul, three, cbrt2);
            graph.binary(BinOp::Div, da, den)
        }

        // d/dx(erf(a)) = (2/sqrt(pi)) * exp(-a^2) * da
        Node::Unary(UnaryOp::Erf, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let a2 = graph.binary(BinOp::Mul, a, a);
            let neg_a2 = graph.unary(UnaryOp::Neg, a2);
            let exp_neg_a2 = graph.unary(UnaryOp::Exp, neg_a2);
            let c = graph.constant(2.0 / std::f64::consts::PI.sqrt());
            let pref = graph.binary(BinOp::Mul, c, exp_neg_a2);
            graph.binary(BinOp::Mul, pref, da)
        }

        // d/dx(erfc(a)) = -(2/sqrt(pi)) * exp(-a^2) * da
        Node::Unary(UnaryOp::Erfc, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let a2 = graph.binary(BinOp::Mul, a, a);
            let neg_a2 = graph.unary(UnaryOp::Neg, a2);
            let exp_neg_a2 = graph.unary(UnaryOp::Exp, neg_a2);
            let c = graph.constant(-2.0 / std::f64::consts::PI.sqrt());
            let pref = graph.binary(BinOp::Mul, c, exp_neg_a2);
            graph.binary(BinOp::Mul, pref, da)
        }

        // d/dx lgamma(a) = ψ(a) · da  where ψ is the digamma function.
        // d/dx tgamma(a) = tgamma(a) · ψ(a) · da.
        // ψ is evaluated at runtime via an internal `UnaryOp::Digamma` op
        // (not exposed as a user-level ufunc).
        Node::Unary(UnaryOp::Lgamma, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let psi = graph.unary(UnaryOp::Digamma, a);
            graph.binary(BinOp::Mul, psi, da)
        }
        Node::Unary(UnaryOp::Tgamma, a) => {
            let da = differentiate(graph, a, wrt, memo);
            let psi = graph.unary(UnaryOp::Digamma, a);
            let g   = graph.unary(UnaryOp::Tgamma, a);
            let gpsi = graph.binary(BinOp::Mul, g, psi);
            graph.binary(BinOp::Mul, gpsi, da)
        }
        // Digamma itself is typically only produced by the AD above; it's
        // symbolically differentiable (trigamma), but we don't need higher-
        // order derivatives, so we treat it as non-differentiable here.
        Node::Unary(UnaryOp::Digamma, _) => graph.constant(0.0),

        // d/dx(hypot(a,b)) = (a*da + b*db) / hypot(a,b)
        Node::Binary(BinOp::Hypot, a, b) => {
            let da = differentiate(graph, a, wrt, memo);
            let db = differentiate(graph, b, wrt, memo);
            let ada = graph.binary(BinOp::Mul, a, da);
            let bdb = graph.binary(BinOp::Mul, b, db);
            let num = graph.binary(BinOp::Add, ada, bdb);
            let h = graph.binary(BinOp::Hypot, a, b);
            graph.binary(BinOp::Div, num, h)
        }

        // d/dx(cond ? a : b) = cond ? da : db  (pass-through)
        Node::Select(c, a, b) => {
            let da = differentiate(graph, a, wrt, memo);
            let db = differentiate(graph, b, wrt, memo);
            graph.select(c, da, db)
        }

        // Comparisons are piecewise constant → derivative is 0
        Node::Cmp(_, _, _) => graph.constant(0.0),

        // d/dx(fma(a,b,c)) = d/dx(a*b+c) = a*db + b*da + dc
        Node::Fma(a, b, c) => {
            let da = differentiate(graph, a, wrt, memo);
            let db = differentiate(graph, b, wrt, memo);
            let dc = differentiate(graph, c, wrt, memo);
            let adb = graph.binary(BinOp::Mul, a, db);
            let bda = graph.binary(BinOp::Mul, b, da);
            let sum = graph.binary(BinOp::Add, adb, bda);
            graph.binary(BinOp::Add, sum, dc)
        }

        // Reductions. Sum is linear; product uses the generalized product
        // rule; min/max flow the derivative of whichever operand is selected
        // (subgradient, matching the scalar Min/Max convention above).
        Node::Reduce(rop, args) => match rop {
            // d/dx Σ aᵢ = Σ daᵢ
            ReduceOp::Sum => {
                let mut dargs = Vec::with_capacity(args.len());
                for &a in &args {
                    dargs.push(differentiate(graph, a, wrt, memo));
                }
                reduce_or_collapse(graph, ReduceOp::Sum, dargs)
            }
            // d/dx Π aᵢ = Σᵢ daᵢ · Π_{j≠i} aⱼ
            ReduceOp::Product => {
                let mut terms = Vec::with_capacity(args.len());
                for i in 0..args.len() {
                    let da_i = differentiate(graph, args[i], wrt, memo);
                    let others: Vec<NodeId> = args
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, &a)| a)
                        .collect();
                    let prod_others = reduce_or_collapse_with_identity(graph, ReduceOp::Product, others, 1.0);
                    terms.push(graph.binary(BinOp::Mul, da_i, prod_others));
                }
                reduce_or_collapse(graph, ReduceOp::Sum, terms)
            }
            // d/dx min/max: subgradient of the running extremum.
            ReduceOp::Min | ReduceOp::Max => {
                let (cmp_op, bin_op) = match rop {
                    ReduceOp::Min => (CmpOp::Lt, BinOp::Min),
                    _ => (CmpOp::Gt, BinOp::Max),
                };
                if args.is_empty() {
                    return graph.constant(0.0);
                }
                let mut running = args[0];
                let mut d = differentiate(graph, args[0], wrt, memo);
                for &ak in &args[1..] {
                    let dak = differentiate(graph, ak, wrt, memo);
                    // ak more extreme than the running value → it wins, take dak.
                    let cond = graph.cmp(cmp_op, ak, running);
                    d = graph.select(cond, dak, d);
                    running = graph.binary(bin_op, running, ak);
                }
                d
            }
        },

        // d/dx Σ aᵢ·bᵢ = Σ (aᵢ·dbᵢ + daᵢ·bᵢ). Expanded into a plain Add-chain
        // of products rather than two derivative Dots, so the optimizer folds
        // the (very common) zero terms — e.g. constant matrix coefficients give
        // daᵢ = 0 — and re-fuses survivors into FMAs, keeping Jacobians clean.
        Node::Dot(a, b) => {
            if a.is_empty() {
                return graph.constant(0.0);
            }
            let mut acc: Option<NodeId> = None;
            for (&ai, &bi) in a.iter().zip(b.iter()) {
                let dai = differentiate(graph, ai, wrt, memo);
                let dbi = differentiate(graph, bi, wrt, memo);
                let adb = graph.binary(BinOp::Mul, ai, dbi);
                let bda = graph.binary(BinOp::Mul, dai, bi);
                let term = graph.binary(BinOp::Add, adb, bda);
                acc = Some(match acc {
                    None => term,
                    Some(prev) => graph.binary(BinOp::Add, prev, term),
                });
            }
            acc.unwrap_or_else(|| graph.constant(0.0))
        }
    }
}

/// Build a `Reduce` node, collapsing the degenerate 0/1-operand cases so the
/// graph stays free of trivial reductions (which the tape would otherwise
/// have to walk). Empty → identity constant; single → the operand itself.
fn reduce_or_collapse(graph: &mut Graph, op: ReduceOp, args: Vec<NodeId>) -> NodeId {
    let identity = match op {
        ReduceOp::Sum => 0.0,
        ReduceOp::Product => 1.0,
        ReduceOp::Min => f64::INFINITY,
        ReduceOp::Max => f64::NEG_INFINITY,
    };
    reduce_or_collapse_with_identity(graph, op, args, identity)
}

fn reduce_or_collapse_with_identity(graph: &mut Graph, op: ReduceOp, args: Vec<NodeId>, identity: f64) -> NodeId {
    match args.len() {
        0 => graph.constant(identity),
        1 => args[0],
        _ => graph.add(Node::Reduce(op, args)),
    }
}

/// Build a full Jacobian graph of all outputs w.r.t. all inputs (flat).
/// Returns a graph with `n_out * n_inputs` outputs in row-major layout:
/// `[∂f0/∂in0, ..., ∂f0/∂in{N-1}, ∂f1/∂in0, ...]`.
pub fn jacobian(graph: &Graph) -> Graph {
    jacobian_wrt_flat_range(graph, 0, graph.signature.total_size)
}

/// Differentiate w.r.t. a named slot.  Useful for multi-slot signatures:
/// `jacobian_wrt_slot(graph, "z")` differentiates all outputs w.r.t. the
/// `z` slot only.  Returns `None` if the slot isn't in the signature.
pub fn jacobian_wrt_slot(graph: &Graph, slot_name: &str) -> Option<Graph> {
    let slot = graph.signature.slot(slot_name)?;
    Some(jacobian_wrt_flat_range(graph, slot.offset, slot.offset + slot.size))
}

/// Core: differentiate w.r.t. a contiguous range of flat input indices.
pub(crate) fn jacobian_wrt_flat_range(graph: &Graph, start: usize, end: usize) -> Graph {
    let mut jac = graph.clone();
    let original_outputs = jac.outputs.clone();
    jac.outputs.clear();

    let wrt_nodes: Vec<NodeId> = (start..end)
        .map(|i| {
            let node = Node::Input(i as u32);
            if let Some(&id) = jac.dedup.get(&node) {
                id
            } else {
                jac.add(node)
            }
        })
        .collect();

    // Sparsity-gated construction: a Jacobian entry ∂out/∂x is identically zero
    // whenever `out`'s expression DAG does not even reach the input `x`. For a
    // banded/block-structured system (the common case: a chain or ring of state
    // blocks) only O(n) of the n_out·n_state entries are structurally nonzero,
    // so differentiating every pair is O(n²) wasted symbolic work. Instead, per
    // output we mark the state inputs it actually reaches (one cheap DFS) and
    // only run the heavy `differentiate` on those pairs; the rest emit a shared
    // constant-zero leaf. The dense row-major n_out·size output layout is
    // preserved exactly, so downstream lowering is unchanged.
    let size = end - start;
    let zero = jac.constant(0.0);
    let mut reached = vec![false; size];
    for &out_id in &original_outputs {
        for r in reached.iter_mut() {
            *r = false;
        }
        collect_reached_inputs(&jac, out_id, start, size, &mut reached);
        for (k, &x_id) in wrt_nodes.iter().enumerate() {
            if reached[k] {
                let mut memo = HashMap::new();
                let deriv = differentiate(&mut jac, out_id, x_id, &mut memo);
                jac.outputs.push(deriv);
            } else {
                jac.outputs.push(zero);
            }
        }
    }

    jac
}

// ======================================================================================
// Jacobian introspection (Step 2 foundation): query the AD-derived Jacobian's
// structure without evaluating it. The implicit solver uses these to pick a
// linear-solve strategy: constant -> factor once per dt; sparse -> sparse LU.
// ======================================================================================

/// `true` if the Jacobian of all outputs w.r.t. the `slot_name` slot does not
/// depend on the state in that slot, i.e. it is state-independent (linear
/// blocks: `∂(Ax + Bu)/∂x = A` is constant). After optimization a constant
/// Jacobian reaches no `Input` from the slot. Returns `false` if the slot is
/// absent (conservative: treat as non-constant).
pub fn jacobian_is_constant(graph: &Graph, slot_name: &str) -> bool {
    let slot = match graph.signature.slot(slot_name) {
        Some(s) => s,
        None => return false,
    };
    let (start, end) = (slot.offset, slot.offset + slot.size);
    let mut jac = match jacobian_wrt_slot(graph, slot_name) {
        Some(g) => g,
        None => return false,
    };
    super::optimize::optimize(&mut jac);
    let outputs = jac.outputs.clone();
    !reaches_input_range(&jac, &outputs, start, end)
}

/// Sparse `∂outputs/∂slot_name`: the Jacobian graph optimized to a fixpoint, with
/// its outputs reduced to only the structurally-nonzero entries, plus their
/// coordinate `(rows, cols)` in the dense row-major `n_out × slot_size` layout.
/// `None` if the slot is absent.
///
/// Interpreting the returned graph yields exactly `rows.len()` values, in pattern
/// order, so the caller can carry a sparse Jacobian (pattern fixed, values per
/// step) instead of materialising a dense matrix full of shared zeros. Backs the
/// compiled blocks' [`crate::solvers::solver::Jacobian::Sparse`].
pub fn jacobian_sparse_wrt_slot(
    graph: &Graph,
    slot_name: &str,
) -> Option<(Graph, Vec<u32>, Vec<u32>)> {
    let slot_size = graph.signature.slot(slot_name)?.size;
    let mut jac = jacobian_wrt_slot(graph, slot_name)?;
    for _ in 0..crate::constants::COMPILE_OPTIMIZE_MAX_PASSES {
        if !super::optimize::optimize(&mut jac) {
            break;
        }
    }
    let dense_outputs = jac.outputs.clone();
    let (mut rows, mut cols, mut nz) = (Vec::new(), Vec::new(), Vec::new());
    for (idx, &o) in dense_outputs.iter().enumerate() {
        if !jac.is_const(o, 0.0) {
            rows.push((idx / slot_size) as u32);
            cols.push((idx % slot_size) as u32);
            nz.push(o);
        }
    }
    jac.outputs = nz;
    Some((jac, rows, cols))
}

/// Build the *optimized* Jacobian graph of `graph`'s outputs w.r.t. `slot_name`
/// and report whether it is globally constant — in ONE construction.
///
/// The static compiler needs both the lowerable Jacobian tape and the LTI flag
/// (whether `∂outputs/∂slot` is a globally constant matrix). Computing them
/// separately builds the (O(n²)) symbolic Jacobian *twice* and optimizes it
/// twice. This folds the two into a single build + fixpoint-optimize, then
/// derives `jac_const` from the already-optimized graph (it reaches no `Input`
/// ⇒ constant). Returns `(optimized_jacobian, jac_const)`, or `None` if the slot
/// is absent.
pub fn jacobian_wrt_slot_optimized(graph: &Graph, slot_name: &str) -> Option<(Graph, bool)> {
    let total = graph.signature.total_size;
    let mut jac = jacobian_wrt_slot(graph, slot_name)?;
    for _ in 0..crate::constants::COMPILE_OPTIMIZE_MAX_PASSES {
        if !super::optimize::optimize(&mut jac) {
            break;
        }
    }
    let outputs = jac.outputs.clone();
    let jac_const = !reaches_input_range(&jac, &outputs, 0, total);
    Some((jac, jac_const))
}

/// `true` if any output algebraically reads the `slot_name` slot, i.e. the block
/// has direct feedthrough from that input to some output. The SSA-derived
/// analogue of the hand-set `BlockRole.is_alg`: pure reachability over the
/// optimized output region (so `*0` terms, e.g. `D = 0` in `y = Cx + Du`, fold
/// away and do not count as feedthrough). Works for every op via pure
/// reachability. `false` if the slot is absent.
pub fn has_feedthrough(graph: &Graph, slot_name: &str) -> bool {
    let slot = match graph.signature.slot(slot_name) {
        Some(s) => s,
        None => return false,
    };
    let (start, end) = (slot.offset, slot.offset + slot.size);
    let mut g = graph.clone();
    for _ in 0..crate::constants::COMPILE_OPTIMIZE_MAX_PASSES {
        if !super::optimize::optimize(&mut g) {
            break;
        }
    }
    let outputs = g.outputs.clone();
    reaches_input_range(&g, &outputs, start, end)
}

/// The full MIMO direct-feedthrough pattern of an output region w.r.t. its input
/// slot: a flat row-major `n_out × slot_size` mask, `true` where output `o`
/// structurally reads input element `j` (after optimization). For an LTI block
/// `y = Cx + Du` this is the nonzero pattern of `D`. Derived by per-output
/// reachability, so it is exact for any op. `None` if the slot is absent.
pub fn feedthrough_pattern(graph: &Graph, slot_name: &str) -> Option<Vec<bool>> {
    let slot = graph.signature.slot(slot_name)?;
    let (start, size) = (slot.offset, slot.size);
    let mut g = graph.clone();
    for _ in 0..crate::constants::COMPILE_OPTIMIZE_MAX_PASSES {
        if !super::optimize::optimize(&mut g) {
            break;
        }
    }
    let outputs = g.outputs.clone();
    let mut mask = vec![false; outputs.len() * size];
    for (oi, &out) in outputs.iter().enumerate() {
        collect_reached_inputs(&g, out, start, size, &mut mask[oi * size..(oi + 1) * size]);
    }
    Some(mask)
}

/// DFS from one output node, marking `mask[i - start]` for each reached
/// `Input(i)` with `start <= i < start + size`.
fn collect_reached_inputs(g: &Graph, root: NodeId, start: usize, size: usize, mask: &mut [bool]) {
    let mut seen = vec![false; g.nodes.len()];
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        let idx = n as usize;
        if seen[idx] {
            continue;
        }
        seen[idx] = true;
        if let Node::Input(i) = &g.nodes[idx] {
            let i = *i as usize;
            if i >= start && i < start + size {
                mask[i - start] = true;
            }
        }
        g.nodes[idx].for_each_child(|c| stack.push(c));
    }
}

/// Does any node in `roots` transitively read an `Input(i)` with `start <= i < end`?
fn reaches_input_range(g: &Graph, roots: &[NodeId], start: usize, end: usize) -> bool {
    let mut seen = vec![false; g.nodes.len()];
    let mut stack: Vec<NodeId> = roots.to_vec();
    while let Some(n) = stack.pop() {
        let idx = n as usize;
        if seen[idx] {
            continue;
        }
        seen[idx] = true;
        if let Node::Input(i) = &g.nodes[idx] {
            let i = *i as usize;
            if i >= start && i < end {
                return true;
            }
        }
        g.nodes[idx].for_each_child(|c| stack.push(c));
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobian_introspection_constant() {
        // Linear: y0 = 2*x0 (no x1), y1 = 4*x0 + 5*x1. J = [[2,0],[4,5]] -> constant.
        let mut g = Graph::with_single_input("x", 2);
        let x0 = g.add(Node::Input(0));
        let x1 = g.add(Node::Input(1));
        let (a00, a10, a11) = (g.constant(2.0), g.constant(4.0), g.constant(5.0));
        let r0 = g.binary(BinOp::Mul, a00, x0);
        let m10 = g.binary(BinOp::Mul, a10, x0);
        let m11 = g.binary(BinOp::Mul, a11, x1);
        let r1 = g.binary(BinOp::Add, m10, m11);
        g.outputs = vec![r0, r1];

        assert!(jacobian_is_constant(&g, "x"), "linear Jacobian must be constant");

        // Nonlinear: y = x0*x1 -> J = [x1, x0], state-dependent.
        let mut h = Graph::with_single_input("x", 2);
        let h0 = h.add(Node::Input(0));
        let h1 = h.add(Node::Input(1));
        let p = h.binary(BinOp::Mul, h0, h1);
        h.outputs = vec![p];
        assert!(!jacobian_is_constant(&h, "x"), "x0*x1 Jacobian is state-dependent");
    }

    #[test]
    fn jacobian_sparse_reconstructs_dense() {
        // A coupled 3-state RHS with a genuinely sparse Jacobian (5 of 9 nonzero):
        //   f0 = x0*x1   -> ∂x0=x1, ∂x1=x0
        //   f1 = sin(x2) -> ∂x2=cos(x2)
        //   f2 = x0 + x2 -> ∂x0=1, ∂x2=1
        let mut g = Graph::with_single_input("x", 3);
        let (x0, x1, x2) = (g.input(0), g.input(1), g.input(2));
        let f0 = g.binary(BinOp::Mul, x0, x1);
        let f1 = g.unary(UnaryOp::Sin, x2);
        let f2 = g.binary(BinOp::Add, x0, x2);
        g.outputs = vec![f0, f1, f2];

        let dense_g = jacobian(&g);
        let (sparse_g, rows, cols) = jacobian_sparse_wrt_slot(&g, "x").unwrap();

        let xin = [0.7, -1.3, 0.4];
        let dense = dense_g.interpret(&[&xin], &[]);
        let vals = sparse_g.interpret(&[&xin], &[]);

        assert_eq!(vals.len(), 5, "5 of 9 entries are structurally nonzero");
        assert_eq!(rows.len(), vals.len());
        assert_eq!(cols.len(), vals.len());

        // Scatter the sparse triples back into a dense matrix; it must match the
        // full dense Jacobian entry-for-entry (zeros included).
        let n = 3;
        let mut recon = vec![0.0; n * n];
        for k in 0..vals.len() {
            recon[rows[k] as usize * n + cols[k] as usize] = vals[k];
        }
        for i in 0..n * n {
            assert!(
                (recon[i] - dense[i]).abs() < 1e-12,
                "entry {i}: sparse {} != dense {}", recon[i], dense[i]
            );
        }
    }

    #[test]
    fn feedthrough_from_ssa_matches_structure() {
        // Output region y over slots x(2), u(2):
        //   y0 = x0            (no input -> no feedthrough)
        //   y1 = 2*u0 + x1     (feedthrough on u0 only)
        let mut g = Graph::new(InputSignature::from_named_sizes([("x", 2), ("u", 2)]));
        let x0 = g.add(Node::Input(0));
        let x1 = g.add(Node::Input(1));
        let u0 = g.add(Node::Input(2));
        let two = g.constant(2.0);
        let m = g.binary(BinOp::Mul, two, u0);
        let y1 = g.binary(BinOp::Add, m, x1);
        g.outputs = vec![x0, y1];

        assert!(has_feedthrough(&g, "u"), "y1 reads u -> block has feedthrough");
        // n_out=2 x u_size=2, row-major: y0 reads no u; y1 reads u0 only.
        assert_eq!(feedthrough_pattern(&g, "u").unwrap(),
                   vec![false, false, true, false]);

        // D = 0 case: y = x0 + 0*u0 must fold away the input -> no feedthrough.
        let mut h = Graph::new(InputSignature::from_named_sizes([("x", 1), ("u", 1)]));
        let hx = h.add(Node::Input(0));
        let hu = h.add(Node::Input(1));
        let zero = h.constant(0.0);
        let z = h.binary(BinOp::Mul, zero, hu);
        let hy = h.binary(BinOp::Add, hx, z);
        h.outputs = vec![hy];
        assert!(!has_feedthrough(&h, "u"), "0*u folds away -> no feedthrough");
        assert_eq!(feedthrough_pattern(&h, "u").unwrap(), vec![false]);
    }

    #[test]
    fn jacobian_wrt_slot_optimized_reports_global_constant() {
        // y = t * x0: ∂y/∂x = t. State-independent but time-dependent, so it is
        // NOT a globally constant matrix (the optimized Jacobian reaches `t`).
        let mut g = Graph::new(InputSignature::from_named_sizes([("x", 1), ("t", 1)]));
        let x0 = g.add(Node::Input(0));
        let t = g.add(Node::Input(1));
        let y = g.binary(BinOp::Mul, t, x0);
        g.outputs = vec![y];
        assert!(jacobian_is_constant(&g, "x"), "∂(t·x)/∂x is state-independent");
        let (_, gconst) = jacobian_wrt_slot_optimized(&g, "x").unwrap();
        assert!(!gconst, "∂(t·x)/∂x = t depends on time");

        // y = 3*x0 + 7*x1: ∂y/∂x = [3, 7], depends on nothing -> globally constant.
        let mut h = Graph::new(InputSignature::from_named_sizes([("x", 2), ("t", 1)]));
        let h0 = h.add(Node::Input(0));
        let h1 = h.add(Node::Input(1));
        let (c3, c7) = (h.constant(3.0), h.constant(7.0));
        let m0 = h.binary(BinOp::Mul, c3, h0);
        let m1 = h.binary(BinOp::Mul, c7, h1);
        let y = h.binary(BinOp::Add, m0, m1);
        h.outputs = vec![y];
        let (_, hconst) = jacobian_wrt_slot_optimized(&h, "x").unwrap();
        assert!(hconst, "constant-coefficient Jacobian is global");
    }

    #[test]
    fn test_diff_constant() {
        let mut g = Graph::with_single_input("x", 1);
        let x = g.add(Node::Input(0));
        let c = g.constant(5.0);
        g.outputs.push(c);

        let mut memo = HashMap::new();
        let dc = differentiate(&mut g, c, x, &mut memo);
        assert!(g.is_const(dc, 0.0));
    }

    #[test]
    fn test_diff_identity() {
        // d/dx(x) = 1
        let mut g = Graph::with_single_input("x", 1);
        let x = g.add(Node::Input(0));
        let mut memo = HashMap::new();
        let dx = differentiate(&mut g, x, x, &mut memo);
        assert!(g.is_const(dx, 1.0));
    }

    #[test]
    fn test_diff_linear() {
        // f(x) = 3*x + 2  → df/dx = 3
        let mut g = Graph::with_single_input("x", 1);
        let x = g.add(Node::Input(0));
        let three = g.constant(3.0);
        let two = g.constant(2.0);
        let three_x = g.binary(BinOp::Mul, three, x);
        let f = g.binary(BinOp::Add, three_x, two);

        let mut memo = HashMap::new();
        let df = differentiate(&mut g, f, x, &mut memo);

        // After optimization, should be 3.0
        super::super::optimize::optimize(&mut g);
        let _result = g.interpret(&[&[99.0]], &[]);
        // df node should evaluate to 3 regardless of x
        g.outputs = vec![df];
        let df_result = g.interpret(&[&[99.0]], &[]);
        assert!((df_result[0] - 3.0).abs() < 1e-10, "d(3x+2)/dx should be 3, got {}", df_result[0]);
    }

    #[test]
    fn test_diff_quadratic() {
        // f(x) = x^2  → df/dx = 2x
        let mut g = Graph::with_single_input("x", 1);
        let x = g.add(Node::Input(0));
        let two = g.constant(2.0);
        let f = g.binary(BinOp::Pow, x, two);

        let mut memo = HashMap::new();
        let df = differentiate(&mut g, f, x, &mut memo);
        g.outputs = vec![df];

        super::super::optimize::optimize(&mut g);

        // At x=3: df/dx = 2*3 = 6
        let r = g.interpret(&[&[3.0]], &[]);
        assert!((r[0] - 6.0).abs() < 1e-10, "d(x^2)/dx at x=3 should be 6, got {}", r[0]);

        // At x=-2: df/dx = 2*(-2) = -4
        let r2 = g.interpret(&[&[-2.0]], &[]);
        assert!((r2[0] - (-4.0)).abs() < 1e-10);
    }

    #[test]
    fn test_diff_product() {
        // f(x0, x1) = x0 * x1  → ∂f/∂x0 = x1, ∂f/∂x1 = x0
        let mut g = Graph::with_single_input("x", 2);
        let x0 = g.add(Node::Input(0));
        let x1 = g.add(Node::Input(1));
        let f = g.binary(BinOp::Mul, x0, x1);

        let mut memo0 = HashMap::new();
        let df_dx0 = differentiate(&mut g, f, x0, &mut memo0);
        let mut memo1 = HashMap::new();
        let df_dx1 = differentiate(&mut g, f, x1, &mut memo1);

        g.outputs = vec![df_dx0, df_dx1];
        super::super::optimize::optimize(&mut g);

        // At (3, 5): ∂f/∂x0 = 5, ∂f/∂x1 = 3
        let r = g.interpret(&[&[3.0, 5.0]], &[]);
        assert!((r[0] - 5.0).abs() < 1e-10, "∂(x0*x1)/∂x0 = x1 = 5, got {}", r[0]);
        assert!((r[1] - 3.0).abs() < 1e-10, "∂(x0*x1)/∂x1 = x0 = 3, got {}", r[1]);
    }

    #[test]
    fn test_diff_sin() {
        // f(x) = sin(x) → df/dx = cos(x)
        let mut g = Graph::with_single_input("x", 1);
        let x = g.add(Node::Input(0));
        let f = g.unary(UnaryOp::Sin, x);

        let mut memo = HashMap::new();
        let df = differentiate(&mut g, f, x, &mut memo);
        g.outputs = vec![df];

        let r = g.interpret(&[&[1.0]], &[]);
        assert!((r[0] - 1.0_f64.cos()).abs() < 1e-10);
    }

    #[test]
    fn test_diff_exp() {
        // f(x) = exp(x) → df/dx = exp(x)
        let mut g = Graph::with_single_input("x", 1);
        let x = g.add(Node::Input(0));
        let f = g.unary(UnaryOp::Exp, x);

        let mut memo = HashMap::new();
        let df = differentiate(&mut g, f, x, &mut memo);
        g.outputs = vec![df];

        let r = g.interpret(&[&[2.0]], &[]);
        assert!((r[0] - 2.0_f64.exp()).abs() < 1e-10);
    }

    #[test]
    fn test_diff_chain_rule() {
        // f(x) = sin(x^2) → df/dx = cos(x^2) * 2x
        let mut g = Graph::with_single_input("x", 1);
        let x = g.add(Node::Input(0));
        let two = g.constant(2.0);
        let x2 = g.binary(BinOp::Pow, x, two);
        let f = g.unary(UnaryOp::Sin, x2);

        let mut memo = HashMap::new();
        let df = differentiate(&mut g, f, x, &mut memo);
        g.outputs = vec![df];
        super::super::optimize::optimize(&mut g);

        // At x=1.5: df/dx = cos(2.25) * 3.0
        let expected = 1.5_f64.powi(2).cos() * 2.0 * 1.5;
        let r = g.interpret(&[&[1.5]], &[]);
        assert!((r[0] - expected).abs() < 1e-8, "got {} expected {}", r[0], expected);
    }

    #[test]
    fn test_jacobian_robertson() {
        // Robertson ODE:
        // f0 = -a*x0 + b*x1*x2
        // f1 =  a*x0 - b*x1*x2 - c*x1^2
        // f2 =  c*x1^2
        //
        // Jacobian:
        // J = [[-a,      b*x2,       b*x1     ],
        //      [ a, -b*x2-2c*x1, -b*x1        ],
        //      [ 0,      2c*x1,       0        ]]

        let mut g = Graph::with_single_input("x", 3);
        let x0 = g.add(Node::Input(0));
        let x1 = g.add(Node::Input(1));
        let x2 = g.add(Node::Input(2));
        let a = g.constant(0.04);
        let b = g.constant(1e4);
        let c = g.constant(3e7);
        let two = g.constant(2.0);

        // f0 = -a*x0 + b*x1*x2
        let ax0 = g.binary(BinOp::Mul, a, x0);
        let neg_ax0 = g.unary(UnaryOp::Neg, ax0);
        let bx1 = g.binary(BinOp::Mul, b, x1);
        let bx1x2 = g.binary(BinOp::Mul, bx1, x2);
        let f0 = g.binary(BinOp::Add, neg_ax0, bx1x2);

        // f1 = a*x0 - b*x1*x2 - c*x1^2
        let x1_2 = g.binary(BinOp::Pow, x1, two);
        let cx1_2 = g.binary(BinOp::Mul, c, x1_2);
        let f1_a = g.binary(BinOp::Sub, ax0, bx1x2);
        let f1 = g.binary(BinOp::Sub, f1_a, cx1_2);

        // f2 = c*x1^2
        let f2 = cx1_2;

        g.outputs = vec![f0, f1, f2];

        let jac = jacobian(&g);
        assert_eq!(jac.outputs.len(), 9); // 3 outputs × 3 inputs

        // Optimize the Jacobian graph
        let mut jac_opt = jac;
        super::super::optimize::optimize(&mut jac_opt);

        // Evaluate at x = [1.0, 0.5, 0.3]
        let x_test = [1.0, 0.5, 0.3];
        let j = jac_opt.interpret(&[&x_test], &[]);

        // Expected Jacobian values
        let a_v = 0.04; let b_v = 1e4; let c_v = 3e7;
        let expected = [
            -a_v,                           // ∂f0/∂x0
            b_v * x_test[2],                // ∂f0/∂x1
            b_v * x_test[1],                // ∂f0/∂x2
            a_v,                            // ∂f1/∂x0
            -b_v * x_test[2] - 2.0 * c_v * x_test[1], // ∂f1/∂x1
            -b_v * x_test[1],               // ∂f1/∂x2
            0.0,                            // ∂f2/∂x0
            2.0 * c_v * x_test[1],          // ∂f2/∂x1
            0.0,                            // ∂f2/∂x2
        ];

        for (i, (got, exp)) in j.iter().zip(expected.iter()).enumerate() {
            let tol = 1e-6 * exp.abs().max(1.0);
            assert!((got - exp).abs() < tol,
                "J[{}]: got {} expected {}", i, got, exp);
        }
    }

    #[test]
    fn test_jacobian_compiled() {
        // Same Robertson but lower the Jacobian through the flat-tape interpreter
        let mut g = Graph::with_single_input("x", 3);
        let x0 = g.add(Node::Input(0));
        let x1 = g.add(Node::Input(1));
        let x2 = g.add(Node::Input(2));
        let a = g.constant(0.04);
        let b = g.constant(1e4);
        let c = g.constant(3e7);
        let two = g.constant(2.0);

        let ax0 = g.binary(BinOp::Mul, a, x0);
        let neg_ax0 = g.unary(UnaryOp::Neg, ax0);
        let bx1 = g.binary(BinOp::Mul, b, x1);
        let bx1x2 = g.binary(BinOp::Mul, bx1, x2);
        let f0 = g.binary(BinOp::Add, neg_ax0, bx1x2);

        let x1_2 = g.binary(BinOp::Pow, x1, two);
        let cx1_2 = g.binary(BinOp::Mul, c, x1_2);
        let f1_a = g.binary(BinOp::Sub, ax0, bx1x2);
        let f1 = g.binary(BinOp::Sub, f1_a, cx1_2);

        let f2 = cx1_2;
        g.outputs = vec![f0, f1, f2];

        let mut jac = jacobian(&g);
        super::super::optimize::optimize(&mut jac);

        {
            let interp = super::super::tape::InterpretedFn::from_graph(jac);
            let j = interp.call(&[&[1.0, 0.5, 0.3]]);

            let a_v = 0.04; let b_v = 1e4; let c_v = 3e7;
            assert!((j[0] - (-a_v)).abs() < 1e-6);
            assert!((j[1] - b_v * 0.3).abs() < 1e-2);
            assert!((j[4] - (-b_v * 0.3 - 2.0 * c_v * 0.5)).abs() / (b_v * 0.3 + 2.0 * c_v * 0.5) < 1e-6);
            assert!((j[7] - 2.0 * c_v * 0.5).abs() < 1e-2);
            assert!((j[6]).abs() < 1e-10); // ∂f2/∂x0 = 0
            assert!((j[8]).abs() < 1e-10); // ∂f2/∂x2 = 0
        }
    }

    #[test]
    fn test_diff_fuzz_vs_finite_diff() {
        // Fuzz: random expressions, verify symbolic derivative matches finite differences
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                self.0 >> 33
            }
        }

        for seed in 0..100 {
            let mut rng = Rng(seed);
            let mut g = Graph::with_single_input("x", 1);
            let x = g.add(Node::Input(0));

            // Build a random expression of depth 2-3
            let ops_bin = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div];
            let mut pool = vec![x, g.constant(2.0), g.constant(0.5)];

            for _ in 0..4 {
                let kind = rng.next() % 4;
                let id = match kind {
                    0 => {
                        let a = pool[rng.next() as usize % pool.len()];
                        let b = pool[rng.next() as usize % pool.len()];
                        g.binary(ops_bin[rng.next() as usize % ops_bin.len()], a, b)
                    }
                    1 => {
                        let a = pool[rng.next() as usize % pool.len()];
                        let ops = [UnaryOp::Sin, UnaryOp::Cos, UnaryOp::Neg, UnaryOp::Exp, UnaryOp::Tanh];
                        g.unary(ops[rng.next() as usize % ops.len()], a)
                    }
                    2 => {
                        // Pow with abs(base) to keep it valid
                        let a = pool[rng.next() as usize % pool.len()];
                        let abs_a = g.unary(UnaryOp::Abs, a);
                        let small = g.constant(0.01);
                        let safe_base = g.binary(BinOp::Add, abs_a, small);
                        let exp = g.constant((rng.next() % 3) as f64 + 0.5);
                        g.binary(BinOp::Pow, safe_base, exp)
                    }
                    _ => g.constant(rng.next() as f64 % 5.0 + 0.1),
                };
                pool.push(id);
            }

            let f_node = pool[rng.next() as usize % pool.len()];

            // Symbolic derivative
            let mut memo = HashMap::new();
            let df_node = differentiate(&mut g, f_node, x, &mut memo);

            // Evaluate at a few points
            for &x_val in &[0.5, 1.0, 2.0, 0.3] {
                g.outputs = vec![f_node];
                let f_val = g.interpret(&[&[x_val]], &[])[0];

                g.outputs = vec![df_node];
                let df_val = g.interpret(&[&[x_val]], &[])[0];

                // Finite difference: (f(x+h) - f(x-h)) / 2h
                let h = 1e-7;
                g.outputs = vec![f_node];
                let f_plus = g.interpret(&[&[x_val + h]], &[])[0];
                let f_minus = g.interpret(&[&[x_val - h]], &[])[0];
                let fd = (f_plus - f_minus) / (2.0 * h);

                if df_val.is_finite() && fd.is_finite() && f_val.is_finite() {
                    let tol = 1e-4 * fd.abs().max(1.0);
                    assert!(
                        (df_val - fd).abs() < tol,
                        "seed={} x={}: symbolic={} fd={}", seed, x_val, df_val, fd
                    );
                }
            }
        }
    }

    /// Check one unary op at a single point: symbolic derivative vs central FD.
    fn check_unary_fd(op: UnaryOp, x_val: f64) {
        let mut g = Graph::with_single_input("x", 1);
        let x = g.add(Node::Input(0));
        let f_node = g.unary(op, x);
        let mut memo = HashMap::new();
        let df_node = differentiate(&mut g, f_node, x, &mut memo);

        g.outputs = vec![df_node];
        let df_val = g.interpret(&[&[x_val]], &[])[0];

        g.outputs = vec![f_node];
        let h = 1e-6_f64.max(1e-6 * x_val.abs());
        let fp = g.interpret(&[&[x_val + h]], &[])[0];
        let fm = g.interpret(&[&[x_val - h]], &[])[0];
        let fd = (fp - fm) / (2.0 * h);

        assert!(df_val.is_finite(), "{:?}@{}: df not finite", op, x_val);
        let tol = 1e-4_f64.max(1e-4 * fd.abs());
        assert!((df_val - fd).abs() < tol,
            "{:?}@{}: symbolic={}, fd={}", op, x_val, df_val, fd);
    }

    /// Check one binary op at a point for the first operand.
    fn check_binary_fd(op: BinOp, a_val: f64, b_val: f64) {
        let sig = InputSignature::from_named_sizes([("a", 1), ("b", 1)]);
        let mut g = Graph::new(sig);
        let a = g.input(0);
        let b = g.input(1);
        let f_node = g.binary(op, a, b);
        let mut memo = HashMap::new();
        let df_node = differentiate(&mut g, f_node, a, &mut memo);

        g.outputs = vec![df_node];
        let df_val = g.interpret(&[&[a_val], &[b_val]], &[])[0];

        g.outputs = vec![f_node];
        let h = 1e-6_f64.max(1e-6 * a_val.abs());
        let fp = g.interpret(&[&[a_val + h], &[b_val]], &[])[0];
        let fm = g.interpret(&[&[a_val - h], &[b_val]], &[])[0];
        let fd = (fp - fm) / (2.0 * h);

        assert!(df_val.is_finite(), "{:?}@({},{}): df not finite", op, a_val, b_val);
        let tol = 1e-4_f64.max(1e-4 * fd.abs());
        assert!((df_val - fd).abs() < tol,
            "{:?}@({},{}): symbolic={}, fd={}", op, a_val, b_val, df_val, fd);
    }

    /// Systematic derivative check for every user-reachable UnaryOp. The
    /// derivative is validated against a central finite difference. Inputs are
    /// picked inside each op's defined domain (no NaNs from log(-1) etc.).
    #[test]
    fn test_diff_all_unary_ops() {
        use UnaryOp::*;
        let cases: &[(UnaryOp, &[f64])] = &[
            (Neg,    &[-1.5, 0.3, 2.0]),
            (Sin,    &[-1.0, 0.5, 1.7]),
            (Cos,    &[-1.0, 0.5, 1.7]),
            (Tan,    &[0.3, 0.7]),   // stay away from π/2
            (Atan,   &[-2.0, 0.0, 1.5]),
            (Asin,   &[-0.5, 0.3, 0.8]),   // |x| < 1
            (Acos,   &[-0.5, 0.3, 0.8]),
            (Sinh,   &[-1.5, 0.0, 1.5]),
            (Cosh,   &[-1.5, 0.0, 1.5]),
            (Tanh,   &[-1.5, 0.0, 1.5]),
            (Asinh,  &[-2.0, 0.5, 3.0]),
            (Acosh,  &[1.5, 3.0]),         // x ≥ 1
            (Atanh,  &[-0.5, 0.3, 0.8]),   // |x| < 1
            (Exp,    &[-1.0, 0.3, 2.0]),
            (Expm1,  &[-1.0, 0.3, 2.0]),
            (Log,    &[0.3, 1.5, 4.0]),    // x > 0
            (Log10,  &[0.3, 1.5, 4.0]),
            (Log2,   &[0.3, 1.5, 4.0]),
            (Log1p,  &[-0.5, 0.5, 2.0]),   // x > -1
            (Abs,    &[-1.3, 1.3]),        // skip x=0 (non-smooth)
            (Sqrt,   &[0.3, 1.5, 4.0]),
            (Cbrt,   &[-1.5, 0.3, 2.0]),
            (Erf,    &[-1.0, 0.3, 1.5]),
            (Erfc,   &[-1.0, 0.3, 1.5]),
            (Lgamma, &[0.5, 1.5, 3.7]),
            (Tgamma, &[0.5, 1.5, 3.7]),
            // Digamma: non-smooth derivative w.r.t. our AD (returns 0) — only
            // tested to assert it doesn't panic; skip FD.
        ];
        for (op, xs) in cases {
            for &x in *xs {
                check_unary_fd(*op, x);
            }
        }
    }

    /// Systematic derivative check for every user-reachable BinOp w.r.t. the
    /// first operand. Second operand is held fixed; symmetric case is covered
    /// implicitly by algebraic structure.
    #[test]
    fn test_diff_all_binary_ops() {
        use BinOp::*;
        let cases: &[(BinOp, &[(f64, f64)])] = &[
            (Add,   &[(1.0, 2.0), (-0.5, 1.0)]),
            (Sub,   &[(1.0, 2.0), (-0.5, 1.0)]),
            (Mul,   &[(1.0, 2.0), (-0.5, 1.0)]),
            (Div,   &[(1.0, 2.0), (-0.5, 1.3)]),
            (Pow,   &[(1.5, 2.0), (0.7, 3.0)]),  // base > 0 so ln(base) is ok
            (Min,   &[(0.5, 1.0), (1.0, 0.5)]),  // one strictly smaller on each side
            (Max,   &[(0.5, 1.0), (1.0, 0.5)]),
            (Atan2, &[(1.0, 1.0), (-0.5, 1.5)]),
            (Hypot, &[(3.0, 4.0), (-1.0, 2.0)]),
            // Mod is piecewise, FD around the seam is unstable → skip.
        ];
        for (op, pairs) in cases {
            for &(a, b) in *pairs {
                check_binary_fd(*op, a, b);
            }
        }
    }

    /// Concrete lgamma/tgamma derivative check: before this commit these were
    /// hardcoded to zero. We compare against ψ and tgamma·ψ at a few points.
    #[test]
    fn test_diff_reduce_sum_product_max() {
        // f = Σ xᵢ → ∂f/∂x₁ = 1
        let mut g = Graph::with_single_input("x", 3);
        let xs: Vec<NodeId> = (0..3).map(|i| g.add(Node::Input(i))).collect();
        let s = g.reduce(ReduceOp::Sum, xs.clone());
        let mut memo = HashMap::new();
        let ds = differentiate(&mut g, s, xs[1], &mut memo);
        g.outputs = vec![ds];
        super::super::optimize::optimize(&mut g);
        assert!((g.interpret(&[&[2.0, 3.0, 4.0]], &[])[0] - 1.0).abs() < 1e-12);

        // f = Π xᵢ → ∂f/∂x₁ = x₀·x₂
        let mut g = Graph::with_single_input("x", 3);
        let xs: Vec<NodeId> = (0..3).map(|i| g.add(Node::Input(i))).collect();
        let p = g.reduce(ReduceOp::Product, xs.clone());
        let mut memo = HashMap::new();
        let dp = differentiate(&mut g, p, xs[1], &mut memo);
        g.outputs = vec![dp];
        super::super::optimize::optimize(&mut g);
        // at (2,3,4): x₀·x₂ = 8
        assert!((g.interpret(&[&[2.0, 3.0, 4.0]], &[])[0] - 8.0).abs() < 1e-12);

        // f = max(x₀,x₁,x₂) → subgradient picks the arg-max operand's deriv.
        let mut g = Graph::with_single_input("x", 3);
        let xs: Vec<NodeId> = (0..3).map(|i| g.add(Node::Input(i))).collect();
        let mx = g.reduce(ReduceOp::Max, xs.clone());
        let mut memo = HashMap::new();
        let dmx = differentiate(&mut g, mx, xs[2], &mut memo);
        g.outputs = vec![dmx];
        super::super::optimize::optimize(&mut g);
        // x₂ is the max → ∂/∂x₂ = 1
        assert!((g.interpret(&[&[1.0, 2.0, 5.0]], &[])[0] - 1.0).abs() < 1e-12);
        // x₂ is not the max → ∂/∂x₂ = 0
        assert!((g.interpret(&[&[1.0, 9.0, 5.0]], &[])[0]).abs() < 1e-12);
    }

    #[test]
    fn test_diff_dot() {
        // y = Σ cᵢ·xᵢ with constant c → ∂y/∂x₁ = c₁ (matvec-row case).
        let mut g = Graph::with_single_input("x", 3);
        let xs: Vec<NodeId> = (0..3).map(|i| g.add(Node::Input(i))).collect();
        let cs: Vec<NodeId> = [2.0, 3.0, 4.0].iter().map(|&c| g.constant(c)).collect();
        let y = g.dot(cs, xs.clone());
        let mut memo = HashMap::new();
        let dy = differentiate(&mut g, y, xs[1], &mut memo);
        g.outputs = vec![dy];
        super::super::optimize::optimize(&mut g);
        assert!((g.interpret(&[&[9.0, 9.0, 9.0]], &[])[0] - 3.0).abs() < 1e-12);

        // y = Σ xᵢ·xᵢ = ‖x‖² → ∂y/∂x₁ = 2·x₁.
        let mut g = Graph::with_single_input("x", 3);
        let xs: Vec<NodeId> = (0..3).map(|i| g.add(Node::Input(i))).collect();
        let y = g.dot(xs.clone(), xs.clone());
        let mut memo = HashMap::new();
        let dy = differentiate(&mut g, y, xs[1], &mut memo);
        g.outputs = vec![dy];
        super::super::optimize::optimize(&mut g);
        // at x₁ = 5 → 2·5 = 10
        assert!((g.interpret(&[&[1.0, 5.0, 2.0]], &[])[0] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_diff_gamma_uses_digamma() {
        let mut g = Graph::with_single_input("x", 1);
        let x = g.add(Node::Input(0));
        let lg = g.unary(UnaryOp::Lgamma, x);
        let tg = g.unary(UnaryOp::Tgamma, x);

        let mut memo = HashMap::new();
        let dlg = differentiate(&mut g, lg, x, &mut memo);
        let dtg = differentiate(&mut g, tg, x, &mut memo);

        for &xv in &[0.5_f64, 1.0, 2.5, 4.7] {
            g.outputs = vec![dlg];
            let d_lg = g.interpret(&[&[xv]], &[])[0];
            let expected_lg = crate::ssa::graph::digamma(xv);
            assert!((d_lg - expected_lg).abs() < 1e-10,
                "d/dx lgamma({}): got {}, expected digamma = {}", xv, d_lg, expected_lg);

            g.outputs = vec![dtg];
            let d_tg = g.interpret(&[&[xv]], &[])[0];
            let expected_tg = libm::tgamma(xv) * crate::ssa::graph::digamma(xv);
            assert!((d_tg - expected_tg).abs() < 1e-10,
                "d/dx tgamma({}): got {}, expected Γ·ψ = {}", xv, d_tg, expected_tg);
        }
    }
}
