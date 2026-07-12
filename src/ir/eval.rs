//! Reference interpreter over the IR `schema`. This is the semantic ground
//! truth for verification: a block's lowered region, evaluated here, must match
//! the block's live runtime closure (and, later, any code generator's output).
//!
//! It is intentionally simple and slow (a `Vec<f64>` value stack per call); the
//! runtime hot path is the native closure, not this.

use crate::ir::schema::{Op, Region, Write};
use crate::ssa::graph::{apply_binary, apply_cmp, apply_unary};

/// Evaluation environment for one region call. Inputs are per-port slices
/// (so `Op::Input { port, elem }` indexes `inputs[port][elem]`); `params` is
/// indexed by `ParamId`, `state` by `StateId`, `memory` by `MemorySlotId`.
pub struct EvalCtx<'a> {
    pub inputs: &'a [&'a [f64]],
    pub state: &'a [f64],
    pub memory: &'a [&'a [f64]],
    pub params: &'a [f64],
    pub t: f64,
}

/// Error raised when a region cannot be evaluated (e.g. references an opaque
/// extern `Op::Call`, which the reference interpreter does not implement).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalError {
    ExternCall,
}

/// Evaluate a region's `ops` into a value stack, then collect the `writes`
/// in declaration order into the returned output vector. For an `alg` region
/// the outputs are the `Write::Output` values; for `dyn`, the `Write::StateDeriv`
/// values; event effects may mix write kinds. The source `NodeId` of each
/// write is what lands in the output, regardless of kind (callers know which
/// region they asked for).
pub fn eval_region(region: &Region, ctx: &EvalCtx) -> Result<Vec<f64>, EvalError> {
    let mut vals: Vec<f64> = Vec::with_capacity(region.ops.len());
    for op in &region.ops {
        let v = match op {
            Op::Const(c) => *c,
            Op::Time => ctx.t,
            Op::Input { port, elem } => ctx.inputs[*port as usize][*elem as usize],
            Op::Param { id } => ctx.params[id.idx()],
            Op::State { id } => ctx.state[id.idx()],
            Op::Memory { slot, offset } => ctx.memory[slot.idx()][*offset as usize],
            Op::Binary { op, a, b } => apply_binary(*op, vals[a.idx()], vals[b.idx()]),
            Op::Unary { op, a } => apply_unary(*op, vals[a.idx()]),
            Op::Cmp { op, a, b } => apply_cmp(*op, vals[a.idx()], vals[b.idx()]),
            Op::Select { c, t, e } => {
                if vals[c.idx()] != 0.0 { vals[t.idx()] } else { vals[e.idx()] }
            }
            Op::Fma { a, b, c } => vals[a.idx()].mul_add(vals[b.idx()], vals[c.idx()]),
            Op::Reduce { op, args } => {
                args.iter().fold(op.identity(), |acc, n| op.combine(acc, vals[n.idx()]))
            }
            Op::Dot { a, b } => a
                .iter()
                .zip(b.iter())
                .fold(0.0, |acc, (na, nb)| vals[na.idx()].mul_add(vals[nb.idx()], acc)),
            Op::Lut1d { input, points, values, clamp } => {
                lut1d(vals[input.idx()], points, values, *clamp)
            }
            Op::Call { .. } => return Err(EvalError::ExternCall),
        };
        vals.push(v);
    }

    let mut out = Vec::with_capacity(region.writes.len());
    for w in &region.writes {
        let src = match w {
            Write::Output { src, .. }
            | Write::StateDeriv { src, .. }
            | Write::StateWrite { src, .. }
            | Write::MemoryWrite { src, .. } => *src,
        };
        out.push(vals[src.idx()]);
    }
    Ok(out)
}

/// 1-D piecewise-linear lookup. Segment `k` is the highest breakpoint index with
/// `points[k] <= x` (else `0`); the value is the linear interpolation across
/// `[points[k], points[k+1]]`, with the boundary segment continuing past the
/// ends unless `clamp` holds the end value. The codegen emission mirrors this
/// op-for-op (counted search + division), so the two agree numerically.
fn lut1d(x: f64, points: &[f64], values: &[f64], clamp: bool) -> f64 {
    let n = points.len();
    if n < 2 {
        return values.first().copied().unwrap_or(0.0);
    }
    let mut k = 0usize;
    for j in 1..n - 1 {
        if x >= points[j] {
            k = j;
        }
    }
    let t = (x - points[k]) / (points[k + 1] - points[k]);
    let mut y = values[k] + t * (values[k + 1] - values[k]);
    if clamp {
        if x > points[n - 1] {
            y = values[n - 1];
        } else if x < points[0] {
            y = values[0];
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::schema::{NodeId, ReduceKind};

    #[test]
    fn reduce_ops_eval() {
        // read three inputs, then sum / product / min / max over them.
        let region = Region {
            ops: vec![
                Op::Input { port: 0, elem: 0 },
                Op::Input { port: 0, elem: 1 },
                Op::Input { port: 0, elem: 2 },
                Op::Reduce { op: ReduceKind::Sum, args: vec![NodeId(0), NodeId(1), NodeId(2)] },
                Op::Reduce { op: ReduceKind::Product, args: vec![NodeId(0), NodeId(1), NodeId(2)] },
                Op::Reduce { op: ReduceKind::Min, args: vec![NodeId(0), NodeId(1), NodeId(2)] },
                Op::Reduce { op: ReduceKind::Max, args: vec![NodeId(0), NodeId(1), NodeId(2)] },
            ],
            writes: vec![
                Write::Output { port: 0, elem: 0, src: NodeId(3) },
                Write::Output { port: 0, elem: 1, src: NodeId(4) },
                Write::Output { port: 0, elem: 2, src: NodeId(5) },
                Write::Output { port: 0, elem: 3, src: NodeId(6) },
            ],
        };
        let u = [2.0, 5.0, -1.0];
        let ctx = EvalCtx { inputs: &[&u], state: &[], memory: &[], params: &[], t: 0.0 };
        // sum=6, product=-10, min=-1, max=5
        assert_eq!(eval_region(&region, &ctx).unwrap(), vec![6.0, -10.0, -1.0, 5.0]);
    }

    #[test]
    fn dot_op_eval() {
        // y = Σ aᵢ·bᵢ with a = inputs[0..2], b = consts [10, 100].
        let region = Region {
            ops: vec![
                Op::Input { port: 0, elem: 0 },
                Op::Input { port: 0, elem: 1 },
                Op::Const(10.0),
                Op::Const(100.0),
                Op::Dot { a: vec![NodeId(0), NodeId(1)], b: vec![NodeId(2), NodeId(3)] },
            ],
            writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(4) }],
        };
        let u = [3.0, 4.0];
        let ctx = EvalCtx { inputs: &[&u], state: &[], memory: &[], params: &[], t: 0.0 };
        // 3*10 + 4*100 = 430
        assert_eq!(eval_region(&region, &ctx).unwrap(), vec![430.0]);
    }
}
