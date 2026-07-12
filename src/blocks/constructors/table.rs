// Lookup-table (LUT) block constructors.
//
// Mirrors pathsim's `LUT1D` (1-D linear interpolation). pathsim reaches for
// `scipy.interpolate.interp1d`; here we inline a native Rust implementation
// with a last-interval cache so smooth inputs hit the fast path in O(1)
// amortised time instead of O(log n) per binary search.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::blocks::block::{Block, BlockRef, BlockRole};
use crate::blocks::blockops::Lut1dSpec;
use crate::error::SimError;
use crate::ssa::build::{Builder, GraphBuilder};
use crate::ssa::graph::{Graph, InputSignature};
use crate::utils::fastcell::FastCell;

use super::out_port_map;

/// Op-graph form of LUT1D: a piecewise-linear function unrolled into a select
/// chain over the fixed breakpoints. Mirrors the runtime arithmetic exactly:
/// `raw_t = (x - points[i]) * inv_dx[i]`, then `y = values[i] + raw_t * (delta)`
/// with `delta = values[i+1] - values[i]`; segment `i` is the highest `j` with
/// `points[j] <= x`. For `Clamp` the ends are flattened to the boundary values.
/// The runtime keeps its cached O(1) search; this graph drives the IR / codegen
/// (equivalence-tested).
fn lut1d_build<B: Builder>(b: &B, points: &[f64], values: &[f64], inv_dx: &[f64], clamp: bool, x: B::N) -> B::N {
    let n = points.len();
    let seg = |i: usize| -> B::N {
        let raw_t = b.mul(b.sub(x, b.cst(points[i])), b.cst(inv_dx[i]));
        b.add(b.cst(values[i]), b.mul(raw_t, b.cst(values[i + 1] - values[i])))
    };
    let mut y = seg(0);
    for j in 1..n - 1 {
        y = b.select(b.ge(x, b.cst(points[j])), seg(j), y);
    }
    if clamp {
        // Past the right edge: values[n-2] + 1*(values[n-1]-values[n-2]) (bit-exact runtime form).
        let v_last = b.add(b.cst(values[n - 2]), b.cst(values[n - 1] - values[n - 2]));
        y = b.select(b.gt(x, b.cst(points[n - 1])), v_last, y);
        // Below the left edge: values[0] (runtime t=0).
        y = b.select(b.lt(x, b.cst(points[0])), b.cst(values[0]), y);
    }
    y
}

/// One interpolation segment `i` built directly on `g` (the `&mut Graph` analogue
/// of `lut1d_build`'s `seg`): `values[i] + (x - points[i])·inv_dx[i]·Δvalues`.
fn seg_to_graph(g: &mut Graph, x: u32, points: &[f64], values: &[f64], inv_dx: &[f64], i: usize) -> u32 {
    use crate::ssa::graph::BinOp;
    let pi = g.constant(points[i]);
    let invi = g.constant(inv_dx[i]);
    let dx = g.binary(BinOp::Sub, x, pi);
    let raw_t = g.binary(BinOp::Mul, dx, invi);
    let vi = g.constant(values[i]);
    let dv = g.constant(values[i + 1] - values[i]);
    let term = g.binary(BinOp::Mul, raw_t, dv);
    g.binary(BinOp::Add, vi, term)
}

/// Expand a LUT into a select-chain on an existing graph. Used by `splice` to
/// inline `Op::Lut1d` back to scalar ops for the fused runtime path. Mirrors
/// `lut1d_build` op-for-op so the fused graph is unchanged by the IR carrying a
/// structured `Op::Lut1d` instead of the unrolled chain.
pub fn lut1d_to_graph(g: &mut Graph, x: u32, points: &[f64], values: &[f64], clamp: bool) -> u32 {
    use crate::ssa::graph::{BinOp, CmpOp};
    let n = points.len();
    let inv_dx: Vec<f64> = points.windows(2).map(|w| 1.0 / (w[1] - w[0])).collect();
    let mut y = seg_to_graph(g, x, points, values, &inv_dx, 0);
    for j in 1..n - 1 {
        let pj = g.constant(points[j]);
        let cond = g.cmp(CmpOp::Ge, x, pj);
        let sj = seg_to_graph(g, x, points, values, &inv_dx, j);
        y = g.select(cond, sj, y);
    }
    if clamp {
        let pn = g.constant(points[n - 1]);
        let vn2 = g.constant(values[n - 2]);
        let dv = g.constant(values[n - 1] - values[n - 2]);
        let v_last = g.binary(BinOp::Add, vn2, dv);
        let cond_hi = g.cmp(CmpOp::Gt, x, pn);
        y = g.select(cond_hi, v_last, y);
        let p0 = g.constant(points[0]);
        let v0 = g.constant(values[0]);
        let cond_lo = g.cmp(CmpOp::Lt, x, p0);
        y = g.select(cond_lo, v0, y);
    }
    y
}

fn lut1d_graph(points: &[f64], values: &[f64], inv_dx: &[f64], clamp: bool) -> Graph {
    let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", 1usize)])));
    let y = {
        let gb = GraphBuilder::new(&cell);
        lut1d_build(&gb, points, values, inv_dx, clamp, gb.input(0))
    };
    let mut g = cell.into_inner();
    g.outputs.push(y);
    g
}

/// Behaviour when the input falls outside `[points[0], points[last]]`.
#[derive(Debug, Clone, Copy, Default)]
pub enum ExtrapMode {
    /// Continue the boundary line linearly past either end (pathsim default).
    #[default]
    Extrapolate,
    /// Clamp the output to the nearest boundary value.
    Clamp,
}

/// 1-D piecewise-linear lookup table: `y = interp(u, points → values)`.
///
/// `points` must be strictly monotonically increasing. `values.len()` must
/// equal `points.len()`.
///
/// Performance: the last used interval is cached, so a smooth input (i.e. one
/// that moves by at most one interval per call) stays on the O(1) fast path.
/// On a cache miss, `partition_point` binary search finds the interval in
/// O(log n). Inverse slopes `1 / (x_{i+1} − x_i)` are precomputed at
/// construction to avoid division in the hot path.
pub fn lut1d(points: Vec<f64>, values: Vec<f64>, mode: ExtrapMode) -> Result<BlockRef, SimError> {
    if points.len() != values.len() {
        return Err(SimError::InvalidBlockParam(format!(
            "LUT1D: points ({}) and values ({}) must have equal length",
            points.len(), values.len())));
    }
    if points.len() < 2 {
        return Err(SimError::InvalidBlockParam(
            "LUT1D: at least two points required".to_string()));
    }
    for w in points.windows(2) {
        if w[1] <= w[0] {
            return Err(SimError::InvalidBlockParam(format!(
                "LUT1D: points must be strictly increasing (got {} before {})", w[0], w[1])));
        }
    }

    // Precompute inverse slopes per interval: inv_dx[i] = 1 / (points[i+1] − points[i]).
    let inv_dx: Vec<f64> = points.windows(2).map(|w| 1.0 / (w[1] - w[0])).collect();

    // Last-interval-hint cache. Shared between closure invocations via Rc<Cell>.
    let last_idx: Rc<Cell<usize>> = Rc::new(Cell::new(0));

    let mut b = Block::new(None, Some(out_port_map()));
    b.type_name = "LUT1D";
    b.role = BlockRole::default();

    // IR op-graph (unrolled piecewise-linear select chain). The runtime below
    // keeps its cached search; this is the IR/codegen representation.
    let clamp = matches!(mode, ExtrapMode::Clamp);
    b.set_alg("LUT1D", lut1d_graph(&points, &values, &inv_dx, clamp));
    // Carry the table structure so the IR builder can emit one `Op::Lut1d`
    // (an efficient `static const` table) rather than the unrolled select chain.
    b.alg_op.as_mut().unwrap().lut1d =
        Some(Lut1dSpec { points: points.clone(), values: values.clone(), clamp });

    b.f_alg = Some(Box::new(move |_x, u, _t, out| {
        let x = u[0];
        let n = points.len();
        let last_interval = n - 1;

        // Fast path: try the cached interval first. Smooth (or slowly varying)
        // inputs land here for essentially every eval.
        let mut i = last_idx.get();
        let hit = if i >= last_interval {
            // Cached index got invalidated (table shrunk? not possible here, but
            // be defensive). Fall through to binary search.
            false
        } else {
            x >= points[i] && x <= points[i + 1]
        };

        if !hit {
            // Cache miss — find the interval containing x.
            // partition_point returns the count of points ≤ x; subtract 1 for
            // the left endpoint index, clamped to [0, n-2].
            i = points.partition_point(|&p| p <= x).saturating_sub(1).min(last_interval - 1);
            last_idx.set(i);
        }

        // Linear interpolation. `t` is the normalised position in [0, 1] for
        // inputs inside the table range; outside, it goes negative or > 1 and
        // the Extrapolate branch uses it directly. Clamp mode saturates.
        let raw_t = (x - points[i]) * inv_dx[i];
        let t = match mode {
            ExtrapMode::Extrapolate => raw_t,
            ExtrapMode::Clamp => {
                if x < points[0] { 0.0 }
                else if x > points[last_interval] { 1.0 }
                else { raw_t }
            }
        };

        // For Clamp mode past the right edge we must also clamp the interval.
        let (i, t) = match mode {
            ExtrapMode::Clamp if x > points[last_interval] => (last_interval - 1, 1.0),
            _ => (i, t),
        };

        let y = values[i] + t * (values[i + 1] - values[i]);
        out.push(y);
    }));

    Ok(Rc::new(FastCell::new(b)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(blk: &BlockRef, x: f64) -> f64 {
        blk.borrow_mut().inputs.set_single(0, x);
        blk.borrow_mut().update(0.0);
        blk.borrow().outputs.get_single(0)
    }

    #[test]
    fn test_lut1d_exact_knot() {
        let blk = lut1d(vec![0.0, 1.0, 2.0], vec![0.0, 10.0, 40.0], ExtrapMode::Extrapolate).unwrap();
        assert_eq!(eval(&blk, 0.0), 0.0);
        assert_eq!(eval(&blk, 1.0), 10.0);
        assert_eq!(eval(&blk, 2.0), 40.0);
    }

    #[test]
    fn test_lut1d_midpoint() {
        let blk = lut1d(vec![0.0, 1.0, 2.0], vec![0.0, 10.0, 40.0], ExtrapMode::Extrapolate).unwrap();
        assert_eq!(eval(&blk, 0.5), 5.0);   // halfway between 0 and 10
        assert_eq!(eval(&blk, 1.5), 25.0);  // halfway between 10 and 40
    }

    #[test]
    fn test_lut1d_extrapolate() {
        let blk = lut1d(vec![0.0, 1.0], vec![0.0, 10.0], ExtrapMode::Extrapolate).unwrap();
        // Linear continuation: slope 10 per unit.
        assert_eq!(eval(&blk, -1.0), -10.0);
        assert_eq!(eval(&blk, 2.5), 25.0);
    }

    #[test]
    fn test_lut1d_clamp() {
        let blk = lut1d(vec![0.0, 1.0], vec![0.0, 10.0], ExtrapMode::Clamp).unwrap();
        assert_eq!(eval(&blk, -5.0), 0.0);
        assert_eq!(eval(&blk, 5.0), 10.0);
        assert_eq!(eval(&blk, 0.5), 5.0);
    }

    #[test]
    fn test_lut1d_non_uniform_spacing() {
        // Unevenly spaced points: 0, 0.5, 10.0. Values: 0, 5, 105.
        // Between 0 and 0.5: slope 10. Between 0.5 and 10: slope (105-5)/9.5 ≈ 10.526.
        let blk = lut1d(vec![0.0, 0.5, 10.0], vec![0.0, 5.0, 105.0], ExtrapMode::Extrapolate).unwrap();
        assert_eq!(eval(&blk, 0.25), 2.5);
        let y = eval(&blk, 5.0);
        let expected = 5.0 + (5.0 - 0.5) * (105.0 - 5.0) / 9.5;
        assert!((y - expected).abs() < 1e-12);
    }

    #[test]
    fn test_lut1d_cache_hit_path() {
        // Walk through the table monotonically — every eval after the first
        // in each interval must hit the cache.
        let blk = lut1d(
            (0..100).map(|i| i as f64).collect(),
            (0..100).map(|i| (i * i) as f64).collect(),
            ExtrapMode::Clamp,
        ).unwrap();
        // Just check we produce monotonic quadratic-ish output without panicking.
        let mut last = eval(&blk, 0.0);
        for i in 1..=99 {
            let x = i as f64 + 0.5;
            let y = eval(&blk, x);
            assert!(y > last);
            last = y;
        }
    }

    #[test]
    fn test_lut1d_rejects_unsorted() {
        let err = lut1d(vec![0.0, 2.0, 1.0], vec![0.0, 1.0, 2.0], ExtrapMode::Extrapolate)
            .err().unwrap();
        assert!(err.to_string().contains("strictly increasing"));
    }

    #[test]
    fn test_lut1d_rejects_single_point() {
        let err = lut1d(vec![0.0], vec![0.0], ExtrapMode::Extrapolate).err().unwrap();
        assert!(err.to_string().contains("at least two points"));
    }
}
