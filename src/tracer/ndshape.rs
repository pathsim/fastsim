// Pure N-D shape / stride helpers for the tracer's array layout.
//
// These are plain integer routines (no PyO3, no graph) — factored out of the
// tracer so the row-major / broadcast index math is one cohesive, directly
// unit-testable concern. The graph is always flat row-major; shape is a view.

/// Row-major (C-order) strides for `shape`. Empty shape → empty strides.
pub(crate) fn strides_row_major(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    if n == 0 { return Vec::new(); }
    let mut strides = vec![1usize; n];
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Flat source-index for a broadcasted read. `out_idx` walks the output multi-
/// index; the operand's axes are right-aligned to the output (with leading 1-
/// axes implied), and axes with `src_shape[d] == 1` collapse to index 0.
pub(crate) fn bcast_src_flat(out_idx: &[usize], src_shape: &[usize], src_strides: &[usize]) -> usize {
    let pad = out_idx.len() - src_shape.len();
    let mut flat = 0usize;
    for d in 0..src_shape.len() {
        let od = d + pad;
        let ix = if src_shape[d] == 1 { 0 } else { out_idx[od] };
        flat += ix * src_strides[d];
    }
    flat
}

/// Insert a size-1 dimension at position `axis` (possibly negative) into a
/// shape. Used to implement `keepdims=True`.
pub(crate) fn shape_with_kept_axis(out_shape: &[usize], ndim_in: usize, axis: isize) -> Vec<usize> {
    let ax = if axis < 0 { ndim_in as isize + axis } else { axis } as usize;
    let mut s = Vec::with_capacity(out_shape.len() + 1);
    s.extend_from_slice(&out_shape[..ax.min(out_shape.len())]);
    s.push(1);
    s.extend_from_slice(&out_shape[ax.min(out_shape.len())..]);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_are_row_major() {
        assert_eq!(strides_row_major(&[]), Vec::<usize>::new());
        assert_eq!(strides_row_major(&[5]), vec![1]);
        assert_eq!(strides_row_major(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn broadcast_collapses_size_one_axes() {
        // src shape [1, 4] broadcast into out [3, 4]: row axis collapses to 0.
        let strides = strides_row_major(&[1, 4]);
        assert_eq!(bcast_src_flat(&[2, 1], &[1, 4], &strides), 1);
        assert_eq!(bcast_src_flat(&[0, 3], &[1, 4], &strides), 3);
    }

    #[test]
    fn kept_axis_inserts_one() {
        assert_eq!(shape_with_kept_axis(&[3], 2, 1), vec![3, 1]);
        assert_eq!(shape_with_kept_axis(&[4], 2, 0), vec![1, 4]);
        // negative axis counts from ndim_in
        assert_eq!(shape_with_kept_axis(&[5], 2, -1), vec![5, 1]);
    }
}
