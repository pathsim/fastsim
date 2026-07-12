// AlgebraicConstraint block: solves `F(x, u) = 0` for `x` at every evaluation.
//
// A pure-algebraic block (no integrated state) that, given the current block
// inputs `u`, runs a warmstarted Newton on the traced residual `F(x, u)` with
// the AD Jacobian `∂F/∂x`, and outputs the converged `x`. It is the standalone
// counterpart of the `semi_explicit_dae` inner z-elimination: the same Newton
// core, exposed as its own block so it can model an instantaneous algebraic
// relation (chemical equilibrium, flash/VLE, steady-state operating point,
// implicit constitutive law). Feeding it a zeroed rate `F := f(x, u)` recovers
// the quasi-steady-state approximation — without the name prescribing it.

use std::cell::RefCell;
use std::rc::Rc;

use crate::blocks::block::{Block, BlockRef};
use crate::constants::{
    DAE_ALGEBRAIC_TOLERANCE, DAE_FULLIMPLICIT_MAX_ITER, DAE_JACOBIAN_SINGULAR_TOL,
};
use crate::optim::linsolve::LinearSolver;
use crate::utils::fastcell::FastCell;

/// `(x, u, out)` — the traced residual `F(x, u)` and its Jacobian `∂F/∂x`
/// (flat row-major `n×n`), with the inputs `u` still free (set dynamically).
type ResFn = dyn Fn(&[f64], &[f64], &mut Vec<f64>);

/// Solve `F(x, u) = 0` for `x` each evaluation (warmstarted, AD Jacobian).
pub fn algebraic_constraint(
    residual: Box<ResFn>,
    jac: Box<ResFn>,
    x0: Vec<f64>,
) -> BlockRef {
    let n = x0.len();
    let warm_x = Rc::new(RefCell::new(x0.clone()));
    let lin = Rc::new(RefCell::new(LinearSolver::new()));

    let mut b = Block::default_block();
    b.type_name = "AlgebraicConstraint";
    b.opaque_feedthrough = false; // solved numerically; declares no direct feedthrough
    b.outputs.resize(n);
    b.len_fn = Some(Box::new(move |_| n));

    let x0_reset = x0.clone();
    let wx_reset = warm_x.clone();

    b.f_alg = Some(Box::new(move |_x, inputs, _t, out| {
        let mut x = warm_x.borrow_mut();
        if x.len() != n {
            x.resize(n, 0.0);
        }
        let mut f_buf: Vec<f64> = Vec::with_capacity(n);
        let mut j_buf: Vec<f64> = Vec::with_capacity(n * n);

        // Warmstarted Newton: x ← x − (∂F/∂x)⁻¹ · F(x, u), via the persistent
        // (scratch-reusing, factorization-caching, in-place) linear solver.
        for _ in 0..DAE_FULLIMPLICIT_MAX_ITER {
            residual(&x, inputs, &mut f_buf);
            let fnorm: f64 = f_buf.iter().map(|v| v * v).sum::<f64>().sqrt();
            if fnorm < DAE_ALGEBRAIC_TOLERANCE {
                break;
            }
            jac(&x, inputs, &mut j_buf);
            // Guard against a structurally singular ∂F/∂x — bail with current best.
            let singular = (0..n)
                .any(|i| (0..n).all(|j| j_buf[i * n + j].abs() < DAE_JACOBIAN_SINGULAR_TOL));
            if singular {
                break;
            }
            let prev = x.clone();
            let res = lin.borrow_mut().newton_solve(&mut x, &f_buf, &j_buf, n);
            if !x.iter().all(|v| v.is_finite()) {
                *x = prev;
                break;
            }
            if res < DAE_ALGEBRAIC_TOLERANCE {
                break;
            }
        }

        out.clear();
        out.extend_from_slice(&x);
    }));

    b.reset_fn = Some(Box::new(move |blk: &mut Block| {
        blk.inputs.reset();
        blk.outputs.reset();
        *wx_reset.borrow_mut() = x0_reset.clone();
    }));

    Rc::new(FastCell::new(b))
}
