// Collocation BVP block (scipy.solve_bvp rebuilt natively) with free parameters
// and interior point conditions. An algebraic block that, per evaluation, solves
//   y'(x) = fun(x, y, p, inputs),   bc(y(a), y(b), p, inputs) = 0,
//   icond(y@ports, p, inputs) = 0   (optional interior/multipoint conditions)
// for the field `y` AND the parameters `p`, with the native collocation solver
// (src/optim/colloc_bvp.rs). `fun`/`bc`/`icond` are traced; AD supplies every
// Jacobian block. The hot path is allocation-free: a reused `BvpWorkspace` and a
// warmstarted mesh/solution/parameters live on the block across evaluations.
// Output is the solution sampled at fixed query points (constant port count).

use std::cell::RefCell;
use std::rc::Rc;

use crate::blocks::block::{Block, BlockRef};
use crate::constants::{PDE_BVP_MAX_NODES, PDE_BVP_MAX_ROUNDS};
use crate::optim::colloc_bvp::{solve_bvp, BvpFns, BvpWorkspace};
use crate::utils::fastcell::FastCell;

/// `(x, y, p, inputs, out)` and `(ya, yb, p, inputs, out)` and
/// `(y_ports, p, inputs, out)` — the traced tapes, with `inputs` still free.
type FFn = dyn Fn(f64, &[f64], &[f64], &[f64], &mut Vec<f64>);
type BFn = dyn Fn(&[f64], &[f64], &[f64], &[f64], &mut Vec<f64>);
type IFn = dyn Fn(&[f64], &[f64], &[f64], &mut Vec<f64>);

/// The ten traced closures a BVP1D block drives (built by the pybinding from
/// AD-derived tapes). No-ops are used for absent parts (`k=0`, no interior conds).
pub struct Bvp1dClosures {
    pub fun: Box<FFn>,
    pub jac_fy: Box<FFn>,
    pub jac_fp: Box<FFn>,
    pub bc: Box<BFn>,
    pub jac_bc_ya: Box<BFn>,
    pub jac_bc_yb: Box<BFn>,
    pub jac_bc_p: Box<BFn>,
    pub icond: Box<IFn>,
    pub jac_ic_y: Box<IFn>,
    pub jac_ic_p: Box<IFn>,
}

#[allow(clippy::too_many_arguments)]
pub fn bvp1d(
    c: Bvp1dClosures,
    n_eq: usize,
    n_params: usize,
    n_bc: usize,
    x_ports: Vec<f64>,
    n_ic: usize,
    x0: Vec<f64>,
    y0: Vec<f64>,
    p0: Vec<f64>,
    x_query: Vec<f64>,
    tol: f64,
) -> BlockRef {
    let n_out = x_query.len() * n_eq;
    let n_total = n_out + n_params; // output = [field samples ; converged params]
    let warm_x = Rc::new(RefCell::new(x0.clone()));
    let warm_y = Rc::new(RefCell::new(y0.clone()));
    let warm_p = Rc::new(RefCell::new(p0.clone()));
    let ws = Rc::new(RefCell::new(BvpWorkspace::new()));

    let mut b = Block::default_block();
    b.type_name = "BVP1D";
    b.opaque_feedthrough = false;
    b.outputs.resize(n_total);
    b.len_fn = Some(Box::new(move |_| n_total));

    // Reset captures.
    let (x0_i, y0_i, p0_i) = (x0.clone(), y0.clone(), p0.clone());
    let (wx_r, wy_r, wp_r) = (warm_x.clone(), warm_y.clone(), warm_p.clone());

    b.f_alg = Some(Box::new(move |_x, inputs, _t, out| {
        // Bind `inputs` into the BvpFns closures (the tapes still carry `inputs`).
        let inp = inputs;
        let fun = |x: f64, y: &[f64], p: &[f64], o: &mut Vec<f64>| (c.fun)(x, y, p, inp, o);
        let jfy = |x: f64, y: &[f64], p: &[f64], o: &mut Vec<f64>| (c.jac_fy)(x, y, p, inp, o);
        let jfp = |x: f64, y: &[f64], p: &[f64], o: &mut Vec<f64>| (c.jac_fp)(x, y, p, inp, o);
        let bc = |ya: &[f64], yb: &[f64], p: &[f64], o: &mut Vec<f64>| (c.bc)(ya, yb, p, inp, o);
        let jba = |ya: &[f64], yb: &[f64], p: &[f64], o: &mut Vec<f64>| (c.jac_bc_ya)(ya, yb, p, inp, o);
        let jbb = |ya: &[f64], yb: &[f64], p: &[f64], o: &mut Vec<f64>| (c.jac_bc_yb)(ya, yb, p, inp, o);
        let jbp = |ya: &[f64], yb: &[f64], p: &[f64], o: &mut Vec<f64>| (c.jac_bc_p)(ya, yb, p, inp, o);
        let icond = |yp: &[f64], p: &[f64], o: &mut Vec<f64>| (c.icond)(yp, p, inp, o);
        let jicy = |yp: &[f64], p: &[f64], o: &mut Vec<f64>| (c.jac_ic_y)(yp, p, inp, o);
        let jicp = |yp: &[f64], p: &[f64], o: &mut Vec<f64>| (c.jac_ic_p)(yp, p, inp, o);

        let fns = BvpFns {
            n_eq, n_params, n_bc, x_ports: &x_ports, n_ic,
            fun: &fun, jac_fy: &jfy, jac_fp: &jfp,
            bc: &bc, jac_bc_ya: &jba, jac_bc_yb: &jbb, jac_bc_p: &jbp,
            icond: &icond, jac_ic_y: &jicy, jac_ic_p: &jicp,
        };

        let (xw, yw, pw) = (warm_x.borrow().clone(), warm_y.borrow().clone(), warm_p.borrow().clone());
        let sol = {
            let mut w = ws.borrow_mut();
            solve_bvp(&fns, &mut w, &xw, &yw, &pw, tol, PDE_BVP_MAX_NODES, PDE_BVP_MAX_ROUNDS)
        };

        // Sample at fixed query points via the 4th-order Hermite interpolant.
        out.clear();
        let xm = &sol.x;
        for &xq in x_query.iter() {
            let mut j = 0;
            while j + 1 < xm.len() - 1 && xm[j + 1] < xq {
                j += 1;
            }
            let h = xm[j + 1] - xm[j];
            let t = if h > 0.0 { (xq - xm[j]) / h } else { 0.0 };
            let (t2, t3) = (t * t, t * t * t);
            let (h00, h01) = (1.0 - 3.0 * t2 + 2.0 * t3, 3.0 * t2 - 2.0 * t3);
            let (h10, h11) = (t - 2.0 * t2 + t3, -t2 + t3);
            for k in 0..n_eq {
                let yj = sol.y[j * n_eq + k];
                let yj1 = sol.y[(j + 1) * n_eq + k];
                let fj = sol.f[j * n_eq + k];
                let fj1 = sol.f[(j + 1) * n_eq + k];
                out.push(h00 * yj + h01 * yj1 + h * (h10 * fj + h11 * fj1));
            }
        }
        out.extend_from_slice(&sol.p); // expose converged parameters

        *warm_x.borrow_mut() = sol.x;
        *warm_y.borrow_mut() = sol.y;
        *warm_p.borrow_mut() = sol.p;
    }));

    b.reset_fn = Some(Box::new(move |blk: &mut Block| {
        blk.inputs.reset();
        blk.outputs.reset();
        *wx_r.borrow_mut() = x0_i.clone();
        *wy_r.borrow_mut() = y0_i.clone();
        *wp_r.borrow_mut() = p0_i.clone();
    }));

    Rc::new(FastCell::new(b))
}
