// Dynamical-system block constructors: ODE, DynamicalSystem, DynamicalFunction.
//
// These accept raw user callbacks for the f_dyn / f_alg paths.  JIT-traced
// variants with analytical Jacobians live in pybindings as `_trace_*`
// entries; both paths wrap the same underlying block layout produced here.

use std::rc::Rc;

use crate::blocks::block::{Block, BlockRef, BlockRole};
use crate::solvers::solver::Solver;
use crate::utils::fastcell::FastCell;

use super::ScalarJacFn;

// ======================================================================================
// ODE: dx/dt = func(x, u, t), y = x — overrides len, update, solve, step
// ======================================================================================

/// ODE: dx/dt = func(x, u, t), y = x
pub fn ode(
    func: impl Fn(&[f64], &[f64], f64) -> Vec<f64> + 'static,
    initial_value: &[f64],
    jac: Option<ScalarJacFn>,
) -> BlockRef {
    let _n = initial_value.len();
    let mut b = Block::default_block();
    b.type_name = "ODE";
    b.role = BlockRole { is_dyn: true, is_src: false, is_rec: false };
    b.opaque_feedthrough = false; // opaque: y = x (no direct input feedthrough)
    b.initial_value = Some(initial_value.to_vec());
    b.engine = Some(Solver::with_defaults(initial_value));
    b.len_fn = Some(Box::new(|_| 0));

    // dx/dt = func(x, u, t)
    b.f_dyn = Some(Box::new(move |x, u, t, out| out.extend(func(x, u, t))));

    // J = jac(x, u, t) if provided (scalar only for legacy compat)
    // If None → automatic numerical Jacobian via compute_jacobian()
    if let Some(jac_fn) = jac {
        b.jac_dyn = Some(Box::new(move |x, u, t, out| out.push(jac_fn(x, u, t))));
    }
    // else: jac_dyn = None → num_jac computed automatically

    // y = x (output = state)
    b.f_alg = Some(Box::new(|x, _u, _t, out| out.extend_from_slice(x)));

    Rc::new(FastCell::new(b))
}

// ======================================================================================
// DynamicalSystem: dx/dt = func_dyn(x, u, t), y = func_alg(x, u, t)
// ======================================================================================

/// DynamicalSystem: dx/dt = func_dyn(x, u, t), y = func_alg(x, u, t)
///
/// `jac_dyn`, when supplied, returns the dense row-major `∂f_dyn/∂x` (length
/// `n·n`) and is installed as the block's analytic Jacobian; `None` leaves the
/// solver to build it by central differencing (`Block::compute_jacobian`).
pub fn dynamical_system(
    func_dyn: impl Fn(&[f64], &[f64], f64) -> Vec<f64> + 'static,
    func_alg: impl Fn(&[f64], &[f64], f64) -> Vec<f64> + 'static,
    initial_value: &[f64],
    has_passthrough: bool,
    jac_dyn: Option<Box<dyn Fn(&[f64], &[f64], f64) -> Vec<f64>>>,
) -> BlockRef {
    let _n = initial_value.len();
    let mut b = Block::default_block();
    b.type_name = "DynamicalSystem";
    b.role = BlockRole { is_dyn: true, is_src: false, is_rec: false };
    b.opaque_feedthrough = has_passthrough; // opaque: caller declares y-on-u feedthrough
    b.initial_value = Some(initial_value.to_vec());
    b.engine = Some(Solver::with_defaults(initial_value));
    let len_val = if has_passthrough { 1 } else { 0 };
    b.len_fn = Some(Box::new(move |_| len_val));
    b.f_dyn = Some(Box::new(move |x, u, t, out| out.extend(func_dyn(x, u, t))));
    b.f_alg = Some(Box::new(move |x, u, t, out| out.extend(func_alg(x, u, t))));
    if let Some(jac) = jac_dyn {
        b.jac_dyn = Some(Box::new(move |x, u, t, out| out.extend(jac(x, u, t))));
    }
    Rc::new(FastCell::new(b))
}

// ======================================================================================
// DynamicalFunction: y = func(u, t) — time-dependent algebraic function
// ======================================================================================

/// DynamicalFunction: y = func(u, t) -- time-dependent algebraic function
pub fn dynamical_function(func: impl Fn(&[f64], f64) -> Vec<f64> + 'static) -> BlockRef {
    let mut b = Block::default_block();
    b.type_name = "DynamicalFunction";
    b.f_alg = Some(Box::new(move |_x, u, t, out| out.extend(func(u, t))));
    Rc::new(FastCell::new(b))
}
