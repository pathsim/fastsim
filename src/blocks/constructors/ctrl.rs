// Control-block constructors: LeadLag, PID, AntiWindupPID, Differentiator.

use std::rc::Rc;

use crate::blocks::block::{Block, BlockRef, BlockRole};
use crate::blocks::blockops::region_graph_xu;
use crate::ssa::build::{Builder, F64Builder};
use crate::solvers::solver::Solver;
use crate::utils::fastcell::FastCell;

use super::statespace;

/// Differentiator math (alg == dyn): f_max * (u0 - x0).
fn diff_eval<B: Builder>(b: &B, f_max: B::N, x0: B::N, u0: B::N) -> B::N {
    b.mul(f_max, b.sub(u0, x0))
}

/// AntiWindupPID controller output y = Kp*u + Ki*x1 + Kd*f_max*(u - x0).
/// `p = [kp, ki, kd, f_max, ks, lo, hi]`.
fn awpid_y<B: Builder>(b: &B, p: &[B::N], x: &[B::N], u: &[B::N]) -> B::N {
    let t1 = b.mul(p[0], u[0]);
    let t2 = b.mul(p[1], x[1]);
    let t3 = b.mul(b.mul(p[2], p[3]), b.sub(u[0], x[0]));
    b.add(b.add(t1, t2), t3)
}

fn awpid_alg_eval<B: Builder>(b: &B, p: &[B::N], x: &[B::N], u: &[B::N], out: &mut Vec<B::N>) {
    out.clear();
    out.push(awpid_y(b, p, x, u));
}

fn awpid_dyn_eval<B: Builder>(b: &B, p: &[B::N], x: &[B::N], u: &[B::N], out: &mut Vec<B::N>) {
    out.clear();
    // x0' = f_max*(u - x0)
    out.push(b.mul(p[3], b.sub(u[0], x[0])));
    // x1' = u - ks*(y - clamp(y, lo, hi))
    let y = awpid_y(b, p, x, u);
    let clamped = b.min(b.max(y, p[5]), p[6]);
    let w = b.mul(p[4], b.sub(y, clamped));
    out.push(b.sub(u[0], w));
}

// ======================================================================================
// LTI / Control blocks
// ======================================================================================

// ======================================================================================
// LTI / Control blocks
// ======================================================================================

/// LeadLag compensator: H(s) = K * (1 + T1*s) / (1 + T2*s)
pub fn lead_lag(k: f64, t1: f64, t2: f64) -> BlockRef {
    statespace(
        vec![-1.0/t2], vec![1.0/t2],
        vec![k*(t2-t1)/t2], vec![k*t1/t2],
        1, 1, 1, None,
    )
}

/// PID controller: y = Kp*u + Ki*integral(u) + Kd*d/dt(u), with derivative filter
pub fn pid(kp: f64, ki: f64, kd: f64, f_max: f64) -> BlockRef {
    statespace(
        vec![-f_max, 0.0, 0.0, 0.0],
        vec![f_max, 1.0], vec![-kd*f_max, ki], vec![kd*f_max + kp],
        2, 1, 1, None,
    )
}

// ======================================================================================
// Differentiator: dx/dt = f_max*(u - x), y = f_max*(u - x)
// ======================================================================================

/// Differentiator: dx/dt = f_max*(u - x), y = f_max*(u - x)
pub fn differentiator(f_max: f64) -> BlockRef {
    let mut b = Block::default_block();
    b.type_name = "Differentiator";
    b.role = BlockRole { is_dyn: true, is_src: false, is_rec: false };
    b.initial_value = Some(vec![0.0]);
    b.engine = Some(Solver::with_defaults(&[0.0]));
    b.len_fn = Some(Box::new(|_| 1));
    b.f_dyn = Some(Box::new(move |x, u, _t, out| {
        out.clear();
        out.push(diff_eval(&F64Builder, f_max, x[0], u[0]));
    }));
    // Jacobian ∂(dx/dt)/∂x = -f_max is derived by AD from the `dyn_` op-graph
    // (the block's Operator), so no hand-written `jac_dyn` is needed.
    b.f_alg = Some(Box::new(move |x, u, _t, out| {
        out.clear();
        out.push(diff_eval(&F64Builder, f_max, x[0], u[0]));
    }));
    let alg = region_graph_xu(1, 1, vec![f_max], &["f_max"], |gb, p, x, u, out| {
        out.push(diff_eval(gb, p[0], x[0], u[0]));
    });
    let dyn_ = region_graph_xu(1, 1, vec![f_max], &["f_max"], |gb, p, x, u, out| {
        out.push(diff_eval(gb, p[0], x[0], u[0]));
    });
    b.set_dynamic("Differentiator", alg, dyn_);
    Rc::new(FastCell::new(b))
}

// ======================================================================================
// AntiWindupPID: PID with back-calculation anti-windup
// ======================================================================================

/// AntiWindupPID: PID controller with back-calculation anti-windup and output clamping
pub fn anti_windup_pid(kp: f64, ki: f64, kd: f64, f_max: f64, ks: f64, limits: (f64, f64)) -> BlockRef {
    let mut b = Block::default_block();
    b.type_name = "AntiWindupPID";
    b.role = BlockRole { is_dyn: true, is_src: false, is_rec: false };
    b.initial_value = Some(vec![0.0, 0.0]);
    b.engine = Some(Solver::with_defaults(&[0.0, 0.0]));
    b.len_fn = Some(Box::new(|_| 1));
    let params = vec![kp, ki, kd, f_max, ks, limits.0, limits.1];
    let names: &[&str] = &["kp", "ki", "kd", "f_max", "ks", "lo", "hi"];
    let p_dyn = params.clone();
    b.f_dyn = Some(Box::new(move |x, u, _t, out| {
        awpid_dyn_eval(&F64Builder, &p_dyn, x, u, out);
    }));
    let p_alg = params.clone();
    b.f_alg = Some(Box::new(move |x, u, _t, out| {
        awpid_alg_eval(&F64Builder, &p_alg, x, u, out);
    }));
    let alg = region_graph_xu(2, 1, params.clone(), names, |gb, p, x, u, out| {
        awpid_alg_eval(gb, p, x, u, out)
    });
    let dyn_ = region_graph_xu(2, 1, params, names, |gb, p, x, u, out| {
        awpid_dyn_eval(gb, p, x, u, out)
    });
    b.set_dynamic("AntiWindupPID", alg, dyn_);
    Rc::new(FastCell::new(b))
}
