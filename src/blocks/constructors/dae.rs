// DAE block constructors: MassMatrix, SemiExplicit (reduced), FullyImplicit.

use std::cell::RefCell;
use std::rc::Rc;

use crate::blocks::block::{Block, BlockRef, BlockRole};
use crate::error::SimError;
use crate::constants::{
    DAE_ALGEBRAIC_TOLERANCE, DAE_JACOBIAN_SINGULAR_TOL, DAE_FULLIMPLICIT_MAX_ITER,
};
use crate::solvers::solver::Solver;
use crate::optim::linsolve::LinearSolver;
use crate::solvers::stage::{
    num_jac_wrt_z, solve_xdot_for_f, FiContext, FiFn,
    FullyImplicitStageBuilder, Mass, MassMatrixStageBuilder,
};
use crate::utils::fastcell::FastCell;

use super::DaeJacFn;

// ======================================================================================
// MassMatrixDAE: M·dx/dt = func(x, u, t), y = x — DAE with constant mass matrix
// ======================================================================================

/// MassMatrixDAE: `M · dx/dt = func(x, u, t)`, `y = x`.
///
/// The right-hand side `func` is identical to the ODE form.  The supplied
/// `Mass` is stored on the block and installed into the solver's
/// `stage_builder` via `engine_postprocess` whenever an implicit solver is
/// attached (explicit solvers see a pure ODE and will fail on singular `M` —
/// capability checks come in a later commit).
pub fn mass_matrix_dae(
    func: impl Fn(&[f64], &[f64], f64) -> Vec<f64> + 'static,
    mass: Mass,
    initial_value: &[f64],
    jac: Option<super::VecJacFn>,
) -> Result<BlockRef, SimError> {
    let n = initial_value.len();
    if mass.n != n {
        return Err(SimError::InvalidBlockParam(format!(
            "MassMatrixDAE: mass matrix size {} does not match initial_value length {}",
            mass.n, n)));
    }

    let mut b = Block::default_block();
    b.type_name = "MassMatrixDAE";
    b.role = BlockRole { is_dyn: true, is_src: false, is_rec: false };
    b.opaque_feedthrough = false; // opaque DAE: solved numerically, declares no direct feedthrough
    b.initial_value = Some(initial_value.to_vec());
    b.engine = Some(Solver::with_defaults(initial_value));
    b.len_fn = Some(Box::new(|_| 0));

    // dx/dt path: expose the raw RHS f; the stage builder handles the
    // M-weighted residual and the M - dt·a_ii·J Newton matrix.
    b.f_dyn = Some(Box::new(move |x, u, t, out| out.extend(func(x, u, t))));

    // Jacobian of RHS w.r.t. x (same shape as ODE).  If None, numerical
    // Jacobian is derived downstream in Block::compute_jacobian.
    if let Some(jac_fn) = jac {
        b.jac_dyn = Some(Box::new(move |x, u, t, out| out.extend(jac_fn(x, u, t))));
    }

    // y = x (output = state)
    b.f_alg = Some(Box::new(|x, _u, _t, out| out.extend_from_slice(x)));

    // Install MassMatrixStageBuilder when an implicit solver is attached.
    // Explicit solvers silently fall back to the ODE path — correct only
    // for `M = I` (documented contract).  `Simulation::with_defaults`
    // installs SSPRK22 during `add_block`, so a panic here would break
    // the normal construction flow; the expectation is that users call
    // `Simulation.set_solver(<implicit>)` before running.
    let mass_for_pp = mass;
    b.engine_postprocess = Some(Box::new(move |solver: &mut Solver| {
        if solver.is_implicit {
            solver.stage_builder = Some(Box::new(
                MassMatrixStageBuilder::new(mass_for_pp.clone())
            ));
        }
    }));

    Ok(Rc::new(FastCell::new(b)))
}

// ======================================================================================
// SemiExplicitDAE: inner-Newton eliminates `z`, block is a plain ODE in `x`.
// ======================================================================================

/// Semi-explicit Index-1 DAE with differential state `x` (length `n_x`) and
/// algebraic state `z` (length `n_z`):
///
/// ```text
/// ẋ = f_dyn(x, z, u, t)
/// 0 = f_alg(x, z, u, t)
/// ```
///
/// The algebraic variable `z` is eliminated by an **inner Newton** on
/// `f_alg(x, z, u, t) = 0` at every RHS evaluation (warmstarted from the
/// previous call).  The outer solver sees only the ODE in `x`, so **any of
/// the 21 fastsim solvers** — explicit or implicit — can be attached.
///
/// The block output is `[x; z]` (with `z` taken from the converged inner
/// Newton), so downstream blocks see both.
///
/// Trade-offs vs formulating the same system as a `mass_matrix_dae` with
/// block-diagonal singular mass:
/// - **+**  Explicit solvers (RKDP54, RKF78, RKV65, …) work.
/// - **+**  Smaller Newton problem per stage (`n_z` instead of `n_x+n_z`).
/// - **−**  Inner Newton cost per RHS call (typically 1–3 iterations once
///   warmstarted).
/// - **−**  Adaptive error control watches only `x`, not `z`.
///
/// `jac_z` (optional): analytical `∂f_alg/∂z` as flat row-major `n_z × n_z`.
/// Falls back to central differences if omitted.  If you need the outer
/// solver to also see `z` in its state/error control, use
/// `mass_matrix_dae` with a block-diagonal singular mass matrix.
pub fn semi_explicit_dae(
    f_dyn: impl Fn(&[f64], &[f64], &[f64], f64) -> Vec<f64> + 'static,
    f_alg: impl Fn(&[f64], &[f64], &[f64], f64) -> Vec<f64> + 'static,
    x0: &[f64],
    z0: &[f64],
    jac_z: Option<DaeJacFn>,
) -> Result<BlockRef, SimError> {
    let n_x = x0.len();
    let n_z = z0.len();
    if n_x == 0 {
        return Err(SimError::InvalidBlockParam(
            "SemiExplicitDAE: x0 (differential state) must be non-empty".to_string()));
    }

    // Wrap user callbacks for cheap cloning across the f_dyn / f_alg closures.
    let f_dyn_rc: FiFn = Rc::new(f_dyn);
    let f_alg_rc: FiFn = Rc::new(f_alg);
    let jac_z_rc: Option<FiFn> = jac_z.map(|b| Rc::from(b) as FiFn);

    // Warmstart store for z.  Shared with f_alg output so downstream blocks
    // can read the converged algebraic values via the block's second slot.
    let z_state: Rc<RefCell<Vec<f64>>> = Rc::new(RefCell::new(z0.to_vec()));

    let mut b = Block::default_block();
    b.type_name = "SemiExplicitDAE";
    b.role = BlockRole { is_dyn: true, is_src: false, is_rec: false };
    b.opaque_feedthrough = false; // opaque DAE: solved numerically, declares no direct feedthrough
    b.initial_value = Some(x0.to_vec());
    b.engine = Some(Solver::with_defaults(x0));
    b.len_fn = Some(Box::new(|_| 0));

    // f_dyn: inner Newton on f_alg(x, z, u, t) = 0 for z, then emit ẋ.
    let f_dyn_for_fdyn = f_dyn_rc.clone();
    let f_alg_for_fdyn = f_alg_rc.clone();
    let jac_z_for_fdyn = jac_z_rc.clone();
    let z_for_fdyn = z_state.clone();
    // Persistent linear solver: reuses its scratch/`b_mat` buffers across calls
    // and skips refactorization when ∂g/∂z is unchanged (constant-in-z algebraic).
    let lin_fdyn = Rc::new(RefCell::new(LinearSolver::new()));
    b.f_dyn = Some(Box::new(move |x, u, t, out| {
        let mut z = z_for_fdyn.borrow_mut();
        if z.len() != n_z { z.resize(n_z, 0.0); }

        // Inner Newton: z ← z − (∂g/∂z)⁻¹ · g(x,z,u,t).  Warmstart from the
        // previous call converges in 1–2 iterations for smooth problems.
        let mut jz = Vec::with_capacity(n_z * n_z);
        for _ in 0..DAE_FULLIMPLICIT_MAX_ITER {
            let g_val = (f_alg_for_fdyn)(x, &z, u, t);
            let g_norm: f64 = g_val.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if g_norm < DAE_ALGEBRAIC_TOLERANCE { break; }
            match &jac_z_for_fdyn {
                Some(jf) => jz = (jf)(x, &z, u, t),
                None     => num_jac_wrt_z(&f_alg_for_fdyn, x, &z, u, t, &mut jz),
            }
            // Guard against singular ∂g/∂z — bail with current best guess.
            let singular = (0..n_z).any(|i|
                (0..n_z).all(|j| jz[i * n_z + j].abs() < DAE_JACOBIAN_SINGULAR_TOL)
            );
            if singular { break; }
            let prev = z.clone();
            let res = lin_fdyn.borrow_mut().newton_solve(&mut z, &g_val, &jz, n_z);
            if !z.iter().all(|v| v.is_finite()) { *z = prev; break; }
            if res < DAE_ALGEBRAIC_TOLERANCE { break; }
        }

        let xdot = (f_dyn_for_fdyn)(x, &z, u, t);
        out.extend(xdot);
    }));

    // y = [x; z_current] — z read from warmstart store.
    let z_for_alg = z_state.clone();
    b.f_alg = Some(Box::new(move |x, _u, _t, out| {
        out.extend_from_slice(x);
        out.extend_from_slice(&z_for_alg.borrow());
    }));

    Ok(Rc::new(FastCell::new(b)))
}

// ======================================================================================
// FullyImplicitDAE: F(x, ẋ, u, t) = 0, y = x — fully-implicit DAE
// ======================================================================================

/// Fully implicit DAE block: `F(x, ẋ, u, t) = 0`, `y = x`.
///
/// Use for systems that can't be cast into semi-explicit or mass-matrix form
/// (implicit constitutive relations, mixed differential/algebraic with
/// non-trivial coupling, etc.).
///
/// The user supplies:
/// - `f`: the residual `F(x, xdot, u, t)`.
/// - `initial_value`: consistent `x_0` (user is responsible for picking it
///   such that there exists an `ẋ_0` with `F(x_0, ẋ_0, u_0, 0) ≈ 0`).
/// - `jac_x` / `jac_xdot` (optional): analytical `∂F/∂x` and `∂F/∂ẋ` as
///   flat row-major `n×n` matrices.  If omitted, numerical Jacobians (central
///   differences) are used.
///
/// Only implicit solvers (ESDIRK/DIRK family) work — the block installs a
/// `FullyImplicitStageBuilder` into the engine via `engine_postprocess`.
pub fn fully_implicit_dae(
    f: impl Fn(&[f64], &[f64], &[f64], f64) -> Vec<f64> + 'static,
    initial_value: &[f64],
    jac_x: Option<DaeJacFn>,
    jac_xdot: Option<DaeJacFn>,
) -> Result<BlockRef, SimError> {
    let n = initial_value.len();
    if n == 0 {
        return Err(SimError::InvalidBlockParam(
            "FullyImplicitDAE: initial_value must be non-empty".to_string()));
    }

    let f_rc: FiFn = Rc::new(f);
    let jac_x_rc: Option<FiFn> = jac_x.map(|b| Rc::from(b) as FiFn);
    let jac_xdot_rc: Option<FiFn> = jac_xdot.map(|b| Rc::from(b) as FiFn);

    let ctx = FiContext::new();

    let mut b = Block::default_block();
    b.type_name = "FullyImplicitDAE";
    b.role = BlockRole { is_dyn: true, is_src: false, is_rec: false };
    b.opaque_feedthrough = false; // opaque DAE: solved numerically, declares no direct feedthrough
    b.initial_value = Some(initial_value.to_vec());
    b.engine = Some(Solver::with_defaults(initial_value));
    b.len_fn = Some(Box::new(|_| 0));

    // f_dyn for this block plays two roles:
    //   1. cache stage time `t` and block inputs `u` into the FiContext so
    //      the stage builder can read them;
    //   2. at ESDIRK's explicit first stage (where `make_dirk_step` copies
    //      `f` straight into `ks[0]`), provide a *consistent* slope by
    //      solving `F(x, ẋ, u, t) = 0` for `ẋ` via a small Newton loop.
    // For later stages `f` is still written into `ks[stage]` by the generic
    // solver path but is immediately overwritten by the builder with the
    // true `K_i`, so the consistency solve there is harmless.
    let ctx_for_fdyn = ctx.clone();
    let f_for_fdyn = f_rc.clone();
    let xdot_warmstart: Rc<RefCell<Vec<f64>>> =
        Rc::new(RefCell::new(vec![0.0; n]));
    // Persistent linear solver for the consistency Newton (scratch reuse + LU
    // cache across steps when ∂F/∂ẋ is unchanged).
    let lin_fdyn = Rc::new(RefCell::new(LinearSolver::new()));
    b.f_dyn = Some(Box::new(move |x, u, t, out| {
        *ctx_for_fdyn.t.borrow_mut() = t;
        {
            let mut u_buf = ctx_for_fdyn.u.borrow_mut();
            u_buf.clear();
            u_buf.extend_from_slice(u);
        }
        let mut guess = xdot_warmstart.borrow_mut();
        let _ = solve_xdot_for_f(&f_for_fdyn, x, &mut guess, u, t,
                                 DAE_ALGEBRAIC_TOLERANCE, DAE_FULLIMPLICIT_MAX_ITER,
                                 &mut lin_fdyn.borrow_mut());
        out.clear();
        out.extend_from_slice(&guess);
    }));

    // Zero jacobian of the correct shape — the stage builder ignores the
    // Jacobian argument and computes its own `∂F/∂x` and `∂F/∂ẋ` internally.
    // We still have to emit a properly-sized slice so `compute_jacobian`
    // (block.rs) doesn't index-out-of-bounds when it picks `Scalar` vs `Matrix`.
    b.jac_dyn = Some(Box::new(|x, _u, _t, out| {
        out.clear();
        if x.len() == 1 { out.push(0.0); } else { out.resize(x.len() * x.len(), 0.0); }
    }));

    // y = x
    b.f_alg = Some(Box::new(|x, _u, _t, out| out.extend_from_slice(x)));

    // Install FullyImplicitStageBuilder on implicit solvers.  Explicit
    // solvers silently fall through — attach an implicit solver before run.
    let f_for_pp = f_rc.clone();
    let jx_for_pp = jac_x_rc.clone();
    let jxd_for_pp = jac_xdot_rc.clone();
    let ctx_for_pp = ctx.clone();
    b.engine_postprocess = Some(Box::new(move |solver: &mut Solver| {
        if solver.is_implicit {
            solver.stage_builder = Some(Box::new(FullyImplicitStageBuilder::new(
                f_for_pp.clone(),
                jx_for_pp.clone(),
                jxd_for_pp.clone(),
                ctx_for_pp.clone(),
            )));
        }
    }));

    Ok(Rc::new(FastCell::new(b)))
}
