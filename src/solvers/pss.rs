// Periodic-steady-state (PSS) shooting solver factory.
//
// Architecture: a PSS engine is a normal inner solver (RK / DIRK / GEAR /
// DAE-extended) augmented with `Solver::pss_ext = Some(PssExt)`.  During
// the transient integration over one period `[0, T]` the engine behaves
// identically to the inner solver — every `step_fn` / `solve_fn` /
// `buffer_fn` / `stage_builder` / `opt` field is whatever the inner
// factory installed, so all existing solver and DAE machinery passes
// through unchanged.
//
// The shooting fixed-point map `g(x_0) = x(T; x_0)` is closed by the
// outer `Simulation::periodic_steady_state` loop, which:
//   1. Calls `sim.run(period, ..., reset=false)` — integrates the period.
//   2. Per dynamic block, calls `engine.pss_close_period()` (defined on
//      `Solver`) which runs one matrix-free Anderson step that mutates
//      `pss_ext.x_start` toward the limit-cycle period-start state and
//      copies it into `solver.x` as the next period's IC.
//   3. Convergence is the max WRMS residual against `NLS_COEF`, same
//      semantics as every other implicit-stage / steady-state residual.
//
// DAE blocks are supported transparently: their `engine_postprocess`
// (set on the Block, fired by `Block::set_solver_from`) installs the
// custom `StageBuilder` on the PSS-augmented engine after this factory
// returns — exactly the same path the DAE blocks take with any other
// solver factory.

use crate::optim::anderson::Anderson;
use crate::solvers::factories::SolverFactory;
use crate::solvers::solver::PssExt;

/// Build a PSS-augmented solver factory from any inner ODE-solver factory.
///
/// The returned factory produces a fully-functional inner solver with
/// `pss_ext = Some(PssExt::new(iv, period, anderson_m))` attached.  Use
/// `Simulation::periodic_steady_state(period, inner_factory, ...)` rather
/// than calling this directly — the outer loop is what drives the
/// shooting iteration.
///
/// `anderson_m` is the Anderson rolling-buffer depth.  `OPT_HISTORY = 4`
/// is the canonical default (matches the inner-stage Anderson and
/// pathsim's historical setting).  Larger values track more secant
/// information but cost an `(m × n)` least-squares solve per iteration.
pub fn periodic_steady_state_factory(
    inner: SolverFactory,
    period: f64,
    anderson_m: usize,
) -> SolverFactory {
    Box::new(move |iv: &[f64]| {
        let mut s = inner(iv);
        s.pss_ext = Some(PssExt {
            x_start: iv.to_vec(),
            anderson: Anderson::new(anderson_m),
            period,
            closed_once: false,
        });
        s.type_name = "PSS";
        s
    })
}

// ======================================================================================
// Tests
// ======================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{OPT_HISTORY, SOL_TOLERANCE_LTE_ABS, SOL_TOLERANCE_LTE_REL};
    use crate::solvers::factories::rkdp54_factory;

    /// PSS-augmented solver delegates all hot-path behaviour to the inner
    /// solver: flags, callbacks, tolerances all carry through unchanged.
    #[test]
    fn pss_factory_preserves_inner_solver_identity() {
        let inner = rkdp54_factory(SOL_TOLERANCE_LTE_ABS, SOL_TOLERANCE_LTE_REL);
        let pss = periodic_steady_state_factory(inner, 1.0, OPT_HISTORY);
        let s = pss(&[1.0, 2.0]);
        assert_eq!(s.type_name, "PSS");
        assert!(s.is_adaptive, "RKDP54 is adaptive — PSS wrapper must preserve");
        assert!(s.is_explicit);
        assert!(!s.is_implicit);
        assert!(s.step_fn.is_some(), "inner step_fn must pass through");
        assert!(s.pss_ext.is_some());
        let ext = s.pss_ext.as_ref().unwrap();
        assert_eq!(ext.x_start, vec![1.0, 2.0]);
        assert_eq!(ext.period, 1.0);
        assert!(!ext.closed_once);
    }

    /// `pss_close_period` on a non-PSS solver is a no-op returning 0.0.
    /// Guarantees zero overhead and no surprise on every existing factory.
    #[test]
    fn pss_close_period_is_noop_without_ext() {
        let mut s = rkdp54_factory(SOL_TOLERANCE_LTE_ABS, SOL_TOLERANCE_LTE_REL)(&[0.0]);
        assert!(s.pss_ext.is_none());
        let r = s.pss_close_period();
        assert_eq!(r, 0.0);
    }

    /// First shooting iteration: x_start == x_end (manually crafted),
    /// residual is 0, x doesn't move.  Sanity check on the bookkeeping.
    #[test]
    fn pss_close_period_zero_residual_at_fixed_point() {
        let inner = rkdp54_factory(SOL_TOLERANCE_LTE_ABS, SOL_TOLERANCE_LTE_REL);
        let pss = periodic_steady_state_factory(inner, 1.0, OPT_HISTORY);
        let mut s = pss(&[3.0]);
        // Simulate "ran the period, x didn't move" (already at fixed point).
        s.x = vec![3.0];
        let r = s.pss_close_period();
        assert!(r < 1e-12, "residual at fixed point should be ~0, got {}", r);
        assert!(s.pss_ext.as_ref().unwrap().closed_once);
        // x_start unchanged, x carries forward.
        assert_eq!(s.x, vec![3.0]);
        assert_eq!(s.pss_ext.as_ref().unwrap().x_start, vec![3.0]);
    }

    /// Mock a shooting iteration on a 1-D contraction map: `x_{n+1} = 0.5·x_n + 1`
    /// has fixed point `x* = 2`.  Drive it by hand through `pss_close_period`
    /// and verify Anderson converges to 2 within a handful of iterations.
    #[test]
    fn pss_anderson_converges_on_contraction_map() {
        let inner = rkdp54_factory(SOL_TOLERANCE_LTE_ABS, SOL_TOLERANCE_LTE_REL);
        let pss = periodic_steady_state_factory(inner, 1.0, OPT_HISTORY);
        let mut s = pss(&[0.0]);
        // x_start starts at 0; we'll inject g(x_start) = 0.5·x_start + 1 as `x`
        // before each pss_close_period call, mimicking what a period
        // integration would produce.
        for _ in 0..30 {
            let x0 = s.pss_ext.as_ref().unwrap().x_start[0];
            s.x = vec![0.5 * x0 + 1.0];
            let r = s.pss_close_period();
            if r < 1e-10 { break; }
        }
        let x_final = s.pss_ext.as_ref().unwrap().x_start[0];
        assert!((x_final - 2.0).abs() < 1e-6,
            "Anderson should drive x_start to fixed point 2.0, got {}", x_final);
    }
}
