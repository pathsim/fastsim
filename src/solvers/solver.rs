// Base numerical integrator — the ONLY solver type
// Ported 1:1 from pathsim/solvers/_solver.py
//
// Same pattern as Block: one Solver struct with callback fields.
// Concrete solvers (SSPRK22, RK4, RKDP54, etc.) are constructor functions
// that create a Solver with the right callbacks.
//
// The Simulation stores Solver instances in Block.engine.
// When block.step(t, dt) is called, it calls engine.step(f, dt)
// which dispatches to the correct solver logic via the callback.

use std::collections::VecDeque;

use smallvec::SmallVec;

use crate::constants::*;
use crate::optim::anderson::{Anderson, Optimizer};
use crate::solvers::stage::StageBuilder;

/// Per-component ARKODE-style WRMS weight `atol + rtol·|ref|`. The single
/// source of truth for the weighted-RMS scaling used by every convergence /
/// error norm (solver inner-Newton, step-size control, algebraic-loop booster).
#[inline]
pub(crate) fn wrms_scale(atol: f64, rtol: f64, reference: f64) -> f64 {
    atol + rtol * reference.abs()
}

/// Jacobian type: scalar (1D systems), flat row-major dense matrix (nD), or a
/// structurally-sparse matrix carrying only its nonzero entries.
#[derive(Clone, Debug)]
pub enum Jacobian {
    Scalar(f64),
    Matrix(Vec<f64>, usize), // (flat data, n_rows) — row-major n×n
    Sparse(SparseJac),
}

/// A structurally-sparse `n×n` Jacobian: a fixed coordinate pattern
/// (`rows[k]`, `cols[k]`) with the matching nonzero `values[k]`.
///
/// The pattern comes from the AD graph's structural sparsity and is constant
/// across evaluations of one block; only `values` change per step. Carrying the
/// pattern lets the linear solver skip the `O(n²)` dense materialisation and the
/// per-refactor nonzero scan, and reuse the symbolic factorization while the
/// pattern is unchanged.
#[derive(Clone, Debug)]
pub struct SparseJac {
    pub n: usize,
    /// Coordinate pattern. Structurally invariant for a given width, so it is
    /// shared via `Rc` — cloning a `SparseJac` (per Newton iteration) bumps a
    /// refcount instead of deep-copying the pattern (issue #43).
    pub rows: std::rc::Rc<[u32]>,
    pub cols: std::rc::Rc<[u32]>,
    pub values: Vec<f64>,
}

impl SparseJac {
    /// Expand to a flat row-major dense `n×n` buffer (zeros off-pattern). Used by
    /// fallback paths that have not yet been taught the sparse form, and by tests.
    pub fn to_dense(&self) -> Vec<f64> {
        let mut m = vec![0.0; self.n * self.n];
        for k in 0..self.values.len() {
            m[self.rows[k] as usize * self.n + self.cols[k] as usize] = self.values[k];
        }
        m
    }
}

// Callback types for overridable solver methods
pub type SolverStepFn = Box<dyn FnMut(&mut Solver, &[f64], f64) -> (bool, f64, Option<f64>)>;
pub type SolverSolveFn = Box<dyn FnMut(&mut Solver, &[f64], Option<Jacobian>, f64) -> f64>;
pub type SolverBufferFn = Box<dyn FnMut(&mut Solver, f64)>;

pub struct Solver {
    // -- State --
    pub x: Vec<f64>,
    pub initial_value: Vec<f64>,
    pub _scalar_initial: bool,
    pub tolerance_lte_abs: f64,
    pub tolerance_lte_rel: f64,
    pub is_adaptive: bool,
    pub is_explicit: bool,
    pub is_implicit: bool,
    pub history: VecDeque<Vec<f64>>,
    pub history_maxlen: usize,
    /// Variable-step BDF/GEAR support: timestep history parallel to `history`.
    /// Maintained only by multistep solvers (their factory sets
    /// `history_dt_maxlen > 0` and pushes alongside `history` in their
    /// `buffer_fn`).  Single-step RK solvers leave it empty and do not pay
    /// allocation or push cost — the default `Solver::buffer` only touches
    /// it when `history_dt_maxlen > 0` so existing solvers are unaffected.
    pub history_dt: VecDeque<f64>,
    pub history_dt_maxlen: usize,
    pub n: usize,
    pub s: usize,
    pub _stage: usize,
    pub eval_stages: Vec<f64>,
    pub opt: Option<Optimizer>,

    // -- RK-specific data (Butcher tableau, slopes, etc.) --
    // Indexed by stage number (0..s). Vec instead of HashMap for O(1) access.
    pub ks: Vec<Vec<f64>>,                  // slopes per stage: Ks[stage] = f
    pub bt: Vec<Option<Vec<f64>>>,          // Butcher tableau: BT[stage] = coefficients (None = explicit stage)
    pub tr: Option<Vec<f64>>,
    pub a_final: Option<Vec<f64>>,
    pub m: usize, // embedded order
    pub beta: f64, // safety factor
    /// Minimum rescale factor applied by `step_factor`.  Defaults to
    /// `SOL_SCALE_MIN` (0.1) for PathSim parity; specific solvers override
    /// to a less-aggressive value when their reject-cascade behaviour would
    /// otherwise underflow `dt_min`.  GEAR52A uses 0.2.
    pub scale_min: f64,

    // -- PI step-size controller state (Gustafsson). --
    // `err_prev = 0.0` sentinel means "no previous err available" — used on the
    // first step, and reset to 0.0 after every rejected step so the retry uses
    // pure I-control.  On accepted steps `error_controller` updates this to the
    // accepted error norm so the next step's P-term has a basis.
    pub err_prev: f64,
    /// Opt-in flag: enable Gustafsson PI step-size control.  Off by default so
    /// existing RK solvers keep their classical I-only behaviour bit-for-bit.
    /// Set to `true` by the GEAR/BDF factory, where damping the step-size
    /// sequence pays off the most (BDF rejection is order-reset expensive).
    /// Wider rollout to RK methods is a follow-up decision once benchmarks
    /// quantify the rejection-rate vs. GTE tradeoff.
    pub use_pi_controller: bool,

    // -- Overridable methods --
    pub step_fn: Option<SolverStepFn>,
    pub solve_fn: Option<SolverSolveFn>,
    pub buffer_fn: Option<SolverBufferFn>,

    /// Problem-form-specific Newton-stage logic.  `None` for explicit solvers
    /// and the legacy special-solvers (EUB, SteadyState).  DIRK/ESDIRK
    /// factories install `OdeStageBuilder` by default — used by all plain
    /// ODE-form blocks (Integrator, StateSpace, PT1, TF, ODE, …) and also by
    /// SemiExplicitDAE blocks after their inner `z`-elimination reduces them
    /// to a pure ODE.  Genuinely DAE blocks swap in dedicated builders:
    /// `MassMatrixStageBuilder` for `M·ẋ = f`, `FullyImplicitStageBuilder`
    /// for `F(x, ẋ, u, t) = 0`.
    ///
    /// All builders return a WRMS residual norm against
    /// `(tolerance_lte_abs + tolerance_lte_rel·|x|)` (via `Solver::wrms_norm`),
    /// not an absolute L2 norm — see `Simulation::_solve` for the threshold.
    pub stage_builder: Option<Box<dyn StageBuilder>>,

    /// Periodic-steady-state shooting extension.  `None` for ordinary
    /// integrators — pays zero overhead on the hot path.  Set by
    /// `periodic_steady_state_factory` to augment a normal inner solver
    /// (RK / DIRK / GEAR / DAE-extended) with the per-block Anderson
    /// shooting state.  See `pss_close_period()` for the per-period hook.
    pub pss_ext: Option<PssExt>,

    // -- Type name for debugging --
    pub type_name: &'static str,

    /// One-shot guard for the NaN-in-error-norm diagnostic (issue #24).  Set
    /// the first time `error_controller` / `scaled_max_norm` see a non-finite
    /// component so the rejection message is emitted once, not per retry.
    pub nan_reported: std::cell::Cell<bool>,
}

/// Per-block state for the periodic-steady-state shooting solver.  Holds
/// the period-start state snapshot `x_start = x(0)`, a matrix-free Anderson
/// accelerator that drives the shooting fixed-point map
/// `g(x_0) = x(T; x_0)`, and the period length `T`.
///
/// The inner ODE solver — whatever it is (explicit RK, DIRK, GEAR52A,
/// DAE-extended via `engine_postprocess`) — is what `Solver` *itself* is;
/// `pss_ext` augments it with the outer shooting state.  During the
/// transient integration over `[0, T]` it sits idle.  `pss_close_period()`
/// is the only consumer.
pub struct PssExt {
    /// Period-start state `x(0)` for the current shooting iteration.
    /// Mutated in-place by `Anderson::step` each time the period closes.
    pub x_start: Vec<f64>,
    /// Matrix-free Anderson accelerator on the shooting map.  Pure
    /// `Anderson` (not `NewtonAnderson`): no Jacobian required, since
    /// we never form the monodromy matrix `Φ = ∂x(T)/∂x(0)`.
    pub anderson: Anderson,
    /// Period length `T`.  Stored for diagnostics / logging; the outer
    /// loop in `Simulation::periodic_steady_state` is what actually
    /// drives the integration over `[0, T]`.
    pub period: f64,
    /// `false` until `pss_close_period` has been called at least once.
    /// On the first call `x_start` already holds `x(0)` (set at
    /// factory time from the block's `initial_value`); subsequent calls
    /// re-snapshot at the start of each shooting iteration via Anderson.
    pub closed_once: bool,
}

impl Solver {
    pub fn new(
        initial_value: &[f64],
        tolerance_lte_abs: f64,
        tolerance_lte_rel: f64,
    ) -> Self {
        Self {
            x: initial_value.to_vec(),
            initial_value: initial_value.to_vec(),
            _scalar_initial: initial_value.len() == 1,
            tolerance_lte_abs,
            tolerance_lte_rel,
            is_adaptive: false,
            is_explicit: true,
            is_implicit: false,
            history: VecDeque::new(),
            history_maxlen: 1,
            history_dt: VecDeque::new(),
            history_dt_maxlen: 0,
            n: 1,
            s: 1,
            _stage: 0,
            eval_stages: vec![0.0],
            opt: None,
            ks: Vec::new(),
            bt: Vec::new(),
            tr: None,
            a_final: None,
            m: 0,
            beta: SOL_BETA,
            scale_min: SOL_SCALE_MIN,
            err_prev: 0.0,
            use_pi_controller: false,
            step_fn: None,
            solve_fn: None,
            buffer_fn: None,
            stage_builder: None,
            pss_ext: None,
            type_name: "Solver",
            nan_reported: std::cell::Cell::new(false),
        }
    }

    pub fn with_defaults(initial_value: &[f64]) -> Self {
        Self::new(initial_value, SOL_TOLERANCE_LTE_ABS, SOL_TOLERANCE_LTE_REL)
    }

    pub fn from_scalar(val: f64) -> Self {
        let mut s = Self::with_defaults(&[val]);
        s._scalar_initial = true;
        s
    }

    pub fn len(&self) -> usize { self.x.len() }
    pub fn is_empty(&self) -> bool { self.x.is_empty() }

    pub fn stage(&self) -> usize { self._stage }
    pub fn set_stage(&mut self, val: usize) { self._stage = val; }
    pub fn is_first_stage(&self) -> bool { self._stage == 0 }
    pub fn is_last_stage(&self) -> bool { self._stage == self.s - 1 }

    pub fn n_stages(&self) -> usize { self.eval_stages.len() }

    pub fn stage_time(&self, stage: usize, t: f64, dt: f64) -> f64 {
        t + self.eval_stages[stage] * dt
    }

    /// Min-correction stage-value predictor (SUNDIALS ARKODE "Type 5").
    ///
    /// Before the Newton/Anderson iteration at stage `i`, initialise `x` to the
    /// closed-form guess
    ///
    /// ```text
    ///   y_i^(0) = y_n + dt · Σ_{j<i} a_ij · k_j
    /// ```
    ///
    /// using only slopes from already-solved stages (`j < i`).  This is the
    /// Newton fixed-point right-hand side *evaluated with `k_i = 0`*, so
    /// starting from here puts Newton into the quadratic-convergence basin
    /// directly and typically saves 1-2 iterations per stage.
    ///
    /// Also resets the Anderson optimiser: its `dx/dr` history is per-stage
    /// and carries no useful information into the next stage's different
    /// fixed-point equation.
    ///
    /// No-op at stage 0 (explicit first stage in ESDIRK, or nothing to
    /// predict in DIRK), when the tableau row is missing, or when no prior
    /// state has been buffered.
    pub fn apply_stage_predictor(&mut self, dt: f64) {
        let stage = self._stage;
        if stage == 0 { return; }
        if self.history.is_empty() { return; }
        let Some(Some(row)) = self.bt.get(stage) else { return };
        let n = self.x.len();

        for idx in 0..n {
            let mut s = 0.0;
            for j in 0..stage {
                if j < row.len() && j < self.ks.len() && !self.ks[j].is_empty() {
                    s += row[j] * self.ks[j][idx];
                }
            }
            self.x[idx] = self.history[0][idx] + dt * s;
        }
        if let Some(opt) = self.opt.as_mut() { opt.reset(); }
    }

    pub fn get(&self) -> &[f64] { &self.x }
    pub fn set(&mut self, x: &[f64]) {
        self.x.resize(x.len(), 0.0);
        self.x.copy_from_slice(x);
    }

    /// Reconfigure the Anderson history depth on the installed optimizer.
    /// No-op when no optimizer is set or when the optimizer is pure Newton.
    pub fn set_optimizer_history(&mut self, m: usize) {
        if let Some(opt) = self.opt.as_mut() {
            opt.set_m(m);
        }
    }

    /// ARKODE-style WRMS norm of `residual`, scaled component-wise by
    /// `(tolerance_lte_abs + tolerance_lte_rel · |x_i|)`.  Used as the
    /// inner-Newton convergence criterion by every implicit StageBuilder.
    /// Allocation-free hot-path: iterates `residual` and `self.x` in
    /// lockstep, accumulates a single scalar.
    #[inline]
    pub fn wrms_norm(&self, residual: &[f64]) -> f64 {
        let n = residual.len();
        debug_assert_eq!(self.x.len(), n,
            "wrms_norm: residual length {} != solver state length {}", n, self.x.len());
        if n == 0 { return 0.0; }
        let atol = self.tolerance_lte_abs;
        let rtol = self.tolerance_lte_rel;
        let mut sum_sq = 0.0;
        for i in 0..n {
            let scale = wrms_scale(atol, rtol, self.x[i]);
            let s = residual[i] / scale;
            sum_sq += s * s;
        }
        (sum_sq / n as f64).sqrt()
    }

    /// Fused `wrms_norm(a − b)` — same semantics as `wrms_norm` applied to
    /// `[a[i] − b[i]]`, but computes the difference on the fly so callers
    /// (notably `OdeStageBuilder`) need no separate residual buffer.
    /// Allocation-free.
    #[inline]
    pub fn wrms_norm_diff(&self, a: &[f64], b: &[f64]) -> f64 {
        let n = a.len();
        debug_assert_eq!(b.len(), n);
        debug_assert_eq!(self.x.len(), n);
        if n == 0 { return 0.0; }
        let atol = self.tolerance_lte_abs;
        let rtol = self.tolerance_lte_rel;
        let mut sum_sq = 0.0;
        for i in 0..n {
            let scale = wrms_scale(atol, rtol, self.x[i]);
            let s = (a[i] - b[i]) / scale;
            sum_sq += s * s;
        }
        (sum_sq / n as f64).sqrt()
    }

    pub fn reset(&mut self, initial_value: Option<&[f64]>) {
        if let Some(iv) = initial_value {
            self.initial_value = iv.to_vec();
        }
        self.x = self.initial_value.clone();
        self.history.clear();
        self.history_dt.clear();
        for k in &mut self.ks { k.clear(); }
        self.err_prev = 0.0;
    }

    // -- buffer (overridable) --

    /// Push the current state `x` to the front of `history`, recycling the
    /// oldest `Vec` allocation once at capacity. Single source of truth for the
    /// state-history ring across every solver's buffer path.
    pub fn push_state_history(&mut self) {
        if self.history.len() >= self.history_maxlen {
            // Recycle the oldest entry to avoid allocation.
            if let Some(mut recycled) = self.history.pop_back() {
                recycled.clear();
                recycled.extend_from_slice(&self.x);
                self.history.push_front(recycled);
            }
        } else {
            self.history.push_front(self.x.clone());
        }
    }

    pub fn buffer(&mut self, dt: f64) {
        if let Some(mut f) = self.buffer_fn.take() {
            f(self, dt);
            self.buffer_fn = Some(f);
        } else {
            self.push_state_history();
            // Mirror the dt history when the solver opts into multistep storage.
            // RK solvers leave `history_dt_maxlen == 0` and skip this branch.
            if self.history_dt_maxlen > 0 {
                while self.history_dt.len() >= self.history_dt_maxlen {
                    self.history_dt.pop_back();
                }
                self.history_dt.push_front(dt);
            }
        }
    }

    /// Buffer for implicit solvers: also resets optimizer.
    pub fn buffer_implicit(&mut self) {
        self.push_state_history();
        self._stage = 0;
        if let Some(ref mut opt) = self.opt {
            opt.reset();
        }
    }

    // -- revert --

    /// Roll back to the last buffered state on step rejection.
    ///
    /// Always restarts the Anderson accelerator: each implicit RK stage has
    /// its own fixed-point equation `g_i(x) = y_n + dt·Σ a_ij·k_j`, and on
    /// rejection `dt` changes (typically halves) so every stage's `g_i`
    /// changes too.  Carrying the previous attempt's `dx/dr` history into
    /// the retry would feed the LS solve secant differences for a *different*
    /// equation and mis-direct convergence.  Restarting on every revert —
    /// LTE-driven or convergence-failure-driven — is the only correct policy.
    pub fn revert(&mut self) {
        if let Some(prev) = self.history.pop_front() {
            self.x = prev;
        }
        // Mirror on the dt-history.  Multistep solvers (`history_dt_maxlen > 0`)
        // need the buffered timestep dropped in lockstep with the state so the
        // next retry recomputes BDF coefficients against the fresh dt.
        if self.history_dt_maxlen > 0 {
            self.history_dt.pop_front();
        }
        if let Some(opt) = self.opt.as_mut() {
            opt.reset();
        }
        // Drop the PI history on rejection: the next retry runs pure I-control
        // (Gustafsson convention).  Mixing the rejected step's error norm into
        // the P-term of the retry would consistently over-shrink.
        self.err_prev = 0.0;
    }

    // -- pss_close_period --

    /// Close one period of the periodic-steady-state shooting iteration.
    ///
    /// Call sequence per outer iteration (driven by
    /// `Simulation::periodic_steady_state`):
    ///   1. Outer loop calls `sim.run(period, false, adaptive)` — integrates
    ///      from `x(0) = self.x` over `[0, T]`, leaving `self.x = x(T)`.
    ///   2. Outer loop calls this method on every dynamic block's engine.
    ///   3. Method computes the WRMS-scaled shooting residual (used as the
    ///      `NLS_COEF` convergence test, same semantics as every other
    ///      implicit-stage solver), runs one matrix-free Anderson step on
    ///      the shooting map `g(x_0) = x(T)` to produce the next `x_0`,
    ///      copies it into `self.x`, and clears the inner solver's
    ///      transient bookkeeping (`history`, `err_prev`, `opt`) so the
    ///      next period starts from a clean slate.
    ///
    /// No-op (returns `0.0`) when `pss_ext` is `None`.
    pub fn pss_close_period(&mut self) -> f64 {
        // Early exit: not a PSS-augmented engine.
        if self.pss_ext.is_none() { return 0.0; }

        // WRMS residual `‖x(T) − x(0)‖` against the engine's own tolerance
        // weights.  Computed *before* the Anderson update mutates x_start.
        // Borrow split: read x_start through the ext, x through self.
        let wrms = {
            let ext = self.pss_ext.as_ref().unwrap();
            self.wrms_norm_diff(&self.x, &ext.x_start)
        };

        // One Anderson step on the shooting map.  `x_start` mutates in
        // place toward the limit cycle's period-start state.
        {
            let ext = self.pss_ext.as_mut().unwrap();
            ext.anderson.step(&mut ext.x_start, &self.x);
            ext.closed_once = true;
        }

        // Hand the new guess to the inner solver as the next period's IC.
        let new_x: Vec<f64> = self.pss_ext.as_ref().unwrap().x_start.to_vec();
        self.x.copy_from_slice(&new_x);

        // Clear inner-solver transient bookkeeping.  Adaptive solvers
        // restart their PI controller; multistep solvers re-bootstrap
        // their history from the new IC; implicit solvers reset their
        // Anderson optimizer (orthogonal to the *outer* Anderson on
        // `pss_ext`).
        self.history.clear();
        self.history_dt.clear();
        for k in &mut self.ks { k.clear(); }
        self.err_prev = 0.0;
        if let Some(opt) = self.opt.as_mut() { opt.reset(); }
        self._stage = 0;

        wrms
    }

    // -- step (overridable) --

    pub fn step(&mut self, f: &[f64], dt: f64) -> (bool, f64, Option<f64>) {
        if let Some(mut step_fn) = self.step_fn.take() {
            let result = step_fn(self, f, dt);
            self.step_fn = Some(step_fn);
            return result;
        }
        (true, 0.0, None) // Default: no-op
    }

    // -- solve (overridable) --

    pub fn solve(&mut self, f: &[f64], j: Option<Jacobian>, dt: f64) -> f64 {
        if let Some(mut solve_fn) = self.solve_fn.take() {
            let result = solve_fn(self, f, j, dt);
            self.solve_fn = Some(solve_fn);
            return result;
        }
        0.0 // Default: no-op
    }

    /// Take one step from `time` over `dt`: `buffer` + RK stages (explicit
    /// `step`, or implicit Newton per stage) + error control. Returns
    /// `(accepted, rescale)` where `rescale` is the adaptive step factor (`None`
    /// for non-embedded methods). For implicit RK-tableau solvers, `jac_fn` and
    /// `opt` are required; `g_buf`/`jac_buf` are caller-owned scratch.
    ///
    /// Mirrors the per-step body of `ExplicitSolver::integrate` /
    /// `ImplicitSolver::integrate` so callers that need to interleave their own
    /// logic between steps (the event-aware compiled run loop) share the exact
    /// stepping semantics. Tableau-less implicit methods (EUB/GEAR/SteadyState)
    /// are not handled here — they drive their own `solve_fn`/`step_fn`.
    #[allow(clippy::too_many_arguments)]
    pub fn take_step(
        &mut self,
        func: &mut dyn FnMut(&[f64], f64, &mut Vec<f64>),
        jac_fn: Option<&dyn Fn(&[f64], f64) -> Jacobian>,
        const_jac: Option<&Jacobian>,
        time: f64,
        dt: f64,
        max_iterations: usize,
        opt: Option<&mut crate::optim::anderson::Optimizer>,
        g_buf: &mut Vec<f64>,
        jac_buf: &mut Vec<f64>,
        f_buf: &mut Vec<f64>,
    ) -> (bool, Option<f64>) {
        self.buffer(dt);

        if !self.is_implicit {
            // Explicit RK: one slope per stage, error control on the last stage.
            // The slope is evaluated into the caller-owned `f_buf` (no per-stage
            // heap allocation — issue #41).
            let mut result = (true, 0.0, None);
            for stage_idx in 0..self.n_stages() {
                let t_stage = self.stage_time(stage_idx, time, dt);
                self._stage = stage_idx;
                func(&self.x, t_stage, f_buf);
                result = self.step(&f_buf[..], dt);
            }
            return (result.0, result.2);
        }

        debug_assert!(
            jac_fn.is_some() || const_jac.is_some(),
            "implicit take_step requires a Jacobian (recompute closure or constant)"
        );

        if self.bt.is_empty() {
            // Tableau-less implicit (EUB / SteadyState / GEAR multistep): drive
            // the installed `solve_fn` to convergence, then `step`. The optimizer
            // is owned by the solver's own closures, so the external `opt` is
            // unused here (matches `Simulation::_solve`).
            for _ in 0..max_iterations {
                func(&self.x, time + dt, f_buf);
                // `solve` consumes the Jacobian by value; a constant one is cloned
                // once per iteration here (tableau-less GEAR/EUB is not the LTI
                // showcase), a state-dependent one is recomputed.
                let j = match const_jac {
                    Some(cj) => cj.clone(),
                    None => jac_fn.expect("implicit take_step requires a Jacobian")(&self.x, time + dt),
                };
                if self.solve(&f_buf[..], Some(j), dt) < NLS_COEF {
                    break;
                }
            }
            func(&self.x, time + dt, f_buf);
            let (success, _err, scale) = self.step(&f_buf[..], dt);
            return (success, scale);
        }

        // Implicit RK tableau: per-stage Newton with WRMS convergence.
        let opt = opt.expect("implicit take_step requires an optimizer");
        opt.reset();
        let mut result = (true, 0.0, None);
        for stage_idx in 0..self.n_stages() {
            let t_stage = self.stage_time(stage_idx, time, dt);
            self._stage = stage_idx;
            let n = self.x.len();

            // ESDIRK: first stage explicit — just store the slope, no solve.
            let is_explicit_stage =
                self.is_first_stage() && stage_idx < self.bt.len() && self.bt[stage_idx].is_none();
            if is_explicit_stage {
                func(&self.x, t_stage, f_buf);
                while self.ks.len() <= stage_idx {
                    self.ks.push(Vec::new());
                }
                self.ks[stage_idx].resize(n, 0.0);
                self.ks[stage_idx].copy_from_slice(&f_buf[..]);
                result = self.step(&f_buf[..], dt);
                continue;
            }

            self.apply_stage_predictor(dt);
            opt.reset();
            if self.history.is_empty() || stage_idx >= self.bt.len() {
                break;
            }
            // Stack-backed copy of this stage's Butcher row (issue #47): the row
            // is read while `self.ks` / `self.x` are mutated in the Newton loop,
            // so it can't be borrowed in place — but a `SmallVec` keeps it off the
            // heap for every tableau (max stage count well under 16).
            let bt_stage: SmallVec<[f64; 16]> = match &self.bt[stage_idx] {
                Some(coeffs) => SmallVec::from_slice(coeffs),
                None => break,
            };
            let b = bt_stage[stage_idx];

            let mut scaled_norm = f64::INFINITY;
            for _ in 0..max_iterations {
                func(&self.x, t_stage, f_buf);
                // A globally constant (LTI) Jacobian is read by reference from the
                // caller's single evaluation — no O(n^2) clone per Newton iteration
                // (issue #42). A state-dependent one is recomputed each iteration.
                let j_owned;
                let j: &Jacobian = match const_jac {
                    Some(cj) => cj,
                    None => {
                        j_owned = jac_fn.expect("implicit take_step requires a Jacobian")(&self.x, t_stage);
                        &j_owned
                    }
                };
                while self.ks.len() <= stage_idx {
                    self.ks.push(Vec::new());
                }
                self.ks[stage_idx].resize(n, 0.0);
                self.ks[stage_idx].copy_from_slice(&f_buf[..]);

                g_buf.resize(n, 0.0);
                {
                    let x_0 = &self.history[0];
                    for idx in 0..n {
                        let mut s = 0.0;
                        for (i, &a) in bt_stage.iter().enumerate() {
                            if i < self.ks.len() && !self.ks[i].is_empty() {
                                s += self.ks[i][idx] * a;
                            }
                        }
                        g_buf[idx] = x_0[idx] + dt * s;
                    }
                }
                scaled_norm = self.wrms_norm_diff(g_buf, &self.x);

                let _l2 = match j {
                    Jacobian::Scalar(jac) => opt.step(&mut self.x, g_buf, Some(dt * b * *jac)),
                    Jacobian::Matrix(jac_flat, n_rows) => {
                        let nn = *n_rows * *n_rows;
                        jac_buf.resize(nn, 0.0);
                        let scale = dt * b;
                        for i in 0..nn {
                            jac_buf[i] = jac_flat[i] * scale;
                        }
                        opt.step_matrix(&mut self.x, g_buf, Some(jac_buf), *n_rows)
                    }
                    // Dense fallback for the sparse Jacobian until the sparse
                    // solve path lands (SAJ-3/4): scatter the scaled nonzeros
                    // into the dense scratch.
                    Jacobian::Sparse(sj) => {
                        let n_rows = sj.n;
                        let scale = dt * b;
                        jac_buf.clear();
                        jac_buf.resize(n_rows * n_rows, 0.0);
                        for k in 0..sj.values.len() {
                            jac_buf[sj.rows[k] as usize * n_rows + sj.cols[k] as usize] = sj.values[k] * scale;
                        }
                        opt.step_matrix(&mut self.x, g_buf, Some(jac_buf), n_rows)
                    }
                };
                if scaled_norm < NLS_COEF {
                    break;
                }
            }
            if scaled_norm > NLS_COEF {
                result = (false, 0.0, Some(0.5));
                break;
            }
            func(&self.x, t_stage, f_buf);
            result = self.step(&f_buf[..], dt);
        }
        (result.0, result.2)
    }

    // -- error_controller (for RK methods) --

    pub fn error_controller(&mut self, dt: f64) -> (bool, f64, Option<f64>) {
        let tr = match &self.tr {
            Some(tr) => tr,
            None => return (true, 0.0, None),
        };
        let n = self.x.len();
        // Compute error norm without allocating slope vector
        let mut max_scaled_error: f64 = TOLERANCE;
        for j in 0..n {
            let mut slope_j = 0.0;
            for (i, &b) in tr.iter().enumerate() {
                if i < self.ks.len() && !self.ks[i].is_empty() {
                    slope_j += self.ks[i][j] * b;
                }
            }
            let scale = wrms_scale(self.tolerance_lte_abs, self.tolerance_lte_rel, self.x[j]);
            let se = (dt * slope_j).abs() / scale;
            // A NaN component slips the `>` comparison (`NaN > x` is false), so
            // the error norm would collapse to the floor and the poisoned step
            // would be ACCEPTED (issue #24).  Detect it explicitly and force a
            // rejection with the smallest step factor so the run either recovers
            // on a shorter step or aborts at `dt_min` — never marches on NaN.
            if se.is_nan() {
                self.report_nan(j);
                return (false, f64::INFINITY, Some(self.scale_min));
            }
            if se > max_scaled_error { max_scaled_error = se; }
        }
        let error_norm = max_scaled_error.max(TOLERANCE);
        let success = error_norm <= 1.0;
        let order = self.m.min(self.n) + 1;
        let rescale = self.step_factor(error_norm, order, success);
        (success, error_norm, Some(rescale))
    }

    /// WRMS-scaled max-norm `max_i |r_i| / (atol + rtol·|x_i|)` of a residual
    /// vector, floored at `TOLERANCE`. Shared by the GEAR error controllers.
    pub fn scaled_max_norm(&self, residual: &[f64]) -> f64 {
        let mut max_scaled_error: f64 = TOLERANCE;
        for (i, &r) in residual.iter().enumerate() {
            let scale = wrms_scale(self.tolerance_lte_abs, self.tolerance_lte_rel, self.x[i]);
            let se = r.abs() / scale;
            // Explicit NaN rejection — see `error_controller` (issue #24). An
            // infinite `se` is handled correctly by the `>` below; only NaN
            // needs forcing, since it would otherwise leave the norm at the
            // floor and let the caller accept a poisoned step.
            if se.is_nan() {
                self.report_nan(i);
                return f64::INFINITY;
            }
            if se > max_scaled_error { max_scaled_error = se; }
        }
        max_scaled_error.max(TOLERANCE)
    }

    /// One-time diagnostic naming the state component whose error norm went
    /// NaN.  Guarded by `nan_reported` so a rejection cascade emits a single
    /// line rather than one per retry.
    fn report_nan(&self, index: usize) {
        if !self.nan_reported.replace(true) {
            crate::utils::sink::error(&format!(
                "error: solver '{}' produced NaN in state component {} of the \
                 error estimate; forcing step rejection (state poisoned — the \
                 run will retry on a shorter step or abort at dt_min)",
                self.type_name, index
            ));
        }
    }

    /// Error controller with external truncation residual (for GEAR methods).
    pub fn error_controller_with_tr(&mut self, tr_residual: &[f64], _dt: f64) -> (bool, f64, Option<f64>) {
        let error_norm = self.scaled_max_norm(tr_residual);
        let success = error_norm <= 1.0;
        let order = self.m.min(self.n) + 1;
        let rescale = self.step_factor(error_norm, order, success);
        (success, error_norm, Some(rescale))
    }

    /// Step-size factor for adaptive solvers.
    ///
    /// When `use_pi_controller` is set, applies the Gustafsson PI controller
    /// (Hairer-Wanner II §IV.2):  the classical I-term `err^(-α/p)` is blended
    /// with a P-term `(err_prev/err)^(β/p)` so the step-size sequence is damped
    /// and rejections become rarer.  After rejection or on the first step
    /// (`err_prev == 0` sentinel) we fall back to pure I-control to avoid
    /// feeding stale error information into the retry.
    ///
    /// When the flag is unset (default for all RK solvers), behaves identically
    /// to the legacy `safety / err^(1/p)` rule so existing trajectories stay
    /// bit-for-bit.
    ///
    /// `accepted` decides whether the just-computed `err` is committed as
    /// `err_prev` for the next step.  Mirrors what `revert()` undoes: any
    /// rejected step zeroes `err_prev` so a subsequent retry runs I-only.
    pub fn step_factor(&mut self, err: f64, order: usize, accepted: bool) -> f64 {
        let p = order.max(1) as f64;
        let factor = if self.use_pi_controller && self.err_prev > 0.0 {
            // PI: factor = beta · err^(-α/p) · (err_prev / err)^(β/p)
            //            = beta · err^(-(α+β)/p) · err_prev^(β/p)
            let exp_err = -(PI_ALPHA + PI_BETA) / p;
            let exp_prev = PI_BETA / p;
            self.beta * err.powf(exp_err) * self.err_prev.powf(exp_prev)
        } else {
            // I-only path.  Phrased as `beta / err^(1/p)` (not the
            // algebraically equivalent `beta · err^(-1/p)`) so existing RK
            // trajectories stay bit-for-bit identical to the legacy
            // controller's floating-point output.
            self.beta / err.powf(1.0 / p)
        };
        if accepted && self.use_pi_controller {
            self.err_prev = err;
        }
        factor.clamp(self.scale_min, SOL_SCALE_MAX)
    }
}

// ======================================================================================
// ExplicitSolver / ImplicitSolver — kept as helper structs for integrate() testing only
// ======================================================================================

pub struct ExplicitSolver { pub solver: Solver }
pub struct ImplicitSolver { pub solver: Solver }

/// Hairer-Wanner II §IV.4 initial-step-size heuristic.
///
/// Caps `user_dt0` against an automatically-detected safe initial step so
/// aggressive user hints (e.g. `t_end / 100` on stiff systems) don't trigger
/// rejection cascades that underflow `dt_min`.  Costs two extra `func`
/// evaluations per integrate call (negligible amortised over hundreds to
/// thousands of steps).
///
/// Returns `min(user_dt0, h_auto)` clamped to `[dt_min, dt_max]`.  If `user_dt0`
/// is already smaller than the heuristic estimate (user knows what they want),
/// it's respected unchanged.  If it's reckless, we shrink it to a value that
/// the controller can grow from.
///
/// `order` is the integrator's accuracy order (`solver.n` for BDF, the
/// integration order for RK).  A higher order tolerates a slightly larger
/// initial step because LTE shrinks faster with `h`.
pub fn auto_initial_step(
    func: &dyn Fn(&[f64], f64) -> Vec<f64>,
    y_0: &[f64],
    t_0: f64,
    user_dt0: f64,
    atol: f64,
    rtol: f64,
    order: usize,
    dt_min: f64,
    dt_max: f64,
) -> f64 {
    let n = y_0.len();
    if n == 0 { return user_dt0.clamp(dt_min, dt_max); }

    // d_0 = ||y_0 / scale||₂ (RMS-scaled).
    let mut sum_sq = 0.0;
    for i in 0..n {
        let scale = wrms_scale(atol, rtol, y_0[i]);
        let s = y_0[i] / scale;
        sum_sq += s * s;
    }
    let d_0 = (sum_sq / n as f64).sqrt();

    // d_1 = ||f(y_0, t_0) / scale||₂.
    let f_0 = func(y_0, t_0);
    if f_0.len() != n { return user_dt0.clamp(dt_min, dt_max); }
    let mut sum_sq = 0.0;
    for i in 0..n {
        let scale = wrms_scale(atol, rtol, y_0[i]);
        let s = f_0[i] / scale;
        sum_sq += s * s;
    }
    let d_1 = (sum_sq / n as f64).sqrt();

    // Initial guess: time it takes for ||y|| to change by 1% of itself.
    let h_0 = if d_0 < 1e-5 || d_1 < 1e-5 { 1e-6 } else { 0.01 * d_0 / d_1 };

    // Take one explicit-Euler step and estimate ||y'' / scale|| via finite
    // difference of f.
    let mut y_1 = vec![0.0; n];
    for i in 0..n { y_1[i] = y_0[i] + h_0 * f_0[i]; }
    let f_1 = func(&y_1, t_0 + h_0);
    if f_1.len() != n { return user_dt0.clamp(dt_min, dt_max); }
    let mut sum_sq = 0.0;
    for i in 0..n {
        let scale = wrms_scale(atol, rtol, y_0[i]);
        let s = (f_1[i] - f_0[i]) / scale;
        sum_sq += s * s;
    }
    let d_2 = (sum_sq / n as f64).sqrt() / h_0;

    // Pick `h_1` such that an order-`p` step of size `h_1` would have an
    // LTE of about 1% (well below atol+rtol·|y|).
    let max_d = d_1.max(d_2);
    let p = order.max(1) as f64;
    let h_1 = if max_d < 1e-15 {
        (h_0 * 1e-3).max(1e-6)
    } else {
        (0.01_f64 / max_d).powf(1.0 / (p + 1.0))
    };

    // Final: most aggressive of "100x h_0" and "h_1", capped by user hint.
    let h_auto = (100.0 * h_0).min(h_1);
    user_dt0.min(h_auto).clamp(dt_min, dt_max)
}

impl ExplicitSolver {
    pub fn from_scalar(val: f64) -> Self {
        let mut s = Solver::from_scalar(val);
        s.is_explicit = true;
        s.is_implicit = false;
        Self { solver: s }
    }
    pub fn with_defaults(initial_value: &[f64]) -> Self {
        let mut s = Solver::with_defaults(initial_value);
        s.is_explicit = true;
        s.is_implicit = false;
        Self { solver: s }
    }

    /// Buffer helper (static, for closures).
    pub fn buffer(solver: &mut Solver, _dt: f64) {
        solver.push_state_history();
    }

    /// Directly integrate f(x, t) from time_start to time_end.
    /// Exact 1:1 copy of PathSim ExplicitSolver.integrate().
    pub fn integrate(
        solver: &mut Solver,
        func: &dyn Fn(&[f64], f64) -> Vec<f64>,
        time_start: f64,
        time_end: f64,
        mut dt: f64,
        dt_min: f64,
        dt_max: f64,
        adaptive: bool,
    ) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut output_times = vec![time_start];
        let mut output_states = vec![solver.x.clone()];
        let mut time = time_start;

        // Initial-step heuristic (Hairer-Wanner II §IV.4): cap user-supplied
        // dt against an automatically-detected safe value derived from
        // ||y_0||, ||f(y_0)||, and a finite-difference y'' estimate.  Only
        // shrinks dt when the user's guess was reckless; reasonable hints
        // pass through unchanged.
        if adaptive {
            let order = solver.n.max(1);
            dt = auto_initial_step(
                func, &solver.x, time_start, dt,
                solver.tolerance_lte_abs, solver.tolerance_lte_rel,
                order, dt_min, dt_max,
            );
        }

        let (mut g_buf, mut jac_buf, mut f_buf) = (Vec::new(), Vec::new(), Vec::new());
        // Adapt the caller's `Vec`-returning RHS to the zero-alloc buffer form
        // `take_step` now expects (issue #41). The Python/standalone RHS still
        // allocates its own return `Vec` (unavoidable across the callback), so
        // this path is unchanged in allocation count; the compiled fast path
        // (`CompiledSimulation::run`) routes through `call_into` directly.
        let mut f_adapter = |x: &[f64], t: f64, out: &mut Vec<f64>| {
            let v = func(x, t);
            out.clear();
            out.extend_from_slice(&v);
        };
        // Loop guard (issue #26): adaptive runs stop the instant time reaches
        // time_end (the overshoot clamp below lands the final step exactly on
        // it, so `< time_end` avoids a trailing dt==0 duplicate sample); fixed
        // steps stop half a step short so the last step ends on time_end
        // without a step beyond it — matches the `ir_rk4` reference loop.
        while (adaptive && time < time_end) || (!adaptive && time < time_end - 0.5 * dt) {
            // Single timestep (buffer + stages + error control).
            let (success, scale) =
                solver.take_step(&mut f_adapter, None, None, time, dt, 0, None, &mut g_buf, &mut jac_buf, &mut f_buf);
            if adaptive && !success {
                solver.revert();
            } else {
                time += dt;
                output_states.push(solver.x.clone());
                output_times.push(time);
            }

            if adaptive {
                if let Some(s) = scale {
                    let new_dt = s * dt;
                    if new_dt < dt_min {
                        // Step controller wants dt below the floor — return
                        // the trajectory accumulated so far instead of
                        // aborting the entire integration via panic.  The
                        // caller sees a truncated `times` / `states` ending
                        // at the last accepted step; user can detect this
                        // by `times[-1] < time_end`.
                        crate::utils::sink::warn(&format!(
                            "warning: solver requires dt < dt_min ({:e}) at t={:.6}; \
                             returning truncated trajectory",
                            dt_min, time
                        ));
                        break;
                    }
                    dt = new_dt.clamp(dt_min, dt_max);
                }
                // Prevent overshoot
                if time + dt > time_end {
                    dt = time_end - time;
                }
            }
        }

        (output_times, output_states)
    }
}

impl ImplicitSolver {
    pub fn from_scalar(val: f64) -> Self {
        let mut s = Solver::from_scalar(val);
        s.is_explicit = false;
        s.is_implicit = true;
        s.eval_stages = vec![1.0];
        s.opt = Some(Optimizer::default_newton_anderson());
        Self { solver: s }
    }
    pub fn with_defaults(initial_value: &[f64]) -> Self {
        let mut s = Solver::with_defaults(initial_value);
        s.is_explicit = false;
        s.is_implicit = true;
        s.eval_stages = vec![1.0];
        s.opt = Some(Optimizer::default_newton_anderson());
        Self { solver: s }
    }

    pub fn buffer(solver: &mut Solver, _dt: f64) {
        while solver.history.len() >= solver.history_maxlen {
            solver.history.pop_back();
        }
        solver.history.push_front(solver.x.clone());
        solver._stage = 0;
        if let Some(ref mut opt) = solver.opt {
            opt.reset();
        }
    }

    /// Directly integrate f(x, t) with implicit solver.
    /// Uses the provided optimizer for the Newton/fixed-point solve.
    pub fn integrate(
        solver: &mut Solver,
        func: &dyn Fn(&[f64], f64) -> Vec<f64>,
        jac_fn: &dyn Fn(&[f64], f64) -> Jacobian,
        time_start: f64,
        time_end: f64,
        mut dt: f64,
        dt_min: f64,
        dt_max: f64,
        adaptive: bool,
        max_iterations: usize,
        opt: &mut crate::optim::anderson::Optimizer,
    ) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut output_times = vec![time_start];
        let mut output_states = vec![solver.x.clone()];
        let mut time = time_start;
        let mut g_buf: Vec<f64> = Vec::new();
        let mut jac_buf: Vec<f64> = Vec::new();
        let mut f_buf: Vec<f64> = Vec::new();

        // Initial-step heuristic — see ExplicitSolver::integrate for rationale.
        if adaptive {
            let order = solver.n.max(1);
            dt = auto_initial_step(
                func, &solver.x, time_start, dt,
                solver.tolerance_lte_abs, solver.tolerance_lte_rel,
                order, dt_min, dt_max,
            );
        }

        // Dispatch: solvers that don't carry an RK tableau (`bt` empty) — i.e.
        // EUB, SteadyState, GEAR52A — drive their installed solve_fn / step_fn
        // closures directly.  The Newton inner loop below is RK-tableau-aware
        // (it builds `g = x_0 + dt·Σa·k` from `bt[stage_idx]`) and would break
        // out immediately for those solvers.  Mirrors how `Simulation::_solve`
        // handles tableau-less implicit blocks.
        if solver.bt.is_empty() {
            return Self::integrate_no_tableau(
                solver, func, jac_fn, time_start, time_end,
                dt, dt_min, dt_max, adaptive, max_iterations,
            );
        }

        // Adapt the caller's `Vec`-returning RHS to the zero-alloc buffer form
        // `take_step` now expects (issue #41).
        let mut f_adapter = |x: &[f64], t: f64, out: &mut Vec<f64>| {
            let v = func(x, t);
            out.clear();
            out.extend_from_slice(&v);
        };
        // Loop guard — see ExplicitSolver::integrate (issue #26).
        while (adaptive && time < time_end) || (!adaptive && time < time_end - 0.5 * dt) {
            let (success, scale) = solver.take_step(
                &mut f_adapter, Some(jac_fn), None, time, dt, max_iterations, Some(&mut *opt), &mut g_buf, &mut jac_buf, &mut f_buf,
            );
            if adaptive && !success {
                solver.revert();
            } else {
                time += dt;
                output_states.push(solver.x.clone());
                output_times.push(time);
            }

            if adaptive {
                if let Some(s) = scale {
                    let new_dt = s * dt;
                    if new_dt < dt_min {
                        // Step controller wants dt below the floor — return
                        // the trajectory accumulated so far instead of
                        // aborting the entire integration via panic.  The
                        // caller sees a truncated `times` / `states` ending
                        // at the last accepted step; user can detect this
                        // by `times[-1] < time_end`.
                        crate::utils::sink::warn(&format!(
                            "warning: solver requires dt < dt_min ({:e}) at t={:.6}; \
                             returning truncated trajectory",
                            dt_min, time
                        ));
                        break;
                    }
                    dt = new_dt.clamp(dt_min, dt_max);
                }
                if time + dt > time_end {
                    dt = time_end - time;
                }
            }
        }

        (output_times, output_states)
    }

    /// Integrate a solver whose Newton stage logic lives entirely in its
    /// installed `solve_fn` / `step_fn` closures (no RK tableau).  Used by
    /// EUB, SteadyState, and GEAR52A — i.e. anything whose state is
    /// advanced via `Σ α_i x_{n−i} = β·dt·f` (multistep BDF) or another
    /// non-Butcher form.
    ///
    /// Mirrors what `Simulation::_solve` does for a single block: repeated
    /// `solver.solve(...)` calls per step until the WRMS-scaled residual
    /// drops below `NLS_COEF`, then `solver.step(...)` for the
    /// success/error/rescale tuple.  Re-entrancy with the optimizer is
    /// owned by the closure (the GEAR factory installs and resets its own
    /// `solver.opt`), so no external `opt` is threaded through here.
    fn integrate_no_tableau(
        solver: &mut Solver,
        func: &dyn Fn(&[f64], f64) -> Vec<f64>,
        jac_fn: &dyn Fn(&[f64], f64) -> Jacobian,
        time_start: f64,
        time_end: f64,
        mut dt: f64,
        dt_min: f64,
        dt_max: f64,
        adaptive: bool,
        max_iterations: usize,
    ) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut output_times = vec![time_start];
        let mut output_states = vec![solver.x.clone()];
        let mut time = time_start;
        let (mut g_buf, mut jac_buf, mut f_buf) = (Vec::new(), Vec::new(), Vec::new());
        // Adapt the caller's `Vec`-returning RHS to the zero-alloc buffer form
        // `take_step` now expects (issue #41).
        let mut f_adapter = |x: &[f64], t: f64, out: &mut Vec<f64>| {
            let v = func(x, t);
            out.clear();
            out.extend_from_slice(&v);
        };

        while time < time_end {
            // Don't overshoot — clamp this step's dt against the remaining
            // window.  Without this the last accepted step lands a hair past
            // `time_end` which downstream consumers (e.g. trajectory-match
            // tests) read as a regression vs scipy/pathsim.
            let dt_step = dt.min(time_end - time);
            let (success, scale) = solver.take_step(
                &mut f_adapter, Some(jac_fn), None, time, dt_step, max_iterations, None, &mut g_buf, &mut jac_buf, &mut f_buf,
            );

            if !success && adaptive {
                solver.revert();
            } else {
                time += dt_step;
                output_states.push(solver.x.clone());
                output_times.push(time);
            }

            // Step-size adaptation.  Same clamp behaviour as the RK path:
            // refuse to go below `dt_min`, cap at `dt_max`.
            if let Some(s) = scale {
                let new_dt = s * dt;
                if new_dt < dt_min && adaptive {
                    crate::utils::sink::warn(&format!(
                        "warning: solver requires dt < dt_min ({:e}) at t={:.6}; \
                         returning truncated trajectory",
                        dt_min, time
                    ));
                    break;
                }
                dt = new_dt.clamp(dt_min, dt_max);
            }
        }

        (output_times, output_states)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_creation_scalar() {
        let s = Solver::from_scalar(1.0);
        assert_eq!(s.x, vec![1.0]);
        assert!(s._scalar_initial);
    }

    #[test]
    fn test_solver_buffer_revert() {
        let mut s = Solver::from_scalar(1.0);
        s.buffer(0.01);
        s.x = vec![2.0];
        s.revert();
        assert_eq!(s.x, vec![1.0]);
    }

    #[test]
    fn test_solver_reset() {
        let mut s = Solver::from_scalar(1.0);
        s.x = vec![99.0];
        s.reset(None);
        assert_eq!(s.x, vec![1.0]);
    }

    /// Issue #24: a derivative that turns NaN partway through the window must
    /// force step rejection — the integrator must not accept the poisoned step
    /// and march NaN to `time_end`.
    #[test]
    fn test_nan_derivative_rejected_not_marched() {
        let factory = crate::solvers::factories::rkf45_factory(1e-8, 0.0);
        let mut s = factory(&[0.0]);
        // dx/dt = 1 for t < 0.5, NaN afterwards.
        let f = |_x: &[f64], t: f64| if t < 0.5 { vec![1.0] } else { vec![f64::NAN] };

        let (times, states) = ExplicitSolver::integrate(
            &mut s, &f, 0.0, 1.0, 0.1, 1e-6, 1.0, /*adaptive=*/true,
        );

        // No emitted sample may be NaN — the poisoned step is never committed.
        for row in &states {
            assert!(row.iter().all(|v| v.is_finite()), "NaN leaked into trajectory");
        }
        // The run must abort (truncate) rather than reach time_end past the NaN.
        assert!(
            *times.last().unwrap() < 1.0,
            "integrator marched past the NaN region to t={}",
            times.last().unwrap()
        );
    }

    /// Issue #26: `integrate()` must not overshoot `time_end` (fixed step) nor
    /// append a duplicate dt==0 final sample (adaptive).
    #[test]
    fn test_integrate_lands_on_time_end_without_overshoot() {
        let factory = crate::solvers::factories::rkf45_factory(1e-6, 0.0);
        let f = |x: &[f64], _t: f64| vec![-x[0]];

        // Adaptive: final sample lands exactly on time_end, no dt==0 duplicate.
        let mut s = factory(&[1.0]);
        let (t, _x) = ExplicitSolver::integrate(&mut s, &f, 0.0, 1.0, 0.1, 1e-9, 1.0, true);
        assert!((t.last().unwrap() - 1.0).abs() < 1e-9, "adaptive end t={}", t.last().unwrap());
        let n = t.len();
        assert!(t[n - 1] > t[n - 2], "duplicate final sample: {} == {}", t[n - 1], t[n - 2]);

        // Fixed step: no sample beyond time_end, last lands on time_end.
        let mut s2 = factory(&[1.0]);
        let (t2, _x2) = ExplicitSolver::integrate(&mut s2, &f, 0.0, 1.0, 0.1, 1e-9, 1.0, false);
        assert!(t2.iter().all(|&ti| ti <= 1.0 + 1e-9), "fixed-step overshoot: last={:?}", t2.last());
        assert!((t2.last().unwrap() - 1.0).abs() < 1e-9, "fixed-step end t={}", t2.last().unwrap());
    }

    #[test]
    fn test_solver_with_step_callback() {
        let mut s = Solver::from_scalar(0.0);
        s.step_fn = Some(Box::new(|solver, f, dt| {
            // Simple Euler: x = x_0 + dt * f
            if let Some(x_0) = solver.history.front() {
                solver.x = x_0.iter().zip(f.iter())
                    .map(|(&x0, &fi)| x0 + dt * fi).collect();
            }
            (true, 0.0, None)
        }));

        // Integrate dx/dt = 1: x should advance
        s.buffer(0.1);
        let (success, _, _) = s.step(&[1.0], 0.1);
        assert!(success);
        assert!((s.x[0] - 0.1).abs() < 1e-10);
    }

    /// Issue #46: the Gustafsson PI step-size controller (now enabled for
    /// adaptive RK) must never REJECT MORE steps than pure I-control on a mildly
    /// stiff problem, and must land on the same solution (accuracy is unchanged —
    /// both are LTE-controlled). Count-based, not timing-based: we count
    /// `!success` returns from `take_step`, so it is robust to machine noise.
    #[test]
    fn pi_controller_does_not_increase_rejections_and_preserves_accuracy() {
        // Van der Pol (mu = 5): mildly stiff, the classic case where I-control's
        // step sequence oscillates and over-rejects.
        fn run(use_pi: bool) -> (usize, [f64; 2]) {
            let mu = 5.0;
            let mut f = |x: &[f64], _t: f64, out: &mut Vec<f64>| {
                out.clear();
                out.push(x[1]);
                out.push(mu * (1.0 - x[0] * x[0]) * x[1] - x[0]);
            };
            let mut s = crate::solvers::factories::rkdp54_factory(1e-7, 1e-7)(&[2.0, 0.0]);
            s.use_pi_controller = use_pi;
            let (mut g, mut j, mut fb) = (Vec::new(), Vec::new(), Vec::new());
            let (dt_min, dt_max) = (1e-10_f64, 1.0_f64);
            let (t_end, mut t, mut dt) = (10.0_f64, 0.0_f64, 1e-3_f64);
            let mut rejected = 0usize;
            let mut guard = 0usize;
            while t < t_end {
                guard += 1;
                if guard > 5_000_000 { break; }
                if t + dt > t_end { dt = t_end - t; }
                let (success, scale) =
                    s.take_step(&mut f, None, None, t, dt, 0, None, &mut g, &mut j, &mut fb);
                if success {
                    t += dt;
                } else {
                    rejected += 1;
                    s.revert();
                }
                if let Some(sc) = scale {
                    dt = (sc * dt).clamp(dt_min, dt_max);
                }
            }
            (rejected, [s.x[0], s.x[1]])
        }

        let (rej_i, x_i) = run(false);
        let (rej_pi, x_pi) = run(true);
        eprintln!("van der Pol rejections: I-only = {rej_i}, PI = {rej_pi}");

        // Same solution regardless of controller (accuracy preserved).
        assert!((x_i[0] - x_pi[0]).abs() < 1e-3 && (x_i[1] - x_pi[1]).abs() < 1e-3,
                "PI changed the solution: I={x_i:?} PI={x_pi:?}");
        // PI never rejects more than pure I-control.
        assert!(rej_pi <= rej_i,
                "PI increased rejections: I-only={rej_i}, PI={rej_pi}");
    }
}
