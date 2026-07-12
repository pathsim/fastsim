//! The compiled runtime: [`CompiledSimulation`], the stateful object produced by
//! [`compile`](super::compile). It holds the fused `dX/dt = F(X, t)` tape (plus a
//! lazily-built symbolic Jacobian and the recorded-signal taps) and drives the
//! standalone solver step by step, mirroring [`crate::simulation::Simulation`]'s
//! run loop / logging. The transformation that builds it lives in the parent
//! module; this module is purely the runtime side of that handoff.

use std::rc::Rc;

use crate::ssa::tape::InterpretedFn;
use super::splice::WriteTarget;
use super::{assemble_sim_event, CompiledEvent};

/// Mutable scratch shared between the event-aware run loop and the event
/// guard/effect closures: the current `(X, M)` at a step boundary.
struct EventScratch {
    x: Vec<f64>,
    m: Vec<f64>,
}

/// A statically compiled system: the fused `dX/dt = F(X, t)` plus the global
/// state layout. Integrate it with [`CompiledSimulation::run`] or sample the
/// derivative directly with [`CompiledSimulation::deriv`].
#[derive(Clone)]
pub struct CompiledSimulation {
    /// Fused derivative function `dX/dt = F(X, t)`. Inputs are `[&x, &[t]]`.
    pub(super) fused: InterpretedFn,
    /// The (unoptimized) derivative graph `dX/dt`, retained so the symbolic
    /// Jacobian `∂F/∂x` can be built LAZILY on first implicit use. Building it
    /// is the (O(n²)) bulk of compile cost, and explicit-solver runs never need
    /// it — so we differentiate on demand, not eagerly at compile time.
    pub(super) jac_src: crate::ssa::graph::Graph,
    /// Jacobian `∂F/∂x` (n×n), used by implicit solvers. Derived from `jac_src`
    /// by symbolic autodiff on first implicit run (cached as an `Rc` so the run
    /// loop's closures can share it), so stiff solvers get an exact native
    /// Jacobian — no Python callback, no finite differences. `None` until built.
    pub(super) jac: std::cell::RefCell<Option<InterpretedFn>>,
    /// `Some(true)` when `∂F/∂x` is a globally constant matrix (linear time-
    /// invariant system, e.g. a `StateSpace` chain), detected by AD introspection
    /// when the Jacobian is first built. `None` until then. The implicit run loop
    /// then evaluates the Jacobian tape once
    /// and reuses the buffer across every Newton iteration, stage and step,
    /// instead of re-running it each iteration. Orthogonal to the linear solver's
    /// factorization cache: that reuses the `O(n³)` factor, this skips the
    /// repeated `O(ops)` Jacobian tape evaluation and matrix assembly.
    pub(super) jac_const: std::cell::Cell<Option<bool>>,
    /// Fused signal-recording function `G(X, t)` producing the tap signals (the
    /// inputs each sink observes). Evaluated only at recorded points, never in
    /// the solver's inner loop.
    pub(super) taps: InterpretedFn,
    /// Label per tap signal (parallel to `taps`' outputs).
    pub(super) tap_labels: Vec<String>,
    /// Number of global continuous-state elements.
    pub n_state: usize,
    /// Number of global discrete-memory elements (event/discrete blocks).
    pub n_mem: usize,
    /// Initial global state (block order).
    pub x0: Vec<f64>,
    /// Initial global discrete memory.
    pub(super) m0: Vec<f64>,
    /// `"<block>.<state>"` label per global state element.
    pub state_labels: Vec<String>,
    /// Block-internal events (empty for pure-continuous models).
    pub(super) events: Vec<CompiledEvent>,

    // -- stateful run, mirroring `Simulation` --
    /// Nominal timestep (initial step for adaptive, fixed step otherwise).
    pub dt: f64,
    /// Explicit solver method name (`solvers::factories::factory_from_name`).
    pub solver: String,
    /// Local-truncation-error tolerances for adaptive stepping.
    pub atol: f64,
    pub rtol: f64,
    /// Current state / memory / time (advanced by `run`, rewound by `reset`).
    pub(super) x: Vec<f64>,
    pub(super) m: Vec<f64>,
    pub(super) time: f64,
    /// Recorded trajectory (times + global states + discrete memory), accumulated
    /// across `run` calls and cleared by `reset` — the static-compile analogue of
    /// scope data. `rec_mems` is parallel to `rec_states` (empty when `n_mem=0`).
    pub(super) rec_times: Vec<f64>,
    /// Flat recorded state trajectory, row-major with stride `n_state` (issue
    /// #44): each accepted sample is one `extend_from_slice` into a bulk-grown
    /// buffer, not a fresh `Vec<f64>` allocation per step — fewer allocations
    /// and contiguous, cache-friendly readout. Invariant:
    /// `rec_states.len() == rec_times.len() * n_state`.
    pub(super) rec_states: Vec<f64>,
    /// Flat recorded discrete memory, stride `n_mem` (parallel to `rec_states`).
    pub(super) rec_mems: Vec<f64>,
    /// Downsampling for long runs: record one of every `output_stride` accepted
    /// steps (issue #44). The initial point and the true final state are always
    /// kept. Default 1 (record every step).
    pub(super) output_stride: usize,
    /// Optional hard cap on the number of recorded samples (bounded memory for
    /// very long runs). Once reached, interior samples are dropped; the final
    /// true state is still recorded. `None` = unbounded. Default `None`.
    pub(super) max_samples: Option<usize>,
    /// Accepted-step counter driving `output_stride` (reset by `reset`).
    pub(super) accepted_steps: usize,
    /// Run-time logger, mirroring `Simulation::logger`. Disabled by default;
    /// `Simulation::compile` (Python) turns it on when the source sim logs, so a
    /// compiled run prints the same `SOLVER` / `TRANSIENT` progress lines.
    pub(super) logger: crate::utils::logger::Logger,
    /// Memoized [`recordings`](CompiledSimulation::recordings) result. Reading
    /// the taps re-evaluates the tap tape per recorded sample — O(samples ·
    /// tape) — so repeated readout of an unchanged trajectory must not pay
    /// that again. Invalidated (set to `None`) by everything that changes the
    /// trajectory or the tap outputs: recording a step, `reset`, `set_param`.
    pub(super) rec_cache: std::cell::RefCell<Option<Vec<Vec<f64>>>>,
}

impl CompiledSimulation {
    /// Evaluate `dX/dt = F(X, M, t)` at the current discrete memory `M`.
    /// `x.len()` must equal `n_state`.
    pub fn deriv(&self, x: &[f64], t: f64) -> Vec<f64> {
        self.fused.call(&[x, &self.m, &[t]])
    }

    /// Parameter labels (`"<block>.<param>"`), in the order `set_param`/the
    /// fused tape use them. Parameters stay *live*: they are baked as graph
    /// `Param` inputs, not folded constants, so they can be retuned without
    /// recompiling. Topology and equations, by contrast, are frozen.
    pub fn param_names(&self) -> &[String] {
        &self.fused.param_names
    }

    /// Current parameter values (parallel to `param_names`).
    pub fn params(&self) -> &[f64] {
        &self.fused.params
    }

    /// Retune a baked parameter by label. Returns `false` if unknown. Applied to
    /// both the derivative and the signal-recording tapes so taps stay correct.
    pub fn set_param(&mut self, name: &str, value: f64) -> bool {
        let a = self.fused.set_param_by_name(name, value);
        let b = self.taps.set_param_by_name(name, value);
        if b {
            // The tap tape changed → memoized readout is stale.
            self.rec_cache.get_mut().take();
        }
        a || b
    }

    /// Labels of the recorded signals (the inputs each sink observes).
    pub fn tap_labels(&self) -> &[String] {
        &self.tap_labels
    }

    /// Lazily build (and cache) the symbolic Jacobian `∂F/∂x` tape and its LTI
    /// flag, differentiating `jac_src` on first call. Only implicit solvers reach
    /// this, so explicit-solver models never pay the (O(n²)) symbolic-Jacobian
    /// construction. Returns `jac_const`. The built tape is cached (only used to
    /// derive the const flag; the run loop evaluates the Jacobian via a separate
    /// `Operator` over `jac_src`).
    fn ensure_jacobian(&self) -> bool {
        if self.jac.borrow().is_none() {
            let (jac_graph, jac_const) =
                crate::ssa::autodiff::jacobian_wrt_slot_optimized(&self.jac_src, "x")
                    .expect("dX/dt is differentiable wrt state (slot validated at compile time)");
            *self.jac.borrow_mut() = Some(InterpretedFn::from_graph(jac_graph));
            // Don't clobber a value already forced by `force_recompute_jacobian`.
            if self.jac_const.get().is_none() {
                self.jac_const.set(Some(jac_const));
            }
        }
        self.jac_const.get().unwrap()
    }

    /// `true` if the AD-derived Jacobian is a globally constant matrix (linear
    /// time-invariant system), so the implicit run loop evaluates it once instead
    /// of per Newton iteration. Builds the Jacobian on first query if needed.
    pub fn jacobian_is_constant(&self) -> bool {
        self.ensure_jacobian()
    }

    /// Diagnostic/benchmark hook: force the implicit run loop onto the general
    /// per-iteration Jacobian-recompute path even when the Jacobian is constant.
    /// Lets a benchmark A/B the eval-once optimization on a single compiled model
    /// without affecting numerical results (both paths compute the same matrix).
    #[doc(hidden)]
    pub fn force_recompute_jacobian(&mut self) {
        self.jac_const.set(Some(false));
    }

    /// Evaluate the recorded signals at one `(x, t)` point (current memory `M`).
    pub fn eval_taps(&self, x: &[f64], t: f64) -> Vec<f64> {
        self.taps.call(&[x, &self.m, &[t]])
    }

    /// Current global state (advanced by `run`, rewound by `reset`).
    pub fn state(&self) -> &[f64] {
        &self.x
    }

    /// Current simulation time.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Recorded sample times (accumulated across `run` calls).
    pub fn times(&self) -> &[f64] {
        &self.rec_times
    }

    /// Recorded global-state trajectory as one owned `Vec` per sample
    /// (materialized from the flat buffer — issue #44). Prefer [`states_flat`]
    /// for a zero-copy row-major view.
    ///
    /// [`states_flat`]: CompiledSimulation::states_flat
    pub fn states(&self) -> Vec<Vec<f64>> {
        if self.n_state == 0 {
            return vec![Vec::new(); self.rec_times.len()];
        }
        self.rec_states.chunks(self.n_state).map(|c| c.to_vec()).collect()
    }

    /// Zero-copy row-major view of the recorded state trajectory: sample `i`
    /// occupies `states_flat()[i*n_state .. (i+1)*n_state]`.
    pub fn states_flat(&self) -> &[f64] {
        &self.rec_states
    }

    /// One recorded state row (empty slice when `n_state == 0`).
    fn state_row(&self, i: usize) -> &[f64] {
        if self.n_state == 0 { &[] } else { &self.rec_states[i * self.n_state..(i + 1) * self.n_state] }
    }

    /// One recorded memory row (empty slice when `n_mem == 0`).
    fn mem_row(&self, i: usize) -> &[f64] {
        if self.n_mem == 0 { &[] } else { &self.rec_mems[i * self.n_mem..(i + 1) * self.n_mem] }
    }

    /// Set the output stride: record one of every `stride` accepted steps
    /// (issue #44). `0` is treated as `1`. Takes effect on the next `run`.
    pub fn set_output_stride(&mut self, stride: usize) {
        self.output_stride = stride.max(1);
    }

    /// Current output stride (see [`set_output_stride`]).
    ///
    /// [`set_output_stride`]: CompiledSimulation::set_output_stride
    pub fn output_stride(&self) -> usize {
        self.output_stride
    }

    /// Cap the number of recorded samples for a long run (bounded memory), or
    /// `None` for unbounded (issue #44).
    pub fn set_max_samples(&mut self, max_samples: Option<usize>) {
        self.max_samples = max_samples;
    }

    /// Current recorded-sample cap (see [`set_max_samples`]), `None` if unbounded.
    ///
    /// [`set_max_samples`]: CompiledSimulation::set_max_samples
    pub fn max_samples(&self) -> Option<usize> {
        self.max_samples
    }

    /// Record one accepted sample, honoring `output_stride` and `max_samples`.
    /// `force` bypasses the stride (used for the guaranteed final sample).
    fn record_step(&mut self, t: f64, x: &[f64], m: &[f64], force: bool) {
        self.accepted_steps += 1;
        if !force && self.output_stride > 1 && !self.accepted_steps.is_multiple_of(self.output_stride) {
            return;
        }
        if let Some(cap) = self.max_samples {
            if !force && self.rec_times.len() >= cap {
                return;
            }
        }
        self.rec_times.push(t);
        self.rec_states.extend_from_slice(x);
        self.rec_mems.extend_from_slice(m);
        self.rec_cache.get_mut().take();
    }

    /// Recorded observed-signal traces: one time series per tap (parallel to
    /// [`tap_labels`], aligned with `times`) — the static-compile analogue of
    /// scope data, ready to plot.
    ///
    /// [`tap_labels`]: CompiledSimulation::tap_labels
    pub fn recordings(&self) -> Vec<Vec<f64>> {
        if let Some(cached) = self.rec_cache.borrow().as_ref() {
            return cached.clone();
        }
        let n = self.tap_labels.len();
        let mut series: Vec<Vec<f64>> = vec![Vec::with_capacity(self.rec_times.len()); n];
        for i in 0..self.rec_times.len() {
            let (x, m, t) = (self.state_row(i), self.mem_row(i), self.rec_times[i]);
            let row = self.taps.call(&[x, m, &[t]]); // taps may depend on t / discrete memory
            for (j, v) in row.into_iter().enumerate() {
                series[j].push(v);
            }
        }
        *self.rec_cache.borrow_mut() = Some(series.clone());
        series
    }

    /// Rewind state `X` and discrete memory `M` to their compile-time initial
    /// values, set the clock to `time`, and reset the recorded trajectory to a
    /// single point. Mirrors `Simulation::reset`; parameters and solver choice are
    /// left untouched. `run(.., reset=true)` calls this implicitly.
    pub fn reset(&mut self, time: f64) {
        self.x = self.x0.clone();
        self.m = self.m0.clone();
        self.time = time;
        self.rec_times = vec![time];
        self.rec_states = self.x0.clone(); // flat, one row of stride n_state
        self.rec_mems = self.m0.clone();
        self.accepted_steps = 0;
        self.rec_cache.get_mut().take();
    }

    /// Select the integration method by solver class name (`factory_from_name`),
    /// with adaptive LTE tolerances `atol`/`rtol`. Mirrors `Simulation::set_solver`;
    /// implicit methods drive the compile-time symbolic Jacobian. An unknown name
    /// falls back to `RKBS32` (the Python `set_solver` rejects it up front).
    pub fn set_solver(&mut self, name: impl Into<String>, atol: f64, rtol: f64) {
        self.solver = name.into();
        self.atol = atol;
        self.rtol = rtol;
    }

    /// Enable or disable run-time logging (mirrors `Simulation`'s logger). When
    /// enabling, logs a one-line `COMPILE` summary; each subsequent `run` then
    /// prints the same `SOLVER` and interleaved `TRANSIENT` progress a `Simulation`
    /// run does. `Simulation::compile` calls this to inherit the source sim's
    /// logging; toggle it via the compiled object's `log` attribute.
    pub fn set_logging(&mut self, enabled: bool) {
        self.logger = crate::utils::logger::Logger::new(enabled, "compiled");
        self.logger.info(&format!(
            "COMPILE (states: {}, memory: {}, events: {})",
            self.n_state, self.n_mem, self.events.len()
        ));
    }

    /// Whether run-time logging is currently on.
    pub fn logging_enabled(&self) -> bool {
        self.logger.enabled
    }

    /// Set the current simulation time without touching state or the recorded
    /// trajectory; the next `run` resumes from `t`. Use `reset` to also rewind
    /// state and clear recordings.
    pub fn set_time(&mut self, t: f64) {
        self.time = t;
    }

    /// Advance the simulation by `duration`, recording the trajectory. Mirrors
    /// `Simulation::run`: stateful (continues from the current state/time) unless
    /// `reset` is set, in which case it rewinds first. Uses the standalone solver
    /// `integrate` over the fused `dX/dt` — explicit RK or implicit (ESDIRK/DIRK,
    /// with the native symbolic Jacobian). Read results via `times` / `states` /
    /// `recordings` afterwards.
    pub fn run(&mut self, duration: f64, reset: bool, adaptive: bool) {
        use crate::utils::logger::ProgressTracker;
        if reset {
            self.reset(0.0);
        }
        // Adaptive stepping requires the chosen solver to carry an embedded error
        // estimate — mirror `Simulation` (`adaptive && engine.is_adaptive`). Under
        // a fixed-step solver, an `adaptive=true` request would otherwise drive the
        // event locator's `dt` down to `dt_min` (a runaway). Resolve the solver's
        // own adaptivity once and gate the run flag with it (this also keeps the
        // SOLVER log line in sync with the interpreted run).
        let solver_adaptive = {
            use crate::solvers::factories::{factory_from_name, rkbs32_factory};
            let factory = factory_from_name(&self.solver, self.atol, self.rtol)
                .unwrap_or_else(|| rkbs32_factory(self.atol, self.rtol));
            factory(&self.x).is_adaptive
        };
        let adaptive = adaptive && solver_adaptive;
        if self.logger.enabled {
            self.logger.info(&format!(
                "SOLVER (states: {}) -> {} (adaptive: {})",
                self.n_state, self.solver, if adaptive { "True" } else { "False" },
            ));
        }
        // Both run loops drive the standalone solver's per-step `take_step` and
        // update the tracker each accepted step, so the `TRANSIENT` progress is
        // interleaved exactly like `Simulation::run`.
        let mut tracker = ProgressTracker::new(duration, "TRANSIENT", self.logger.enabled);
        tracker.start();
        if self.events.is_empty() {
            self.run_continuous(duration, adaptive, &mut tracker);
        } else {
            self.run_with_events(duration, adaptive, &mut tracker);
        }
        tracker.close();
    }

    /// Run `param_sets.len()` parameter variations in parallel (rayon), each on
    /// an independent clone with its own scratch and trajectory buffers — the
    /// throughput lever for parameter sweeps and Monte Carlo (issue #45).
    ///
    /// Each entry of `param_sets` is a list of `(param_name, value)` overrides
    /// (unknown names are ignored, matching `set_param`); every run starts from
    /// the compile-time initial state (`reset=true`). Returns the final global
    /// state of each run, in input order (deterministic — identical to running
    /// the sweep serially). `duration` and `adaptive` match [`run`].
    ///
    /// The base simulation is cloned once per run up front (it is `!Sync`, so it
    /// cannot be shared across threads); the clones are then integrated
    /// concurrently with no shared mutable state, giving near-linear speedup up
    /// to the core count. For very long trajectories, set an output stride / cap
    /// on the base first ([`set_output_stride`], [`set_max_samples`]) to bound
    /// per-run memory.
    ///
    /// [`run`]: CompiledSimulation::run
    /// [`set_output_stride`]: CompiledSimulation::set_output_stride
    /// [`set_max_samples`]: CompiledSimulation::set_max_samples
    #[cfg(feature = "parallel")]
    pub fn run_batch(
        &self,
        param_sets: &[Vec<(String, f64)>],
        duration: f64,
        adaptive: bool,
    ) -> Vec<Vec<f64>> {
        use rayon::prelude::*;
        // Clone sequentially (self is `!Sync`), then run the owned, `Send` clones
        // across the rayon pool. Logging is silenced on the clones so parallel
        // runs don't interleave progress output.
        //
        // Chunked: at most `chunk` clones (tape + trajectory buffers) are alive
        // at a time, so a 10k-run sweep holds O(threads) copies in memory, not
        // O(runs). 4× oversubscription keeps the pool busy across uneven run
        // times; results are per-chunk in input order, so the output is
        // identical to the unchunked (and serial) sweep.
        let chunk = rayon::current_num_threads().max(1) * 4;
        let mut out: Vec<Vec<f64>> = Vec::with_capacity(param_sets.len());
        for batch in param_sets.chunks(chunk) {
            let mut sims: Vec<CompiledSimulation> = batch
                .iter()
                .map(|_| {
                    let mut c = self.clone();
                    c.logger = crate::utils::logger::Logger::disabled();
                    c
                })
                .collect();
            sims.par_iter_mut()
                .zip(batch.par_iter())
                .for_each(|(sim, params)| {
                    for (name, val) in params {
                        sim.set_param(name, *val);
                    }
                    sim.run(duration, true, adaptive);
                });
            out.extend(sims.into_iter().map(|s| s.x));
        }
        out
    }

    /// Pure-continuous fast path: step `dX/dt = F(X, M, t)` (M constant) over the
    /// span with the standalone solver's `take_step` — the shared per-step
    /// primitive (explicit RK, implicit RK with the native symbolic Jacobian, and
    /// tableau-less implicit). Uses `*Solver::integrate`'s initial-step heuristic
    /// and adaptive `dt` control, but `Simulation::run`'s loop bound and end-time
    /// semantics (`while time < t_end`; adaptive lands exactly on `t_end`,
    /// fixed-step ends within one step past it) — a compiled run must end where
    /// the interpreted run of the same model ends. Note the standalone
    /// `integrate()` itself has a stricter no-overshoot contract (issue #26);
    /// that contract is deliberately NOT this one.
    fn run_continuous(
        &mut self,
        duration: f64,
        adaptive: bool,
        tracker: &mut crate::utils::logger::ProgressTracker,
    ) {
        use crate::optim::anderson::{NewtonAnderson, Optimizer};
        use crate::solvers::factories::{factory_from_name, rkbs32_factory};
        use crate::solvers::solver::{auto_initial_step, Jacobian};

        let factory = factory_from_name(&self.solver, self.atol, self.rtol)
            .unwrap_or_else(|| rkbs32_factory(self.atol, self.rtol));
        let mut s = factory(&self.x);
        let m = self.m.clone(); // constant (no events)
        let dt_min = 1e-12;
        let t_start = self.time;
        let t_end = t_start + duration;
        let mut dt = self.dt;
        let dt_max = t_end.max(dt);
        let n = self.n_state;

        // Implicit scratch: a sparse-aware Jacobian operator over the derivative
        // graph, plus a once-evaluated value when the Jacobian is a globally
        // constant (LTI) matrix. Explicit methods build neither. The operator
        // gates Scalar / dense Matrix / Sparse exactly like the static-compile and
        // interpreted paths, so a large nonlinear sparse system gets the sparse
        // Newton solve here too (not just a dense matrix).
        let jac_op = if s.is_implicit {
            Some(crate::blocks::operator::Operator::jac_only(
                crate::blocks::blockops::RegionGraph::Fixed(self.jac_src.clone()),
            ))
        } else {
            None
        };
        let const_jac: Option<crate::solvers::solver::Jacobian> =
            if s.is_implicit && self.jacobian_is_constant() {
                jac_op.as_ref().and_then(|op| op.jacobian_wrt_state(&self.x, &[], t_start, &m))
            } else {
                None
            };
        let mut opt = Optimizer::NewtonAnderson(NewtonAnderson::with_defaults());
        let (mut g_buf, mut jac_buf, mut f_buf) = (Vec::new(), Vec::new(), Vec::new());
        let n_out = self.fused.n_out;

        // Initial-step heuristic (adaptive) — matches `*Solver::integrate`.
        if adaptive {
            let order = s.n.max(1);
            let rhs = |x: &[f64], t: f64| self.fused.call(&[x, &m, &[t]]);
            dt = auto_initial_step(
                &rhs, &s.x, t_start, dt, s.tolerance_lte_abs, s.tolerance_lte_rel,
                order, dt_min, dt_max,
            );
        }

        // Loop bound — the single semantic source of truth is `Simulation::run`
        // (`while self.time < end_time`), which `run` documents this loop as
        // mirroring: a fixed-step run takes the first step that reaches or
        // passes `t_end` (final time in `[t_end, t_end + dt)`, pathsim parity),
        // an adaptive run lands exactly on `t_end` via the overshoot clamp
        // below. The previous `time < t_end + dt` bound took one full step
        // MORE than the interpreted engine on the same model — compiled and
        // interpreted trajectories must end identically.
        let mut time = t_start;
        while time < t_end {
            // `rhs` (and `jac`) borrow `self.fused`; scope them to the step so the
            // recording push below can borrow `self.rec_*` mutably. The RHS writes
            // straight into the caller-owned `f_buf` via `call_into` — the marketed
            // compiled fast path is now zero-alloc per RK stage (issue #41).
            let (success, scale) = {
                let mut rhs = |x: &[f64], t: f64, out: &mut Vec<f64>| {
                    out.resize(n_out, 0.0);
                    self.fused.call_into(&[x, &m, &[t]], out);
                };
                if let Some(ref op) = jac_op {
                    if let Some(ref cj) = const_jac {
                        // LTI system: the once-evaluated Jacobian is read by
                        // reference every Newton iteration — no O(n^2) clone (#42).
                        s.take_step(&mut rhs, None, Some(cj), time, dt, 200, Some(&mut opt), &mut g_buf, &mut jac_buf, &mut f_buf)
                    } else {
                        let jac = |x: &[f64], t: f64| -> Jacobian {
                            op.jacobian_wrt_state(x, &[], t, &m)
                                .unwrap_or_else(|| Jacobian::Matrix(vec![0.0; n * n], n))
                        };
                        s.take_step(&mut rhs, Some(&jac), None, time, dt, 200, Some(&mut opt), &mut g_buf, &mut jac_buf, &mut f_buf)
                    }
                } else {
                    s.take_step(&mut rhs, None, None, time, dt, 0, None, &mut g_buf, &mut jac_buf, &mut f_buf)
                }
            };
            if adaptive && !success {
                s.revert();
            } else {
                time += dt;
                self.record_step(time, &s.x, &m, false);
                tracker.update(((time - t_start) / duration).clamp(0.0, 1.0), true);
            }
            if adaptive {
                if let Some(sc) = scale {
                    let new_dt = sc * dt;
                    if new_dt < dt_min {
                        crate::utils::sink::warn(&format!(
                            "warning: solver requires dt < dt_min ({:e}) at t={:.6}; \
                             returning truncated trajectory",
                            dt_min, time
                        ));
                        break;
                    }
                    dt = new_dt.clamp(dt_min, dt_max);
                }
                if time + dt > t_end {
                    dt = t_end - time;
                }
            }
        }

        self.time = time;
        self.x = s.x.clone();
        // Guarantee the final true state is in the trajectory even under
        // downsampling / capping (issue #44): append it if the last recorded
        // sample isn't already this step.
        if self.rec_times.last() != Some(&time) {
            self.record_step(time, &s.x, &m, true);
        }
    }

    /// Build runtime `SimEvent`s from the compiled events, with guard/effect
    /// closures over a shared `(X, M)` scratch — reusing the per-block runtime's
    /// detect/locate/resolve machinery verbatim.
    fn build_sim_events(
        &self,
        shared: &Rc<crate::utils::fastcell::FastCell<EventScratch>>,
    ) -> Vec<crate::simulation::SimEventRef> {
        let mut out: Vec<crate::simulation::SimEventRef> = Vec::with_capacity(self.events.len());
        for ce in &self.events {
            // Guard: evaluate the pre-built guard tape at the current (X, M, t).
            let make_guard = || -> Box<dyn Fn(f64) -> f64> {
                let sh = shared.clone();
                let guard = ce.guard.clone().expect("ZeroCross/Condition event has a guard");
                Box::new(move |t: f64| {
                    let s = sh.borrow();
                    guard.call(&[&s.x, &s.m, &[t]])[0]
                })
            };
            // Effect: evaluate the effect tape at the current (X, M, t), then
            // write its outputs into the global M / X slots (reads old state,
            // applies all writes — matches the IR's "writes after ops" rule).
            let make_act = || -> Box<dyn FnMut(f64)> {
                let sh = shared.clone();
                let eff = ce.effect.clone();
                let tgts = ce.targets.clone();
                Box::new(move |t: f64| {
                    let outs = {
                        let s = sh.borrow();
                        eff.call(&[&s.x, &s.m, &[t]])
                    };
                    let s = sh.borrow_mut();
                    for (val, tgt) in outs.iter().zip(tgts.iter()) {
                        match tgt {
                            WriteTarget::Mem(i) => s.m[*i] = *val,
                            WriteTarget::State(i) => s.x[*i] = *val,
                        }
                    }
                })
            };
            out.push(assemble_sim_event(&ce.kind, make_guard, make_act));
        }
        out
    }

    /// Event-aware run loop (explicit stepping over the fused `dX/dt`): step the
    /// continuous state with `M` held constant, then detect / locate (adaptive
    /// step-shrink) / resolve block-internal events, mutating `M` (and `X` for
    /// discrete resets). Mirrors `Simulation::timestep`'s event handling.
    fn run_with_events(
        &mut self,
        duration: f64,
        adaptive: bool,
        tracker: &mut crate::utils::logger::ProgressTracker,
    ) {
        use crate::optim::anderson::{NewtonAnderson, Optimizer};
        use crate::solvers::factories::{factory_from_name, rkbs32_factory};
        use crate::solvers::solver::Jacobian;
        use crate::utils::fastcell::FastCell;

        let factory = factory_from_name(&self.solver, self.atol, self.rtol)
            .unwrap_or_else(|| rkbs32_factory(self.atol, self.rtol));
        let mut s = factory(&self.x);
        let dt_min = 1e-12;
        let t_end = self.time + duration;
        let dt_max = t_end.max(self.dt);
        let mut dt = self.dt;
        let mut time = self.time;
        let n_state = self.n_state;

        // Implicit-solver scratch (unused for explicit methods).
        let mut opt = Optimizer::NewtonAnderson(NewtonAnderson::with_defaults());
        let (mut g_buf, mut jac_buf, mut f_buf) = (Vec::new(), Vec::new(), Vec::new());
        let n_out = self.fused.n_out;

        // Build the symbolic Jacobian on first implicit use (cached); explicit
        // methods never need it. A globally constant Jacobian is independent of
        // state, memory and time, so evaluate the tape once for the whole run
        // (events mutate M/X, never the constant matrix) and reuse the buffer
        // across every Newton step.
        let jac_op = if s.is_implicit {
            Some(crate::blocks::operator::Operator::jac_only(
                crate::blocks::blockops::RegionGraph::Fixed(self.jac_src.clone()),
            ))
        } else {
            None
        };
        let const_jac: Option<crate::solvers::solver::Jacobian> =
            if s.is_implicit && self.jacobian_is_constant() {
                jac_op.as_ref().and_then(|op| op.jacobian_wrt_state(&self.x, &[], time, &self.m))
            } else {
                None
            };

        let shared = Rc::new(FastCell::new(EventScratch { x: self.x.clone(), m: self.m.clone() }));
        let sim_events = self.build_sim_events(&shared);

        // Resolve events already active at the start time (e.g. Schedule@phase 0).
        for e in &sim_events {
            e.borrow_mut().buffer(time);
        }
        for e in &sim_events {
            let fired = e.borrow_mut().detect(time).0;
            if fired {
                e.borrow_mut().resolve(time);
            }
        }
        {
            let sh = shared.borrow();
            s.x.copy_from_slice(&sh.x);
        }

        // Discrete memory is constant during a step and only changes when an
        // event resolves. Hold one owned copy and refresh it only after a resolve
        // (issue #47) — the common no-event step no longer clones `M` each time.
        let mut m_step = shared.borrow().m.clone();

        // Loop bound + end-time semantics: mirror `Simulation::run` exactly like
        // the no-event loop above — adaptive runs clamp the last step onto
        // `t_end`, fixed-step runs take the first step that reaches or passes it.
        // A model must end at the same time whether or not it happens to contain
        // block-internal events (and whether or not it was compiled).
        let mut iters = 0usize;
        while time < t_end {
            iters += 1;
            if iters > 50_000_000 {
                break; // runaway guard (e.g. an event firing every step)
            }
            if adaptive && time + dt > t_end {
                dt = t_end - time;
            }

            {
                let sh = shared.borrow_mut();
                sh.x.copy_from_slice(&s.x);
            }
            for e in &sim_events {
                e.borrow_mut().buffer(time);
            }

            // One step over the fused dX/dt (M held constant). Explicit RK or
            // implicit Newton — identical event handling around it. The RHS
            // writes into the caller-owned `f_buf` via `call_into` (issue #41).
            let mut rhs = |x: &[f64], t: f64, out: &mut Vec<f64>| {
                out.resize(n_out, 0.0);
                self.fused.call_into(&[x, &m_step, &[t]], out);
            };
            let (success, scale) = if let Some(ref op) = jac_op {
                if let Some(ref cj) = const_jac {
                    // LTI system: read the once-evaluated Jacobian by reference
                    // every Newton iteration — no O(n^2) clone (issue #42).
                    s.take_step(&mut rhs, None, Some(cj), time, dt, 200, Some(&mut opt), &mut g_buf, &mut jac_buf, &mut f_buf)
                } else {
                    let jac = |x: &[f64], t: f64| -> Jacobian {
                        op.jacobian_wrt_state(x, &[], t, &m_step)
                            .unwrap_or_else(|| Jacobian::Matrix(vec![0.0; n_state * n_state], n_state))
                    };
                    s.take_step(&mut rhs, Some(&jac), None, time, dt, 200, Some(&mut opt), &mut g_buf, &mut jac_buf, &mut f_buf)
                }
            } else {
                s.take_step(&mut rhs, None, None, time, dt, 0, None, &mut g_buf, &mut jac_buf, &mut f_buf)
            };
            if adaptive && !success {
                s.revert();
                if let Some(sc) = scale {
                    dt = (sc * dt).clamp(dt_min, dt_max);
                }
                continue;
            }

            let t_new = time + dt;
            {
                let sh = shared.borrow_mut();
                sh.x.copy_from_slice(&s.x);
            }

            // Detect events over the step.
            let mut detected: Vec<(usize, bool, f64)> = Vec::new();
            for (i, e) in sim_events.iter().enumerate() {
                let (det, close, ratio) = e.borrow_mut().detect(t_new);
                if det {
                    detected.push((i, close, ratio));
                }
            }
            if !detected.is_empty() {
                if adaptive {
                    // Locate: if any event isn't close yet, shrink the step to the
                    // earliest one and retry (root-find via the step controller).
                    let earliest = detected
                        .iter()
                        .filter(|(_, c, _)| !*c)
                        .map(|(_, _, r)| *r)
                        .fold(f64::INFINITY, f64::min);
                    if earliest.is_finite() {
                        s.revert();
                        dt = (earliest * dt).max(dt_min);
                        continue;
                    }
                }
                // Resolve: apply effects → mutate M / X. The effect tape reads
                // `shared.x` (the step-end state) and re-evaluates inputs at the
                // event time, so resolve at `t_new` (adaptive landed the step on
                // the event; fixed-step resolves at the boundary).
                for (i, _, _) in &detected {
                    sim_events[*i].borrow_mut().resolve(t_new);
                }
                let sh = shared.borrow();
                s.x.copy_from_slice(&sh.x);
                // Memory may have changed; refresh the step-constant copy (only
                // paid on steps where an event actually fired — issue #47).
                m_step.clear();
                m_step.extend_from_slice(&sh.m);
            }

            time = t_new;
            let mem_now = shared.borrow().m.clone();
            self.record_step(time, &s.x, &mem_now, false);
            tracker.update(((time - (t_end - duration)) / duration).clamp(0.0, 1.0), true);

            dt = if adaptive {
                scale.map(|sc| (sc * dt).clamp(dt_min, dt_max)).unwrap_or(dt)
            } else {
                self.dt // restore nominal step after a possible event-driven shrink
            };
        }

        self.time = time;
        self.x = s.x.clone();
        self.m = shared.borrow().m.clone();
        // Guarantee the final true state is recorded even under downsampling /
        // capping (issue #44).
        if self.rec_times.last() != Some(&time) {
            let mem_now = self.m.clone();
            self.record_step(time, &self.x.clone(), &mem_now, true);
        }
    }
}
