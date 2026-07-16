//! Dense output — a continuous extension of the last completed step.
//!
//! After a solver finishes a step `x₀ → x₁` over `dt`, `Solver::interpolate`
//! evaluates the solution at any `θ ∈ [0, 1]` inside that step *without*
//! re-integrating. Consumers: output sampling at arbitrary `t_eval` points
//! (no step-size locking), event localisation on the interpolant (a polynomial
//! evaluation instead of a bisection re-integration), and — later — interval
//! hulls over the step.
//!
//! Everything is zero-copy: the interpolant reads the live solver buffers
//! (`history[0]` = x₀, `x` = x₁, `ks` = stage slopes), which stay intact from
//! the end of a step until the next step's buffer. The validity window is
//! owned by exactly two methods — `Solver::dense_open` (called by the `step`
//! wrapper on a successful final stage) and `Solver::dense_close` (called by
//! `push_state_history`, `revert`, `reset`, and the external `set`) — so no
//! per-step snapshot is ever taken and every state mutation that breaks
//! coherence closes the window through the same choke point.
//!
//! Interpolant selection, best first:
//! 0. **Multistep history polynomial** (`dense_hist > 0`, BDF/GEAR): the
//!    Lagrange polynomial through `x₁` and the newest `dense_hist` history
//!    states at their variable-step nodes — the polynomial the BDF formula
//!    itself fits, so it is order-consistent with the step.
//! 1. **Tableau interpolant** (`Tableau::di`, e.g. RKDP54's Shampine
//!    continuous extension): `x(θ) = x₀ + dt·Σᵢ kᵢ·Σ_q di[i][q]·θ^(q+1)`.
//! 2. **Cubic Hermite** when both endpoint slopes exist (`dense_f0 && dense_f1`:
//!    FSAL explicit RK, stiffly-accurate DIRK/ESDIRK): O(dt⁴) local error.
//! 3. **Quadratic Hermite** on one endpoint slope (either side): O(dt³).
//! 4. **Linear** between x₀ and x₁ otherwise (EUF/EUB).

use crate::solvers::solver::Solver;

impl Solver {
    /// True while [`interpolate`](Self::interpolate) can evaluate the last
    /// completed step (between a step's final stage and the next buffer).
    pub fn can_interpolate(&self) -> bool {
        self.dense_valid && !self.history.is_empty()
    }

    /// Interpolation order of the active dense output (the polynomial's local
    /// accuracy order `p` in `O(dt^p)`): 1 = none/linear fallback data.
    pub fn dense_order(&self) -> usize {
        // Mirror interp_history's own availability conditions so the reported
        // order never disagrees with the tier actually evaluated.
        if self.dense_hist > 0
            && self.history.len() >= self.dense_hist
            && self.history_dt.len() >= self.dense_hist
            && self.dense_dt > 0.0
        {
            // Multistep history polynomial: degree dense_hist (= BDF order).
            return self.dense_hist + 1;
        }
        if !self.dense_di.is_empty() {
            // θ-polynomial degree + 1 (RKDP54: 4 columns → O(dt⁵) local).
            return self.dense_di.iter().map(|r| r.len()).max().unwrap_or(0) + 1;
        }
        match (self.dense_f0 && self.has_k(0), self.dense_f1 && self.has_k(self.s.saturating_sub(1))) {
            (true, true) => 4,
            (true, false) | (false, true) => 3,
            (false, false) => 2,
        }
    }

    #[inline]
    fn has_k(&self, i: usize) -> bool {
        self.ks.get(i).is_some_and(|k| k.len() == self.x.len())
    }

    /// Evaluate the continuous extension of the last completed step at
    /// `θ ∈ [0, 1]` (θ = 0 → step start `x₀`, θ = 1 → step end `x₁`), writing
    /// the interpolated state into `out`. Returns `false` (leaving `out`
    /// untouched) when no completed step is available — before the first step,
    /// after a rejection, or once the next step has begun.
    pub fn interpolate(&self, theta: f64, out: &mut Vec<f64>) -> bool {
        if !self.can_interpolate() {
            return false;
        }
        // While seeking, `x` holds a repositioned state; the true x₁ is stashed.
        let x1: &[f64] = if self.dense_seeking { &self.dense_stash } else { &self.x };
        out.clear();
        out.resize(x1.len(), 0.0);
        self.interp_core(theta, x1, out)
    }

    /// Temporarily reposition `x` at the interpolant value for `θ` (event
    /// localisation). The first seek stashes `x₁`; further seeks re-evaluate
    /// from the stash, and [`dense_seek_end`](Self::dense_seek_end) restores
    /// `x₁`. Returns `false` (state untouched) when no interpolant is valid.
    pub fn dense_seek(&mut self, theta: f64) -> bool {
        if !self.can_interpolate() {
            return false;
        }
        if !self.dense_seeking {
            self.dense_stash.clear();
            self.dense_stash.extend_from_slice(&self.x);
            self.dense_seeking = true;
        }
        // `interp_core` reads x₁ from the stash, so `x` is free to receive the
        // seeked state directly (taken out to satisfy the borrow checker).
        let mut buf = std::mem::take(&mut self.x);
        buf.clear();
        buf.resize(self.dense_stash.len(), 0.0);
        let ok = self.interp_core(theta, &self.dense_stash, &mut buf);
        if !ok {
            buf.clear();
            buf.extend_from_slice(&self.dense_stash);
        }
        self.x = buf;
        ok
    }

    /// Restore `x₁` after one or more [`dense_seek`](Self::dense_seek) calls.
    /// No-op when not seeking.
    pub fn dense_seek_end(&mut self) {
        if self.dense_seeking {
            self.x.clear();
            self.x.extend_from_slice(&self.dense_stash);
            self.dense_seeking = false;
        }
    }

    /// Shared interpolation core: evaluate at `θ` with an explicit right
    /// endpoint `x1` (the live `x`, or the seek stash), writing into `out`
    /// (already sized to `n`). Reads `x₀ = history[0]` and the stage slopes.
    fn interp_core(&self, theta: f64, x1: &[f64], out: &mut [f64]) -> bool {
        let x0 = &self.history[0];
        let n = x1.len();
        if x0.len() != n || out.len() != n {
            return false;
        }
        let h = self.dense_dt;
        // Slope availability against `n`, NOT `self.x.len()`: during a seek the
        // live `x` is temporarily taken/repositioned while x₁ is the stash.
        let has_k = |i: usize| self.ks.get(i).is_some_and(|k| k.len() == n);

        // 0. Multistep (BDF/GEAR) history interpolant — the polynomial the BDF
        // formula itself assumes; falls through on insufficient history.
        if self.dense_hist > 0 && self.interp_history(theta, x1, out) {
            return true;
        }

        // 1. Tableau continuous extension over the stage slopes.
        if !self.dense_di.is_empty() {
            let mut ok = true;
            for (i, row) in self.dense_di.iter().enumerate() {
                if !has_k(i) {
                    ok = false;
                    break;
                }
                // w = Σ_q di[i][q] · θ^(q+1), Horner over θ.
                let mut w = 0.0;
                for &c in row.iter().rev() {
                    w = theta * (c + w);
                }
                let k = &self.ks[i];
                for j in 0..n {
                    out[j] += w * k[j];
                }
            }
            if ok {
                for j in 0..n {
                    out[j] = x0[j] + h * out[j];
                }
                return true;
            }
            // Incomplete slope data (e.g. dynamic resize mid-run): fall through
            // to the Hermite tiers below.
        }

        let f0 = (self.dense_f0 && has_k(0)).then(|| &self.ks[0]);
        let f1 = (self.dense_f1 && self.s >= 1 && has_k(self.s - 1))
            .then(|| &self.ks[self.s - 1]);

        match (f0, f1) {
            // 2. Cubic Hermite from both endpoint slopes.
            (Some(f0), Some(f1)) => {
                let t2 = theta * theta;
                let t3 = t2 * theta;
                let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
                let h10 = t3 - 2.0 * t2 + theta;
                let h01 = -2.0 * t3 + 3.0 * t2;
                let h11 = t3 - t2;
                for j in 0..n {
                    out[j] = h00 * x0[j] + h * h10 * f0[j] + h01 * x1[j] + h * h11 * f1[j];
                }
            }
            // 3a. Quadratic Hermite anchored at the left endpoint slope.
            (Some(f0), None) => {
                for j in 0..n {
                    let c = x1[j] - x0[j] - h * f0[j];
                    out[j] = x0[j] + theta * (h * f0[j] + theta * c);
                }
            }
            // 3b. Quadratic Hermite anchored at the right endpoint slope.
            (None, Some(f1)) => {
                let u = theta - 1.0;
                for j in 0..n {
                    let c = x0[j] - x1[j] + h * f1[j];
                    out[j] = x1[j] + u * (h * f1[j] + u * c);
                }
            }
            // 4. Linear fallback.
            (None, None) => {
                for j in 0..n {
                    out[j] = x0[j] + theta * (x1[j] - x0[j]);
                }
            }
        }
        true
    }

    /// Multistep dense output: evaluate the degree-`m` Lagrange polynomial
    /// through `x₁` (θ = 1) and the `m = dense_hist` newest history states at
    /// their actual variable-step node positions, `m ≤ 5` in practice (BDF
    /// order cap). This is exactly the polynomial the variable-step BDF
    /// formula fits, so the continuous extension is order-consistent with the
    /// step itself. Returns `false` (caller falls through to the single-step
    /// tiers) when history or `history_dt` are too shallow.
    fn interp_history(&self, theta: f64, x1: &[f64], out: &mut [f64]) -> bool {
        const MAX_NODES: usize = 8;
        let n = x1.len();
        let h = self.dense_dt;
        // BDF caps the order at 5, so 6 nodes suffice; clamp defensively so a
        // misconfigured dense_hist degrades instead of indexing out of range.
        let m = self.dense_hist.min(MAX_NODES - 1);
        // Node positions need the current step (dense_dt) plus the `m − 1`
        // past steps `history_dt[1..m]` (`history_dt[0]` IS the current step,
        // pushed by the multistep buffer).
        if h <= 0.0 || self.history.len() < m || self.history_dt.len() < m {
            return false;
        }
        // τ over the last step: x₁ at 1, history[0] at 0, then each older
        // state one (variable) step further left. This is the same cumulative
        // dt-ratio node convention as `bdf::compute_bdf_alphas` (θ anchored at
        // x_n = 0 there; shifted by +1 here so θ parametrises the last step) —
        // a change to the dt/history pairing must be mirrored in both.
        let mut tau = [0.0f64; MAX_NODES];
        tau[0] = 1.0;
        tau[1] = 0.0;
        for j in 1..m {
            // A zero or negative buffered step would collapse two nodes and
            // divide the Lagrange weights by zero — fall through instead.
            let hd = self.history_dt[j];
            if hd <= 0.0 {
                return false;
            }
            tau[j + 1] = tau[j] - hd / h;
        }
        let value = |k: usize| -> &[f64] {
            if k == 0 { x1 } else { &self.history[k - 1] }
        };
        for k in 0..=m {
            if value(k).len() != n {
                return false;
            }
        }
        // Lagrange basis weights at θ (≤ 6 nodes: direct product form). `out`
        // arrives zeroed from `interpolate`/`dense_seek`.
        for k in 0..=m {
            let mut w = 1.0;
            for i in 0..=m {
                if i != k {
                    w *= (theta - tau[i]) / (tau[k] - tau[i]);
                }
            }
            let v = value(k);
            for j in 0..n {
                out[j] += w * v[j];
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::solvers::factories::*;
    use crate::solvers::solver::Solver;
    use crate::solvers::tableaus;

    /// Drive one explicit step through [`Solver::take_step`] — the same
    /// buffer/stage/step sequence `Simulation::timestep` performs, via the
    /// shared API instead of a hand-rolled copy of the stage loop.
    fn drive_step(s: &mut Solver, rhs: &dyn Fn(f64, &[f64]) -> Vec<f64>, t: f64, dt: f64) {
        let (mut g_buf, mut jac_buf, mut f_buf) = (Vec::new(), Vec::new(), Vec::new());
        s.take_step(
            &mut |x, ts, out| {
                out.clear();
                out.extend(rhs(ts, x));
            },
            None,
            None,
            t,
            dt,
            0,
            None,
            &mut g_buf,
            &mut jac_buf,
            &mut f_buf,
        );
    }

    fn exp_rhs(_t: f64, x: &[f64]) -> Vec<f64> {
        x.to_vec() // x' = x, exact solution e^t from x(0) = 1
    }

    #[test]
    fn classification_per_tableau() {
        // (factory, expect_f0, expect_f1, expect_di)
        let cases: Vec<(Solver, bool, bool, bool)> = vec![
            (rk4_factory()(&[1.0]), true, false, false),
            (ssprk22_factory()(&[1.0]), true, false, false),
            (rkf45_factory(1e6, 1e6)(&[1.0]), true, false, false),
            (rkck54_factory(1e6, 1e6)(&[1.0]), true, false, false),
            (rkbs32_factory(1e6, 1e6)(&[1.0]), true, true, true),
            (rkdp54_factory(1e6, 1e6)(&[1.0]), true, true, true),
            (rkv65_factory(1e6, 1e6)(&[1.0]), true, true, false),
            (esdirk32_factory(1e6, 1e6)(&[1.0]), true, true, false),
        ];
        for (s, f0, f1, di) in cases {
            assert_eq!(s.dense_f0, f0, "{}: dense_f0", s.type_name);
            assert_eq!(s.dense_f1, f1, "{}: dense_f1", s.type_name);
            assert_eq!(!s.dense_di.is_empty(), di, "{}: dense_di", s.type_name);
        }
    }

    #[test]
    fn di_row_sums_equal_b() {
        // θ = 1 must reproduce x₁: the interpolant matrix rows must sum to the
        // propagating b row. Guards every current and future `di` definition.
        for t in tableaus::ALL {
            if t.di.is_empty() {
                continue;
            }
            let b: &[f64] = if t.a_final.is_empty() { t.bt[t.s - 1] } else { t.a_final };
            assert_eq!(t.di.len(), t.s, "{}: di must have one row per stage", t.name);
            for (i, row) in t.di.iter().enumerate() {
                let sum: f64 = row.iter().sum();
                let bi = b.get(i).copied().unwrap_or(0.0);
                assert!(
                    (sum - bi).abs() < 1e-12,
                    "{}: di row {} sums to {} != b_{} = {}",
                    t.name, i, sum, i, bi
                );
            }
        }
    }

    #[test]
    fn endpoints_are_exact() {
        let mk: Vec<Box<dyn Fn(&[f64]) -> Solver>> = vec![
            Box::new(|iv| rk4_factory()(iv)),
            Box::new(|iv| rkbs32_factory(1e6, 1e6)(iv)),
            Box::new(|iv| rkdp54_factory(1e6, 1e6)(iv)),
            Box::new(|iv| rkv65_factory(1e6, 1e6)(iv)),
            Box::new(|iv| euf_factory()(iv)),
        ];
        for f in mk {
            let mut s = f(&[1.0, 2.0]);
            let rhs = |_t: f64, x: &[f64]| vec![x[0], -0.5 * x[1]];
            drive_step(&mut s, &rhs, 0.0, 0.1);
            assert!(s.can_interpolate(), "{}", s.type_name);
            let x0 = s.history[0].clone();
            let x1 = s.x.clone();
            let mut out = Vec::new();
            assert!(s.interpolate(0.0, &mut out));
            for j in 0..2 {
                assert!(
                    (out[j] - x0[j]).abs() < 1e-13 * x0[j].abs().max(1.0),
                    "{}: θ=0 must give x₀ ({} vs {})", s.type_name, out[j], x0[j]
                );
            }
            assert!(s.interpolate(1.0, &mut out));
            for j in 0..2 {
                assert!(
                    (out[j] - x1[j]).abs() < 1e-13 * x1[j].abs().max(1.0),
                    "{}: θ=1 must give x₁ ({} vs {})", s.type_name, out[j], x1[j]
                );
            }
        }
    }

    #[test]
    fn validity_window() {
        let mut s = rkdp54_factory(1e6, 1e6)(&[1.0]);
        assert!(!s.can_interpolate(), "no step completed yet");
        drive_step(&mut s, &exp_rhs, 0.0, 0.1);
        assert!(s.can_interpolate(), "window opens after the final stage");
        s.buffer(0.1);
        assert!(!s.can_interpolate(), "the next buffer closes the window");

        let mut s = rkdp54_factory(1e6, 1e6)(&[1.0]);
        drive_step(&mut s, &exp_rhs, 0.0, 0.1);
        s.revert();
        assert!(!s.can_interpolate(), "rejection closes the window");
    }

    /// Measured convergence order of the interpolant against the analytic
    /// solution of x' = x over one step from an exact initial state: the
    /// worst-case error over interior θ must shrink as O(dt^p).
    fn measured_order(mk: &dyn Fn(&[f64]) -> Solver) -> f64 {
        let steps = [0.4, 0.2, 0.1, 0.05];
        let mut errs = Vec::new();
        for &h in &steps {
            let mut s = mk(&[1.0]);
            drive_step(&mut s, &exp_rhs, 0.0, h);
            assert!(s.can_interpolate(), "{}", s.type_name);
            let mut out = Vec::new();
            let mut worst = 0.0f64;
            for &theta in &[0.25, 0.5, 0.75] {
                assert!(s.interpolate(theta, &mut out));
                worst = worst.max((out[0] - (theta * h).exp()).abs());
            }
            errs.push(worst);
        }
        // Mean observed order over the halvings.
        let mut p = 0.0;
        for w in errs.windows(2) {
            p += (w[0] / w[1]).log2();
        }
        p / (errs.len() - 1) as f64
    }

    #[test]
    fn interpolant_orders() {
        // (solver, expected local order p, slack)
        let cases: Vec<(Box<dyn Fn(&[f64]) -> Solver>, f64, &str)> = vec![
            // Shampine continuous extension: O(dt⁵) local.
            (Box::new(|iv: &[f64]| rkdp54_factory(1e6, 1e6)(iv)), 4.5, "RKDP54 di"),
            // Bogacki-Shampine extension: O(dt⁴) local.
            (Box::new(|iv: &[f64]| rkbs32_factory(1e6, 1e6)(iv)), 3.5, "RKBS32 di"),
            // Cubic Hermite (FSAL slopes, no di): O(dt⁴) local.
            (Box::new(|iv: &[f64]| rkv65_factory(1e6, 1e6)(iv)), 3.5, "RKV65 cubic Hermite"),
            // Quadratic Hermite (left slope only): O(dt³) local.
            (Box::new(|iv: &[f64]| rk4_factory()(iv)), 2.5, "RK4 quadratic Hermite"),
            // Linear between endpoints: O(dt²) local.
            (Box::new(|iv: &[f64]| euf_factory()(iv)), 1.6, "EUF linear"),
        ];
        for (mk, expect, label) in cases {
            let p = measured_order(mk.as_ref());
            assert!(p > expect, "{label}: measured order {p:.2} <= expected {expect}");
        }
    }

    #[test]
    fn history_interpolant_reproduces_polynomial_exactly() {
        // A degree-3 Lagrange polynomial through 4 nodes must reproduce a cubic
        // exactly, including variable past step sizes — this is the BDF/GEAR
        // dense-output tier driven purely by (history, history_dt, x₁).
        let p = |t: f64| [t * t * t, 2.0 * t * t * t - t]; // two components
        let (h, h1, h2) = (0.4, 0.3, 0.5); // current + two variable past steps
        let t1 = 2.0;
        let t0 = t1 - h;

        let mut s = Solver::with_defaults(&p(t1));
        s.history_maxlen = 6;
        s.history_dt_maxlen = 6;
        s.history.push_back(p(t0).to_vec());
        s.history.push_back(p(t0 - h1).to_vec());
        s.history.push_back(p(t0 - h1 - h2).to_vec());
        s.history_dt.push_back(h);
        s.history_dt.push_back(h1);
        s.history_dt.push_back(h2);
        s.dense_hist = 3;
        s.dense_dt = h;
        s.dense_valid = true;

        assert_eq!(s.dense_order(), 4);
        let mut out = Vec::new();
        for &theta in &[0.0, 0.25, 0.5, 0.75, 1.0, -0.5] {
            assert!(s.interpolate(theta, &mut out));
            let want = p(t0 + theta * h);
            for j in 0..2 {
                assert!(
                    (out[j] - want[j]).abs() < 1e-12 * want[j].abs().max(1.0),
                    "θ={theta} comp {j}: {} vs exact {}",
                    out[j], want[j]
                );
            }
        }
    }

    #[test]
    fn gear_dense_output_tracks_solution() {
        // Sim-level: x' = -x with GEAR52A; after the run the last completed
        // step's interpolant must track e^{-t} to tolerance scale.
        use crate::blocks::constructors::{amplifier, integrator, scope};
        use crate::connection::Connection;
        use crate::simulation::Simulation;

        let int_x = integrator(1.0);
        let amp = amplifier(-1.0);
        let s = scope(None, 0.0, vec![]);
        let conns = vec![
            Connection::single(&int_x, &amp),
            Connection::single(&amp, &int_x),
            Connection::single(&int_x, &s),
        ];
        let int_ref = int_x.clone();
        let mut sim = Simulation::with_defaults(vec![int_x, amp, s], conns);
        sim.set_solver(gear52a_factory(1e-9, 1e-9));
        sim.dt = 0.01;
        sim.run(2.0, true, true);

        let block = int_ref.borrow();
        let engine = block.engine.as_ref().expect("integrator engine");
        assert!(engine.can_interpolate(), "window must be open after the final step");
        assert!(engine.dense_hist >= 1, "GEAR must set its active order");
        let h = engine.dense_dt;
        let mut out = Vec::new();
        for &theta in &[0.25, 0.5, 0.75] {
            assert!(engine.interpolate(theta, &mut out));
            let t = sim.time - (1.0 - theta) * h;
            let want = (-t).exp();
            assert!(
                (out[0] - want).abs() < 1e-6,
                "θ={theta}: interp {} vs exact {} (t={t}, h={h})",
                out[0], want
            );
        }
    }

    #[test]
    fn vector_state_oscillator() {
        // 2D harmonic oscillator x'' = -x via RKDP54; mid-step interpolant must
        // track (cos t, -sin t) to interpolant accuracy.
        let rhs = |_t: f64, x: &[f64]| vec![x[1], -x[0]];
        let mut s = rkdp54_factory(1e6, 1e6)(&[1.0, 0.0]);
        let h = 0.2;
        let mut t = 0.0;
        let mut out = Vec::new();
        for _ in 0..10 {
            drive_step(&mut s, &rhs, t, h);
            for &theta in &[0.3, 0.7] {
                assert!(s.interpolate(theta, &mut out));
                let tt = t + theta * h;
                assert!(
                    (out[0] - tt.cos()).abs() < 1e-5 && (out[1] + tt.sin()).abs() < 1e-5,
                    "t={tt}: interp ({}, {}) vs exact ({}, {})",
                    out[0], out[1], tt.cos(), -tt.sin()
                );
            }
            t += h;
        }
    }
}
