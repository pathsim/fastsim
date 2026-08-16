//! Tableau-driven integrator emission for the struct ("rtModel") API.
//!
//! The C integrator is generated from a Butcher [`Tableau`] (the same data the
//! runtime solvers in [`crate::solvers`] consume), so codegen and the runtime
//! share one solver registry instead of a hand-written stage kernel per method.
//! One generic explicit-RK driver covers every explicit tableau:
//!
//! - **Fixed-step** (`tr` empty): the static `fs_stages_step` kernel runs the
//!   stages and advances `dt`; `<name>_run` is the same fixed-`dt` loop as
//!   before, and the public `<name>_step` wraps one kernel step with event
//!   handling and output refresh (the RTOS/ISR entry point).
//! - **Adaptive** (`tr` non-empty): the static `fs_trial_step` kernel runs the
//!   stages and returns the
//!   embedded WRMS error norm; `<name>_run` is an accept/reject loop with the same
//!   I-controller as [`crate::solvers::solver::Solver::step_factor`]
//!   (`factor = beta / err^(1/p)`, clamped to `[scale_min, scale_max]`), carrying
//!   the step `fs_h` across calls in the model struct.
//!
//! The emitted stage loop mirrors `make_explicit_rk_step` + `take_step`: at stage
//! `s`, `model_deriv` is evaluated at `t0 + c[s]·dt` over the current state, then
//! the state is set to `x0 + dt·Σ_j a[s][j]·k[j]` (the last row is the output `b`).
//! Tolerances and controller constants are inlined from [`crate::constants`].

use crate::constants;
use crate::solvers::tableaus::{Tableau, TableauKind};

use super::{fmt_lit, Numeric, Tolerances, R};

/// Forward Euler as a one-stage explicit Butcher tableau, so it flows through the
/// same generic emitter as every other explicit method. Not in the runtime
/// registry (`tableaus::ALL`) — the runtime drives EUF through a dedicated
/// `euf_factory` — but numerically identical: `x_{n+1} = x_n + dt·f(x_n, t_n)`.
pub(crate) const EUF_TABLEAU: Tableau = Tableau {
    name: "EUF",
    kind: TableauKind::ExplicitRK,
    n: 1,
    m: 0,
    s: 1,
    eval_stages: &[0.0],
    bt: &[&[1.0]],
    tr: &[],
    a_final: &[],
    di: &[],
};

/// Backward Euler as a one-stage DIRK tableau, so it flows through the same
/// generic implicit emitter as DIRK/ESDIRK. Not in the runtime registry
/// (`tableaus::ALL`) — the runtime drives EUB through a dedicated `eub_factory`
/// — but numerically identical: `x_{n+1} = x_n + dt·f(x_{n+1}, t_{n+1})`.
pub(crate) const EUB_TABLEAU: Tableau = Tableau {
    name: "EUB",
    kind: TableauKind::DIRK,
    n: 1,
    m: 0,
    s: 1,
    eval_stages: &[1.0],
    bt: &[&[1.0]],
    tr: &[],
    a_final: &[],
    di: &[],
};

/// Everything the integrator emitter needs beyond the tableau itself.
pub(crate) struct SolverCtx<'a> {
    pub name: &'a str,
    pub n_state: usize,
    pub real: &'static str,
    pub numeric: Numeric,
    pub has_events: bool,
    pub has_sig: bool,
    /// LTE tolerances inlined into the adaptive step controller. Unused by the
    /// fixed-step emitter.
    pub tolerances: Tolerances,
}

impl SolverCtx<'_> {
    fn lit(&self, x: f64) -> String {
        fmt_lit(x, self.numeric)
    }
    /// `0.5 * dt`, numeric-aware: a Q shift under fixed point (a plain
    /// `half * dt` int multiply would double-apply the 2^frac scale).
    fn half_dt(&self) -> String {
        match self.numeric.frac() {
            Some(_) => "(dt >> 1)".to_string(),
            None => format!("{} * dt", self.lit(0.5)),
        }
    }
    /// A `<math.h>` call with the numeric-type suffix (`pow`/`powf`, `fabs`/`fabsf`).
    fn mfn(&self, f: &str, args: &str) -> String {
        format!("{f}{}({args})", self.numeric.suffix())
    }
}

/// `true` if this tableau is emitted with the adaptive accept/reject loop.
pub(crate) fn is_adaptive(t: &Tableau) -> bool {
    t.is_adaptive()
}

/// One extra `model_t` field (`fs_h`: the carried adaptive step size) for adaptive
/// solvers, so chunked `model_run` calls keep their step history. Empty otherwise.
pub(crate) fn struct_fields(t: &Tableau, real: &str) -> Vec<String> {
    if is_adaptive(t) {
        vec![format!("    {real} fs_h;       /* carried adaptive step size (0 = use dt) */")]
    } else {
        Vec::new()
    }
}

/// Initializer line for the adaptive step field, injected into `model_init`.
pub(crate) fn init_body(t: &Tableau) -> String {
    if is_adaptive(t) {
        "    m->fs_h = 0;\n".to_string()
    } else {
        String::new()
    }
}

/// Emit the continuous-model integrator (stage kernel + `<name>_step` + `<name>_run`)
/// for `tableau`. The caller injects the result where the old hand-written
/// `solver_impl_struct` body went (Compact: into `model.c`; Library: into `solver.c`).
pub(crate) fn emit(t: &Tableau, cx: &SolverCtx) -> R<String> {
    if t.is_implicit() && cx.numeric.frac().is_some() {
        // The stage Newton needs division, `fabs` and a pivoted LU — none of
        // which have an integer lowering.
        return Err(super::CodegenError::Unsupported(format!(
            "implicit tableau '{}' under fixed point (the stage solve needs              division and fabs); use an explicit fixed-step solver",
            t.name
        )));
    }
    // The stage kernel differs (Newton per stage vs a plain evaluation); the
    // helpers it calls are file-static and emitted once, ahead of it.
    let helpers = if t.is_implicit() { implicit_helpers(cx) } else { String::new() };
    if is_adaptive(t) {
        if cx.numeric.frac().is_some() {
            // The embedded-error controller needs pow/fabs on the error norm —
            // there is no integer lowering. Fixed point is fixed-step.
            return Err(super::CodegenError::Unsupported(format!(
                "adaptive tableau '{}' under fixed point (the step controller \
                 needs pow); use a fixed-step solver (rk4, euler, ssprk22/33/34)",
                t.name
            )));
        }
        Ok(helpers + &emit_adaptive(t, cx))
    } else {
        Ok(helpers + &emit_fixed(t, cx))
    }
}

/// The shared Butcher-tableau `static const` arrays: stage times `fs_c[S]` and the
/// (zero-padded, lower-triangular) coefficient matrix `fs_a[S][S]`.
fn tableau_arrays(t: &Tableau, cx: &SolverCtx) -> String {
    let s = t.s;
    let real = cx.real;
    let c = t.eval_stages.iter().map(|v| cx.lit(*v)).collect::<Vec<_>>().join(", ");
    let mut rows = Vec::with_capacity(s);
    for row in t.bt.iter() {
        let mut vals: Vec<String> = row.iter().map(|v| cx.lit(*v)).collect();
        while vals.len() < s {
            vals.push(cx.lit(0.0));
        }
        rows.push(format!("        {{ {} }}", vals.join(", ")));
    }
    format!(
        "    static const {real} fs_c[{s}] = {{ {c} }};\n    \
         static const {real} fs_a[{s}][{s}] = {{\n{}\n    }};\n",
        rows.join(",\n"),
    )
}

/// File-static helpers the implicit stage kernel needs: a dense Jacobian and a
/// dense LU solve. Emitted once, ahead of the stage kernel.
///
/// The Jacobian is a forward difference of `<name>_deriv`. An analytic
/// `<name>_jvp` is emitted for most models and would give exact columns in one
/// call each, but it is not available for every op, whereas differencing the
/// derivative always is — and a Newton iteration converges on an approximate
/// Jacobian regardless (it only slows convergence, it does not move the answer,
/// which is pinned by the residual).
fn implicit_helpers(cx: &SolverCtx) -> String {
    let real = cx.real;
    let n = cx.n_state;
    let name = cx.name;
    let fabs_xj = cx.mfn("fabs", "xs");
    format!(
        "/* Dense d(dx/dt)/dx at the current state, row-major. Forward differences. */\n\
         static void fs_jacobian({name}_t * restrict m, {real} J[{n}][{n}]) {{\n\
         \x20   {real} f0[{n}], f1[{n}];\n\
         \x20   {name}_deriv(m, f0);\n\
         \x20   for (size_t j = 0; j < {n}; j++) {{\n\
         \x20       const {real} xs = m->x[j];\n\
         \x20       const {real} a = {fabs_xj};\n\
         \x20       const {real} h = {eps} * (a > {one} ? a : {one});\n\
         \x20       m->x[j] = xs + h;\n\
         \x20       {name}_deriv(m, f1);\n\
         \x20       m->x[j] = xs;\n\
         \x20       for (size_t i = 0; i < {n}; i++) J[i][j] = (f1[i] - f0[i]) / h;\n\
         \x20   }}\n\
         }}\n\n\
         /* Solve A z = b in place (LU, partial pivoting). Returns 0 on success,\n\
         \x20  1 if A is numerically singular — the caller then keeps its iterate. */\n\
         static int fs_lu_solve({real} A[{n}][{n}], {real} b[{n}]) {{\n\
         \x20   for (size_t c = 0; c < {n}; c++) {{\n\
         \x20       size_t piv = c;\n\
         \x20       {real} best = {fabs_ac};\n\
         \x20       for (size_t r = c + 1; r < {n}; r++) {{\n\
         \x20           const {real} v = {fabs_arc};\n\
         \x20           if (v > best) {{ best = v; piv = r; }}\n\
         \x20       }}\n\
         \x20       if (best <= {tiny}) return 1;\n\
         \x20       if (piv != c) {{\n\
         \x20           for (size_t j = 0; j < {n}; j++) {{ {real} t = A[c][j]; A[c][j] = A[piv][j]; A[piv][j] = t; }}\n\
         \x20           {real} t = b[c]; b[c] = b[piv]; b[piv] = t;\n\
         \x20       }}\n\
         \x20       for (size_t r = c + 1; r < {n}; r++) {{\n\
         \x20           const {real} f = A[r][c] / A[c][c];\n\
         \x20           if (f == {zero}) continue;\n\
         \x20           for (size_t j = c; j < {n}; j++) A[r][j] -= f * A[c][j];\n\
         \x20           b[r] -= f * b[c];\n\
         \x20       }}\n\
         \x20   }}\n\
         \x20   for (size_t ri = {n}; ri-- > 0; ) {{\n\
         \x20       {real} acc = b[ri];\n\
         \x20       for (size_t j = ri + 1; j < {n}; j++) acc -= A[ri][j] * b[j];\n\
         \x20       b[ri] = acc / A[ri][ri];\n\
         \x20   }}\n\
         \x20   return 0;\n\
         }}\n\n",
        eps = cx.lit(1e-7),
        one = cx.lit(1.0),
        zero = cx.lit(0.0),
        tiny = cx.lit(constants::TOLERANCE),
        fabs_ac = cx.mfn("fabs", "A[c][c]"),
        fabs_arc = cx.mfn("fabs", "A[r][c]"),
    )
}

/// The implicit stage loop: same contract as [`stage_loop`] (fills `k[s][n]`,
/// leaves the new state in `m->x` and `m->time` at `t0 + dt`), so the fixed and
/// adaptive emitters drive it unchanged.
///
/// Each stage solves its own slope. For stage `i` with diagonal coefficient
/// `a_ii`, the unknown `k_i` satisfies
///
/// ```text
/// k_i = f(x0 + dt·Σ_{j<i} a_ij k_j + dt·a_ii·k_i,  t0 + c_i·dt)
/// ```
///
/// which Newton solves as `R(k_i) = k_i − f(z) = 0` with `R' = I − dt·a_ii·J(z)`.
/// A stage with `a_ii == 0` (an ESDIRK's explicit first stage) is a plain
/// evaluation, no solve.
fn stage_loop_implicit(t: &Tableau, cx: &SolverCtx) -> String {
    let real = cx.real;
    let n = cx.n_state;
    let s = t.s;
    let name = cx.name;
    // Converge the stage well inside the local error the step controller will
    // accept, so the Newton residual never becomes the limiting error term.
    let newton_tol = (cx.tolerances.abs * 1e-3).max(1e-14);
    format!(
        "    {real} base[{n}], resid[{n}], zvec[{n}], fz[{n}];\n\
         \x20   {real} jac[{n}][{n}], amat[{n}][{n}];\n\
         \x20   for (size_t fs_s = 0; fs_s < {s}u; fs_s++) {{\n\
         \x20       const {real} aii = fs_a[fs_s][fs_s];\n\
         \x20       m->time = t0 + fs_c[fs_s] * dt;\n\
         \x20       for (size_t i = 0; i < {n}; i++) {{\n\
         \x20           {real} acc = {zero};\n\
         \x20           for (size_t j = 0; j < fs_s; j++) acc += fs_a[fs_s][j] * k[j][i];\n\
         \x20           base[i] = x0[i] + dt * acc;\n\
         \x20       }}\n\
         \x20       for (size_t i = 0; i < {n}; i++) m->x[i] = base[i];\n\
         \x20       {name}_deriv(m, k[fs_s]);\n\
         \x20       if (aii != {zero}) {{\n\
         \x20           for (size_t it = 0; it < {maxit}u; it++) {{\n\
         \x20               for (size_t i = 0; i < {n}; i++) zvec[i] = base[i] + dt * aii * k[fs_s][i];\n\
         \x20               for (size_t i = 0; i < {n}; i++) m->x[i] = zvec[i];\n\
         \x20               {name}_deriv(m, fz);\n\
         \x20               {real} worst = {zero};\n\
         \x20               for (size_t i = 0; i < {n}; i++) {{\n\
         \x20                   resid[i] = k[fs_s][i] - fz[i];\n\
         \x20                   const {real} av = {fabs_r};\n\
         \x20                   if (av > worst) worst = av;\n\
         \x20               }}\n\
         \x20               if (worst <= {ntol}) break;\n\
         \x20               fs_jacobian(m, jac);\n\
         \x20               for (size_t i = 0; i < {n}; i++)\n\
         \x20                   for (size_t j = 0; j < {n}; j++)\n\
         \x20                       amat[i][j] = (i == j ? {one} : {zero}) - dt * aii * jac[i][j];\n\
         \x20               for (size_t i = 0; i < {n}; i++) resid[i] = -resid[i];\n\
         \x20               if (fs_lu_solve(amat, resid) != 0) break;\n\
         \x20               for (size_t i = 0; i < {n}; i++) k[fs_s][i] += resid[i];\n\
         \x20           }}\n\
         \x20       }}\n\
         \x20       for (size_t i = 0; i < {n}; i++) {{\n\
         \x20           {real} acc = {zero};\n\
         \x20           for (size_t j = 0; j <= fs_s; j++) acc += fs_a[fs_s][j] * k[j][i];\n\
         \x20           m->x[i] = x0[i] + dt * acc;\n\
         \x20       }}\n\
         \x20   }}\n\
         \x20   m->time = t0 + dt;\n",
        zero = cx.lit(0.0),
        one = cx.lit(1.0),
        maxit = 25,
        ntol = cx.lit(newton_tol),
        fabs_r = cx.mfn("fabs", "resid[i]"),
    )
}

/// The stage loop body (shared by fixed and adaptive). Assumes locals `x0`, `k`,
/// `t0` and the `fs_c`/`fs_a` arrays are already declared; leaves the new state in
/// `m->x` and `m->time` at `t0 + dt`. `s` is the stage count, inlined as a literal.
fn stage_loop(t: &Tableau, cx: &SolverCtx) -> String {
    if t.is_implicit() {
        return stage_loop_implicit(t, cx);
    }
    let real = cx.real;
    let n = cx.n_state;
    let s = t.s;
    let name = cx.name;
    if let Some(frac) = cx.numeric.frac() {
        // Q kernel: the stage accumulator stays in int64 (headroom for the
        // weighted slope sum), every product carries one >> frac rescale, and
        // stores truncate back to int32 (the documented wrap).
        return format!(
            "    for (size_t fs_s = 0; fs_s < {s}u; fs_s++) {{\n\
             \x20       m->time = (int32_t)((int64_t)t0 + (((int64_t)fs_c[fs_s] * (int64_t)dt) >> {frac}));\n\
             \x20       {name}_deriv(m, k[fs_s]);\n\
             \x20       for (size_t i = 0; i < {n}; i++) {{\n\
             \x20           int64_t acc = 0;\n\
             \x20           for (size_t j = 0; j <= fs_s; j++) acc += ((int64_t)fs_a[fs_s][j] * (int64_t)k[j][i]) >> {frac};\n\
             \x20           m->x[i] = (int32_t)((int64_t)x0[i] + (((int64_t)dt * acc) >> {frac}));\n\
             \x20       }}\n\
             \x20   }}\n\
             \x20   m->time = (int32_t)((int64_t)t0 + (int64_t)dt);\n",
        );
    }
    format!(
        "    for (size_t fs_s = 0; fs_s < {s}u; fs_s++) {{\n\
         \x20       m->time = t0 + fs_c[fs_s] * dt;\n\
         \x20       {name}_deriv(m, k[fs_s]);\n\
         \x20       for (size_t i = 0; i < {n}; i++) {{\n\
         \x20           {real} acc = {zero};\n\
         \x20           for (size_t j = 0; j <= fs_s; j++) acc += fs_a[fs_s][j] * k[j][i];\n\
         \x20           m->x[i] = x0[i] + dt * acc;\n\
         \x20       }}\n\
         \x20   }}\n\
         \x20   m->time = t0 + dt;\n",
        zero = cx.lit(0.0),
    )
}

/// The public single-step entry point `<name>_step`: initial-event guard, one
/// integrator step (`inner` — the file-static stage kernel), events at the new
/// time, output refresh. Shared by the fixed and adaptive emitters; the
/// adaptive kernel's embedded error estimate is deliberately ignored here
/// (fixed-rate stepping has no step-size control — that lives in `run`).
fn emit_public_step(cx: &SolverCtx, inner: &str) -> String {
    let real = cx.real;
    let name = cx.name;
    let mut out = String::new();
    out.push_str(&format!(
        "/* Advance by exactly ONE step of `dt`: fire events due now (first call\n\
         \x20  only — later calls fire them at the new time), take one RK step,\n\
         \x20  handle events, refresh outputs. Fixed work per call (bounded stage\n\
         \x20  count, no loops over time, no allocation): suitable for a periodic\n\
         \x20  real-time task or timer ISR at rate 1/dt. N calls compose exactly\n\
         \x20  to run(t0 + N*dt, dt). */\n\
         void {name}_step({name}_t * restrict m, {real} dt) {{\n"
    ));
    if cx.has_events {
        out.push_str(&format!(
            "    if (!m->fs_started) {{ {name}_handle_events(m, dt); m->fs_started = 1; }}\n"
        ));
    }
    out.push_str(&format!("    {inner};\n"));
    if cx.has_events {
        out.push_str(&format!("    {name}_handle_events(m, dt);\n"));
    }
    if cx.has_sig {
        out.push_str(&format!("    {name}_outputs(m);\n"));
    }
    out.push_str("}\n\n");
    out
}

/// Fixed-step explicit integrator: the file-static stage kernel
/// `fs_stages_step`, the public single-step `<name>_step`, and the fixed-`dt`
/// `<name>_run` loop.
fn emit_fixed(t: &Tableau, cx: &SolverCtx) -> String {
    let real = cx.real;
    let n = cx.n_state;
    let s = t.s;
    let name = cx.name;
    let half_dt = cx.half_dt();
    let mut out = String::new();

    // File-static so it never collides with the public `<name>_step` (the
    // default model name IS "model").
    out.push_str(&format!(
        "static void fs_stages_step({name}_t * restrict m, {real} dt) {{\n\
         \x20   const {real} t0 = m->time;\n\
         \x20   {real} x0[{n}], k[{s}][{n}];\n\
         \x20   for (size_t i = 0; i < {n}; i++) x0[i] = m->x[i];\n",
    ));
    out.push_str(&tableau_arrays(t, cx));
    out.push_str(&stage_loop(t, cx));
    out.push_str("}\n\n");

    out.push_str(&emit_public_step(cx, "fs_stages_step(m, dt)"));

    out.push_str(&format!(
        "/* Integrate the model from its current time to `t_end` in fixed `dt` steps. */\n\
         void {name}_run({name}_t * restrict m, {real} t_end, {real} dt) {{\n"
    ));
    if cx.has_events {
        out.push_str(&format!("    {name}_handle_events(m, dt);\n"));
        out.push_str("    m->fs_started = 1;\n");
    }
    out.push_str(&format!("    while (m->time < t_end - {half_dt}) {{\n"));
    out.push_str("        fs_stages_step(m, dt);\n");
    if cx.has_events {
        out.push_str(&format!("        {name}_handle_events(m, dt);\n"));
    }
    out.push_str("    }\n");
    if cx.has_sig {
        out.push_str(&format!("    {name}_outputs(m);\n"));
    }
    out.push_str("}\n");
    out
}

/// Adaptive explicit integrator: `fs_trial_step` (returns the embedded WRMS error
/// norm), the public single-step `<name>_step`, and an accept/reject `<name>_run`
/// loop with the I-controller. Mirrors
/// `Solver::error_controller` / `Solver::step_factor` for the default
/// (`use_pi_controller == false`) RK path.
fn emit_adaptive(t: &Tableau, cx: &SolverCtx) -> String {
    let real = cx.real;
    let n = cx.n_state;
    let s = t.s;
    let name = cx.name;

    // LTE tolerances come from the caller (Simulation.to_c inherits the
    // simulation's), so the emitted controller and the reference run accept the
    // same steps. The remaining controller constants mirror the Solver defaults.
    let atol = cx.lit(cx.tolerances.abs);
    let rtol = cx.lit(cx.tolerances.rel);
    let beta = cx.lit(constants::SOL_BETA);
    let smin = cx.lit(constants::SOL_SCALE_MIN);
    let smax = cx.lit(constants::SOL_SCALE_MAX);
    let floor = cx.lit(constants::TOLERANCE);
    let hmin = cx.lit(constants::SIM_TIMESTEP_MIN);
    let zero = cx.lit(0.0);
    let one = cx.lit(1.0);
    // order p = min(m, n) + 1, exactly as Solver::error_controller.
    let inv_p = cx.lit(1.0 / (t.m.min(t.n) + 1) as f64);
    let fabs_e = cx.mfn("fabs", "dt * e");
    let fabs_x = cx.mfn("fabs", "m->x[i]");
    let pow_call = cx.mfn("pow", &format!("err, {inv_p}"));

    let mut out = String::new();

    // fs_trial_step: stages + embedded error norm (file-static; the public
    // `<name>_step` and the accept/reject `run` loop both drive it).
    out.push_str(&format!(
        "/* One trial step over `dt`; leaves the new state in `m` and returns the\n\
         \x20  embedded WRMS error norm (floored at the reference tolerance). */\n\
         static {real} fs_trial_step({name}_t * restrict m, {real} dt) {{\n\
         \x20   const {real} t0 = m->time;\n\
         \x20   {real} x0[{n}], k[{s}][{n}];\n\
         \x20   for (size_t i = 0; i < {n}; i++) x0[i] = m->x[i];\n",
    ));
    out.push_str(&tableau_arrays(t, cx));
    let tr = t.tr.iter().map(|v| cx.lit(*v)).collect::<Vec<_>>().join(", ");
    out.push_str(&format!("    static const {real} fs_tr[{s}] = {{ {tr} }};\n"));
    out.push_str(&stage_loop(t, cx));
    out.push_str(&format!(
        "    {real} err = {floor};\n\
         \x20   for (size_t i = 0; i < {n}; i++) {{\n\
         \x20       {real} e = {zero};\n\
         \x20       for (size_t fs_s = 0; fs_s < {s}u; fs_s++) e += fs_tr[fs_s] * k[fs_s][i];\n\
         \x20       {real} scale = {atol} + {rtol} * {fabs_x};\n\
         \x20       {real} se = {fabs_e} / scale;\n\
         \x20       if (se > err) err = se;\n\
         \x20   }}\n\
         \x20   return err;\n\
         }}\n\n",
    ));

    // Public single-step: one trial step of exactly `dt`, error estimate
    // discarded (no step-size control at a fixed call rate).
    out.push_str(&emit_public_step(cx, "(void)fs_trial_step(m, dt)"));

    // <name>_run: accept/reject loop with the I-controller.
    out.push_str(&format!(
        "/* Integrate adaptively from the current time to `t_end`; `dt` seeds the\n\
         \x20  first step. The accepted step size is carried in `m->fs_h`. */\n\
         void {name}_run({name}_t * restrict m, {real} t_end, {real} dt) {{\n"
    ));
    if cx.has_events {
        out.push_str(&format!("    {name}_handle_events(m, dt);\n"));
        out.push_str("    m->fs_started = 1;\n");
    }
    out.push_str(&format!(
        "    {real} h = (m->fs_h > {zero}) ? m->fs_h : dt;\n\
         \x20   while (t_end - m->time > {floor} * (t_end > {zero} ? t_end : -t_end) + {hmin}) {{\n\
         \x20       int clamped = 0;\n\
         \x20       {real} hh = h;\n\
         \x20       if (m->time + hh >= t_end) {{ hh = t_end - m->time; clamped = 1; }}\n\
         \x20       {real} x0[{n}];\n\
         \x20       const {real} t0 = m->time;\n\
         \x20       for (size_t i = 0; i < {n}; i++) x0[i] = m->x[i];\n\
         \x20       {real} err = fs_trial_step(m, hh);\n\
         \x20       {real} fac = {beta} / {pow_call};\n\
         \x20       if (fac < {smin}) fac = {smin};\n\
         \x20       if (fac > {smax}) fac = {smax};\n\
         \x20       if (err <= {one} || hh <= {hmin}) {{\n",
    ));
    if cx.has_events {
        out.push_str(&format!("            {name}_handle_events(m, hh);\n"));
    }
    out.push_str(&format!(
        "            if (!clamped) h = hh * fac;\n\
         \x20       }} else {{\n\
         \x20           for (size_t i = 0; i < {n}; i++) m->x[i] = x0[i];\n\
         \x20           m->time = t0;\n\
         \x20           h = hh * fac;\n\
         \x20       }}\n\
         \x20   }}\n\
         \x20   m->fs_h = h;\n",
    ));
    if cx.has_sig {
        out.push_str(&format!("    {name}_outputs(m);\n"));
    }
    out.push_str("}\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solvers::tableaus;

    fn ctx(tolerances: Tolerances) -> SolverCtx<'static> {
        SolverCtx {
            name: "m",
            n_state: 1,
            real: "double",
            numeric: Numeric::Double,
            has_events: false,
            has_sig: false,
            tolerances,
        }
    }

    fn tableau(name: &str) -> &'static Tableau {
        tableaus::ALL.iter().find(|t| t.name == name).expect("tableau in the registry")
    }

    /// The caller's tolerances reach the emitted error scale. Regression guard:
    /// these used to be the crate constants, so a model with custom LTE
    /// tolerances generated C that stepped differently than its own reference.
    #[test]
    fn adaptive_controller_uses_the_supplied_tolerances() {
        let c = emit(tableau("RKBS32"), &ctx(Tolerances { abs: 1e-9, rel: 1.5e-7 })).unwrap();
        assert!(c.contains("1e-9"), "atol missing from the emitted scale:\n{c}");
        assert!(c.contains("1.5e-7"), "rtol missing from the emitted scale:\n{c}");
    }

    /// Fixed-step tableaus have no error control, so the tolerances must not
    /// leak into their output at all.
    #[test]
    fn fixed_step_ignores_tolerances() {
        let c = emit(tableau("RK4"), &ctx(Tolerances { abs: 1e-9, rel: 1.5e-7 })).unwrap();
        assert!(!c.contains("1e-9"));
        assert!(!c.contains("1.5e-7"));
    }
}
