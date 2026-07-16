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

use super::{fmt_lit, Numeric, R};

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

/// Everything the integrator emitter needs beyond the tableau itself.
pub(crate) struct SolverCtx<'a> {
    pub name: &'a str,
    pub n_state: usize,
    pub real: &'static str,
    pub numeric: Numeric,
    pub has_events: bool,
    pub has_sig: bool,
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
    debug_assert!(t.is_explicit(), "codegen solver: only explicit tableaus are emitted");
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
        Ok(emit_adaptive(t, cx))
    } else {
        Ok(emit_fixed(t, cx))
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

/// The stage loop body (shared by fixed and adaptive). Assumes locals `x0`, `k`,
/// `t0` and the `fs_c`/`fs_a` arrays are already declared; leaves the new state in
/// `m->x` and `m->time` at `t0 + dt`. `s` is the stage count, inlined as a literal.
fn stage_loop(t: &Tableau, cx: &SolverCtx) -> String {
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

    // Controller constants (mirror crate::constants + Solver defaults).
    let atol = cx.lit(constants::SOL_TOLERANCE_LTE_ABS);
    let rtol = cx.lit(constants::SOL_TOLERANCE_LTE_REL);
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
