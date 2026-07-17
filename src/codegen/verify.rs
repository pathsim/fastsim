//! Software-in-the-loop (SIL) verification of the generated C.
//!
//! [`verify_c`] closes the loop the codegen test suite exercises internally,
//! as a product API: it compiles the emitted C with a local C compiler, drives
//! the compiled binary and the reference engine ([`CompiledSimulation`], the
//! flat-tape twin of the same IR) over the SAME fixed-step trajectory, and
//! compares the state trajectories sample by sample. The result is a
//! [`VerifyCReport`] with the worst scaled error — evidence that the C shipped
//! to a target computes what the simulation computed, on this machine, today.
//!
//! Scope (v1): fixed-step explicit solvers, models with continuous state (the
//! `Simulation.compile()` subset). Adaptive tableaus adapt their step sequence
//! independently on each side, so their trajectories are not comparable
//! sample-by-sample; they are rejected up front rather than compared loosely.
//!
//! The C side steps through `<name>_run(m, m->time + dt, dt)` (exactly one
//! step per call — the generated fixed-step loop runs while
//! `time < t_end - dt/2`), the reference through
//! [`CompiledSimulation::run`]`(dt, ..)` (exactly one step per call — the run
//! loop takes the first step that reaches `t_end`). Both sides accumulate
//! `time += dt` identically, so the sample times line up bit-for-bit and the
//! comparison never mixes trajectories at different times.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{file_base, struct_layout, CodegenError, CodegenOptions, VarKind};
use crate::compile::CompiledSimulation;
use crate::ir::schema::Module;

/// Knobs for [`verify_c`]. All fields have sensible defaults.
#[derive(Debug, Clone)]
pub struct VerifyCOptions {
    /// Simulated time span; the step count is `round(duration / dt)` (min 1).
    pub duration: f64,
    /// Fixed step size for both sides.
    pub dt: f64,
    /// Absolute weight of the per-sample scaled error `|c - ref| / (atol + rtol·|ref|)`.
    pub atol: f64,
    /// Relative weight (see `atol`). The check passes when the worst scaled
    /// error is ≤ 1. The C expressions and the tape evaluate the same ops in
    /// different association orders, so ULP-level drift accumulates through
    /// the integration — the defaults leave room for that, not for bugs.
    pub rtol: f64,
    /// Keep the build directory (sources + binary) instead of deleting it.
    pub keep_build: bool,
}

impl Default for VerifyCOptions {
    fn default() -> Self {
        Self { duration: 1.0, dt: 1e-3, atol: 1e-9, rtol: 1e-6, keep_build: false }
    }
}

/// Outcome of a [`verify_c`] run. `passed` is `max_scaled_error <= 1.0`.
#[derive(Debug, Clone)]
pub struct VerifyCReport {
    pub passed: bool,
    /// The C compiler that built the harness (resolved by [`find_compiler`]).
    pub compiler: String,
    /// Steps taken (samples compared: `n_steps + 1`, including t = 0).
    pub n_steps: usize,
    pub n_states: usize,
    /// Worst `|c - ref| / (atol + rtol·|ref|)` over all states and samples.
    pub max_scaled_error: f64,
    /// State label (`<block>` / `<block>_x<k>`) of the worst deviation.
    pub worst_state: Option<String>,
    /// Sample time of the worst deviation.
    pub worst_time: f64,
    /// Generated file names that were compiled.
    pub files: Vec<String>,
    /// Build directory, when kept (`keep_build`) — `None` after cleanup.
    pub build_dir: Option<PathBuf>,
}

static UNIQ: AtomicU64 = AtomicU64::new(0);

fn unique_dir() -> PathBuf {
    let n = UNIQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("fastsim_verify_{}_{n}", std::process::id()))
}

fn verr(msg: impl Into<String>) -> CodegenError {
    CodegenError::Verify(msg.into())
}

/// Build a `Command` for a compiler spec that may carry leading arguments
/// (`"zig cc"`, `"python -m ziglang cc"`) — whitespace-split, first token is
/// the program. Lets `$FASTSIM_CC` point at multi-word toolchain launchers.
/// Public so external harnesses (the test suite) invoke the compiler the same
/// way [`find_compiler`] probed it.
pub fn cc_command(cc: &str) -> Command {
    let mut parts = cc.split_whitespace();
    let mut cmd = Command::new(parts.next().unwrap_or(cc));
    cmd.args(parts);
    cmd
}

/// [`cc_command`] with the working directory set — and, for zig-based specs,
/// zig's caches isolated INTO that directory. Concurrent `zig cc` processes
/// contend on one global cache lock and can deadlock each other (observed:
/// four parallel test-thread compiles hung for hours); a per-build cache
/// removes the shared lock entirely, at the price of a cold cache per build.
/// Every compiler spawn against a build directory should go through this.
pub fn cc_command_in(cc: &str, dir: &std::path::Path) -> Command {
    let mut cmd = cc_command(cc);
    cmd.current_dir(dir);
    if cc.contains("zig") {
        cmd.env("ZIG_LOCAL_CACHE_DIR", dir.join(".zig-local-cache"));
        cmd.env("ZIG_GLOBAL_CACHE_DIR", dir.join(".zig-global-cache"));
    }
    cmd
}

/// Locate a working C99 compiler: `$FASTSIM_CC`, `$CC`, then `cc`/`clang`/`gcc`
/// on PATH. A candidate must compile AND run a floating-point + libm probe —
/// `--version` succeeding is not enough (broken cross/MSYS toolchains, e.g.
/// Anaconda's bundled MinGW gcc 5.3 ICEs on any double op). A spec may carry
/// arguments (`FASTSIM_CC="zig cc"`). Returns `None` when nothing usable is
/// found.
pub fn find_compiler() -> Option<String> {
    let mut candidates = Vec::new();
    for var in ["FASTSIM_CC", "CC"] {
        if let Ok(cc) = std::env::var(var) {
            if !cc.is_empty() {
                candidates.push(cc);
            }
        }
    }
    candidates.extend(["cc", "clang", "gcc", "zig cc"].iter().map(|s| s.to_string()));
    candidates.into_iter().find(|cc| compiler_is_sane(cc))
}

/// Probe with a representative double + libm program so a compiler broken for
/// floating point is skipped, not mistaken for a codegen bug later.
fn compiler_is_sane(cc: &str) -> bool {
    let probe = "#include <stdio.h>\n#include <math.h>\n\
                 int main(void){double a[1]; a[0]=2.0; printf(\"%.17g\", sqrt(a[0])); return 0;}\n";
    let dir = unique_dir();
    if std::fs::create_dir_all(&dir).is_err() {
        return false;
    }
    let ok = (|| {
        std::fs::write(dir.join("probe.c"), probe).ok()?;
        let out = cc_command_in(cc, &dir)
            .args(["probe.c", "-o", "probe.exe", "-lm"])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let run = Command::new(dir.join("probe.exe")).output().ok()?;
        let text = String::from_utf8_lossy(&run.stdout);
        let v: f64 = text.trim().parse().ok()?;
        ((v - 2f64.sqrt()).abs() < 1e-9).then_some(())
    })()
    .is_some();
    let _ = std::fs::remove_dir_all(&dir);
    ok
}

/// Compile the generated C for `module`, run it and the reference engine over
/// the same fixed-step trajectory, and compare the state trajectories. See the
/// module docs for scope and mechanics. `reference` is reset to `t = 0` and
/// integrated; its solver/`dt` are configured from `cg`/`opts` internally.
pub fn verify_c(
    module: &Module,
    reference: &mut CompiledSimulation,
    cg: &CodegenOptions,
    opts: &VerifyCOptions,
    log: &crate::utils::logger::Logger,
) -> Result<VerifyCReport, CodegenError> {
    let tableau = cg.solver.tableau();
    if cg.numeric.frac().is_some() {
        return Err(verr(
            "verification compares against the f64 reference engine; a fixed-point              trajectory deviates by quantization design. Verify the double build              (same model, numeric=\"double\") and rely on the Q kernel's dedicated              fixed-point tests.",
        ));
    }
    if tableau.is_adaptive() {
        return Err(verr(format!(
            "solver '{}' is adaptive; verification compares fixed-step trajectories \
             sample-by-sample (adaptive step sequences diverge between backends). \
             Use a fixed-step solver (rk4, euler, ssprk22/33/34).",
            tableau.name
        )));
    }
    let positive = |v: f64| v.is_finite() && v > 0.0;
    if !positive(opts.dt) || !positive(opts.duration) {
        return Err(verr("duration and dt must be positive and finite"));
    }

    let layout = struct_layout(module, cg)?;
    if layout.n_state == 0 {
        return Err(verr("model has no continuous state; nothing to verify"));
    }
    if layout.n_input > 0 {
        return Err(verr(
            "model has external inputs (open system); verification drives closed models only",
        ));
    }
    if reference.n_state != layout.n_state {
        return Err(verr(format!(
            "reference/codegen state-count mismatch: tape has {}, generated C has {} \
             (the reference must be compiled from the same model)",
            reference.n_state, layout.n_state
        )));
    }
    let state_names: Vec<String> = layout
        .vars
        .iter()
        .filter(|v| matches!(v.kind, VarKind::State))
        .map(|v| v.name.clone())
        .collect();

    let n_steps = (opts.duration / opts.dt).round().max(1.0) as usize;
    let n = layout.n_state;
    log.info(&format!(
        "VERIFY_C (model: {}, solver: {}, duration: {}, dt: {}, steps: {}, states: {})",
        module.name, tableau.name, opts.duration, opts.dt, n_steps, n
    ));

    let files = crate::codegen::generate_logged(module, cg, log)?;

    let compiler = find_compiler().ok_or_else(|| {
        verr(
            "no working C compiler found (tried $FASTSIM_CC, $CC, cc, clang, gcc). \
             Point FASTSIM_CC at a C99 compiler with libm.",
        )
    })?;

    // -- write sources + harness -------------------------------------------------------
    let base = file_base(&module.name);
    let entry_header = files
        .iter()
        .map(|f| f.name.as_str())
        .find(|nm| nm.ends_with("_solver.h"))
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{base}.h"));
    // Symbol prefix: identical to the file base except for a degenerate empty
    // model name, which `file_base` maps to "model" while the symbols keep the
    // sanitized original. Require a proper name instead of guessing.
    let sym = base.clone();
    if module.name.is_empty() {
        return Err(verr("model needs a non-empty name for verification"));
    }

    let dir = unique_dir();
    std::fs::create_dir_all(&dir).map_err(|e| verr(format!("create build dir: {e}")))?;
    for f in &files {
        std::fs::write(dir.join(&f.name), &f.contents)
            .map_err(|e| verr(format!("write {}: {e}", f.name)))?;
    }
    let main_c = format!(
        "#include <stdio.h>\n\
         #include \"{entry_header}\"\n\
         int main(void) {{\n\
         \x20   static {sym}_t m;\n\
         \x20   {sym}_init(&m);\n\
         \x20   for (unsigned long k = 0; k <= {n_steps}ul; ++k) {{\n\
         \x20       if (k) {sym}_run(&m, m.time + {dt:.17e}, {dt:.17e});\n\
         \x20       printf(\"%.17g\", (double)m.time);\n\
         \x20       for (unsigned long i = 0; i < {n}ul; ++i) printf(\" %.17g\", (double)m.x[i]);\n\
         \x20       printf(\"\\n\");\n\
         \x20   }}\n\
         \x20   return 0;\n\
         }}\n",
        dt = opts.dt,
    );
    std::fs::write(dir.join("main.c"), main_c).map_err(|e| verr(format!("write main.c: {e}")))?;

    // -- compile + run ------------------------------------------------------------------
    let mut args: Vec<String> = files
        .iter()
        .filter(|f| f.name.ends_with(".c"))
        .map(|f| f.name.clone())
        .collect();
    args.push("main.c".into());
    // `-ffp-contract=off`: the generated C carries `#pragma STDC FP_CONTRACT OFF`,
    // but gcc ignores that pragma (and contracts to FMA under -O2 by default),
    // which can flip a discrete event's fire/no-fire decision at a step boundary
    // by one ULP. The explicit flag makes the SiL comparison compiler-independent.
    args.extend(
        ["-std=c99", "-O2", "-ffp-contract=off", "-o", "verify.exe", "-lm"]
            .iter()
            .map(|s| s.to_string()),
    );
    let t_compile = std::time::Instant::now();
    let out = cc_command_in(&compiler, &dir)
        .args(&args)
        .output()
        .map_err(|e| verr(format!("spawn {compiler}: {e}")))?;
    if !out.status.success() {
        return Err(verr(format!(
            "C compile failed (sources kept in {}):\n{}",
            dir.display(),
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    log.info(&format!(
        "VERIFY_C COMPILE (cc: {compiler}) in {:.1} ms",
        t_compile.elapsed().as_secs_f64() * 1000.0
    ));
    let t_run = std::time::Instant::now();
    let run = Command::new(dir.join("verify.exe"))
        .output()
        .map_err(|e| verr(format!("run harness (kept in {}): {e}", dir.display())))?;
    if !run.status.success() {
        return Err(verr(format!(
            "harness exited with {:?} (kept in {})",
            run.status.code(),
            dir.display()
        )));
    }
    let text = String::from_utf8_lossy(&run.stdout);
    let rows: Vec<Vec<f64>> = text
        .lines()
        .map(|l| l.split_whitespace().filter_map(|t| t.parse::<f64>().ok()).collect())
        .filter(|r: &Vec<f64>| !r.is_empty())
        .collect();
    if rows.len() != n_steps + 1 || rows.iter().any(|r| r.len() != n + 1) {
        return Err(verr(format!(
            "harness output malformed: expected {} rows of {} values, got {} rows \
             (kept in {})",
            n_steps + 1,
            n + 1,
            rows.len(),
            dir.display()
        )));
    }

    // -- reference trajectory + comparison ---------------------------------------------
    reference.set_solver(tableau.name, opts.atol, opts.rtol);
    reference.dt = opts.dt;
    reference.reset(0.0);

    let mut max_scaled = 0.0f64;
    let mut worst: Option<(usize, f64)> = None; // (state index, time)
    for (k, row) in rows.iter().enumerate() {
        if k > 0 {
            reference.run(opts.dt, false, false);
        }
        let (t_c, x_c) = (row[0], &row[1..]);
        let t_r = reference.time();
        if (t_c - t_r).abs() > 1e-9 * (1.0 + t_r.abs()) {
            return Err(verr(format!(
                "sample-time misalignment at step {k}: C t={t_c}, reference t={t_r} \
                 — the two run loops disagree; this is a fastsim bug, please report it"
            )));
        }
        let x_r = reference.state();
        for i in 0..n {
            let scaled = (x_c[i] - x_r[i]).abs() / (opts.atol + opts.rtol * x_r[i].abs());
            if scaled > max_scaled {
                max_scaled = scaled;
                worst = Some((i, t_r));
            }
        }
    }

    let build_dir = if opts.keep_build {
        Some(dir)
    } else {
        let _ = std::fs::remove_dir_all(&dir);
        None
    };

    let passed = max_scaled <= 1.0;
    let worst_state =
        worst.map(|(i, _)| state_names.get(i).cloned().unwrap_or_else(|| format!("x[{i}]")));
    let worst_time = worst.map(|(_, t)| t).unwrap_or(0.0);
    let summary = format!(
        "(max scaled error: {:.2e}, worst state: {}, at t: {:.6}, samples: {}) in {:.1} ms",
        max_scaled,
        worst_state.as_deref().unwrap_or("-"),
        worst_time,
        n_steps + 1,
        t_run.elapsed().as_secs_f64() * 1000.0
    );
    if passed {
        log.info(&format!("VERIFY_C PASSED {summary}"));
    } else {
        log.warning(&format!("VERIFY_C FAILED {summary}"));
    }

    Ok(VerifyCReport {
        passed,
        compiler,
        n_steps,
        n_states: n,
        max_scaled_error: max_scaled,
        worst_state,
        worst_time,
        files: files.iter().map(|f| f.name.clone()).collect(),
        build_dir,
    })
}
