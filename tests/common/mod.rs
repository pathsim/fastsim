//! Shared test support: locate a working C compiler and compile-and-run C.
//!
//! Skips gracefully when no usable compiler is present. The machine's PATH
//! `gcc`/`cc` may be an ancient MSYS2 5.3 that ICEs on any double op, so a
//! candidate must pass a representative floating-point probe, not just
//! `--version`. Point `$FASTSIM_CC` (or `$CC`) at a modern compiler to run the
//! verification. Set `$FASTSIM_REQUIRE_CC=1` on a machine that is *meant* to
//! verify (CI, release gate) so a missing compiler FAILS loudly instead of
//! silently green-passing the numeric checks.

// Shared across several test binaries; each uses a different subset of helpers.
#![allow(dead_code)]
// Only used by the codegen verification tests, which are themselves gated.
#![cfg(feature = "codegen")]

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use fastsim::codegen::GeneratedFile;

/// Process-global counter, combined with the OS process id, so neither
/// concurrent threads within one `cargo test` process nor two *separate*
/// `cargo test` processes (each has its own `AtomicU64` starting at 0) ever
/// share a temp directory.
static UNIQ: AtomicU64 = AtomicU64::new(0);

/// A collision-free temp directory name for one compile-and-run case. Includes
/// the process id so concurrent `cargo test` invocations do not race on the same
/// `_{idx}_{counter}` path.
fn unique_dir(idx: usize) -> PathBuf {
    let uniq = UNIQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("fastsim_cg_verify_{idx}_{pid}_{uniq}"))
}

/// Whether a missing compiler must FAIL rather than skip (set on CI / the release
/// gate via `FASTSIM_REQUIRE_CC`).
fn require_cc() -> bool {
    std::env::var("FASTSIM_REQUIRE_CC").map(|v| !v.is_empty() && v != "0").unwrap_or(false)
}

/// Concatenate every generated file's contents, so `.contains()` assertions can
/// look for a snippet without caring which file (`model.c` / `model.h` / ...) it
/// landed in.
pub fn concat_sources(files: &[GeneratedFile]) -> String {
    files.iter().map(|f| f.contents.as_str()).collect::<Vec<_>>().join("\n")
}

/// Find a working gcc-style C compiler (honours `$FASTSIM_CC`, then `$CC`), or
/// `None` to skip — unless `$FASTSIM_REQUIRE_CC` is set, in which case a missing
/// compiler panics so the verification cannot silently be skipped.
///
/// Discovery + sanity probe live in the PRODUCT (`codegen::verify`, the same
/// resolution `Simulation.verify_c` uses), so the tests exercise the shipped
/// path; only the fail-loud policy is test-specific.
pub fn find_cc() -> Option<String> {
    let found = fastsim::codegen::verify::find_compiler();
    if found.is_none() && require_cc() {
        panic!(
            "FASTSIM_REQUIRE_CC is set but no working C compiler was found. \
             Point $FASTSIM_CC at a C99 compiler with libm (tried $FASTSIM_CC, $CC, cc, clang, gcc)."
        );
    }
    found
}

/// Compile + run a C source, returning the whitespace-separated f64s it prints.
/// `Ok(None)` means the exe wouldn't launch (environmental, skip); `Err` means
/// the C didn't compile (a real bug in whatever produced it).
pub fn compile_and_run(cc: &str, idx: usize, src: &str) -> Result<Option<Vec<f64>>, String> {
    compile_and_run_aux(cc, idx, src, &[])
}

/// As [`compile_and_run`], but first writes `aux` files (e.g. a `model.h` the
/// source `#include`s) into the same directory.
pub fn compile_and_run_aux(
    cc: &str,
    idx: usize,
    src: &str,
    aux: &[(&str, &str)],
) -> Result<Option<Vec<f64>>, String> {
    let dir = unique_dir(idx);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    for (name, content) in aux {
        std::fs::write(dir.join(name), content).map_err(|e| e.to_string())?;
    }
    std::fs::write(dir.join("m.c"), src).map_err(|e| e.to_string())?;
    compile_and_run_units(cc, &dir, &["m.c".to_string()])
}

/// Generate-and-run a whole model: write every file in `files`, build a `main.c`
/// from `main`, then compile all `.c` translation units together and run.
///
/// When the set has a `model.h` (the two-file Compact / Library layouts), `main`
/// is its own TU that `#include`s the header. Otherwise the single
/// self-contained `.c` is concatenated with `main` into one TU (the reentrant /
/// struct emissions that still inline their header).
pub fn compile_and_run_files(
    cc: &str,
    idx: usize,
    main: &str,
    files: &[GeneratedFile],
) -> Result<Option<Vec<f64>>, String> {
    let has_header = files.iter().any(|f| f.name.ends_with(".h"));
    // Files are named `<base>.{h,c}` (+ `<base>_blocks.*` / `<base>_solver.*`
    // under the Library layout). Library splits the integrator into
    // `<base>_solver.h` (which pulls in `<base>.h`), so that is the single
    // header a consumer includes; otherwise it is the model header itself.
    let entry_header = files
        .iter()
        .map(|f| f.name.as_str())
        .find(|n| n.ends_with("_solver.h"))
        .or_else(|| {
            files
                .iter()
                .map(|f| f.name.as_str())
                .find(|n| n.ends_with(".h") && !n.ends_with("_blocks.h"))
        })
        .unwrap_or("model.h")
        .to_string();
    let dir = unique_dir(idx);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let mut units: Vec<String> = Vec::new();
    if has_header {
        for f in files {
            std::fs::write(dir.join(&f.name), &f.contents).map_err(|e| e.to_string())?;
            if f.name.ends_with(".c") {
                units.push(f.name.clone());
            }
        }
        let main_c = format!("#include <stdio.h>\n#include \"{entry_header}\"\n{main}");
        std::fs::write(dir.join("main.c"), main_c).map_err(|e| e.to_string())?;
        units.push("main.c".to_string());
    } else {
        let model = files
            .iter()
            .find(|f| f.name.ends_with(".c"))
            .map(|f| f.contents.as_str())
            .unwrap_or("");
        let main_c = format!("#include <stdio.h>\n{model}\n{main}");
        std::fs::write(dir.join("main.c"), main_c).map_err(|e| e.to_string())?;
        units.push("main.c".to_string());
    }
    compile_and_run_units(cc, &dir, &units)
}

/// Compile several already-named C sources plus a `main.c` body into one binary
/// and run it. Every `.c` in `sources` is its own translation unit; the `.h`
/// files are written alongside. Used to prove two independently generated models
/// link into a single binary (the HIL plant+controller scenario) without symbol
/// or include-guard collisions.
pub fn compile_and_run_named(
    cc: &str,
    idx: usize,
    sources: &[(String, String)],
    main: &str,
) -> Result<Option<Vec<f64>>, String> {
    let dir = unique_dir(idx);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let mut units: Vec<String> = Vec::new();
    for (name, content) in sources {
        std::fs::write(dir.join(name), content).map_err(|e| e.to_string())?;
        if name.ends_with(".c") {
            units.push(name.clone());
        }
    }
    std::fs::write(dir.join("main.c"), main).map_err(|e| e.to_string())?;
    units.push("main.c".to_string());
    compile_and_run_units(cc, &dir, &units)
}

/// Compile the named `.c` units (already written into `dir`) into one exe and
/// run it, returning the whitespace-separated f64s it prints. `Ok(None)` = the
/// exe wouldn't launch (environmental); `Err` = the C didn't compile.
fn compile_and_run_units(cc: &str, dir: &std::path::Path, units: &[String]) -> Result<Option<Vec<f64>>, String> {
    let mut args: Vec<String> = units.to_vec();
    args.extend(["-O0", "-o", "m.exe", "-lm"].iter().map(|s| s.to_string()));
    // `cc` may carry arguments (`zig cc`) — spawn it exactly the way the
    // product's `find_compiler` probed it, with zig's caches isolated per
    // build dir (parallel test threads deadlock on zig's global cache lock).
    let compile = fastsim::codegen::verify::cc_command_in(cc, dir)
        .args(&args)
        .output()
        .map_err(|e| e.to_string())?;
    if !compile.status.success() {
        let srcs: String = units
            .iter()
            .map(|u| {
                let body = std::fs::read_to_string(dir.join(u)).unwrap_or_default();
                format!("--- {u} ---\n{body}\n")
            })
            .collect();
        return Err(format!(
            "C compile failed:\n{}\n{srcs}",
            String::from_utf8_lossy(&compile.stderr)
        ));
    }

    match Command::new(dir.join("m.exe")).output() {
        Ok(run) if run.status.success() => {
            let text = String::from_utf8_lossy(&run.stdout);
            Ok(Some(text.split_whitespace().filter_map(|t| t.parse::<f64>().ok()).collect()))
        }
        _ => Ok(None),
    }
}
