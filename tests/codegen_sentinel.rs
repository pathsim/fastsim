//! Always-compiled sentinel for the codegen verification suite.
//!
//! Every codegen test binary (`codegen_op_coverage`, `codegen_verify_c`,
//! `codegen_verify_system`) is `#![cfg(feature = "codegen")]`, and `codegen` is
//! NOT a default feature. So a plain `cargo test` compiles ZERO of them and
//! green-passes with the commercial code-generation differentiator entirely
//! untested — nothing would fail if the `--features codegen` line were dropped
//! from CI. This file is compiled in BOTH configurations to make that gap
//! impossible to miss: with the feature off it raises a compile-time deprecation
//! warning plus a stderr banner; with it on it confirms the generator is linked.

#[cfg(not(feature = "codegen"))]
#[deprecated(
    note = "the `codegen` feature is OFF: the codegen verification suite was NOT compiled. \
            Run `cargo test --features codegen` to exercise the C code generator."
)]
const CODEGEN_SUITE_DISABLED: () = ();

/// With `codegen` off, emit a loud reminder (the `let _ =` below trips the
/// deprecation lint at compile time; the banner shows on a failing/`--nocapture`
/// run) so a default `cargo test` cannot quietly skip the differentiator.
#[cfg(not(feature = "codegen"))]
#[test]
fn codegen_suite_is_disabled_without_feature() {
    #[allow(clippy::let_unit_value)]
    let _ = CODEGEN_SUITE_DISABLED;
    eprintln!(
        "\n============================================================\n\
         WARNING: `codegen` feature OFF — the codegen verification suite\n\
         (op coverage, C equivalence, whole-system trajectories) was NOT\n\
         compiled or run. Use `cargo test --features codegen`.\n\
         ============================================================\n"
    );
}

/// With `codegen` on, confirm the generator entry point is actually linked into
/// the test build (guards against the feature silently pulling in nothing).
#[cfg(feature = "codegen")]
#[test]
fn codegen_suite_is_enabled() {
    let _ = fastsim::codegen::CodegenOptions::default();
}
