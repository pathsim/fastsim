//! Code generation from the fastsim IR.
//!
//! The fastsim IR (`crate::ir::schema`) lowers every block to scalar SSA ops.
//! This module lowers those ops to a target language; C99 is the first backend.
//! Because the op vocabulary is small (~60 kinds), a target is *one* op-lowering
//! rather than a template per block, which is what lets the same IR drive many
//! targets and settings.
//!
//! **Everything runs over the IR.** codegen is a pure function of a `Module`; it
//! never calls back into `crate::compile`. It consumes only the IR schema (data)
//! and `crate::ir::eval` (the reference evaluator used to verify generated code,
//! closing the loop against the source of truth) — exactly like `crate::compile`,
//! the other IR backend. It is gated behind the `codegen` feature (pulls in the
//! minijinja template engine).
//!
//! ## Settings ([`CodegenOptions`])
//!
//! Each setting binds to one pipeline stage:
//! - [`Numeric`] (double/float/fixed) — the emitter's scalar type and math calls.
//! - [`Reductions`] (unrolled/vectorized) — how `Reduce`/`Dot` ops lower.
//! - [`Structure`] (flat/hierarchical) — whole-system shape: one fused `dx/dt`
//!   vs one named function per block/subsystem (the generated code mirrors the
//!   model, so you can see what belongs to which block).
//! - [`Layout`] (compact/library) — output file assembly (see [`generate`]).
//! - [`SolverChoice`] — the integration-loop Butcher table.
//!
//! This module currently realizes the per-region op-lowering for `Numeric`
//! (Double/Float) and `Reductions::Unrolled`; the whole-system stages build on
//! top of `emit_region_fn`.

use std::fmt;

use crate::ir::schema::{
    BinOpKind, CmpKind, NodeId, Op, ReduceKind, Region, UnaryOpKind, Write,
};
use serde::Serialize;

mod solver;
mod system;
mod templates;
/// SIL verification of the generated C against the reference engine. Needs a
/// local C compiler and `std::process`, so it is absent from WASM builds.
#[cfg(not(target_family = "wasm"))]
pub mod verify;
pub use system::{
    event_layout, file_base, generate, struct_layout, EventInfo, EventKindInfo,
    EventLayout, LayoutVar, ModelLayout, VarKind,
};
use templates::render;

/// One emitted source artifact: a file name (e.g. `"model.c"`) and its contents.
/// [`generate`] returns the full set a build needs; the caller writes each to
/// disk (or feeds them to a compiler) under its `name`.
#[derive(Debug, Clone)]
pub struct GeneratedFile {
    pub name: String,
    pub contents: String,
}

// ======================================================================================
// Settings
// ======================================================================================

/// Full code-generation configuration. One field per pipeline stage.
#[derive(Debug, Clone, Default)]
pub struct CodegenOptions {
    pub numeric: Numeric,
    pub reductions: Reductions,
    pub structure: Structure,
    pub layout: Layout,
    pub solver: SolverChoice,
    pub api: ModelApi,
    /// Additionally emit the build scaffold: a `CMakeLists.txt` (static model
    /// library + demo executable) and an EDITABLE `<name>_main.c` demo driver
    /// that steps the model via `<name>_step` and prints a CSV trajectory,
    /// with marked HAL hook points for real I/O. Off by default — the model
    /// sources alone stay the stable, regeneratable artifact; the scaffold is
    /// a starting point the user owns.
    pub scaffold: bool,
    /// Additionally emit `<name>_trace.json`: the model-to-code trace map
    /// (block → emitted functions with file/line, block → states/outputs/
    /// params with their `SIG_*` ids, block → events) plus static metrics
    /// (RAM/stack estimates, IR op counts, per-step work). Machine-readable —
    /// the substrate for traceability audits, calibration maps (A2L) and CI
    /// size gates. Off by default.
    pub trace: bool,
    /// Additionally emit `<name>.a2l`: an ASAP2 measurement/calibration
    /// description (MEASUREMENT for time/states/outputs/inputs/memory,
    /// CHARACTERISTIC for tunable parameters) addressed via `SYMBOL_LINK`
    /// against one global model instance plus computed struct offsets —
    /// ready for XCP tooling (CANape, INCA). Offsets follow the
    /// natural-alignment layout shared by mainstream ABIs; 64-bit `size_t`
    /// assumed. Off by default.
    pub a2l: bool,
}

/// The public shape of the generated model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelApi {
    /// A single `model_t` struct holding time / states / signals / parameters /
    /// memory, with `model_get_signal` / `model_set_signal` accessors by id.
    /// Naturally reentrant; the embedded-friendly "rtModel" shape. The only API
    /// — the one that can actually be embedded (set inputs via `set_signal`).
    #[default]
    Struct,
}

/// Scalar real type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Numeric {
    #[default]
    Double,
    Float,
    /// Fixed-point Q(31-frac).frac on `int32_t` with `int64_t` intermediates
    /// (`frac` fractional bits, 1..=30). Arithmetic wraps through a defined
    /// int64 truncation (never C signed-overflow UB). Ops without an integer
    /// lowering (transcendentals, `pow`, `atan2`, ...) are rejected with a
    /// precise message — model them with `LUT1D` (the embedded pattern) or
    /// generate `double`/`float`. Fixed-step tableaus only.
    Fixed {
        /// Fractional bits: resolution `2^-frac`, range `±2^(31-frac)`.
        frac: u8,
    },
}

/// Default Q format for the plain `"fixed"` spelling: Q16.16 — resolution
/// ~1.5e-5, range ±32768. A sensible middle ground for control loops.
pub const DEFAULT_FIXED_FRAC: u8 = 16;

impl Numeric {
    /// The C scalar type.
    pub fn real(self) -> &'static str {
        match self {
            Numeric::Double => "double",
            Numeric::Float => "float",
            Numeric::Fixed { .. } => "int32_t",
        }
    }
    /// Suffix on `<math.h>` functions and float literals (`sinf`, `1.5f`).
    fn suffix(self) -> &'static str {
        match self {
            Numeric::Float => "f",
            _ => "",
        }
    }
    /// Fractional bits when fixed-point, else `None`.
    pub(crate) fn frac(self) -> Option<u8> {
        match self {
            Numeric::Fixed { frac } => Some(frac),
            _ => None,
        }
    }
}

/// How `Reduce`/`Dot` ops lower: inline expression vs a counted loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Reductions {
    #[default]
    Unrolled,
    /// Gather the operands into a local array and fold in a counted `for` loop
    /// (see `CTarget::reduce_block` / `dot_block`).
    Vectorized,
}

/// Whole-system shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Structure {
    /// One named function per block/subsystem, wired via connections; the
    /// generated code mirrors the model structure (readable, auditable).
    #[default]
    Hierarchical,
    /// One fused `dx/dt = F(X, t)`; block boundaries dissolve (compact).
    Flat,
}

/// Output file assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Layout {
    /// Two files: `model.c` (implementation) and `model.h` (public interface:
    /// dimensions, the model/state struct, prototypes). The default.
    #[default]
    Compact,
    /// Multiple files split by concern: `model.{c,h}` (the `F(x, t)` model),
    /// `solver.{c,h}` (the tableau-driven integrator, fixed-step or adaptive), and under
    /// [`Structure::Hierarchical`] `blocks.{c,h}` (the per-block functions).
    Library,
}

/// Integrator for the generated loop, selected by Butcher tableau. Every
/// solver — forward Euler, RK4, and every adaptive method alike — is one entry
/// in `crate::solvers::tableaus` (the same registry the runtime uses), so there
/// is no special-cased method: codegen and runtime cannot drift, and the emitter
/// has a single tableau-driven path. Fixed-step tableaus get a plain stage kernel;
/// adaptive ones (`tr` non-empty) get the embedded-error step controller.
#[derive(Debug, Clone, Copy)]
pub struct SolverChoice {
    tableau: &'static crate::solvers::tableaus::Tableau,
}

impl Default for SolverChoice {
    /// Classical RK4 — the historical codegen default.
    fn default() -> Self {
        SolverChoice { tableau: &crate::solvers::tableaus::RK4 }
    }
}

impl SolverChoice {
    /// Forward Euler (one-stage explicit; codegen-only, see [`solver::EUF_TABLEAU`]).
    pub const EULER: SolverChoice = SolverChoice { tableau: &solver::EUF_TABLEAU };
    /// Classical RK4.
    pub const RK4: SolverChoice = SolverChoice { tableau: &crate::solvers::tableaus::RK4 };

    /// Select a solver by tableau name. Accepts every explicit tableau in the
    /// runtime registry (`"RK4"`, `"RKDP54"`, `"RKCK54"`, ...) plus `"EUF"`
    /// (forward Euler). Returns `None` for an unknown name or an implicit tableau
    /// (DIRK/ESDIRK codegen needs a generated Newton/linear solve — not yet).
    pub fn by_name(name: &str) -> Option<SolverChoice> {
        let t = if name.eq_ignore_ascii_case("EUF") {
            &solver::EUF_TABLEAU
        } else {
            crate::solvers::tableaus::by_name(name)?
        };
        t.is_explicit().then_some(SolverChoice { tableau: t })
    }

    /// The Butcher tableau the emitter lowers (always explicit by construction).
    pub(crate) fn tableau(self) -> &'static crate::solvers::tableaus::Tableau {
        self.tableau
    }
}

// ======================================================================================
// Errors
// ======================================================================================

/// An error during code generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodegenError {
    /// An op or setting the backend does not (yet) lower.
    Unsupported(String),
    /// A template failed to render (internal: a context/template mismatch).
    Template(String),
    /// SIL verification could not run or produced malformed output (missing
    /// compiler, compile/run failure, harness mismatch) — see [`verify`].
    Verify(String),
}

impl fmt::Display for CodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodegenError::Unsupported(what) => write!(f, "codegen: unsupported {what}"),
            CodegenError::Template(e) => write!(f, "codegen: template error: {e}"),
            CodegenError::Verify(e) => write!(f, "codegen: verification: {e}"),
        }
    }
}

impl std::error::Error for CodegenError {}

type R<T> = Result<T, CodegenError>;

// ======================================================================================
// C emission (per region)
// ======================================================================================

/// Stateless PRNG helper, the C twin of `jit::graph::rand_uniform` (splitmix64
/// finalizer over the key's bits → uniform `[0, 1)`). Emitted into the generated
/// header so every region function that calls `UnaryOp::RandUniform` sees it, and
/// reused by `c_prelude` for the region-verify harness. Bit-identical to the
/// interpreter and tape (same constants, same `>> 11` mantissa extraction), so a
/// code-generated noise source replays exactly like the compiled model.
pub const RNG_HELPER_C: &str = "\
#include <stdint.h>
#include <string.h>
static inline double fastsim_rand_uniform(double fs_key) {
    uint64_t z;
    memcpy(&z, &fs_key, sizeof z);
    z += (uint64_t)0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * (uint64_t)0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * (uint64_t)0x94D049BB133111EBULL;
    z ^= z >> 31;
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}
";

/// Digamma ψ(x) helper, the C twin of `ssa::op::digamma` (recursion into the
/// asymptotic regime + Bernoulli expansion). It is the derivative of `lgamma`,
/// so the forward-AD `model_jvp` emits a call to it when differentiating
/// `lgamma`/`tgamma`; C's `<math.h>` has no digamma. Bit-identical to the Rust
/// kernel, so a generated Jacobian matches the interpreter. Needs `<math.h>`
/// (for `log`), already included wherever this is emitted.
pub const DIGAMMA_HELPER_C: &str = "\
static inline double fastsim_digamma(double x) {
    double result = 0.0;
    while (x < 6.0) { result -= 1.0 / x; x += 1.0; }
    double inv = 1.0 / x;
    double inv2 = inv * inv;
    result += log(x) - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0)));
    return result;
}
";

/// Shared C prelude: the math include, the equality tolerance (sourced from
/// fastsim's constant so the generated `Eq`/`Ne` matches the reference exactly),
/// the stateless PRNG helper, and the digamma helper (gamma-derivative kernel).
pub fn c_prelude() -> String {
    format!(
        "#include <math.h>\n#define FASTSIM_EQ_TOL {}\n{}{}",
        fmt_lit(crate::constants::JIT_FLOAT_EQ_TOL, Numeric::Double),
        RNG_HELPER_C,
        DIGAMMA_HELPER_C,
    )
}

/// Emit a C function `name` that evaluates `region`'s writes into `out[]`.
///
/// Signature: `static void name(const T* u, const T* x, const T* p,
/// const T* m, T t, T* out)`, where `out[i]` is the i-th write in declaration
/// order (matching `crate::ir::eval::eval_region`'s output vector). `T` is the
/// configured [`Numeric`] type.
pub fn emit_region_fn(name: &str, region: &Region, opts: &CodegenOptions, mem_off: &[usize], in_sizes: &[u32]) -> R<String> {
    emit_region_fn_linkage(name, region, opts, mem_off, in_sizes, "static ", String::new())
}

/// As [`emit_region_fn`], but with an explicit storage-class prefix (`"static "`
/// for an inlined function, `""` for one that is externally linked and declared
/// in a header, e.g. the Library layout's `blocks.c`).
pub(crate) fn emit_region_fn_linkage(
    name: &str,
    region: &Region,
    opts: &CodegenOptions,
    mem_off: &[usize],
    in_sizes: &[u32],
    linkage: &'static str,
    doc: String,
) -> R<String> {
    let target = CTarget::new(opts)?;
    let (stmts, writes) = lower_region(region, &target, mem_off, in_sizes)?;
    render("region.c", RegionCtx { name, real: target.scalar_ty(), stmts, writes, linkage, doc })
}

/// One SSA op lowered for emission, keyed by its node index (`vN`). Normally a C
/// rvalue in `expr` (rendered as `const T vN = expr;`); a vectorized reduction
/// instead sets `block` to a self-contained multi-line block that declares `vN`
/// (a counted loop over a gathered operand array).
#[derive(Serialize)]
pub(crate) struct Stmt {
    pub id: usize,
    pub expr: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block: Option<String>,
}

/// One region write, as `out[idx] = vN` (the write kind is irrelevant here: the
/// caller picks the region whose writes it wants, matching `ir::eval`).
#[derive(Serialize)]
pub(crate) struct OutWrite {
    pub idx: usize,
    pub src: u32,
}

#[derive(Serialize)]
struct RegionCtx<'a> {
    name: &'a str,
    real: &'static str,
    stmts: Vec<Stmt>,
    writes: Vec<OutWrite>,
    /// Storage-class prefix: `"static "` (inlined) or `""` (extern, header-declared).
    linkage: &'static str,
    /// One-line provenance comment above the function (block type + name + which
    /// region), or empty for none.
    doc: String,
}

/// Lower a region's ops to its statement list (`vN = <expr>`), in the configured
/// target language. Shared by region and effect emission (their *writes* differ,
/// but the op-statements are identical).
pub(crate) fn lower_stmts(region: &Region, target: &CTarget, mem_off: &[usize], in_sizes: &[u32]) -> R<Vec<Stmt>> {
    let leaves = HierLeaves { target, mem_off, in_sizes };
    region
        .ops
        .iter()
        .enumerate()
        .map(|(id, op)| lower_one(op, id, target, &leaves))
        .collect()
}

/// Lower one op into a `Stmt`. Under `Reductions::Vectorized`, a non-empty
/// `Reduce`/`Dot` becomes a counted-loop block; everything else (and the empty
/// case) is the single-expression form. Shared by the hierarchical and flat
/// drivers (both pass their own `Leaves`).
pub(crate) fn lower_one<L: Leaves>(op: &Op, id: usize, target: &CTarget, leaves: &L) -> R<Stmt> {
    // A lookup table always emits a block (static const arrays + counted search),
    // independent of the reduction setting.
    if let Op::Lut1d { input, points, values, clamp } = op {
        let block = target.lut1d_block(id, &leaves.temp(*input), points, values, *clamp)?;
        return Ok(Stmt { id, expr: String::new(), block: Some(block) });
    }
    if target.vectorized() {
        match op {
            Op::Reduce { op: rop, args } if !args.is_empty() => {
                let parts: Vec<String> = args.iter().map(|n| leaves.temp(*n)).collect();
                return Ok(Stmt { id, expr: String::new(), block: Some(target.reduce_block(id, *rop, &parts)) });
            }
            Op::Dot { a, b } if !a.is_empty() => {
                let pa: Vec<String> = a.iter().map(|n| leaves.temp(*n)).collect();
                let pb: Vec<String> = b.iter().map(|n| leaves.temp(*n)).collect();
                return Ok(Stmt { id, expr: String::new(), block: Some(target.dot_block(id, &pa, &pb)) });
            }
            _ => {}
        }
    }
    Ok(Stmt { id, expr: op_expr(op, target, leaves)?, block: None })
}

/// Lower a region to its statement list and output (`out[]`) writes.
pub(crate) fn lower_region(
    region: &Region,
    target: &CTarget,
    mem_off: &[usize],
    in_sizes: &[u32],
) -> R<(Vec<Stmt>, Vec<OutWrite>)> {
    let stmts = lower_stmts(region, target, mem_off, in_sizes)?;
    let writes = region
        .writes
        .iter()
        .enumerate()
        .map(|(idx, w)| {
            let src = match w {
                Write::Output { src, .. }
                | Write::StateDeriv { src, .. }
                | Write::StateWrite { src, .. }
                | Write::MemoryWrite { src, .. } => src.0,
            };
            Ok(OutWrite { idx, src })
        })
        .collect::<R<Vec<_>>>()?;
    Ok((stmts, writes))
}

/// A backend language. The op-walking driver ([`op_expr`]) is target-agnostic;
/// each `Target` owns the surface syntax for one language (its scalar type,
/// literal format, array indexing, and the per-op-kind expression strings). C99
/// is [`CTarget`]; a second target (e.g. Python) is another impl, not a fork of
/// the driver. This is the seam that keeps op lowering shared across targets.
pub(crate) trait Target {
    /// The scalar real type (`double`, `float`, ...).
    fn scalar_ty(&self) -> &'static str;
    /// A numeric literal of the scalar type.
    fn literal(&self, x: f64) -> String;
    /// An SSA temporary reference (`vN`).
    fn temp(&self, id: u32) -> String;
    /// The simulation-time variable.
    fn time(&self) -> String;
    /// An array element `base[idx]` (`u`/`x`/`p`/`m`).
    fn index(&self, base: &str, idx: usize) -> String;
    fn binary(&self, op: BinOpKind, a: &str, b: &str) -> String;
    /// Pre-flight for `binary`: reject an op this target cannot lower (the
    /// emitter itself is infallible). Default: everything is lowerable.
    fn check_fixed_binary(&self, _op: BinOpKind) -> R<()> {
        Ok(())
    }
    fn unary(&self, op: UnaryOpKind, a: &str) -> R<String>;
    fn cmp(&self, op: CmpKind, a: &str, b: &str) -> String;
    fn select(&self, c: &str, t: &str, e: &str) -> String;
    fn fma(&self, a: &str, b: &str, c: &str) -> String;
    /// Unrolled variadic reduction over already-rendered operand strings.
    fn reduce(&self, op: ReduceKind, parts: &[String]) -> String;
    /// Unrolled dot product over already-rendered operand-pair strings.
    fn dot(&self, a: &[String], b: &[String]) -> String;
}

/// Resolves an op's *leaf* operands (reads and node references) to target-
/// language strings. The operator structure is shared by [`op_expr`]; only how a
/// node-ref / `Input` / `State` / `Param` / `Memory` / `Const` / `Time` renders
/// differs between the *hierarchical* driver (per-block: `vN`, `u[]`, `x[]`,
/// `p[]`, `m[]`) and the *flat* driver (one fused function: globally-renumbered
/// temps, inputs resolved to the producing temp, params inlined).
pub(crate) trait Leaves {
    /// A reference to an earlier op's value (a node id local to the region).
    fn temp(&self, node: NodeId) -> String;
    fn constant(&self, c: f64) -> String;
    fn time(&self) -> String;
    fn input(&self, port: u32, elem: u32) -> R<String>;
    fn state(&self, id: u32) -> String;
    fn param(&self, id: u32) -> String;
    fn memory(&self, slot: usize, offset: u32) -> R<String>;
}

/// The rvalue for a single op in the `target`'s language, with leaf operands
/// resolved by `leaves`. The operator dispatch (binary/unary/.../dot) is shared
/// across every target and both drivers; only the leaves vary.
pub(crate) fn op_expr<T: Target, L: Leaves>(op: &Op, target: &T, leaves: &L) -> R<String> {
    let v = |node: &NodeId| leaves.temp(*node);
    Ok(match op {
        Op::Const(c) => leaves.constant(*c),
        Op::Time => leaves.time(),
        Op::Input { port, elem } => leaves.input(*port, *elem)?,
        Op::State { id } => leaves.state(id.0),
        Op::Param { id } => leaves.param(id.0),
        Op::Memory { slot, offset } => leaves.memory(slot.idx(), *offset)?,
        Op::Binary { op, a, b } => {
            target.check_fixed_binary(*op)?;
            target.binary(*op, &v(a), &v(b))
        }
        Op::Unary { op, a } => target.unary(*op, &v(a))?,
        Op::Cmp { op, a, b } => target.cmp(*op, &v(a), &v(b)),
        Op::Select { c, t, e } => target.select(&v(c), &v(t), &v(e)),
        Op::Fma { a, b, c } => target.fma(&v(a), &v(b), &v(c)),
        Op::Reduce { op, args } => target.reduce(*op, &args.iter().map(&v).collect::<Vec<_>>()),
        Op::Dot { a, b } => target.dot(
            &a.iter().map(&v).collect::<Vec<_>>(),
            &b.iter().map(&v).collect::<Vec<_>>(),
        ),
        // Lut1d carries a table and emits a multi-line block, not an rvalue;
        // `lower_one` intercepts it before this driver is reached.
        Op::Lut1d { .. } => {
            return Err(CodegenError::Unsupported("Op::Lut1d must lower as a block".into()))
        }
        Op::Call { .. } => return Err(CodegenError::Unsupported("Op::Call (opaque extern)".into())),
    })
}

/// Hierarchical leaf resolution: each block becomes a function whose reads index
/// its `u`/`x`/`p`/`m` parameters and whose node-refs are `vN`. `mem_off[slot]`
/// is the slot's base in the global `m[]`.
pub(crate) struct HierLeaves<'a, T> {
    target: &'a T,
    mem_off: &'a [usize],
    /// Input-port element sizes, so a multi-port `Input { port, elem }` resolves
    /// to the right flat index in the gathered `u[]` (matching `StructLeaves`).
    in_sizes: &'a [u32],
}

impl<T: Target> Leaves for HierLeaves<'_, T> {
    fn temp(&self, node: NodeId) -> String {
        self.target.temp(node.0)
    }
    fn constant(&self, c: f64) -> String {
        self.target.literal(c)
    }
    fn time(&self) -> String {
        self.target.time()
    }
    fn input(&self, port: u32, elem: u32) -> R<String> {
        // `u[]` concatenates the input ports element-wise (the call site's gather
        // lays them out the same way), so the flat index is `port_offset + elem`.
        let flat = system::port_offset(self.in_sizes, port)? + elem as usize;
        Ok(self.target.index("u", flat))
    }
    fn state(&self, id: u32) -> String {
        self.target.index("x", id as usize)
    }
    fn param(&self, id: u32) -> String {
        self.target.index("p", id as usize)
    }
    fn memory(&self, slot: usize, offset: u32) -> R<String> {
        let base = self
            .mem_off
            .get(slot)
            .copied()
            .ok_or_else(|| CodegenError::Unsupported(format!("memory slot {slot} has no layout")))?;
        Ok(self.target.index("m", base + offset as usize))
    }
}

// ======================================================================================
// C99 target
// ======================================================================================

/// C99 backend. Carries the numeric type (so `<math.h>` suffixes and literals
/// render correctly: `sinf`, `1.5f`) and the reduction style (unrolled
/// expression vs counted loop).
pub(crate) struct CTarget {
    numeric: Numeric,
    reductions: Reductions,
}

impl CTarget {
    /// Build the C target for these options, rejecting settings not yet emitted.
    pub(crate) fn new(opts: &CodegenOptions) -> R<Self> {
        Ok(Self { numeric: opts.numeric, reductions: opts.reductions })
    }

    /// A suffixed `<math.h>` call: `f(args)` for double, `ff(args)` for float.
    fn mfn(&self, f: &str, args: &str) -> String {
        format!("{f}{}({args})", self.numeric.suffix())
    }

    /// Nested binary-function fold: `f(f(a,b),c)`. Empty -> `identity`.
    fn fold_call(&self, f: &str, parts: &[String], identity: &str) -> String {
        let f = format!("{f}{}", self.numeric.suffix());
        let mut it = parts.iter();
        match it.next() {
            None => identity.to_string(),
            Some(first) => it.fold(first.clone(), |acc, p| format!("{f}({acc}, {p})")),
        }
    }

    /// Whether reductions lower to a counted loop (vs an unrolled expression).
    pub(crate) fn vectorized(&self) -> bool {
        self.reductions == Reductions::Vectorized
    }

    /// A vectorized reduction: gather the operands into a local array and fold
    /// over them in a counted loop. Declares `v{id}`; the operand list is never
    /// empty (the caller keeps that case on the unrolled path).
    fn reduce_block(&self, id: usize, op: ReduceKind, parts: &[String]) -> String {
        let t = self.numeric.real();
        let arr = format!("_r{id}");
        let (init, step) = if self.numeric.frac().is_some() {
            // Fixed point: fold through the Q-aware binary emitter (widened
            // adds, shift-scaled products, ternary min/max).
            let acc = format!("v{id}");
            let elem = format!("{arr}[_i]");
            let fold = |b: BinOpKind| format!("v{id} = {};", self.binary(b, &acc, &elem));
            match op {
                ReduceKind::Sum => (self.literal(0.0), fold(BinOpKind::Add)),
                ReduceKind::Product => (self.literal(1.0), fold(BinOpKind::Mul)),
                ReduceKind::Min => ("INT32_MAX".to_string(), fold(BinOpKind::Min)),
                ReduceKind::Max => ("INT32_MIN".to_string(), fold(BinOpKind::Max)),
            }
        } else {
            match op {
                ReduceKind::Sum => (self.literal(0.0), format!("v{id} += {arr}[_i];")),
                ReduceKind::Product => (self.literal(1.0), format!("v{id} *= {arr}[_i];")),
                ReduceKind::Min => ("INFINITY".to_string(), format!("v{id} = {};", self.mfn("fmin", &format!("v{id}, {arr}[_i]")))),
                ReduceKind::Max => ("(-INFINITY)".to_string(), format!("v{id} = {};", self.mfn("fmax", &format!("v{id}, {arr}[_i]")))),
            }
        };
        let n = parts.len();
        let joined = parts.join(", ");
        format!(
            "    {t} v{id};\n    {{\n        const {t} {arr}[] = {{ {joined} }};\n        \
             v{id} = {init};\n        for (size_t _i = 0; _i < {n}; _i++) {step}\n    }}\n"
        )
    }

    /// A 1-D lookup table: the breakpoints and values as `static const` arrays
    /// (read-only, shared across instances), a counted search for the segment,
    /// and linear interpolation. Mirrors `ir::eval::lut1d` op-for-op.
    fn lut1d_block(&self, id: usize, input: &str, points: &[f64], values: &[f64], clamp: bool) -> R<String> {
        let n = points.len();
        if n < 2 || values.len() != n {
            return Err(CodegenError::Unsupported("Lut1d needs >= 2 points and matching values".into()));
        }
        let t = self.numeric.real();
        let (lx, ly) = (format!("_lx{id}"), format!("_ly{id}"));
        let px = points.iter().map(|v| self.literal(*v)).collect::<Vec<_>>().join(", ");
        let py = values.iter().map(|v| self.literal(*v)).collect::<Vec<_>>().join(", ");
        // Segment interpolation, numeric-aware: the fraction `_tt` and the
        // final blend go through the Q-aware Div/Mul under fixed point.
        let tt_expr = self.binary(
            BinOpKind::Div,
            &format!("({input} - {lx}[_k])"),
            &format!("({lx}[_k + 1] - {lx}[_k])"),
        );
        let blend = self.binary(BinOpKind::Mul, "_tt", &format!("({ly}[_k + 1] - {ly}[_k])"));
        let mut s = format!(
            "    {t} v{id};\n    {{\n        \
             static const {t} {lx}[] = {{ {px} }};\n        \
             static const {t} {ly}[] = {{ {py} }};\n        \
             size_t _k = 0;\n        \
             for (size_t _j = 1; _j + 1 < {n}; _j++) if ({input} >= {lx}[_j]) _k = _j;\n        \
             {t} _tt = {tt_expr};\n        \
             v{id} = {ly}[_k] + {blend};\n"
        );
        if clamp {
            let last = n - 1;
            s.push_str(&format!(
                "        if ({input} > {lx}[{last}]) v{id} = {ly}[{last}];\n        \
                 else if ({input} < {lx}[0]) v{id} = {ly}[0];\n"
            ));
        }
        s.push_str("    }\n");
        Ok(s)
    }

    /// Forward-AD tangent of a 1-D lookup table: `d = slope(segment) * d_input`,
    /// where `slope` is the active piecewise-linear segment's gradient. Declares
    /// `d{id}` from the input's primal (`input`) and tangent (`dinput`). With
    /// clamping the tangent is 0 outside the breakpoint range (flat extrapolation).
    pub(crate) fn lut1d_tangent_block(
        &self,
        id: usize,
        input: &str,
        dinput: &str,
        points: &[f64],
        values: &[f64],
        clamp: bool,
    ) -> R<String> {
        let n = points.len();
        if n < 2 || values.len() != n {
            return Err(CodegenError::Unsupported("Lut1d needs >= 2 points and matching values".into()));
        }
        let t = self.numeric.real();
        let (lx, ly) = (format!("_dlx{id}"), format!("_dly{id}"));
        let px = points.iter().map(|v| self.literal(*v)).collect::<Vec<_>>().join(", ");
        let py = values.iter().map(|v| self.literal(*v)).collect::<Vec<_>>().join(", ");
        let mut s = format!(
            "    {t} d{id};\n    {{\n        \
             static const {t} {lx}[] = {{ {px} }};\n        \
             static const {t} {ly}[] = {{ {py} }};\n        \
             size_t _dk = 0;\n        \
             for (size_t _dj = 1; _dj + 1 < {n}; _dj++) if ({input} >= {lx}[_dj]) _dk = _dj;\n        \
             {t} _dslope = {slope_expr};\n        \
             d{id} = {dmul_expr};\n",
            slope_expr = self.binary(
                BinOpKind::Div,
                &format!("({ly}[_dk + 1] - {ly}[_dk])"),
                &format!("({lx}[_dk + 1] - {lx}[_dk])"),
            ),
            dmul_expr = self.binary(BinOpKind::Mul, "_dslope", dinput),
        );
        if clamp {
            let last = n - 1;
            s.push_str(&format!(
                "        if ({input} > {lx}[{last}] || {input} < {lx}[0]) d{id} = {};\n",
                self.literal(0.0)
            ));
        }
        s.push_str("    }\n");
        Ok(s)
    }

    /// A vectorized dot product: gather both operand lists and accumulate with
    /// `fma` in a counted loop. Declares `v{id}`; lists are non-empty.
    fn dot_block(&self, id: usize, a: &[String], b: &[String]) -> String {
        let t = self.numeric.real();
        let (aa, bb) = (format!("_a{id}"), format!("_b{id}"));
        let n = a.len();
        let (ja, jb) = (a.join(", "), b.join(", "));
        let fma = self.fma(&format!("{aa}[_i]"), &format!("{bb}[_i]"), &format!("v{id}"));
        let _ = &fma;
        format!(
            "    {t} v{id};\n    {{\n        const {t} {aa}[] = {{ {ja} }};\n        \
             const {t} {bb}[] = {{ {jb} }};\n        v{id} = {};\n        \
             for (size_t _i = 0; _i < {n}; _i++) v{id} = {fma};\n    }}\n",
            self.literal(0.0)
        )
    }
}

impl Target for CTarget {
    fn scalar_ty(&self) -> &'static str {
        self.numeric.real()
    }

    fn literal(&self, x: f64) -> String {
        fmt_lit(x, self.numeric)
    }

    fn temp(&self, id: u32) -> String {
        format!("v{id}")
    }

    fn time(&self) -> String {
        "t".to_string()
    }

    fn index(&self, base: &str, idx: usize) -> String {
        format!("{base}[{idx}]")
    }

    fn binary(&self, op: BinOpKind, a: &str, b: &str) -> String {
        if let Some(frac) = self.numeric.frac() {
            // Q arithmetic on int32 through int64 intermediates: sums/products
            // widen first (a signed overflow in C is UB; the truncating cast
            // back to int32 is the DEFINED wrap this backend documents), and
            // multiplication/division carry the 2^frac scale explicitly.
            return match op {
                BinOpKind::Add => format!("(int32_t)((int64_t){a} + (int64_t){b})"),
                BinOpKind::Sub => format!("(int32_t)((int64_t){a} - (int64_t){b})"),
                BinOpKind::Mul => format!("(int32_t)(((int64_t){a} * (int64_t){b}) >> {frac})"),
                BinOpKind::Div => format!("(int32_t)((((int64_t){a}) << {frac}) / (int64_t){b})"),
                // C integer % truncates toward zero — the same convention as
                // fmod, so Q remainders keep the float semantics.
                BinOpKind::Mod => format!("(int32_t)((int64_t){a} % (int64_t){b})"),
                BinOpKind::Min => format!("({a} < {b} ? {a} : {b})"),
                BinOpKind::Max => format!("({a} > {b} ? {a} : {b})"),
                // Unreachable: `check_fixed_binary` rejected these in
                // `lower_one` with a precise error. The undeclared call is a
                // hard compile error should a future path ever slip through.
                BinOpKind::Pow | BinOpKind::Atan2 | BinOpKind::Hypot => {
                    format!("fastsim_fixed_unsupported_{op:?}({a}, {b})")
                }
            };
        }
        // Plain libm twin (pow/fmod/fmin/...) comes from the op manifest; only
        // the infix operators are hand-written here.
        if let Some(f) = crate::ssa::op::binary_c_fn(op) {
            return self.mfn(f, &format!("{a}, {b}"));
        }
        match op {
            BinOpKind::Add => format!("({a} + {b})"),
            BinOpKind::Sub => format!("({a} - {b})"),
            BinOpKind::Mul => format!("({a} * {b})"),
            BinOpKind::Div => format!("({a} / {b})"),
            _ => unreachable!("binary_c_fn covers all non-infix BinOps"),
        }
    }

    /// Reject an op that has no integer lowering under fixed point, with a
    /// message naming the alternative. Called from `lower_one` BEFORE the
    /// (infallible) `binary` emitter; the fixed `unary` arm rejects its own.
    fn check_fixed_binary(&self, op: BinOpKind) -> R<()> {
        if self.numeric.frac().is_some()
            && matches!(op, BinOpKind::Pow | BinOpKind::Atan2 | BinOpKind::Hypot)
        {
            return Err(CodegenError::Unsupported(format!(
                "op {op:?} has no integer lowering under fixed point; model it \
                 with a LUT1D block or generate numeric=\"double\"/\"float\""
            )));
        }
        Ok(())
    }

    fn unary(&self, op: UnaryOpKind, a: &str) -> R<String> {
        use UnaryOpKind as U;
        if let Some(frac) = self.numeric.frac() {
            // Integer-expressible unaries; everything transcendental is
            // rejected with the LUT1D pointer. Bit masks assume two's
            // complement (universal on the targets C99 code reaches).
            let mask = (1i64 << frac) - 1; // fractional-bit mask
            let half = 1i64 << (frac - 1);
            let one = one(self.numeric);
            return match op {
                U::Neg => Ok(format!("(-{a})")),
                U::Abs => Ok(format!("({a} < 0 ? -{a} : {a})")),
                U::Sign => Ok(format!("({a} > 0 ? {one} : ({a} < 0 ? -{one} : {a}))")),
                // floor: clear the fractional bits (rounds toward -inf on
                // two's complement).
                U::Floor => Ok(format!("(int32_t)((int64_t){a} & ~{mask}LL)")),
                U::Ceil => Ok(format!("(int32_t)(((int64_t){a} + {mask}LL) & ~{mask}LL)")),
                // round half away from zero, matching Rust f64::round.
                U::Round => Ok(format!(
                    "(int32_t)({a} < 0 ? -((-(int64_t){a} + {half}LL) & ~{mask}LL) : (((int64_t){a} + {half}LL) & ~{mask}LL))"
                )),
                U::Trunc => Ok(format!(
                    "(int32_t)({a} < 0 ? -((-(int64_t){a}) & ~{mask}LL) : ((int64_t){a} & ~{mask}LL))"
                )),
                _ => Err(CodegenError::Unsupported(format!(
                    "op {op:?} has no integer lowering under fixed point; model it \
                     with a LUT1D block or generate numeric=\"double\"/\"float\""
                ))),
            };
        }
        // Plain libm twin (sin/cos/fabs/...) comes from the op manifest; only
        // the special forms below are hand-written here.
        if let Some(f) = crate::ssa::op::unary_c_fn(op) {
            return Ok(self.mfn(f, a));
        }
        let s = match op {
            U::Neg => format!("(-{a})"),
            // numpy `sign` semantics, matching `ssa::op::numpy_sign` (0 at ±0,
            // NaN passthrough) — the ternary chain is C's spelling of numpy's
            // own kernel `in > 0 ? 1 : (in < 0 ? -1 : in)`.
            U::Sign => {
                let (one, zero) = (one(self.numeric), zero(self.numeric));
                format!("({a} > {zero} ? {one} : ({a} < {zero} ? -{one} : {a}))")
            }
            // No C stdlib digamma; emitted via the `fastsim_digamma` helper
            // (see `DIGAMMA_HELPER_C`). Produced as the gamma/lgamma derivative.
            U::Digamma => format!("fastsim_digamma({a})"),
            // Stateless PRNG hash → the `fastsim_rand_uniform` helper emitted in
            // the generated header (see `RNG_HELPER_C`). Bit-identical to the tape.
            U::RandUniform => format!("fastsim_rand_uniform({a})"),
            _ => unreachable!("unary_c_fn covers all plain-libm UnaryOps"),
        };
        Ok(s)
    }

    fn cmp(&self, op: CmpKind, a: &str, b: &str) -> String {
        let (one, zero) = (one(self.numeric), zero(self.numeric));
        if let Some(frac) = self.numeric.frac() {
            // Comparisons on the raw Q ints; Eq/Ne against the float-parity
            // tolerance quantized to Q, floored at one tick (a zero tolerance
            // would make Eq exact-integer equality — stricter than the float
            // semantics this mirrors).
            let tol = ((crate::constants::JIT_FLOAT_EQ_TOL * (1i64 << frac) as f64).round()
                as i64)
                .max(1);
            let diff = format!("((int64_t){a} - (int64_t){b})");
            return match op {
                CmpKind::Gt => format!("({a} > {b} ? {one} : {zero})"),
                CmpKind::Ge => format!("({a} >= {b} ? {one} : {zero})"),
                CmpKind::Lt => format!("({a} < {b} ? {one} : {zero})"),
                CmpKind::Le => format!("({a} <= {b} ? {one} : {zero})"),
                CmpKind::Eq => format!("(({diff} < 0 ? -{diff} : {diff}) < {tol}LL ? {one} : {zero})"),
                CmpKind::Ne => format!("(({diff} < 0 ? -{diff} : {diff}) >= {tol}LL ? {one} : {zero})"),
            };
        }
        match op {
            CmpKind::Gt => format!("({a} > {b} ? {one} : {zero})"),
            CmpKind::Ge => format!("({a} >= {b} ? {one} : {zero})"),
            CmpKind::Lt => format!("({a} < {b} ? {one} : {zero})"),
            CmpKind::Le => format!("({a} <= {b} ? {one} : {zero})"),
            CmpKind::Eq => format!("({} < FASTSIM_EQ_TOL ? {one} : {zero})", self.mfn("fabs", &format!("{a} - {b}"))),
            CmpKind::Ne => format!("({} >= FASTSIM_EQ_TOL ? {one} : {zero})", self.mfn("fabs", &format!("{a} - {b}"))),
        }
    }

    fn select(&self, c: &str, t: &str, e: &str) -> String {
        format!("({c} != {} ? {t} : {e})", zero(self.numeric))
    }

    fn fma(&self, a: &str, b: &str, c: &str) -> String {
        if let Some(frac) = self.numeric.frac() {
            return format!("(int32_t)((((int64_t){a} * (int64_t){b}) >> {frac}) + (int64_t){c})");
        }
        self.mfn("fma", &format!("{a}, {b}, {c}"))
    }

    fn reduce(&self, op: ReduceKind, parts: &[String]) -> String {
        if self.numeric.frac().is_some() {
            // Fold through the fixed-aware binary emitter (widened adds,
            // shift-scaled products, ternary min/max).
            let (bop, identity) = match op {
                ReduceKind::Sum => (BinOpKind::Add, zero(self.numeric)),
                ReduceKind::Product => (BinOpKind::Mul, one(self.numeric)),
                ReduceKind::Min => (BinOpKind::Min, "INT32_MAX".to_string()),
                ReduceKind::Max => (BinOpKind::Max, "INT32_MIN".to_string()),
            };
            let mut it = parts.iter();
            return match it.next() {
                None => identity,
                Some(first) => it.fold(first.clone(), |acc, p| self.binary(bop, &acc, p)),
            };
        }
        match op {
            ReduceKind::Sum => {
                if parts.is_empty() { zero(self.numeric) } else { format!("({})", parts.join(" + ")) }
            }
            ReduceKind::Product => {
                if parts.is_empty() { one(self.numeric) } else { format!("({})", parts.join(" * ")) }
            }
            ReduceKind::Min => self.fold_call("fmin", parts, "INFINITY"),
            ReduceKind::Max => self.fold_call("fmax", parts, "(-INFINITY)"),
        }
    }

    fn dot(&self, a: &[String], b: &[String]) -> String {
        a.iter().zip(b.iter()).fold(zero(self.numeric), |acc, (ai, bi)| {
            self.fma(ai, bi, &acc)
        })
    }
}

/// `1.0` / `1.0f` literal for the numeric type.
fn one(n: Numeric) -> String {
    fmt_lit(1.0, n)
}
/// `0.0` / `0.0f` literal for the numeric type.
fn zero(n: Numeric) -> String {
    fmt_lit(0.0, n)
}

/// Format an f64 as a C literal of the given numeric type (round-trip via
/// Rust's shortest repr; non-finite via `<math.h>` macros; `f` suffix for float).
pub(crate) fn fmt_lit(x: f64, n: Numeric) -> String {
    if let Numeric::Fixed { frac } = n {
        // Q-scaled integer literal, rounded to nearest, saturated to the
        // int32 range (a constant outside the format is a modelling error the
        // clamp keeps finite; NaN maps to 0). The original value rides along
        // as a comment so the generated C stays reviewable.
        let scaled = (x * (1i64 << frac) as f64).round();
        let q = if scaled.is_nan() {
            0i64
        } else {
            scaled.clamp(i32::MIN as f64, i32::MAX as f64) as i64
        };
        return format!("{q} /* {x:?} */");
    }
    if x.is_nan() {
        return "NAN".into();
    }
    if x.is_infinite() {
        return if x < 0.0 { "(-INFINITY)".into() } else { "INFINITY".into() };
    }
    let s = format!("{x:?}"); // shortest round-tripping decimal
    let base = if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{s}.0")
    };
    format!("{base}{}", n.suffix())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::schema::{NodeId, ParamId, Region, StateId};

    fn dbl() -> CodegenOptions {
        CodegenOptions::default()
    }

    #[test]
    fn amplifier_region_emits_c() {
        let region = Region {
            ops: vec![
                Op::Input { port: 0, elem: 0 },
                Op::Param { id: ParamId(0) },
                Op::Binary { op: BinOpKind::Mul, a: NodeId(0), b: NodeId(1) },
            ],
            writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(2) }],
        };
        let c = emit_region_fn("amp_alg", &region, &dbl(), &[], &[1]).unwrap();
        assert!(c.contains("const double v0 = u[0];"), "{c}");
        assert!(c.contains("const double v2 = (v0 * v1);"), "{c}");
        assert!(c.contains("out[0] = v2;"), "{c}");
    }

    #[test]
    fn float_numeric_suffixes_type_and_math() {
        // y = sin(u) with Numeric::Float → float type + sinf.
        let region = Region {
            ops: vec![
                Op::Input { port: 0, elem: 0 },
                Op::Unary { op: UnaryOpKind::Sin, a: NodeId(0) },
            ],
            writes: vec![Write::Output { port: 0, elem: 0, src: NodeId(1) }],
        };
        let opts = CodegenOptions { numeric: Numeric::Float, ..Default::default() };
        let c = emit_region_fn("f", &region, &opts, &[], &[1]).unwrap();
        assert!(c.contains("const float v0 = u[0];"), "{c}");
        assert!(c.contains("const float v1 = sinf(v0);"), "{c}");
        assert!(c.contains("(const float* u"), "{c}");
    }

    #[test]
    fn integrator_dyn_reads_state_and_input() {
        let region = Region {
            ops: vec![Op::Input { port: 0, elem: 0 }],
            writes: vec![Write::StateDeriv { id: StateId(0), src: NodeId(0) }],
        };
        let c = emit_region_fn("int_dyn", &region, &dbl(), &[], &[1]).unwrap();
        assert!(c.contains("const double v0 = u[0];"), "{c}");
        assert!(c.contains("out[0] = v0;"), "{c}");
    }

    #[test]
    fn reduce_and_dot_emit_unrolled() {
        let region = Region {
            ops: vec![
                Op::Input { port: 0, elem: 0 },
                Op::Input { port: 0, elem: 1 },
                Op::Reduce { op: ReduceKind::Sum, args: vec![NodeId(0), NodeId(1)] },
                Op::Dot { a: vec![NodeId(0), NodeId(1)], b: vec![NodeId(1), NodeId(0)] },
            ],
            writes: vec![
                Write::Output { port: 0, elem: 0, src: NodeId(2) },
                Write::Output { port: 0, elem: 1, src: NodeId(3) },
            ],
        };
        let c = emit_region_fn("r", &region, &dbl(), &[], &[2]).unwrap();
        assert!(c.contains("const double v2 = (v0 + v1);"), "{c}");
        assert!(c.contains("const double v3 = fma(v1, v0, fma(v0, v1, 0.0));"), "{c}");
    }

    #[test]
    fn solver_choice_by_name_accepts_explicit_rejects_implicit() {
        // Default is RK4 (explicit).
        assert_eq!(SolverChoice::default().tableau().name, "RK4");
        // Explicit fixed-step and adaptive tableaus resolve.
        assert_eq!(SolverChoice::by_name("RK4").unwrap().tableau().name, "RK4");
        assert!(SolverChoice::by_name("RKDP54").unwrap().tableau().is_adaptive());
        assert!(SolverChoice::by_name("SSPRK33").is_some());
        // Forward Euler via the codegen-only EUF tableau, case-insensitively.
        assert_eq!(SolverChoice::by_name("euf").unwrap().tableau().name, "EUF");
        assert_eq!(SolverChoice::EULER.tableau().name, "EUF");
        // Implicit tableaus are not yet emitted; unknown names are rejected.
        assert!(SolverChoice::by_name("ESDIRK43").is_none());
        assert!(SolverChoice::by_name("DIRK2").is_none());
        assert!(SolverChoice::by_name("NotASolver").is_none());
    }

    #[test]
    fn unsupported_op_reports_error() {
        // An opaque extern call has no static lowering (RNG/noise/DAE/Python).
        // (Digamma is now supported via the `fastsim_digamma` helper.)
        let region = Region {
            ops: vec![Op::Call { id: crate::ir::schema::ExternId(0), args: vec![], out_idx: 0 }],
            writes: vec![],
        };
        let err = emit_region_fn("d", &region, &dbl(), &[], &[]).unwrap_err();
        assert!(matches!(err, CodegenError::Unsupported(_)));
    }
}
