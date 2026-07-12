//! Op manifest: the SSA op vocabulary and its canonical semantics in one place.
//!
//! The four op enums (`BinOp`/`UnaryOp`/`CmpOp`/`ReduceOp`) ARE the IR's op
//! kinds (re-exported by `ir::schema`) and drive the flat-tape opcodes. This
//! module is their single home: the enum definitions, the canonical f64
//! semantics (`apply_*`), the stateless PRNG / digamma kernels, the tape opcode
//! numbering (`code`), and the `Node`-kind to opcode mapping. `graph` re-exports
//! the enums + `apply_*` for back-compat; `tape` aliases `code` as `op` and uses
//! the `*_opcode` mappers. The tape's hot-loop arithmetic stays inlined in
//! `tape.rs`, pinned bit-exact to `apply_*` by the differential fuzzer.

use crate::constants::JIT_FLOAT_EQ_TOL;

/// Binary operations.
///
/// This is also the IR's `schema::BinOpKind` (re-exported there): the jit graph
/// and the serializable IR share one op vocabulary, so the enum carries serde
/// derives and lives here as the single source of truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BinOp {
    Add, Sub, Mul, Div, Pow, Mod, Min, Max, Atan2, Hypot,
}

/// Unary operations. Also the IR's `schema::UnaryOpKind` (see `BinOp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum UnaryOp {
    Neg, Sin, Cos, Tan, Atan, Sinh, Cosh, Tanh,
    Exp, Log, Log10, Abs, Sqrt, Sign, Floor,
    Asin, Acos, Asinh, Acosh, Atanh,
    Ceil, Round, Trunc,
    Log2, Log1p, Expm1, Cbrt,
    Erf, Erfc, Lgamma, Tgamma,
    /// Digamma ψ(x) = d/dx lgamma(x). Used only as derivative of gamma/lgamma;
    /// numpy has no direct digamma ufunc (scipy.special.digamma exists but is
    /// outside our ufunc route). Kept as an internal op.
    Digamma,
    /// Stateless counter-based PRNG: maps the operand's bit pattern through a
    /// splitmix64 finalizer to a uniform value in `[0, 1)`. `rand_uniform(key)`
    /// is a *pure function* of its argument, so noise sources trace, replay, and
    /// (unlike `np.random`) stay reproducible. Derivative is 0 (see `autodiff`).
    /// Exposed to Python as `fastsim.random_uniform` / `fastsim.random_normal`.
    RandUniform,
}

/// Digamma function ψ(x) via recursion into the asymptotic regime + Bernoulli
/// expansion. Accurate to ~1e-12 for x > 0 and reasonable negative x. Not
/// exposed as a numpy ufunc — it is emitted by the autodiff pass as the
/// derivative of `lgamma`/`tgamma`.
pub fn digamma(mut x: f64) -> f64 {
    // Recursion ψ(x) = ψ(x+1) - 1/x pushes x into the asymptotic zone.
    let mut result = 0.0;
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic: ψ(x) ≈ ln x - 1/(2x) - 1/(12x²) + 1/(120x⁴) - 1/(252x⁶) + ...
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result += x.ln() - 0.5 * inv
        - inv2 * (1.0 / 12.0
                  - inv2 * (1.0 / 120.0
                            - inv2 * (1.0 / 252.0)));
    result
}

/// Stateless counter-based PRNG (the `UnaryOp::RandUniform` kernel). Maps the
/// key's IEEE bit pattern through a splitmix64 finalizer and takes the top 53
/// bits as a uniform double in `[0, 1)`. Deterministic and identical across the
/// interpreter, the tape and (eventually) codegen, so a model replays bit-for-
/// bit. For step-wise noise the caller floors the key (e.g. `floor(t/dt)`); a
/// continuously varying key gives fresh noise at every distinct float.
#[inline]
pub fn rand_uniform(key: f64) -> f64 {
    let mut z = key.to_bits().wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Top 53 bits → exact integer in [0, 2^53) → [0, 1).
    ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

/// Comparison operations (result is 0.0 or 1.0). Also the IR's
/// `schema::CmpKind` (see `BinOp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum CmpOp {
    Gt, Ge, Lt, Le, Eq, Ne,
}

/// Variadic reduction operations over a list of operands. The single
/// structured op in an otherwise scalar IR: it collapses an N-element fold
/// (a `sum`, a dot-product accumulation, a `min`/`max` over a vector) into
/// one node instead of an N-deep tree of scalar `Binary` ops. This keeps the
/// trace graph small (one node per `np.sum`/`np.dot`, not O(N)), gives the
/// tape a tight inner loop, and lets codegen emit a `for` loop rather than
/// re-discovering vector structure from unrolled scalars.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ReduceOp {
    Sum, Product, Min, Max,
}

impl ReduceOp {
    /// Identity element: the reduction of an empty operand list.
    pub fn identity(self) -> f64 {
        match self {
            ReduceOp::Sum => 0.0,
            ReduceOp::Product => 1.0,
            ReduceOp::Min => f64::INFINITY,
            ReduceOp::Max => f64::NEG_INFINITY,
        }
    }
    /// Combine the running accumulator with the next operand.
    #[inline]
    pub fn combine(self, acc: f64, v: f64) -> f64 {
        match self {
            ReduceOp::Sum => acc + v,
            ReduceOp::Product => acc * v,
            ReduceOp::Min => acc.min(v),
            ReduceOp::Max => acc.max(v),
        }
    }
}

/// Canonical f64 binary semantics. The single source of truth shared by the
/// recursive `interpret`, the IR reference evaluator, and constant folding (the
/// `Tape` interpreter keeps its own inlined copy for the hot loop). The native
/// `F64Builder` is locked to this by test.
#[inline]
pub fn apply_binary(op: BinOp, a: f64, b: f64) -> f64 {
    match op {
        BinOp::Add => a + b, BinOp::Sub => a - b,
        BinOp::Mul => a * b, BinOp::Div => a / b,
        BinOp::Pow => a.powf(b), BinOp::Mod => a % b,
        BinOp::Min => a.min(b), BinOp::Max => a.max(b),
        BinOp::Atan2 => a.atan2(b), BinOp::Hypot => a.hypot(b),
    }
}

/// Canonical f64 comparison semantics (result `1.0`/`0.0`). `Eq`/`Ne` use the
/// `JIT_FLOAT_EQ_TOL` band. See [`apply_binary`] for the sharing contract.
#[inline]
pub fn apply_cmp(op: CmpOp, a: f64, b: f64) -> f64 {
    let cond = match op {
        CmpOp::Gt => a > b, CmpOp::Ge => a >= b,
        CmpOp::Lt => a < b, CmpOp::Le => a <= b,
        CmpOp::Eq => (a - b).abs() < JIT_FLOAT_EQ_TOL,
        CmpOp::Ne => (a - b).abs() >= JIT_FLOAT_EQ_TOL,
    };
    if cond { 1.0 } else { 0.0 }
}

/// numpy `sign` semantics: `1.0` for positive, `-1.0` for negative; zeros and
/// NaN pass through unchanged (exactly numpy's kernel
/// `in > 0 ? 1 : (in < 0 ? -1 : in)`). The single definition every backend
/// (recursive interpreter, flat tape, C emission) mirrors.
#[inline]
pub fn numpy_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x // ±0.0 and NaN pass through
    }
}

/// Canonical f64 unary semantics. See [`apply_binary`] for the sharing contract.
#[inline]
pub fn apply_unary(op: UnaryOp, va: f64) -> f64 {
    match op {
        UnaryOp::Neg => -va, UnaryOp::Sin => va.sin(),
        UnaryOp::Cos => va.cos(), UnaryOp::Tan => va.tan(),
        UnaryOp::Atan => va.atan(), UnaryOp::Sinh => va.sinh(),
        UnaryOp::Cosh => va.cosh(), UnaryOp::Tanh => va.tanh(),
        UnaryOp::Exp => va.exp(), UnaryOp::Log => va.ln(),
        UnaryOp::Log10 => va.log10(), UnaryOp::Abs => va.abs(),
        UnaryOp::Sqrt => va.sqrt(),
        // numpy `sign` semantics (the tracer maps `np.sign` here): 0 at ±0,
        // NaN for NaN — NOT Rust's `signum` (±1, never 0). Pinned by the
        // tape/codegen parity tests; keep all three backends in sync.
        UnaryOp::Sign => numpy_sign(va),
        UnaryOp::Floor => va.floor(),
        UnaryOp::Asin => va.asin(), UnaryOp::Acos => va.acos(),
        UnaryOp::Asinh => va.asinh(), UnaryOp::Acosh => va.acosh(),
        UnaryOp::Atanh => va.atanh(),
        UnaryOp::Ceil => va.ceil(), UnaryOp::Round => va.round(),
        UnaryOp::Trunc => va.trunc(),
        UnaryOp::Log2 => va.log2(), UnaryOp::Log1p => va.ln_1p(),
        UnaryOp::Expm1 => va.exp_m1(), UnaryOp::Cbrt => va.cbrt(),
        UnaryOp::Erf => libm::erf(va), UnaryOp::Erfc => libm::erfc(va),
        UnaryOp::Lgamma => libm::lgamma(va), UnaryOp::Tgamma => libm::tgamma(va),
        UnaryOp::Digamma => digamma(va),
        UnaryOp::RandUniform => rand_uniform(va),
    }
}

// =====================================================================================
// Flat-tape opcode numbering + Node-kind -> opcode mapping
// =====================================================================================

pub mod code {
    pub const CONST: u8 = 0;
    pub const INPUT: u8 = 1;     // arg0=slot, arg1=element
    pub const PARAM: u8 = 2;     // arg0=param index
    pub const ADD: u8 = 3;   pub const SUB: u8 = 4;
    pub const MUL: u8 = 5;   pub const DIV: u8 = 6;
    pub const POW: u8 = 7;   pub const MOD: u8 = 8;
    pub const MIN: u8 = 9;   pub const MAX: u8 = 10;
    pub const ATAN2: u8 = 11; pub const HYPOT: u8 = 12;
    pub const NEG: u8 = 20;  pub const SIN: u8 = 21;
    pub const COS: u8 = 22;  pub const TAN: u8 = 23;
    pub const ATAN: u8 = 24; pub const SINH: u8 = 25;
    pub const COSH: u8 = 26; pub const TANH: u8 = 27;
    pub const EXP: u8 = 28;  pub const LOG: u8 = 29;
    pub const LOG10: u8 = 30; pub const ABS: u8 = 31;
    pub const SQRT: u8 = 32; pub const SIGN: u8 = 33;
    pub const FLOOR: u8 = 34;
    pub const ASIN: u8 = 35; pub const ACOS: u8 = 36;
    pub const ASINH: u8 = 37; pub const ACOSH: u8 = 38;
    pub const ATANH: u8 = 39;
    pub const CEIL: u8 = 40; pub const ROUND: u8 = 41;
    pub const TRUNC: u8 = 42;
    pub const LOG2: u8 = 43; pub const LOG1P: u8 = 44;
    pub const EXPM1: u8 = 45; pub const CBRT: u8 = 46;
    pub const ERF: u8 = 47;  pub const ERFC: u8 = 48;
    pub const LGAMMA: u8 = 49; pub const TGAMMA: u8 = 50;
    pub const DIGAMMA: u8 = 51;
    pub const RAND_UNIFORM: u8 = 52;
    pub const CMP_GT: u8 = 60; pub const CMP_GE: u8 = 61;
    pub const CMP_LT: u8 = 62; pub const CMP_LE: u8 = 63;
    pub const CMP_EQ: u8 = 64; pub const CMP_NE: u8 = 65;
    pub const SELECT: u8 = 70;
    pub const FMA: u8 = 80;
    /// Variadic reduction. `arg0` = reduce-op code (see `reduce`), `arg1` =
    /// start offset into the tape's `arg_pool`, `arg2` = operand count.
    pub const REDUCE: u8 = 90;
    /// Fused dot product. `arg0` = term count `k`, `arg1` = `arg_pool` offset of
    /// the a-list (the b-list follows immediately at `arg1 + k`).
    pub const DOT: u8 = 100;

    pub mod reduce {
        pub const SUM: u32 = 0;
        pub const PRODUCT: u32 = 1;
        pub const MIN: u32 = 2;
        pub const MAX: u32 = 3;
    }
}

/// `BinOp` -> tape opcode.
pub fn binary_opcode(op: BinOp) -> u8 {
    match op {
                        BinOp::Add => code::ADD, BinOp::Sub => code::SUB,
                        BinOp::Mul => code::MUL, BinOp::Div => code::DIV,
                        BinOp::Pow => code::POW, BinOp::Mod => code::MOD,
                        BinOp::Min => code::MIN, BinOp::Max => code::MAX,
                        BinOp::Atan2 => code::ATAN2, BinOp::Hypot => code::HYPOT,
    }
}

/// `UnaryOp` -> tape opcode.
pub fn unary_opcode(op: UnaryOp) -> u8 {
    match op {
                        UnaryOp::Neg => code::NEG, UnaryOp::Sin => code::SIN,
                        UnaryOp::Cos => code::COS, UnaryOp::Tan => code::TAN,
                        UnaryOp::Atan => code::ATAN, UnaryOp::Sinh => code::SINH,
                        UnaryOp::Cosh => code::COSH, UnaryOp::Tanh => code::TANH,
                        UnaryOp::Exp => code::EXP, UnaryOp::Log => code::LOG,
                        UnaryOp::Log10 => code::LOG10, UnaryOp::Abs => code::ABS,
                        UnaryOp::Sqrt => code::SQRT, UnaryOp::Sign => code::SIGN,
                        UnaryOp::Floor => code::FLOOR,
                        UnaryOp::Asin => code::ASIN, UnaryOp::Acos => code::ACOS,
                        UnaryOp::Asinh => code::ASINH, UnaryOp::Acosh => code::ACOSH,
                        UnaryOp::Atanh => code::ATANH,
                        UnaryOp::Ceil => code::CEIL, UnaryOp::Round => code::ROUND,
                        UnaryOp::Trunc => code::TRUNC,
                        UnaryOp::Log2 => code::LOG2, UnaryOp::Log1p => code::LOG1P,
                        UnaryOp::Expm1 => code::EXPM1, UnaryOp::Cbrt => code::CBRT,
                        UnaryOp::Erf => code::ERF, UnaryOp::Erfc => code::ERFC,
                        UnaryOp::Lgamma => code::LGAMMA, UnaryOp::Tgamma => code::TGAMMA,
                        UnaryOp::Digamma => code::DIGAMMA,
                        UnaryOp::RandUniform => code::RAND_UNIFORM,
    }
}

/// `CmpOp` -> tape opcode.
pub fn cmp_opcode(op: CmpOp) -> u8 {
    match op {
                        CmpOp::Gt => code::CMP_GT, CmpOp::Ge => code::CMP_GE,
                        CmpOp::Lt => code::CMP_LT, CmpOp::Le => code::CMP_LE,
                        CmpOp::Eq => code::CMP_EQ, CmpOp::Ne => code::CMP_NE,
    }
}

/// `ReduceOp` -> tape reduction sub-code (see `code::reduce`).
pub fn reduce_code(op: ReduceOp) -> u32 {
    match op {
        ReduceOp::Sum => code::reduce::SUM,
        ReduceOp::Product => code::reduce::PRODUCT,
        ReduceOp::Min => code::reduce::MIN,
        ReduceOp::Max => code::reduce::MAX,
    }
}

// =====================================================================================
// Codegen C/libm mapping
// =====================================================================================
//
// The C math-function name per op, so the codegen `Target` is a thin backend:
// it emits `f(args)` from the name and only hand-writes the structural forms
// (infix operators, comparisons, and the few ops with no plain libm twin).

/// libm function name for a `BinOp` emitted as `f(a, b)`, or `None` for ops
/// codegen emits as an infix operator (`+ - * /`).
pub fn binary_c_fn(op: BinOp) -> Option<&'static str> {
    match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => None,
        BinOp::Pow => Some("pow"),
        BinOp::Mod => Some("fmod"),
        BinOp::Min => Some("fmin"),
        BinOp::Max => Some("fmax"),
        BinOp::Atan2 => Some("atan2"),
        BinOp::Hypot => Some("hypot"),
    }
}

/// libm function name for a `UnaryOp` emitted as `f(a)`, or `None` for ops
/// codegen emits specially: `Neg` as `-a`, `Sign` via `copysign`, `RandUniform`
/// via the emitted PRNG helper, `Digamma` unsupported in C (gamma-derivative
/// only). Note the C names differ from the Rust method names for a few ops
/// (`Abs` -> `fabs`, `Log` -> `log`, `Log1p` -> `log1p`, `Expm1` -> `expm1`).
pub fn unary_c_fn(op: UnaryOp) -> Option<&'static str> {
    match op {
        UnaryOp::Neg | UnaryOp::Sign | UnaryOp::Digamma | UnaryOp::RandUniform => None,
        UnaryOp::Sin => Some("sin"),
        UnaryOp::Cos => Some("cos"),
        UnaryOp::Tan => Some("tan"),
        UnaryOp::Atan => Some("atan"),
        UnaryOp::Sinh => Some("sinh"),
        UnaryOp::Cosh => Some("cosh"),
        UnaryOp::Tanh => Some("tanh"),
        UnaryOp::Exp => Some("exp"),
        UnaryOp::Log => Some("log"),
        UnaryOp::Log10 => Some("log10"),
        UnaryOp::Abs => Some("fabs"),
        UnaryOp::Sqrt => Some("sqrt"),
        UnaryOp::Floor => Some("floor"),
        UnaryOp::Asin => Some("asin"),
        UnaryOp::Acos => Some("acos"),
        UnaryOp::Asinh => Some("asinh"),
        UnaryOp::Acosh => Some("acosh"),
        UnaryOp::Atanh => Some("atanh"),
        UnaryOp::Ceil => Some("ceil"),
        UnaryOp::Round => Some("round"),
        UnaryOp::Trunc => Some("trunc"),
        UnaryOp::Log2 => Some("log2"),
        UnaryOp::Log1p => Some("log1p"),
        UnaryOp::Expm1 => Some("expm1"),
        UnaryOp::Cbrt => Some("cbrt"),
        UnaryOp::Erf => Some("erf"),
        UnaryOp::Erfc => Some("erfc"),
        UnaryOp::Lgamma => Some("lgamma"),
        UnaryOp::Tgamma => Some("tgamma"),
    }
}

// =====================================================================================
// Canonical vector reductions (4-lane multi-accumulator)
// =====================================================================================
//
// The fused `Dot` and variadic `Reduce` ops evaluate through these, so the
// native `F64Builder`, the interpreter, and the flat tape all agree bit-for-bit.
// Four independent accumulators break the reduction's dependency chain so the
// compiler can pipeline / pack the body (SSE2 is x86-64 baseline, so this widens
// without an `+fma` target feature). We use plain `a*b + acc`, NOT `f64::mul_add`:
// without a hardware FMA target feature `mul_add` lowers to a libm `fma()` CALL
// per element, far slower than two native instructions. The four-lane split
// reassociates the sum, so a result can differ from a serial fold by a few ULPs
// (within the parity tolerance, the tradeoff every BLAS makes); tails of length
// < 4 stay on the serial fold.

/// Canonical fused dot product `Σ aᵢ·bᵢ` (4-lane plain multiply-add).
///
/// On x86-64 with runtime AVX2 this dispatches to a 256-bit kernel; everywhere
/// else (and on non-AVX2 CPUs) it runs the portable 4-lane scalar fold. The two
/// are bit-for-bit identical by construction (same 4-lane grouping, same `mul`
/// then `add` with two roundings, same horizontal merge), so the parity the
/// interpreter / tape / native builder rely on is preserved. Wheels stay
/// portable: AVX2 is chosen at runtime, never at compile time.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // Safety: gated on runtime AVX2 detection.
            return unsafe { dot_avx2(a, b) };
        }
    }
    dot_scalar(a, b)
}

/// Portable 4-lane scalar fold. Four independent accumulators break the
/// dependency chain so the compiler packs the body (SSE2 baseline). Plain
/// `a*b + acc` (two roundings), NOT `mul_add`: without a hardware FMA target
/// feature `mul_add` lowers to a libm `fma()` call per element.
#[inline]
fn dot_scalar(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut acc = [0.0f64; 4];
    let mut i = 0;
    while i + 4 <= n {
        acc[0] += a[i] * b[i];
        acc[1] += a[i + 1] * b[i + 1];
        acc[2] += a[i + 2] * b[i + 2];
        acc[3] += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

/// AVX2 dot: one 256-bit accumulator over chunks of 4. Lane `j` accumulates the
/// same elements (`j, j+4, ...`) as `dot_scalar`'s `acc[j]`, with a separate
/// `_mm256_mul_pd` then `_mm256_add_pd` (two roundings, NOT `fmadd`), and the
/// identical `(l0+l1)+(l2+l3)` horizontal merge and scalar tail. So it returns
/// bit-for-bit what `dot_scalar` does, just 4-wide instead of relying on SSE2
/// auto-vectorisation of the four scalar accumulators.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let mut acc = _mm256_setzero_pd();
    let mut i = 0;
    while i + 4 <= n {
        let va = _mm256_loadu_pd(a.as_ptr().add(i));
        let vb = _mm256_loadu_pd(b.as_ptr().add(i));
        acc = _mm256_add_pd(acc, _mm256_mul_pd(va, vb));
        i += 4;
    }
    let mut lanes = [0.0f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), acc);
    let mut s = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

/// Canonical variadic reduction. Sum/Product use a 4-lane multi-accumulator
/// (reassociated, ULP-different from a serial fold); Min/Max are exact either
/// way but also widen for shorter latency.
///
/// On x86-64 with runtime AVX2, Sum and Product dispatch to bit-identical
/// 256-bit kernels (same lane grouping and merge as the scalar fold). Min/Max
/// stay on the scalar path: `_mm256_min_pd`/`_mm256_max_pd` differ from
/// `f64::min`/`max` on NaN operands, which would break the bit-exact parity the
/// fuzzer pins.
#[inline]
pub fn reduce(op: ReduceOp, xs: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            match op {
                // Safety: gated on runtime AVX2 detection.
                ReduceOp::Sum => return unsafe { reduce_sum_avx2(xs) },
                ReduceOp::Product => return unsafe { reduce_prod_avx2(xs) },
                ReduceOp::Min | ReduceOp::Max => {}
            }
        }
    }
    reduce_scalar(op, xs)
}

/// Portable 4-lane scalar reduction (see [`reduce`]).
#[inline]
fn reduce_scalar(op: ReduceOp, xs: &[f64]) -> f64 {
    let n = xs.len();
    macro_rules! lanes4 {
        ($init:expr, $comb:expr, $merge:expr) => {{
            let mut acc = [$init; 4];
            let mut i = 0;
            while i + 4 <= n {
                acc[0] = $comb(acc[0], xs[i]);
                acc[1] = $comb(acc[1], xs[i + 1]);
                acc[2] = $comb(acc[2], xs[i + 2]);
                acc[3] = $comb(acc[3], xs[i + 3]);
                i += 4;
            }
            let mut s = $merge($merge(acc[0], acc[1]), $merge(acc[2], acc[3]));
            while i < n {
                s = $comb(s, xs[i]);
                i += 1;
            }
            s
        }};
    }
    match op {
        ReduceOp::Sum => lanes4!(0.0f64, |a: f64, b: f64| a + b, |a: f64, b: f64| a + b),
        ReduceOp::Product => lanes4!(1.0f64, |a: f64, b: f64| a * b, |a: f64, b: f64| a * b),
        ReduceOp::Min => lanes4!(f64::INFINITY, |a: f64, b: f64| a.min(b), |a: f64, b: f64| a.min(b)),
        ReduceOp::Max => lanes4!(f64::NEG_INFINITY, |a: f64, b: f64| a.max(b), |a: f64, b: f64| a.max(b)),
    }
}

/// AVX2 sum: one 256-bit accumulator over chunks of 4, bit-identical to the
/// scalar `Sum` fold (same lane grouping, `(l0+l1)+(l2+l3)` merge, scalar tail).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn reduce_sum_avx2(xs: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    let n = xs.len();
    let mut acc = _mm256_setzero_pd();
    let mut i = 0;
    while i + 4 <= n {
        acc = _mm256_add_pd(acc, _mm256_loadu_pd(xs.as_ptr().add(i)));
        i += 4;
    }
    let mut lanes = [0.0f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), acc);
    let mut s = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    while i < n {
        s += xs[i];
        i += 1;
    }
    s
}

/// AVX2 product: one 256-bit accumulator over chunks of 4, bit-identical to the
/// scalar `Product` fold (lanes init to 1.0, `(l0*l1)*(l2*l3)` merge, scalar tail).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn reduce_prod_avx2(xs: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    let n = xs.len();
    let mut acc = _mm256_set1_pd(1.0);
    let mut i = 0;
    while i + 4 <= n {
        acc = _mm256_mul_pd(acc, _mm256_loadu_pd(xs.as_ptr().add(i)));
        i += 4;
    }
    let mut lanes = [0.0f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), acc);
    let mut s = (lanes[0] * lanes[1]) * (lanes[2] * lanes[3]);
    while i < n {
        s *= xs[i];
        i += 1;
    }
    s
}

#[cfg(all(test, target_arch = "x86_64"))]
mod simd_parity {
    use super::*;

    /// xorshift64* PRNG with occasional special values so NaN/inf/zeros are hit.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn val(&mut self) -> f64 {
            match self.next() % 16 {
                0 => 0.0,
                1 => -0.0,
                2 => f64::NAN,
                3 => f64::INFINITY,
                4 => f64::NEG_INFINITY,
                _ => {
                    let u = (self.next() as f64) / (u64::MAX as f64);
                    (u * 2.0 - 1.0) * 1e3
                }
            }
        }
    }

    fn bits_eq(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    /// The AVX2 dot/sum/product kernels must return bit-for-bit what the scalar
    /// folds return, for every length 0..=33 and across special values. This is
    /// what lets the runtime dispatch in `dot`/`reduce` stay invisible to the
    /// interpret-vs-tape parity fuzzer and the golden trajectories.
    #[test]
    fn avx2_kernels_match_scalar_bit_exact() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 not available on this CPU; skipping parity check");
            return;
        }
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        for len in 0..=33usize {
            for _ in 0..64 {
                let a: Vec<f64> = (0..len).map(|_| rng.val()).collect();
                let b: Vec<f64> = (0..len).map(|_| rng.val()).collect();

                let d_s = dot_scalar(&a, &b);
                let d_v = unsafe { dot_avx2(&a, &b) };
                assert!(bits_eq(d_s, d_v), "dot len {len}: scalar {d_s:?} != avx2 {d_v:?}");

                let s_s = reduce_scalar(ReduceOp::Sum, &a);
                let s_v = unsafe { reduce_sum_avx2(&a) };
                assert!(bits_eq(s_s, s_v), "sum len {len}: scalar {s_s:?} != avx2 {s_v:?}");

                let p_s = reduce_scalar(ReduceOp::Product, &a);
                let p_v = unsafe { reduce_prod_avx2(&a) };
                assert!(bits_eq(p_s, p_v), "product len {len}: scalar {p_s:?} != avx2 {p_v:?}");
            }
        }
    }
}
