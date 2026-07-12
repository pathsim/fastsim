//! `src/ir/` — Hierarchical, serializable model IR derived from a live
//! `Simulation`.
//!
//! The IR is a point-in-time snapshot of the running model, lowered to scalar
//! SSA ops per block (the lowest layer). It is the substrate for code
//! generation and verification, and never sits on the live runtime hot path.
//!
//! The IR is the stable *interface*: code generators are downstream consumers
//! that live in their own module/crate and build on the serialized `Module`
//! (or the `schema` types) — they are deliberately NOT part of fastsim.
//!
//! Layers:
//!   - `schema`  — the pure data model (`Module`, `Subsystem`, `Block`, ...).
//!   - `builder` — walks a `Simulation` and produces a `Module`.
//!   - `eval`    — reference interpreter over a `Module` (verification basis).
//!
//! `Block`/`Connection`/`Port` here are the IR's own data types and are
//! intentionally NOT glob-re-exported, to avoid clashing with the runtime
//! `blocks::block::Block`, `connection::Connection`, and
//! `utils::portreference::Port`. Refer to them as `ir::schema::Block`, etc.

pub mod schema;
pub mod eval;
pub mod builder;
