# fastsim IR (v1)

A hierarchical, serializable **snapshot** of a live `Simulation`, lowered to
scalar SSA ops per block. It is the stable, language-agnostic *interface* on
which downstream tools (code generation, verification, static compilation)
build. The IR never sits on the runtime hot path.

- **Source of truth:** the running model. The IR is derived from it, never the
  other way around. Build it with `sim.to_ir(name)` (Python) /
  `ir::builder::module_from_sim` (Rust).
- **Transport:** JSON only. There is no separate IR crate; consumers read the
  serialized `Module` or the typed `fastsim.ir` dataclasses. Round-trips are
  lossless and textually stable (see *Versioning*).
- **Granularity:** block boundaries are preserved by design (blocks are mutable,
  stateful runtime instances). Each block is *atomized* one level below pathsim's
  `op_alg`/`op_dyn` operators into scalar SSA ops.

```
Module
 ├─ root: Subsystem ──(recursive)──┐
 │   ├─ interface: Interface       │
 │   ├─ children: [Block | Subsystem]
 │   ├─ connections: [Connection]   ← source of truth for wiring
 │   └─ schedule: Schedule          ← derived/advisory (see below)
 ├─ events: [Event]                 ← simulation-level (global) events
 └─ extern_decls: [ExternDecl]      ← opaque Op::Call signatures
```

## Block

```
Block { id, name, type_name, role, ports, params, state, memory, regions, events }
```

- `role` ∈ `Algebraic | Dynamic | Source | Sink`.
- `ports`: input/output ports, each with a scalar element `size` (1 = SISO).
- `params`: runtime-mutable parameters, snapshotted by value. Kept as `Op::Param`
  nodes (not baked constants) so the IR is pathsim-faithful — a backend can emit
  them as live coefficients.
- `state`: continuous state variables (`init` values), integrated by a solver.
- `memory`: discrete persistent slots (sampled / event-driven blocks).
- `regions`: the block's math (below).
- `events`: block-internal events (below).

### Regions

Mirror pathsim's `op_alg` / `op_dyn`, one level lower (scalar SSA):

```
Regions { alg, dyn_ }
Region  { ops: [Op], writes: [Write] }
```

- `alg` — output equation `y = f(x, u, t, mem)`.
- `dyn_` — state derivative `dx/dt = g(x, u, t, mem)`.
- `ops` are SSA nodes: `NodeId(i)` refers to `ops[i]`, reads only. Effects happen
  through `writes`, applied in order *after* all `ops` execute.

### Op vocabulary

The compute ops share `jit::graph`'s op enums directly (the IR's `BinOpKind` /
`UnaryOpKind` / `CmpKind` / `ReduceKind` are re-exports of `jit::graph`'s
`BinOp` / `UnaryOp` / `CmpOp` / `ReduceOp`, one source of truth, no copy), plus
IR-level reads, the structured `Dot` / `Lut1d` ops, and an extern escape:

| Op | meaning |
|----|---------|
| `Const(f64)` | literal |
| `Time` | simulation time `t` |
| `Input { port, elem }` | input port element |
| `Param { id }` | block parameter |
| `State { id }` | continuous state element |
| `Memory { slot, offset }` | discrete memory element |
| `Binary { op, a, b }` | `Add Sub Mul Div Pow Mod Min Max Atan2 Hypot` |
| `Unary { op, a }` | `Neg Sin … Erf …` (see `UnaryOpKind`) |
| `Cmp { op, a, b }` | `Gt Ge Lt Le Eq Ne` → 1.0 / 0.0 |
| `Select { c, t, e }` | `c != 0 ? t : e` |
| `Fma { a, b, c }` | `a*b + c` |
| `Reduce { op, args }` | variadic fold over an operand list: `Sum Product Min Max` |
| `Dot { a, b }` | fused dot product `Σ aᵢ·bᵢ` over two equal-length lists |
| `Lut1d { input, points, values, clamp }` | 1-D piecewise-linear lookup table (inline breakpoints) |
| `Call { id, args, out_idx }` | opaque extern (RNG, FMU, callback); `id` → `ExternDecl` |

### Writes

```
Output { port, elem, src }        // alg regions + event effects
StateDeriv { id, src }            // dyn regions only
StateWrite { id, src }            // event effects (discrete state mutation)
MemoryWrite { slot, offset, src } // event effects
```

## Connections vs Schedule

**Connections are the source of truth** for dataflow and cannot be dropped:

```
Connection { id, src: PortRef, targets: [PortRef] }
PortRef    { block, port, elems: Option<[u32]> }
```

- `block == BlockId::INTERFACE` (`0xFFFFFFFF`) refers to the enclosing
  subsystem's interface ports.
- `elems = None` means the whole port; `Some([..])` is MIMO element slicing.

**The `Schedule` is derived/advisory** — everything in it is recomputable from
`connections` + block roles (that is exactly what `utils::schedule` does, and what
the builder reuses verbatim, so the IR schedule and the live evaluation order
can never drift). It exists only so a consumer need not reimplement the graph
analysis:

```
Schedule {
  topo: [BlockId],          // full linear order (DAG depths, then loop members)
  groups: [DagGroup],       // acyclic part, grouped by algebraic-feedthrough depth
  sccs: [Scc],              // algebraic loops (each solved by fixed-point iteration)
  back_edges: [ConnectionId]// deduped union of every SCC's cut set
}
```

Notes:

- "Depth" is **algebraic feedthrough depth**, not naive topological distance:
  dynamic/source blocks sit at depth 0 (their outputs depend on state/time, not
  the current-step inputs).
- Block ids in the schedule may include `BlockId::INTERFACE` (the interface
  forwards the subsystem's inputs and participates in the order).
- `groups` covers only the acyclic part; loop members live in `sccs`. `topo`
  covers everyone.

## Events

Three firing kinds, mirroring `events/` at runtime:

```
Event    { id, kind, effect: Region, opaque: bool }
EventKind = ZeroCross { guard: Region, direction } | Schedule { times } | Condition { guard }
ScheduleTimes = Fixed([f64]) | Periodic { period, phase }
```

- **Op-bearing events** (discrete blocks, relay, comparator): fully represented.
  The `effect` is a real op-graph; guards are op-graphs whose scalar value is the
  last op. `opaque = false`.
- **Opaque events** (`opaque = true`): the guard and/or action is host code (RNG
  draw, scope recording, arbitrary callback) not expressible as ops. The
  `effect` (and any guard `Region`) is empty; `kind` still carries the
  statically-known structure (Schedule timing, ZeroCross direction). This is how
  opaque blocks' sampling events and simulation-level events are surfaced — the
  event's *existence and timing* are honest, its *effect* is explicitly marked
  unmodeled rather than silently dropped.
- **`Module.events`**: simulation-level (global) events not attached to a block.
  Always opaque (host guards/actions).

## Externs

```
ExternDecl { id, name, arity_in, arity_out }
```

Opaque blocks (Scope, RNG, FMU, arbitrary Python callables, DAE) are represented
honestly: their `alg` region reads its inputs and emits one `Op::Call` per output
element referencing an `ExternDecl`. The IR records that the block exists with
its arity, but makes no claim about its math. A backend that cannot supply the
extern flags the block as non-lowerable rather than failing the whole module.

## IDs and sentinels

All ids are `u32` index newtypes (`BlockId`, `ConnectionId`, `StateId`, …),
serialized as bare integers, so a `Module` round-trips losslessly. The only
sentinel is `BlockId::INTERFACE = u32::MAX`.

## Versioning

- `Module.ir_version` (currently **1**) is the schema version.
- A committed golden module (`tests/golden/`) pins the exact JSON. The golden
  test (`tests/test_ir_golden.rs`) fails on any schema change, forcing a
  deliberate `IR_VERSION` bump + golden refresh (`UPDATE_IR_GOLDEN=1 cargo test`).
- Serialization is field-order deterministic and uses `skip_serializing_if` for
  empty collections / default flags, so absent JSON keys mean "default", and
  re-serializing a parsed module is byte-identical.

## Verification

`ir::eval` is a reference interpreter over a `Region` (`eval_region(region, ctx)`).
Every op-bearing block has a test asserting `graph → Region → eval` equals the
native runtime closure (`f_alg` / `f_dyn`) to 1e-12 — the IR is provably faithful
for the entire op-expressible block library.

## Python surface

`fastsim.ir` provides typed dataclasses mirroring this schema 1:1
(`Module`, `Subsystem`, `Block`, `Region`, `Op` via `Tagged(kind, fields)`, …),
with `Module.from_json` / `to_json` (lossless), recursive `blocks()`, `find()`,
`extern_blocks()`, and `summary()`. Build it with `sim.to_ir(name)`.
