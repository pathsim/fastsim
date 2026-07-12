"""Hierarchical model IR — the first-class Python view of fastsim's intermediate
representation.

A :class:`Module` is a serializable, language-agnostic snapshot of an assembled
``Simulation``: ``Module -> Subsystem (recursive) -> Block -> Regions{alg, dyn}
-> Region{ops, writes}``. Each block is either lowered to its scalar SSA op-graph
(for code generation / verification) or recorded as a typed ``extern`` call.

Get one from a simulation::

    sim = fastsim.Simulation(blocks, connections)
    ir = sim.to_ir()                # -> fastsim.ir.Module
    print(ir.summary())
    ir.to_json_file("model.json")

The Rust core produces the IR (it owns the block op-graphs); this module gives it
a typed, inspectable Python surface. The leaf sum types (``Op``, ``Write``,
``EventKind``, ``ParamValue``) are kept as light tagged objects since they are
rarely hand-built in Python; the container types are full dataclasses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from typing import Any, Iterator, Optional, Union

# ----------------------------------------------------------------------------
# Leaf sum types (serde external tagging: "Unit" | {"Newtype": v} | {"Struct": {...}})
# ----------------------------------------------------------------------------


@dataclass
class Tagged:
    """A serde-tagged enum value: a ``kind`` plus its payload fields.

    Unit variants (e.g. ``Op::Time``) carry an empty ``fields``; newtype variants
    (e.g. ``Op::Const(1.5)``) expose the value under ``fields["value"]``; struct
    variants (e.g. ``Op::Binary{op,a,b}``) expose their named fields directly.
    """

    kind: str
    fields: dict = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.fields[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __repr__(self) -> str:
        if not self.fields:
            return f"{self.kind}"
        inner = ", ".join(f"{k}={v!r}" for k, v in self.fields.items())
        return f"{self.kind}({inner})"

    @classmethod
    def parse(cls, v: Any) -> "Tagged":
        if isinstance(v, str):
            return cls(v, {})
        if isinstance(v, dict) and len(v) == 1:
            (k, payload), = v.items()
            if isinstance(payload, dict):
                return cls(k, dict(payload))
            return cls(k, {"value": payload})
        raise ValueError(f"not a tagged enum value: {v!r}")


# Op / Write / EventKind / ParamValue are all serde-tagged -> Tagged.
Op = Tagged
Write = Tagged
EventKind = Tagged
ParamValue = Tagged


# ----------------------------------------------------------------------------
# Container types (full dataclasses)
# ----------------------------------------------------------------------------


@dataclass
class Port:
    name: str
    size: int = 1

    @classmethod
    def parse(cls, d: dict) -> "Port":
        return cls(d["name"], d.get("size", 1))


@dataclass
class Ports:
    inputs: list[Port] = field(default_factory=list)
    outputs: list[Port] = field(default_factory=list)

    @classmethod
    def parse(cls, d: dict) -> "Ports":
        return cls(
            [Port.parse(p) for p in d.get("inputs", [])],
            [Port.parse(p) for p in d.get("outputs", [])],
        )


@dataclass
class Param:
    id: int
    name: str
    value: ParamValue

    @classmethod
    def parse(cls, d: dict) -> "Param":
        return cls(d["id"], d["name"], Tagged.parse(d["value"]))


@dataclass
class StateVar:
    id: int
    name: str
    init: float

    @classmethod
    def parse(cls, d: dict) -> "StateVar":
        return cls(d["id"], d["name"], d["init"])


@dataclass
class MemorySlot:
    id: int
    name: str
    size: int
    init: list[float]

    @classmethod
    def parse(cls, d: dict) -> "MemorySlot":
        return cls(d["id"], d["name"], d["size"], list(d.get("init", [])))


@dataclass
class Region:
    ops: list[Op] = field(default_factory=list)
    writes: list[Write] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.ops and not self.writes

    @classmethod
    def parse(cls, d: dict) -> "Region":
        return cls(
            [Tagged.parse(o) for o in d.get("ops", [])],
            [Tagged.parse(w) for w in d.get("writes", [])],
        )


@dataclass
class Regions:
    alg: Region = field(default_factory=Region)
    dyn: Region = field(default_factory=Region)

    @classmethod
    def parse(cls, d: dict) -> "Regions":
        # serde field is `dyn_` (dyn is a Rust keyword).
        return cls(Region.parse(d.get("alg", {})), Region.parse(d.get("dyn_", {})))


@dataclass
class Event:
    id: int
    kind: EventKind
    effect: Region = field(default_factory=Region)
    opaque: bool = False
    """True when the guard/action is host code (RNG, scope recording, callback)
    not expressible as ops. ``effect`` (and any guard region in ``kind``) is then
    empty/advisory; ``kind`` still carries the known structure (Schedule timing,
    ZeroCross direction)."""

    @classmethod
    def parse(cls, d: dict) -> "Event":
        return cls(
            d["id"],
            Tagged.parse(d["kind"]),
            Region.parse(d.get("effect", {})),
            bool(d.get("opaque", False)),
        )


@dataclass
class Block:
    id: int
    name: str
    type_name: str
    role: str  # "Algebraic" | "Dynamic" | "Source" | "Sink"
    ports: Ports
    params: list[Param] = field(default_factory=list)
    state: list[StateVar] = field(default_factory=list)
    memory: list[MemorySlot] = field(default_factory=list)
    regions: Regions = field(default_factory=Regions)
    events: list[Event] = field(default_factory=list)

    @property
    def is_extern(self) -> bool:
        """True if the block is represented as an opaque extern call (its alg
        region contains a ``Call`` op rather than a lowered op-graph)."""
        return any(o.kind == "Call" for o in self.regions.alg.ops)

    @classmethod
    def parse(cls, d: dict) -> "Block":
        return cls(
            id=d["id"],
            name=d["name"],
            type_name=d["type_name"],
            role=d["role"],
            ports=Ports.parse(d.get("ports", {})),
            params=[Param.parse(p) for p in d.get("params", [])],
            state=[StateVar.parse(s) for s in d.get("state", [])],
            memory=[MemorySlot.parse(m) for m in d.get("memory", [])],
            regions=Regions.parse(d.get("regions", {})),
            events=[Event.parse(e) for e in d.get("events", [])],
        )


@dataclass
class PortRef:
    block: int  # 0xFFFFFFFF (u32::MAX) == enclosing subsystem interface
    port: int = 0
    elems: Optional[list[int]] = None

    INTERFACE = 0xFFFFFFFF

    @property
    def is_interface(self) -> bool:
        return self.block == PortRef.INTERFACE

    @classmethod
    def parse(cls, d: dict) -> "PortRef":
        return cls(d["block"], d.get("port", 0), d.get("elems"))


@dataclass
class Connection:
    id: int
    src: PortRef
    targets: list[PortRef]

    @classmethod
    def parse(cls, d: dict) -> "Connection":
        return cls(d["id"], PortRef.parse(d["src"]), [PortRef.parse(t) for t in d["targets"]])


@dataclass
class Interface:
    inputs: list[Port] = field(default_factory=list)
    outputs: list[Port] = field(default_factory=list)

    @classmethod
    def parse(cls, d: dict) -> "Interface":
        return cls(
            [Port.parse(p) for p in d.get("inputs", [])],
            [Port.parse(p) for p in d.get("outputs", [])],
        )


@dataclass
class DagGroup:
    depth: int
    blocks: list[int]

    @classmethod
    def parse(cls, d: dict) -> "DagGroup":
        return cls(d["depth"], list(d["blocks"]))


@dataclass
class Scc:
    blocks: list[int]
    back_edges: list[int]

    @classmethod
    def parse(cls, d: dict) -> "Scc":
        return cls(list(d["blocks"]), list(d.get("back_edges", [])))


@dataclass
class Schedule:
    topo: list[int] = field(default_factory=list)
    groups: list[DagGroup] = field(default_factory=list)
    sccs: list[Scc] = field(default_factory=list)
    back_edges: list[int] = field(default_factory=list)

    @classmethod
    def parse(cls, d: dict) -> "Schedule":
        return cls(
            list(d.get("topo", [])),
            [DagGroup.parse(g) for g in d.get("groups", [])],
            [Scc.parse(s) for s in d.get("sccs", [])],
            list(d.get("back_edges", [])),
        )


# Child = Block | Subsystem (serde-tagged: {"Block": {...}} | {"Subsystem": {...}})


@dataclass
class Subsystem:
    id: int
    name: str
    interface: Interface
    children: list[Union[Block, "Subsystem"]]
    connections: list[Connection]
    schedule: Schedule

    @classmethod
    def parse(cls, d: dict) -> "Subsystem":
        children: list[Union[Block, Subsystem]] = []
        for c in d.get("children", []):
            (tag, payload), = c.items()
            children.append(Block.parse(payload) if tag == "Block" else Subsystem.parse(payload))
        return cls(
            id=d["id"],
            name=d["name"],
            interface=Interface.parse(d.get("interface", {})),
            children=children,
            connections=[Connection.parse(c) for c in d.get("connections", [])],
            schedule=Schedule.parse(d.get("schedule", {})),
        )

    def blocks(self) -> Iterator[Block]:
        """Iterate every leaf :class:`Block` in this scope, recursing into
        nested subsystems."""
        for c in self.children:
            if isinstance(c, Subsystem):
                yield from c.blocks()
            else:
                yield c

    def subsystems(self) -> Iterator["Subsystem"]:
        for c in self.children:
            if isinstance(c, Subsystem):
                yield c
                yield from c.subsystems()


@dataclass
class ExternDecl:
    id: int
    name: str
    arity_in: int
    arity_out: int

    @classmethod
    def parse(cls, d: dict) -> "ExternDecl":
        return cls(d["id"], d["name"], d["arity_in"], d["arity_out"])


@dataclass
class Module:
    """Top-level hierarchical IR of a model."""

    name: str
    root: Subsystem
    ir_version: int = 1
    description: str = ""
    events: list[Event] = field(default_factory=list)
    """Simulation-level (global) events not attached to any block. Always
    opaque (host guards/actions); only kind/timing is recorded."""
    extern_decls: list[ExternDecl] = field(default_factory=list)

    # -- construction --

    @classmethod
    def from_dict(cls, d: dict) -> "Module":
        return cls(
            name=d["name"],
            root=Subsystem.parse(d["root"]),
            ir_version=d.get("ir_version", 1),
            description=d.get("description", ""),
            events=[Event.parse(e) for e in d.get("events", [])],
            extern_decls=[ExternDecl.parse(e) for e in d.get("extern_decls", [])],
        )

    @classmethod
    def from_json(cls, s: str) -> "Module":
        m = cls.from_dict(json.loads(s))
        m._raw_json = s
        return m

    @classmethod
    def from_json_file(cls, path: str) -> "Module":
        with open(path) as f:
            s = f.read()
        return cls.from_json(s)

    # -- export --
    # The canonical JSON is what the Rust core emits; we round-trip through it.

    def to_json(self, indent: int = 2) -> str:
        return self._raw_json if self._raw_json is not None else json.dumps(_to_plain(self), indent=indent)

    def to_json_file(self, path: str, indent: int = 2) -> None:
        with open(path, "w") as f:
            f.write(self.to_json(indent=indent))

    # -- code generation --

    def to_c(
        self,
        *,
        numeric: str = "double",
        reductions: str = "unrolled",
        structure: str = "hierarchical",
        layout: str = "compact",
        solver: str = "rk4",
        api: str = "struct",
        scaffold: bool = False,
        trace: bool = False,
        a2l: bool = False,
    ) -> dict[str, str]:
        """Generate standalone C99 source from this IR.

        Lowers the model's scalar op-graph to C (the same path the verification
        suite checks against :func:`fastsim.ir.eval`). Returns a dict mapping
        each file name to its source; files are named after the model
        (``<name>.h`` + ``<name>.c``, default name ``model``).
        ``layout="compact"`` (the default) gives the two model files;
        ``layout="library"`` additionally splits out ``<name>_solver.{c,h}``
        (the integrator) and, under the hierarchical structure,
        ``<name>_blocks.{c,h}`` (the per-block functions). Compile the ``.c``
        files together.

        The code generator is built into the ``fastsim`` extension under the
        ``codegen`` feature; raises :class:`ImportError` if the installed wheel
        was built without it. (With a live :class:`~fastsim.Simulation`, prefer
        ``sim.to_c(...)``, which lowers in-process without this JSON round-trip.)

        Parameters
        ----------
        numeric : str
            scalar type: ``"double"`` (default), ``"float"``, ``"fixed"`` (reserved)
        reductions : str
            ``"unrolled"`` (default) or ``"vectorized"`` (Reduce/Dot as a counted loop)
        structure : str
            ``"hierarchical"`` (default; one function per block) or ``"flat"``
            (one fused ``dx/dt``)
        layout : str
            ``"compact"`` (default; ``.c`` + ``.h``) or ``"library"`` (multi-file)
        solver : str
            integrator tableau by name (case-insensitive): ``"rk4"`` (default) and
            ``"euler"`` are fixed-step; ``"rkdp54"``/``"rkck54"``/``"rkf45"``/
            ``"rkf78"``/``"rkv65"``/``"rkbs32"``/``"rkf21"``/``"rkdp87"`` are
            adaptive (embedded-error step control); ``"ssprk22"``/``"ssprk33"``/
            ``"ssprk34"`` are fixed-step. Implicit (DIRK/ESDIRK) not yet emitted.
        api : str
            ``"struct"`` (the only API): a single ``model_t`` holding state /
            signals / parameters / memory, with ``get_signal`` / ``set_signal``
            accessors. Reentrant and embeddable (inputs set via ``set_signal``).

        Returns
        -------
        dict[str, str]
            file name -> C source

        Raises
        ------
        ValueError
            an unknown option value, or malformed IR
        RuntimeError
            the model uses a construct the backend cannot lower (e.g. an opaque
            ``extern`` block, or an unsupported option combination)
        """
        from fastsim import _fastsim

        if not hasattr(_fastsim, "generate_c"):  # pragma: no cover - optional build
            raise ImportError(
                "this fastsim build has no code generator; rebuild the extension "
                "with the `codegen` feature enabled (`maturin develop --features "
                "python,codegen`)"
            )

        return _fastsim.generate_c(
            self.to_json(),
            numeric=numeric,
            reductions=reductions,
            structure=structure,
            layout=layout,
            solver=solver,
            api=api,
            scaffold=scaffold,
            trace=trace,
            a2l=a2l,
        )

    # -- inspection --

    def blocks(self) -> Iterator[Block]:
        """Every leaf block in the whole model (recurses subsystems)."""
        return self.root.blocks()

    def find(self, type_name: str) -> list[Block]:
        return [b for b in self.blocks() if b.type_name == type_name]

    def extern_blocks(self) -> list[Block]:
        return [b for b in self.blocks() if b.is_extern]

    def summary(self) -> str:
        blks = list(self.blocks())
        n_extern = sum(1 for b in blks if b.is_extern)
        n_sub = len(list(self.root.subsystems()))
        return (
            f"Module '{self.name}' (ir v{self.ir_version}): "
            f"{len(blks)} blocks ({len(blks) - n_extern} with ops, {n_extern} extern), "
            f"{n_sub} nested subsystems, {len(self.extern_decls)} extern decls, "
            f"{len(self.events)} global events"
        )

    # raw JSON the module was parsed from, so to_json is a lossless round-trip.
    _raw_json: Optional[str] = field(default=None, repr=False, compare=False)


def _to_plain(obj: Any) -> Any:
    """Serialize an IR dataclass tree back to plain JSON-compatible values
    (issue #33).

    The inverse of the ``parse``/``from_dict`` constructors, so a module built
    via the public :meth:`Module.from_dict` (no cached ``_raw_json``) can still
    be exported with :meth:`Module.to_json`. ``Tagged`` (serde externally-tagged
    enums) round-trip through the same three forms :meth:`Tagged.parse` accepts;
    regular dataclasses map field-for-field (private ``_`` fields and ``None``
    optionals are dropped, matching the tolerant ``d.get(...)`` parsers).
    """
    # Tagged first — it is itself a dataclass, but serializes to the serde form.
    if isinstance(obj, Tagged):
        if not obj.fields:
            return obj.kind  # unit variant  -> "Kind"
        if list(obj.fields.keys()) == ["value"]:
            return {obj.kind: _to_plain(obj.fields["value"])}  # newtype -> {"Kind": v}
        return {obj.kind: {k: _to_plain(v) for k, v in obj.fields.items()}}  # struct
    if is_dataclass(obj) and not isinstance(obj, type):
        out: dict = {}
        for f in dataclass_fields(obj):
            if f.name.startswith("_"):
                continue
            val = getattr(obj, f.name)
            if val is None:
                continue
            # `Subsystem.children` is an externally-tagged Union[Block, Subsystem]
            # (`{"Block": {...}}` / `{"Subsystem": {...}}`); wrap each element by
            # its type name so it re-parses. Root subsystems are untagged.
            if f.name == "children" and isinstance(val, list):
                out[f.name] = [{type(el).__name__: _to_plain(el)} for el in val]
            else:
                out[f.name] = _to_plain(val)
        return out
    if isinstance(obj, (list, tuple)):
        return [_to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj
