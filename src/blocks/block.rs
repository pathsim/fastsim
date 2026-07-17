// Base Block: the ONLY block type in fastsim
// Ported 1:1 from pathsim/blocks/_block.py
//
// In Python, Integrator(Block) overrides update(), solve(), step() etc.
// In Rust, Block has callback fields for these methods. Concrete block types
// (Integrator, Amplifier, etc.) are constructor functions that create a Block
// with the right callbacks. Everything is BlockRef = Rc<FastCell<Block>>.

use std::collections::HashMap;
use std::rc::Rc;

use smallvec::SmallVec;

use crate::constants::JACOBIAN_ZERO_THRESHOLD;
use crate::solvers::solver::Solver;
use crate::utils::fastcell::FastCell;
use crate::utils::register::Register;

/// Stack-allocated small vector for block outputs/derivatives.
/// Up to 8 elements on the stack, falls back to heap for larger.
pub type SVec = SmallVec<[f64; 8]>;

/// Shared reference to a Block — the single reference type used everywhere.
/// Used by PortReference, Connection, Simulation, ConnectionBooster.
pub type BlockRef = Rc<FastCell<Block>>;

/// Topology role of a block. Set explicitly by constructors. Source of truth
/// for graph classification (replaces `len() > 0` heuristics).
///
/// Flags are orthogonal — a block can be both `is_dyn` and `is_alg`
/// (e.g. StateSpace with non-zero D matrix).
#[derive(Debug, Clone, Copy)]
pub struct BlockRole {
    /// Has solver state (engine). Stepped/solved per RK stage.
    pub is_dyn: bool,
    /// No inputs (Source, Constant). Always at DAG depth 0 input-side.
    pub is_src: bool,
    /// No outputs (Scope, recorder). Sink — never participates downstream.
    pub is_rec: bool,
}

impl Default for BlockRole {
    /// Default = a plain leaf block (not dynamic, not a source, not a sink).
    /// Algebraic feedthrough is no longer part of the role: it is derived from
    /// the block's SSA (`Block::feedthrough_mask`), with opaque blocks declaring
    /// it via `Block::opaque_feedthrough`.
    fn default() -> Self {
        Self { is_dyn: false, is_src: false, is_rec: false }
    }
}

// Callback types for overridable methods
pub type LenFn = Box<dyn Fn(&Block) -> usize>;
pub type UpdateFn = Box<dyn FnMut(&mut Block, f64)>;
pub type SolveFn = Box<dyn FnMut(&mut Block, f64, f64) -> f64>;
pub type StepFn = Box<dyn FnMut(&mut Block, f64, f64) -> (bool, f64, Option<f64>)>;
pub type ResetFn = Box<dyn FnMut(&mut Block)>;
pub type SampleFn = Box<dyn FnMut(&mut Block, f64, f64)>;
pub type BufferFn = Box<dyn FnMut(&mut Block, f64)>;
pub type RevertFn = Box<dyn FnMut(&mut Block)>;
pub type SetSolverFn = Box<dyn FnMut(&dyn Fn(&[f64]) -> Solver)>;
pub type OnOffFn = Box<dyn FnMut()>;
/// Returns a block's child blocks (Subsystem only; None elsewhere). Lets hosts
/// recurse into composites — e.g. to find Scopes nested in a Subsystem.
pub type ChildrenFn = Box<dyn Fn() -> Vec<BlockRef>>;
/// Called on a freshly-constructed `Solver` after `set_solver_from` assigns it.
/// Used by DAE blocks to install a custom `StageBuilder` into the engine.
pub type EnginePostprocessFn = Box<dyn Fn(&mut Solver)>;

/// Write-into-buffer callback used by all block dynamic / algebraic / Jacobian
/// paths.  The callee appends (or writes) into `out`; the caller clears it
/// before the call.  Signature: `(x, u, t, out)`.
pub type BlockFn = Box<dyn Fn(&[f64], &[f64], f64, &mut Vec<f64>)>;

/// Base Block object — the ONLY block type.
///
/// Specific block types (Integrator, Amplifier, etc.) are created via
/// constructor functions that set the appropriate callbacks.
pub struct Block {
    // -- Core fields (same as Python Block) --
    pub inputs: Register,
    pub outputs: Register,
    pub engine: Option<Solver>,
    pub _active: bool,
    pub events: Vec<crate::simulation::SimEventRef>,
    pub initial_value: Option<Vec<f64>>,
    pub input_port_labels: Option<HashMap<String, usize>>,
    pub output_port_labels: Option<HashMap<String, usize>>,

    // -- Unified dynamic/algebraic paths (mirrors pathsim op_dyn / op_alg) --
    /// Dynamic path: dx/dt = f_dyn(x, u, t). Writes into `out` (cleared by caller).
    pub f_dyn: Option<BlockFn>,
    /// Jacobian df_dyn/dx. Writes into `out` (cleared by caller). If None → numerical.
    pub jac_dyn: Option<BlockFn>,
    /// `true` when `jac_dyn` returns a Jacobian that does not depend on the state
    /// (e.g. linear blocks: ∂(Ax+Bu)/∂x = A). Lets the implicit solver factor the
    /// Newton matrix `(I - dt·a_ii·J)` once per `dt` and reuse it across Newton
    /// iterations, stages (SDIRK shares the diagonal `a_ii`) and steps, instead
    /// of refactoring every iteration. Default `false` (conservative: refactor).
    pub jacobian_const: bool,
    /// Algebraic path: y = f_alg(x, u, t). Writes into `out` (cleared by caller).
    pub f_alg: Option<BlockFn>,
    /// Persistent scratch buffers (avoid per-call allocation).
    pub _f_buf: Vec<f64>,
    pub _jac_buf: Vec<f64>,

    // -- Custom data storage (for block-specific state like gain, recordings, etc.) --
    pub data_f64: HashMap<String, f64>,
    pub data_vec: HashMap<String, Vec<f64>>,
    pub data_vec2: HashMap<String, Vec<Vec<f64>>>,
    pub data_strings: HashMap<String, Vec<String>>,
    // -- Overridable methods (set by constructor functions) --
    pub len_fn: Option<LenFn>,
    pub update_fn: Option<UpdateFn>,
    pub solve_fn: Option<SolveFn>,
    pub step_fn: Option<StepFn>,
    pub reset_fn: Option<ResetFn>,
    pub sample_fn: Option<SampleFn>,
    pub buffer_fn: Option<BufferFn>,
    pub revert_fn: Option<RevertFn>,
    pub set_solver_fn: Option<SetSolverFn>,
    pub on_fn: Option<OnOffFn>,
    pub off_fn: Option<OnOffFn>,
    /// Child blocks of a composite (Subsystem). None for leaf blocks.
    pub children_fn: Option<ChildrenFn>,
    /// Optional engine post-processor: invoked on the newly-built solver
    /// inside `set_solver_from()` after `old.x` has been copied across.
    /// DAE blocks use this to install their `StageBuilder`.
    pub engine_postprocess: Option<EnginePostprocessFn>,

    // -- Block type name (for debugging/logging) --
    pub type_name: &'static str,

    /// Topology role — source of truth for graph classification.
    pub role: BlockRole,

    /// Dynamic-path Jacobian operator (the dyn op-graph). Supplies the
    /// sparse-aware AD Jacobian, so the runtime no longer needs a hand-written
    /// `jac_dyn` + `jac_pattern`. Set at construction by `set_dynamic` (or by
    /// the compiled-subsystem fuse); `None` for opaque dynamic blocks (raw Rust
    /// `ode`/`dynamical_system`), which fall back to the numerical Jacobian.
    pub dyn_op: Option<crate::blocks::operator::Operator>,
    /// Algebraic-path operator (op-graph for IR / codegen + the alg Jacobian).
    /// The block's sole algebraic graph representation. `None` for opaque blocks
    /// (RNG, FMU) and shape-poly discrete blocks (see `op_discrete_builder`).
    pub alg_op: Option<crate::blocks::operator::Operator>,
    /// Discrete-memory specs for sampled / event-driven blocks.
    pub op_memory: Vec<crate::blocks::blockops::MemSpec>,
    /// Block-internal event specs (Schedule / ZeroCross / Condition).
    pub op_events: Vec<crate::blocks::blockops::EventSpec>,
    /// Shape-poly discrete resolver: yields `(alg, memory, events)` at the
    /// connected input width (used in place of `alg_op` for those blocks).
    /// Returns `None` when the block cannot lower at this width (e.g. a
    /// JIT-traced effect whose Python callable does not trace) — the block then
    /// surfaces as an opaque extern `Op::Call`, same as any unported block.
    pub op_discrete_builder:
        Option<std::rc::Rc<dyn Fn(usize) -> Option<crate::blocks::blockops::DiscreteResolved>>>,
    /// IR type name, which can differ from the runtime `type_name` (e.g.
    /// "Source" vs "SinusoidalSource"). Set by the `set_*` operator helpers.
    pub op_type_name: Option<&'static str>,
    /// Live discrete-memory cell for blocks whose `dyn_op` graph reads memory
    /// (the fused compiled-subsystem block): the Jacobian tape needs the current
    /// memory as a fixed input. `None` for memory-free continuous blocks.
    pub mem_ref: Option<std::rc::Rc<crate::utils::fastcell::FastCell<Vec<f64>>>>,

    /// For `Subsystem` wrapper blocks: a handle to the inner block graph, so the
    /// IR builder can recurse into nested subsystems (the runtime accesses the
    /// inner via captured closures instead). `None` for leaf blocks.
    pub subsystem_inner: Option<std::rc::Rc<crate::utils::fastcell::FastCell<crate::subsystem::SubsystemInner>>>,

    /// Declared algebraic feedthrough for OPAQUE blocks only (no SSA `ops` and
    /// not a subsystem): the conservative `u(t) → y(t)` assumption the scheduler
    /// falls back to when it cannot derive the pattern. `true` (the default)
    /// over-approximates safely (never misses a real loop); opaque blocks that
    /// genuinely have no feedthrough (e.g. a DAE or noise source) set it `false`.
    /// Ignored for op-expressible and subsystem blocks, which derive the exact
    /// per-port matrix in `feedthrough_mask`.
    pub opaque_feedthrough: bool,

    /// Cached port-granular direct-feedthrough mask, keyed by `(n_in, n_out)`.
    /// The pattern is derived from the block's SSA (`feedthrough_pattern`), which
    /// is expensive (clones + optimizes the op-graph) and structural — it depends
    /// only on the block's equations and port dimensions, not on the surrounding
    /// topology. The scheduler rebuilds it on every `_assemble_graph` (i.e. every
    /// live mutation), so caching it here turns that into an `O(1)` lookup. Reset
    /// to `None` when the recorded `(n_in, n_out)` no longer matches.
    feedthrough_cache: std::cell::RefCell<Option<(usize, usize, Vec<bool>)>>,
}

impl Block {
    /// Create a new base Block.
    pub fn new(
        input_port_labels: Option<HashMap<String, usize>>,
        output_port_labels: Option<HashMap<String, usize>>,
    ) -> Self {
        Self {
            inputs: Register::new(None, input_port_labels.clone()),
            outputs: Register::new(None, output_port_labels.clone()),
            engine: None,
            _active: true,
            events: Vec::new(),
            initial_value: None,
            input_port_labels,
            output_port_labels,
            f_dyn: None,
            jac_dyn: None,
            jacobian_const: false,
            dyn_op: None,
            alg_op: None,
            op_memory: Vec::new(),
            op_events: Vec::new(),
            op_discrete_builder: None,
            op_type_name: None,
            mem_ref: None,
            f_alg: None,
            _f_buf: Vec::new(),
            _jac_buf: Vec::new(),
            data_f64: HashMap::new(),
            data_vec: HashMap::new(),
            data_vec2: HashMap::new(),
            data_strings: HashMap::new(),
            len_fn: None,
            children_fn: None,
            update_fn: None,
            solve_fn: None,
            step_fn: None,
            reset_fn: None,
            sample_fn: None,
            buffer_fn: None,
            revert_fn: None,
            set_solver_fn: None,
            on_fn: None,
            off_fn: None,
            engine_postprocess: None,
            type_name: "Block",
            role: BlockRole::default(),
            subsystem_inner: None,
            opaque_feedthrough: true,
            feedthrough_cache: std::cell::RefCell::new(None),
        }
    }

    pub fn default_block() -> Self {
        Self::new(None, None)
    }

    /// Port-granular direct-feedthrough mask (flat row-major `n_out × n_in`) for
    /// the scheduler, cached by `(n_in, n_out)`. Op-expressible blocks get the
    /// exact pattern from their algebraic SSA region; opaque blocks (or any size
    /// mismatch) fall back to the declared `role.is_alg` as an all-or-nothing
    /// matrix. The expensive SSA derivation runs once per `(n_in, n_out)`; later
    /// `_assemble_graph` calls (every live mutation) hit the cache.
    pub fn feedthrough_mask(&self, n_in: usize, n_out: usize) -> Vec<bool> {
        if let Some((ci, co, mask)) = self.feedthrough_cache.borrow().as_ref() {
            if *ci == n_in && *co == n_out {
                return mask.clone();
            }
        }
        let mask = self.compute_feedthrough_mask(n_in, n_out);
        *self.feedthrough_cache.borrow_mut() = Some((n_in, n_out, mask.clone()));
        mask
    }

    fn compute_feedthrough_mask(&self, n_in: usize, n_out: usize) -> Vec<bool> {
        // Op-expressible leaf block: exact per-port pattern from its algebraic
        // operator's SSA region.
        let alg_rg = self.alg_op.as_ref().and_then(|o| o.graph_ref());
        // Shape-poly discrete leaf (`set_discrete_lazy`): no standalone `alg_op`;
        // resolve the builder at the connected width to get its alg region. By
        // construction this region reads memory (not input), so it has no
        // feedthrough, but we derive it exactly rather than assume.
        let resolved_alg = if alg_rg.is_none() {
            self.op_discrete_builder
                .as_ref()
                .and_then(|b| b(n_in.max(1)))
                .map(|(alg, _, _)| alg)
        } else {
            None
        };
        // A `Lazy` region that cannot lower at this width (e.g. a Python callable
        // with data-dependent control flow) yields `None` here and falls through
        // to the opaque (conservative) feedthrough below, exactly like any other
        // opaque block — never a panic.
        let alg_graph = alg_rg
            .and_then(|rg| rg.resolve(n_in.max(1)))
            .or(resolved_alg);
        if let Some(alg) = alg_graph {
            return match crate::ssa::autodiff::feedthrough_pattern(&alg, "u") {
                // Exact derived pattern.
                Some(mask) if mask.len() == n_in * n_out => mask,
                // The region has no input ("u") slot: it reads no input, so it
                // has no feedthrough (e.g. a source: y = f(t)). NOT the opaque
                // fallback, which would spuriously mark it algebraic.
                None => vec![false; n_in * n_out],
                // Resolved-shape mismatch (rare, shape-poly edge): be conservative.
                Some(_) => vec![true; n_in * n_out],
            };
        }
        // Subsystem (composite) block: derive the exact per-port pattern from its
        // interior graph (interface-in -> interface-out reachability), so the
        // enclosing scheduler treats it like any other block.
        if let Some(inner) = &self.subsystem_inner {
            let mask = inner.borrow().feedthrough_matrix();
            if mask.len() == n_in * n_out {
                return mask;
            }
        }
        // Opaque block (RNG/FMU/DAE): a source or sink never has input→output
        // feedthrough; otherwise use the declared (conservative) opaque flag.
        if self.role.is_src || self.role.is_rec {
            return vec![false; n_in * n_out];
        }
        vec![self.opaque_feedthrough; n_in * n_out]
    }

    // -- __len__: algebraic path length --

    pub fn len(&self) -> usize {
        if let Some(ref f) = self.len_fn {
            f(self)
        } else {
            if self._active { 1 } else { 0 }
        }
    }

    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn is_active(&self) -> bool { self._active }

    // -- on/off --

    pub fn on(&mut self) {
        self._active = true;
        if let Some(mut f) = self.on_fn.take() {
            f();
            self.on_fn = Some(f);
        }
    }

    pub fn off(&mut self) {
        self._active = false;
        // Deactivate internal events
        for event in &self.events {
            event.borrow_mut().off();
        }
        // Reset outputs to zero (matches pathsim behavior)
        self.outputs.reset();
        if let Some(mut f) = self.off_fn.take() {
            f();
            self.off_fn = Some(f);
        }
    }

    // -- size/shape --

    pub fn size(&self) -> (usize, usize) {
        let nx = self.engine.as_ref().map(|e| e.len()).unwrap_or(0);
        (1, nx)
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.inputs.len(), self.outputs.len())
    }

    // -- state access --

    pub fn state(&self) -> Option<&[f64]> {
        self.engine.as_ref().map(|e| e.get())
    }

    pub fn set_state(&mut self, val: &[f64]) {
        if let Some(ref mut engine) = self.engine {
            engine.set(val);
        }
    }

    pub fn get_all(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let inputs = self.inputs.to_array();
        let outputs = self.outputs.to_array();
        let states = self.engine.as_ref().map(|e| e.get().to_vec()).unwrap_or_default();
        (inputs, outputs, states)
    }

    // -- reset --

    pub fn reset(&mut self) {
        // Reset internal events regardless of the reset path (matches pathsim
        // Block.reset, which always resets self.events).
        for event in &self.events {
            event.borrow_mut().reset();
        }

        // Custom reset first (e.g. Scope clears recordings)
        if let Some(mut f) = self.reset_fn.take() {
            f(self);
            self.reset_fn = Some(f);
            return; // custom reset handles everything
        }

        // Default reset
        self.inputs.reset();
        self.outputs.reset();
        if let Some(ref mut engine) = self.engine {
            if let Some(ref iv) = self.initial_value {
                engine.reset(Some(iv));
            } else {
                engine.reset(None);
            }
        }
    }

    // -- set_solver --

    pub fn set_solver_from(&mut self, make_solver: &dyn Fn(&[f64]) -> Solver) {
        // If subsystem/composite, also delegate to internal blocks
        if let Some(mut f) = self.set_solver_fn.take() {
            f(make_solver);
            self.set_solver_fn = Some(f);
        }
        // Set own engine (for subsystem: dummy engine so Simulation treats it as dynamic)
        if let Some(ref iv) = self.initial_value {
            let mut new_engine = make_solver(iv);
            if let Some(ref old_engine) = self.engine {
                new_engine.set(old_engine.get());
            }
            // DAE blocks patch the freshly-built engine with their custom
            // stage builder (or other per-form configuration).
            if let Some(ref postprocess) = self.engine_postprocess {
                postprocess(&mut new_engine);
            }
            self.engine = Some(new_engine);
        }
    }

    // -- buffer/revert --

    pub fn buffer(&mut self, dt: f64) {
        if let Some(mut f) = self.buffer_fn.take() {
            f(self, dt);
            self.buffer_fn = Some(f);
        } else if let Some(ref mut engine) = self.engine {
            engine.buffer(dt);
        }
    }

    pub fn revert(&mut self) {
        if let Some(mut f) = self.revert_fn.take() {
            f(self);
            self.revert_fn = Some(f);
        } else if let Some(ref mut engine) = self.engine {
            engine.revert();
        }
    }

    // -- update: f_alg path or custom callback --

    pub fn update(&mut self, t: f64) {
        // Priority 1: custom callback (Scope, Delay, Switch, etc.)
        // Use take/put to satisfy borrow checker (callback needs &mut self)
        if let Some(mut f) = self.update_fn.take() {
            f(self, t);
            self.update_fn = Some(f);
            return;
        }
        // Priority 2: unified f_alg — write directly into outputs (no intermediate buffer)
        if let Some(ref f_alg) = self.f_alg {
            let x: &[f64] = self.engine.as_ref().map(|e| e.get()).unwrap_or(&[]);
            self.outputs._data.clear();
            f_alg(x, &self.inputs._data, t, &mut self.outputs._data);
        }
    }

    // -- solve: f_dyn path with Jacobian, or custom callback --

    // -- Operator construction helpers (the per-path SSA op-graphs). These set
    //    `alg_op` / `dyn_op` + the discrete fields directly: the operator is the
    //    block's sole graph representation. Native eval stays on `f_alg`/`f_dyn`.
    //    `op_type_name` carries the IR type name (which may differ from the
    //    runtime `type_name`, e.g. "Source"/"SinusoidalSource").

    /// Shape-fixed algebraic-only block: `y = f(x, u, t)`.
    pub fn set_alg(&mut self, type_name: &'static str, alg: crate::ssa::graph::Graph) {
        self.op_type_name = Some(type_name);
        self.alg_op = Some(crate::blocks::operator::Operator::graph_only(
            crate::blocks::blockops::RegionGraph::Fixed(alg),
        ));
    }

    /// Shape-poly algebraic-only block (graph resolved at the connected width).
    pub fn set_alg_lazy(
        &mut self,
        type_name: &'static str,
        alg: std::rc::Rc<crate::blocks::blockops::ShapeLazyGraph>,
    ) {
        self.op_type_name = Some(type_name);
        self.alg_op = Some(crate::blocks::operator::Operator::graph_only(
            crate::blocks::blockops::RegionGraph::Lazy(alg),
        ));
    }

    /// Dynamic block: `y = f_alg`, `dx/dt = f_dyn`. State init comes from
    /// `initial_value` (set separately by the constructor).
    pub fn set_dynamic(
        &mut self,
        type_name: &'static str,
        alg: crate::ssa::graph::Graph,
        dyn_: crate::ssa::graph::Graph,
    ) {
        use crate::blocks::blockops::RegionGraph;
        use crate::blocks::operator::Operator;
        self.op_type_name = Some(type_name);
        self.alg_op = Some(Operator::graph_only(RegionGraph::Fixed(alg)));
        self.dyn_op = Some(Operator::graph_only(RegionGraph::Fixed(dyn_)));
    }

    /// Discrete block: algebraic output plus memory slots + the events that
    /// update them.
    pub fn set_discrete(
        &mut self,
        type_name: &'static str,
        alg: crate::ssa::graph::Graph,
        memory: Vec<crate::blocks::blockops::MemSpec>,
        events: Vec<crate::blocks::blockops::EventSpec>,
    ) {
        self.op_type_name = Some(type_name);
        self.alg_op = Some(crate::blocks::operator::Operator::graph_only(
            crate::blocks::blockops::RegionGraph::Fixed(alg),
        ));
        self.op_memory = memory;
        self.op_events = events;
    }

    /// Shape-poly discrete block: resolves `(alg, memory, events)` at the
    /// connected width.
    pub fn set_discrete_lazy(
        &mut self,
        type_name: &'static str,
        builder: impl Fn(usize) -> Option<crate::blocks::blockops::DiscreteResolved> + 'static,
    ) {
        self.op_type_name = Some(type_name);
        self.op_discrete_builder = Some(std::rc::Rc::new(builder));
    }

    pub fn solve(&mut self, t: f64, dt: f64) -> f64 {
        // Priority 1: custom callback
        if let Some(mut f) = self.solve_fn.take() {
            let result = f(self, t, dt);
            self.solve_fn = Some(f);
            return result;
        }
        if self.f_dyn.is_none() {
            return 0.0;
        }
        // Priority 2: unified f_dyn path — write into persistent scratch buffer.
        let x: &[f64] = self.engine.as_ref().map(|e| e.get()).unwrap_or(&[]);
        self._f_buf.clear();
        (self.f_dyn.as_ref().unwrap())(x, &self.inputs._data, t, &mut self._f_buf);
        // Jacobian: prefer the operator's sparse-aware AD Jacobian (derived from
        // the dyn op-graph); fall back to the legacy analytical/numerical path for
        // opaque blocks (FMU) without an operator. Memory-bearing operator graphs
        // (the fused compiled-subsystem block) read the live memory cell.
        let mem_guard = self.mem_ref.as_ref().map(|c| c.borrow());
        let mem: &[f64] = match &mem_guard {
            Some(g) => g.as_slice(),
            None => &[],
        };
        let jac = if let Some(op) = &self.dyn_op {
            op.jacobian_wrt_state(x, &self.inputs._data, t, mem)
        } else {
            Self::compute_jacobian(
                self.jac_dyn.as_ref(), self.f_dyn.as_ref(), self.engine.is_some(),
                &mut self._jac_buf, x, &self.inputs._data, t,
            )
        };
        if let Some(ref mut engine) = self.engine {
            return engine.solve(&self._f_buf, jac, dt);
        }
        0.0
    }

    // -- step: f_dyn path, or custom callback --

    pub fn step(&mut self, t: f64, dt: f64) -> (bool, f64, Option<f64>) {
        // Priority 1: custom callback
        if let Some(mut f) = self.step_fn.take() {
            let result = f(self, t, dt);
            self.step_fn = Some(f);
            return result;
        }
        // Priority 2: unified f_dyn path — write into persistent scratch buffer
        if let Some(ref f_dyn) = self.f_dyn {
            let x: &[f64] = self.engine.as_ref().map(|e| e.get()).unwrap_or(&[]);
            self._f_buf.clear();
            f_dyn(x, &self.inputs._data, t, &mut self._f_buf);
            if let Some(ref mut engine) = self.engine {
                return engine.step(&self._f_buf, dt);
            }
        }
        (true, 0.0, None)
    }

    // -- Jacobian computation --

    /// Legacy Jacobian path for OPAQUE blocks only (no op-graph): an analytical
    /// `jac_dyn` closure (dense `n×n`, e.g. FMU directional derivatives) or a
    /// central-difference fallback. Traceable blocks get their (sparse-aware)
    /// Jacobian from `dyn_op` instead, so this no longer carries a sparse pattern.
    fn compute_jacobian(
        jac_dyn: Option<&BlockFn>,
        f_dyn: Option<&BlockFn>,
        has_engine: bool,
        jac_buf: &mut Vec<f64>,
        x: &[f64],
        u: &[f64],
        t: f64,
    ) -> Option<crate::solvers::solver::Jacobian> {
        if let Some(jac_fn) = jac_dyn {
            jac_buf.clear();
            jac_fn(x, u, t, jac_buf);
            // An analytical Jacobian callback may produce nothing (e.g. a traced
            // RHS with no x-dependence, where AD yields an empty graph). Treat an
            // empty/short result as "no analytical Jacobian" and fall through to
            // the numerical path rather than indexing out of bounds.
            if jac_buf.len() == x.len() * x.len() && !jac_buf.is_empty() {
                return if x.len() == 1 {
                    Some(crate::solvers::solver::Jacobian::Scalar(jac_buf[0]))
                } else {
                    Some(crate::solvers::solver::Jacobian::Matrix(std::mem::take(jac_buf), x.len()))
                };
            }
        }
        if has_engine {
            if let Some(f_dyn) = f_dyn {
                let jac = crate::utils::numerical::num_jac(
                    &|xx, uu, tt, out| f_dyn(xx, uu, tt, out), x, u, t);
                match &jac {
                    crate::solvers::solver::Jacobian::Scalar(v) if v.abs() < JACOBIAN_ZERO_THRESHOLD => None,
                    crate::solvers::solver::Jacobian::Matrix(m, _) if m.iter().all(|v| v.abs() < JACOBIAN_ZERO_THRESHOLD) => None,
                    _ => Some(jac),
                }
            } else { None }
        } else { None }
    }

    // -- sample (overridable) --

    pub fn sample(&mut self, t: f64, dt: f64) {
        if let Some(mut f) = self.sample_fn.take() {
            f(self, t, dt);
            self.sample_fn = Some(f);
        }
    }

    // -- plot --

    pub fn plot(&self) {
        // Visualization blocks override this; base is no-op
    }

    // -- has_engine / has_events --

    pub fn has_engine(&self) -> bool { self.engine.is_some() }
    pub fn has_events(&self) -> bool { !self.events.is_empty() }
}

/// Create a new BlockRef.
pub fn new_block_ref(
    input_port_labels: Option<HashMap<String, usize>>,
    output_port_labels: Option<HashMap<String, usize>>,
) -> BlockRef {
    Rc::new(FastCell::new(Block::new(input_port_labels, output_port_labels)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_init() {
        let b = Block::default_block();
        assert_eq!(b.inputs.len(), 1);
        assert_eq!(b.outputs.len(), 1);
        assert!(b.engine.is_none());
        assert!(b._active);
        assert_eq!(b.len(), 1);
        assert_eq!(b.type_name, "Block");
    }

    #[test]
    fn test_block_len_override() {
        let mut b = Block::default_block();
        b.len_fn = Some(Box::new(|_| 0));
        assert_eq!(b.len(), 0);
    }

    #[test]
    fn test_block_on_off() {
        let mut b = Block::default_block();
        assert!(b.is_active());
        b.off();
        assert!(!b.is_active());
        assert_eq!(b.len(), 0);
        b.on();
        assert!(b.is_active());
    }

    #[test]
    fn test_block_reset() {
        let mut b = Block::default_block();
        b.inputs.update_from_array(&[1.0, 2.0]);
        b.outputs.update_from_array(&[3.0]);
        b.reset();
        assert_eq!(b.inputs.get_single(0), 0.0);
        assert_eq!(b.outputs.get_single(0), 0.0);
    }

    #[test]
    fn test_block_update_override() {
        let mut b = Block::default_block();
        b.update_fn = Some(Box::new(|blk, _t| {
            blk.outputs.set_single(0, 42.0);
        }));
        b.update(0.0);
        assert_eq!(b.outputs.get_single(0), 42.0);
    }

    #[test]
    fn test_block_ref_shared() {
        let br = Rc::new(FastCell::new(Block::default_block()));
        let br2 = br.clone();
        br.borrow_mut().inputs.set_single(0, 99.0);
        assert_eq!(br2.borrow().inputs.get_single(0), 99.0);
    }

    #[test]
    fn test_block_with_engine() {
        let mut b = Block::default_block();
        b.initial_value = Some(vec![1.0]);
        b.engine = Some(Solver::from_scalar(1.0));
        assert_eq!(b.size(), (1, 1));
        assert_eq!(b.state(), Some(&[1.0][..]));
    }

    #[test]
    fn test_block_data_storage() {
        let mut b = Block::default_block();
        b.data_f64.insert("gain".to_string(), 3.0);
        assert_eq!(b.data_f64["gain"], 3.0);
    }
}
