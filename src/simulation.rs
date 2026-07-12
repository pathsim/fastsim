// Main simulation engine — ported from pathsim/simulation.py
//
// Core methods ported: run, run_streaming, run_realtime, steadystate, reset, stop,
// timestep, _update, _dag, _loops, _solve, _step, _buffer, _revert, _sample,
// add/remove block/connection/event, _set_solver, _assemble_graph.
//
// Not ported: linearize/delinearize, checkpoint save/load, Duplex connections.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use crate::blocks::block::{Block, BlockRef};
use crate::connection::Connection;
use crate::constants::*;
use crate::error::SimError;
use crate::events::eventtype::SimEvent;
use crate::optim::booster::ConnectionBooster;
use crate::solvers::solver::Solver;
use crate::utils::diagnostics::{ConvergenceTracker, StepTracker, Diagnostics};
use crate::utils::fastcell::FastCell;
use crate::utils::schedule::Schedule;

/// Shared reference types for simulation components.
pub type SimEventRef = Rc<FastCell<dyn SimEvent>>;
pub use crate::connection::ConnectionRef;

// Cooperative stop signal raised from inside a running simulation.
//
// A user callback (e.g. a `Function` block or an event action) that raises
// pathsim's `StopSimulation` cannot unwind through the Rust closure boundary —
// the pybindings wrapper catches it and calls `request_stop()` instead. The run
// loop observes it via `take_stop_requested()` at the next safe checkpoint and
// terminates cleanly, exactly as if `stop()` had been called. Thread-local
// because a simulation runs single-threaded; cleared at the start of each run.
thread_local! {
    static STOP_REQUESTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

    /// First data-dependent runtime fault raised from deep inside an eval
    /// closure that cannot return `Err` (e.g. a zero denominator in Divider's
    /// `f_alg` under `ZeroDiv::Raise`, issue #28). The closure records the
    /// fault and requests a cooperative stop; the run wrapper drains it after
    /// the loop unwinds and re-raises it as a catchable Python exception
    /// instead of a `PanicException`.
    static RUNTIME_FAULT: std::cell::RefCell<Option<crate::error::SimError>> =
        const { std::cell::RefCell::new(None) };
}

/// Signal that the running simulation should stop at the next checkpoint.
pub fn request_stop() {
    STOP_REQUESTED.with(|c| c.set(true));
}

/// Read and clear the cooperative stop signal.
pub fn take_stop_requested() -> bool {
    STOP_REQUESTED.with(|c| c.replace(false))
}

/// Clear any stale stop signal. Called at the start of each run.
pub fn clear_stop_requested() {
    STOP_REQUESTED.with(|c| c.set(false));
    RUNTIME_FAULT.with(|c| *c.borrow_mut() = None);
}

/// Record a data-dependent runtime fault and request a cooperative stop. First
/// fault wins (later faults in the same step are dropped). Used by eval
/// closures that cannot propagate `Err` through the Rust `Fn` boundary.
pub fn record_runtime_fault(err: crate::error::SimError) {
    RUNTIME_FAULT.with(|c| {
        let mut slot = c.borrow_mut();
        if slot.is_none() { *slot = Some(err); }
    });
    request_stop();
}

/// Take the pending runtime fault, if any. Called by the run wrappers to
/// re-raise it after the loop has unwound.
pub fn take_runtime_fault() -> Option<crate::error::SimError> {
    RUNTIME_FAULT.with(|c| c.borrow_mut().take())
}

/// Mutation requests queued by Schedule (or other) callbacks while the
/// simulation is running.  Because pyo3 holds a `&mut self` borrow on
/// `PySimulation` for the entire duration of `run()`, the callback can't
/// call `sim.add_block` etc. directly — that would re-enter the borrow
/// and panic.  The callback pushes `PendingOp` entries into
/// `Simulation::_pending_ops` (interior mutability via `Rc<RefCell<…>>`,
/// which is NOT subject to pyo3's borrow checker), and the engine drains
/// the queue at the next timestep boundary via `_apply_pending_ops`.
pub enum PendingOp {
    AddBlock(BlockRef),
    RemoveBlock(BlockRef),
    AddConnection(ConnectionRef),
    RemoveConnection(ConnectionRef),
    AddEvent(SimEventRef),
    RemoveEvent(SimEventRef),
}

/// Statistics returned by run().
pub struct RunStats {
    pub total_steps: usize,
    pub successful_steps: usize,
    pub rejected_steps: usize,
    pub total_evals: usize,
    pub total_solver_its: usize,
    pub wall_time_secs: f64,

    // Structured numeric outcome (issue #27). Populated truthfully every run so
    // headless/optimizer callers can inspect convergence instead of trusting a
    // green summary. `converged` is false if any implicit solve or algebraic
    // loop exhausted its iteration budget during the run; `truncated_at` is
    // `Some(t)` if the trajectory ended before the requested end (step floor);
    // `max_residual` / `worst_block` carry the worst WRMS residual observed and
    // the block index that produced it (data lifted from ConvergenceTracker,
    // previously discarded).
    pub converged: bool,
    pub truncated_at: Option<f64>,
    pub max_residual: f64,
    pub worst_block: Option<usize>,
}

impl Default for RunStats {
    fn default() -> Self {
        Self {
            total_steps: 0,
            successful_steps: 0,
            rejected_steps: 0,
            total_evals: 0,
            total_solver_its: 0,
            wall_time_secs: 0.0,
            converged: true,
            truncated_at: None,
            max_residual: 0.0,
            worst_block: None,
        }
    }
}

/// Main simulation engine.
///
/// Performs transient analysis of dynamical systems defined by blocks and connections.
/// Fixed-point iteration distributes information through the system each timestep.
/// Handles algebraic loops via Anderson-accelerated fixed-point iteration.
/// Supports adaptive timestepping, implicit solvers, and discrete events.
pub struct Simulation {
    // System definition
    pub blocks: Vec<BlockRef>,
    pub connections: Vec<ConnectionRef>,
    pub events: Vec<SimEventRef>,

    // Simulation timestep and bounds
    pub dt: f64,
    pub dt_min: f64,
    pub dt_max: Option<f64>,

    // Global dummy engine for stage synchronization and solver attributes
    pub engine: Solver,

    // Solver constructor (for _set_solver and add_block)
    solver_factory: Option<Box<dyn Fn(&[f64]) -> Solver>>,
    pub solver_kwargs: Vec<(String, f64)>,

    // Internal system graph
    graph: Option<Schedule>,
    _graph_dirty: bool,

    // Algebraic loop solvers
    boosters: Vec<ConnectionBooster>,

    // Fixed-point iteration parameters.
    //
    // Convergence threshold for the algebraic-loop solver, the implicit-stage
    // Newton-Anderson solve, and the steady-state operating-point loop is the
    // unitless ARKODE-style `NLS_COEF = 0.1` applied to a WRMS-scaled
    // residual; the solver-level `tolerance_lte_abs` / `tolerance_lte_rel`
    // provide the WRMS weights.  The legacy absolute `tolerance_fpi` kwarg
    // has been retired in favour of this scale-aware criterion — see
    // `Simulation::_solve`, `Simulation::_loops`, `make_dirk_solve`.
    pub iterations_max: usize,

    // Simulation time
    pub time: f64,

    // Cached block sublists (indices into self.blocks)
    _blocks_dyn_indices: Vec<usize>,
    _blocks_evt_indices: Vec<usize>,

    // Event-loop scratch buffer, reused across timesteps to avoid per-step allocs
    _detected_scratch: Vec<(SimEventRef, bool, f64)>,


    // Active flag
    _active: bool,

    // Set by `_assemble_graph` when a connection's port aliases fail to
    // resolve (a user configuration error). `_update` bails instead of
    // panicking deep in the data transfer; the run wrappers surface it.
    _assembly_error: Option<crate::error::SimError>,

    // Persistent working timestep for the chunked streaming generator
    // (`run_begin`/`run_until`).  The blocking `run`/`run_streaming` keep
    // their adaptive dt in a local; the generator pauses between ticks, so
    // its dt must survive across `run_until` calls.
    _run_dt: f64,

    // Progress tracker for the chunked streaming generator. Created in
    // `run_begin` (logs STARTING), updated per step in `run_until` (progress
    // bar / it/s), finalized in `run_end` (logs FINISHED). None outside a
    // streaming run.
    _stream_tracker: Option<crate::utils::logger::ProgressTracker>,

    // Convergence trackers (mirrors Python's three trackers)
    _loop_tracker: ConvergenceTracker,
    _solve_tracker: ConvergenceTracker,
    _step_tracker: StepTracker,

    // Per-run numeric outcome (issue #27). `_reset_run_outcome` clears these at
    // the start of every run; the convergence loops set them on failure; the
    // run methods copy them into RunStats. Previously this data was rendered
    // into a log string and discarded, so a headless caller could not tell a
    // clean run from one that silently proceeded on an unconverged state.
    _run_converged: bool,
    _run_max_residual: f64,
    _run_worst_block: Option<usize>,
    _run_truncated_at: Option<f64>,

    // Diagnostics (None when disabled)
    pub diagnostics: Option<Diagnostics>,
    pub diagnostics_history: Option<Vec<Diagnostics>>,

    // Per-timestep wall-clock trace.  When `Some`, every `timestep()` call
    // appends its elapsed wall-clock duration in seconds.  Off by default
    // so steady-state runs pay zero overhead; enable via the Python
    // `enable_wct_trace()` getter to inspect mutation hickups, jitter, or
    // any other per-step timing characteristic that the aggregate
    // `RunStats.wall_time_secs` summary loses.
    pub _wct_trace: Option<Vec<f64>>,

    /// Mutation queue.  Drained by `apply_pending_ops` at the start of
    /// every `timestep()`.  Schedule (and other event) callbacks push
    /// entries into this queue from Python or Rust to add/remove blocks,
    /// connections, or events without re-entering the `&mut self` borrow
    /// that `run()` holds during simulation.
    pub _pending_ops: Rc<RefCell<Vec<PendingOp>>>,

    // Logging
    pub logger: crate::utils::logger::Logger,
}

/// Assemble the port-granular scheduling graph for a set of blocks and
/// connections. This is the single source of truth for the algebraic-feedthrough
/// schedule, shared by the top-level [`Simulation`], nested `SubsystemInner`, and
/// the IR schedule export, so every layer schedules identically. Ports must be
/// resolved (`Connection::resolve_ports`) before calling: the per-block
/// feedthrough matrices (`Block::feedthrough_mask`) and the element-granular
/// `PortEdge`s read the resolved register widths and port indices.
pub(crate) fn assemble_graph_from(
    blocks: &[crate::blocks::block::BlockRef],
    connections: &[crate::connection::ConnectionRef],
) -> Schedule {
    use crate::utils::feedthrough_loops::{Feedthrough, PortEdge};
    let block_index: std::collections::HashMap<*const FastCell<crate::blocks::block::Block>, usize> =
        blocks.iter().enumerate().map(|(i, b)| (Rc::as_ptr(b), i)).collect();

    // Block-level connection specs (src, tgt, conn_id) for the loop processor /
    // schedule export.
    let mut conn_specs = Vec::new();
    for (conn_idx, conn) in connections.iter().enumerate() {
        for target in &conn.targets {
            let si = block_index.get(&Rc::as_ptr(&conn.source.block)).copied();
            let ti = block_index.get(&Rc::as_ptr(&target.block)).copied();
            if let (Some(si), Some(ti)) = (si, ti) {
                conn_specs.push((si, ti, conn_idx));
            }
        }
    }

    // Per-block direct-feedthrough matrices (exact per-port pattern from each
    // block's SSA, or its declared fallback for opaque blocks).
    let feedthrough: Vec<Feedthrough> = blocks
        .iter()
        .map(|b| {
            let b = b.borrow();
            let n_in = b.inputs._data.len();
            let n_out = b.outputs._data.len();
            Feedthrough::new(n_in, n_out, b.feedthrough_mask(n_in, n_out))
        })
        .collect();

    // Element-granular port edges from the resolved connection indices.
    let mut port_edges = Vec::new();
    for conn in connections {
        let Some(src) = block_index.get(&Rc::as_ptr(&conn.source.block)).copied() else { continue };
        let out_idx = conn.source._get_output_indices();
        for target in &conn.targets {
            let Some(tgt) = block_index.get(&Rc::as_ptr(&target.block)).copied() else { continue };
            let in_idx = target._get_input_indices();
            for k in 0..out_idx.len().min(in_idx.len()) {
                port_edges.push(PortEdge { src, out_port: out_idx[k], tgt, in_port: in_idx[k] });
            }
        }
    }

    Schedule::new_with_feedthrough(&conn_specs, feedthrough, port_edges)
}

impl Simulation {
    /// Create a new Simulation.
    pub fn new(
        blocks: Vec<BlockRef>,
        connections: Vec<ConnectionRef>,
        events: Vec<SimEventRef>,
        dt: f64,
        dt_min: f64,
        dt_max: Option<f64>,
        iterations_max: usize,
        enable_diagnostics: bool,
    ) -> Self {
        let engine = Solver::with_defaults(&[0.0]);

        let mut sim = Self {
            blocks: Vec::new(),
            connections: Vec::new(),
            events: Vec::new(),
            dt, dt_min, dt_max,
            engine,
            solver_factory: None,
            solver_kwargs: Vec::new(),
            graph: None,
            _graph_dirty: false,
            boosters: Vec::new(),
            iterations_max,
            time: 0.0,
            _blocks_dyn_indices: Vec::new(),
            _blocks_evt_indices: Vec::new(),
            _detected_scratch: Vec::new(),
            _active: true,
            _assembly_error: None,
            _run_dt: dt,
            _stream_tracker: None,
            _loop_tracker: ConvergenceTracker::new(),
            _solve_tracker: ConvergenceTracker::new(),
            _step_tracker: StepTracker::new(),
            _run_converged: true,
            _run_max_residual: 0.0,
            _run_worst_block: None,
            _run_truncated_at: None,
            diagnostics: if enable_diagnostics { Some(Diagnostics::new()) } else { None },
            diagnostics_history: None,
            _wct_trace: None,
            _pending_ops: Rc::new(RefCell::new(Vec::new())),
            logger: crate::utils::logger::Logger::disabled(),
        };

        for block in blocks { let _ = sim.add_block(block); }
        for conn in connections { sim.add_connection(conn); }
        for evt in events { sim.add_event(evt); }

        sim._check_blocks_are_managed();
        sim._assemble_graph();

        sim
    }

    /// Create with default parameters (SSPRK22 solver, like pathsim).
    pub fn with_defaults(
        blocks: Vec<BlockRef>,
        connections: Vec<ConnectionRef>,
    ) -> Self {
        Self::with_solver(
            blocks, connections,
            crate::solvers::factories::ssprk22_factory(),
            SIM_TIMESTEP,
        )
    }

    /// Create with a specific solver factory.
    /// The factory is called for each dynamic block to create its integration engine.
    /// Also creates a dummy engine for stage synchronization.
    pub fn with_solver(
        blocks: Vec<BlockRef>,
        connections: Vec<ConnectionRef>,
        solver_factory: Box<dyn Fn(&[f64]) -> Solver>,
        dt: f64,
    ) -> Self {
        Self::with_solver_and_logger(blocks, connections, solver_factory, dt, false)
    }

    pub fn with_solver_and_logger(
        blocks: Vec<BlockRef>,
        connections: Vec<ConnectionRef>,
        solver_factory: Box<dyn Fn(&[f64]) -> Solver>,
        dt: f64,
        log: bool,
    ) -> Self {
        let logger = crate::utils::logger::Logger::new(log, "simulation");
        if log {
            logger.info(&format!("LOGGING (log: {})", if log { "True" } else { "False" }));
        }

        // Create dummy engine from factory
        let engine = solver_factory(&[0.0]);

        let mut sim = Self {
            blocks: Vec::new(),
            connections: Vec::new(),
            events: Vec::new(),
            dt,
            dt_min: SIM_TIMESTEP_MIN,
            dt_max: None,
            engine,
            solver_factory: Some(solver_factory),
            solver_kwargs: Vec::new(),
            graph: None,
            _graph_dirty: false,
            boosters: Vec::new(),
            iterations_max: SIM_ITERATIONS_MAX,
            time: 0.0,
            _blocks_dyn_indices: Vec::new(),
            _blocks_evt_indices: Vec::new(),
            _detected_scratch: Vec::new(),
            _active: true,
            _assembly_error: None,
            _run_dt: dt,
            _stream_tracker: None,
            _loop_tracker: ConvergenceTracker::new(),
            _solve_tracker: ConvergenceTracker::new(),
            _step_tracker: StepTracker::new(),
            _run_converged: true,
            _run_max_residual: 0.0,
            _run_worst_block: None,
            _run_truncated_at: None,
            diagnostics: None,
            diagnostics_history: None,
            _wct_trace: None,
            _pending_ops: Rc::new(RefCell::new(Vec::new())),
            logger,
        };

        for block in blocks { let _ = sim.add_block(block); }
        for conn in connections { sim.add_connection(conn); }

        sim._check_blocks_are_managed();
        sim._assemble_graph();

        // Log solver info (matches pathsim constructor logging)
        sim.logger.info(&format!(
            "SOLVER (dyn. blocks: {}) -> {} (adaptive: {}, explicit: {})",
            sim._blocks_dyn_indices.len(), sim.engine.type_name,
            if sim.engine.is_adaptive { "True" } else { "False" },
            if sim.engine.is_explicit { "True" } else { "False" },
        ));

        sim
    }

    // -- __contains__ equivalent --

    pub fn contains_block(&self, block: &BlockRef) -> bool {
        self.blocks.iter().any(|b| Rc::ptr_eq(b, block))
    }

    pub fn is_active(&self) -> bool { self._active }

    /// Take a port-resolution error recorded during graph assembly, if any.
    /// The pybindings run wrappers surface it as a Python exception instead of
    /// the engine panicking on a bad port alias.
    pub fn take_assembly_error(&mut self) -> Option<crate::error::SimError> {
        self._assembly_error.take()
    }

    pub fn has_loops(&self) -> bool {
        self.graph.as_ref().map(|g| g.has_loops).unwrap_or(false)
    }

    pub fn dag_depth(&self) -> usize {
        self.graph.as_ref().map(|g| g.depth().0).unwrap_or(0)
    }

    pub fn loop_depth(&self) -> usize {
        self.graph.as_ref().map(|g| g.depth().1).unwrap_or(0)
    }

    pub fn num_dynamic_blocks(&self) -> usize {
        self._blocks_dyn_indices.len()
    }

    // -- size --

    pub fn size(&self) -> (usize, usize) {
        let mut total_n = 0;
        let mut total_nx = 0;
        for block in &self.blocks {
            let (n, nx) = block.borrow().size();
            total_n += n;
            total_nx += nx;
        }
        (total_n, total_nx)
    }

    // -- plot --

    pub fn plot(&self) {
        for block in &self.blocks {
            if block.borrow().is_active() { block.borrow().plot(); }
        }
    }

    // ==================================================================================
    // Adding/removing system components
    // ==================================================================================

    pub fn add_block(&mut self, block: BlockRef) -> Result<(), SimError> {
        // Duplicate check
        if self.contains_block(&block) {
            return Err(SimError::DuplicateBlock);
        }

        let idx = self.blocks.len();

        // Initialize solver on block if factory is set
        if let Some(ref factory) = self.solver_factory {
            block.borrow_mut().set_solver_from(&**factory);
        }

        // Track dynamic and eventful blocks by index
        {
            let b = block.borrow();
            if b.has_engine() {
                self._blocks_dyn_indices.push(idx);
            }
            if b.has_events() {
                self._blocks_evt_indices.push(idx);
            }
        }

        self.blocks.push(block);

        if self.graph.is_some() {
            self._graph_dirty = true;
        }
        Ok(())
    }

    pub fn remove_block(&mut self, block: &BlockRef) -> Result<(), SimError> {
        if let Some(pos) = self.blocks.iter().position(|b| Rc::ptr_eq(b, block)) {
            self.blocks.remove(pos);
            self._rebuild_block_indices();
            if self.graph.is_some() {
                self._graph_dirty = true;
            }
            Ok(())
        } else {
            Err(SimError::BlockNotFound)
        }
    }

    pub fn add_connection(&mut self, connection: ConnectionRef) {
        self.connections.push(connection);
        if self.graph.is_some() { self._graph_dirty = true; }
    }

    pub fn remove_connection(&mut self, connection: &ConnectionRef) -> Result<(), SimError> {
        let len_before = self.connections.len();
        self.connections.retain(|c| !Rc::ptr_eq(c, connection));
        if self.connections.len() == len_before {
            Err(SimError::ConnectionNotFound)
        } else {
            if self.graph.is_some() { self._graph_dirty = true; }
            Ok(())
        }
    }

    pub fn add_event(&mut self, event: SimEventRef) {
        self.events.push(event);
    }

    pub fn remove_event(&mut self, event: &SimEventRef) -> Result<(), SimError> {
        let len_before = self.events.len();
        self.events.retain(|e| !Rc::ptr_eq(e, event));
        if self.events.len() == len_before {
            Err(SimError::EventNotFound)
        } else {
            Ok(())
        }
    }

    fn _rebuild_block_indices(&mut self) {
        self._blocks_dyn_indices.clear();
        self._blocks_evt_indices.clear();
        for (i, block) in self.blocks.iter().enumerate() {
            let b = block.borrow();
            if b.has_engine() { self._blocks_dyn_indices.push(i); }
            if b.has_events() { self._blocks_evt_indices.push(i); }
        }
    }

    // ==================================================================================
    // Schedule assembly
    // ==================================================================================

    fn _assemble_graph(&mut self) {
        // Reset all block inputs to clear stale values.
        for block in &self.blocks {
            let b = block.borrow_mut();
            b.inputs.reset();
        }

        // Resolve all port indices FIRST: this sizes the input/output registers
        // and resolves element indices, which the port-granular feedthrough
        // assembly below reads. A failure here is a bad port alias (user error):
        // remember it so `_update` can bail cleanly and the run wrappers surface it.
        let mut port_err = None;
        for conn in &self.connections {
            if let Err(e) = conn.resolve_ports() {
                port_err = Some(e);
                break;
            }
        }
        self._assembly_error = port_err;

        // A bad port alias was found: stop before `assemble_graph_from`, which
        // would call `_get_input_indices` on the unresolved port and panic in the
        // hot path. Leaving the graph unbuilt lets `_update` bail at its
        // `_assembly_error` check and the run wrappers surface a clean ValueError.
        if let Some(ref e) = self._assembly_error {
            // Log the configuration error through the sink (Logger::error, which
            // reports regardless of the `log` flag — issue #29). The run wrapper
            // still raises it as a Python exception.
            self.logger.error(&format!("graph assembly failed: {e}"));
            self._graph_dirty = false;
            return;
        }

        // Single shared port-granular assembly (see `assemble_graph_from`).
        self.graph = Some(assemble_graph_from(&self.blocks, &self.connections));
        self._graph_dirty = false;

        // Create boosters for loop-closing connections.  Each booster
        // inherits its WRMS weights from the simulation-level outer
        // tolerances (kept on the dummy `engine`) so the algebraic-loop
        // convergence test scales consistently with the time integrator.
        let booster_atol = self.engine.tolerance_lte_abs;
        let booster_rtol = self.engine.tolerance_lte_rel;
        self.boosters.clear();
        if let Some(ref graph) = self.graph {
            if graph.has_loops {
                for &conn_idx in graph.loop_closing_connections() {
                    if let Some(conn) = self.connections.get(conn_idx) {
                        self.boosters.push(ConnectionBooster::new(
                            conn.clone(), booster_atol, booster_rtol,
                        ));
                    }
                }
            }
        }

        // Log system info (matches pathsim: logged at graph assembly time).
        // Skip the format! allocations entirely when logging is disabled —
        // every live mutation re-runs _assemble_graph, so this is on the
        // hot mutation path with `log=False`.
        if self.logger.enabled {
            let num_dyn = self._blocks_dyn_indices.len();
            let num_evt = self._blocks_evt_indices.len();
            self.logger.info(&format!(
                "BLOCKS (total: {}, dynamic: {}, static: {}, eventful: {})",
                self.blocks.len(), num_dyn, self.blocks.len() - num_dyn, num_evt
            ));
            if let Some(ref graph) = self.graph {
                let (ad, ld) = graph.depth();
                self.logger.info(&format!(
                    "GRAPH (nodes: {}, edges: {}, alg. depth: {}, loop depth: {})",
                    self.blocks.len(), self.connections.len(), ad, ld
                ));
            }
        }
    }

    // ==================================================================================
    // Topology checks
    // ==================================================================================

    fn _check_blocks_are_managed(&self) {
        for conn in &self.connections {
            let conn_blocks = conn.get_blocks();
            for conn_block in &conn_blocks {
                let found = self.blocks.iter().any(|b| Rc::ptr_eq(b, conn_block));
                if !found {
                    crate::utils::sink::warn("[fastsim WARNING] block in connection but not in simulation blocks list");
                }
            }
        }
    }

    // ==================================================================================
    // Solver management
    // ==================================================================================

    /// Reconfigure the Anderson history depth on every dynamic block's
    /// optimizer (and on the dummy stage-sync engine).  No-op for blocks with
    /// no optimizer or with a pure Newton optimizer.
    pub fn set_optimizer_history(&mut self, m: usize) {
        self.engine.set_optimizer_history(m);
        for block in &self.blocks {
            if let Some(engine) = block.borrow_mut().engine.as_mut() {
                engine.set_optimizer_history(m);
            }
        }
    }

    /// Change the solver for all blocks.
    /// Mirrors Python `_set_solver`.
    pub fn set_solver(&mut self, factory: Box<dyn Fn(&[f64]) -> Solver>) {
        // Create new dummy engine
        self.engine = factory(&[0.0]);
        self.solver_factory = Some(factory);

        // Reinitialize all blocks
        self._blocks_dyn_indices.clear();
        for (i, block) in self.blocks.iter().enumerate() {
            if let Some(ref f) = self.solver_factory {
                block.borrow_mut().set_solver_from(&**f);
            }
            if block.borrow().has_engine() {
                self._blocks_dyn_indices.push(i);
            }
        }

        self.logger.info(&format!(
            "SOLVER (dyn. blocks: {}) -> {} (adaptive: {}, explicit: {})",
            self._blocks_dyn_indices.len(), self.engine.type_name,
            if self.engine.is_adaptive { "True" } else { "False" },
            if self.engine.is_explicit { "True" } else { "False" },
        ));
    }

    // ==================================================================================
    // Reset
    // ==================================================================================

    pub fn reset(&mut self, time: f64) {
        self.logger.info(&format!("RESET (time: {})", time));
        self._active = true;
        self.time = time;
        self.engine.reset(None);

        for block in &self.blocks {
            block.borrow_mut().reset();
        }
        for event in &self.events {
            event.borrow_mut().reset();
        }

        // Reset trackers
        self._loop_tracker.reset();
        self._solve_tracker.reset();
        self._step_tracker.reset();
        if self.diagnostics.is_some() {
            self.diagnostics = Some(Diagnostics::new());
        }
        if let Some(ref mut hist) = self.diagnostics_history {
            hist.clear();
        }

        self._update(self.time);
    }

    // ==================================================================================
    // Steady state
    // ==================================================================================

    /// Find DC operating point by temporarily switching to SteadyState solver.
    pub fn steadystate(&mut self, reset: bool) {
        if reset { self.reset(0.0); }

        // Save current solver factory
        let saved_factory = self.solver_factory.take();

        // Switch to SteadyState solver
        let ss_factory = crate::solvers::factories::steadystate_factory();
        self.set_solver(ss_factory);

        // Iterate solve loop until converged.  Per-block residuals come back
        // WRMS-scaled (matching the time-integrator's convergence criterion);
        // we check the max against the same `NLS_COEF` used by `_solve` /
        // `_loops` so steady-state and dynamic runs share a single tolerance
        // semantics.
        for _ in 0..self.iterations_max {
            self._update(self.time);

            let mut max_error: f64 = 0.0;
            for &idx in &self._blocks_dyn_indices {
                if let Some(block) = self.blocks.get(idx) {
                    if !block.borrow().is_active() { continue; }
                    let err = block.borrow_mut().solve(self.time, self.dt);
                    if err > max_error { max_error = err; }
                }
            }

            if max_error < NLS_COEF {
                break;
            }
        }

        // Sample result
        self._sample(self.time, self.dt);

        // Restore original solver
        if let Some(factory) = saved_factory {
            self.set_solver(factory);
        }
    }

    // ==================================================================================
    // Periodic steady state (shooting)
    // ==================================================================================

    /// Find the periodic steady-state limit cycle of period `T` by Anderson-
    /// accelerated shooting on the period map `g(x_0) = x(T; x_0)`.
    ///
    /// Algorithm (one outer iteration):
    ///   1. Integrate the system over `[0, T]` via the inner ODE solver
    ///      (just a regular transient `run(period, ...)`).
    ///   2. Per dynamic block, run one matrix-free Anderson step on the
    ///      shooting map via `Solver::pss_close_period`.  Mutates the
    ///      block's `pss_ext.x_start` toward the fixed point and copies
    ///      it into `engine.x` as the IC for the next period.
    ///   3. Check WRMS-scaled residual `‖x(T) − x(0)‖` against `NLS_COEF`
    ///      across all dynamic blocks (same convergence semantics as
    ///      every other implicit-stage / steady-state residual).
    ///   4. If not converged: reset `sim.time = 0`, reset event schedules,
    ///      reset the dummy stage-sync engine.  Crucially do NOT call
    ///      `block.reset()` — that would clobber the Anderson-updated
    ///      state with `initial_value`.
    ///
    /// After convergence one final transient run is performed over `[0, T]`
    /// so Scope blocks record the limit-cycle trajectory.  The original
    /// solver factory is restored on exit.
    ///
    /// `inner_factory` is the ODE solver used during each period integration
    /// (e.g. `rkdp54_factory`, `esdirk43_factory`).  DAE blocks pass through
    /// transparently — their `engine_postprocess` installs the appropriate
    /// `StageBuilder` on the PSS-augmented engine after the factory returns.
    ///
    /// `reset=true` calls `self.reset(0.0)` first, restoring every block to
    /// its `initial_value`.  `reset=false` starts shooting from whatever
    /// state the engine currently holds (the typical case after running a
    /// transient warm-up to seed a good initial guess).
    pub fn periodic_steady_state(
        &mut self,
        period: f64,
        inner_factory: crate::solvers::factories::SolverFactory,
        adaptive: bool,
        reset: bool,
    ) -> RunStats {
        assert!(period > 0.0, "periodic_steady_state: period must be positive, got {}", period);

        if reset { self.reset(0.0); }

        // Save current solver factory for restoration on exit.  `take()`
        // because `set_solver` would otherwise drop it.
        let saved_factory = self.solver_factory.take();

        // Swap to PSS-augmented inner factory.  `set_solver` rebuilds every
        // dyn block's engine, copying the *current* `x` across via
        // `Block::set_solver_from` (block.rs:271-274).
        let pss_factory = crate::solvers::pss::periodic_steady_state_factory(
            inner_factory, period, OPT_HISTORY,
        );
        self.set_solver(pss_factory);

        // Sync `pss_ext.x_start` to the engine's current `x` on every dyn
        // block.  The factory itself initializes `x_start` from the block's
        // `initial_value`, but after `set_solver_from` runs, `engine.x` may
        // hold a different state (e.g. the result of a warm-up transient).
        // We want shooting to start from *that* state, not from `initial_value`.
        for &idx in &self._blocks_dyn_indices {
            let block = self.blocks[idx].borrow_mut();
            if let Some(engine) = block.engine.as_mut() {
                let current = engine.x.clone();
                if let Some(ext) = engine.pss_ext.as_mut() {
                    ext.x_start = current;
                    ext.anderson.reset();
                    ext.closed_once = false;
                }
            }
        }

        let mut total_stats = RunStats::default();

        // Silence the simulation logger across the inner `run(period, ...)`
        // calls so the user sees one summary line per PSS call, not one
        // STARTING/FINISHED TRANSIENT block per shooting iteration.  We
        // restore the flag before our own summary line and the warning
        // (if any), and before any subsequent code on the simulation.
        let logger_was_enabled = self.logger.enabled;
        let pss_start = std::time::Instant::now();
        self.logger.enabled = logger_was_enabled;
        self.logger.info(&format!(
            "STARTING -> PSS (period: {:.4}, iter cap: {})",
            period, SIM_PSS_ITERATIONS_MAX,
        ));
        self.logger.enabled = false;

        let mut converged = false;
        let mut iterations_used = 0usize;

        for it in 0..SIM_PSS_ITERATIONS_MAX {
            iterations_used = it + 1;

            // Integrate one period from the current x_start.  `reset=false`:
            // we manage time and events explicitly below.
            let stats = self.run(period, false, adaptive);
            total_stats.total_steps += stats.total_steps;
            total_stats.successful_steps += stats.successful_steps;
            total_stats.rejected_steps += stats.rejected_steps;
            total_stats.total_evals += stats.total_evals;
            total_stats.total_solver_its += stats.total_solver_its;
            total_stats.wall_time_secs += stats.wall_time_secs;

            // Per-block shooting update + WRMS residual collection.
            self._solve_tracker.begin_iteration();
            let mut max_err: f64 = 0.0;
            for &idx in &self._blocks_dyn_indices {
                let block = self.blocks[idx].borrow_mut();
                if !block.is_active() { continue; }
                if let Some(engine) = block.engine.as_mut() {
                    let err = engine.pss_close_period();
                    self._solve_tracker.record(idx, err);
                    if err > max_err { max_err = err; }
                }
            }

            if max_err < NLS_COEF {
                converged = true;
                self._solve_tracker.iterations = iterations_used;
                break;
            }

            // Not converged — reset time + events for the next iteration.
            // NOT block.reset() (would clobber Anderson state with IV).
            self.time = 0.0;
            for event in &self.events {
                event.borrow_mut().reset();
            }
            for &idx in &self._blocks_evt_indices {
                let b = self.blocks[idx].borrow();
                for event in &b.events {
                    event.borrow_mut().reset();
                }
            }
            self.engine.reset(None);
        }

        self.logger.enabled = logger_was_enabled;
        let pss_ms = pss_start.elapsed().as_secs_f64() * 1000.0;
        if converged {
            self.logger.info(&format!(
                "FINISHED -> PSS (iterations: {}, runtime: {:.1} ms)",
                iterations_used, pss_ms,
            ));
        } else {
            self.logger.warning(&format!(
                "PSS did not converge (iterations: {}, WRMS residual: {:.2e}, threshold: {:.2e}, runtime: {:.1} ms)",
                iterations_used, self._solve_tracker.max_error, NLS_COEF, pss_ms,
            ));
        }

        // Final transient run over one period so Scope blocks record the
        // converged limit-cycle trajectory.  `pss_close_period` is *never*
        // called inside `run()` — it's only invoked by this outer loop —
        // so leaving the PSS-augmented engine in place is fine.
        //
        // Reset recorder blocks (`role.is_rec`) first so their accumulated
        // samples from the shooting iterations don't pollute the final
        // trajectory.  Their `reset_fn` only clears recordings — no engine
        // state involved (recorders are not dynamic).
        for block in &self.blocks {
            if block.borrow().role.is_rec {
                block.borrow_mut().reset();
            }
        }
        self.time = 0.0;
        for event in &self.events {
            event.borrow_mut().reset();
        }
        for &idx in &self._blocks_evt_indices {
            let b = self.blocks[idx].borrow();
            for event in &b.events {
                event.borrow_mut().reset();
            }
        }
        self.engine.reset(None);
        // Final sample-run is part of the PSS call from the user's POV —
        // silence its inner transient log so the user sees only the PSS
        // summary line emitted by the tracker above.
        self.logger.enabled = false;
        let final_stats = self.run(period, false, adaptive);
        self.logger.enabled = logger_was_enabled;
        total_stats.total_steps += final_stats.total_steps;
        total_stats.successful_steps += final_stats.successful_steps;
        total_stats.rejected_steps += final_stats.rejected_steps;
        total_stats.total_evals += final_stats.total_evals;
        total_stats.total_solver_its += final_stats.total_solver_its;
        total_stats.wall_time_secs += final_stats.wall_time_secs;

        // Restore original solver factory.
        if let Some(factory) = saved_factory {
            self.set_solver(factory);
        }

        // Report the shooting-loop outcome (not the last inner transient's),
        // so a non-converged PSS surfaces via RunStats + a Python warning.
        total_stats.converged = converged;
        total_stats.max_residual = self._solve_tracker.max_error;
        total_stats
    }

    // ==================================================================================
    // Active events (including internal block events)
    // ==================================================================================

    fn _estimate_events(&self, t: f64) -> Option<f64> {
        let mut min_dt: Option<f64> = None;
        let mut consider = |event: &SimEventRef| {
            let e = event.borrow();
            if !e.is_active() { return; }
            if let Some(dt_est) = e.estimate(t) {
                if dt_est > 0.0 {
                    min_dt = Some(min_dt.map_or(dt_est, |m: f64| m.min(dt_est)));
                }
            }
        };
        for event in &self.events { consider(event); }
        for &idx in &self._blocks_evt_indices {
            if let Some(block) = self.blocks.get(idx) {
                let b = block.borrow();
                for event in &b.events { consider(event); }
            }
        }
        min_dt
    }

    /// Resolve any events that are already detected at the current time. Used
    /// once per `run()` before the main loop starts, so the Vec alloc for the
    /// clones is amortized across the entire run.
    fn _resolve_initial_events(&mut self) {
        self._detect_into_scratch(self.time);
        for i in 0..self._detected_scratch.len() {
            let event = self._detected_scratch[i].0.clone();
            event.borrow_mut().resolve(self.time);
            self._update(self.time);
        }
        // A StopSimulation raised during the initial system evaluation or an
        // initial event action lands here (the run loop won't start).
        if take_stop_requested() {
            self._active = false;
        }
    }

    /// Populate `self._detected_scratch` with `(event, close, ratio)` tuples
    /// for every active event that fired at time `t`, sorted by `ratio` ascending.
    fn _detect_into_scratch(&mut self, t: f64) {
        self._detected_scratch.clear();

        // External events
        for event in &self.events {
            if !event.borrow().is_active() { continue; }
            let (det, close, ratio) = event.borrow_mut().detect(t);
            if det {
                self._detected_scratch.push((event.clone(), close, ratio));
            }
        }

        // Block-internal events
        for &idx in &self._blocks_evt_indices {
            if let Some(block) = self.blocks.get(idx) {
                let b = block.borrow();
                for event in &b.events {
                    if !event.borrow().is_active() { continue; }
                    let (det, close, ratio) = event.borrow_mut().detect(t);
                    if det {
                        self._detected_scratch.push((event.clone(), close, ratio));
                    }
                }
            }
        }

        self._detected_scratch.sort_by(|a, b| {
            a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // ==================================================================================
    // System equation evaluation
    // ==================================================================================

    fn _update(&mut self, t: f64) {
        if self._graph_dirty {
            self._assemble_graph();
        }
        // A bad port alias was found during assembly: stop before the DAG
        // transfer (which would otherwise panic resolving the alias).
        if self._assembly_error.is_some() {
            self._active = false;
            return;
        }
        self._dag(t);
        if let Some(ref graph) = self.graph {
            if graph.has_loops {
                self._loops(t);
            }
        }
    }

    fn _dag(&self, t: f64) {
        if let Some(ref graph) = self.graph {
            for (block_ids, conn_ids) in graph.dag_iter() {
                for &block_idx in block_ids {
                    let b = self.blocks[block_idx].borrow_mut();
                    if b._active { b.update(t); }
                }
                for &conn_idx in conn_ids {
                    self.connections[conn_idx].update();
                }
            }
        }
    }

    fn _loops(&mut self, t: f64) {
        for booster in &mut self.boosters {
            booster.reset();
        }

        let graph = self.graph.as_ref().unwrap();
        let (_, loop_depth) = graph.depth();

        // NOTE: 1-based `1..iterations_max` runs one fewer pass than `_solve`'s
        // `0..iterations_max`. This is deliberate pathsim parity (its loop is
        // `range(1, iterations_max)`); do not "fix" the bound without re-checking
        // trajectory-match, as it changes the fixed-point iteration count.
        for iteration in 1..self.iterations_max {
            for d in 0..loop_depth {
                for &block_idx in graph.loop_blocks(d) {
                    if let Some(block) = self.blocks.get(block_idx) {
                        if block.borrow().is_active() {
                            block.borrow_mut().update(t);
                        }
                    }
                }
                for &conn_idx in graph.loop_connections(d) {
                    if let Some(conn) = self.connections.get(conn_idx) {
                        if conn.is_active() { conn.update(); }
                    }
                }
            }

            // Step boosters and track convergence.  Boosters return a
            // WRMS-scaled residual (matching the implicit-stage criterion);
            // we check it against the same unitless `NLS_COEF` so both the
            // algebraic loop and the inner Newton-Anderson solve converge
            // proportionally to `(atol + rtol·|x|)`.
            self._loop_tracker.begin_iteration();
            for (i, booster) in self.boosters.iter_mut().enumerate() {
                let res = booster.update();
                self._loop_tracker.record(i, res);
            }

            if self._loop_tracker.converged(NLS_COEF) {
                self._loop_tracker.iterations = iteration;
                return;
            }
        }

        self._loop_tracker.iterations = self.iterations_max;
        // Record the failure into the per-run outcome so RunStats reports it and
        // the Python layer can emit an unconditional FastSimConvergenceWarning
        // (issue #27) — instead of only this log line, which is off by default.
        let worst_booster = self._loop_tracker.errors.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&id, _)| id);
        let max_err = self._loop_tracker.max_error;
        self._record_nonconvergence(max_err, worst_booster);
        let details = self._loop_tracker.details(&|id| format!("Booster_{}", id));
        self.logger.warning(&format!(
            "algebraic loop not converged (iters: {}, err: {:.2e})\n{}",
            self.iterations_max, max_err, details.join("\n")
        ));
    }

    // ==================================================================================
    // Implicit solver loop
    // ==================================================================================

    /// Inner Newton-Anderson loop on every dynamic block's stage builder.
    ///
    /// Each block's `solve(t, dt)` returns a WRMS-scaled residual norm
    /// (computed via `Solver::wrms_norm` / `wrms_norm_diff` against the
    /// per-block `tolerance_lte_abs` / `tolerance_lte_rel`).  We track the
    /// max across blocks and check it against the unitless `NLS_COEF` —
    /// the same threshold used by `_loops` for the algebraic-loop boosters
    /// and by the steady-state operating-point loop.  Tying inner-Newton
    /// accuracy to the outer step's `(atol + rtol·|x|)` weights replaces
    /// the legacy absolute `tolerance_fpi` knob, which over-converged
    /// well-scaled state components on multi-scale stiff problems.
    fn _solve(&mut self, t: f64, dt: f64) -> (bool, usize, usize) {
        let mut total_evals: usize = 0;
        for it in 0..self.iterations_max {
            self._update(t);
            total_evals += 1;

            self._solve_tracker.begin_iteration();
            for &idx in &self._blocks_dyn_indices {
                if let Some(block) = self.blocks.get(idx) {
                    if !block.borrow().is_active() { continue; }
                    let err = block.borrow_mut().solve(t, dt);
                    self._solve_tracker.record(idx, err);
                }
            }

            if self._solve_tracker.converged(NLS_COEF) {
                self._solve_tracker.iterations = it + 1;
                return (true, total_evals, it + 1);
            }
        }

        self._solve_tracker.iterations = self.iterations_max;
        (false, total_evals, self.iterations_max)
    }

    // ==================================================================================
    // Timestepping helpers
    // ==================================================================================

    fn _revert(&mut self, t: f64) {
        self.engine.revert();
        for &idx in &self._blocks_dyn_indices {
            if let Some(block) = self.blocks.get(idx) {
                if block.borrow().is_active() {
                    block.borrow_mut().revert();
                }
            }
        }
        self._update(t);
    }

    fn _sample(&self, t: f64, dt: f64) {
        for block in &self.blocks {
            if block.borrow().is_active() {
                block.borrow_mut().sample(t, dt);
            }
        }
    }

    fn _buffer(&mut self, t: f64, dt: f64) {
        // External events
        for event in &self.events {
            if event.borrow().is_active() {
                event.borrow_mut().buffer(t);
            }
        }
        // Block-internal events
        for &idx in &self._blocks_evt_indices {
            if let Some(block) = self.blocks.get(idx) {
                let b = block.borrow();
                for event in &b.events {
                    if event.borrow().is_active() {
                        event.borrow_mut().buffer(t);
                    }
                }
            }
        }
        self.engine.buffer(dt);
        for &idx in &self._blocks_dyn_indices {
            if let Some(block) = self.blocks.get(idx) {
                if block.borrow().is_active() {
                    block.borrow_mut().buffer(dt);
                }
            }
        }
    }

    fn _step(&mut self, t: f64, dt: f64) -> (bool, f64, Option<f64>) {
        self._step_tracker.reset();
        for &idx in &self._blocks_dyn_indices {
            if let Some(block) = self.blocks.get(idx) {
                if !block.borrow().is_active() { continue; }
                let (suc, err, scl) = block.borrow_mut().step(t, dt);
                self._step_tracker.record(idx, suc, err, scl);
            }
        }
        (self._step_tracker.success, self._step_tracker.max_error, self._step_tracker.min_scale)
    }

    // ==================================================================================
    // Main timestep
    // ==================================================================================

    /// Apply any mutations that callbacks have queued via `_pending_ops`.
    /// Called at the top of every `timestep` so the previous step's
    /// callbacks land before the next step runs (and the next step's
    /// graph rebuild reflects the new topology).
    fn _apply_pending_ops(&mut self) {
        if self._pending_ops.borrow().is_empty() { return; }
        let ops: Vec<PendingOp> = self._pending_ops.borrow_mut().drain(..).collect();

        // Bucket by op type so we can apply removes en bloc (single retain
        // pass + single index rebuild) instead of per-call O(N) work.
        let mut add_blocks: Vec<BlockRef> = Vec::new();
        let mut rm_block_ptrs: std::collections::HashSet<*const FastCell<Block>> =
            std::collections::HashSet::new();
        let mut add_conns: Vec<ConnectionRef> = Vec::new();
        let mut rm_conn_ptrs: std::collections::HashSet<*const Connection> =
            std::collections::HashSet::new();
        let mut add_events: Vec<SimEventRef> = Vec::new();
        let mut rm_event_ptrs: std::collections::HashSet<*const FastCell<dyn SimEvent>> =
            std::collections::HashSet::new();

        for op in ops {
            match op {
                PendingOp::AddBlock(b) => add_blocks.push(b),
                PendingOp::RemoveBlock(b) => { rm_block_ptrs.insert(Rc::as_ptr(&b)); }
                PendingOp::AddConnection(c) => add_conns.push(c),
                PendingOp::RemoveConnection(c) => { rm_conn_ptrs.insert(Rc::as_ptr(&c)); }
                PendingOp::AddEvent(e) => add_events.push(e),
                PendingOp::RemoveEvent(e) => { rm_event_ptrs.insert(Rc::as_ptr(&e)); }
            }
        }

        let mut topology_changed = false;

        // Removes first (in case the user adds-then-removes the same item).
        if !rm_block_ptrs.is_empty() {
            let before = self.blocks.len();
            self.blocks.retain(|b| !rm_block_ptrs.contains(&Rc::as_ptr(b)));
            if self.blocks.len() != before { topology_changed = true; }
        }
        if !rm_conn_ptrs.is_empty() {
            let before = self.connections.len();
            self.connections.retain(|c| !rm_conn_ptrs.contains(&Rc::as_ptr(c)));
            if self.connections.len() != before { topology_changed = true; }
        }
        if !rm_event_ptrs.is_empty() {
            self.events.retain(|e| !rm_event_ptrs.contains(&Rc::as_ptr(e)));
        }

        // Adds: tight loop, push directly.  Solver-init is per-block (intrinsic
        // cost; cannot be batched because each block owns its own solver state).
        for block in add_blocks {
            if self.contains_block(&block) { continue; }
            if let Some(ref factory) = self.solver_factory {
                block.borrow_mut().set_solver_from(&**factory);
            }
            self.blocks.push(block);
            topology_changed = true;
        }
        for conn in add_conns {
            self.connections.push(conn);
            topology_changed = true;
        }
        for evt in add_events {
            self.events.push(evt);
        }

        // Single index rebuild + dirty mark, regardless of how many ops landed.
        if topology_changed {
            self._rebuild_block_indices();
            if self.graph.is_some() {
                self._graph_dirty = true;
            }
        }
    }

    /// Advance the simulation by one timestep.
    pub fn timestep(&mut self, dt: Option<f64>, adaptive: bool) -> (bool, f64, f64, usize, usize) {
        // Drain any callback-queued mutations from the previous step before
        // measuring this step's timing.  The graph rebuild that follows is
        // part of this timestep's cost, which is exactly what the WCT trace
        // should attribute to it.
        self._apply_pending_ops();

        // Per-step wall-clock timing — only sampled when `_wct_trace` is
        // enabled.  `Instant::now()` is on the order of 30–50 ns on Apple
        // Silicon, so the overhead is well below any per-step cost we care
        // to measure.  When the trace is disabled (`None`) the branch
        // predicts trivially and the entire instrumentation compiles to
        // a single `if let None`.
        let _wct_start = if self._wct_trace.is_some() { Some(Instant::now()) } else { None };

        let is_adaptive = adaptive && self.engine.is_adaptive;
        let is_implicit = self.engine.is_implicit;

        let mut total_evals: usize = 0;
        let mut total_solver_its: usize = 0;
        let mut error_norm: f64 = 0.0;
        let mut scale: f64 = 1.0;
        let mut success = true;

        let dt = dt.unwrap_or(self.dt);

        // Buffer events and dynamic blocks
        self._buffer(self.time, dt);

        // Solver stages iteration (skip if no dynamic blocks)
        if !self._blocks_dyn_indices.is_empty() {
            let n_stages = self.engine.n_stages();
            for stage_idx in 0..n_stages {
                let time_stage = self.engine.stage_time(stage_idx, self.time, dt);
                self.engine._stage = stage_idx;
                // Synchronize stage counter + apply min-correction predictor
                // (for implicit stages) so Newton starts close to the fixed point.
                for &idx in &self._blocks_dyn_indices {
                    if let Some(block) = self.blocks.get(idx) {
                        if let Some(ref mut engine) = block.borrow_mut().engine {
                            engine._stage = stage_idx;
                            if is_implicit {
                                engine.apply_stage_predictor(dt);
                            }
                        }
                    }
                }

                if is_implicit {
                    let (sol_success, evals, solver_its) = self._solve(time_stage, dt);
                    total_evals += evals;
                    total_solver_its += solver_its;
                    if !sol_success {
                        if is_adaptive {
                            self._revert(self.time);
                            if let (Some(start), Some(trace)) = (_wct_start, self._wct_trace.as_mut()) {
                                trace.push(start.elapsed().as_secs_f64());
                            }
                            return (false, 0.0, 0.5, total_evals + 1, total_solver_its);
                        }
                        // Non-adaptive (or floor) implicit non-convergence: the
                        // step proceeds on the best-so-far state (fail-open per
                        // issue #27 policy), but record it so RunStats is
                        // truthful and the Python layer warns.
                        let worst = self._solve_tracker.errors.iter()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                            .map(|(&id, _)| id);
                        let max_err = self._solve_tracker.max_error;
                        self._record_nonconvergence(max_err, worst);
                        self.logger.warning(&format!(
                            "implicit solver not converged at t={:.6}", time_stage
                        ));
                    }
                } else {
                    self._update(time_stage);
                    total_evals += 1;
                }

                // Step dynamic blocks (last stage's result wins, mirrors Python)
                let (step_success, step_error, _) = self._step(time_stage, dt);
                success = step_success;
                error_norm = step_error;
                scale = self._step_tracker.scale();

                if !success && is_adaptive {
                    self._revert(self.time);
                    if let (Some(start), Some(trace)) = (_wct_start, self._wct_trace.as_mut()) {
                        trace.push(start.elapsed().as_secs_f64());
                    }
                    return (false, error_norm, scale, total_evals + 1, total_solver_its);
                }
            }
        }

        let time_dt = self.time + dt;

        // Evaluate system equation before event check
        self._update(time_dt);
        total_evals += 1;

        // Handle detected events chronologically.  When `dt` is already at the
        // `dt_min` floor, force-resolve at the bracketed crossing rather than
        // reverting into another non-close retry (livelock guard, issue #25).
        let at_step_floor = is_adaptive && dt <= self.dt_min * (1.0 + 1e-9);
        let evt_ratio =
            self._resolve_events(time_dt, dt, is_adaptive, at_step_floor, &mut total_evals);

        // A user callback (event action or block update/source) may have raised
        // pathsim's StopSimulation, which the pybindings layer converts into a
        // cooperative stop request. Observe it here, after all of this step's
        // callbacks have run, and terminate cleanly via `_active`.
        if take_stop_requested() {
            self._active = false;
        }

        if let Some(ratio) = evt_ratio {
            if let (Some(start), Some(trace)) = (_wct_start, self._wct_trace.as_mut()) {
                trace.push(start.elapsed().as_secs_f64());
            }
            return (false, error_norm, ratio, total_evals + 1, total_solver_its);
        }

        self._update_diagnostics(time_dt);

        // Sample data
        self._sample(time_dt, dt);

        // Increment global time
        self.time = time_dt;

        // Record per-step wall-clock if tracing enabled.
        if let (Some(start), Some(trace)) = (_wct_start, self._wct_trace.as_mut()) {
            trace.push(start.elapsed().as_secs_f64());
        }

        (success, error_norm, scale, total_evals, total_solver_its)
    }

    /// Detect and resolve events scheduled for this step. Returns `Some(ratio)`
    /// if the caller must revert and early-return (adaptive mode with a
    /// non-close event — see two-phase comment below); returns `None` when
    /// events are fully resolved (or none detected).
    ///
    /// Increments `*evals` for each extra `_update` performed (one, if any
    /// events were resolved).
    fn _resolve_events(
        &mut self, time_dt: f64, dt: f64, is_adaptive: bool, force_resolve: bool, evals: &mut usize,
    ) -> Option<f64> {
        self._detect_into_scratch(time_dt);
        if self._detected_scratch.is_empty() {
            return None;
        }
        if is_adaptive {
            // Two-phase: if any event is not yet close, revert without resolving
            // anything — otherwise a close event in the same step would get its
            // func_act called twice (once here, once after the retry step).
            let non_close_ratio = self._detected_scratch
                .iter()
                .find(|(_, close, _)| !*close)
                .map(|(_, _, r)| *r);
            match non_close_ratio {
                // Normal case: shrink dt and retry to bring the event within
                // tolerance on the next step.
                Some(ratio) if !force_resolve => {
                    self._revert(self.time);
                    return Some(ratio);
                }
                // `force_resolve`: dt is already pinned at `dt_min`, so shrinking
                // again would reproduce the same non-close detection forever
                // (livelock, issue #25). Accept the bracketed crossing at the
                // step floor instead — resolve each event at its interpolated
                // crossing time `t + ratio·dt`, exactly like the non-adaptive
                // path, then advance.
                Some(_) => {
                    let time = self.time;
                    for (event, _, ratio) in &self._detected_scratch {
                        event.borrow_mut().resolve(time + ratio * dt);
                    }
                }
                None => {
                    for (event, _, _) in &self._detected_scratch {
                        event.borrow_mut().resolve(time_dt);
                    }
                }
            }
        } else {
            let time = self.time;
            for (event, _, ratio) in &self._detected_scratch {
                event.borrow_mut().resolve(time + ratio * dt);
            }
        }
        // Single update after all events resolved
        self._update(time_dt);
        *evals += 1;
        None
    }

    /// Capture current diagnostics snapshot (if enabled) and append to history.
    fn _update_diagnostics(&mut self, time_dt: f64) {
        if self.diagnostics.is_none() {
            return;
        }
        let diag = Diagnostics::from_trackers(
            time_dt, &self._loop_tracker, &self._solve_tracker, &self._step_tracker,
        );
        if let Some(ref mut hist) = self.diagnostics_history {
            hist.push(diag.clone());
        }
        self.diagnostics = Some(diag);
    }

    // ==================================================================================
    // Run outcome tracking (issue #27)
    // ==================================================================================

    /// Clear the per-run numeric outcome. Called at the start of every run so
    /// `converged` / `max_residual` / `worst_block` / `truncated_at` reflect
    /// only the run just performed.
    fn _reset_run_outcome(&mut self) {
        self._run_converged = true;
        self._run_max_residual = 0.0;
        self._run_worst_block = None;
        self._run_truncated_at = None;
    }

    /// Record a convergence failure (implicit solve or algebraic loop) with its
    /// worst WRMS residual and the block/booster index that produced it.
    fn _record_nonconvergence(&mut self, residual: f64, worst: Option<usize>) {
        self._run_converged = false;
        if residual > self._run_max_residual {
            self._run_max_residual = residual;
            self._run_worst_block = worst;
        }
    }

    /// Fold the per-run numeric outcome into a freshly counted `RunStats`.
    fn _apply_run_outcome(&self, stats: &mut RunStats) {
        stats.converged = self._run_converged;
        stats.max_residual = self._run_max_residual;
        stats.worst_block = self._run_worst_block;
        stats.truncated_at = self._run_truncated_at;
    }

    /// Public accessor for the last run's numeric outcome (issue #29): the data
    /// the convergence trackers gather every step, previously discarded. Returns
    /// `(converged, max_residual, worst_block, truncated_at)`.
    pub fn run_outcome(&self) -> (bool, f64, Option<usize>, Option<f64>) {
        (self._run_converged, self._run_max_residual, self._run_worst_block, self._run_truncated_at)
    }

    /// The most recent per-timestep diagnostics snapshot, if diagnostics are
    /// enabled. Exposes the structured `Diagnostics` (loop/solve iterations,
    /// residuals, worst block) that was formerly only rendered into a log
    /// string and thrown away.
    pub fn last_diagnostics(&self) -> Option<&Diagnostics> {
        self.diagnostics.as_ref()
    }

    // ==================================================================================
    // Stop
    // ==================================================================================

    pub fn stop(&mut self) {
        self._active = false;
    }

    // ==================================================================================
    // Run methods
    // ==================================================================================

    /// Adaptive step-size update shared by every run loop: reset to the nominal
    /// `dt` when there is no error estimate, rescale by `scale`, clamp to the next
    /// event and to `end_time`, then to the `[dt_min, dt_max]` bounds. Returns the
    /// dt for the next step.
    fn _advance_dt(&self, mut dt: f64, error_norm: f64, scale: f64, end_time: f64) -> f64 {
        if error_norm == 0.0 && scale == 1.0 {
            dt = self.dt;
        }
        dt *= scale;
        if let Some(dt_evt) = self._estimate_events(self.time) {
            if dt_evt < dt { dt = dt_evt; }
        }
        if self.time + dt > end_time {
            dt = end_time - self.time;
        }
        dt.clamp(self.dt_min, self.dt_max.unwrap_or(f64::MAX))
    }

    /// Run the simulation for a given duration.
    pub fn run(&mut self, duration: f64, reset: bool, adaptive: bool) -> RunStats {
        use crate::utils::logger::ProgressTracker;

        self._active = true;
        if reset { self.reset(0.0); }

        let mut tracker = ProgressTracker::new(duration, "TRANSIENT", self.logger.enabled);
        tracker.start();

        let is_adaptive = adaptive && self.engine.is_adaptive;
        let end_time = self.time + duration;
        let start_time = self.time;
        let mut dt = self.dt;
        let mut stats = RunStats::default();

        // Reset per-run convergence tracking so this run reports its own outcome.
        self._reset_run_outcome();

        // Initial system evaluation
        self._update(self.time);

        // Catch and resolve initial events
        self._resolve_initial_events();

        // Sample initial state
        self._sample(self.time, dt);

        // Main simulation loop
        while self.time < end_time && self._active {
            let (success, error_norm, scale, evals, solver_its) =
                self.timestep(Some(dt), is_adaptive);

            stats.total_steps += 1;
            stats.total_evals += evals;
            stats.total_solver_its += solver_its;

            if success {
                stats.successful_steps += 1;
            } else {
                stats.rejected_steps += 1;
            }

            if is_adaptive {
                dt = self._advance_dt(dt, error_norm, scale, end_time);
            }

            // Update progress tracker
            let progress = ((self.time - start_time) / duration).clamp(0.0, 1.0);
            tracker.update(progress, success);
        }

        if !self._active { tracker.interrupt(); }
        tracker.close();

        stats.wall_time_secs = tracker.stats.runtime_ms / 1000.0;
        stats.total_steps = tracker.stats.total_steps;
        stats.successful_steps = tracker.stats.successful_steps;
        stats.rejected_steps = tracker.stats.rejected_steps;
        self._apply_run_outcome(&mut stats);
        stats
    }

    /// Begin a chunked (cooperative) run for the streaming generator.
    ///
    /// Performs the one-time run setup (optional reset, initial system
    /// evaluation, initial event resolution, initial sample), seeds the
    /// persistent working timestep `_run_dt`, and starts a progress tracker
    /// (logs STARTING; updated per step in `run_until`). `duration` is the
    /// total run length, used for the progress percentage. Pairs with repeated
    /// `run_until` calls and a final `run_end`.
    pub fn run_begin(&mut self, reset: bool, duration: f64) {
        use crate::utils::logger::ProgressTracker;
        self._active = true;
        if reset { self.reset(0.0); }
        self._run_dt = self.dt;
        self._reset_run_outcome();

        // Initial system evaluation, event resolution, and sample — mirrors run().
        self._update(self.time);
        self._resolve_initial_events();
        self._sample(self.time, self._run_dt);

        let mut tracker = ProgressTracker::new(duration, "STREAMING", self.logger.enabled);
        tracker.start();
        self._stream_tracker = Some(tracker);
    }

    /// Finalize a chunked streaming run: logs FINISHED (or INTERRUPTED if the
    /// run was stopped) and drops the tracker. Call once after the last
    /// `run_until`.
    pub fn run_end(&mut self) {
        if let Some(mut tracker) = self._stream_tracker.take() {
            if !self._active { tracker.interrupt(); }
            tracker.close();
        }
    }

    /// Advance the simulation up to `target_time` (a chunk boundary), without
    /// resetting and without re-running the initial setup. `end_time` is the
    /// true run end, used only for adaptive overshoot prevention so the final
    /// chunk lands exactly on the end. The working timestep persists in
    /// `_run_dt` across calls. Returns step counts for the chunk
    /// (`wall_time_secs` is always 0.0 — no wall-clock timing on this path).
    pub fn run_until(&mut self, target_time: f64, end_time: f64, adaptive: bool) -> RunStats {
        let is_adaptive = adaptive && self.engine.is_adaptive;
        let mut dt = self._run_dt;
        let mut stats = RunStats::default();

        // Progress fraction needs the run start time; total_duration is fixed
        // for the run, so derive the start from the true end.
        let total_dur = self._stream_tracker.as_ref().map(|t| t.total_duration).unwrap_or(0.0);
        let run_t0 = end_time - total_dur;

        // Step until the chunk boundary. A chunk may overshoot `target_time`
        // slightly (one step); the next chunk continues from there. Only the
        // true `end_time` is enforced exactly via overshoot prevention.
        while self.time < target_time && self._active {
            let (success, error_norm, scale, evals, solver_its) =
                self.timestep(Some(dt), is_adaptive);

            stats.total_steps += 1;
            stats.total_evals += evals;
            stats.total_solver_its += solver_its;
            if success { stats.successful_steps += 1; } else { stats.rejected_steps += 1; }

            // Drive the streaming progress tracker (logs bar / it/s at its own
            // cadence). Copy time first so the tracker field can borrow self.
            let now_time = self.time;
            if let Some(t) = self._stream_tracker.as_mut() {
                let prog = if total_dur > 0.0 {
                    ((now_time - run_t0) / total_dur).clamp(0.0, 1.0)
                } else { 0.0 };
                t.update(prog, success);
            }

            if is_adaptive {
                dt = self._advance_dt(dt, error_norm, scale, end_time);
            }
        }

        self._run_dt = dt;
        self._apply_run_outcome(&mut stats);
        stats
    }

    /// Run with streaming output at fixed wall-clock rate.
    /// Yields progress info at each tick.
    pub fn run_streaming<F>(
        &mut self,
        duration: f64,
        reset: bool,
        adaptive: bool,
        tickrate: f64,
        mut callback: F,
    ) where F: FnMut(f64, bool, f64) {
        self._active = true;
        if reset { self.reset(0.0); }

        let is_adaptive = adaptive && self.engine.is_adaptive;
        let start_time = self.time;
        let end_time = self.time + duration;
        let mut dt = self.dt;
        let tick_interval = 1.0 / tickrate;
        let mut last_tick = Instant::now();

        self._update(self.time);
        self._resolve_initial_events();
        self._sample(self.time, dt);

        while self.time < end_time && self._active {
            let (success, error_norm, scale, _, _) = self.timestep(Some(dt), is_adaptive);

            if is_adaptive {
                dt = self._advance_dt(dt, error_norm, scale, end_time);
            }

            // Yield at tick rate
            if last_tick.elapsed().as_secs_f64() >= tick_interval {
                let progress = ((self.time - start_time) / duration).clamp(0.0, 1.0);
                callback(progress, success, dt);
                last_tick = Instant::now();
            }
        }

        // Final callback
        callback(1.0, true, dt);
    }

    /// Run synchronized to wall-clock time.
    pub fn run_realtime<F>(
        &mut self,
        duration: f64,
        reset: bool,
        adaptive: bool,
        tickrate: f64,
        speed: f64,
        mut callback: F,
    ) where F: FnMut(f64, bool, f64) {
        self._active = true;
        if reset { self.reset(0.0); }

        let is_adaptive = adaptive && self.engine.is_adaptive;
        let start_time = self.time;
        let end_time = self.time + duration;
        let mut dt = self.dt;
        let tick_interval = 1.0 / tickrate;
        let wall_start = Instant::now();

        self._update(self.time);
        self._resolve_initial_events();
        self._sample(self.time, dt);

        let mut last_tick = Instant::now();

        while self.time < end_time && self._active {
            // Target simulation time based on wall clock
            let wall_elapsed = wall_start.elapsed().as_secs_f64();
            let target_time = start_time + wall_elapsed * speed;

            // Advance simulation until caught up
            while self.time < target_time.min(end_time) && self._active {
                let (_success, error_norm, scale, _, _) = self.timestep(Some(dt), is_adaptive);

                if is_adaptive {
                    dt = self._advance_dt(dt, error_norm, scale, end_time);
                }
            }

            // Yield at tick rate
            if last_tick.elapsed().as_secs_f64() >= tick_interval {
                let progress = ((self.time - start_time) / duration).clamp(0.0, 1.0);
                callback(progress, true, dt);
                last_tick = Instant::now();
            }

            // Small sleep to avoid busy-wait
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        callback(1.0, true, dt);
    }
}

// ==================================================================================
// Tests
// ==================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_empty() {
        let sim = Simulation::with_defaults(Vec::new(), Vec::new());
        assert_eq!(sim.time, 0.0);
        assert!(sim.blocks.is_empty());
        assert!(sim.connections.is_empty());
    }

    #[test]
    fn test_simulation_size() {
        let sim = Simulation::with_defaults(Vec::new(), Vec::new());
        assert_eq!(sim.size(), (0, 0));
    }

    #[test]
    fn test_simulation_run_empty() {
        let mut sim = Simulation::with_defaults(Vec::new(), Vec::new());
        let stats = sim.run(1.0, false, false);
        assert!(stats.total_steps > 0);
        assert!((sim.time - 1.0).abs() < 0.02);
    }

    #[test]
    fn test_simulation_stop() {
        let mut sim = Simulation::with_defaults(Vec::new(), Vec::new());
        assert!(sim.is_active());
        sim.stop();
        assert!(!sim.is_active());
    }

    #[test]
    fn bad_port_alias_surfaces_cleanly_without_panic() {
        // A connection to a non-existent input alias must NOT panic in the
        // assembly hot path; it is remembered as an assembly error for the run
        // wrappers to raise as a clean exception. Mirrors the Python regression
        // `test_bad_port_alias_raises_at_run`.
        use crate::blocks::constructors::{constant, scope};
        use crate::connection::Connection;
        use crate::utils::portreference::{Port, PortReference};
        let c = constant(1.0);
        let sco = scope(None, 0.0, vec![]);
        let conn = Rc::new(Connection::new(
            PortReference::new(c.clone(), None),
            vec![PortReference::new(sco.clone(), Some(vec![Port::Name("nonexistent".to_string())]))],
        ));
        let mut sim = Simulation::with_defaults(vec![c, sco], vec![conn]);
        sim.run(0.1, false, false); // must not panic
        assert!(sim.take_assembly_error().is_some(), "bad alias must surface as an assembly error");
    }

    #[test]
    fn test_event_livelock_force_resolves_at_dt_min() {
        // Issue #25: a steep guard whose |func_evt| can never fall within
        // tolerance across a dt_min bracket previously pinned dt at dt_min and
        // retried forever without advancing time. The floor force-resolve must
        // break the loop. Bounded by a step counter (not wall-clock) so a
        // regression fails via the cap instead of hanging.
        use crate::blocks::constructors::{constant, scope};
        use crate::connection::Connection;
        use crate::events::zerocrossing::ZeroCrossing;
        use crate::utils::fastcell::FastCell;
        use crate::utils::portreference::PortReference;

        let fired = Rc::new(FastCell::new(0usize));
        let fired_c = fired.clone();
        // Step guard: |func_evt| == 1 on both sides of t=1, so localization can
        // never get within tolerance.
        let evt = ZeroCrossing::new(
            |t| if t < 1.0 { -1.0 } else { 1.0 },
            Some(Box::new(move |_t| { *fired_c.borrow_mut() += 1; })),
            1e-9,
        );
        let evt_ref: SimEventRef = Rc::new(FastCell::new(evt));

        let c = constant(1.0);
        let s = scope(None, 0.0, vec![]);
        let conn = Rc::new(Connection::new(
            PortReference::new(c.clone(), None),
            vec![PortReference::new(s.clone(), None)],
        ));
        let mut sim = Simulation::with_defaults(vec![c, s], vec![conn]);
        sim.add_event(evt_ref);
        sim.set_solver(crate::solvers::factories::rkf45_factory(1e-6, 0.0));
        sim.dt = 0.1;
        sim.dt_min = 0.02;

        // Mirror the adaptive run loop with a hard step cap.
        sim.reset(0.0);
        sim._active = true;
        let end_time = 2.0;
        let mut dt = sim.dt;
        let is_adaptive = sim.engine.is_adaptive;
        assert!(is_adaptive, "test requires an adaptive engine");
        sim._update(0.0);
        sim._resolve_initial_events();
        sim._sample(sim.time, dt);

        let mut steps = 0usize;
        const MAX_STEPS: usize = 100_000;
        while sim.time < end_time {
            let (_ok, err, scale, _e, _i) = sim.timestep(Some(dt), is_adaptive);
            dt = sim._advance_dt(dt, err, scale, end_time);
            steps += 1;
            assert!(
                steps < MAX_STEPS,
                "event localization livelocked (no time progress) at t={}",
                sim.time
            );
        }
        assert!(*fired.borrow() >= 1, "steep guard event never resolved");
        assert!(
            (sim.time - end_time).abs() < 0.15,
            "run did not reach end_time, stuck at t={}",
            sim.time
        );
    }

    #[test]
    fn test_simulation_reset() {
        let mut sim = Simulation::with_defaults(Vec::new(), Vec::new());
        sim.run(1.0, false, false);
        assert!(sim.time > 0.0);
        sim.reset(0.0);
        assert_eq!(sim.time, 0.0);
    }

    #[test]
    fn test_convergence_tracker_integration() {
        let mut ct = ConvergenceTracker::new();
        ct.begin_iteration();
        ct.record(0, 1e-11);
        // Threshold is now the WRMS-scaled `NLS_COEF` everywhere; the tracker
        // itself just stores residual norms and runs a max-vs-threshold check.
        assert!(ct.converged(NLS_COEF));
    }

    #[test]
    fn test_step_tracker_integration() {
        let mut st = StepTracker::new();
        st.record(0, true, 0.0, None);
        assert!(st.success);
        assert_eq!(st.scale(), 1.0);
    }
}
