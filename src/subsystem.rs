// Subsystem: hierarchical block composition
// Ported 1:1 from pathsim/subsystem.py
//
// A Subsystem is a Block that internally holds blocks + connections + graph.
// It delegates update/solve/step/buffer/revert/reset to internal blocks.
// I/O is handled via an Interface block (reversed: subsystem.inputs = interface.outputs).

use std::rc::Rc;

use crate::blocks::block::{Block, BlockRef, BlockRole};
use crate::constants::NLS_COEF;
use crate::error::SimError;
use crate::optim::booster::ConnectionBooster;
use crate::simulation::ConnectionRef;
use crate::solvers::solver::Solver;
use crate::utils::fastcell::FastCell;
use crate::utils::schedule::Schedule;

/// Create an Interface block (bare-bone block with len=0).
pub fn interface() -> BlockRef {
    let mut b = Block::default_block();
    b.type_name = "Interface";
    // Interface block is a passive port-bridge: no engine, no internal feedthrough
    // (it forwards between subsystem and inner blocks, never feeds an input back to
    // an output itself), and not a source or sink in its own right.
    b.role = BlockRole { is_dyn: false, is_src: false, is_rec: false };
    b.opaque_feedthrough = false; // passive bridge: declares no input->output feedthrough
    b.len_fn = Some(Box::new(|_| 0));
    Rc::new(FastCell::new(b))
}

/// Check if a block is an Interface block.
pub fn is_interface(block: &BlockRef) -> bool {
    block.borrow().type_name == "Interface"
}

/// Internal state for a Subsystem.
pub struct SubsystemInner {
    pub blocks: Vec<BlockRef>,         // regular blocks (NOT interface)
    pub connections: Vec<ConnectionRef>,
    pub interface: BlockRef,           // the single interface block
    /// `blocks` followed by `interface` (interface last == its graph node id).
    /// Cached at `assemble_graph` time so the per-timestep `dag`/`loops` walks
    /// don't reclone the whole block list on every call.
    all_blocks: Vec<BlockRef>,
    graph: Option<Schedule>,
    graph_dirty: bool,
    boosters: Vec<ConnectionBooster>,
    blocks_dyn_indices: Vec<usize>,    // indices into self.blocks
    iterations_max: usize,
}

impl SubsystemInner {
    /// Build from blocks (must contain exactly one Interface) and connections.
    pub fn new(
        all_blocks: Vec<BlockRef>,
        connections: Vec<ConnectionRef>,
        iterations_max: usize,
    ) -> Result<Self, SimError> {
        // Separate interface from regular blocks
        let mut blocks = Vec::new();
        let mut interface: Option<BlockRef> = None;

        for block in all_blocks {
            if is_interface(&block) {
                if interface.is_some() {
                    return Err(SimError::MultipleInterfaces);
                }
                interface = Some(block);
            } else {
                blocks.push(block);
            }
        }

        let interface = interface.ok_or(SimError::MissingInterface)?;

        let mut inner = Self {
            blocks,
            connections,
            interface,
            all_blocks: Vec::new(),
            graph: None,
            graph_dirty: true,
            boosters: Vec::new(),
            blocks_dyn_indices: Vec::new(),
            iterations_max,
        };

        inner.assemble_graph();
        Ok(inner)
    }

    /// Interface node ID in the graph (= last index).
    fn interface_node_id(&self) -> usize {
        self.blocks.len() // interface is appended at end
    }

    fn assemble_graph(&mut self) {
        // Reset all block inputs
        for block in &self.blocks {
            block.borrow_mut().inputs.reset();
        }

        self.all_blocks = self.blocks.to_vec();
        self.all_blocks.push(self.interface.clone());

        // Resolve internal port indices FIRST so the register widths and port
        // indices the port-granular assembly reads are sized. Best-effort: a bad
        // port alias surfaces on the first update like before.
        for conn in &self.connections {
            let _ = conn.resolve_ports();
        }

        // Single shared port-granular assembly — identical to the top-level
        // `Simulation` and the IR export, so subsystem interiors break the same
        // false algebraic loops as everything else (see `assemble_graph_from`).
        self.graph = Some(crate::simulation::assemble_graph_from(&self.all_blocks, &self.connections));
        self.graph_dirty = false;

        // Create boosters for loop-closing connections.  Subsystems don't
        // own per-instance outer tolerances yet; default to the
        // simulation-wide constants so the WRMS scale is consistent with
        // the rest of the integrator.
        self.boosters.clear();
        if let Some(ref graph) = self.graph {
            if graph.has_loops {
                for &conn_idx in graph.loop_closing_connections() {
                    if let Some(conn) = self.connections.get(conn_idx) {
                        self.boosters.push(ConnectionBooster::new(
                            conn.clone(),
                            crate::constants::SOL_TOLERANCE_LTE_ABS,
                            crate::constants::SOL_TOLERANCE_LTE_REL,
                        ));
                    }
                }
            }
        }
    }

    /// Evaluate internal DAG.
    fn dag(&self, t: f64) {
        if let Some(ref graph) = self.graph {
            let all = &self.all_blocks;
            let (alg_depth, _) = graph.depth();
            let iface_id = self.interface_node_id();

            // Update interface outgoing connections first
            for &conn_idx in graph.outgoing_connections(iface_id) {
                if let Some(conn) = self.connections.get(conn_idx) {
                    conn.update();
                }
            }

            // DAG evaluation by depth
            for depth in 0..=alg_depth {
                for &bi in graph.dag_blocks(depth) {
                    if let Some(block) = all.get(bi) {
                        if block.borrow().is_active() {
                            block.borrow_mut().update(t);
                        }
                    }
                }
                for &ci in graph.dag_connections(depth) {
                    if let Some(conn) = self.connections.get(ci) {
                        conn.update();
                    }
                }
            }
        }
    }

    /// Solve algebraic loops via fixed-point iteration.
    fn loops(&mut self, t: f64) {
        if let Some(ref graph) = self.graph {
            let all = &self.all_blocks;
            let (_, loop_depth) = graph.depth();

            for _iteration in 0..self.iterations_max {
                // Iterate loop DAG
                for depth in 0..=loop_depth {
                    for &bi in graph.loop_blocks(depth) {
                        if let Some(block) = all.get(bi) {
                            if block.borrow().is_active() {
                                block.borrow_mut().update(t);
                            }
                        }
                    }
                    for &ci in graph.loop_connections(depth) {
                        if let Some(conn) = self.connections.get(ci) {
                            conn.update();
                        }
                    }
                }

                // Step boosters and check convergence.  Booster residuals
                // are WRMS-scaled; threshold matches the simulation-level
                // `NLS_COEF` used by `Simulation::_solve` / `_loops`.
                let mut max_err: f64 = 0.0;
                for booster in &mut self.boosters {
                    let err = booster.update();
                    if err > max_err { max_err = err; }
                }

                if max_err <= NLS_COEF {
                    return;
                }
            }
        }
    }

    // -- Block protocol methods --

    pub fn update(&mut self, t: f64) {
        if self.graph_dirty { self.assemble_graph(); }
        self.dag(t);
        if self.graph.as_ref().is_some_and(|g| g.has_loops) {
            self.loops(t);
        }
    }

    pub fn solve(&self, t: f64, dt: f64) -> f64 {
        let mut max_error: f64 = 0.0;
        for &idx in &self.blocks_dyn_indices {
            if let Some(block) = self.blocks.get(idx) {
                if block.borrow().is_active() {
                    let err = block.borrow_mut().solve(t, dt);
                    if err > max_error { max_error = err; }
                }
            }
        }
        max_error
    }

    pub fn step(&self, t: f64, dt: f64) -> (bool, f64, Option<f64>) {
        let mut success = true;
        let mut max_error: f64 = 0.0;
        let mut min_scale: Option<f64> = None;
        for &idx in &self.blocks_dyn_indices {
            if let Some(block) = self.blocks.get(idx) {
                if block.borrow().is_active() {
                    let (suc, err, scl) = block.borrow_mut().step(t, dt);
                    success = success && suc;
                    if err > max_error { max_error = err; }
                    if let Some(s) = scl {
                        min_scale = Some(min_scale.map_or(s, |m: f64| m.min(s)));
                    }
                }
            }
        }
        (success, max_error, min_scale)
    }

    pub fn buffer(&self, dt: f64) {
        for &idx in &self.blocks_dyn_indices {
            if let Some(block) = self.blocks.get(idx) {
                if block.borrow().is_active() {
                    block.borrow_mut().buffer(dt);
                }
            }
        }
    }

    pub fn revert(&self) {
        for &idx in &self.blocks_dyn_indices {
            if let Some(block) = self.blocks.get(idx) {
                block.borrow_mut().revert();
            }
        }
    }

    pub fn reset(&self) {
        self.interface.borrow_mut().reset();
        for block in &self.blocks {
            block.borrow_mut().reset();
        }
    }

    pub fn sample(&self, t: f64, dt: f64) {
        for block in &self.blocks {
            if block.borrow().is_active() {
                block.borrow_mut().sample(t, dt);
            }
        }
    }

    pub fn set_solver(&mut self, factory: &dyn Fn(&[f64]) -> Solver) {
        self.blocks_dyn_indices.clear();
        for (i, block) in self.blocks.iter().enumerate() {
            block.borrow_mut().set_solver_from(factory);
            if block.borrow().has_engine() {
                self.blocks_dyn_indices.push(i);
            }
        }
    }

    pub fn has_dynamic_blocks(&self) -> bool {
        !self.blocks_dyn_indices.is_empty()
    }

    /// The subsystem's port-granular direct-feedthrough matrix (row-major
    /// `n_out × n_in`), derived by reachability through the interior graph. This
    /// is the per-port generalisation of [`has_passthrough`]: it lets the
    /// enclosing scheduler treat the subsystem exactly like a leaf block, so a
    /// feedback into a subsystem input that no output algebraically reads does
    /// not manufacture a false loop. Empty if the interior graph is absent.
    pub fn feedthrough_matrix(&self) -> Vec<bool> {
        match &self.graph {
            Some(g) => g.interface_feedthrough(self.interface_node_id()),
            None => Vec::new(),
        }
    }

    /// Check algebraic passthrough (interface -> interface path).
    pub fn has_passthrough(&self) -> bool {
        if let Some(ref graph) = self.graph {
            let iface_id = self.interface_node_id();
            graph.is_algebraic_path(iface_id, iface_id)
        } else {
            false
        }
    }
}

/// Create a Subsystem block from blocks (including Interface) and connections.
///
/// Returns a single BlockRef that acts as the subsystem in the parent simulation.
/// The subsystem's inputs/outputs map to the Interface block (reversed).
pub fn subsystem(
    all_blocks: Vec<BlockRef>,
    connections: Vec<ConnectionRef>,
    iterations_max: usize,
) -> Result<BlockRef, SimError> {
    let inner = Rc::new(FastCell::new(SubsystemInner::new(
        all_blocks, connections, iterations_max,
    )?));

    // Build the wrapper Block
    let mut blk = Block::default_block();
    blk.type_name = "Subsystem";

    // I/O: subsystem inputs = interface outputs, subsystem outputs = interface inputs
    // We sync these in update_fn
    let has_dyn = inner.borrow().has_dynamic_blocks();
    if has_dyn {
        blk.engine = Some(Solver::with_defaults(&[0.0]));
    }

    // Outer role mirrors inner topology: `is_dyn` if any internal block has an
    // engine. Algebraic feedthrough is no longer a role flag: the subsystem
    // block derives its exact per-port feedthrough matrix from the interior
    // graph (`SubsystemInner::feedthrough_matrix`, via `Block::feedthrough_mask`).
    blk.role = BlockRole {
        is_dyn: has_dyn,
        is_src: false,
        is_rec: false,
    };

    // len: algebraic passthrough detection
    let inner_len = inner.clone();
    blk.len_fn = Some(Box::new(move |_| {
        if inner_len.borrow().has_passthrough() { 1 } else { 0 }
    }));

    // children: expose internal blocks (excluding the Interface) so hosts can
    // recurse into the subsystem, e.g. to read Scopes nested inside it.
    let inner_children = inner.clone();
    blk.children_fn = Some(Box::new(move || inner_children.borrow().blocks.clone()));

    // update: copy inputs to interface, run internal DAG+loops, copy outputs
    let inner_upd = inner.clone();
    blk.update_fn = Some(Box::new(move |blk, t| {
        // Subsystem inputs -> interface outputs
        {
            let inner_ref = inner_upd.borrow();
            let u = blk.inputs.to_array();
            inner_ref.interface.borrow_mut().outputs.update_from_array(&u);
        }

        // Run internal graph
        inner_upd.borrow_mut().update(t);

        // Interface inputs -> subsystem outputs
        {
            let inner_ref = inner_upd.borrow();
            let y = inner_ref.interface.borrow().inputs.to_array();
            blk.outputs.update_from_array(&y);
        }
    }));

    // solve: sync stage counter from wrapper engine, then delegate
    let inner_sol = inner.clone();
    blk.solve_fn = Some(Box::new(move |blk, t, dt| {
        // Propagate stage counter from wrapper engine to internal dynamic blocks
        let stage = blk.engine.as_ref().map(|e| e._stage).unwrap_or(0);
        let inner_ref = inner_sol.borrow();
        for &idx in &inner_ref.blocks_dyn_indices {
            if let Some(block) = inner_ref.blocks.get(idx) {
                if let Some(ref mut engine) = block.borrow_mut().engine {
                    engine._stage = stage;
                }
            }
        }
        inner_ref.solve(t, dt)
    }));

    // step: sync stage counter, then delegate
    let inner_stp = inner.clone();
    blk.step_fn = Some(Box::new(move |blk, t, dt| {
        let stage = blk.engine.as_ref().map(|e| e._stage).unwrap_or(0);
        let inner_ref = inner_stp.borrow();
        for &idx in &inner_ref.blocks_dyn_indices {
            if let Some(block) = inner_ref.blocks.get(idx) {
                if let Some(ref mut engine) = block.borrow_mut().engine {
                    engine._stage = stage;
                }
            }
        }
        inner_ref.step(t, dt)
    }));

    // buffer
    let inner_buf = inner.clone();
    blk.buffer_fn = Some(Box::new(move |_blk, dt| {
        inner_buf.borrow().buffer(dt);
    }));

    // revert
    let inner_rev = inner.clone();
    blk.revert_fn = Some(Box::new(move |_blk| {
        inner_rev.borrow().revert();
    }));

    // sample
    let inner_samp = inner.clone();
    blk.sample_fn = Some(Box::new(move |_blk, t, dt| {
        inner_samp.borrow().sample(t, dt);
    }));

    // reset
    let inner_rst = inner.clone();
    blk.reset_fn = Some(Box::new(move |blk| {
        blk.inputs.reset();
        blk.outputs.reset();
        inner_rst.borrow().reset();
    }));

    // set_solver: propagate to internal blocks, update engine
    let inner_ss = inner.clone();
    blk.set_solver_fn = Some(Box::new(move |factory: &dyn Fn(&[f64]) -> Solver| {
        let inner_ref = inner_ss.borrow_mut();
        inner_ref.set_solver(factory);
        // Don't set engine here — it's done on the wrapper block by Simulation
    }));

    // on/off: propagate recursively to all internal blocks (matches pathsim)
    let inner_on = inner.clone();
    blk.on_fn = Some(Box::new(move || {
        let inner_ref = inner_on.borrow();
        for block in &inner_ref.blocks {
            block.borrow_mut().on();
        }
    }));

    let inner_off = inner.clone();
    blk.off_fn = Some(Box::new(move || {
        let inner_ref = inner_off.borrow();
        for block in &inner_ref.blocks {
            block.borrow_mut().off();
        }
    }));

    // Propagate internal block events to the wrapper so the outer Simulation
    // can detect and resolve them (e.g., Schedule events in WhiteNoise, Delay)
    {
        let inner_ref = inner.borrow();
        for block in &inner_ref.blocks {
            let b = block.borrow();
            for event in &b.events {
                blk.events.push(event.clone());
            }
        }
    }

    // Set initial_value so Simulation creates an engine on the wrapper
    blk.initial_value = Some(vec![0.0]);

    // Expose the inner graph for IR recursion (the runtime uses the closures
    // above; the IR builder needs blocks + connections + interface).
    blk.subsystem_inner = Some(inner);

    Ok(Rc::new(FastCell::new(blk)))
}
