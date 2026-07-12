// Schedule: directed graph analysis with DFS, Tarjan SCC, topological sort
// Ported from pathsim/utils/graph.py
//
// Optimized: all HashMap/HashSet replaced with Vec-based direct indexing
// since NodeIds are always 0..n (contiguous block indices).

/// Identifier for blocks and connections in the graph.
pub type NodeId = usize;

/// Topology role passed to `Schedule::new` for each node.
///
/// Decoupled from `BlockRole` in `blocks::block` so `utils::schedule` stays free
/// of block-layer dependencies. Only the two flags that actually drive graph
/// logic live here: feedthrough (DAG layering) and engine-state.
#[derive(Debug, Clone, Copy, Default)]
pub struct NodeRole {
    /// Algebraic feedthrough u(t) → y(t). Drives DAG depth computation.
    pub is_alg: bool,
}

/// Optimized graph representation with efficient assembly and cycle detection.
pub struct Schedule {
    /// Number of blocks
    n_blocks: usize,

    /// Per-block: true = has *some* algebraic feedthrough u(t) → y(t). Derived
    /// from `feedthrough` (any nonzero entry). Used by the DAG-depth-0 placement
    /// of dynamic blocks and the block-level `is_algebraic_path` query.
    is_alg: Vec<bool>,

    /// Per-block direct-feedthrough matrix (output-port × input-port). The
    /// port-granular scheduler (`compute_depths`) walks these instead of the
    /// block-level `is_alg` so a MIMO block whose output `o` does not read input
    /// `j` does not create a spurious dependency / loop. `Schedule::new` builds the
    /// block-level-equivalent 1×1 matrix from `is_alg`; `new_with_feedthrough`
    /// gets the precise per-port matrices (derived from each block's SSA).
    feedthrough: Vec<crate::utils::feedthrough_loops::Feedthrough>,

    /// Port-aware connection edges (`out(src,o) -> in(tgt,j)`), parallel to the
    /// block-level `edges` but carrying the ports the scheduler needs.
    port_edges: Vec<crate::utils::feedthrough_loops::PortEdge>,

    /// Downstream adjacency: block -> list of successor block ids
    dnst: Vec<Vec<NodeId>>,

    /// Outgoing connections: block -> list of connection indices
    outg_conns: Vec<Vec<usize>>,

    /// Connection edges stored as (src, tgt, conn_id)
    edges: Vec<(NodeId, NodeId, usize)>,

    /// Flag indicating presence of algebraic loops
    pub has_loops: bool,

    // DAG evaluation order — precomputed flat arrays
    dag_flat_blocks: Vec<NodeId>,
    dag_flat_conns: Vec<usize>,
    dag_offsets: Vec<(usize, usize, usize, usize)>,
    alg_depth: usize,

    // Loop evaluation order
    loop_flat_blocks: Vec<NodeId>,
    loop_flat_conns: Vec<usize>,
    loop_offsets: Vec<(usize, usize, usize, usize)>,
    loop_closing_conns: Vec<usize>,
    loop_depth: usize,

    // Per-SCC algebraic-loop decomposition (parallel arrays: `loop_sccs[i]` is
    // the block set of the i-th loop, `loop_scc_back_edges[i]` the connection
    // ids cut to break it). Retained for IR/schedule export; the runtime hot
    // path reads only the flat `loop_*` arrays above.
    loop_sccs: Vec<Vec<NodeId>>,
    loop_scc_back_edges: Vec<Vec<usize>>,
}

impl Schedule {
    /// Build from explicit per-node role flags.
    ///
    /// `blocks[i]` is the `NodeRole` for node `i`. Nodes are implicitly
    /// indexed by position — the caller owns the mapping from its own
    /// indices to graph indices.
    pub fn new(
        blocks: &[NodeRole],
        connections: &[(NodeId, NodeId, usize)],
    ) -> Self {
        // Block-level-equivalent feedthrough: collapse each block to a single
        // input and output port with a 1×1 matrix `[is_alg]`, and each
        // connection to the port pair (0, 0). The port scheduler then reduces
        // exactly to the legacy block-level behaviour.
        use crate::utils::feedthrough_loops::{Feedthrough, PortEdge};
        let feedthrough: Vec<Feedthrough> = blocks
            .iter()
            .map(|r| Feedthrough::new(1, 1, vec![r.is_alg]))
            .collect();
        let port_edges: Vec<PortEdge> = connections
            .iter()
            .map(|&(src, tgt, _)| PortEdge { src, out_port: 0, tgt, in_port: 0 })
            .collect();
        Self::assemble_from(connections, feedthrough, port_edges)
    }

    /// Build with precise per-block feedthrough matrices and port-aware edges
    /// (the port-granular scheduler). `connections` is the block-level
    /// `(src, tgt, conn_id)` view (for loop processing / schedule export);
    /// `port_edges` is the same connections with their resolved ports.
    pub fn new_with_feedthrough(
        connections: &[(NodeId, NodeId, usize)],
        feedthrough: Vec<crate::utils::feedthrough_loops::Feedthrough>,
        port_edges: Vec<crate::utils::feedthrough_loops::PortEdge>,
    ) -> Self {
        Self::assemble_from(connections, feedthrough, port_edges)
    }

    fn assemble_from(
        connections: &[(NodeId, NodeId, usize)],
        feedthrough: Vec<crate::utils::feedthrough_loops::Feedthrough>,
        port_edges: Vec<crate::utils::feedthrough_loops::PortEdge>,
    ) -> Self {
        let n = feedthrough.len();

        // `is_alg` is now a derived summary: the block has feedthrough iff any
        // output reads any input.
        let is_alg: Vec<bool> = feedthrough.iter().map(|f| f.mat.iter().any(|&b| b)).collect();

        let mut dnst = vec![Vec::new(); n];
        let mut outg_conns = vec![Vec::new(); n];

        for &(src, tgt, con_id) in connections {
            if src < n && tgt < n {
                // Avoid duplicate entries in adjacency lists
                if !dnst[src].contains(&tgt) { dnst[src].push(tgt); }
                outg_conns[src].push(con_id);
            }
        }

        let mut graph = Self {
            n_blocks: n,
            is_alg,
            feedthrough,
            port_edges,
            dnst,
            outg_conns,
            edges: connections.to_vec(),
            has_loops: false,
            dag_flat_blocks: Vec::new(),
            dag_flat_conns: Vec::new(),
            dag_offsets: Vec::new(),
            alg_depth: 0,
            loop_flat_blocks: Vec::new(),
            loop_flat_conns: Vec::new(),
            loop_offsets: Vec::new(),
            loop_closing_conns: Vec::new(),
            loop_depth: 0,
            loop_sccs: Vec::new(),
            loop_scc_back_edges: Vec::new(),
        };

        graph.assemble();
        graph
    }

    pub fn depth(&self) -> (usize, usize) {
        (self.alg_depth, self.loop_depth)
    }

    fn assemble(&mut self) {
        self.has_loops = false;

        if self.n_blocks == 0 {
            self.alg_depth = 0;
            self.loop_depth = 0;
            return;
        }

        // Temporary depth-level storage: Vec<Vec<NodeId>> indexed by depth
        let mut dag_blocks_by_depth: Vec<Vec<NodeId>> = Vec::new();
        let mut dag_conns_by_depth: Vec<Vec<usize>> = Vec::new();

        // Ensure at least depth 0
        dag_blocks_by_depth.push(Vec::new());
        dag_conns_by_depth.push(Vec::new());

        // Dynamic blocks go to depth 0
        for id in 0..self.n_blocks {
            if !self.is_alg[id] {
                dag_blocks_by_depth[0].push(id);
                for &con_id in &self.outg_conns[id] {
                    dag_conns_by_depth[0].push(con_id);
                }
            }
        }

        // Check if there are any algebraic blocks
        let has_alg = self.is_alg.iter().any(|&a| a);

        if has_alg {
            // Compute depths for algebraic blocks
            let depths = self.compute_depths();

            let mut loop_blocks = Vec::new();

            for id in 0..self.n_blocks {
                if !self.is_alg[id] { continue; }
                match depths[id] {
                    Some(d) => {
                        // Ensure depth level exists
                        while dag_blocks_by_depth.len() <= d {
                            dag_blocks_by_depth.push(Vec::new());
                            dag_conns_by_depth.push(Vec::new());
                        }
                        dag_blocks_by_depth[d].push(id);
                        for &con_id in &self.outg_conns[id] {
                            dag_conns_by_depth[d].push(con_id);
                        }
                    }
                    None => {
                        loop_blocks.push(id);
                        self.has_loops = true;
                    }
                }
            }

            if self.has_loops {
                self.process_loops(&loop_blocks);
            } else {
                self.loop_depth = 0;
                self.loop_flat_blocks.clear();
                self.loop_flat_conns.clear();
                self.loop_offsets.clear();
                self.loop_closing_conns.clear();
                self.loop_sccs.clear();
                self.loop_scc_back_edges.clear();
            }
        } else {
            self.loop_depth = 0;
        }

        // Flatten DAG into precomputed arrays
        self.alg_depth = dag_blocks_by_depth.len();
        self.dag_flat_blocks.clear();
        self.dag_flat_conns.clear();
        self.dag_offsets.clear();

        for d in 0..self.alg_depth {
            let bs = self.dag_flat_blocks.len();
            self.dag_flat_blocks.extend_from_slice(&dag_blocks_by_depth[d]);
            let be = self.dag_flat_blocks.len();
            let cs = self.dag_flat_conns.len();
            self.dag_flat_conns.extend_from_slice(&dag_conns_by_depth[d]);
            let ce = self.dag_flat_conns.len();
            self.dag_offsets.push((bs, be, cs, ce));
        }
    }

    /// Port-granular algebraic depths over a signal graph of input/output ports
    /// built from the per-block feedthrough matrices and the port-aware edges.
    /// Each block's depth is the longest feedthrough path to any of its outputs;
    /// a block is `None` (an algebraic loop, or downstream of one) exactly when
    /// one of its outputs lies on — or is fed by — a true port-level cycle. So a
    /// MIMO block whose output `o` does not read input `j` does not create a
    /// spurious dependency or loop. Reduces to the legacy block-level depths when
    /// every block is collapsed to a single 1×1-feedthrough port (`Schedule::new`).
    fn compute_depths(&self) -> Vec<Option<usize>> {
        let n = self.n_blocks;
        let ft = &self.feedthrough;

        // Signal-node ids: all input ports (block-major), then all output ports.
        let mut in_off = vec![0usize; n + 1];
        let mut out_off = vec![0usize; n + 1];
        for b in 0..n { in_off[b + 1] = in_off[b] + ft[b].n_in; }
        let n_in = in_off[n];
        for b in 0..n { out_off[b + 1] = out_off[b] + ft[b].n_out; }
        let n_nodes = n_in + out_off[n];
        let in_id = |b: usize, j: usize| in_off[b] + j;
        let out_id = |b: usize, o: usize| n_in + out_off[b] + o;

        // Weighted incoming edges per signal node (and forward adjacency for SCC):
        // internal `in(b,j) -> out(b,o)` weight 1 (a feedthrough step), connection
        // `out(src,o) -> in(tgt,j)` weight 0.
        // Atomic block evaluation: a read input couples to every output, so
        // `in(b,j) -> out(b,o)` for all `o` when input `j` is read by any output.
        // (Unread inputs get no edge — connections feeding them are not deps.)
        let mut incoming: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_nodes];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        for b in 0..n {
            for j in 0..ft[b].n_in {
                if ft[b].reads_input(j) {
                    for o in 0..ft[b].n_out {
                        incoming[out_id(b, o)].push((in_id(b, j), 1));
                        adj[in_id(b, j)].push(out_id(b, o));
                    }
                }
            }
        }
        for e in &self.port_edges {
            if e.src < n && e.tgt < n && e.out_port < ft[e.src].n_out && e.in_port < ft[e.tgt].n_in {
                incoming[in_id(e.tgt, e.in_port)].push((out_id(e.src, e.out_port), 0));
                adj[out_id(e.src, e.out_port)].push(in_id(e.tgt, e.in_port));
            }
        }

        // True port-level cycles: signal nodes in a strongly-connected component
        // of size > 1 (the port graph has no node self-loops).
        let (comp, _) = crate::utils::scc::components(&adj);
        let mut comp_size = vec![0usize; n_nodes];
        for &c in &comp { comp_size[c] += 1; }
        let on_cycle: Vec<bool> = (0..n_nodes).map(|i| comp_size[comp[i]] > 1).collect();

        // Memoized longest-path depth per signal node; `None` = on/after a cycle.
        // The non-cycle subgraph is a DAG, so the recursion terminates.
        fn depth_of(
            node: usize,
            incoming: &[Vec<(usize, usize)>],
            on_cycle: &[bool],
            memo: &mut [Option<Option<usize>>],
        ) -> Option<usize> {
            if let Some(d) = memo[node] { return d; }
            if on_cycle[node] {
                memo[node] = Some(None);
                return None;
            }
            memo[node] = Some(Some(0)); // guard (unused in a DAG)
            let mut best: Option<usize> = Some(0);
            for &(pred, w) in &incoming[node] {
                match depth_of(pred, incoming, on_cycle, memo) {
                    None => { best = None; break; }
                    Some(d) => if let Some(b) = best { best = Some(b.max(d + w)); },
                }
            }
            memo[node] = Some(best);
            best
        }

        // Block depth = max over its output ports; `None` if any output is `None`.
        // A block with no outputs (sink) resolves at depth 0.
        let mut memo: Vec<Option<Option<usize>>> = vec![None; n_nodes];
        let mut depths = vec![Some(0usize); n];
        for b in 0..n {
            let mut bd: Option<usize> = Some(0);
            for o in 0..ft[b].n_out {
                match depth_of(out_id(b, o), &incoming, &on_cycle, &mut memo) {
                    None => { bd = None; break; }
                    Some(d) => if let Some(c) = bd { bd = Some(c.max(d)); },
                }
            }
            depths[b] = bd;
        }
        depths
    }

    /// Direct-feedthrough matrix of a subsystem whose boundary is the block
    /// `iface_node`: its outputs carry the subsystem's inputs into the interior,
    /// its inputs collect the subsystem's outputs. Returns a row-major
    /// `n_out × n_in` mask (subsystem ports), where entry `[o*n_in + j]` is true
    /// iff subsystem output `o` algebraically depends on subsystem input `j`, by
    /// reachability through this (port-granular) interior graph. Lets a subsystem
    /// block declare the same exact per-port feedthrough as a leaf op-block.
    pub fn interface_feedthrough(&self, iface_node: NodeId) -> Vec<bool> {
        let n = self.n_blocks;
        let ft = &self.feedthrough;
        if iface_node >= n {
            return Vec::new();
        }
        let n_sub_in = ft[iface_node].n_out; // interface outputs == subsystem inputs
        let n_sub_out = ft[iface_node].n_in; // interface inputs == subsystem outputs

        // Signal-port node ids (same layout as `compute_depths`).
        let mut in_off = vec![0usize; n + 1];
        let mut out_off = vec![0usize; n + 1];
        for b in 0..n { in_off[b + 1] = in_off[b] + ft[b].n_in; }
        let n_in_total = in_off[n];
        for b in 0..n { out_off[b + 1] = out_off[b] + ft[b].n_out; }
        let n_nodes = n_in_total + out_off[n];
        let in_id = |b: usize, j: usize| in_off[b] + j;
        let out_id = |b: usize, o: usize| n_in_total + out_off[b] + o;

        // Forward adjacency: atomic internal feedthrough `in(b,j) -> out(b,o)` for
        // every read input, plus connection edges `out(src,o) -> in(tgt,j)`.
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        for b in 0..n {
            for j in 0..ft[b].n_in {
                if ft[b].reads_input(j) {
                    for o in 0..ft[b].n_out { adj[in_id(b, j)].push(out_id(b, o)); }
                }
            }
        }
        for e in &self.port_edges {
            if e.src < n && e.tgt < n && e.out_port < ft[e.src].n_out && e.in_port < ft[e.tgt].n_in {
                adj[out_id(e.src, e.out_port)].push(in_id(e.tgt, e.in_port));
            }
        }

        // Reachability from each interface-output (subsystem input) to each
        // interface-input (subsystem output).
        let mut mat = vec![false; n_sub_out * n_sub_in];
        for j in 0..n_sub_in {
            let mut seen = vec![false; n_nodes];
            let start = out_id(iface_node, j);
            seen[start] = true;
            let mut stack = vec![start];
            while let Some(node) = stack.pop() {
                for &nb in &adj[node] {
                    if !seen[nb] { seen[nb] = true; stack.push(nb); }
                }
            }
            for o in 0..n_sub_out {
                if seen[in_id(iface_node, o)] {
                    mat[o * n_sub_in + j] = true;
                }
            }
        }
        mat
    }

    /// Process algebraic loops: find SCCs and assign local depths.
    fn process_loops(&mut self, loop_blocks: &[NodeId]) {
        if loop_blocks.is_empty() {
            self.loop_depth = 0;
            return;
        }

        // Membership bitset
        let mut in_loop = vec![false; self.n_blocks];
        for &id in loop_blocks { in_loop[id] = true; }

        let sccs = self.find_sccs(loop_blocks, &in_loop);

        let mut loop_blocks_by_depth: Vec<Vec<NodeId>> = Vec::new();
        let mut loop_conns_by_depth: Vec<Vec<usize>> = Vec::new();
        self.loop_closing_conns.clear();
        self.loop_sccs.clear();
        self.loop_scc_back_edges.clear();

        let mut current_depth: usize = 0;

        for scc in &sccs {
            let mut in_scc = vec![false; self.n_blocks];
            for &id in scc { in_scc[id] = true; }

            // Intra-SCC directed edges (tgt, conn_id) per source block, deduped.
            let mut sn: Vec<Vec<(NodeId, usize)>> = vec![Vec::new(); self.n_blocks];
            for &(src, tgt, cid) in &self.edges {
                if in_scc[src] && in_scc[tgt] && !sn[src].contains(&(tgt, cid)) {
                    sn[src].push((tgt, cid));
                }
            }

            // Minimal tear set: the smallest set of connections whose removal
            // makes this SCC acyclic. For small SCCs this is the *exact* minimum
            // (subset enumeration by ascending size); larger SCCs fall back to a
            // DFS back-edge feedback arc set. Tearing the fewest connections
            // minimizes the number of loop-closing residuals (ConnectionBoosters)
            // the algebraic-loop solver must drive to zero.
            let torn: std::collections::HashSet<usize> =
                min_feedback_conns(scc, self.n_blocks, &sn).into_iter().collect();

            // Longest-path local depth on the torn (acyclic) DAG: a block sits one
            // level below its deepest non-torn intra-SCC predecessor.
            let mut preds: Vec<Vec<NodeId>> = vec![Vec::new(); self.n_blocks];
            for &(src, tgt, cid) in &self.edges {
                if in_scc[src] && in_scc[tgt] && !torn.contains(&cid) && !preds[tgt].contains(&src) {
                    preds[tgt].push(src);
                }
            }
            fn local_depth_of(b: NodeId, preds: &[Vec<NodeId>], memo: &mut [usize]) -> usize {
                if memo[b] != usize::MAX { return memo[b]; }
                memo[b] = 0; // guard (the torn graph is acyclic)
                let mut d = 0;
                for &p in &preds[b] {
                    d = d.max(local_depth_of(p, preds, memo) + 1);
                }
                memo[b] = d;
                d
            }
            let mut local_depth = vec![usize::MAX; self.n_blocks];
            let mut max_local = 0;
            for &blk in scc {
                let d = local_depth_of(blk, &preds, &mut local_depth);
                if d > max_local { max_local = d; }
            }

            // Assign global depths; torn connections close the loop (boosters),
            // the rest fire at their source block's depth.
            let mut scc_closing: Vec<usize> = Vec::new();
            for &blk in scc {
                let global_d = current_depth + local_depth[blk];
                while loop_blocks_by_depth.len() <= global_d {
                    loop_blocks_by_depth.push(Vec::new());
                    loop_conns_by_depth.push(Vec::new());
                }
                loop_blocks_by_depth[global_d].push(blk);

                let mut seen: Vec<usize> = Vec::new();
                for &con_id in &self.outg_conns[blk] {
                    if seen.contains(&con_id) { continue; }
                    seen.push(con_id);
                    if torn.contains(&con_id) {
                        self.loop_closing_conns.push(con_id);
                        scc_closing.push(con_id);
                    } else {
                        loop_conns_by_depth[global_d].push(con_id);
                    }
                }
            }

            // Retain this SCC and its (deduped) cut set for schedule export.
            scc_closing.sort_unstable();
            scc_closing.dedup();
            self.loop_sccs.push(scc.clone());
            self.loop_scc_back_edges.push(scc_closing);

            current_depth += max_local + 1;
        }
        self.loop_closing_conns.sort_unstable();
        self.loop_closing_conns.dedup();

        // Flatten loop ordering
        self.loop_depth = loop_blocks_by_depth.len();
        self.loop_flat_blocks.clear();
        self.loop_flat_conns.clear();
        self.loop_offsets.clear();

        for d in 0..self.loop_depth {
            let bs = self.loop_flat_blocks.len();
            self.loop_flat_blocks.extend_from_slice(&loop_blocks_by_depth[d]);
            let be = self.loop_flat_blocks.len();
            let cs = self.loop_flat_conns.len();
            self.loop_flat_conns.extend_from_slice(&loop_conns_by_depth[d]);
            let ce = self.loop_flat_conns.len();
            self.loop_offsets.push((bs, be, cs, ce));
        }
    }

    /// Algebraic-loop SCCs over the block subgraph restricted to `in_set`: the
    /// set-internal successor adjacency fed to the shared Tarjan
    /// ([`crate::utils::scc::tarjan_sccs`]), keeping only components that are real
    /// cycles (size > 1, or a single block with a self-edge).
    fn find_sccs(&self, blocks: &[NodeId], in_set: &[bool]) -> Vec<Vec<NodeId>> {
        if blocks.is_empty() { return Vec::new(); }

        // Successors restricted to the set, as a full-width adjacency so node ids
        // stay the block ids (out-of-set blocks are empty -> singletons, dropped).
        let mut succ: Vec<Vec<NodeId>> = vec![Vec::new(); self.n_blocks];
        for &blk in blocks {
            for &nb in &self.dnst[blk] {
                if in_set[nb] { succ[blk].push(nb); }
            }
        }

        crate::utils::scc::tarjan_sccs(&succ)
            .into_iter()
            .filter(|scc| scc.len() > 1 || succ[scc[0]].contains(&scc[0]))
            .collect()
    }

    // ==================================================================================
    // Public accessors
    // ==================================================================================

    /// Get blocks at a given DAG depth level.
    pub fn dag_blocks(&self, depth: usize) -> &[NodeId] {
        if depth < self.dag_offsets.len() {
            let (bs, be, _, _) = self.dag_offsets[depth];
            &self.dag_flat_blocks[bs..be]
        } else { &[] }
    }

    /// Get connection IDs at a given DAG depth level.
    pub fn dag_connections(&self, depth: usize) -> &[usize] {
        if depth < self.dag_offsets.len() {
            let (_, _, cs, ce) = self.dag_offsets[depth];
            &self.dag_flat_conns[cs..ce]
        } else { &[] }
    }

    /// Iterate flat DAG evaluation order: yields (block_slice, conn_slice) per depth.
    pub fn dag_iter(&self) -> impl Iterator<Item = (&[NodeId], &[usize])> {
        self.dag_offsets.iter().map(move |&(bs, be, cs, ce)| {
            (&self.dag_flat_blocks[bs..be], &self.dag_flat_conns[cs..ce])
        })
    }

    /// Get blocks at a given loop depth level.
    pub fn loop_blocks(&self, depth: usize) -> &[NodeId] {
        if depth < self.loop_offsets.len() {
            let (bs, be, _, _) = self.loop_offsets[depth];
            &self.loop_flat_blocks[bs..be]
        } else { &[] }
    }

    /// Get connection IDs at a given loop depth level.
    pub fn loop_connections(&self, depth: usize) -> &[usize] {
        if depth < self.loop_offsets.len() {
            let (_, _, cs, ce) = self.loop_offsets[depth];
            &self.loop_flat_conns[cs..ce]
        } else { &[] }
    }

    /// Returns loop-closing connection IDs.
    pub fn loop_closing_connections(&self) -> &[usize] {
        &self.loop_closing_conns
    }

    /// Full linear evaluation order in node ids: DAG depths first (0..alg_depth),
    /// then loop depths. The order a sequential backend would evaluate blocks in.
    /// Note: "depth" is *algebraic feedthrough* depth, so dynamic/source blocks
    /// sit at depth 0 (their outputs depend on state, not current inputs), not at
    /// naive topological distance.
    pub fn topo_order(&self) -> Vec<NodeId> {
        let mut order = Vec::with_capacity(self.n_blocks);
        for (blocks, _) in self.dag_iter() {
            order.extend_from_slice(blocks);
        }
        for d in 0..self.loop_depth {
            order.extend_from_slice(self.loop_blocks(d));
        }
        order
    }

    /// Iterate the algebraic loops: yields `(scc_blocks, back_edge_conn_ids)` per
    /// strongly-connected component. Empty when the graph is acyclic.
    pub fn algebraic_loops(&self) -> impl Iterator<Item = (&[NodeId], &[usize])> {
        self.loop_sccs
            .iter()
            .zip(self.loop_scc_back_edges.iter())
            .map(|(s, b)| (s.as_slice(), b.as_slice()))
    }

    pub fn size(&self) -> (usize, usize) {
        (self.n_blocks, self.edges.len())
    }

    pub fn outgoing_connections(&self, block: NodeId) -> &[usize] {
        if block < self.n_blocks { &self.outg_conns[block] } else { &[] }
    }

    /// Check if blocks are connected through an algebraic path.
    pub fn is_algebraic_path(&self, start: NodeId, end: NodeId) -> bool {
        if start >= self.n_blocks || end >= self.n_blocks { return false; }

        // Self-loop
        if start == end {
            return self._has_algebraic_self_loop(start);
        }

        if !self.is_alg[end] { return false; }

        let mut visited = vec![false; self.n_blocks];
        let mut stack = vec![start];

        while let Some(node) = stack.pop() {
            if visited[node] { continue; }
            visited[node] = true;

            for &nbr in &self.dnst[node] {
                if nbr == end { return true; }
                if !self.is_alg[nbr] { continue; }
                if !visited[nbr] { stack.push(nbr); }
            }
        }
        false
    }

    fn _has_algebraic_self_loop(&self, block: NodeId) -> bool {
        if !self.is_alg[block] { return false; }
        if self.dnst[block].is_empty() { return false; }

        let mut visited = vec![false; self.n_blocks];
        let mut stack: Vec<NodeId> = self.dnst[block].clone();

        while let Some(node) = stack.pop() {
            if visited[node] { continue; }
            if node == block { return true; }
            visited[node] = true;
            if !self.is_alg[node] { continue; }
            for &nbr in &self.dnst[node] {
                if !visited[nbr] { stack.push(nbr); }
            }
        }
        false
    }

}

/// Exact-minimum feedback connection set for a single algebraic-loop SCC: the
/// smallest set of connection ids whose removal makes the SCC acyclic. Tearing
/// the fewest connections minimizes the number of loop-closing residuals
/// (ConnectionBoosters) the algebraic-loop solver has to drive to zero, and it
/// keeps the torn remainder a valid DAG for depth layering.
///
/// `sn[b]` lists block `b`'s deduped intra-SCC `(tgt, conn_id)` successor edges.
/// When the SCC has at most [`crate::constants::TEAR_EXACT_MAX_CONNS`] distinct
/// connections this enumerates connection subsets by ascending size (Gosper's
/// hack) and returns the first whose removal breaks every cycle, i.e. the exact
/// minimum. Larger SCCs fall back to [`dfs_back_edge_conns`] (a valid feedback
/// arc set, not guaranteed minimum) to stay within the `O(2^c)` budget.
fn min_feedback_conns(scc: &[NodeId], n_blocks: usize, sn: &[Vec<(NodeId, usize)>]) -> Vec<usize> {
    // Remap SCC blocks to dense local indices so the cycle check works on
    // arrays sized by the SCC, not the whole block count.
    let m = scc.len();
    let mut local = vec![usize::MAX; n_blocks];
    for (i, &b) in scc.iter().enumerate() {
        local[b] = i;
    }

    // Distinct intra-SCC connections (bit positions) and local edge list.
    let mut conns: Vec<usize> = Vec::new();
    let mut edges: Vec<(usize, usize, usize)> = Vec::new(); // (lsrc, ltgt, bit)
    for &b in scc {
        for &(tgt, cid) in &sn[b] {
            let bit = match conns.iter().position(|&x| x == cid) {
                Some(i) => i,
                None => {
                    conns.push(cid);
                    conns.len() - 1
                }
            };
            edges.push((local[b], local[tgt], bit));
        }
    }
    let c = conns.len();
    if c == 0 {
        return Vec::new();
    }

    if c <= crate::constants::TEAR_EXACT_MAX_CONNS {
        // Enumerate connection subsets by ascending size; the first acyclic one
        // is an exact-minimum feedback connection set. `k == c` (remove all
        // edges) is always acyclic, so the loop always returns.
        for k in 0..=c {
            if k == 0 {
                if is_acyclic_local(m, &edges, 0) {
                    return Vec::new();
                }
                continue;
            }
            let mut mask: u32 = (1u32 << k) - 1;
            let limit: u32 = 1u32 << c;
            while mask < limit {
                if is_acyclic_local(m, &edges, mask) {
                    return (0..c).filter(|&i| mask & (1 << i) != 0).map(|i| conns[i]).collect();
                }
                // Gosper's hack: next integer with the same popcount.
                let lowest = mask & mask.wrapping_neg();
                let ripple = mask + lowest;
                mask = (((mask ^ ripple) >> 2) / lowest) | ripple;
            }
        }
        // Unreachable (k == c removes every edge), but stay total.
        return conns;
    }

    dfs_back_edge_conns(scc, n_blocks, sn)
}

/// Is the SCC acyclic once the connections selected by `removed_mask` are cut?
/// `edges` are `(local_src, local_tgt, bit)` over `0..m` local block indices;
/// an edge is active iff its `bit` is clear in `removed_mask`. Iterative
/// white/gray/black DFS over the `m` local nodes.
fn is_acyclic_local(m: usize, edges: &[(usize, usize, usize)], removed_mask: u32) -> bool {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); m];
    for &(s, t, bit) in edges {
        if removed_mask & (1 << bit) == 0 {
            adj[s].push(t);
        }
    }
    const WHITE: u8 = 0;
    const GRAY: u8 = 1;
    const BLACK: u8 = 2;
    let mut color = vec![WHITE; m];
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for root in 0..m {
        if color[root] != WHITE {
            continue;
        }
        color[root] = GRAY;
        stack.push((root, 0));
        while let Some(&(blk, ci)) = stack.last() {
            if ci < adj[blk].len() {
                stack.last_mut().unwrap().1 += 1;
                let t = adj[blk][ci];
                match color[t] {
                    GRAY => return false, // edge to a stack ancestor -> cycle
                    WHITE => {
                        color[t] = GRAY;
                        stack.push((t, 0));
                    }
                    _ => {}
                }
            } else {
                color[blk] = BLACK;
                stack.pop();
            }
        }
    }
    true
}

/// DFS back-edge feedback connection set for large SCCs: a connection is torn
/// iff one of its edges points to a block still on the DFS stack (a gray
/// ancestor). Only true back-edges are cut, so the result is a valid feedback
/// arc set (the remainder is a DAG) though not necessarily of minimum size.
fn dfs_back_edge_conns(scc: &[NodeId], n_blocks: usize, sn: &[Vec<(NodeId, usize)>]) -> Vec<usize> {
    const WHITE: u8 = 0;
    const GRAY: u8 = 1;
    const BLACK: u8 = 2;
    let mut color = vec![WHITE; n_blocks];
    let mut torn: Vec<usize> = Vec::new();
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut stack: Vec<(NodeId, usize)> = Vec::new();
    for &root in scc {
        if color[root] != WHITE {
            continue;
        }
        color[root] = GRAY;
        stack.push((root, 0));
        while let Some(&(blk, ci)) = stack.last() {
            if ci < sn[blk].len() {
                stack.last_mut().unwrap().1 += 1;
                let (tgt, cid) = sn[blk][ci];
                match color[tgt] {
                    GRAY if seen.insert(cid) => torn.push(cid),
                    GRAY => {}
                    WHITE => {
                        color[tgt] = GRAY;
                        stack.push((tgt, 0));
                    }
                    _ => {}
                }
            } else {
                color[blk] = BLACK;
                stack.pop();
            }
        }
    }
    torn
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build roles from `is_alg` flags.
    /// Tests here only exercise DAG/loop layering, not engine classification.
    fn alg_roles(flags: &[bool]) -> Vec<NodeRole> {
        flags.iter().map(|&a| NodeRole { is_alg: a }).collect()
    }

    #[test]
    fn test_empty_graph() {
        let g = Schedule::new(&[], &[]);
        assert_eq!(g.depth(), (0, 0));
        assert!(!g.has_loops);
    }

    #[test]
    fn test_simple_dag() {
        // A(non-alg) -> B(alg) -> C(alg)
        let blocks = alg_roles(&[false, true, true]);
        let connections = vec![(0, 1, 0), (1, 2, 1)];
        let g = Schedule::new(&blocks, &connections);
        assert!(!g.has_loops);
        assert_eq!(g.alg_depth, 3);
    }

    #[test]
    fn test_parallel_dag() {
        let blocks = alg_roles(&[false, true, true]);
        let connections = vec![(0, 1, 0), (0, 2, 1)];
        let g = Schedule::new(&blocks, &connections);
        assert!(!g.has_loops);
        assert_eq!(g.alg_depth, 2);
    }

    #[test]
    fn test_simple_loop() {
        let blocks = alg_roles(&[true, true]);
        let connections = vec![(0, 1, 0), (1, 0, 1)];
        let g = Schedule::new(&blocks, &connections);
        assert!(g.has_loops);
        assert!(g.loop_depth > 0);
        assert!(!g.loop_closing_connections().is_empty());
    }

    #[test]
    fn port_feedthrough_breaks_unread_input_loop() {
        use crate::utils::feedthrough_loops::{Feedthrough, PortEdge};
        // Same topology two ways. Block 0 = M (2-in, 1-out, out reads in0 only);
        // block 1 = A (1-in, 1-out). Edges M.out0 -> A.in0, A.out0 -> M.in1.
        let edges = vec![(0usize, 1usize, 0usize), (1, 0, 1)];

        // Block-level (both algebraic): the M<->A cycle is an algebraic loop.
        let block_level = Schedule::new(&alg_roles(&[true, true]), &edges);
        assert!(block_level.has_loops, "block-level: M<->A is an algebraic loop");

        // Port-level: M's in1 is unread, so the feedback into it is no dependency.
        let ft = vec![
            Feedthrough::new(2, 1, vec![true, false]), // M: out0<-in0, in1 unread
            Feedthrough::new(1, 1, vec![true]),        // A: full feedthrough
        ];
        let port_edges = vec![
            PortEdge { src: 0, out_port: 0, tgt: 1, in_port: 0 }, // M.out0 -> A.in0
            PortEdge { src: 1, out_port: 0, tgt: 0, in_port: 1 }, // A.out0 -> M.in1 (unread)
        ];
        let port_level = Schedule::new_with_feedthrough(&edges, ft, port_edges);
        assert!(!port_level.has_loops, "port-level: feedback into unread in1 is not a loop");
    }

    #[test]
    fn minimal_tear_set_cuts_only_back_edges() {
        // SCC 0->1->2->0 (cycle) plus the forward chord 0->2. A single back-edge
        // (2->0) breaks the cycle; the chord is not part of any cycle. The DFS
        // tear cuts exactly one connection, where a BFS-layering heuristic would
        // over-cut the chord as a cross-edge too.
        let blocks = alg_roles(&[true, true, true]);
        let connections = vec![(0, 1, 0), (1, 2, 1), (2, 0, 2), (0, 2, 3)];
        let g = Schedule::new(&blocks, &connections);
        assert!(g.has_loops);
        assert_eq!(g.loop_closing_connections().len(), 1,
                   "only the genuine back-edge is torn, not the forward chord");
    }

    #[test]
    fn test_topo_order_feedforward() {
        // A(non-alg) -> B(alg) -> C(alg): depths 0,1,2 -> order [0,1,2].
        let blocks = alg_roles(&[false, true, true]);
        let connections = vec![(0, 1, 0), (1, 2, 1)];
        let g = Schedule::new(&blocks, &connections);
        assert_eq!(g.topo_order(), vec![0, 1, 2]);
        assert!(g.algebraic_loops().next().is_none(), "acyclic: no loops");
    }

    #[test]
    fn test_algebraic_loops_exposed() {
        // Two algebraic blocks in a cycle: one SCC {0,1}, with back-edge cut.
        let blocks = alg_roles(&[true, true]);
        let connections = vec![(0, 1, 0), (1, 0, 1)];
        let g = Schedule::new(&blocks, &connections);
        let loops: Vec<_> = g.algebraic_loops().collect();
        assert_eq!(loops.len(), 1, "exactly one algebraic loop");
        let (blks, back) = loops[0];
        let mut bs = blks.to_vec();
        bs.sort_unstable();
        assert_eq!(bs, vec![0, 1]);
        assert!(!back.is_empty(), "loop must report a back-edge cut");
        // topo order covers every block (loop members appended after the DAG part).
        let mut order = g.topo_order();
        order.sort_unstable();
        assert_eq!(order, vec![0, 1]);
    }

    #[test]
    fn test_only_dynamic_blocks() {
        let blocks = alg_roles(&[false, false]);
        let connections = vec![(0, 1, 0)];
        let g = Schedule::new(&blocks, &connections);
        assert!(!g.has_loops);
        assert_eq!(g.alg_depth, 1);
    }

    #[test]
    fn test_self_loop() {
        let blocks = alg_roles(&[true]);
        let connections = vec![(0, 0, 0)];
        let g = Schedule::new(&blocks, &connections);
        assert!(g.has_loops);
    }

    #[test]
    fn test_dag_level_accessors() {
        let blocks = alg_roles(&[false, true, true]);
        let connections = vec![(0, 1, 0), (1, 2, 1)];
        let g = Schedule::new(&blocks, &connections);
        // Verify blocks are in correct depth levels via dag_iter
        let levels: Vec<_> = g.dag_iter().collect();
        assert!(levels[0].0.contains(&0)); // depth 0: non-alg block
        assert!(levels[1].0.contains(&1)); // depth 1: B
        assert!(levels[2].0.contains(&2)); // depth 2: C
    }

    #[test]
    fn test_size() {
        let blocks = alg_roles(&[false, true, true]);
        let connections = vec![(0, 1, 0), (1, 2, 1)];
        let g = Schedule::new(&blocks, &connections);
        assert_eq!(g.size(), (3, 2));
    }

    #[test]
    fn test_outgoing_connections() {
        let blocks = alg_roles(&[false, true, true]);
        let connections = vec![(0, 1, 0), (0, 2, 1), (1, 2, 2)];
        let g = Schedule::new(&blocks, &connections);
        let out_0 = g.outgoing_connections(0);
        assert_eq!(out_0.len(), 2);
        assert!(out_0.contains(&0));
        assert!(out_0.contains(&1));
        assert_eq!(g.outgoing_connections(1), &[2]);
        assert!(g.outgoing_connections(2).is_empty());
    }

    #[test]
    fn test_is_algebraic_path() {
        let blocks = alg_roles(&[false, true, true]);
        let connections = vec![(0, 1, 0), (1, 2, 1)];
        let g = Schedule::new(&blocks, &connections);
        assert!(g.is_algebraic_path(1, 2));
        assert!(g.is_algebraic_path(0, 2));
        assert!(!g.is_algebraic_path(2, 1));
        assert!(!g.is_algebraic_path(2, 0));
    }

    #[test]
    fn test_is_algebraic_path_to_dynamic() {
        let blocks = alg_roles(&[true, false]);
        let connections = vec![(0, 1, 0)];
        let g = Schedule::new(&blocks, &connections);
        assert!(!g.is_algebraic_path(0, 1));
    }

    #[test]
    fn test_algebraic_self_loop() {
        let blocks = alg_roles(&[true, true]);
        let connections = vec![(0, 1, 0), (1, 0, 1)];
        let g = Schedule::new(&blocks, &connections);
        assert!(g.is_algebraic_path(0, 0));
        assert!(g.is_algebraic_path(1, 1));
    }

    #[test]
    fn test_no_algebraic_self_loop_dynamic() {
        let blocks = alg_roles(&[false, true]);
        let connections = vec![(0, 1, 0), (1, 0, 1)];
        let g = Schedule::new(&blocks, &connections);
        assert!(!g.is_algebraic_path(0, 0));
    }
}
