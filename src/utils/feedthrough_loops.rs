//! Port-granular algebraic-loop detection (FT2).
//!
//! The block-level scheduler (`utils::schedule`) treats every algebraic block as a
//! single feedthrough node: any cycle through algebraic blocks is an algebraic
//! loop. That over-approximates. A MIMO block whose output port `o` does not read
//! input port `j` does *not* carry an algebraic path from `j` to `o`, so a cycle
//! that only routes through such non-feedthrough port pairs is a *false* loop and
//! can be scheduled as a DAG.
//!
//! This module decides loops at port granularity using each block's direct-
//! feedthrough matrix (derived from its SSA via
//! [`crate::ssa::autodiff::feedthrough_pattern`]). It builds a directed graph over
//! signal ports — internal edges `in(b,j) -> out(b,o)` where block `b` feeds `j`
//! through to `o`, plus connection edges `out(src,o) -> in(tgt,j)` — and finds the
//! blocks that lie on a true cycle via Tarjan SCC. The result is a subset of the
//! block-level loop set (it only ever *removes* false loops).

use super::schedule::NodeId;

/// One block's direct-feedthrough matrix: `mat[o * n_in + j]` is true iff output
/// port `o` algebraically reads input port `j`. A block with no feedthrough has
/// an all-false matrix (or `n_out == 0`).
#[derive(Debug, Clone, Default)]
pub struct Feedthrough {
    pub n_in: usize,
    pub n_out: usize,
    pub mat: Vec<bool>,
}

impl Feedthrough {
    pub fn new(n_in: usize, n_out: usize, mat: Vec<bool>) -> Self {
        debug_assert_eq!(mat.len(), n_in * n_out, "feedthrough matrix shape");
        Self { n_in, n_out, mat }
    }

    /// A block with no direct feedthrough at all (pure dynamic / source / sink).
    pub fn none(n_in: usize, n_out: usize) -> Self {
        Self { n_in, n_out, mat: vec![false; n_in * n_out] }
    }

    /// Does output port `o` directly read input port `j`?
    #[inline]
    pub fn feeds(&self, o: usize, j: usize) -> bool {
        self.mat[o * self.n_in + j]
    }

    /// Is input port `j` read by *any* output? Blocks evaluate atomically
    /// (`update()` computes all outputs from all current inputs), so for
    /// scheduling/loop purposes a read input couples to every output: a
    /// connection feeding an *unread* input creates no dependency, but a read
    /// input does — even toward an output that does not individually read it.
    #[inline]
    pub fn reads_input(&self, j: usize) -> bool {
        (0..self.n_out).any(|o| self.feeds(o, j))
    }
}

/// A port-aware connection edge: output port `out_port` of `src` feeds input port
/// `in_port` of `tgt`.
#[derive(Debug, Clone, Copy)]
pub struct PortEdge {
    pub src: NodeId,
    pub out_port: usize,
    pub tgt: NodeId,
    pub in_port: usize,
}

/// Blocks that lie on a *true* (port-granular) algebraic cycle, sorted ascending.
///
/// `ft[b]` is block `b`'s feedthrough matrix; `edges` the port-aware connections.
/// A block is included iff one of its internal feedthrough edges `in(b,j) ->
/// out(b,o)` is part of a directed cycle through the connection graph.
pub fn algebraic_loop_blocks(ft: &[Feedthrough], edges: &[PortEdge]) -> Vec<NodeId> {
    let n_blocks = ft.len();

    // Node id layout: all input ports first (block-major), then all output ports.
    let mut in_off = vec![0usize; n_blocks + 1];
    let mut out_off = vec![0usize; n_blocks + 1];
    for b in 0..n_blocks {
        in_off[b + 1] = in_off[b] + ft[b].n_in;
    }
    let n_in_total = in_off[n_blocks];
    for b in 0..n_blocks {
        out_off[b + 1] = out_off[b] + ft[b].n_out;
    }
    let n_nodes = n_in_total + out_off[n_blocks];
    let in_id = |b: usize, j: usize| in_off[b] + j;
    let out_id = |b: usize, o: usize| n_in_total + out_off[b] + o;

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    // Internal feedthrough edges (atomic block evaluation): a read input couples
    // to EVERY output, so `in(b,j) -> out(b,o)` for all `o` whenever input `j` is
    // read by any output. Unread inputs get no edge -> cycles through them break.
    for (b, f) in ft.iter().enumerate() {
        for j in 0..f.n_in {
            if f.reads_input(j) {
                for o in 0..f.n_out {
                    adj[in_id(b, j)].push(out_id(b, o));
                }
            }
        }
    }
    // Connection edges: out(src,o) -> in(tgt,j).
    for e in edges {
        if e.src < n_blocks && e.tgt < n_blocks
            && e.out_port < ft[e.src].n_out && e.in_port < ft[e.tgt].n_in
        {
            adj[out_id(e.src, e.out_port)].push(in_id(e.tgt, e.in_port));
        }
    }

    let (comp, _n_scc) = super::scc::components(&adj);

    // A block is on a true loop iff some internal edge has both endpoints in the
    // same SCC that actually contains a cycle (size > 1 — the port graph is
    // bipartite-ish and has no node self-loops, so size 1 is never a cycle).
    let mut scc_size = vec![0usize; n_nodes];
    for &c in &comp {
        scc_size[c] += 1;
    }
    let mut flagged = vec![false; n_blocks];
    for (b, f) in ft.iter().enumerate() {
        for j in 0..f.n_in {
            if f.reads_input(j) {
                for o in 0..f.n_out {
                    let (ci, co) = (comp[in_id(b, j)], comp[out_id(b, o)]);
                    if ci == co && scc_size[ci] > 1 {
                        flagged[b] = true;
                    }
                }
            }
        }
    }
    (0..n_blocks).filter(|&b| flagged[b]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cycle that routes through an UNREAD input port is not a true algebraic
    /// loop: block A's single output reads only in0, so a feedback into A.in1
    /// (which no output reads) creates no real dependency.
    #[test]
    fn unread_input_breaks_false_loop() {
        // A: 2-in, 1-out, out0 reads in0 only (in1 unread). B: full feedthrough.
        // Block-level sees a cycle A.out0 -> B -> A.in1, but in1 is dead.
        let a = Feedthrough::new(2, 1, vec![true, false]); // out0<-in0, in1 unread
        let ft = vec![a, Feedthrough::new(1, 1, vec![true])];
        let edges = vec![
            PortEdge { src: 0, out_port: 0, tgt: 1, in_port: 0 }, // A.out0 -> B.in0
            PortEdge { src: 1, out_port: 0, tgt: 0, in_port: 1 }, // B.out0 -> A.in1 (unread)
        ];
        assert!(algebraic_loop_blocks(&ft, &edges).is_empty(),
                "feedback into an unread input -> no true loop");
    }

    /// Atomic evaluation: a diagonal MIMO block whose feedback enters a READ
    /// input IS a true loop — to compute the block you need that input, which the
    /// cycle makes depend on the block's own (atomic) output.
    #[test]
    fn diagonal_read_input_is_true_loop() {
        // A diagonal 2x2 (out0<-in0, out1<-in1); cycle A.out0 -> B -> A.in1.
        // in1 is read (by out1) so atomically it couples to out0 too -> real loop.
        let a = Feedthrough::new(2, 2, vec![true, false, false, true]);
        let ft = vec![a, Feedthrough::new(1, 1, vec![true])];
        let edges = vec![
            PortEdge { src: 0, out_port: 0, tgt: 1, in_port: 0 }, // A.out0 -> B.in0
            PortEdge { src: 1, out_port: 0, tgt: 0, in_port: 1 }, // B.out0 -> A.in1 (read)
        ];
        assert_eq!(algebraic_loop_blocks(&ft, &edges), vec![0, 1],
                   "read input under atomic evaluation -> true algebraic loop");
    }

    /// A dynamic block (no feedthrough) breaks any loop through it.
    #[test]
    fn dynamic_block_breaks_loop() {
        // A full-feedthrough, B dynamic (no feedthrough): cycle A<->B is broken.
        let ft = vec![Feedthrough::new(1, 1, vec![true]), Feedthrough::none(1, 1)];
        let edges = vec![
            PortEdge { src: 0, out_port: 0, tgt: 1, in_port: 0 },
            PortEdge { src: 1, out_port: 0, tgt: 0, in_port: 0 },
        ];
        assert!(algebraic_loop_blocks(&ft, &edges).is_empty(),
                "B has no feedthrough -> no algebraic cycle");
    }
}
