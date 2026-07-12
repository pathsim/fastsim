//! Tarjan strongly-connected-components, the single shared implementation.
//!
//! Both the block-level scheduler (`utils::schedule`) and the port-level algebraic-
//! loop detector (`utils::feedthrough_loops`) need SCCs over a `Vec<Vec<usize>>`
//! adjacency. This is that one iterative, allocation-light Tarjan; the callers add
//! their own filtering (real-cycle-only, component map, ...) on top.

/// Strongly-connected components of `adj` (adjacency by node index), each as a
/// node list. Components are returned in finalization order (the order their root
/// closes), with nodes in the order Tarjan pops them off its stack — matching the
/// legacy in-place scheduler behaviour. Singletons are included (callers filter).
pub fn tarjan_sccs(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adj.len();
    let mut index = vec![usize::MAX; n];
    let mut lowlink = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut sccs: Vec<Vec<usize>> = Vec::new();
    let mut next_index = 0usize;

    // Explicit DFS work stack of (node, next-child-cursor); usize::MAX = pre-visit.
    let mut work: Vec<(usize, usize)> = Vec::new();
    for start in 0..n {
        if index[start] != usize::MAX {
            continue;
        }
        work.push((start, usize::MAX));
        while let Some(entry) = work.last_mut() {
            let v = entry.0;
            if entry.1 == usize::MAX {
                index[v] = next_index;
                lowlink[v] = next_index;
                next_index += 1;
                stack.push(v);
                on_stack[v] = true;
                entry.1 = 0;
                continue;
            }
            let ci = entry.1;
            if ci < adj[v].len() {
                entry.1 += 1;
                let w = adj[v][ci];
                if index[w] == usize::MAX {
                    work.push((w, usize::MAX));
                } else if on_stack[w] && index[w] < lowlink[v] {
                    lowlink[v] = index[w];
                }
            } else {
                // Post-visit: close an SCC if v is a root, then fold into parent.
                if lowlink[v] == index[v] {
                    let mut scc = Vec::new();
                    loop {
                        let w = stack.pop().unwrap();
                        on_stack[w] = false;
                        scc.push(w);
                        if w == v {
                            break;
                        }
                    }
                    sccs.push(scc);
                }
                work.pop();
                if let Some(&(parent, _)) = work.last() {
                    if lowlink[v] < lowlink[parent] {
                        lowlink[parent] = lowlink[v];
                    }
                }
            }
        }
    }
    sccs
}

/// Component-id-per-node view of [`tarjan_sccs`] (`comp[node]` = its SCC index),
/// convenient when the caller needs same-component / component-size queries.
pub fn components(adj: &[Vec<usize>]) -> (Vec<usize>, usize) {
    let sccs = tarjan_sccs(adj);
    let mut comp = vec![usize::MAX; adj.len()];
    for (ci, scc) in sccs.iter().enumerate() {
        for &node in scc {
            comp[node] = ci;
        }
    }
    (comp, sccs.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_cycles_and_singletons() {
        // 0->1->2->0 (cycle), 3->0 (feeds in), 3 singleton.
        let adj = vec![vec![1], vec![2], vec![0], vec![0]];
        let sccs = tarjan_sccs(&adj);
        let cyc: Vec<_> = sccs.iter().filter(|s| s.len() > 1).collect();
        assert_eq!(cyc.len(), 1);
        let mut c = cyc[0].clone();
        c.sort();
        assert_eq!(c, vec![0, 1, 2]);
        assert_eq!(sccs.iter().filter(|s| s.len() == 1).count(), 1); // node 3
    }

    #[test]
    fn components_view_consistent() {
        let adj = vec![vec![1], vec![0], vec![]];
        let (comp, n) = components(&adj);
        assert_eq!(comp[0], comp[1]);
        assert_ne!(comp[0], comp[2]);
        assert_eq!(n, 2);
    }
}
