//! Operating-point linearization: local block models and their assembly into a
//! single global state space model for an interconnected diagram.
//!
//! Linearization here is a **query**, never a mode the system is switched into.
//! Asking a block or a simulation for its linear model leaves everything
//! evaluating its original functions afterwards. This mirrors how the Jacobian
//! is treated everywhere else in the engine (`Operator::jacobian_wrt_state` is
//! recomputed per Newton step rather than linearized-and-reused) and matches
//! pathsim's `to_statespace` API, so models port between the two projects.
//!
//! The per-block Jacobians come from AD over the SSA op-graph
//! (`Operator::dense_jacobian_wrt`), so `(A, B, C, D)` are exact for every
//! traceable block; only genuinely opaque operators (FMU, RNG, arbitrary
//! Python) fall back to central differences.

// ======================================================================================
// Local model
// ======================================================================================

/// A block's local linear model around the current operating point:
///
/// ```text
/// dx/dt = A·x + B·u
///     y = C·x + D·u
/// ```
///
/// All matrices are dense row-major, shaped `A: nx×nx`, `B: nx×nu`,
/// `C: ny×nx`, `D: ny×nu`. Any of the dimensions may be zero (a source has no
/// inputs, a sink no outputs, an algebraic block no states); the corresponding
/// matrices are then empty rather than absent.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LocalModel {
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    pub d: Vec<f64>,
    pub nx: usize,
    pub nu: usize,
    pub ny: usize,
}

impl LocalModel {
    /// An all-zero model of the given shape — the correct linearization of a
    /// block whose output depends on neither its state nor its inputs.
    pub fn zeros(nx: usize, nu: usize, ny: usize) -> Self {
        Self {
            a: vec![0.0; nx * nx],
            b: vec![0.0; nx * nu],
            c: vec![0.0; ny * nx],
            d: vec![0.0; ny * nu],
            nx,
            nu,
            ny,
        }
    }

    /// Read entry `(i, j)` of a row-major `rows × cols` block of this model.
    #[inline]
    pub fn at(m: &[f64], cols: usize, i: usize, j: usize) -> f64 {
        m[i * cols + j]
    }
}

/// Reshape a `dense_jacobian_wrt` result into `rows × cols`, padding with zeros
/// when the operator reports fewer rows/columns than the block's registers
/// declare. A block whose graph does not mention a port still occupies the
/// corresponding row/column in the assembled system, so the shapes have to come
/// from the block, not from the graph.
pub(crate) fn fit(
    jac: Option<(Vec<f64>, usize, usize)>,
    rows: usize,
    cols: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    let Some((values, n_rows, n_cols)) = jac else {
        return out;
    };
    for i in 0..n_rows.min(rows) {
        for j in 0..n_cols.min(cols) {
            out[i * cols + j] = values[i * n_cols + j];
        }
    }
    out
}

// ======================================================================================
// Global assembly
// ======================================================================================

use crate::blocks::block::BlockRef;
use crate::connection::ConnectionRef;
use crate::error::{Result, SimError};

/// The assembled linear model of a whole interconnected diagram, plus the block
/// names for its rows and columns. The labels are what `control.StateSpace`
/// takes as `states` / `inputs` / `outputs`, so the model hands over to
/// python-control without an adapter.
#[derive(Debug, Clone, Default)]
pub struct GlobalModel {
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    pub d: Vec<f64>,
    pub nx: usize,
    pub nu: usize,
    pub ny: usize,
    pub state_labels: Vec<String>,
    pub input_labels: Vec<String>,
    pub output_labels: Vec<String>,
}

/// Row-major `(rows_a × k) · (k × cols_b)`.
fn matmul(a: &[f64], b: &[f64], rows_a: usize, k: usize, cols_b: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows_a * cols_b];
    if k == 0 || rows_a == 0 || cols_b == 0 {
        return out;
    }
    for i in 0..rows_a {
        for p in 0..k {
            let aip = a[i * k + p];
            if aip == 0.0 {
                continue;
            }
            for j in 0..cols_b {
                out[i * cols_b + j] += aip * b[p * cols_b + j];
            }
        }
    }
    out
}

/// Assemble a global linear state space model of an interconnected diagram
/// around its current operating point.
///
/// Every block contributes its local model through [`crate::blocks::block::Block::to_statespace`],
/// which is a pure query. The local models are stacked block diagonally, the
/// connections become an interconnection matrix, and the internal signals are
/// eliminated in one linear solve:
///
/// ```text
/// dx/dt = A_b·x + B_b·v
///     w = C_b·x + D_b·v
///     v = L·w + M·u
/// ```
///
/// substituting gives `(I − L·D_b)·v = L·C_b·x + M·u`, which is solved directly.
/// Unlike substitution along a topological order this resolves algebraic loops
/// that survive the input break instead of rejecting them, and well-posedness of
/// the diagram becomes the invertibility of that matrix.
///
/// `in_cols` holds one entry per external input column, each listing the
/// `(block index, input row)` pairs that column drives — one column may feed
/// several ports, which is what a subsystem interface needs. Existing incoming
/// connections at those ports are cut. `out_rows` holds one
/// `(block index, output row)` per output row, or `None` for an output nothing
/// drives.
///
/// # Errors
/// [`SimError::NotLinearizable`] if a block has no linear model, and
/// [`SimError::LinearizationIllPosed`] if the interconnection is singular.
/// The exogenous-independent part of an assembly: the block-diagonal stack, the
/// interconnection matrix, and the eliminated `G·L·C_b` map. Built once, then
/// reused for however many exogenous quantities are driven through it.
struct Assembly {
    v_off: Vec<usize>,
    w_off: Vec<usize>,
    n_v: usize,
    n_w: usize,
    n_x: usize,
    x_layout: Vec<(usize, usize)>,
    b_b: Vec<f64>,
    d_b: Vec<f64>,
    l: Vec<f64>,
    /// `I − L·D_b`, the interconnection matrix.
    sys: Vec<f64>,
    /// Closed-loop `A`, independent of what drives the system.
    a: Vec<f64>,
    /// Closed-loop `C` over the full internal output vector `w`.
    c_full: Vec<f64>,
}

impl Assembly {
    /// Stack the local models, wire the connections, and eliminate the internal
    /// signals. `broken` names the `(block, input row)` pairs whose incoming
    /// connections are cut because something external drives them.
    fn build(
        blocks: &[BlockRef],
        connections: &[ConnectionRef],
        broken: &std::collections::HashSet<(usize, usize)>,
        t: f64,
    ) -> Result<Self> {
        let n_blocks = blocks.len();

        // Local models first: the state layout follows from what the blocks
        // report, not from their integration engines. A Subsystem carries only a
        // dummy engine whose state is never written, so only its model knows its
        // states.
        let models: Vec<LocalModel> = blocks
            .iter()
            .map(|b| b.borrow().to_statespace(t))
            .collect::<Result<_>>()?;

        let (mut v_off, mut w_off, mut x_off) = (
            vec![0usize; n_blocks],
            vec![0usize; n_blocks],
            vec![0usize; n_blocks],
        );
        let (mut n_v, mut n_w, mut n_x) = (0usize, 0usize, 0usize);
        let mut x_layout: Vec<(usize, usize)> = Vec::new();
        for i in 0..n_blocks {
            v_off[i] = n_v;
            w_off[i] = n_w;
            x_off[i] = n_x;
            n_v += models[i].nu;
            n_w += models[i].ny;
            if models[i].nx > 0 {
                x_layout.push((i, models[i].nx));
                n_x += models[i].nx;
            }
        }

        let mut a_b = vec![0.0; n_x * n_x];
        let mut b_b = vec![0.0; n_x * n_v];
        let mut c_b = vec![0.0; n_w * n_x];
        let mut d_b = vec![0.0; n_w * n_v];
        for i in 0..n_blocks {
            let m = &models[i];
            let (vo, wo, xo) = (v_off[i], w_off[i], x_off[i]);
            for r in 0..m.nx {
                for cc in 0..m.nx {
                    a_b[(xo + r) * n_x + xo + cc] = m.a[r * m.nx + cc];
                }
                for cc in 0..m.nu {
                    b_b[(xo + r) * n_v + vo + cc] = m.b[r * m.nu + cc];
                }
            }
            for r in 0..m.ny {
                for cc in 0..m.nx {
                    c_b[(wo + r) * n_x + xo + cc] = m.c[r * m.nx + cc];
                }
                for cc in 0..m.nu {
                    d_b[(wo + r) * n_v + vo + cc] = m.d[r * m.nu + cc];
                }
            }
        }

        // Interconnection matrix `L`; ports driven from outside are cut.
        let index_of = |target: &BlockRef| -> Option<usize> {
            blocks.iter().position(|b| std::rc::Rc::ptr_eq(b, target))
        };
        let mut l = vec![0.0; n_v * n_w];
        for con in connections {
            let Some(src_i) = index_of(&con.source.block) else {
                continue;
            };
            let src_rows = con.source._get_output_indices();
            for trg in &con.targets {
                let Some(tgt_i) = index_of(&trg.block) else {
                    continue;
                };
                for (s, d) in src_rows.iter().zip(trg._get_input_indices()) {
                    if broken.contains(&(tgt_i, d)) {
                        continue;
                    }
                    if v_off[tgt_i] + d < n_v && w_off[src_i] + s < n_w {
                        l[(v_off[tgt_i] + d) * n_w + w_off[src_i] + s] = 1.0;
                    }
                }
            }
        }

        let lc = matmul(&l, &c_b, n_v, n_w, n_x);
        let ld = matmul(&l, &d_b, n_v, n_w, n_v);

        let mut sys = vec![0.0; n_v * n_v];
        for i in 0..n_v {
            for j in 0..n_v {
                sys[i * n_v + j] = if i == j { 1.0 } else { 0.0 } - ld[i * n_v + j];
            }
        }

        // Well-posedness is a property of the interconnection alone, so it is
        // checked independently of whether there happen to be states or
        // exogenous columns to solve for.
        check_well_posed(&sys, n_v)?;

        let glc = solve_columns(&sys, &lc, n_v, n_x)?;

        let a = {
            let mut a = matmul(&b_b, &glc, n_x, n_v, n_x);
            for i in 0..n_x * n_x {
                a[i] += a_b[i];
            }
            a
        };
        let c_full = {
            let mut c = matmul(&d_b, &glc, n_w, n_v, n_x);
            for i in 0..n_w * n_x {
                c[i] += c_b[i];
            }
            c
        };

        Ok(Self { v_off, w_off, n_v, n_w, n_x, x_layout, b_b, d_b, l, sys, a, c_full })
    }

    /// Row of the internal input vector `v` for `(block index, input port)`.
    fn v_row(&self, block: usize, port: usize) -> usize {
        self.v_off[block] + port
    }

    /// Drive `n_q` exogenous columns through the eliminated interconnection and
    /// return their `(B_q, D_q_full)` contributions.
    ///
    /// The three entry points cover every way something can act on the system:
    /// `ev` injects at the block inputs `v` (an external input replacing a cut
    /// connection), `ex` at the derivatives `dx/dt` and `ew` at the block
    /// outputs `w` (a model parameter, which acts inside the blocks' own
    /// equations rather than through a port). With
    /// `(I − L·D_b)·v = L·C_b·x + (L·E_w + E_v)·q` this gives
    /// `B_q = E_x + B_b·G·(L·E_w + E_v)` and `D_q = E_w + D_b·G·(L·E_w + E_v)`.
    fn drive(
        &self,
        ev: &[f64],
        ex: &[f64],
        ew: &[f64],
        n_q: usize,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        if n_q == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        // rhs = L·E_w + E_v
        let mut rhs = if ew.is_empty() {
            vec![0.0; self.n_v * n_q]
        } else {
            matmul(&self.l, ew, self.n_v, self.n_w, n_q)
        };
        if !ev.is_empty() {
            for i in 0..self.n_v * n_q {
                rhs[i] += ev[i];
            }
        }

        let gq = solve_columns(&self.sys, &rhs, self.n_v, n_q)?;

        let mut b_q = matmul(&self.b_b, &gq, self.n_x, self.n_v, n_q);
        if !ex.is_empty() {
            for i in 0..self.n_x * n_q {
                b_q[i] += ex[i];
            }
        }

        let mut d_q = matmul(&self.d_b, &gq, self.n_w, self.n_v, n_q);
        if !ew.is_empty() {
            for i in 0..self.n_w * n_q {
                d_q[i] += ew[i];
            }
        }

        Ok((b_q, d_q))
    }

    /// Pick the tapped rows out of the full `w`-indexed `C` and `D`.
    fn select(
        &self,
        d_full: &[f64],
        n_q: usize,
        out_rows: &[Option<(usize, usize)>],
    ) -> (Vec<f64>, Vec<f64>) {
        let n_y = out_rows.len();
        let mut c = vec![0.0; n_y * self.n_x];
        let mut d = vec![0.0; n_y * n_q];
        for (row, tap) in out_rows.iter().enumerate() {
            let Some((bi, out)) = tap else { continue };
            let w = self.w_off[*bi] + out;
            c[row * self.n_x..(row + 1) * self.n_x]
                .copy_from_slice(&self.c_full[w * self.n_x..(w + 1) * self.n_x]);
            d[row * n_q..(row + 1) * n_q].copy_from_slice(&d_full[w * n_q..(w + 1) * n_q]);
        }
        (c, d)
    }
}

pub fn assemble_from_ports(
    blocks: &[BlockRef],
    connections: &[ConnectionRef],
    in_cols: &[Vec<(usize, usize)>],
    out_rows: &[Option<(usize, usize)>],
    t: f64,
) -> Result<GlobalModel> {
    let broken: std::collections::HashSet<(usize, usize)> =
        in_cols.iter().flatten().copied().collect();

    let asm = Assembly::build(blocks, connections, &broken, t)?;

    // An external input replaces a cut connection, so it injects at `v`.
    let n_u = in_cols.len();
    let mut ev = vec![0.0; asm.n_v * n_u];
    for (col, pairs) in in_cols.iter().enumerate() {
        for &(bi, row) in pairs {
            let v = asm.v_row(bi, row);
            if v < asm.n_v {
                ev[v * n_u + col] = 1.0;
            }
        }
    }

    let (b, d_full) = asm.drive(&ev, &[], &[], n_u)?;
    let (c, d) = asm.select(&d_full, n_u, out_rows);

    let keys = block_keys(blocks);
    Ok(GlobalModel {
        a: asm.a.clone(),
        b,
        c,
        d,
        nx: asm.n_x,
        nu: n_u,
        ny: out_rows.len(),
        state_labels: state_labels(&asm.x_layout, &keys),
        input_labels: port_labels(in_cols.iter().map(|c| c.first().map(|p| p.0)), &keys),
        output_labels: port_labels(out_rows.iter().map(|r| r.map(|p| p.0)), &keys),
    })
}
/// The steady-state sensitivity of the tapped outputs to a set of model
/// parameters, `dy/dp`, row-major `n_y × n_p`.
///
/// At an operating point the state satisfies `0 = A·x + B_p·p`, so
/// `dx/dp = −A⁻¹·B_p` and `dy/dp = C·dx/dp + D_p`. Both `B_p` and `D_p` come out
/// of the same interconnection elimination as the input columns — a parameter
/// simply enters at `dx/dt` and `w` instead of at `v`.
///
/// `params` names `(block index, parameter name)` pairs. A parameter a block
/// does not carry contributes a zero column rather than an error, so a caller
/// can sweep one name across a whole diagram.
///
/// # Errors
/// [`SimError::NotLinearizable`] if a block has no linear model,
/// [`SimError::LinearizationIllPosed`] if the interconnection is singular, and
/// [`SimError::SingularJacobian`] if `A` is singular, so the operating point is
/// not isolated and the sensitivity is undefined rather than merely large.
pub fn assemble_sensitivity(
    blocks: &[BlockRef],
    connections: &[ConnectionRef],
    params: &[(usize, String)],
    out_rows: &[Option<(usize, usize)>],
    t: f64,
) -> Result<Vec<f64>> {
    let broken = std::collections::HashSet::new();
    let asm = Assembly::build(blocks, connections, &broken, t)?;

    let n_p = params.len();
    let (n_x, n_w, n_y) = (asm.n_x, asm.n_w, out_rows.len());

    // A parameter acts inside the blocks' own equations: `∂f/∂p` at `dx/dt`
    // and `∂g/∂p` at `w`.
    let mut ex = vec![0.0; n_x * n_p];
    let mut ew = vec![0.0; n_w * n_p];
    for (col, (bi, name)) in params.iter().enumerate() {
        let (dfdp, dgdp) = blocks[*bi].borrow().param_sensitivity(name, t)?;

        let x_start: usize = asm
            .x_layout
            .iter()
            .take_while(|(b, _)| b != bi)
            .map(|(_, nx)| *nx)
            .sum();
        for (r, v) in dfdp.iter().enumerate() {
            if x_start + r < n_x {
                ex[(x_start + r) * n_p + col] = *v;
            }
        }
        for (r, v) in dgdp.iter().enumerate() {
            let w = asm.w_off[*bi] + r;
            if w < n_w {
                ew[w * n_p + col] = *v;
            }
        }
    }

    let (b_p, d_p_full) = asm.drive(&[], &ex, &ew, n_p)?;
    let (c, d_p) = asm.select(&d_p_full, n_p, out_rows);

    // dx/dp = −A⁻¹·B_p
    let dxdp = if n_x == 0 || n_p == 0 {
        vec![0.0; n_x * n_p]
    } else {
        let neg_b: Vec<f64> = b_p.iter().map(|v| -v).collect();
        solve_columns(&asm.a, &neg_b, n_x, n_p)
            .map_err(|_| SimError::SingularJacobian { time: t })?
    };

    // dy/dp = C·dx/dp + D_p
    let mut out = matmul(&c, &dxdp, n_y, n_x, n_p);
    for i in 0..n_y * n_p {
        out[i] += d_p[i];
    }
    Ok(out)
}

/// `true` when `sys · z` reproduces `rhs` to a scaled tolerance. The dense LU
/// does not report a singular factorization — a zero pivot surfaces as a
/// non-finite or simply wrong solution — so the solution is verified instead of
/// trusted.
fn residual_ok(sys: &[f64], z: &[f64], rhs: &[f64], n: usize) -> bool {
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..n {
            acc += sys[i * n + j] * z[j];
        }
        if !acc.is_finite() || (acc - rhs[i]).abs() > 1e-8 * (1.0 + rhs[i].abs()) {
            return false;
        }
    }
    true
}

/// Reject a diagram whose interconnection matrix `I − L·D` is singular: an
/// algebraic loop with unity gain survived the input break, so the internal
/// signals are not determined and no linear model exists.
fn check_well_posed(sys: &[f64], n: usize) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let mut ls = crate::optim::linsolve::LinearSolver::default();
    let mut rhs = vec![0.0f64; n];
    rhs[0] = 1.0;
    let mut z = vec![0.0f64; n];
    ls.solve(sys, &rhs, &mut z, n);

    if residual_ok(sys, &z, &rhs, n) {
        Ok(())
    } else {
        Err(SimError::LinearizationIllPosed)
    }
}

/// Solve `sys · X = rhs` column by column. `sys` is row-major `n×n`, `rhs`
/// row-major `n×cols`; the result is row-major `n×cols`.
fn solve_columns(sys: &[f64], rhs: &[f64], n: usize, cols: usize) -> Result<Vec<f64>> {
    let mut out = vec![0.0; n * cols];
    if n == 0 || cols == 0 {
        return Ok(out);
    }

    let mut ls = crate::optim::linsolve::LinearSolver::default();
    let (mut col_rhs, mut col_out) = (vec![0.0f64; n], vec![0.0f64; n]);

    for j in 0..cols {
        for i in 0..n {
            col_rhs[i] = rhs[i * cols + j];
        }
        // The factorization is keyed on `sys`, so only the first column factors.
        ls.solve(sys, &col_rhs, &mut col_out, n);
        if !residual_ok(sys, &col_out, &col_rhs, n) {
            return Err(SimError::LinearizationIllPosed);
        }
        for i in 0..n {
            out[i * cols + j] = col_out[i];
        }
    }
    Ok(out)
}

/// Canonical `TypeName_index` identifier per block, assigned once over all
/// blocks so the state, input and output labels all name the same block the
/// same way.
fn block_keys(blocks: &[BlockRef]) -> Vec<String> {
    let mut counters: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    blocks
        .iter()
        .map(|b| {
            let name = b.borrow().type_name;
            let idx = counters.entry(name).or_insert(0);
            let key = format!("{name}_{idx}");
            *idx += 1;
            key
        })
        .collect()
}

fn state_labels(x_layout: &[(usize, usize)], keys: &[String]) -> Vec<String> {
    let mut labels = Vec::new();
    for &(bi, nx) in x_layout {
        if nx == 1 {
            labels.push(keys[bi].clone());
        } else {
            labels.extend((0..nx).map(|i| format!("{}[{i}]", keys[bi])));
        }
    }
    labels
}

fn port_labels(owners: impl Iterator<Item = Option<usize>>, keys: &[String]) -> Vec<String> {
    owners
        .map(|o| o.map(|bi| keys[bi].clone()).unwrap_or_else(|| "-".into()))
        .collect()
}

// ======================================================================================
// Tests
// ======================================================================================

#[cfg(test)]
mod tests {
    use crate::blocks::constructors::{amplifier, comparator, constant, integrator, sample_hold};
    use crate::error::SimError;

    /// A gain is pure feedthrough: no states, `D = [gain]`.
    #[test]
    fn algebraic_block_is_feedthrough_only() {
        let blk = amplifier(3.0);
        blk.borrow_mut().inputs.resize(1);
        blk.borrow_mut().inputs._data[0] = 1.0;

        let m = blk.borrow().to_statespace(0.0).unwrap();

        assert_eq!((m.nx, m.nu, m.ny), (0, 1, 1));
        assert!(m.a.is_empty() && m.b.is_empty() && m.c.is_empty());
        assert!((m.d[0] - 3.0).abs() < 1e-12, "D = {:?}", m.d);
    }

    /// The integrator model is exact and operating-point independent:
    /// `A = 0, B = I, C = I, D = 0`.
    #[test]
    fn integrator_model_is_exact() {
        let blk = integrator(0.0);
        blk.borrow_mut().set_solver_from(&|iv| crate::solvers::factories::ssprk22_factory()(iv));
        blk.borrow_mut().inputs.resize(1);
        blk.borrow_mut().outputs.resize(1);

        let m = blk.borrow().to_statespace(0.0).unwrap();

        assert_eq!((m.nx, m.nu, m.ny), (1, 1, 1));
        assert!((m.a[0] - 0.0).abs() < 1e-12, "A = {:?}", m.a);
        assert!((m.b[0] - 1.0).abs() < 1e-12, "B = {:?}", m.b);
        assert!((m.c[0] - 1.0).abs() < 1e-12, "C = {:?}", m.c);
        assert!((m.d[0] - 0.0).abs() < 1e-12, "D = {:?}", m.d);
    }

    /// A source has no input-to-output path, so its model is all zeros rather
    /// than absent.
    #[test]
    fn source_has_zero_model() {
        let blk = constant(2.0);
        let m = blk.borrow().to_statespace(0.0).unwrap();
        assert!(m.d.iter().all(|v| *v == 0.0), "D = {:?}", m.d);
    }

    /// Switching blocks refuse rather than returning a differenced number.
    #[test]
    fn switching_block_refuses() {
        let blk = comparator(0.0, (-1.0, 1.0));
        match blk.borrow().to_statespace(0.0) {
            Err(SimError::NotLinearizable(name)) => assert_eq!(name, "Comparator"),
            other => panic!("expected NotLinearizable, got {other:?}"),
        }
    }

    /// A named parameter is addressable and its sensitivity is exact:
    /// `y = gain * u`, so `dy/dgain = u`.
    #[test]
    fn param_sensitivity_matches_analytic() {
        let blk = amplifier(3.0);
        blk.borrow_mut().inputs.resize(1);
        blk.borrow_mut().inputs._data[0] = 4.0;
        blk.borrow_mut().outputs.resize(1);

        assert_eq!(blk.borrow().param_names(), vec!["gain".to_string()]);

        let (dfdp, dgdp) = blk.borrow().param_sensitivity("gain", 0.0).unwrap();
        assert!(dfdp.is_empty(), "stateless block has no dynamics: {dfdp:?}");
        assert_eq!(dgdp.len(), 1);
        assert!((dgdp[0] - 4.0).abs() < 1e-12, "dy/dgain should be u = 4, got {}", dgdp[0]);
    }

    /// An unknown parameter name contributes zeros rather than erroring — the
    /// two operator paths carry independent namespaces, so absence is normal.
    #[test]
    fn unknown_param_is_zero() {
        let blk = amplifier(3.0);
        blk.borrow_mut().inputs.resize(1);
        blk.borrow_mut().inputs._data[0] = 4.0;
        blk.borrow_mut().outputs.resize(1);

        let (_, dgdp) = blk.borrow().param_sensitivity("nope", 0.0).unwrap();
        assert!(dgdp.iter().all(|v| *v == 0.0), "got {dgdp:?}");
    }

    /// Discrete-time blocks are covered by the `set_discrete` choke point, so a
    /// new discrete block cannot silently become linearizable.
    #[test]
    fn discrete_block_refuses() {
        let blk = sample_hold(1.0, 0.0);
        assert!(!blk.borrow().linearizable);
        assert!(matches!(
            blk.borrow().to_statespace(0.0),
            Err(SimError::NotLinearizable(_))
        ));
    }
}

#[cfg(test)]
mod assembly_tests {
    use super::*;
    use crate::blocks::constructors::{adder, amplifier, constant, integrator};
    use crate::connection::Connection;
    use crate::simulation::Simulation;
    use crate::utils::portreference::{Port, PortReference};
    use std::rc::Rc;

    /// `src.out -> tgt.in[port]`
    fn wire(src: &crate::blocks::block::BlockRef, tgt: &crate::blocks::block::BlockRef, port: usize) -> ConnectionRef {
        Rc::new(Connection::new(
            PortReference::new(src.clone(), None),
            vec![PortReference::new(tgt.clone(), Some(vec![Port::Index(port)]))],
        ))
    }

    /// `blk`'s output port `p` as a `PortReference`.
    fn port(blk: &crate::blocks::block::BlockRef, p: usize) -> PortReference {
        PortReference::new(blk.clone(), Some(vec![Port::Index(p)]))
    }

    fn close(got: &[f64], want: &[f64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: {got:?} vs {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-9, "{what}: {got:?} != {want:?}");
        }
    }

    fn assemble(
        sim: &Simulation,
        in_cols: &[Vec<(usize, usize)>],
        out_rows: &[Option<(usize, usize)>],
    ) -> Result<GlobalModel> {
        assemble_from_ports(&sim.blocks, &sim.connections, in_cols, out_rows, 0.0)
    }

    /// u -> gain(3) -> int -> gain(2) -> int, against the analytic model.
    #[test]
    fn cascade_matches_analytic_model() {
        let (src, g1, i1, g2, i2) =
            (constant(0.0), amplifier(3.0), integrator(0.0), amplifier(2.0), integrator(0.0));
        let sim = Simulation::with_defaults(
            vec![src.clone(), g1.clone(), i1.clone(), g2.clone(), i2.clone()],
            vec![wire(&src, &g1, 0), wire(&g1, &i1, 0), wire(&i1, &g2, 0), wire(&g2, &i2, 0)],
        );

        // break at gain1's input (block 1), tap integrator2's output (block 4)
        let m = assemble(&sim, &[vec![(1, 0)]], &[Some((4, 0))]).unwrap();

        assert_eq!((m.nx, m.nu, m.ny), (2, 1, 1));
        close(&m.a, &[0.0, 0.0, 2.0, 0.0], "A");
        close(&m.b, &[3.0, 0.0], "B");
        close(&m.c, &[0.0, 1.0], "C");
        close(&m.d, &[0.0], "D");
    }

    /// Closing a loop with gain `k` around an integrator gives `A = -k`.
    #[test]
    fn negative_feedback_gives_minus_k() {
        let k = 5.0;
        let (r, add, integ, gain) =
            (constant(0.0), adder(Some("+-")), integrator(0.0), amplifier(k));
        let sim = Simulation::with_defaults(
            vec![r.clone(), add.clone(), integ.clone(), gain.clone()],
            vec![wire(&r, &add, 0), wire(&integ, &gain, 0), wire(&gain, &add, 1), wire(&add, &integ, 0)],
        );

        let m = assemble(&sim, &[vec![(1, 0)]], &[Some((2, 0))]).unwrap();

        assert_eq!((m.nx, m.nu, m.ny), (1, 1, 1));
        close(&m.a, &[-k], "A");
        close(&m.b, &[1.0], "B");
        close(&m.c, &[1.0], "C");
        close(&m.d, &[0.0], "D");
    }

    /// An algebraic loop surviving the break is eliminated, not rejected: two
    /// gains of 0.5 around an adder give a closed-loop gain `0.5 / (1 - 0.25)`.
    #[test]
    fn algebraic_loop_is_resolved() {
        let (src, add, a, b) =
            (constant(0.0), adder(Some("++")), amplifier(0.5), amplifier(0.5));
        let sim = Simulation::with_defaults(
            vec![src.clone(), add.clone(), a.clone(), b.clone()],
            vec![wire(&src, &add, 0), wire(&add, &a, 0), wire(&a, &b, 0), wire(&b, &add, 1)],
        );

        let m = assemble(&sim, &[vec![(1, 0)]], &[Some((2, 0))]).unwrap();
        close(&m.d, &[0.5 / (1.0 - 0.25)], "D");
    }

    /// A unity-gain algebraic loop has no linear model and must be rejected.
    #[test]
    fn ill_posed_loop_is_rejected() {
        let (a, b) = (amplifier(1.0), amplifier(1.0));
        let sim = Simulation::with_defaults(
            vec![a.clone(), b.clone()],
            vec![wire(&a, &b, 0), wire(&b, &a, 0)],
        );

        match assemble(&sim, &[], &[Some((0, 0))]) {
            Err(SimError::LinearizationIllPosed) => {}
            other => panic!("expected LinearizationIllPosed, got {other:?}"),
        }
    }

    /// Steady-state parameter sensitivity against a hand-computed reference.
    ///
    /// `src -> gain(k) -> int -> feedback gain(-1)` closes as `dx/dt = k*c - x`,
    /// so the steady state is `x = k*c` and `dy/dk = c`.
    #[test]
    fn steadystate_sensitivity_matches_analytic() {
        let (c_val, k) = (2.0, 3.0);
        let (src, g, integ, fb, add) = (
            constant(c_val),
            amplifier(k),
            integrator(0.0),
            amplifier(1.0),
            adder(Some("+-")),
        );
        // add = g*src - fb(x);  dx/dt = add;  y = x
        let sim = Simulation::with_defaults(
            vec![src.clone(), g.clone(), add.clone(), integ.clone(), fb.clone()],
            vec![
                wire(&src, &g, 0),
                wire(&g, &add, 0),
                wire(&integ, &fb, 0),
                wire(&fb, &add, 1),
                wire(&add, &integ, 0),
            ],
        );

        // `sensitivity` settles the network first: `dy/dgain = u` depends on
        // the actual input value, unlike the constant Jacobians above.
        let mut sim = sim;
        let dydp = sim
            .sensitivity(&[(g.clone(), "gain".to_string())], &[port(&integ, 0)], Some(0.0))
            .unwrap();

        // dx/dt = k*c - x  =>  x_ss = k*c  =>  dy/dk = c
        assert_eq!(dydp.len(), 1);
        assert!(
            (dydp[0] - c_val).abs() < 1e-9,
            "dy/dgain should be the source value {c_val}, got {}",
            dydp[0]
        );
    }

    /// A parameter that does not act on the tapped output has zero sensitivity,
    /// and an unknown name is a zero column rather than an error.
    #[test]
    fn sensitivity_of_unrelated_parameter_is_zero() {
        let (src, g, integ, fb, add) = (
            constant(2.0),
            amplifier(3.0),
            integrator(0.0),
            amplifier(1.0),
            adder(Some("+-")),
        );
        let sim = Simulation::with_defaults(
            vec![src.clone(), g.clone(), add.clone(), integ.clone(), fb.clone()],
            vec![
                wire(&src, &g, 0),
                wire(&g, &add, 0),
                wire(&integ, &fb, 0),
                wire(&fb, &add, 1),
                wire(&add, &integ, 0),
            ],
        );

        let mut sim = sim;
        let dydp = sim
            .sensitivity(
                &[(g.clone(), "not_a_parameter".to_string())],
                &[port(&integ, 0)],
                Some(0.0),
            )
            .unwrap();

        assert!(dydp.iter().all(|v| v.abs() < 1e-12), "got {dydp:?}");
    }

    /// One counter across all label categories, so the same block is named the
    /// same way whether it appears as a state, an input or an output.
    #[test]
    fn labels_are_consistent_across_categories() {
        let (src, g1, i1, g2, i2) =
            (constant(0.0), amplifier(3.0), integrator(0.0), amplifier(2.0), integrator(0.0));
        let sim = Simulation::with_defaults(
            vec![src.clone(), g1.clone(), i1.clone(), g2.clone(), i2.clone()],
            vec![wire(&src, &g1, 0), wire(&g1, &i1, 0), wire(&i1, &g2, 0), wire(&g2, &i2, 0)],
        );

        // break at the SECOND amplifier, tap the SECOND integrator
        let m = assemble(&sim, &[vec![(3, 0)]], &[Some((4, 0))]).unwrap();

        assert_eq!(m.state_labels, vec!["Integrator_0", "Integrator_1"]);
        assert_eq!(m.output_labels, vec!["Integrator_1"]);
        assert_eq!(m.input_labels, vec!["Amplifier_1"]);
    }
}
