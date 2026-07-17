// Standalone JIT / autodiff Python API and traced-block constructors.

use std::rc::Rc;

use pyo3::prelude::*;
use pyo3::exceptions::{PyTypeError, PyValueError};

use crate::blocks::constructors;
use crate::utils::fastcell::FastCell;

use super::PyBlock;
use super::helpers::{extract_initial_value, extract_matrix};

// ======================================================================================
// Traced-block constructors — dynamical-system / function / wrapper / DAE variants
// ======================================================================================

/// Interpret a JIT trace probe. Returns `Ok(true)` to proceed with lazy tracing,
/// `Ok(false)` to bail (the caller returns `Ok(None)` — untraceable), and `Err`
/// to propagate a genuine (non-shape) error. Shape errors (ValueError /
/// IndexError) are treated as "not traceable at this shape, fall back".
fn probe_proceed<T>(py: Python<'_>, probe: PyResult<Option<T>>) -> PyResult<bool> {
    match probe {
        Ok(Some(_)) => Ok(true),
        Ok(None) => Ok(false),
        Err(e) => {
            let is_shape = e.is_instance_of::<pyo3::exceptions::PyValueError>(py)
                || e.is_instance_of::<pyo3::exceptions::PyIndexError>(py);
            if is_shape { Ok(true) } else { Err(e) }
        }
    }
}

/// `probe_proceed` variant that keeps the probe graph on success so the
/// caller can seed it into the `LazyTraced` cache (saving the first runtime
/// retrace when the probe shape matches the resolved shape — the SISO case).
/// `Ok(None)` = not traceable (caller falls back); `Ok(Some(None))` = proceed
/// without a seed (shape error, retrace at the real width); `Ok(Some(Some(g)))`
/// = proceed with the probe graph as seed.
fn probe_keep<T>(py: Python<'_>, probe: PyResult<Option<T>>) -> PyResult<Option<Option<T>>> {
    match probe {
        Ok(Some(g)) => Ok(Some(Some(g))),
        Ok(None) => Ok(None),
        Err(e) => {
            let is_shape = e.is_instance_of::<pyo3::exceptions::PyValueError>(py)
                || e.is_instance_of::<pyo3::exceptions::PyIndexError>(py);
            if is_shape { Ok(Some(None)) } else { Err(e) }
        }
    }
}

/// JIT-compile a DynamicalSystem: traces `func_dyn(x, u, t)` and
/// `func_alg(x, u, t)` into separate graphs; derives `∂f_dyn/∂x` via AD.
#[pyfunction]
#[pyo3(signature = (func_dyn, func_alg, initial_value=None, has_passthrough=false))]
#[allow(non_snake_case)]
pub fn _trace_dynamical_system(
    py: Python<'_>,
    func_dyn: &Bound<'_, PyAny>,
    func_alg: &Bound<'_, PyAny>,
    initial_value: Option<&Bound<'_, PyAny>>,
    has_passthrough: bool,
) -> PyResult<Option<PyBlock>> {
    use super::lazy::{LazyTraced, SigArg};
    use crate::tracer::{trace_with_signature, TraceArg};

    let iv = extract_initial_value(initial_value)?;
    let n = iv.len();
    if n == 0 { return Ok(None); }

    let sig = &[
        TraceArg::Array { name: "x", size: n },
        TraceArg::Array { name: "u", size: 1 },
        TraceArg::Scalar { name: "t" },
    ];
    // Probe both callables: surface genuine (non-shape) errors, but DON'T bail
    // when one operator is untraceable. Each operator is wrapped in its own
    // LazyTraced, which JIT-runs when traceable and transparently falls back to
    // a per-operator Python call otherwise — so a gnarly op_alg no longer forces
    // a traceable op_dyn into Python (and vice versa). Trace everything possible.
    // A trace failure of one operator (Ok(None) OR any error — e.g. branching on
    // a value, an unsupported op) just means THAT operator runs in Python via its
    // LazyTraced fallback; it must not disqualify the other operator. So tolerate
    // every per-operator outcome here and let each LazyTraced self-manage.
    let dyn_probe = trace_with_signature(py, func_dyn, sig).ok().flatten();
    let alg_probe = trace_with_signature(py, func_alg, sig).ok().flatten();
    let dyn_traceable = dyn_probe.is_some();
    let alg_traceable = alg_probe.is_some();
    let any_traceable = dyn_traceable || alg_traceable;
    // Nothing traced → let the caller use its plain Python-callback constructor
    // (identical behaviour, one less LazyTraced layer).
    if !any_traceable { return Ok(None); }

    let sig_args = || vec![SigArg::Array("x"), SigArg::Array("u"), SigArg::Scalar("t")];
    let traced_dyn = LazyTraced::new(func_dyn.clone().unbind(), sig_args(), Some("x"));
    let traced_alg = LazyTraced::new(func_alg.clone().unbind(), sig_args(), None);
    // Seed the caches with the probe graphs — a block whose resolved width
    // matches the probe (SISO) never traces twice.
    if let Some(g) = dyn_probe { traced_dyn.seed(g); }
    if let Some(g) = alg_probe { traced_alg.seed(g); }

    let mut blk = crate::blocks::block::Block::default_block();
    blk.type_name = "DynamicalSystem";
    blk.role = crate::blocks::block::BlockRole {
        is_dyn: true, is_src: false, is_rec: false,
    };
    blk.opaque_feedthrough = has_passthrough; // caller-declared u->y feedthrough
    blk.initial_value = Some(iv.to_vec());
    blk.engine = Some(crate::solvers::solver::Solver::with_defaults(&iv));
    blk.len_fn = Some(Box::new(move |b| {
        if !b._active { 0 } else if has_passthrough { 1 } else { 0 }
    }));

    let traced_dyn_ir = traced_dyn.clone();
    let traced_alg_ir = traced_alg.clone();
    let td = traced_dyn.clone();
    blk.f_dyn = Some(Box::new(move |x, u, t, out| {
        td.call_into(&[x, u, &[t]], out);
    }));
    let ta = traced_alg;
    blk.f_alg = Some(Box::new(move |x, u, t, out| {
        ta.call_into(&[x, u, &[t]], out);
    }));
    let tj = traced_dyn;
    blk.jac_dyn = Some(Box::new(move |x, u, t, out| {
        tj.call_jacobian_into(&[x, u, &[t]], out);
    }));

    // Op atomization for IR / static compile — only when BOTH operators are
    // op-traceable (otherwise the fused math would be incomplete, so the block
    // stays opaque and runs via its per-operator LazyTraced/Python fallback).
    // Both regions are served from the SAME trace caches as the runtime tapes.
    if dyn_traceable && alg_traceable {
        let alg_lazy = traced_alg_ir.op_graph(move |w| vec![n, w, 1]);
        let dyn_lazy = traced_dyn_ir.op_graph(move |w| vec![n, w, 1]);
        blk.op_type_name = Some("DynamicalSystem");
        blk.alg_op = Some(crate::blocks::operator::Operator::graph_only(
            crate::blocks::blockops::RegionGraph::Lazy(alg_lazy),
        ));
        blk.dyn_op = Some(crate::blocks::operator::Operator::graph_only(
            crate::blocks::blockops::RegionGraph::Lazy(dyn_lazy),
        ));
    }

    Ok(Some(PyBlock::wrap(Rc::new(FastCell::new(blk)))))
}

/// JIT-compile a DynamicalFunction: `y = func(u, t)` with lazy-rejit trace.
#[pyfunction]
#[allow(non_snake_case)]
pub fn _trace_dynamical_function(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
) -> PyResult<Option<PyBlock>> {
    use super::lazy::{LazyTraced, SigArg};
    use crate::tracer::{trace_with_signature, TraceArg};

    let probe = trace_with_signature(py, func, &[
        TraceArg::Array { name: "u", size: 1 },
        TraceArg::Scalar { name: "t" },
    ]);
    let Some(seed) = probe_keep(py, probe)? else {
        return Ok(None);
    };

    let callable = func.clone().unbind();
    let traced = LazyTraced::new(
        callable,
        vec![SigArg::Array("u"), SigArg::Scalar("t")],
        None,
    );
    if let Some(g) = seed { traced.seed(g); }
    let mut blk = crate::blocks::block::Block::default_block();
    blk.type_name = "DynamicalFunction";
    let t = traced.clone();
    blk.f_alg = Some(Box::new(move |_x, u, ti, out| {
        t.call_into(&[u, &[ti]], out);
    }));
    // Op atomization for IR / static compile (shape-lazy on `u`, reads time),
    // served from the same trace cache as the runtime tape.
    blk.set_alg_lazy("DynamicalFunction", traced.op_graph(|w| vec![w, 1]));
    Ok(Some(PyBlock::wrap(Rc::new(FastCell::new(blk)))))
}

/// JIT-compile a Wrapper: traces `func(u)` (scheduled-evaluation only).
#[pyfunction]
#[pyo3(signature = (func, period=1.0, tau=0.0))]
#[allow(non_snake_case)]
pub fn _trace_wrapper(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
    period: f64,
    tau: f64,
) -> PyResult<Option<PyBlock>> {
    use super::lazy::{LazyTraced, SigArg};
    use crate::tracer::{trace_with_signature, TraceArg};

    let probe = trace_with_signature(py, func, &[
        TraceArg::Array { name: "u", size: 1 },
    ]);
    let Some(seed) = probe_keep(py, probe)? else {
        return Ok(None);
    };

    let traced = LazyTraced::new(func.clone().unbind(), vec![SigArg::Array("u")], None);
    if let Some(g) = seed { traced.seed(g); }

    // Output buffer sizes itself at the first scheduled evaluation.
    let output: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(Vec::new()));
    let input_snap: Rc<FastCell<Vec<f64>>> = Rc::new(FastCell::new(Vec::new()));

    let mut b = crate::blocks::block::Block::default_block();
    b.type_name = "Wrapper";
    b.role = crate::blocks::block::BlockRole {
        is_dyn: false, is_src: false, is_rec: false,
    };
    b.opaque_feedthrough = false; // wrapper declares no direct feedthrough
    b.len_fn = Some(Box::new(|_| 0));

    // Op-expressible representation for the IR / codegen / compiled-fusion path:
    // a memory (ZOH) slot written every period by the traced effect graph and
    // read back by the alg pass — the same discrete shape SampleHold uses, but
    // the effect is `func(u)` instead of identity. Returns `None` at any width
    // the Python effect does not trace, so an untraceable wrapper degrades
    // gracefully to an opaque extern (interpreted-only) instead of lowering.
    let traced_disc = traced.clone();
    b.set_discrete_lazy("Wrapper", move |ni| {
        use crate::blocks::blockops::{
            mem_read_alg_graph, EventKindSpec, EventSpec, MemSpec, MemTarget,
        };
        let effect = traced_disc.graph_for_key(&[ni])?;
        let no = effect.outputs.len();
        let alg = mem_read_alg_graph(0, no);
        let memory = vec![MemSpec { name: "held".into(), init: vec![0.0; no] }];
        let targets: Vec<MemTarget> =
            (0..no as u32).map(|i| MemTarget { slot: 0, offset: i }).collect();
        let events = vec![EventSpec {
            kind: EventKindSpec::SchedulePeriodic { period, phase: tau },
            effect,
            targets,
        }];
        Some((alg, memory, events))
    });

    let input_upd = input_snap.clone();
    let output_upd = output.clone();
    b.update_fn = Some(Box::new(move |blk, _t| {
        let data = &blk.inputs._data;
        let snap = input_upd.borrow_mut();
        snap.resize(data.len(), 0.0);
        snap.copy_from_slice(data);
        let out = output_upd.borrow();
        for (i, &v) in out.iter().enumerate() {
            blk.outputs.set_single(i, v);
        }
    }));

    let t_evt = traced.clone();
    let input_evt = input_snap.clone();
    let output_evt = output.clone();
    use crate::events::schedule::Schedule;
    let evt = Schedule::new(
        tau, None, period,
        Some(Box::new(move |_t| {
            let u = input_evt.borrow();
            let out = output_evt.borrow_mut();
            t_evt.call_into(&[u], out);
        })),
        crate::constants::TOLERANCE,
    );
    b.events.push(Rc::new(FastCell::new(evt)));

    let output_reset = output.clone();
    b.reset_fn = Some(Box::new(move |blk| {
        blk.inputs.reset();
        blk.outputs.reset();
        let out = output_reset.borrow_mut();
        for v in out.iter_mut() { *v = 0.0; }
    }));

    Ok(Some(PyBlock::wrap(Rc::new(FastCell::new(b)))))
}

/// JIT-compile a MassMatrixDAE: traces `func(x, u, t)`, derives `∂f/∂x` via
/// AD, then constructs a `mass_matrix_dae` block with compiled callbacks.
///
/// Returns `None` on trace failure — Python falls back to the callback path.
#[pyfunction]
#[allow(non_snake_case)]
pub fn _trace_mass_matrix_dae(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
    mass: &Bound<'_, PyAny>,
    initial_value: &Bound<'_, PyAny>,
) -> PyResult<Option<PyBlock>> {
    use super::lazy::{LazyTraced, SigArg};
    use crate::tracer::{trace_with_signature, TraceArg};
    use crate::solvers::stage::Mass;

    let iv = extract_initial_value(Some(initial_value))?;
    let n = iv.len();
    if n == 0 { return Ok(None); }

    let mass_flat = extract_matrix(Some(mass), &[])?;
    if mass_flat.len() != n * n {
        return Err(PyValueError::new_err(format!(
            "mass matrix must be {}×{}; got {} entries", n, n, mass_flat.len()
        )));
    }
    let m = Mass::from_flat(mass_flat, n);

    let probe = trace_with_signature(py, func, &[
        TraceArg::Array { name: "x", size: n },
        TraceArg::Array { name: "u", size: 1 },
        TraceArg::Scalar { name: "t" },
    ]);
    if !probe_proceed(py, probe)? {
        return Ok(None);
    }

    let traced = LazyTraced::new(
        func.clone().unbind(),
        vec![SigArg::Array("x"), SigArg::Array("u"), SigArg::Scalar("t")],
        Some("x"),
    );

    let tc = traced.clone();
    let f_closure = move |x: &[f64], u: &[f64], t: f64| -> Vec<f64> {
        let mut out = Vec::new();
        tc.call_into(&[x, u, &[t]], &mut out);
        out
    };
    let tj = traced;
    let jac_closure: Box<dyn Fn(&[f64], &[f64], f64) -> Vec<f64>> =
        Box::new(move |x: &[f64], u: &[f64], t: f64| -> Vec<f64> {
            let mut out = Vec::new();
            tj.call_jacobian_into(&[x, u, &[t]], &mut out);
            out
        });

    Ok(Some(PyBlock::wrap(constructors::mass_matrix_dae(
        f_closure, m, &iv, Some(jac_closure),
    )?)))
}

/// JIT-compile a FullyImplicitDAE: traces `F(x, xdot, u, t)`, derives both
/// `∂F/∂x` and `∂F/∂ẋ` via AD, and builds a `fully_implicit_dae` block.
#[pyfunction]
#[allow(non_snake_case)]
pub fn _trace_fully_implicit_dae(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
    initial_value: &Bound<'_, PyAny>,
) -> PyResult<Option<PyBlock>> {
    use super::lazy::{LazyTraced, SigArg};
    use crate::tracer::{trace_with_signature, TraceArg};

    let iv = extract_initial_value(Some(initial_value))?;
    let n = iv.len();
    if n == 0 { return Ok(None); }

    let probe = trace_with_signature(py, func, &[
        TraceArg::Array { name: "x", size: n },
        TraceArg::Array { name: "xdot", size: n },
        TraceArg::Array { name: "u", size: 1 },
        TraceArg::Scalar { name: "t" },
    ]);
    if !probe_proceed(py, probe)? {
        return Ok(None);
    }

    let traced = LazyTraced::new_multi_jac(
        func.clone().unbind(),
        vec![
            SigArg::Array("x"),
            SigArg::Array("xdot"),
            SigArg::Array("u"),
            SigArg::Scalar("t"),
        ],
        vec!["x", "xdot"],
    );

    let tc = traced.clone();
    let f_closure = move |x: &[f64], xdot: &[f64], u: &[f64], t: f64| -> Vec<f64> {
        let mut out = Vec::new();
        tc.call_into(&[x, xdot, u, &[t]], &mut out);
        out
    };
    let tjx = traced.clone();
    let jac_x: Box<dyn Fn(&[f64], &[f64], &[f64], f64) -> Vec<f64>> =
        Box::new(move |x: &[f64], xdot: &[f64], u: &[f64], t: f64| {
            let mut out = Vec::new();
            tjx.call_jacobian_idx_into(0, &[x, xdot, u, &[t]], &mut out);
            out
        });
    let tjxd = traced;
    let jac_xdot: Box<dyn Fn(&[f64], &[f64], &[f64], f64) -> Vec<f64>> =
        Box::new(move |x: &[f64], xdot: &[f64], u: &[f64], t: f64| {
            let mut out = Vec::new();
            tjxd.call_jacobian_idx_into(1, &[x, xdot, u, &[t]], &mut out);
            out
        });

    Ok(Some(PyBlock::wrap(constructors::fully_implicit_dae(
        f_closure, &iv, Some(jac_x), Some(jac_xdot),
    )?)))
}

/// JIT-compile a SemiExplicitDAE: traces `f_dyn` and `f_alg` into SSA graphs,
/// differentiates `f_alg` w.r.t. `z` for an analytical `∂g/∂z`, then wraps
/// three interpreted flat-tape functions into a `semi_explicit_dae` block.
///
/// Returns `None` if tracing fails (unsupported op, branching on values, ...).
/// Python falls back to the non-JIT constructor in that case.
#[pyfunction]
#[allow(non_snake_case)]
pub fn _trace_semi_explicit_dae(
    py: Python<'_>,
    f_dyn: &Bound<'_, PyAny>,
    f_alg: &Bound<'_, PyAny>,
    x0: &Bound<'_, PyAny>,
    z0: &Bound<'_, PyAny>,
) -> PyResult<Option<PyBlock>> {
    use super::lazy::{LazyTraced, SigArg};
    use crate::tracer::{trace_with_signature, TraceArg};

    let x0v = extract_initial_value(Some(x0))?;
    let z0v = extract_initial_value(Some(z0))?;
    let n_x = x0v.len();
    let n_z = z0v.len();
    if n_x == 0 { return Ok(None); }

    let sig = &[
        TraceArg::Array { name: "x", size: n_x },
        TraceArg::Array { name: "z", size: n_z },
        TraceArg::Array { name: "u", size: 1 },
        TraceArg::Scalar { name: "t" },
    ];
    for callable in [f_dyn, f_alg] {
        if !probe_proceed(py, trace_with_signature(py, callable, sig))? {
            return Ok(None);
        }
    }

    let sig_args = || vec![
        SigArg::Array("x"),
        SigArg::Array("z"),
        SigArg::Array("u"),
        SigArg::Scalar("t"),
    ];
    let traced_dyn = LazyTraced::new(f_dyn.clone().unbind(), sig_args(), None);
    let traced_alg = LazyTraced::new(f_alg.clone().unbind(), sig_args(), Some("z"));

    let td = traced_dyn;
    let f_dyn_closure = move |x: &[f64], z: &[f64], u: &[f64], t: f64| -> Vec<f64> {
        let mut out = Vec::new();
        td.call_into(&[x, z, u, &[t]], &mut out);
        out
    };
    let ta = traced_alg.clone();
    let f_alg_closure = move |x: &[f64], z: &[f64], u: &[f64], t: f64| -> Vec<f64> {
        let mut out = Vec::new();
        ta.call_into(&[x, z, u, &[t]], &mut out);
        out
    };
    let tj = traced_alg;
    let jac_z_closure: Box<dyn Fn(&[f64], &[f64], &[f64], f64) -> Vec<f64>> =
        Box::new(move |x: &[f64], z: &[f64], u: &[f64], t: f64| {
            let mut out = Vec::new();
            tj.call_jacobian_into(&[x, z, u, &[t]], &mut out);
            out
        });

    Ok(Some(PyBlock::wrap(constructors::semi_explicit_dae(
        f_dyn_closure, f_alg_closure, &x0v, &z0v, Some(jac_z_closure),
    )?)))
}

/// JIT-compile an AlgebraicConstraint: trace `residual(x, u)` and derive its
/// Jacobian `∂F/∂x` via AD. Inputs `u` are dynamic (probed at size 1, retraced
/// at runtime), so no input count is declared — like every other traced block.
#[pyfunction]
#[pyo3(signature = (residual, x0))]
pub fn _trace_algebraic_constraint(
    py: Python<'_>,
    residual: &Bound<'_, PyAny>,
    x0: &Bound<'_, PyAny>,
) -> PyResult<Option<PyBlock>> {
    use super::lazy::{LazyTraced, SigArg};
    use crate::tracer::{trace_with_signature, TraceArg};

    let x0v = extract_initial_value(Some(x0))?;
    let n = x0v.len();
    if n == 0 { return Ok(None); }

    let sig = &[
        TraceArg::Array { name: "x", size: n },
        TraceArg::Array { name: "u", size: 1 },
    ];
    if !probe_proceed(py, trace_with_signature(py, residual, sig))? {
        return Ok(None);
    }

    let sig_args = || vec![SigArg::Array("x"), SigArg::Array("u")];
    let traced = LazyTraced::new(residual.clone().unbind(), sig_args(), Some("x"));

    let tr = traced.clone();
    let res_c: Box<dyn Fn(&[f64], &[f64], &mut Vec<f64>)> =
        Box::new(move |x: &[f64], u: &[f64], o: &mut Vec<f64>| tr.call_into(&[x, u], o));
    let tj = traced;
    let jac_c: Box<dyn Fn(&[f64], &[f64], &mut Vec<f64>)> =
        Box::new(move |x: &[f64], u: &[f64], o: &mut Vec<f64>| tj.call_jacobian_into(&[x, u], o));

    Ok(Some(PyBlock::wrap(constructors::algebraic_constraint(res_c, jac_c, x0v.to_vec()))))
}

/// JIT-compile a BVP1D (native scipy.solve_bvp with free parameters + interior
/// conditions): trace `fun(x, y, p, inputs)`, `bc(ya, yb, p, inputs)` and the
/// optional `icond(y_ports, p, inputs)`; AD supplies every Jacobian block. Builds
/// a `bvp1d` block whose native collocation solver runs each evaluation with the
/// current inputs as boundary/parameter data.
#[pyfunction]
#[pyo3(signature = (fun, bc, icond, n_eq, n_params, x0, y0, p0, x_query, x_ports, tol=1e-6))]
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn _trace_bvp1d(
    py: Python<'_>,
    fun: &Bound<'_, PyAny>,
    bc: &Bound<'_, PyAny>,
    icond: Option<&Bound<'_, PyAny>>,
    n_eq: usize,
    n_params: usize,
    x0: &Bound<'_, PyAny>,
    y0: &Bound<'_, PyAny>,
    p0: &Bound<'_, PyAny>,
    x_query: &Bound<'_, PyAny>,
    x_ports: &Bound<'_, PyAny>,
    tol: f64,
) -> PyResult<Option<PyBlock>> {
    use super::lazy::{LazyTraced, SigArg};
    use crate::blocks::constructors::Bvp1dClosures;
    use crate::tracer::{trace_with_signature, TraceArg};

    let x0v = extract_initial_value(Some(x0))?;
    let y0v = extract_initial_value(Some(y0))?;
    let p0v = if n_params > 0 { extract_initial_value(Some(p0))? } else { vec![] };
    let xqv = extract_initial_value(Some(x_query))?;
    let xports: Vec<f64> = if x_ports.is_none() { vec![] } else { extract_initial_value(Some(x_ports))? };
    let n = n_eq;
    let k = n_params;
    let n_ports = xports.len();
    if n < 1 || x0v.len() < 2 {
        return Ok(None);
    }
    let kp = k.max(1); // probe sizes (≥1 so the trace never sees a 0-length array)
    let mp = 1usize; // inputs probed at size 1; the real shape is retraced at runtime

    let fun_sig = &[
        TraceArg::Scalar { name: "x" },
        TraceArg::Array { name: "y", size: n },
        TraceArg::Array { name: "p", size: kp },
        TraceArg::Array { name: "inputs", size: mp },
    ];
    let bc_sig = &[
        TraceArg::Array { name: "ya", size: n },
        TraceArg::Array { name: "yb", size: n },
        TraceArg::Array { name: "p", size: kp },
        TraceArg::Array { name: "inputs", size: mp },
    ];
    if !probe_proceed(py, trace_with_signature(py, fun, fun_sig))? {
        return Ok(None);
    }
    // Infer the number of boundary conditions from the bc trace's output count
    // (input-size-independent), so no input/BC counts need to be declared.
    let n_bc = match trace_with_signature(py, bc, bc_sig) {
        Ok(Some(g)) => g.outputs.len(),
        Ok(None) => return Ok(None),
        Err(e) => {
            let is_shape = e.is_instance_of::<PyValueError>(py)
                || e.is_instance_of::<pyo3::exceptions::PyIndexError>(py);
            if is_shape { return Ok(None); } else { return Err(e); }
        }
    };
    let n_ic = (n_eq + n_params).saturating_sub(n_bc);

    let fun_targets = if k > 0 { vec!["y", "p"] } else { vec!["y"] };
    let fun_t = LazyTraced::new_multi_jac(
        fun.clone().unbind(),
        vec![SigArg::Scalar("x"), SigArg::Array("y"), SigArg::Array("p"), SigArg::Array("inputs")],
        fun_targets,
    );
    let bc_targets = if k > 0 { vec!["ya", "yb", "p"] } else { vec!["ya", "yb"] };
    let bc_t = LazyTraced::new_multi_jac(
        bc.clone().unbind(),
        vec![SigArg::Array("ya"), SigArg::Array("yb"), SigArg::Array("p"), SigArg::Array("inputs")],
        bc_targets,
    );

    let f0 = fun_t.clone();
    let fun_c = Box::new(move |x: f64, y: &[f64], p: &[f64], i: &[f64], o: &mut Vec<f64>| {
        f0.call_into(&[&[x], y, p, i], o)
    });
    let f1 = fun_t.clone();
    let jfy_c = Box::new(move |x: f64, y: &[f64], p: &[f64], i: &[f64], o: &mut Vec<f64>| {
        f1.call_jacobian_idx_into(0, &[&[x], y, p, i], o)
    });
    let jfp_c: Box<dyn Fn(f64, &[f64], &[f64], &[f64], &mut Vec<f64>)> = if k > 0 {
        let f2 = fun_t;
        Box::new(move |x: f64, y: &[f64], p: &[f64], i: &[f64], o: &mut Vec<f64>| {
            f2.call_jacobian_idx_into(1, &[&[x], y, p, i], o)
        })
    } else {
        Box::new(|_x, _y, _p, _i, o: &mut Vec<f64>| o.clear())
    };

    let b0 = bc_t.clone();
    let bc_c = Box::new(move |a: &[f64], b: &[f64], p: &[f64], i: &[f64], o: &mut Vec<f64>| {
        b0.call_into(&[a, b, p, i], o)
    });
    let b1 = bc_t.clone();
    let jba_c = Box::new(move |a: &[f64], b: &[f64], p: &[f64], i: &[f64], o: &mut Vec<f64>| {
        b1.call_jacobian_idx_into(0, &[a, b, p, i], o)
    });
    let b2 = bc_t.clone();
    let jbb_c = Box::new(move |a: &[f64], b: &[f64], p: &[f64], i: &[f64], o: &mut Vec<f64>| {
        b2.call_jacobian_idx_into(1, &[a, b, p, i], o)
    });
    let jbp_c: Box<dyn Fn(&[f64], &[f64], &[f64], &[f64], &mut Vec<f64>)> = if k > 0 {
        let b3 = bc_t;
        Box::new(move |a: &[f64], b: &[f64], p: &[f64], i: &[f64], o: &mut Vec<f64>| {
            b3.call_jacobian_idx_into(2, &[a, b, p, i], o)
        })
    } else {
        Box::new(|_a, _b, _p, _i, o: &mut Vec<f64>| o.clear())
    };

    let (ic_c, jicy_c, jicp_c): (
        Box<dyn Fn(&[f64], &[f64], &[f64], &mut Vec<f64>)>,
        Box<dyn Fn(&[f64], &[f64], &[f64], &mut Vec<f64>)>,
        Box<dyn Fn(&[f64], &[f64], &[f64], &mut Vec<f64>)>,
    ) = if n_ports > 0 {
        let ic = icond.ok_or_else(|| PyValueError::new_err("x_ports given but icond is None"))?;
        let ic_sig = &[
            TraceArg::Array { name: "y_ports", size: n_ports * n },
            TraceArg::Array { name: "p", size: kp },
            TraceArg::Array { name: "inputs", size: mp },
        ];
        if !probe_proceed(py, trace_with_signature(py, ic, ic_sig))? {
            return Ok(None);
        }
        let ic_targets = if k > 0 { vec!["y_ports", "p"] } else { vec!["y_ports"] };
        let ic_t = LazyTraced::new_multi_jac(
            ic.clone().unbind(),
            vec![SigArg::Array("y_ports"), SigArg::Array("p"), SigArg::Array("inputs")],
            ic_targets,
        );
        let i0 = ic_t.clone();
        let i1 = ic_t.clone();
        let jicp: Box<dyn Fn(&[f64], &[f64], &[f64], &mut Vec<f64>)> = if k > 0 {
            let i2 = ic_t;
            Box::new(move |yp: &[f64], p: &[f64], i: &[f64], o: &mut Vec<f64>| {
                i2.call_jacobian_idx_into(1, &[yp, p, i], o)
            })
        } else {
            Box::new(|_y, _p, _i, o: &mut Vec<f64>| o.clear())
        };
        (
            Box::new(move |yp: &[f64], p: &[f64], i: &[f64], o: &mut Vec<f64>| i0.call_into(&[yp, p, i], o)),
            Box::new(move |yp: &[f64], p: &[f64], i: &[f64], o: &mut Vec<f64>| {
                i1.call_jacobian_idx_into(0, &[yp, p, i], o)
            }),
            jicp,
        )
    } else {
        (
            Box::new(|_y, _p, _i, o: &mut Vec<f64>| o.clear()),
            Box::new(|_y, _p, _i, o: &mut Vec<f64>| o.clear()),
            Box::new(|_y, _p, _i, o: &mut Vec<f64>| o.clear()),
        )
    };

    let closures = Bvp1dClosures {
        fun: fun_c, jac_fy: jfy_c, jac_fp: jfp_c,
        bc: bc_c, jac_bc_ya: jba_c, jac_bc_yb: jbb_c, jac_bc_p: jbp_c,
        icond: ic_c, jac_ic_y: jicy_c, jac_ic_p: jicp_c,
    };

    Ok(Some(PyBlock::wrap(constructors::bvp1d(
        closures, n_eq, n_params, n_bc, xports, n_ic, x0v, y0v, p0v, xqv, tol,
    ))))
}

// ======================================================================================
// Standalone JIT API: jit() and jacobian()
// ======================================================================================

/// A JIT-compiled function. Lazy: traces on first call to infer dimensions.
/// Re-traces transparently when a later call passes an `x` of different
/// length (the tape is specialized to the traced shape; evaluating it with a
/// mismatched slice would read out of bounds).
#[pyclass(name = "JitFunction", unsendable)]
pub struct PyJitFunction {
    func: Py<PyAny>,
    interp: Option<crate::ssa::tape::InterpretedFn>,
    /// `x` length the cached tape was traced with.
    traced_n: usize,
}

#[pymethods]
impl PyJitFunction {
    #[pyo3(signature = (*args))]
    fn __call__(&mut self, py: Python<'_>, args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Py<PyAny>> {
        let (x, t) = parse_jit_call_args(args)?;
        if self.interp.is_none() || self.traced_n != x.len() {
            let func = self.func.bind(py);
            let graph = crate::tracer::trace_with_signature(py, func, &[
                crate::tracer::TraceArg::Array { name: "x", size: x.len() },
                crate::tracer::TraceArg::Scalar { name: "t" },
            ])?.ok_or_else(|| PyValueError::new_err("Failed to trace function"))?;
            self.interp = Some(crate::ssa::tape::InterpretedFn::from_graph(graph));
            self.traced_n = x.len();
        }
        let result = self.interp.as_ref().unwrap().call(&[&x, &[t]]);
        let np = py.import("numpy")?;
        if result.len() == 1 {
            Ok(result[0].into_pyobject(py)?.into_any().unbind())
        } else {
            Ok(np.call_method1("array", (result,))?.unbind())
        }
    }

    fn __repr__(&self) -> String {
        match &self.interp {
            Some(i) => format!("JitFunction(compiled, n_out={})", i.n_out),
            None => "JitFunction(pending)".to_string(),
        }
    }
}

/// A JIT-compiled Jacobian function. Lazy: traces + AD on first call.
/// Re-traces transparently when a later call passes an `x` of different
/// length (same shape-specialization contract as `JitFunction`).
#[pyclass(name = "JitJacobian", unsendable)]
pub struct PyJitJacobian {
    func: Py<PyAny>,
    interp: Option<crate::ssa::tape::InterpretedFn>,
    n_x: usize,
    n_out: usize,
}

#[pymethods]
impl PyJitJacobian {
    #[pyo3(signature = (*args))]
    fn __call__(&mut self, py: Python<'_>, args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Py<PyAny>> {
        let (x, t) = parse_jit_call_args(args)?;
        if self.interp.is_none() || self.n_x != x.len() {
            let func = self.func.bind(py);
            let graph = crate::tracer::trace_with_signature(py, func, &[
                crate::tracer::TraceArg::Array { name: "x", size: x.len() },
                crate::tracer::TraceArg::Scalar { name: "t" },
            ])?.ok_or_else(|| PyValueError::new_err("Failed to trace function"))?;
            self.n_x = x.len();
            self.n_out = graph.outputs.len();
            let mut jac_graph = crate::ssa::autodiff::jacobian_wrt_slot(&graph, "x")
                .ok_or_else(|| PyValueError::new_err("function has no x argument"))?;
            crate::ssa::optimize::optimize(&mut jac_graph);
            self.interp = Some(crate::ssa::tape::InterpretedFn::from_graph(jac_graph));
        }
        let result = self.interp.as_ref().unwrap().call(&[&x, &[t]]);
        let np = py.import("numpy")?;
        if self.n_x == 1 && self.n_out == 1 {
            Ok(result[0].into_pyobject(py)?.into_any().unbind())
        } else {
            let arr = np.call_method1("array", (result,))?;
            let reshaped = arr.call_method1("reshape", ((self.n_out, self.n_x),))?;
            Ok(reshaped.unbind())
        }
    }

    fn __repr__(&self) -> String {
        match &self.interp {
            Some(_) => format!("JitJacobian(compiled, shape=({}, {}))", self.n_out, self.n_x),
            None => "JitJacobian(pending)".to_string(),
        }
    }
}

/// Parse call args: f(x), f(x, t), or f(scalar) → (x_vec, t_val)
fn parse_jit_call_args(args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<(Vec<f64>, f64)> {
    let x: Vec<f64> = if let Ok(v) = args.get_item(0)?.extract::<Vec<f64>>() {
        v
    } else if let Ok(v) = args.get_item(0)?.extract::<f64>() {
        vec![v]
    } else {
        let list = args.get_item(0)?.call_method0("tolist")?;
        list.extract::<Vec<f64>>()?
    };
    let t: f64 = if args.len() > 1 {
        // A non-numeric `t` must raise, not silently evaluate at t=0.
        let item = args.get_item(1)?;
        item.extract().map_err(|_| {
            PyTypeError::new_err(format!(
                "t must be a number, got {}",
                item.get_type().name().map(|n| n.to_string()).unwrap_or_else(|_| "?".into())
            ))
        })?
    } else { 0.0 };
    Ok((x, t))
}

/// jit(func, n_x=None) → JitFunction.
/// If n_x is given, traces eagerly. Otherwise lazy on first call.
#[pyfunction]
#[pyo3(signature = (func, n_x=None))]
pub(super) fn jit_compile(py: Python<'_>, func: Py<PyAny>, n_x: Option<usize>) -> PyResult<PyJitFunction> {
    let mut jf = PyJitFunction { func, interp: None, traced_n: 0 };
    if let Some(nx) = n_x {
        let f = jf.func.bind(py);
        let graph = crate::tracer::trace_with_signature(py, f, &[
            crate::tracer::TraceArg::Array { name: "x", size: nx },
            crate::tracer::TraceArg::Scalar { name: "t" },
        ])?.ok_or_else(|| PyValueError::new_err("Failed to trace function"))?;
        jf.interp = Some(crate::ssa::tape::InterpretedFn::from_graph(graph));
        jf.traced_n = nx;
    }
    Ok(jf)
}

/// jacobian(func, n_x=None) → JitJacobian.
/// If n_x is given, traces + AD eagerly. Otherwise lazy on first call.
#[pyfunction]
#[pyo3(signature = (func, n_x=None))]
pub(super) fn jit_jacobian(py: Python<'_>, func: Py<PyAny>, n_x: Option<usize>) -> PyResult<PyJitJacobian> {
    let mut jj = PyJitJacobian { func, interp: None, n_x: 0, n_out: 0 };
    if let Some(nx) = n_x {
        let f = jj.func.bind(py);
        let graph = crate::tracer::trace_with_signature(py, f, &[
            crate::tracer::TraceArg::Array { name: "x", size: nx },
            crate::tracer::TraceArg::Scalar { name: "t" },
        ])?.ok_or_else(|| PyValueError::new_err("Failed to trace function"))?;
        jj.n_x = nx;
        jj.n_out = graph.outputs.len();
        let mut jac_graph = crate::ssa::autodiff::jacobian_wrt_slot(&graph, "x")
            .ok_or_else(|| PyValueError::new_err("function has no x argument"))?;
        crate::ssa::optimize::optimize(&mut jac_graph);
        jj.interp = Some(crate::ssa::tape::InterpretedFn::from_graph(jac_graph));
    }
    Ok(jj)
}
