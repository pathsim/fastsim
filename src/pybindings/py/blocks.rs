// PyO3 constructor wrappers for all block types (simple algebraic, LTI, DAE, noise, LUT, subsystem).

use std::rc::Rc;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use num_complex::Complex64;

use crate::blocks::block::BlockRef;
use crate::blocks::constructors;
use crate::connection::Connection;

use super::{PyBlock, PyConnection};
use super::helpers::{
    extract_initial_value, extract_matrix, extract_scalar_f64,
    extract_vec_f64, on_callback_err, to_numpy, warn_param_ignored,
};

/// Declarative helper for the two most common block-constructor shapes.
///
/// - `block!(Name => ctor)` — zero-arg 1:1 wrapper.
/// - `block!(Name => ctor(a = default, b = default, ...))` — scalar f64 args
///   with defaults, passed positionally to the Rust constructor.
///
/// Only use for constructors whose Python signature is a 1:1 mapping;
/// anything that massages arguments (vec-wrapping, tuple splats, non-f64
/// types) must remain a plain `fn`.
macro_rules! block {
    ($name:ident => $ctor:ident) => {
        #[pyfunction]
        #[allow(non_snake_case)]
        pub(super) fn $name() -> PyBlock { PyBlock::wrap(constructors::$ctor()) }
    };
    ($name:ident => $ctor:ident ( $($arg:ident = $default:expr),+ $(,)? )) => {
        #[pyfunction]
        #[pyo3(signature = ( $($arg = $default),+ ))]
        #[allow(non_snake_case)]
        pub(super) fn $name($($arg: f64),+) -> PyBlock {
            PyBlock::wrap(constructors::$ctor($($arg),+))
        }
    };
}

/// Integrator block: dx/dt = u, y = x.
///
/// State dimension adapts dynamically to the number of connected inputs.
///
/// Parameters
/// ----------
/// initial_value : float or list, optional
///     Initial state (default 0.0).
#[pyfunction]
#[pyo3(signature = (initial_value=None))]
#[allow(non_snake_case)]
pub(super) fn Integrator(initial_value: Option<&Bound<'_, PyAny>>) -> PyResult<PyBlock> {
    let iv = extract_initial_value(initial_value)?;
    Ok(PyBlock::wrap(constructors::integrator_vec(&iv)))
}

// Amplifier block: y = gain * u.
block!(Amplifier => amplifier(gain = 1.0));

/// Adder block: y = sum of inputs with optional sign operations.
#[pyfunction]
#[pyo3(signature = (operations=None))]
#[allow(non_snake_case)]
pub(super) fn Adder(operations: Option<&str>) -> PyBlock { PyBlock::wrap(constructors::adder(operations)) }

// Multiplier block: y = product of all inputs.
block!(Multiplier => multiplier);

// Constant block: y = value.
block!(Constant => constant(value = 1.0));

/// Source block: y = func(t).
#[pyfunction]
#[allow(non_snake_case)]
pub(super) fn Source(func: Py<PyAny>) -> PyBlock {
    PyBlock::wrap(constructors::source(move |t: f64| {
        Python::attach(|py| {
            match func.call1(py, (t,)) {
                Ok(r) => extract_scalar_f64(py, &r).unwrap_or(0.0),
                Err(e) => on_callback_err(py, e, 0.0),
            }
        })
    }))
}

/// Scope block: records input signals over time.
#[pyfunction]
#[pyo3(signature = (sampling_period=None, t_wait=0.0, labels=None))]
#[allow(non_snake_case)]
pub(super) fn Scope(sampling_period: Option<f64>, t_wait: f64, labels: Option<Vec<String>>) -> PyBlock {
    PyBlock::wrap(constructors::scope(sampling_period, t_wait, labels.unwrap_or_default()))
}

/// Spectrum block: running Fourier transform of input signals.
#[pyfunction]
#[pyo3(signature = (freq=vec![], t_wait=0.0, alpha=0.0, labels=None))]
#[allow(non_snake_case)]
pub(super) fn Spectrum(freq: Vec<f64>, t_wait: f64, alpha: f64, labels: Option<Vec<String>>) -> PyBlock {
    PyBlock::wrap(constructors::spectrum(freq, t_wait, alpha, 1, labels.unwrap_or_default()))
}

/// Function block: y = func(*u).
#[pyfunction]
#[allow(non_snake_case)]
pub(super) fn Function(func: Py<PyAny>) -> PyBlock {
    PyBlock::wrap(constructors::function(move |u: &[f64]| {
        Python::attach(|py| {
            let py_args = pyo3::types::PyTuple::new(py, u.iter().copied()).unwrap();
            match func.call1(py, py_args) {
                Ok(r) => {
                    if let Ok(v) = r.extract::<f64>(py) { return vec![v]; }
                    if let Ok(v) = r.extract::<Vec<f64>>(py) { return v; }
                    vec![0.0]
                }
                Err(e) => on_callback_err(py, e, vec![0.0]),
            }
        })
    }))
}

/// ODE block: dx/dt = func(x, u, t), y = x.
#[pyfunction]
#[pyo3(signature = (func, initial_value=None, jac=None))]
#[allow(non_snake_case)]
pub(super) fn ODE(func: Py<PyAny>, initial_value: Option<&Bound<'_, PyAny>>, jac: Option<Py<PyAny>>) -> PyResult<PyBlock> {
    let iv = extract_initial_value(initial_value)?;
    let iv_clone = iv.clone();

    let jac_fn: Option<Box<dyn Fn(&[f64], &[f64], f64) -> f64>> = jac.map(|j| {
        Box::new(move |x: &[f64], u: &[f64], t: f64| -> f64 {
            Python::attach(|py| {
                let x_np = to_numpy(py, x);
                let u_np = to_numpy(py, u);
                match j.call1(py, (x_np, u_np, t)) {
                    Ok(r) => r.extract::<f64>(py).unwrap_or(0.0),
                    Err(e) => on_callback_err(py, e, 0.0),
                }
            })
        }) as Box<dyn Fn(&[f64], &[f64], f64) -> f64>
    });

    Ok(PyBlock::wrap(constructors::ode(
        move |x: &[f64], u: &[f64], t: f64| -> Vec<f64> {
            let n = x.len();
            Python::attach(|py| {
                let x_np = to_numpy(py, x);
                let u_np = to_numpy(py, u);
                match func.call1(py, (x_np, u_np, t)) {
                    // The returned derivative must match the state dimension;
                    // a wrong-length return now fails at the boundary with a
                    // clear message (issue #33) instead of deep in the core.
                    Ok(r) => super::helpers::extract_vec_f64_checked(py, &r, n),
                    Err(e) => on_callback_err(py, e, vec![0.0; n]),
                }
            })
        },
        &iv_clone,
        jac_fn,
    )))
}

/// MassMatrixDAE: M·dx/dt = func(x, u, t), y = x.
///
/// Only implicit solvers (ESDIRK32/43/54, DIRK2/3, EUB) handle singular mass
/// matrices; explicit solvers will misbehave.
#[pyfunction]
#[pyo3(signature = (func, mass, initial_value, jac=None))]
#[allow(non_snake_case)]
pub(super) fn MassMatrixDAE(
    func: Py<PyAny>,
    mass: &Bound<'_, PyAny>,
    initial_value: &Bound<'_, PyAny>,
    jac: Option<Py<PyAny>>,
) -> PyResult<PyBlock> {
    use crate::solvers::stage::Mass;

    let iv = extract_initial_value(Some(initial_value))?;
    let n = iv.len();
    if n == 0 {
        return Err(PyValueError::new_err("initial_value must be non-empty"));
    }

    let mass_flat = extract_matrix(Some(mass), &[])?;
    if mass_flat.len() != n * n {
        return Err(PyValueError::new_err(format!(
            "mass matrix must be {}×{} (={} entries), got {}",
            n, n, n * n, mass_flat.len()
        )));
    }
    let m = Mass::from_flat(mass_flat, n);
    let iv_clone = iv.clone();

    let jac_fn: Option<Box<dyn Fn(&[f64], &[f64], f64) -> Vec<f64>>> = jac.map(|j| {
        let expected = n * n;
        Box::new(move |x: &[f64], u: &[f64], t: f64| -> Vec<f64> {
            Python::attach(|py| {
                let x_np = to_numpy(py, x);
                let u_np = to_numpy(py, u);
                match j.call1(py, (x_np, u_np, t)) {
                    Ok(r) => {
                        let v = extract_vec_f64(py, &r);
                        if v.len() == expected { v } else { vec![0.0; expected] }
                    }
                    Err(e) => on_callback_err(py, e, vec![0.0; expected]),
                }
            })
        }) as Box<dyn Fn(&[f64], &[f64], f64) -> Vec<f64>>
    });

    Ok(PyBlock::wrap(constructors::mass_matrix_dae(
        move |x: &[f64], u: &[f64], t: f64| -> Vec<f64> {
            Python::attach(|py| {
                let x_np = to_numpy(py, x);
                let u_np = to_numpy(py, u);
                match func.call1(py, (x_np, u_np, t)) {
                    Ok(r) => extract_vec_f64(py, &r),
                    Err(e) => on_callback_err(py, e, vec![0.0; n]),
                }
            })
        },
        m,
        &iv_clone,
        jac_fn,
    )?))
}

/// SemiExplicitDAE: Index-1 DAE `ẋ = f_dyn(x, z, u, t)`, `0 = f_alg(x, z, u, t)`.
#[pyfunction]
#[pyo3(signature = (f_dyn, f_alg, x0, z0, jac_z=None))]
#[allow(non_snake_case)]
pub(super) fn SemiExplicitDAE(
    f_dyn: Py<PyAny>,
    f_alg: Py<PyAny>,
    x0: &Bound<'_, PyAny>,
    z0: &Bound<'_, PyAny>,
    jac_z: Option<Py<PyAny>>,
) -> PyResult<PyBlock> {
    let x0v = extract_initial_value(Some(x0))?;
    let z0v = extract_initial_value(Some(z0))?;
    let n_x = x0v.len();
    let n_z = z0v.len();
    if n_x == 0 { return Err(PyValueError::new_err("x0 must be non-empty")); }

    let f_dyn_fn = move |x: &[f64], z: &[f64], u: &[f64], t: f64| -> Vec<f64> {
        Python::attach(|py| {
            let x_np = to_numpy(py, x);
            let z_np = to_numpy(py, z);
            let u_np = to_numpy(py, u);
            match f_dyn.call1(py, (x_np, z_np, u_np, t)) {
                Ok(r) => extract_vec_f64(py, &r),
                Err(e) => on_callback_err(py, e, vec![0.0; n_x]),
            }
        })
    };

    let f_alg_fn = move |x: &[f64], z: &[f64], u: &[f64], t: f64| -> Vec<f64> {
        Python::attach(|py| {
            let x_np = to_numpy(py, x);
            let z_np = to_numpy(py, z);
            let u_np = to_numpy(py, u);
            match f_alg.call1(py, (x_np, z_np, u_np, t)) {
                Ok(r) => extract_vec_f64(py, &r),
                Err(e) => on_callback_err(py, e, vec![0.0; n_z]),
            }
        })
    };

    let jac_z_fn: Option<Box<dyn Fn(&[f64], &[f64], &[f64], f64) -> Vec<f64>>> =
        jac_z.map(|j| {
            let expected = n_z * n_z;
            Box::new(move |x: &[f64], z: &[f64], u: &[f64], t: f64| -> Vec<f64> {
                Python::attach(|py| {
                    let x_np = to_numpy(py, x);
                    let z_np = to_numpy(py, z);
                    let u_np = to_numpy(py, u);
                    match j.call1(py, (x_np, z_np, u_np, t)) {
                        Ok(r) => {
                            let v = extract_vec_f64(py, &r);
                            if v.len() == expected { v } else { vec![0.0; expected] }
                        }
                        Err(e) => on_callback_err(py, e, vec![0.0; expected]),
                    }
                })
            }) as Box<dyn Fn(&[f64], &[f64], &[f64], f64) -> Vec<f64>>
        });

    Ok(PyBlock::wrap(constructors::semi_explicit_dae(
        f_dyn_fn, f_alg_fn, &x0v, &z0v, jac_z_fn,
    )?))
}

/// FullyImplicitDAE: `F(x, ẋ, u, t) = 0`, `y = x`.
#[pyfunction]
#[pyo3(signature = (func, initial_value, jac_x=None, jac_xdot=None))]
#[allow(non_snake_case)]
pub(super) fn FullyImplicitDAE(
    func: Py<PyAny>,
    initial_value: &Bound<'_, PyAny>,
    jac_x: Option<Py<PyAny>>,
    jac_xdot: Option<Py<PyAny>>,
) -> PyResult<PyBlock> {
    let iv = extract_initial_value(Some(initial_value))?;
    let n = iv.len();
    if n == 0 {
        return Err(PyValueError::new_err("initial_value must be non-empty"));
    }

    let f_fn = move |x: &[f64], xdot: &[f64], u: &[f64], t: f64| -> Vec<f64> {
        Python::attach(|py| {
            let x_np = to_numpy(py, x);
            let xd_np = to_numpy(py, xdot);
            let u_np = to_numpy(py, u);
            match func.call1(py, (x_np, xd_np, u_np, t)) {
                Ok(r) => extract_vec_f64(py, &r),
                Err(e) => on_callback_err(py, e, vec![0.0; n]),
            }
        })
    };

    let jac_x_fn: Option<Box<dyn Fn(&[f64], &[f64], &[f64], f64) -> Vec<f64>>> =
        jac_x.map(|j| {
            let expected = n * n;
            Box::new(move |x: &[f64], xdot: &[f64], u: &[f64], t: f64| -> Vec<f64> {
                Python::attach(|py| {
                    let x_np = to_numpy(py, x);
                    let xd_np = to_numpy(py, xdot);
                    let u_np = to_numpy(py, u);
                    match j.call1(py, (x_np, xd_np, u_np, t)) {
                        Ok(r) => {
                            let v = extract_vec_f64(py, &r);
                            if v.len() == expected { v } else { vec![0.0; expected] }
                        }
                        Err(e) => on_callback_err(py, e, vec![0.0; expected]),
                    }
                })
            }) as Box<dyn Fn(&[f64], &[f64], &[f64], f64) -> Vec<f64>>
        });

    let jac_xdot_fn: Option<Box<dyn Fn(&[f64], &[f64], &[f64], f64) -> Vec<f64>>> =
        jac_xdot.map(|j| {
            let expected = n * n;
            Box::new(move |x: &[f64], xdot: &[f64], u: &[f64], t: f64| -> Vec<f64> {
                Python::attach(|py| {
                    let x_np = to_numpy(py, x);
                    let xd_np = to_numpy(py, xdot);
                    let u_np = to_numpy(py, u);
                    match j.call1(py, (x_np, xd_np, u_np, t)) {
                        Ok(r) => {
                            let v = extract_vec_f64(py, &r);
                            if v.len() == expected { v } else { vec![0.0; expected] }
                        }
                        Err(e) => on_callback_err(py, e, vec![0.0; expected]),
                    }
                })
            }) as Box<dyn Fn(&[f64], &[f64], &[f64], f64) -> Vec<f64>>
        });

    Ok(PyBlock::wrap(constructors::fully_implicit_dae(
        f_fn, &iv, jac_x_fn, jac_xdot_fn,
    )?))
}

// LTI / Control blocks
/// State-space block: dx/dt = Ax + Bu, y = Cx + Du.
#[pyfunction]
#[pyo3(signature = (A=None, B=None, C=None, D=None, initial_value=None))]
#[allow(non_snake_case)]
pub(super) fn StateSpace(
    A: Option<&Bound<'_, PyAny>>,
    B: Option<&Bound<'_, PyAny>>,
    C: Option<&Bound<'_, PyAny>>,
    D: Option<&Bound<'_, PyAny>>,
    initial_value: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyBlock> {
    let a_flat = extract_matrix(A, &[-1.0])?;
    let b_flat = extract_matrix(B, &[1.0])?;
    let c_flat = extract_matrix(C, &[1.0])?;
    let d_flat = extract_matrix(D, &[0.0])?;

    let n = (a_flat.len() as f64).sqrt() as usize;
    let n = if n * n == a_flat.len() { n } else { 1 };
    let n_in = b_flat.len().checked_div(n).unwrap_or(1);
    let n_out = c_flat.len().checked_div(n).unwrap_or(1);

    let iv = match initial_value {
        Some(v) => {
            if let Ok(f) = v.extract::<f64>() { Some(vec![f]) }
            else {
                v.extract::<Vec<f64>>().ok()
            }
        }
        None => None,
    };

    Ok(PyBlock::wrap(constructors::statespace(
        a_flat, b_flat, c_flat, d_flat, n, n_in, n_out, iv,
    )))
}

block!(PT1 => pt1(K = 1.0, T = 1.0));
block!(PT2 => pt2(K = 1.0, T = 1.0, d = 1.0));
block!(LeadLag => lead_lag(K = 1.0, T1 = 1.0, T2 = 1.0));
block!(PID => pid(Kp = 0.0, Ki = 0.0, Kd = 0.0, f_max = 100.0));

// Source variants
#[pyfunction]
#[pyo3(signature = (amplitude=None, tau=None))]
#[allow(non_snake_case)]
pub(super) fn StepSource(amplitude: Option<&Bound<'_, PyAny>>, tau: Option<&Bound<'_, PyAny>>) -> PyResult<PyBlock> {
    let amps: Vec<f64> = match amplitude {
        None => vec![1.0],
        Some(a) => if let Ok(v) = a.extract::<f64>() { vec![v] } else { a.extract::<Vec<f64>>()? },
    };
    let times: Vec<f64> = match tau {
        None => vec![0.0],
        Some(t) => if let Ok(v) = t.extract::<f64>() { vec![v] } else { t.extract::<Vec<f64>>()? },
    };
    if amps.len() != times.len() {
        return Err(PyValueError::new_err("'amplitude' and 'tau' must have same length"));
    }
    Ok(PyBlock::wrap(constructors::step_source(amps, times)?))
}
#[pyfunction]
#[pyo3(signature = (amplitude=1.0, tau=0.0))]
#[allow(non_snake_case)]
pub(super) fn Step(amplitude: f64, tau: f64) -> PyBlock {
    // Single-element vecs are always equal-length, so this constructor never
    // errors — an internal invariant, safe to expect().
    PyBlock::wrap(constructors::step_source(vec![amplitude], vec![tau])
        .expect("Step builds equal-length amplitude/tau vecs"))
}
block!(SinusoidalSource => sinusoidal_source(frequency = 1.0, amplitude = 1.0, phase = 0.0));
block!(TriangleWaveSource => triangle_wave_source(frequency = 1.0, amplitude = 1.0, phase = 0.0));
block!(SquareWaveSource => square_wave_source(amplitude = 1.0, frequency = 1.0, phase = 0.0));

#[pyfunction]
#[pyo3(signature = (operations="*/", zero_div="warn"))]
#[allow(non_snake_case)]
pub(super) fn Divider(operations: &str, zero_div: &str) -> PyResult<PyBlock> {
    for c in operations.chars() {
        if c != '*' && c != '/' {
            return Err(PyValueError::new_err(
                format!("invalid operation '{c}', must be '*' or '/'")));
        }
    }
    let zd = match zero_div {
        "warn"  => constructors::ZeroDiv::Warn,
        "clamp" => constructors::ZeroDiv::Clamp,
        "raise" => constructors::ZeroDiv::Raise,
        other => return Err(PyValueError::new_err(
            format!("zero_div must be one of warn/clamp/raise, got '{}'", other))),
    };
    Ok(PyBlock::wrap(constructors::divider(Some(operations), zd)?))
}

// Math blocks
block!(Sin => sin_block);
block!(Cos => cos_block);
block!(Exp => exp_block);
block!(Abs => abs_block);
block!(Sqrt => sqrt_block);
block!(Log => log_block);
block!(Tanh => tanh_block);
block!(Tan => tan_block);
block!(Atan => atan_block);
block!(Sinh => sinh_block);
block!(Cosh => cosh_block);
block!(Log10 => log10_block);
block!(Pow => pow_block(exponent = 2.0));
block!(Clip => clip_block(min_val = -1.0, max_val = 1.0));
#[pyfunction]
#[pyo3(signature = (i0=0.0, i1=1.0, o0=0.0, o1=1.0, saturate=false))]
#[allow(non_snake_case)]
pub(super) fn Rescale(i0: f64, i1: f64, o0: f64, o1: f64, saturate: bool) -> PyBlock {
    let s = (o1 - o0) / (i1 - i0);
    let o = o0 - s * i0;
    let lo = o0.min(o1);
    let hi = o0.max(o1);
    PyBlock::wrap(constructors::rescale_block(s, o, saturate, lo, hi))
}
block!(Atan2 => atan2_block);
block!(Mod => mod_single(modulus = 1.0));
block!(Norm => norm_block);
#[pyfunction]
#[pyo3(signature = (exponents=vec![1.0, 1.0]))]
#[allow(non_snake_case)]
pub(super) fn PowProd(exponents: Vec<f64>) -> PyBlock { PyBlock::wrap(constructors::pow_prod(exponents)) }

/// Polynomial: y = c[0]·u^n + … + c[n] (numpy.polyval convention, vector input).
#[pyfunction]
#[pyo3(signature = (coeffs=vec![1.0, 0.0]))]
#[allow(non_snake_case)]
pub(super) fn Polynomial(coeffs: Vec<f64>) -> PyBlock { PyBlock::wrap(constructors::polynomial(coeffs)) }

#[pyfunction]
#[pyo3(signature = (A))]
#[allow(non_snake_case)]
pub(super) fn Matrix(A: Vec<Vec<f64>>) -> PyBlock {
    let rows = A.len();
    let cols = if rows > 0 { A[0].len() } else { 0 };
    let flat: Vec<f64> = A.into_iter().flatten().collect();
    PyBlock::wrap(constructors::matrix_block(flat, rows, cols))
}
block!(Alias => alias_block);

// Logic blocks
block!(GreaterThan => greater_than);
block!(LessThan => less_than);
block!(Equal => equal_block(tolerance = 1e-12));
block!(LogicAnd => logic_and);
block!(LogicOr => logic_or);
block!(LogicNot => logic_not);

// Source variants
#[pyfunction]
#[pyo3(signature = (amplitude=1.0, T=1.0, t_rise=0.0, t_fall=0.0, tau=0.0, duty=0.5))]
#[allow(non_snake_case)]
pub(super) fn Pulse(amplitude: f64, T: f64, t_rise: f64, t_fall: f64, tau: f64, duty: f64) -> PyBlock {
    PyBlock::wrap(constructors::pulse_source(amplitude, T, t_rise, t_fall, tau, duty))
}
#[pyfunction]
#[pyo3(signature = (T=1.0, tau=0.0))]
#[allow(non_snake_case)]
pub(super) fn Clock(py: Python<'_>, T: f64, tau: f64) -> PyResult<PyBlock> {
    if tau != 0.0 {
        warn_param_ignored(py, "Clock", "tau",
            "the runtime clock source toggles on the raw period T with no phase/delay offset")?;
    }
    Ok(PyBlock::wrap(constructors::clock_source(T)))
}
#[pyfunction]
#[pyo3(signature = (T=1.0, tau=0.0))]
#[allow(non_snake_case)]
pub(super) fn ClockSource(py: Python<'_>, T: f64, tau: f64) -> PyResult<PyBlock> {
    if tau != 0.0 {
        warn_param_ignored(py, "ClockSource", "tau",
            "the runtime clock source toggles on the raw period T with no phase/delay offset")?;
    }
    Ok(PyBlock::wrap(constructors::clock_source(T)))
}
#[pyfunction]
#[pyo3(signature = (amplitude=1.0, T=1.0, t_rise=0.0, t_fall=0.0, tau=0.0, duty=0.5))]
#[allow(non_snake_case)]
pub(super) fn PulseSource(amplitude: f64, T: f64, t_rise: f64, t_fall: f64, tau: f64, duty: f64) -> PyBlock {
    PyBlock::wrap(constructors::pulse_source(amplitude, T, t_rise, t_fall, tau, duty))
}
#[pyfunction]
#[pyo3(signature = (amplitude=1.0, f0=1.0, BW=1.0, T=1.0, phase=0.0,
    sig_cum=0.0, sig_white=0.0, sampling_period=0.1, seed=None))]
#[allow(non_snake_case)]
pub(super) fn ChirpSource(
    amplitude: f64, f0: f64, BW: f64, T: f64, phase: f64,
    sig_cum: f64, sig_white: f64, sampling_period: f64, seed: Option<u64>,
) -> PyBlock {
    PyBlock::wrap(constructors::chirp_phase_noise_source(
        amplitude, f0, BW, T, phase, sig_cum, sig_white, Some(sampling_period), seed,
    ))
}

#[pyfunction]
#[pyo3(signature = (amplitude=1.0, f0=1.0, BW=1.0, T=1.0, phase=0.0,
    sig_cum=0.0, sig_white=0.0, sampling_period=0.1, seed=None))]
#[allow(non_snake_case)]
pub(super) fn ChirpPhaseNoiseSource(
    amplitude: f64, f0: f64, BW: f64, T: f64, phase: f64,
    sig_cum: f64, sig_white: f64, sampling_period: f64, seed: Option<u64>,
) -> PyBlock {
    PyBlock::wrap(constructors::chirp_phase_noise_source(
        amplitude, f0, BW, T, phase, sig_cum, sig_white, Some(sampling_period), seed,
    ))
}

#[pyfunction]
#[pyo3(signature = (frequency=1.0, amplitude=1.0, phase=0.0,
    sig_cum=0.0, sig_white=0.0, sampling_period=0.1, seed=None))]
#[allow(non_snake_case)]
pub(super) fn SinusoidalPhaseNoiseSource(
    frequency: f64, amplitude: f64, phase: f64,
    sig_cum: f64, sig_white: f64, sampling_period: f64, seed: Option<u64>,
) -> PyBlock {
    PyBlock::wrap(constructors::sinusoidal_phase_noise_source(
        frequency, amplitude, phase, sig_cum, sig_white, Some(sampling_period), seed,
    ))
}
block!(GaussianPulseSource => gaussian_pulse_source(amplitude = 1.0, f_max = 1000.0, tau = 0.0));

#[pyfunction]
#[pyo3(signature = (standard_deviation=1.0, spectral_density=None, sampling_period=None, seed=None))]
#[allow(non_snake_case)]
pub(super) fn WhiteNoise(standard_deviation: f64, spectral_density: Option<f64>, sampling_period: Option<f64>, seed: Option<u64>) -> PyBlock {
    PyBlock::wrap(constructors::white_noise(standard_deviation, spectral_density, sampling_period, seed))
}
#[pyfunction]
#[pyo3(signature = (standard_deviation=1.0, spectral_density=None, num_octaves=16, sampling_period=None, seed=None))]
#[allow(non_snake_case)]
pub(super) fn PinkNoise(standard_deviation: f64, spectral_density: Option<f64>, num_octaves: usize, sampling_period: Option<f64>, seed: Option<u64>) -> PyBlock {
    PyBlock::wrap(constructors::pink_noise(standard_deviation, spectral_density, num_octaves, sampling_period, seed))
}
#[pyfunction]
#[pyo3(signature = (sampling_period=None, seed=None))]
#[allow(non_snake_case)]
pub(super) fn RandomNumberGenerator(sampling_period: Option<f64>, seed: Option<u64>) -> PyBlock {
    PyBlock::wrap(constructors::random_number_generator(sampling_period, seed))
}

/// DynamicalSystem: dx/dt = func_dyn(x, u, t), y = func_alg(x, u, t)
#[pyfunction]
#[pyo3(signature = (func_dyn, func_alg, initial_value=None, has_passthrough=false, jac_dyn=None))]
#[allow(non_snake_case)]
pub(super) fn DynamicalSystem(
    func_dyn: Py<PyAny>,
    func_alg: Py<PyAny>,
    initial_value: Option<&Bound<'_, PyAny>>,
    has_passthrough: bool,
    jac_dyn: Option<Py<PyAny>>,
) -> PyResult<PyBlock> {
    let iv = extract_initial_value(initial_value)?;
    let iv_clone = iv.clone();

    let dyn_fn = Python::attach(|py| func_dyn.clone_ref(py));
    let alg_fn = Python::attach(|py| func_alg.clone_ref(py));

    // Wire a user-supplied analytic Jacobian: `jac_dyn(x, u, t)` returns the
    // dense ∂f_dyn/∂x, which `extract_vec_f64` flattens row-major to length
    // n·n — exactly the shape `Block::compute_jacobian` consumes.  Absent, the
    // solver falls back to central differencing.
    let jac_closure: Option<Box<dyn Fn(&[f64], &[f64], f64) -> Vec<f64>>> = jac_dyn.map(|jf| {
        let jac_fn = Python::attach(|py| jf.clone_ref(py));
        Box::new(move |x: &[f64], u: &[f64], t: f64| -> Vec<f64> {
            Python::attach(|py| {
                let x_np = to_numpy(py, x);
                let u_np = to_numpy(py, u);
                match jac_fn.call1(py, (x_np, u_np, t)) {
                    Ok(r) => extract_vec_f64(py, &r),
                    Err(e) => on_callback_err(py, e, Vec::new()),
                }
            })
        }) as Box<dyn Fn(&[f64], &[f64], f64) -> Vec<f64>>
    });

    Ok(PyBlock::wrap(constructors::dynamical_system(
        move |x: &[f64], u: &[f64], t: f64| -> Vec<f64> {
            Python::attach(|py| {
                let x_np = to_numpy(py, x);
                let u_np = to_numpy(py, u);
                match dyn_fn.call1(py, (x_np, u_np, t)) {
                    Ok(r) => extract_vec_f64(py, &r),
                    Err(e) => on_callback_err(py, e, vec![0.0]),
                }
            })
        },
        move |x: &[f64], u: &[f64], t: f64| -> Vec<f64> {
            Python::attach(|py| {
                let x_np = to_numpy(py, x);
                let u_np = to_numpy(py, u);
                match alg_fn.call1(py, (x_np, u_np, t)) {
                    Ok(r) => extract_vec_f64(py, &r),
                    Err(e) => on_callback_err(py, e, vec![0.0]),
                }
            })
        },
        &iv_clone,
        has_passthrough,
        jac_closure,
    )))
}

#[pyfunction]
#[pyo3(signature = (func))]
#[allow(non_snake_case)]
pub(super) fn DynamicalFunction(func: Py<PyAny>) -> PyBlock {
    PyBlock::wrap(constructors::dynamical_function(
        move |u: &[f64], t: f64| -> Vec<f64> {
            Python::attach(|py| {
                let u_np = to_numpy(py, u);
                match func.call1(py, (u_np, t)) {
                    Ok(r) => extract_vec_f64(py, &r),
                    Err(e) => on_callback_err(py, e, vec![0.0]),
                }
            })
        },
    ))
}

block!(Differentiator => differentiator(f_max = 100.0));

#[pyfunction]
#[pyo3(signature = (tau=0.001, sampling_period=None))]
#[allow(non_snake_case)]
pub(super) fn Delay(tau: f64, sampling_period: Option<f64>) -> PyBlock {
    let effective_sp = sampling_period.unwrap_or(0.0);
    PyBlock::wrap(constructors::delay(tau, effective_sp))
}

block!(SampleHold => sample_hold(T = 1.0, tau = 0.0));
block!(ZeroOrderHold => sample_hold(T = 1.0, tau = 0.0));
block!(FirstOrderHold => first_order_hold(T = 1.0, tau = 0.0));
block!(DiscreteDerivative => discrete_derivative(T = 1.0, tau = 0.0));

#[pyfunction]
#[pyo3(signature = (func, T=1.0, tau=0.0))]
#[allow(non_snake_case)]
pub(super) fn Wrapper(func: Py<PyAny>, T: f64, tau: f64) -> PyBlock {
    PyBlock::wrap(constructors::wrapper(move |u: &[f64]| {
        Python::attach(|py| {
            let py_args = pyo3::types::PyTuple::new(py, u.iter().copied()).unwrap();
            match func.call1(py, py_args) {
                Ok(r) => {
                    if let Ok(v) = r.extract::<f64>(py) { return vec![v]; }
                    if let Ok(v) = r.extract::<Vec<f64>>(py) { return v; }
                    vec![0.0]
                }
                Err(e) => on_callback_err(py, e, vec![0.0]),
            }
        })
    }, T, tau))
}

#[pyfunction]
#[pyo3(signature = (coeffs, T=1.0, tau=0.0))]
#[allow(non_snake_case)]
pub(super) fn FIR(coeffs: Vec<f64>, T: f64, tau: f64) -> PyBlock {
    PyBlock::wrap(constructors::fir(coeffs, T, tau))
}

#[pyfunction]
#[pyo3(signature = (T=1.0, tau=0.0, initial_value=None))]
#[allow(non_snake_case)]
pub(super) fn DiscreteIntegrator(
    T: f64, tau: f64, initial_value: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyBlock> {
    let iv = match initial_value {
        None => Vec::new(),
        Some(obj) => {
            if let Ok(v) = obj.extract::<f64>() { vec![v] }
            else if let Ok(v) = obj.extract::<Vec<f64>>() { v }
            else { return Err(PyValueError::new_err("initial_value must be a float or list of floats")); }
        }
    };
    Ok(PyBlock::wrap(constructors::discrete_integrator(T, tau, iv)))
}

/// DiscreteStateSpace: x[k+1]=A·x[k]+B·u[k], y[k]=C·x[k]+D·u[k].
#[pyfunction]
#[pyo3(signature = (A=None, B=None, C=None, D=None, T=1.0, tau=0.0, initial_value=None))]
#[allow(non_snake_case)]
pub(super) fn DiscreteStateSpace(
    A: Option<&Bound<'_, PyAny>>,
    B: Option<&Bound<'_, PyAny>>,
    C: Option<&Bound<'_, PyAny>>,
    D: Option<&Bound<'_, PyAny>>,
    T: f64, tau: f64,
    initial_value: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyBlock> {
    let a_flat = extract_matrix(A, &[0.0])?;
    let b_flat = extract_matrix(B, &[1.0])?;
    let c_flat = extract_matrix(C, &[1.0])?;
    let d_flat = extract_matrix(D, &[0.0])?;

    let n = (a_flat.len() as f64).sqrt() as usize;
    let n = if n * n == a_flat.len() { n } else { 1 };
    let n_in = b_flat.len().checked_div(n).unwrap_or(1);
    let n_out = c_flat.len().checked_div(n).unwrap_or(1);

    let iv = match initial_value {
        Some(v) => {
            if let Ok(f) = v.extract::<f64>() { Some(vec![f]) }
            else {
                v.extract::<Vec<f64>>().ok()
            }
        }
        None => None,
    };

    Ok(PyBlock::wrap(constructors::discrete_state_space(
        a_flat, b_flat, c_flat, d_flat, n, n_in, n_out, T, tau, iv,
    )?))
}

#[pyfunction]
#[pyo3(signature = (Num=None, Den=None, T=1.0, tau=0.0))]
#[allow(non_snake_case)]
pub(super) fn DiscreteTransferFunction(
    Num: Option<Vec<f64>>, Den: Option<Vec<f64>>, T: f64, tau: f64,
) -> PyResult<PyBlock> {
    let num = Num.unwrap_or_else(|| vec![1.0]);
    let den = Den.unwrap_or_else(|| vec![1.0, 0.0]);
    Ok(PyBlock::wrap(constructors::discrete_transfer_function(&num, &den, T, tau)?))
}

#[pyfunction]
#[pyo3(signature = (N=2, T=1.0, tau=0.0))]
#[allow(non_snake_case)]
pub(super) fn TappedDelay(N: usize, T: f64, tau: f64) -> PyResult<PyBlock> {
    Ok(PyBlock::wrap(constructors::tapped_delay(N, T, tau)?))
}

#[pyfunction]
#[pyo3(signature = (n_bits=4, span=None, T=1.0, tau=0.0))]
#[allow(non_snake_case)]
pub(super) fn ADC(n_bits: usize, span: Option<[f64; 2]>, T: f64, tau: f64) -> PyBlock {
    let [lo, hi] = span.unwrap_or([-1.0, 1.0]);
    PyBlock::wrap(constructors::adc(n_bits, lo, hi, T, tau))
}

#[pyfunction]
#[pyo3(signature = (n_bits=4, span=None, T=1.0, tau=0.0))]
#[allow(non_snake_case)]
pub(super) fn DAC(n_bits: usize, span: Option<[f64; 2]>, T: f64, tau: f64) -> PyBlock {
    let [lo, hi] = span.unwrap_or([-1.0, 1.0]);
    PyBlock::wrap(constructors::dac(n_bits, lo, hi, T, tau))
}

#[pyfunction]
#[pyo3(signature = (threshold=0.0, tolerance=None, span=None))]
#[allow(non_snake_case)]
pub(super) fn Comparator(py: Python<'_>, threshold: f64, tolerance: Option<f64>, span: Option<&Bound<'_, PyAny>>) -> PyResult<PyBlock> {
    // `tolerance` is accepted for pathsim API parity but the runtime comparator
    // is a stateless exact select (`u >= threshold`) with no hysteresis band,
    // so a user-supplied tolerance cannot be honored — warn instead of dropping
    // it silently (issue #30).
    if tolerance.is_some() {
        warn_param_ignored(py, "Comparator", "tolerance",
            "the runtime comparator switches exactly at threshold with no hysteresis band")?;
    }
    let s: (f64, f64) = match span {
        None => (-1.0, 1.0),
        Some(obj) => if let Ok(t) = obj.extract::<(f64, f64)>() { t }
            else if let Ok(v) = obj.extract::<Vec<f64>>() {
                if v.len() >= 2 { (v[0], v[1]) } else { return Err(PyValueError::new_err("span must have 2 elements")); }
            } else { return Err(PyValueError::new_err("span must be a tuple or list of 2 floats")); },
    };
    Ok(PyBlock::wrap(constructors::comparator(threshold, s)))
}

#[pyfunction]
#[pyo3(signature = (switch_state=None))]
#[allow(non_snake_case)]
pub(super) fn Switch(switch_state: Option<usize>) -> PyBlock {
    PyBlock::wrap(constructors::switch(2, switch_state))
}

block!(Relay => relay(threshold_up = 1.0, threshold_down = 0.0, value_up = 1.0, value_down = 0.0));
block!(Counter => counter(start = 0.0, threshold = 0.0));
block!(CounterUp => counter_up(start = 0.0, threshold = 0.0));
block!(CounterDown => counter_down(start = 0.0, threshold = 0.0));

#[pyfunction]
#[pyo3(signature = (Kp=0.0, Ki=0.0, Kd=0.0, f_max=100.0, Ks=10.0, limits=(-10.0, 10.0)))]
#[allow(non_snake_case)]
pub(super) fn AntiWindupPID(Kp: f64, Ki: f64, Kd: f64, f_max: f64, Ks: f64, limits: (f64, f64)) -> PyBlock {
    PyBlock::wrap(constructors::anti_windup_pid(Kp, Ki, Kd, f_max, Ks, limits))
}

block!(RateLimiter => rate_limiter(rate = 1.0, f_max = 100.0));
block!(Backlash => backlash(width = 1.0, f_max = 100.0));
block!(Deadband => deadband(lower = -1.0, upper = 1.0));

#[pyfunction]
#[pyo3(signature = (Num=None, Den=None))]
#[allow(non_snake_case)]
pub(super) fn TransferFunctionNumDen(Num: Option<Vec<f64>>, Den: Option<Vec<f64>>) -> PyBlock {
    let n = Num.unwrap_or(vec![1.0]);
    let d = Den.unwrap_or(vec![1.0, 1.0]);
    PyBlock::wrap(constructors::transfer_function_num_den(&n, &d))
}

/// TransferFunctionPRC — pole-residue-constant form.
#[pyfunction]
#[pyo3(signature = (Poles, Residues, Const=0.0))]
#[allow(non_snake_case)]
pub(super) fn TransferFunctionPRC(
    Poles: Vec<Complex64>,
    Residues: Vec<Complex64>,
    Const: f64,
) -> PyResult<PyBlock> {
    if Poles.is_empty() {
        return Err(PyValueError::new_err("TransferFunctionPRC: Poles must be non-empty"));
    }
    if Poles.len() != Residues.len() {
        return Err(PyValueError::new_err(
            "TransferFunctionPRC: Poles and Residues must have equal length"));
    }
    Ok(PyBlock::wrap(constructors::transfer_function_prc(&Poles, &Residues, Const)))
}

/// TransferFunction — alias for TransferFunctionPRC (matches pathsim).
#[pyfunction]
#[pyo3(signature = (Poles, Residues, Const=0.0))]
#[allow(non_snake_case)]
pub(super) fn TransferFunction(
    Poles: Vec<Complex64>,
    Residues: Vec<Complex64>,
    Const: f64,
) -> PyResult<PyBlock> {
    TransferFunctionPRC(Poles, Residues, Const)
}

/// TransferFunctionZPG — zero-pole-gain form.
#[pyfunction]
#[pyo3(signature = (Zeros=None, Poles=None, Gain=1.0))]
#[allow(non_snake_case)]
pub(super) fn TransferFunctionZPG(
    Zeros: Option<Vec<Complex64>>,
    Poles: Option<Vec<Complex64>>,
    Gain: f64,
) -> PyResult<PyBlock> {
    let zeros = Zeros.unwrap_or_default();
    let poles = Poles.unwrap_or_else(|| vec![Complex64::new(-1.0, 0.0)]);
    if poles.is_empty() {
        return Err(PyValueError::new_err("TransferFunctionZPG: Poles must be non-empty"));
    }
    Ok(PyBlock::wrap(constructors::transfer_function_zpg(&zeros, &poles, Gain)))
}

#[pyfunction]
#[pyo3(signature = (Fc=100.0, n=2))]
#[allow(non_snake_case)]
pub(super) fn ButterworthLowpassFilter(Fc: f64, n: usize) -> PyResult<PyBlock> {
    Ok(PyBlock::wrap(constructors::butter_lowpass(Fc, n)?))
}

#[pyfunction]
#[pyo3(signature = (Fc=100.0, n=2))]
#[allow(non_snake_case)]
pub(super) fn ButterworthHighpassFilter(Fc: f64, n: usize) -> PyResult<PyBlock> {
    Ok(PyBlock::wrap(constructors::butter_highpass(Fc, n)?))
}

#[pyfunction]
#[pyo3(signature = (Fc=None, n=2))]
#[allow(non_snake_case)]
pub(super) fn ButterworthBandpassFilter(Fc: Option<Vec<f64>>, n: usize) -> PyResult<PyBlock> {
    let fc = Fc.unwrap_or_else(|| vec![50.0, 100.0]);
    if fc.len() != 2 {
        return Err(PyValueError::new_err(
            "ButterworthBandpassFilter: Fc must be [f_lo, f_hi]"));
    }
    Ok(PyBlock::wrap(constructors::butter_bandpass(fc[0], fc[1], n)?))
}

#[pyfunction]
#[pyo3(signature = (Fc=None, n=2))]
#[allow(non_snake_case)]
pub(super) fn ButterworthBandstopFilter(Fc: Option<Vec<f64>>, n: usize) -> PyResult<PyBlock> {
    let fc = Fc.unwrap_or_else(|| vec![50.0, 100.0]);
    if fc.len() != 2 {
        return Err(PyValueError::new_err(
            "ButterworthBandstopFilter: Fc must be [f_lo, f_hi]"));
    }
    Ok(PyBlock::wrap(constructors::butter_bandstop(fc[0], fc[1], n)?))
}

#[pyfunction]
#[pyo3(signature = (fs=100.0, n=1))]
#[allow(non_snake_case)]
pub(super) fn AllpassFilter(fs: f64, n: usize) -> PyResult<PyBlock> {
    Ok(PyBlock::wrap(constructors::allpass_filter(fs, n)?))
}

#[pyfunction]
#[pyo3(signature = (points, values, fill_value="extrapolate"))]
#[allow(non_snake_case)]
pub(super) fn LUT1D(points: Vec<f64>, values: Vec<f64>, fill_value: &str) -> PyResult<PyBlock> {
    let mode = match fill_value {
        "extrapolate" => constructors::ExtrapMode::Extrapolate,
        "clamp" | "clip" => constructors::ExtrapMode::Clamp,
        other => return Err(PyValueError::new_err(
            format!("LUT1D: fill_value must be 'extrapolate' or 'clamp', got '{}'", other))),
    };
    Ok(PyBlock::wrap(constructors::lut1d(points, values, mode)?))
}

#[pyfunction]
#[allow(non_snake_case)]
pub(super) fn Interface() -> PyBlock { PyBlock::wrap(crate::subsystem::interface()) }

/// Hierarchical subsystem: encapsulates blocks and connections as a single block.
///
/// Convergence of the inner algebraic loop uses the unitless `NLS_COEF`
/// against a WRMS-scaled residual; the legacy `tolerance_fpi` kwarg has
/// been retired (see `Simulation` for migration).  A typo'd kwarg raises
/// `TypeError` here — the Rust boundary is the single validation point, so
/// the generated `subsystem.py` facade's `params.update(kwargs)` can no
/// longer double-swallow an unknown key (issue #31).
#[pyfunction]
#[pyo3(signature = (blocks=vec![], connections=vec![], events=None, iterations_max=200, **kwargs))]
#[allow(non_snake_case)]
pub(super) fn Subsystem(
    py: Python<'_>,
    blocks: Vec<PyBlock>,
    connections: Vec<Bound<'_, PyConnection>>,
    events: Option<Vec<Py<PyAny>>>,
    iterations_max: usize,
    kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<PyBlock> {
    let _ = events;
    if let Some(kw) = kwargs {
        // Validate remaining kwargs against the known-key set: a typo'd knob
        // raises TypeError instead of being silently dropped by `**kwargs`.
        for key in kw.keys() {
            let k: String = key.extract()?;
            if k != "tolerance_fpi" {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "Subsystem() got an unexpected keyword argument '{k}'"
                )));
            }
        }
        if kw.contains("tolerance_fpi")? {
            let warnings = py.import("warnings")?;
            warnings.call_method1(
                "warn",
                (
                    "tolerance_fpi has been retired on Subsystem; the algebraic-\
                     loop solver now uses NLS_COEF against a WRMS-scaled residual.",
                    py.get_type::<pyo3::exceptions::PyDeprecationWarning>(),
                ),
            )?;
        }
    }
    let block_refs: Vec<BlockRef> = blocks.iter().map(|b| b.inner.clone()).collect();
    let conn_refs: Vec<Rc<Connection>> = connections.iter()
        .map(|c| c.borrow().inner.clone()).collect();

    let subsys = crate::subsystem::subsystem(block_refs, conn_refs, iterations_max)?;

    Ok(PyBlock { inner: subsys })
}
