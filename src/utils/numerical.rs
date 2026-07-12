// Numerical Jacobian via central finite differences
// Ported from pathsim/optim/numerical.py num_jac()

use crate::solvers::solver::Jacobian;
use crate::constants::{NUM_JAC_REL, NUM_JAC_TOL};

/// Compute numerical Jacobian df/dx via central finite differences.
///
/// Uses adaptive step size h = max(|r * x_j|, tol) per dimension,
/// and central differences: J_ij = (f_i(x+h_j) - f_i(x-h_j)) / (2*h_j)
///
/// `f` writes into the provided `&mut Vec<f64>` buffer (caller-cleared).
///
/// For n=1: returns Jacobian::Scalar
/// For n>1: returns Jacobian::Matrix (flat row-major n×n)
pub fn num_jac(
    f: &dyn Fn(&[f64], &[f64], f64, &mut Vec<f64>),
    x: &[f64],
    u: &[f64],
    t: f64,
) -> Jacobian {
    let n = x.len();
    let mut f_p: Vec<f64> = Vec::with_capacity(n);
    let mut f_m: Vec<f64> = Vec::with_capacity(n);

    if n == 1 {
        let h = (NUM_JAC_REL * x[0].abs()).max(NUM_JAC_TOL);
        let xp = [x[0] + h];
        let xm = [x[0] - h];
        f_p.clear(); f(&xp, u, t, &mut f_p);
        f_m.clear(); f(&xm, u, t, &mut f_m);
        Jacobian::Scalar(0.5 * (f_p[0] - f_m[0]) / h)
    } else {
        let mut jac = vec![0.0; n * n];
        let mut x_buf = x.to_vec();
        for j in 0..n {
            let h = (NUM_JAC_REL * x[j].abs()).max(NUM_JAC_TOL);
            x_buf[j] = x[j] + h;
            f_p.clear(); f(&x_buf, u, t, &mut f_p);
            x_buf[j] = x[j] - h;
            f_m.clear(); f(&x_buf, u, t, &mut f_m);
            x_buf[j] = x[j]; // restore
            let inv_2h = 0.5 / h;
            for i in 0..n {
                jac[i * n + j] = (f_p[i] - f_m[i]) * inv_2h;
            }
        }
        Jacobian::Matrix(jac, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_jac_scalar() {
        // f(x) = -2*x, J = -2
        let j = num_jac(&|x, _u, _t, out| out.push(-2.0 * x[0]), &[3.0], &[], 0.0);
        match j {
            Jacobian::Scalar(v) => assert!((v + 2.0).abs() < 1e-5),
            _ => panic!("Expected scalar"),
        }
    }

    #[test]
    fn test_num_jac_matrix() {
        // f(x) = [-x[0] + x[1], x[0] - 2*x[1]]
        // J = [[-1, 1], [1, -2]]
        let j = num_jac(
            &|x, _u, _t, out| { out.push(-x[0] + x[1]); out.push(x[0] - 2.0 * x[1]); },
            &[1.0, 1.0], &[], 0.0,
        );
        match j {
            Jacobian::Matrix(m, n) => {
                assert_eq!(n, 2);
                assert!((m[0] + 1.0).abs() < 1e-5); // J[0,0] = -1
                assert!((m[1] - 1.0).abs() < 1e-5); // J[0,1] = 1
                assert!((m[2] - 1.0).abs() < 1e-5); // J[1,0] = 1
                assert!((m[3] + 2.0).abs() < 1e-5); // J[1,1] = -2
            }
            _ => panic!("Expected matrix"),
        }
    }
}
