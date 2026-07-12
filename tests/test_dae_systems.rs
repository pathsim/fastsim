// End-to-end tests for DAE blocks — exercises the MassMatrixStageBuilder
// through the full Block ← Solver ← Simulation stack.

use fastsim::blocks::constructors::{
    fully_implicit_dae, mass_matrix_dae, semi_explicit_dae,
};
use fastsim::simulation::Simulation;
use fastsim::solvers::factories::{dirk3_factory, esdirk43_factory, rkdp54_factory};
use fastsim::solvers::stage::Mass;

#[test]
fn mass_matrix_identity_recovers_exp_decay() {
    // With M = I, the DAE `M·ẋ = -x` is just the plain ODE `ẋ = -x`.
    // Exact solution: x(t) = x(0) · exp(-t).  Integrating to t=1 with
    // ESDIRK43 at tight tolerance should match to several digits.
    let blk = mass_matrix_dae(
        |x, _u, _t| vec![-x[0]],
        Mass::identity(1),
        &[1.0],
        None,
    ).unwrap();

    let mut sim = Simulation::with_defaults(vec![blk.clone()], Vec::new());
    sim.set_solver(esdirk43_factory(1e-10, 1e-10));
    sim.run(1.0, false, true);

    let x_final = blk.borrow().engine.as_ref().unwrap().get()[0];
    let expected = (-1.0_f64).exp();
    assert!((x_final - expected).abs() < 1e-5,
        "identity mass matrix: got {}, expected {} (exp(-1))", x_final, expected);
}

#[test]
fn mass_matrix_diagonal_scales_dynamics() {
    // Diagonal M = diag(2) with `M·ẋ = -x` means `ẋ = -x/2`.
    // Exact solution at t=1:  x(1) = x(0) · exp(-0.5).
    let data = vec![2.0];  // 1×1 diagonal
    let m = Mass::from_flat(data, 1);
    let blk = mass_matrix_dae(
        |x, _u, _t| vec![-x[0]],
        m,
        &[1.0],
        None,
    ).unwrap();

    let mut sim = Simulation::with_defaults(vec![blk.clone()], Vec::new());
    sim.set_solver(esdirk43_factory(1e-10, 1e-10));
    sim.run(1.0, false, true);

    let x_final = blk.borrow().engine.as_ref().unwrap().get()[0];
    let expected = (-0.5_f64).exp();
    assert!((x_final - expected).abs() < 1e-5,
        "diagonal mass matrix: got {}, expected {} (exp(-0.5))", x_final, expected);
}

#[test]
fn mass_matrix_singular_holds_algebraic_constraint() {
    // Index-1 DAE:  ẋ = -x + z,  0 = x - z.
    // State = [x, z], M = diag(1, 0).  Algebraic row forces z = x, so the
    // reduced ODE is ẋ = 0 → trajectory is stationary at the initial point.
    let m_data = vec![1.0, 0.0,
                       0.0, 0.0];
    let m = Mass::from_flat(m_data, 2);
    assert!(m.is_singular);
    assert_eq!(m.n_alg, 1);

    let blk = mass_matrix_dae(
        |x, _u, _t| vec![-x[0] + x[1], x[0] - x[1]],
        m,
        &[1.0, 1.0],  // consistent init: x = z
        None,
    ).unwrap();

    let mut sim = Simulation::with_defaults(vec![blk.clone()], Vec::new());
    sim.set_solver(esdirk43_factory(1e-8, 1e-8));
    sim.run(1.0, false, true);

    let x_final = blk.borrow().engine.as_ref().unwrap().get();
    // Both components should stay pinned to their initial value.
    assert!((x_final[0] - 1.0).abs() < 1e-4, "x drifted: {}", x_final[0]);
    assert!((x_final[1] - 1.0).abs() < 1e-4, "z drifted: {}", x_final[1]);
    // Constraint x = z must hold.
    assert!((x_final[0] - x_final[1]).abs() < 1e-4,
        "constraint broken: x={}, z={}", x_final[0], x_final[1]);
}

#[test]
fn fully_implicit_exp_decay() {
    // F(x, ẋ, u, t) = ẋ + x = 0  →  ẋ = -x  →  x(t) = x(0)·exp(-t).
    let blk = fully_implicit_dae(
        |x, xdot, _u, _t| vec![xdot[0] + x[0]],
        &[1.0],
        None, None,
    ).unwrap();

    let mut sim = Simulation::with_defaults(vec![blk.clone()], Vec::new());
    sim.set_solver(esdirk43_factory(1e-10, 1e-10));
    sim.run(1.0, false, true);

    let x_final = blk.borrow().engine.as_ref().unwrap().get()[0];
    let expected = (-1.0_f64).exp();
    assert!((x_final - expected).abs() < 1e-4,
        "FI exp decay: got {}, expected {} (exp(-1))", x_final, expected);
}

#[test]
fn fully_implicit_nonlinear_scaling() {
    // F(x, ẋ, u, t) = 2·ẋ + x·(1 + x²) = 0  (x scalar, well-defined for small |x|).
    // For small x, behaves like ẋ ≈ -x/2, so x(t) ≈ x0·exp(-t/2).
    // With x0 = 0.01 (small), comparison against exp(-t/2) should be close.
    let blk = fully_implicit_dae(
        |x, xdot, _u, _t| vec![2.0 * xdot[0] + x[0] * (1.0 + x[0].powi(2))],
        &[0.01],
        None, None,
    ).unwrap();

    let mut sim = Simulation::with_defaults(vec![blk.clone()], Vec::new());
    sim.set_solver(esdirk43_factory(1e-10, 1e-10));
    sim.run(1.0, false, true);

    let x_final = blk.borrow().engine.as_ref().unwrap().get()[0];
    let linear_approx = 0.01 * (-0.5_f64).exp();
    // Nonlinearity contributes O(x³), so error is within 1e-5 for x0=0.01.
    assert!((x_final - linear_approx).abs() < 1e-5,
        "FI nonlinear: got {}, linear approx {}", x_final, linear_approx);
}

#[test]
fn fully_implicit_index1_dae() {
    // 2-state fully-implicit form of the same index-1 DAE as earlier tests:
    //   F1 = ẋ - (-x + z) = ẋ + x - z = 0
    //   F2 = x - z = 0   (algebraic, no ẋ-dependence)
    // Jacobians: ∂F/∂x = [[1, -1], [1, -1]], ∂F/∂ẋ = [[1, 0], [0, 0]].
    // Initial (1, 1) is consistent.  Trajectory stays pinned at (1, 1).
    let blk = fully_implicit_dae(
        |x, xdot, _u, _t| vec![xdot[0] + x[0] - x[1], x[0] - x[1]],
        &[1.0, 1.0],
        Some(Box::new(|_x, _xd, _u, _t| vec![1.0, -1.0, 1.0, -1.0])),
        Some(Box::new(|_x, _xd, _u, _t| vec![1.0, 0.0, 0.0, 0.0])),
    ).unwrap();

    // DIRK3 has no explicit first stage, so every stage passes through the
    // FullyImplicitStageBuilder — the safest choice for singular `∂F/∂ẋ`.
    let mut sim = Simulation::with_defaults(vec![blk.clone()], Vec::new());
    sim.set_solver(dirk3_factory());
    sim.run(1.0, false, true);

    let state = blk.borrow().engine.as_ref().unwrap().get().to_vec();
    assert!((state[0] - 1.0).abs() < 1e-3, "x drifted: {}", state[0]);
    assert!((state[1] - 1.0).abs() < 1e-3, "z drifted: {}", state[1]);
    assert!((state[0] - state[1]).abs() < 1e-3,
        "constraint broken: {} vs {}", state[0], state[1]);
}

// =========================================================================
// SemiExplicitDAE — reduced form, solvable with any solver.
// =========================================================================

#[test]
fn semi_explicit_dae_pinned_constraint_explicit_solver() {
    // ẋ = -x + z,  0 = x - z  →  z = x  →  ẋ = 0  →  stationary at 1.
    let blk = semi_explicit_dae(
        |x, z, _u, _t| vec![-x[0] + z[0]],
        |x, z, _u, _t| vec![x[0] - z[0]],
        &[1.0], &[1.0],
        None,
    ).unwrap();

    let mut sim = Simulation::with_defaults(vec![blk.clone()], Vec::new());
    sim.set_solver(rkdp54_factory(1e-8, 1e-8));  // EXPLICIT solver.
    sim.run(1.0, false, true);

    let x_final = blk.borrow().engine.as_ref().unwrap().get()[0];
    assert!((x_final - 1.0).abs() < 1e-4,
        "x drifted to {}", x_final);
    // z should equal x in the converged output — check the second output port.
    let z_final = blk.borrow().outputs.get_single(1);
    assert!((z_final - 1.0).abs() < 1e-4, "z drifted to {}", z_final);
}

#[test]
fn semi_explicit_dae_driven_constraint_explicit_solver() {
    // ẋ = z,  0 = z - sin(t)  →  z(t) = sin(t),  x(t) = x0 + 1 - cos(t).
    let blk = semi_explicit_dae(
        |_x, z, _u, _t| vec![z[0]],
        |_x, z, _u, t| vec![z[0] - t.sin()],
        &[0.0], &[0.0],
        Some(Box::new(|_x, _z, _u, _t| vec![1.0])),   // ∂g/∂z = 1
    ).unwrap();

    let mut sim = Simulation::with_defaults(vec![blk.clone()], Vec::new());
    sim.set_solver(rkdp54_factory(1e-10, 1e-10));
    let t_end = std::f64::consts::PI;
    sim.run(t_end, false, true);

    let x_final = blk.borrow().engine.as_ref().unwrap().get()[0];
    let expected = 0.0 + 1.0 - t_end.cos();  // = 2
    assert!((x_final - expected).abs() < 1e-5,
        "got {}, expected {}", x_final, expected);
}

#[test]
fn semi_explicit_dae_works_with_implicit_solver_too() {
    // Sanity: the same block must also run with an implicit solver.
    let blk = semi_explicit_dae(
        |x, z, _u, _t| vec![-x[0] + z[0]],
        |x, z, _u, _t| vec![x[0] - z[0]],
        &[1.0], &[1.0],
        None,
    ).unwrap();
    let mut sim = Simulation::with_defaults(vec![blk.clone()], Vec::new());
    sim.set_solver(esdirk43_factory(1e-8, 1e-8));
    sim.run(1.0, false, true);
    let x_final = blk.borrow().engine.as_ref().unwrap().get()[0];
    assert!((x_final - 1.0).abs() < 1e-4);
}
