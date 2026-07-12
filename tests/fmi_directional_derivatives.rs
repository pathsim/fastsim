// End-to-end test of FMI 3.0 `fmi3GetDirectionalDerivative`.
//
// Uses the VanDerPol reference FMU which advertises
// `providesDirectionalDerivatives="true"` and has 2 states (x0, x1).  The
// Jacobian ∂ẋ/∂x is known analytically for mu=1, x=[2, 0]:
//
//   ẋ0 = x1                            ∂ẋ0/∂x0 = 0
//   ẋ1 = mu (1 - x0²) x1 − x0          ∂ẋ0/∂x1 = 1
//                                      ∂ẋ1/∂x0 = -2·mu·x0·x1 − 1 = -1
//                                      ∂ẋ1/∂x1 = mu (1 - x0²) = -3

use fastsim::fmi::instance::{Instance, Me};
use fastsim::fmi::model_description::ModelDescription;
use fastsim::fmi::unzip::FmuArchive;

const VDP_FMU: &str = "tests/fixtures/fmi/VanDerPol.fmu";

#[test]
fn vanderpol_directional_derivatives_match_analytical_jacobian() {
    let archive = FmuArchive::extract(VDP_FMU).expect("extract");
    let md = ModelDescription::from_file(archive.model_description()).expect("parse xml");

    assert!(md.model_exchange.as_ref().unwrap().provides_directional_derivatives);
    assert_eq!(md.n_continuous_states(), 2);

    let inst = Instance::<Me>::new_model_exchange(archive, &md, "vdp", false)
        .expect("instantiate");

    assert!(inst.supports_directional_derivatives(),
        "FMU advertises providesDirectionalDerivatives but symbol is absent");

    inst.enter_initialization_mode(Some(1e-6), 0.0, None).expect("enter init");
    inst.exit_initialization_mode().expect("exit init");

    // Drain discrete-state updates.
    loop {
        let u = inst.update_discrete_states().expect("update discrete");
        if !u.discrete_states_need_update { break; }
    }
    inst.enter_continuous_time_mode().expect("enter cont");

    // State VRs in the order the ContinuousStateDerivative structure lists them.
    let state_vrs: Vec<_> = md.continuous_states().iter().map(|v| v.value_reference).collect();
    let state_deriv_vrs = md.model_structure.continuous_state_derivatives.clone();
    assert_eq!(state_vrs.len(), 2);
    assert_eq!(state_deriv_vrs.len(), 2);

    // Pin state at [2, 0] (FMU default start values — no set needed, but do it
    // explicitly so the test is independent of defaults).
    inst.set_time(0.0).expect("set time");
    inst.set_continuous_states(&[2.0, 0.0]).expect("set states");

    // Column 0: seed = [1, 0] → sensitivity = column 0 of J = [∂ẋ0/∂x0, ∂ẋ1/∂x0]
    let mut col0 = [0.0f64; 2];
    inst.get_directional_derivative(&state_deriv_vrs, &state_vrs, &[1.0, 0.0], &mut col0)
        .expect("directional deriv col 0");
    // Column 1: seed = [0, 1]
    let mut col1 = [0.0f64; 2];
    inst.get_directional_derivative(&state_deriv_vrs, &state_vrs, &[0.0, 1.0], &mut col1)
        .expect("directional deriv col 1");

    // Analytical column 0 at x=[2,0], mu=1:  [0, -1]
    assert!((col0[0] - 0.0).abs() < 1e-10, "∂ẋ0/∂x0 expected 0, got {}", col0[0]);
    assert!((col0[1] - -1.0).abs() < 1e-10, "∂ẋ1/∂x0 expected -1, got {}", col0[1]);
    // Analytical column 1 at x=[2,0], mu=1:  [1, -3]
    assert!((col1[0] - 1.0).abs() < 1e-10, "∂ẋ0/∂x1 expected 1, got {}", col1[0]);
    assert!((col1[1] - -3.0).abs() < 1e-10, "∂ẋ1/∂x1 expected -3, got {}", col1[1]);

    inst.terminate().expect("terminate");
}
