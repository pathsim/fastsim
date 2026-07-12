// End-to-end test of FMI 3.0 Model-Exchange import against the Modelica
// Reference-FMUs. Uses the Dahlquist model (dx/dt = -k*x) because it has one
// state, no events, and a trivially verifiable analytical solution.

use fastsim::fmi::instance::{Instance, Me};
use fastsim::fmi::model_description::ModelDescription;
use fastsim::fmi::unzip::FmuArchive;

const DAHLQUIST_FMU: &str = "tests/fixtures/fmi/Dahlquist.fmu";

#[test]
fn dahlquist_me_forward_euler() {
    let archive = FmuArchive::extract(DAHLQUIST_FMU).expect("extract");
    let md = ModelDescription::from_file(archive.model_description()).expect("parse xml");

    assert_eq!(md.model_name, "Dahlquist");
    assert_eq!(md.n_continuous_states(), 1);

    let inst = Instance::<Me>::new_model_exchange(archive, &md, "dq", false)
        .expect("instantiate");

    // Initialize — start values from modelDescription (x=1, k=1 via initial=exact).
    inst.enter_initialization_mode(Some(1e-6), 0.0, Some(10.0))
        .expect("enter init");
    inst.exit_initialization_mode().expect("exit init");

    // Drain initial discrete-state updates.
    loop {
        let u = inst.update_discrete_states().expect("update discrete");
        assert!(!u.terminate_simulation);
        if !u.discrete_states_need_update {
            break;
        }
    }
    inst.enter_continuous_time_mode().expect("enter cont");

    // Fixed-step forward Euler: dx/dt = -k*x, k=1, x0=1 → x(t)=exp(-t).
    let x_vr = md.variable_by_name("x").expect("x").value_reference;
    let mut x = [1.0_f64; 1];
    let mut dx = [0.0_f64; 1];
    let dt = 1e-3_f64;
    let steps = 1000;
    let mut t = 0.0;
    for _ in 0..steps {
        inst.set_time(t).expect("set time");
        inst.set_continuous_states(&x).expect("set states");
        inst.get_continuous_state_derivatives(&mut dx)
            .expect("get deriv");
        x[0] += dt * dx[0];
        t += dt;
        // CompletedIntegratorStep — Dahlquist declares needsCompletedIntegratorStep
        // by default, so call it after each step.
        let cs = inst
            .completed_integrator_step(true)
            .expect("completed step");
        assert!(!cs.terminate_simulation);
    }

    // Verify via fmi3GetFloat64 that FMU state matches our buffer after final set.
    inst.set_time(t).expect("set time");
    inst.set_continuous_states(&x).expect("set states");
    let mut got = [0.0_f64];
    inst.get_float64(&[x_vr], &mut got).expect("get float");
    assert!((got[0] - x[0]).abs() < 1e-12);

    // x(1.0) ≈ exp(-1) ≈ 0.3679; forward Euler at dt=1e-3 is within ~0.02%.
    let expected = (-1.0_f64).exp();
    assert!(
        (x[0] - expected).abs() < 5e-4,
        "got {} expected {}",
        x[0],
        expected
    );

    inst.terminate().expect("terminate");
}
