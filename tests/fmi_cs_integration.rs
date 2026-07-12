// End-to-end test of FMI 3.0 Co-Simulation import. We exercise `fmi3DoStep`
// against the Dahlquist reference FMU (fixedInternalStepSize=0.1).

use fastsim::fmi::instance::{Cs, Instance};
use fastsim::fmi::model_description::ModelDescription;
use fastsim::fmi::unzip::FmuArchive;

const DAHLQUIST_FMU: &str = "tests/fixtures/fmi/Dahlquist.fmu";

#[test]
fn dahlquist_cs_do_step() {
    let archive = FmuArchive::extract(DAHLQUIST_FMU).expect("extract");
    let md = ModelDescription::from_file(archive.model_description()).expect("parse xml");

    let inst = Instance::<Cs>::new_co_simulation(archive, &md, "dq_cs", false, false, false)
        .expect("instantiate");

    inst.enter_initialization_mode(Some(1e-6), 0.0, Some(1.0))
        .expect("enter init");
    inst.exit_initialization_mode().expect("exit init");

    // For CS without event mode, we step directly from init → step mode is
    // entered automatically by the FMU on exit init (per FMI 3.0 §4.2.5).

    let x_vr = md.variable_by_name("x").expect("x").value_reference;
    let mut x_out = [0.0_f64];

    let dt = 0.01_f64;
    let n_steps = 100;
    let mut t = 0.0;
    for _ in 0..n_steps {
        let r = inst.do_step(t, dt).expect("do_step");
        assert!(!r.terminate_simulation);
        t += dt;
    }

    inst.get_float64(&[x_vr], &mut x_out).expect("get x");
    // Dahlquist CS advances with forward Euler at fixedInternalStepSize=0.1,
    // so over t=1.0 we expect (1-0.1)^10 = 0.3486784... (not the analytical
    // exp(-1)). We accept the Euler result as the FMU's own ground truth.
    let euler_expected = 0.9_f64.powi(10);
    assert!(
        (x_out[0] - euler_expected).abs() < 1e-9,
        "got {} expected {}",
        x_out[0],
        euler_expected
    );

    inst.terminate().expect("terminate");
}
