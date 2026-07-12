// Profiling: Lorenz system — run with `cargo run --release --example profile_lorenz`
// Use with `cargo flamegraph --example profile_lorenz` for flame graphs

use std::rc::Rc;
use std::time::Instant;

use fastsim::blocks::constructors::*;
use fastsim::connection::Connection;
use fastsim::simulation::Simulation;
use fastsim::solvers::factories;
use fastsim::utils::portreference::{PortReference, Port};

fn connect(src: &fastsim::blocks::block::BlockRef, targets: Vec<(&fastsim::blocks::block::BlockRef, Option<usize>)>) -> Rc<Connection> {
    let src_ref = PortReference::new(src.clone(), None);
    let tgt_refs: Vec<PortReference> = targets.iter().map(|(b, port)| {
        match port {
            Some(p) => PortReference::new((*b).clone(), Some(vec![Port::Index(*p)])),
            None => PortReference::new((*b).clone(), None),
        }
    }).collect();
    Rc::new(Connection::new(src_ref, tgt_refs))
}

fn main() {
    let sigma: f64 = 10.0;
    let rho: f64 = 28.0;
    let beta: f64 = 8.0 / 3.0;

    let itg_x = integrator(1.0);
    let itg_y = integrator(1.0);
    let itg_z = integrator(1.0);
    let amp_sigma = amplifier(sigma);
    let add_x = adder(Some("+-"));
    let cns_rho = constant(rho);
    let add_rho_z = adder(Some("+-"));
    let mul_x_rho_z = multiplier();
    let add_y = adder(Some("-+"));
    let mul_xy = multiplier();
    let amp_beta = amplifier(beta);
    let add_z = adder(Some("+-"));
    let sco = scope(None, 0.0, vec![]);

    let blocks = vec![
        itg_x.clone(), itg_y.clone(), itg_z.clone(),
        amp_sigma.clone(), add_x.clone(), cns_rho.clone(),
        add_rho_z.clone(), mul_x_rho_z.clone(), add_y.clone(),
        mul_xy.clone(), amp_beta.clone(), add_z.clone(), sco.clone(),
    ];

    let connections = vec![
        connect(&itg_x, vec![(&add_x, Some(1)), (&mul_x_rho_z, Some(0)), (&mul_xy, Some(0)), (&sco, Some(0))]),
        connect(&itg_y, vec![(&add_x, Some(0)), (&add_y, Some(0)), (&mul_xy, Some(1)), (&sco, Some(1))]),
        connect(&itg_z, vec![(&add_rho_z, Some(1)), (&amp_beta, None), (&sco, Some(2))]),
        connect(&add_x, vec![(&amp_sigma, None)]),
        connect(&amp_sigma, vec![(&itg_x, None)]),
        connect(&cns_rho, vec![(&add_rho_z, Some(0))]),
        connect(&add_rho_z, vec![(&mul_x_rho_z, Some(1))]),
        connect(&mul_x_rho_z, vec![(&add_y, Some(1))]),
        connect(&add_y, vec![(&itg_y, None)]),
        connect(&mul_xy, vec![(&add_z, Some(0))]),
        connect(&amp_beta, vec![(&add_z, Some(1))]),
        connect(&add_z, vec![(&itg_z, None)]),
    ];

    let factory = factories::rkdp54_factory(1e-6, 0.0);
    let mut sim = Simulation::with_solver(blocks, connections, factory, 0.01);

    // Warmup
    sim.run(1.0, true, true);

    // Benchmark: 100 runs of 50s simulation
    let n_runs = 100;
    let t0 = Instant::now();
    for _ in 0..n_runs {
        sim.run(50.0, true, true);
    }
    let elapsed = t0.elapsed();

    let (times, _data) = scope_read(sco.borrow());
    println!("Lorenz 50s × {} runs: {:.1} ms total, {:.2} ms/run, {} time points",
        n_runs, elapsed.as_millis(), elapsed.as_secs_f64() * 1000.0 / n_runs as f64, times.len());
}
