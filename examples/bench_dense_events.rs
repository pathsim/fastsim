//! A/B benchmark: dense-output event localisation vs legacy secant retries.
//!
//! Harmonic oscillator (x'' = -x) with a zero-crossing event on x, adaptive
//! RKDP54. Every crossing forces the legacy path into a revert + re-integrate
//! secant cascade; the dense path root-finds on the step interpolant instead.
//!
//!   cargo run --release --example bench_dense_events [duration]

use std::rc::Rc;
use std::time::Instant;

use fastsim::blocks::constructors::{amplifier, integrator, scope};
use fastsim::connection::Connection;
use fastsim::events::zerocrossing::ZeroCrossing;
use fastsim::simulation::Simulation;
use fastsim::solvers::factories::rkdp54_factory;
use fastsim::utils::fastcell::FastCell;

fn build(dense: bool, omega: f64, crossings: Rc<FastCell<usize>>) -> Simulation {
    let int_v = integrator(0.0);
    let int_x = integrator(1.0);
    let amp = amplifier(-omega * omega);
    let s = scope(None, 0.0, vec![]);

    let conns = vec![
        Connection::single(&int_v, &int_x),
        Connection::single(&int_x, &amp),
        Connection::single(&amp, &int_v),
        Connection::single(&int_x, &s),
    ];

    let x_read = int_x.clone();
    let evt = ZeroCrossing::new(
        move |_t| x_read.borrow().outputs.get_single(0),
        Some(Box::new(move |_t| { *crossings.borrow_mut() += 1; })),
        1e-8,
    );

    let mut sim = Simulation::with_defaults(vec![int_v, int_x, amp, s], conns);
    sim.set_solver(rkdp54_factory(1e-9, 1e-9));
    sim.add_event(Rc::new(FastCell::new(evt)));
    sim.dt = 0.05;
    sim.dense_events = dense;
    sim
}

fn main() {
    let duration: f64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200.0);
    let omega: f64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1.0);

    for (label, dense) in [("dense interpolant", true), ("secant retries  ", false)] {
        let crossings = Rc::new(FastCell::new(0usize));
        let mut sim = build(dense, omega, crossings.clone());
        let t0 = Instant::now();
        let stats = sim.run(duration, true, true);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        println!(
            "{label}: {:5} crossings  {:6} steps ({:5} rejected)  {:7} evals  {:8.1} ms",
            crossings.borrow(),
            stats.total_steps,
            stats.rejected_steps,
            stats.total_evals,
            ms
        );
    }
}
