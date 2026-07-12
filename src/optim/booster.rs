// ConnectionBooster: wraps Connection with Anderson acceleration
// Ported 1:1 from pathsim/optim/booster.py

use std::rc::Rc;

use crate::connection::Connection;
use crate::optim::anderson::Anderson;

/// Wraps a `Connection` instance and injects a fixed-point accelerator.
///
/// This class is part of the solver structure and intended to improve the
/// algebraic loop solver of the simulation.
///
/// Mirrors Python's ConnectionBooster exactly:
/// - Holds a direct reference to the Connection
/// - `get()` reads source outputs via `connection.source.get_outputs()`
/// - `set()` writes to all targets via `trg.set_inputs()`
/// - `update()` is all-in-one: get → accelerate → set → return residual
pub struct ConnectionBooster {
    /// Connection instance being boosted
    pub connection: Rc<Connection>,
    /// Previous evaluation of the connection value (stack-allocated)
    pub history: Vec<f64>,
    /// Scratch buffer for current values (reused)
    _current: Vec<f64>,
    /// Internal fixed-point accelerator
    pub accelerator: Anderson,
    /// Absolute tolerance for the WRMS scale `(atol + rtol·|signal|)`.
    /// Must match the simulation-level outer step tolerance so the algebraic
    /// loop converges in step with the time integrator.
    atol: f64,
    /// Relative tolerance for the WRMS scale.
    rtol: f64,
}

impl ConnectionBooster {
    /// Create a new ConnectionBooster.
    ///
    /// Mirrors Python: `ConnectionBooster(connection)`
    /// Initializes history by calling `get()` on the source.
    /// `atol`/`rtol` are the WRMS-norm weights for the algebraic-loop
    /// convergence test — typically copied from the simulation's
    /// `tolerance_lte_abs/rel`.
    pub fn new(connection: Rc<Connection>, atol: f64, rtol: f64) -> Self {
        let history = connection.source.get_outputs();
        let current = history.clone();
        Self {
            connection,
            history,
            _current: current,
            accelerator: Anderson::with_defaults(),
            atol,
            rtol,
        }
    }

    /// Return the output values of the source block referenced in the connection.
    ///
    /// Mirrors Python `get()`.
    pub fn get(&self) -> Vec<f64> {
        self.connection.source.get_outputs()
    }

    /// Set targets input values.
    ///
    /// Mirrors Python `set(val)`.
    pub fn set(&self, val: &[f64]) {
        for trg in &self.connection.targets {
            trg.set_inputs(val);
        }
    }

    /// Reset the internal fixed-point accelerator and update the history
    /// to the most recent value.
    ///
    /// Mirrors Python `reset()`.
    pub fn reset(&mut self) {
        self.accelerator.reset();
        self.connection.source.get_outputs_into(&mut self.history);
    }

    /// Wraps the `Connection.update` method for data transfer from source
    /// to targets and injects a solver step of the fixed-point accelerator.
    /// Updates the history and returns the **WRMS-scaled** fixed-point
    /// residual `||(current − history) / (atol + rtol·|history|)||_RMS`.
    ///
    /// The downstream `_loop_tracker` checks this against the unitless
    /// `NLS_COEF` (matching the implicit-stage convergence test) so the
    /// algebraic-loop tolerance scales with the signal magnitudes — small
    /// loop-closing signals don't get over-converged just because they
    /// happen to live near zero.
    ///
    /// Allocation-free hot-path: scratch buffers are reused across calls.
    pub fn update(&mut self) -> f64 {
        // Read current values directly into the reusable scratch buffer
        // (no per-iteration Vec allocation).
        self.connection.source.get_outputs_into(&mut self._current);

        // Anderson step (in-place on history buffer); resize if signal
        // dimension grew/shrank between calls.
        if self.history.len() != self._current.len() {
            self.history.resize(self._current.len(), 0.0);
        }

        // WRMS norm of (current − history) BEFORE the accelerator overwrites
        // history.  Scale is component-wise `(atol + rtol·|history|)`.
        let n = self.history.len();
        let res_wrms = if n == 0 {
            0.0
        } else {
            let mut sum_sq = 0.0;
            for i in 0..n {
                let scale = crate::solvers::solver::wrms_scale(self.atol, self.rtol, self.history[i]);
                let s = (self._current[i] - self.history[i]) / scale;
                sum_sq += s * s;
            }
            (sum_sq / n as f64).sqrt()
        };

        // Anderson step — discards its own L2 return; the WRMS norm above
        // is what the convergence tracker consumes.
        let _l2 = self.accelerator.step(&mut self.history, &self._current);

        // Set accelerated values to targets
        self.set(&self.history);

        res_wrms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::block::new_block_ref;
    use crate::connection::Connection;
    use crate::utils::portreference::PortReference;

    #[test]
    fn test_booster_creation() {
        let b1 = new_block_ref(None, None);
        let b2 = new_block_ref(None, None);

        b1.borrow_mut().outputs.set_single(0, 5.0);

        let conn = Rc::new(Connection::new(
            PortReference::new(b1.clone(), None),
            vec![PortReference::new(b2.clone(), None)],
        ));

        let booster = ConnectionBooster::new(conn, 1e-8, 1e-5);
        // History initialized from source outputs
        assert_eq!(booster.history, vec![5.0]);
    }

    #[test]
    fn test_booster_get_set() {
        let b1 = new_block_ref(None, None);
        let b2 = new_block_ref(None, None);

        b1.borrow_mut().outputs.set_single(0, 10.0);

        let conn = Rc::new(Connection::new(
            PortReference::new(b1.clone(), None),
            vec![PortReference::new(b2.clone(), None)],
        ));

        let booster = ConnectionBooster::new(conn, 1e-8, 1e-5);

        // get() reads source outputs
        assert_eq!(booster.get(), vec![10.0]);

        // set() writes to targets
        booster.set(&[42.0]);
        assert_eq!(b2.borrow().inputs.get_single(0), 42.0);
    }

    #[test]
    fn test_booster_update() {
        let b1 = new_block_ref(None, None);
        let b2 = new_block_ref(None, None);

        b1.borrow_mut().outputs.set_single(0, 1.0);

        let conn = Rc::new(Connection::new(
            PortReference::new(b1.clone(), None),
            vec![PortReference::new(b2.clone(), None)],
        ));

        let mut booster = ConnectionBooster::new(conn, 1e-8, 1e-5);

        // First update
        b1.borrow_mut().outputs.set_single(0, 2.0);
        let res = booster.update();
        assert!(res >= 0.0);

        // Target should have received accelerated value
        let target_val = b2.borrow().inputs.get_single(0);
        assert!(target_val != 0.0);
    }

    #[test]
    fn test_booster_reset() {
        let b1 = new_block_ref(None, None);
        let b2 = new_block_ref(None, None);

        b1.borrow_mut().outputs.set_single(0, 1.0);

        let conn = Rc::new(Connection::new(
            PortReference::new(b1.clone(), None),
            vec![PortReference::new(b2.clone(), None)],
        ));

        let mut booster = ConnectionBooster::new(conn, 1e-8, 1e-5);

        // Do some updates
        b1.borrow_mut().outputs.set_single(0, 5.0);
        booster.update();

        // Reset
        b1.borrow_mut().outputs.set_single(0, 3.0);
        booster.reset();
        assert_eq!(booster.history, vec![3.0]);
    }
}
