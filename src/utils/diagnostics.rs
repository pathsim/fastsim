// Convergence tracking and diagnostics
// Ported 1:1 from pathsim/utils/diagnostics.py

use std::collections::HashMap;

/// Tracks per-object scalar errors and convergence for fixed-point loops.
///
/// Used by algebraic loop solver (keyed by booster index) and
/// implicit ODE solver (keyed by block index).
pub struct ConvergenceTracker {
    pub errors: HashMap<usize, f64>,
    pub max_error: f64,
    pub iterations: usize,
}

impl Default for ConvergenceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvergenceTracker {
    pub fn new() -> Self {
        Self { errors: HashMap::new(), max_error: 0.0, iterations: 0 }
    }

    pub fn reset(&mut self) {
        self.errors.clear();
        self.max_error = 0.0;
        self.iterations = 0;
    }

    pub fn begin_iteration(&mut self) {
        self.errors.clear();
        self.max_error = 0.0;
    }

    pub fn record(&mut self, obj_id: usize, error: f64) {
        self.errors.insert(obj_id, error);
        if error > self.max_error {
            self.max_error = error;
        }
    }

    pub fn converged(&self, tolerance: f64) -> bool {
        self.max_error <= tolerance
    }

    /// Format per-object error breakdown for error messages.
    /// `label_fn` maps object IDs to human-readable labels.
    pub fn details(&self, label_fn: &dyn Fn(usize) -> String) -> Vec<String> {
        self.errors.iter()
            .map(|(&obj, &err)| format!("  {}: {:.2e}", label_fn(obj), err))
            .collect()
    }
}

/// Tracks per-block adaptive step results.
pub struct StepTracker {
    pub errors: HashMap<usize, (bool, f64, Option<f64>)>,
    pub success: bool,
    pub max_error: f64,
    pub min_scale: Option<f64>,
}

impl Default for StepTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl StepTracker {
    pub fn new() -> Self {
        Self { errors: HashMap::new(), success: true, max_error: 0.0, min_scale: None }
    }

    pub fn reset(&mut self) {
        self.errors.clear();
        self.success = true;
        self.max_error = 0.0;
        self.min_scale = None;
    }

    pub fn record(&mut self, block_id: usize, success: bool, err_norm: f64, scale: Option<f64>) {
        self.errors.insert(block_id, (success, err_norm, scale));
        if !success { self.success = false; }
        if err_norm > self.max_error { self.max_error = err_norm; }
        if let Some(s) = scale {
            self.min_scale = Some(self.min_scale.map_or(s, |m: f64| m.min(s)));
        }
    }

    pub fn scale(&self) -> f64 {
        self.min_scale.unwrap_or(1.0)
    }
}

/// Per-timestep diagnostics snapshot.
#[derive(Clone)]
pub struct Diagnostics {
    pub time: f64,
    pub loop_residuals: HashMap<usize, f64>,
    pub loop_iterations: usize,
    pub solve_residuals: HashMap<usize, f64>,
    pub solve_iterations: usize,
    pub step_errors: HashMap<usize, (bool, f64, Option<f64>)>,
}

impl Default for Diagnostics {
    fn default() -> Self {
        Self::new()
    }
}

impl Diagnostics {
    pub fn new() -> Self {
        Self {
            time: 0.0,
            loop_residuals: HashMap::new(),
            loop_iterations: 0,
            solve_residuals: HashMap::new(),
            solve_iterations: 0,
            step_errors: HashMap::new(),
        }
    }

    pub fn from_trackers(
        time: f64,
        loop_tracker: &ConvergenceTracker,
        solve_tracker: &ConvergenceTracker,
        step_tracker: &StepTracker,
    ) -> Self {
        Self {
            time,
            loop_residuals: loop_tracker.errors.clone(),
            loop_iterations: loop_tracker.iterations,
            solve_residuals: solve_tracker.errors.clone(),
            solve_iterations: solve_tracker.iterations,
            step_errors: step_tracker.errors.clone(),
        }
    }

    /// Identify the block with the highest residual.
    pub fn worst_block(&self) -> Option<(usize, f64)> {
        let mut worst: Option<(usize, f64)> = None;
        for (&obj, &err) in &self.solve_residuals {
            if worst.is_none_or(|(_, e)| err > e) { worst = Some((obj, err)); }
        }
        for (&obj, &(_, err_norm, _)) in &self.step_errors {
            if worst.is_none_or(|(_, e)| err_norm > e) { worst = Some((obj, err_norm)); }
        }
        worst
    }

    /// Identify the booster with the highest algebraic loop residual.
    pub fn worst_booster(&self) -> Option<(usize, f64)> {
        self.loop_residuals.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(&id, &err)| (id, err))
    }

    /// Formatted summary of this diagnostics snapshot.
    pub fn summary(&self, label_fn: &dyn Fn(usize) -> String) -> String {
        let mut lines = vec![format!("Diagnostics at t = {:.6}", self.time)];
        if !self.step_errors.is_empty() {
            lines.push("  Adaptive step errors:".to_string());
            for (&obj, &(suc, err, scl)) in &self.step_errors {
                let status = if suc { "OK" } else { "FAIL" };
                let scl_str = scl.map_or("N/A".to_string(), |s| format!("{:.3}", s));
                lines.push(format!("    {} {}: err={:.2e}, scale={}", status, label_fn(obj), err, scl_str));
            }
        }
        if !self.solve_residuals.is_empty() {
            lines.push(format!("  Implicit solver residuals ({} iterations):", self.solve_iterations));
            for (&obj, &err) in &self.solve_residuals {
                lines.push(format!("    {}: {:.2e}", label_fn(obj), err));
            }
        }
        if !self.loop_residuals.is_empty() {
            lines.push(format!("  Algebraic loop residuals ({} iterations):", self.loop_iterations));
            for (&obj, &err) in &self.loop_residuals {
                lines.push(format!("    {}: {:.2e}", label_fn(obj), err));
            }
        }
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_tracker() {
        let mut ct = ConvergenceTracker::new();
        ct.begin_iteration();
        ct.record(0, 1e-5);
        ct.record(1, 1e-3);
        assert_eq!(ct.max_error, 1e-3);
        assert!(!ct.converged(1e-4));
        assert!(ct.converged(1e-2));
    }

    #[test]
    fn test_convergence_details() {
        let mut ct = ConvergenceTracker::new();
        ct.record(0, 1e-5);
        ct.record(1, 1e-3);
        let details = ct.details(&|id| format!("Block_{}", id));
        assert_eq!(details.len(), 2);
    }

    #[test]
    fn test_step_tracker() {
        let mut st = StepTracker::new();
        st.record(0, true, 1e-5, Some(0.9));
        st.record(1, true, 1e-3, Some(0.5));
        assert!(st.success);
        assert_eq!(st.max_error, 1e-3);
        assert_eq!(st.scale(), 0.5);

        st.record(2, false, 1e-1, None);
        assert!(!st.success);
    }

    #[test]
    fn test_diagnostics_summary() {
        let d = Diagnostics {
            time: 1.5,
            solve_residuals: HashMap::from([(0, 1e-5), (1, 1e-3)]),
            solve_iterations: 5,
            ..Diagnostics::new()
        };
        let summary = d.summary(&|id| format!("Block_{}", id));
        assert!(summary.contains("t = 1.5"));
        assert!(summary.contains("5 iterations"));
    }

    #[test]
    fn test_diagnostics_worst_block() {
        let d = Diagnostics {
            time: 1.0,
            solve_residuals: HashMap::from([(0, 1e-5), (1, 1e-3)]),
            ..Diagnostics::new()
        };
        let (id, err) = d.worst_block().unwrap();
        assert_eq!(id, 1);
        assert_eq!(err, 1e-3);
    }
}
