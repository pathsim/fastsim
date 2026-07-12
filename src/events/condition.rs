// Condition events with bisection refinement
// Ported 1:1 from pathsim/events/condition.py

use crate::constants::EVT_TOLERANCE;

/// Condition event: triggers when event function evaluates to true.
///
/// Uses bisection for event location (non-smooth event function).
/// Deactivates after first resolution (one-shot).
pub struct Condition {
    pub func_evt: Box<dyn Fn(f64) -> bool>,
    pub func_act: Option<Box<dyn FnMut(f64)>>,
    pub tolerance: f64,
    pub _history: (Option<bool>, f64),
    pub _times: Vec<f64>,
    pub _active: bool,
}

impl Condition {
    pub fn new(
        func_evt: impl Fn(f64) -> bool + 'static,
        func_act: Option<Box<dyn FnMut(f64)>>,
        tolerance: f64,
    ) -> Self {
        Self {
            func_evt: Box::new(func_evt),
            func_act, tolerance,
            _history: (None, 0.0), _times: Vec::new(), _active: true,
        }
    }

    pub fn from_evt(func_evt: impl Fn(f64) -> bool + 'static) -> Self {
        Self::new(func_evt, None, EVT_TOLERANCE)
    }

    pub fn len(&self) -> usize { self._times.len() }
    pub fn is_empty(&self) -> bool { self._times.is_empty() }
    pub fn is_active(&self) -> bool { self._active }
    pub fn on(&mut self) { self._active = true; }
    pub fn off(&mut self) { self._active = false; }

    pub fn reset(&mut self) {
        self._history = (None, 0.0);
        self._times.clear();
        self._active = true;
    }

    pub fn buffer(&mut self, t: f64) {
        self._history = (Some((self.func_evt)(t)), t);
    }

    /// Detect: bisection method for non-smooth event function.
    ///
    /// Returns (detected, close, ratio).
    /// ratio = 0.5 for bisection (halves the timestep).
    /// ratio = 1.0 when close enough.
    pub fn detect(&self, t: f64) -> (bool, bool, f64) {
        let (_result, _t) = self._history;
        let result = (self.func_evt)(t);

        // Check if interval narrowed down sufficiently
        let close = result && (t - _t) < self.tolerance;

        if close {
            return (true, true, 1.0);
        }

        // Half the stepsize (bisection)
        (result, false, 0.5)
    }

    /// Resolve: record time, call action, deactivate (one-shot).
    pub fn resolve(&mut self, t: f64) {
        self._times.push(t);
        if let Some(ref mut func) = self.func_act {
            func(t);
        }
        self.off();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_condition_detect_false() {
        let mut e = Condition::from_evt(|t| t > 5.0);
        e.buffer(1.0);
        let (de, cl, ra) = e.detect(2.0);
        assert!(!de); assert!(!cl); assert_eq!(ra, 0.5);
    }

    #[test]
    fn test_condition_detect_true_not_close() {
        let mut e = Condition::new(|t| t > 5.0, None, 0.1);
        e.buffer(4.0);
        let (de, cl, ra) = e.detect(6.0);
        assert!(de); assert!(!cl); assert_eq!(ra, 0.5);
    }

    #[test]
    fn test_condition_detect_true_close() {
        let mut e = Condition::new(|t| t > 5.0, None, 0.1);
        e.buffer(5.05);
        let (de, cl, ra) = e.detect(5.08);
        assert!(de); assert!(cl); assert_eq!(ra, 1.0);
    }

    #[test]
    fn test_condition_resolve_deactivates() {
        let mut e = Condition::from_evt(|t| t > 5.0);
        assert!(e._active);
        e.resolve(5.5);
        assert!(!e._active);
        assert_eq!(e._times.len(), 1);
        assert_eq!(e._times[0], 5.5);
    }

    #[test]
    fn test_condition_resolve_with_action() {
        use crate::utils::fastcell::FastCell;
        use std::rc::Rc;
        let called = Rc::new(FastCell::new(Vec::<f64>::new()));
        let called_clone = called.clone();
        let mut e = Condition::new(
            |t| t > 5.0,
            Some(Box::new(move |t| { called_clone.borrow_mut().push(t); })),
            EVT_TOLERANCE,
        );
        assert!(e._active);
        e.resolve(5.5);
        assert!(!e._active);
        assert_eq!(called.borrow().len(), 1);
        assert_eq!(called.borrow()[0], 5.5);
    }

    #[test]
    fn test_condition_bisection() {
        let mut e = Condition::new(|t| t > 10.0, None, 0.1);

        // Large gap
        e.buffer(9.0);
        let (de, cl, ra) = e.detect(11.0);
        assert!(de); assert!(!cl); assert_eq!(ra, 0.5);

        // Small gap
        e.buffer(10.05);
        let (de, cl, ra) = e.detect(10.08);
        assert!(de); assert!(cl); assert_eq!(ra, 1.0);
    }

    #[test]
    fn test_condition_len() {
        let mut e = Condition::from_evt(|t| t > 5.0);
        assert_eq!(e.len(), 0);
        e.resolve(5.5);
        assert_eq!(e.len(), 1);
    }
}
