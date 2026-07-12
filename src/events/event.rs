// Event base class for event detection and resolution
// Ported 1:1 from pathsim/events/_event.py

use crate::constants::EVT_TOLERANCE;

/// Base class of the event handling system.
///
/// Monitors system state by evaluating an event function (func_evt).
/// If an event is detected, some action (func_act) is performed.
///
/// Methods are structured such that event detection can be separated from
/// resolution — required for adaptive timestep solvers to approach the event.
///
/// NOTE: this is the generic base / null-event template (its `detect` never
/// fires on its own). Production code uses the concrete `ZeroCrossing`,
/// `Schedule`, `ScheduleList`, and `Condition` types; this type is currently
/// exercised only by its own unit tests.
pub struct Event {
    /// Event function: evaluates system state, zeros/true = events
    pub func_evt: Option<Box<dyn Fn(f64) -> f64>>,
    /// Action function for event resolution
    pub func_act: Option<Box<dyn FnMut(f64)>>,
    /// Tolerance for event location
    pub tolerance: f64,
    /// History: (evaluation_result, evaluation_time)
    pub _history: (Option<f64>, f64),
    /// Recorded event times
    pub _times: Vec<f64>,
    /// Active flag
    pub _active: bool,
}

impl Event {
    pub fn new(
        func_evt: Option<Box<dyn Fn(f64) -> f64>>,
        func_act: Option<Box<dyn FnMut(f64)>>,
        tolerance: f64,
    ) -> Self {
        Self {
            func_evt, func_act, tolerance,
            _history: (None, 0.0),
            _times: Vec::new(),
            _active: true,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(None, None, EVT_TOLERANCE)
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

    /// Buffer event function evaluation before timestep.
    pub fn buffer(&mut self, t: f64) {
        if let Some(ref func) = self.func_evt {
            self._history = (Some(func(t)), t);
        }
    }

    /// Estimate time until next event (base: None = no estimate).
    pub fn estimate(&self, _t: f64) -> Option<f64> {
        None
    }

    /// Detect if an event occurred.
    /// Returns (detected, close, ratio).
    /// Base implementation: no event.
    pub fn detect(&self, _t: f64) -> (bool, bool, f64) {
        (false, false, 1.0)
    }

    /// Resolve the event: record time and call action function.
    pub fn resolve(&mut self, t: f64) {
        self._times.push(t);
        if let Some(ref mut func) = self.func_act {
            func(t);
        }
    }

    /// Iterator over recorded event times.
    pub fn times(&self) -> &[f64] {
        &self._times
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_init_default() {
        let e = Event::with_defaults();
        assert!(e.func_evt.is_none());
        assert!(e.func_act.is_none());
        assert_eq!(e.tolerance, EVT_TOLERANCE);
        assert_eq!(e._history, (None, 0.0));
        assert!(e._active);
    }

    #[test]
    fn test_event_init_custom() {
        let e = Event::new(
            Some(Box::new(|_t| 1.0)),
            Some(Box::new(|_t| {})),
            1e-6,
        );
        assert!(e.func_evt.is_some());
        assert!(e.func_act.is_some());
        assert_eq!(e.tolerance, 1e-6);
    }

    #[test]
    fn test_event_on_off() {
        let mut e = Event::with_defaults();
        assert!(e._active);
        e.off();
        assert!(!e._active);
        e.on();
        assert!(e._active);
    }

    #[test]
    fn test_event_len() {
        let mut e = Event::with_defaults();
        assert_eq!(e.len(), 0);
        e._times = vec![1.0, 2.0, 3.0];
        assert_eq!(e.len(), 3);
    }

    #[test]
    fn test_event_iter() {
        let mut e = Event::with_defaults();
        e._times = vec![1.0, 2.0, 3.0];
        for (i, &t) in e.times().iter().enumerate() {
            assert_eq!(t, (i + 1) as f64);
        }
    }

    #[test]
    fn test_event_detect_base() {
        let e = Event::new(Some(Box::new(|_| 0.0)), None, EVT_TOLERANCE);
        let (de, cl, ra) = e.detect(0.0);
        assert!(!de);
        assert!(!cl);
        assert_eq!(ra, 1.0);
    }

    #[test]
    fn test_event_resolve() {
        let mut e = Event::new(Some(Box::new(|_| 0.0)), None, EVT_TOLERANCE);
        for t in 0..5 {
            e.resolve(t as f64);
            assert_eq!(e.len(), t + 1);
        }
    }

    #[test]
    fn test_event_reset() {
        let mut e = Event::with_defaults();
        e._times = vec![1.0, 2.0];
        e._active = false;
        e.reset();
        assert!(e._times.is_empty());
        assert!(e._active);
        assert_eq!(e._history, (None, 0.0));
    }
}
