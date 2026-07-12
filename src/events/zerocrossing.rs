// Zero-crossing event detection
// Unified implementation with CrossingDirection enum

use crate::constants::TOLERANCE;

/// Direction of zero-crossing to detect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CrossingDirection {
    /// Triggers on any sign change (+ to - or - to +)
    Both,
    /// Triggers only on negative-to-positive transitions
    Up,
    /// Triggers only on positive-to-negative transitions
    Down,
}

/// Zero-crossing event detector.
///
/// Detects when the event function crosses zero. Direction controls
/// whether bidirectional, upward-only, or downward-only crossings trigger.
/// Uses linear interpolation (secant method) to estimate event location.
pub struct ZeroCrossing {
    pub direction: CrossingDirection,
    pub func_evt: Box<dyn Fn(f64) -> f64>,
    pub func_act: Option<Box<dyn FnMut(f64)>>,
    pub tolerance: f64,
    pub _history: (Option<f64>, f64),
    pub _times: Vec<f64>,
    pub _active: bool,
}

impl ZeroCrossing {
    pub fn new(
        func_evt: impl Fn(f64) -> f64 + 'static,
        func_act: Option<Box<dyn FnMut(f64)>>,
        tolerance: f64,
    ) -> Self {
        Self {
            direction: CrossingDirection::Both,
            func_evt: Box::new(func_evt),
            func_act, tolerance,
            _history: (None, 0.0), _times: Vec::new(), _active: true,
        }
    }

    pub fn with_direction(
        direction: CrossingDirection,
        func_evt: impl Fn(f64) -> f64 + 'static,
        func_act: Option<Box<dyn FnMut(f64)>>,
        tolerance: f64,
    ) -> Self {
        Self {
            direction,
            func_evt: Box::new(func_evt),
            func_act, tolerance,
            _history: (None, 0.0), _times: Vec::new(), _active: true,
        }
    }

    pub fn from_evt(func_evt: impl Fn(f64) -> f64 + 'static) -> Self {
        Self::new(func_evt, None, crate::constants::EVT_TOLERANCE)
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

    pub fn detect(&self, t: f64) -> (bool, bool, f64) {
        let (prev_val, _prev_t) = self._history;
        let prev = match prev_val {
            Some(r) => r,
            None => return (false, false, 1.0),
        };

        let current = (self.func_evt)(t);

        // Exactly hit zero
        match self.direction {
            CrossingDirection::Both => {
                if current == 0.0 && prev != 0.0 {
                    return (true, true, 1.0);
                }
            }
            CrossingDirection::Up => {
                if current == 0.0 && prev < 0.0 {
                    return (true, true, 1.0);
                }
            }
            CrossingDirection::Down => {
                if current == 0.0 && prev > 0.0 {
                    return (true, true, 1.0);
                }
            }
        }

        let close = current.abs() <= self.tolerance;

        // Check for sign change with direction filter
        let sign_change = (current * prev).signum() < 0.0;
        let is_event = sign_change && match self.direction {
            CrossingDirection::Both => true,
            CrossingDirection::Up => current > prev && prev < 0.0,
            CrossingDirection::Down => current < prev && prev > 0.0,
        };

        if !is_event {
            return (false, false, 1.0);
        }

        // Linear interpolation for event location ratio
        let ratio = prev.abs() / (prev - current).abs().max(TOLERANCE);
        (true, close, ratio)
    }

    pub fn resolve(&mut self, t: f64) {
        self._times.push(t);
        if let Some(ref mut func) = self.func_act {
            func(t);
        }
    }

    pub fn estimate(&self, _t: f64) -> Option<f64> { None }
}

// Backward-compatibility aliases. NOTE: both resolve to the same `ZeroCrossing`
// type — the crossing direction is chosen at construction (`new_up` / `new_down`
// set the runtime `direction` field), NOT by the alias. They are not distinct
// types, so the alias does not enforce a direction at compile time.
pub type ZeroCrossingUp = ZeroCrossing;
pub type ZeroCrossingDown = ZeroCrossing;

// Constructor functions matching old API
impl ZeroCrossing {
    pub fn new_up(
        func_evt: impl Fn(f64) -> f64 + 'static,
        func_act: Option<Box<dyn FnMut(f64)>>,
        tolerance: f64,
    ) -> Self {
        Self::with_direction(CrossingDirection::Up, func_evt, func_act, tolerance)
    }

    pub fn new_down(
        func_evt: impl Fn(f64) -> f64 + 'static,
        func_act: Option<Box<dyn FnMut(f64)>>,
        tolerance: f64,
    ) -> Self {
        Self::with_direction(CrossingDirection::Down, func_evt, func_act, tolerance)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    // -- ZeroCrossing (bidirectional) --

    #[test]
    fn test_zc_detect_up() {
        let mut e = ZeroCrossing::from_evt(|t| t - 2.0);

        e.buffer(0.0);
        let (de, cl, ra) = e.detect(1.0);
        assert!(!de); assert!(!cl); assert_eq!(ra, 1.0);

        let (de, cl, ra) = e.detect(3.0);
        assert!(de); assert!(!cl);
        assert!((ra - 2.0 / 3.0).abs() < 1e-10);

        e.buffer(1.0);
        let (de, cl, ra) = e.detect(3.0);
        assert!(de); assert!(!cl);
        assert!((ra - 0.5).abs() < 1e-10);

        e.buffer(3.0);
        let (de, cl, ra) = e.detect(4.0);
        assert!(!de); assert!(!cl); assert_eq!(ra, 1.0);
    }

    // -- ZeroCrossingUp --

    #[test]
    fn test_zc_up_detect_up() {
        let mut e = ZeroCrossing::new_up(|t| t - 2.0, None, crate::constants::EVT_TOLERANCE);

        e.buffer(0.0);
        let (de, _, _) = e.detect(1.0);
        assert!(!de);

        let (de, _, ra) = e.detect(3.0);
        assert!(de);
        assert!((ra - 2.0 / 3.0).abs() < 1e-10);

        e.buffer(1.0);
        let (de, _, ra) = e.detect(3.0);
        assert!(de);
        assert!((ra - 0.5).abs() < 1e-10);

        e.buffer(3.0);
        let (de, _, _) = e.detect(4.0);
        assert!(!de);
    }

    #[test]
    fn test_zc_up_no_down() {
        let mut e = ZeroCrossing::new_up(|t| t - 2.0, None, crate::constants::EVT_TOLERANCE);
        e.buffer(3.0); // val = 1 (positive)
        let (de, _, _) = e.detect(0.0); // val = -2 (crossing downward)
        assert!(!de); // should NOT trigger
    }

    // -- ZeroCrossingDown --

    #[test]
    fn test_zc_down_no_up() {
        let mut e = ZeroCrossing::new_down(|t| t - 2.0, None, crate::constants::EVT_TOLERANCE);

        e.buffer(0.0);
        // Crossing upward -> should NOT trigger
        let (de, _, _) = e.detect(3.0);
        assert!(!de);

        e.buffer(1.0);
        let (de, _, _) = e.detect(3.0);
        assert!(!de);
    }

    #[test]
    fn test_zc_down_detect_down() {
        let mut e = ZeroCrossing::new_down(|t| t - 2.0, None, crate::constants::EVT_TOLERANCE);
        e.buffer(3.0); // val = 1 (positive)
        let (de, _, _) = e.detect(0.0); // val = -2 (crossing downward)
        assert!(de); // SHOULD trigger
    }

    #[test]
    fn test_resolve_and_len() {
        let mut e = ZeroCrossing::from_evt(|t| t - 1.0);
        assert_eq!(e.len(), 0);
        e.resolve(1.0);
        assert_eq!(e.len(), 1);
        e.resolve(2.0);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn test_on_off_reset() {
        let mut e = ZeroCrossing::from_evt(|t| t);
        assert!(e.is_active());
        e.off();
        assert!(!e.is_active());
        e.on();
        assert!(e.is_active());
        e.resolve(1.0);
        assert_eq!(e.len(), 1);
        e.reset();
        assert_eq!(e.len(), 0);
        assert!(e.is_active());
    }
}
