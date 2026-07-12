// Time-scheduled events
// Ported 1:1 from pathsim/events/schedule.py

use crate::constants::TOLERANCE;

/// Shared `detect` tail for time-scheduled events: given the next scheduled time
/// `t_next`, the current time `t`, the buffered previous time `history_t`, and
/// the close-enough `tolerance`, decide `(detected, close_enough, ratio)`. Both
/// `Schedule` and `ScheduleList` call this after computing their own `t_next`
/// and end-of-schedule condition.
fn detect_at(t_next: f64, t: f64, history_t: f64, tolerance: f64) -> (bool, bool, f64) {
    // No event yet.
    if t_next > t {
        return (false, false, 1.0);
    }
    // Close enough to the sample.
    if (t_next - t).abs() <= tolerance {
        return (true, true, 0.0);
    }
    // Already passed (buffered time is at/after the next sample).
    if history_t >= t_next {
        return (true, true, 0.0);
    }
    let ratio = (t_next - history_t) / (t - history_t).abs().max(TOLERANCE);
    (true, false, ratio)
}

/// Periodic time-based event. Triggers at t_start + n * t_period.
pub struct Schedule {
    pub func_act: Option<Box<dyn FnMut(f64)>>,
    pub tolerance: f64,
    pub t_start: f64,
    pub t_period: f64,
    pub t_end: Option<f64>,
    pub _history: (Option<f64>, f64),
    pub _times: Vec<f64>,
    pub _active: bool,
}

impl Schedule {
    pub fn new(
        t_start: f64,
        t_end: Option<f64>,
        t_period: f64,
        func_act: Option<Box<dyn FnMut(f64)>>,
        tolerance: f64,
    ) -> Self {
        Self {
            func_act, tolerance, t_start, t_period, t_end,
            _history: (None, 0.0), _times: Vec::new(), _active: true,
        }
    }

    pub fn periodic(t_start: f64, t_period: f64) -> Self {
        Self::new(t_start, None, t_period, None, TOLERANCE)
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

    /// Next scheduled event time.
    pub fn _next(&self) -> f64 {
        self.t_start + self._times.len() as f64 * self.t_period
    }

    pub fn estimate(&self, t: f64) -> f64 {
        self._next() - t
    }

    pub fn buffer(&mut self, t: f64) {
        self._history = (None, t);
    }

    pub fn detect(&mut self, t: f64) -> (bool, bool, f64) {
        let t_next = self._next();

        // End time reached?
        if let Some(t_end) = self.t_end {
            if t_next > t_end {
                self.off();
                return (false, false, 1.0);
            }
        }

        detect_at(t_next, t, self._history.1, self.tolerance)
    }

    pub fn resolve(&mut self, t: f64) {
        self._times.push(t);
        if let Some(ref mut func) = self.func_act {
            func(t);
        }
    }
}

/// List-based scheduled events. Triggers at specific times from a list.
pub struct ScheduleList {
    pub func_act: Option<Box<dyn FnMut(f64)>>,
    pub tolerance: f64,
    pub times_evt: Vec<f64>,
    pub _history: (Option<f64>, f64),
    pub _times: Vec<f64>,
    pub _active: bool,
}

impl ScheduleList {
    pub fn new(
        times_evt: Vec<f64>,
        func_act: Option<Box<dyn FnMut(f64)>>,
        tolerance: f64,
    ) -> Self {
        // Ensure ascending order
        let mut times_evt = times_evt;
        times_evt.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Self {
            func_act, tolerance, times_evt,
            _history: (None, 0.0), _times: Vec::new(), _active: true,
        }
    }

    pub fn from_times(times_evt: Vec<f64>) -> Self {
        Self::new(times_evt, None, TOLERANCE)
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

    pub fn _next(&self) -> f64 {
        let n = self._times.len();
        if n < self.times_evt.len() {
            self.times_evt[n]
        } else {
            *self.times_evt.last().unwrap()
        }
    }

    pub fn estimate(&self, t: f64) -> f64 {
        self._next() - t
    }

    pub fn buffer(&mut self, t: f64) {
        self._history = (None, t);
    }

    pub fn detect(&mut self, t: f64) -> (bool, bool, f64) {
        let n = self._times.len();
        if n >= self.times_evt.len() {
            self.off();
            return (false, false, 1.0);
        }

        let t_next = self._next();
        detect_at(t_next, t, self._history.1, self.tolerance)
    }

    pub fn resolve(&mut self, t: f64) {
        self._times.push(t);
        if let Some(ref mut func) = self.func_act {
            func(t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_init() {
        let s = Schedule::new(0.1, Some(200.0), 20.0, None, TOLERANCE);
        assert_eq!(s.t_start, 0.1);
        assert_eq!(s.t_end, Some(200.0));
        assert_eq!(s.t_period, 20.0);
    }

    #[test]
    fn test_schedule_next() {
        let mut s = Schedule::periodic(0.0, 20.0);
        assert_eq!(s._next(), 0.0);
        s.resolve(0.0);
        assert_eq!(s._next(), 20.0);
    }

    #[test]
    fn test_schedule_estimate() {
        let mut s = Schedule::periodic(2.0, 20.0);
        assert_eq!(s.estimate(0.0), 2.0);
        assert_eq!(s.estimate(1.0), 1.0);
        s.resolve(2.0);
        assert_eq!(s.estimate(2.0), 20.0);
        assert_eq!(s.estimate(13.0), 9.0);
    }

    #[test]
    fn test_schedule_detect() {
        let mut s = Schedule::periodic(2.0, 20.0);
        s.buffer(0.0);

        let (d, c, _r) = s.detect(0.0);
        assert!(!d); assert!(!c);

        let (d, c, r) = s.detect(4.0);
        assert!(d); assert!(!c);
        assert_eq!(r, 0.5);
    }

    #[test]
    fn test_schedule_list_init() {
        let s = ScheduleList::from_times(vec![1.0, 3.0, 5.0, 7.0]);
        assert_eq!(s.times_evt, vec![1.0, 3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_schedule_list_auto_sorts() {
        let s = ScheduleList::from_times(vec![1.0, 3.0, 5.0, 2.0, 7.0]);
        assert_eq!(s.times_evt, vec![1.0, 2.0, 3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_schedule_list_next() {
        let mut s = ScheduleList::from_times(vec![1.0, 3.0, 5.0, 7.0]);
        assert_eq!(s._next(), 1.0);
        s.resolve(1.0);
        assert_eq!(s._next(), 3.0);
        s.resolve(3.0);
        assert_eq!(s._next(), 5.0);
    }

    #[test]
    fn test_schedule_list_estimate() {
        let mut s = ScheduleList::from_times(vec![1.0, 3.0, 5.0, 7.0]);
        assert_eq!(s.estimate(0.0), 1.0);
        assert_eq!(s.estimate(0.5), 0.5);
        s.resolve(1.0);
        assert_eq!(s.estimate(1.0), 2.0);
        assert_eq!(s.estimate(2.0), 1.0);
    }

    #[test]
    fn test_schedule_list_detect() {
        let mut s = ScheduleList::from_times(vec![1.0, 3.0, 5.0, 7.0]);
        s.buffer(0.0);

        let (d, c, _r) = s.detect(0.0);
        assert!(!d); assert!(!c);

        let (d, c, r) = s.detect(2.0);
        assert!(d); assert!(!c);
        assert_eq!(r, 0.5);
    }

    #[test]
    fn test_schedule_list_func_act() {
        let s = ScheduleList::new(
            vec![1.0, 2.0, 3.0],
            Some(Box::new(|_t| {})),
            TOLERANCE,
        );
        assert!(s.func_act.is_some());
    }
}
