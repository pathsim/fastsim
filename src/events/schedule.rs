// Time-scheduled events
// Ported 1:1 from pathsim/events/schedule.py

use crate::constants::TOLERANCE;
use crate::events::active::{active_flag, ActiveFlag};

/// Effective detection tolerance at evaluation time `t`.
///
/// The simulation time is accumulated by repeated addition of the timestep, so
/// it drifts away from the exact schedule times by a floating point error that
/// grows with `t`. The configured absolute tolerance cannot cover this, which
/// makes ticks that land within a few ulp of a step boundary flip between
/// neighboring timesteps and lose the final tick of a run. The widened
/// tolerance absorbs that drift while staying far below the schedule spacing,
/// so no genuine tick can be swallowed. Upstream: pathsim#249.
fn tolerance_at(tolerance: f64, t: f64, spacing: f64) -> f64 {
    tolerance.max((1e-10 * t.abs()).min(0.1 * spacing))
}

/// Shared `detect` tail for time-scheduled events: given the next scheduled time
/// `t_next`, the current time `t`, the buffered previous time `history_t`, and
/// the close-enough `tolerance` (already widened via `tolerance_at`), decide
/// `(detected, close_enough, ratio)`. Both `Schedule` and `ScheduleList` call
/// this after computing their own `t_next` and end-of-schedule condition.
fn detect_at(t_next: f64, t: f64, history_t: f64, tolerance: f64) -> (bool, bool, f64) {
    // No event yet. A tick within tolerance of the step end counts as inside
    // the step — otherwise clock drift pushes it into the next step and the
    // final tick of a run is lost (pathsim#249).
    if t_next > t + tolerance {
        return (false, false, 1.0);
    }
    // Close enough to the sample. `t` is the END of the step and the caller
    // resolves at `step_start + ratio * dt`, so an event sitting on `t` is at
    // ratio 1 — ratio 0 would place it a full `dt` early. `ZeroCrossing` uses
    // the same convention for its exact hit. Upstream: pathsim#248.
    if (t_next - t).abs() <= tolerance {
        return (true, true, 1.0);
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
    pub _active: ActiveFlag,
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
            _history: (None, 0.0), _times: Vec::new(), _active: active_flag(),
        }
    }

    pub fn periodic(t_start: f64, t_period: f64) -> Self {
        Self::new(t_start, None, t_period, None, TOLERANCE)
    }

    pub fn len(&self) -> usize { self._times.len() }
    pub fn is_empty(&self) -> bool { self._times.is_empty() }
    pub fn is_active(&self) -> bool { self._active.get() }
    pub fn on(&self) { self._active.set(true); }
    pub fn off(&self) { self._active.set(false); }
    /// Handle on the activation flag, for callers that must toggle it without
    /// borrowing the event (see `events::active`).
    pub fn active_flag(&self) -> ActiveFlag { self._active.clone() }

    pub fn reset(&mut self) {
        self._history = (None, 0.0);
        self._times.clear();
        self._active.set(true);
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

        let tol = tolerance_at(self.tolerance, t, self.t_period);
        detect_at(t_next, t, self._history.1, tol)
    }

    /// Resolve at the exact scheduled time. The caller passes the numerically
    /// reached time, which carries the accumulated drift of the simulation
    /// clock; the schedule knows its own exact event time, so that is what
    /// gets recorded and handed to the action — timestamps land exactly on
    /// the schedule. Upstream: pathsim#249.
    pub fn resolve(&mut self, _t: f64) {
        let t_evt = self._next();
        self._times.push(t_evt);
        if let Some(ref mut func) = self.func_act {
            func(t_evt);
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
    pub _active: ActiveFlag,
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
            _history: (None, 0.0), _times: Vec::new(), _active: active_flag(),
        }
    }

    pub fn from_times(times_evt: Vec<f64>) -> Self {
        Self::new(times_evt, None, TOLERANCE)
    }

    pub fn len(&self) -> usize { self._times.len() }
    pub fn is_empty(&self) -> bool { self._times.is_empty() }
    pub fn is_active(&self) -> bool { self._active.get() }
    pub fn on(&self) { self._active.set(true); }
    pub fn off(&self) { self._active.set(false); }
    /// Handle on the activation flag, for callers that must toggle it without
    /// borrowing the event (see `events::active`).
    pub fn active_flag(&self) -> ActiveFlag { self._active.clone() }

    pub fn reset(&mut self) {
        self._history = (None, 0.0);
        self._times.clear();
        self._active.set(true);
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

    /// Gap between the upcoming event and its successor in the time list,
    /// used to bound the effective detection tolerance.
    fn spacing(&self) -> f64 {
        let n = self._times.len();
        if n + 1 < self.times_evt.len() {
            self.times_evt[n + 1] - self.times_evt[n]
        } else {
            f64::INFINITY
        }
    }

    pub fn detect(&mut self, t: f64) -> (bool, bool, f64) {
        let n = self._times.len();
        if n >= self.times_evt.len() {
            self.off();
            return (false, false, 1.0);
        }

        let t_next = self._next();
        let tol = tolerance_at(self.tolerance, t, self.spacing());
        detect_at(t_next, t, self._history.1, tol)
    }

    /// Resolve at the exact scheduled time — see `Schedule::resolve`.
    pub fn resolve(&mut self, _t: f64) {
        let t_evt = self._next();
        self._times.push(t_evt);
        if let Some(ref mut func) = self.func_act {
            func(t_evt);
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

    /// An event landing exactly on `t` is at the end of the step, not its start.
    #[test]
    fn test_schedule_detect_exact_hit_is_end_of_step() {
        let mut s = Schedule::periodic(2.0, 20.0);
        s.buffer(1.0);

        let (d, c, r) = s.detect(2.0);
        assert!(d); assert!(c);
        assert_eq!(r, 1.0);
    }

    #[test]
    fn test_schedule_list_detect_exact_hit_is_end_of_step() {
        let mut s = ScheduleList::from_times(vec![1.0, 3.0, 5.0, 7.0]);
        s.buffer(0.5);

        let (d, c, r) = s.detect(1.0);
        assert!(d); assert!(c);
        assert_eq!(r, 1.0);
    }

    /// The other close branch: the event was already passed before this step,
    /// so it belongs at the step start and keeps ratio 0.
    #[test]
    fn test_schedule_detect_already_passed_is_start_of_step() {
        let mut s = Schedule::periodic(2.0, 20.0);
        s.buffer(3.0);

        let (d, c, r) = s.detect(4.0);
        assert!(d); assert!(c);
        assert_eq!(r, 0.0);
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

    // --- drift handling (pathsim#249) ----------------------------------

    #[test]
    fn resolve_records_the_exact_scheduled_time() {
        // The caller passes the numerically reached time, which carries the
        // accumulated drift of the simulation clock; the schedule records its
        // own exact event time instead.
        let mut s = Schedule::new(0.0, None, 0.01, None, TOLERANCE);
        s.resolve(0.0);
        s.resolve(0.010000000000000002); // one ulp of drift
        s.resolve(0.020000000000000004);
        assert_eq!(s._times, vec![0.0, 0.01, 0.02]);
    }

    #[test]
    fn detect_absorbs_clock_drift_at_the_step_boundary() {
        // A tick landing a few ulp behind the end of the step must still be
        // detected in that step (close, ratio 1), otherwise clock drift pushes
        // it into the next step and the final tick of a run is lost.
        let mut s = Schedule::new(0.0, None, 0.01, None, TOLERANCE);
        s._times = vec![0.0; 10]; // ten ticks resolved, next is t=0.1
        let t_drifted = f64::from_bits(0.1f64.to_bits() - 1); // one ulp below
        assert!(t_drifted < 0.1);
        s.buffer(t_drifted - 0.01);
        let (d, c, r) = s.detect(t_drifted);
        assert!(d);
        assert!(c);
        assert_eq!(r, 1.0);
    }

    #[test]
    fn drift_tolerance_is_capped_by_the_spacing() {
        // The allowance must stay far below the schedule spacing so no genuine
        // tick can be absorbed by the widened tolerance.
        assert!(tolerance_at(TOLERANCE, 1e6, 1e-12) <= 0.1e-12);
    }

    #[test]
    fn schedule_list_resolves_at_the_listed_times() {
        let mut s = ScheduleList::new(vec![1.0, 2.0], None, TOLERANCE);
        s.resolve(1.0000000000000002);
        s.resolve(2.0000000000000004);
        assert_eq!(s._times, vec![1.0, 2.0]);
    }
}
