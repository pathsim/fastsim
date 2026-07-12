// SimEvent trait implementations for all concrete event types

use crate::events::eventtype::{CrossDir, EventDescriptor, SimEvent};
use crate::events::event::Event;
use crate::events::zerocrossing::{CrossingDirection, ZeroCrossing};
use crate::events::schedule::{Schedule, ScheduleList};
use crate::events::condition::Condition;

// -- Event (base) --

impl SimEvent for Event {
    fn is_active(&self) -> bool { self._active }
    fn on(&mut self) { self._active = true; }
    fn off(&mut self) { self._active = false; }
    fn reset(&mut self) { Event::reset(self); }
    fn buffer(&mut self, t: f64) { Event::buffer(self, t); }
    fn estimate(&self, _t: f64) -> Option<f64> { None }
    fn detect(&mut self, t: f64) -> (bool, bool, f64) { Event::detect(self, t) }
    fn resolve(&mut self, t: f64) { Event::resolve(self, t); }
    fn len(&self) -> usize { Event::len(self) }
    fn times(&self) -> &[f64] { &self._times }
    // Base Event monitors a host guard (func_evt) -> condition-like, opaque.
    fn ir_descriptor(&self) -> EventDescriptor { EventDescriptor::Condition }
}

// -- ZeroCrossing (unified: Both/Up/Down via direction field) --

impl SimEvent for ZeroCrossing {
    fn is_active(&self) -> bool { self._active }
    fn on(&mut self) { self._active = true; }
    fn off(&mut self) { self._active = false; }
    fn reset(&mut self) { ZeroCrossing::reset(self); }
    fn buffer(&mut self, t: f64) { ZeroCrossing::buffer(self, t); }
    fn estimate(&self, _t: f64) -> Option<f64> { None }
    fn detect(&mut self, t: f64) -> (bool, bool, f64) { ZeroCrossing::detect(self, t) }
    fn resolve(&mut self, t: f64) { ZeroCrossing::resolve(self, t); }
    fn len(&self) -> usize { ZeroCrossing::len(self) }
    fn times(&self) -> &[f64] { &self._times }
    fn ir_descriptor(&self) -> EventDescriptor {
        let direction = match self.direction {
            CrossingDirection::Both => CrossDir::Both,
            CrossingDirection::Up => CrossDir::Rising,
            CrossingDirection::Down => CrossDir::Falling,
        };
        EventDescriptor::ZeroCross { direction }
    }
}

// -- Schedule --

impl SimEvent for Schedule {
    fn is_active(&self) -> bool { self._active }
    fn on(&mut self) { self._active = true; }
    fn off(&mut self) { self._active = false; }
    fn reset(&mut self) { Schedule::reset(self); }
    fn buffer(&mut self, t: f64) { Schedule::buffer(self, t); }
    fn estimate(&self, t: f64) -> Option<f64> { Some(Schedule::estimate(self, t)) }
    fn detect(&mut self, t: f64) -> (bool, bool, f64) { Schedule::detect(self, t) }
    fn resolve(&mut self, t: f64) { Schedule::resolve(self, t); }
    fn len(&self) -> usize { Schedule::len(self) }
    fn times(&self) -> &[f64] { &self._times }
    fn ir_descriptor(&self) -> EventDescriptor {
        EventDescriptor::SchedulePeriodic { period: self.t_period, phase: self.t_start }
    }
}

impl SimEvent for ScheduleList {
    fn is_active(&self) -> bool { self._active }
    fn on(&mut self) { self._active = true; }
    fn off(&mut self) { self._active = false; }
    fn reset(&mut self) { ScheduleList::reset(self); }
    fn buffer(&mut self, t: f64) { ScheduleList::buffer(self, t); }
    fn estimate(&self, t: f64) -> Option<f64> { Some(ScheduleList::estimate(self, t)) }
    fn detect(&mut self, t: f64) -> (bool, bool, f64) { ScheduleList::detect(self, t) }
    fn resolve(&mut self, t: f64) { ScheduleList::resolve(self, t); }
    fn len(&self) -> usize { ScheduleList::len(self) }
    fn times(&self) -> &[f64] { &self._times }
    fn ir_descriptor(&self) -> EventDescriptor {
        EventDescriptor::ScheduleFixed { times: self.times_evt.clone() }
    }
}

// -- Condition --

impl SimEvent for Condition {
    fn is_active(&self) -> bool { self._active }
    fn on(&mut self) { self._active = true; }
    fn off(&mut self) { self._active = false; }
    fn reset(&mut self) { Condition::reset(self); }
    fn buffer(&mut self, t: f64) { Condition::buffer(self, t); }
    fn estimate(&self, _t: f64) -> Option<f64> { None }
    fn detect(&mut self, t: f64) -> (bool, bool, f64) { Condition::detect(self, t) }
    fn resolve(&mut self, t: f64) { Condition::resolve(self, t); }
    fn len(&self) -> usize { Condition::len(self) }
    fn times(&self) -> &[f64] { &self._times }
    fn ir_descriptor(&self) -> EventDescriptor { EventDescriptor::Condition }
}
