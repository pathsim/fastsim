// SimEvent trait: polymorphic interface for all event types
// Mirrors Python's Event class with virtual methods (detect, resolve, buffer, etc.)

/// Zero-crossing direction, mirrored neutrally for IR export so this module
/// stays independent of `events::zerocrossing`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossDir {
    Rising,
    Falling,
    Both,
}

/// Statically-known structure of an event, for IR export. Captures the firing
/// *kind* and any concrete timing; the guard / action are host closures and are
/// deliberately NOT captured — a backend treats them as opaque.
#[derive(Debug, Clone)]
pub enum EventDescriptor {
    /// Fires at `phase`, then every `period`.
    SchedulePeriodic { period: f64, phase: f64 },
    /// Fires at the given absolute times.
    ScheduleFixed { times: Vec<f64> },
    /// Fires when a host guard crosses zero in `direction` (guard opaque).
    ZeroCross { direction: CrossDir },
    /// Fires while a host guard is non-zero (guard opaque).
    Condition,
}

/// Trait that all simulation events must implement.
pub trait SimEvent {
    fn is_active(&self) -> bool;
    fn on(&mut self);
    fn off(&mut self);
    fn reset(&mut self);
    fn buffer(&mut self, t: f64);
    fn estimate(&self, t: f64) -> Option<f64>;
    fn detect(&mut self, t: f64) -> (bool, bool, f64);
    fn resolve(&mut self, t: f64);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn times(&self) -> &[f64];
    /// Statically-known structure for IR export (kind + timing; guard/action
    /// stay opaque). Used when lowering opaque blocks and simulation-level
    /// events into the IR.
    fn ir_descriptor(&self) -> EventDescriptor;
}
