// Shared activation flag for events.

use std::cell::Cell;
use std::rc::Rc;

/// The `_active` flag of an event, held behind an `Rc<Cell<_>>` rather than as a
/// plain field.
///
/// Events are stored in `FastCell`s, which do not track borrows: the simulation
/// takes a `&mut` for the whole of `resolve`, and `resolve` calls the user's
/// action function. That action routinely switches event tracking — including
/// on the very event being resolved:
///
/// ```text
/// def slip_to_stick_act(t):          # the action of E_slip_to_stick
///     E_slip_to_stick.off()          # ... turns itself off
///     E_stick_to_slip.on()
/// ```
///
/// (from pathsim's `example_stickslip_event.py`; `example_bouncingball_switched.py`
/// does the same through a `Condition`.)
///
/// Reaching the flag through the cell would mean a second `&mut` to an event
/// that is already mutably borrowed further up the stack. Handing the Python
/// wrapper its own handle on the flag instead keeps `on()`/`off()` off the
/// event body entirely, so they are safe to call from inside any callback.
///
/// This applies to `on`/`off` only. Every other method — `reset`, `buffer`,
/// `detect`, `resolve` — goes through the cell as usual and must not be called
/// re-entrantly on the event being resolved.
pub type ActiveFlag = Rc<Cell<bool>>;

/// A fresh flag, active.
pub fn active_flag() -> ActiveFlag {
    Rc::new(Cell::new(true))
}
