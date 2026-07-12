//! Zero-overhead interior mutability cell.
//!
//! Drop-in replacement for `RefCell<T>` that eliminates runtime borrow checking.
//! Uses `UnsafeCell` internally. Safe because fastsim's simulation loop guarantees
//! that no two mutable borrows of the same block exist simultaneously:
//! - Blocks are borrowed one at a time in DAG order
//! - Python callbacks run synchronously (no concurrency)
//! - Event callbacks don't re-borrow the same block

use std::cell::UnsafeCell;

pub struct FastCell<T: ?Sized>(UnsafeCell<T>);

// FastCell is !Sync (like RefCell), but we need Send for some patterns.
// The bound mirrors `RefCell`'s: a cell may only move to another thread when
// its contents may. An unconditional impl would let e.g. `FastCell<Rc<T>>`
// cross threads through the type system — unsound, even if no current caller
// does so (the `parallel` feature ships rayon in this very crate).
unsafe impl<T: Send + ?Sized> Send for FastCell<T> {}

impl<T> FastCell<T> {
    #[inline(always)]
    pub fn new(value: T) -> Self {
        Self(UnsafeCell::new(value))
    }

    /// Consume the cell, returning the wrapped value.
    #[inline(always)]
    pub fn into_inner(self) -> T {
        self.0.into_inner()
    }
}

impl<T: ?Sized> FastCell<T> {
    /// Immutable borrow — zero overhead.
    ///
    /// Named `borrow`/`borrow_mut` to mirror `RefCell`; the `should_implement_trait`
    /// lint wants `std::borrow::Borrow`, but this is an inherent unchecked accessor.
    #[inline(always)]
    #[allow(clippy::should_implement_trait)]
    pub fn borrow(&self) -> &T {
        unsafe { &*self.0.get() }
    }

    /// Mutable borrow — zero overhead.
    ///
    /// Returning `&mut T` from `&self` is the whole point of this primitive:
    /// `FastCell` is a deliberate, single-threaded interior-mutability cell
    /// (an unchecked `RefCell`). The `mut_from_ref` lint flags exactly this
    /// pattern, so it is allowed here by design.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub fn borrow_mut(&self) -> &mut T {
        unsafe { &mut *self.0.get() }
    }

}

impl<T: std::fmt::Debug + ?Sized> std::fmt::Debug for FastCell<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FastCell({:?})", self.borrow())
    }
}
