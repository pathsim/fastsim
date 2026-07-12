// Connection class: transfers data between blocks
// Ported 1:1 from pathsim/connection.py

use std::rc::Rc;
use std::cell::Cell;

use crate::blocks::block::BlockRef;
use crate::error::SimError;
use crate::utils::portreference::PortReference;

/// Shared reference to a Connection.
pub type ConnectionRef = Rc<Connection>;

/// Class to handle input-output relations of blocks by connecting them
/// (directed graph) and transferring data from the output port of the
/// source block to the input port of the target block.
///
/// The default ports for connection are (0) -> (0).
///
/// Supports:
/// - Single source, multiple targets (broadcast)
/// - MIMO connections (port slicing/lists)
/// - Self-connections (feedback)
pub struct Connection {
    /// Source block and output port(s)
    pub source: PortReference,
    /// Target blocks and input port(s)
    pub targets: Vec<PortReference>,
    /// Flag to set Connection as active or inactive (Cell for interior mutability)
    pub _active: Cell<bool>,
}

impl Connection {
    /// Create a new Connection from source and targets.
    ///
    /// Mirrors Python `Connection.__init__(source, *targets)`.
    /// Validates port dimensions (source ports count must match each target's).
    pub fn new(source: PortReference, targets: Vec<PortReference>) -> Self {
        // Validate port dimensions
        let n_src = source.len();
        for trg in &targets {
            if trg.len() != n_src {
                crate::utils::sink::warn(&format!(
                    "[fastsim WARNING] Connection port count mismatch: source={}, target={} — using minimum",
                    n_src, trg.len()
                ));
            }
        }

        // Validate port aliases
        source._validate_output_ports();
        for trg in &targets {
            trg._validate_input_ports();
        }

        Self {
            source,
            targets,
            _active: Cell::new(true),
        }
    }

    /// Number of ports in the connection.
    pub fn len(&self) -> usize {
        self.source.len()
    }

    pub fn is_empty(&self) -> bool {
        self.source.is_empty()
    }

    /// Is the connection active?
    pub fn is_active(&self) -> bool {
        self._active.get()
    }

    pub fn on(&self) {
        self._active.set(true);
    }

    pub fn off(&self) {
        self._active.set(false);
    }

    /// Returns all unique internal source and target blocks.
    ///
    /// Mirrors Python `get_blocks()`.
    pub fn get_blocks(&self) -> Vec<BlockRef> {
        let mut blocks: Vec<BlockRef> = vec![self.source.block.clone()];
        for trg in &self.targets {
            let already_in = blocks.iter().any(|b| Rc::ptr_eq(b, &trg.block));
            if !already_in {
                blocks.push(trg.block.clone());
            }
        }
        blocks
    }

    /// Transfers data from source output ports to all target input ports.
    /// Only transfers if the connection is active.
    ///
    /// Mirrors Python `Connection.update()`.
    pub fn update(&self) {
        if !self._active.get() { return; }
        for trg in &self.targets {
            self.source.to(trg);
        }
    }

    /// Eagerly resolve source and target port indices, growing the
    /// underlying input/output `Register`s to their final size.
    ///
    /// Normally registers grow lazily on the first `connection.update()`
    /// via `PortReference::_get_input_indices`. In the DAG case that is
    /// fine: the source-side connection fires before the target block's
    /// `update`, so by the time the block reads its inputs, the register
    /// has the right size. Inside an algebraic loop the order is reversed
    /// (`block.update()` first, then `conn.update()` in the same
    /// iteration), so the first iteration sees a register still at the
    /// `Block::default_block()` size and any block accessing inputs by
    /// position (e.g. a `function` closure indexing `u[1]`, or `adder`
    /// summing `u.iter()`) misreads — the function variant panics on
    /// out-of-bounds index, the adder variant silently sums fewer terms
    /// for one iteration before subsequent connection updates resize
    /// the register.
    ///
    /// Calling `resolve_ports` once during graph assembly closes that
    /// gap. Idempotent. New register slots are zero-initialised by
    /// `Vec::resize` in `Register::resize`.
    ///
    /// Mirrors pathsim `Connection.resolve_ports()` (PR pathsim#214).
    ///
    /// Fallible: a port name that does not resolve to an index on its block is
    /// a user configuration error, returned as `SimError` (surfaced at run
    /// setup) rather than panicking deep in the data-transfer hot path.
    pub fn resolve_ports(&self) -> Result<(), SimError> {
        self.source.resolve_output_indices()?;
        for trg in &self.targets {
            trg.resolve_input_indices()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::block::new_block_ref;
    use crate::utils::portreference::Port;

    #[test]
    fn test_basic_connection() {
        let b1 = new_block_ref(None, None);
        let b2 = new_block_ref(None, None);

        b1.borrow_mut().outputs.set_single(0, 42.0);

        let conn = Connection::new(
            PortReference::new(b1.clone(), None),
            vec![PortReference::new(b2.clone(), None)],
        );

        conn.update();
        assert_eq!(b2.borrow().inputs.get_single(0), 42.0);
    }

    #[test]
    fn test_multi_target_connection() {
        let b1 = new_block_ref(None, None);
        let b2 = new_block_ref(None, None);
        let b3 = new_block_ref(None, None);

        b1.borrow_mut().outputs.set_single(0, 7.0);

        let conn = Connection::new(
            PortReference::new(b1.clone(), None),
            vec![
                PortReference::new(b2.clone(), None),
                PortReference::new(b3.clone(), None),
            ],
        );

        conn.update();
        assert_eq!(b2.borrow().inputs.get_single(0), 7.0);
        assert_eq!(b3.borrow().inputs.get_single(0), 7.0);
    }

    #[test]
    fn test_mimo_connection() {
        let b1 = new_block_ref(None, None);
        let b2 = new_block_ref(None, None);

        b1.borrow_mut().outputs.set_single(0, 10.0);
        b1.borrow_mut().outputs.set_single(1, 20.0);

        let conn = Connection::new(
            PortReference::new(b1.clone(), Some(vec![Port::Index(0), Port::Index(1)])),
            vec![PortReference::new(b2.clone(), Some(vec![Port::Index(0), Port::Index(1)]))],
        );

        conn.update();
        assert_eq!(b2.borrow().inputs.get_single(0), 10.0);
        assert_eq!(b2.borrow().inputs.get_single(1), 20.0);
    }

    #[test]
    fn test_get_blocks() {
        let b1 = new_block_ref(None, None);
        let b2 = new_block_ref(None, None);

        let conn = Connection::new(
            PortReference::new(b1.clone(), None),
            vec![PortReference::new(b2.clone(), None)],
        );

        let blocks = conn.get_blocks();
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn test_get_blocks_self_connection() {
        let b1 = new_block_ref(None, None);

        let conn = Connection::new(
            PortReference::new(b1.clone(), None),
            vec![PortReference::new(b1.clone(), None)],
        );

        // Self-connection: only 1 unique block
        let blocks = conn.get_blocks();
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn test_on_off() {
        let b1 = new_block_ref(None, None);
        let b2 = new_block_ref(None, None);

        let conn = Connection::new(
            PortReference::new(b1, None),
            vec![PortReference::new(b2, None)],
        );

        assert!(conn.is_active());
        conn.off();
        assert!(!conn.is_active());
        conn.on();
        assert!(conn.is_active());
    }

    #[test]
    fn test_dimension_mismatch_warns() {
        // Mismatched port counts should warn but not panic
        let b1 = new_block_ref(None, None);
        let b2 = new_block_ref(None, None);

        let _conn = Connection::new(
            PortReference::new(b1, Some(vec![Port::Index(0), Port::Index(1)])),
            vec![PortReference::new(b2, Some(vec![Port::Index(0)]))],
        );
        // Should not panic — just warns to stderr
    }

    #[test]
    fn test_resolve_ports_grows_input_register() {
        // Fresh blocks have size-1 input/output registers from
        // Block::default_block(). resolve_ports must size them to fit
        // the connection's port indices BEFORE any update() runs.
        let b1 = new_block_ref(None, None);
        let b2 = new_block_ref(None, None);

        assert_eq!(b1.borrow().outputs.len(), 1);
        assert_eq!(b2.borrow().inputs.len(), 1);

        let conn = Connection::new(
            PortReference::new(b1.clone(), Some(vec![Port::Index(2)])),
            vec![PortReference::new(b2.clone(), Some(vec![Port::Index(3)]))],
        );

        // Lazy: ports validated but registers not yet resized
        assert_eq!(b1.borrow().outputs.len(), 1);
        assert_eq!(b2.borrow().inputs.len(), 1);

        conn.resolve_ports().unwrap();

        assert_eq!(b1.borrow().outputs.len(), 3);
        assert_eq!(b2.borrow().inputs.len(), 4);
        // New slots are zero-initialised
        assert_eq!(b2.borrow().inputs.get_single(3), 0.0);
    }

    #[test]
    fn test_resolve_ports_is_idempotent() {
        let b1 = new_block_ref(None, None);
        let b2 = new_block_ref(None, None);

        let conn = Connection::new(
            PortReference::new(b1.clone(), Some(vec![Port::Index(5)])),
            vec![PortReference::new(b2.clone(), Some(vec![Port::Index(7)]))],
        );

        conn.resolve_ports().unwrap();
        let size_in = b2.borrow().inputs.len();
        let size_out = b1.borrow().outputs.len();

        // Seed a value to detect data loss on a second resolve
        b2.borrow_mut().inputs.set_single(7, 42.0);

        conn.resolve_ports().unwrap();
        assert_eq!(b2.borrow().inputs.len(), size_in);
        assert_eq!(b1.borrow().outputs.len(), size_out);
        assert_eq!(b2.borrow().inputs.get_single(7), 42.0);
    }

    #[test]
    fn test_resolve_ports_multi_target() {
        // All targets must be resolved, not just the first.
        let b1 = new_block_ref(None, None);
        let b2 = new_block_ref(None, None);
        let b3 = new_block_ref(None, None);

        let conn = Connection::new(
            PortReference::new(b1.clone(), Some(vec![Port::Index(1)])),
            vec![
                PortReference::new(b2.clone(), Some(vec![Port::Index(4)])),
                PortReference::new(b3.clone(), Some(vec![Port::Index(6)])),
            ],
        );

        conn.resolve_ports().unwrap();

        assert_eq!(b1.borrow().outputs.len(), 2);
        assert_eq!(b2.borrow().inputs.len(), 5);
        assert_eq!(b3.borrow().inputs.len(), 7);
    }
}
