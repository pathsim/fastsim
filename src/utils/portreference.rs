// PortReference: block + port list reference with cached indices
// Ported 1:1 from pathsim/utils/portreference.py

use std::collections::HashSet;
use std::rc::Rc;

use crate::blocks::block::BlockRef;
use crate::error::SimError;
use crate::utils::fastcell::FastCell;

/// Port specifier: either an integer index or a named alias.
/// Mirrors Python's mixed int/str port keys.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Port {
    Index(usize),
    Name(String),
}

/// Container class that holds a reference to a block and a list of ports.
/// Optimized with cached integer indices for ultra-fast transfers.
///
/// Note: The default port, when no ports are defined in the arguments is `0`.
///
/// Mirrors Python's PortReference exactly:
/// - Holds a direct reference to the Block (not an ID)
/// - Caches resolved integer indices lazily
/// - Validates port existence at construction time
/// - Provides `to()` for direct data transfer between blocks
pub struct PortReference {
    /// Direct reference to the block (matches Python's `self.block`)
    pub block: BlockRef,
    /// List of port specifiers
    pub ports: Vec<Port>,
    /// Cached resolved input indices (lazily initialized)
    _input_indices: FastCell<Option<Vec<usize>>>,
    /// Cached resolved output indices (lazily initialized)
    _output_indices: FastCell<Option<Vec<usize>>>,
}

impl PortReference {
    /// Create a new PortReference with the given block and ports.
    ///
    /// Mirrors Python `PortReference.__init__(block, ports)`.
    /// Default port is [0] if ports is None/empty.
    /// Validates port types, positivity, existence in block, and uniqueness.
    pub fn new(block: BlockRef, ports: Option<Vec<Port>>) -> Self {
        let ports = match ports {
            Some(p) if !p.is_empty() => p,
            _ => vec![Port::Index(0)],
        };

        // Type validation + existence check against block
        {
            let blk = block.borrow();
            for p in &ports {
                match p {
                    Port::Index(_i) => {
                        // int ports are always valid (Python: isinstance(p, int))
                    }
                    Port::Name(name) => {
                        // Key existence validation for string ports
                        if !blk.inputs.contains_str(name) && !blk.outputs.contains_str(name) {
                            crate::utils::sink::warn(&format!("[WARNING] Port alias '{}' not defined for Block — ignored", name));
                        }
                    }
                }
            }
        }

        // Port uniqueness validation
        let mut seen = HashSet::new();
        for p in &ports {
            if !seen.insert(p.clone()) {
                crate::utils::sink::warn(&format!("[WARNING] Duplicate port {:?} — ignoring duplicate", p));
            }
        }

        Self {
            block,
            ports,
            _input_indices: FastCell::new(None),
            _output_indices: FastCell::new(None),
        }
    }

    /// Number of ports managed by this PortReference.
    pub fn len(&self) -> usize {
        self.ports.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ports.is_empty()
    }

    /// Resolve input indices (string aliases → integers), resizing the input
    /// register and caching the result. Fallible: a missing alias is a user
    /// configuration error surfaced as `SimError`, not a panic. Called at graph
    /// assembly (`Connection::resolve_ports`) so the hot path reads the cache.
    pub fn resolve_input_indices(&self) -> Result<Vec<usize>, SimError> {
        {
            let cache = self._input_indices.borrow();
            if let Some(ref indices) = *cache {
                return Ok(indices.clone());
            }
        }

        let indices: Vec<usize> = {
            let blk = self.block.borrow();
            let mut out = Vec::with_capacity(self.ports.len());
            for p in &self.ports {
                out.push(match p {
                    Port::Index(i) => *i,
                    Port::Name(name) => blk.inputs._map(name)
                        .ok_or_else(|| SimError::InputPortAlias(name.clone()))?,
                });
            }
            out
        };

        let max_idx = *indices.iter().max().unwrap();
        self.block.borrow_mut().inputs.resize(max_idx + 1);
        *self._input_indices.borrow_mut() = Some(indices.clone());
        Ok(indices)
    }

    /// Resolve output indices (string aliases → integers), resizing the output
    /// register and caching the result. See `resolve_input_indices`.
    pub fn resolve_output_indices(&self) -> Result<Vec<usize>, SimError> {
        {
            let cache = self._output_indices.borrow();
            if let Some(ref indices) = *cache {
                return Ok(indices.clone());
            }
        }

        let indices: Vec<usize> = {
            let blk = self.block.borrow();
            let mut out = Vec::with_capacity(self.ports.len());
            for p in &self.ports {
                out.push(match p {
                    Port::Index(i) => *i,
                    Port::Name(name) => blk.outputs._map(name)
                        .ok_or_else(|| SimError::OutputPortAlias(name.clone()))?,
                });
            }
            out
        };

        let max_idx = *indices.iter().max().unwrap();
        self.block.borrow_mut().outputs.resize(max_idx + 1);
        *self._output_indices.borrow_mut() = Some(indices.clone());
        Ok(indices)
    }

    /// Infallible cached accessor for the hot path / tests. Port aliases are
    /// validated at graph assembly (`Connection::resolve_ports`), so by the
    /// time the simulation loop calls this the cache is populated and the
    /// resolution branch (which could fail on a bad alias) is not reached.
    pub fn _get_input_indices(&self) -> Vec<usize> {
        self.resolve_input_indices()
            .expect("input port aliases must be validated at graph assembly")
    }

    /// Infallible cached accessor for the hot path / tests. See `_get_input_indices`.
    pub fn _get_output_indices(&self) -> Vec<usize> {
        self.resolve_output_indices()
            .expect("output port aliases must be validated at graph assembly")
    }

    /// Validate that all ports exist as input ports of the block.
    pub fn _validate_input_ports(&self) {
        let blk = self.block.borrow();
        for p in &self.ports {
            if let Port::Name(name) = p {
                if !blk.inputs.contains_str(name) {
                    crate::utils::sink::warn(&format!("[WARNING] Input port '{}' not defined for Block!", name));
                }
            }
        }
    }

    /// Validate that all ports exist as output ports of the block.
    pub fn _validate_output_ports(&self) {
        let blk = self.block.borrow();
        for p in &self.ports {
            if let Port::Name(name) = p {
                if !blk.outputs.contains_str(name) {
                    crate::utils::sink::warn(&format!("[WARNING] Output port '{}' not defined for Block!", name));
                }
            }
        }
    }

    /// Transfer data from self (outputs) to other (inputs).
    ///
    /// Mirrors Python `PortReference.to(other)`:
    /// `other.block.inputs._data[dst_indices] = self.block.outputs._data[src_indices]`
    ///
    /// Uses cached integer indices for fast transfer.
    pub fn to(&self, other: &PortReference) {
        let same_block = Rc::ptr_eq(&self.block, &other.block);

        // Ensure indices are resolved (lazy init)
        self._ensure_output_indices();
        other._ensure_input_indices();

        if same_block {
            // Self-connection: read/write on the same block.  For SISO we skip
            // the SmallVec buffer entirely; for MIMO we read into a scratch
            // buffer because we can't hold both an immutable and mutable borrow.
            let src_cache = self._output_indices.borrow();
            let dst_cache = other._input_indices.borrow();
            let src_idx = src_cache.as_ref().unwrap();
            let dst_idx = dst_cache.as_ref().unwrap();

            if src_idx.len() == 1 && dst_idx.len() == 1 {
                // SISO self-connection: one borrow to read, one to write.
                let val = self.block.borrow().outputs._data[src_idx[0]];
                self.block.borrow_mut().inputs._data[dst_idx[0]] = val;
                return;
            }

            let values: smallvec::SmallVec<[f64; 8]> = {
                let blk = self.block.borrow();
                src_idx.iter().map(|&i| blk.outputs._data[i]).collect()
            };
            let blk = self.block.borrow_mut();
            for (&di, &val) in dst_idx.iter().zip(values.iter()) {
                blk.inputs._data[di] = val;
            }
            return;
        }

        let src_blk = self.block.borrow();
        let dst_blk = other.block.borrow_mut();
        let src_cache = self._output_indices.borrow();
        let dst_cache = other._input_indices.borrow();
        let src_idx = src_cache.as_ref().unwrap();
        let dst_idx = dst_cache.as_ref().unwrap();

        // SISO cross-block fast path: skip iterator/zip machinery.  This is
        // >80% of connections in typical block diagrams, and the loop
        // structure below compiles to more setup than a plain assignment.
        if src_idx.len() == 1 && dst_idx.len() == 1 {
            dst_blk.inputs._data[dst_idx[0]] = src_blk.outputs._data[src_idx[0]];
            return;
        }

        for (&si, &di) in src_idx.iter().zip(dst_idx.iter()) {
            dst_blk.inputs._data[di] = src_blk.outputs._data[si];
        }
    }

    /// Ensure output indices are resolved and cached (no clone).
    fn _ensure_output_indices(&self) {
        let needs_init = self._output_indices.borrow().is_none();
        if needs_init {
            let _ = self._get_output_indices(); // populates cache
        }
    }

    /// Ensure input indices are resolved and cached (no clone).
    fn _ensure_input_indices(&self) {
        let needs_init = self._input_indices.borrow().is_none();
        if needs_init {
            let _ = self._get_input_indices(); // populates cache
        }
    }

    /// Return the input values of the block at specified ports.
    ///
    /// Mirrors Python `get_inputs()`.
    pub fn get_inputs(&self) -> Vec<f64> {
        let indices = self._get_input_indices();
        let blk = self.block.borrow();
        indices.iter().map(|&i| blk.inputs._data[i]).collect()
    }

    /// Set the block inputs with values at specified ports.
    ///
    /// Mirrors Python `set_inputs(vals)`.
    pub fn set_inputs(&self, vals: &[f64]) {
        // Borrow the cached indices directly (the index- and data-registers are
        // distinct cells), avoiding a per-call clone — this is hot in the
        // algebraic-loop solver (ConnectionBooster::set every iteration).
        self._ensure_input_indices();
        let cache = self._input_indices.borrow();
        let indices = cache.as_ref().unwrap();
        let blk = self.block.borrow_mut();
        for (&idx, &val) in indices.iter().zip(vals.iter()) {
            blk.inputs._data[idx] = val;
        }
    }

    /// Return the output values of the block at specified ports.
    ///
    /// Mirrors Python `get_outputs()`.
    pub fn get_outputs(&self) -> Vec<f64> {
        let indices = self._get_output_indices();
        let blk = self.block.borrow();
        indices.iter().map(|&i| blk.outputs._data[i]).collect()
    }

    /// Allocation-free variant: writes output values into the caller's buffer.
    /// Used in the algebraic-loop hot path (ConnectionBooster) where the same
    /// buffer is reused every iteration.
    pub fn get_outputs_into(&self, out: &mut Vec<f64>) {
        self._ensure_output_indices();
        let cache = self._output_indices.borrow();
        let indices = cache.as_ref().unwrap();
        let blk = self.block.borrow();
        out.clear();
        out.reserve(indices.len());
        for &i in indices {
            out.push(blk.outputs._data[i]);
        }
    }

    /// Set the block outputs with values at specified ports.
    ///
    /// Mirrors Python `set_outputs(vals)`.
    pub fn set_outputs(&self, vals: &[f64]) {
        self._ensure_output_indices();
        let cache = self._output_indices.borrow();
        let indices = cache.as_ref().unwrap();
        let blk = self.block.borrow_mut();
        for (&idx, &val) in indices.iter().zip(vals.iter()) {
            blk.outputs._data[idx] = val;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::block::new_block_ref;

    #[test]
    fn test_default_port() {
        let block = new_block_ref(None, None);
        let pr = PortReference::new(block, None);
        assert_eq!(pr.len(), 1);
        assert_eq!(pr.ports, vec![Port::Index(0)]);
    }

    #[test]
    fn test_multiple_ports() {
        let block = new_block_ref(None, None);
        let pr = PortReference::new(block, Some(vec![
            Port::Index(0), Port::Index(1), Port::Index(2)
        ]));
        assert_eq!(pr.len(), 3);
    }

    #[test]
    fn test_duplicate_ports_warning() {
        // Duplicate ports now emit a warning instead of panicking
        let block = new_block_ref(None, None);
        let pr = PortReference::new(block, Some(vec![Port::Index(1), Port::Index(1)]));
        // Should still create the PortReference (with duplicates preserved)
        assert!(!pr.is_empty());
    }

    #[test]
    fn test_named_ports() {
        let mut in_labels = std::collections::HashMap::new();
        in_labels.insert("signal".to_string(), 0);
        let block = new_block_ref(Some(in_labels), None);
        let pr = PortReference::new(block, Some(vec![Port::Name("signal".to_string())]));
        assert_eq!(pr.len(), 1);
    }

    #[test]
    fn test_invalid_named_port_warns() {
        // Invalid named ports now emit a warning instead of panicking
        let block = new_block_ref(None, None);
        let pr = PortReference::new(block, Some(vec![Port::Name("nonexistent".to_string())]));
        assert_eq!(pr.len(), 1);
    }

    #[test]
    fn test_to_transfer() {
        let src_block = new_block_ref(None, None);
        let dst_block = new_block_ref(None, None);

        // Set source output
        src_block.borrow_mut().outputs.set_single(0, 42.0);

        let src_pr = PortReference::new(src_block.clone(), None);
        let dst_pr = PortReference::new(dst_block.clone(), None);

        // Transfer: src.outputs[0] -> dst.inputs[0]
        src_pr.to(&dst_pr);

        assert_eq!(dst_block.borrow().inputs.get_single(0), 42.0);
    }

    #[test]
    fn test_to_transfer_mimo() {
        let src_block = new_block_ref(None, None);
        let dst_block = new_block_ref(None, None);

        // Set source outputs
        src_block.borrow_mut().outputs.set_single(0, 10.0);
        src_block.borrow_mut().outputs.set_single(1, 20.0);
        src_block.borrow_mut().outputs.set_single(2, 30.0);

        let src_pr = PortReference::new(src_block.clone(), Some(vec![
            Port::Index(0), Port::Index(1), Port::Index(2)
        ]));
        let dst_pr = PortReference::new(dst_block.clone(), Some(vec![
            Port::Index(0), Port::Index(1), Port::Index(2)
        ]));

        src_pr.to(&dst_pr);

        let dst = dst_block.borrow();
        assert_eq!(dst.inputs.get_single(0), 10.0);
        assert_eq!(dst.inputs.get_single(1), 20.0);
        assert_eq!(dst.inputs.get_single(2), 30.0);
    }

    #[test]
    fn test_self_connection() {
        let block = new_block_ref(None, None);

        // Set output
        block.borrow_mut().outputs.set_single(0, 99.0);

        let src_pr = PortReference::new(block.clone(), Some(vec![Port::Index(0)]));
        let dst_pr = PortReference::new(block.clone(), Some(vec![Port::Index(0)]));

        // Self-connection: outputs[0] -> inputs[0]
        src_pr.to(&dst_pr);

        assert_eq!(block.borrow().inputs.get_single(0), 99.0);
    }

    #[test]
    fn test_get_set_inputs() {
        let block = new_block_ref(None, None);
        let pr = PortReference::new(block.clone(), Some(vec![
            Port::Index(0), Port::Index(1)
        ]));

        pr.set_inputs(&[5.0, 10.0]);
        assert_eq!(pr.get_inputs(), vec![5.0, 10.0]);

        // Verify directly on block
        let blk = block.borrow();
        assert_eq!(blk.inputs.get_single(0), 5.0);
        assert_eq!(blk.inputs.get_single(1), 10.0);
    }

    #[test]
    fn test_get_set_outputs() {
        let block = new_block_ref(None, None);
        let pr = PortReference::new(block.clone(), Some(vec![
            Port::Index(0), Port::Index(1)
        ]));

        pr.set_outputs(&[7.0, 14.0]);
        assert_eq!(pr.get_outputs(), vec![7.0, 14.0]);
    }

    #[test]
    fn test_auto_resize_on_index_resolve() {
        let block = new_block_ref(None, None);
        assert_eq!(block.borrow().inputs.len(), 1);

        let pr = PortReference::new(block.clone(), Some(vec![
            Port::Index(0), Port::Index(5)
        ]));

        // Resolving input indices should resize inputs to accommodate index 5
        let _indices = pr._get_input_indices();
        assert!(block.borrow().inputs.len() >= 6);
    }

    #[test]
    fn test_cached_indices() {
        let block = new_block_ref(None, None);
        let pr = PortReference::new(block, Some(vec![
            Port::Index(0), Port::Index(1)
        ]));

        let idx1 = pr._get_input_indices();
        let idx2 = pr._get_input_indices();
        assert_eq!(idx1, idx2);
        assert_eq!(idx1, vec![0, 1]);
    }
}
