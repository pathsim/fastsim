// Register: port value container backed by Vec<f64>
// Ported 1:1 from pathsim/utils/register.py

use std::collections::HashMap;

/// Port value container backed by `Vec<f64>`.
///
/// Basic functionality is similar to a `dict` but with some additional methods
/// and implemented as a contiguous array for fast data transfer.
///
/// Values can be added dynamically and the size of the register doesn't have
/// to be specified. It also implements methods to interact with arrays and
/// to streamline convergence checks.
pub struct Register {
    /// Internal data array holding port values (equivalent to np.ndarray)
    pub _data: Vec<f64>,
    /// String aliases for integer ports
    pub _mapping: HashMap<String, usize>,
}

impl Register {
    /// Create a new Register.
    ///
    /// Mirrors Python: `Register(size=None, mapping=None)`
    pub fn new(size: Option<usize>, mapping: Option<HashMap<String, usize>>) -> Self {
        Self {
            _data: vec![0.0; size.unwrap_or(1)],
            _mapping: mapping.unwrap_or_default(),
        }
    }

    /// Map string keys to integers defined in '_mapping'.
    ///
    /// Mirrors Python: `self._mapping.get(key, key)`
    /// Returns the mapped index if found, or None if the string is not mapped.
    pub fn _map(&self, key: &str) -> Option<usize> {
        self._mapping.get(key).copied()
    }

    pub fn len(&self) -> usize {
        self._data.len()
    }

    pub fn is_empty(&self) -> bool {
        self._data.is_empty()
    }

    /// Get value(s) — mirrors Python `__getitem__`.
    ///
    /// For Int: returns single f64 (0.0 if out of bounds).
    /// For Name: maps via _mapping, returns 0.0 if unmapped.
    /// For Slice/List: returns Vec<f64>.
    pub fn get_single(&self, index: usize) -> f64 {
        if index >= self._data.len() {
            0.0
        } else {
            self._data[index]
        }
    }

    pub fn get_by_name(&self, name: &str) -> f64 {
        match self._map(name) {
            Some(idx) => self.get_single(idx),
            None => 0.0,
        }
    }

    pub fn get_slice(&self, start: usize, stop: usize, step: usize) -> Vec<f64> {
        (start..stop)
            .step_by(step)
            .map(|i| self.get_single(i))
            .collect()
    }

    pub fn get_indices(&self, indices: &[usize]) -> Vec<f64> {
        indices.iter().map(|&i| self.get_single(i)).collect()
    }

    /// Set value at integer index with auto-resize.
    /// Mirrors Python `__setitem__` for int keys.
    pub fn set_single(&mut self, index: usize, value: f64) {
        self.resize(index + 1);
        self._data[index] = value;
    }

    /// Set value at named port.
    pub fn set_by_name(&mut self, name: &str, value: f64) {
        if let Some(idx) = self._map(name) {
            self.set_single(idx, value);
        }
    }

    /// Set values at slice indices.
    pub fn set_slice(&mut self, start: usize, stop: usize, step: usize, values: &[f64]) {
        let indices: Vec<usize> = (start..stop).step_by(step).collect();
        if let Some(&max_idx) = indices.last() {
            self.resize(max_idx + 1);
        }
        for (idx, &val) in indices.iter().zip(values.iter()) {
            self._data[*idx] = val;
        }
    }

    /// Set values at specific indices (fancy indexing).
    pub fn set_indices(&mut self, indices: &[usize], values: &[f64]) {
        if let Some(&max_idx) = indices.iter().max() {
            self.resize(max_idx + 1);
        }
        for (&idx, &val) in indices.iter().zip(values.iter()) {
            self._data[idx] = val;
        }
    }

    /// Resize the internal data array to accommodate more entries.
    ///
    /// Mirrors Python: creates new zero-filled array and copies old data.
    /// In Rust, Vec::resize achieves the same effect.
    pub fn resize(&mut self, size: usize) {
        if size > self._data.len() {
            self._data.resize(size, 0.0);
        }
    }

    /// Set all stored values to zero.
    pub fn reset(&mut self) {
        self._data.fill(0.0);
    }

    /// Returns a copy of the internal array.
    pub fn to_array(&self) -> Vec<f64> {
        self._data.clone()
    }

    /// Update the register values from a slice in place.
    ///
    /// Mirrors Python `update_from_array` for array-like inputs.
    pub fn update_from_array(&mut self, arr: &[f64]) {
        let n = arr.len();
        if n > self._data.len() {
            self.resize(n);
        }
        self._data[..n].copy_from_slice(arr);
    }

    /// Update from a single scalar (sets port 0).
    ///
    /// Mirrors Python `update_from_array` for scalar inputs.
    pub fn update_from_scalar(&mut self, val: f64) {
        self._data[0] = val;
    }

    /// Check if a key is in mapping or is a valid integer index.
    ///
    /// Mirrors Python `__contains__`: `key in self._mapping or isinstance(key, int)`
    /// For strings: checks mapping. For int: always True (matches Python behavior).
    pub fn contains_str(&self, key: &str) -> bool {
        self._mapping.contains_key(key)
    }

    pub fn contains_int(&self, _index: usize) -> bool {
        // Python: isinstance(key, int) always returns True
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_size() {
        let reg = Register::new(None, None);
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.get_single(0), 0.0);
    }

    #[test]
    fn test_with_size() {
        let reg = Register::new(Some(4), None);
        assert_eq!(reg.len(), 4);
        for i in 0..4 {
            assert_eq!(reg.get_single(i), 0.0);
        }
    }

    #[test]
    fn test_set_get() {
        let mut reg = Register::new(None, None);
        reg.set_single(0, 3.2);
        assert_eq!(reg.get_single(0), 3.2);
    }

    #[test]
    fn test_auto_resize() {
        let mut reg = Register::new(None, None);
        assert_eq!(reg.len(), 1);
        reg.set_single(5, 42.0);
        assert_eq!(reg.len(), 6);
        assert_eq!(reg.get_single(5), 42.0);
        assert_eq!(reg.get_single(3), 0.0);
    }

    #[test]
    fn test_out_of_bounds_returns_zero() {
        let reg = Register::new(Some(2), None);
        assert_eq!(reg.get_single(100), 0.0);
    }

    #[test]
    fn test_named_ports() {
        let mut mapping = HashMap::new();
        mapping.insert("voltage".to_string(), 0);
        mapping.insert("current".to_string(), 1);
        let mut reg = Register::new(Some(2), Some(mapping));

        reg.set_by_name("voltage", 5.0);
        reg.set_by_name("current", 2.5);

        assert_eq!(reg.get_by_name("voltage"), 5.0);
        assert_eq!(reg.get_by_name("current"), 2.5);
        assert_eq!(reg.get_by_name("nonexistent"), 0.0);
    }

    #[test]
    fn test_reset() {
        let mut reg = Register::new(Some(3), None);
        reg.set_single(0, 1.0);
        reg.set_single(1, 2.0);
        reg.set_single(2, 3.0);
        reg.reset();
        for i in 0..3 {
            assert_eq!(reg.get_single(i), 0.0);
        }
    }

    #[test]
    fn test_to_array() {
        let mut reg = Register::new(Some(3), None);
        reg.set_single(0, 1.0);
        reg.set_single(1, 2.0);
        reg.set_single(2, 3.0);
        assert_eq!(reg.to_array(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_update_from_array() {
        let mut reg = Register::new(Some(2), None);
        reg.update_from_array(&[10.0, 20.0, 30.0]);
        assert_eq!(reg.len(), 3);
        assert_eq!(reg.get_single(0), 10.0);
        assert_eq!(reg.get_single(1), 20.0);
        assert_eq!(reg.get_single(2), 30.0);
    }

    #[test]
    fn test_update_from_scalar() {
        let mut reg = Register::new(None, None);
        reg.update_from_scalar(7.5);
        assert_eq!(reg.get_single(0), 7.5);
    }

    #[test]
    fn test_contains() {
        let mut mapping = HashMap::new();
        mapping.insert("x".to_string(), 0);
        let reg = Register::new(Some(3), Some(mapping));

        assert!(reg.contains_str("x"));
        assert!(!reg.contains_str("y"));
        // Python: isinstance(key, int) is always True
        assert!(reg.contains_int(999));
    }

    #[test]
    fn test_slice_access() {
        let mut reg = Register::new(Some(5), None);
        for i in 0..5 {
            reg.set_single(i, (i + 1) as f64 * 10.0);
        }
        // get_slice(1, 4, 1) -> [20, 30, 40]
        assert_eq!(reg.get_slice(1, 4, 1), vec![20.0, 30.0, 40.0]);
        // get_slice(0, 5, 2) -> [10, 30, 50]
        assert_eq!(reg.get_slice(0, 5, 2), vec![10.0, 30.0, 50.0]);
    }

    #[test]
    fn test_fancy_indexing() {
        let mut reg = Register::new(Some(5), None);
        for i in 0..5 {
            reg.set_single(i, (i + 1) as f64 * 10.0);
        }
        // get_indices([0, 2, 4]) -> [10, 30, 50]
        assert_eq!(reg.get_indices(&[0, 2, 4]), vec![10.0, 30.0, 50.0]);
    }

    #[test]
    fn test_set_indices() {
        let mut reg = Register::new(Some(5), None);
        reg.set_indices(&[0, 2, 4], &[100.0, 300.0, 500.0]);
        assert_eq!(reg.get_single(0), 100.0);
        assert_eq!(reg.get_single(1), 0.0);
        assert_eq!(reg.get_single(2), 300.0);
        assert_eq!(reg.get_single(3), 0.0);
        assert_eq!(reg.get_single(4), 500.0);
    }

    #[test]
    fn test_set_slice() {
        let mut reg = Register::new(Some(5), None);
        reg.set_slice(1, 4, 1, &[10.0, 20.0, 30.0]);
        assert_eq!(reg.get_single(0), 0.0);
        assert_eq!(reg.get_single(1), 10.0);
        assert_eq!(reg.get_single(2), 20.0);
        assert_eq!(reg.get_single(3), 30.0);
    }

    #[test]
    fn test_map_key() {
        let mut mapping = HashMap::new();
        mapping.insert("x".to_string(), 3);
        let reg = Register::new(Some(5), Some(mapping));

        assert_eq!(reg._map("x"), Some(3));
        assert_eq!(reg._map("y"), None);
    }
}
