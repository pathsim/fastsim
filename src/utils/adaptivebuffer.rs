//! Rolling time buffer with linear interpolation, the signal history of the
//! continuous `Delay` (pathsim `utils/adaptivebuffer.py`).

use std::collections::VecDeque;

use crate::constants::DELAY_BUFFER_SAFETY;
use crate::utils::register::Register;

/// Samples `(t, u)` of a signal with any number of channels, looked up at
/// `t - delay` by linear interpolation.
pub struct AdaptiveBuffer {
    delay: f64,
    times: VecDeque<f64>,
    /// channel values of the samples back to back, `width` per sample
    values: VecDeque<f64>,
    width: usize,
}

impl AdaptiveBuffer {
    pub fn new(delay: f64) -> Self {
        Self { delay, times: VecDeque::new(), values: VecDeque::new(), width: 0 }
    }

    /// Add a sample, then drop the ones older than the delay, keeping
    /// `DELAY_BUFFER_SAFETY` of them for the interpolation. A sample of another
    /// width restarts the history.
    pub fn add(&mut self, t: f64, value: &[f64]) {
        if value.len() != self.width {
            self.clear();
            self.width = value.len();
        }
        self.times.push_back(t);
        self.values.extend(value);
        while self.times.len() > DELAY_BUFFER_SAFETY && t - self.times[DELAY_BUFFER_SAFETY] > self.delay {
            self.times.pop_front();
            self.values.drain(..self.width);
        }
    }

    /// Write the signal at `t - delay` into `out`: zero up to the first sample,
    /// the last sample after the last one, linear in between.
    pub fn get(&self, t: f64, out: &mut Register) {
        let target = t - self.delay;
        let w = self.width;
        let n = self.times.len();
        if n == 0 || target <= self.times[0] {
            out.update_from_array(&[0.0]);
            return;
        }
        if target >= self.times[n - 1] {
            for c in 0..w { out.set_single(c, self.values[(n - 1) * w + c]); }
            return;
        }
        let i = self.times.partition_point(|&s| s < target);
        let (t0, t1) = (self.times[i - 1], self.times[i]);
        let frac = (target - t0) / (t1 - t0);
        for c in 0..w {
            let (v0, v1) = (self.values[(i - 1) * w + c], self.values[i * w + c]);
            out.set_single(c, v0 + frac * (v1 - v0));
        }
    }

    pub fn clear(&mut self) {
        self.times.clear();
        self.values.clear();
    }
}
