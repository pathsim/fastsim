// Scope block constructor + companion reader helper.
//
// Scope is a sink: it records timestamped input samples at fixed
// `sampling_period` intervals (or every timestep if None).  `scope_read`
// returns the recorded `(times, channels)` tuple for post-processing.

use std::collections::HashMap;
use std::rc::Rc;

use crate::blocks::block::{Block, BlockRef, BlockRole};
use crate::utils::fastcell::FastCell;

// ======================================================================================
// Scope: recording block — overrides len, reset, sample
// ======================================================================================

/// Scope: records input signals over time for later retrieval
pub fn scope(sampling_period: Option<f64>, t_wait: f64, labels: Vec<String>) -> BlockRef {
    let mut b = Block::new(None, Some(HashMap::new()));
    b.type_name = "Scope";
    b.role = BlockRole { is_dyn: false, is_src: false, is_rec: true };
    b.data_f64.insert("t_wait".to_string(), t_wait);
    if !labels.is_empty() {
        b.data_strings.insert("labels".to_string(), labels);
    }
    b.data_vec.insert("recording_time".to_string(), Vec::new());
    b.data_vec2.insert("recording_data".to_string(), Vec::new());
    // Cursor for incremental reads: index of the first not-yet-read sample.
    b.data_f64.insert("_read_idx".to_string(), 0.0);

    b.len_fn = Some(Box::new(|_| 0));

    let has_sampling_period = sampling_period.is_some();
    if let Some(sp) = sampling_period {
        b.data_f64.insert("sampling_period".to_string(), sp);
    }

    b.reset_fn = Some(Box::new(|blk| {
        blk.inputs.reset();
        blk.outputs.reset();
        if let Some(v) = blk.data_vec.get_mut("recording_time") { v.clear() }
        if let Some(v) = blk.data_vec2.get_mut("recording_data") { v.clear() }
        if let Some(idx) = blk.data_f64.get_mut("_read_idx") { *idx = 0.0 }
    }));

    // Per-timestep recording; only active without a sampling period. With one,
    // the Schedule event below records in its action instead, stamped with the
    // scheduled time rather than the end of the enclosing timestep.
    b.sample_fn = Some(Box::new(move |blk, t, _dt| {
        if has_sampling_period { return; }
        let t_wait = blk.data_f64.get("t_wait").copied().unwrap_or(0.0);
        if t >= t_wait {
            record(blk, t);
        }
    }));

    let blk_ref: BlockRef = Rc::new(FastCell::new(b));

    // Event-based sampling records directly in the Schedule action so the
    // timestamps are the scheduled times and no tick can be lost (a flag
    // deferred to the end of the timestep collapses two ticks that land in
    // one step into a single sample). Upstream: pathsim#249. The action holds
    // the block, forming the (documented) closure/event Rc cycle used
    // throughout the constructors.
    if let Some(sp) = sampling_period {
        use crate::events::schedule::Schedule;
        let blk_evt = blk_ref.clone();
        let evt = Schedule::new(
            t_wait, None, sp,
            Some(Box::new(move |t| record(blk_evt.borrow_mut(), t))),
            crate::constants::TOLERANCE,
        );
        blk_ref.borrow_mut().events.push(Rc::new(FastCell::new(evt)));
    }

    blk_ref
}

/// Append one sample of all inputs at time `t`, skipping duplicate timestamps
/// to keep the recording's time points unique.
fn record(blk: &mut Block, t: f64) {
    if let Some(times) = blk.data_vec.get("recording_time") {
        if times.last() == Some(&t) {
            return;
        }
    }
    let data = blk.inputs._data.clone();
    if let Some(v) = blk.data_vec.get_mut("recording_time") { v.push(t) }
    if let Some(v) = blk.data_vec2.get_mut("recording_data") { v.push(data) }
}

/// Read recorded data from a Scope block.
pub fn scope_read(block: &Block) -> (Vec<f64>, Vec<Vec<f64>>) {
    let times = block.data_vec.get("recording_time").cloned().unwrap_or_default();
    let data = block.data_vec2.get("recording_data").cloned().unwrap_or_default();
    (times, data)
}

/// Read only the samples recorded since the last incremental read, advancing
/// the read cursor to the current end. Used for live streaming so each tick
/// transfers only new data instead of the full (growing) history. The cursor
/// is reset to 0 by the block's `reset`.
pub fn scope_read_incremental(block: &mut Block) -> (Vec<f64>, Vec<Vec<f64>>) {
    let total = block.data_vec.get("recording_time").map(|v| v.len()).unwrap_or(0);
    let start = block.data_f64.get("_read_idx").copied().unwrap_or(0.0) as usize;
    // Defensive: a reset can shrink the buffer below a stale cursor.
    let start = start.min(total);

    let times = block.data_vec.get("recording_time")
        .map(|v| v[start..].to_vec()).unwrap_or_default();
    let data = block.data_vec2.get("recording_data")
        .map(|v| v[start..].to_vec()).unwrap_or_default();

    block.data_f64.insert("_read_idx".to_string(), total as f64);
    (times, data)
}
