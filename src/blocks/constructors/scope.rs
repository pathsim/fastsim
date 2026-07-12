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

    // If sampling_period is set, use Schedule event for discrete sampling
    let sample_flag = Rc::new(FastCell::new(false));
    let has_sampling_period = sampling_period.is_some();

    if let Some(sp) = sampling_period {
        b.data_f64.insert("sampling_period".to_string(), sp);

        use crate::events::schedule::Schedule;
        let flag = sample_flag.clone();
        let evt = Schedule::new(
            t_wait, None, sp,
            Some(Box::new(move |_t| { *flag.borrow_mut() = true; })),
            crate::constants::TOLERANCE,
        );
        b.events.push(Rc::new(FastCell::new(evt)));
    }

    b.reset_fn = Some(Box::new(|blk| {
        blk.inputs.reset();
        blk.outputs.reset();
        if let Some(v) = blk.data_vec.get_mut("recording_time") { v.clear() }
        if let Some(v) = blk.data_vec2.get_mut("recording_data") { v.clear() }
        if let Some(idx) = blk.data_f64.get_mut("_read_idx") { *idx = 0.0 }
    }));

    let sample_flag_fn = sample_flag.clone();
    b.sample_fn = Some(Box::new(move |blk, t, _dt| {
        let t_wait = blk.data_f64.get("t_wait").copied().unwrap_or(0.0);

        // Determine if we should sample
        let should_sample = if has_sampling_period {
            let flag = *sample_flag_fn.borrow();
            if flag { *sample_flag_fn.borrow_mut() = false; true } else { false }
        } else {
            t >= t_wait
        };

        if !should_sample { return; }

        // Skip duplicate timestamps
        if let Some(times) = blk.data_vec.get("recording_time") {
            if let Some(&last) = times.last() {
                if last == t { return; }
            }
        }

        let data = blk.inputs._data.clone();
        if let Some(v) = blk.data_vec.get_mut("recording_time") { v.push(t) }
        if let Some(v) = blk.data_vec2.get_mut("recording_data") { v.push(data) }
    }));

    Rc::new(FastCell::new(b))
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
