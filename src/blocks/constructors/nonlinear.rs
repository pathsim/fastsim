// Nonlinear / event-effect block constructors:
// Comparator, Switch, Relay, Counter family, RateLimiter, Backlash, Deadband.

use std::cell::RefCell;
use std::rc::Rc;

use crate::blocks::block::{Block, BlockRef, BlockRole};
use crate::blocks::blockops::{
    mem_read_alg_graph, region_graph_xu, DirSpec, EventKindSpec, EventSpec, MemSpec,
    MemTarget,
};

/// RateLimiter derivative: clamp(f_max*(u - x), -rate, rate).
fn rate_limiter_dyn<B: Builder>(b: &B, rate: B::N, f_max: B::N, x0: B::N, u0: B::N) -> B::N {
    let v = b.mul(f_max, b.sub(u0, x0));
    b.min(b.max(v, b.neg(rate)), rate)
}

/// Backlash derivative: f_max*((u-x) - clamp(u-x, -hw, hw)).
fn backlash_dyn<B: Builder>(b: &B, hw: B::N, f_max: B::N, x0: B::N, u0: B::N) -> B::N {
    let diff = b.sub(u0, x0);
    let clamped = b.min(b.max(diff, b.neg(hw)), hw);
    b.mul(f_max, b.sub(diff, clamped))
}
use crate::ssa::build::{Builder, F64Builder, GraphBuilder};
use crate::ssa::graph::{Graph, InputSignature};
use crate::solvers::solver::Solver;
use crate::utils::fastcell::FastCell;

// ======================================================================================
// Comparator: y = max(span) if u >= threshold else min(span)
// ======================================================================================

/// Comparator: y = span.1 if u >= threshold, else span.0
pub fn comparator(threshold: f64, span: (f64, f64)) -> BlockRef {
    let mut b = Block::default_block();
    b.type_name = "Comparator";
    // Algebraic feedthrough: the output is the instantaneous select
    // `(u >= threshold) ? hi : lo` (see `update_fn` / the `ops.alg` region below).
    // The ZeroCrossing registered further down is only a solver step-size hint,
    // not what produces the output, so the block DOES participate in the
    // algebraic FPI loop (is_alg = true, derived-from-SSA consistent).
    b.role = BlockRole { is_dyn: false, is_src: false, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));

    b.update_fn = Some(Box::new(move |blk, _t| {
        let u = blk.inputs._data[0];
        blk.outputs._data[0] = if u >= threshold { span.1 } else { span.0 };
    }));

    // IR: purely algebraic select y = (u >= threshold) ? span.1 : span.0. The
    // runtime keeps its zero-crossing event as a solver step-size hint only.
    {
        fn build<B: Builder>(b: &B, u0: B::N, thr: B::N, hi: B::N, lo: B::N) -> B::N {
            b.select(b.ge(u0, thr), hi, lo)
        }
        let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", 1usize)])));
        {
            let mut g = cell.borrow_mut();
            g.n_params = 3;
            g.param_defaults = vec![threshold, span.1, span.0];
            g.param_names = vec!["threshold".into(), "value_high".into(), "value_low".into()];
        }
        let y = {
            let gb = GraphBuilder::new(&cell);
            build(&gb, gb.input(0), gb.param(0), gb.param(1), gb.param(2))
        };
        let mut g = cell.into_inner();
        g.outputs.push(y);
        b.set_alg("Comparator", g);
    }

    let blk_ref: BlockRef = Rc::new(FastCell::new(b));
    let blk_evt = blk_ref.clone();

    use crate::events::zerocrossing::ZeroCrossing;
    let evt = ZeroCrossing::new(
        move |_t| blk_evt.borrow().inputs._data[0] - threshold,
        None,
        crate::constants::EVT_TOLERANCE,
    );
    blk_ref.borrow_mut().events.push(Rc::new(FastCell::new(evt)));

    blk_ref
}

// ======================================================================================
// Switch: routes one of N inputs to output based on switch_state
// ======================================================================================

/// Switch: routes one of N inputs to the output based on switch_state index
pub fn switch(n_inputs: usize, initial_state: Option<usize>) -> BlockRef {
    let mut b = Block::default_block();
    b.type_name = "Switch";
    // Mirror current len_fn: closed (state >= 0) → algebraic feedthrough, open → no feedthrough.
    // Captured at construction time from initial_state (matches Graph assembly behaviour).
    b.role = BlockRole { is_dyn: false, is_src: false, is_rec: false };
    b.inputs.resize(n_inputs);

    // Store switch_state as f64 in data_f64 (-1 = None, >= 0 = active index)
    let state_val = initial_state.map(|s| s as f64).unwrap_or(-1.0);
    b.data_f64.insert("switch_state".to_string(), state_val);

    b.len_fn = Some(Box::new(|blk| {
        if blk.data_f64.get("switch_state").copied().unwrap_or(-1.0) >= 0.0 { 1 } else { 0 }
    }));

    b.update_fn = Some(Box::new(move |blk, _t| {
        let s = blk.data_f64.get("switch_state").copied().unwrap_or(-1.0);
        let y = if s >= 0.0 {
            let idx = s as usize;
            if idx < blk.inputs._data.len() { blk.inputs._data[idx] } else { 0.0 }
        } else { 0.0 };
        blk.outputs._data[0] = y;
    }));

    // IR: the routing index is a held `state` memory slot (no event; set
    // externally). alg = select-chain: y = (state==k) ? u[k] : ... : 0.
    {
        let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([
            (format!("mem{}", 0u32), 1usize),
            ("u".to_string(), n_inputs),
        ])));
        let y = {
            let gb = GraphBuilder::new(&cell);
            let state = gb.input(0);
            let u: Vec<u32> = (0..n_inputs as u32).map(|i| gb.input(1 + i)).collect();
            let mut acc = gb.cst(0.0);
            for k in (0..n_inputs).rev() {
                let hit = gb.eq(state, gb.cst(k as f64));
                acc = gb.select(hit, u[k], acc);
            }
            acc
        };
        let mut g = cell.into_inner();
        g.outputs.push(y);
        let memory = vec![MemSpec { name: "state".into(), init: vec![state_val] }];
        b.set_discrete("Switch", g, memory, vec![]);
    }

    Rc::new(FastCell::new(b))
}

// ======================================================================================
// Relay: hysteresis switching via thresholds
// ======================================================================================

/// Relay: hysteresis switch between value_up and value_down at two thresholds
pub fn relay(threshold_up: f64, threshold_down: f64, value_up: f64, value_down: f64) -> BlockRef {
    let output = Rc::new(FastCell::new(value_down));
    let out_upd = output.clone();
    let out_evt_up = output.clone();
    let out_evt_dn = output.clone();
    let out_reset = output.clone();

    let mut b = Block::default_block();
    b.type_name = "Relay";
    b.role = BlockRole { is_dyn: false, is_src: false, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));

    b.update_fn = Some(Box::new(move |blk, _t| {
        blk.outputs._data[0] = *out_upd.borrow();
    }));

    // IR (Memory + Event): one `out` slot. Two zero-crossing events:
    // u - threshold_up rising -> out = value_up; u - threshold_down falling ->
    // out = value_down. alg output y = out.
    {
        let guard = |thr: f64| {
            let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", 1usize)])));
            let y = {
                let gb = GraphBuilder::new(&cell);
                gb.sub(gb.input(0), gb.cst(thr))
            };
            let mut g = cell.into_inner();
            g.outputs.push(y);
            g
        };
        let effect = |val: f64| {
            let cell = RefCell::new(Graph::new(InputSignature::empty()));
            let y = {
                let gb = GraphBuilder::new(&cell);
                gb.cst(val)
            };
            let mut g = cell.into_inner();
            g.outputs.push(y);
            g
        };
        let memory = vec![MemSpec { name: "out".into(), init: vec![value_down] }];
        let events = vec![
            EventSpec {
                kind: EventKindSpec::ZeroCross { guard: guard(threshold_up), direction: DirSpec::Rising },
                effect: effect(value_up),
                targets: vec![MemTarget { slot: 0, offset: 0 }],
            },
            EventSpec {
                kind: EventKindSpec::ZeroCross { guard: guard(threshold_down), direction: DirSpec::Falling },
                effect: effect(value_down),
                targets: vec![MemTarget { slot: 0, offset: 0 }],
            },
        ];
        b.set_discrete("Relay", mem_read_alg_graph(0, 1), memory, events);
    }

    let blk_ref: BlockRef = Rc::new(FastCell::new(b));

    // Event closures read directly from the block's inputs (not a cached copy).
    // This ensures correct values after adaptive solver reverts.
    let blk_evt_up = blk_ref.clone();
    let blk_evt_dn = blk_ref.clone();

    use crate::events::zerocrossing::ZeroCrossing;
    let evt_up = ZeroCrossing::new_up(
        move |_t| blk_evt_up.borrow().inputs._data[0] - threshold_up,
        Some(Box::new(move |_t| { *out_evt_up.borrow_mut() = value_up; })),
        crate::constants::EVT_TOLERANCE,
    );
    let evt_dn = ZeroCrossing::new_down(
        move |_t| blk_evt_dn.borrow().inputs._data[0] - threshold_down,
        Some(Box::new(move |_t| { *out_evt_dn.borrow_mut() = value_down; })),
        crate::constants::EVT_TOLERANCE,
    );
    blk_ref.borrow_mut().events.push(Rc::new(FastCell::new(evt_up)));
    blk_ref.borrow_mut().events.push(Rc::new(FastCell::new(evt_dn)));

    let _blk_reset = blk_ref.clone();
    blk_ref.borrow_mut().reset_fn = Some(Box::new(move |blk| {
        blk.inputs.reset();
        blk.outputs.reset();
        *out_reset.borrow_mut() = value_down;
    }));

    blk_ref
}

// ======================================================================================
// Counter variants: count zero-crossings of input
// ======================================================================================

fn counter_with_direction(name: &'static str, start: f64, threshold: f64,
                          direction: crate::events::zerocrossing::CrossingDirection) -> BlockRef {
    use crate::events::zerocrossing::ZeroCrossing;

    let mut b = Block::default_block();
    b.type_name = name;
    b.role = BlockRole { is_dyn: false, is_src: false, is_rec: false };
    b.len_fn = Some(Box::new(|_| 0));

    let blk_ref: BlockRef = Rc::new(FastCell::new(b));
    let blk_evt = blk_ref.clone();

    let evt = ZeroCrossing::with_direction(direction,
        move |_t| blk_evt.borrow().inputs.get_single(0) - threshold,
        None, crate::constants::EVT_TOLERANCE);
    let evt_ref = Rc::new(FastCell::new(evt));
    let evt_for_update = evt_ref.clone();
    blk_ref.borrow_mut().events.push(evt_ref);

    blk_ref.borrow_mut().update_fn = Some(Box::new(move |blk, _t| {
        blk.outputs.set_single(0, start + evt_for_update.borrow().len() as f64);
    }));

    // IR (Memory + Event): a `count` slot incremented on each qualifying
    // zero-crossing of (u - threshold). alg output y = start + count.
    {
        use crate::events::zerocrossing::CrossingDirection;
        let dir = match direction {
            CrossingDirection::Up => DirSpec::Rising,
            CrossingDirection::Down => DirSpec::Falling,
            _ => DirSpec::Both,
        };
        let make = |f: &dyn Fn(&GraphBuilder) -> u32, slot_name: &str, n: usize| {
            let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([(slot_name.to_string(), n)])));
            let y = {
                let gb = GraphBuilder::new(&cell);
                f(&gb)
            };
            let mut g = cell.into_inner();
            g.outputs.push(y);
            g
        };
        let alg = make(&|gb| gb.add(gb.cst(start), gb.input(0)), "mem0", 1);
        let guard = make(&|gb| gb.sub(gb.input(0), gb.cst(threshold)), "u", 1);
        let effect = make(&|gb| gb.add(gb.input(0), gb.cst(1.0)), "mem0", 1);
        let memory = vec![MemSpec { name: "count".into(), init: vec![0.0] }];
        let events = vec![EventSpec {
            kind: EventKindSpec::ZeroCross { guard, direction: dir },
            effect,
            targets: vec![MemTarget { slot: 0, offset: 0 }],
        }];
        blk_ref.borrow_mut().set_discrete(name, alg, memory, events);
    }

    blk_ref
}

/// Counter: counts all zero-crossings of (u - threshold)
pub fn counter(start: f64, threshold: f64) -> BlockRef {
    counter_with_direction("Counter", start, threshold, crate::events::zerocrossing::CrossingDirection::Both)
}
/// CounterUp: counts upward zero-crossings of (u - threshold)
pub fn counter_up(start: f64, threshold: f64) -> BlockRef {
    counter_with_direction("CounterUp", start, threshold, crate::events::zerocrossing::CrossingDirection::Up)
}
/// CounterDown: counts downward zero-crossings of (u - threshold)
pub fn counter_down(start: f64, threshold: f64) -> BlockRef {
    counter_with_direction("CounterDown", start, threshold, crate::events::zerocrossing::CrossingDirection::Down)
}

// ======================================================================================
// RateLimiter: dx/dt = clip(f_max*(u-x), -rate, rate), y = x
// ======================================================================================

/// RateLimiter: dx/dt = clamp(f_max*(u - x), -rate, rate), y = x
pub fn rate_limiter(rate: f64, f_max: f64) -> BlockRef {
    let mut b = Block::default_block();
    b.type_name = "RateLimiter";
    b.role = BlockRole { is_dyn: true, is_src: false, is_rec: false };
    b.initial_value = Some(vec![0.0]);
    b.engine = Some(Solver::with_defaults(&[0.0]));
    b.len_fn = Some(Box::new(|_| 0));
    b.f_dyn = Some(Box::new(move |x, u, _t, out| {
        out.clear();
        out.push(rate_limiter_dyn(&F64Builder, rate, f_max, x[0], u[0]));
    }));
    b.f_alg = Some(Box::new(|x, _u, _t, out| {
        out.clear();
        out.push(x[0]);
    }));
    let names: &[&str] = &["rate", "f_max"];
    let alg = region_graph_xu(1, 1, vec![rate, f_max], names, |_gb, _p, x, _u, out| out.push(x[0]));
    let dyn_ = region_graph_xu(1, 1, vec![rate, f_max], names, |gb, p, x, u, out| {
        out.push(rate_limiter_dyn(gb, p[0], p[1], x[0], u[0]))
    });
    b.set_dynamic("RateLimiter", alg, dyn_);
    Rc::new(FastCell::new(b))
}

// ======================================================================================
// Backlash: dx/dt = f_max*((u-x) - clip(u-x, -w/2, w/2)), y = x
// ======================================================================================

/// Backlash: dead-zone nonlinearity, dx/dt = f_max*((u-x) - clamp(u-x, -w/2, w/2)), y = x
pub fn backlash(width: f64, f_max: f64) -> BlockRef {
    let hw = width / 2.0;
    let mut b = Block::default_block();
    b.type_name = "Backlash";
    b.role = BlockRole { is_dyn: true, is_src: false, is_rec: false };
    b.initial_value = Some(vec![0.0]);
    b.engine = Some(Solver::with_defaults(&[0.0]));
    b.len_fn = Some(Box::new(|_| 0));
    b.f_dyn = Some(Box::new(move |x, u, _t, out| {
        out.clear();
        out.push(backlash_dyn(&F64Builder, hw, f_max, x[0], u[0]));
    }));
    b.f_alg = Some(Box::new(|x, _u, _t, out| {
        out.clear();
        out.push(x[0]);
    }));
    let names: &[&str] = &["hw", "f_max"];
    let alg = region_graph_xu(1, 1, vec![hw, f_max], names, |_gb, _p, x, _u, out| out.push(x[0]));
    let dyn_ = region_graph_xu(1, 1, vec![hw, f_max], names, |gb, p, x, u, out| {
        out.push(backlash_dyn(gb, p[0], p[1], x[0], u[0]))
    });
    b.set_dynamic("Backlash", alg, dyn_);
    Rc::new(FastCell::new(b))
}

// ======================================================================================
// Deadband: y = u - clip(u, lower, upper)
// ======================================================================================

/// Deadband: y = u - clamp(u, lower, upper)
pub fn deadband(lower: f64, upper: f64) -> BlockRef {
    // y = u - clamp(u, lower, upper)  (dead zone between lower and upper)
    fn build<B: Builder>(b: &B, u0: B::N, lo: B::N, hi: B::N) -> B::N {
        b.sub(u0, b.min(b.max(u0, lo), hi))
    }
    let mut b = Block::default_block();
    b.type_name = "Deadband";
    b.f_alg = Some(Box::new(move |_x, u: &[f64], _t, out: &mut Vec<f64>| {
        out.clear();
        out.push(build(&F64Builder, u[0], lower, upper));
    }));
    let cell = RefCell::new(Graph::new(InputSignature::from_named_sizes([("u", 1usize)])));
    {
        let mut g = cell.borrow_mut();
        g.n_params = 2;
        g.param_defaults = vec![lower, upper];
        g.param_names = vec!["lower".into(), "upper".into()];
    }
    let y = {
        let gb = GraphBuilder::new(&cell);
        build(&gb, gb.input(0), gb.param(0), gb.param(1))
    };
    let mut g = cell.into_inner();
    g.outputs.push(y);
    b.set_alg("Deadband", g);
    Rc::new(FastCell::new(b))
}
