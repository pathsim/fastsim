// FMU block constructors: ModelExchangeFMU and CoSimulationFMU.
//
// An FMU (Functional Mock-up Unit) appears to FastSim as a regular Block with
// input/output ports. We wrap a loaded `Instance<Me>` or `Instance<Cs>` in an
// interior-mutable backend (`Rc<FastCell<Backend>>`) that is shared between:
//   - the block's `f_dyn`/`f_alg`/`update_fn`/`sample_fn` closures
//   - state-event and time-event callbacks (`ZeroCrossing.func_act`,
//     `Schedule.func_act`, `ScheduleList.func_act`)
//
// ## Error policy in hot-path closures
//
// The `Fn`/`FnMut` signatures of FastSim block and event callbacks do not
// return `Result`, so FMI calls inside them cannot propagate errors back to
// the simulation loop. We intentionally swallow their `Result` via
// `let _ = backend.instance.<call>()`: an FMU that reports an error during
// a step has logged via the logger callback (see `fmi::callbacks`) and has
// left its state inconsistent; crashing the whole simulation is worse than
// emitting an error log and letting the outer loop continue. Init-time
// calls *do* propagate via `?` because they run before the block is wired up.
//
// ## References (audited projects)
//   - PathSim    `pathsim/blocks/fmu.py`          — Python API shape
//   - fmpy       `src/fmpy/fmi3.py`, simulation.py — lifecycle sequence
//   - Reference- `fmusim/FMI3MESimulation.c`       — ME loop structure
//     FMUs      `fmusim/FMI3CSSimulation.c`       — CS loop structure

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use crate::constants::FMI_INITIALIZATION_TOL;
use crate::events::schedule::{Schedule, ScheduleList};
use crate::utils::fastcell::FastCell;
use crate::utils::register::Register;

use crate::fmi::bindings::fmi3ValueReference;
use crate::fmi::instance::{Cs, DiscreteStateUpdate, Instance, Me};
use crate::fmi::model_description::{
    Causality, ModelDescription, StartValue, VarType, Variable,
};
use crate::fmi::unzip::FmuArchive;
use crate::fmi::Result;

use super::block::{Block, BlockRef};

// =========================================================================
// Common backend helpers
// =========================================================================

/// Expand a user-supplied override to exactly one value per array element.
///
/// A single value is broadcast across the whole array — the same rule FMI 3.0
/// §2.4.7.5 defines for the XML's own `start` attribute, so a scalar override
/// behaves identically whether the target is a scalar or an array. Any other
/// length mismatch is a mistake worth reporting rather than silently padding.
fn expand_override(name: &str, given: &[f64], n: usize) -> Result<Vec<f64>> {
    if given.len() > 1 && given.len() != n {
        return Err(crate::fmi::FmiError::ModelDescription(format!(
            "start value for `{name}` has {} entries but the variable has {n} \
             element(s); pass one value per element, or a single value to \
             broadcast across all of them",
            given.len()
        )));
    }
    Ok(StartValue::Float64(given.to_vec())
        .expand_f64(n)
        .expect("a Float64 start always expands to f64"))
}

/// Write one variable's values through the `fmi3Set*` call matching its declared
/// type.
///
/// FastSim's public API speaks `f64` throughout, so integer and boolean
/// variables are converted here. That keeps `start_values` usable for every
/// numeric FMI type instead of silently ignoring everything but `Float64`.
/// `String`, `Binary` and `Clock` have no meaningful `f64` representation and
/// are rejected rather than approximated.
fn set_variable_f64<K>(
    inst: &Instance<K>,
    name: &str,
    v: &Variable,
    values: &[f64],
) -> Result<()> {
    let vr = [v.value_reference];
    macro_rules! set_as {
        ($call:ident, $ty:ty) => {{
            let converted: Vec<$ty> = values.iter().map(|x| *x as $ty).collect();
            inst.$call(&vr, &converted)
        }};
    }
    match v.var_type {
        VarType::Float64 => inst.set_float64(&vr, values),
        VarType::Float32 => set_as!(set_float32, f32),
        VarType::Int8 => set_as!(set_int8, i8),
        VarType::UInt8 => set_as!(set_uint8, u8),
        VarType::Int16 => set_as!(set_int16, i16),
        VarType::UInt16 => set_as!(set_uint16, u16),
        VarType::Int32 => set_as!(set_int32, i32),
        VarType::UInt32 => set_as!(set_uint32, u32),
        // FMI 3.0 §2.2.6: `Enumeration` variables are accessed through the
        // Int64 pair, not through an accessor of their own.
        VarType::Int64 | VarType::Enumeration => set_as!(set_int64, i64),
        VarType::UInt64 => set_as!(set_uint64, u64),
        VarType::Boolean => {
            let converted: Vec<bool> = values.iter().map(|x| *x != 0.0).collect();
            inst.set_boolean(&vr, &converted)
        }
        VarType::String | VarType::Binary | VarType::Clock => {
            Err(crate::fmi::FmiError::ModelDescription(format!(
                "`{name}` is a {:?} variable and cannot be set from a number",
                v.var_type
            )))
        }
    }
}

/// Push the `start_values` entries that name structural parameters into the FMU
/// through Configuration Mode, then record the new sizes on the model
/// description.
///
/// Structural parameters are the only way to resize an FMI 3.0 array, and
/// §2.3.2 makes Configuration Mode the only state in which they may be set — a
/// plain `fmi3Set*` in Initialization Mode is rejected. Because the array shapes
/// derived from them feed port counts and every `nValues` argument afterwards,
/// this has to run before initialization and before port discovery.
///
/// Returns the names it consumed, so the caller does not apply them a second
/// time during initialization.
fn apply_structural_overrides<K>(
    inst: &Instance<K>,
    md: &mut ModelDescription,
    overrides: &HashMap<String, Vec<f64>>,
) -> Result<Vec<String>> {
    let mut pending: Vec<(String, fmi3ValueReference, Vec<f64>)> = Vec::new();
    for (name, given) in overrides {
        let v = md
            .variable_by_name(name)
            .ok_or_else(|| crate::fmi::FmiError::UnknownVariable(name.clone()))?;
        if v.causality != Causality::StructuralParameter {
            continue;
        }
        let n = md.n_values(v);
        pending.push((name.clone(), v.value_reference, expand_override(name, given, n)?));
    }
    if pending.is_empty() {
        return Ok(Vec::new());
    }
    if !inst.supports_configuration_mode() {
        return Err(crate::fmi::FmiError::ModelDescription(
            "FMU declares structural parameters but does not export \
             fmi3EnterConfigurationMode"
                .into(),
        ));
    }

    inst.enter_configuration_mode()?;
    for (name, vr, values) in &pending {
        let v = md
            .variable_by_vr(*vr)
            .expect("value reference came from this description");
        set_variable_f64(inst, name, v, values)?;
    }
    inst.exit_configuration_mode()?;

    // Re-derive every array shape from the values we just wrote. Only the first
    // element can act as a dimension (a dimension is a single UInt64), which is
    // also all `Dimension::Referenced` ever reads.
    for (_, vr, values) in &pending {
        if let Some(first) = values.first() {
            md.set_dimension_size(*vr, first.max(0.0) as u64);
        }
    }
    Ok(pending.into_iter().map(|(name, _, _)| name).collect())
}

/// The flat `Float64` value layout of a group of FMI variables.
///
/// `vrs` is what goes to `fmi3Get*`/`fmi3Set*` as `nValueReferences`, while
/// `n_values` is the length of the value buffer those calls read or write. The
/// two are equal only when every variable is a scalar; an array contributes one
/// reference and several elements.
///
/// Used for the quantities FMI defines as `Float64` by construction — the
/// continuous states and their derivatives. Ports go through `PortLayout`
/// instead, because they may be of any numeric type.
struct ValueLayout {
    vrs: Vec<fmi3ValueReference>,
    n_values: usize,
}

impl ValueLayout {
    fn collect<'a>(
        md: &ModelDescription,
        vars: impl Iterator<Item = &'a Variable>,
    ) -> Self {
        let selected: Vec<&Variable> =
            vars.filter(|v| v.var_type == VarType::Float64).collect();
        Self {
            vrs: selected.iter().map(|v| v.value_reference).collect(),
            n_values: selected.iter().map(|v| md.n_values(v)).sum(),
        }
    }

    fn is_empty(&self) -> bool {
        self.vrs.is_empty()
    }
}

/// A typed scratch buffer plus the `fmi3Get*` / `fmi3Set*` pair that fills it.
///
/// FastSim ports are `f64`, but an FMU's inputs and outputs may be of any FMI
/// numeric type — `Stair` publishes only an `Int32`, and `Feedthrough` mixes
/// `Float32`, `Int32` and `Boolean` with its `Float64` signals. Widening every
/// numeric type to `f64` at the port boundary makes those FMUs usable while
/// keeping one FFI call per type per step.
macro_rules! typed_scratch {
    ($( $variant:ident, $elem:ty, $get:ident, $set:ident, $( $vt:ident )|+ ; )*) => {
        pub enum Scratch { $( $variant(Vec<$elem>), )* }

        impl Scratch {
            /// A buffer of `n` elements for `ty`, or `None` for the types that
            /// have no meaningful `f64` representation (`String`, `Binary`,
            /// `Clock`) and therefore cannot become ports.
            fn new(ty: VarType, n: usize) -> Option<Self> {
                match ty {
                    $( $( VarType::$vt )|+ => Some(Scratch::$variant(vec![Default::default(); n])), )*
                    _ => None,
                }
            }

            fn read<K>(&mut self, inst: &Instance<K>, vrs: &[fmi3ValueReference]) -> Result<()> {
                match self { $( Scratch::$variant(b) => inst.$get(vrs, b), )* }
            }

            fn write<K>(&self, inst: &Instance<K>, vrs: &[fmi3ValueReference]) -> Result<()> {
                match self { $( Scratch::$variant(b) => inst.$set(vrs, b), )* }
            }

            /// Element `i` widened to `f64`.
            fn get(&self, i: usize) -> f64 {
                match self { $( Scratch::$variant(b) => b[i].into_f64(), )* }
            }

            /// Store `v` into element `i`, narrowing to the FMI type.
            fn put(&mut self, i: usize, v: f64) {
                match self { $( Scratch::$variant(b) => b[i] = FromF64::from_f64(v), )* }
            }
        }
    };
}

/// Widening and narrowing between an FMI element type and FastSim's `f64` port
/// representation. `bool` maps to 0.0/1.0 and back through `!= 0.0`, which is
/// the convention FastSim's own boolean-valued blocks use.
pub trait IntoF64 {
    fn into_f64(self) -> f64;
}
pub trait FromF64 {
    fn from_f64(v: f64) -> Self;
}

macro_rules! numeric_conv {
    ($($ty:ty),*) => {$(
        impl IntoF64 for $ty {
            fn into_f64(self) -> f64 { self as f64 }
        }
        impl FromF64 for $ty {
            fn from_f64(v: f64) -> Self { v as $ty }
        }
    )*};
}
numeric_conv!(f32, f64, i8, u8, i16, u16, i32, u32, i64, u64);

impl IntoF64 for bool {
    fn into_f64(self) -> f64 {
        if self { 1.0 } else { 0.0 }
    }
}
impl FromF64 for bool {
    fn from_f64(v: f64) -> Self {
        v != 0.0
    }
}

typed_scratch! {
    F64, f64, get_float64, set_float64, Float64;
    F32, f32, get_float32, set_float32, Float32;
    I8,  i8,  get_int8,    set_int8,    Int8;
    U8,  u8,  get_uint8,   set_uint8,   UInt8;
    I16, i16, get_int16,   set_int16,   Int16;
    U16, u16, get_uint16,  set_uint16,  UInt16;
    I32, i32, get_int32,   set_int32,   Int32;
    U32, u32, get_uint32,  set_uint32,  UInt32;
    // FMI 3.0 §2.2.6 routes `Enumeration` through the Int64 accessors.
    I64, i64, get_int64,   set_int64,   Int64 | Enumeration;
    U64, u64, get_uint64,  set_uint64,  UInt64;
    B,   bool, get_boolean, set_boolean, Boolean;
}

/// One FMI type's contribution to a block's port vector.
pub struct PortGroup {
    vrs: Vec<fmi3ValueReference>,
    /// Port index of each variable's first element, parallel to `vrs`.
    offsets: Vec<usize>,
    /// Element count of each variable, parallel to `vrs`.
    counts: Vec<usize>,
    buf: Scratch,
}

/// A block's ports, backed by FMI variables of possibly differing types.
///
/// Ports are numbered in the order the variables are declared, whatever their
/// type, and an array variable occupies one port per element. The groups below
/// scatter that flat numbering into one typed FFI call per type.
pub struct PortLayout {
    groups: Vec<PortGroup>,
    n_ports: usize,
}

impl PortLayout {
    fn collect<'a>(
        md: &ModelDescription,
        vars: impl Iterator<Item = &'a Variable>,
    ) -> Self {
        // (type, vrs, offsets, counts) accumulated in first-appearance order so
        // the layout is deterministic.
        let mut by_type: Vec<(VarType, Vec<fmi3ValueReference>, Vec<usize>, Vec<usize>)> =
            Vec::new();
        let mut n_ports = 0usize;

        for v in vars {
            let n = md.n_values(v);
            if n == 0 || Scratch::new(v.var_type, 0).is_none() {
                continue; // zero-sized array, or a type with no f64 meaning
            }
            let entry = match by_type.iter_mut().find(|(t, ..)| *t == v.var_type) {
                Some(e) => e,
                None => {
                    by_type.push((v.var_type, Vec::new(), Vec::new(), Vec::new()));
                    by_type.last_mut().expect("just pushed")
                }
            };
            entry.1.push(v.value_reference);
            entry.2.push(n_ports);
            entry.3.push(n);
            n_ports += n;
        }

        let groups = by_type
            .into_iter()
            .map(|(ty, vrs, offsets, counts)| PortGroup {
                buf: Scratch::new(ty, counts.iter().sum())
                    .expect("type was accepted during collection"),
                vrs,
                offsets,
                counts,
            })
            .collect();
        Self { groups, n_ports }
    }

    fn is_empty(&self) -> bool {
        self.n_ports == 0
    }

    /// Where a value reference's elements sit in the port numbering, as
    /// `(first port, element count)`. `None` if it is not one of these ports.
    fn locate(&self, vr: fmi3ValueReference) -> Option<(usize, usize)> {
        self.groups.iter().find_map(|g| {
            g.vrs
                .iter()
                .position(|&x| x == vr)
                .map(|i| (g.offsets[i], g.counts[i]))
        })
    }

    /// Read every group from the FMU and scatter the values into `out`, which
    /// the caller sizes to `n_ports`. A failing group leaves its ports at their
    /// previous values — see the error policy in the file header.
    fn read<K>(&mut self, inst: &Instance<K>, out: &mut [f64]) {
        for g in &mut self.groups {
            if g.buf.read(inst, &g.vrs).is_err() {
                continue;
            }
            let mut src = 0usize;
            for (&off, &count) in g.offsets.iter().zip(&g.counts) {
                for k in 0..count {
                    out[off + k] = g.buf.get(src + k);
                }
                src += count;
            }
        }
    }

    /// Gather `values` (indexed by port) into the typed buffers and push each
    /// group to the FMU. A short `values` leaves the remaining ports at zero.
    fn write<K>(&mut self, inst: &Instance<K>, values: &[f64]) {
        for g in &mut self.groups {
            let mut dst = 0usize;
            for (&off, &count) in g.offsets.iter().zip(&g.counts) {
                for k in 0..count {
                    g.buf.put(dst + k, values.get(off + k).copied().unwrap_or(0.0));
                }
                dst += count;
            }
            let _ = g.buf.write(inst, &g.vrs);
        }
    }
}

/// Drain the `UpdateDiscreteStates` fixed-point loop. FMI 3.0 §3.2.1 requires
/// iterating until `discreteStatesNeedUpdate=false`. Returns the last update so
/// the caller can inspect `next_event_time` and `values_changed`.
fn drain_discrete_state_updates<K>(inst: &Instance<K>) -> Result<DiscreteStateUpdate> {
    loop {
        let u = inst.update_discrete_states()?;
        if u.terminate_simulation {
            return Err(crate::fmi::FmiError::ModelDescription(
                "FMU requested termination during initialization".into(),
            ));
        }
        if !u.discrete_states_need_update {
            return Ok(u);
        }
    }
}

/// Shared FMI 3.0 initialization sequence used by both ME and CS block
/// constructors: `EnterInit → apply user overrides → ExitInit → drain
/// UpdateDiscreteStates`. Returns the final discrete-state update so the caller
/// can seed time events from `next_event_time_defined`.
///
/// The values declared in `modelDescription.xml` are deliberately *not* pushed
/// back into the FMU. FMI 3.0 §2.4.7.5 is explicit that the FMU has already
/// initialized its variables from those values at instantiation, so calling
/// `fmi3Set{VariableType}` is "only necessary if a different value as stored in
/// the XML file is desired" — which is precisely what `start_values` is for.
/// Re-applying them is not just redundant: an FMU is free to reject a set on a
/// variable it considers its own to compute, and PMSF's `DynamicArrayTest`
/// answers a write to its `initial="exact"` output with `fmi3Error`.
///
/// `enters_event_mode` says which state `fmi3ExitInitializationMode` leaves the
/// FMU in (FMI 3.0 §2.3.1): Model Exchange always lands in Event Mode, while
/// Co-Simulation lands in Event Mode only when instantiated with
/// `eventModeUsed=true` and in Step Mode otherwise. `fmi3UpdateDiscreteStates`
/// is an Event Mode function, so draining it in Step Mode is a state-machine
/// violation that a strict FMU answers with `fmi3Error` — which is exactly what
/// PMSF's `SimpleVariableTest` (`hasEventMode="false"`) does.
///
/// Ref: PathSim `blocks/fmu.py::_initialize`, Reference-FMUs
/// `fmusim/FMI3MESimulation.c:53-96` and `FMI3CSSimulation.c:70-123`.
fn run_initialization<K>(
    inst: &Instance<K>,
    md: &ModelDescription,
    start_values: &Option<HashMap<String, Vec<f64>>>,
    already_applied: &[String],
    tolerance: Option<f64>,
    enters_event_mode: bool,
) -> Result<Option<DiscreteStateUpdate>> {
    inst.enter_initialization_mode(tolerance, 0.0, None)?;

    if let Some(overrides) = start_values {
        for (name, given) in overrides {
            if already_applied.iter().any(|n| n == name) {
                continue; // a structural parameter, set in Configuration Mode
            }
            let v = md
                .variable_by_name(name)
                .ok_or_else(|| crate::fmi::FmiError::UnknownVariable(name.clone()))?;
            let n = md.n_values(v);
            if n == 0 {
                continue; // array currently sized to zero elements
            }
            let values = expand_override(name, given, n)?;
            set_variable_f64(inst, name, v, &values)?;
        }
    }

    inst.exit_initialization_mode()?;
    if enters_event_mode {
        drain_discrete_state_updates(inst).map(Some)
    } else {
        Ok(None)
    }
}

// =========================================================================
// ModelExchangeFMU
// =========================================================================

/// FMI 3.0 Model-Exchange backend — owns the instantiated FMU plus cached
/// metadata needed by block/event callbacks. Shared between closures via
/// `Rc<FastCell<MeBackend>>`; `block` and `time_events` are filled in after
/// the `Block` is wrapped in `Rc`, closing the back-reference loop used by
/// `handle_event` to mutate the block's engine and schedule new time events.
pub struct MeBackend {
    pub instance: Instance<Me>,
    pub md: ModelDescription,
    /// Ports, grouped by FMI type. Every numeric type is widened to the block's
    /// `f64` ports; see `PortLayout`.
    pub inputs: PortLayout,
    pub outputs: PortLayout,
    pub state_vrs: Vec<fmi3ValueReference>,
    /// Flat element count behind `state_vrs`. Equal to `state_vrs.len()` unless
    /// a state is an array, in which case the FMI call's `nValues` argument must
    /// be the element count while `nValueReferences` stays the variable count.
    pub n_state_values: usize,
    pub n_event_indicators: usize,
    /// Back-reference to the owning block, set after block construction.
    /// Used by `handle_event` to call `engine.set(x_new)` when the FMU
    /// signals `values_changed` from `UpdateDiscreteStates`.
    pub block: Option<BlockRef>,
    /// ScheduleList that carries FMU-announced time events; filled in after
    /// `Block` is wrapped in `Rc`. `handle_event` appends `next_event_time`
    /// here when the FMU reports one.
    pub time_events: Option<Rc<FastCell<ScheduleList>>>,
    /// Pre-allocated scratch for `GetEventIndicators`. Sized to
    /// `n_event_indicators` at construction; re-read on every ZeroCrossing
    /// `func_evt` call without allocating.
    pub ei_buf: Vec<f64>,
    /// Pre-allocated scratch for `GetFloat64(state_vrs)` used by
    /// `handle_event` when the FMU signals `values_changed`.
    pub state_buf: Vec<f64>,
    /// Pre-allocated seed vector for `fmi3GetDirectionalDerivative` in
    /// `jac_dyn`. Sized to `n_states` at construction; the closure flips one
    /// element to 1.0 and back per column without allocating.
    pub jac_seed_buf: Vec<f64>,
    /// Pre-allocated column buffer for the sensitivity output of each
    /// directional-derivative call.
    pub jac_col_buf: Vec<f64>,
    /// Derivative VRs in the order their corresponding states appear in
    /// `state_vrs`. Captured once at construction so the `jac_dyn` closure
    /// has immediate access without hitting `ModelStructure`.
    pub state_deriv_vrs: Vec<fmi3ValueReference>,
    /// Flat element count behind `state_deriv_vrs`; the length of the
    /// `sensitivity` buffer `fmi3GetDirectionalDerivative` writes.
    pub n_deriv_values: usize,
}

impl MeBackend {
    /// Latch the block's inputs into the FMU. A short `u` (block not fully
    /// wired) leaves the remaining ports at zero.
    fn apply_inputs(&mut self, u: &[f64]) {
        let Self { instance, inputs, .. } = self;
        inputs.write(instance, u);
    }

    /// Pull the FMU's outputs into `out`, sized to the port count.
    fn read_outputs(&mut self, out: &mut Vec<f64>) {
        let Self { instance, outputs, .. } = self;
        out.clear();
        out.resize(outputs.n_ports, 0.0);
        outputs.read(instance, out);
    }

    /// Run the full event-handling sequence in response to a detected state
    /// or time event:
    /// `EnterEventMode → drain UpdateDiscreteStates → EnterContinuousTimeMode`
    /// plus engine-state reset if `values_changed` and time-event insertion
    /// if `next_event_time_defined`.
    ///
    /// The FMU's time is advanced to the event time first. Without it the FMU
    /// is still standing at the end of the last integrator step, which is
    /// strictly before the event, so `UpdateDiscreteStates` performs no
    /// transition and re-announces the same `nextEventTime` — the event then
    /// repeats, `ScheduleList` sees no new entry and deactivates itself, and the
    /// FMU's discrete state silently stops advancing. `Stair` shows this as a
    /// counter frozen at 3.
    ///
    /// Ref: `reference-fmus/fmusim/FMI3MESimulation.c:186-227` (`FMI3SetTime`
    /// before `FMI3EnterEventMode`) + PathSim
    /// `blocks/fmu.py::ModelExchangeFMU._handle_event`.
    fn handle_event(&mut self, t: f64) {
        if self.instance.set_time(t).is_err() {
            return;
        }
        if self.instance.enter_event_mode().is_err() {
            return;
        }
        let u = match drain_discrete_state_updates(&self.instance) {
            Ok(u) => u,
            Err(_) => return,
        };
        let _ = self.instance.enter_continuous_time_mode();

        if u.values_changed && !self.state_vrs.is_empty() {
            // Re-use the pre-allocated scratch buffer.
            if self
                .instance
                .get_float64(&self.state_vrs, &mut self.state_buf)
                .is_ok()
            {
                if let Some(blk) = &self.block {
                    if let Some(engine) = blk.borrow_mut().engine.as_mut() {
                        engine.set(&self.state_buf);
                    }
                }
            }
        }

        if u.next_event_time_defined {
            self.insert_time_event(u.next_event_time);
        }
    }

    /// `bisect.insort` analog — append a FMU-announced time event into
    /// `time_events.times_evt` in ascending order, deduplicating within
    /// tolerance.
    fn insert_time_event(&self, t: f64) {
        let Some(tel) = &self.time_events else { return };
        let tel = tel.borrow_mut();
        let tol = crate::constants::TOLERANCE;
        let pos = tel.times_evt.partition_point(|&v| v < t - tol);
        if pos < tel.times_evt.len() && (tel.times_evt[pos] - t).abs() <= tol {
            return;
        }
        tel.times_evt.insert(pos, t);
    }
}

/// Construct a Model-Exchange FMU block.
///
/// Mirrors PathSim's `ModelExchangeFMU(fmu_path, instance_name, start_values,
/// tolerance, verbose)` signature.
pub fn model_exchange_fmu(
    fmu_path: impl AsRef<Path>,
    instance_name: &str,
    start_values: Option<HashMap<String, Vec<f64>>>,
    tolerance: f64,
    verbose: bool,
) -> Result<BlockRef> {
    // --- 1. extract + parse + instantiate ---
    let archive = FmuArchive::extract(fmu_path.as_ref())?;
    let mut md = ModelDescription::from_file(archive.model_description())?;
    let inst = Instance::<Me>::new_model_exchange(archive, &md, instance_name, verbose)?;

    // --- 2. structural parameters, then the shared init sequence, then the
    // ME-specific transition to continuous time. Structural parameters go first
    // because they decide the array shapes every later step depends on.
    let structural = match &start_values {
        Some(o) => apply_structural_overrides(&inst, &mut md, o)?,
        None => Vec::new(),
    };
    // ME always exits initialization into Event Mode (FMI 3.0 §2.3.1), so the
    // drain runs and the returned update is always `Some`.
    let init_update =
        run_initialization(&inst, &md, &start_values, &structural, Some(tolerance), true)?
            .expect("Model Exchange always enters Event Mode after ExitInitializationMode");
    inst.enter_continuous_time_mode()?;

    // --- 3. discover port / state VRs ---
    let inputs = PortLayout::collect(&md, md.inputs());
    let outputs = PortLayout::collect(&md, md.outputs());
    let states = ValueLayout::collect(&md, md.continuous_states().into_iter());
    let derivs = ValueLayout::collect(&md, md.continuous_state_derivatives());
    let n_event_indicators = md.n_event_indicators();

    // --- 4. initial state from FMU (post-init) via GetFloat64 on state VRs ---
    let mut initial_state = vec![0.0; states.n_values];
    if !states.is_empty() {
        inst.get_float64(&states.vrs, &mut initial_state)?;
    }

    // --- 5. assemble backend + Block ---
    let ei_buf = vec![0.0; n_event_indicators];
    let state_buf = vec![0.0; states.n_values];
    let jac_seed_buf = vec![0.0; states.n_values];
    let jac_col_buf = vec![0.0; states.n_values];
    let backend = Rc::new(FastCell::new(MeBackend {
        instance: inst,
        md,
        inputs,
        outputs,
        state_vrs: states.vrs,
        n_state_values: states.n_values,
        n_event_indicators,
        block: None,
        time_events: None,
        ei_buf,
        state_buf,
        jac_seed_buf,
        jac_col_buf,
        state_deriv_vrs: derivs.vrs,
        n_deriv_values: derivs.n_values,
    }));

    let mut b = Block::default_block();
    b.type_name = "ModelExchangeFMU";
    b.role = crate::blocks::block::BlockRole {
        is_dyn: true, is_src: false, is_rec: false,
    };
    b.opaque_feedthrough = true; // opaque FMU: conservatively assume y-on-u feedthrough
    b.initial_value = Some(initial_state.clone());
    b.engine = Some(crate::solvers::solver::Solver::with_defaults(&initial_state));

    // One port per flat element, so an array port exposes all of its elements.
    let n_in = { backend.borrow().inputs.n_ports };
    let n_out = { backend.borrow().outputs.n_ports };
    b.inputs = Register::new(Some(n_in), None);
    b.outputs = Register::new(Some(n_out), None);

    // f_dyn: set_time + set_states + set_inputs + get_derivatives
    let be = backend.clone();
    b.f_dyn = Some(Box::new(move |x, u, t, out| {
        let backend = be.borrow_mut();
        let _ = backend.instance.set_time(t);
        if !backend.state_vrs.is_empty() {
            let _ = backend.instance.set_continuous_states(x);
        }
        backend.apply_inputs(u);
        let n = backend.n_state_values;
        out.resize(n, 0.0);
        if n > 0 {
            let _ = backend.instance.get_continuous_state_derivatives(out);
        }
    }));

    // f_alg: set_time + set_states + set_inputs + get_outputs
    let be = backend.clone();
    b.f_alg = Some(Box::new(move |x, u, t, out| {
        let backend = be.borrow_mut();
        let _ = backend.instance.set_time(t);
        if !backend.state_vrs.is_empty() {
            let _ = backend.instance.set_continuous_states(x);
        }
        backend.apply_inputs(u);
        backend.read_outputs(out);
    }));

    // jac_dyn: if the FMU advertises `providesDirectionalDerivatives` AND
    // exports the symbol, assemble ∂ẋ/∂x column-by-column via directional
    // derivatives (seed = e_j, column j of the Jacobian).  Output layout is
    // row-major: `jac[i * n + j] = ∂ẋ_i/∂x_j` — matches the JIT AD convention
    // in `src/jit/autodiff.rs` so implicit solvers treat both paths uniformly.
    // Absent/erroring → leave `jac_dyn` None → falls back to the FD Jacobian
    // in `Block::compute_jacobian`.
    //
    // Scratch buffers (`jac_seed_buf`, `jac_col_buf`) live on `MeBackend`
    // sized at construction, so this closure allocates nothing in steady
    // state.
    let (provides_dd, has_dd_symbol, n_states, derivs_match) = {
        let be = backend.borrow();
        (
            be.md.model_exchange.as_ref()
                .map(|me| me.provides_directional_derivatives).unwrap_or(false),
            be.instance.supports_directional_derivatives(),
            be.n_state_values,
            // The Jacobian is square only if the derivative block has as many
            // flat elements as the state block — the natural case, but worth
            // checking before indexing `out[i * n + j]`.
            be.n_deriv_values == be.n_state_values,
        )
    };
    if provides_dd && has_dd_symbol && n_states > 0 && derivs_match {
        let be = backend.clone();
        b.jac_dyn = Some(Box::new(move |x, u, t, out| {
            let backend = be.borrow_mut();
            let _ = backend.instance.set_time(t);
            let _ = backend.instance.set_continuous_states(x);
            backend.apply_inputs(u);
            let n = backend.n_state_values;
            out.clear();
            out.resize(n * n, 0.0);
            if n == 0 { return; }
            // Split-borrow across disjoint fields of `MeBackend`.
            let MeBackend {
                instance, state_vrs, state_deriv_vrs, jac_seed_buf, jac_col_buf, ..
            } = &mut *backend;
            for v in jac_seed_buf.iter_mut() { *v = 0.0; }
            for j in 0..n {
                jac_seed_buf[j] = 1.0;
                if instance.get_directional_derivative(
                    state_deriv_vrs, state_vrs, jac_seed_buf, jac_col_buf,
                ).is_ok() {
                    for i in 0..n { out[i * n + j] = jac_col_buf[i]; }
                }
                // On error: column stays zero (already initialized).  Newton
                // degrades gracefully to a partial Jacobian; FD fallback
                // isn't possible here without a re-entrant handle to `f_dyn`.
                jac_seed_buf[j] = 0.0;
            }
        }));
    }

    // --- 6. wrap block + install events -------------------------------
    let blk_ref: BlockRef = Rc::new(FastCell::new(b));

    // Time-event ScheduleList: starts empty. Call `handle_event` which will
    // pull/store new times into it during event resolution.
    let time_events = Rc::new(FastCell::new(ScheduleList::new(
        Vec::new(),
        None,
        tolerance,
    )));
    {
        let be = backend.clone();
        time_events.borrow_mut().func_act =
            Some(Box::new(move |t| be.borrow_mut().handle_event(t)));
    }

    // Close the back-references so `handle_event` can reach the block's
    // engine and the time-event list. This creates the (existing fastsim)
    // closure/event Rc cycle documented in `constructors::sample_hold`.
    {
        let be = backend.borrow_mut();
        be.block = Some(blk_ref.clone());
        be.time_events = Some(time_events.clone());
    }

    // Seed initial time event if the FMU announced one during init.
    if init_update.next_event_time_defined {
        backend
            .borrow_mut()
            .insert_time_event(init_update.next_event_time);
    }
    // Register the time-event list as a block event.
    blk_ref
        .borrow_mut()
        .events
        .push(time_events as crate::simulation::SimEventRef);

    // ZeroCrossing per event indicator. `func_evt(t)` fetches the i-th
    // indicator from the FMU; `func_act(t)` runs the full event handler.
    let n_ei = backend.borrow().n_event_indicators;
    for i in 0..n_ei {
        let be_evt = backend.clone();
        let func_evt = move |t: f64| -> f64 {
            let be = be_evt.borrow_mut();
            let _ = be.instance.set_time(t);
            // Split-borrow: `instance` and `ei_buf` are disjoint fields, so
            // the borrow checker allows the simultaneous &self / &mut slice.
            if be.instance.get_event_indicators(&mut be.ei_buf).is_err() {
                return 0.0;
            }
            be.ei_buf[i]
        };

        let be_act = backend.clone();
        let func_act = Box::new(move |t: f64| be_act.borrow_mut().handle_event(t));

        let zc = crate::events::zerocrossing::ZeroCrossing::new(
            func_evt,
            Some(func_act),
            tolerance,
        );
        blk_ref.borrow_mut().events.push(Rc::new(FastCell::new(zc)));
    }

    // Install sample_fn: after each successful RK timestep, call
    // CompletedIntegratorStep; if the FMU signals event mode, run the event
    // handler.  Ref: FMI3MESimulation.c:179-227.
    //
    // Skip entirely when the FMU declares `needsCompletedIntegratorStep=false`
    // (FMI 3.0 §3.2.2): the spec permits omitting the call in that case,
    // which saves one FFI round-trip per successful step.
    let needs_cis = backend.borrow().md
        .model_exchange.as_ref()
        .map(|me| me.needs_completed_integrator_step)
        .unwrap_or(true);
    if needs_cis {
        let be_sample = backend.clone();
        blk_ref.borrow_mut().sample_fn = Some(Box::new(move |_blk, t, _dt| {
            let be = be_sample.borrow_mut();
            match be.instance.completed_integrator_step(true) {
                Ok(r) if r.enter_event_mode => be.handle_event(t),
                _ => {}
            }
        }));
    }

    Ok(blk_ref)
}

// =========================================================================
// CoSimulationFMU
// =========================================================================

/// An output variable that supplies Taylor derivatives.
///
/// `fmi3GetOutputDerivatives` is requested per value reference and returns all
/// of that variable's elements at once, so the correction works in units of
/// variables while the block's ports are units of elements. This carries the
/// mapping between the two. Only floating-point outputs that declare
/// `maxOutputDerivativeOrder > 0` appear here.
pub struct OutputVar {
    pub vr: fmi3ValueReference,
    /// Port index of this variable's first element.
    pub offset: usize,
    /// Number of elements it occupies (1 for a scalar).
    pub n_values: usize,
    /// Its declared `maxOutputDerivativeOrder`, always at least 1.
    pub max_order: u32,
}

pub struct CsBackend {
    pub instance: Instance<Cs>,
    pub md: ModelDescription,
    /// Ports, grouped by FMI type; every numeric type is widened to the block's
    /// `f64` ports (see `PortLayout`).
    pub inputs: PortLayout,
    pub outputs: PortLayout,
    /// The outputs that carry Taylor derivatives, used to interpolate at block
    /// times between FMU communication points. Empty for most FMUs.
    pub taylor_vars: Vec<OutputVar>,
    pub dt: f64,
    /// Whether the FMU was instantiated with `eventModeUsed = true`. Drives
    /// post-init transition and in-step event handling.
    pub event_mode_used: bool,
    /// Whether the FMU was instantiated with `earlyReturnAllowed = true`. When
    /// true, DoStep may return before the requested step completes; the
    /// Schedule callback loops until the target time is reached.
    pub early_return_allowed: bool,
    /// Set to true once the FMU signals `terminateSimulation` from DoStep or
    /// UpdateDiscreteStates. Subsequent DoStep/I/O calls are skipped.
    pub terminated: bool,
    /// The last time the FMU successfully advanced to. We pass this as
    /// `currentCommunicationPoint` on the next DoStep, decoupling the Schedule
    /// event cadence from the FMU's actual time cursor.
    pub current_time: f64,

    // ----- hot-path scratch buffers (sized at construction) --------------
    /// Block inputs gathered from the input register before being handed to
    /// `PortLayout::write`.
    pub input_buf: Vec<f64>,
    /// FMU outputs read through `PortLayout` and optionally Taylor-extrapolated
    /// in `update_fn`.
    pub output_buf: Vec<f64>,
    /// Indices into `taylor_vars` whose declared `maxOutputDerivativeOrder`
    /// reaches the current Taylor order; rebuilt in-place each call.
    pub taylor_idx: Vec<usize>,
    /// Value references of that subset, in the same order.
    pub taylor_vrs: Vec<fmi3ValueReference>,
    /// Order vector passed to `GetOutputDerivatives`, one entry per value
    /// reference (all entries identical within a call).
    pub taylor_orders: Vec<i32>,
    /// Derivative values returned by `GetOutputDerivatives` — the flat
    /// concatenation of the selected variables' elements.
    pub taylor_deriv: Vec<f64>,
}

impl CsBackend {
    /// Run the CS event-handling sequence: EnterEventMode → drain
    /// UpdateDiscreteStates → EnterStepMode. Invoked when `fmi3DoStep`
    /// returns `eventHandlingNeeded` and `event_mode_used=true`.
    /// Ref: `reference-fmus/fmusim/FMI3CSSimulation.c:205-233`.
    fn handle_event(&mut self) {
        if self.instance.enter_event_mode().is_err() {
            return;
        }
        loop {
            let u = match self.instance.update_discrete_states() {
                Ok(u) => u,
                Err(_) => return,
            };
            if u.terminate_simulation {
                self.terminated = true;
                return;
            }
            if !u.discrete_states_need_update {
                break;
            }
        }
        let _ = self.instance.enter_step_mode();
    }
}

/// Construct a Co-Simulation FMU block.
///
/// Mirrors PathSim's `CoSimulationFMU(fmu_path, instance_name, start_values,
/// dt)` signature, plus a `verbose` flag (symmetric to `ModelExchangeFMU`).
/// If `dt` is `None`, the FMU's `DefaultExperiment.stepSize` is used;
/// otherwise an error is raised.
pub fn cosimulation_fmu(
    fmu_path: impl AsRef<Path>,
    instance_name: &str,
    start_values: Option<HashMap<String, Vec<f64>>>,
    dt: Option<f64>,
    verbose: bool,
) -> Result<BlockRef> {
    // --- 1. extract + parse ---
    let archive = FmuArchive::extract(fmu_path.as_ref())?;
    let mut md = ModelDescription::from_file(archive.model_description())?;

    // Communication step: explicit dt overrides DefaultExperiment.stepSize.
    let dt = dt.or(md.default_experiment.step_size).ok_or_else(|| {
        crate::fmi::FmiError::ModelDescription(
            "no communication step size: neither `dt` argument nor DefaultExperiment.stepSize"
                .into(),
        )
    })?;

    // Auto-detect FMI 3.0 CS capabilities from ModelDescription:
    //   - event_mode_used: opt in if FMU supports it (handles state/time
    //     events detected during DoStep).
    //   - early_return_allowed: opt in if FMU might return early (allows
    //     precise advance up to an internal event instead of the requested
    //     step boundary — improves bounce/event precision).
    let (event_mode_used, early_return_allowed) = md
        .co_simulation
        .as_ref()
        .map(|cs| (cs.has_event_mode, cs.might_return_early_from_do_step))
        .unwrap_or((false, false));

    // --- 2. instantiate + shared init sequence ---
    let inst = Instance::<Cs>::new_co_simulation(
        archive,
        &md,
        instance_name,
        event_mode_used,
        early_return_allowed,
        verbose,
    )?;
    // After ExitInit the FMU is in Event Mode if `event_mode_used`, else in
    // Step Mode (FMI 3.0 §2.3.1). `run_initialization` drains discrete states
    // only in the former case; here we add the explicit transition back to
    // Step Mode. Ref: reference-fmus/fmusim/FMI3CSSimulation.c:103-122.
    let structural = match &start_values {
        Some(o) => apply_structural_overrides(&inst, &mut md, o)?,
        None => Vec::new(),
    };
    let _init_update = run_initialization(
        &inst,
        &md,
        &start_values,
        &structural,
        Some(FMI_INITIALIZATION_TOL),
        event_mode_used,
    )?;
    if event_mode_used {
        inst.enter_step_mode()?;
    }

    // --- 4. port discovery ---
    let inputs = PortLayout::collect(&md, md.inputs());
    let outputs = PortLayout::collect(&md, md.outputs());
    // Taylor interpolation applies only to floating-point outputs that declare
    // derivatives; `fmi3GetOutputDerivatives` returns `fmi3Float64` values and
    // has no meaning for the integer and boolean ports.
    let taylor_vars: Vec<OutputVar> = md
        .outputs()
        .filter(|v| {
            matches!(v.var_type, VarType::Float64 | VarType::Float32)
                && v.max_output_derivative_order > 0
        })
        .filter_map(|v| {
            outputs.locate(v.value_reference).map(|(offset, n_values)| OutputVar {
                vr: v.value_reference,
                offset,
                n_values,
                max_order: v.max_output_derivative_order,
            })
        })
        .collect();

    // --- 5. assemble backend + Block: one port per flat element ---
    let n_in = inputs.n_ports;
    let n_out = outputs.n_ports;
    let n_taylor = taylor_vars.len();
    let backend = Rc::new(FastCell::new(CsBackend {
        instance: inst,
        md,
        inputs,
        outputs,
        taylor_vars,
        dt,
        event_mode_used,
        early_return_allowed,
        terminated: false,
        current_time: 0.0,
        input_buf: vec![0.0; n_in],
        output_buf: vec![0.0; n_out],
        // Taylor scratch: worst case every derivative-carrying output
        // contributes at every order.
        taylor_idx: Vec::with_capacity(n_taylor),
        taylor_vrs: Vec::with_capacity(n_taylor),
        taylor_orders: Vec::with_capacity(n_taylor),
        taylor_deriv: Vec::with_capacity(n_out),
    }));

    let mut b = Block::default_block();
    b.type_name = "CoSimulationFMU";
    b.role = crate::blocks::block::BlockRole {
        is_dyn: false, is_src: false, is_rec: false,
    };
    b.opaque_feedthrough = true; // opaque FMU: conservatively assume y-on-u feedthrough
    b.inputs = Register::new(Some(n_in), None);
    b.outputs = Register::new(Some(n_out), None);

    // Populate initial outputs from the post-init FMU state via the
    // backend's pre-allocated output_buf.
    {
        let backend = backend.borrow_mut();
        if !backend.outputs.is_empty() {
            let CsBackend { instance, outputs, output_buf, .. } = &mut *backend;
            outputs.read(instance, output_buf);
            for (i, v) in output_buf.iter().enumerate() {
                b.outputs.set_single(i, *v);
            }
        }
    }

    // Scheduled communication step. On each tick we drive the FMU up to
    // the target time `t`, possibly via multiple DoStep calls when the FMU
    // returns early (FMI 3.0 §4.2.4).
    //
    // Sequence per iteration:
    //   1. step_size = t - current_time
    //   2. DoStep(current_time, step_size)
    //   3. If earlyReturn: advance = last_successful_time - current_time
    //      else:           advance = step_size  (with earlyReturnAllowed=false
    //                                             the spec allows FMUs to skip
    //                                             writing last_successful_time)
    //   4. Handle terminate_simulation (sticky flag) and eventHandlingNeeded
    //      (EnterEventMode → drain UpdateDiscreteStates → EnterStepMode).
    //   5. Loop until current_time >= t or FMU fails to advance.
    let be = backend.clone();
    let schedule = Schedule::new(
        0.0,
        None,
        dt,
        Some(Box::new(move |t: f64| {
            let backend = be.borrow_mut();
            loop {
                if backend.terminated {
                    return;
                }
                let current = backend.current_time;
                let remaining = t - current;
                if remaining < crate::constants::TOLERANCE {
                    return;
                }
                let r = match backend.instance.do_step(current, remaining) {
                    Ok(r) => r,
                    Err(_) => return,
                };
                let advance = if backend.early_return_allowed && r.early_return {
                    (r.last_successful_time - current).max(0.0)
                } else {
                    remaining
                };
                backend.current_time = current + advance;

                if r.terminate_simulation {
                    backend.terminated = true;
                    return;
                }
                if r.event_handling_needed && backend.event_mode_used {
                    backend.handle_event();
                }

                // Safety: if the FMU neither terminated nor advanced, bail
                // instead of spinning forever.
                if advance < crate::constants::TOLERANCE {
                    return;
                }
            }
        })),
        crate::constants::TOLERANCE,
    );
    b.events.push(Rc::new(FastCell::new(schedule)));

    // update_fn: latch current inputs into FMU, read outputs back into block.
    // Runs on every FastSim update pass. When terminated, skip the I/O.
    //
    // When the FMU declares `maxOutputDerivativeOrder>0` for any output and
    // the block's current time `t` is ahead of the FMU's last communication
    // point (`backend.current_time`), we Taylor-extrapolate outputs using
    // `fmi3GetOutputDerivatives`:
    //     y(t) = y(tc) + Σ_{k=1..max_k}  y^(k)(tc) · (t-tc)^k / k!
    // Outputs with lower `max_output_derivative_order` contribute only up to
    // their declared order.
    //
    // All scratch buffers are fields of `CsBackend` pre-allocated at
    // construction, so this closure allocates nothing on the hot path.
    let be = backend.clone();
    b.update_fn = Some(Box::new(move |blk, t| {
        let backend = be.borrow_mut();
        if backend.terminated {
            return;
        }

        // --- inputs: block register → FMU (reuses input_buf) ---
        if !backend.inputs.is_empty() {
            for i in 0..backend.input_buf.len() {
                backend.input_buf[i] = blk.inputs.get_single(i);
            }
            let CsBackend { instance, inputs, input_buf, .. } = &mut *backend;
            inputs.write(instance, input_buf);
        }
        if backend.outputs.is_empty() {
            return;
        }

        // --- outputs: FMU → output_buf (zero-alloc) ---
        {
            let CsBackend { instance, outputs, output_buf, .. } = &mut *backend;
            outputs.read(instance, output_buf);
        }

        // --- Taylor interpolation (only if any output declares derivatives) ---
        let max_order_global = backend
            .taylor_vars
            .iter()
            .map(|o| o.max_order)
            .max()
            .unwrap_or(0);
        let dt_offset = t - backend.current_time;
        if max_order_global > 0 && dt_offset > crate::constants::TOLERANCE {
            let mut factorial: f64 = 1.0;
            for order in 1..=max_order_global {
                factorial *= order as f64;
                let factor = dt_offset.powi(order as i32) / factorial;

                // Refill the scratch lists in place for this order. Selection is
                // per variable — a variable's declared order covers all of its
                // elements.
                backend.taylor_idx.clear();
                backend.taylor_vrs.clear();
                let mut n_deriv_values = 0usize;
                for (i, o) in backend.taylor_vars.iter().enumerate() {
                    if o.max_order >= order {
                        backend.taylor_idx.push(i);
                        backend.taylor_vrs.push(o.vr);
                        n_deriv_values += o.n_values;
                    }
                }
                if backend.taylor_idx.is_empty() {
                    break;
                }
                backend.taylor_orders.clear();
                backend
                    .taylor_orders
                    .resize(backend.taylor_vrs.len(), order as i32);
                backend.taylor_deriv.clear();
                backend.taylor_deriv.resize(n_deriv_values, 0.0);

                // Split-borrow: `instance` is read-only; the three taylor_*
                // slices alias disjoint fields on self.
                if backend
                    .instance
                    .get_output_derivatives(
                        &backend.taylor_vrs,
                        &backend.taylor_orders,
                        &mut backend.taylor_deriv,
                    )
                    .is_err()
                {
                    break;
                }
                // Scatter the flat derivative block back to each variable's
                // slice of `output_buf`.
                let mut src = 0usize;
                for &i in &backend.taylor_idx {
                    let o = &backend.taylor_vars[i];
                    for k in 0..o.n_values {
                        backend.output_buf[o.offset + k] +=
                            backend.taylor_deriv[src + k] * factor;
                    }
                    src += o.n_values;
                }
            }
        }

        for (i, v) in backend.output_buf.iter().enumerate() {
            blk.outputs.set_single(i, *v);
        }
    }));

    Ok(Rc::new(FastCell::new(b)))
}
