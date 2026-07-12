// Value-reference mapping: bridge the struct-API codegen layout
// (`codegen::ModelLayout`) to the FMI variable set, value references, and the
// `<ModelStructure>`.
//
// The layout's `signal_id` is exactly the id the generated `*_get_signal` /
// `*_set_signal` accessors take. We *reuse* that id space as the FMI value
// reference for everything addressable through a signal (states, outputs,
// params), so the C wrapper can forward `fmi3GetFloat64`/`fmi3SetFloat64`
// straight to `get_signal`/`set_signal` with no extra table. Two variable
// kinds have no signal id, so they get value references *above* the signal
// space:
//   - one continuous-state *derivative* per state (read via `model_deriv`), and
//   - the independent variable `time`.
//
// Layout of the value-reference space (all contiguous):
//   [0,                       n_state)                 states   (== signal id)
//   [n_state,                 n_state+n_sig)           outputs  (== signal id)
//   [n_state+n_sig,           n_state+n_sig+n_param)   params   (== signal id)
//   [der_base,                der_base+n_state)        der(x_i) (der_base = above)
//   time_vr                                            time     (der_base+n_state)

use crate::codegen::ModelLayout;

use crate::fmi::bindings::fmi3ValueReference;
use crate::fmi::model_description::{
    Causality, Initial, ModelStructure, StartValue, Variability, VarType, Variable,
};

/// The contiguous value-reference ranges of an exported model, derived from the
/// codegen layout. The C wrapper consumes the same constants so its
/// `fmi3GetFloat64` dispatch matches these ranges exactly.
#[derive(Debug, Clone)]
pub struct VrLayout {
    pub n_state: usize,
    pub n_sig: usize,
    pub n_param: usize,
    /// First value reference of the derivative block: `n_state+n_sig+n_param`.
    pub der_base: fmi3ValueReference,
    /// Value reference of the independent variable `time`: `der_base+n_state`.
    pub time_vr: fmi3ValueReference,
    /// First value reference of the event-indicator block: `time_vr+1`.
    pub ind_base: fmi3ValueReference,
    /// First value reference of the external-input block (open system):
    /// `ind_base + n_indicators`. Inputs map to `m->u[vr - input_base]`.
    pub input_base: fmi3ValueReference,
    /// Number of external inputs (`m->u[]`).
    pub n_input: usize,
}

impl VrLayout {
    /// `n_indicators` shifts the input block above the event indicators.
    pub fn new(layout: &ModelLayout, n_indicators: usize) -> Self {
        let der_base = (layout.n_state + layout.n_sig + layout.n_param) as fmi3ValueReference;
        let time_vr = der_base + layout.n_state as fmi3ValueReference;
        let ind_base = time_vr + 1;
        let input_base = ind_base + n_indicators as fmi3ValueReference;
        Self {
            n_state: layout.n_state,
            n_sig: layout.n_sig,
            n_param: layout.n_param,
            der_base,
            time_vr,
            ind_base,
            input_base,
            n_input: layout.n_input,
        }
    }

    /// The value reference of the derivative of the state at x-index `i`.
    pub fn der_vr(&self, i: usize) -> fmi3ValueReference {
        self.der_base + i as fmi3ValueReference
    }

    /// The value reference of the event indicator at index `i`.
    pub fn ind_vr(&self, i: usize) -> fmi3ValueReference {
        self.ind_base + i as fmi3ValueReference
    }

    /// The value reference of the external input at flat index `i`.
    pub fn input_vr(&self, i: usize) -> fmi3ValueReference {
        self.input_base + i as fmi3ValueReference
    }
}

/// The FMI translation of a struct-API model: the typed variable list and the
/// model structure to feed `ModelDescription::new`, plus the value-reference
/// ranges the C wrapper binds to.
pub struct FmiVars {
    pub variables: Vec<Variable>,
    pub model_structure: ModelStructure,
    pub vr: VrLayout,
}

/// Build the FMI variable set from the codegen layout.
///
/// Every state, output and parameter becomes a `Float64` whose value reference
/// equals its signal id. Each state additionally gets a derivative variable
/// `der(<name>)` (value reference in the derivative block) referenced from the
/// state via FMI's `derivative=` attribute. Then the independent variable `time`,
/// and one `event_indicator_<i>` variable per state event (`n_indicators`).
/// `<ModelStructure>` lists outputs, state derivatives, event indicators, and the
/// calculated initial unknowns (outputs and derivatives).
pub fn build(layout: &ModelLayout, n_indicators: usize) -> FmiVars {
    let vr = VrLayout::new(layout, n_indicators);
    let mut variables = Vec::with_capacity(layout.vars.len() + layout.n_state + 1);
    let mut structure = ModelStructure::default();

    // The independent variable. FMI 3.0 requires exactly one; `time` is the
    // conventional name and carries no start.
    variables.push(Variable {
        name: "time".into(),
        value_reference: vr.time_vr,
        var_type: VarType::Float64,
        causality: Causality::Independent,
        variability: Variability::Continuous,
        initial: None,
        description: Some("Simulation time".into()),
        start: None,
        derivative_of: None,
        max_output_derivative_order: 0,
    });

    // States: local continuous variables with an exact start. Their derivatives
    // are emitted right after so `der(name)` can point back via `derivative=`.
    // `layout.states()` yields states in x-index order (signal id 0..n_state),
    // which is also the order `model_deriv` writes `dxdt[]`.
    for (i, st) in layout.states().enumerate() {
        let state_vr = st.signal_id as fmi3ValueReference;
        variables.push(Variable {
            name: st.name.clone(),
            value_reference: state_vr,
            var_type: VarType::Float64,
            causality: Causality::Local,
            variability: Variability::Continuous,
            initial: Some(Initial::Exact),
            description: None,
            start: st.start.map(StartValue::Float64),
            derivative_of: None,
            max_output_derivative_order: 0,
        });
        let dvr = vr.der_vr(i);
        variables.push(Variable {
            name: format!("der({})", st.name),
            value_reference: dvr,
            var_type: VarType::Float64,
            causality: Causality::Local,
            variability: Variability::Continuous,
            initial: Some(Initial::Calculated),
            description: None,
            start: None,
            derivative_of: Some(state_vr),
            max_output_derivative_order: 0,
        });
        structure.continuous_state_derivatives.push(dvr);
        structure.initial_unknowns.push(dvr);
    }

    // Block outputs: observable, calculated continuous variables.
    for out in layout.outputs() {
        let out_vr = out.signal_id as fmi3ValueReference;
        variables.push(Variable {
            name: out.name.clone(),
            value_reference: out_vr,
            var_type: VarType::Float64,
            causality: Causality::Output,
            variability: Variability::Continuous,
            initial: Some(Initial::Calculated),
            description: None,
            start: None,
            derivative_of: None,
            max_output_derivative_order: 0,
        });
        structure.outputs.push(out_vr);
        structure.initial_unknowns.push(out_vr);
    }

    // Internal observable signals: `local` (not part of the interface, so not in
    // `<Output>` / initial unknowns), but still readable through `get_signal`.
    for loc in layout.locals() {
        variables.push(Variable {
            name: loc.name.clone(),
            value_reference: loc.signal_id as fmi3ValueReference,
            var_type: VarType::Float64,
            causality: Causality::Local,
            variability: Variability::Continuous,
            initial: Some(Initial::Calculated),
            description: None,
            start: None,
            derivative_of: None,
            max_output_derivative_order: 0,
        });
    }

    // Parameters: tunable (the wrapper routes their value references to
    // `set_signal`, and `model_deriv` re-reads `m->p[]` each call), with an exact
    // start.
    for p in layout.params() {
        variables.push(Variable {
            name: p.name.clone(),
            value_reference: p.signal_id as fmi3ValueReference,
            var_type: VarType::Float64,
            causality: Causality::Parameter,
            variability: Variability::Tunable,
            initial: None,
            description: None,
            start: p.start.map(StartValue::Float64),
            derivative_of: None,
            max_output_derivative_order: 0,
        });
    }

    // External inputs (open system / subsystem export): settable continuous
    // variables routed to `m->u[]`. `layout.inputs()` is in `u[]` order; its
    // `signal_id` is the flat `u[]` index.
    for inp in layout.inputs() {
        variables.push(Variable {
            name: inp.name.clone(),
            value_reference: vr.input_vr(inp.signal_id),
            var_type: VarType::Float64,
            causality: Causality::Input,
            variability: Variability::Continuous,
            initial: None,
            description: None,
            start: Some(StartValue::Float64(0.0)),
            derivative_of: None,
            max_output_derivative_order: 0,
        });
    }

    // Event indicators: one `Float64` per state event, referenced by the
    // `<EventIndicator>` model-structure entries. Read via fmi3GetEventIndicators
    // (and fmi3GetFloat64 on their value references).
    for i in 0..n_indicators {
        let ivr = vr.ind_vr(i);
        variables.push(Variable {
            name: format!("event_indicator_{i}"),
            value_reference: ivr,
            var_type: VarType::Float64,
            causality: Causality::Local,
            variability: Variability::Continuous,
            initial: Some(Initial::Calculated),
            description: None,
            start: None,
            derivative_of: None,
            max_output_derivative_order: 0,
        });
        structure.event_indicators.push(ivr);
    }

    FmiVars { variables, model_structure: structure, vr }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{LayoutVar, ModelLayout, VarKind};

    /// A layout with 2 states, 1 output, 1 param. Mirrors what `struct_layout`
    /// would produce: states 0..2, output 2, param 3.
    fn layout() -> ModelLayout {
        ModelLayout {
            name: "M".into(),
            n_state: 2,
            n_sig: 1,
            n_param: 1,
            vars: vec![
                LayoutVar { name: "h".into(), signal_id: 0, kind: VarKind::State, start: Some(1.0) },
                LayoutVar { name: "v".into(), signal_id: 1, kind: VarKind::State, start: Some(0.0) },
                LayoutVar { name: "y".into(), signal_id: 2, kind: VarKind::Output, start: None },
                LayoutVar { name: "g".into(), signal_id: 3, kind: VarKind::Param, start: Some(-9.81) },
            ],
            n_input: 0,
            has_events: false,
            jvp: true,
        }
    }

    #[test]
    fn value_references_are_contiguous_and_typed() {
        let v = build(&layout(), 0);
        // der_base = n_state + n_sig + n_param = 4; time = der_base + n_state = 6.
        assert_eq!(v.vr.der_base, 4);
        assert_eq!(v.vr.time_vr, 6);
        assert_eq!(v.vr.der_vr(0), 4);
        assert_eq!(v.vr.der_vr(1), 5);

        let by_name = |n: &str| v.variables.iter().find(|x| x.name == n).unwrap();
        // States keep their signal id as value reference.
        assert_eq!(by_name("h").value_reference, 0);
        assert_eq!(by_name("h").causality, Causality::Local);
        assert_eq!(by_name("h").start.as_ref().and_then(|s| s.as_f64()), Some(1.0));
        // Derivatives point back at their state and live in the der block.
        assert_eq!(by_name("der(h)").value_reference, 4);
        assert_eq!(by_name("der(h)").derivative_of, Some(0));
        assert_eq!(by_name("der(v)").value_reference, 5);
        assert_eq!(by_name("der(v)").derivative_of, Some(1));
        // Output and param keep their signal ids; output is observable.
        assert_eq!(by_name("y").value_reference, 2);
        assert_eq!(by_name("y").causality, Causality::Output);
        assert_eq!(by_name("g").value_reference, 3);
        assert_eq!(by_name("g").causality, Causality::Parameter);
        // The independent variable is unique and named `time`.
        assert_eq!(by_name("time").causality, Causality::Independent);
    }

    #[test]
    fn model_structure_lists_derivatives_outputs_and_initial_unknowns() {
        let v = build(&layout(), 0);
        assert_eq!(v.model_structure.continuous_state_derivatives, vec![4, 5]);
        assert_eq!(v.model_structure.outputs, vec![2]);
        // Initial unknowns are the calculated variables: derivatives + outputs.
        assert_eq!(v.model_structure.initial_unknowns, vec![4, 5, 2]);
    }

    #[test]
    fn round_trips_through_model_description() {
        // The exporter's variables + structure must parse back through the
        // importer's `ModelDescription` consistently (states ordered by der).
        use crate::fmi::model_description::{
            DefaultExperiment, ModelDescription, ModelExchangeInfo,
        };
        let v = build(&layout(), 0);
        let md = ModelDescription::new(
            "M",
            "{tok}",
            Some(ModelExchangeInfo {
                model_identifier: "M".into(),
                needs_completed_integrator_step: false,
                provides_directional_derivatives: false,
                can_get_and_set_fmu_state: false,
            }),
            None,
            DefaultExperiment::default(),
            v.variables,
            v.model_structure,
        );
        let reparsed = ModelDescription::from_str(&md.to_xml()).unwrap();
        assert_eq!(reparsed.n_continuous_states(), 2);
        let states: Vec<_> = reparsed.continuous_states().iter().map(|s| s.name.clone()).collect();
        assert_eq!(states, vec!["h", "v"]);
    }
}
