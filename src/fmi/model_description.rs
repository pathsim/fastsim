// FMI 3.0 `modelDescription.xml` parser.
//
// Covers the subset needed for import: the root attributes, `<ModelExchange>`
// and `<CoSimulation>` interfaces, `<DefaultExperiment>`, `<ModelVariables>`
// (typed elements: Float64, Float32, Int8..Int64, UInt8..UInt64, Boolean,
// String), and `<ModelStructure>` (Output, ContinuousStateDerivative,
// EventIndicator, InitialUnknown).
//
// Convenience accessors (inspired by FMIL) pre-filter variables by role so
// callers don't repeat the filtering logic.

use std::collections::HashMap;
use std::path::Path;

use roxmltree::{Document, Node};

use super::bindings::fmi3ValueReference;
use super::{FmiError, Result};

// --- enums -----------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarType {
    Float32,
    Float64,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Boolean,
    String,
    Binary,
    Clock,
    Enumeration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Causality {
    Parameter,
    CalculatedParameter,
    Input,
    Output,
    Local,
    Independent,
    StructuralParameter,
}

impl Causality {
    fn parse(s: &str) -> Self {
        match s {
            "parameter" => Self::Parameter,
            "calculatedParameter" => Self::CalculatedParameter,
            "input" => Self::Input,
            "output" => Self::Output,
            "independent" => Self::Independent,
            "structuralParameter" => Self::StructuralParameter,
            _ => Self::Local, // default per FMI 3.0 §2.4.7.4
        }
    }

    fn to_str(self) -> &'static str {
        match self {
            Self::Parameter => "parameter",
            Self::CalculatedParameter => "calculatedParameter",
            Self::Input => "input",
            Self::Output => "output",
            Self::Local => "local",
            Self::Independent => "independent",
            Self::StructuralParameter => "structuralParameter",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variability {
    Constant,
    Fixed,
    Tunable,
    Discrete,
    Continuous,
}

impl Variability {
    fn parse(s: &str, ty: VarType) -> Self {
        match s {
            "constant" => Self::Constant,
            "fixed" => Self::Fixed,
            "tunable" => Self::Tunable,
            "discrete" => Self::Discrete,
            "continuous" => Self::Continuous,
            _ => {
                // Default per FMI 3.0 §2.4.7.4: continuous for Float*, discrete otherwise.
                if matches!(ty, VarType::Float32 | VarType::Float64) {
                    Self::Continuous
                } else {
                    Self::Discrete
                }
            }
        }
    }

    fn to_str(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::Fixed => "fixed",
            Self::Tunable => "tunable",
            Self::Discrete => "discrete",
            Self::Continuous => "continuous",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Initial {
    Exact,
    Approx,
    Calculated,
}

impl Initial {
    fn to_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Approx => "approx",
            Self::Calculated => "calculated",
        }
    }
}

// --- array dimensions -------------------------------------------------------

/// One dimension of an array variable (FMI 3.0 §2.4.7.4). A `<Dimension>`
/// element carries exactly one of the two attributes, never both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    /// `<Dimension start="N"/>` — a constant size baked into the XML.
    Fixed(u64),
    /// `<Dimension valueReference="vr"/>` — the size is the current value of the
    /// `UInt64` variable with this value reference, which the spec requires to
    /// be either a constant or a structural parameter. Structural parameters can
    /// be changed in Configuration Mode, so this size is not fixed at parse time
    /// — resolve it through `ModelDescription::dimension_size`.
    Referenced(fmi3ValueReference),
}

// --- variable ---------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Variable {
    pub name: String,
    pub value_reference: fmi3ValueReference,
    pub var_type: VarType,
    pub causality: Causality,
    pub variability: Variability,
    pub initial: Option<Initial>,
    pub description: Option<String>,
    pub start: Option<StartValue>,
    /// `valueReference` of the state this variable is a derivative of.
    /// Only set on continuous-state-derivative variables.
    pub derivative_of: Option<fmi3ValueReference>,
    /// Highest order of the Taylor polynomial available via
    /// `fmi3GetOutputDerivatives` (FMI 3.0 §2.4.7.5). 0 means no derivatives
    /// are provided. Applies to output variables.
    pub max_output_derivative_order: u32,
    /// Array shape. Empty for a scalar; one entry per `<Dimension>` element
    /// otherwise. The number of values the variable occupies in `fmi3Get*` /
    /// `fmi3Set*` is the product of the dimension sizes — see
    /// `ModelDescription::n_values`.
    pub dimensions: Vec<Dimension>,
}

impl Variable {
    /// A scalar variable with no start value, no derivative link and no output
    /// derivatives — the shape the exporter emits. Callers override the
    /// remaining fields with struct-update syntax, which keeps them compiling
    /// when new optional fields are added here.
    pub fn scalar(
        name: impl Into<String>,
        value_reference: fmi3ValueReference,
        var_type: VarType,
        causality: Causality,
        variability: Variability,
    ) -> Self {
        Self {
            name: name.into(),
            value_reference,
            var_type,
            causality,
            variability,
            initial: None,
            description: None,
            start: None,
            derivative_of: None,
            max_output_derivative_order: 0,
            dimensions: Vec::new(),
        }
    }

    /// True when the variable declares at least one `<Dimension>`.
    pub fn is_array(&self) -> bool {
        !self.dimensions.is_empty()
    }
}

/// A variable's declared start values.
///
/// Every FMI 3.0 variable is potentially an array, so each variant carries a
/// list. A scalar is the one-element case. Per FMI 3.0 §2.4.7.5 the `start`
/// attribute of an array may also hold a *single* value, which then applies to
/// every element — `expand_*` implements that broadcast.
#[derive(Debug, Clone, PartialEq)]
pub enum StartValue {
    Float64(Vec<f64>),
    Int64(Vec<i64>),
    Boolean(Vec<bool>),
    String(Vec<String>),
    /// Raw bytes per element, decoded from the hex-encoded `<Start value=".."/>`
    /// children of a `<Binary>` variable.
    Binary(Vec<Vec<u8>>),
}

/// Apply the FMI 3.0 broadcast rule: a single declared value fills the whole
/// array; otherwise the list is used as-is, padded with its last element (or
/// truncated) so the caller always gets exactly `n`.
fn expand<T: Clone + Default>(values: &[T], n: usize) -> Vec<T> {
    match values.len() {
        0 => vec![T::default(); n],
        1 => vec![values[0].clone(); n],
        len if len >= n => values[..n].to_vec(),
        _ => {
            let mut out = values.to_vec();
            let last = out.last().cloned().unwrap_or_default();
            out.resize(n, last);
            out
        }
    }
}

impl StartValue {
    pub fn scalar_f64(v: f64) -> Self {
        Self::Float64(vec![v])
    }

    /// Number of declared values. Note this is the count in the XML, which for a
    /// broadcast array start is 1 regardless of the array's size.
    pub fn len(&self) -> usize {
        match self {
            Self::Float64(v) => v.len(),
            Self::Int64(v) => v.len(),
            Self::Boolean(v) => v.len(),
            Self::String(v) => v.len(),
            Self::Binary(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The first declared value as an `f64`, for scalar consumers.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float64(v) => v.first().copied(),
            Self::Int64(v) => v.first().map(|x| *x as f64),
            Self::Boolean(v) => v.first().map(|b| if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    /// The first declared value as a `u64`, for array dimension sizes.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Int64(v) => v.first().and_then(|x| u64::try_from(*x).ok()),
            Self::Float64(v) => v.first().map(|x| *x as u64),
            _ => None,
        }
    }

    /// `n` float values, converting from the integer and boolean variants so a
    /// caller driving `fmi3SetFloat64` can accept any numeric declaration.
    pub fn expand_f64(&self, n: usize) -> Option<Vec<f64>> {
        match self {
            Self::Float64(v) => Some(expand(v, n)),
            Self::Int64(v) => Some(expand(v, n).into_iter().map(|x| x as f64).collect()),
            Self::Boolean(v) => Some(
                expand(v, n)
                    .into_iter()
                    .map(|b| if b { 1.0 } else { 0.0 })
                    .collect(),
            ),
            _ => None,
        }
    }

    pub fn expand_i64(&self, n: usize) -> Option<Vec<i64>> {
        match self {
            Self::Int64(v) => Some(expand(v, n)),
            Self::Boolean(v) => Some(expand(v, n).into_iter().map(i64::from).collect()),
            Self::Float64(v) => Some(expand(v, n).into_iter().map(|x| x as i64).collect()),
            _ => None,
        }
    }

    pub fn expand_bool(&self, n: usize) -> Option<Vec<bool>> {
        match self {
            Self::Boolean(v) => Some(expand(v, n)),
            Self::Int64(v) => Some(expand(v, n).into_iter().map(|x| x != 0).collect()),
            Self::Float64(v) => Some(expand(v, n).into_iter().map(|x| x != 0.0).collect()),
            _ => None,
        }
    }

    pub fn expand_string(&self, n: usize) -> Option<Vec<String>> {
        match self {
            Self::String(v) => Some(expand(v, n)),
            _ => None,
        }
    }

    pub fn expand_binary(&self, n: usize) -> Option<Vec<Vec<u8>>> {
        match self {
            Self::Binary(v) => Some(expand(v, n)),
            _ => None,
        }
    }
}

// --- interface sections -----------------------------------------------------

#[derive(Debug, Clone)]
pub struct ModelExchangeInfo {
    pub model_identifier: String,
    pub needs_completed_integrator_step: bool,
    pub provides_directional_derivatives: bool,
    pub can_get_and_set_fmu_state: bool,
}

// Several capability flags below are parsed for FMI 3.0 completeness and
// round-trip fidelity; not every one is consumed by the importer yet.
#[derive(Debug, Clone)]
pub struct CoSimulationInfo {
    pub model_identifier: String,
    pub can_handle_variable_communication_step_size: bool,
    pub fixed_internal_step_size: Option<f64>,
    pub has_event_mode: bool,
    pub provides_intermediate_update: bool,
    pub can_return_early_after_intermediate_update: bool,
    pub might_return_early_from_do_step: bool,
    pub can_get_and_set_fmu_state: bool,
}

#[derive(Debug, Clone, Default)]
pub struct DefaultExperiment {
    pub start_time: Option<f64>,
    pub stop_time: Option<f64>,
    pub tolerance: Option<f64>,
    pub step_size: Option<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct ModelStructure {
    pub outputs: Vec<fmi3ValueReference>,
    pub continuous_state_derivatives: Vec<fmi3ValueReference>,
    pub event_indicators: Vec<fmi3ValueReference>,
    pub initial_unknowns: Vec<fmi3ValueReference>,
}

// --- root -------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ModelDescription {
    pub fmi_version: String,
    pub model_name: String,
    pub instantiation_token: String,
    pub description: Option<String>,
    pub generation_tool: Option<String>,

    pub model_exchange: Option<ModelExchangeInfo>,
    pub co_simulation: Option<CoSimulationInfo>,
    pub default_experiment: DefaultExperiment,

    pub variables: Vec<Variable>,
    pub model_structure: ModelStructure,

    name_to_index: HashMap<String, usize>,
    vr_to_index: HashMap<fmi3ValueReference, usize>,
    /// Current size contributed by every `UInt64` variable that a
    /// `<Dimension valueReference=".."/>` may point at, seeded from the `start`
    /// attributes. Kept separate from `variables` because structural parameters
    /// are mutable: `set_dimension_size` records a Configuration Mode override
    /// so `n_values` keeps reporting the FMU's live array shape.
    dimension_sizes: HashMap<fmi3ValueReference, u64>,
}

/// Seed the dimension-size table from the declared starts of every `UInt64`
/// variable that could serve as an array bound (FMI 3.0 §2.4.7.4 restricts
/// those to constants and structural parameters, but indexing every `UInt64`
/// costs nothing and tolerates FMUs that stretch the rule).
fn seed_dimension_sizes(variables: &[Variable]) -> HashMap<fmi3ValueReference, u64> {
    variables
        .iter()
        .filter(|v| v.var_type == VarType::UInt64)
        .filter_map(|v| {
            v.start
                .as_ref()
                .and_then(|s| s.as_u64())
                .map(|n| (v.value_reference, n))
        })
        .collect()
}

impl ModelDescription {
    /// Build a `ModelDescription` from its parts (the export path). Defaults
    /// `fmiVersion` to "3.0" and `generationTool` to "fastsim", and computes the
    /// private name/VR lookup indices. The caller owns the value-reference
    /// assignment so it stays consistent with the generated C wrapper.
    pub fn new(
        model_name: impl Into<String>,
        instantiation_token: impl Into<String>,
        model_exchange: Option<ModelExchangeInfo>,
        co_simulation: Option<CoSimulationInfo>,
        default_experiment: DefaultExperiment,
        variables: Vec<Variable>,
        model_structure: ModelStructure,
    ) -> Self {
        let mut name_to_index = HashMap::with_capacity(variables.len());
        let mut vr_to_index = HashMap::with_capacity(variables.len());
        for (i, v) in variables.iter().enumerate() {
            name_to_index.insert(v.name.clone(), i);
            vr_to_index.insert(v.value_reference, i);
        }
        let dimension_sizes = seed_dimension_sizes(&variables);
        Self {
            fmi_version: "3.0".to_owned(),
            model_name: model_name.into(),
            instantiation_token: instantiation_token.into(),
            description: None,
            generation_tool: Some("fastsim".to_owned()),
            model_exchange,
            co_simulation,
            default_experiment,
            variables,
            model_structure,
            name_to_index,
            vr_to_index,
            dimension_sizes,
        }
    }

    /// Parse a `modelDescription.xml` from disk.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_str(&text)
    }

    /// Parse a `modelDescription.xml` from a string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(xml: &str) -> Result<Self> {
        let doc = Document::parse(xml)?;
        let root = doc.root_element();
        if root.tag_name().name() != "fmiModelDescription" {
            return Err(FmiError::ModelDescription(format!(
                "root element is <{}>, expected <fmiModelDescription>",
                root.tag_name().name()
            )));
        }

        let fmi_version = required_attr(root, "fmiVersion")?.to_owned();
        if !fmi_version.starts_with("3.") {
            return Err(FmiError::UnsupportedFmiVersion(fmi_version));
        }

        let model_name = required_attr(root, "modelName")?.to_owned();
        let instantiation_token = required_attr(root, "instantiationToken")?.to_owned();

        let mut me = None;
        let mut cs = None;
        let mut default_experiment = DefaultExperiment::default();
        let mut variables: Vec<Variable> = Vec::new();
        let mut model_structure = ModelStructure::default();

        for child in root.children().filter(|n| n.is_element()) {
            match child.tag_name().name() {
                "ModelExchange" => me = Some(parse_me(child)?),
                "CoSimulation" => cs = Some(parse_cs(child)?),
                "DefaultExperiment" => default_experiment = parse_default_experiment(child),
                "ModelVariables" => variables = parse_variables(child)?,
                "ModelStructure" => model_structure = parse_model_structure(child),
                _ => {} // ignore UnitDefinitions, TypeDefinitions, LogCategories, ...
            }
        }

        let mut name_to_index = HashMap::with_capacity(variables.len());
        let mut vr_to_index = HashMap::with_capacity(variables.len());
        for (i, v) in variables.iter().enumerate() {
            name_to_index.insert(v.name.clone(), i);
            vr_to_index.insert(v.value_reference, i);
        }
        let dimension_sizes = seed_dimension_sizes(&variables);

        Ok(Self {
            fmi_version,
            model_name,
            instantiation_token,
            description: root.attribute("description").map(String::from),
            generation_tool: root.attribute("generationTool").map(String::from),
            model_exchange: me,
            co_simulation: cs,
            default_experiment,
            variables,
            model_structure,
            name_to_index,
            vr_to_index,
            dimension_sizes,
        })
    }

    // --- lookups -----------------------------------------------------------

    pub fn variable_by_name(&self, name: &str) -> Option<&Variable> {
        self.name_to_index.get(name).map(|&i| &self.variables[i])
    }

    pub fn variable_by_vr(&self, vr: fmi3ValueReference) -> Option<&Variable> {
        self.vr_to_index.get(&vr).map(|&i| &self.variables[i])
    }

    // --- array shape -------------------------------------------------------

    /// Resolve one dimension to its current size. A `Referenced` dimension whose
    /// target carries no start value is malformed XML — the spec requires those
    /// starts to exist and be positive (§2.4.7.4) — so we fall back to 1, which
    /// degrades an unparseable array to a scalar instead of silently reporting
    /// zero elements and skipping it everywhere.
    pub fn dimension_size(&self, d: Dimension) -> u64 {
        match d {
            Dimension::Fixed(n) => n,
            Dimension::Referenced(vr) => self.dimension_sizes.get(&vr).copied().unwrap_or(1),
        }
    }

    /// The number of values a variable occupies in `fmi3Get*` / `fmi3Set*`: the
    /// product of its dimension sizes, and 1 for a scalar (the empty product).
    /// This is the per-VR contribution to the `nValues` argument, which is why
    /// `nValues` and `nValueReferences` differ for arrays. A dimension resized
    /// to zero yields zero, which the spec permits (§2.4.7.4).
    pub fn n_values(&self, v: &Variable) -> usize {
        v.dimensions
            .iter()
            .map(|&d| self.dimension_size(d) as usize)
            .product()
    }

    /// Total `nValues` across a set of variables, i.e. the length of the flat
    /// value buffer that a single `fmi3Get*`/`fmi3Set*` call over their value
    /// references reads or writes.
    pub fn total_n_values<'a>(&self, vars: impl Iterator<Item = &'a Variable>) -> usize {
        vars.map(|v| self.n_values(v)).sum()
    }

    /// Record a new size for a `UInt64` variable used as an array bound, after
    /// the importer has pushed it into the FMU through Configuration Mode.
    /// Subsequent `n_values` calls reflect the new shape.
    pub fn set_dimension_size(&mut self, vr: fmi3ValueReference, size: u64) {
        self.dimension_sizes.insert(vr, size);
    }

    /// Structural parameters, in declaration order. These are the variables an
    /// importer may change in Configuration Mode to resize arrays.
    pub fn structural_parameters(&self) -> impl Iterator<Item = &Variable> {
        self.variables
            .iter()
            .filter(|v| v.causality == Causality::StructuralParameter)
    }

    // --- convenience filters (pre-filtered once, iterated many times) ------

    pub fn inputs(&self) -> impl Iterator<Item = &Variable> {
        self.variables
            .iter()
            .filter(|v| v.causality == Causality::Input)
    }

    pub fn outputs(&self) -> impl Iterator<Item = &Variable> {
        self.model_structure
            .outputs
            .iter()
            .filter_map(|vr| self.variable_by_vr(*vr))
    }

    pub fn continuous_state_derivatives(&self) -> impl Iterator<Item = &Variable> {
        self.model_structure
            .continuous_state_derivatives
            .iter()
            .filter_map(|vr| self.variable_by_vr(*vr))
    }

    pub fn event_indicators(&self) -> impl Iterator<Item = &Variable> {
        self.model_structure
            .event_indicators
            .iter()
            .filter_map(|vr| self.variable_by_vr(*vr))
    }

    /// The continuous states, in the order given by `ContinuousStateDerivative`
    /// entries in the ModelStructure (FMI 3.0 §2.4.8). Each derivative variable
    /// has a `derivative="<VR of state>"` attribute that we follow here.
    pub fn continuous_states(&self) -> Vec<&Variable> {
        self.continuous_state_derivatives()
            .filter_map(|d| d.derivative_of.and_then(|vr| self.variable_by_vr(vr)))
            .collect()
    }

    pub fn n_continuous_states(&self) -> usize {
        self.model_structure.continuous_state_derivatives.len()
    }

    pub fn n_event_indicators(&self) -> usize {
        self.model_structure.event_indicators.len()
    }

    // --- writer ------------------------------------------------------------

    /// Serialize to a FMI 3.0 `modelDescription.xml` string. The inverse of
    /// `from_str`: emitting then re-parsing yields an equivalent description
    /// (see the round-trip test). Phase 1 covers the subset the exporter
    /// produces: root attributes, `<ModelExchange>` / `<CoSimulation>`,
    /// `<DefaultExperiment>`, typed `<ModelVariables>`, and `<ModelStructure>`.
    pub fn to_xml(&self) -> String {
        let mut s = String::new();
        s.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");

        // Root element with its attributes.
        s.push_str("<fmiModelDescription\n");
        s.push_str(&format!("  fmiVersion=\"{}\"\n", xml_escape(&self.fmi_version)));
        s.push_str(&format!("  modelName=\"{}\"\n", xml_escape(&self.model_name)));
        s.push_str(&format!(
            "  instantiationToken=\"{}\"",
            xml_escape(&self.instantiation_token)
        ));
        if let Some(d) = &self.description {
            s.push_str(&format!("\n  description=\"{}\"", xml_escape(d)));
        }
        if let Some(t) = &self.generation_tool {
            s.push_str(&format!("\n  generationTool=\"{}\"", xml_escape(t)));
        }
        s.push_str(">\n");

        if let Some(me) = &self.model_exchange {
            s.push_str(&format!(
                "  <ModelExchange modelIdentifier=\"{}\" needsCompletedIntegratorStep=\"{}\" providesDirectionalDerivatives=\"{}\" canGetAndSetFMUState=\"{}\"/>\n",
                xml_escape(&me.model_identifier),
                me.needs_completed_integrator_step,
                me.provides_directional_derivatives,
                me.can_get_and_set_fmu_state,
            ));
        }
        if let Some(cs) = &self.co_simulation {
            s.push_str(&format!(
                "  <CoSimulation modelIdentifier=\"{}\" canHandleVariableCommunicationStepSize=\"{}\"",
                xml_escape(&cs.model_identifier),
                cs.can_handle_variable_communication_step_size,
            ));
            if let Some(fs) = cs.fixed_internal_step_size {
                s.push_str(&format!(" fixedInternalStepSize=\"{}\"", fmt_f64(fs)));
            }
            s.push_str(&format!(
                " hasEventMode=\"{}\" providesIntermediateUpdate=\"{}\" canReturnEarlyAfterIntermediateUpdate=\"{}\" mightReturnEarlyFromDoStep=\"{}\" canGetAndSetFMUState=\"{}\"/>\n",
                cs.has_event_mode,
                cs.provides_intermediate_update,
                cs.can_return_early_after_intermediate_update,
                cs.might_return_early_from_do_step,
                cs.can_get_and_set_fmu_state,
            ));
        }

        // DefaultExperiment (only emit attributes that are set).
        let de = &self.default_experiment;
        if de.start_time.is_some()
            || de.stop_time.is_some()
            || de.tolerance.is_some()
            || de.step_size.is_some()
        {
            s.push_str("  <DefaultExperiment");
            if let Some(v) = de.start_time {
                s.push_str(&format!(" startTime=\"{}\"", fmt_f64(v)));
            }
            if let Some(v) = de.stop_time {
                s.push_str(&format!(" stopTime=\"{}\"", fmt_f64(v)));
            }
            if let Some(v) = de.tolerance {
                s.push_str(&format!(" tolerance=\"{}\"", fmt_f64(v)));
            }
            if let Some(v) = de.step_size {
                s.push_str(&format!(" stepSize=\"{}\"", fmt_f64(v)));
            }
            s.push_str("/>\n");
        }

        // ModelVariables.
        s.push_str("  <ModelVariables>\n");
        for v in &self.variables {
            s.push_str("    <");
            s.push_str(var_type_to_tag(v.var_type));
            s.push_str(&format!(" name=\"{}\"", xml_escape(&v.name)));
            s.push_str(&format!(" valueReference=\"{}\"", v.value_reference));
            s.push_str(&format!(" causality=\"{}\"", v.causality.to_str()));
            s.push_str(&format!(" variability=\"{}\"", v.variability.to_str()));
            if let Some(init) = v.initial {
                s.push_str(&format!(" initial=\"{}\"", init.to_str()));
            }
            if let Some(vr) = v.derivative_of {
                s.push_str(&format!(" derivative=\"{}\"", vr));
            }
            if v.max_output_derivative_order > 0 {
                s.push_str(&format!(
                    " maxOutputDerivativeOrder=\"{}\"",
                    v.max_output_derivative_order
                ));
            }
            // `<String>` and `<Binary>` carry their starts as child elements,
            // never as an attribute (FMI 3.0 §2.4.7.5).
            let start_as_children =
                matches!(v.var_type, VarType::String | VarType::Binary);
            if let Some(start) = &v.start {
                if !start_as_children {
                    s.push_str(&format!(" start=\"{}\"", fmt_start(start)));
                }
            }
            if let Some(d) = &v.description {
                s.push_str(&format!(" description=\"{}\"", xml_escape(d)));
            }

            let start_children: Vec<String> = match (&v.start, start_as_children) {
                (Some(StartValue::String(items)), true) => items
                    .iter()
                    .map(|t| format!("<Start value=\"{}\"/>", xml_escape(t)))
                    .collect(),
                (Some(StartValue::Binary(items)), true) => items
                    .iter()
                    .map(|b| format!("<Start value=\"{}\"/>", to_hex(b)))
                    .collect(),
                _ => Vec::new(),
            };
            if v.dimensions.is_empty() && start_children.is_empty() {
                s.push_str("/>\n");
                continue;
            }
            s.push_str(">\n");
            for d in &v.dimensions {
                match d {
                    Dimension::Fixed(n) => {
                        s.push_str(&format!("      <Dimension start=\"{n}\"/>\n"))
                    }
                    Dimension::Referenced(vr) => {
                        s.push_str(&format!("      <Dimension valueReference=\"{vr}\"/>\n"))
                    }
                }
            }
            for child in &start_children {
                s.push_str(&format!("      {child}\n"));
            }
            s.push_str(&format!("    </{}>\n", var_type_to_tag(v.var_type)));
        }
        s.push_str("  </ModelVariables>\n");

        // ModelStructure.
        s.push_str("  <ModelStructure>\n");
        let ms = &self.model_structure;
        for &vr in &ms.outputs {
            s.push_str(&format!("    <Output valueReference=\"{}\"/>\n", vr));
        }
        for &vr in &ms.continuous_state_derivatives {
            s.push_str(&format!(
                "    <ContinuousStateDerivative valueReference=\"{}\"/>\n",
                vr
            ));
        }
        for &vr in &ms.event_indicators {
            s.push_str(&format!("    <EventIndicator valueReference=\"{}\"/>\n", vr));
        }
        for &vr in &ms.initial_unknowns {
            s.push_str(&format!("    <InitialUnknown valueReference=\"{}\"/>\n", vr));
        }
        s.push_str("  </ModelStructure>\n");

        s.push_str("</fmiModelDescription>\n");
        s
    }
}

/// Format an f64 attribute so it round-trips through `f64::parse`. Rust's
/// default `Display` already emits the shortest round-trippable decimal.
fn fmt_f64(v: f64) -> String {
    format!("{}", v)
}

/// The `start` attribute value: a whitespace-separated list, which collapses to
/// a plain scalar for the single-element case. Only reached for the numeric and
/// boolean types — `<String>` and `<Binary>` starts are emitted as children.
fn fmt_start(start: &StartValue) -> String {
    fn join<T>(items: &[T], f: impl Fn(&T) -> String) -> String {
        items.iter().map(f).collect::<Vec<_>>().join(" ")
    }
    match start {
        StartValue::Float64(v) => join(v, |x| fmt_f64(*x)),
        StartValue::Int64(v) => join(v, |x| x.to_string()),
        StartValue::Boolean(v) => join(v, |x| x.to_string()),
        StartValue::String(v) => join(v, |x| xml_escape(x)),
        StartValue::Binary(v) => join(v, |x| to_hex(x)),
    }
}

// --- element parsers -------------------------------------------------------

fn required_attr<'a>(n: Node<'a, 'a>, name: &str) -> Result<&'a str> {
    n.attribute(name).ok_or_else(|| {
        FmiError::ModelDescription(format!(
            "<{}> missing required attribute {name}",
            n.tag_name().name()
        ))
    })
}

fn parse_bool_attr(n: Node, name: &str) -> bool {
    n.attribute(name).map(parse_xs_bool).unwrap_or(false)
}

fn parse_f64_attr(n: Node, name: &str) -> Option<f64> {
    n.attribute(name).and_then(|s| s.parse().ok())
}

fn parse_me(n: Node) -> Result<ModelExchangeInfo> {
    Ok(ModelExchangeInfo {
        model_identifier: required_attr(n, "modelIdentifier")?.to_owned(),
        needs_completed_integrator_step: n
            .attribute("needsCompletedIntegratorStep")
            .map(|s| s == "true")
            .unwrap_or(true), // default per FMI 3.0 spec
        provides_directional_derivatives: parse_bool_attr(n, "providesDirectionalDerivatives"),
        can_get_and_set_fmu_state: parse_bool_attr(n, "canGetAndSetFMUState"),
    })
}

fn parse_cs(n: Node) -> Result<CoSimulationInfo> {
    Ok(CoSimulationInfo {
        model_identifier: required_attr(n, "modelIdentifier")?.to_owned(),
        can_handle_variable_communication_step_size: parse_bool_attr(
            n,
            "canHandleVariableCommunicationStepSize",
        ),
        fixed_internal_step_size: parse_f64_attr(n, "fixedInternalStepSize"),
        has_event_mode: parse_bool_attr(n, "hasEventMode"),
        provides_intermediate_update: parse_bool_attr(n, "providesIntermediateUpdate"),
        can_return_early_after_intermediate_update: parse_bool_attr(
            n,
            "canReturnEarlyAfterIntermediateUpdate",
        ),
        might_return_early_from_do_step: parse_bool_attr(n, "mightReturnEarlyFromDoStep"),
        can_get_and_set_fmu_state: parse_bool_attr(n, "canGetAndSetFMUState"),
    })
}

fn parse_default_experiment(n: Node) -> DefaultExperiment {
    DefaultExperiment {
        start_time: parse_f64_attr(n, "startTime"),
        stop_time: parse_f64_attr(n, "stopTime"),
        tolerance: parse_f64_attr(n, "tolerance"),
        step_size: parse_f64_attr(n, "stepSize"),
    }
}

fn var_type_from_tag(tag: &str) -> Option<VarType> {
    Some(match tag {
        "Float32" => VarType::Float32,
        "Float64" => VarType::Float64,
        "Int8" => VarType::Int8,
        "UInt8" => VarType::UInt8,
        "Int16" => VarType::Int16,
        "UInt16" => VarType::UInt16,
        "Int32" => VarType::Int32,
        "UInt32" => VarType::UInt32,
        "Int64" => VarType::Int64,
        "UInt64" => VarType::UInt64,
        "Boolean" => VarType::Boolean,
        "String" => VarType::String,
        "Binary" => VarType::Binary,
        "Clock" => VarType::Clock,
        "Enumeration" => VarType::Enumeration,
        _ => return None,
    })
}

fn var_type_to_tag(ty: VarType) -> &'static str {
    match ty {
        VarType::Float32 => "Float32",
        VarType::Float64 => "Float64",
        VarType::Int8 => "Int8",
        VarType::UInt8 => "UInt8",
        VarType::Int16 => "Int16",
        VarType::UInt16 => "UInt16",
        VarType::Int32 => "Int32",
        VarType::UInt32 => "UInt32",
        VarType::Int64 => "Int64",
        VarType::UInt64 => "UInt64",
        VarType::Boolean => "Boolean",
        VarType::String => "String",
        VarType::Binary => "Binary",
        VarType::Clock => "Clock",
        VarType::Enumeration => "Enumeration",
    }
}

/// Escape the five XML predefined entities for safe placement inside an
/// attribute value (double-quoted).
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

fn parse_variables(n: Node) -> Result<Vec<Variable>> {
    let mut out = Vec::new();
    for v in n.children().filter(|n| n.is_element()) {
        let tag = v.tag_name().name();
        let Some(ty) = var_type_from_tag(tag) else {
            continue;
        };
        let name = required_attr(v, "name")?.to_owned();
        let value_reference: fmi3ValueReference = required_attr(v, "valueReference")?
            .parse()
            .map_err(|_| FmiError::ModelDescription(format!("{name}: invalid valueReference")))?;
        let causality = Causality::parse(v.attribute("causality").unwrap_or(""));
        let variability = Variability::parse(v.attribute("variability").unwrap_or(""), ty);
        let initial = v.attribute("initial").and_then(|s| match s {
            "exact" => Some(Initial::Exact),
            "approx" => Some(Initial::Approx),
            "calculated" => Some(Initial::Calculated),
            _ => None,
        });
        let description = v.attribute("description").map(String::from);
        let start = parse_start_value(v, ty);
        let derivative_of: Option<fmi3ValueReference> =
            v.attribute("derivative").and_then(|s| s.parse().ok());
        let max_output_derivative_order: u32 = v
            .attribute("maxOutputDerivativeOrder")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let dimensions = parse_dimensions(v);

        out.push(Variable {
            name,
            value_reference,
            var_type: ty,
            causality,
            variability,
            initial,
            description,
            start,
            derivative_of,
            max_output_derivative_order,
            dimensions,
        });
    }
    Ok(out)
}

/// `<Dimension start="N"/>` / `<Dimension valueReference="vr"/>` children, in
/// document order (FMI 3.0 §2.4.7.4). Elements carrying neither attribute are
/// malformed and skipped rather than treated as size 0.
fn parse_dimensions(n: Node) -> Vec<Dimension> {
    n.children()
        .filter(|c| c.is_element() && c.tag_name().name() == "Dimension")
        .filter_map(|c| {
            if let Some(s) = c.attribute("start") {
                s.parse::<u64>().ok().map(Dimension::Fixed)
            } else {
                c.attribute("valueReference")
                    .and_then(|s| s.parse().ok())
                    .map(Dimension::Referenced)
            }
        })
        .collect()
}

/// Parse a boolean the way the FMI 3.0 XSD does (`xs:boolean` accepts both
/// spellings of each value).
fn parse_xs_bool(s: &str) -> bool {
    s == "true" || s == "1"
}

/// Decode a hex string into bytes, as used by the `value` attribute of a
/// `<Start>` element under a `<Binary>` variable. An odd length or a non-hex
/// digit yields `None`.
fn parse_hex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02X}")).collect()
}

/// Start values, in either of the two forms FMI 3.0 defines (§2.4.7.5):
///
///  - numeric and boolean variables use a `start` attribute holding a
///    whitespace-separated list (a single entry broadcasts across an array);
///  - `<String>` and `<Binary>` variables use a sequence of `<Start value=".."/>`
///    child elements, one per array element.
fn parse_start_value(n: Node, ty: VarType) -> Option<StartValue> {
    if matches!(ty, VarType::String | VarType::Binary) {
        let items: Vec<&str> = n
            .children()
            .filter(|c| c.is_element() && c.tag_name().name() == "Start")
            .filter_map(|c| c.attribute("value"))
            .collect();
        if items.is_empty() {
            return None;
        }
        return Some(match ty {
            VarType::String => StartValue::String(items.iter().map(|s| (*s).to_owned()).collect()),
            _ => StartValue::Binary(
                items
                    .iter()
                    .map(|s| parse_hex(s).unwrap_or_default())
                    .collect(),
            ),
        });
    }

    let s = n.attribute("start")?;
    let items: Vec<&str> = s.split_whitespace().collect();
    if items.is_empty() {
        return None;
    }
    match ty {
        VarType::Float32 | VarType::Float64 => items
            .iter()
            .map(|t| t.parse::<f64>().ok())
            .collect::<Option<Vec<_>>>()
            .map(StartValue::Float64),
        VarType::Int8
        | VarType::UInt8
        | VarType::Int16
        | VarType::UInt16
        | VarType::Int32
        | VarType::UInt32
        | VarType::Int64
        | VarType::UInt64 => items
            .iter()
            .map(|t| t.parse::<i64>().ok())
            .collect::<Option<Vec<_>>>()
            .map(StartValue::Int64),
        VarType::Boolean => Some(StartValue::Boolean(
            items.iter().map(|t| parse_xs_bool(t)).collect(),
        )),
        // Enumeration values are integers on the wire; Clock has no start value.
        VarType::Enumeration => items
            .iter()
            .map(|t| t.parse::<i64>().ok())
            .collect::<Option<Vec<_>>>()
            .map(StartValue::Int64),
        _ => None,
    }
}

fn parse_model_structure(n: Node) -> ModelStructure {
    let mut s = ModelStructure::default();
    for c in n.children().filter(|n| n.is_element()) {
        let vr: Option<fmi3ValueReference> = c.attribute("valueReference").and_then(|s| s.parse().ok());
        let Some(vr) = vr else { continue };
        match c.tag_name().name() {
            "Output" => s.outputs.push(vr),
            "ContinuousStateDerivative" => s.continuous_state_derivatives.push(vr),
            "EventIndicator" => s.event_indicators.push(vr),
            "InitialUnknown" => s.initial_unknowns.push(vr),
            _ => {}
        }
    }
    // dependencies attribute is parsed lazily if needed; phase 1 ignores it.
    s
}

// --- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const DAHLQUIST: &str = include_str!("../../tests/fixtures/fmi/Dahlquist.xml");
    const BOUNCING_BALL: &str = include_str!("../../tests/fixtures/fmi/BouncingBall.xml");

    #[test]
    fn parses_dahlquist() {
        let md = ModelDescription::from_str(DAHLQUIST).unwrap();
        assert_eq!(md.fmi_version, "3.0");
        assert_eq!(md.model_name, "Dahlquist");
        assert!(md.model_exchange.is_some());
        assert!(md.co_simulation.is_some());
        assert_eq!(
            md.model_exchange.as_ref().unwrap().model_identifier,
            "Dahlquist"
        );
        assert_eq!(md.variables.len(), 4);
        assert_eq!(md.n_continuous_states(), 1);
        assert_eq!(md.n_event_indicators(), 0);

        // x is the state; der(x) is the derivative.
        let states = md.continuous_states();
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].name, "x");
        let ders: Vec<_> = md.continuous_state_derivatives().collect();
        assert_eq!(ders.len(), 1);
        assert_eq!(ders[0].name, "der(x)");
        assert_eq!(ders[0].derivative_of, Some(1));
    }

    #[test]
    fn parses_bouncing_ball() {
        let md = ModelDescription::from_str(BOUNCING_BALL).unwrap();
        assert_eq!(md.model_name, "BouncingBall");
        assert_eq!(md.n_continuous_states(), 2);
        assert_eq!(md.n_event_indicators(), 1);

        let cs = md.co_simulation.as_ref().unwrap();
        assert!(cs.has_event_mode);
        assert!(cs.might_return_early_from_do_step);
        assert_eq!(cs.fixed_internal_step_size, Some(1e-3));

        // g has start = -9.81
        let g = md.variable_by_name("g").unwrap();
        assert_eq!(g.start.as_ref().and_then(|s| s.as_f64()), Some(-9.81));
        assert_eq!(g.causality, Causality::Parameter);

        // States ordered by derivative appearance: h (vr=1), v (vr=3)
        let states = md.continuous_states();
        assert_eq!(states.iter().map(|v| v.name.as_str()).collect::<Vec<_>>(),
                   vec!["h", "v"]);
    }

    #[test]
    fn rejects_fmi_2_0() {
        let err =
            ModelDescription::from_str(r#"<fmiModelDescription fmiVersion="2.0" modelName="x" instantiationToken="t"/>"#)
                .unwrap_err();
        assert!(matches!(err, FmiError::UnsupportedFmiVersion(_)));
    }

    #[test]
    fn lookups() {
        let md = ModelDescription::from_str(DAHLQUIST).unwrap();
        assert_eq!(md.variable_by_name("x").unwrap().value_reference, 1);
        assert_eq!(md.variable_by_vr(3).unwrap().name, "k");
    }

    // --- array variables ---------------------------------------------------

    /// Both `<Dimension>` forms in one description: a constant size, a size read
    /// from a structural parameter, and a two-dimensional array combining them.
    /// Modelled on the Reference-FMUs' `StateSpace` and PMSF's `DynamicArrayTest`.
    const ARRAYS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<fmiModelDescription fmiVersion="3.0" modelName="Arrays" instantiationToken="t">
  <ModelVariables>
    <UInt64 name="n" valueReference="1" causality="structuralParameter" variability="tunable" start="4"/>
    <Float64 name="scalar" valueReference="2" causality="parameter" variability="tunable" start="2.5"/>
    <Float64 name="fixed" valueReference="3" causality="parameter" variability="tunable" start="1 2 3">
      <Dimension start="3"/>
    </Float64>
    <Float64 name="dynamic" valueReference="4" causality="input" start="7">
      <Dimension valueReference="1"/>
    </Float64>
    <Float64 name="matrix" valueReference="5" causality="parameter" variability="tunable" start="0.5">
      <Dimension start="2"/>
      <Dimension valueReference="1"/>
    </Float64>
    <String name="label" valueReference="6" causality="parameter" variability="tunable">
      <Start value="alpha"/>
      <Start value="beta"/>
    </String>
    <Binary name="blob" valueReference="7" causality="parameter" variability="tunable">
      <Start value="BEEF"/>
    </Binary>
  </ModelVariables>
  <ModelStructure/>
</fmiModelDescription>
"#;

    #[test]
    fn parses_both_dimension_forms() {
        let md = ModelDescription::from_str(ARRAYS).unwrap();
        let dims = |name: &str| md.variable_by_name(name).unwrap().dimensions.clone();

        assert!(dims("scalar").is_empty());
        assert_eq!(dims("fixed"), vec![Dimension::Fixed(3)]);
        assert_eq!(dims("dynamic"), vec![Dimension::Referenced(1)]);
        assert_eq!(
            dims("matrix"),
            vec![Dimension::Fixed(2), Dimension::Referenced(1)]
        );
        assert!(!md.variable_by_name("scalar").unwrap().is_array());
        assert!(md.variable_by_name("fixed").unwrap().is_array());
    }

    #[test]
    fn n_values_is_the_product_of_the_dimension_sizes() {
        let md = ModelDescription::from_str(ARRAYS).unwrap();
        let n = |name: &str| md.n_values(md.variable_by_name(name).unwrap());

        assert_eq!(n("scalar"), 1, "a scalar is the empty product");
        assert_eq!(n("fixed"), 3);
        assert_eq!(n("dynamic"), 4, "resolved through structural parameter n=4");
        assert_eq!(n("matrix"), 2 * 4);

        let vars = ["fixed", "dynamic"].map(|s| md.variable_by_name(s).unwrap());
        assert_eq!(md.total_n_values(vars.into_iter()), 7);
    }

    #[test]
    fn structural_parameter_change_resizes_dependent_arrays() {
        let mut md = ModelDescription::from_str(ARRAYS).unwrap();
        md.set_dimension_size(1, 2);

        assert_eq!(md.n_values(md.variable_by_name("dynamic").unwrap()), 2);
        assert_eq!(md.n_values(md.variable_by_name("matrix").unwrap()), 2 * 2);
        // A constant dimension is unaffected by the structural parameter.
        assert_eq!(md.n_values(md.variable_by_name("fixed").unwrap()), 3);

        // A dimension may legitimately go to zero (FMI 3.0 §2.4.7.4).
        md.set_dimension_size(1, 0);
        assert_eq!(md.n_values(md.variable_by_name("dynamic").unwrap()), 0);

        assert_eq!(
            md.structural_parameters().map(|v| v.name.as_str()).collect::<Vec<_>>(),
            vec!["n"]
        );
    }

    #[test]
    fn start_values_parse_as_lists_and_broadcast() {
        let md = ModelDescription::from_str(ARRAYS).unwrap();
        let start = |name: &str| md.variable_by_name(name).unwrap().start.clone().unwrap();

        // A list start is kept element by element.
        assert_eq!(start("fixed"), StartValue::Float64(vec![1.0, 2.0, 3.0]));
        assert_eq!(start("fixed").expand_f64(3).unwrap(), vec![1.0, 2.0, 3.0]);

        // A single value fills the whole array (FMI 3.0 §2.4.7.5).
        assert_eq!(start("dynamic"), StartValue::Float64(vec![7.0]));
        assert_eq!(start("dynamic").expand_f64(4).unwrap(), vec![7.0; 4]);
        assert_eq!(start("matrix").expand_f64(8).unwrap(), vec![0.5; 8]);

        // Scalars still behave like one-element arrays.
        assert_eq!(start("scalar").as_f64(), Some(2.5));
        assert_eq!(start("scalar").expand_f64(1).unwrap(), vec![2.5]);
    }

    #[test]
    fn string_and_binary_starts_come_from_child_elements() {
        let md = ModelDescription::from_str(ARRAYS).unwrap();
        assert_eq!(
            md.variable_by_name("label").unwrap().start,
            Some(StartValue::String(vec!["alpha".into(), "beta".into()]))
        );
        assert_eq!(
            md.variable_by_name("blob").unwrap().start,
            Some(StartValue::Binary(vec![vec![0xBE, 0xEF]]))
        );
    }

    #[test]
    fn round_trips_array_variables() {
        // The writer has to emit `<Dimension>` and `<Start>` children, which
        // means the variable element stops being self-closing.
        let a = ModelDescription::from_str(ARRAYS).unwrap();
        let b = ModelDescription::from_str(&a.to_xml()).unwrap();

        for name in ["scalar", "fixed", "dynamic", "matrix", "label", "blob"] {
            let va = a.variable_by_name(name).unwrap();
            let vb = b.variable_by_name(name).unwrap();
            assert_eq!(va.dimensions, vb.dimensions, "{name} dimensions");
            assert_eq!(va.start, vb.start, "{name} start");
            assert_eq!(a.n_values(va), b.n_values(vb), "{name} n_values");
        }
    }

    // --- writer round-trips ------------------------------------------------

    /// Parse a fixture, serialize it back out, re-parse, and assert the
    /// description survives the round-trip. This pins the writer as the exact
    /// inverse of the parser over the supported subset.
    fn assert_round_trip(xml: &str) {
        let a = ModelDescription::from_str(xml).unwrap();
        let b = ModelDescription::from_str(&a.to_xml()).unwrap();

        assert_eq!(a.fmi_version, b.fmi_version);
        assert_eq!(a.model_name, b.model_name);
        assert_eq!(a.instantiation_token, b.instantiation_token);
        assert_eq!(a.n_continuous_states(), b.n_continuous_states());
        assert_eq!(a.n_event_indicators(), b.n_event_indicators());
        assert_eq!(a.variables.len(), b.variables.len());
        assert_eq!(
            a.model_exchange.is_some(),
            b.model_exchange.is_some()
        );
        assert_eq!(a.co_simulation.is_some(), b.co_simulation.is_some());

        for va in &a.variables {
            let vb = b.variable_by_name(&va.name).unwrap();
            assert_eq!(va.value_reference, vb.value_reference);
            assert_eq!(va.var_type, vb.var_type);
            assert_eq!(va.causality, vb.causality);
            assert_eq!(va.variability, vb.variability);
            assert_eq!(va.initial, vb.initial);
            assert_eq!(va.derivative_of, vb.derivative_of);
            assert_eq!(
                va.start.as_ref().and_then(|s| s.as_f64()),
                vb.start.as_ref().and_then(|s| s.as_f64()),
            );
        }

        assert_eq!(a.model_structure.outputs, b.model_structure.outputs);
        assert_eq!(
            a.model_structure.continuous_state_derivatives,
            b.model_structure.continuous_state_derivatives
        );
        assert_eq!(
            a.model_structure.event_indicators,
            b.model_structure.event_indicators
        );
    }

    #[test]
    fn round_trips_dahlquist() {
        assert_round_trip(DAHLQUIST);
    }

    #[test]
    fn round_trips_bouncing_ball() {
        assert_round_trip(BOUNCING_BALL);
    }

    #[test]
    fn writer_emits_constructed_model() {
        // Build a minimal ME model by hand (the export path) and check the
        // emitted XML re-parses into the same shape.
        let f64_var = |name: &str, vr, causality, variability| {
            Variable::scalar(name, vr, VarType::Float64, causality, variability)
        };
        let vars = vec![
            f64_var(
                "time",
                0,
                Causality::Independent,
                Variability::Continuous,
            ),
            Variable {
                initial: Some(Initial::Exact),
                description: Some("state & output".into()),
                start: Some(StartValue::scalar_f64(1.0)),
                ..f64_var("x", 1, Causality::Output, Variability::Continuous)
            },
            Variable {
                initial: Some(Initial::Calculated),
                derivative_of: Some(1),
                ..f64_var("der(x)", 2, Causality::Local, Variability::Continuous)
            },
        ];
        let md = ModelDescription::new(
            "Test",
            "{fastsim-test}",
            Some(ModelExchangeInfo {
                model_identifier: "Test".into(),
                needs_completed_integrator_step: false,
                provides_directional_derivatives: false,
                can_get_and_set_fmu_state: false,
            }),
            None,
            DefaultExperiment {
                start_time: Some(0.0),
                stop_time: Some(10.0),
                tolerance: Some(1e-6),
                step_size: None,
            },
            vars,
            ModelStructure {
                outputs: vec![1],
                continuous_state_derivatives: vec![2],
                event_indicators: vec![],
                initial_unknowns: vec![1],
            },
        );

        let reparsed = ModelDescription::from_str(&md.to_xml()).unwrap();
        assert_eq!(reparsed.model_name, "Test");
        assert_eq!(reparsed.fmi_version, "3.0");
        assert_eq!(reparsed.n_continuous_states(), 1);
        assert!(reparsed.model_exchange.is_some());
        assert_eq!(reparsed.continuous_states()[0].name, "x");
        let x = reparsed.variable_by_name("x").unwrap();
        assert_eq!(x.start.as_ref().and_then(|s| s.as_f64()), Some(1.0));
        assert_eq!(x.description.as_deref(), Some("state & output"));
        assert_eq!(reparsed.default_experiment.stop_time, Some(10.0));
    }
}
