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
}

#[derive(Debug, Clone)]
pub enum StartValue {
    Float64(f64),
    Int64(i64),
    Boolean(bool),
    String(String),
}

impl StartValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float64(v) => Some(*v),
            Self::Int64(v) => Some(*v as f64),
            Self::Boolean(v) => Some(if *v { 1.0 } else { 0.0 }),
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
        })
    }

    // --- lookups -----------------------------------------------------------

    pub fn variable_by_name(&self, name: &str) -> Option<&Variable> {
        self.name_to_index.get(name).map(|&i| &self.variables[i])
    }

    pub fn variable_by_vr(&self, vr: fmi3ValueReference) -> Option<&Variable> {
        self.vr_to_index.get(&vr).map(|&i| &self.variables[i])
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
            if let Some(start) = &v.start {
                s.push_str(&format!(" start=\"{}\"", fmt_start(start)));
            }
            if let Some(d) = &v.description {
                s.push_str(&format!(" description=\"{}\"", xml_escape(d)));
            }
            s.push_str("/>\n");
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

fn fmt_start(start: &StartValue) -> String {
    match start {
        StartValue::Float64(v) => fmt_f64(*v),
        StartValue::Int64(v) => v.to_string(),
        StartValue::Boolean(v) => v.to_string(),
        StartValue::String(s) => xml_escape(s),
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
    n.attribute(name) == Some("true")
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
        });
    }
    Ok(out)
}

fn parse_start_value(n: Node, ty: VarType) -> Option<StartValue> {
    let s = n.attribute("start")?;
    match ty {
        VarType::Float32 | VarType::Float64 => s.parse::<f64>().ok().map(StartValue::Float64),
        VarType::Int8
        | VarType::UInt8
        | VarType::Int16
        | VarType::UInt16
        | VarType::Int32
        | VarType::UInt32
        | VarType::Int64
        | VarType::UInt64 => s.parse::<i64>().ok().map(StartValue::Int64),
        VarType::Boolean => Some(StartValue::Boolean(s == "true" || s == "1")),
        VarType::String => Some(StartValue::String(s.to_owned())),
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
        let vars = vec![
            Variable {
                name: "time".into(),
                value_reference: 0,
                var_type: VarType::Float64,
                causality: Causality::Independent,
                variability: Variability::Continuous,
                initial: None,
                description: None,
                start: None,
                derivative_of: None,
                max_output_derivative_order: 0,
            },
            Variable {
                name: "x".into(),
                value_reference: 1,
                var_type: VarType::Float64,
                causality: Causality::Output,
                variability: Variability::Continuous,
                initial: Some(Initial::Exact),
                description: Some("state & output".into()),
                start: Some(StartValue::Float64(1.0)),
                derivative_of: None,
                max_output_derivative_order: 0,
            },
            Variable {
                name: "der(x)".into(),
                value_reference: 2,
                var_type: VarType::Float64,
                causality: Causality::Local,
                variability: Variability::Continuous,
                initial: Some(Initial::Calculated),
                description: None,
                start: None,
                derivative_of: Some(1),
                max_output_derivative_order: 0,
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
