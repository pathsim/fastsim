//! FMI 3.0 *source* FMU export (Model Exchange + Co-Simulation).
//!
//! A fastsim model is compiled to the struct-API C (`codegen` with
//! `ModelApi::Struct`), wrapped in a thin FMI 3.0 Model-Exchange C layer, and
//! packaged with a generated `modelDescription.xml` into a `.fmu` (a zip). The
//! result is a *source* FMU: it ships the C sources plus a `buildDescription.xml`
//! and the vendored FMI header, so an importer compiles it on its own platform
//! (no prebuilt binaries, no toolchain assumptions here).
//!
//! The exporter is the mirror image of the importer in this module: value
//! references, variable names, and the `<ModelStructure>` are all derived from
//! `codegen::struct_layout`, so the generated C and the `modelDescription.xml`
//! agree by construction (see [`vrmap`]).
//!
//! Scope: models with continuous state. Zero-crossing, condition and scheduled
//! events are exported too — Model Exchange hands the indicators to the importer,
//! Co-Simulation resolves them inside `fmi3DoStep`. A model with no continuous
//! state is rejected up front.

pub mod headers;
pub mod package;
pub mod vrmap;
pub mod wrapper;

use std::path::Path;

use crate::codegen::{self, CodegenOptions, ModelApi};
use crate::fmi::model_description::{
    CoSimulationInfo, DefaultExperiment, ModelDescription, ModelExchangeInfo,
};
use crate::fmi::{FmiError, Result};
use crate::ir::schema::Module;

/// Knobs for [`export_fmu`]. All optional: the model name defaults to the IR
/// module name, the instantiation token to a deterministic `{fastsim-<id>}`,
/// and the default experiment fields are emitted only when set.
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// `modelName` attribute; defaults to the module's name.
    pub model_name: Option<String>,
    /// `instantiationToken`; defaults to `{fastsim-<modelIdentifier>}`.
    pub instantiation_token: Option<String>,
    pub start_time: Option<f64>,
    pub stop_time: Option<f64>,
    pub tolerance: Option<f64>,
    pub step_size: Option<f64>,
    /// Solver baked into the emitted C. Model Exchange never runs it — the
    /// importer integrates — but Co-Simulation does, so this is what the FMU
    /// will actually solve with. `None` leaves the codegen default (RK4), and
    /// `None` from a caller that *has* a solver means the FMU is exported
    /// without the Co-Simulation interface rather than with the wrong one.
    pub solver: Option<codegen::SolverChoice>,
    /// LTE tolerances for the emitted adaptive step controller, inherited from
    /// the source simulation so the FMU runs the same error budget.
    pub tolerances: Option<codegen::Tolerances>,
    /// Advertise `<CoSimulation>`. Off when the source solver has no emittable
    /// equivalent: an FMU that claims Co-Simulation must integrate with the
    /// solver it was built from, not silently with a different one.
    pub co_simulation: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            model_name: None,
            instantiation_token: None,
            start_time: None,
            stop_time: None,
            tolerance: None,
            step_size: None,
            solver: None,
            tolerances: None,
            co_simulation: true,
        }
    }
}

/// One file in the FMU layout: an archive path (e.g. `"sources/fmu.c"`) and its
/// text contents. [`fmu_files`] returns the full set; [`export_fmu_bytes`] zips
/// it. Exposed so callers (and tests) can inspect or compile the sources without
/// unpacking a zip.
#[derive(Debug, Clone)]
pub struct FmuFile {
    pub name: String,
    pub contents: String,
}

/// Build every file of the source FMU (FMI 3.0, Model Exchange) for `module`:
/// `modelDescription.xml` at the root and the C sources under `sources/`. This
/// is the single place the artifacts are assembled; [`export_fmu_bytes`] just
/// zips the result.
pub fn fmu_files(module: &Module, opts: &ExportOptions) -> Result<Vec<FmuFile>> {
    // The struct API is the FMI substrate: a `model_t` with `model_init` /
    // `model_deriv` / `*_get_signal` / `*_set_signal`.
    let mut cg = CodegenOptions { api: ModelApi::Struct, ..Default::default() };
    if let Some(solver) = opts.solver {
        cg.solver = solver;
    }
    if let Some(tol) = opts.tolerances {
        cg.tolerances = tol;
    }

    let layout = codegen::struct_layout(module, &cg).map_err(cg_err)?;
    let events = codegen::event_layout(module, &cg).map_err(cg_err)?;
    // `model_identifier` must be a C identifier (matches the wrapper's struct
    // prefix); `struct_layout` already sanitized it.
    let model_identifier = layout.name.clone();
    let model_name = opts.model_name.clone().unwrap_or_else(|| module.name.clone());
    let token = opts
        .instantiation_token
        .clone()
        .unwrap_or_else(|| format!("{{fastsim-{model_identifier}}}"));

    // Value references + variables + structure, derived from `layout` (+ one
    // event indicator per state event).
    let fmi = vrmap::build(&layout, events.n_indicators());

    // The generated struct-API C. Files carry the model name (`<base>.h` /
    // `<base>.c`, see `codegen::file_base`) so two exported models unzipped
    // into one workspace don't collide; the wrapper includes `<base>.c` and
    // the FMU packages the sources under the same names.
    let base = codegen::file_base(&module.name);
    let files = codegen::generate(module, &cg).map_err(cg_err)?;
    let model_h = file_named(&files, &format!("{base}.h"))?;
    let model_c = file_named(&files, &format!("{base}.c"))?;

    // The FMI ME wrapper over that C. Directional derivatives are advertised and
    // wired only when codegen emitted an analytic `model_jvp`; the event
    // interface is generated from `events`.
    let fmu_c = wrapper::emit_wrapper(
        &model_identifier,
        &base,
        &fmi.vr,
        &token,
        layout.jvp,
        &events,
        fmi.vr.ind_base,
        opts.step_size,
    );

    // The modelDescription.xml. `needsCompletedIntegratorStep` is required for
    // state events (the FMU records the previous indicator values there).
    let md = ModelDescription::new(
        model_name,
        token,
        Some(ModelExchangeInfo {
            model_identifier: model_identifier.clone(),
            needs_completed_integrator_step: events.n_indicators() > 0,
            provides_directional_derivatives: layout.jvp,
            can_get_and_set_fmu_state: false,
        }),
        // Co-Simulation as well: the generated C already integrates itself
        // (`<name>_step`), so `fmi3DoStep` is a loop over it. One FMU offering
        // both interfaces is what the Reference-FMUs do, and it lets an importer
        // pick whichever fits — its own solver, or ours.
        opts.co_simulation.then(|| CoSimulationInfo {
            model_identifier: model_identifier.clone(),
            // The internal step is subdivided to fit whatever the master asks for.
            can_handle_variable_communication_step_size: true,
            fixed_internal_step_size: None,
            // Events are resolved inside `fmi3DoStep`, so the master is never
            // asked to enter Event Mode and never needs an early return.
            has_event_mode: false,
            provides_intermediate_update: false,
            can_return_early_after_intermediate_update: false,
            might_return_early_from_do_step: false,
            can_get_and_set_fmu_state: false,
        }),
        DefaultExperiment {
            start_time: opts.start_time,
            stop_time: opts.stop_time,
            tolerance: opts.tolerance,
            step_size: opts.step_size,
        },
        fmi.variables,
        fmi.model_structure,
    );

    let f = |name: &str, contents: String| FmuFile { name: name.into(), contents };
    Ok(vec![
        f("modelDescription.xml", md.to_xml()),
        f("sources/buildDescription.xml", package::build_description_xml(&model_identifier)),
        f("sources/fmi3.h", headers::FMI3_HEADER.to_string()),
        f(&format!("sources/{base}.h"), model_h),
        f(&format!("sources/{base}.c"), model_c),
        f("sources/fmu.c", fmu_c),
    ])
}

/// Build a source FMU (FMI 3.0, Model Exchange) for `module` and return its
/// `.fmu` bytes. See [`export_fmu`] to write it to a path.
pub fn export_fmu_bytes(module: &Module, opts: &ExportOptions) -> Result<Vec<u8>> {
    let entries: Vec<(String, Vec<u8>)> = fmu_files(module, opts)?
        .into_iter()
        .map(|f| (f.name, f.contents.into_bytes()))
        .collect();
    package::zip_fmu(&entries)
}

/// Build a source FMU for `module` and write it to `path` (conventionally
/// `*.fmu`).
pub fn export_fmu(module: &Module, path: impl AsRef<Path>, opts: &ExportOptions) -> Result<()> {
    let bytes = export_fmu_bytes(module, opts)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

fn cg_err(e: codegen::CodegenError) -> FmiError {
    FmiError::Export(format!("codegen: {e}"))
}

/// Pull one generated file's contents out by name (the struct API always emits
/// `model.h` and `model.c`).
fn file_named(files: &[codegen::GeneratedFile], name: &str) -> Result<String> {
    files
        .iter()
        .find(|f| f.name == name)
        .map(|f| f.contents.clone())
        .ok_or_else(|| FmiError::Export(format!("codegen did not emit {name}")))
}
