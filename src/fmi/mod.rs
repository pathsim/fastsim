// FMI 3.0 FMU import — Model Exchange and Co-Simulation
//
// Self-contained Rust-native implementation. Draws design inspiration from:
//   - rust-fmi (jondo2010)       — Instance<Tag> type-state, platform detection
//   - Reference-FMUs (Modelica)  — ME/CS lifecycle, logger callback bridging
//   - FMPy (CATIA-Systems)       — start-values mapping, resourcePath URI
//   - FMIL (Modelon)             — convenience list accessors

pub mod bindings;
pub mod callbacks;
// FMU *export* (source-FMU generation) needs the C code generator and the
// struct-API layout, so it is additionally gated behind the `codegen` feature.
#[cfg(feature = "codegen")]
pub mod export;
pub mod instance;
pub mod model_description;
pub mod platform;
pub mod unzip;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum FmiError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("ZIP error: {0}")]
    Zip(#[from] zip::result::ZipError),

    #[error("XML parse error: {0}")]
    Xml(#[from] roxmltree::Error),

    #[error("libloading error: {0}")]
    Loading(#[from] libloading::Error),

    #[error("unsupported platform: no binary for {tuple} in FMU (available: {available:?})")]
    UnsupportedPlatform { tuple: String, available: Vec<String> },

    #[error("invalid FMU archive: {0}")]
    InvalidArchive(String),

    #[error("FMI {0} is not supported (only FMI 3.0)")]
    UnsupportedFmiVersion(String),

    #[error("modelDescription.xml: {0}")]
    ModelDescription(String),

    #[error("FMI call {call} returned status {status:?}")]
    FmiStatus { call: &'static str, status: FmiStatus },

    #[error("variable not found: {0}")]
    UnknownVariable(String),

    #[error("FMU export: {0}")]
    Export(String),
}

pub type Result<T> = std::result::Result<T, FmiError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum FmiStatus {
    Ok = 0,
    Warning = 1,
    Discard = 2,
    Error = 3,
    Fatal = 4,
}

impl FmiStatus {
    pub fn from_raw(v: i32) -> Self {
        match v {
            0 => Self::Ok,
            1 => Self::Warning,
            2 => Self::Discard,
            3 => Self::Error,
            _ => Self::Fatal,
        }
    }

    pub fn is_ok_or_warning(self) -> bool {
        matches!(self, Self::Ok | Self::Warning)
    }
}
