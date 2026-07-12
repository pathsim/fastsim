// FMU archive extraction
//
// An `.fmu` is a ZIP containing `modelDescription.xml`, `binaries/{platform}/...`,
// and optionally `resources/`. We extract to a `TempDir` whose cleanup is tied to
// the `FmuArchive`'s lifetime (RAII).
//
// Path validation rejects absolute paths, parent-directory traversal (`..`), and
// Windows-style drive letters — same rules as fmpy `__init__.py:213-219`.

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

use tempfile::TempDir;
use zip::ZipArchive;

use super::{FmiError, Result};

/// An extracted FMU archive. The temporary directory is removed when this is dropped.
pub struct FmuArchive {
    dir: TempDir,
}

impl FmuArchive {
    /// Extract the given `.fmu` file to a new temporary directory.
    pub fn extract(fmu_path: impl AsRef<Path>) -> Result<Self> {
        let fmu_path = fmu_path.as_ref();
        let dir = tempfile::Builder::new().prefix("fastsim-fmu-").tempdir()?;
        let file = File::open(fmu_path)?;
        let mut archive = ZipArchive::new(file)?;

        for i in 0..archive.len() {
            let mut entry = archive.by_index(i)?;
            let raw = entry
                .enclosed_name()
                .ok_or_else(|| FmiError::InvalidArchive(format!("unsafe path: {}", entry.name())))?
                .to_owned();
            validate_path(&raw)?;
            let out_path = dir.path().join(&raw);

            if entry.is_dir() {
                fs::create_dir_all(&out_path)?;
                continue;
            }
            if let Some(parent) = out_path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut out = File::create(&out_path)?;
            io::copy(&mut entry, &mut out)?;
        }

        Ok(Self { dir })
    }

    /// Root directory of the extracted archive.
    pub fn root(&self) -> &Path {
        self.dir.path()
    }

    /// Path to `modelDescription.xml`.
    pub fn model_description(&self) -> PathBuf {
        self.root().join("modelDescription.xml")
    }

    /// Path to the `binaries/{platform_tuple}/` directory (may not exist).
    pub fn binaries_dir(&self, platform_tuple: &str) -> PathBuf {
        self.root().join("binaries").join(platform_tuple)
    }

    /// List the platform tuples that the FMU ships binaries for (directory names
    /// under `binaries/`). Returns an empty vec if `binaries/` is missing.
    pub fn available_platforms(&self) -> Vec<String> {
        let binaries = self.root().join("binaries");
        let Ok(rd) = fs::read_dir(&binaries) else {
            return Vec::new();
        };
        rd.filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .filter_map(|e| e.file_name().into_string().ok())
            .collect()
    }

    /// Absolute path to the `resources/` directory, with a trailing slash, as
    /// required by FMI 3.0.2 §4.2.1 for `resourcePath` in `fmi3Instantiate*`.
    /// Returns `None` when the FMU has no resources directory.
    ///
    /// Note: FMI 3.0 expects a plain path string (e.g. `/tmp/foo/resources/`),
    /// not a `file://` URI. Reference-FMUs concatenate a filename onto this
    /// string directly, so the trailing slash is mandatory.
    pub fn resource_uri(&self) -> Option<String> {
        let p = self.root().join("resources");
        if !p.exists() {
            return None;
        }
        let abs = p.canonicalize().ok()?;
        let mut s = abs.to_string_lossy().into_owned();
        // On Windows, `canonicalize` returns an extended-length path with the
        // `\\?\` verbatim prefix. Under that prefix Win32 does NOT translate
        // forward slashes, so when an FMU concatenates a filename with `/` onto
        // resourcePath (the Reference-FMUs do exactly that), the resulting
        // `...\resources/y.txt` mixes separators and `fopen` fails. Strip the
        // prefix (handling the `\\?\UNC\` network form) and normalize to forward
        // slashes so the joined path is a clean, openable string.
        #[cfg(windows)]
        {
            s = if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
                format!(r"\\{rest}")
            } else if let Some(rest) = s.strip_prefix(r"\\?\") {
                rest.to_string()
            } else {
                s
            };
            s = s.replace('\\', "/");
        }
        if !s.ends_with('/') {
            s.push('/');
        }
        Some(s)
    }
}

fn validate_path(p: &Path) -> Result<()> {
    if p.is_absolute() {
        return Err(FmiError::InvalidArchive(format!(
            "absolute path in archive: {}",
            p.display()
        )));
    }
    for comp in p.components() {
        use std::path::Component;
        match comp {
            Component::ParentDir => {
                return Err(FmiError::InvalidArchive(format!(
                    "parent-dir traversal in archive: {}",
                    p.display()
                )));
            }
            Component::Prefix(_) | Component::RootDir => {
                return Err(FmiError::InvalidArchive(format!(
                    "prefixed/root path in archive: {}",
                    p.display()
                )));
            }
            _ => {}
        }
    }
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_fmu_zip(entries: &[(&str, &[u8])]) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new().suffix(".fmu").tempfile().unwrap();
        let mut zw = zip::ZipWriter::new(f.as_file_mut());
        let opts: zip::write::SimpleFileOptions =
            zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        for (name, data) in entries {
            zw.start_file(*name, opts).unwrap();
            zw.write_all(data).unwrap();
        }
        zw.finish().unwrap();
        f
    }

    #[test]
    fn extracts_modeldescription_and_binary() {
        let fmu = make_fmu_zip(&[
            ("modelDescription.xml", b"<fmiModelDescription/>"),
            ("binaries/aarch64-darwin/Foo.dylib", b"\x7FELF"),
        ]);
        let arch = match FmuArchive::extract(fmu.path()) {
            Ok(a) => a,
            Err(e) => panic!("extract failed: {e}"),
        };
        assert!(arch.model_description().exists());
        assert!(arch.binaries_dir("aarch64-darwin").exists());
        assert_eq!(arch.available_platforms(), vec!["aarch64-darwin"]);
    }

    #[test]
    fn rejects_parent_traversal() {
        let fmu = make_fmu_zip(&[("../evil.txt", b"x")]);
        // Either `enclosed_name` strips it to `None` (InvalidArchive) or our
        // validator rejects — both map to InvalidArchive.
        let err = match FmuArchive::extract(fmu.path()) {
            Ok(_) => panic!("expected extraction to fail"),
            Err(e) => e,
        };
        assert!(matches!(err, FmiError::InvalidArchive(_)), "got {err:?}");
    }

    #[test]
    fn resource_uri_present_when_dir_exists() {
        let fmu = make_fmu_zip(&[
            ("modelDescription.xml", b"<x/>"),
            ("resources/data.csv", b"a,b\n"),
        ]);
        let arch = match FmuArchive::extract(fmu.path()) {
            Ok(a) => a,
            Err(e) => panic!("extract failed: {e}"),
        };
        let uri = arch.resource_uri().unwrap();
        // Cross-platform: on Linux this is `/.../resources/`, on Windows a
        // canonicalized `\\?\C:\...\resources/`. Assert it is absolute and
        // points at the resources dir without baking in the `/` separator.
        assert!(
            std::path::Path::new(&uri).is_absolute(),
            "expected absolute path, got {uri}"
        );
        assert!(
            uri.trim_end_matches(['/', '\\']).ends_with("resources"),
            "uri = {uri}"
        );
    }
}
