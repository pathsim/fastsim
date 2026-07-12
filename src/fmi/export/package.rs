// `.fmu` packaging: the `buildDescription.xml` and the zip assembly.
//
// A source FMU is a zip with `modelDescription.xml` at the root and the C
// sources under `sources/`, including a `buildDescription.xml` that tells the
// importer which translation units to compile. We list only `fmu.c` (the
// wrapper), which `#include`s `model.c`, so the whole model is one TU.

use std::io::{Cursor, Write};

use crate::fmi::Result;

/// The `sources/buildDescription.xml`: a single source-file set compiling
/// `fmu.c`. `model_identifier` must match the `<ModelExchange modelIdentifier>`
/// in `modelDescription.xml` (FMI 3.0 §2.4.2).
pub fn build_description_xml(model_identifier: &str) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <fmiBuildDescription fmiVersion=\"3.0\">\n\
         \x20 <BuildConfiguration modelIdentifier=\"{model_identifier}\">\n\
         \x20   <SourceFileSet language=\"C99\">\n\
         \x20     <SourceFile name=\"fmu.c\"/>\n\
         \x20   </SourceFileSet>\n\
         \x20 </BuildConfiguration>\n\
         </fmiBuildDescription>\n"
    )
}

/// Zip a set of (archive-path, bytes) entries into an in-memory `.fmu`. Entries
/// are stored (no compression) for simplicity and to match how the importer's
/// fixtures are built; an FMU is a plain zip either way.
pub fn zip_fmu(entries: &[(String, Vec<u8>)]) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    {
        let mut zw = zip::ZipWriter::new(Cursor::new(&mut buf));
        let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for (name, data) in entries {
            zw.start_file(name, opts)?;
            zw.write_all(data)?;
        }
        zw.finish()?;
    }
    Ok(buf)
}
