//! Jinja template environment for structural code emission.
//!
//! The templates (`templates/*.c.jinja`) own the structural shape of the
//! generated code: function signatures, the integrator loops, the file
//! assembly. The recursive op->expression lowering stays in Rust (the `Target`
//! trait) and is fed in as data (statement and write lists). Templates are
//! embedded at compile time, so the crate stays self-contained (no runtime file
//! dependency, WASM/Pyodide-friendly).

use std::sync::LazyLock;

use minijinja::Environment;
use serde::Serialize;

use super::{CodegenError, R};

static ENV: LazyLock<Environment<'static>> = LazyLock::new(|| {
    let mut env = Environment::new();
    // Jinja2-style whitespace control: drop the newline after a block tag and
    // the leading whitespace before it, so the templates can be indented for
    // readability without that indentation leaking into the generated C.
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.add_template("region.c", include_str!("templates/region.c.jinja"))
        .expect("region.c.jinja must parse");
    env.add_template("model_struct.c", include_str!("templates/model_struct.c.jinja"))
        .expect("model_struct.c.jinja must parse");
    env.add_template("model_struct.h", include_str!("templates/model_struct.h.jinja"))
        .expect("model_struct.h.jinja must parse");
    env.add_template("solver_impl_struct.c", include_str!("templates/solver_impl_struct.c.jinja"))
        .expect("solver_impl_struct.c.jinja must parse");
    env.add_template("solver_struct.c", include_str!("templates/solver_struct.c.jinja"))
        .expect("solver_struct.c.jinja must parse");
    env.add_template("solver_struct.h", include_str!("templates/solver_struct.h.jinja"))
        .expect("solver_struct.h.jinja must parse");
    env.add_template("blocks_struct.c", include_str!("templates/blocks_struct.c.jinja"))
        .expect("blocks_struct.c.jinja must parse");
    env.add_template("blocks_struct.h", include_str!("templates/blocks_struct.h.jinja"))
        .expect("blocks_struct.h.jinja must parse");
    env.add_template("scaffold_main.c", include_str!("templates/scaffold_main.c.jinja"))
        .expect("scaffold_main.c.jinja must parse");
    env.add_template("scaffold_cmake", include_str!("templates/scaffold_cmake.jinja"))
        .expect("scaffold_cmake.jinja must parse");
    env.add_template("a2l", include_str!("templates/a2l.jinja"))
        .expect("a2l.jinja must parse");
    env
});

/// Render a registered template with a serializable context. A failure here is
/// an internal mismatch between the Rust context and the template, not an
/// unsupported model, so it carries its own error variant.
pub(crate) fn render<C: Serialize>(name: &str, ctx: C) -> R<String> {
    let tmpl = ENV
        .get_template(name)
        .map_err(|e| CodegenError::Template(format!("{name}: {e}")))?;
    tmpl.render(ctx)
        .map_err(|e| CodegenError::Template(format!("{name}: {e}")))
}
