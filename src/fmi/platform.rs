// FMI 3.0 binary platform tuple detection
//
// Maps the host platform to the directory name inside `binaries/` of an FMU.
// Reference: FMI 3.0 spec §6.2 "Platform Tuple" and rust-fmi `fmi3/mod.rs:138-156`.

/// Returns the FMI 3.0 platform tuple for the current host, e.g. `"aarch64-darwin"`.
///
/// FMI 3.0 uses `{arch}-{os}` where:
///   - arch ∈ { x86_64, aarch64, x86 }
///   - os   ∈ { linux, darwin, windows }
pub fn platform_tuple() -> &'static str {
    const ARCH: &str = std::env::consts::ARCH;
    const OS: &str = std::env::consts::OS;

    match (ARCH, OS) {
        ("x86_64", "linux") => "x86_64-linux",
        ("x86_64", "macos") => "x86_64-darwin",
        ("x86_64", "windows") => "x86_64-windows",
        ("aarch64", "linux") => "aarch64-linux",
        ("aarch64", "macos") => "aarch64-darwin",
        ("aarch64", "windows") => "aarch64-windows",
        ("x86", "linux") => "x86-linux",
        ("x86", "windows") => "x86-windows",
        _ => "unknown",
    }
}

/// Dynamic library file extension for the current host (`.so`, `.dylib`, `.dll`).
pub fn library_extension() -> &'static str {
    if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_tuple_is_known() {
        // On any supported dev/CI host we should never hit "unknown".
        assert_ne!(platform_tuple(), "unknown");
    }

    #[test]
    fn library_extension_matches_host() {
        let ext = library_extension();
        assert!(["so", "dylib", "dll"].contains(&ext));
    }
}
