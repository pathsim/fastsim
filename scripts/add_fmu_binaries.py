#!/usr/bin/env python
"""Embed platform binaries into the published source FMUs.

FastSim exports source FMUs; importing tools without a C toolchain cannot use
them directly. This script compiles the generated C for the platforms it can
reach and adds the results under ``binaries/{platform}/`` next to the sources,
so the same archive works either way:

- ``x86_64-windows`` — built on this machine with FMPy's CMake project, the
  same path ``check_exported_fmus.py`` exercises.
- ``x86_64-linux`` — built in a ``gcc`` Docker container. The generated model
  is a single translation unit (see ``sources/buildDescription.xml``), so a
  plain ``gcc -shared -fPIC`` is the whole build.

Run after ``export_reference_fmus.py``::

    python scripts/export_reference_fmus.py
    python scripts/add_fmu_binaries.py

Platforms that cannot be built here (e.g. darwin) are simply absent; the
sources always remain in the archive, so any importer can still compile
locally.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FMUS = ROOT / "fmus"

LINUX_IMAGE = "gcc:13"


def model_identifier(unpacked: Path) -> str:
    """The modelIdentifier names the shared library inside the FMU."""
    import xml.etree.ElementTree as ET

    root = ET.parse(unpacked / "modelDescription.xml").getroot()
    for tag in ("ModelExchange", "CoSimulation"):
        node = root.find(tag)
        if node is not None:
            return node.get("modelIdentifier")
    raise ValueError("no interface element in modelDescription.xml")


def build_windows(unpacked: Path) -> Path:
    """x86_64-windows via FMPy's CMake project for source FMUs."""
    from fmpy.build import build_platform_binary

    cmake_options = {}
    try:
        from fastsim._fastsim import find_c_compiler

        cc = find_c_compiler()
        if cc:
            cmake_options["CMAKE_C_COMPILER"] = cc
    except Exception:
        pass

    build_platform_binary(unpacked, cmake_options=cmake_options)
    out = unpacked / "binaries" / "x86_64-windows" / f"{model_identifier(unpacked)}.dll"
    if not out.is_file():
        raise RuntimeError(f"windows build produced no {out.name}")
    return out


def build_linux(unpacked: Path) -> Path:
    """x86_64-linux via a gcc container; one translation unit, one command."""
    ident = model_identifier(unpacked)
    outdir = unpacked / "binaries" / "x86_64-linux"
    outdir.mkdir(parents=True, exist_ok=True)

    # Windows paths go straight to `docker -v`; invoking through subprocess
    # (not a POSIX shell) avoids MSYS path mangling.
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{unpacked}:/w",
        LINUX_IMAGE,
        "bash", "-c",
        f"gcc -shared -fPIC -O2 /w/sources/fmu.c "
        f"-o /w/binaries/x86_64-linux/{ident}.so -lm "
        # sanity check: the required FMI 3.0 entry points must be exported
        f"&& nm -D /w/binaries/x86_64-linux/{ident}.so "
        f"| grep -q fmi3InstantiateModelExchange",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"linux build failed:\n{res.stderr[-800:]}")
    out = outdir / f"{ident}.so"
    if not out.is_file():
        raise RuntimeError(f"linux build produced no {out.name}")
    return out


def repack(unpacked: Path, fmu_path: Path) -> None:
    """Zip the tree back and replace the FMU atomically."""
    tmp = fmu_path.with_suffix(".fmu.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(unpacked.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(unpacked).as_posix())
    tmp.replace(fmu_path)


def main() -> int:
    model_dirs = sorted(p for p in FMUS.iterdir() if p.is_dir())
    if not model_dirs:
        print(f"no model directories under {FMUS}")
        return 1

    workdir = Path(tempfile.mkdtemp(prefix="fastsim_fmu_bin_"))
    try:
        for d in model_dirs:
            fmu_path = d / f"{d.name}.fmu"
            unpacked = workdir / d.name
            with zipfile.ZipFile(fmu_path) as z:
                z.extractall(unpacked)
            # rebuild from a clean state even if binaries were embedded before
            shutil.rmtree(unpacked / "binaries", ignore_errors=True)

            dll = build_windows(unpacked)
            so = build_linux(unpacked)
            repack(unpacked, fmu_path)

            size = fmu_path.stat().st_size / 1024
            print(
                f"{d.name}: + {dll.parent.name}/{dll.name} ({dll.stat().st_size//1024} KiB)"
                f", {so.parent.name}/{so.name} ({so.stat().st_size//1024} KiB)"
                f" -> {fmu_path.name} {size:.0f} KiB"
            )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
