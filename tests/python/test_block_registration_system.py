########################################################################################
##
##              Registration-consistency guard for the block surface
##
##  Adding a block touches several places (Rust constructor, constructors/mod.rs
##  re-export, #[pyfunction], #[pymodule] registration, fastsim.blocks class).
##  Nothing else asserts these agree, so it is easy to register a constructor
##  that is never exposed in `fastsim.blocks` (a silent gap). These tests close
##  that loop at the Python boundary.
##
########################################################################################

import inspect
import unittest

import fastsim._fastsim as _fastsim
import fastsim.blocks as blocks
from fastsim._fastsim import Block

# `Interface` and `Subsystem` are structural (top-level fastsim.*), not blocks.
_NON_BLOCK_FACTORIES = {"Interface", "Subsystem"}


def _exposed_block_classes():
    return {
        name
        for name in dir(blocks)
        if isinstance(getattr(blocks, name), type)
        and issubclass(getattr(blocks, name), Block)
        and getattr(blocks, name) is not Block
    }


def _registered_factories():
    # PascalCase callables in the extension module that are NOT classes are the
    # block-constructor `#[pyfunction]`s.
    return {
        name
        for name in dir(_fastsim)
        if not name.startswith("_")
        and name[0].isupper()
        and callable(getattr(_fastsim, name))
        and not isinstance(getattr(_fastsim, name), type)
    }


class TestBlockRegistrationConsistency(unittest.TestCase):

    def test_every_registered_factory_is_exposed(self):
        """Every block `#[pyfunction]` registered in `_fastsim` must be exposed
        as a `fastsim.blocks` class (minus the structural Interface/Subsystem)."""
        orphaned = _registered_factories() - _exposed_block_classes() - _NON_BLOCK_FACTORIES
        self.assertEqual(
            orphaned, set(),
            f"block factories registered in _fastsim but not exposed in "
            f"fastsim.blocks: {sorted(orphaned)}",
        )

    def test_every_exposed_block_introspects(self):
        """Every exposed block that carries info() must round-trip through it —
        this fails if a class is wired to a missing or mis-signatured Rust
        factory. The hand-written classes (Scope, Spectrum, BVP1D,
        AlgebraicConstraint) are unified onto the same info() path via
        _finalize_block_class, so they round-trip too."""
        for name in sorted(_exposed_block_classes()):
            cls = getattr(blocks, name)
            if not hasattr(cls, "info"):
                continue
            info = cls.info()
            self.assertEqual(info["type"], name)
            self.assertIn("parameters", info)
            self.assertIsInstance(info["parameters"], dict)

    def test_block_surface_is_non_trivial(self):
        """Regression guard: an accidental de-registration would shrink this."""
        self.assertGreaterEqual(len(_exposed_block_classes()), 90)


if __name__ == "__main__":
    unittest.main()
