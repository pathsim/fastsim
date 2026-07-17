"""Port parity: every pathsim base block with a fastsim namesake must rebase
thin (`__init__`-only) subclasses natively, and blocks without a native path
must fall back to a FAITHFUL Python shim — including algebraic/internal-solver
blocks without ``initial_value`` (issue: pathsim-chem's ``GLC``).

The registry/synthetic tests require only pathsim; the toolbox tests skip when
the toolbox is not installed, so the main suite passes on a clean install.
"""

import unittest
import warnings

import numpy as np

import fastsim
from fastsim import Simulation, Connection, port
from fastsim.blocks import Constant, Scope

try:
    import pathsim.blocks as ps_blocks
    from pathsim.blocks._block import Block as PsBlock
    _HAS_PATHSIM = True
except ImportError:
    _HAS_PATHSIM = False

try:
    from pathsim_chem.tritium.glc import GLC
    _HAS_CHEM = True
except ImportError:
    _HAS_CHEM = False


@unittest.skipUnless(_HAS_PATHSIM, "pathsim not installed")
class TestRegistryNamesakeComplete(unittest.TestCase):
    """The adapt registry must cover EVERY pathsim/fastsim namesake pair."""

    def test_every_namesake_is_registered(self):
        from fastsim.adapter import _build_registry
        import fastsim.blocks as fs_blocks

        reg = _build_registry()
        missing = []
        for name in dir(ps_blocks):
            if name.startswith("_") or name == "Block":
                continue
            ps_cls = getattr(ps_blocks, name, None)
            fs_cls = getattr(fs_blocks, name, None)
            if not (isinstance(ps_cls, type) and isinstance(fs_cls, type)):
                continue
            if not issubclass(ps_cls, PsBlock) or ps_cls is PsBlock:
                continue
            if not issubclass(fs_cls, fs_blocks.Block):
                continue
            if ps_cls not in reg:
                missing.append(name)
        self.assertEqual(missing, [], f"namesakes missing from the registry: {missing}")
        # The motivating case: toolboxes subclass "end-user" blocks too.
        self.assertIn(ps_blocks.BVP1D, reg)

    def test_thin_subclass_of_every_registered_base_rebases(self):
        """A synthetic `__init__`-only subclass of EVERY registry key must
        rebase onto the fastsim namesake (the user-facing parity contract)."""
        from fastsim.adapter import _build_registry, adapt

        failures = []
        for ps_cls, fs_cls in _build_registry().items():
            thin = type("Thin" + ps_cls.__name__, (ps_cls,), {})
            try:
                adapted = adapt(thin, strict=True)
            except Exception as e:  # noqa: BLE001 - collected for the report
                failures.append(f"{ps_cls.__name__}: {type(e).__name__}: {e}")
                continue
            if not (isinstance(adapted, type) and issubclass(adapted, fs_cls)):
                failures.append(f"{ps_cls.__name__}: not rebased onto {fs_cls.__name__}")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_hook_override_still_refuses_rebase(self):
        """The correctness guard survives the registry expansion: a subclass
        overriding an engine hook must NOT be class-accelerable."""
        from fastsim.port import _class_accelerable

        class Custom(ps_blocks.BVP1D):
            def update(self, t):  # custom post-processing => shim territory
                return super().update(t)

        self.assertFalse(_class_accelerable(Custom))


@unittest.skipUnless(_HAS_PATHSIM, "pathsim not installed")
class TestAlgebraicShim(unittest.TestCase):
    """Blocks without ``initial_value`` get a working Python fallback."""

    def _make_algebraic_block(self):
        class Doubler(PsBlock):
            """Stateless: y = 2*u + t, computed in a custom update()."""

            def update(self, t):
                u = self.inputs.to_array()
                self.outputs.update_from_array(2.0 * np.atleast_1d(u) + t)
                return 0.0

        return Doubler()

    def test_port_returns_working_block(self):
        blk = port(self._make_algebraic_block())
        self.assertIsInstance(blk, fastsim.blocks.Block)

    def test_runs_in_simulation_with_correct_values(self):
        blk = port(self._make_algebraic_block())
        src = Constant(3.0)
        sco = Scope()
        sim = Simulation(
            blocks=[src, blk, sco],
            connections=[Connection(src, blk), Connection(blk, sco)],
            log=False,
        )
        sim.run(1.0)
        t, ch = sco.read()
        expected = 2.0 * 3.0 + np.asarray(t)
        np.testing.assert_allclose(np.asarray(ch[0]), expected, rtol=1e-12)


@unittest.skipUnless(_HAS_CHEM, "pathsim-chem not installed")
class TestGLCPortParity(unittest.TestCase):
    """pathsim-chem's GLC (BVP1D subclass with custom update) ports via the
    algebraic shim and reproduces the pathsim block bit-for-bit."""

    _BASE = dict(P_in=2e5, L=1.0, D=0.1, T=623.0, g=9.81)
    _INPUT = (1e-3, 1.0, 0.0, 1e-4)

    def test_glc_ports_and_matches_pathsim(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = GLC(BCs="C-C", **self._BASE)
            ref.inputs.update_from_array(np.array(self._INPUT))
            ref.update(0.0)
            ref_out = ref.outputs.to_array()

            Ported = port(GLC)
            glc = Ported(BCs="C-C", **self._BASE)
            srcs = [Constant(v) for v in self._INPUT]
            conns = [Connection(s, glc[i]) for i, s in enumerate(srcs)]
            sim = Simulation(blocks=[*srcs, glc], connections=conns, log=False)
            sim.run(0.02)

        out = np.asarray(glc.outputs, dtype=float)[: len(ref_out)]
        np.testing.assert_allclose(out, ref_out, rtol=1e-9)
