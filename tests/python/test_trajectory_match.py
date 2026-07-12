"""Trajectory matching tests: fastsim vs pathsim for every block type.

For each block, builds a minimal simulation in both fastsim and pathsim,
runs them with identical parameters, and asserts the trajectories match
within numerical tolerance.

Run: python -m pytest tests/python/test_trajectory_match.py -v
"""

import unittest
import numpy as np
import pytest

# Import both engines — the pathsim-dependent classes carry the shared
# `@pytest.mark.pathsim` marker (conftest skips them when pathsim is absent, or
# fails hard under PATHSIM_REQUIRED), replacing the old per-class
# `unittest.skipUnless` so there is one skip mechanism.
import fastsim
try:
    import pathsim
    HAS_PATHSIM = True
except ImportError:
    HAS_PATHSIM = False

from fastsim import Simulation as FSim, Connection as FConn

if HAS_PATHSIM:
    from pathsim import Simulation as PSim, Connection as PConn
    from fastsim.solvers import RKCK54 as F_RKCK54, ESDIRK43 as F_ESDIRK43
    from pathsim.solvers import RKCK54 as P_RKCK54, ESDIRK43 as P_ESDIRK43


# ======================================================================================
# Helpers
# ======================================================================================

def compare_trajectories(test_case, t_fs, d_fs, t_ps, d_ps, tol=1e-6, label=""):
    """Compare two trajectories by interpolating to common time grid."""
    # Must have data
    test_case.assertGreater(len(t_fs), 0, f"{label}: fastsim produced no data")
    test_case.assertGreater(len(t_ps), 0, f"{label}: pathsim produced no data")

    # Interpolate to common grid
    t_end = min(t_fs[-1], t_ps[-1])
    t_common = np.linspace(0, t_end, 200)

    n_channels = min(len(d_fs), len(d_ps))
    for ch in range(n_channels):
        fs_interp = np.interp(t_common, t_fs, d_fs[ch])
        ps_interp = np.interp(t_common, t_ps, d_ps[ch])
        max_diff = np.max(np.abs(fs_interp - ps_interp))
        test_case.assertLess(
            max_diff, tol,
            f"{label} channel {ch}: max diff = {max_diff:.2e} (tol={tol})")


def run_siso_block_test(test_case, fs_block_factory, ps_block_factory,
                        source_fn=None, duration=5.0, tol=1e-6,
                        solver="explicit", label=""):
    """Run a SISO block test: Source → Block → Scope, compare fastsim vs pathsim."""
    if source_fn is None:
        source_fn = lambda t: np.sin(2 * np.pi * t)

    # fastsim
    from fastsim.blocks import Source as FSource, Scope as FScope
    fs_src = FSource(source_fn)
    fs_blk = fs_block_factory()
    fs_sco = FScope()
    fs_sim = FSim([fs_src, fs_blk, fs_sco],
                  [FConn(fs_src, fs_blk), FConn(fs_blk, fs_sco)], log=False)
    if solver == "implicit":
        fs_sim._set_solver(F_ESDIRK43)
    else:
        fs_sim._set_solver(F_RKCK54, tolerance_lte_abs=1e-8, tolerance_lte_rel=0.0)
    fs_sim.run(duration)
    t_fs, d_fs = fs_sco.read()

    # pathsim
    from pathsim.blocks import Source as PSource, Scope as PScope
    ps_src = PSource(source_fn)
    ps_blk = ps_block_factory()
    ps_sco = PScope()
    ps_sim = PSim([ps_src, ps_blk, ps_sco],
                  [PConn(ps_src, ps_blk), PConn(ps_blk, ps_sco)], log=False)
    if solver == "implicit":
        ps_sim._set_solver(P_ESDIRK43)
    else:
        ps_sim._set_solver(P_RKCK54, tolerance_lte_abs=1e-8, tolerance_lte_rel=0.0)
    ps_sim.run(duration)
    t_ps, d_ps = ps_sco.read()

    compare_trajectories(test_case, t_fs, d_fs, t_ps, d_ps, tol=tol, label=label)


def run_source_test(test_case, fs_source_factory, ps_source_factory,
                    duration=5.0, tol=1e-6, label=""):
    """Run a Source block test: Source → Scope, compare fastsim vs pathsim."""
    # fastsim
    from fastsim.blocks import Scope as FScope
    fs_src = fs_source_factory()
    fs_sco = FScope()
    fs_sim = FSim([fs_src, fs_sco], [FConn(fs_src, fs_sco)], log=False)
    fs_sim._set_solver(F_RKCK54)
    fs_sim.run(duration)
    t_fs, d_fs = fs_sco.read()

    # pathsim
    from pathsim.blocks import Scope as PScope
    ps_src = ps_source_factory()
    ps_sco = PScope()
    ps_sim = PSim([ps_src, ps_sco], [PConn(ps_src, ps_sco)], log=False)
    ps_sim._set_solver(P_RKCK54)
    ps_sim.run(duration)
    t_ps, d_ps = ps_sco.read()

    compare_trajectories(test_case, t_fs, d_fs, t_ps, d_ps, tol=tol, label=label)


# ======================================================================================
# Test Classes
# ======================================================================================

@pytest.mark.pathsim
class TestSISOBlocks(unittest.TestCase):
    """Test SISO blocks: Source → Block → Scope, trajectory match."""

    def test_amplifier(self):
        run_siso_block_test(self,
            lambda: fastsim.blocks.Amplifier(2.5),
            lambda: pathsim.blocks.Amplifier(2.5),
            label="Amplifier")

    def test_integrator(self):
        run_siso_block_test(self,
            lambda: fastsim.blocks.Integrator(1.0),
            lambda: pathsim.blocks.Integrator(1.0),
            label="Integrator")

    def test_differentiator(self):
        run_siso_block_test(self,
            lambda: fastsim.blocks.Differentiator(),
            lambda: pathsim.blocks.Differentiator(),
            label="Differentiator", tol=1e-3)

    def test_pt1(self):
        run_siso_block_test(self,
            lambda: fastsim.blocks.PT1(K=2.0, T=0.1),
            lambda: pathsim.blocks.PT1(K=2.0, T=0.1),
            label="PT1")

    def test_pt2(self):
        run_siso_block_test(self,
            lambda: fastsim.blocks.PT2(K=1.0, T=0.5, d=0.3),
            lambda: pathsim.blocks.PT2(K=1.0, T=0.5, d=0.3),
            label="PT2")

    def test_lead_lag(self):
        run_siso_block_test(self,
            lambda: fastsim.blocks.LeadLag(T1=0.1, T2=0.5),
            lambda: pathsim.blocks.LeadLag(T1=0.1, T2=0.5),
            label="LeadLag")

    def test_delay(self):
        run_siso_block_test(self,
            lambda: fastsim.blocks.Delay(tau=0.1),
            lambda: pathsim.blocks.Delay(tau=0.1),
            label="Delay", tol=1e-2)

    def test_clip(self):
        run_siso_block_test(self,
            lambda: fastsim.blocks.Clip(min_val=-0.5, max_val=0.5),
            lambda: pathsim.blocks.Clip(min_val=-0.5, max_val=0.5),
            label="Clip")

    def test_deadband(self):
        run_siso_block_test(self,
            lambda: fastsim.blocks.Deadband(lower=-0.3, upper=0.3),
            lambda: pathsim.blocks.Deadband(lower=-0.3, upper=0.3),
            label="Deadband")

    def test_rate_limiter(self):
        run_siso_block_test(self,
            lambda: fastsim.blocks.RateLimiter(rate=5.0),
            lambda: pathsim.blocks.RateLimiter(rate=5.0),
            label="RateLimiter")

    def test_rescale(self):
        run_siso_block_test(self,
            lambda: fastsim.blocks.Rescale(i0=-1, i1=1, o0=0, o1=10),
            lambda: pathsim.blocks.Rescale(i0=-1, i1=1, o0=0, o1=10),
            label="Rescale")

    def test_backlash(self):
        run_siso_block_test(self,
            lambda: fastsim.blocks.Backlash(width=0.2),
            lambda: pathsim.blocks.Backlash(width=0.2),
            label="Backlash")


@pytest.mark.pathsim
class TestMathBlocks(unittest.TestCase):
    """Test math/transcendental blocks."""

    def _run(self, name, fs_cls, ps_cls, source_fn=None, tol=1e-6):
        run_siso_block_test(self,
            lambda: fs_cls(), lambda: ps_cls(),
            source_fn=source_fn or (lambda t: 0.5 + 0.3 * np.sin(2 * np.pi * t)),
            label=name, tol=tol)

    def test_sin(self):
        self._run("Sin", fastsim.blocks.Sin, pathsim.blocks.Sin)

    def test_cos(self):
        self._run("Cos", fastsim.blocks.Cos, pathsim.blocks.Cos)

    def test_exp(self):
        self._run("Exp", fastsim.blocks.Exp, pathsim.blocks.Exp,
                  source_fn=lambda t: 0.1 * np.sin(t))

    def test_abs(self):
        self._run("Abs", fastsim.blocks.Abs, pathsim.blocks.Abs)

    def test_sqrt(self):
        self._run("Sqrt", fastsim.blocks.Sqrt, pathsim.blocks.Sqrt,
                  source_fn=lambda t: 1.0 + 0.5 * np.sin(t))

    def test_log(self):
        self._run("Log", fastsim.blocks.Log, pathsim.blocks.Log,
                  source_fn=lambda t: 1.0 + 0.5 * np.sin(t))

    def test_tanh(self):
        self._run("Tanh", fastsim.blocks.Tanh, pathsim.blocks.Tanh)

    def test_sinh(self):
        self._run("Sinh", fastsim.blocks.Sinh, pathsim.blocks.Sinh,
                  source_fn=lambda t: 0.3 * np.sin(t))

    def test_cosh(self):
        self._run("Cosh", fastsim.blocks.Cosh, pathsim.blocks.Cosh,
                  source_fn=lambda t: 0.3 * np.sin(t))

    def test_atan(self):
        self._run("Atan", fastsim.blocks.Atan, pathsim.blocks.Atan)

    def test_pow(self):
        self._run("Pow", fastsim.blocks.Pow, pathsim.blocks.Pow,
                  source_fn=lambda t: 0.5 + 0.3 * np.sin(t))


@pytest.mark.pathsim
class TestSourceBlocks(unittest.TestCase):
    """Test source blocks: Source → Scope, trajectory match."""

    def test_constant(self):
        run_source_test(self,
            lambda: fastsim.blocks.Constant(3.14),
            lambda: pathsim.blocks.Constant(3.14),
            label="Constant")

    def test_step_source(self):
        run_source_test(self,
            lambda: fastsim.blocks.StepSource(amplitude=2.0, tau=1.0),
            lambda: pathsim.blocks.StepSource(amplitude=2.0, tau=1.0),
            label="StepSource")

    def test_step_source_multi(self):
        run_source_test(self,
            lambda: fastsim.blocks.StepSource(amplitude=[1.0, 2.0, 0.5], tau=[0.0, 1.0, 3.0]),
            lambda: pathsim.blocks.StepSource(amplitude=[1.0, 2.0, 0.5], tau=[0.0, 1.0, 3.0]),
            label="StepSource(multi)")

    def test_sinusoidal_source(self):
        run_source_test(self,
            lambda: fastsim.blocks.SinusoidalSource(frequency=2.0, amplitude=3.0),
            lambda: pathsim.blocks.SinusoidalSource(frequency=2.0, amplitude=3.0),
            label="SinusoidalSource")

    def test_clock(self):
        run_source_test(self,
            lambda: fastsim.blocks.Clock(),
            lambda: pathsim.blocks.Clock(),
            label="Clock")

    def test_square_wave(self):
        run_source_test(self,
            lambda: fastsim.blocks.SquareWaveSource(frequency=2.0, amplitude=1.0),
            lambda: pathsim.blocks.SquareWaveSource(frequency=2.0, amplitude=1.0),
            label="SquareWaveSource", tol=0.1)  # Discontinuous

    def test_triangle_wave(self):
        run_source_test(self,
            lambda: fastsim.blocks.TriangleWaveSource(frequency=2.0, amplitude=1.0),
            lambda: pathsim.blocks.TriangleWaveSource(frequency=2.0, amplitude=1.0),
            label="TriangleWaveSource", tol=1e-3)


@pytest.mark.pathsim
class TestSystemBlocks(unittest.TestCase):
    """Test ODE, Function, StateSpace blocks in system context."""

    def test_ode_brusselator(self):
        """Brusselator ODE system."""
        a, b = 0.4, 1.2
        def f(x, u, t):
            return np.array([a - x[0] - b*x[0] + x[0]**2*x[1],
                             b*x[0] - x[0]**2*x[1]])

        # fastsim
        fs_ode = fastsim.blocks.ODE(f, initial_value=np.zeros(2))
        fs_sco = fastsim.blocks.Scope()
        fs_sim = FSim([fs_ode, fs_sco],
                      [FConn(fs_ode[:2], fs_sco[:2])], log=False)
        fs_sim._set_solver(F_RKCK54, tolerance_lte_abs=1e-8, tolerance_lte_rel=0.0)
        fs_sim.run(50)
        t_fs, d_fs = fs_sco.read()

        # pathsim
        ps_ode = pathsim.blocks.ODE(f, initial_value=np.zeros(2))
        ps_sco = pathsim.blocks.Scope()
        ps_sim = PSim([ps_ode, ps_sco],
                      [PConn(ps_ode[:2], ps_sco[:2])], log=False)
        ps_sim._set_solver(P_RKCK54, tolerance_lte_abs=1e-8, tolerance_lte_rel=0.0)
        ps_sim.run(50)
        t_ps, d_ps = ps_sco.read()

        compare_trajectories(self, t_fs, d_fs, t_ps, d_ps, tol=1e-4, label="Brusselator")

    def test_function_block(self):
        """Function block: y = x^2."""
        run_siso_block_test(self,
            lambda: fastsim.blocks.Function(lambda x: x**2),
            lambda: pathsim.blocks.Function(lambda x: x**2),
            label="Function(x^2)")

    def test_state_space(self):
        """StateSpace: first-order lowpass."""
        A = [[-10.0]]
        B = [[10.0]]
        C = [[1.0]]
        D = [[0.0]]
        run_siso_block_test(self,
            lambda: fastsim.blocks.StateSpace(A, B, C, D),
            lambda: pathsim.blocks.StateSpace(A, B, C, D),
            label="StateSpace(LP)")

    def test_pid(self):
        """PID controller block."""
        run_siso_block_test(self,
            lambda: fastsim.blocks.PID(Kp=2.0, Ki=0.5, Kd=0.1),
            lambda: pathsim.blocks.PID(Kp=2.0, Ki=0.5, Kd=0.1),
            label="PID", tol=1e-3)


@pytest.mark.pathsim
class TestMultiBlockSystems(unittest.TestCase):
    """Test complete multi-block systems."""

    def test_harmonic_oscillator(self):
        """Damped harmonic oscillator: 7 blocks, feedback loop."""
        m, c, k = 0.8, 0.2, 1.5

        # fastsim
        I1 = fastsim.blocks.Integrator(5.0)
        I2 = fastsim.blocks.Integrator(2.0)
        A1 = fastsim.blocks.Amplifier(c)
        A2 = fastsim.blocks.Amplifier(k)
        A3 = fastsim.blocks.Amplifier(-1/m)
        P1 = fastsim.blocks.Adder()
        S = fastsim.blocks.Scope()
        fs_sim = FSim([I1,I2,A1,A2,A3,P1,S], [
            FConn(I1,I2,A1,S), FConn(I2,A2,S[1]),
            FConn(A1,P1), FConn(A2,P1[1]), FConn(P1,A3), FConn(A3,I1)], log=False)
        fs_sim._set_solver(F_RKCK54, tolerance_lte_abs=1e-8, tolerance_lte_rel=0.0)
        fs_sim.run(30)
        t_fs, d_fs = S.read()

        # pathsim
        I1 = pathsim.blocks.Integrator(5.0)
        I2 = pathsim.blocks.Integrator(2.0)
        A1 = pathsim.blocks.Amplifier(c)
        A2 = pathsim.blocks.Amplifier(k)
        A3 = pathsim.blocks.Amplifier(-1/m)
        P1 = pathsim.blocks.Adder()
        S = pathsim.blocks.Scope()
        ps_sim = PSim([I1,I2,A1,A2,A3,P1,S], [
            PConn(I1,I2,A1,S), PConn(I2,A2,S[1]),
            PConn(A1,P1), PConn(A2,P1[1]), PConn(P1,A3), PConn(A3,I1)], log=False)
        ps_sim._set_solver(P_RKCK54, tolerance_lte_abs=1e-8, tolerance_lte_rel=0.0)
        ps_sim.run(30)
        t_ps, d_ps = S.read()

        compare_trajectories(self, t_fs, d_fs, t_ps, d_ps, tol=1e-4,
                             label="Harmonic Oscillator")

    def test_lorenz(self):
        """Lorenz attractor: 13 blocks, chaotic."""
        sigma, rho, beta = 10, 28, 8/3

        def build(Sim, Conn, blk):
            i1=blk.Integrator(1.0);i2=blk.Integrator(1.0);i3=blk.Integrator(1.0)
            a1=blk.Amplifier(sigma);ax=blk.Adder('+-');cr=blk.Constant(rho);ar=blk.Adder('+-')
            mr=blk.Multiplier();ay=blk.Adder('-+');mxy=blk.Multiplier()
            ab=blk.Amplifier(beta);az=blk.Adder('+-');s=blk.Scope()
            sim=Sim([i1,i2,i3,a1,ax,cr,ar,mr,ay,mxy,ab,az,s],[
                Conn(i1,ax[1],mr[0],mxy[0],s[0]),Conn(i2,ax[0],ay[0],mxy[1],s[1]),
                Conn(i3,ar[1],ab,s[2]),Conn(ax,a1),Conn(a1,i1),
                Conn(cr,ar[0]),Conn(ar,mr[1]),Conn(mr,ay[1]),Conn(ay,i2),
                Conn(mxy,az[0]),Conn(ab,az[1]),Conn(az,i3)],log=False)
            return sim, s

        fs_sim, fs_sco = build(FSim, FConn, fastsim.blocks)
        fs_sim._set_solver(F_RKCK54, tolerance_lte_abs=1e-8, tolerance_lte_rel=0.0)
        fs_sim.run(10)
        t_fs, d_fs = fs_sco.read()

        ps_sim, ps_sco = build(PSim, PConn, pathsim.blocks)
        ps_sim._set_solver(P_RKCK54, tolerance_lte_abs=1e-8, tolerance_lte_rel=0.0)
        ps_sim.run(10)
        t_ps, d_ps = ps_sco.read()

        compare_trajectories(self, t_fs, d_fs, t_ps, d_ps, tol=1e-3,
                             label="Lorenz Attractor")


@pytest.mark.pathsim
class TestFixedBlocks(unittest.TestCase):
    """Tests for blocks that were reimplemented or had signatures fixed."""

    def test_mod(self):
        run_siso_block_test(self,
            lambda: fastsim.blocks.Mod(modulus=2.0),
            lambda: pathsim.blocks.Mod(modulus=2.0),
            source_fn=lambda t: 3.0 * np.sin(2 * np.pi * t),
            label="Mod")

    def test_gaussian_pulse(self):
        run_source_test(self,
            lambda: fastsim.blocks.GaussianPulseSource(amplitude=1.0, f_max=10.0, tau=0.5),
            lambda: pathsim.blocks.GaussianPulseSource(amplitude=1.0, f_max=10.0, tau=0.5),
            label="GaussianPulse", duration=1.0)

    def test_step_source_multi(self):
        run_source_test(self,
            lambda: fastsim.blocks.StepSource(amplitude=[1.0, 2.0, 0.5], tau=[0.0, 1.0, 3.0]),
            lambda: pathsim.blocks.StepSource(amplitude=[1.0, 2.0, 0.5], tau=[0.0, 1.0, 3.0]),
            label="StepSource(multi)")

    def test_rescale(self):
        run_siso_block_test(self,
            lambda: fastsim.blocks.Rescale(i0=-1.0, i1=1.0, o0=0.0, o1=10.0),
            lambda: pathsim.blocks.Rescale(i0=-1.0, i1=1.0, o0=0.0, o1=10.0),
            label="Rescale")

    def test_counter(self):
        """Counter block: count zero crossings of sinusoid."""
        from fastsim.blocks import SinusoidalSource as FSS, Counter as FC, Scope as FS
        from pathsim.blocks import SinusoidalSource as PSS, Counter as PC, Scope as PS

        fs = FSS(frequency=2.0); fc = FC(); fsc = FS()
        fsim = FSim([fs,fc,fsc], [FConn(fs,fc), FConn(fc,fsc)], log=False)
        fsim._set_solver(F_RKCK54, tolerance_lte_abs=1e-6)
        fsim.run(2.0)
        t1, d1 = fsc.read()

        ps = PSS(frequency=2.0); pc = PC(); psc = PS()
        psim = PSim([ps,pc,psc], [PConn(ps,pc), PConn(pc,psc)], log=False)
        psim._set_solver(P_RKCK54, tolerance_lte_abs=1e-6)
        psim.run(2.0)
        t2, d2 = psc.read()

        # Counter output is discrete — compare final count
        self.assertEqual(d1[0][-1], d2[0][-1], "Counter final count mismatch")

    def test_comparator_list_span(self):
        """Comparator accepts list for span (not just tuple)."""
        c = fastsim.blocks.Comparator(threshold=0.0, span=[0.0, 1.0])
        self.assertIsNotNone(c)

    def test_pow_exponent_name(self):
        """Pow uses 'exponent' parameter name."""
        p = fastsim.blocks.Pow(exponent=3.0)
        self.assertIsNotNone(p)


@pytest.mark.pathsim
class TestPortLabels(unittest.TestCase):
    """Test that port labels match pathsim's API."""

    def test_siso_labels(self):
        for name in ['Amplifier', 'Integrator', 'PT1', 'PID', 'Sin', 'Cos']:
            with self.subTest(block=name):
                fs_cls = getattr(fastsim.blocks, name)
                ps_cls = getattr(pathsim.blocks, name)
                # Compare class-level port labels
                fs_in = getattr(fs_cls, '_input_port_labels', None) or getattr(fs_cls, 'input_port_labels', None)
                ps_in = getattr(ps_cls, 'input_port_labels', None)
                fs_out = getattr(fs_cls, '_output_port_labels', None) or getattr(fs_cls, 'output_port_labels', None)
                ps_out = getattr(ps_cls, 'output_port_labels', None)
                self.assertEqual(fs_in, ps_in, f"{name} input_port_labels mismatch")
                self.assertEqual(fs_out, ps_out, f"{name} output_port_labels mismatch")

    def test_source_labels(self):
        for name in ['Constant', 'SinusoidalSource', 'Clock']:
            with self.subTest(block=name):
                fs_cls = getattr(fastsim.blocks, name)
                ps_cls = getattr(pathsim.blocks, name)
                fs_in = getattr(fs_cls, '_input_port_labels', None) or getattr(fs_cls, 'input_port_labels', None)
                ps_in = getattr(ps_cls, 'input_port_labels', None)
                fs_out = getattr(fs_cls, '_output_port_labels', None) or getattr(fs_cls, 'output_port_labels', None)
                ps_out = getattr(ps_cls, 'output_port_labels', None)
                self.assertEqual(fs_in, ps_in, f"{name} input_port_labels mismatch")
                self.assertEqual(fs_out, ps_out, f"{name} output_port_labels mismatch")

    def test_dual_input_labels(self):
        for name in ['GreaterThan', 'LessThan', 'Equal', 'LogicAnd', 'LogicOr']:
            with self.subTest(block=name):
                fs_cls = getattr(fastsim.blocks, name)
                ps_cls = getattr(pathsim.blocks, name)
                fs_in = getattr(fs_cls, '_input_port_labels', None) or getattr(fs_cls, 'input_port_labels', None)
                ps_in = getattr(ps_cls, 'input_port_labels', None)
                self.assertEqual(fs_in, ps_in, f"{name} input_port_labels mismatch")

    def test_string_indexing_siso(self):
        """Verify string indexing on blocks with fixed ports."""
        pid = fastsim.blocks.PID(1.0, 0.5, 0.1)
        pr = pid["in"]
        self.assertIsNotNone(pr)
        pr = pid["out"]
        self.assertIsNotNone(pr)

    def test_string_indexing_dual_input(self):
        """Verify dual-input blocks support named ports."""
        gt = fastsim.blocks.GreaterThan()
        self.assertIsNotNone(gt["a"])
        self.assertIsNotNone(gt["b"])
        self.assertIsNotNone(gt["y"])

    def test_string_on_vectorial_raises(self):
        """Vectorial blocks (None labels) reject string indexing."""
        amp = fastsim.blocks.Amplifier(1.0)
        with self.assertRaises(ValueError):
            amp["in"]

    def test_invalid_string_raises(self):
        pid = fastsim.blocks.PID(1.0, 0.5, 0.1)
        with self.assertRaises(ValueError):
            pid["nonexistent"]


# ======================================================================================
# Noise Sources
# ======================================================================================

class TestNoiseBlocks(unittest.TestCase):
    """Test noise blocks produce correct statistical properties."""

    def test_white_noise_continuous_statistics(self):
        """WhiteNoise continuous mode: mean ≈ 0, std ≈ standard_deviation."""
        from fastsim.blocks import WhiteNoise, Scope
        wn = WhiteNoise(standard_deviation=2.0, seed=42)
        sco = Scope()
        sim = FSim([wn, sco], [FConn(wn, sco)], dt=0.001, log=False)
        sim.run(10.0)
        _, data = sco.read()
        vals = np.array(data[0])
        self.assertAlmostEqual(np.mean(vals), 0.0, delta=0.2)
        self.assertAlmostEqual(np.std(vals), 2.0, delta=0.3)

    def test_white_noise_discrete_statistics(self):
        """WhiteNoise discrete mode: produces held samples."""
        from fastsim.blocks import WhiteNoise, Scope
        wn = WhiteNoise(standard_deviation=1.0, sampling_period=0.01, seed=42)
        sco = Scope()
        sim = FSim([wn, sco], [FConn(wn, sco)], dt=0.001, log=False)
        sim.run(1.0)
        _, data = sco.read()
        vals = np.array(data[0])
        # In discrete mode, values should change at sampling_period intervals
        self.assertGreater(len(vals), 10)
        self.assertFalse(np.all(vals == vals[0]), "All values identical")

    def test_white_noise_spectral_density(self):
        """WhiteNoise spectral density mode: amplitude scales with 1/sqrt(dt)."""
        from fastsim.blocks import WhiteNoise, Scope
        wn = WhiteNoise(spectral_density=1.0, seed=42)
        sco = Scope()
        sim = FSim([wn, sco], [FConn(wn, sco)], dt=0.01, log=False)
        sim.run(10.0)
        _, data = sco.read()
        vals = np.array(data[0])
        # With dt=0.01, expected std ≈ sqrt(1.0/0.01) = 10
        self.assertGreater(np.std(vals), 5.0)

    def test_pink_noise_continuous(self):
        """PinkNoise continuous mode: produces non-zero output."""
        from fastsim.blocks import PinkNoise, Scope
        pn = PinkNoise(standard_deviation=1.0, seed=42)
        sco = Scope()
        sim = FSim([pn, sco], [FConn(pn, sco)], dt=0.001, log=False)
        sim.run(5.0)
        _, data = sco.read()
        vals = np.array(data[0])
        self.assertGreater(len(vals), 100)
        self.assertGreater(np.std(vals), 0.1)
        self.assertAlmostEqual(np.mean(vals), 0.0, delta=0.5)

    def test_pink_noise_discrete(self):
        """PinkNoise discrete mode: held samples."""
        from fastsim.blocks import PinkNoise, Scope
        pn = PinkNoise(standard_deviation=1.0, sampling_period=0.01, seed=42)
        sco = Scope()
        sim = FSim([pn, sco], [FConn(pn, sco)], dt=0.001, log=False)
        sim.run(1.0)
        _, data = sco.read()
        vals = np.array(data[0])
        self.assertGreater(len(vals), 10)

    def test_random_number_generator_continuous(self):
        """RandomNumberGenerator continuous: uniform [0,1)."""
        from fastsim.blocks import RandomNumberGenerator, Scope
        rng = RandomNumberGenerator(seed=42)
        sco = Scope()
        sim = FSim([rng, sco], [FConn(rng, sco)], dt=0.001, log=False)
        sim.run(5.0)
        _, data = sco.read()
        vals = np.array(data[0])
        self.assertGreater(len(vals), 100)
        self.assertTrue(np.all(vals >= 0.0))
        self.assertTrue(np.all(vals < 1.0))
        self.assertAlmostEqual(np.mean(vals), 0.5, delta=0.05)

    def test_random_number_generator_discrete(self):
        """RandomNumberGenerator discrete: held uniform samples."""
        from fastsim.blocks import RandomNumberGenerator, Scope
        rng = RandomNumberGenerator(sampling_period=0.01, seed=42)
        sco = Scope()
        sim = FSim([rng, sco], [FConn(rng, sco)], dt=0.001, log=False)
        sim.run(1.0)
        _, data = sco.read()
        vals = np.array(data[0])
        self.assertGreater(len(vals), 10)
        self.assertTrue(np.all(vals >= 0.0))
        self.assertTrue(np.all(vals < 1.0))

    def test_white_noise_seed_reproducibility(self):
        """Same seed → same output."""
        from fastsim.blocks import WhiteNoise, Scope
        results = []
        for _ in range(2):
            wn = WhiteNoise(seed=123)
            sco = Scope()
            sim = FSim([wn, sco], [FConn(wn, sco)], dt=0.01, log=False)
            sim.run(1.0)
            _, data = sco.read()
            results.append(np.array(data[0]))
        np.testing.assert_array_equal(results[0], results[1])


# ======================================================================================
# Filters and TransferFunctionZPG
# ======================================================================================

@pytest.mark.pathsim
class TestFilterBlocks(unittest.TestCase):
    """Test filter blocks match pathsim trajectory."""

    def test_butterworth_lowpass(self):
        """ButterworthLowpassFilter: fastsim vs pathsim trajectory match."""
        run_siso_block_test(
            self,
            lambda: fastsim.blocks.ButterworthLowpassFilter(Fc=10, n=2),
            lambda: pathsim.blocks.ButterworthLowpassFilter(Fc=10, n=2),
            source_fn=lambda t: np.sin(2 * np.pi * t),
            duration=2.0, tol=1e-5, label="ButterworthLowpass")

    def test_butterworth_highpass(self):
        """ButterworthHighpassFilter: fastsim vs pathsim trajectory match."""
        run_siso_block_test(
            self,
            lambda: fastsim.blocks.ButterworthHighpassFilter(Fc=10, n=2),
            lambda: pathsim.blocks.ButterworthHighpassFilter(Fc=10, n=2),
            source_fn=lambda t: np.sin(2 * np.pi * t),
            duration=2.0, tol=1e-5, label="ButterworthHighpass")

    def test_butterworth_bandpass(self):
        """ButterworthBandpassFilter: fastsim vs pathsim trajectory match."""
        run_siso_block_test(
            self,
            lambda: fastsim.blocks.ButterworthBandpassFilter(Fc=[5, 15], n=2),
            lambda: pathsim.blocks.ButterworthBandpassFilter(Fc=[5, 15], n=2),
            source_fn=lambda t: np.sin(2 * np.pi * 10 * t),
            duration=2.0, tol=1e-5, label="ButterworthBandpass")

    def test_butterworth_bandstop(self):
        """ButterworthBandstopFilter: fastsim vs pathsim trajectory match."""
        run_siso_block_test(
            self,
            lambda: fastsim.blocks.ButterworthBandstopFilter(Fc=[5, 15], n=2),
            lambda: pathsim.blocks.ButterworthBandstopFilter(Fc=[5, 15], n=2),
            source_fn=lambda t: np.sin(2 * np.pi * 10 * t),
            duration=2.0, tol=1e-5, label="ButterworthBandstop")

    def test_allpass_filter(self):
        """AllpassFilter: fastsim vs pathsim trajectory match."""
        run_siso_block_test(
            self,
            lambda: fastsim.blocks.AllpassFilter(fs=10, n=1),
            lambda: pathsim.blocks.AllpassFilter(fs=10, n=1),
            source_fn=lambda t: np.sin(2 * np.pi * t),
            duration=2.0, tol=1e-5, label="AllpassFilter")

    def test_transfer_function_zpg(self):
        """TransferFunctionZPG: fastsim vs pathsim trajectory match."""
        run_siso_block_test(
            self,
            lambda: fastsim.blocks.TransferFunctionZPG(Zeros=[], Poles=[-1, -2], Gain=2.0),
            lambda: pathsim.blocks.TransferFunctionZPG(Zeros=[], Poles=[-1, -2], Gain=2.0),
            source_fn=lambda t: np.sin(2 * np.pi * t),
            duration=3.0, tol=1e-5, label="TransferFunctionZPG")

    def test_transfer_function_zpg_complex_poles(self):
        """TransferFunctionZPG with complex conjugate poles."""
        run_siso_block_test(
            self,
            lambda: fastsim.blocks.TransferFunctionZPG(
                Zeros=[], Poles=[-1+2j, -1-2j], Gain=5.0),
            lambda: pathsim.blocks.TransferFunctionZPG(
                Zeros=[], Poles=[-1+2j, -1-2j], Gain=5.0),
            source_fn=lambda t: np.sin(2 * np.pi * t),
            duration=3.0, tol=1e-5, label="TransferFunctionZPG_complex")


# ======================================================================================
# Main
# ======================================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
