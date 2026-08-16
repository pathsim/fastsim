"""Drop-in surface that pathsim's own examples reach for.

Each group below is a gap that kept an example from running at all, found by
running pathsim's `examples/` under fastsim (``test_example_parity.py``). They
are collected here because they are API contracts in their own right, not
properties of any one example: what ``Scope.read`` returns, what a block written
in Python can do, what a linear block reports about itself.
"""
import numpy as np
import pytest

import fastsim as fs
from fastsim._fastsim import Block
from fastsim.blocks._block import Block as BlockFromPathsimPath
from fastsim.events import Schedule

try:
    import pathsim as ps
    from pathsim.blocks._block import Block as PBlock
    from pathsim.events import Schedule as PSchedule
    HAS_PATHSIM = True
except ImportError:                                       # pragma: no cover
    HAS_PATHSIM = False


# -- Scope.read ------------------------------------------------------------------------

def _recorded(engine, duration=0.5, dt=0.1):
    B = engine.blocks
    src = B.SinusoidalSource(frequency=1.0, amplitude=1.0)
    integ = B.Integrator(0.0)
    sco = B.Scope(labels=["u", "x"])
    sim = engine.Simulation(
        [src, integ, sco],
        [engine.Connection(src, integ, sco[0]), engine.Connection(integ, sco[1])],
        dt=dt, log=False)
    sim.run(duration, reset=True, adaptive=False)
    return sco


def test_read_returns_an_array_not_a_list():
    """`data` supports array arithmetic.

    pathsim's `example_solar.py` subtracts two recordings (`data_moon -
    data_earth`); against a list of channels that is a TypeError.
    """
    time, data = _recorded(fs).read()
    assert isinstance(time, np.ndarray)
    assert isinstance(data, np.ndarray)
    assert data.shape == (2, len(time))
    assert (data - data).shape == data.shape


def test_read_reports_an_empty_recording_as_none():
    """`(None, None)`, which is what pathsim's examples test for."""
    assert fs.blocks.Scope().read() == (None, None)


@pytest.mark.skipif(not HAS_PATHSIM, reason="pathsim not installed")
def test_read_matches_pathsim_shape_and_type():
    f_t, f_d = _recorded(fs).read()
    p_t, p_d = _recorded(ps).read()
    assert type(f_d) is type(p_d)
    assert f_d.shape == p_d.shape
    assert f_t.shape == p_t.shape


# -- plotting ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="module")
def _headless():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")


def test_plot_variants_return_figure_and_axis():
    """`plot`, `plot2D` and `plot3D` all return `(fig, ax)`.

    Two examples unpack `fig, ax = sc.plot2D()`, which is a TypeError against a
    method that returns `None` — or does not exist.
    """
    import matplotlib.pyplot as plt

    B = fs.blocks
    lorenz = B.ODE(func=lambda x, u, t: np.array([
        10.0 * (x[1] - x[0]),
        x[0] * (28.0 - x[2]) - x[1],
        x[0] * x[1] - 8.0 / 3.0 * x[2]]), initial_value=[1.0, 1.0, 1.0])
    sco = B.Scope(labels=["x", "y", "z"])
    sim = fs.Simulation([lorenz, sco], [fs.Connection(lorenz, sco[0], sco[1], sco[2])],
                        dt=0.01, log=False)
    sim.run(1.0, reset=True, adaptive=False)

    for name in ("plot", "plot2D", "plot3D"):
        fig, ax = getattr(sco, name)()
        assert fig is not None and ax is not None, name
        assert (sco.fig, sco.ax) == (fig, ax), f"{name} should keep them on the block"
        plt.close(fig)


def test_plot_on_an_empty_scope_warns_and_returns_none():
    sco = fs.blocks.Scope()
    for name in ("plot", "plot2D", "plot3D"):
        with pytest.warns(UserWarning, match="no recording"):
            assert getattr(sco, name)() == (None, None), name


def test_spectrum_keeps_its_axis():
    """`Spc.ax.set_yscale("log")` — pathsim's `example_noise.py`."""
    import matplotlib.pyplot as plt

    src = fs.blocks.SinusoidalSource(frequency=1.0, amplitude=1.0)
    spc = fs.blocks.Spectrum()
    sim = fs.Simulation([src, spc], [fs.Connection(src, spc)], dt=0.01, log=False)
    sim.run(1.0, reset=True, adaptive=False)

    fig, ax = spc.plot()
    assert (spc.fig, spc.ax) == (fig, ax)
    spc.ax.set_yscale("log")
    plt.close(fig)


# -- a block written in Python -----------------------------------------------------------

def _counter_class(Block_, Schedule_):
    class Counter(Block_):
        """Accumulates its input every `T` and reports the running total."""

        def __init__(self, T=0.1):
            super().__init__()
            self.total = 0.0

            def _step(t):
                self.total += float(np.atleast_1d(self.inputs[0])[0])
                self.outputs[0] = self.total

            self.events = [Schedule_(t_start=0.0, t_period=T, func_act=_step)]

        def __len__(self):
            return 0

    return Counter


def _run_counter(engine, Block_, Schedule_, duration=0.5):
    cnt = _counter_class(Block_, Schedule_)(T=0.1)
    src, sco = engine.blocks.Constant(1.0), engine.blocks.Scope()
    sim = engine.Simulation([src, cnt, sco],
                            [engine.Connection(src, cnt), engine.Connection(cnt, sco)],
                            dt=0.01, log=False)
    sim.run(duration, reset=True, adaptive=False)
    return np.asarray(sco.read()[1])[0]


def test_block_importable_from_the_pathsim_path():
    """`from fastsim.blocks._block import Block`, as pathsim's examples spell it."""
    assert BlockFromPathsimPath is Block


def test_outputs_assignment_reaches_the_block():
    """A block reports its result by assigning to `outputs`.

    Against a snapshot array those writes went nowhere and the block silently
    emitted zeros.
    """
    blk = Block()
    blk.outputs[0] = 3.0
    assert blk.outputs[0] == 3.0

    # past the end: pathsim's register grows rather than raising
    blk.outputs[4] = 7.0
    assert blk.outputs[4] == 7.0
    assert len(blk.outputs) == 5

    blk.outputs = [1.0, 2.0]
    assert list(blk.outputs[:2]) == [1.0, 2.0]


def test_outputs_is_still_an_array():
    """The write-through view must not cost the array behaviour it replaced."""
    blk = Block()
    blk.outputs = [1.0, 2.0, 3.0]
    out = blk.outputs
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert list(out * 2) == [2.0, 4.0, 6.0]
    assert float(np.sum(out)) == 6.0


def test_a_block_declaring_its_own_events_is_driven_by_them():
    """`self.events = [...]` is how pathsim's `Block` publishes them."""
    got = _run_counter(fs, Block, Schedule)
    assert got[-1] > 0.0, "the block's own schedule never fired"
    assert np.all(np.diff(got) >= 0.0), "a running total should not decrease"


def test_declared_events_are_not_registered_twice():
    """A block may both attach an event and publish it; it must fire once."""
    fired = []
    evt = Schedule(t_start=0.0, t_period=0.1, func_act=fired.append)

    class Both(Block):
        def __init__(self):
            super().__init__()
            self.add_event(evt)     # the way a fastsim block attaches one
            self.events = [evt]     # the way pathsim publishes one

        def __len__(self):
            return 0

    blk = Both()
    src, sco = fs.blocks.Constant(1.0), fs.blocks.Scope()
    sim = fs.Simulation([src, blk, sco],
                        [fs.Connection(src, blk), fs.Connection(blk, sco)],
                        dt=0.01, log=False)
    sim.run(0.5, reset=True, adaptive=False)
    assert len(fired) == len(set(fired)) == 6, f"fired at {fired}"


@pytest.mark.skipif(not HAS_PATHSIM, reason="pathsim not installed")
def test_python_block_matches_pathsim():
    got = _run_counter(fs, Block, Schedule)
    ref = _run_counter(ps, PBlock, PSchedule)
    assert np.allclose(got, ref), f"{got} != {ref}"


# -- Wrapper subclassing -----------------------------------------------------------------

def _discrete_gain(engine, T=0.1):
    """A `Wrapper` subclass supplying its callable as a method, not an argument."""
    class DoubleIt(engine.blocks.Wrapper):
        def __init__(self, T=T):
            super().__init__(T=T)

        def func(self, u):
            return 2.0 * u

    src, sco = engine.blocks.Constant(1.5), engine.blocks.Scope()
    blk = DoubleIt()
    sim = engine.Simulation([src, blk, sco],
                            [engine.Connection(src, blk), engine.Connection(blk, sco)],
                            dt=0.01, log=False)
    sim.run(0.5, reset=True, adaptive=False)
    return np.asarray(sco.read()[1])[0]


def test_wrapper_subclass_may_define_func_as_a_method():
    """pathsim assigns `self.func` only when the argument is callable, so a
    subclass's own `func` survives — its `example_pid_vs_discretePID.py`
    builds a DiscretePID that way."""
    got = _discrete_gain(fs)
    assert np.isclose(got[-1], 3.0), f"expected 2*1.5, got {got[-1]}"


@pytest.mark.skipif(not HAS_PATHSIM, reason="pathsim not installed")
def test_wrapper_subclass_matches_pathsim():
    assert np.allclose(_discrete_gain(fs), _discrete_gain(ps))


# -- LTI realization ---------------------------------------------------------------------

LTI_BLOCKS = {
    "ButterworthLowpassFilter": dict(Fc=100.0, n=2),
    "ButterworthHighpassFilter": dict(Fc=100.0, n=2),
    "ButterworthBandpassFilter": dict(Fc=(10.0, 40.0), n=2),
    "ButterworthBandstopFilter": dict(Fc=(10.0, 40.0), n=2),
    "TransferFunctionNumDen": dict(Num=[1.0], Den=[1.0, 2.0, 1.0]),
    "PT2": dict(K=1.0, T=0.5, d=0.7),
}


@pytest.mark.parametrize("name", sorted(LTI_BLOCKS))
def test_lti_blocks_report_their_realization(name):
    """`A`/`B`/`C`/`D` of any block built from a state-space realization.

    pathsim's filters subclass `StateSpace` outright, so they carry the
    matrices; its `example_spectrum.py` builds the ideal frequency response
    from them.
    """
    blk = getattr(fs.blocks, name)(**LTI_BLOCKS[name])
    A, B, C, D = (np.asarray(getattr(blk, k)) for k in "ABCD")
    n = A.shape[0]
    assert A.shape == (n, n)
    assert B.shape[0] == n
    assert C.shape[1] == n
    assert D.shape == (C.shape[0], B.shape[1])


@pytest.mark.skipif(not HAS_PATHSIM, reason="pathsim not installed")
@pytest.mark.parametrize("name", sorted(LTI_BLOCKS))
def test_realization_matches_pathsim(name):
    kwargs = LTI_BLOCKS[name]
    f = getattr(fs.blocks, name)(**kwargs)
    try:
        p = getattr(ps.blocks, name)(**kwargs)
    except Exception as e:                                # pragma: no cover
        pytest.skip(f"pathsim cannot build {name}: {e}")
    for k in "ABCD":
        got, ref = np.asarray(getattr(f, k)), np.asarray(getattr(p, k), dtype=float)
        assert got.shape == ref.shape, f"{name}.{k}: {got.shape} vs {ref.shape}"
        # Scaled by the matrix, not elementwise: a coefficient that is
        # analytically zero lands on rounding noise (~1e-7 next to entries of
        # ~4e6 in the bandstop C row), and elementwise `allclose` would call
        # two numerical zeros different while ignoring the entries that matter.
        scale = max(1.0, float(np.max(np.abs(ref))))
        worst = float(np.max(np.abs(got - ref))) / scale
        assert worst < 1e-12, f"{name}.{k}: worst scaled deviation {worst:.3e}\n{got}\nvs\n{ref}"


def test_frequency_response_from_the_realization():
    """The realization is the one being integrated, checked against the
    analytic Butterworth magnitude |H| = 1/sqrt(1 + (f/Fc)^(2n))."""
    Fc, n = 100.0, 3
    blk = fs.blocks.ButterworthLowpassFilter(Fc=Fc, n=n)
    A, B, C, D = (np.asarray(getattr(blk, k)) for k in "ABCD")

    for f in (10.0, 50.0, Fc, 200.0, 1000.0):
        s = 2j * np.pi * f
        H = (C @ np.linalg.solve(s * np.eye(A.shape[0]) - A, B) + D).item()
        want = 1.0 / np.sqrt(1.0 + (f / Fc) ** (2 * n))
        assert np.isclose(abs(H), want, rtol=1e-9), f"{f} Hz: {abs(H)} vs {want}"


def test_non_lti_blocks_have_no_realization():
    """`state_space()` reports None rather than inventing matrices."""
    assert fs.blocks.Multiplier().state_space() is None
    with pytest.raises(AttributeError):
        fs.blocks.Multiplier().A
