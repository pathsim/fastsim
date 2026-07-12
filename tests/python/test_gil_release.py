"""The compiled run paths release the GIL (pure-Rust integration).

While `CompiledSimulation.run` / `run_batch` execute on the main thread, a
background Python thread must keep making progress. If the GIL were held for
the whole integration, the background counter would advance ~0 times during
the run (only at the boundaries).

`CompiledSimulation` is `unsendable` (thread-affine), so the sim runs on the
thread that created it — the realistic pattern — and the counter thread
measures interpreter responsiveness.

The thresholds are deliberately loose — the test distinguishes "GIL released"
(counter advances thousands of times) from "GIL held" (counter frozen), not
precise timing.
"""

import threading

import numpy as np

from fastsim import Simulation, Connection
from fastsim.blocks import Integrator, Amplifier
from fastsim.solvers import RKBS32


def _compiled_oscillator():
    """Undamped oscillator, compiled — cheap per step, many steps."""
    int_v = Integrator(1.0)
    int_x = Integrator(0.0)
    amp = Amplifier(-1.0)
    sim = Simulation(
        blocks=[int_v, int_x, amp],
        connections=[
            Connection(int_v, int_x),
            Connection(int_x, amp),
            Connection(amp, int_v),
        ],
        dt=1e-5,
        log=False,
        Solver=RKBS32,
    )
    c = sim.compile()
    c.output_stride = 1000  # keep recording memory small
    return c


class _Counter:
    """Background thread incrementing a counter until stopped."""

    def __init__(self):
        self.count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            self.count += 1

    def __enter__(self):
        self._thread.start()
        # Let the thread warm up, then zero the count so we only measure
        # progress made DURING the foreground work.
        while self.count == 0:
            pass
        self.count = 0
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        return False


def test_compiled_run_releases_gil():
    c = _compiled_oscillator()
    with _Counter() as bg:
        c.run(20.0, reset=True, adaptive=False)
        n = bg.count
    assert n > 1000, f"background thread starved during compiled run (count={n})"
    assert c.time >= 20.0 - 1e-9


def test_run_batch_releases_gil():
    c = _compiled_oscillator()
    with _Counter() as bg:
        finals = c.run_batch([{} for _ in range(8)], 5.0, False)
        n = bg.count
    assert n > 1000, f"background thread starved during run_batch (count={n})"
    assert len(finals) == 8
    arr = np.array(finals)
    assert np.allclose(arr, arr[0]), "identical params -> identical finals"
