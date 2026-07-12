########################################################################################
##
##                  Testing the new discrete-time blocks at system level
##
##   FirstOrderHold, DiscreteIntegrator, DiscreteDerivative, DiscreteStateSpace,
##   DiscreteTransferFunction, TappedDelay, plus the Polynomial math block and
##   the ZeroOrderHold alias.
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from fastsim import Simulation, Connection
from fastsim.blocks import (
    Constant,
    Source,
    Scope,
    SampleHold,
    ZeroOrderHold,
    FirstOrderHold,
    DiscreteIntegrator,
    DiscreteDerivative,
    DiscreteStateSpace,
    DiscreteTransferFunction,
    TappedDelay,
    Polynomial,
    )


# HELPERS =============================================================================

def _run(blocks, connections, duration, dt=0.01):
    sim = Simulation(blocks=blocks, connections=connections, dt=dt, log=False)
    sim.run(duration=duration, reset=True)


# POLYNOMIAL ==========================================================================

class TestPolynomialSystem(unittest.TestCase):

    def test_quadratic(self):
        """y = 2 u² + 3 u + 1 evaluated on a ramp"""
        Src = Source(lambda t: t)
        P = Polynomial(coeffs=[2.0, 3.0, 1.0])
        Sco = Scope()

        _run([Src, P, Sco], [Connection(Src, P), Connection(P, Sco)], duration=2.0)

        t, [y] = Sco.read()
        expected = 2.0 * t**2 + 3.0 * t + 1.0
        self.assertLess(np.max(np.abs(np.array(y) - expected)), 1e-6)


# ZERO-ORDER HOLD ALIAS ===============================================================

class TestZeroOrderHoldAlias(unittest.TestCase):

    def test_alias_runs(self):
        """ZeroOrderHold should behave identically to SampleHold"""
        Src = Constant(value=3.0)
        SH  = SampleHold(T=0.1)
        ZOH = ZeroOrderHold(T=0.1)
        S1  = Scope()
        S2  = Scope()

        _run(
            [Src, SH, ZOH, S1, S2],
            [Connection(Src, SH), Connection(Src, ZOH),
             Connection(SH, S1), Connection(ZOH, S2)],
            duration=0.5,
            )
        _, [a] = S1.read()
        _, [b] = S2.read()
        np.testing.assert_array_equal(a, b)


# FIRST-ORDER HOLD ====================================================================

class TestFirstOrderHoldSystem(unittest.TestCase):

    def test_extrapolation_tracks_ramp(self):
        """For a linear input, FOH extrapolation should track the slope"""
        Src = Source(lambda t: 2.0 * t)
        FOH = FirstOrderHold(T=0.1)
        Sco = Scope()

        _run([Src, FOH, Sco], [Connection(Src, FOH), Connection(FOH, Sco)], duration=1.0)

        t, [y] = Sco.read()
        t = np.array(t); y = np.array(y)
        #after the first two samples, output should be very close to 2t
        mask = t > 0.25
        if np.any(mask):
            err = np.max(np.abs(y[mask] - 2.0 * t[mask]))
            self.assertLess(err, 0.2)


# DISCRETE INTEGRATOR =================================================================

class TestDiscreteIntegratorSystem(unittest.TestCase):

    def test_constant_input_accumulates(self):
        """y[k+1] = y[k] + T·u[k] with constant u=1 reaches ~duration"""
        Src = Constant(value=1.0)
        DI  = DiscreteIntegrator(T=0.05)
        Sco = Scope()

        _run([Src, DI, Sco], [Connection(Src, DI), Connection(DI, Sco)], duration=1.0)

        t, [y] = Sco.read()
        #last sample should be roughly t (forward Euler with one-sample lag)
        self.assertAlmostEqual(y[-1], t[-1], delta=0.1)


    def test_initial_value(self):
        """Output starts at the supplied IC"""
        Src = Constant(value=0.0)
        DI  = DiscreteIntegrator(T=0.5, initial_value=5.0)
        Sco = Scope()

        _run([Src, DI, Sco], [Connection(Src, DI), Connection(DI, Sco)], duration=0.05)

        _, [y] = Sco.read()
        self.assertAlmostEqual(y[0], 5.0)


# DISCRETE DERIVATIVE =================================================================

class TestDiscreteDerivativeSystem(unittest.TestCase):

    def test_ramp_slope(self):
        """For a 2t ramp, d/dt ≈ 2 (in steady state)"""
        Src = Source(lambda t: 2.0 * t)
        DD  = DiscreteDerivative(T=0.05)
        Sco = Scope()

        _run([Src, DD, Sco], [Connection(Src, DD), Connection(DD, Sco)], duration=1.0)

        t, [y] = Sco.read()
        t = np.array(t); y = np.array(y)
        #after warmup, y should be close to 2.0
        mask = t > 0.2
        self.assertLess(np.max(np.abs(y[mask] - 2.0)), 0.5)


# DISCRETE STATE SPACE ================================================================

class TestDiscreteStateSpaceSystem(unittest.TestCase):

    def test_first_order(self):
        """x[k+1] = 0.5 x[k] + u[k], y[k] = x[k]; with u=1: y converges to 2"""
        Src = Constant(value=1.0)
        DSS = DiscreteStateSpace(A=[[0.5]], B=[[1.0]], C=[[1.0]], D=[[0.0]], T=0.1)
        Sco = Scope()

        _run([Src, DSS, Sco], [Connection(Src, DSS), Connection(DSS, Sco)], duration=2.0)

        _, [y] = Sco.read()
        self.assertAlmostEqual(y[-1], 2.0, delta=0.05)


# DISCRETE TRANSFER FUNCTION ==========================================================

class TestDiscreteTransferFunctionSystem(unittest.TestCase):

    def test_first_order(self):
        """H(z) = 1/(z - 0.5) → y[k+1] = 0.5 y[k] + u[k], steady state 2"""
        Src = Constant(value=1.0)
        DTF = DiscreteTransferFunction(Num=[1.0], Den=[1.0, -0.5], T=0.1)
        Sco = Scope()

        _run([Src, DTF, Sco], [Connection(Src, DTF), Connection(DTF, Sco)], duration=2.0)

        _, [y] = Sco.read()
        self.assertAlmostEqual(y[-1], 2.0, delta=0.1)


# TAPPED DELAY ========================================================================

class TestTappedDelaySystem(unittest.TestCase):

    def test_shift_register(self):
        """y_i[k] = u[k - i] for i=0..N-1"""
        Src = Source(lambda t: t)
        TD  = TappedDelay(N=3, T=0.1)
        Sco = Scope()

        _run(
            [Src, TD, Sco],
            [Connection(Src, TD),
             Connection(TD[0], Sco[0]),
             Connection(TD[1], Sco[1]),
             Connection(TD[2], Sco[2])],
            duration=1.0,
            )
        t, data = Sco.read()
        y0 = np.array(data[0])
        y1 = np.array(data[1])
        y2 = np.array(data[2])
        #y0 leads y1 by ~T=0.1, y1 leads y2 by ~T
        #compare in steady state where shifts are stable
        mask = np.array(t) > 0.5
        self.assertGreater(np.mean(y0[mask] - y1[mask]), 0.05)
        self.assertGreater(np.mean(y1[mask] - y2[mask]), 0.05)


# RUN =================================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
