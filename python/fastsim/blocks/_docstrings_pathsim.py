# Block docstrings mirrored 1:1 from pathsim — DO NOT EDIT BY HAND.
# Regenerate from pathsim source via scripts/extract_docstrings.py.

DOCS = {
    'ADC': """Models an ideal Analog-to-Digital Converter (ADC).

This block samples an analog input signal periodically, quantizes it
according to the specified number of bits and input span, and outputs
the resulting digital code on multiple output ports. The sampling
is triggered by a scheduled event.

Functionality:

1. Samples the analog input `inputs[0]` at intervals of `T`, starting after delay `tau`.
2. Clips the input voltage to the defined `span` [min_voltage, max_voltage].
3. Scales the clipped voltage to the range [0, 1].
4. Quantizes the scaled value to an integer code between 0 and 2^n_bits - 1 using flooring.
5. Converts the integer code to an n_bits binary representation.
6. Outputs the binary code on ports 0 (LSB) to n_bits-1 (MSB).

Ideal characteristics:

- Instantaneous sampling at scheduled times.
- Perfect, noise-free quantization.
- No aperture jitter or other dynamic errors.


Parameters
----------
n_bits : int, optional
    Number of bits for the digital output code. Default is 4.
span : list[float] or tuple[float], optional
    The valid analog input value range [min_voltage, max_voltage].
    Inputs outside this range will be clipped. Default is [-1, 1].
T : float, optional
    Sampling period (time between samples). Default is 1 time unit.
tau : float, optional
    Initial delay before the first sample is taken. Default is 0.


Attributes
----------
events : list[Schedule]
    Internal scheduled event responsible for periodic sampling and conversion.
""",
    'Abs': """Absolute value operator block.

This block supports vector inputs. This is the operation it does:
    
.. math::
    
    \\vec{y} = \\vert| \\vec{u} \\vert| 
    
Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'Adder': """Summs / adds up all input signals to a single output signal (MISO)

This is how it works in the default case

.. math::
    
    y(t) = \\sum_i u_i(t)

and like this when additional operations are defined

.. math::
    
    y(t) = \\sum_i \\mathrm{op}_i \\cdot u_i(t)


Example
-------
This is the default initialization that just adds up all the inputs:

.. code-block:: python

    A = Adder()

and this is the initialization with specific operations that subtracts 
the second from first input and neglects all others:

.. code-block:: python

    A = Adder('+-')


Note
----
This block is purely algebraic and its operation (`op_alg`) will be called 
multiple times per timestep, each time when `Simulation._update(t)` is 
called in the global simulation loop.


Parameters
----------
operations : str, optional
    optional string of operations to be applied before 
    summation, i.e. '+-' will compute the difference, 
    'None' will just perform regular sum


Attributes
----------
_ops : dict
    dict that maps string operations to numerical
_ops_array : array_like
    operations converted to array
op_alg : Operator
    internal algebraic operator
""",
    'Alias': """Signal alias / pass-through block.

Passes the input directly to the output without modification.
This is useful for signal renaming in model composition.

.. math::

    y = x

This block supports vector inputs.

Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'AllpassFilter': """Direct implementation of a first order allpass filter, or a cascade 
of n 1st order allpass filters

.. math:: 

    H(s) = \\frac{s - 2\\pi f_s}{s + 2\\pi f_s}

where f_s is the frequency, where the 1st order allpass has a 90 deg phase shift.

Parameters
----------
fs : float
    frequency for 90 deg phase shift of 1st order allpass
n : int
    number of cascades
""",
    'Amplifier': """Amplifies the input signal by multiplication with a constant gain term.

Like this:

.. math::
    
    y(t) = \\mathrm{gain} \\cdot u(t)


Note
----
This block is purely algebraic and its operation (`op_alg`) will be called 
multiple times per timestep, each time when `Simulation._update(t)` is 
called in the global simulation loop.

    
Example
-------
The block is initialized like this:

.. code-block:: python
    
    #amplification by factor 5
    A = Amplifier(gain=5)


Parameters
----------
gain : float
    amplifier gain

    
Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'AntiWindupPID': """Proportional-Integral-Differentiation (PID) controller with anti-windup mechanism (back-calculation).

Anti-windup mechanisms are needed when the magnitude of the control signal
from the PID controller is limited by some real world saturation. In these cases,
the integrator will continue to accumulate the control error and "wind itself up".
Once the setpoint is reached, this can result in significant overshoots. This
implementation adds a conditional feedback term to the internal integrator that
"unwinds" it when the PID output crosses some limits. This is pretty much a
deadzone feedback element for the integrator.

Mathematically, this block implements the following set of ODEs

.. math::

    \\begin{align}
    \\dot{x}_1 &= f_\\mathrm{max} (u - x_1) \\\\
    \\dot{x}_2 &= u - w
    \\end{align}

with the anti-windup feedback (depending on the pid output)

.. math::

    w = K_s (y - \\min(\\max(y, y_\\mathrm{min}), y_\\mathrm{max}))

and the output itself

.. math::

    y = K_p u + K_d f_\\mathrm{max} (u - x_1) + K_i x_2


Note
----
Depending on `f_max`, the resulting system might become stiff or ill conditioned!
As a practical choice set `f_max` to 3x the highest expected signal frequency.
Since this block uses an approximation of real differentiation, the approximation will
not hold if there are high frequency components present in the signal. For example if
you have discontinuities such as steps or square waves.


Example
-------
The block is initialized like this:

.. code-block:: python

    #cutoff at 1kHz, windup limits at [-5, 5]
    pid = AntiWindupPID(Kp=2, Ki=0.5, Kd=0.1, f_max=1e3, limits=[-5, 5])


Parameters
----------
Kp : float
    proportional controller coefficient
Ki : float
    integral controller coefficient
Kd : float
    differentiator controller coefficient
f_max : float
    highest expected signal frequency
Ks : float
    feedback term for back calculation for anti-windup control of integrator
limits : array_like[float]
    lower and upper limit for PID output that triggers anti-windup of integrator
""",
    'Atan': """Arctangent operator block.

This block supports vector inputs. This is the operation it does:
    
.. math::
    
    \\vec{y} = \\arctan(\\vec{u}) 
    
Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'Atan2': """Two-argument arctangent block.

Computes the four-quadrant arctangent of two inputs:

.. math::

    y = \\mathrm{atan2}(a, b)

Note
----
This block takes exactly two inputs (a, b) and produces one output.
The first input is the y-coordinate, the second is the x-coordinate,
matching the convention of ``numpy.arctan2(y, x)``.

Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'Backlash': """Backlash (mechanical play) element.

Models the hysteresis-like behavior of mechanical backlash in gears,
couplings and other systems with play. The output only tracks the input
after the input has moved through the full backlash width.

.. math::

    \\dot{x} = f_\\mathrm{max} \\left((u - x) - \\mathrm{clip}(u - x,\\; -w/2,\\; w/2)\\right)

where `w` is the total backlash width. Inside the dead zone :math:`|u - x| \\leq w/2`
the output does not move. Once the input pushes past the edge, the output
tracks with bandwidth `f_max`.


Example
-------
The block is initialized like this:

.. code-block:: python

    #backlash with 0.5 units of total play
    bl = Backlash(width=0.5, f_max=1e3)


Parameters
----------
width : float
    total backlash width (play)
f_max : float
    tracking bandwidth parameter when engaged
""",
    'ButterworthBandpassFilter': """Direct implementation of a bandpass butterworth filter block.

Follows the same structure as the 'StateSpace' block in the 
'pathsim.blocks' module. The numerator and denominator of the 
filter transfer function are generated and then the transfer 
function is realized as a state space model. 

Parameters
----------
Fc : list[float]
    corner frequencies (left, right) of the filter in [Hz]
n : int
    filter order
""",
    'ButterworthBandstopFilter': """Direct implementation of a bandstop butterworth filter block.

Follows the same structure as the 'StateSpace' block in the 
'pathsim.blocks' module. The numerator and denominator of the 
filter transfer function are generated and then the transfer 
function is realized as a state space model. 

Parameters
----------
Fc : tuple[float], list[float]
    corner frequencies (left, right) of the filter in [Hz]
n : int
    filter order
""",
    'ButterworthHighpassFilter': """Direct implementation of a high pass butterworth filter block.

Follows the same structure as the 'StateSpace' block in the 
'pathsim.blocks' module. The numerator and denominator of the 
filter transfer function are generated and then the transfer 
function is realized as a state space model. 

Parameters
----------
Fc : float
    corner frequency of the filter in [Hz]
n : int
    filter order
""",
    'ButterworthLowpassFilter': """Direct implementation of a low pass butterworth filter block.

Follows the same structure as the 'StateSpace' block in the 
'pathsim.blocks' module. The numerator and denominator of the 
filter transfer function are generated and then the transfer 
function is realized as a state space model. 

Parameters
----------
Fc : float
    corner frequency of the filter in [Hz]
n : int
    filter order
""",
    'ChirpPhaseNoiseSource': """Chirp source, sinusoid with frequency ramp up and ramp down, plus phase noise.

This works by using a time dependent triangle wave for the frequency 
and integrating it with a numerical integration engine to get a 
continuous phase. This phase is then used to evaluate a sinusoid.

Additionally the chirp source can have white and cumulative phase noise. 
Mathematically it looks like this for the contributions to the phase from 
the triangular wave:

.. math::

    \\varphi_t(t) = \\int_0^t \\mathrm{tri}_{f_0, B, T}(\\tau) \\, d\\tau

And from the white (w) and cumulative (c) noise:

.. math::

    \\varphi_n(t) = \\sigma_w \\, n_w(t) + \\sigma_c \\int_0^t n_c(\\tau) \\, d\\tau

The phase contributions are then used to evaluate a sinusoid to get the final chirp signal:

.. math::

    y(t) = A \\sin(\\varphi_t(t) + \\varphi_n(t) + \\varphi_0)

Parameters
----------
amplitude : float
    amplitude of the chirp signal
f0 : float
    start frequency of the chirp signal
BW : float
    bandwidth of the frequency ramp of the chirp signal
T : float
    period of the frequency ramp of the chirp signal
phase : float
    phase of sinusoid (initial, radians)
sig_cum : float
    weight for cumulative phase noise contribution
sig_white : float
    weight for white phase noise contribution
sampling_period : float, None
    time between phase noise samples. If None,
    noise is sampled every timestep (default is 0.1)

Attributes
----------
noise_1 : float
    internal noise value for white phase noise
noise_2 : float
    internal noise value for cumulative phase noise
events : list[Schedule]
    scheduled event for periodic sampling (only if sampling_period is set)
""",
    'ChirpSource': """Alias for ChirpPhaseNoiseSource.

.. deprecated:: 1.0.0
   Use :func:`ChirpPhaseNoiseSource` instead.""",
    'Clip': """Clipping/saturation operator block.

This block supports vector inputs. This is the operation it does:
    
.. math::
    
    \\vec{y} = \\text{clip}(\\vec{u}, u_{min}, u_{max}) 

Parameters
----------
min_val : float, array_like
    minimum clipping value
max_val : float, array_like
    maximum clipping value
    
Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'Clock': """Alias for ClockSource.

.. deprecated:: 1.0.0
   Use :func:`ClockSource` instead.""",
    'ClockSource': """Discrete time clock source block.

Utilizes scheduled events to periodically set 
the block output to 0 or 1 at discrete times.

Parameters
----------
T : float
    period of the clock
tau : float
    clock delay

Attributes
----------
events : list[Schedule]
    internal scheduled event list 
""",
    'Comparator': """Comparator block that sets output depending on predefined thresholds for the input.

Sets the output to '1' if the input signal crosses a predefined threshold and to '-1' 
if it crosses in the reverse direction. 

This is realized by the block spawning a zero-crossing event detector that watches 
the input of the block and locates the transition up to a tolerance. 

The block output is determined by a simple sign check in
the 'update' method.

Parameters
----------
threshold : float
    threshold value for the comparator
tolerance : float
    tolerance for zero crossing detection    
span : list[float] or tuple[float], optional
    output value range [min, max]

Attributes
----------
events : list[ZeroCrossing]
    internal zero crossing event
""",
    'Constant': """Produces a constant output signal (SISO).

.. math::

    y(t) = const.

    
Parameters
----------
value : float
    constant defining block output
""",
    'Cos': """Cosine operator block.

This block supports vector inputs. This is the operation it does:
    
.. math::
    
    \\vec{y} = \\cos(\\vec{u}) 
    
Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'Cosh': """Hyperbolic cosine operator block.

This block supports vector inputs. This is the operation it does:
    
.. math::
    
    \\vec{y} = \\cosh(\\vec{u}) 
    
Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'Counter': """Counts the number of detected bidirectional threshold crossings.

Uses zero-crossing events for the detection and sets the output 
accordingly.

Parameters
----------
start : int
    counter start (initial condition)
threshold : float
    threshold for zero crossing

Attributes
----------
E : ZeroCrossing
    internal event manager
events : list[ZeroCrossing]
    internal zero crossing event
""",
    'CounterDown': """Counts the number of detected unidirectional (hi->lo) threshold crossings.

Note
----
This is a modification of 'Counter' which only counts 
unidirectional zero-crossings (high -> low)

Parameters
----------
start : int
    counter start (initial condition)
threshold : float
    threshold for zero crossing

Attributes
----------
E : ZeroCrossingDown
    internal event manager
events : list[ZeroCrossing]
    internal zero crossing event
""",
    'CounterUp': """Counts the number of detected unidirectional (lo->hi) threshold crossings.

Note
----
This is a modification of 'Counter' which only counts 
unidirectional zero-crossings (low -> high)

Parameters
----------
start : int
    counter start (initial condition)
threshold : float
    threshold for zero crossing

Attributes
----------
E : ZeroCrossingUp
    internal event manager
events : list[ZeroCrossing]
    internal zero crossing event
""",
    'DAC': """Models an ideal Digital-to-Analog Converter (DAC).

This block reads a digital input code periodically from its input ports,
reconstructs the corresponding analog value based on the number of bits
and output span, and holds the output constant between updates. The update
is triggered by a scheduled event.

Functionality:

1. Reads the digital code from input ports 0 (LSB) to n_bits-1 (MSB) at intervals of `T`, starting after delay `tau`.
2. Interprets the inputs as an unsigned binary integer code.
3. Converts the integer code to a fractional value between 0 and (2^n_bits - 1) / 2^n_bits.
4. Scales this fractional value to the specified analog output `span`.
5. Outputs the resulting analog value on `outputs[0]`.
6. Holds the output value constant until the next scheduled update.

Ideal characteristics:

- Instantaneous update at scheduled times.
- Perfect, noise-free reconstruction.
- No glitches or settling time.


Parameters
----------
n_bits : int, optional
    Number of digital input bits expected. Default is 4.
span : list[float] or tuple[float], optional
    The analog output value range [min_voltage, max_voltage] corresponding
    to the digital codes 0 and 2^n_bits - 1, respectively (approximately).
    Default is [-1, 1].
T : float, optional
    Update period (time between output updates). Default is 1 time unit.
tau : float, optional
    Initial delay before the first output update. Default is 0.


Attributes
----------
events : list[Schedule]
    Internal scheduled event responsible for periodic updates.
""",
    'Deadband': """Deadband (dead zone) element.

Outputs zero when the input is within the dead zone, and passes
the signal shifted by the zone boundary otherwise:

.. math::

    y = \\begin{cases}
        u - u_\\mathrm{upper} & \\text{if } u > u_\\mathrm{upper} \\\\
        0 & \\text{if } u_\\mathrm{lower} \\leq u \\leq u_\\mathrm{upper} \\\\
        u - u_\\mathrm{lower} & \\text{if } u < u_\\mathrm{lower}
    \\end{cases}

or equivalently :math:`y = u - \\mathrm{clip}(u,\\; u_\\mathrm{lower},\\; u_\\mathrm{upper})`.


Example
-------
The block is initialized like this:

.. code-block:: python

    #symmetric dead zone of width 0.2
    db = Deadband(lower=-0.1, upper=0.1)


Parameters
----------
lower : float
    lower bound of the dead zone
upper : float
    upper bound of the dead zone
""",
    'Delay': """Delays the input signal by a time constant 'tau' in seconds.

Supports two modes of operation:

**Continuous mode** (default, ``sampling_period=None``):
Uses an adaptive interpolating buffer for continuous-time delay.

.. math::

    y(t) =
    \\begin{cases}
    x(t - \\tau) & , t \\geq \\tau \\\\
    0            & , t < \\tau
    \\end{cases}

**Discrete mode** (``sampling_period`` provided):
Uses a ring buffer with scheduled sampling events for N-sample delay,
where ``N = round(tau / sampling_period)``.

.. math::

    y[k] = x[k - N]

Note
----
In continuous mode, the internal adaptive buffer uses interpolation for
the evaluation. This is required to be compatible with variable step solvers.
It has a drawback however. The order of the ode solver used will degrade
when this block is used, due to the interpolation.


Note
----
This block supports vector input, meaning we can have multiple parallel
delay paths through this block.


Example
-------
Continuous-time delay:

.. code-block:: python

    #5 time units delay
    D = Delay(tau=5)

Discrete-time N-sample delay (10 samples):

.. code-block:: python

    D = Delay(tau=0.01, sampling_period=0.001)

Parameters
----------
tau : float
    delay time constant in seconds
sampling_period : float, None
    sampling period for discrete mode, default is continuous mode

Attributes
----------
_buffer : AdaptiveBuffer
    internal interpolatable adaptive rolling buffer (continuous mode)
_ring : deque
    internal ring buffer for N-sample delay (discrete mode)
""",
    'Differentiator': """Differentiates the input signal. 

Uses a first order transfer function with a pole at the origin which implements 
a high pass filter. Supports vector input. 
    
.. math::
    
    H_\\mathrm{diff}(s) = \\frac{s}{1 + s / f_\\mathrm{max}} 

The approximation holds for signals up to a frequency of approximately f_max.

Note
-----
Depending on `f_max`, the resulting system might become stiff or ill conditioned!
As a practical choice set `f_max` to 3x the highest expected signal frequency.

Note
----
Since this is an approximation of real differentiation, the approximation will not hold 
if there are high frequency components present in the signal. For example if you have 
discontinuities such as steps or squere waves.

Example
-------
The block is initialized like this:

.. code-block:: python
    
    #cutoff at 1kHz
    D = Differentiator(f_max=1e3)

Parameters
----------
f_max : float
    highest expected signal frequency

Attributes
----------
op_dyn : DynamicOperator
    internal dynamic operator for ODE component
op_alg : DynamicOperator
    internal algebraic operator

""",
    'DiscreteDerivative': """Discrete-time backward-difference derivative.

.. math::

    y[k] = \\frac{u[k] - u[k-1]}{T}

Note
----
Supports vector input — each channel is differentiated independently.

Parameters
----------
T : float
    sampling period
tau : float
    delay before first sample

Attributes
----------
events : list[Schedule]
    internal scheduled event for periodic update
""",
    'DiscreteIntegrator': """Discrete-time integrator (forward Euler).

.. math::

    y[k+1] = y[k] + T \\, u[k]

The output at sample ``k`` is the accumulated sum of past inputs;
the current input ``u[k]`` only enters the next sample.

Note
----
Supports vector input — each channel is integrated independently.
Pass an array as ``initial_value`` to set per-channel initial values.

Parameters
----------
T : float
    sampling period
tau : float
    delay before first sample
initial_value : float, array_like
    initial integrator output ``y[0]``

Attributes
----------
events : list[Schedule]
    internal scheduled event for periodic update
""",
    'DiscreteStateSpace': """Discrete-time MIMO state space block.

.. math::

    \\begin{align}
        x[k+1] &= \\mathbf{A}\\, x[k] + \\mathbf{B}\\, u[k] \\\\
        y[k]   &= \\mathbf{C}\\, x[k] + \\mathbf{D}\\, u[k]
    \\end{align}

Note
----
The output port reflects ``y[k]`` for the duration of the current
sample interval (zero-order hold between updates). The direct
feedthrough term ``D u[k]`` is computed at the sample event, so the
block has no algebraic passthrough between updates.

Parameters
----------
A, B, C, D : array_like
    discrete state space matrices
T : float
    sampling period
tau : float
    delay before first sample
initial_value : array_like, None
    initial state ``x[0]``

Attributes
----------
events : list[Schedule]
    internal scheduled event for periodic update
""",
    'DiscreteTransferFunction': """Discrete-time SISO transfer function in numerator/denominator form.

.. math::

    H(z) = \\frac{b_0 z^M + b_1 z^{M-1} + \\dots + b_M}{a_0 z^N + a_1 z^{N-1} + \\dots + a_N}

Realized internally as a ``DiscreteStateSpace`` via the controllable
canonical form returned by ``scipy.signal.tf2ss``.

Parameters
----------
Num : array_like
    numerator polynomial coefficients (highest power of z first)
Den : array_like
    denominator polynomial coefficients (highest power of z first)
T : float
    sampling period
tau : float
    delay before first sample
""",
    'Divider': """Multiplies and divides input signals (MISO).

This is the default behavior (multiply all):

.. math::

    y(t) = \\prod_i u_i(t)

and this is the behavior with an operations string:

.. math::

    y(t) = \\frac{\\prod_{i \\in M} u_i(t)}{\\prod_{j \\in D} u_j(t)}

where :math:`M` is the set of inputs with ``*`` and :math:`D` the set with ``/``.


Example
-------
Default initialization multiplies the first input and divides by the second:

.. code-block:: python

    D = Divider()

Multiply the first two inputs and divide by the third:

.. code-block:: python

    D = Divider('**/')

Raise an error instead of producing ``inf`` when a denominator input is zero:

.. code-block:: python

    D = Divider('**/', zero_div='raise')

Clamp the denominator to machine epsilon so the output stays finite:

.. code-block:: python

    D = Divider('**/', zero_div='clamp')


Note
----
This block is purely algebraic and its operation (``op_alg``) will be called
multiple times per timestep, each time when ``Simulation._update(t)`` is
called in the global simulation loop.


Parameters
----------
operations : str, optional
    String of ``*`` and ``/`` characters indicating which inputs are
    multiplied (``*``) or divided (``/``). Inputs beyond the length of
    the string default to ``*``. Defaults to ``'*/'`` (divide second
    input by first).
zero_div : str, optional
    Behaviour when a denominator input is zero. One of:

    ``'warn'`` *(default)*
        Propagates ``inf`` and emits a ``RuntimeWarning`` — numpy's
        standard behaviour.
    ``'raise'``
        Raises ``ZeroDivisionError``.
    ``'clamp'``
        Clamps the denominator magnitude to machine epsilon
        (``numpy.finfo(float).eps``), preserving sign, so the output
        stays large-but-finite rather than ``inf``.


Attributes
----------
_ops : dict
    Maps operation characters to exponent values (``+1`` or ``-1``).
_ops_array : numpy.ndarray
    Exponents (+1 for ``*``, -1 for ``/``) converted to an array.
op_alg : Operator
    Internal algebraic operator.
""",
    'DynamicalFunction': """Arbitrary MIMO time and input dependent function block.

The function signature needs two arguments `f(u, t)` where `u` is 
the (possibly vectorial) block input and `t` is a time dependency.

.. math::

    \\vec{y} = \\mathrm{func}(\\vec{u}, t)


Note
----
This block does essentially the same as `Function` but with different 
requirements for the signature of the function to be wrapped. 
Block inputs are packed into an array `u` and this block additionally 
accepts time dependency in the function provided. 
Thats where the prefix `Dynamical..` comes from.


Example
-------
Lets say we want to implement a super simple model for a voltage controlled 
oscillator (VCO), where the block input controls the frequency of a sine wave 
at the output. 

.. code-block:: python
    
    import numpy as np
    from pathsim.blocks import DynamicalFunction
    
    f_0 = 100

    def f_vco(u, t):
        return np.sin(2*np.pi*f_0*u*t)

    vco = DynamicalFunction(f_vco)        


Using it as a decorator also works:

.. code-block:: python
    
    import numpy as np
    from pathsim.blocks import DynamicalFunction
    
    f_0 = 100
    
    @DynamicalFunction
    def vco(u, t):
        return np.sin(2*np.pi*f_0*u*t)

    #'vco' is now a PathSim block 


Parameters
----------
func : callable
    function that defines algebraic block IO behaviour with time dependency, 
    signature `func(u, t)` where `u` is `numpy.ndarray` and `t` is `float`


Attributes
----------
op_alg : DynamicOperator
    internal operator that wraps `func`

""",
    'DynamicalSystem': """This block implements a nonlinear dynamical system / nonlinear state space model.

Its basically the same as the `ODE` block with the addition of an output equation
that takes the state, input and time as arguments:

.. math::

    \\begin{align}
        \\dot{x}(t) &= \\mathrm{func}_\\mathrm{dyn}(x(t), u(t), t) \\\\
               y(t) &= \\mathrm{func}_\\mathrm{alg}(x(t), u(t), t)
    \\end{align}

    
Parameters
----------
func_dyn : callable
    right hand side function of ode-part of the system
func_alg : callable
    output function of the system
initial_value : array[float]
    initial state / initial condition
jac_dyn : callable | None
    optional jacobian of `func_dyn` to improve convergence 
    for implicit ode solvers


Attributes
----------
op_dyn : DynamicOperator
    internal dynamic operator for `func_dyn`
op_alg : DynamicOperator
    internal dynamic operator for `func_alg`
""",
    'Equal': """Equality comparison block.

Compares two inputs and outputs 1.0 if |a - b| <= tolerance, else 0.0.

.. math::

    y =
    \\begin{cases}
    1 & , |a - b| \\leq \\epsilon \\\\
    0 & , |a - b| > \\epsilon
    \\end{cases}

Parameters
----------
tolerance : float
    comparison tolerance for floating point equality

Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'Exp': """Exponential operator block.

This block supports vector inputs. This is the operation it does:
    
.. math::
    
    \\vec{y} = e^{\\vec{u}} 
    
Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'FIR': """Discrete-time Finite-Impulse-Response (FIR) filter.

Applies an FIR filter to a periodically sampled input signal.

.. math::

    y[n] = b_0 x[n] + b_1 x[n-1] + \\dots + b_N x[n-N]

where ``b`` are the filter coefficients and ``N`` is the filter order
(number of coefficients minus one). The output is held constant
between sample times.

Note
----
Supports vector input — the same coefficients are applied to each
channel in parallel.

Parameters
----------
coeffs : array_like
    FIR filter coefficients ``[b0, b1, ..., bN]``
T : float
    sampling period
tau : float
    delay before first sample

Attributes
----------
events : list[Schedule]
    internal scheduled event for periodic filter evaluation
""",
    'FirstOrderHold': """First-order hold reconstructor.

Reconstructs a continuous signal from periodic samples using linear
extrapolation across one sampling interval. Causal (one-sample-lag)
variant matching the Simulink ``First-Order Hold`` block.

Between two consecutive sample times :math:`t_{k-1}` and :math:`t_k`,
the output is

.. math::

    y(t) = u_{k-1} + \\frac{u_{k-1} - u_{k-2}}{T} (t - t_{k-1})

During the very first interval (only one sample captured) the output
is held at the most recent sample.

Note
----
Supports vector input — each channel is extrapolated independently.

Parameters
----------
T : float
    sampling period
tau : float
    delay before first sample

Attributes
----------
events : list[Schedule]
    internal scheduled event for periodic sampling
""",
    'Function': """Arbitrary MIMO function block, defined by a function or `lambda` expression.

The function can have multiple arguments that are then provided 
by the input channels of the function block.

Form multi input, the function has to specify multiple arguments
and for multi output, the aoutputs have to be provided as a 
tuple or list. 

In the context of the global system, this block implements algebraic 
components of the global system ODE/DAE.

.. math::

    \\vec{y} = \\mathrm{func}(\\vec{u})


Note
----
This block is purely algebraic and its operation (`op_alg`) will be called 
multiple times per timestep, each time when `Simulation._update(t)` is 
called in the global simulation loop.
Therefore `func` must be purely algebraic and not introduce states, 
delay, etc. For interfacing with external stateful APIs, use the 
`Wrapper` block.


Note
-----
If the outputs are provided as a single numpy array, they are 
considered a single output. For MIMO, output has to be tuple.


Example
-------
consider the function: 

.. code-block:: python

    from pathsim.blocks import Function

    def f(a, b, c):
        return a**2, a*b, b/c

    fn = Function(f)
    

then, when the block is updated, the input channels of the block are 
assigned to the function arguments following this scheme:

.. code-block::

    inputs[0] -> a
    inputs[1] -> b
    inputs[2] -> c

and the function outputs are assigned to the 
output channels of the block in the same way:

.. code-block::

    a**2 -> outputs[0]
    a*b  -> outputs[1]
    b/c  -> outputs[2]

Because the `Function` block only has a single argument, it can be 
used to decorate a function and make it a `PathSim` block. This might 
be handy in some cases to keep definitions concise and localized 
in the code:

.. code-block:: python

    from pathsim.blocks import Function

    #does the same as the definition above
        
    @Function
    def fn(a, b, c):
        return a**2, a*b, b/c

    #'fn' is now a PathSim block


Parameters
---------- 
func : callable
    MIMO function that defines algebraic block IO behaviour, signature `func(*tuple)`


Attributes
----------
op_alg : Operator
    internal algebraic operator that wraps `func`

""",
    'GaussianPulseSource': """Source block that generates a gaussian pulse
    
Parameters
----------
amplitude : float
    amplitude of the gaussian pulse
f_max : float
    maximum frequency component of the gaussian pulse (steepness)
tau : float
    time delay of the gaussian pulse 
""",
    'GreaterThan': """Greater-than comparison block.

Compares two inputs and outputs 1.0 if a > b, else 0.0.

.. math::

    y =
    \\begin{cases}
    1 & , a > b \\\\
    0 & , a \\leq b
    \\end{cases}

Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'Integrator': """Integrates the input signal.

Uses a numerical integration engine like this:

.. math::

    y(t) = \\int_0^t u(\\tau) \\ d \\tau

or in differential form like this:

.. math::
    \\begin{align}
        \\dot{x}(t) &= u(t) \\\\
               y(t) &= x(t)
    \\end{align}

The Integrator block is inherently MIMO capable, so `u` 
and `y` can be vectors.

Example
-------
This is how to initialize the integrator: 

.. code-block:: python

    #initial value 0.0
    i1 = Integrator()

    #initial value 2.5
    i2 = Integrator(2.5)


Parameters
----------
initial_value : float, array
    initial value of integrator
""",
    'LUT1D': """One-dimensional lookup table with linear interpolation functionality.

This class implements a 1-dimensional lookup table that uses scipy's interp1d [#scipy]_
for piecewise linear interpolation along a single axis. The interpolation
provides linear interpolation between adjacent data points and supports
extrapolation beyond the input data range using the 'extrapolate' fill mode.

The LUT1D acts as a Function block.


References
----------
.. [#scipy] https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.interpolate.interp1d.html


Parameters
----------
points : array_like of shape (n,)
    1-D array of monotonically increasing data point coordinates where n 
    is the number of points. These represent the independent variable values
    at which the dependent values are known.
values : array_like of shape (n,) or (n, m)
    1-D or 2-D array of data values at the corresponding points. If 1-D,
    represents scalar values at each point. If 2-D with shape (n, m), 
    each column represents a different output dimension, allowing the
    lookup table to return m-dimensional vectors.
fill_value : float or str, optional
    The value to use for points outside the interpolation range. If "extrapolate",
    the interpolator will use linear extrapolation. Default is "extrapolate".
    See https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.interpolate.interp1d.html for more details


Attributes
----------
points : ndarray
    Flattened array of input point coordinates, stored as 1-D array.
values : ndarray
    Stored array of output values at each point, preserving original shape.
inter : scipy.interpolate.interp1d
    The scipy 1D interpolator object used for linear interpolation with
    extrapolation enabled beyond the data range.
""",
    'LeadLag': """Lead-Lag compensator.

The transfer function is defined as

.. math::

    H(s) = K \\frac{T_1 s + 1}{T_2 s + 1}

where `K` is the static gain, `T1` is the lead time constant
and `T2` is the lag time constant.

- :math:`T_1 > T_2`: lead compensator (phase advance)
- :math:`T_1 < T_2`: lag compensator (phase lag)
- :math:`T_1 = T_2`: pure gain


Example
-------
The block is initialized like this:

.. code-block:: python

    #lead compensator
    ll = LeadLag(K=1.0, T1=0.5, T2=0.1)


Parameters
----------
K : float
    static gain
T1 : float
    lead (numerator) time constant in seconds
T2 : float
    lag (denominator) time constant in seconds (must be > 0)
""",
    'LessThan': """Less-than comparison block.

Compares two inputs and outputs 1.0 if a < b, else 0.0.

.. math::

    y =
    \\begin{cases}
    1 & , a < b \\\\
    0 & , a \\geq b
    \\end{cases}

Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'Log': """Natural logarithm operator block.

This block supports vector inputs. This is the operation it does:
    
.. math::
    
    \\vec{y} = \\ln(\\vec{u}) 
    
Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'Log10': """Base-10 logarithm operator block.

This block supports vector inputs. This is the operation it does:
    
.. math::
    
    \\vec{y} = \\log_{10}(\\vec{u}) 
    
Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'LogicAnd': """Logical AND block.

Outputs 1.0 if both inputs are nonzero, else 0.0.

.. math::

    y = a \\land b

Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'LogicNot': """Logical NOT block.

Outputs 1.0 if input is zero, else 0.0.

.. math::

    y = \\lnot x

Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'LogicOr': """Logical OR block.

Outputs 1.0 if either input is nonzero, else 0.0.

.. math::

    y = a \\lor b

Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'Matrix': """Linear matrix operation (matrix-vector product).

This block supports vector inputs. This is the operation it does:
    
.. math::
    
    \\vec{y} = \\mathbf{A} \\vec{u} 

Parameters
----------
A : np.ndarray
    matrix, 2d array with dim=2
    
Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'Mod': """Modulo operator block.

This block supports vector inputs. This is the operation it does:
    
.. math::
    
    \\vec{y} = \\vec{u} \\bmod m


Note
----
modulo is not differentiable at discontinuities

Parameters
----------
modulus : float
    modulus value
    
Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'Multiplier': """Multiplies all signals from all input ports (MISO).
  
.. math::
    
    y(t) = \\prod_i u_i(t)

        
Note
----
This block is purely algebraic and its operation (`op_alg`) will be called 
multiple times per timestep, each time when `Simulation._update(t)` is 
called in the global simulation loop.


Attributes
----------
op_alg : Operator
    internal algebraic operator that wraps 'prod'
""",
    'Norm': """Vector norm operator block.

This block computes the Euclidean norm of the input vector:
    
.. math::
    
    y = \\|\\vec{u}\\|_2 = \\sqrt{\\sum_i u_i^2}
    
Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'ODE': """Ordinary differential equation (ODE) defined by its right hand side function.

.. math::

    \\begin{align}
        \\dot{x}(t) &= \\mathrm{func}(x(t), u(t), t) \\\\
               y(t) &= x(t)
    \\end{align}

with inhomogenity (input) `u` and state vector `x`. The function can be nonlinear 
and the ODE can be of arbitrary order. The block utilizes the integration engine 
to solve the ODE by integrating the `func`, which is the right hand side function.

Example
-------

For example a linear 1st order ODE:

.. code-block:: python
    
    ode = ODE(lambda x, u, t: -x)

Or something more complex like the `Van der Pol` system, where it makes sense to 
also specify the jacobian, which improves convergence for implicit solvers but is 
not needed in most cases: 

.. code-block:: python
    
    import numpy as np
        
    #initial condition
    x0 = np.array([2, 0])

    #van der Pol parameter
    mu = 1000

    def func(x, u, t):
        return np.array([x[1], mu*(1 - x[0]**2)*x[1] - x[0]])

    #analytical jacobian (optional)
    def jac(x, u, t):
        return np.array(
            [[0                , 1               ], 
             [-mu*2*x[0]*x[1]-1, mu*(1 - x[0]**2)]]
             )

    #finally the block
    vdp = ODE(func, x0, jac) 
    
Parameters
----------
func : callable
    right hand side function of ODE
initial_value : array[float]
    initial state / initial condition
jac : callable, None
    jacobian of 'func' or 'None'

Attributes
----------
op_dyn : DynamicOperator
    internal dynamic operator for ODE right hand side 'func'
""",
    'PID': """Proportional-Integral-Differentiation (PID) controller.

The transfer function is defined as

.. math::

    H(s) = K_p + K_i \\frac{1}{s} + K_d \\frac{s}{1 + s / f_\\mathrm{max}}

where the differentiation is approximated by a high pass filter that holds
for signals up to a frequency of approximately `f_max`.

Internally realized as a linear state space model with two states
(differentiator filter state and integrator state).


Note
----
Depending on `f_max`, the resulting system might become stiff or ill conditioned!
As a practical choice set `f_max` to 3x the highest expected signal frequency.
Since this block uses an approximation of real differentiation, the approximation will
not hold if there are high frequency components present in the signal. For example if
you have discontinuities such as steps or square waves.


Example
-------
The block is initialized like this:

.. code-block:: python

    #cutoff at 1kHz
    pid = PID(Kp=2, Ki=0.5, Kd=0.1, f_max=1e3)


Parameters
----------
Kp : float
    proportional controller coefficient
Ki : float
    integral controller coefficient
Kd : float
    differentiator controller coefficient
f_max : float
    highest expected signal frequency
""",
    'PT1': """First-order lag element (PT1).

The transfer function is defined as

.. math::

    H(s) = \\frac{K}{1 + T s}

where `K` is the static gain and `T` is the time constant.


Example
-------
The block is initialized like this:

.. code-block:: python

    pt1 = PT1(K=2.0, T=0.5)


Parameters
----------
K : float
    static gain
T : float
    time constant in seconds (must be > 0)
""",
    'PT2': """Second-order lag element (PT2).

The transfer function is defined as

.. math::

    H(s) = \\frac{K}{1 + 2 d T s + T^2 s^2}

where `K` is the static gain, `T` is the time constant
(related to the natural frequency by :math:`\\omega_n = 1/T`)
and `d` is the damping ratio.

The damping ratio `d` controls the transient behavior:

- :math:`d < 1`: underdamped (oscillatory)
- :math:`d = 1`: critically damped
- :math:`d > 1`: overdamped


Example
-------
The block is initialized like this:

.. code-block:: python

    #underdamped second-order system
    pt2 = PT2(K=1.0, T=0.1, d=0.3)


Parameters
----------
K : float
    static gain
T : float
    time constant in seconds (must be > 0)
d : float
    damping ratio (must be >= 0)
""",
    'PinkNoise': """Pink noise (1/f noise) source using the Voss-McCartney algorithm.

Generates noise with power spectral density proportional to 1/f, where
lower frequencies have more power than higher frequencies.

The algorithm maintains ``num_octaves`` independent random values representing
different frequency bands. At each sample, one octave is updated based on the
binary representation of the sample counter, creating the characteristic 1/f
spectrum through the superposition of different update rates.


Note
----
If ``spectral_density`` is provided, it takes precedence over ``standard_deviation``.
If ``sampling_period`` is set, noise is sampled at fixed intervals (zero-order hold).


Parameters
----------
standard_deviation : float
    approximate output standard deviation (default: 1.0)
spectral_density : float, optional
    power spectral density, output scaled as √(S₀/(N·dt))
num_octaves : int
    number of frequency bands in algorithm (default: 16)
sampling_period : float, optional
    time between samples, if None samples every timestep
seed : int, optional
    random seed for reproducibility
""",
    'Polynomial': """Polynomial operator block.

Evaluates a polynomial in the input. The coefficients follow the
`numpy.polyval` convention, with the highest order term first:

.. math::

    \\vec{y} = c_0 \\vec{u}^n + c_1 \\vec{u}^{n-1} + \\dots + c_{n-1} \\vec{u} + c_n

This block supports vector inputs (the polynomial is evaluated
element-wise).

Example
-------
Quadratic :math:`y = 2 u^2 + 3 u + 1`:

.. code-block:: python

    p = Polynomial(coeffs=[2, 3, 1])

Parameters
----------
coeffs : array_like
    polynomial coefficients in descending order of power,
    following the ``numpy.polyval`` convention

Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'Pow': """Raise to power operator block.

This block supports vector inputs. This is the operation it does:
    
.. math::
    
    \\vec{y} = \\vec{u}^{p} 

Parameters
----------
exponent : float, array_like
    exponent to raise the input to the power of
    
Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'PowProd': """Power-Product operator block.

This block raises each input to a power and then multiplies all results together:
    
.. math::
    
    y = \\prod_i u_i^{p_i}

Parameters
----------
exponents : float, array_like
    exponent(s) to raise the inputs to the power of. If scalar, 
    applies same exponent to all inputs.
    
Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'Pulse': """Alias for PulseSource.

.. deprecated:: 1.0.0
   Use :func:`PulseSource` instead.""",
    'PulseSource': """Generates a periodic pulse waveform with defined rise and fall times.

Scheduled events trigger phase changes (low, rising, high, falling),
and the `update` method calculates the output value based on the
current phase, performing linear interpolation during rise and fall.

Parameters
----------
amplitude : float, optional
    Peak amplitude of the pulse. Default is 1.0.
T : float, optional
    Period of the pulse train. Must be positive. Default is 1.0.
t_rise : float, optional
    Duration of the rising edge. Default is 0.0.
t_fall : float, optional
    Duration of the falling edge. Default is 0.0.
tau : float, optional
    Initial delay before the first pulse cycle begins. Default is 0.0.
duty : float, optional
    Duty cycle, ratio of the pulse ON duration (plateau time only)
    to the total period T (must be between 0 and 1). Default is 0.5.
    The high plateau duration is `T * duty`.

Attributes
----------
events : list[Schedule]
    Internal scheduled events triggering phase transitions.
_phase : str
    Current phase of the pulse ('low', 'rising', 'high', 'falling').
_phase_start_time : float
    Simulation time when the current phase began.
""",
    'RandomNumberGenerator': """Generates a random output value using `numpy.random.rand`.

If no `sampling_period` (None) is specified, every simulation timestep gets
a random value. Otherwise an internal `Schedule` event is used to periodically
sample a random value and set the output like a zero-order-hold stage.

Parameters
----------
sampling_period : float, None
    time between random samples

Attributes
----------
_sample : float
    internal random number state in case that
    no `sampling_period` is provided
Evt : Schedule
    internal event that periodically samples a random
    value in case `sampling_period` is provided
""",
    'RateLimiter': """Rate limiter block that limits the rate of change of a signal.

Implements a continuous-time rate limiter as a first-order tracking system
with clipped rate of change:

.. math::

    \\dot{x} = \\mathrm{clip}\\left(f_\\mathrm{max} (u - x),\\; -r,\\; r\\right)

where `r` is the maximum allowed rate and `f_max` controls the tracking
bandwidth when the signal is not rate-limited. The output is the state
:math:`y = x`.


Note
----
The parameter `f_max` should be set high enough that the output tracks
the input without lag when the rate is within limits.


Example
-------
The block is initialized like this:

.. code-block:: python

    #max rate of 10 units/s
    rl = RateLimiter(rate=10.0, f_max=1e3)


Parameters
----------
rate : float
    maximum rate of change (positive value)
f_max : float
    tracking bandwidth parameter
""",
    'Relay': """Relay block with hysteresis (Schmitt trigger).

Switches output between two values based on input crossing upper and lower 
thresholds. The hysteresis prevents rapid switching when input is noisy.

When input rises above `threshold_up`, output switches to `value_up`.
When input falls below `threshold_down`, output switches to `value_down`.

Examples
--------
Basic thermostat that turns heater on below 19°C, off above 21°C:

.. code-block:: python

    from pathsim.blocks import Relay
    
    thermostat = Relay(
        threshold_up=21.0, 
        threshold_down=19.0,
        value_up=0.0, 
        value_down=1.0
        )

Parameters
----------
threshold_up : float
    threshold for transitioning to upper relay state `value_up` (default: 1.0)
threshold_down : float
    threshold for transitioning to lower relay state `value_down` (default: 0.0)
value_up : float
    value for upper relay state (default: 1.0)
value_down : float
    value for lower relay state (default: 0.0)

Attributes
----------
events : list[ZeroCrossingUp, ZeroCrossingDown]
    internal zero crossing events for relay state transitions
""",
    'Rescale': """Linear rescaling / mapping block.

Maps the input linearly from range ``[i0, i1]`` to range ``[o0, o1]``.
Optionally saturates the output to ``[o0, o1]``.

.. math::

    y = o_0 + \\frac{(x - i_0) \\cdot (o_1 - o_0)}{i_1 - i_0}

This block supports vector inputs.

Parameters
----------
i0 : float
    input range lower bound
i1 : float
    input range upper bound
o0 : float
    output range lower bound
o1 : float
    output range upper bound
saturate : bool
    if True, clamp output to [min(o0,o1), max(o0,o1)]

Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'SampleHold': """Zero-order hold: samples the input periodically and holds it at the output.

.. math::

    y(t) = u(k T + \\tau), \\quad k T + \\tau \\leq t < (k+1) T + \\tau

Note
----
Supports vector input — each channel is sampled independently.

Parameters
----------
T : float
    sampling period
tau : float
    delay before first sample

Attributes
----------
events : list[Schedule]
    internal scheduled event for periodic sampling
""",
    'Scope': """Block for recording time domain data with variable sampling period.

A time threshold can be set by `t_wait` to start recording data after the simulation
time is larger then the specified waiting time, i.e. `t - t_wait > 0`.
This is useful for recording data only after all the transients have settled.

The block uses an internal `Schedule` event, when `sampling_period` is provided,
otherwise it just samples at every simulation timestep.

Parameters
----------
sampling_period : float, None
    time between samples, default is every timestep
t_wait : float
    wait time before starting recording, optional
labels : list[str]
    labels for the scope traces, and for the csv, optional

Attributes
----------
recording_time : list[float]
    recorded time points
recording_data : list[float]
    recorded data points
_incremental_idx : int
    index for incremental reading of accumulated data since last
    call of incremental read
_sample_next_timestep : bool
    flag to indicate this is a timestep to sample, only used for
    event based sampling when `sampling_period` is provided as an arg
events : list[Schedule]
    internal scheduled event for periodic input sampling when
    `sampling_period` is provided
""",
    'Sin': """Sine operator block.

This block supports vector inputs. This is the operation it does:
    
.. math::
    
    \\vec{y} = \\sin(\\vec{u}) 


Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'Sinh': """Hyperbolic sine operator block.

This block supports vector inputs. This is the operation it does:
    
.. math::
    
    \\vec{y} = \\sinh(\\vec{u}) 
    
Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'SinusoidalPhaseNoiseSource': """Sinusoidal source with cumulative and white phase noise.

Generates a sinusoid with additive phase noise from two components:

- White phase noise: sampled from a normal distribution at each sample
- Cumulative phase noise: integrated random walk process

The output is given by:

.. math::

    y(t) = A \\sin\\left(\\omega t + \\varphi_0 + \\sigma_w n_w(t) + \\sigma_c \\int_0^t n_c(\\tau) d\\tau\\right)

where :math:`A` is amplitude, :math:`\\omega = 2\\pi f` is angular frequency,
:math:`\\varphi_0` is initial phase, :math:`\\sigma_w` and :math:`\\sigma_c` are
the white and cumulative noise weights, and :math:`n_w(t)` and :math:`n_c(t)` are
independent standard normal random processes sampled at the specified sampling period.

Parameters
----------
frequency : float
    frequency of the sinusoid
amplitude : float
    amplitude of the sinusoid
phase : float
    initial phase of the sinusoid (radians)
sig_cum : float
    weight for cumulative phase noise contribution
sig_white : float
    weight for white phase noise contribution
sampling_period : float, None
    time between phase noise samples. If None,
    noise is sampled every timestep (default is 0.1)

Attributes
----------
omega : float
    angular frequency of the sinusoid, derived from `frequency`
noise_1 : float
    internal noise value for white phase noise
noise_2 : float
    internal noise value for cumulative phase noise
events : list[Schedule]
    scheduled event for periodic sampling (only if sampling_period is set)
""",
    'SinusoidalSource': """Source block that generates a sinusoid wave
    
Parameters
----------
frequency : float
    frequency of the sinusoid
amplitude : float
    amplitude of the sinusoid
phase : float
    phase of the sinusoid
""",
    'Source': """Source that produces an arbitrary time dependent output defined by `func` (callable).

.. math::

    y(t) = \\mathrm{func}(t)


Note
----
This block is purely algebraic and its internal function (`func`) will 
be called multiple times per timestep, each time when `Simulation._update(t)` 
is called in the global simulation loop.


Example
-------
For example a ramp:

.. code-block:: python

    from pathsim.blocks import Source

    src = Source(lambda t : t)

or a simple sinusoid with some frequency:

.. code-block:: python
    
    import numpy as np
    from pathsim.blocks import Source

    #some parameter
    omega = 100

    #the function that gets evaluated
    def f(t):
        return np.sin(omega * t)

    src = Source(f)
 
Because the `Source` block only has a single argument, it can be 
used to decorate a function and make it a `PathSim` block. This might 
be handy in some cases to keep definitions concise and localized 
in the code:

.. code-block:: python
    
    import numpy as np
    from pathsim.blocks import Source

    #does the same as the definition above
        
    @Source
    def src(t):
        omega = 100
        return np.sin(omega * t)

    #'src' is now a PathSim block


Parameters
---------- 
func : callable
    function defining time dependent block output
""",
    'Spectrum': """Block for fourier spectrum analysis (spectrum analyzer).

Computes continuous time running fourier transform (RFT) of the incoming signal.

A time threshold can be set by 't_wait' to start recording data only after the 
simulation time is larger then the specified waiting time, i.e. 't - t_wait > dt'. 
This is useful for recording the steady state after all the transients have settled.

An exponential forgetting factor 'alpha' can be specified for realtime spectral 
analysis. It biases the spectral components exponentially to the most recent signal 
values by applying a single sided exponential window like this:

.. math::

    \\int_0^t u(\\tau) \\exp(\\alpha (t-\\tau))  \\exp(-j \\omega \\tau)\\ d \\tau

It is also known as the 'exponentially forgetting transform' (EFT) and a form of 
short time fourier transform (STFT). It is implemented as a 1st order statespace model 
    
.. math::

    \\dot{x} = - \\alpha  x +  \\exp(-j \\omega t) u

where 'u' is the input signal and 'x' is the state variable that represents the 
complex fourier coefficient to the frequency 'omega'. The ODE is integrated using the 
numerical integration engine of the block.

Example
-------
This is how to initialize it: 

.. code-block:: python

    import numpy as np
    
    #linear frequencies (0Hz, DC -> 1kHz)
    sp1 = Spectrum(
        freq=np.linspace(0, 1e3, 100),
        labels=['x1', 'x2'] #labels for two inputs
        )

    #log frequencies (1Hz -> 1kHz)
    sp2 = Spectrum(
        freq=np.logspace(0, 3, 100)
        )
    
    #log frequencies including DC (0Hz, DC + 1Hz -> 1kHz)
    sp3 = Spectrum(
        freq=np.hstack([0.0, np.logspace(0, 3, 100)])
        )

    #arbitrary frequencies
    sp4 = Spectrum(
        freq=np.array([0, 0.5, 20, 1e3])
        )

Note
----
This block is relatively slow! But it is valuable for long running simulations 
with few evaluation frequencies, where just FFT'ing the time series data 
wouldnt be efficient OR if only the evaluation at weirdly spaced frequencies 
is required. Otherwise its more efficient to just do an FFT on the time 
series recording after the simulation has finished.


Parameters
----------
freq : array[float] 
    list of evaluation frequencies for RFT, can be arbitrarily spaced
t_wait : float 
    wait time before starting RFT
alpha : float
    exponential forgetting factor for realtime spectrum
labels : list[str]
    labels for the inputs
""",
    'Sqrt': """Square root operator block.

This block supports vector inputs. This is the operation it does:
    
.. math::
    
    \\vec{y} = \\sqrt{|\\vec{u}|} 
    
Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'SquareWaveSource': """Discrete time square wave source.

Utilizes scheduled events to periodically set 
the block output at discrete times.

Parameters
----------
amplitude : float
    amplitude of the square wave signal
frequency : float
    frequency of the square wave signal
phase : float
    phase of the square wave signal

Attributes
----------
events : list[Schedule]
    internal scheduled events 
""",
    'StateSpace': """Linear time invariant (LTI) multi input multi output (MIMO) state space model.

.. math::

    \\begin{align}
        \\dot{x} &= \\mathbf{A} x + \\mathbf{B} u \\\\
               y &= \\mathbf{C} x + \\mathbf{D} u
    \\end{align}

where `A`, `B`, `C` and `D` are the state space matrices, `x` is the state, 
`u` the input and `y` the output vector.

Example
-------
A SISO state space block with two internal states can be initialized 
like this:

.. code-block:: python

    S = StateSpace(
        A=-np.eye(2), 
        B=np.ones((2, 1)), 
        C=np.ones((1, 2)), 
        D=1.0
        )

and a MIMO (2 in, 2 out) state space block with three internal states 
can be initialized like this:

.. code-block:: python

    S = StateSpace(
        A=-np.eye(3), 
        B=np.ones((3, 2)), 
        C=np.ones((2, 3)), 
        D=np.ones((2, 2))
        )

Parameters
----------
A, B, C, D : array_like
    real valued state space matrices
initial_value : array_like, None
    initial state / initial condition

Attributes
----------
op_dyn : DynamicOperator
    internal dynamic operator for state equation
op_alg : DynamicOperator
    internal algebraic operator for mapping to outputs
""",
    'Step': """Alias for StepSource.

.. deprecated:: 1.0.0
   Use :func:`StepSource` instead.""",
    'StepSource': """Discrete time unit step (or multi step) source block.

Utilizes a scheduled event to set the block output 
to the specified output levels at the defined event times.

The arguments can be vectorial and in that case, the output is set to the 
amplitude that corresponds to the defined delay like a zero-order-hold stage. 
This functionality enables adding external or time series measurement data 
into the system.


Examples
--------

This is how to use the source as a unit step source:

.. code-block:: python

    from pathsim.blocks import StepSource
    
    #default, starts at 0, jumps to 1
    stp = StepSource()


And this is how to configure it with multiple consecutive steps:

.. code-block:: python

    from pathsim.blocks import StepSource
    
    #starts at 0, jumps to 1 at 1, jumps to -1 at 2 and jumps back to 0 at 3
    stp = StepSource(amplitude=[1, -1, 0], tau=[1, 2, 3])


Similarly implementing measured time series data via zoh:

.. code-block:: python

    import numpy as np
    from pathsim.blocks import StepSource
    
    #some random time series arrays
    times, data = np.linspace(0, 100, 1000), np.random.rand(1000)
    
    #pass them to the block
    stp = StepSource(amplitude=data, tau=times)


Parameters
----------
amplitude : float | list[float]
    amplitude of the step signal, or amplitudes / output 
    levels of the multiple steps
tau : float | list[float]
    delay of the step, or delays of the different steps

Attributes
----------
Evt : ScheduleList
    internal scheduled event directly accessible
events : list[ScheduleList]
    list of interna events
""",
    'Switch': """Switch block that selects between its inputs.

Example
-------
The block is initialized like this:

.. code-block:: python 
    
    #default None -> no passthrough 
    s1 = Switch()

    #selecting port 2 as passthrough
    s2 = Switch(2)

    #change the state of the switch to port 3
    s2.select(3)

Sets block output depending on `self.switch_state` like this:

.. code-block::

    switch_state == None -> outputs[0] = 0

    switch_state == 0 -> outputs[0] = inputs[0]

    switch_state == 1 -> outputs[0] = inputs[1]

    switch_state == 2 -> outputs[0] = inputs[2]

    ...

Parameters
----------
switch_state : int, None
    state of the switch

""",
    'Tan': """Tangent operator block.

This block supports vector inputs. This is the operation it does:
    
.. math::
    
    \\vec{y} = \\tan(\\vec{u}) 
    
Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'Tanh': """Hyperbolic tangent operator block.

This block supports vector inputs. This is the operation it does:
    
.. math::
    
    \\vec{y} = \\tanh(\\vec{u}) 
    
Attributes
----------
op_alg : Operator
    internal algebraic operator
""",
    'TappedDelay': """Tapped delay line.

Outputs the current and ``N-1`` past samples of the input as parallel
signals. The block has ``N`` outputs:

.. math::

    y_i[k] = u[k - i], \\quad i = 0, 1, \\dots, N-1

Parameters
----------
N : int
    number of taps (output ports)
T : float
    sampling period
tau : float
    delay before first sample

Attributes
----------
events : list[Schedule]
    internal scheduled event for periodic shift
""",
    'TransferFunction': """Alias for TransferFunctionPRC.

.. deprecated:: 1.0.0
   Use :func:`TransferFunctionPRC` instead.""",
    'TransferFunctionNumDen': """This block defines a LTI (SISO) transfer function.

The transfer function is defined in polynomial (numerator-denominator) form

.. math::
    
    \\mathbf{H}(s) = \\frac{b_n + b_{n-1} s + \\dots + b_{0} s^n}{a_m + a_{m-1} s + \\dots + a_{0} s^m}

where `Num` is the list of numerator polynomial coefficients and `Den` the 
list of denominator coefficients.

Upon initialization, the state space realization of the transfer function is 
computed using `scipy.signal.TransferFunction(Num, Den).to_ss()`.

The resulting state space model of the form

.. math::
    
    \\begin{align}
        \\dot{x} &= \\mathbf{A} x + \\mathbf{B} u \\\\
               y &= \\mathbf{C} x + \\mathbf{D} u
    \\end{align}

is handled the same as the 'StateSpace' block, where `A`, `B`, `C` and `D` 
are the state space matrices, `x` is the internal state, `u` the input and 
`y` the output vector.
    
Parameters
----------
Num : array_like
    numerator polynomial coefficients
Den : array_like
    denominator polynomial coefficients
""",
    'TransferFunctionPRC': """This block defines a LTI (MIMO for pole residue) transfer function.

The transfer function is defined in pole-residue-constant (PRC) form

.. math::
    
    \\mathbf{H}(s) = \\mathbf{C} + \\sum_n^N \\frac{\\mathbf{R}_n}{s - p_n}

where 'Poles' are the scalar (possibly complex conjugate) poles of the 
transfer function and 'Residues' are the possibly matrix valued (in MIMO case) 
and complex conjugate residues of the transfer function. 'Const' has same 
shape as 'Residues'.

Upon initialization, the state space realization of the transfer 
function is computed using a minimal gilbert realization.

The resulting state space model of the form

.. math::
    
    \\begin{align}
        \\dot{x} &= \\mathbf{A} x + \\mathbf{B} u \\\\
               y &= \\mathbf{C} x + \\mathbf{D} u
    \\end{align}

is handled the same as the 'StateSpace' block, where `A`, `B`, `C` and `D` 
are the state space matrices, `x` is the internal state, `u` the input and 
`y` the output vector.
    
Parameters
----------
Poles : array
    transfer function poles
Residues : array
    transfer function residues
Const : array, float
    constant term of transfer function
""",
    'TransferFunctionZPG': """This block defines a LTI (SISO) transfer function.

The transfer function is defined in zeros-poles-gain (ZPG) form

.. math::
    
    \\mathbf{H}(s) = k \\frac{(s - z_1)(s - z_2)\\cdots(s - z_m)}{(s - p_1)(s - p_2)\\cdots(s - p_n)}

where `Zeros` are the scalar (possibly complex conjugate) zeros of the 
transfer function, and `Poles` are the poles (denominator zeros) of the 
transfer function. `Gain` is the scalar factor `k`.

Upon initialization, the state space realization of the transfer function is 
computed using `scipy.signal.ZerosPolesGain(Zeros, Poles, Gain).to_ss()`.

The resulting state space model of the form

.. math::
    
    \\begin{align}
        \\dot{x} &= \\mathbf{A} x + \\mathbf{B} u \\\\
               y &= \\mathbf{C} x + \\mathbf{D} u
    \\end{align}

is handled the same as the 'StateSpace' block, where `A`, `B`, `C` and `D` 
are the state space matrices, `x` is the internal state, `u` the input and 
`y` the output vector.
    
Parameters
----------
Poles : array_like
    transfer function poles
Zeros : array_like
    transfer function zeros
Gain : float
    gain term of transfer function 
""",
    'TriangleWaveSource': """Source block that generates an analog triangle wave
    
Parameters
----------
frequency : float
    frequency of the triangle wave
amplitude : float
    amplitude of the triangle wave
phase : float
    phase of the triangle wave
""",
    'WhiteNoise': """White noise source with Gaussian distribution.

Generates uncorrelated random samples with either constant amplitude
(``standard_deviation`` mode) or timestep-scaled amplitude for stochastic
integration (``spectral_density`` mode).

In spectral density mode, output is scaled as √(S₀/dt) so that integrating
the noise yields correct statistical properties (Wiener process).


Note
----
If ``spectral_density`` is provided, it takes precedence over ``standard_deviation``.
If ``sampling_period`` is set, noise is sampled at fixed intervals (zero-order hold).


Parameters
----------
standard_deviation : float
    output standard deviation for constant-amplitude mode (default: 1.0)
spectral_density : float, optional
    power spectral density S₀ in [signal²/Hz]
sampling_period : float, optional
    time between samples, if None samples every timestep
seed : int, optional
    random seed for reproducibility
""",
    'Wrapper': """Wrapper block for discrete implementation and external code integration.

The `Wrapper` class is designed to call the internal `func` at fixed intervals 
using an internal `Schedule` event. This makes it particularly useful for wrapping 
external code or implementing discrete-time systems.

Essentially this block does the same as `Function` with the difference that its 
not evaluated continuously but periodically at discrete times.


Example
-------
There are two ways to setup the `Wrapper`, first and standard way is to define 
a function to be wrapped and pass it to the block initializer:

.. code-block:: python
    
    from pathsim.blocks import Wrapper
    
    #function to be wrapped
    def func(a, b, c):
        return a * (b + c)

    wrp = Wrapper(func, T=0.1)


Another option is to use the `dec` classmethod, which might be more convenient 
in some situations:

.. code-block:: python
    
    from pathsim.blocks import Wrapper
    
    @Wrapper.dec(T=0.1)
    def wrp(a, b, c):
        return a * (b + c)


This way the internal function of the block `wrp` will be evaluated with a period 
of `T=0.1` and its outputs updated accordingly.


Parameters
----------
func : callable
    function that defines algebraic block IO behaviour
T : float
    sampling period for the wrapped function
tau : float
    delay time for the start time of the event
    
Attributes
----------
Evt : Schedule
    internal event. Used for periodic sampling the wrapped method
""",
    'ZeroOrderHold': """Zero-order hold: samples the input periodically and holds it at the output.

.. math::

    y(t) = u(k T + \\tau), \\quad k T + \\tau \\leq t < (k+1) T + \\tau

Note
----
Supports vector input — each channel is sampled independently.

Parameters
----------
T : float
    sampling period
tau : float
    delay before first sample

Attributes
----------
events : list[Schedule]
    internal scheduled event for periodic sampling
""",
}
