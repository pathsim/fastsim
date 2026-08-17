# fastsim — Drop-in replacement for pathsim
#
# Module layout mirrors pathsim: the core classes live in their own modules
# (connection, subsystem, simulation, exceptions) and are re-exported here so
# `from fastsim import Simulation, Connection, Interface, Subsystem` works.

from fastsim.connection import Connection
from fastsim.subsystem import Interface, Subsystem
from fastsim.simulation import Simulation
from fastsim.exceptions import StopSimulation
from fastsim.adapter import adapt
from fastsim.port import port
from fastsim.random import random_uniform, random_normal

# Submodules, imported so they are attributes of the package rather than only
# importable by name. pathsim exposes `pathsim.solvers` / `pathsim.events` /
# `pathsim.blocks` after a bare `import pathsim`, and code written against it
# reaches them that way — `Solver=engine.solvers.RK4` in a factory that takes
# the engine module is the idiom this exists for. Without these, such a script
# ran under pathsim and raised `AttributeError` under fastsim.
from fastsim import blocks, events, solvers  # noqa: F401,E402
# fastsim-only: the standalone tracer/JIT and the IR types.
from fastsim import ir, jit  # noqa: F401,E402

__version__ = "0.28.0"
