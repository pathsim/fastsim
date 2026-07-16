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

__version__ = "0.20.0"
