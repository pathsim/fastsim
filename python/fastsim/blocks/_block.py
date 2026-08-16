# fastsim.blocks._block — import-path parity with pathsim.blocks._block.
#
# pathsim keeps its `Block` base class here, and that is the path its own
# examples import from when they define a block of their own:
#
#     from pathsim.blocks._block import Block
#
#     class SAR(Block):
#         ...
#
# fastsim's `Block` is the Rust class re-exported from `fastsim.blocks`, so this
# module only makes the pathsim spelling work; there is no second class.

from fastsim._fastsim import Block

__all__ = ["Block"]
