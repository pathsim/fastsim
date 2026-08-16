# The array returned by `Block.outputs`, with assignments that reach the block.
#
# `inputs`/`outputs`/`state` return numpy arrays rather than pathsim's dict-like
# `Register` (see the note on the getters in src/pybindings/py/core.rs): one
# array type, no `block.outputs * 2` footgun. That decision stands — this is the
# same ndarray, only no longer a dead-end snapshot.
#
# A block written in Python reports its result by assigning to its outputs:
#
#     def _step(t):
#         for i in range(self.n_bits):
#             self.outputs[i] = (self.register >> i) & 1
#
# (pathsim's `example_sar.py`). Against a plain snapshot those writes went
# nowhere and the block silently emitted zeros.

import numpy as np


class OutputRegister(np.ndarray):
    """A block's outputs as an ndarray; item assignment writes through.

    Indexing past the end grows the register, as pathsim's `Register` does —
    which is why an out-of-range index is forwarded to the block instead of
    being rejected by numpy's bounds check.
    """

    def __array_finalize__(self, obj):
        # Views and slices of this array carry the block along, so `arr[:2][0] = x`
        # still lands. numpy calls this for every derived array.
        if obj is not None:
            self._block = getattr(obj, "_block", None)

    def __setitem__(self, index, value):
        block = getattr(self, "_block", None)
        if block is None:
            super().__setitem__(index, value)
            return
        if isinstance(index, (int, np.integer)) and int(index) >= self.size:
            block._set_output(int(index), float(value))
            return
        super().__setitem__(index, value)
        block.outputs = np.asarray(self, dtype=float).ravel().tolist()

    def __reduce__(self):
        # Pickling a live handle on a Rust block is not meaningful; degrade to
        # the plain array so `copy`/`pickle` of recorded data keeps working.
        return np.asarray(self).__reduce__()


def _view(array, block):
    """Wrap `array` as this block's writable output register."""
    out = np.asarray(array).view(OutputRegister)
    out._block = block
    return out
