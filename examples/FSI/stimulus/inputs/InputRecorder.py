"""
Created on 00:11, May. 23rd, 2021
Author: fassial
Filename: InputRecorder.py
"""
import brainpy as bp

__all__ = [
    "InputRecorder",
]

class InputRecorder(bp.NeuGroup):
    target_backend = "general"

    def __init__(self, size, **kwargs):
        # init params
        self.size = (size,) if isinstance(size, int) else tuple(size)

        # init vars
        self.input = bp.ops.zeros(bp.size2len(self.size))
        self.Iext = bp.ops.zeros(bp.size2len(self.size))

        # init super
        super(InputRecorder, self).__init__(size = size, **kwargs)

    def update(self, _t):
        # update Iext
        self.Iext = bp.ops.vstack((self.Iext, self.input))
        # reset input
        self.input[:] = 0.

    def get_Iext(self):
        return self.Iext[1:,:]

