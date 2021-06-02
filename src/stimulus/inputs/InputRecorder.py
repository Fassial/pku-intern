"""
Created on 00:11, May. 23rd, 2021
Author: fassial
Filename: InputRecorder.py
"""
import brainpy as bp
from copy import deepcopy

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
        self.Iext = []

        # init super
        super(InputRecorder, self).__init__(size = size, **kwargs)

    def update(self, _t):
        # update Iext
        self.Iext.append(deepcopy(self.input))
        # reset input
        self.input[:] = 0.

    def get_Iext(self):
        return bp.ops.as_tensor(self.Iext)

