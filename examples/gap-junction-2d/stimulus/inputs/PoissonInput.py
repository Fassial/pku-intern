"""
Created on 20:53, May. 22nd, 2021
Author: fassial
Filename: PoissonInput.py
"""
import numpy as np
import brainpy as bp

__all__ = [
    "PoissonInput",
]

class PoissonInput(bp.NeuGroup):
    target_backend = "general"

    def __init__(self, size, freqs, **kwargs):
        # init params
        self.freqs = freqs
        self.size = (size,) if isinstance(size, int) else tuple(size)

        # init vars
        self.dt = bp.backend.get_dt() / 1000.
        self.num = bp.size2len(size)
        self.spike = bp.ops.zeros(self.num, dtype = bool)
        self.t_last_spike = -1e7 * bp.ops.ones(self.num)

        # init super
        super(PoissonInput, self).__init__(size = size, steps = {
            "update": self.update,
        }, **kwargs)

    def update(self, _t):
        self.spike = np.random.random(self.num) <= (self.freqs * self.dt)
        self.t_last_spike = np.where(self.spike, _t, self.t_last_spike)

if __name__ == "__main__":
    # inst PoissonInput
    pi_inst = PoissonInput(
        size = (20,),
        freqs = 20,
        # monitor
        monitors = ["spike"]
    )
    # run pi_inst
    pi_inst.run(duration = 1000)
    # display state
    print(pi_inst.mon.spike)   # (duration / dt, size)

