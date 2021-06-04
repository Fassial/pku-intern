"""
Created on 14:31, June. 4th, 2021
Author: fassial
Filename: LIF.py
"""
import numpy as np
import brainpy as bp
from numba import prange

__all__ = [
    "LIF",
]

class LIF(bp.NeuGroup):
    target_backend = ['numpy', 'numba', 'numba-parallel', 'numba-cuda']

    @staticmethod
    def derivative(V, t, Iext, V_rest, R, tau):
        dVdt = (-(V - V_rest) + R * Iext) / tau
        return dVdt

    def __init__(self, size,
        V_rest = 0., V_reset = -5., V_th = 20., V_init = "reset",
        R = 1., tau = 10., t_refractory = 1., noise = 0., **kwargs
    ):
        # init params
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.V_init = V_init
        self.R = R
        self.tau = tau
        self.t_refractory = t_refractory

        # init vars
        num = bp.size2len(size)
        self.t_last_spike = bp.ops.ones(num) * -1e7
        self.input = bp.ops.zeros(num)
        self.refractory = bp.ops.zeros(num, dtype=bool)
        self.spike = bp.ops.zeros(num, dtype=bool)
        self._init_V(num = num)

        # init integral
        if noise == 0.:
            self.integral = bp.odeint(
                f = LIF.derivative,
                method = "euler"
            )
        else:
            self.integral = bp.sdeint(
                f = LIF.derivative,
                g = lambda V, t, Iext, V_rest, R, tau: (noise / tau),
                method = "euler"
            )

        # init super
        super(LIF, self).__init__(size = size, **kwargs)

    def _init_V(self, num):
        if self.V_init == "gaussian":
            a = (self.V_th - self.V_reset) / 6
            b = (self.V_th - self.V_reset) / 2
            self.V = np.random.randn(num) * a + b
            self.V = np.clip(self.V, self.V_reset, self.V_th - 0.1)
        elif self.V_init == "uniform":
            self.V = np.random.rand(num) * (self.V_th - self.V_reset) + self.V_reset
        else:
            # default reset
            self.V = bp.ops.ones(num) * self.V_reset

    def update(self, _t):
        # update vars
        for i in prange(self.size[0]):
            spike = 0.
            refractory = (_t - self.t_last_spike[i] <= self.t_refractory)
            if not refractory:
                V = self.integral(
                    V = self.V[i],
                    t = _t,
                    Iext = self.input[i],
                    V_rest = self.V_rest,
                    R = self.R,
                    tau = self.tau
                )
                spike = (V >= self.V_th)
                if spike:
                    V = self.V_reset
                    self.t_last_spike[i] = _t
                self.V[i] = V
            self.spike[i] = spike
            self.refractory[i] = refractory or spike
            self.input[i] = 0.

if __name__ == "__main__":
    # set backend.dt
    bp.backend.set(dt = 0.01)
    bp.backend.set(backend = "numba")
    # inst lif & run lif
    lif_inst = LIF(size = (100, 100), monitors = ["V"])
    lif_inst.run(
        duration = 200.,
        inputs = [("input", 26.),],
        report = True
    )
    bp.visualize.line_plot(lif_inst.mon.ts, lif_inst.mon.V, show = True)

