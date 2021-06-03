"""
Created on 21:01, Apr. 5th, 2021
Author: fassial
Filename: LIF.py
"""
import numpy as np
import brainpy as bp

__all__ = [
    "LIF",
]

class LIF(bp.NeuGroup):
    target_backend = "general"

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

        # def kinetic func
        def diff(V, t, Iext, V_rest, R, tau):
            dVdt = (-(V - V_rest) + R * Iext) / tau
            return dVdt
        if noise == 0.:
            self.integral = bp.odeint(
                f = diff,
                method = "euler"
            )
        else:
            self.integral = bp.sdeint(
                f = diff,
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
        refractory = (_t - self.t_last_spike) <= self.t_refractory
        V = self.integral(
            V = self.V,
            t = _t,
            Iext = self.input,
            V_rest = self.V_rest,
            R = self.R,
            tau = self.tau
        )
        V = bp.ops.where(refractory, self.V, V)
        spike = (self.V_th <= V) & ~refractory
        self.t_last_spike = bp.ops.where(spike, _t, self.t_last_spike)
        self.V = bp.ops.where(spike, self.V_reset, V)
        self.refractory = refractory | spike
        self.input[:] = 0.
        self.spike = spike

if __name__ == "__main__":
    # set backend.dt
    bp.backend.set(dt = 0.01)
    # inst lif & run lif
    lif_inst = LIF(size = (100, 100), monitors = ["V"])
    lif_inst.run(
        duration = 200.,
        inputs = [("input", 26.),],
        report = True
    )
    bp.visualize.line_plot(lif_inst.mon.ts, lif_inst.mon.V, show = True)

