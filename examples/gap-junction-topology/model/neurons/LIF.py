"""
Created on 21:01, Apr. 5th, 2021
Author: fassial
Filename: LIF.py
"""
import brainpy as bp

__all__ = [
    "LIF",
]

class LIF(bp.NeuGroup):
    target_backend = "general"

    def __init__(self, size,
        V_rest = 0., V_reset = -5., V_th = 20.,
        R = 1., tau = 10., t_refractory = 1.,
        noise = 0., **kwargs
    ):
        # init params
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.R = R
        self.tau = tau
        self.t_refractory = t_refractory

        # init vars
        num = bp.size2len(size)
        self.t_last_spike = bp.backend.ones(num) * -1e7
        self.input = bp.backend.zeros(num)
        self.V = bp.backend.ones(num) * V_reset
        self.refractory = bp.backend.zeros(num, dtype=bool)
        self.spike = bp.backend.zeros(num, dtype=bool)

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
        V = bp.backend.where(refractory, self.V, V)
        spike = self.V_th <= V
        self.t_last_spike = bp.backend.where(spike, _t, self.t_last_spike)
        self.V = bp.backend.where(spike, self.V_reset, V)
        self.refractory = refractory
        self.input[:] = 0.
        self.spike = spike

if __name__ == "__main__":
    # set backend.dt
    bp.backend.set(dt = 0.01)
    # inst lif & run lif
    lif_inst = LIF(size = 100, monitors = ["V"])
    lif_inst.run(
        duration = 200.,
        inputs = [("input", 26.),],
        report = True
    )
    bp.visualize.line_plot(lif_inst.mon.ts, lif_inst.mon.V, show = True)
