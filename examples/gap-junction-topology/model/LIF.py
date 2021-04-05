"""
Created on 21:01, Apr. 5th, 2021
Author: fassial
Filename: LIF.py
"""
import brainpy as bp

class LIF(bp.NeuGroup):
    target_backend = "general"

    def __init__(self, size,
        V_rest = 0., V_reset = -5., V_th = 20.,
        R = 1., tau = 10., tau_refractory = 1., **kwargs
    ):
        # init params
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.R = R
        self.tau = tau
        self.tau_refractory = tau_refractory

        # init vars
        self.t_last_spike = bp.backend.ones(size) * -1e7
        self.refractory = bp.backend.zeros(size)
        self.input = bp.backend.zeros(size)
        self.spike = bp.backend.zeros(size)
        self.V = bp.backend.ones(size) * V_reset

        # def kinetic func
        def diff(V, t, Iext, V_rest, R, tau):
            dVdt = (-(V - V_rest) + R * Iext) / tau
            return dVdt
        self.integral = bp.odeint(
            f = diff,
            method = "rk4"
        )

        # init super
        super(LIF, self).__init__(size = size, **kwargs)

    def update(self, _t):
        # update vars
        not_refractory = (_t - self.t_last_spike > self.tau_refractory)
        self.V[not_refractory] = self.integral(
            V = self.V[not_refractory],
            t = _t,
            Iext = self.input[not_refractory],
            V_rest = self.V_rest,
            R = self.R,
            tau = self.tau
        )
        spike = (self.V > self.V_th)
        self.V[spike] = self.V_reset
        self.t_last_spike[spike] = _t
        self.spike = spike
        self.refractory = ~not_refractory
        self.input[:] = 0.

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
