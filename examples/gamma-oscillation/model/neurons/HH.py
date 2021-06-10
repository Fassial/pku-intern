"""
Created on 16:49, June. 9th, 2021
Author: fassial
Filename: HH.py
"""
import brainpy as bp

__all__ = [
    "HH",
]

class HH(bp.NeuGroup):
    target_backend = "general"

    @staticmethod
    def derivative(V, h, n, t, Iext, gNa, ENa, gK, EK, gL, EL, C, phi):
        # set dhdt
        alpha = .07 * bp.ops.exp(-(V + 58.) / 20.)
        beta = 1 / (bp.ops.exp(-.1 * (V + 28.)) + 1.)
        dhdt = alpha * (1. - h) - beta * h

        # set dndt
        alpha = -.01 * (V + 34.) / (bp.ops.exp(-.1 * (V + 34.)) - 1.)
        beta = .125 * bp.ops.exp(-(V + 44.) / 80.)
        dndt = alpha * (1. - n) - beta * n

        # set m
        alpha = -.1 * (V + 35.) / (bp.ops.exp(-.1 * (V + 35.)) -1.)
        beta = 4. * bp.ops.exp(-(V + 60.) / 18.)
        m = alpha / (alpha + beta)

        # set dVdt
        INa = gNa * m ** 3 * h * (V - ENa)
        IK = gK * n ** 4 * (V - EK)
        IL = gL * (V - EL)
        dVdt = (-INa - IK - IL + Iext) / C

        return dVdt, phi * dhdt, phi * dndt

    def __init__(self, size,
        ENa = 55., EK = -90., EL = -65.,
        gNa = 35., gK = 9., gL = .1,
        C = 1., phi = 5., V_th = 20., **kwargs
    ):
        # init params
        self.ENa = ENa
        self.EK = EK
        self.EL = EL
        self.gNa = gNa
        self.gK = gK
        self.gL = gL
        self.C = C
        self.phi = phi
        self.V_th = V_th

        # init vars
        self.V = bp.ops.ones(size) * -65.
        self.h = bp.ops.ones(size) * .6
        self.n = bp.ops.ones(size) *.32
        self.spike = bp.ops.zeros(size)
        self.input = bp.ops.zeros(size)

        # init integral
        self.integral = bp.odeint(
            f = HH.derivative,
            method = "rk4"
        )

        # init super
        super(HH, self).__init__(size = size, **kwargs)

    def update(self, _t):
        # update vars
        V, h, n = self.integral(
            V = self.V,
            h = self.h,
            n = self.n,
            t = _t,
            Iext = self.input,
            gNa = self.gNa,
            ENa = self.ENa,
            gK = self.gK,
            EK = self.EK,
            gL = self.gL,
            EL = self.EL,
            C = self.C,
            phi = self.phi
        )
        self.spike = (self.V < self.V_th) * (V >= self.V_th)
        self.V = V
        self.h = h
        self.n = n
        self.input[:] = 0.

if __name__ == "__main__":
    # set backend.dt
    bp.backend.set(dt = 0.01)
    bp.backend.set(backend = "numba")
    # inst hh & run hh
    hh_inst = HH(size = (100, 100), monitors = ["V"])
    hh_inst.run(
        duration = 200.,
        inputs = [("input", 1.),],
        report = True
    )
    bp.visualize.line_plot(hh_inst.mon.ts, hh_inst.mon.V, show = True)

