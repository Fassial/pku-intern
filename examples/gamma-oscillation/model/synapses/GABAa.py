"""
Created on 23:46, June. 9th, 2021
Author: fassial
Filename: GABAa.py
"""
import brainpy as bp

__all__ = [
    "GABAa",
]

class GABAa(bp.TwoEndConn):
    target_backend = ["numpy", "numba"]

    @staticmethod
    def derivative(s, t, TT, alpha, beta):
        dsdt = alpha * TT * (1. - s) - beta * s
        return dsdt

    def __init__(self, pre, post, conn,
        delay = 0., g_max = .1, E = -75.,
        alpha = 12., beta = .1, T = 1., T_duration = 1., **kwargs
    ):
        # init params
        self.delay = delay
        self.g_max = g_max
        self.E = E
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.T_duration = T_duration

        # init connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = self.conn.requires("conn_mat")
        self.size = bp.ops.shape(self.conn_mat)

        # init vars
        self.s = bp.ops.zeros(self.size)
        self.t_last_pre_spike = bp.ops.ones(self.size) * -1e7
        self.g = self.register_constant_delay("g",
            size = self.size,
            delay_time = self.delay
        )

        # init integral
        self.integral = bp.odeint(
            f = GABAa.derivative,
            method = "rk4"
        )

        # init super
        super(GABAa, self).__init__(pre = pre, post = post, **kwargs)

    def update(self, _t):
        # update vars
        for i in range(self.pre.size[0]):
            if self.pre.spike[i] > 0.:
                self.t_last_pre_spike[i] = _t
        TT = ((_t - self.t_last_pre_spike) < self.T_duration) * self.T
        self.s = self.integral(
            s = self.s,
            t = _t,
            TT = TT,
            alpha = self.alpha,
            beta = self.beta
        )
        self.g.push(self.g_max * self.s)
        g = self.g.pull()
        self.post.input -= bp.ops.sum(g, axis = 0) * (self.post.V - self.E)

