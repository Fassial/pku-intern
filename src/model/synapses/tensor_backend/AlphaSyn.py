"""
Created on 16:58, Apr. 22nd, 2021
Author: fassial
Filename: AlphaSyn.py
"""
import brainpy as bp

__all__ = [
    "AlphaSyn",
]

class AlphaSyn(bp.TwoEndConn):
    target_backend = "general"

    @staticmethod
    def derivative(s, x, t, tau):
        dxdt = (-2 * tau * x - s) / (tau ** 2)
        dsdt = x
        return dsdt, dxdt

    def __init__(self, pre, post, conn, weight = .2, delay = 0., tau = 2., **kwargs):
        # init params
        self.tau = tau
        self.delay = delay
        self.weight = weight

        # init connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = self.conn.requires("conn_mat")
        self.size = bp.ops.shape(self.conn_mat)

        # init vars
        self.s = bp.ops.zeros(self.size)
        self.x = bp.ops.zeros(self.size)
        self.w = bp.ops.ones(self.size) * self.weight
        self.Isyn = self.register_constant_delay("Isyn",
            size = self.size,
            delay_time = self.delay
        )

        # init integral
        self.integral = bp.odeint(
            f = AlphaSyn.derivative,
            method = "euler"
        )

        # init super
        super(AlphaSyn, self).__init__(pre = pre, post = post, **kwargs)

    def update(self, _t):
        self.s, self.x = self.integral(self.s, self.x, _t, self.tau)
        self.x += bp.ops.unsqueeze(self.pre.spike, 1) * self.conn_mat
        self.Isyn.push(self.w * self.s)
        self.post.input += bp.ops.sum(self.Isyn.pull(), axis = 0)

