"""
Created on 16:11, Apr. 22nd, 2021
Author: fassial
Filename: ExpSyn.py
"""
import brainpy as bp

__all__ = [
    "ExpSyn",
]

class ExpSyn(bp.TwoEndConn):
    target_backend = "general"

    @staticmethod
    def derivative(s, t, tau):
        dsdt = -s / tau
        return dsdt

    def __init__(self, pre, post, conn, weight = .1, delay = 0., neighbors = 1, tau = 8., **kwargs):
        # init params
        self.tau = tau
        self.delay = delay
        self.weight = weight
        self.neighbors = neighbors

        # init connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires("conn_mat")
        self.size = bp.ops.shape(self.conn_mat)

        # init vars
        self.s = bp.ops.zeros(self.size)
        self.w = bp.ops.ones(self.size) * weight
        self.Isyn = self.register_constant_delay("Isyn",
            size = self.size,
            delay_time = self.delay
        )

        # init integral
        self.integral = bp.odeint(
            f = ExpSyn.derivative,
            method = "exponential_euler"
        )

        # init super
        super(ExpSyn, self).__init__(pre = pre, post = post, **kwargs)

    def update(self, _t):
        self.s = self.integral(self.s, _t, self.tau)
        self.s += bp.ops.unsqueeze(self.pre.spike, 1) * self.conn_mat
        self.Isyn.push(self.w * self.s / self.neighbors)
        self.post.input += bp.ops.sum(self.Isyn.pull(), axis = 0)

