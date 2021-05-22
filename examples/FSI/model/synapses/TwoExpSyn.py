"""
Created on 20:09, May. 22nd, 2021
Author: fassial
Filename: TwoExpSyn.py
"""
import brainpy as bp

__all__ = [
    "TwoExpSyn",
]

class TwoExpSyn(bp.TwoEndConn):
    target_backend = "general"

    @staticmethod
    def derivative(s, x, t, tau1, tau2):
        dxdt = (-(tau1 + tau2) * x - s) / (tau1 * tau2)
        dsdt = x
        return dsdt, dxdt

    def __init__(self, pre, post, conn,
        weight = .3, delay = 0., tau1 = .3, tau2 = 3., **kwargs
    ):
        # init params
        self.weight = weight
        self.delay = delay
        self.tau1 = tau1
        self.tau2 = tau2

        # init connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = self.conn.requires("conn_mat")
        self.size = bp.ops.shape(self.conn_mat)

        # init vars
        self.w = bp.ops.ones(self.size) * self.weight
        self.s = bp.ops.zeros(self.size)
        self.x = bp.ops.zeros(self.size)
        self.Isyn = self.register_constant_delay("Isyn",
            size = self.size,
            delay_time = self.delay
        )

        # init integral
        self.integral = bp.odeint(
            f = TwoExpSyn.derivative,
            method = "euler"
        )

        # init super
        super(TwoExpSyn, self).__init__(pre = pre, post = post, **kwargs)

    def update(self, _t):
        # get Isyn
        self.s, self.x = self.integral(
            s = self.s,
            x = self.x,
            t = _t,
            tau1 = self.tau1,
            tau2 = self.tau2
        )
        self.x += bp.ops.unsqueeze(self.pre.spike, 1) * self.conn_mat
        self.Isyn.push(self.w * self.s)

        # update post.input
        self.post.input += bp.ops.sum(self.Isyn.pull(), axis = 0)

