"""
Created on 12:55, June. 4th, 2021
Author: fassial
Filename: TwoExpSyn.py
"""
import brainpy as bp

__all__ = [
    "TwoExpSyn",
]

class TwoExpSyn(bp.TwoEndConn):
    target_backend = ['numpy', 'numba', 'numba-parallel', 'numba-cuda']

    @staticmethod
    def derivative(s, x, t, tau1, tau2):
        dxdt = (-(tau1 + tau2) * x - s) / (tau1 * tau2)
        dsdt = x
        return dsdt, dxdt

    def __init__(self, pre, post, conn,
        weight = 2., delay = 0., tau1 = .3, tau2 = 3., **kwargs
    ):
        # init params
        self.weight = weight
        self.delay = delay
        self.tau1 = tau1
        self.tau2 = tau2

        # init connections
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = self.conn.requires("pre_ids", "post_ids")
        self.size = len(self.pre_ids)

        # init vars
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
        # set post.input
        for i in range(self.size):
            pre_id, post_id = self.pre_ids[i], self.post_ids[i]
            # get Isyn
            self.s[i], self.x[i] = self.integral(
                s = self.s[i],
                x = self.x[i],
                t = _t,
                tau1 = self.tau1,
                tau2 = self.tau2
            )
            self.x[i] += self.pre.spike[pre_id]
            self.Isyn.push(i,
                self.weight * self.s[i]
            )
            # update post.input
            self.post.input[post_id] += self.Isyn.pull(i)

