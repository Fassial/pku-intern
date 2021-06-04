"""
Created on 12:58, June. 4th, 2021
Author: fassial
Filename: ExpSyn.py
"""
import brainpy as bp

__all__ = [
    "ExpSyn",
]

class ExpSyn(bp.TwoEndConn):
    target_backend = ['numpy', 'numba', 'numba-parallel', 'numba-cuda']

    @staticmethod
    def derivative(s, t, tau):
        dsdt = -s / tau
        return dsdt

    def __init__(self, pre, post, conn, weight = .1, delay = 0., tau = 8., **kwargs):
        # init params
        self.tau = tau
        self.delay = delay
        self.weight = weight

        # init connections
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = self.conn.requires("pre_ids", "post_ids")
        self.size = len(self.pre_ids)

        # init vars
        self.s = bp.ops.zeros(self.size)
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
        for i in range(self.size):
            pre_id, post_id = self.pre_ids[i], self.post_ids[i]
            # get Isyn
            self.s[i] = self.integral(
                s = self.s[i],
                t = _t,
                tau = self.tau
            )
            self.s[i] += self.pre.spike[pre_id]
            self.Isyn.push(i,
                self.weight * self.s[i]
            )
            # update post.input
            self.post.input[post_id] += self.Isyn.pull(i)

