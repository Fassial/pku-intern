"""
Created on 13:00, June. 4th, 2021
Author: fassial
Filename: AlphaSyn.py
"""
import brainpy as bp

__all__ = [
    "AlphaSyn",
]

class AlphaSyn(bp.TwoEndConn):
    target_backend = ['numpy', 'numba', 'numba-parallel', 'numba-cuda']

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
            f = AlphaSyn.derivative,
            method = "euler"
        )

        # init super
        super(AlphaSyn, self).__init__(pre = pre, post = post, **kwargs)

    def update(self, _t):
        for i in range(self.size):
            pre_id, post_id = self.pre_ids[i], self.post_ids[i]
            # get Isyn
            self.s[i], self.x[i] = self.integral(
                s = self.s[i],
                x = self.x[i],
                t = _t,
                tau = self.tau
            )
            self.x[i] += self.pre.spike[pre_id]
            self.Isyn.push(i,
                self.weight * self.s[i]
            )
            # update post.input
            self.post.input[post_id] += self.Isyn.pull(i)

