"""
Created on 14:12, May 22nd, 2021
Author: fassial
Filename: GABAergic.py
"""
import brainpy as bp

__all__ = [
    "GABAergic",
]

class GABAergic(bp.TwoEndConn):
    target_backend = "general"

    @staticmethod
    def integral(t_last_spike, t, tau_s, tau_f):
        Isyn = bp.ops.exp(-(t - t_last_spike) / tau_s) -\
            bp.ops.exp(-(t - t_last_spike) / tau_f)
        return Isyn

    def __init__(self, pre, post, conn,
        weight = .3, delay = 0., tau_s = 3, tau_f = .3, **kwargs
    ):
        # init params
        self.weight = weight
        self.delay = delay
        self.tau_s = tau_s
        self.tau_f = tau_f

        # init varsconnections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = self.conn.requires("conn_mat")
        self.size = bp.ops.shape(self.conn_mat)

        # init vars
        self.w = bp.ops.ones(self.size) * self.weight
        self.Isyn = self.register_constant_delay("Isyn",
            size = self.size,
            delay_time = self.delay
        )

        # init integral
        self.integral = GABAergic.integral

        # init super
        super(GABAergic, self).__init__(pre = pre, post = post, **kwargs)

    def update(self, _t):
        # set Isyn & post.input
        self.Isyn.push(self.w * self.integral(
            t_last_spike = self.pre.t_last_spike,
            t = _t,
            tau_s = self.tau_s,
            tau_f = self.tau_f
        ) * self.conn_mat)
        self.post.input += bp.ops.sum(self.Isyn.pull(), axis = 0)

