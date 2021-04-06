"""
Created on 00:37, Apr. 6th, 2021
Author: fassial
Filename: VoltageJump.py
"""
import brainpy as bp

__all__ = [
    "VoltageJump",
]

class VoltageJump(bp.TwoEndConn):
    target_backend = "general"

    def __init__(self, pre, post, conn, weight = 1., delay = 0., post_refractory = False, **kwargs):
        # init params
        self.delay = delay
        self.post_refractory = post_refractory

        # init connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = self.conn.requires("conn_mat")
        self.size = bp.backend.shape(self.conn_mat)

        # init vars
        self.s = bp.backend.zeros(self.size)
        self.w = bp.backend.ones(self.size) * weight
        self.Isyn = self.register_constant_delay("Isyn",
            size = self.size,
            delay_time = delay
        )

        # init super
        super(VoltageJump, self).__init__(pre = pre, post = post, **kwargs)

    def update(self, _t):
        # set s
        self.s = bp.backend.unsqueeze(self.pre.spike, 1) * self.conn_mat

        # check post_refractory
        if self.post_refractory:
            refractor_map = (1. - bp.backend.unsqueeze(self.post.refractory, 0)) * self.conn_mat
            self.Isyn.push(self.s * refractor_map * self.w)
        else:
            self.Isyn.push(self.s * self.w)

        # set post.V
        self.post.V += bp.backend.sum(self.Isyn.pull(), axis = 0)
