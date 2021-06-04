"""
Created on 12:39, June. 4th, 2021
Author: fassial
Filename: VoltageJump.py
"""
import brainpy as bp

__all__ = [
    "VoltageJump",
]

class VoltageJump(bp.TwoEndConn):
    target_backend = "general"

    def __init__(self, pre, post, conn,
        weight = 1., delay = 0., **kwargs
    ):
        # init params
        self.weight = weight
        self.delay = delay

        # init connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = self.conn.requires("conn_mat")
        self.size = bp.ops.shape(self.conn_mat)

        # init vars
        self.w = bp.ops.ones(self.size) * self.weight
        self.Isyn = self.register_constant_delay("Isyn",
            size = self.size,
            delay_time = self.delay
        )

        # init super
        super(VoltageJump, self).__init__(pre = pre, post = post, **kwargs)

    def update(self, _t):
        # set Isyn & post.V
        Isyn = self.w * bp.ops.unsqueeze(self.pre.spike, 1) * self.conn_mat
        self.post.V += bp.ops.sum(Isyn * (1. - self.post.refractory), axis = 0)

