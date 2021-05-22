"""
Created on 23:57, Apr. 5th, 2021
Author: fassial
Filename: GapJunction.py
"""
import brainpy as bp

__all__ = [
    "GapJunction",
    "GapJunction_LIF",
]

class GapJunction(bp.TwoEndConn):
    target_backend = "general"

    def __init__(self, pre, post, conn, weight = 1., delay = 0., **kwargs):
        # init params
        self.delay = delay
        self.weight = weight

        # init connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires("conn_mat")
        self.size = bp.ops.shape(self.conn_mat)

        # init vars
        self.w = bp.ops.ones(self.size) * self.weight

        # init super
        super(GapJunction, self).__init__(pre = pre, post = post, **kwargs)

    def update(self, _t):
        # get V_post & V_pre
        V_post = bp.ops.vstack((self.post.V,) * self.size[0])
        V_pre = bp.ops.vstack((self.pre.V,) * self.size[1]).T

        # set Isyn & post.input
        Isyn = self.w * (V_pre - V_post) * self.conn_mat
        self.post.input += bp.ops.sum(Isyn, axis = 0)

class GapJunction_LIF(bp.TwoEndConn):
    target_backend = "general"

    def __init__(self, pre, post, conn, weight = 1., delay = 0.,
        k_spikelet = .1, post_refractory = True, **kwargs
    ):
        # init params
        self.delay = delay
        self.weight = weight
        self.k_spikelet = k_spikelet
        self.post_refractory = post_refractory

        # init connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = self.conn.requires("conn_mat")
        self.size = bp.ops.shape(self.conn_mat)

        # init vars
        self.w = bp.ops.ones(self.size) * self.weight
        self.spikelet = self.register_constant_delay("spikelet",
            size = self.size,
            delay_time = delay
        )

        # init super
        super(GapJunction_LIF, self).__init__(pre = pre, post = post, **kwargs)

    def update(self, _t):
        # get V_post & V_pre
        V_post = bp.ops.vstack((self.post.V,) * self.size[0])
        V_pre = bp.ops.vstack((self.pre.V,) * self.size[1]).T

        # set Isyn & post.input
        Isyn = self.w * (V_pre - V_post) * self.conn_mat
        self.post.input += bp.ops.sum(Isyn, axis = 0)

        # push spikelet
        self.spikelet.push(self.w * self.k_spikelet *\
            bp.ops.unsqueeze(self.pre.spike, 1) *\
            self.conn_mat
        )

        # set post.V & check post_refractory
        if self.post_refractory:
            self.post.V += bp.ops.sum(self.spikelet.pull() * (1. - self.post.refractory), axis = 0)
        else:
            self.post.V += bp.ops.sum(self.spikelet.pull(), axis = 0)

