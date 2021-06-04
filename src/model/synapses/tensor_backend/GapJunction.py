"""
Created on 12:41, June. 4th, 2021
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

    def __init__(self, pre, post, conn, weight = 1., **kwargs):
        # init params
        self.weight = weight

        # init connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = self.conn.requires('conn_mat')
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
        Isyn = self.w * (V_pre - V_post) * self.conn_mat / self.neighbors
        self.post.input += bp.ops.sum(Isyn, axis = 0)

class GapJunction_LIF(bp.TwoEndConn):
    target_backend = "general"

    def __init__(self, pre, post, conn,
        weight = 1., k_spikelet = .1, **kwargs
    ):
        # init params
        self.weight = weight
        self.k_spikelet = k_spikelet

        # init connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = self.conn.requires("conn_mat")
        self.size = bp.ops.shape(self.conn_mat)

        # init vars
        self.w = bp.ops.ones(self.size) * self.weight

        # init super
        super(GapJunction_LIF, self).__init__(pre = pre, post = post, **kwargs)

    def update(self, _t):
        # get V_post & V_pre
        V_post = bp.ops.vstack((self.post.V,) * self.size[0])
        V_pre = bp.ops.vstack((self.pre.V,) * self.size[1]).T

        # set Isyn & post.input
        Isyn = self.w * (V_pre - V_post) * self.conn_mat
        self.post.input += bp.ops.sum(Isyn, axis = 0)

        # set spikelet & post.V
        spikelet = self.w * self.k_spikelet *\
            bp.ops.unsqueeze(self.pre.spike, 1) * self.conn_mat
        self.post.V += bp.ops.sum(spikelet * (1. - self.post.refractory), axis = 0)

