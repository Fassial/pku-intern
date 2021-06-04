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
        self.pre_ids, self.post_ids = self.conn.requires("pre_ids", "post_ids")
        self.size = len(self.pre_ids)

        # init super
        super(GapJunction, self).__init__(pre = pre, post = post, **kwargs)

    def update(self, _t):
        # set post.V
        for i in range(self.size):
            pre_id, post_id = self.pre_ids[i], self.post_ids[i]
            self.post.input[post_id] += self.weight *\
                (self.pre.V[pre_id] - self.post.V[post_id])

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
        self.pre_ids, self.post_ids = self.conn.requires("pre_ids", "post_ids")
        self.size = len(self.pre_ids)

        # init super
        super(GapJunction_LIF, self).__init__(pre = pre, post = post, **kwargs)

    def update(self, _t):
        # set post.V
        for i in range(self.size):
            pre_id, post_id = self.pre_ids[i], self.post_ids[i]
            self.post.input[post_id] += self.weight *\
                (self.pre.V[pre_id] - self.post.V[post_id])
            if self.pre.spike[pre_id] and not self.post.refractory[post_id]:
                self.post.V[post_id] += self.weight * self.k_spikelet

