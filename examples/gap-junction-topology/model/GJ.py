"""
Created on 23:57, Apr. 5th, 2021
Author: fassial
Filename: GJ.py
"""
import brainpy as bp

class GJ(bp.TwoEndConn):
    target_backend = "general"

    def __init__(self, pre, post, conn, delay = 0., **kwargs):
        # init params
        self.delay = delay

        # init connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires("conn_mat")
        self.size = bp.backend.shape(self.conn_mat)

        # init vars
        self.w = bp.backend.ones(self.size)

        # init super
        super(GJ, self).__init__(pre = pre, post = post, **kwargs)

    def update(self, _t):
        # get V_post & V_pre
        V_post = bp.backend.vstack((self.post.V,) * self.size[0])
        V_pre = bp.backend.vstack((self.pre.V,) * self.size[1]).T

        # set Isyn & post.input
        Isyn = self.w * (V_pre - V_post) * self.conn_mat
        self.post.input += bp.backend.sum(Isyn, axis = 0)

class GJ_LIF(bp.TwoEndConn):
    target_backend = "general"

    def __init__(self, pre, post, conn, delay = 0.,
        k_spikelet = 0.1, post_refractory = False, **kwargs
    ):
        # init params
        self.delay = delay
        self.k_spikelet = k_spikelet
        self.post_refractory = post_refractory

        # init connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = self.conn.requires("conn_mat")
        self.size = bp.backend.shape(self.conn_mat)

        # init vars
        self.w = bp.backend.ones(self.size)
        self.spikelet = self.register_constant_delay("spikelet",
            size = self.size,
            delay_time = delay
        )

        # init super
        super(GJ_LIF, self).__init__(pre = pre, post = post, **kwargs)

    def update(self, _t):
        # get V_post & V_pre
        V_post = bp.backend.vstack((self.post.V,) * self.size[0])
        V_pre = bp.backend.vstack((self.pre.V,) * self.size[1]).T

        # set Isyn & post.input
        Isyn = self.w * (V_pre - V_post) * self.conn_mat
        self.post.input += bp.backend.sum(Isyn, axis = 0)

        # check post_refractory
        if self.post_refractory:
            self.spikelet.push(self.w * self.k_spikelet *\
                bp.backend.unsqueeze(self.pre.spike, 1) *\
                self.conn_mat * (1. - self.post.refractory)
            )
        else:
            self.spikelet.push(self.w * self.k_spikelet *\
                bp.backend.unsqueeze(self.pre.spike, 1) *\
                self.conn_mat
            )

        # set post.V
        self.post.V += bp.backend.sum(self.spikelet.pull(), axis = 0)
