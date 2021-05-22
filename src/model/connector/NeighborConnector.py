"""
Created on 21:40, Apr. 22nd, 2021
Author: fassial
Filename: NeighborConnector.py
"""
import brainpy as bp

__all__ = [
    "NeighborConnector",
]

class NeighborConnector(bp.connect.Connector):

    @staticmethod
    def gen_neighbors(num_neuron, neighbors = 1):
        # init pre_idxs & post_idxs
        pre_idxs, post_idxs = [], []

        # gen neighbors
        for pre_idx in range(num_neuron):
            for neighbor in range(neighbors):
                # get left neighbor
                post_idx = pre_idx - (neighbor + 1)
                if post_idx < 0: post_idx += num_neuron
                pre_idxs.append(pre_idx); post_idxs.append(post_idx)
                # get right neighbor
                post_idx = pre_idx + (neighbor + 1)
                if post_idx >= num_neuron: post_idx -= num_neuron
                pre_idxs.append(pre_idx); post_idxs.append(post_idx)

        return pre_idxs, post_idxs

    def __init__(self, neighbors = 1):
        # init params
        self.neighbors = neighbors

        # init super
        super(NeighborConnector, self).__init__()

    def __call__(self, pre_size, post_size):
        assert pre_size == post_size

        # init params
        self.num_neuron = bp.size2len(pre_size)
        self.num_pre = self.num_neuron
        self.num_post = self.num_neuron

        # init vars
        pre_idxs, post_idxs = NeighborConnector.gen_neighbors(
            num_neuron = self.num_neuron,
            neighbors = self.neighbors
        )
        self.pre_ids = bp.ops.as_tensor(pre_idxs)
        self.post_ids = bp.ops.as_tensor(post_idxs)
        self.conn_mat = bp.connect.ij2mat(
            i = self.pre_ids,
            j = self.post_ids,
            num_pre = self.num_pre,
            num_post = self.num_post
        )

        return self

if __name__ == "__main__":
    # inst NeighborConnector
    nc = NeighborConnector(
        neighbors = 2
    )
    # show conn_mat(num_neuron = 10)
    nc = nc(
        pre_size = (10,),
        post_size = (10,)
    ); print(nc.requires("conn_mat"))
    # show conn_mat(num_neuron = 15)
    nc = nc(
        pre_size = (15,),
        post_size = (15,)
    ); print(nc.requires("conn_mat"))

