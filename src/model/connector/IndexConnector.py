"""
Created on 15:36, June. 4th, 2021
Author: fassial
Filename: IndexConnector.py
"""
import numpy as np
import brainpy as bp

__all__ = [
    "IndexConnector",
]

class IndexConnector(bp.connect.Connector):

    def __init__(self):
        # init params
        self.pre_ids = bp.ops.as_tensor([])
        self.post_ids = bp.ops.as_tensor([])

        # init super
        super(IndexConnector, self).__init__()

    def __call__(self, pre_size, post_size, pre_ids = None, post_ids = None):
        # init params
        self.num_pre = bp.size2len(pre_size)
        self.num_post = bp.size2len(post_size)
        if pre_ids != None: self.pre_ids = bp.ops.as_tensor(pre_ids)
        if post_ids != None: self.post_ids = bp.ops.as_tensor(post_ids)

        # init vars
        self.conn_mat = bp.connect.ij2mat(
            i = self.pre_ids,
            j = self.post_ids,
            num_pre = self.num_pre,
            num_post = self.num_post
        )

        return self

if __name__ == "__main__":
    # inst IndexConnector
    ic = IndexConnector(
        pre_ids = [0,0,0,1,1,2,7,8,8,9,9,9],
        post_ids = [0,1,2,3,4,5,6,7,8,9,0,1]
    )
    # show conn_mat
    ic = ic(
        pre_size = (10,),
        post_size = (10,)
    ); print(ic.requires("conn_mat"))

