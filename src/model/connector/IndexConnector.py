"""
Created on 00:35, Apr. 23rd, 2021
Author: fassial
Filename: IndexConnector.py
"""
import brainpy as bp

__all__ = [
    "IndexConnector",
]

class IndexConnector(bp.connect.Connector):

    def __init__(self):
        # init super
        super(IndexConnector, self).__init__()

    def __call__(self, pre_size, post_size, pre_ids = [], post_ids = []):
        # init params
        self.num_pre = bp.size2len(pre_size)
        self.num_post = bp.size2len(post_size)

        # init vars
        self.pre_ids = pre_ids
        self.post_ids = post_ids
        self.conn_mat = bp.connect.ij2mat(
            i = self.pre_ids,
            j = self.post_ids,
            num_pre = self.num_pre,
            num_post = self.num_post
        )

        return self

if __name__ == "__main__":
    # inst IndexConnector
    ic = IndexConnector()
    # show conn_mat
    ic = ic(
        pre_size = (10,),
        post_size = (10,),
        pre_ids = [0,0,0,1,1,2,7,8,8,9,9,9],
        post_ids = [0,1,2,3,4,5,6,7,8,9,0,1]
    ); print(ic.requires("conn_mat"))

