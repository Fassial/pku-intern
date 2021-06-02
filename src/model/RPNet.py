"""
Created on 18:52, Apr. 22nd, 2021
Author: fassial
Filename: RPNet.py
"""
import os
import numpy as np
import brainpy as bp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# local dep
from . import neurons
from . import synapses
from . import connector

__all__ = [
    "RPNet",
]

class RPNet(bp.Network):
    """
    Network composed of ipRGC and PAC.
    """

    @staticmethod
    def gen_neighbors(num_i, num_e, neighbors = 1):
        num_ratio = int(num_e / num_i)
        assert (num_e / num_i) == int(num_e / num_i)

        neighbors_ii, neighbors_ie, neighbors_ei, neighbors_ee = [[],[]], [[],[]], [[],[]], [[],[]]
        # set neighbors_ii & neighbors_ie
        for pre_idx in range(num_i):
            for neighbor in range(neighbors):
                # get pre_id on circle
                pre_id = pre_idx * (num_ratio + 1)
                ## set right neighbors
                # get post_id on circle
                post_id = pre_id + (neighbor + 1)
                if post_id >= (num_i + num_e): post_id -= (num_i + num_e)
                # get post_idx
                if (post_id / (num_ratio + 1)) == int(post_id / (num_ratio + 1)):
                    post_idx = int(post_id / (num_ratio + 1))
                    neighbors_ii[0].append(pre_idx); neighbors_ii[1].append(post_idx)
                else:
                    post_idx = num_ratio * (post_id // (num_ratio + 1)) + (post_id % (num_ratio + 1)) - 1
                    neighbors_ie[0].append(pre_idx); neighbors_ie[1].append(post_idx)
                ## set left neighbors
                # get post_id on circle
                post_id = pre_id - (neighbor + 1)
                if post_id < 0: post_id += (num_i + num_e)
                # get post_idx
                if (post_id / (num_ratio + 1)) == int(post_id / (num_ratio + 1)):
                    post_idx = int(post_id / (num_ratio + 1))
                    neighbors_ii[0].append(pre_idx); neighbors_ii[1].append(post_idx)
                else:
                    post_idx = num_ratio * (post_id // (num_ratio + 1)) + (post_id % (num_ratio + 1)) - 1
                    neighbors_ie[0].append(pre_idx); neighbors_ie[1].append(post_idx)
        # set neighbors_ei & neighbors_ee
        for pre_idx in range(num_e):
            for neighbor in range(neighbors):
                # get pre_id on circle
                pre_id = ((num_ratio + 1) * (pre_idx // num_ratio)) + (pre_idx % num_ratio) + 1
                ## set right neighbors
                # get post_id on circle
                post_id = pre_id + (neighbor + 1)
                if post_id >= (num_i + num_e): post_id -= (num_i + num_e)
                # get post_idx
                if (post_id / (num_ratio + 1)) == int(post_id / (num_ratio + 1)):
                    post_idx = int(post_id / (num_ratio + 1))
                    neighbors_ei[0].append(pre_idx); neighbors_ei[1].append(post_idx)
                else:
                    post_idx = num_ratio * (post_id // (num_ratio + 1)) + (post_id % (num_ratio + 1)) - 1
                    neighbors_ee[0].append(pre_idx); neighbors_ee[1].append(post_idx)
                ## set left neighbors
                # get post_id on circle
                post_id = pre_id - (neighbor + 1)
                if post_id < 0: post_id += (num_i + num_e)
                # get post_idx
                if (post_id / (num_ratio + 1)) == int(post_id / (num_ratio + 1)):
                    post_idx = int(post_id / (num_ratio + 1))
                    neighbors_ei[0].append(pre_idx); neighbors_ei[1].append(post_idx)
                else:
                    post_idx = num_ratio * (post_id // (num_ratio + 1)) + (post_id % (num_ratio + 1)) - 1
                    neighbors_ee[0].append(pre_idx); neighbors_ee[1].append(post_idx)

        return neighbors_ii, neighbors_ie, neighbors_ei, neighbors_ee

    def __init__(self, net_params = {
        "ipRGC": {
            ## neurons params
            # shape params
            "size": (40,),
            # dynamic params
            "V_init": "reset",
            "tau": .5,
            "t_refractory": 5.,
        },
        "PAC": {
            ## neurons params
            # shape params
            "size": (20,),
            # dynamic params
            "V_init": "reset",
            "tau": .5,
            "t_refractory": 5.,
        },
        "GJ_RP": {
            # gap junction
            "neighbors": 1,
            "weight": .5,
            "k_spikelet": .1,
            "conn": connector.IndexConnector(),
        },
        "ES_RP": {
            # exp synapses
            "neighbors": 2,
            "weight": .5,
            "delay": .1,
            "tau": .5,
            "conn": connector.IndexConnector(),
        },
    }, run_params = {
        "inputs": {
            "ipRGC": 0.,
            "PAC": 0.,
        },
        "dt": 0.01,
        "duration": 20,
    }):
        # init params
        self.net_params = net_params
        self.run_params = run_params

        # init backend
        bp.backend.set(dt = run_params["dt"])

        ## init comps of network
        # init iprgc
        self.iprgc = neurons.LIF(
            size = net_params["ipRGC"]["size"],
            V_rest = 0.,
            V_reset = 0.,
            V_th = 1.,
            V_init = net_params["ipRGC"]["V_init"],
            R = 1.,
            tau = net_params["ipRGC"]["tau"],
            t_refractory = net_params["ipRGC"]["t_refractory"],
            # monitor
            monitors = ["V", "spike"]
        )
        # init pac
        self.pac = neurons.LIF(
            size = net_params["PAC"]["size"],
            V_rest = 0.,
            V_reset = 0.,
            V_th = 1.,
            V_init = net_params["PAC"]["V_init"],
            R = 1.,
            tau = net_params["PAC"]["tau"],
            t_refractory = net_params["PAC"]["t_refractory"],
            # monitor
            monitors = ["V", "spike"]
        )
        ## init gj
        # get neighbors
        neighbors_pp, neighbors_pr, neighbors_rp, neighbors_rr = RPNet.gen_neighbors(
            num_i = bp.size2len(net_params["PAC"]["size"]),
            num_e = bp.size2len(net_params["ipRGC"]["size"]),
            neighbors = net_params["GJ_RP"]["neighbors"]
        )
        # init gj_pp
        self.gj_pp = synapses.GapJunction_LIF(
            pre = self.pac,
            post = self.pac,
            conn = net_params["GJ_RP"]["conn"](
                pre_size = net_params["PAC"]["size"],
                post_size = net_params["PAC"]["size"],
                pre_ids = neighbors_pp[0],
                post_ids = neighbors_pp[1]
            ),
            weight = net_params["GJ_RP"]["weight"],
            k_spikelet = net_params["GJ_RP"]["k_spikelet"]
        )
        # init gj_pr
        self.gj_pr = synapses.GapJunction_LIF(
            pre = self.pac,
            post = self.iprgc,
            conn = net_params["GJ_RP"]["conn"](
                pre_size = net_params["PAC"]["size"],
                post_size = net_params["ipRGC"]["size"],
                pre_ids = neighbors_pr[0],
                post_ids = neighbors_pr[1]
            ),
            weight = net_params["GJ_RP"]["weight"],
            k_spikelet = net_params["GJ_RP"]["k_spikelet"]
        )
        # init gj_rp
        self.gj_rp = synapses.GapJunction_LIF(
            pre = self.iprgc,
            post = self.pac,
            conn = net_params["GJ_RP"]["conn"](
                pre_size = net_params["ipRGC"]["size"],
                post_size = net_params["PAC"]["size"],
                pre_ids = neighbors_rp[0],
                post_ids = neighbors_rp[1]
            ),
            weight = net_params["GJ_RP"]["weight"],
            k_spikelet = net_params["GJ_RP"]["k_spikelet"]
        )
        # init gj_rr
        self.gj_rr = synapses.GapJunction_LIF(
            pre = self.iprgc,
            post = self.iprgc,
            conn = net_params["GJ_RP"]["conn"](
                pre_size = net_params["ipRGC"]["size"],
                post_size = net_params["ipRGC"]["size"],
                pre_ids = neighbors_rr[0],
                post_ids = neighbors_rr[1]
            ),
            weight = net_params["GJ_RP"]["weight"],
            k_spikelet = net_params["GJ_RP"]["k_spikelet"]
        )
        ## init es
        # get neighbors
        _, neighbors_pr, _, _ = RPNet.gen_neighbors(
            num_i = bp.size2len(net_params["PAC"]["size"]),
            num_e = bp.size2len(net_params["ipRGC"]["size"]),
            neighbors = net_params["ES_RP"]["neighbors"]
        )
        # init es_pr
        self.es_pr = synapses.ExpSyn(
            pre = self.pac,
            post = self.iprgc,
            conn = net_params["ES_RP"]["conn"](
                pre_size = net_params["PAC"]["size"],
                post_size = net_params["ipRGC"]["size"],
                pre_ids = neighbors_pr[0],
                post_ids = neighbors_pr[1]
            ),
            weight = -net_params["ES_RP"]["weight"],
            delay = net_params["ES_RP"]["delay"],
            tau = net_params["ES_RP"]["tau"]
        )

        # integrate network
        self.network = super(RPNet, self).__init__(
            ## neurons
            self.iprgc, self.pac,
            ## synapses
            # gap junction
            self.gj_pp, self.gj_pr, self.gj_rp, self.gj_rr,
            # exp syn
            self.es_pr
        )

    def run(self, run_params = None, report = True, report_percent = 0.1):
        # update run_params
        if run_params != None: self.run_params = run_params

        # excute super.run
        super(RPNet, self).run(
            duration = self.run_params["duration"],
            inputs = (
                (self.iprgc, "input", self.run_params["inputs"]["ipRGC"]),
                (self.pac, "input", self.run_params["inputs"]["PAC"]),
            ),
            report = report,
            report_percent = report_percent
        )

    def get_monitors(self):
        monitors = {
            "ipRGC": self.iprgc.mon,
            "PAC": self.pac.mon,
        }
        return monitors

    def save(self, spike_fname = None):
        # get spike_flist
        spike_flist = os.path.splitext(spike_fname)

        # save iprgc.spike
        np.savetxt(
            fname = spike_flist[0] + "-iprgc" + spike_flist[1],
            X = self.iprgc.mon.spike,
            fmt = "%1d",
            delimiter = ","
        )
        # save pac.spike
        np.savetxt(
            fname = spike_flist[0] + "-pac" + spike_flist[1],
            X = self.pac.mon.spike,
            fmt = "%1d",
            delimiter = ","
        )

    def show(self, img_size = None, img_fname = None):
        # init fig & gs
        fig = plt.figure(
            figsize = img_size,
            constrained_layout = True
        )
        gs = GridSpec(3, 1, figure = fig)

        ## axes 11: show ipRGC spikes
        ax11 = fig.add_subplot(gs[0:2,:])
        # get neu_idx & t_spike
        neu_idx, t_spike = bp.measure.raster_plot(
            sp_matrix = self.iprgc.mon.spike,
            times = self.iprgc.mon.ts
        )
        # plot ax11
        ax11.plot(
            # plot 1
            t_spike, neu_idx, ".",
            # plot settings
            markersize = 2, color = "red"
        )
        ax11.set_xlim(left = -0.1, right = self.run_params["duration"] + 0.1)
        ax11.set_ylim(bottom = -0.1, top = bp.size2len(self.net_params["ipRGC"]["size"]) + 0.1)
        ax11.set_ylabel(ylabel = "ipRGCs")
        # ax11.set_xlabel(xlabel = "Time [{} ms]".format(self.run_params["duration"]))
        # ax11.set_title(label = "raster_plot(spike) of RGCs")

        ## axes 12: show PAC spikes
        ax12 = fig.add_subplot(gs[2:,:])
        # get neu_idx & t_spike
        neu_idx, t_spike = bp.measure.raster_plot(
            sp_matrix = self.pac.mon.spike,
            times = self.pac.mon.ts
        )
        # plot ax12
        ax12.plot(
            # plot 1
            t_spike, neu_idx, ".",
            # plot settings
            markersize = 2, color = "blue"
        )
        ax12.set_xlim(left = -0.1, right = self.run_params["duration"] + 0.1)
        ax12.set_ylim(bottom = -0.1, top = bp.size2len(self.net_params["PAC"]["size"]) + 0.1)
        ax12.set_ylabel(ylabel = "PACs")
        ax12.set_xlabel(xlabel = "Time [{} ms]".format(self.run_params["duration"]))
        # ax12.set_title(label = "line_plot(V) of RGCs")

        # integrate fig
        fig.align_ylabels([ax11, ax12])

        # img show or save
        if img_fname:
            plt.savefig(fname = img_fname)
            plt.close(fig)
        else:
            plt.show()

