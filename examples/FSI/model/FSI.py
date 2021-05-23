"""
Created on 15:18, May. 22nd, 2021
Author: fassial
Filename: FSI.py
"""
import numpy as np
import brainpy as bp
import matplotlib.pyplot as plt
# local dep
from . import neurons
from . import synapses
from . import connector

__all__ = [
    "FSI",
]

class FSI(bp.Network):
    """
    Network FSI.
    """

    @staticmethod
    def _gen_links_mat(num_neu, r, p, biconn = False):
        assert num_neu > r * 4

        # init links_mat
        links_mat = bp.ops.zeros((num_neu, num_neu))

        # set links_mat
        for i in range(num_neu):
            # init neighbors_i & links_i
            neighbors_i = []; links_i = []; neighbors_i.append(i)
            for j in range(r):
                # set left & right
                left = (i - (j + 1)) if (i - (j + 1)) >= 0 else (i - (j + 1) +  num_neu)
                right = (i + (j + 1)) if (i + (j + 1)) < num_neu else (i + (j + 1) - num_neu)
                # update neighbors_i & links_i
                neighbors_i.append(left); neighbors_i.append(right)
                links_i.append(left); links_i.append(right)
            # random links_mat
            for j in range(len(links_i)):
                if np.random.random_sample() < p:
                    link_n = np.random.randint(num_neu)
                    while (link_n in neighbors_i) or (link_n in links_i):
                        link_n = np.random.randint(num_neu)
                    links_i[j] = link_n
            # update links_mat
            links_mat[i, links_i] = 1
            if biconn: links_mat[links_i, i] = 1

        return links_mat

    @staticmethod
    def gen_links(num_neu, r, p, biconn = False):
        # init links
        links = [[], []]

        # get links_mat
        links_mat = FSI._gen_links_mat(num_neu = num_neu, r = r, p = p, biconn = biconn)
        # set links
        for i in range(links_mat.shape[0]):
            for j in range(links_mat.shape[1]):
                if links_mat[i, j] == 1: links[0].append(i); links[1].append(j)

        return links

    def __init__(self, net_params = {
        "neurons" : {
            "size": (200,),
            "V_init": "reset",
        },
        "GJ": {
            "r": 1,
            "p": 1.,
            "weight": .3,
            "conn": connector.IndexConnector(),
        },
        "CHEMS": {
            "r": 1,
            "p": 1.,
            "weight": 2.,
            "conn": connector.IndexConnector(),
        }
    }, run_params = {
        "inputs": 0.,
        "dt": 0.01,
        "duration": 20,
    }):
        # init params
        self.net_params = net_params
        self.run_params = run_params

        # init backend
        bp.backend.set(dt = run_params["dt"])

        ## init comps of network
        # init neurons
        self.neurons = neurons.LIF(
            size = net_params["neurons"]["size"],
            V_rest = 0.,
            V_reset = 0.,
            V_th = 1.,
            V_init = net_params["neurons"]["V_init"],
            alpha = np.random.random_sample(net_params["neurons"]["size"]) * .3 + 1,
            R = 1.,
            tau = .5,
            t_refractory = 5.,
            # monitor
            monitors = ["V", "spike"]
        )
        # init neighbors
        neighbors_gj = FSI.gen_links(
            num_neu = bp.size2len(net_params["neurons"]["size"]),
            r = net_params["GJ"]["r"],
            p = net_params["GJ"]["p"],
            biconn = True
        )
        neighbors_es = FSI.gen_links(
            num_neu = bp.size2len(net_params["neurons"]["size"]),
            r = net_params["CHEMS"]["r"],
            p = net_params["CHEMS"]["p"],
            biconn = False
        )
        # init gj
        self.gj = synapses.GapJunction_LIF(
            pre = self.neurons,
            post = self.neurons,
            conn = net_params["GJ"]["conn"](
                pre_size = net_params["neurons"]["size"],
                post_size = net_params["neurons"]["size"],
                pre_ids = neighbors_gj[0],
                post_ids = neighbors_gj[1]
            ),
            weight = net_params["GJ"]["weight"],
            delay = 0.,
            k_spikelet = .1,
            post_refractory = True
        )
        # init chems
        self.chems = synapses.TwoExpSyn(
            pre = self.neurons,
            post = self.neurons,
            conn = net_params["CHEMS"]["conn"](
                pre_size = net_params["neurons"]["size"],
                post_size = net_params["neurons"]["size"],
                pre_ids = neighbors_es[0],
                post_ids = neighbors_es[1]
            ),
            weight = -net_params["CHEMS"]["weight"],
            delay = 0.,
            tau1 = .3,
            tau2 = 3.
        )

        # integrate network
        self.network = super(FSI, self).__init__(
            ## neurons
            self.neurons,
            ## synapses
            # gap junction
            self.gj,
            # chem syn
            self.chems
        )

    def run(self, report = True, report_percent = 0.1):
        super(FSI, self).run(
            duration = self.run_params["duration"],
            inputs = (
                (self.neurons, "input", self.run_params["inputs"]),
            ),
            report = report,
            report_percent = report_percent
        )

    def get_monitors(self):
        return self.neurons.mon

    def save(self, spike_fname = None):
        np.savetxt(
            fname = spike_fname,
            X = self.neurons.mon.spike,
            fmt = "%1d",
            delimiter = ","
        )

    def show(self, img_size = None, img_fname = None):
        # init fig & gs
        fig = plt.figure(
            figsize = img_size,
            constrained_layout = True
        )

        # get neu_idx & t_spike
        neu_idx, t_spike = bp.measure.raster_plot(
            sp_matrix = self.neurons.mon.spike,
            times = self.neurons.mon.ts
        )
        # plot fig
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(
            # plot 1
            t_spike, neu_idx, ".",
            # plot settings
            markersize = 2, color = "blue"
        )
        ax.set_xlim(left = -0.1, right = self.run_params["duration"] + 0.1)
        ax.set_ylim(bottom = -0.1, top = bp.size2len(self.net_params["neurons"]["size"]) + 0.1)
        ax.set_ylabel(ylabel = "neurons")
        ax.set_xlabel(xlabel = "Time [{} ms]".format(self.run_params["duration"]))
        ax.set_title(label = "raster_plot(spike) of neurons")

        # img show or save
        if img_fname:
            plt.savefig(fname = img_fname)
            plt.close(fig)
        else:
            plt.show()

