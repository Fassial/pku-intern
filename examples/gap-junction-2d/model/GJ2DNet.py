"""
Created on 01:07, June. 4th, 2021
Author: fassial
Filename: GJ2DNet.py
"""
import numpy as np
import brainpy as bp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# local dep
from . import neurons
from . import synapses

__all__ = [
    "GJ2DNet",
]

class GJ2DNet(bp.Network):

    def __init__(self, net_params = {
        "neurons": {
            ## neurons params
            # shape params
            "size": (60, 60),
            # dynamic params
            "V_init": "gaussian",
            "tau": 5.,
            "t_refractory": 1.,
            "noise": .2,
        },
        "GJ": {
            # gap junction
            "weight": .5,
            "k_spikelet": .1,
            "conn": bp.connect.GridEight(include_self = False),
        },
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
            R = 1.,
            tau = net_params["neurons"]["tau"],
            t_refractory = net_params["neurons"]["t_refractory"],
            noise = net_params["neurons"]["noise"],
            # monitor
            monitors = ["V", "spike"]
        )
        # init gj
        self.gj = synapses.GapJunction_LIF(
            pre = self.neurons,
            post = self.neurons,
            conn = net_params["GJ"]["conn"],
            weight = net_params["GJ"]["weight"],
            k_spikelet = net_params["GJ"]["k_spikelet"]
        )

        # integrate network
        self.network = super(GJ2DNet, self).__init__(
            self.neurons, self.gj
        )

    def run(self, report = True, report_percent = 0.1):
        # excute super.run
        super(GJ2DNet, self).run(
            duration = self.run_params["duration"],
            inputs = (
                (self.neurons, "input", self.run_params["inputs"]),
            ),
            report = report,
            report_percent = report_percent
        )

    def get_monitors(self):
        monitors = self.neurons.mon
        return monitors

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


