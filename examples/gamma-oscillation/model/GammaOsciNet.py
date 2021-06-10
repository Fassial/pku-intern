"""
Created on 00:01, June. 9th, 2021
Author: fassial
Filename: GammaOsciNet.py
"""
import numpy as np
import brainpy as bp
# local dep
from . import neurons
from . import synapses

__all__ = [
    "GammaOsciNet",
]

class GammaOsciNet(bp.Network):

    def __init__(self, net_params = {
        "neurons": {
            ## neurons params
            # shape params
            "size": (100,),
        },
        "GABAa": {
            # GABAa es
            "conn": bp.connect.All2All(include_self = False),
        },
    }, run_params = {
        "inputs": 1.,
        "dt": 0.01,
        "duration": 500,
    }):
        # init params
        self.net_params = net_params
        self.run_params = run_params

        # init backend
        bp.backend.set(dt = run_params["dt"])

        ## init comps of network
        # init neurons
        self.neurons = neurons.HH(
            size = net_params["neurons"]["size"],
            # monitor
            monitors = ["spike", "V"]
        )
        self.neurons.V = -70. + bp.ops.normal(size = net_params["neurons"]["size"]) * 20.
        # init es
        self.es = synapses.GABAa(
            pre = self.neurons,
            post = self.neurons,
            conn = net_params["GABAa"]["conn"]
        )
        self.es.g_max = .1 / bp.size2len(net_params["neurons"]["size"])

        # integrate network
        self.network = super(GammaOsciNet, self).__init__(
            self.neurons, self.es
        )

    def run(self, report = True, report_percent = 0.1):
        # excute super.run
        super(GammaOsciNet, self).run(
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
        fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
        xlim = (self.t_start - 0.1, self.t_end + 0.1)

        fig.add_subplot(gs[0, 0])
        bp.visualize.line_plot(
            ts = self.ts,
            val_matrix = self.neurons.mon.V,
            xlim = xlim,
            ylabel = "Membrane potential (N0)"
        )

        fig.add_subplot(gs[1, 0])
        bp.visualize.raster_plot(
            ts = self.ts,
            sp_matrix = self.neurons.mon.spike,
            xlim = xlim,
            show = True
        )

        # img show or save
        if img_fname:
            fig.savefig(fname = img_fname)
        else:
            fig.show()


