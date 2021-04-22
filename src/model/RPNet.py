"""
Created on 18:52, Apr. 22nd, 2021
Author: fassial
Filename: RPNet.py
"""
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

    def __init__(self, net_params = {
        "ipRGC": {
            ## neurons params
            # shape params
            "size": (60, 60),
            # dynamic params
            "V_reset": 0,
            "V_th": 10,
            "V_init": "reset",
            "tau": 5,
            "t_refractory": 3.5,
            "noise_sigma": 0.5,
            ## synapses params
            # gap junction
            "gj_w": 0.5,
            "gj_spikelet": 0.15,
            "gj_conn": bp.connect.GridEight(include_self = False),
            # exp synapses
            "es_w": 0.5,
            "es_delay": 0.1,
            "es_conn": connector.NeighborConnector(neighbors = 1)
        },
        "PAC": {
            ## neurons params
            # shape params
            "size": 1,
            # dynamic params
            "V_reset": 0,
            "V_th": 10,
            "V_init": "reset",
            "tau": 5,
            "t_refractory": 0.5,
            "noise_sigma": 0.1,
            ## synapses params
            # gap junction
            "gj_w": 0.5,
            "gj_spikelet": 0.15,
            "gj_conn": bp.connect.GridEight(include_self = False),
        }
    }, run_params = {
        "inputs": {
            "RGC": 0.,
            "SC": 0.,
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
        # init rgc
        self.rgc = neurons.LIF(
            size = net_params["RGC"]["size"],
            V_rest = net_params["RGC"]["V_reset"],
            V_reset = net_params["RGC"]["V_reset"],
            V_th = net_params["RGC"]["V_th"],
            V_init = net_params["RGC"]["V_init"],
            R = 1.,
            tau = net_params["RGC"]["tau"],
            t_refractory = net_params["RGC"]["t_refractory"],
            noise = net_params["RGC"]["noise_sigma"],
            # monitor
            monitors = ["V", "spike"]
        )
        # init gj in rgc
        self.gj = synapses.GapJunction_LIF(
            pre = self.rgc,
            post = self.rgc,
            conn = net_params["RGC"]["gj_conn"],
            weight = net_params["RGC"]["gj_w"],
            delay = 0.,
            k_spikelet = net_params["RGC"]["gj_spikelet"],
            post_refractory = True
        )
        # init sc
        self.sc = neurons.LIF(
            size = net_params["SC"]["size"],
            V_rest = net_params["SC"]["V_reset"],
            V_reset = net_params["SC"]["V_reset"],
            V_th = net_params["SC"]["V_th"],
            V_init = net_params["SC"]["V_init"],
            R = 1.,
            tau = net_params["SC"]["tau"],
            t_refractory = net_params["SC"]["t_refractory"],
            noise = net_params["SC"]["noise_sigma"],
            # monitor
            monitors = ["V", "spike"]
        )
        # init chem betweem rgc and sc
        self.chem = synapses.VoltageJump(
            pre = self.rgc,
            post = self.sc,
            conn = bp.connect.All2All(include_self = True),
            weight = net_params["SC"]["R2N_w"],
            delay = net_params["SC"]["R2N_delay"],
            post_refractory = True
        )

        # integrate network
        self.network = super(RGC_SC_Net, self).__init__(
            self.rgc, self.sc, self.gj, self.chem
        )

    def run(self, report = True, report_percent = 0.1):
        # excute super.run
        super(RGC_SC_Net, self).run(
            duration = self.run_params["duration"],
            inputs = (
                (self.rgc, "input", self.run_params["inputs"]["RGC"]),
                (self.sc, "input", self.run_params["inputs"]["SC"]),
            ),
            report = report,
            report_percent = report_percent
        )

    def get_monitors(self):
        monitors = {
            "RGC": self.rgc.mon,
            "SC": self.sc.mon,
        }
        return monitors

    def show(self, img_size = None, img_fname = None):
        # init fig & gs
        fig = plt.figure(
            figsize = img_size,
            constrained_layout = True
        )
        gs = GridSpec(3, 1, figure = fig)

        ## axes 11: show RGC spikes
        ax11 = fig.add_subplot(gs[0:2,:])
        # get neu_idx & t_spike
        neu_idx, t_spike = bp.measure.raster_plot(
            sp_matrix = self.rgc.mon.spike,
            times = self.rgc.mon.ts
        )
        # plot ax11
        ax11.plot(
            # plot 1
            t_spike, neu_idx, ".",
            # plot settings
            markersize = 1
        )
        ax11.set_xlim(left = -0.1, right = self.run_params["duration"] + 0.1)
        ax11.set_ylim(bottom = -0.1, top = self.net_params["RGC"]["size"][0] * self.net_params["RGC"]["size"][1] + 0.1)
        ax11.set_ylabel(ylabel = "RGCs")
        # ax11.set_xlabel(xlabel = "Time [{} ms]".format(self.run_params["duration"]))
        # ax11.set_title(label = "raster_plot(spike) of RGCs")

        ## axes 12: show RON membrane potentials
        ax12 = fig.add_subplot(gs[2:,:])
        # plot ax12
        ax12.plot(
            # plot 1
            self.sc.mon.ts, self.sc.mon.V[:,0]
        )
        ax12.set_xlim(left = -0.1, right = self.run_params["duration"] + 0.1)
        ax12.set_ylim(bottom = -0.1, top = self.net_params["RGC"]["V_th"] + 0.1)
        ax12.set_ylabel(ylabel = "SC")
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

