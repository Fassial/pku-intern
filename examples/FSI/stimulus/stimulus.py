"""
Created on 16:50, May. 22nd, 2021
Author: fassial
Filename: stimulus.py
"""
import numpy as np
import brainpy as bp
import matplotlib.pyplot as plt
# local dep
from . import inputs

__all__ = [
    "stimulus",
    "stim_params",
]

class stimulus(object):

    @staticmethod
    def get(stim_params):
        # get corresponding func in stimulus
        try:
            func = getattr(stimulus, "_{}".format(stim_params.name))
        except Exception:
            raise ValueError("ERROR: Unknown function in stimulus.get.")
        # get stim & spike
        stim, spike = func(
            height = stim_params.height,
            width = stim_params.width,
            duration = stim_params.duration,
            stim_params = stim_params.others
        )
        return stim, spike

    @staticmethod
    def show(stim_params, img_size = None, img_fname = None):
        # get corresponding arr & idxs & stim
        _, spike = stimulus.get(stim_params = stim_params)

        # init fig & gs
        fig = plt.figure(
            figsize = img_size,
            constrained_layout = True
        )

        # get neu_idx & t_spike
        neu_idx, t_spike = bp.measure.raster_plot(
            sp_matrix = spike,
            times = np.arange(0., stim_params.duration, stim_params.duration / spike.shape[0]).astype(np.float32)
        )
        # plot fig
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(
            # plot 1
            t_spike, neu_idx, ".",
            # plot settings
            markersize = 2, color = "blue"
        )
        ax.set_xlim(left = -0.1, right = stim_params.duration + 0.1)
        ax.set_ylim(bottom = -0.1, top = bp.size2len(spike.shape[1]) + 0.1)
        ax.set_ylabel(ylabel = "neurons")
        ax.set_xlabel(xlabel = "Time [{} ms]".format(stim_params.duration))
        ax.set_title(label = "raster_plot(spike) of neurons")

        # img show or save
        if img_fname:
            plt.savefig(fname = img_fname)
            plt.close(fig)
        else:
            plt.show()

    """
    tool funcs
    """
    @staticmethod
    def _dist(i, j, center):
        return np.power((i - center[0]) ** 2 + (j - center[1]) ** 2, 0.5)

    """
    stimulus funcs
    """
    # black stimulus funcs
    @staticmethod
    def _black(height = 100, width = 1, stim_params = {
        "noise": 0.,
    }):
        # set stim & spike
        stim = np.zeros((height * width,), dtype=np.float32)
        spike = np.zeros((height * width,), dtype=np.float32)
        # add noise to stim
        stim *= np.random.normal(
            loc = 1.,
            scale = stim_params["noise"],
            size = stim.shape
        )

        return stim, spike

    # black stimulus funcs
    @staticmethod
    def _normal(height = 100, width = 1, duration = 100, stim_params = {
        "freqs": np.full((100,), 20., dtype = np.float32),
        "noise": 0.,
    }):
        # set stim & spike
        stim, spike = inputs.poisson_input(duration = duration, net_params = {
            "neurons": {
                "size": (height, width),
            },
            "synapses": {
                "weight": .3,
                "delay": 0.,
                "tau1": .3,
                "tau2": 3.,
            }
        }, others = {
            "freqs": stim_params["freqs"],
        })
        # add noise to stim
        stim *= np.random.normal(
            loc = 1.,
            scale = stim_params["noise"],
            size = stim.shape
        )

        return stim, spike

class stim_params:

    def __init__(self, name, height, width, duration, others):
        # init params
        self.name = name
        self.height = height
        self.width = width
        self.duration = duration
        self.others = others

default_stim_params = {
    "normal": stim_params(
        name = "normal",
        height = 100,
        width = 1,
        duration = 100,
        others = {
            "freqs": np.full((100,), 20., dtype = np.float32),
            "noise": 0.,
        }
    ),
}

