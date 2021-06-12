"""
Created on 16:50, May. 22nd, 2021
Author: fassial
Filename: stimulus.py
"""
import numpy as np
import brainpy as bp
import matplotlib.pyplot as plt
from copy import deepcopy
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
    def show_stim(stim_params, img_size = None, img_fname = None):
        # get corresponding stim
        stim, _ = stimulus.get(stim_params = stim_params)

        # reshape stim
        stim = stim[0,:].reshape((stim_params.height, stim_params.width))

        # gen plt
        plt.imshow(stim[::-1,:], cmap = "gray")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(pad = 1.)

        # save img
        if img_fname is None:
            plt.show()
        else:
            plt.savefig(fname = img_fname)

    @staticmethod
    def show_spike(stim_params, img_size = None, img_fname = None):
        # get corresponding spike
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
    def _black(height = 100, width = 1, duration = 100, stim_params = {
        "noise": 0.,
    }):
        # set stim & spike
        stim = np.zeros((int(duration / bp.backend.get_dt()), height * width), dtype=np.float32)
        spike = np.zeros((int(duration / bp.backend.get_dt()), height * width), dtype=np.float32)
        # add noise to stim
        stim *= np.random.normal(
            loc = 1.,
            scale = stim_params["noise"],
            size = stim.shape
        )

        return stim, spike

    # white stimulus funcs
    @staticmethod
    def _white(height = 100, width = 1, duration = 100, stim_params = {
        "noise": 0.,
    }):
        # set stim & spike
        stim = np.ones((int(duration / bp.backend.get_dt()), height * width), dtype=np.float32)
        spike = np.zeros((int(duration / bp.backend.get_dt()), height * width), dtype=np.float32)
        # add noise to stim
        stim *= np.random.normal(
            loc = 1.,
            scale = stim_params["noise"],
            size = stim.shape
        )

        return stim, spike

    # normal stimulus funcs
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
                "weight": 8.,
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

    @staticmethod
    def _frate_increase(height = 100, width = 1, duration = 100, stim_params = {
        "freqs": np.full((100,), 20., dtype = np.float32),
        "factor": 4.,   # (1,16)
        "ratio": .2,
        "noise": 0.,
    }):
        ## normal stim
        # init freqs
        freqs = deepcopy(stim_params["freqs"])
        # set stim1 & spike1
        stim1, spike1 = inputs.poisson_input(duration = duration // 2, net_params = {
            "neurons": {
                "size": (height, width),
            },
            "synapses": {
                "weight": 4.,
                "delay": 0.,
                "tau1": .3,
                "tau2": 3.,
            }
        }, others = {
            "freqs": freqs,
        })

        ## frate_increase stim
        # init idxs & freqs
        idxs = [i for i in range(int(height*(.5-stim_params["ratio"]/2)), int(height*(.5+stim_params["ratio"]/2)))]
        freqs[idxs] *= stim_params["factor"]
        # set stim2, spike2
        stim2, spike2 = inputs.poisson_input(duration = duration // 2, net_params = {
            "neurons": {
                "size": (height, width),
            },
            "synapses": {
                "weight": 4.,
                "delay": 0.,
                "tau1": .3,
                "tau2": 3.,
            }
        }, others = {
            "freqs": freqs,
        })

        # set stim, spike
        stim = np.vstack((stim1, stim2))
        spike = np.vstack((spike1, spike2))
        # add noise to stim
        stim *= np.random.normal(
            loc = 1.,
            scale = stim_params["noise"],
            size = stim.shape
        )

        return stim, spike

    # hole stimulus funcs
    @staticmethod
    def _one_hole(height = 50, width = 50, duration = 100, stim_params = {
        "radius": 5,
        "position": "center",   # ["center", "corner", "line_middle"]
        "noise": 0.,
    }):
        # init stim & spike
        stim = np.zeros((int(duration / bp.backend.get_dt()), height, width), dtype=np.float32)
        spike = np.zeros((int(duration / bp.backend.get_dt()), height * width), dtype=np.float32)

        # set center
        if stim_params["position"] == "center":
            center = (height / 2, width / 2)
        elif stim_params["position"] == "corner":
            center = (stim_params["radius"] - 1, stim_params["radius"] - 1)
        elif stim_params["position"] == "line_middle":
            center = (stim_params["radius"] - 1, width / 2)
        else:
            raise ValueError("ERROR: Unknown stim_params[\"position\"] in stimulus._one_hole.")

        for i in range(height):
            for j in range(width):
                dist = stimulus._dist(i, j, center)
                if dist < stim_params["radius"]:
                    stim[:, i, j] = 1.

        # reshape stim
        stim = stim.reshape((-1,height * width))
        # add noise to stim
        stim *= np.random.normal(
            loc = 1.,
            scale = stim_params["noise"],
            size = stim.shape
        )
        return stim, spike

    @staticmethod
    def _two_holes(height = 50, width = 50, duration = 100, stim_params = {
        "radius_left": 5,
        "radius_right": 5,
        "interval": 15,
        "noise": 0.,
    }):
        # init stim & spike
        stim = np.zeros((int(duration / bp.backend.get_dt()), height, width), dtype=np.float32)
        spike = np.zeros((int(duration / bp.backend.get_dt()), height * width), dtype=np.float32)

        # set center
        center_left = (height / 2, width / 2 - stim_params["interval"] / 2)
        center_right = (height / 2, width / 2 + stim_params["interval"] / 2)

        for i in range(height):
            for j in range(width):
                dist_left = stimulus._dist(i, j, center_left)
                dist_right = stimulus._dist(i, j, center_right)
                if dist_left < stim_params["radius_left"]:
                    stim[:, i, j] = 1.
                if dist_right < stim_params["radius_right"]:
                    stim[:, i, j] = 1.

        # reshape stim
        stim = stim.reshape((-1,height * width))
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
    "white": stim_params(
        name = "white",
        height = 100,
        width = 1,
        duration = 100,
        others = {
            "noise": 0.,
        }
    ),
    "black": stim_params(
        name = "black",
        height = 100,
        width = 1,
        duration = 100,
        others = {
            "noise": 0.,
        }
    ),
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
    "frate_increase": stim_params(
        name = "frate_increase",
        height = 100,
        width = 1,
        duration = 100,
        others = {
            "freqs": np.full((100,), 20., dtype = np.float32),
            "factor": 4.,    # (1,16)
            "ratio": .2,
            "noise": 0.,
        }
    ),
    "one_hole": stim_params(
        name = "one_hole",
        height = 50,
        width = 50,
        duration = 100,
        others = {
            "radius": 5,
            "position": "center",
            "noise": 0.,
        }
    ),
    "two_holes": stim_params(
        name = "two_holes",
        height = 50,
        width = 50,
        duration = 100,
        others = {
            "radius_left": 5,
            "radius_right": 5,
            "interval": 15,
            "noise": 0.,
        }
    ),
}

