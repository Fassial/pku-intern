"""
Created on 17:33, Apr. 6th, 2021
Author: fassial
Filename: main.py
"""
import os
import numpy as np
import brainpy as bp
# local dep
import utils
import model
import stimulus

# macro
DIR_ROOT = os.getcwd()
DIR_FIGS = os.path.join(DIR_ROOT, "figs")
if not os.path.exists(DIR_FIGS): os.mkdir(DIR_FIGS)
DIR_OUTPUTS = os.path.join(DIR_ROOT, "outputs")
if not os.path.exists(DIR_OUTPUTS): os.mkdir(DIR_OUTPUTS)
DIR_OUTPUTS_STIM = os.path.join(DIR_OUTPUTS, "stimulus")
if not os.path.exists(DIR_OUTPUTS_STIM): os.mkdir(DIR_OUTPUTS_STIM)
DIR_OUTPUTS_SPIKE = os.path.join(DIR_OUTPUTS, "spike")
if not os.path.exists(DIR_OUTPUTS_SPIKE): os.mkdir(DIR_OUTPUTS_SPIKE)

## default params
# default stim_params
default_stim_params = {
    "white": stimulus.stim_params(
        name = "white",
        height = 50,
        width = 50,
        duration = 100,
        others = {
            "noise": 0.,
        }
    ),
    "black": stimulus.stim_params(
        name = "black",
        height = 50,
        width = 50,
        duration = 100,
        others = {
            "noise": 0.,
        }
    ),
    "normal": stimulus.stim_params(
        name = "normal",
        height = 50,
        width = 50,
        duration = 100,
        others = {
            "freqs": np.full((50, 50), 20., dtype = np.float32),
            "noise": 0.,
        }
    ),
    "frate_increase": stimulus.stim_params(
        name = "frate_increase",
        height = 50,
        width = 50,
        duration = 100,
        others = {
            "freqs": np.full((50, 50), 20., dtype = np.float32),
            "factor": 4.,    # (1,16)
            "ratio": .2,
            "noise": 0.,
        }
    ),
}
# default net_params
default_net_params = {
    "neurons": {
        ## neurons params
        # shape params
        "size": (50, 50),
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
}

def main(dt = 0.01):
    # init seed & expr_curr
    np.random.seed(0)
    expr_curr = "white"
    # init backend
    bp.backend.set(dt = dt)
    bp.backend.set(backend = "numba")

    ## get stim
    # get stim_fname
    stim_fname = os.path.join(
        DIR_OUTPUTS_STIM,
        expr_curr + "-" + str(default_stim_params[expr_curr].duration) + ".csv"
    )
    # get stim
    if os.path.exists(stim_fname):
        # load stim
        stim = np.loadtxt(
            fname = stim_fname,
            delimiter = ","
        )
    else:
        # get stim
        stim, _ = stimulus.stimulus.get(
            stim_params = default_stim_params[expr_curr]
        )
        # save stim
        np.savetxt(fname = stim_fname, X = stim, delimiter = ",")
    # rescale stim
    stim *= .95; print(stim.shape)

    ## exec sim
    # inst GJ2DNet
    net = model.GJ2DNet(net_params = default_net_params, run_params = {
        "inputs": stim,
        "dt": 0.01,
        "duration": default_stim_params[expr_curr].duration,
    })
    # net run
    net.run(report = True)
    # show net.mon
    net_monitors = net.get_monitors()
    net.show(img_fname = os.path.join(DIR_FIGS, expr_curr + ".png"))

if __name__ == "__main__":
    main()

