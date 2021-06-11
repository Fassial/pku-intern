"""
Created on 23:07, Apr. 22nd, 2021
Author: fassial
Filename: main.py
"""
import os
import copy
import pickle
import numpy as np
import brainpy as bp
# local dep
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
    "ipRGC": {
        "white": stimulus.stim_params(
            name = "white",
            height = 100,
            width = 1,
            duration = 1000,
            others = {
                "noise": .0,
            }
        ),
        "black": stimulus.stim_params(
            name = "black",
            height = 100,
            width = 1,
            duration = 1000,
            others = {
                "noise": 0.,
            }
        ),
        "normal": stimulus.stim_params(
            name = "normal",
            height = 100,
            width = 1,
            duration = 1000,
            others = {
                "freqs": np.full((100,), 20., dtype = np.float32),
                "noise": 0.,
            }
        ),
        "frate_increase": stimulus.stim_params(
            name = "frate_increase",
            height = 100,
            width = 1,
            duration = 1000,
            others = {
                "freqs": np.full((100,), 20., dtype = np.float32),
                "factor": 4.,   # (1,16)
                "ratio": .2,
                "noise": 0.,
            }
        ),
    },
    "PAC": {
        "white": stimulus.stim_params(
            name = "white",
            height = 50,
            width = 1,
            duration = 1000,
            others = {
                "noise": .0,
            }
        ),
        "black": stimulus.stim_params(
            name = "black",
            height = 50,
            width = 1,
            duration = 1000,
            others = {
                "noise": 0.,
            }
        ),
        "normal": stimulus.stim_params(
            name = "normal",
            height = 50,
            width = 1,
            duration = 1000,
            others = {
                "freqs": np.full((50,), 20., dtype = np.float32),
                "noise": 0.,
            }
        ),
        "frate_increase": stimulus.stim_params(
            name = "frate_increase",
            height = 50,
            width = 1,
            duration = 1000,
            others = {
                "freqs": np.full((50,), 20., dtype = np.float32),
                "factor": 4.,   # (1,16)
                "ratio": .2,
                "noise": 0.,
            }
        ),
    },
}
# default net_params
default_net_params = {
    "ipRGC": {
        ## neurons params
        # shape params
        "size": (100,),
        # dynamic params
        "V_init": "gaussian",
        "tau": 5.,  # tau > t_refractory
        "t_refractory": 1.,
        "noise": .2,
    },
    "PAC": {
        ## neurons params
        # shape params
        "size": (50,),
        # dynamic params
        "V_init": "gaussian",
        "tau": 5.,
        "t_refractory": 1.,
        "noise": .2,
    },
    "GJ_RP": {
        # gap junction
        "neighbors": 3,
        "weight": .3,
        "k_spikelet": .1,
        "conn": model.connector.IndexConnector(),
    },
    "ES_RP": {
        # exp synapses
        "neighbors": 5,
        "weight": .5,
        "delay": .5,    # random dists
        "tau": .5,
        "conn": model.connector.IndexConnector(),
    },
}

def main(dt = 0.01):
    # init seed & expr_curr
    np.random.seed(0)
    expr_curr = "white"
    # init backend
    bp.backend.set(dt = dt)
    bp.backend.set(backend = "numba")
    model.set_backend(backend = "numba")

    ## get stim
    # get stim_fname
    stim_iprgc_fname = os.path.join(
        DIR_OUTPUTS_STIM,
        expr_curr + "-" + str(default_stim_params["ipRGC"][expr_curr].duration) + "-iprgc" + ".csv"
    )
    stim_pac_fname = os.path.join(
        DIR_OUTPUTS_STIM,
        expr_curr + "-" + str(default_stim_params["PAC"][expr_curr].duration) + "-pac" + ".csv"
    )
    # get stim_iprgc
    if os.path.exists(stim_iprgc_fname):
        # load stim
        stim_iprgc = np.loadtxt(
            fname = stim_iprgc_fname,
            delimiter = ","
        )
    else:
        # get stim
        stim_iprgc, _ = stimulus.stimulus.get(
            stim_params = default_stim_params["ipRGC"][expr_curr]
        )
        # save stim
        np.savetxt(fname = stim_iprgc_fname, X = stim_iprgc, delimiter = ",")
    # get stim_pac
    if os.path.exists(stim_pac_fname):
        # load stim
        stim_pac = np.loadtxt(
            fname = stim_pac_fname,
            delimiter = ","
        )
    else:
        # get stim
        stim_pac, _ = stimulus.stimulus.get(
            stim_params = default_stim_params["PAC"][expr_curr]
        )
        # save stim
        np.savetxt(fname = stim_pac_fname, X = stim_pac, delimiter = ",")
    # rescale stim
    stim_pac *= .9; stim_iprgc *= .9

    ## exec sim
    # inst RPNet
    net = model.RPNet(net_params = default_net_params, run_params = {
        "inputs": {
            "ipRGC": stim_iprgc,
            "PAC": stim_pac,
        },
        "dt": 0.01,
        "duration": default_stim_params["ipRGC"][expr_curr].duration,
    })
    # net run
    net.run(report = True)
    # show net.mon
    net_monitors = net.get_monitors()
    net.show(img_fname = os.path.join(DIR_FIGS, expr_curr + ".png"))

if __name__ == "__main__":
    main()

