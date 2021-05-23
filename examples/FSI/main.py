"""
Created on 16:46, May. 22nd, 2021
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

## default params
# default stim_params
default_stim_params = {
    "normal": stimulus.stim_params(
        name = "normal",
        height = 200,
        width = 1,
        duration = 100,
        others = {
            "freqs": np.full((200,), 20., dtype = np.float32),
            "noise": 0.,
        }
    ),
}
# default net_params
default_net_params = {
    "neurons" : {
        "size": (200,),
        "V_init": "reset",
    },
    "GJ": {
        "r": 1,
        "p": 1.,
        "weight": .3,
        "conn": model.connector.IndexConnector(),
    },
    "CHEMS": {
        "r": 1,
        "p": 1.,
        "weight": 2.,
        "conn": model.connector.IndexConnector(),
    }
}

def main(dt = 0.01):
    # init seed
    np.random.seed(0)
    # init backend
    bp.backend.set(dt = dt)
    bp.backend.set(backend = "numpy")

    # init expr_curr
    expr_curr = "normal"

    # init inputs
    inputs_neurons = []
    # get stimulus
    stim_neurons, _ = stimulus.stimulus.get(
        stim_params = default_stim_params[expr_curr]
    ); stim_neurons += 1.
    # inst FSI
    net = model.FSI(net_params = default_net_params, run_params = {
        "inputs": stim_neurons,
        "dt": dt,
        "duration": default_stim_params[expr_curr].duration,
    })
    # net run
    net.run(report = True)
    # show net.mon
    net_monitors = net.get_monitors()
    net.show(img_fname = os.path.join(DIR_FIGS, expr_curr + ".png"))

if __name__ == "__main__":
    main()

