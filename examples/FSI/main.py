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
    "ES": {
        "r": 1,
        "p": 1.,
        "weight": 2.,
        "conn": model.connector.IndexConnector(),
    }
}

def main():
    # init seed
    np.random.seed(0)
    # init backend
    bp.backend.set(backend = "numpy")

    # init expr_curr
    expr_curr = "normal"

    # init inputs
    inputs_neurons = []
    # get stimulus
    _, _, stim_neurons = stimulus.stimulus.get(
        stim_params = default_stim_params[expr_curr]
    ); inputs_neurons.append([stim_neurons, 1000])
    # gen inputs
    inputs_neurons, duration = bp.inputs.constant_current(inputs_neurons, dt = 0.01)
    # inst FSI
    net = model.FSI(net_params = default_net_params, run_params = {
        "inputs": {
            "ipRGC": inputs_iprgc,
            "PAC": inputs_pac,
        },
        "dt": 0.01,
        "duration": duration,
    })
    # net run
    net.run(report = True)
    # show net.mon
    net_monitors = net.get_monitors()
    net.show(img_fname = os.path.join(DIR_FIGS, expr_curr + ".png"))

if __name__ == "__main__":
    main()

