"""
Created on 23:07, Apr. 22nd, 2021
Author: fassial
Filename: main.py
"""
import os
import pickle
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
    "ipRGC": {
        "white": stimulus.stim_params(
            name = "white",
            height = 40,
            width = 1,
            intensity = [15,],
            others = None
        ),
    },
    "PAC": {
        "white": stimulus.stim_params(
            name = "white",
            height = 20,
            width = 1,
            intensity = [15,],
            others = None
        ),
    },
}
# default net_params
default_net_params = {
    "ipRGC": {
        ## neurons params
        # shape params
        "size": (40,),
        # dynamic params
        "V_reset": 0,
        "V_th": 10,
        "V_init": "reset",
        "tau": 5,
        "t_refractory": 3.5,
        "noise_sigma": 0.5,
    },
    "PAC": {
        ## neurons params
        # shape params
        "size": (20,),
        # dynamic params
        "V_reset": 0,
        "V_th": 10,
        "V_init": "reset",
        "tau": 5,
        "t_refractory": 0.5,
        "noise_sigma": 0.1,
    },
    "GJ_RP": {
        # gap junction
        "neighbors": 1,
        "weight": 0.5,
        "k_spikelet": 0.15,
        "conn": model.connector.IndexConnector(),
    },
    "ES_RP": {
        # exp synapses
       "neighbors": 2,
        "weight": 0.5,
        "delay": 0.1,
        "tau": 8.,
        "conn": model.connector.IndexConnector(),
    },
}

def main():
    # init expr_curr
    expr_curr = "white"
    # init backend
    bp.backend.set(backend = "numpy")
    # init stimulus
    _, _, stim_iprgc = stimulus.stimulus.get(
        stim_params = default_stim_params["ipRGC"][expr_curr]
    )
    _, _, stim_pac = stimulus.stimulus.get(
        stim_params = default_stim_params["PAC"][expr_curr]
    )
    # inst RPNet
    net = model.RPNet(net_params = default_net_params, run_params = {
        "inputs": {
            "ipRGC": stim_iprgc,
            "PAC": stim_pac,
        },
        "dt": 0.01,
        "duration": 8,
    })
    # net run & show
    net.run(report = True)
    net_monitors = net.get_monitors()
    net.show(img_fname = os.path.join(DIR_FIGS, expr_curr + ".png"))

if __name__ == "__main__":
    main()

