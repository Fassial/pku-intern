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

## default params
# default stim_params
default_stim_params = {
    "ipRGC": {
        "white": stimulus.stim_params(
            name = "white",
            height = 100,
            width = 1,
            intensity = [9,],
            others = {
                "noise": .2,
            }
        ),
        "black": stimulus.stim_params(
            name = "black",
            height = 100,
            width = 1,
            intensity = [9,],
            others = {
                "noise": 0,
            }
        ),
    },
    "PAC": {
        "white": stimulus.stim_params(
            name = "white",
            height = 50,
            width = 1,
            intensity = [9,],
            others = {
                "noise": .2,
            }
        ),
        "black": stimulus.stim_params(
            name = "black",
            height = 50,
            width = 1,
            intensity = [9,],
            others = {
                "noise": 0,
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
        "V_reset": 0,
        "V_th": 10,
        "V_init": "gaussian",
        "tau": 5,
        "t_refractory": 0.5,
    },
    "PAC": {
        ## neurons params
        # shape params
        "size": (50,),
        # dynamic params
        "V_reset": 0,
        "V_th": 10,
        "V_init": "gaussian",
        "tau": 5,
        "t_refractory": 0.5,
    },
    "GJ_RP": {
        # gap junction
        "neighbors": 1,
        "weight": .1,
        "k_spikelet": 0.15,
        "conn": model.connector.IndexConnector(),
    },
    "ES_RP": {
        # exp synapses
       "neighbors": 1,
        "weight": .1,
        "delay": .1,
        "tau": 8.,
        "conn": model.connector.IndexConnector(),
    },
}

def tuning(params = {
    "GJ_RP": {
        "neighbors": 1,
        "weight": .1,
    },
    "ES_RP": {
       "neighbors": 1,
        "weight": .1,
    }
}):
    ## set net_params
    net_params = copy.deepcopy(default_net_params)
    net_params["GJ_RP"]["neighbors"] = params["GJ_RP"]["neighbors"]
    net_params["GJ_RP"]["weight"] = params["GJ_RP"]["weight"]
    net_params["ES_RP"]["neighbors"] = params["ES_RP"]["neighbors"]
    net_params["ES_RP"]["weight"] = params["ES_RP"]["weight"]

    ## expr
    # init seed & expr_curr
    np.random.seed(0)
    expr_curr = "white"
    # init backend
    bp.backend.set(backend = "numpy")

    # init inputs
    inputs_iprgc, inputs_pac = [], []
    # get stimulus(white)
    _, _, stim_iprgc = stimulus.stimulus.get(
        stim_params = default_stim_params["ipRGC"][expr_curr]
    ); inputs_iprgc.append([stim_iprgc, 180])
    _, _, stim_pac = stimulus.stimulus.get(
        stim_params = default_stim_params["PAC"][expr_curr]
    ); inputs_pac.append([stim_pac, 180])
    # gen inputs
    inputs_iprgc, duration = bp.inputs.constant_current(inputs_iprgc, dt = 0.01)
    inputs_pac, duration = bp.inputs.constant_current(inputs_pac, dt = 0.01)
    # inst RPNet
    net = model.RPNet(net_params = net_params, run_params = {
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
    net.show(img_fname = os.path.join(DIR_FIGS, expr_curr +\
        "_" + str(params["GJ_RP"]["neighbors"]) +\
        "_" + str(params["GJ_RP"]["weight"]) +\
        "_" + str(params["ES_RP"]["neighbors"]) +\
        "_" + str(params["ES_RP"]["weight"]) +\
        ".png"))

def main():
    # init seed & expr_curr
    np.random.seed(0)
    expr_curr = "white"
    # init backend
    bp.backend.set(backend = "numpy")

    # init inputs
    inputs_iprgc, inputs_pac = [], []
    # get stimulus(white)
    _, _, stim_iprgc = stimulus.stimulus.get(
        stim_params = default_stim_params["ipRGC"][expr_curr]
    ); inputs_iprgc.append([stim_iprgc, 180])
    _, _, stim_pac = stimulus.stimulus.get(
        stim_params = default_stim_params["PAC"][expr_curr]
    ); inputs_pac.append([stim_pac, 180])
    # gen inputs
    inputs_iprgc, duration = bp.inputs.constant_current(inputs_iprgc, dt = 0.01)
    inputs_pac, duration = bp.inputs.constant_current(inputs_pac, dt = 0.01)
    # inst RPNet
    net = model.RPNet(net_params = default_net_params, run_params = {
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
    # main()
    for gj_neighbors in np.arange(1, 10, 1).astype(np.uint32):
        for gj_weight in np.arange(.1, 1., .1).astype(np.float32):
            for es_neighbors in np.arange(1, 10, 1).astype(np.uint32):
                for es_weight in np.arange(.1, 1., .1).astype(np.float32):
                    print("processing(" +\
                        str(gj_neighbors) + "," +\
                        str(gj_weight) + "," +\
                        str(es_neighbors) + "," +\
                        str(es_weight) + "," +\
                        ")")
                    tuning(params = {
                        "GJ_RP": {
                            "neighbors": gj_neighbors,
                            "weight": gj_weight,
                        },
                        "ES_RP": {
                           "neighbors": es_neighbors,
                            "weight": es_weight,
                        }
                    })

