"""
Created on 14:30, May. 30th, 2021
Author: fassial
Filename: expr1.py
Description:
    TODO
"""
import numpy as np
import brainpy as bp
from copy import deepcopy
# local dep
import os
import sys
sys.path.append(os.path.join("..", ".."))
import model
import stimulus
import utils

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
                "noise": .2,
            }
        ),
        "black": stimulus.stim_params(
            name = "black",
            height = 100,
            width = 1,
            duration = 1000,
            others = {
                "noise": .2,
            }
        ),
        "normal": stimulus.stim_params(
            name = "normal",
            height = 100,
            width = 1,
            duration = 1000,
            others = {
                "freqs": np.full((100,), 20., dtype = np.float32),
                "noise": .2,
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
                "noise": .2,
            }
        ),
        "black": stimulus.stim_params(
            name = "black",
            height = 50,
            width = 1,
            duration = 1000,
            others = {
                "noise": 0,
            }
        ),
        "normal": stimulus.stim_params(
            name = "normal",
            height = 50,
            width = 1,
            duration = 1000,
            others = {
                "freqs": np.full((50,), 20., dtype = np.float32),
                "noise": 0,
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
        "tau": .5,
        "t_refractory": 5.,
    },
    "PAC": {
        ## neurons params
        # shape params
        "size": (50,),
        # dynamic params
        "V_init": "gaussian",
        "tau": .5,
        "t_refractory": 5.,
    },
    "GJ_RP": {
        # gap junction
        "neighbors": 1,
        "weight": .5,
        "k_spikelet": .1,
        "conn": model.connector.IndexConnector(),
    },
    "ES_RP": {
        # exp synapses
       "neighbors": 2,
        "weight": .5,
        "delay": .1,
        "tau": .5,
        "conn": model.connector.IndexConnector(),
    },
}

def expr(gj_neigh, gj_w, es_neigh, es_w, dt = 0.01):
    print("processing expr(" +\
        str(gj_neigh) + "," + str(gj_w) + "," +\
        str(es_neigh) + "," + str(es_w) + ")..."
    )

    # init seed
    np.random.seed(0)
    # init backend
    bp.backend.set(dt = dt)
    bp.backend.set(backend = "numpy")

    # init expr_curr
    expr_curr = "white"

    ## prepare expr
    # init net_params
    net_params = deepcopy(default_net_params)
    net_params["GJ_RP"]["weight"] = gj_w; net_params["GJ_RP"]["neighbors"] = gj_neigh
    net_params["ES_RP"]["weight"] = es_w; net_params["ES_RP"]["neighbors"] = es_neigh; print(net_params)

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
    stim_pac *= .95; stim_iprgc *= .95

    ## exec expr
    # inst RPNet
    net = model.RPNet(net_params = net_params, run_params = {
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
    net.show(img_fname = os.path.join(DIR_FIGS,
        expr_curr + "-" + str(gj_neigh) + "-" + str(gj_w) + "-" +\
        str(es_neigh) + "-" + str(es_w) + ".png"
    ))
    net.save(spike_fname = os.path.join(DIR_OUTPUTS_SPIKE,
        expr_curr + "-" + str(gj_neigh) + "-" + str(gj_w) + "-" +\
        str(es_neigh) + "-" + str(es_w) + ".csv"
    ))

    ## compute omega
    spike = bp.ops.vstack((net_monitors["ipRGC"].spike.T, net_monitors["PAC"].spike.T))
    print(spike.shape)
    omega = utils.get_omega(spike = spike, dt = dt)

    return omega

def main(dt = 0.01):
    # init omegas
    omegas = []

    # init gj_neighs & gj_ws & es_neighs & es_ws
    gj_neighs = [1, 3, 5, 10]
    gj_ws = [.1, .3, .5]
    es_neighs = [1, 3, 5, 10]
    es_ws = [.1, .3, .5]

    # set omegas
    for gj_neigh in gj_neighs:
        for gj_w in gj_ws:
            for es_neigh in es_neighs:
                for es_w in es_ws:
                    expr(
                        gj_neigh = gj_neigh,
                        gj_w = gj_w,
                        es_neigh = es_neigh,
                        es_w = es_w,
                        dt = dt
                    )
    omegas = np.array(omegas, dtype = np.float32)

    # save omegas
    np.savetxt(
        fname = os.path.join(DIR_OUTPUTS, "omegas.csv"),
        X = omegas,
        delimiter = ","
    )

if __name__ == "__main__":
    main()

