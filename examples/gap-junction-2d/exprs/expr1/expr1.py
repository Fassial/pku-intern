"""
Created on 01:54, June. 4th, 2021
Author: fassial
Filename: expr1.py
Description:
    TODO
"""
import gc
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
    "white": stimulus.stim_params(
        name = "white",
        height = 50,
        width = 50,
        duration = 300,
        others = {
            "noise": 0.,
        }
    ),
    "black": stimulus.stim_params(
        name = "black",
        height = 50,
        width = 50,
        duration = 300,
        others = {
            "noise": 0.,
        }
    ),
    "normal": stimulus.stim_params(
        name = "normal",
        height = 50,
        width = 50,
        duration = 300,
        others = {
            "freqs": np.full((50, 50), 20., dtype = np.float32),
            "noise": 0.,
        }
    ),
    "frate_increase": stimulus.stim_params(
        name = "frate_increase",
        height = 50,
        width = 50,
        duration = 300,
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

def expr(gj_w, gj_k, run_params, dt = 0.01):
    print("processing expr(" +\
        str(gj_w) + "," + str(gj_k) + ")..."
    )

    # init seed
    np.random.seed(0)

    # init expr_curr & stim
    expr_curr = run_params["expr_curr"]
    stim = run_params["stim"]

    ## prepare expr
    # init net_params
    net_params = deepcopy(default_net_params)
    net_params["GJ"]["weight"] = gj_w; net_params["GJ"]["k_spikelet"] = gj_k; print(net_params)

    ## exec expr
    # inst GJ2DNet
    net = model.GJ2DNet(net_params = net_params, run_params = {
        "inputs": stim,
        "dt": 0.01,
        "duration": default_stim_params[expr_curr].duration,
    })
    # net run
    net.run(report = True)
    # show net.mon
    net_monitors = net.get_monitors()
    net.show(img_fname = os.path.join(DIR_FIGS,
        expr_curr + "-" +\
        str(gj_w) + "-" + str(gj_k) + ".png"
    ))
    net.save(spike_fname = os.path.join(DIR_OUTPUTS_SPIKE,
        expr_curr + "-" +\
        str(gj_w) + "-" + str(gj_k) + ".csv"
    ))

    ## compute omega
    spike = net_monitors.spike.T
    print(spike.shape)
    omega = 0.  # omega = utils.get_omega(spike = spike, dt = dt)

    # rm vars
    del(net_params); del(net); del(net_monitors); del(spike); gc.collect()

    return omega

def main(dt = 0.01):
    # init omegas
    omegas = []

    # init gj_ws & gj_ks
    gj_ws = [.05, .1, .2, .3, .4, .5]
    gj_ks = [.05, .1, .15, .2, .3, .5]

    ## init backend
    bp.backend.set(dt = dt)
    bp.backend.set(backend = "numba")
    ## init expr_curr
    expr_curr = "white"
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
    stim *= .8; print(stim.shape)

    # set omegas
    for gj_w in gj_ws:
        omega = []
        for gj_k in gj_ks:
            omega.append(expr(
                gj_w = gj_w,
                gj_k = gj_k,
                run_params = {
                    "expr_curr": expr_curr,
                    "stim": stim,
                },
                dt = dt
            ))
        omegas.append(omega)
    omegas = np.array(omegas, dtype = np.float32)

    # save omegas
    np.savetxt(
        fname = os.path.join(DIR_OUTPUTS, "omegas.csv"),
        X = omegas,
        delimiter = ","
    )

if __name__ == "__main__":
    main()

