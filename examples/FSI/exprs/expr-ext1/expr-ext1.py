"""
Created on 16:13, May. 25th, 2021
Author: fassial
Filename: expr-ext1.py
Description:
    Excitatory chemical connections cannot substitute for GJ in the formation of local active zones.
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
    "normal": stimulus.stim_params(
        name = "normal",
        height = 200,
        width = 1,
        duration = 1000,
        others = {
            "freqs": np.full((200,), 20., dtype = np.float32),
            "noise": .2,
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
        "p": 0.,
        "weight": .3,
        "conn": model.connector.IndexConnector(),
    },
    "CHEMS": {
        "r": 30,
        "p": 1.,
        "weight": 2.,
        "conn": model.connector.IndexConnector(),
    }
}

def expr(r_g, p_g, w_g, r_i, p_i, w_i, dt = 0.01):
    print("processing expr(" +\
        str(r_g) + "," + str(p_g) + "," + str(w_g) + "," +\
        str(r_i) + "," + str(p_i) + "," + str(w_i) + ")...")

    # init seed
    np.random.seed(0)
    # init backend
    bp.backend.set(dt = dt)
    bp.backend.set(backend = "numpy")

    # init expr_curr
    expr_curr = "normal"

    ## prepare expr
    # init net_params
    net_params = deepcopy(default_net_params)
    net_params["GJ"]["r"] = r_g; net_params["GJ"]["p"] = p_g; net_params["GJ"]["weight"] = w_g
    net_params["CHEMS"]["r"] = r_i; net_params["CHEMS"]["p"] = p_i; net_params["CHEMS"]["weight"] = w_i; print(net_params)

    # init inputs
    inputs_neurons = []
    # get stimulus
    stim_fname = os.path.join(
        DIR_OUTPUTS_STIM,
        expr_curr + "-" + str(default_stim_params[expr_curr].duration) + ".csv"
    )
    if os.path.exists(stim_fname):
        # load stim
        stim_neurons = np.loadtxt(
            fname = stim_fname,
            delimiter = ","
        )
    else:
        # get stim
        stim_neurons, _ = stimulus.stimulus.get(
            stim_params = default_stim_params[expr_curr]
        ); stim_neurons += .5
        # save stim
        np.savetxt(fname = stim_fname, X = stim_neurons, delimiter = ",")

    ## exec expr
    # inst FSI
    net = model.FSI(net_params = net_params, run_params = {
        "inputs": stim_neurons,
        "dt": dt,
        "duration": default_stim_params[expr_curr].duration,
    })
    # net run
    net.run(report = True)
    # show net.mon
    net_monitors = net.get_monitors()
    net.show(img_fname = os.path.join(DIR_FIGS,
        expr_curr +\
        "-" + str(r_g) + "-" + str(p_g) + "-" + str(w_g) +\
        "-" + str(r_i) + "-" + str(p_i) + "-" + str(w_i) + ".png"
    ))
    net.save(spike_fname = os.path.join(DIR_OUTPUTS_SPIKE,
        expr_curr + "-" + str(dt) +\
        "-" + str(r_g) + "-" + str(p_g) + "-" + str(w_g) +\
        "-" + str(r_i) + "-" + str(p_i) + "-" + str(w_i) + ".csv"
    ))

    ## compute omega
    print(net_monitors.spike.T.shape)
    omega = utils.get_omega(spike = net_monitors.spike.T, dt = dt)

    return omega

def main(dt = 0.01):
    # init expr_params
    expr_params = [
        # [r_g, p_g, w_g, r_i, p_i, w_i]
        [5, 0., .3, 30, 1., 0.],
        [5, 0., .3, 30, 1., .3],
        [5, 0., .3, 30, 1., 1.],
        [5, 0., 2., 30, 1., 0.],
        [5, 0., 2., 30, 1., .3],
        [5, 0., 2., 30, 1., 1.],
    ]

    # exec expr
    for expr_param in expr_params:
        # set r_g et.al
        r_g = expr_param[0]; p_g = expr_param[1]; w_g = expr_param[2]
        r_i = expr_param[3]; p_i = expr_param[4]; w_i = expr_param[5]
        expr(
            r_g = r_g, p_g = p_g, w_g = w_g,
            r_i = r_i, p_i = p_i, w_i = w_i,
            dt = dt
        )

if __name__ == "__main__":
    main()

