"""
Created on 15:25, May. 24th, 2021
Author: fassial
Filename: expr7.py
Description:
    Spatial network patterning (Ω) as a function of inhibitory network topology (p_i)
    for r_i = 5, 10, 30.
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
        "r": 5,
        "p": 0.,
        "weight": .3,
        "conn": model.connector.IndexConnector(),
    },
    "CHEMS": {
        "r": 1,
        "p": 0.,
        "weight": 2.,
        "conn": model.connector.IndexConnector(),
    }
}

def expr(r_i, p_i, dt = 0.01):
    print("processing expr(" + str(r_i) + "," + str(p_i) + ")...")

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
    net_params["CHEMS"]["r"] = r_i; net_params["CHEMS"]["p"] = p_i; print(net_params)

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
        expr_curr + "-" + str(r_i) + "-" + str(p_i) + ".png"
    ))
    net.save(spike_fname = os.path.join(DIR_OUTPUTS_SPIKE,
        expr_curr + "-" + str(dt) + "-" + str(r_i) + "-" + str(p_i) + ".csv"
    ))

    ## compute omega
    print(net_monitors.spike.T.shape)
    omega = utils.get_omega(spike = net_monitors.spike.T, dt = dt)

    return omega

def main(dt = 0.01):
    # init omegas
    omegas = []

    # init r_is & p_is
    r_is = [5, 10, 30]
    p_is = np.arange(0., 1.01, 0.05).astype(dtype = np.float32)

    # exec expr
    for r_i in r_is:
        omega = []
        for p_i in p_is:
            omega.append(expr(
                r_i = r_i,
                p_i = p_i,
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

