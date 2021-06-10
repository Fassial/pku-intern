"""
Created on 00:17, June. 9th, 2021
Author: fassial
Filename: main.py
"""
import os
import numpy as np
import brainpy as bp
# local dep
import model

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
    "white": {
        "duration": 500,
    },
}
# default net_params
default_net_params = {
    "neurons": {
        ## neurons params
        # shape params
        "size": (100,),
    },
    "GABAa": {
        # GABAa es
        "conn": bp.connect.All2All(include_self = False),
    },
}

def main(dt = 0.01):
    # init seed & expr_curr
    np.random.seed(0)
    expr_curr = "white"
    # init backend
    bp.backend.set(dt = dt)
    bp.backend.set(backend = "numba")

    ## exec sim
    # set stim
    stim = 1.
    # inst GammaOsciNet
    net = model.GammaOsciNet(net_params = default_net_params, run_params = {
        "inputs": stim,
        "dt": 0.01,
        "duration": default_stim_params[expr_curr]["duration"],
    })
    # net run
    net.run(report = True)
    # show net.mon
    net_monitors = net.get_monitors()
    net.show(img_fname = os.path.join(DIR_FIGS, expr_curr + ".png"))

if __name__ == "__main__":
    main()

