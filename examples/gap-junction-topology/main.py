"""
Created on 17:33, Apr. 6th, 2021
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
    "black": stimulus.stim_params(
        name = "black",
        height = 50,
        width = 50,
        intensity = [15,],
        others = None
    ),
    "circle": stimulus.stim_params(
        name = "circle",
        height = 95,
        width = 95,
        intensity = [12, 20],
        others = {
            "radius": 20,
        }
    ),
    "one_hole": stimulus.stim_params(
        name = "one_hole",
        height = 95,
        width = 95,
        intensity = [12, 20, 12],
        others = {
            "inner_radius": 9,
            "outer_radius": 28,
            "position": "center",
        }
    ),
    "two_holes": stimulus.stim_params(
        name = "two_holes",
        height = 95,
        width = 95,
        intensity = [12, 12, 12, 20],
        others = {
            "inner_radius": 9,
            "outer_radius": 28,
        }
    ),
}
# default net_params
default_net_params = {
    "RGC": {
        "size": (95, 95),
        # dynamic params
        "V_reset": 0,
        "V_th": 10,
        "V_init": "gaussian",
        "tau": 5,
        "t_refractory": 3.5,
        "noise_sigma": 1.7,
        # local gap junction
        "gj_w": 2.8,
        "gj_spikelet": 0.15,
        "gj_conn": bp.connect.GridEight(include_self = False),
    },
    "SC": {
        "size": 1,
        # dynamic params
        "V_reset": 0,
        "V_th": 10,
        "V_init": "reset",
        "tau": 1,
        "t_refractory": 0.35,
        "noise_sigma": 0.1,
        # params of conn between RGCs and RONs
        "R2N_w": 0.24,
        "R2N_delay": 0.1,
    }
}

def main():
    # init expr_curr
    expr_curr = "circle"
    # init backend
    bp.backend.set(backend = "numpy")
    # init stimulus
    _, _, stim_rgc = stimulus.stimulus.get(
        stim_params = default_stim_params[expr_curr]
    )
    stim_sc = 2
    # inst RGC_SC_Net
    net = model.RGC_SC_Net(net_params = default_net_params, run_params = {
        "inputs": {
            "RGC": stim_rgc,
            "SC": stim_sc,
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

