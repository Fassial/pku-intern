"""
Created on 17:33, Apr. 6th, 2021
Author: fassial
Filename: main.py
"""
import brainpy as bp
# local dep
import model
import stimulus

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
        "tau": 5,
        "t_refractory": 3.5,
        "noise_sigma": 0.5,
        # local gap junction
        "gj_w": 0.5,
        "gj_spikelet": 0.15,
        "gj_conn": bp.connect.GridEight(include_self = False),
    },
    "SC": {
        "size": 1,
        # dynamic params
        "V_reset": 0,
        "V_th": 10,
        "tau": 5,
        "t_refractory": 0.5,
        "noise_sigma": 0.1,
        # params of conn between RGCs and RONs
        "R2N_w": 1.0,
        "R2N_delay": 0.1,
    }
}

def main():
    # init backend
    bp.backend.set(backend = "numpy")
    # init stimulus
    _, _, stim_rgc = stimulus.stimulus.get(
        stim_params = default_stim_params["circle"]
    )
    stim_sc = 1.0
    # inst RGC_SC_Net
    net = model.RGC_SC_Net(net_params = default_net_params, run_params = {
        "inputs": {
            "RGC": stim_rgc,
            "SC": stim_sc,
        },
        "dt": 0.01,
    })
    # net run & show
    net.run(duration = 20, report = True)
    net_monitors = net.get_monitors()

if __name__ == "__main__":
    main()
