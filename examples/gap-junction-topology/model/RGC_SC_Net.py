"""
Created on 20:35, Apr. 5th, 2021
Author: fassial
Filename: RGC_SC_Net.py
"""
import brainpy as bp

class RGC_SC_Net(bp):

    def __init__(self, net_params = {
        "RGC": {
            "seed": -1,
            # dynamic params
            V_reset = 0,
            V_th = 10,
            V_initial = "reset",
            tau = 5,
            tau_refractory = 3.5,
            noise_sigma = 0.5,
            noise_correlated_with_input = False,
            # local gap junction
            gj_w = 0.5,
            gj_spikelet = 0.15,
            gj_conn = "grid_eight",
        },
        "SC": {
            # dynamic params
            V_reset = 0,
            V_th = 10,
            V_initial = "random",
            tau = 5,
            tau_refractory = 0.5,
            noise_mean = 1.0,
            noise_sigma = 0.1,
            # params of conn between RGCs and RONs
            R2N_delay = 0.1,
            R2N_current = 1.0,
        }
    }):
        

if __name__ == "__main__":
    pass
