"""
Created on 00:34, June. 13rd, 2021
Author: fassial
Filename: expr2.py
Description:
    Changes in spatial patterning (Ω), temporal patterning (cv) and
    cross correlation patterning (cor) as a function of noise and
    range of stimulus.
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
    "one_hole": stimulus.stim_params(
        name = "one_hole",
        height = 50,
        width = 50,
        duration = 100,
        others = {
            "radius": 5,
            "position": "center",
            "noise": 0.,
        }
    ),
    "two_holes": stimulus.stim_params(
        name = "two_holes",
        height = 50,
        width = 50,
        duration = 100,
        others = {
            "radius_left": 5,
            "radius_right": 5,
            "interval": 15,
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

def expr(noise, range_, run_params, dt = 0.01):
    print("processing expr(" +\
        str(noise) + "," + str(range_) + ")..."
    )

    # init seed
    np.random.seed(0)

    # init expr_curr & stim
    expr_curr = run_params["expr_curr"]
    stim = run_params["stim"]; stim = stim * range_ + (1 - range_) / 2

    ## prepare expr
    # init net_params
    net_params = deepcopy(default_net_params)
    net_params["neurons"]["noise"] = noise; print(net_params)

    ## exec expr
    # inst GJ2DNet
    net = model.GJ2DNet(net_params = net_params, run_params = {
        "inputs": stim,
        "dt": dt,
        "duration": default_stim_params[expr_curr].duration,
    })
    # net run
    net.run(report = True)
    # show net.mon
    net_monitors = net.get_monitors()
    net.show(img_fname = os.path.join(DIR_FIGS,
        expr_curr + "-" +\
        str(noise) + "-" + str(range_) + ".png"
    ))
    net.save(spike_fname = os.path.join(DIR_OUTPUTS_SPIKE,
        expr_curr + "-" +\
        str(noise) + "-" + str(range_) + ".csv"
    ))

    ## compute statistic-values
    # compute cor
    cor = bp.measure.cross_correlation(
        spikes = net_monitors.spike,
        bin = 10,
        dt = dt
    )
    # compute omega & cv
    spike = net_monitors.spike.T; print(spike.shape)
    omega = utils.get_omega(
        spike = spike,
        bin = 100,
        dt = dt,
        N = 100
    )
    cv = np.mean(utils.get_cv(
        spike = spike,
        dt = dt
    ))

    # rm vars
    del(net_params); del(net); del(net_monitors); del(spike); gc.collect()

    return (omega, cv, cor)

def main(dt = 0.01):
    # init omegas & cvs & cors
    omegas = []; cvs = []; cors = []

    # init gj_ws & gj_ks
    noises = [.05, .1, .15, .2, .25, .3, .35, .4, .45, .5]
    ranges_ = [.1, .3, .5, .7, 1.]

    ## init backend
    bp.backend.set(dt = dt)
    bp.backend.set(backend = "numba")
    ## init expr_curr
    expr_curr = "two_holes"
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
    stim *= 1.; print(stim.shape)

    # set omegas
    for noise in noises:
        omega = []; cv = []; cor = []
        for range_ in ranges_:
            res = expr(
                noise = noise,
                range_ = range_,
                run_params = {
                    "expr_curr": expr_curr,
                    "stim": stim,
                },
                dt = dt
            ); omega.append(res[0]); cv.append(res[1]); cor.append(res[2])
        omegas.append(omega); cvs.append(cv); cors.append(cor)
    omegas = np.array(omegas); cvs = np.array(cvs); cors = np.array(cors)

    # save omegas & cvs & cors
    np.savetxt(
        fname = os.path.join(DIR_OUTPUTS, "omegas.csv"),
        X = omegas,
        delimiter = ","
    )
    np.savetxt(
        fname = os.path.join(DIR_OUTPUTS, "cvs.csv"),
        X = cvs,
        delimiter = ","
    )
    np.savetxt(
        fname = os.path.join(DIR_OUTPUTS, "cors.csv"),
        X = cors,
        delimiter = ","
    )

if __name__ == "__main__":
    main()

