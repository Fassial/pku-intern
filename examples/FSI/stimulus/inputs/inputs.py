"""
Created on 22:00, May. 22nd, 2021
Author: fassial
Filename: inputs.py
"""
import brainpy as bp
# local dep
import model
from .PoissonInput import *
from .InputRecorder import *

__all__ = [
    "poisson_input",
]

## define input func
# define poisson_input func
def poisson_input(duration, net_params = {
    "neurons": {
        "size": (5,),
    },
    "synapses": {
        "weight": 2.,
        "delay": 0.,
        "tau1": .3,
        "tau2": 3.,
    }
}, others = {
    "freqs": 20,
}):
    ## init comps of network
    # inst PoissonInput
    pi_inst = PoissonInput(size = net_params["neurons"]["size"], freqs = others["freqs"], monitors = ["spike"])
    # inst InputRecorder
    ir_inst = InputRecorder(size = net_params["neurons"]["size"])
    # inst TwoExpSyn
    tes_inst = model.synapses.TwoExpSyn(
        pre = pi_inst,
        post = ir_inst,
        conn = bp.connect.One2One(),
        weight = net_params["synapses"]["weight"],
        delay = net_params["synapses"]["delay"],
        tau1 = net_params["synapses"]["tau1"],
        tau2 = net_params["synapses"]["tau2"]
    )
    # inst network
    net_inst = bp.Network(
        # neurons
        pi_inst, ir_inst,
        # synapses
        tes_inst
    )

    # run network
    net_inst.run(
        duration = duration,
        report = True,
        report_percent = 0.1
    )

    # get stim & spike
    stim = ir_inst.get_Iext()
    spike = pi_inst.mon.spike

    return stim, spike

