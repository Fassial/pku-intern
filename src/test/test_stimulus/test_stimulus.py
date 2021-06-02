"""
Created on 12:11, May. 23rd, 2021
Author: fassial
Filename: test_stimulus.py
"""
import numpy as np
# local dep
import stimulus

__all__ = [
    "test_stimulus",
    "test_stimulus_normal",
    "test_stimulus_frate_increase",
]

# macro
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
    "frate_increase": stimulus.stim_params(
        name = "frate_increase",
        height = 200,
        width = 1,
        duration = 2000,
        others = {
            "freqs": np.full((200,), 20., dtype = np.float32),
            "factor": 4.,    # (1,16)
            "ratio": .2,
            "noise": 0.,
        }
    ),
}

## define test func
# define test_stimulus func
def test_stimulus():
    pass

# define test_stimulus_normal func
def test_stimulus_normal():
    stimulus.stimulus.show(default_stim_params["normal"])

# define test_stimulus_frate_increase func
def test_stimulus_frate_increase():
    stimulus.stimulus.show(default_stim_params["frate_increase"])

