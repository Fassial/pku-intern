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
]

# macro
default_stim_params = {
    "normal": stimulus.stim_params(
        name = "normal",
        height = 100,
        width = 1,
        duration = 1000,
        others = {
            "freqs": np.full((100,), 20., dtype = np.float32),
            "noise": 0.,
        }
    ),
}

## define test func
# define test_stimulus func
def test_stimulus():
    stimulus.stimulus.show(default_stim_params["normal"])

