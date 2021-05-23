"""
Created on 12:08, May. 23rd, 2021
Author: fassial
Filename: test_inputs.py
"""
# local dep
import stimulus

__all__ = [
    "test_poisson_input",
]

## define test func
# define test_poisson_input func
def test_poisson_input():
    # get stim
    stim, _ = stimulus.inputs.poisson_input(
        duration = 100
    )
    # display stim
    print(stim.shape)   # (duration / dt, size)

