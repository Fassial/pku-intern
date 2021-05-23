"""
Created on 21:39, May. 23rd, 2021
Author: fassial
Filename: test_utils.py
"""
import numpy as np
# local dep
import utils

__all__ = [
    "test_get_omega",
    "test_get_cv",
]

# macro
spike = np.array([
    [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    [1, 1, 1, 1, 0, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0]
])

## define test func
# define test_get_omega func
def test_get_omega():
    omega = utils.get_omega(
        spike = spike,
        dt = 1000,  # 0.01
    ); print(omega)

# define test_get_cv func
def test_get_cv():
    cv = utils.get_cv(
        spike = spike,
        dt = 1000,  # 0.01
    ); print(cv)

