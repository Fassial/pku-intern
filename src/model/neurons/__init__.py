"""
Created on 14:28, June. 4th, 2021
Author: fassial
Filename: __init__.py
"""

# numba_backend model
from . import numba_backend

# tensor_backend model
from . import tensor_backend
from .tensor_backend import LIF

# def set_backend func
def set_backend(backend):
    global LIF

    if backend in ['tensor', 'numpy', 'pytorch', 'tensorflow', 'jax']:
        LIF = tensor_backend.LIF

    elif backend in ['numba', 'numba-parallel', 'numba-cuda']:
        LIF = numba_backend.LIF

    else:
        raise ValueError(f'Unknown backend "{backend}".')

