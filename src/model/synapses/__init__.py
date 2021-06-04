"""
Created on 14:04, June. 4th, 2021
Author: fassial
Filename: __init__.py
"""

# numba_backend model
from . import numba_backend

# tensor_backend model
from . import tensor_backend
from .tensor_backend import AlphaSyn
from .tensor_backend import ExpSyn
from .tensor_backend import GapJunction
from .tensor_backend import GapJunction_LIF
from .tensor_backend import TwoExpSyn
from .tensor_backend import VoltageJump

# def set_backend func
def set_backend(backend):
    global AlphaSyn
    global ExpSyn
    global GapJunction
    global GapJunction_LIF
    global TwoExpSyn
    global VoltageJump

    if backend in ['tensor', 'numpy', 'pytorch', 'tensorflow', 'jax']:
        AlphaSyn = tensor_backend.AlphaSyn
        ExpSyn = tensor_backend.ExpSyn
        GapJunction = tensor_backend.GapJunction
        GapJunction_LIF = tensor_backend.GapJunction_LIF
        TwoExpSyn = tensor_backend.TwoExpSyn
        VoltageJump = tensor_backend.VoltageJump

    elif backend in ['numba', 'numba-parallel', 'numba-cuda']:
        AlphaSyn = numba_backend.AlphaSyn
        ExpSyn = numba_backend.ExpSyn
        GapJunction = numba_backend.GapJunction
        GapJunction_LIF = numba_backend.GapJunction_LIF
        TwoExpSyn = numba_backend.TwoExpSyn
        VoltageJump = numba_backend.VoltageJump

    else:
        raise ValueError(f'Unknown backend "{backend}".')

