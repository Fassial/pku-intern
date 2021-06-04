"""
Created on 18:53, Apr. 22nd, 2021
Author: fassial
Filename: __init__.py
"""

# neurons model
from . import neurons

# synapses model
from . import synapses

# connector model
from . import connector

# network model
from .RPNet import *

# def set_backend func
def set_backend(backend):
    neurons.set_backend(backend = backend)
    synapses.set_backend(backend = backend)

