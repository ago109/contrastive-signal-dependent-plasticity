#from ngclib.component import Component
from jax import numpy as jnp, random, jit
#from functools import partial
import time, sys

@jit
def add(x, y):
    return x + y

def fast_add(component, value, destination_compartment):
    curr_val = component.compartments[destination_compartment]
    new_val = add(curr_val, value)
    component.compartments[destination_compartment] = new_val
