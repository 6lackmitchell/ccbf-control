"""
#! docstring
"""
import jax.numpy as jnp
from .barriers import cbfs, cbf_grads
from systems.nonlinear_1d.black2023consolidated.controllers.controller_1 import (
    controller_1,
)

# Time params
T0 = 0.0
TF = 10.0
DT = 1e-2
N_STEPS = int((TF - T0) / DT) + 1

# State and control params
INITIAL_STATE = jnp.array([0.0])
ACTUATION_LIMITS = jnp.array([1.0])
LINEAR_CLASS_K = 1.0

# Adaptation Settings
W0 = 1.0
W_MIN = 0.01
W_MAX = 5.0

# Params
KV = 2.0
U0_PERIOD = 5.0
K_DYNAMICS = 0.14
X_LIMIT = 2.0

# CBFs and Gradients
cbfs = [cbf(X_LIMIT) for cbf in cbfs]
cbf_grads = [grad(X_LIMIT) for grad in cbf_grads]

# Nominal Controller
nominal_controller = controller_1(KV, U0_PERIOD)
