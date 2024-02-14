"""
#! docstring
"""

import jax.numpy as jnp
from .barriers import barriers
from ccbf.systems.nonlinear_1d.models.black2024consolidated.controllers.controller_1 import (
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
S_VAL = 10.0
EPS = 1e-1

# Params
KV = 2.0
U0_PERIOD = 5.0
K_DYNAMICS = 0.14
X_LIMIT = 2.0

# CBFs and Gradients
cbfs = barriers(X_LIMIT, alpha=1.0, idxs=[0])
cbf_func, cbf_grad, cbf_hess, cbf_part, cbf_cond = cbfs

# Nominal Controller
nominal_controller = controller_1(KV, U0_PERIOD)

# Dimensions
N_STATES = len(INITIAL_STATE)
N_CONTROLS = len(ACTUATION_LIMITS)
N_CBFS = len(cbf_func)

# Define indices for augmented state vector
IDX_X = jnp.arange(0, N_STATES)
IDX_W = jnp.arange(N_STATES, N_STATES + N_CBFS)
IDX_WDOT = jnp.arange(N_STATES + N_CBFS, N_STATES + 2 * N_CBFS)
IDX_U = jnp.arange(N_STATES + 2 * N_CBFS, N_STATES + 2 * N_CBFS + N_CONTROLS)
IDX_T = jnp.arange(
    N_STATES + 2 * N_CBFS + N_CONTROLS, N_STATES + 2 * N_CBFS + N_CONTROLS + 1
)

AUGMENTED_INITIAL_STATE = jnp.hstack(
    [
        INITIAL_STATE,
        W0 * jnp.ones((N_CBFS,)),
        jnp.zeros((N_CBFS,)),
        jnp.zeros((N_CONTROLS,)),
        jnp.zeros((1,)),
    ]
)
