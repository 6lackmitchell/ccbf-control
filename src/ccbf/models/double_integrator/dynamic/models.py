import symengine as se
import numpy as np
import builtins
from core.dynamics_wrappers import (
    dyn_wrapper,
    control_affine_system_deterministic,
    control_affine_system_stochastic,
    first_order_forward_euler,
)

vehicle = builtins.PROBLEM_CONFIG["vehicle"]
control_level = builtins.PROBLEM_CONFIG["control_level"]
situation = builtins.PROBLEM_CONFIG["situation"]
mod = "models." + vehicle + "." + control_level + "." + situation
dt = getattr(__import__(mod + ".timing_params", fromlist=["dt"]), "dt")

# Define Symbolic State
sym_state = se.symbols(["x", "y", "vx", "vy"], real=True)

# Define symbolic system dynamics
f_symbolic = se.DenseMatrix(
    [
        sym_state[2],
        sym_state[3],
        0.0,
        0.0,
    ]
)
g_symbolic = se.DenseMatrix([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

# Need to be fixed
s_symbolic_deterministic = se.Matrix(
    [[0 for i in range(len(sym_state))] for j in range(len(sym_state))]
)
s_symbolic_stochastic = (
    0.25
    * dt
    * se.Matrix(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
)

# Callable Functions
f = dyn_wrapper(f_symbolic, sym_state)
g = dyn_wrapper(g_symbolic, sym_state)
sigma_deterministic = dyn_wrapper(s_symbolic_deterministic, sym_state)
sigma_stochastic = dyn_wrapper(s_symbolic_stochastic, sym_state)

# Partial Derivative
dfdx_symbolic = (se.DenseMatrix([f_symbolic]).jacobian(se.DenseMatrix(sym_state))).T
dfdx = dyn_wrapper(dfdx_symbolic, sym_state)
# dgdx_symbolic = (se.DenseMatrix([g_symbolic]).jacobian(se.DenseMatrix(sym_state))).T
# dgdx = dyn_wrapper(dgdx_symbolic, sym_state)
dgdx = lambda x: np.zeros((4, 4, 2))

# System Dynamics
deterministic_dynamics = control_affine_system_deterministic(f, g)
stochastic_dynamics = control_affine_system_stochastic(f, g, sigma_stochastic, dt)

# Step Forward
deterministic_step = first_order_forward_euler(deterministic_dynamics, dt)
stochastic_step = first_order_forward_euler(stochastic_dynamics, dt)

# Determine dimensions
nControls = g(np.zeros((len(sym_state),))).shape[1]
u0 = np.zeros((nControls,))
