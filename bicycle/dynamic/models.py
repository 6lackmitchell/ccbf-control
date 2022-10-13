import symengine as se
import numpy as np
import builtins
from core.dynamics_wrappers import dyn_wrapper, control_affine_system_deterministic, control_affine_system_stochastic, \
    first_order_forward_euler

vehicle = builtins.PROBLEM_CONFIG['vehicle']
control_level = builtins.PROBLEM_CONFIG['control_level']
situation = builtins.PROBLEM_CONFIG['situation']
mod = vehicle + '.' + control_level + '.' + situation
ar_max = getattr(__import__(mod + '.physical_params', fromlist=['ar_max']), 'ar_max')
w_max = getattr(__import__(mod + '.physical_params', fromlist=['w_max']), 'w_max')
Lr = getattr(__import__(mod + '.physical_params', fromlist=['Lr']), 'Lr')
dt = getattr(__import__(mod + '.timing_params', fromlist=['dt']), 'dt')

# from .physical_params import ar_max, w_max
# from .physical_params import Lr
# from .timing_params import dt



# Define Symbolic State
sym_state = se.symbols(['x', 'y', 'psi', 'vr', 'beta'], real=True)

# Define symbolic system dynamics
f_symbolic = se.DenseMatrix([sym_state[3] * (se.cos(sym_state[2]) - se.sin(sym_state[2]) * se.tan(sym_state[4])),
                             sym_state[3] * (se.sin(sym_state[2]) + se.cos(sym_state[2]) * se.tan(sym_state[4])),
                             sym_state[3] * se.tan(sym_state[4]) / Lr,
                             0.0,
                             0.0])
g_symbolic = se.DenseMatrix([[0.0, 0.0],
                             [0.0, 0.0],
                             [0.0, 0.0],
                             [0.0, 1.0],
                             [1.0, 0.0]])

# Need to be fixed
ar_max = 1
w_max = 1
s_symbolic_deterministic = se.Matrix([[0 for i in range(5)] for j in range(5)])
s_symbolic_stochastic = 0.25 * dt * se.Matrix([[0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0],
                                               [0, 0, 0, ar_max, 0],
                                               [0, 0, 0, 0, w_max]])

# Callable Functions
f = dyn_wrapper(f_symbolic, sym_state)
g = dyn_wrapper(g_symbolic, sym_state)
sigma_deterministic = dyn_wrapper(s_symbolic_deterministic, sym_state)
sigma_stochastic = dyn_wrapper(s_symbolic_stochastic, sym_state)

# System Dynamics
deterministic_dynamics = control_affine_system_deterministic(f, g)
stochastic_dynamics = control_affine_system_stochastic(f, g, sigma_stochastic, dt)

# Step Forward
deterministic_step = first_order_forward_euler(deterministic_dynamics, dt)
stochastic_step = first_order_forward_euler(stochastic_dynamics, dt)

# Determine dimensions
nControls = g(np.zeros((len(sym_state),))).shape[1]
u0 = np.zeros((nControls,))