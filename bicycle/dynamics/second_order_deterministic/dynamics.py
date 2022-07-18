from bicycle.dynamics.second_order_deterministic.physical_params import *
from bicycle.timing import dt
from dynamics_wrappers import dyn_wrapper, control_affine_system_deterministic, first_order_forward_euler
import sympy as sp

# Define Symbolic State
sym_state = sp.symbols(['x', 'y', 'psi', 'vr', 'beta'])

# Define symbolic system dynamics
# this is really where the dynamics are defined -- the functions f, g, sigma are just wrappers
f_symbolic = sp.Matrix([sym_state[3] * (sp.cos(sym_state[2]) - sp.sin(sym_state[2]) * sp.tan(sym_state[4])),
                        sym_state[3] * (sp.sin(sym_state[2]) + sp.cos(sym_state[2]) * sp.tan(sym_state[4])),
                        sym_state[3] * sp.tan(sym_state[4]) / Lr,
                        0.0,
                        0.0])
g_symbolic = sp.Matrix([[0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0]])
s_symbolic = sp.Matrix(np.eye(5)) * 3 * dt

f = dyn_wrapper(f_symbolic, sym_state)
g = dyn_wrapper(g_symbolic, sym_state)
sigma = dyn_wrapper(s_symbolic, sym_state)
system_dynamics = control_affine_system_deterministic(f, g)
step_dynamics = first_order_forward_euler(system_dynamics, dt)

# Determine dimensions
nControls = g(np.zeros((len(sym_state),))).shape[1]

# Initial parameters for agents
u0 = np.zeros((g_symbolic.shape[1],))
