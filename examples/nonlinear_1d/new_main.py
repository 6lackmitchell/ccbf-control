"""
This module provides a template for example simulations designed to be run from
the interpreter via ''python examples/template.py''.

It does not define any new functions, and primarily loads modules from the
src/cbfkit tree.

"""
from typing import List, Callable, Tuple
from jax import jit, Array, jacfwd, jacrev
import jax.numpy as jnp

# Load CBFkit dependencies
import cbfkit.simulation.simulator as sim
from cbfkit.sensors import perfect as sensor
from cbfkit.estimators.naive import naive as estimator
from cbfkit.utils.numerical_integration import forward_euler as integrator
from cbfkit.optimization.dynamically_defined.optimization_interior_point import (
    generate_predictor_corrector_dynamical_solution,
)
from cbfkit.utils.user_types import DynamicsCallable
from cbfkit.utils.matrix_vector_operations import invert_array
from staliro.models import blackbox
from staliro.core import Trace, BasicResult


# Load setup and barrier modules
from examples.nonlinear_1d import setup

# Load dynamics modules
import systems.nonlinear_1d.black2023consolidated as system

N_STATES = len(setup.INITIAL_STATE)
N_CONTROLS = len(setup.ACTUATION_LIMITS)
N_CBFS = len(setup.cbfs)
S_VAL = 10.0
EPS = 1e-1  # for numerical stability

IDX_X = jnp.arange(0, N_STATES)
IDX_W = jnp.arange(N_STATES, N_STATES + N_CBFS)
IDX_WDOT = jnp.arange(N_STATES + N_CBFS, N_STATES + 2 * N_CBFS)
IDX_U = jnp.arange(N_STATES + 2 * N_CBFS, N_STATES + 2 * N_CBFS + N_CONTROLS)
IDX_T = jnp.arange(
    N_STATES + 2 * N_CBFS + N_CONTROLS, N_STATES + 2 * N_CBFS + N_CONTROLS + 1
)

x_dynamics = system.plant(setup.K_DYNAMICS)
nominal_controller = setup.nominal_controller


# Define augmented system dynamics (autonomous system)
def generate_augmented_dynamics(
    n_states: int,
    list_of_dynamics: List[DynamicsCallable],
) -> Callable[[Array], Array]:
    """_summary_

    Args:
        n_states (int): _description_
        list_of_dynamics (List[DynamicsCallable]): _description_

    Returns:
        Callable[[Array], Array]: _description_
    """

    @jit
    def dynamics(state: Array) -> Array:
        """_summary_

        Args:
            z (Array): _description_

        Returns:
            Array: _description_
        """
        f, g, _ = list_of_dynamics[0](state)
        for dynamics in list_of_dynamics[1:]:
            fd, gd, _ = dynamics(state)
            f = jnp.hstack([f, fd])
            g = jnp.vstack([g, gd])

        return f, g, None

    return dynamics


def generate_compute_w2dot_udot(
    augmented_cost_wdot,
    augmented_cost_u,
    dynamics,
):
    jacobian_wdot = jacfwd(augmented_cost_wdot)
    hessian_wdot = jacrev(jacfwd(augmented_cost_wdot))
    jacobian_u = jacfwd(augmented_cost_u)
    hessian_u = jacrev(jacfwd(augmented_cost_u))

    def generate(pd_matrix):
        def compute_w2dot_udot(z: Array) -> Tuple[Array, Array]:
            f, g, _ = dynamics(z)
            grad_wdot = jacobian_wdot(z)
            hess_wdot = hessian_wdot(z)
            grad_u = jacobian_u(z)
            hess_u = hessian_u(z)

            w2dot_f, w2dot_g, w2dot_s = compute_w2dot(
                z, f, g, grad_wdot, hess_wdot, grad_u, hess_u
            )
            udot_f, udot_g, udot_s = compute_udot(
                z, f, g, grad_wdot, hess_wdot, grad_u, hess_u
            )

            return (
                jnp.hstack([w2dot_f, udot_f]),
                jnp.vstack([w2dot_g, udot_g]),
                jnp.vstack([w2dot_s, udot_s]),
            )

        def compute_w2dot(z, f, g, grad_wdot, hess_wdot, grad_u, hess_u) -> Array:
            """_summary_

            Args:
                z (Array): _description_

            Returns:
                Array: _description_
            """

            term1 = -hess_wdot[IDX_WDOT, IDX_WDOT]
            term2a = jnp.array(
                [
                    jnp.matmul(pd_matrix[IDX_U, IDX_U], grad_u[IDX_U])
                    + jnp.matmul(
                        hess_u[IDX_U, IDX_X], f[IDX_X] + jnp.matmul(g[IDX_X], z[IDX_U])
                    )
                    + jnp.matmul(hess_u[IDX_U, IDX_W], z[IDX_WDOT])
                ]
            )
            term2 = jnp.array(
                [
                    jnp.matmul(pd_matrix[IDX_WDOT, IDX_WDOT], grad_wdot[IDX_WDOT])
                    + jnp.matmul(
                        hess_wdot[IDX_WDOT, IDX_X],
                        f[IDX_X] + jnp.matmul(g[IDX_X], z[IDX_U]),
                    )
                    + jnp.matmul(hess_wdot[IDX_WDOT, IDX_W], z[IDX_WDOT])
                    - jnp.matmul(
                        hess_wdot[IDX_WDOT, IDX_U],
                        jnp.matmul(invert_array(hess_u[IDX_U, IDX_U]), term2a),
                    )
                ]
            )
            term3 = jnp.eye(N_CBFS) - jnp.matmul(
                jnp.matmul(
                    invert_array(hess_wdot[IDX_WDOT, IDX_WDOT]),
                    hess_wdot[IDX_WDOT, IDX_U],
                ),
                jnp.matmul(invert_array(hess_u[IDX_U, IDX_U]), hess_u[IDX_U, IDX_WDOT]),
            )

            return (
                jnp.matmul(invert_array(term1), jnp.matmul(term2, invert_array(term3))),
                jnp.zeros((len(IDX_WDOT), len(IDX_U))),
                jnp.zeros((len(IDX_WDOT), N_STATES)),
            )

        def compute_udot(z, f, g, grad_wdot, hess_wdot, grad_u, hess_u) -> Array:
            """_summary_

            Args:
                z (Array): _description_

            Returns:
                Array: _description_
            """
            term1 = -hess_u[IDX_U, IDX_U]
            term2a = jnp.array(
                [
                    jnp.matmul(pd_matrix[IDX_WDOT, IDX_WDOT], grad_wdot[IDX_WDOT])
                    + jnp.matmul(
                        hess_wdot[IDX_WDOT, IDX_X],
                        f[IDX_X] + jnp.matmul(g[IDX_X], z[IDX_U]),
                    )
                    + jnp.matmul(hess_wdot[IDX_WDOT, IDX_W], z[IDX_WDOT])
                ]
            )
            term2 = jnp.array(
                [
                    jnp.matmul(pd_matrix[IDX_U, IDX_U], grad_u[IDX_U])
                    + jnp.matmul(
                        hess_u[IDX_U, IDX_X], f[IDX_X] + jnp.matmul(g[IDX_X], z[IDX_U])
                    )
                    + jnp.matmul(hess_u[IDX_U, IDX_W], z[IDX_WDOT])
                    - jnp.matmul(
                        hess_u[IDX_U, IDX_WDOT],
                        jnp.matmul(invert_array(hess_wdot[IDX_WDOT, IDX_WDOT]), term2a),
                    )
                ]
            )
            term3 = jnp.eye(N_CBFS) - jnp.matmul(
                jnp.matmul(
                    invert_array(hess_u[IDX_U, IDX_U]),
                    hess_u[IDX_U, IDX_WDOT],
                ),
                jnp.matmul(
                    invert_array(hess_wdot[IDX_WDOT, IDX_WDOT]),
                    hess_wdot[IDX_WDOT, IDX_U],
                ),
            )

            return (
                jnp.matmul(invert_array(term1), jnp.matmul(term2, invert_array(term3))),
                jnp.zeros((len(IDX_U), len(IDX_U))),
                jnp.zeros((len(IDX_WDOT), N_STATES)),
            )

        return compute_w2dot_udot

    return generate


def generate_augmented_cost(cost, constraints, scale):
    def compute_augmented_cost(z):
        return cost(z) + 1 / scale * jnp.sum(
            jnp.array([jnp.log(constraint(z)) for constraint in constraints])
        )

    return compute_augmented_cost


def cost_wdot(z):
    w0 = 1.0
    wdot_des = -1.0 * (z[IDX_W] - w0)
    return 0.5 * jnp.linalg.norm(wdot_des - z[IDX_WDOT] + EPS)


def cost_u(z):
    u_nom, _ = nominal_controller(z[IDX_T], z)
    return 0.5 * jnp.linalg.norm(u_nom - z[IDX_U] + EPS)


w_constraint_funcs = [
    lambda z: -1 * (z[IDX_WDOT] + (z[IDX_W] - setup.W_MIN)),
    lambda z: -1 * (z[IDX_WDOT] + (setup.W_MAX - z[IDX_W])),
] + [
    lambda z: jnp.array(
        [
            jnp.matmul(
                cbf_grad(z)[IDX_X],
                x_dynamics(z)[0] + jnp.matmul(x_dynamics(z)[1], z[IDX_U]),
            )
            + jnp.matmul(cbf_grad(z)[IDX_X], z[IDX_WDOT])
            + setup.LINEAR_CLASS_K * cbf(z)
            for cbf, cbf_grad in zip(setup.cbfs, setup.cbf_grads)
        ]
    )
]

u_constraint_funcs = [
    lambda z: -1 * (z[IDX_U] + setup.ACTUATION_LIMITS),
    lambda z: -1 * (setup.ACTUATION_LIMITS - z[IDX_U]),
] + [
    lambda z: jnp.array(
        [
            jnp.matmul(
                cbf_grad(z)[IDX_X],
                x_dynamics(z)[0] + jnp.matmul(x_dynamics(z)[1], z[IDX_U]),
            )
            + jnp.matmul(cbf_grad(z)[IDX_X], z[IDX_WDOT])
            + setup.LINEAR_CLASS_K * cbf(z)
            for cbf, cbf_grad in zip(setup.cbfs, setup.cbf_grads)
        ]
    )
]


# Generate dynamics functions
w_dynamics = lambda z: (
    z[N_STATES + N_CBFS : N_STATES + 2 * N_CBFS],
    jnp.zeros((N_CBFS, N_CONTROLS)),
    jnp.zeros((N_CBFS, N_STATES)),
)
wdot_and_u_dynamics = generate_compute_w2dot_udot(
    generate_augmented_cost(cost_wdot, w_constraint_funcs, S_VAL),
    generate_augmented_cost(cost_u, u_constraint_funcs, S_VAL),
    generate_augmented_dynamics(N_STATES, [x_dynamics, w_dynamics]),
)
t_dynamics = lambda z: (
    jnp.array([1.0]),
    jnp.zeros((1, N_CONTROLS)),
    jnp.zeros((1, N_STATES)),
)


# Augmented System
augmented_initial_state = jnp.hstack(
    [
        setup.INITIAL_STATE,
        setup.W0 * jnp.ones((N_CBFS,)),
        jnp.zeros((N_CBFS,)),
        jnp.zeros((N_CONTROLS,)),
        jnp.zeros((1,)),
    ]
)
modified_x_dynamics = lambda s: (
    x_dynamics(s)[0] + jnp.matmul(x_dynamics(s)[1], s[IDX_U]),
    jnp.zeros((N_STATES, N_CONTROLS)),
    jnp.zeros((N_STATES, N_STATES)),
)
augmented_dynamics = lambda params: generate_augmented_dynamics(
    N_STATES, [modified_x_dynamics, w_dynamics, wdot_and_u_dynamics(params), t_dynamics]
)

import numpy as np


def execute(params: Array) -> List[Array]:
    """_summary_

    Args:
        int (ii): _description_

    Returns:
        List[Array]: _description_
    """
    x, u, z, p, dkeys, dvalues = sim.execute(
        x0=augmented_initial_state,
        dynamics=augmented_dynamics(jnp.diag(jnp.array(params))),
        sensor=sensor,
        estimator=estimator,
        integrator=integrator,
        dt=setup.DT,
        num_steps=setup.N_STEPS,
    )

    # Reformat results as numpy arrays
    x = np.array(x)
    u = np.array(u)
    z = np.array(z)
    p = np.array(p)

    return x, u, z, p, dkeys, dvalues


# @blackbox
def execute_for_staliro(
    params,
):
    states, _, _, _, _, _ = execute(params)
    times = jnp.arange(setup.T0, setup.TF, setup.DT)
    trace = Trace(times, list(states))

    return BasicResult(trace)


# state, control, estimate, covariance, data_keys, data_values = execute_for_staliro(
#     [1, 1]
# )

import matplotlib.pyplot as plt

# Create subplots
fig1, axs1 = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
time = np.linspace(setup.T0, setup.TF, setup.N_STEPS)

# Plot signal1 on the first subplot
axs1[0].plot(time, state[:, IDX_X], color="blue", label="X")
axs1[0].set_title("State")
axs1[0].set(ylim=[-2.5, 2.5])
axs1[0].legend()

# Plot signal2 on the second subplot
axs1[1].plot(time, state[:, IDX_W], color="green", label="W")
axs1[1].set_title("Weight")
axs1[1].set(ylim=[0.0, 5.0])
axs1[1].legend()

# Plot signal3 on the third subplot
axs1[2].plot(time, state[:, IDX_U], color="black", label="U")
axs1[2].set_title("Control")
axs1[2].set(ylim=[-1.25, 1.25])
axs1[2].legend()

# Add labels to the x-axis and y-axis for the last subplot
axs1[2].set_xlabel("Time")
axs1[2].set_ylabel("Amplitude")

# Adjust layout for better spacing
plt.tight_layout()

# Create subplots
fig2, axs2 = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
time = np.linspace(setup.T0, setup.TF, setup.N_STEPS)

# Plot signal1 on the first subplot
axs2[0].plot(time, [w_constraint_funcs[0](s) for s in state], color="blue", label="C1")
axs2[0].plot(time, [w_constraint_funcs[1](s) for s in state], color="green", label="C2")
# axs2[0].plot(time, [w_constraint_funcs[2](s) for s in state], color="black", label="C3")
axs2[0].set_title("WDOT Constraints")
axs2[0].legend()

# Plot signal2 on the second subplot
axs2[1].plot(time, [u_constraint_funcs[0](s) for s in state], color="blue", label="C1")
axs2[1].plot(time, [u_constraint_funcs[1](s) for s in state], color="green", label="C2")
axs2[1].plot(time, [u_constraint_funcs[2](s) for s in state], color="black", label="C3")
axs2[1].set_title("UDOT Constraints")
axs2[1].legend()

# Plot signal3 on the third subplot
axs2[2].plot(time, state[:, IDX_U], color="black", label="U")
axs2[2].set_title("Control")
axs2[2].legend()

# Add labels to the x-axis and y-axis for the last subplot
axs2[2].set_xlabel("Time")
axs2[2].set_ylabel("Amplitude")

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
