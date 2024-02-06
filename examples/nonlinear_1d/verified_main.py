"""
This module provides a template for example simulations designed to be run from
the interpreter via ''python examples/template.py''.

It does not define any new functions, and primarily loads modules from the
src/cbfkit tree.

"""
from typing import List, Callable, Tuple
from jax import jit, Array, jacfwd, jacrev
import jax.numpy as jnp
import logging
import plotly.graph_objects as go


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
from staliro.core import Trace, BasicResult, best_eval, best_run
from staliro.specifications import RTAMTDense
from staliro.options import Options
from staliro.optimizers import DualAnnealing, UniformRandom
from staliro.staliro import simulate_model, staliro


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


@blackbox
def execute_for_staliro(params, _signal_times, _signal_traces):
    states, _, _, _, _, _ = execute(params)
    times = jnp.arange(setup.T0, setup.TF + setup.DT, setup.DT)
    trace = Trace(times, list(states.T))

    return BasicResult(trace)


TUNE = True
if TUNE:
    initial_params = [(0, 5), (0, 5)]
    # phi = r"(always (a >= -2 and a <= 2))"
    phi = r"(always (c >= -100 and c <= 100 and d >= -100 and d <= 100))"
    specification = RTAMTDense(phi, {"a": 0, "b": 1, "c": 2, "d": 3})
    options = Options(
        runs=1,
        iterations=20,
        interval=(setup.T0, setup.TF),
        static_parameters=initial_params,
    )
    # optimizer = DualAnnealing()
    optimizer = UniformRandom()
    logging.basicConfig(level=logging.DEBUG)

    result = staliro(execute_for_staliro, specification, optimizer, options)

    best_run_ = best_run(result)
    best_sample = best_eval(best_run_).sample
    best_result = simulate_model(execute_for_staliro, options, best_sample)

    success_sample_pd_wdot = []
    success_sample_pd_u = []
    failure_sample_pd_wdot = []
    failure_sample_pd_u = []
    max_cost = 0
    min_cost = jnp.inf
    for evaluation in best_run_.history:
        if evaluation.cost > 0:
            if evaluation.cost > max_cost:
                best_x, best_y = evaluation.sample[0], evaluation.sample[1]
                max_cost = evaluation.cost
            if evaluation.cost < min_cost:
                worst_x, worst_y = evaluation.sample[0], evaluation.sample[1]
                min_cost = evaluation.cost
            success_sample_pd_wdot.append(evaluation.sample[0])
            success_sample_pd_u.append(evaluation.sample[1])
        else:
            failure_sample_pd_wdot.append(evaluation.sample[0])
            failure_sample_pd_u.append(evaluation.sample[1])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            name="Parameter Region",
            x=[0, 0, 5, 5, 0],
            y=[0, 5, 5, 0, 0],
            fill="toself",
            fillcolor="lightsteelblue",
            line_color="steelblue",
            mode="lines+markers",
        )
    )
    figure.add_trace(
        go.Scatter(
            name="Successes",
            x=success_sample_pd_wdot,
            y=success_sample_pd_u,
            mode="markers",
            marker=go.scatter.Marker(color="green", symbol="circle"),
        )
    )
    figure.add_trace(
        go.Scatter(
            name="Failures",
            x=failure_sample_pd_wdot,
            y=failure_sample_pd_u,
            mode="markers",
            marker=go.scatter.Marker(color="black", symbol="x"),
        )
    )
    figure.add_trace(
        go.Scatter(
            name="Most Conservative Result",
            x=[best_x],
            y=[best_y],
            mode="markers",
            marker=go.scatter.Marker(color="lemonchiffon", symbol="square"),
        )
    )
    figure.add_trace(
        go.Scatter(
            name="Most Aggressive Result",
            x=[worst_x],
            y=[worst_y],
            mode="markers",
            marker=go.scatter.Marker(color="darkcyan", symbol="star"),
        )
    )
    figure.update_layout(xaxis_title=r"Gain $\dot w$", yaxis_title=r"Gain $\dot u$")
    # figure.add_hline(y=0, line_color="red")
    # figure.write_image("examples/nonlinear_1d/nonlinear_1d.jpeg")
    figure.show()

    figure2 = go.Figure()
    figure2.add_trace(
        go.Scatter(
            x=best_result.trace.times,
            y=best_result.trace.states[0],
            mode="lines",
            line_color="green",
            name="x",
        )
    )
    figure2.update_layout(xaxis_title="time (s)", yaxis_title="x")
    figure2.add_hline(y=0, line_color="red")
    figure2.write_image("examples/nonlinear_1d/nonlinear_1d.jpeg")

    values = best_sample.values
else:
    values = [1.0, 1.0]

    #! To Do: download lyznet source and modify to be compatible with JAX

# # Verify with Lyapunov function
# dynamics_to_verify = augmented_dynamics(jnp.diag(jnp.array(values)))
# from examples.nonlinear_1d.setup.learn_lyapunov_nn import model, test_lnn

# mod = model(dynamics_to_verify)
# test_lnn(mod)
