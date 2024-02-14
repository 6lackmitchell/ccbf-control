from typing import List, Callable, Tuple
from jax import jit, Array, jacfwd, jacrev
import jax.numpy as jnp
import numpy as np

from cbfkit.utils.user_types import DynamicsCallable
from cbfkit.codegen.create_new_system import generate_model

# Define dynamics
drift_dynamics = "[x[0] * (exp(k * x[0] ** 2) - 1)]"  # drift dynamics f(x)
control_matrix = "[[(4 - x[0] ** 2)]]"  # control matrix g(x)

# State constraints
constraint_funcs = [
    "limit**2 - x[0] ** 2",
    "limit - x[0]",
    "limit + x[0]",
]

# Nominal controller
nominal_control = (
    "kv * (4 * sin(2 * pi * t / period) - x[0]) - 0.5 * x[0] / (4 - x[0] ** 2)"
)

# Define dynamics parameters for code-gen
params = {
    "dynamics": {"k: float": 1.0},
    "controller": {"kv: float": 1.0, "period: float": 1.0},
    "cbf": 3 * [{"limit: float": 1.0}],
}

# Define target dir and name for code-gen
target_directory = "src/ccbf/systems/nonlinear_1d/models/"
model_name = "black2024consolidated"

# Generate new model code
generate_model.generate_model(
    directory=target_directory,
    model_name=model_name,
    drift_dynamics=drift_dynamics,
    control_matrix=control_matrix,
    barrier_funcs=constraint_funcs,
    nominal_controller=nominal_control,
    params=params,
)

from ccbf.systems.nonlinear_1d.models import black2024consolidated
from examples.nonlinear_1d import setup

controlled_x_dynamics = black2024consolidated.plant(k=setup.K_DYNAMICS)
modified_x_dynamics = lambda s: (
    controlled_x_dynamics(s)[0]
    + jnp.matmul(controlled_x_dynamics(s)[1], s[setup.IDX_U]),
    jnp.zeros((setup.N_STATES, setup.N_CONTROLS)),
)

from cbfkit.controllers.model_based.cbf_clf_controllers.dynamically_defined_consolidated_cbf_controller import (
    generate_compute_w2dot_udot,
)
from cbfkit.modeling.augmented_dynamics import (
    generate_w_dynamics,
    generate_augmented_dynamics,
)


# Define CBF constraint function
@jit
def cbf_constraint(z: Array) -> Array:
    return jnp.array(
        [
            jnp.matmul(
                cbf_grad(z[-1], z[:-1])[setup.IDX_X],
                controlled_x_dynamics(z)[0]
                + jnp.matmul(controlled_x_dynamics(z)[1], z[setup.IDX_U]),
            )
            + jnp.matmul(cbf_grad(z[-1], z[:-1])[setup.IDX_X], z[setup.IDX_WDOT])
            + setup.LINEAR_CLASS_K * cbf(z[-1], z[:-1])
            for cbf, cbf_grad in zip(setup.cbf_func, setup.cbf_grad)
        ]
    )


# Define pure adaptation weight dynamics
w_dynamics = generate_w_dynamics(setup.N_STATES, setup.N_CBFS, setup.N_CONTROLS)


# Define adaptation optimization problem
@jit
def cost_wdot(z):
    w0 = 1.0
    wdot_des = -1.0 * (z[setup.IDX_W] - w0)
    return 0.5 * jnp.linalg.norm(wdot_des - z[setup.IDX_WDOT] + setup.EPS)


#! Change setup.cbfs to imported from code-gen
@jit
def adaptation_lower_bound(z: Array) -> Array:
    return -1 * (z[setup.IDX_WDOT] + (z[setup.IDX_W] - setup.W_MIN))


@jit
def adaptation_upper_bound(z: Array) -> Array:
    return -1 * (z[setup.IDX_WDOT] + (setup.W_MAX - z[setup.IDX_W]))


w_constraint_funcs = [
    adaptation_lower_bound,
    adaptation_upper_bound,
    cbf_constraint,
]


# Define control optimization problem constraint functions
@jit
def cost_u(z):
    u_nom, _ = setup.nominal_controller(z[setup.IDX_T], z)
    return 0.5 * jnp.linalg.norm(u_nom - z[setup.IDX_U] + setup.EPS)


@jit
def actuation_lower_bound(z: Array) -> Array:
    return -1 * (z[setup.IDX_U] + setup.ACTUATION_LIMITS)


@jit
def actuation_upper_bound(z: Array) -> Array:
    return -1 * (setup.ACTUATION_LIMITS - z[setup.IDX_U])


u_constraint_funcs = [
    actuation_lower_bound,
    actuation_upper_bound,
    cbf_constraint,
]


# Define augmented cost functions
def generate_augmented_cost(
    cost: Callable[[Array], Array], constraints: Callable[[Array], Array], scale: float
) -> Callable[[Array], Array]:
    @jit
    def compute_augmented_cost(z: Array) -> Array:
        return cost(z) + 1 / scale * jnp.sum(
            jnp.array([jnp.log(constraint(z)) for constraint in constraints])
        )

    return compute_augmented_cost


adaptation_cost_function = generate_augmented_cost(
    cost_wdot, w_constraint_funcs, setup.S_VAL
)
control_cost_function = generate_augmented_cost(cost_u, u_constraint_funcs, setup.S_VAL)

# Define augmented dynamics for states and adaptation weights
x_and_w_augmented_dynamics = generate_augmented_dynamics(
    [controlled_x_dynamics, w_dynamics],
)

# Define interconnected adaptation and control dynamics
wdot_and_u_dynamics = generate_compute_w2dot_udot(
    adaptation_cost_function,
    control_cost_function,
    x_and_w_augmented_dynamics,
    setup.IDX_X,
    setup.IDX_W,
    setup.IDX_WDOT,
    setup.IDX_U,
)

from cbfkit.modeling.augmented_dynamics import generate_t_dynamics

t_dynamics = generate_t_dynamics(setup.N_CONTROLS)


def augmented_dynamics(params):
    return generate_augmented_dynamics(
        [modified_x_dynamics, w_dynamics, wdot_and_u_dynamics(params), t_dynamics]
    )


# Load CBFkit dependencies
import cbfkit.simulation.simulator as sim
from cbfkit.sensors import perfect as sensor
from cbfkit.estimators.naive import naive as estimator
from cbfkit.utils.numerical_integration import forward_euler as integrator

# from cbfkit.utils.matrix_vector_operations import invert_array
from staliro.models import blackbox
from staliro.core import Trace, BasicResult, best_eval, best_run
from staliro.specifications import RTAMTDense
from staliro.options import Options
from staliro.optimizers import UniformRandom
from staliro.staliro import simulate_model, staliro

# Execution function


def execute(params: Array) -> List[Array]:
    """_summary_

    Args:
        int (ii): _description_

    Returns:
        List[Array]: _description_
    """
    x, u, z, p, dkeys, dvalues = sim.execute(
        x0=setup.AUGMENTED_INITIAL_STATE,
        dynamics=augmented_dynamics(jnp.diag(jnp.array(params))),
        sensor=sensor,
        estimator=estimator,
        integrator=integrator,
        dt=setup.DT,
        num_steps=setup.N_STEPS,
    )

    return np.array(x), np.array(u), np.array(z), np.array(p), dkeys, dvalues


@blackbox
def execute_for_staliro(params, _signal_times, _signal_traces):
    states, _, _, _, _, _ = execute(params)
    times = jnp.arange(setup.T0, setup.TF + setup.DT, setup.DT)
    trace = Trace(times, list(states.T))

    return BasicResult(trace)


# This is where the robustness tool runs
import logging
import plotly.graph_objects as go

STABILITY_REGION = 1000

initial_params = [(0, 10), (0, 2)]
phi = rf"(always (c >= -{STABILITY_REGION} and c <= {STABILITY_REGION} and d >= -{STABILITY_REGION} and d <= {STABILITY_REGION}))"
specification = RTAMTDense(phi, {"a": 0, "b": 1, "c": 2, "d": 3})
options = Options(
    runs=1,
    iterations=500,
    interval=(setup.T0, setup.TF),
    static_parameters=initial_params,
)
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
        x=[0, 0, initial_params[0][1], initial_params[0][1], 0],
        y=[0, initial_params[1][1], initial_params[1][1], 0, 0],
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
figure.write_image("examples/nonlinear_1d/nonlinear_1d.jpeg")
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

#! To Do: download lyznet source and modify to be compatible with JAX

# # Verify with Lyapunov function
# dynamics_to_verify = augmented_dynamics(jnp.diag(jnp.array(values)))
# from examples.nonlinear_1d.setup.learn_lyapunov_nn import model, test_lnn

# mod = model(dynamics_to_verify)
# test_lnn(mod)
