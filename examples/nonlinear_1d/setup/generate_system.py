"""
#! docstring
"""
from examples.nonlinear_1d.setup import KV, U0_PERIOD, K_DYNAMICS, X_LIMIT
from cbfkit.systems.create_new_system.generate_model import generate_model

model_name = "black2023consolidated"
drift_dynamics = "[x[0] * (exp(k*x[0]**2) -1)]"
control_matrix = "[[(4 - x[0]**2)]]"
barrier_functions = ["limit + x[0]", "limit - x[0]", "(limit**2 - x[0]**2)"]
nominal_controller = [
    "kv * (4 * sin(2 * pi * t / period) - x[0]) - 0.5 * x[0] / (4 - x[0]**2)"
]
params = {
    "dynamics": {"k": K_DYNAMICS},
    "cbf": [{"limit": X_LIMIT}, {"limit": X_LIMIT}, {"limit": X_LIMIT}],
    "controller": {"kv: float": KV, "period: float": U0_PERIOD},
}
generate_model(
    "src/systems/nonlinear_1d/",
    model_name=model_name,
    drift_dynamics=drift_dynamics,
    control_matrix=control_matrix,
    barrier_funcs=barrier_functions,
    nominal_controller=nominal_controller,
    params=params,
)
