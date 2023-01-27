import builtins
from sys import platform
from importlib import import_module
from core.agent import Agent
from core.cbfs import (
    cbfs_individual,
    cbfs_individual1,
    cbfs_individual2,
    cbfs_individual3,
    cbfs_individual4,
    cbfs_pairwise,
    cbf0,
)

# from core.controllers.cbf_qp_controller_breeden_hocbf import CbfQpController
from core.controllers.cbf_qp_controller_exponential_hocbf import CbfQpController
from core.controllers.consolidated_cbf_controller import ConsolidatedCbfController
from ..models import f, g, nControls
from .timing_params import *
from .physical_params import u_max
from .objective_functions import objective_accel_and_steering
from .nominal_controllers import ProportionalController
from .initial_conditions import *

if builtins.PROBLEM_CONFIG["system_model"] == "stochastic":
    from ..models import (
        sigma_stochastic as sigma,
        stochastic_dynamics as system_dynamics,
        stochastic_step as step_dynamics,
    )
else:
    from ..models import (
        sigma_deterministic as sigma,
        deterministic_dynamics as system_dynamics,
        deterministic_step as step_dynamics,
    )

# Configure parameters
nAgents = len(z0)
time = [dt, tf]

if platform == "linux" or platform == "linux2":
    # linux
    pre_path = "/home/6lackmitchell/"
elif platform == "darwin":
    # OS X
    pre_path = "/Users/mblack/"
elif platform == "win32":
    # Windows...
    pass

save_path = (
    pre_path + "Documents/git/ccbf-control/data/double_integrator/dynamic/toy_example/test.pkl"
)


# Define controllers
def consolidated_cbf_controller(idx: int) -> ConsolidatedCbfController:
    return ConsolidatedCbfController(
        u_max,
        nAgents,
        objective_accel_and_steering,
        ProportionalController(idx),
        cbfs_individual,
        cbfs_pairwise,
    )


def cbf_controller1(idx: int) -> CbfQpController:
    return CbfQpController(
        u_max,
        nAgents,
        objective_accel_and_steering,
        ProportionalController(idx),
        cbfs_individual1,
        cbfs_pairwise,
    )


def cbf_controller2(idx: int) -> CbfQpController:
    return CbfQpController(
        u_max,
        nAgents,
        objective_accel_and_steering,
        ProportionalController(idx),
        cbfs_individual2,
        cbfs_pairwise,
    )


def cbf_controller3(idx: int) -> CbfQpController:
    return CbfQpController(
        u_max,
        nAgents,
        objective_accel_and_steering,
        ProportionalController(idx),
        cbfs_individual3,
        cbfs_pairwise,
    )


def cbf_controller4(idx: int) -> CbfQpController:
    return CbfQpController(
        u_max,
        nAgents,
        objective_accel_and_steering,
        ProportionalController(idx),
        cbfs_individual4,
        cbfs_pairwise,
    )


# Define CBF Controlled Agents
cbf_controlled_agents = [
    Agent(0, z0[0, :], u0, cbf0, time, step_dynamics, consolidated_cbf_controller(0), save_path),
    Agent(1, z0[1, :], u0, cbf0, time, step_dynamics, ProportionalController(1), save_path),
    Agent(2, z0[2, :], u0, cbf0, time, step_dynamics, cbf_controller1(2), save_path),
    Agent(3, z0[3, :], u0, cbf0, time, step_dynamics, cbf_controller2(3), save_path),
    Agent(4, z0[4, :], u0, cbf0, time, step_dynamics, cbf_controller3(4), save_path),
    Agent(5, z0[5, :], u0, cbf0, time, step_dynamics, cbf_controller4(5), save_path),
]
# human_agents = [
#     Agent(i, z0[i, :], u0, cbf0, time, step_dynamics, ZeroController(i), save_path)
#     for i in range(3, 9)
# ]


centralized_agents = None
decentralized_agents = cbf_controlled_agents  # + human_agents
