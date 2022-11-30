import platform
import builtins
from importlib import import_module
from core.agent import Agent
from core.cbfs import cbfs_individual, cbfs_pairwise, cbf0
from core.controllers.cbf_qp_controller import CbfQpController
from core.controllers.consolidated_cbf_controller import ConsolidatedCbfController
from ..models import f, g, nControls
from .timing_params import *
from .physical_params import u_max
from .objective_functions import objective_accel_and_steering
from .nominal_controllers import LqrController, ZeroController
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

if platform.machine() == "aarch64":
    save_path = "/home/6lackmitchell/Documents/datastore/warehouse/test.pkl"
else:
    save_path = "/Users/mblack/Documents/git/ccbf-control/data/warehouse/test.pkl"


# Define controllers
def consolidated_cbf_controller(idx: int) -> ConsolidatedCbfController:
    return ConsolidatedCbfController(
        u_max,
        nAgents,
        objective_accel_and_steering,
        LqrController(idx),
        cbfs_individual,
        cbfs_pairwise,
    )


# Define CBF Controlled Agents
cbf_controlled_agents = [
    Agent(i, z0[i, :], u0, cbf0, time, step_dynamics, consolidated_cbf_controller(i), save_path)
    for i in range(3)
]
human_agents = [
    Agent(i, z0[i, :], u0, cbf0, time, step_dynamics, ZeroController(i), save_path)
    for i in range(3, 9)
]


centralized_agents = None
decentralized_agents = cbf_controlled_agents + human_agents
