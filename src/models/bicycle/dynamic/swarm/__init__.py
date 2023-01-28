import builtins
from sys import platform
from importlib import import_module
from core.agent import Agent
from core.cbfs import cbfs_individual, cbfs_pairwise, cbf0
from core.controllers.consolidated_cbf_controller import ConsolidatedCbfController
from ..models import f, g, nControls
from .timing_params import *
from .physical_params import u_max
from .objective_functions import objective_accel_and_steering
from .nominal_controllers import LqrController
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

save_path = pre_path + "Documents/git/ccbf-control/data/bicycle/dynamic/swarm/test.pkl"


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
    for i in range(nAgents)
]

centralized_agents = None
decentralized_agents = cbf_controlled_agents
