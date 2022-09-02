import builtins
from importlib import import_module
from core.agent import Agent
from core.controllers.cbfs import cbfs_individual, cbfs_pairwise, cbf0
from core.controllers.cbf_qp_controller import CbfQpController
from core.controllers.consolidated_cbf_controller import ConsolidatedCbfController
from core.controllers.centralized_controller import CentralizedController
from .timing_params import *
from .physical_params import u_max
from .models import f, g, nControls
from .objective_functions import objective_accel_and_steering
from .intersection.nominal_controllers import LqrController

vehicle = builtins.PROBLEM_CONFIG['vehicle']
control_level = builtins.PROBLEM_CONFIG['control_level']
system_model = builtins.PROBLEM_CONFIG['system_model']
situation = builtins.PROBLEM_CONFIG['situation']
mod = vehicle + '.' + control_level + '.' + situation + '.initial_conditions'

# Programmatic version of 'from situation import *'
try:
    module = import_module(mod)
    globals().update(
        {n: getattr(module, n) for n in module.__all__} if hasattr(module, '__all__')
        else {k: v for (k, v) in module.__dict__.items() if not k.startswith('_')}
    )
except ModuleNotFoundError as e:
    print('No module named \'{}\' -- exiting.'.format(mod))
    raise e

if system_model == 'stochastic':
    from .models import sigma_stochastic as sigma, \
        stochastic_dynamics as system_dynamics, stochastic_step as step_dynamics
else:
    from .models import sigma_deterministic as sigma, \
        deterministic_dynamics as system_dynamics, deterministic_step as step_dynamics

# Configure parameters
nAgents = len(z0)
time = [dt, tf]
save_path = '/Users/mblack/Documents/datastore/swarm/test.pkl'


# Define controllers
def deterministic_cbf_controller(idx: int) -> CbfQpController:
    return CbfQpController(
        u_max,
        nAgents,
        objective_accel_and_steering,
        LqrController(idx).compute_control,
        cbfs_individual,
        cbfs_pairwise
    )


def consolidated_cbf_controller(idx: int) -> ConsolidatedCbfController:
    return ConsolidatedCbfController(
        u_max,
        nAgents,
        objective_accel_and_steering,
        LqrController(idx).compute_control,
        cbfs_individual,
        cbfs_pairwise
    )


# Define CBF Controlled Agents
cbf_controlled_agents = [
    Agent(i, z0[i, :], u0, cbf0, time, step_dynamics, consolidated_cbf_controller(i), save_path) for i in range(nAgents)
]
# + [Agent(3, z0[3, :], u0, cbf0, time, step_dynamics, deterministic_cbf_controller(3), save_path)]
# l_cbf_ag = len(cbf_controlled_agents)

# centralized_agents = CentralizedController(
#     [age for age in cbf_controlled_agents[:3]],
#     u_max
# )


centralized_agents = None
decentralized_agents = cbf_controlled_agents
