"""main simulation file with customizable settings"""
# package imports
import jax.numpy as jnp

# generic simulation imports
from core.agent import Agent
from core.simulate import simulate

# settings for simulation
from models.bicycle import DynamicBicycleModel
from models.bicycle.nominal_controllers import LqrController
from models.bicycle.cbfs.obstacle_avoidance import cbfs as cbfs_obstacle_avoidance
from models.bicycle.cbfs.speed_limit import cbfs as cbfs_speed
from models.bicycle.cbfs.slip_limit import cbfs as cbfs_slip
from models.bicycle.cbfs.reach_target import cbfs as cbfs_reach


# Not Complete
from core.controllers.consolidated_cbf_controller import ConsolidatedCbfController
from core.controllers.objective_functions import minimum_deviation

# bicycle dynamics model
n_controls = 2
goal_state = jnp.array([2.0, 2.0, 0.0, 0.0, 0.0])
xi = 0.0
yi = 0.0
psii = jnp.arctan2(goal_state[1] - yi, goal_state[0] - xi)
vi = 0.1
x0 = jnp.array([xi, yi, psii, vi, 0.0])
bicycle_model = DynamicBicycleModel(initial_state=x0, n_controls=n_controls)

# consolidated cbf controller
nominal_controller = LqrController(goal_state=goal_state)
objective = minimum_deviation
cbfs_individual = [
    cbfs_obstacle_avoidance,
    cbfs_speed,
    cbfs_slip,
    cbfs_reach,
]
cbfs_pairwise = []
ccbf_controller = ConsolidatedCbfController(
    nominal_controller=nominal_controller,
    objective_function=minimum_deviation,
    cbfs_individual=cbfs_individual,
    cbfs_pairwise=cbfs_pairwise,
)
agents = [
    Agent(dynamics=bicycle_model, controller=ccbf_controller),
]

# set agent identifiers
for aa, agent in enumerate(agents):
    agent.id = aa


end_time = 50.0
timestep = 0.02

success = simulate(end_time, timestep, vehicle, level, situation)

import sys

sys.exit(int(success))
