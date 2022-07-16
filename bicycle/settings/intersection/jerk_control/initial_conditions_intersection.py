import os
import numpy as np
from sys import platform
from .physical_params import LW
from agent import Agent
from .dynamics_rdrive import step_dynamics
from bicycle.prb_cbf_controller import compute_control as prb_cbf_controller
from bicycle.ffcbf_qp_controller import compute_control as ff_cbf_controller
from bicycle.cbf_qp_controller import compute_control as nominal_cbf_controller
from bicycle.nominal_controllers import jerk_intersection_controller_lqr

# Initial States
xi = np.array([LW/2, 20.0, -LW/2, -19.0])  # Fit in between two cars
yi = np.array([-19.0, LW/2, 18.0, -LW/2])
vi = np.array([5.0, 5.0, 5.0, 5.0])
di = ['+y', '-x', '-y', '+x']
hi = {'+y': np.pi / 2, '-x': np.pi, '-y': -np.pi / 2, '+x': 0.0}
z1 = np.array([xi[0],      # x position
               yi[0],      # y position
               hi[di[0]],  # heading angle
               vi[0],      # rear-wheel velocity
               0.0,        # body slip-angle
               0.0,        # rear-wheel acceleration
               0.0])       # body slip-angle rate

z2 = np.array([xi[1],      # x position
               yi[1],      # y position
               hi[di[1]],  # heading angle
               vi[1],      # rear-wheel velocity
               0.0,        # body slip-angle
               0.0,        # rear-wheel acceleration
               0.0])       # body slip-angle rate

z3 = np.array([xi[2],      # x position
               yi[2],      # y position
               hi[di[2]],  # heading angle
               vi[2],      # rear-wheel velocity
               0.0,        # body slip-angle
               0.0,        # rear-wheel acceleration
               0.0])       # body slip-angle rate

z4 = np.array([xi[3],      # x position
               yi[3],      # y position
               hi[di[3]],  # heading angle
               vi[3],      # rear-wheel velocity
               0.0,        # body slip-angle
               0.0,        # rear-wheel acceleration
               0.0])       # body slip-angle rate

z0 = np.array([z1, z2, z3, z4])

# Get Filepath
delimiter = '\\' if platform == "win32" else '/'
folder_ending = delimiter + '..' + delimiter + 'datastore' + delimiter + 'intersection' + delimiter + 'jerk' + delimiter
save_path = os.path.dirname(os.path.abspath(__file__)) + folder_ending

# Define Agents
nAgents, nStates = z0.shape
intersection_agents = [Agent(z1, step_dynamics, nominal_cbf_controller),
                       Agent(z2, step_dynamics, jerk_intersection_controller_lqr),
                       Agent(z3, step_dynamics, nominal_cbf_controller),
                       Agent(z4, step_dynamics, jerk_intersection_controller_lqr)]
