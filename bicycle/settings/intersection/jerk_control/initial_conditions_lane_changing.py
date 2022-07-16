import os
from random import random, randint
import numpy as np
from .timing_params import tf
from .physical_params import LW
from agent import Agent
from .dynamics_rdrive import step_dynamics
from bicycle.ffcbf_qp_controller import compute_control as ff_cbf_controller
from bicycle.ffcbf_qp_controller import highway_controller_lqr


def v_rand() -> float:
    """ Generates random initial velocity. """
    v0 = 32.0
    v_range = 5.0

    return v0 + v_range * 2 * (random() - 0.5)


lane_0 = 0
lane_2 = 2

# # Randomized Initial Conditions
# tfs = int(tf / 2)
# vi = np.array([v_rand(), v_rand(), v_rand(), v_rand(), v_rand(), v_rand()])
# xi = np.array([0.0, 3.0, -5.0, 6.0, 12.0, -10.0])
# yi = np.array([0.0, -LW, LW, LW, 0.0, -LW])
# gl = np.array([randint(lane_0, lane_2), 0, randint(lane_0, lane_2),
#                randint(lane_0, lane_2), 1, 0])
# st = np.array([randint(0, tfs), randint(0, tfs), randint(0, tfs), randint(0, tfs), randint(0, tfs), randint(0, tfs)])
#
#
# z1 = np.array([xi[0],  # x position
#                yi[0],  # y position
#                0.0,    # heading angle
#                vi[0],  # rear-wheel velocity
#                0.0])   # body slip-angle
#
# z2 = np.array([xi[1],  # x position
#                yi[1],  # y position
#                0.0,    # heading angle
#                vi[1],  # rear-wheel velocity
#                0.0])   # body slip-angle
#
# z3 = np.array([xi[2],  # x position
#                yi[2],  # y position
#                0.0,    # heading angle
#                vi[2],  # rear-wheel velocity
#                0.0])   # body slip-angle
#
# z4 = np.array([xi[3],  # x position
#                yi[3],  # y position
#                0.0,    # heading angle
#                vi[3],  # rear-wheel velocity
#                0.0])   # body slip-angle
#
# z5 = np.array([xi[4],  # x position
#                yi[4],  # y position
#                0.0,    # heading angle
#                vi[4],  # rear-wheel velocity
#                0.0])   # body slip-angle
#
# z6 = np.array([xi[5],  # x position
#                yi[5],  # y position
#                0.0,    # heading angle
#                vi[5],  # rear-wheel velocity
#                0.0])   # body slip-angle
#

# Demonstrative CBF Lane Changing Example
vi = np.array([30.0, 33.0, 34.0])
xi = np.array([0.0, -5.0, -25.0])  # Fit in between two cars
# xi = np.array([0.0, -10.0, -25.0])  # Accel to get in front of first car
yi = np.array([0.0, LW, LW])
gl = np.array([2, 2, 2])
st = np.array([0.5, 0, 0])
z1 = np.array([xi[0],  # x position
               yi[0],  # y position
               0.0,    # heading angle
               vi[0],  # rear-wheel velocity
               0.0])   # body slip-angle
z2 = np.array([xi[1],  # x position
               yi[1],  # y position
               0.0,    # heading angle
               vi[1],  # rear-wheel velocity
               0.0])   # body slip-angle
z3 = np.array([xi[2],  # x position
               yi[2],  # y position
               0.0,    # heading angle
               vi[2],  # rear-wheel velocity
               0.0])   # body slip-angle
z4 = np.array([xi[0],  # x position
               yi[0],  # y position
               0.0,    # heading angle
               vi[0],  # rear-wheel velocity
               0.0])   # body slip-angle
z_unsafe = z4
vi = np.append(vi, vi[0])
xi = np.append(xi, xi[0])
yi = np.append(yi, yi[0])
gl = np.append(gl, gl[0])
st = np.append(st, st[0])

z0 = np.array([z1, z2, z3, z4])

theta = np.array([0.5, -0.9, 1.0])      # True Theta parameters
thetaHat = np.array([5.0, -5.0, -5.0])  # Initial estimates of Theta parameters
thetaMax = np.array([5.0, 5.0, 5.0])    # Polytopic set in which theta can exist
thetaMin = -thetaMax
nAgents, nStates = z0.shape
# nStates = len(z0)
nParams = len(theta)

# Get Filepath
from sys import platform
if platform == "linux" or platform == "linux2":
    # linux
    save_path = os.path.dirname(os.path.abspath(__file__)) + '/../datastore/highway/'
elif platform == "darwin":
    # OS X
    pass
elif platform == "win32":
    # Windows
    save_path = os.path.dirname(os.path.abspath(__file__)) + '\\..\\datastore\\highway\\'




# Define Agents
lane_changing_agents = [Agent(z1, step_dynamics, ff_cbf_controller),
                        Agent(z2, step_dynamics, highway_controller_lqr),
                        Agent(z3, step_dynamics, ff_cbf_controller),
                        Agent(z4, step_dynamics, highway_controller_lqr)]
