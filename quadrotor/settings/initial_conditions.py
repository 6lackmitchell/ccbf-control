import numpy as np
from .physical_params import M, G, ELLIPSE_AX
from .control_params import f_max
filepath = '/home/dasc/MB/Code/sim_env/simdata/quadrotor/'
x0       = np.array([  0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0])      # Initial State

theta    = np.array([0.5,-0.9,1.0])  # True Theta parameters
thetaHat = np.array([5.0,-5.0,-5.0]) # Initial estimates of Theta parameters
thetaMax = np.array([5.0,5.0,5.0])
thetaMin = -thetaMax
nStates  = len(x0)