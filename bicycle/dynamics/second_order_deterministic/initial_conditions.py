import os
import numpy as np
from sys import platform
from bicycle.dynamics.second_order_deterministic.physical_params import LW

# Initial States
xi = np.array([LW/2, 20.0, -LW/2, -19.0])  # Fit in between two cars
yi = np.array([-19.0, LW/2, 18.0, -LW/2])
vi = np.array([5.0, 5.0, 5.0, 5.0])
di = ['+y', '-x', '-y', '+x']
hi = {'+y': np.pi / 2, '-x': np.pi, '-y': -np.pi / 2, '+x': 0.0}
z1 = np.array([xi[0],      # x position
               yi[0],     # y position
               hi[di[0]],   # heading angle
               vi[0],       # rear-wheel velocity
               0.0])      # body slip-angle

z2 = np.array([xi[1],      # x position
               yi[1],      # y position
               hi[di[1]],     # heading angle
               vi[1],       # rear-wheel velocity
               0.0])      # body slip-angle

z3 = np.array([xi[2],     # x position
               yi[2],      # y position
               hi[di[2]],  # heading angle
               vi[2],       # rear-wheel velocity
               0.0])      # body slip-angle

z4 = np.array([xi[3],     # x position
               yi[3],     # y position
               hi[di[3]],       # heading angle
               vi[3],       # rear-wheel velocity
               0.0])      # body slip-angle

z0 = np.array([z1, z2, z3, z4])
nAgents, nStates = z0.shape

# Get Filepath
delimiter = '/'  # linux or darwin or linux2
if platform == "win32":
    # Windows
    delimiter = '\\'

folder_ending = delimiter + '..' + \
                delimiter + 'datastore' + \
                delimiter + 'intersection' + \
                delimiter + 'accel_control' + \
                delimiter
save_path = os.path.dirname(os.path.abspath(__file__)) + folder_ending