import numpy as np

# Acceleration due to gravity
G = 9.81  # meters / sec^2

# Taken from AscTec Hummingbird (https://productz.com/en/asctec-hummingbird/p/xKp1)
R = 0.27  # meters
M = 0.71  # kg
arm_length = 0.17  # meters
Jx = 0.00365  # kg m^2
Jy = 0.00368  # kg m^2
Jz = 0.00703  # kg m^2

# Actuation Parameters
k1 = 0.1
k2 = 1e-1

# Control Constraints
f_max = 4.0 * M * G  # propeller force control constraint
d_max = f_max / (4 * k1)
tx_max = arm_length * (f_max / 4.0)
ty_max = arm_length * (f_max / 4.0)
tz_max = 2 * k2 * d_max
# tau_max               = 100#2 * np.pi    # torque control constraint
# TIME_CONSTANT         = 0.5#1.0#0.4

# Circle-Tracking Trajectory
ELLIPSE_AX = 5.0
ELLIPSE_BY = 7.0

# Gerono Lemniscate Trajectory
F_GERONO = 0.1
A_GERONO = 3.0

# Regressor Matrix Parameters
freq_x = 0.01  # 1/100 m^(-1) -- Full cycle every 100m
freq_y = 0.02  # 1/50 m^(-1)  -- Full cycle every 50m

# Load Wind Profile
from .winds import *
