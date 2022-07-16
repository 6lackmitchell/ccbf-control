import numpy as np
from .physical_params import *
from .state_derivatives import *

# CX = 0.0
# PX = 10.0
# NX = 4

# CY = 0.0
# PY = 10.0
# NY = 4

CXo = 0.0
PXo = 12.0
NXo = 2

CYo = 0.0
PYo = 12.0
NYo = 2

CXi = 0.0
PXi = 3.0
NXi = 2

CYi = 0.0
PYi = 3.0
NYi = 2

CZ = 2.25
PZ = 2.5
NZ = 2


def cbf(x):
    return np.array([cbf_altitude(x),cbf_attitude(x)])


############################# Altitude Safety #############################


def cbf_altitude(x):
    return 1 - ((x[2] - CZ)/PZ)**NZ


def cbfdot_altitude(x):
    hdot_z = NZ * (x[2] - CZ)**(NZ-1) * zdot(x)
    return -hdot_z / PZ**NZ


def cbf2dot_altitude_uncontrolled(x):
    h2dot_z = (NZ * (NZ-1) * (x[2] - CZ)**(NZ-2) * zdot(x)**2) + \
              (NZ*(x[2] - CZ)**(NZ-1) * z2dot(x,1)[0])
    return -h2dot_z / PZ**NZ


def cbf2dot_altitude_controlled(x):
    h2dot_z = NZ*(x[2] - CZ)**(NZ-1) * (z2dot(x,1)[1])
    return -h2dot_z / PZ**NZ


############################# Attitude Safety #############################


def cbf_attitude_0(x):
    angle_in_deg = 90.0
    return np.cos(x[6])*np.cos(x[7]) - np.cos(np.pi/180 * angle_in_deg)


def cbfdot_attitude_0(x):
    return -phidot(x)*np.sin(x[6])*np.cos(x[7]) - thedot(x)*np.cos(x[6])*np.sin(x[7])


def cbf2dot_attitude_uncontrolled_0(x):
    return -phi2dot(x)[0]*np.sin(x[6])*np.cos(x[7]) - the2dot(x)[0]*np.cos(x[6])*np.sin(x[7])\
           -(phidot(x)**2 + thedot(x)**2)*np.cos(x[6])*np.cos(x[7]) + 2*phidot(x)*thedot(x)*np.sin(x[6])*np.sin(x[7])


def cbf2dot_attitude_controlled_0(x):
    return -phi2dot(x)[1]*np.sin(x[6])*np.cos(x[7]) - the2dot(x)[1]*np.cos(x[6])*np.sin(x[7])


def cbf_attitude(x):
    return cbf_attitude_0(x)


def cbfdot_attitude(x):
    return cbfdot_attitude_0(x)


def cbf2dot_attitude_uncontrolled(x):
    return cbf2dot_attitude_uncontrolled_0(x)


def cbf2dot_attitude_controlled(x):
    return cbf2dot_attitude_controlled_0(x)


############################# Details #############################

nCBFs = cbf(np.zeros((12,))).shape[0]