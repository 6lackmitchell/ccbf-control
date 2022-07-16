import numpy as np
from .physical_params import *
from .cbfs import *

###############################################################################
############################# Control Constraints #############################
###############################################################################
F_MAX    = f_max
TX_MAX   = tx_max
TY_MAX   = ty_max
TZ_MAX   = tz_max

###############################################################################
############################ CBF-CLF-QP Parameters ############################
###############################################################################
qf0      = 1.0
qt0      = 1.0
qt1      = 1.0
qt2      = 1.0
qa       = 1e3
POWER    = 1.0
kB       = 1.0

F_DECISION_VARS   = [{'name':'F',   'lb':0.0,    'ub':F_MAX},
                     {'name':'a0',  'lb':1.0,    'ub':1e+6}]

a0ub = 1e+6
a1ub = 1e+4
a2ub = 1e+2
a3ub = 1e+0
TAU_DECISION_VARS = [{'name':'tau1','lb':-TX_MAX,'ub':TX_MAX},
                     {'name':'tau2','lb':-TY_MAX,'ub':TY_MAX},
                     {'name':'tau3','lb':-TZ_MAX,'ub':TZ_MAX},
                     {'name':'a00',  'lb':0.0,   'ub':a0ub},
                     {'name':'a01',  'lb':1.0,   'ub':a1ub},
                     {'name':'a02',  'lb':1.0,   'ub':a2ub},
                     {'name':'a03',  'lb':1.0,   'ub':a3ub},
                     {'name':'a10',  'lb':0.0,   'ub':a0ub},
                     {'name':'a11',  'lb':1.0,   'ub':a1ub},
                     {'name':'a12',  'lb':1.0,   'ub':a2ub},
                     {'name':'a13',  'lb':1.0,   'ub':a3ub},
                     {'name':'a20',  'lb':0.0,   'ub':1e0},
                     {'name':'a21',  'lb':1.0,   'ub':1e0},
                     {'name':'a22',  'lb':1.0,   'ub':1e0},
                     {'name':'a30',  'lb':0.0,   'ub':1e0},
                     {'name':'a31',  'lb':1.0,   'ub':1e0},
                     {'name':'a32',  'lb':1.0,   'ub':1e0},
                     {'name':'a40',  'lb':0.0,   'ub':a0ub},
                     {'name':'a41',  'lb':1.0,   'ub':a1ub}]

nSols = 1 + 3#len(F_DECISION_VARS)+len(TAU_DECISION_VARS)

###############################################################################
################################## Functions ##################################
###############################################################################


def thrust_objective(thrust_nom):
    return np.diag([qf0, qa]), \
           np.array([-2*qf0*thrust_nom, 0], dtype=np.float64)
    # return qf0*v[0]*v[0] - 2*qf0*v[0]*F_nom + qf0*F_nom**2 \
    #      + qa*v[1]*v[1]


def moments_objective(moment_nom):
    return np.diag([qt0, qt1, qt2, qa, qa]), \
           np.array([-2*qt0*moment_nom[0], -2*qt1*moment_nom[1], -2*qt2*moment_nom[2], 0, 0], dtype=np.float64)
    # return qt0*v[0]*v[0] - 2*qt0*v[0]*tau_nom[0] + qt0*tau_nom[0]**2 \
    #      + qt1*v[1]*v[1] - 2*qt1*v[1]*tau_nom[1] + qt1*tau_nom[1]**2 \
    #      + qt2*v[2]*v[2] - 2*qt2*v[2]*tau_nom[2] + qt2*tau_nom[1]**2 \
    #      + qa*v[3]*v[3]

###############################################################################
############################# General Parameters ##############################
###############################################################################
ERROR    = -999*np.ones((nSols,))