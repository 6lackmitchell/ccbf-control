import numpy as np
from .quadrotor_dynamics import R_body_to_inertial
from .physical_params import G, M, Jx, Jy, Jz

s   = np.sin
c   = np.cos
Rot = R_body_to_inertial


###############################################################################
#################### Rotations and Rotational Derivatives #####################
###############################################################################

def Omega(state: np.ndarray) -> np.ndarray:
    """ Represents the skew-symmetric matrix for the angular velocity such that
    w^x w = w x w (cross products).

    INPUTS
    ------
    state: (np.ndarray) - state vector (n = 12)

    OUTPUTS
    -------
    Omega (3x3) array

    """
    return np.array([[         0,-state[11], state[10]],
                     [ state[11],         0,-state[ 9]],
                     [-state[10], state[ 9],         0]])

def OmegaDot(state: np.ndarray) -> np.ndarray:
    """ Represents the time-derivative of the skew-symmetric matrix for the 
    angular velocity such that w^x w = w x w (cross products).

    INPUTS
    ------
    state: (np.ndarray) - state vector (n = 12)

    OUTPUTS
    -------
    OmegaDot (3x3) array

    """
    pdot_u = pdot_uncontrolled(state)
    qdot_u = qdot_uncontrolled(state)
    rdot_u = rdot_uncontrolled(state)
    OmegaDotUncontrolled = np.array([[      0,-rdot_u, qdot_u],
                                     [ rdot_u,      0,-pdot_u],
                                     [-qdot_u, pdot_u,      0]])
    pdot_c = pdot_controlled(state)
    qdot_c = qdot_controlled(state)
    rdot_c = rdot_controlled(state)
    OmegaDotControlled   = np.array([[      0,-rdot_c, qdot_c],
                                     [ rdot_c,      0,-pdot_c],
                                     [-qdot_c, pdot_c,      0]])
    return np.array([OmegaDotUncontrolled,OmegaDotControlled])#,dtype=object)

def RotDot(state: np.ndarray) -> np.ndarray:
    """ The time-derivative of the rotation matrix as a function of the state.

    INPUTS
    ------
    state: (np.ndarray) - state vector (n = 12)

    OUTPUTS
    -------
    RotDot (3x3 array)

    """
    return Rot(state) @ Omega(state)

def Rot2Dot_uncontrolled(state: np.ndarray) -> np.ndarray:
    """ The second time-derivative of the rotation matrix as a function of the
    state, however, the uncontrolled dynamics only.

    INPUTS
    ------
    state: (np.ndarray) - state vector (n = 12)

    OUTPUTS
    -------
    Rot2Dot_uncontrolled (3x3 array)

    """
    return RotDot(state)@Omega(state) + Rot(state)@OmegaDot(state)[0]

def Rot2Dot_controlled(state: np.ndarray) -> np.ndarray:
    """ The time-derivative of the rotation matrix as a function of the 
    state, however, the controlled dynamics only.

    INPUTS
    ------
    state: (np.ndarray) - state vector (n = 12)

    OUTPUTS
    -------
    RotDot (3x3 array)

    """
    return np.einsum('ij,jk->ikj',Rot(state),OmegaDot(state)[1])


###############################################################################
####################### Uncontrolled State Derivatives ########################
###############################################################################

##################################### XYZ #####################################

def xdot(state):
    return (Rot(state) @ state[3:6])[0]

def ydot(state):
    return (Rot(state) @ state[3:6])[1]

def zdot(state):
    return (Rot(state) @ state[3:6])[2]

##################################### UVW #####################################

def udot(state):
    return state[4]*state[11] - state[5]*state[10] - G*s(state[7])

def vdot(state):
    return state[5]*state[9] - state[3]*state[11] + G*s(state[6])*c(state[7])

def wdot_uncontrolled(state):
    return state[3]*state[10] - state[4]*state[9] + G*c(state[6])*c(state[7])

def wdot_controlled(state):
    return -1 / M

################################ phi/theta/psi ################################

def phidot(state):
    return state[9] + \
           state[10]*s(state[6])*s(state[7])/c(state[7]) + \
           state[11]*c(state[6])*s(state[7])/c(state[7])

def thedot(state):
    return state[10]*c(state[6]) - state[11]*s(state[6])

def psidot(state):
    return state[10]*s(state[6])/c(state[7]) + state[11]*c(state[6])/c(state[7])

##################################### PQR #####################################

def pdot_uncontrolled(state):
    return (Jy - Jz)/Jx * state[10] * state[11]

def qdot_uncontrolled(state):
    return (Jz - Jx)/Jy * state[9] * state[11]

def rdot_uncontrolled(state):
    return (Jx - Jy)/Jz * state[9] * state[10]

def pdot_controlled(state):
    return 1/Jx

def qdot_controlled(state):
    return 1/Jy

def rdot_controlled(state):
    return 1/Jz

###############################################################################
############################ 2nd Order Derivatives ############################
###############################################################################

##################################### XYZ #####################################

def x2dot(state: np.ndarray,
          F:     float) -> float:
    """ Second time-derivative of the x-coordinate wrt inertial frame

    INPUTS
    ------
    state: (np.ndarray) - state vector (n = 12)
    F:     (float) - force input - set this to 1 if F is control variable

    OUTPUTS
    -------
    x2dot: (float) - d2x/dt2 of x-coordinate wrt inertial frame

    """
    return Rot(state)[0,2] * -F / M

def y2dot(state: np.ndarray,
          F:     float) -> float:
    """ Second time-derivative of the y-coordinate wrt inertial frame

    INPUTS
    ------
    state: (np.ndarray) - state vector (n = 12)
    F:     (float) - force input - set this to 1 if F is control variable

    OUTPUTS
    -------
    y2dot: (float) - d2y/dt2 of y-coordinate wrt inertial frame
    
    """
    return Rot(state)[1,2] * -F / M

def z2dot(state: np.ndarray,
          F:     float) -> float:
    """ Second time-derivative of the z-coordinate wrt inertial frame

    INPUTS
    ------
    state: (np.ndarray) - state vector (n = 12)
    F:     (float) - force input - set this to 1 if F is control variable

    OUTPUTS
    -------
    z2dot: (float) - d2z/dt2 of z-coordinate wrt inertial frame
    
    """
    z2dot_uncontrolled = -G
    z2dot_controlled   = -Rot(state)[2,2] * F / M

    return np.array([z2dot_uncontrolled,z2dot_controlled])

def phi2dot(state):
    phi2dot_uncontrolled = state[10]*(phidot(state)*c(state[6])*np.tan(state[7]) + thedot(state)*s(state[6])/c(state[7])**2) \
                         + state[11]*(-phidot(state)*s(state[6])*np.tan(state[7]) + thedot(state)*c(state[6])/c(state[7])**2) \
                         + (Jy - Jz)/Jx*state[10]*state[11] + (Jz - Jx)/Jy*state[9]*state[11]*s(state[6])*np.tan(state[7])
    phi2dot_controlled = np.array([1/Jx,s(state[6])*np.tan(state[7])/Jy,0])
    return np.array([phi2dot_uncontrolled,phi2dot_controlled],dtype=object)

def the2dot(state):
    the2dot_uncontrolled = -state[10]*phidot(state)*s(state[6]) - state[11]*phidot(state)*c(state[6]) \
                         + (Jz - Jx)/Jy*state[9]*state[11]*c(state[6])
    the2dot_controlled = np.array([0,c(state[6])/Jy,0])
    return np.array([the2dot_uncontrolled,the2dot_controlled],dtype=object)

###############################################################################
####################### 3rd Order Uncontrolled Derivatives ########################
###############################################################################

##################################### XYZ #####################################

def x3dot(state: np.ndarray,
          F:     float) -> float:
    """ Third time-derivative of the z-coordinate wrt inertial frame

    INPUTS
    ------
    state: (np.ndarray) - state vector (n = 12)
    F:     (float) - force input - set this to 1 if F is control variable

    OUTPUTS
    -------
    x3dot: (float) - d3x/dt3 of x-coordinate wrt inertial frame
    
    """
    return -RotDot(state)[0,2] * F / M

def y3dot(state: np.ndarray,
          F:     float) -> float:
    """ Third time-derivative of the z-coordinate wrt inertial frame

    INPUTS
    ------
    state: (np.ndarray) - state vector (n = 12)
    F:     (float) - force input - set this to 1 if F is control variable

    OUTPUTS
    -------
    y3dot: (float) - d3y/dt3 of y-coordinate wrt inertial frame
    
    """
    return -RotDot(state)[1,2] * F / M

def z3dot(state: np.ndarray,
          F:     float) -> float:
    """ Third time-derivative of the z-coordinate wrt inertial frame

    INPUTS
    ------
    state: (np.ndarray) - state vector (n = 12)
    F:     (float) - force input - set this to 1 if F is control variable

    OUTPUTS
    -------
    z3dot: (float) - d3z/dt3 of z-coordinate wrt inertial frame
    
    """
    return -RotDot(state)[2,2] * F / M

###############################################################################
####################### 4th Order Uncontrolled Derivatives ########################
###############################################################################

##################################### XYZ #####################################

def x4dot(state: np.ndarray,
          F:     float) -> float:
    """ Fourth time-derivative of the z-coordinate wrt inertial frame

    INPUTS
    ------
    state: (np.ndarray) - state vector (n = 12)
    F:     (float) - force input - set this to 1 if F is control variable

    OUTPUTS
    -------
    x4dot: (float) - d4x/dt4 of x-coordinate wrt inertial frame
    
    """
    x4dot_uncontrolled = -Rot2Dot_uncontrolled(state)[0,2] * F / M
    x4dot_controlled   = -Rot2Dot_controlled(state)[0,2] * F / M

    return np.array([x4dot_uncontrolled,x4dot_controlled],dtype=object)

def y4dot(state: np.ndarray,
          F:     float) -> float:
    """ Fourth time-derivative of the z-coordinate wrt inertial frame

    INPUTS
    ------
    state: (np.ndarray) - state vector (n = 12)
    F:     (float) - force input - set this to 1 if F is control variable

    OUTPUTS
    -------
    y4dot: (float) - d4y/dt4 of y-coordinate wrt inertial frame
    
    """
    y4dot_uncontrolled = -Rot2Dot_uncontrolled(state)[1,2] * F / M
    y4dot_controlled   = -Rot2Dot_controlled(state)[1,2] * F / M
    
    return np.array([y4dot_uncontrolled,y4dot_controlled],dtype=object)

def z4dot(state: np.ndarray,
          F:     float) -> float:
    """ Fourth time-derivative of the z-coordinate wrt inertial frame

    INPUTS
    ------
    state: (np.ndarray) - state vector (n = 12)
    F:     (float) - force input - set this to 1 if F is control variable

    OUTPUTS
    -------
    z4dot: (float) - d4z/dt4 of z-coordinate wrt inertial frame
    
    """
    z4dot_uncontrolled = -Rot2Dot_uncontrolled(state)[2,2] * F / M
    z4dot_controlled   = -Rot2Dot_controlled(state)[2,2] * F / M
    
    return np.array([z4dot_uncontrolled,z4dot_controlled],dtype=object)

# ##################################### XYZ #####################################

# def xdot(state):
#     return state[3] * c(state[7])*c(state[8]) + \
#            state[4] * (s(state[6])*s(state[7])*c(state[8]) - c(state[6])*s(state[8])) + \
#            state[5] * (c(state[6])*s(state[7])*c(state[8]) + s(state[6])*s(state[8]))

# def ydot(state):
#     return state[3] * c(state[7])*s(state[8]) + \
#            state[4] * (s(state[6])*s(state[7])*s(state[8]) + c(state[6])*c(state[8])) + \
#            state[5] * (c(state[6])*s(state[7])*s(state[8]) - s(state[6])*c(state[8]))

# def zdot(state):
#     return state[3] * s(state[7]) + \
#            state[4] * -s(state[6])*c(state[7]) + \
#            state[5] * -c(state[6])*c(state[7])

# ##################################### UVW #####################################

# def udot(state):
#     return state[4]*state[11] - state[5]*state[10] - G*s(state[7])

# def vdot(state):
#     return state[5]*state[9] - state[3]*state[11] + G*s(state[6])*c(state[7])

# def wdot(state):
#     return state[3]*state[10] - state[4]*state[9] + G*c(state[6])*c(state[7])

# ################################ phi/theta/psi ################################

# def phidot(state):
#     return state[9] + \
#            state[10]*s(state[6])*s(state[7])/c(state[7]) + \
#            state[11]*c(state[6])*s(state[7])/c(state[7])

# def thedot(state):
#     return state[10]*c(state[6]) - state[11]*s(state[6])

# def psidot(state):
#     return state[10]*s(state[6])/c(state[7]) + state[11]*c(state[6])/c(state[7])

# ##################################### PQR #####################################

# def pdot(state):
#     return (Jy - Jz)/Jx * state[10] * state[11]

# def qdot(state):
#     return (Jz - Jx)/Jy * state[9] * state[11]

# def rdot(state):
#     return (Jx - Jy)/Jz * state[9] * state[10]

# ###############################################################################
# ############################ 2nd Order Derivatives ############################
# ###############################################################################

# ##################################### XYZ #####################################

# def x2dot(state: np.ndarray,
#           F:     float) -> float:
#     """ Second time-derivative of the x-coordinate wrt inertial frame

#     INPUTS
#     ------
#     state: (np.ndarray) - state vector (n = 12)
#     F:     (float) - force input - set this to 1 if F is control variable

#     OUTPUTS
#     -------
#     x2dot: (float) - d2x/dt2 of x-coordinate wrt inertial frame

#     """
#     return state[3] * c(state[7])*c(state[8]) + \
#            state[4] * (s(state[6])*s(state[7])*c(state[8]) - c(state[6])*s(state[8])) + \
#            state[5] * (c(state[6])*s(state[7])*c(state[8]) + s(state[6])*s(state[8]))

# def y2dot(state: np.ndarray,
#           F:     float) -> float:
#     """ Second time-derivative of the y-coordinate wrt inertial frame

#     INPUTS
#     ------
#     state: (np.ndarray) - state vector (n = 12)
#     F:     (float) - force input - set this to 1 if F is control variable

#     OUTPUTS
#     -------
#     y2dot: (float) - d2y/dt2 of y-coordinate wrt inertial frame
    
#     """
#     return state[3] * c(state[7])*s(state[8]) + \
#            state[4] * (s(state[6])*s(state[7])*s(state[8]) + c(state[6])*c(state[8])) + \
#            state[5] * (c(state[6])*s(state[7])*s(state[8]) - s(state[6])*c(state[8]))

# def z2dot(state: np.ndarray,
#           F:     float) -> float:
#     """ Second time-derivative of the z-coordinate wrt inertial frame

#     INPUTS
#     ------
#     state: (np.ndarray) - state vector (n = 12)
#     F:     (float) - force input - set this to 1 if F is control variable

#     OUTPUTS
#     -------
#     z2dot: (float) - d2z/dt2 of z-coordinate wrt inertial frame
    
#     """
#     return state[3] * s(state[7]) + \
#            state[4] * -s(state[6])*c(state[7]) + \
#            state[5] * -c(state[6])*c(state[7])

# ###############################################################################
# ####################### 3rd Order Uncontrolled Derivatives ########################
# ###############################################################################

# ##################################### XYZ #####################################

# def x2dot(state):
#     return state[3] * c(state[7])*c(state[8]) + \
#            state[4] * (s(state[6])*s(state[7])*c(state[8]) - c(state[6])*s(state[8])) + \
#            state[5] * (c(state[6])*s(state[7])*c(state[8]) + s(state[6])*s(state[8]))

# def y2dot(state):
#     return state[3] * c(state[7])*s(state[8]) + \
#            state[4] * (s(state[6])*s(state[7])*s(state[8]) + c(state[6])*c(state[8])) + \
#            state[5] * (c(state[6])*s(state[7])*s(state[8]) - s(state[6])*c(state[8]))

# def z2dot(state):
#     return state[3] * s(state[7]) + \
#            state[4] * -s(state[6])*c(state[7]) + \
#            state[5] * -c(state[6])*c(state[7])

# ###############################################################################
# ####################### 4th Order Uncontrolled Derivatives ########################
# ###############################################################################

# ##################################### XYZ #####################################

# def x2dot(state):
#     return state[3] * c(state[7])*c(state[8]) + \
#            state[4] * (s(state[6])*s(state[7])*c(state[8]) - c(state[6])*s(state[8])) + \
#            state[5] * (c(state[6])*s(state[7])*c(state[8]) + s(state[6])*s(state[8]))

# def y2dot(state):
#     return state[3] * c(state[7])*s(state[8]) + \
#            state[4] * (s(state[6])*s(state[7])*s(state[8]) + c(state[6])*c(state[8])) + \
#            state[5] * (c(state[6])*s(state[7])*s(state[8]) - s(state[6])*c(state[8]))

# def z2dot(state):
#     return state[3] * s(state[7]) + \
#            state[4] * -s(state[6])*c(state[7]) + \
#            state[5] * -c(state[6])*c(state[7])
