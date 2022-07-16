from .physical_params import *
from .timing_params import dt
from nptyping import NDArray


def f(z: NDArray) -> NDArray:
    """ Drift dynamics for bicycle model:

    INPUTS
    ------
    z: state vector (Ax5)

    OUTPUTS
    -------
    f(z): np.ndarray -- Ax5

    States (z[agent])
    ------
    0: (x)    inertial x-position (in m)
    1: (y)    inertial y-position (in m)
    2: (psi)  heading angle (in rad)
    3: (vr)   rear-wheel velocity (in m/s)
    4: (beta) body slip-angle (in rad)

    """

    if np.sum(z.shape) == z.shape[0]:
        return np.array([z[3] * (np.cos(z[2]) - np.sin(z[2]) * np.tan(z[4])),
                         z[3] * (np.sin(z[2]) + np.cos(z[2]) * np.tan(z[4])),
                         z[3] * np.tan(z[4]) / Lr,
                         0.0,
                         0.0])
    else:
        return np.array([[zz[3] * (np.cos(zz[2]) - np.sin(zz[2]) * np.tan(zz[4])),
                          zz[3] * (np.sin(zz[2]) + np.cos(zz[2]) * np.tan(zz[4])),
                          zz[3] * np.tan(zz[4]) / Lr,
                          0.0,
                          0.0] for i, zz in enumerate(z)])


def g(z: NDArray) -> NDArray:
    """ Controlled dynamics for bicycle model:

    INPUTS
    ------
    z: state vector (5x1)

    OUTPUTS
    -------
    g(z): np.ndarray -- 5x2

    """
    g = np.array([[0.0, 0.0],
                  [0.0, 0.0],
                  [0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0]])

    if np.sum(z.shape) == z.shape[0]:
        return g
    else:
        return np.array(z.shape[0]*[g])


def system_dynamics(t: float,
                    z: NDArray,
                    u: NDArray,
                    **kwargs: dict) -> NDArray:
    """ Dynamical model for the full bicycle system.

    INPUTS
    ------
    t: time (in sec)
    z: state vector (5x1)
    u: control input vector (2x1)

    OPTIONAL INPUT KEYS
    ---------------
    theta: true affine system parameters influencing dynamics

    OUTPUTS
    -------
    f(z): np.ndarray -- 5x1

    States (z)
    ------
    0: (x)    inertial x-position (in m)
    1: (y)    inertial y-position (in m)
    2: (psi)  heading angle (in rad)
    3: (vr)   rear-wheel velocity (in m/s)
    4: (beta) body slip-angle (in rad)

    Controls (u)
    --------
    0: (ar)    longitudinal acceleration of the rear-wheel (in m/s^2)
    1: (omega) time-derivative of the body slip-angle (in rad/s)

    """

    try:
        zdot = f(z) + np.einsum('ijk,ik->ij', g(z), u)
    except ValueError:
        zdot = f(z) + np.dot(g(z), u)

    if 'theta' in kwargs.keys():
        zdot = zdot + regressor(t,z) @ kwargs['theta']

    return zdot


def feasibility_dynamics(t, p, v):
    return v


def step_dynamics(t: float,
                  x: NDArray,
                  u: NDArray) -> NDArray:
    """ Uses the current time, state, and control action to advance the state forward in time according
    to the (in this case bicycle) dynamics.

    INPUTS
    ------
    x: state vector (Ax5)
    u: control input vector (Ax2)

    OUTPUTS
    -------
    updated state vector (Ax5)

    """
    return x + dt * system_dynamics(t, x, u)
    # return x + dt * system_dynamics(0, x, u, theta=theta)

nControls = g(np.zeros((1,))).shape[1]
# nParams = regressor(np.zeros((12,))).shape[1]