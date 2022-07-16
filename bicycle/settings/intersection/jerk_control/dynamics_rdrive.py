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
    0: (x)     inertial x-position (in m)
    1: (y)     inertial y-position (in m)
    2: (psi)   heading angle (in rad)
    3: (vr)    rear-wheel velocity (in m/s)
    4: (beta)  body slip-angle (in rad)
    5: (ar)    rear-wheel acceleration (in m/s^2)
    6: (omega) body slip-angle rate (in rad/s)

    """

    if np.sum(z.shape) == z.shape[0]:
        return f_single_agent(z)
    else:
        return np.array([f_single_agent(zz) for zz in z])


def f_single_agent(state: NDArray) -> NDArray:
    """ Drift dynamics for single-agent:

    INPUTS
    ------
    state: state vector (Ax5)

    OUTPUTS
    -------
    f(state): np.ndarray (Ax5)

    States
    ------
    0: (x)     inertial x-position (in m)
    1: (y)     inertial y-position (in m)
    2: (psi)   heading angle (in rad)
    3: (vr)    rear-wheel velocity (in m/s)
    4: (beta)  body slip-angle (in rad)
    5: (ar)    rear-wheel acceleration (in m/s^2)
    6: (omega) body slip-angle rate (in rad/s)

    """

    return np.array([state[3] * (np.cos(state[2]) - np.sin(state[2]) * np.tan(state[4])),
                     state[3] * (np.sin(state[2]) + np.cos(state[2]) * np.tan(state[4])),
                     state[3] * np.tan(state[4]) / Lr,
                     state[5],
                     state[6],
                     0.0,
                     0.0])


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
                  [0.0, 0.0],
                  [0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0]])

    if np.sum(z.shape) == z.shape[0]:
        return g_single_agent(z)
    else:
        return np.array([g_single_agent(zz) for zz in z])


def g_single_agent(state: NDArray) -> NDArray:
    """ Controlled dynamics for single-agent:

    INPUTS
    ------
    state: state vector (5x1)

    OUTPUTS
    -------
    g(state): np.ndarray (5x2)

    Controls
    --------
    u(0) = jr -- rear-wheel jerk in (m/s^3)
    u(1) = alpha -- angular acceleration of slip-angle (in rad/s^2)

    """
    return np.array([[0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [1.0, 0.0],
                     [0.0, 1.0]])


def x_accel(state: NDArray) -> float:
    """Computes the acceleration in the x-direction based on the current state.

    INPUTS
    ------
    state: state vector (7x1)

    OUTPUTS
    -------
    ax: x-acceleration (float)

    """
    phi = state[2]
    v = state[3]
    beta = state[4]
    ar = state[5]
    omega = state[6]

    phi_dot = f_single_agent(state)[2]

    ax = ar * (np.cos(phi) - np.sin(phi) * np.tan(beta)) - \
         v * (phi_dot * (np.sin(phi) + np.cos(phi) * np.tan(beta)) + omega * np.sin(phi) / np.cos(beta)**2)

    return ax


def y_accel(state: NDArray) -> float:
    """Computes the acceleration in the y-direction based on the current state.

    INPUTS
    ------
    state: state vector (7x1)

    OUTPUTS
    -------
    ay: y-acceleration (float)

    """
    phi = state[2]
    v = state[3]
    beta = state[4]
    ar = state[5]
    omega = state[6]

    phi_dot = f_single_agent(state)[2]

    ay = ar * (np.sin(phi) + np.cos(phi) * np.tan(beta)) + \
         v * (phi_dot * (np.cos(phi) - np.sin(phi) * np.tan(beta)) + omega * np.cos(phi) / np.cos(beta)**2)

    return ay


def x_jerk_uncontrolled(state: NDArray) -> float:
    """Computes the drift jerk in the x-direction based on the current state.

    INPUTS
    ------
    state: state vector (7x1)

    OUTPUTS
    -------
    jx_unc: x-jerk uncontrolled (float)

    """
    phi = state[2]
    vr = state[3]
    beta = state[4]
    ar = state[5]
    omega = state[6]

    phi_dot = f_single_agent(state)[2]
    phi2_dot = (ar * np.tan(beta) + vr * omega / np.cos(beta)**2) / Lr

    jx_unc = -2 * ar * (phi_dot * (np.sin(phi) + np.cos(phi) * np.tan(beta)) + omega * np.sin(phi) / np.cos(beta)**2) \
             - vr * (phi2_dot * np.sin(phi) + phi_dot**2 * np.cos(phi) + phi2_dot * np.cos(phi) * np.tan(beta) -
                     phi_dot**2 * np.sin(phi) * np.tan(beta) + 2 * phi_dot * omega * np.cos(phi) / np.cos(beta)**2 +
                     2 * omega**2 * np.sin(phi) * np.tan(beta) / np.cos(beta)**2)

    return jx_unc


def x_jerk_controlled(state: NDArray) -> NDArray:
    """Computes the control multipliers based on the current state such that:

    jx_con' * u = x_jerk_controlled

    INPUTS
    ------
    state: state vector (7x1)

    OUTPUTS
    -------
    jx_con: x-jerk controlled multipliers(float)

    """
    phi = state[2]
    vr = state[3]
    beta = state[4]

    jx_con = np.array([np.cos(phi) - np.sin(phi) * np.tan(beta),  # multiplies rear-wheel jerk
                       -vr * np.sin(phi) / np.cos(beta)**2])      # multiplies angular acceleration

    return jx_con


def y_jerk_uncontrolled(state: NDArray) -> float:
    """Computes the drift jerk in the y-direction based on the current state.

    INPUTS
    ------
    state: state vector (7x1)

    OUTPUTS
    -------
    jy_unc: y-jerk uncontrolled (float)

    """
    phi = state[2]
    vr = state[3]
    beta = state[4]
    ar = state[5]
    omega = state[6]

    phi_dot = f_single_agent(state)[2]
    phi2_dot = (ar * np.tan(beta) + vr * omega / np.cos(beta)**2) / Lr

    jy_unc = 2 * ar * (phi_dot * (np.cos(phi) - np.sin(phi) * np.tan(beta)) + omega * np.cos(phi) / np.cos(beta)**2) \
             + vr * (phi2_dot * np.cos(phi) - phi_dot**2 * np.sin(phi) - phi2_dot * np.sin(phi) * np.tan(beta) -
                     phi_dot**2 * np.cos(phi) * np.tan(beta) - 2 * phi_dot * omega * np.sin(phi) / np.cos(beta)**2 +
                     2 * omega**2 * np.cos(phi) * np.tan(beta) / np.cos(beta)**2)

    return jy_unc


def y_jerk_controlled(state: NDArray) -> NDArray:
    """Computes the control multipliers based on the current state such that:

    jy_con' * u = y_jerk_controlled

    INPUTS
    ------
    state: state vector (7x1)

    OUTPUTS
    -------
    jy_con: y-jerk controlled multipliers (float)

    """
    phi = state[2]
    vr = state[3]
    beta = state[4]

    jy_con = np.array([np.sin(phi) + np.cos(phi) * np.tan(beta),  # multiplies rear-wheel jerk
                       vr * np.cos(phi) / np.cos(beta)**2])      # multiplies angular acceleration

    return jy_con


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