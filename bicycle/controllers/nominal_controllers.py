from control import lqr
import numpy as np
from nptyping import NDArray
from ..dynamics import f, Lr, LW, xi, yi, vi, di, hi


def u_nom_wrapper(func):

    def u_nom(t, z, identifier):
        return func(z)

    return u_nom

###############################################################################
################################## Functions ##################################
###############################################################################


def proportional_velocity_control(z: NDArray) -> (float, float):
    """ Computes the control inputs based on a desired velocity (assumed no change in heading).

    INPUTS
    ------
    t: time (in sec)
    z: state vector (5x1)

    OUTPUTS
    -------
    omega: time-derivative of body slip-angle (in rad/sec)
    ar:    rear-wheel acceleration (in m/s^2)

    """
    vd = 10.0  # Desired velocity
    kv = 2.0   # Proportional gain

    ar = kv * (vd - z[3])
    omega = 0.0

    return omega, ar


def proportional_lane_changing_controller(t: float,
                                          z: NDArray,
                                          a: int) -> (float, float):
    """ Computes the control inputs based on a desired velocity and heading angle.

    INPUTS
    ------
    t: time (in sec)
    z: state vector (5x1)
    a: index for agent

    OUTPUTS
    -------
    omega: time-derivative of body slip-angle (in rad/sec)
    ar:    rear-wheel acceleration (in m/s^2)

    """
    from .bicycle_settings.initial_conditions_lane_changing import vi, yi, gl, st

    dims = z.shape
    if len(dims) > 1:  # Check whether one agent was passed or full state
        z = z[a]

    # Proportional control for acceleration
    vd = vi[a]
    kv = 0.25  # Proportional gain for acceleration
    ar = kv * (vd - z[3])

    kp = 5.0  # Proportional gain for change in heading
    kb = 4.0  # Proportional gain for omega

    omega = 0.0  # Nominally do not change heading

    # Check lane condition -- update dy if need to change lanes
    lane = {0: -LW, 1: 0.0, 2: LW}
    if t > st[a]:
        dy = lane[gl[a]] - z[1]
    else:
        dy = yi[a] - z[1]

    if abs(dy) > 0.01:
        power = 2.0
        change_time = 0.25 * abs(dy)**power / LW**power  # time to execute lane change
        dx = z[0] + change_time * z[3]  # Compute dx to get desired heading
        phi_d = np.arctan2(dy, dx)  # Desired heading
        delta_phi = kp * (phi_d - z[2])  # Change in heading
        beta_d = np.arctan2(Lr * delta_phi, z[3])  # Desired slip angle
        omega = kb * (beta_d - z[4])

    return omega, ar


def intersection_controller_lqr(t: float,
                                z: NDArray,
                                a: int) -> (float, float):
    """ Computes the control inputs based on a desired velocity and heading angle.

    INPUTS
    ------
    t: time (in sec)
    z: state vector (5x1)
    a: index for agent

    OUTPUTS
    -------
    omega: time-derivative of body slip-angle (in rad/sec)
    ar:    rear-wheel acceleration (in m/s^2)

    """
    # from .bicycle_settings.initial_conditions_intersection import vi, xi, yi, di

    dims = z.shape
    if len(dims) > 1:
        z = z[a]

    # Get desired x, y, vx, vy
    if di[a] == '+x':
        xd = 50.0
        yd = yi[a]
        vxd = vi[a]
        vyd = 0.0

    elif di[a] == '-x':
        xd = -50.0
        yd = yi[a]
        vxd = -vi[a]
        vyd = 0.0

    elif di[a] == '+y':
        xd = xi[a]
        yd = 50.0
        vxd = 0.0
        vyd = vi[a]

    else:
        xd = xi[a]
        yd = -50.0
        vxd = 0.0
        vyd = -vi[a]

    # LQR Controller from Black et al. 2022 (https://arxiv.org/pdf/2204.00127v1.pdf)
    q_star = np.array([xd, yd, vxd, vyd])  # desired state
    zeta = np.array([z[0], z[1], f(z)[0], f(z)[1]])  # double integrator state
    tracking_error = zeta - q_star
    A_di = np.array([[0, 0, 1, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]])
    B_di = np.array([[0, 0],
                     [0, 0],
                     [1, 0],
                     [0, 1]])
    Q = np.diag([0.001, 0.001, 0.01, 0.01])
    R = np.eye(2)
    K, _, _ = lqr(A_di, B_di, Q, R)  # Compute LQR control input for double integrator model
    mu = -K @ tracking_error  # ax, ay control inputs based on double integrator

    # Create transformation matrix to map from ax, ay to omega, ar
    S = np.array([[-z[3] * np.sin(z[2]) / np.cos(z[4])**2, np.cos(z[2]) - np.sin(z[2]) * np.tan(z[4])],
                  [z[3] * np.cos(z[2]) / np.cos(z[4])**2, np.sin(z[2]) + np.cos(z[2]) * np.tan(z[4])]])

    # If S is close to singular it means that vr is small, apply only acceleration command
    if z[3] > 0.1:
        vec = np.array([mu[0] + f(z)[1] * f(z)[2], mu[1] - f(z)[0] * f(z)[2]])
        u = np.linalg.inv(S) @ vec
        omega = u[0]
        ar = u[1]
    else:
        omega = 0.0
        ar = np.linalg.norm(mu)

    return omega, ar


def jerk_intersection_controller_lqr(t: float,
                                     z: NDArray,
                                     a: int) -> (float, float):
    """ Computes the control inputs for the jerk dynamics based on a desired velocity and heading angle.

    INPUTS
    ------
    t: time (in sec)
    z: state vector (5x1)
    a: index for agent

    OUTPUTS
    -------
    jr:    rear-wheel jerk (in m/s^3)
    alpha: 2nd time-derivative of body slip-angle (in rad/sec)

    """
    from bicycle.settings.intersection.jerk_control import vi, xi, yi, di

    dims = z.shape
    if len(dims) > 1:
        z = z[a]

    # Get desired x, y, vx, vy, ax, ay
    kv = 10.0
    if di[a] == '+x':
        xd = 50.0
        yd = yi[a]
        vxd = vi[a]
        vyd = 0.0
        axd = kv * (vxd - f(z)[0])
        ayd = kv * (vyd - f(z)[1])

    elif di[a] == '-x':
        xd = -50.0
        yd = yi[a]
        vxd = -vi[a]
        vyd = 0.0
        axd = kv * (vxd - f(z)[0])
        ayd = kv * (vyd - f(z)[1])

    elif di[a] == '+y':
        xd = xi[a]
        yd = 50.0
        vxd = 0.0
        vyd = vi[a]
        axd = kv * (vxd - f(z)[0])
        ayd = kv * (vyd - f(z)[1])

    else:
        xd = xi[a]
        yd = -50.0
        vxd = 0.0
        vyd = -vi[a]
        axd = kv * (vxd - f(z)[0])
        ayd = kv * (vyd - f(z)[1])

    # LQR Controller from Black et al. 2022 (https://arxiv.org/pdf/2204.00127v1.pdf)
    q_star = np.array([xd, yd, vxd, vyd, axd, ayd])  # desired state
    zeta = np.array([z[0], z[1], f(z)[0], f(z)[1], x_accel(z), y_accel(z)])  # double integrator state
    tracking_error = zeta - q_star
    A_di = np.array([[0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]])
    B_di = np.array([[0, 0],
                     [0, 0],
                     [0, 0],
                     [0, 0],
                     [1, 0],
                     [0, 1]])
    Q = np.diag([0.001, 0.001, 0.01, 0.01, 0.001, 0.001])
    R = np.eye(2)
    K, _, _ = lqr(A_di, B_di, Q, R)  # Compute LQR control input for double integrator model
    mu = -K @ tracking_error  # ax, ay control inputs based on double integrator

    # Create transformation matrix to map from ax, ay to omega, ar
    vector = np.array([x_jerk_uncontrolled(z), y_jerk_uncontrolled(z)])
    matrix = np.array([x_jerk_controlled(z), y_jerk_controlled(z)])

    # If matrix is close to singular it means that ar is small, apply only jerk command
    if z[5] > 0.1:
        vec = np.array([mu[0] - vector[0], mu[1] - vector[1]])
        u = np.linalg.inv(matrix) @ vec
        jr = u[0]
        alpha = u[1]
    else:
        jr = np.linalg.norm(mu)
        alpha = 0.0

    return jr, alpha


def highway_controller_lqr(t: float,
                           z: NDArray,
                           a: int) -> (float, float):
    """ Computes the control inputs based on a desired velocity and heading angle.

    INPUTS
    ------
    t: time (in sec)
    z: state vector (5x1)
    a: index for agent

    OUTPUTS
    -------
    omega: time-derivative of body slip-angle (in rad/sec)
    ar:    rear-wheel acceleration (in m/s^2)

    """
    # from .bicycle_settings.initial_conditions_lane_changing import vi, yi, gl, st


    lane = {0: -LW, 1: 0.0, 2: LW}
    dims = z.shape
    if len(dims) > 1:
        z = z[a]

    # LQR Controller from Black et al. 2022 (https://arxiv.org/pdf/2204.00127v1.pdf)
    vd = vi[a]
    vxd = vd
    vyd = 0.0
    xd = z[0]
    if t > st[a]:
        yd = lane[gl[a]]
    else:
        yd = yi[a]

    q_star = np.array([xd, yd, vxd, vyd])  # desired state
    zeta = np.array([z[0], z[1], f(z)[0], f(z)[1]])  # double integrator state
    tracking_error = zeta - q_star

    if abs(tracking_error[1]) < 1e-6:
        y_gain = 1.0
    else:
        quant = 1 / (abs(tracking_error[1]) + 1e-3)
        y_gain = 3.0 * np.sqrt(quant)

    # y_gain = 100.0

    A_di = np.array([[0, 0, 1, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]])
    B_di = np.array([[0, 0],
                     [0, 0],
                     [1, 0],
                     [0, 1]])
    Q = np.diag([0.001, y_gain, 0.001, 0.01])
    R = np.eye(2)

    # Compute LQR control input for double integrator model
    K, _, _ = lqr(A_di, B_di, Q, R)
    mu = -K @ tracking_error

    # Create transformation matrix
    S = np.array([[-z[3] * np.sin(z[2]) / np.cos(z[4])**2, np.cos(z[2]) - np.sin(z[2]) * np.tan(z[4])],
                  [z[3] * np.cos(z[2]) / np.cos(z[4])**2, np.sin(z[2]) + np.cos(z[2]) * np.tan(z[4])]])

    if z[3] > 0.1:
        vec = np.array([mu[0] + f(z)[1] * f(z)[2], mu[1] - f(z)[0] * f(z)[2]])
        u = np.linalg.inv(S) @ vec
        omega = u[0]
        ar = u[1]
    else:
        omega = 0.0
        ar = np.linalg.norm(mu)

    return omega, ar


def merging_controller_lqr(t: float,
                           z: NDArray,
                           a: int) -> (float, float):
    """ Computes the control inputs based on a desired velocity and heading angle.

    INPUTS
    ------
    t: time (in sec)
    z: state vector (5x1)
    a: index for agent

    OUTPUTS
    -------
    omega: time-derivative of body slip-angle (in rad/sec)
    ar:    rear-wheel acceleration (in m/s^2)

    """
    # Modified LQR Controller from Black et al. 2022 (https://arxiv.org/pdf/2204.00127v1.pdf)

    cruise_speed = 30.0
    dims = z.shape
    if len(dims) > 1:
        z = z[a]

    if mi[a] == 1 and z[0] < -LW:
        vd = cruise_speed
        vxd = vd * np.cos(z[2])
        vyd = vd * np.sin(z[2])
        xd = 0.0
        yd = 0.0
    elif mi[a] == 1:
        vd = cruise_speed
        vxd = vd
        vyd = 0.0
        xd = z[0]
        yd = 0.0
    else:
        vd = vi[a]
        vxd = vd
        vyd = 0.0
        xd = z[0]
        yd = 0.0

    q_star = np.array([xd, yd, vxd, vyd])  # desired state
    zeta = np.array([z[0], z[1], f(z)[0], f(z)[1]])  # double integrator state
    tracking_error = zeta - q_star

    # if abs(tracking_error[1]) < 1e-6:
    #     y_gain = 1.0
    # else:
    #     quant = 1 / (abs(tracking_error[1]) + 1e-3)
    #     y_gain = 3.0 * np.sqrt(quant)

    A_di = np.array([[0, 0, 1, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]])
    B_di = np.array([[0, 0],
                     [0, 0],
                     [1, 0],
                     [0, 1]])
    # Q = np.diag([0.001, y_gain, 0.001, 0.01])
    Q = 100 * np.eye(4)
    R = np.eye(2)

    # Compute LQR control input for double integrator model
    K, _, _ = lqr(A_di, B_di, Q, R)
    mu = -K @ tracking_error

    # Create transformation matrix
    S = np.array([[-z[3] * np.sin(z[2]) / np.cos(z[4])**2, np.cos(z[2]) - np.sin(z[2]) * np.tan(z[4])],
                  [z[3] * np.cos(z[2]) / np.cos(z[4])**2, np.sin(z[2]) + np.cos(z[2]) * np.tan(z[4])]])

    if z[3] > 0.1:
        vec = np.array([mu[0] + f(z)[1] * f(z)[2], mu[1] - f(z)[0] * f(z)[2]])
        u = np.linalg.inv(S) @ vec
        omega = u[0]
        ar = u[1]
    else:
        omega = 0.0
        ar = np.linalg.norm(mu)

    omega = np.clip(omega, -4 * np.pi, 4 * np.pi)
    ar = np.clip(ar, -9.81, 9.81)

    return omega, ar
