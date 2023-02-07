import numpy as np
from .physical_params import u_max

# q0 = 1.0 / u_max[0]**2
# q1 = 100.0 / u_max[1]**2
# q2 = 2 * q0

q0 = 1 / u_max[0] ** 2
q1 = 100 / u_max[1] ** 2
q2 = 100 * np.max([q0, q1])


def objective_accel_and_steering(u_nom, x=None):

    # if len(u_nom) % 2 == 0:
    #     Q = np.diag(
    #         int(len(u_nom) / 2)
    #         * [np.max([1, 100 * (2 - x[0]) ** 3]), np.max([1, (100 * x[1]) ** 3])]
    #     )
    # else:
    #     Q = np.diag(
    #         int(len(u_nom) / 2) * [np.max([1, 100 * (2 - x[0])]), np.max([1, 100 * x[1]])] + [q2]
    #     )

    exp = 4
    # q0_new = q0 * ((x[0] - 2) / (x[1] - 2)) ** exp
    # q1_new = q1 * ((x[1] - 2) / (x[0] - 2)) ** exp
    # if abs(x[0] - 2) > abs(x[1] - 2):
    #     q0_new = 1e3
    #     q1_new = 1e-3
    # else:
    #     q0_new = 1e-3
    #     q1_new = 1e3

    vdes = np.array([2 - x[0], 2 - x[1]])
    q0_new = q0  # * ((x[2] - vdes[0]) / (x[3] - vdes[1])) ** exp
    q1_new = q1  # * ((x[3] - vdes[1]) / (x[2] - vdes[0])) ** exp

    if len(u_nom) % 2 == 0:
        Q = np.diag(int(len(u_nom) / 2) * [q0_new, q1_new])
    else:
        Q = np.diag(int(len(u_nom) / 2) * [q0_new, q1_new] + [q2])

    Q = 1 / 2 * Q

    p = -Q @ u_nom
    return Q, p


def objective_accel_only(u_nom, agent):
    Q = q0 * np.eye(len(u_nom))
    Q[agent, agent] = Q[agent, agent] / 50.0
    p = -q0 * u_nom
    return 1 / 2 * Q, p
