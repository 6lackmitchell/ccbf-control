import numpy as np
from control import lqr
from nptyping import NDArray
from core.controllers.controller import Controller
from bicycle.dynamic.physical_params import LW, u_max
from bicycle.dynamic.models import f
from bicycle.dynamic.intersection.initial_conditions import *


class LqrController(Controller):

    def __init__(self,
                 ego_id: int):
        super().__init__()
        self.ego_id = ego_id

    def _compute_control(self,
                         t: float,
                         z: NDArray) -> (int, str):
        """Computes the nominal input for a vehicle in the intersection situation.

        INPUTS
        ------
        t: time (in sec)
        z: full state vector

        OUTPUTS
        -------
        code: success (1) / error (0, -1, ...) code
        status: more informative success / error flag

        """
        # Modified LQR Controller from Black et al. 2022 (https://arxiv.org/pdf/2204.00127v1.pdf)
        ze = z[self.ego_id]

        # if t == 0.0 and self.ego_id > 2:
        #     vi[self.ego_id] = cruise_speed + np.random.uniform(low=-1.0, high=1.0)

        xd = xg[self.ego_id]
        yd = yg[self.ego_id]
        vxd = xd - ze[0]
        vyd = yd - ze[1]

        q_star = np.array([xd, yd, vxd, vyd])  # desired state
        zeta = np.array([ze[0], ze[1], f(ze)[0], f(ze)[1]])  # double integrator state
        tracking_error = zeta - q_star

        A_di = np.array([[0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]])
        B_di = np.array([[0, 0],
                         [0, 0],
                         [1, 0],
                         [0, 1]])

        gain = np.min([0.01 / (0.01 + (tracking_error[0])**2 + (tracking_error[1])**2), 1.0])
        gain = 0.001
        Q = gain * np.eye(4)
        R = np.eye(2)

        # Compute LQR control input for double integrator model
        K, _, _ = lqr(A_di, B_di, Q, R)
        mu = -K @ tracking_error

        # Create transformation matrix
        S = np.array([[-ze[3] * np.sin(ze[2]) / np.cos(ze[4]) ** 2, np.cos(ze[2]) - np.sin(ze[2]) * np.tan(ze[4])],
                      [ze[3] * np.cos(ze[2]) / np.cos(ze[4]) ** 2, np.sin(ze[2]) + np.cos(ze[2]) * np.tan(ze[4])]])

        if ze[3] > 0.05:
            vec = np.array([mu[0] + f(ze)[1] * f(ze)[2], mu[1] - f(ze)[0] * f(ze)[2]])
            u = np.linalg.inv(S) @ vec
            omega = u[0]
            ar = u[1]
        else:
            omega = 0.0
            ar = np.linalg.norm(mu)

        omega = np.clip(omega, -u_max[0], u_max[0])
        ar = np.clip(ar, -u_max[1], u_max[1])

        self.u = np.array([omega, ar])

        return self.u, 1, "Optimal"
