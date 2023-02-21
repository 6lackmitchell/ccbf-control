"""nominal_controllers.py

Contains classes for different nominal (safety-agnostic) controllers
for the bicycle model.

"""
import jax.numpy as jnp
from control import lqr
from nptyping import NDArray
from models.model import Model
from core.controllers.controller import Controller


class LqrController(Controller):
    def __init__(self, xg: NDArray, model: Model):
        super().__init__()
        self.xg = xg
        self.model = model
        self.u_max = model.u_max

        self.id = None
        self.complete = False
        self.u_actual = None

    def _compute_control(self, t: float, z: NDArray) -> (int, str):
        """Computes the nominal input for a vehicle in the intersection situation.

        Arguments:
            t (float): time in sec
            z (NDArray): full state vector

        Returns:
            code: success (1) / error (0, -1, ...) code
            status: more informative success / error flag

        """
        # Modified LQR Controller from Black et al. 2022 (https://arxiv.org/pdf/2204.00127v1.pdf)
        ze = z[self.id]
        model_f = self.model.f()

        xd = self.xg[0]
        yd = self.xg[1]

        speed_d = 0.25
        pos_err = jnp.array([ze[0] - xd, ze[1] - yd])
        vd = speed_d * jnp.array([1, 1 / 2 * jnp.linalg.norm(pos_err)]).min()
        th = jnp.arctan2(yd - ze[1], xd - ze[0])
        vxd = vd * jnp.cos(th)
        vyd = vd * jnp.sin(th)

        q_star = jnp.array([xd, yd, vxd, vyd])  # desired state
        zeta = jnp.array([ze[0], ze[1], model_f[0], model_f[1]])  # double integrator state
        tracking_error = zeta - q_star
        if jnp.linalg.norm(tracking_error[:2]) < (0.075):
            tracking_error = jnp.array([0, 0, model_f[0], model_f[1]])
            self.complete = True

        A_di = jnp.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        B_di = jnp.array([[0, 0], [0, 0], [1, 0], [0, 1]])

        gain = 1.0 / (0.01 + (tracking_error[0]) ** 2 + (tracking_error[1]) ** 2)
        gain = jnp.array([gain, 10.0]).min()
        Q = 0.5 * gain * jnp.eye(4)
        R = 5 * jnp.eye(2)

        # Compute LQR control ijnput for double integrator model
        K, _, _ = lqr(A_di, B_di, Q, R)
        mu = -K @ tracking_error

        # Create transformation matrix
        S = jnp.array(
            [
                [
                    -ze[3] * jnp.sin(ze[2]) / jnp.cos(ze[4]) ** 2,
                    jnp.cos(ze[2]) - jnp.sin(ze[2]) * jnp.tan(ze[4]),
                ],
                [
                    ze[3] * jnp.cos(ze[2]) / jnp.cos(ze[4]) ** 2,
                    jnp.sin(ze[2]) + jnp.cos(ze[2]) * jnp.tan(ze[4]),
                ],
            ]
        )

        if ze[3] > 0.01:
            vec = jnp.array([mu[0] + model_f[1] * model_f[2], mu[1] - model_f[0] * model_f[2]])
            u = jnp.linalg.inv(S) @ vec
            omega = u[0]
            ar = u[1]
        else:
            omega = 0.0
            theta = jnp.arctan2(mu[1], mu[0]) - ze[2]
            sign_ar = 1 if (-jnp.pi / 2 < theta < jnp.pi / 2) else -1
            ar = jnp.linalg.norm(mu) * sign_ar

        omega = jnp.clip(omega, -self.u_max[0], self.u_max[0])
        ar = jnp.clip(ar, -self.u_max[1], self.u_max[1])

        # # To reduce potential chattering
        # beta = 0.25
        # if self.u_actual is not None:
        #     omega = beta * self.u_actual[0] + (1 - beta) * omega
        # ar = beta * self.u_actual[1] + (1 - beta) * ar

        self.u = jnp.array([omega, ar])

        return self.u, 1, "Optimal"
