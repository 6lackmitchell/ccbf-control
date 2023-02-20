import jax.numpy as jnp
from nptyping import NDArray
from typing import Optional
from ..model import Model

# Physical Parameters
LR = 1.0


class RearDriveDynamicBicycleModel(Model):
    """Class for the dynamic bicycle, the dynamics of which are as follows:

    x_dot = vr * (cos(psi) - sin(psi) * tan(beta))
    y_dot = vr * (sin(psi) + cos(psi) * tan(beta))
    psi_dot = vr * tan(beta) / lr
    vr_dot = a
    beta_dot = omega

    with states x (lateral position), y (longitudinal position), psi (heading angle),
    vr (rear wheel velocity), and beta (slip angle)

    and control inputs a (rear wheel acceleration) and omega (slip angle rate)
    """

    def __init__(
        self,
        initial_state: NDArray,
        u_max: NDArray,
        dt: float,
        tf: float,
        centralized: Optional[bool] = False,
    ):
        """Class constructor.

        Arguments:
            initial_state (NDArray): initial state vector at t0
            u_max (NDArray): maximum control inputs
            dt (float): length of timestep in sec
            tf (float): final time in sec

        """
        super().__init__(initial_state, u_max, dt, tf, centralized)
        self.lr = LR

    def xdot(self) -> NDArray:
        """Computes the state derivative as a function of the time (t), state (x), and input (u).

        Arguments:
            None -- all required information is internal

        Returns:
            xdot (NDArray): time derivative of the state

        """
        return self._deterministic_control_affine_dynamics()

    def _f(self, z: NDArray) -> NDArray:
        """Drift (uncontrolled) dynamics.

        x_dot = vr * (cos(psi) - sin(psi) * tan(beta))
        y_dot = vr * (sin(psi) + cos(psi) * tan(beta))
        psi_dot = vr * tan(beta) / lr
        vr_dot = 0
        beta_dot = 0

        Arguments:
            z (NDArray): concatenated time t and state vector x

        Returns:
            drift (NDArray): drift dynamics of the state

        """
        psi, vr, beta = z[3], z[4], z[5]
        drift = jnp.array(
            [
                vr * (jnp.cos(psi) - jnp.sin(psi) * jnp.tan(beta)),
                vr * (jnp.sin(psi) + jnp.cos(psi) * jnp.tan(beta)),
                vr * jnp.tan(beta) / self.lr,
                0.0,
                0.0,
            ]
        )

        return drift

    def _g(self, z: NDArray) -> NDArray:
        """Control matrix for system dynamics.

        x_dot = [0, 0]
        y_dot = [0, 0]
        psi_dot = [0, 0]
        vr_dot = [0, 1]
        beta_dot = [1, 0]

        Arguments:
            z (NDArray): concatenated time and state vector

        Returns:
            drift (NDArray): drift dynamics of the state

        """
        control_mat = jnp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )

        return control_mat
