import jax.numpy as jnp
from nptyping import NDArray
from typing import Optional
from ..model import Model

K = jnp.sqrt(jnp.log(2)) / 1.5616


class Unstable1DModel(Model):
    """Class for an unstable 1d nonlinear model, the dynamics of which are as follows:

    xdot = x*(exp((kx)^2) - 1) + (4 - x^2) * u

    with state x

    and control input u.
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
        self.k = K

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

        xdot = x*(exp((kx)^2) - 1

        Arguments:
            z (NDArray): concatenated time t and state vector x

        Returns:
            drift (NDArray): drift dynamics of the state

        """
        x = z[1]
        drift = jnp.array([x * (jnp.exp((self.k * x) ** 2) - 1)])

        return drift

    def _g(self, z: NDArray) -> NDArray:
        """Control matrix for system dynamics.

        x_dot = (4 - x^2)

        Arguments:
            z (NDArray): concatenated time and state vector

        Returns:
            drift (NDArray): drift dynamics of the state

        """
        x = z[1]
        control_mat = jnp.array([[4 - x**2]])

        return control_mat
