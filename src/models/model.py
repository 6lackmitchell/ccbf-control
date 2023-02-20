"""model.py

Contains class Model, parent class for all dynamical system models.

"""
import jax.numpy as jnp
from jax import jacfwd
from nptyping import NDArray
from typing import Optional


class Model:
    """Generic class for dynamics models.

    Class Attributes:
        t (float): time
        x (NDArray): state vector
        u (NDArray): control vector

    Class Methods
        dynamics: computes xdot as function of t, x, u
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
            centralized (bool, optional): true if model belongs to group of centrally controlled agents

        """
        # time params
        self.t = 0.0
        self.dt = dt
        self.tf = tf

        # state
        self.x = initial_state

        # controls
        self.u = 0 * u_max
        self.u_max = u_max
        self.centralized = centralized

        # dimensions
        self.n_states = len(self.x)
        self.n_controls = len(u_max)

    def _deterministic_nonlinear_dynamics(self) -> NDArray:
        """Computes the state derivative as a function of the time (t), state (x), and input (u).

        Arguments:
            None -- all required information is internal

        Returns:
            xdot (NDArray): time derivative of the state

        """

        xdot = self.F()

        return xdot

    def _deterministic_control_affine_dynamics(self) -> NDArray:
        """Computes the state derivative as a function of the time (t), state (x), and input (u).

        Arguments:
            None -- all required information is internal

        Returns:
            xdot (NDArray): time derivative of the state

        """

        xdot = self.f() + self.g() @ self.u

        return xdot

    def F(self, z=None):
        if z is None:
            z = jnp.hstack([self.t, self.x])
        return self._F(z)

    def f(self, z=None):
        if z is None:
            z = jnp.hstack([self.t, self.x])
        return self._f(z)

    def g(self, z=None):
        if z is None:
            z = jnp.hstack([self.t, self.x])
        return self._g(z)

    def dfdt(self, z=None):
        if z is None:
            z = jnp.hstack([self.t, self.x])
        return self._dfdz(z)[:, 0]

    def dfdx(self, z=None):
        if z is None:
            z = jnp.hstack([self.t, self.x])
        return self._dfdz(z)[:, 1:]

    def dgdt(self, z=None):
        if z is None:
            z = jnp.hstack([self.t, self.x])
        return self._dgdz(z)[:, 0]

    def dgdx(self, z=None):
        if z is None:
            z = jnp.hstack([self.t, self.x])
        return self._dgdz(z)[:, 1:]

    def _F(self, z: NDArray) -> NDArray:
        """Generic nonlinear system dynamics. This will be overloaded by child model.

        Arguments:
            z (NDArray): concatenated time, state, and control vectors

        Returns:
            xdot (NDArray): time derivative of state

        """
        return jnp.zeros((self.n_states,))

    def _f(self, z: NDArray) -> NDArray:
        """Drift term for nonlinear, control-affine system dynamics. This will be overloaded by child model.

        Arguments:
            z (NDArray): concatenated time and state vector

        Returns:
            xdot_drift (NDArray): drift term in time derivative of state

        """
        return jnp.zeros((self.n_states,))

    def _g(self, z: NDArray) -> NDArray:
        """Control matrix term for nonlinear, control-affine system dynamics. This will be overloaded by child model.

        Arguments:
            z (NDArray): concatenated time and state vector

        Returns:
            xdot_control_matrix (NDArray): control matrix for time derivative of state

        """
        return jnp.zeros((self.n_states, self.n_controls))

    def _dfdz(self, z: NDArray) -> NDArray:
        """Partial derivative of the drift vector with respect to time.

        Arguments:
            z (NDArray): concatenated time and state vector

        Returns:
            dfdt (NDArray): partial f partial t

        """
        return jacfwd(self._f)(z)

    def _dgdz(self, z: NDArray) -> NDArray:
        """Partial derivative of the drift vector with respect to time.

        Arguments:
            z (NDArray): concatenated time and state vector

        Returns:
            dfdt (NDArray): partial f partial t

        """
        return jacfwd(self._g)(z)


if __name__ == "__main__":
    from bicycle.dynamic_bicycle_model import RearDriveDynamicBicycleModel

    x0 = jnp.array([1.0, 0.0, 0.0, 1.0, 0.0])
    n_controls = 2

    bicycle = RearDriveDynamicBicycleModel(x0, n_controls)
    print(bicycle.xdot())
    print(bicycle.dfdt())
    print(bicycle.dfdx())
