import jax.numpy as jnp
from jax import jit, Array
from cbfkit.utils.user_types import ControllerCallable, ControllerCallableReturns


def controller_1(
    kv: float,
    period: float,
) -> ControllerCallable:
    """
    Create a controller for the given dynamics.

    Args:
        #! USER-POPULATE

    Returns:
        controller (Callable): handle to function computing control

    """

    @jit
    def controller(t: float, x: Array) -> ControllerCallableReturns:
        """Computes control input (1x1).

        Args:
            t (float): time in sec
            x (Array): state vector (or estimate if using observer/estimator)

        Returns:
            unom (Array): 1x1 vector
            data: (dict): empty dictionary
        """
        # logging data
        u_nom = kv * (4 * jnp.sin(2 * jnp.pi * t / period) - x[0]) - 0.5 * x[0] / (
            4 - x[0] ** 2
        )
        data = {"u_nom": u_nom}

        return jnp.array([u_nom]), data

    return controller
