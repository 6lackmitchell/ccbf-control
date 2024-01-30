import jax.numpy as jnp
from jax import jit, Array
from typing import Optional, Union
from cbfkit.utils.user_types import DynamicsCallable, DynamicsCallableReturns


def plant(k, sigma: Optional[Union[Array, None]] = None, **kwargs) -> DynamicsCallable:
    """
    Returns a function that represents the plant model,
    which computes the drift vector 'f', control matrix 'g', and diffusion matrix
    's' (the argument sigma) based on the given state.

    States are the following:
        #! MANUALLY POPULATE

    Control inputs are the following:
        #! MANUALLY POPULATE

    Args:
        sigma (Optional, Array): diffusion term in stochastic differential equation
        kwargs: keyword arguments

    Returns:
        dynamics (Callable): takes state as input and returns dynamics components
            f, g, and s of the form dx = (f(x) + g(x)u)dt + s(x)dw

    """
    if sigma is not None:
        s = sigma
    else:
        s = jnp.zeros((1, 1))

    @jit
    def dynamics(x: Array) -> DynamicsCallableReturns:
        """
        Computes the drift vector 'f' and control matrix 'g' based on the given state x.

        Args:
            x (Array): state vector

        Returns:
            f, g, s (Tuple of Arrays): drift vector f, control matrix g, diffusion matrix s

        """
        nonlocal s

        f = jnp.array([x[0] * (jnp.exp(k * x[0] ** 2) - 1)])
        g = jnp.array([[(4 - x[0] ** 2)]])

        return f, g, s

    return dynamics
