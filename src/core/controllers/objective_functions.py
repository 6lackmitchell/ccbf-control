import jax.numpy as jnp
from typing import Optional, Tuple
from nptyping import NDArray

q1 = 1.0
q2 = 1.0
q3 = 100 * jnp.array([q1, q2]).max()


def minimum_deviation(u_nom: NDArray, x: Optional[NDArray] = None) -> Tuple[NDArray, NDArray]:
    """Constructs a minimum-deviation objective function based on the desired input u_nom.

    Args:
        u_nom (NDArray): nominal solution to qp
        x (Optional[NDArray], optional): state vector (if desired). Defaults to None.

    Returns:
        Tuple[NDArray, NDArray]: Q matrix and p vector for J=1/2u.T@Q@u + p@u
    """

    if len(u_nom) % 2 == 0:
        Q = jnp.diag(int(len(u_nom) / 2) * [q1, q2])
    else:
        Q = jnp.diag(int(len(u_nom) / 2) * [q1, q2] + [q3])

    Q = 1 / 2 * Q

    p = -Q @ u_nom
    return Q, p
