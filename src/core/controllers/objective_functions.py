import jax.numpy as jnp
from typing import Optional, Tuple
from nptyping import NDArray

# works with 2*pi, 0.1 limits
# q1 = 1
# q2 = 2

q1 = 1
q2 = 10
q3 = 5 * jnp.array([q1, q2]).max()


def minimum_deviation(
    u_nom: NDArray, x: Optional[NDArray] = None, t: Optional[float] = None
) -> Tuple[NDArray, NDArray]:
    """Constructs a minimum-deviation objective function based on the desired input u_nom.

    Args:
        u_nom (NDArray): nominal solution to qp
        x (Optional[NDArray], optional): state vector (if desired). Defaults to None.

    Returns:
        Tuple[NDArray, NDArray]: Q matrix and p vector for J=1/2u.T@Q@u + p@u
    """
    q1n = 1.0
    q2n = 10.0
    q3n = 50.0
    if t > 2:
        q1n = 100.0
        q2n = 0.01
        q3n = 1e3

    if len(u_nom) % 2 == 0:
        Qlist = int(len(u_nom) / 2) * [q1n, q2n]
    else:
        Qlist = int(len(u_nom) / 2) * [q1n, q2n] + [q3n]

    Q = 1 / 2 * jnp.diag(jnp.array(Qlist))

    p = -Q @ u_nom
    return Q, p
