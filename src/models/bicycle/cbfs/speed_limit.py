import jax.numpy as jnp
from jax import jacfwd, jacrev, jit
from nptyping import NDArray
from core.cbfs.cbf import Cbf

gain = 1.0


@jit
def h(z: NDArray, speed_limit: float) -> float:
    """Generic speed limit constraint function. Super-level set convention.

    Arguments:
        z (NDArray): concatenated time and state vector
        speed_limit (float): speed limit in m/s

    Returns:
        ret (float): value of constraint function evaluated at time and state

    """
    ret = (speed_limit - z[4]) * (speed_limit + z[4])

    return ret * gain


@jit
def dhdz(z: NDArray, speed_limit: float) -> NDArray:
    """Generic speed limit Jacobian function.

    Arguments:
        z (NDArray): concatenated time and state vector
        speed_limit (float): speed limit in m/s

    Returns:
        ret (NDArray): value of jacobian evaluated at time and state
    """
    return jacfwd(h)(z, speed_limit)


@jit
def d2hdz2(z: NDArray, speed_limit: float) -> NDArray:
    """Generic obstacle avoidance Hessian function.

    Arguments:
        z (NDArray): concatenated time and state vector
        speed_limit (float): speed limit in m/s

    Returns:
        ret (NDArray): value of Hessian evaluated at time and state
    """
    return jacfwd(jacrev(h))(z, speed_limit)


def linear_class_k(k):
    def alpha(val):
        return k * val

    return alpha


# Speed Constraint
speed_limit = 2.0
h1 = lambda t, x: h(jnp.hstack([t, x]), speed_limit)
dh1dt = lambda t, x: dhdz(jnp.hstack([t, x]), speed_limit)[0]
dh1dx = lambda t, x: dhdz(jnp.hstack([t, x]), speed_limit)[1:]
d2h1dtdx = lambda t, x: d2hdz2(jnp.hstack([t, x]), speed_limit)[0, 1:]
d2h1dx2 = lambda t, x: d2hdz2(jnp.hstack([t, x]), speed_limit)[1:, 1:]

cbf = Cbf(h1, None, dh1dx, None, d2h1dx2, linear_class_k(1.0))

cbfs = [cbf]


if __name__ == "__main__":
    # This is a unit test
    x = jnp.array([0.25, 0.0, 0.0, 1.0, 0.0])
    t = 1.0
    z = jnp.hstack([t, x])

    print(h1(t, x))
    print(dh1dt(t, x))
    print(dh1dx(t, x))
    print(d2h1dtdx(t, x))
    print(d2h1dx2(t, x))
