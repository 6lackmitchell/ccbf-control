import jax.numpy as jnp
from jax import jacfwd, jacrev, jit
from nptyping import NDArray
from core.cbfs.cbf import Cbf


@jit
def h(z: NDArray, x: float) -> float:
    """Generic obstacle avoidance constraint function. Super-level set convention.

    Arguments:
        z (NDArray): concatenated time and state vector
        x (float): x-coordinate of obstacle

    Returns:
        ret (float): value of constraint function evaluated at time and state

    """
    ret = (x > 0) * (x - z[1]) + (x <= 0) * (z[1] - x)

    return ret * gain


@jit
def dhdz(z: NDArray, x: float) -> NDArray:
    """Generic obstacle avoidance Jacobian function.

    Arguments:
        z (NDArray): concatenated time and state vector
        x (float): x-coordinate of obstacle

    Returns:
        ret (NDArray): value of jacobian evaluated at time and state
    """
    return jacfwd(h)(z, x)


@jit
def d2hdz2(z: NDArray, x: float) -> NDArray:
    """Generic obstacle avoidance Hessian function.

    Arguments:
        z (NDArray): concatenated time and state vector
        x (float): x-coordinate of obstacle

    Returns:
        ret (NDArray): value of Hessian evaluated at time and state
    """
    return jacfwd(jacrev(h))(z, x)


def linear_class_k(k):
    def alpha(val):
        return k * val

    return alpha


# Global params
gain = 1.0

# Obstacle 1
x1 = 2.0
h1 = lambda t, x: h(jnp.hstack([t, x]), x1)
dh1dt = lambda t, x: dhdz(jnp.hstack([t, x]), x1)[0]
dh1dx = lambda t, x: dhdz(jnp.hstack([t, x]), x1)[1:]
d2h1dtdx = lambda t, x: d2hdz2(jnp.hstack([t, x]), x1)[0, 1:]
d2h1dx2 = lambda t, x: d2hdz2(jnp.hstack([t, x]), x1)[1:, 1:]

# Obstacle 2
x2 = -2.0
h2 = lambda t, x: h(jnp.hstack([t, x]), x2)
dh2dt = lambda t, x: dhdz(jnp.hstack([t, x]), x2)[0]
dh2dx = lambda t, x: dhdz(jnp.hstack([t, x]), x2)[1:]
d2h2dtdx = lambda t, x: d2hdz2(jnp.hstack([t, x]), x2)[0, 1:]
d2h2dx2 = lambda t, x: d2hdz2(jnp.hstack([t, x]), x2)[1:, 1:]

# cbfs
cbf1 = Cbf(h1, None, dh1dx, None, d2h1dx2, linear_class_k(1.0))
cbf2 = Cbf(h2, None, dh2dx, None, d2h2dx2, linear_class_k(1.0))

cbfs = [cbf1, cbf2]


if __name__ == "__main__":
    # This is a unit test
    x = jnp.array([0.5, 1.0, 0.0, 0.0, 0.0])
    t = 10.0
    z = jnp.hstack([t, x])

    print(h1(t, x))
    print(h2(t, x))
    print(h3(t, x))
    print(h4(t, x))
    print(h5(t, x))

    print(dh1dx(t, x))
    print(dh2dx(t, x))
    print(dh3dx(t, x))
    print(dh4dx(t, x))
    print(dh5dx(t, x))
