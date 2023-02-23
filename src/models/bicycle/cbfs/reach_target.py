import jax.numpy as jnp
from jax import jacfwd, jacrev, jit
from nptyping import NDArray
from core.cbfs.cbf import Cbf

gain = 0.25


@jit
def h(z: NDArray, cx: float, cy: float, ri: float, rf: float, T: float) -> float:
    """Generic speed limit constraint function. Super-level set convention.

    Arguments:
        z (NDArray): concatenated time and state vector
        slip_limit (float): speed limit in m/s

    Returns:
        ret (float): value of constraint function evaluated at time and state

    """
    ret = rf**2 + ri**2 * (1 - z[0] / T) - (z[1] - cx) ** 2 - (z[2] - cy) ** 2

    return ret * gain


@jit
def dhdz(z: NDArray, cx: float, cy: float, ri: float, rf: float, T: float) -> NDArray:
    """Generic speed limit Jacobian function.

    Arguments:
        z (NDArray): concatenated time and state vector
        slip_limit (float): speed limit in m/s

    Returns:
        ret (NDArray): value of jacobian evaluated at time and state
    """
    return jacfwd(h)(z, cx, cy, ri, rf, T)


@jit
def d2hdz2(z: NDArray, cx: float, cy: float, ri: float, rf: float, T: float) -> NDArray:
    """Generic obstacle avoidance Hessian function.

    Arguments:
        z (NDArray): concatenated time and state vector
        slip_limit (float): speed limit in m/s

    Returns:
        ret (NDArray): value of Hessian evaluated at time and state
    """
    return jacfwd(jacrev(h))(z, cx, cy, ri, rf, T)


def linear_class_k(k):
    def alpha(val):
        return k * val

    return alpha


# Speed Constraint
ri = 4.0
rf = 0.1
T = 5.0
cx = 2.0
cy = 2.0
h1 = lambda t, x: h(jnp.hstack([t, x]), cx, cy, ri, rf, T)
dh1dt = lambda t, x: dhdz(jnp.hstack([t, x]), cx, cy, ri, rf, T)[0]
dh1dx = lambda t, x: dhdz(jnp.hstack([t, x]), cx, cy, ri, rf, T)[1:]
d2h1dtdx = lambda t, x: d2hdz2(jnp.hstack([t, x]), cx, cy, ri, rf, T)[0, 1:]
d2h1dx2 = lambda t, x: d2hdz2(jnp.hstack([t, x]), cx, cy, ri, rf, T)[1:, 1:]

cbf = Cbf(h1, None, dh1dx, None, d2h1dx2, linear_class_k(1.0))

cbfs = [cbf]


if __name__ == "__main__":
    # This is a unit test
    x = jnp.array([0.25, 0.0, 0.0, 1.0, 0.2])
    t = 1.0
    z = jnp.hstack([t, x])

    print(h1(t, x))
    print(dh1dt(t, x))
    print(dh1dx(t, x))
    print(d2h1dtdx(t, x))
    print(d2h1dx2(t, x))
