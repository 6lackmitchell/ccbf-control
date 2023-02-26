import jax.numpy as jnp
from jax import jacfwd, jacrev, jit
from nptyping import NDArray
from core.cbfs.cbf import Cbf


@jit
def h(z: NDArray, cx: float, cy: float, r: float, g: float) -> float:
    """Generic obstacle avoidance constraint function. Super-level set convention.

    Arguments:
        z (NDArray): concatenated time and state vector
        cx (float): x-coordinate of center of obstacle
        cy (float): y-coordinate of center of obstacle
        r (float): radius of obstacle

    Returns:
        ret (float): value of constraint function evaluated at time and state

    """
    ret = (z[1] - (cx + 0.25)) ** 2 + (z[2] - (cy + 0.25)) ** 2 - r**2

    return ret * g


@jit
def dhdz(z: NDArray, cx: float, cy: float, r: float, g: float) -> NDArray:
    """Generic obstacle avoidance Jacobian function.

    Arguments:
        z (NDArray): concatenated time and state vector
        cx (float): x-coordinate of center of obstacle
        cy (float): y-coordinate of center of obstacle
        r (float): radius of obstacle

    Returns:
        ret (NDArray): value of jacobian evaluated at time and state
    """
    return jacfwd(h)(z, cx, cy, r, g)


@jit
def d2hdz2(z: NDArray, cx: float, cy: float, r: float, g: float) -> NDArray:
    """Generic obstacle avoidance Hessian function.

    Arguments:
        z (NDArray): concatenated time and state vector
        cx (float): x-coordinate of center of obstacle
        cy (float): y-coordinate of center of obstacle
        r (float): radius of obstacle

    Returns:
        ret (NDArray): value of Hessian evaluated at time and state
    """
    return jacfwd(jacrev(h))(z, cx, cy, r, g)


def linear_class_k(k):
    def alpha(val):
        return k * val

    return alpha


# Global params
gain = 3.0
R = 0.45

# Obstacle 1
cx1 = 0.8
cy1 = 1.1
g1 = gain
h1 = lambda t, x: h(jnp.hstack([t, x]), cx1, cy1, R, g1)
dh1dt = lambda t, x: dhdz(jnp.hstack([t, x]), cx1, cy1, R, g1)[0]
dh1dx = lambda t, x: dhdz(jnp.hstack([t, x]), cx1, cy1, R, g1)[1:]
d2h1dtdx = lambda t, x: d2hdz2(jnp.hstack([t, x]), cx1, cy1, R, g1)[0, 1:]
d2h1dx2 = lambda t, x: d2hdz2(jnp.hstack([t, x]), cx1, cy1, R, g1)[1:, 1:]

# Obstacle 2
cx2 = 1.25
cy2 = 2.25
g2 = gain * 3.0
h2 = lambda t, x: h(jnp.hstack([t, x]), cx2, cy2, R, g2)
dh2dt = lambda t, x: dhdz(jnp.hstack([t, x]), cx2, cy2, R, g2)[0]
dh2dx = lambda t, x: dhdz(jnp.hstack([t, x]), cx2, cy2, R, g2)[1:]
d2h2dtdx = lambda t, x: d2hdz2(jnp.hstack([t, x]), cx2, cy2, R, g2)[0, 1:]
d2h2dx2 = lambda t, x: d2hdz2(jnp.hstack([t, x]), cx2, cy2, R, g2)[1:, 1:]

# Obstacle 3
cx3 = 2.5
cy3 = 1.75
g3 = gain * 3.0
h3 = lambda t, x: h(jnp.hstack([t, x]), cx3, cy3, R, g3)
dh3dt = lambda t, x: dhdz(jnp.hstack([t, x]), cx3, cy3, R, g3)[0]
dh3dx = lambda t, x: dhdz(jnp.hstack([t, x]), cx3, cy3, R, g3)[1:]
d2h3dtdx = lambda t, x: d2hdz2(jnp.hstack([t, x]), cx3, cy3, R, g3)[0, 1:]
d2h3dx2 = lambda t, x: d2hdz2(jnp.hstack([t, x]), cx3, cy3, R, g3)[1:, 1:]

# Obstacle 4
cx4 = 2.0
cy4 = 0.25
g4 = gain
h4 = lambda t, x: h(jnp.hstack([t, x]), cx4, cy4, R, g4)
dh4dt = lambda t, x: dhdz(jnp.hstack([t, x]), cx4, cy4, R, g4)[0]
dh4dx = lambda t, x: dhdz(jnp.hstack([t, x]), cx4, cy4, R, g4)[1:]
d2h4dtdx = lambda t, x: d2hdz2(jnp.hstack([t, x]), cx4, cy4, R, g4)[0, 1:]
d2h4dx2 = lambda t, x: d2hdz2(jnp.hstack([t, x]), cx4, cy4, R, g4)[1:, 1:]

# Obstacle 5
cx5 = 0.8
cy5 = -0.25
g5 = gain
h5 = lambda t, x: h(jnp.hstack([t, x]), cx5, cy5, R, g5)
dh5dt = lambda t, x: dhdz(jnp.hstack([t, x]), cx5, cy5, R, g5)[0]
dh5dx = lambda t, x: dhdz(jnp.hstack([t, x]), cx5, cy5, R, g5)[1:]
d2h5dtdx = lambda t, x: d2hdz2(jnp.hstack([t, x]), cx5, cy5, g5)[0, 1:]
d2h5dx2 = lambda t, x: d2hdz2(jnp.hstack([t, x]), cx5, cy5, R, g5)[1:, 1:]


cbf1 = Cbf(h1, None, dh1dx, None, d2h1dx2, linear_class_k(1.0))
cbf2 = Cbf(h2, None, dh2dx, None, d2h2dx2, linear_class_k(1.0))
cbf3 = Cbf(h3, None, dh3dx, None, d2h3dx2, linear_class_k(1.0))
cbf4 = Cbf(h4, None, dh4dx, None, d2h4dx2, linear_class_k(1.0))
cbf5 = Cbf(h5, None, dh5dx, None, d2h5dx2, linear_class_k(1.0))

cbfs = [
    cbf1,
    cbf2,
    cbf3,
    cbf4,
    cbf5,
]


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
