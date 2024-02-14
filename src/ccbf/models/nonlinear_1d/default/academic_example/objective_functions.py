import numpy as np

q0 = 1.0
q1 = 10 * q0


def objective(u_nom, x=None):
    """Formulates matrix and vector for quadratic objective function."""

    Q = 1 / 2 * np.diag([q0, q1])
    p = -Q @ u_nom

    return Q, p
