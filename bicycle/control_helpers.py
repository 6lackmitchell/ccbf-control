import numpy as np
import sympy as sp


def saturate_solution(sol):
    """Applies saturation function to control solution. """
    saturated_sol = np.zeros((len(sol),))
    for ii, s in enumerate(sol):
        saturated_sol[ii] = np.clip(s.x, s.lb, s.ub)

    return saturated_sol


def sigmoid(x, k, d):
    return 0.5 * (1 + sp.tanh(k * (x - d)))


def heavyside_approx(x: float,
                     k: float,
                     d: float) -> float:
    """ Approximation to the unit heavyside function.

    INPUTS
    ------
    x: independent variable
    k: scale factor
    d: offset factor

    OUTPUTS
    -------
    y = 1/2 + 1/2 * tanh(k * (x - d))

    """
    return 0.5 * (1 + np.tanh(k * (x - d)))


def dheavyside_approx(x: float,
                      k: float,
                      d: float) -> float:
    """ Derivative of approximation to the unit heavyside function.

    INPUTS
    ------
    x: independent variable
    k: scale factor
    d: offset factor

    OUTPUTS
    -------
    dy = k/2 * (sech(k * (x - d))) ** 2

    """
    return k / (2 * (np.cosh(k * (x - d)))**2)
