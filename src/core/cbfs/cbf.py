import numpy as np
from typing import Callable
from nptyping import NDArray


class Cbf:
    """Object used to package CBFs and their partial derivatives."""

    def __init__(
        self,
        h: Callable,
        dhdt: Callable,
        dhdx: Callable,
        d2hdtdx: Callable,
        d2hdx2: Callable,
        alpha: Callable,
        h0: Callable = None,
    ):
        """Initialization."""
        self._h0 = h0 if h0 is not None else h
        self._h = h
        self._dhdt = dhdt if dhdt is not None else lambda *x: 0
        self._dhdx = dhdx
        self._d2hdtdx = d2hdtdx if d2hdtdx is not None else lambda *x: 0 * dhdx(*x)
        self._d2hdx2 = d2hdx2
        self.alpha = alpha

        # values
        self.h0_value = None
        self.h_value = None
        self.dhdt_value = None
        self.dhdx_value = None
        self.d2hdtdx_value = None
        self.d2hdx2_value = None

        # symbolic expressions
        self.h_sym = None
        self.dhdt_sym = None
        self.dhdx_sym = None
        self.d2hdtdx_sym = None
        self.d2hdx2_sym = None

    def set_symbolics(self, h_sym, dhdt_sym, dhdx_sym, d2hdtdx_sym, d2hdx2_sym) -> None:
        """Sets the symbolic features of the CBF object."""
        self.h_sym = h_sym
        self.dhdt_sym = dhdt_sym
        self.dhdx_sym = dhdx_sym
        self.d2hdtdx_sym = d2hdtdx_sym
        self.d2hdx2_sym = d2hdx2_sym

    def h0(self, *args) -> float:
        self.h0_value = float(self._h0(*args))
        return self.h0_value

    def h(self, *args) -> float:
        self.h_value = float(self._h(*args))
        return self.h_value

    def dhdt(self, *args) -> NDArray:
        self.dhdt_value = self._dhdt(*args)
        return self.dhdt_value

    def dhdx(self, *args) -> NDArray:
        self.dhdx_value = self._dhdx(*args)
        return self.dhdx_value

    def d2hdtdx(self, *args) -> NDArray:
        self.d2hdtdx_value = self._d2hdtdx(*args)
        return self.d2hdtdx_value

    def d2hdx2(self, *args) -> NDArray:
        self.d2hdx2_value = self._d2hdx2(*args)
        return self.d2hdx2_value

    def generate_cbf_condition(
        self, h: float, Lfh: float, Lgh: NDArray, adaptive: bool = False
    ) -> (NDArray, float):
        """Takes the CBF condition of the form
        Lfh + Lgh*u + alpha(h) >= 0
        and converts it into the form
        A*u <= b
        INPUTS
        ------
        alpha: class K function
        OUTPUTS
        -------
        A: left-hand side of Au <= b
        b: right-hand side of Au <= b
        """
        if adaptive:
            alpha_term = -self.alpha(h) if h > 0 else 0
            A = np.concatenate([-Lgh, np.array([alpha_term])])
            b = Lfh
        else:
            A = np.concatenate([-Lgh, np.array([0])])
            b = Lfh + self.alpha(h)

        return A, b

    def generate_stochastic_cbf_condition(
        self, B: float, LfB: float, LgB: NDArray, beta: float, adaptive: bool = False
    ) -> (NDArray, float):
        """Takes the CBF condition of the form
        Lfh + Lgh*u + alpha(h) >= 0
        and converts it into the form
        A*u <= b

        Note: Stochastic CBF uses 1 / alpha(1 / B) so that a more aggressive class K function
        results in more aggressive behavior (AB <= -alpha*B + beta)

        INPUTS
        ------
        alpha: class K function
        OUTPUTS
        -------
        A: left-hand side of Au <= b
        b: right-hand side of Au <= b
        """
        if B > 1e-10:
            alpha_B = 1 / self.alpha(1 / B)
        else:
            alpha_B = 0

        if adaptive:
            A = np.concatenate([LgB, np.array([alpha_B])])
            b = -LfB + beta
        else:
            A = np.concatenate([LgB, np.array([0])])
            b = -LfB - alpha_B + beta

        return A, b
