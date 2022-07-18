from typing import Callable
from nptyping import NDArray


class Cbf():
    """ Object used to package CBFs and their partial derivatives. """

    def __init__(self,
                 h: Callable,
                 dhdx: Callable,
                 d2hdx2: Callable,
                 alpha: Callable):
        """ Initialization. """
        self.h = h
        self.dhdx = dhdx
        self.d2hdx2 = d2hdx2
        self.alpha = alpha

    def generate_cbf_condition(self,
                               h: float,
                               Lfh: float,
                               Lgh: NDArray) -> (NDArray, float):
        """ Takes the CBF condition of the form

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
        A = -Lgh
        b = Lfh + self.alpha(h)

        return A, b
