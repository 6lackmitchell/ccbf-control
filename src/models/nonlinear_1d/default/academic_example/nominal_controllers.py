import numpy as np

from typing import Tuple
from nptyping import NDArray
from core.controllers.controller import Controller

# from ..physical_params import u_max
from .initial_conditions import xg


class ProportionalController(Controller):
    """_summary_

    Args:
        Controller (_type_): _description_
    """

    def __init__(self, ego_id: int):
        super().__init__()
        self.ego_id = ego_id
        self.complete = False

    def _compute_control(self, t: float, z: NDArray) -> Tuple[int, str]:
        """_summary_

        Args:
            t (float): _description_
            z (NDArray): _description_

        Returns:
            Tuple[int, str]: _description_
        """
        kv = 2.0

        period = 5.0
        xdes = -4 * np.sin(2 * np.pi * t / period)
        vdes = (kv * (xdes - z[self.ego_id, 0]) - 1 / 2 * z[self.ego_id, 0]) / (
            (2 - z[self.ego_id, 0]) * (z[self.ego_id, 0] + 2)
        )

        self.u = np.array([vdes])

        return self.u, 1, "Optimal"
