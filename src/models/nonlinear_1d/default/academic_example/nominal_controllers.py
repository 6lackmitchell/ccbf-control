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
        kv = 1.0
        vdes = -kv * (xg[self.ego_id] - z[self.ego_id, 0])

        self.u = np.array([vdes])

        return self.u, 1, "Optimal"
