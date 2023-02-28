"""nominal_controllers.py

Contains classes for different nominal (safety-agnostic) controllers
for the bicycle model.

"""
import jax.numpy as jnp
from jax import jit
from nptyping import NDArray
from typing import Tuple
from core.controllers.controller import Controller


class SinController(Controller):
    """_summary_

    Args:
        Controller (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        self.id = None
        self.complete = False

        self.setup()

    def setup(self) -> None:
        """Sets up Sin controller."""
        # params
        kv = 2.0
        period = 5.0

        def control_expr(z):
            xdes = 4 * jnp.sin(2 * jnp.pi * z[0] / period)
            vdes = (kv * (xdes - z[1]) - 1 / 2 * z[1]) / ((2 - z[1]) * (z[1] + 2))

            return vdes

        self.control = jit(control_expr)

    def _compute_control(self, t: float, z: NDArray) -> Tuple[int, str]:
        """_summary_

        Args:
            t (float): _description_
            z (NDArray): _description_

        Returns:
            Tuple[int, str]: _description_
        """
        kv = 2.0
        x = z[self.id, 0]

        period = 5.0
        xdes = 4 * jnp.sin(2 * jnp.pi * t / period)
        vdes = (kv * (xdes - x) - 1 / 2 * x) / ((2 - x) * (x + 2))

        self.u = jnp.array(vdes)

        return self.u, 1, "Optimal"


# class SinController(Controller):
#     """_summary_

#     Args:
#         Controller (_type_): _description_
#     """

#     def __init__(self):
#         super().__init__()
#         self.id = None
#         self.complete = False

#     def _compute_control(self, t: float, z: NDArray) -> Tuple[int, str]:
#         """_summary_

#         Args:
#             t (float): _description_
#             z (NDArray): _description_

#         Returns:
#             Tuple[int, str]: _description_
#         """
#         kv = 2.0
#         x = z[self.id, 0]

#         period = 5.0
#         xdes = 4 * jnp.sin(2 * jnp.pi * t / period)
#         vdes = (kv * (xdes - x) - 1 / 2 * x) / ((2 - x) * (x + 2))

#         self.u = jnp.array(vdes)

#         return self.u, 1, "Optimal"
