import numpy as np
from typing import Callable
from nptyping import NDArray


###############################################################################
#################################### Class ####################################
###############################################################################


class Agent:

    def __init__(self,
                 x0: NDArray,
                 dynamics: Callable,
                 controller: Callable):
        self.t = 0.0
        self.x = x0
        self.u = None
        self.u_nom = None
        self.dynamics = dynamics
        self.controller = controller

    def compute_control(self,
                        t: float,
                        x: NDArray,
                        extras: dict) -> (NDArray, NDArray, int, str):
        """ Computes the control input for the Agent.

        INPUTS
        ------
        t: time (in sec)
        x: state vector
        extras: any extra parameters the specified controller needs

        OUTPUTS
        -------
        u_actual: the control input implemented in the system dynamics
        u_nominal: the control input computed by the nominal (potentially unsafe) controller

        """
        self.t = t
        code = None
        status = None
        mas = None
        eta = None

        # Compute control inputs
        outputs = self.controller(t, x, extras)
        if len(outputs) == 2:
            self.u = np.array(outputs)
            self.u_nom = self.u
            code = 1
            status = 'optimal'

        elif len(outputs) == 5:
            self.u = outputs[0]
            self.u_nom = outputs[1]
            code = outputs[2]
            status = outputs[3]
            mas = outputs[4]

        elif len(outputs) == 6:
            self.u = outputs[0]
            self.u_nom = outputs[1]
            code = outputs[2]
            status = outputs[3]
            mas = outputs[4]
            eta = outputs[5]

        return self.u, self.u_nom, code, status, mas

    def step_dynamics(self) -> NDArray:
        """ Advances the Agent's dynamics forward in time using the current state and control input.

        INPUTS
        ------
        None -- Requires only class variables

        OUTPUTS
        -------
        x_updated: new state vector

        """
        x_updated = self.dynamics(self.t, self.x, self.u)
        self.update(x_updated)

        return x_updated

    def update(self,
               x_new: NDArray) -> None:
        """ Updates the class time and state vector.

        INPUTS
        ------
        x_new: new state vector

        OUTPUTS
        -------
        None

        """
        self.x = x_new
