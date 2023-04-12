import numpy as np
from typing import Callable, List, Optional
from nptyping import NDArray
from pathlib import Path
from pickle import dump, load
from .controllers.controller import Controller
from models.model import Model


class Agent:

    t = 0.0

    @property
    def timestep(self):
        assert type(self._timestep) == int
        return self._timestep

    @timestep.setter
    def timestep(self, val):
        if type(val) != int:
            raise ValueError("Timestep must be an integer.")
        self._timestep = val

    @property
    def complete(self):
        return self.controller.nominal_controller.complete

    def __init__(
        self,
        model: Model,
        controller: Controller,
        save_file: str,
    ):
        """Class initializer."""
        self.model = model
        self.controller = controller

        #
        self.id = None
        self.u = self.model.u
        self.u_nom = self.model.u
        self.cbf = self.controller.cbf_vals
        self.save_file = save_file
        self.centralized = self.model.centralized

        # Extract timing data
        self.dt = self.model.dt
        self.tf = self.model.tf
        self.nTimesteps = int((self.tf - self.t) / self.dt) + 1
        self.controller._dt = self.dt

        # Additional class variables
        self.x = None
        self.data = None
        self.x_trajectory = None
        self.u_trajectory = None
        self.u0_trajectory = None
        self.cbf_trajectory = None
        self.consolidated_cbf_trajectory = None
        self.k_gains_trajectory = None
        self.w_desired_trajectory = None
        self.w_dot_trajectory = None
        self.w_dot_f_trajectory = None
        self.b3_trajectory = None

        self.safety = None
        self._timestep = None

        self.reset(self.model.x)  # For simulations

    def set_id(self, id: int) -> None:
        """Sets the identifier for the agent.

        Arguments:
            id (int): agent identifier (place in state vector)

        Returns:
            None
        """
        self.id = id
        self.controller.ego_id = id
        if hasattr(self.controller, "nominal_controller"):
            self.controller.nominal_controller.id = id

    def reset(self, x0: NDArray) -> None:
        """Resets run variables for new trial."""
        self.x = x0
        self.t = 0.0
        self.timestep = 0
        self.x_trajectory = np.zeros((self.nTimesteps, x0.shape[0]))
        self.u_trajectory = np.zeros((self.nTimesteps, self.u_nom.shape[0]))
        self.u0_trajectory = np.zeros((self.nTimesteps, self.u_nom.shape[0]))
        self.safety = np.zeros((self.nTimesteps,))

        if hasattr(self.controller, "cbf_vals"):
            self.cbf_trajectory = np.zeros((self.nTimesteps, len(self.controller.cbf_vals)))
            self.consolidated_cbf_trajectory = np.zeros((self.nTimesteps,))
            self.k_gains_trajectory = np.zeros((self.nTimesteps, len(self.controller.cbf_vals)))
            self.w_desired_trajectory = np.zeros((self.nTimesteps, len(self.controller.cbf_vals)))
            self.w_dot_trajectory = np.zeros((self.nTimesteps, len(self.controller.cbf_vals)))
            self.w_dot_f_trajectory = np.zeros((self.nTimesteps, len(self.controller.cbf_vals)))
            self.b3_trajectory = np.zeros((self.nTimesteps,))
        else:
            self.cbf_trajectory = np.zeros((self.nTimesteps,))
            self.consolidated_cbf_trajectory = np.zeros((self.nTimesteps,))
            self.k_gains_trajectory = np.zeros((self.nTimesteps,))
            self.w_desired_trajectory = np.zeros((self.nTimesteps,))
            self.w_dot_trajectory = np.zeros((self.nTimesteps,))
            self.w_dot_f_trajectory = np.zeros((self.nTimesteps,))
            self.b3_trajectory = np.zeros((self.nTimesteps,))

        # Save data object -- auto-updating since defined by reference
        self.data = {
            "x": self.x_trajectory,
            "u": self.u_trajectory,
            "u0": self.u0_trajectory,
            "cbf": self.cbf_trajectory,
            "ccbf": self.consolidated_cbf_trajectory,
            "kgains": self.k_gains_trajectory,
            "kdes": self.w_desired_trajectory,
            "kdot": self.w_dot_trajectory,
            "kdotf": self.w_dot_f_trajectory,
            "b3": self.b3_trajectory,
            "ii": self.t,
        }

    def compute_control(self, full_state: NDArray) -> (int, str):
        """Computes the control input for the Agent.
        INPUTS
        ------
        full_state: full state vector for all agents
        OUTPUTS
        -------
        code: success / error flag
        status: more informative success / error information
        """
        misc = None

        self.u, code, status = self.controller.compute_control(self.t, full_state)
        self.model.u = self.u

        # Update Control and CBF Trajectories
        self.u_trajectory[self.timestep, :] = self.controller.u
        self.u0_trajectory[self.timestep, :] = self.controller.u_nominal
        self.data["ii"] = self.t
        if hasattr(self.controller, "cbf_vals"):
            self.cbf_trajectory[self.timestep, :] = self.controller.cbf_vals
        if hasattr(self.controller, "c_cbf"):
            self.consolidated_cbf_trajectory[self.timestep] = self.controller.c_cbf
        if hasattr(self.controller, "w_weights"):
            self.k_gains_trajectory[self.timestep, :] = self.controller.w_weights
        if hasattr(self.controller, "w_des"):
            self.w_desired_trajectory[self.timestep, :] = self.controller.w_des
        if hasattr(self.controller, "w_dot"):
            self.w_dot_trajectory[self.timestep, :] = self.controller.w_dot
        if hasattr(self.controller, "w_dot_f"):
            self.w_dot_f_trajectory[self.timestep, :] = self.controller.w_dot_f
        if hasattr(self.controller, "b"):
            self.b3_trajectory[self.timestep] = self.controller.b

        if misc is not None:
            print(misc)

        return code, status

    def step_dynamics(self) -> NDArray:
        """Advances the Agent's dynamics forward in time using the current state and control input.
        INPUTS
        ------
        None -- Requires only class variables
        OUTPUTS
        -------
        x_updated: new state vector
        x_updated: new state vector
        """
        xdot = self.model.xdot()
        x_updated = self.x + self.dt * xdot
        self.update(x_updated)

        return x_updated

    def update(self, x_new: NDArray) -> None:
        """Updates the class time and state vector.
        INPUTS
        ------
        x_new: new state vector
        OUTPUTS
        -------
        None
        """
        self.x = x_new
        self.model.x = x_new
        self.x_trajectory[self.timestep, :] = self.x
        self.timestep = self.timestep + 1
        self.t = self.timestep * self.dt

    def save_data(self, identity: int, fname: Optional[str] = None) -> None:
        """Saves the agent's individual simulation data out to a .pkl file.
        INPUTS
        ------
        identity: agent identifier
        filename: name of file to which data is saved
        OUTPUTS
        -------
        None
        """
        if fname is None:
            filename = self.save_file
        else:
            filename = fname
        file = Path(filename)
        if file.is_file():
            # Load data, then add to it
            with open(filename, "rb") as f:
                try:
                    data = load(f)
                    data[identity] = self.data
                except EOFError:
                    data = {identity: self.data}

        else:
            data = {identity: self.data}

        # Write data to file
        with open(filename, "wb") as f:
            dump(data, f)
