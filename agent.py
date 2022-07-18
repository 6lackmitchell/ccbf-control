import numpy as np
from typing import Callable
from nptyping import NDArray
from pathlib import Path
from pickle import dump, load


class Agent:

    @property
    def timestep(self):
        assert(type(self._timestep) == int)
        return self._timestep

    @timestep.setter
    def timestep(self, val):
        if type(val) != int:
            raise ValueError("Timestep must be an integer.")
        self._timestep = val

    def __init__(self,
                 identifier: int,
                 x0: NDArray,
                 u0: NDArray,
                 cbf0: NDArray,
                 timing: NDArray,
                 dynamics: Callable,
                 controller: Callable,
                 nominal_controller: Callable = None):
        """ Class initializer. """
        self.id = identifier
        self.x = x0
        self.u = u0
        self.u_nom = u0
        self.cbf = cbf0
        self.dynamics = dynamics
        self.controller = controller
        self.nominal_controller = nominal_controller

        # Extract timing data
        self.t = 0.0
        self.dt = timing[0]
        self.tf = timing[1]
        self.timestep = 0
        self.nTimesteps = int((self.tf - self.t) / self.dt) + 1

        # Construct object for saving data
        self.x_trajectory = np.zeros((self.nTimesteps, x0.shape[0]))
        self.u_trajectory = np.zeros((self.nTimesteps, u0.shape[0]))
        self.u0_trajectory = np.zeros((self.nTimesteps, u0.shape[0]))
        self.cbf_trajectory = np.zeros((self.nTimesteps, cbf0.shape[0]))

        # Save data object -- auto-updating since defined by reference
        self.data = {'x': self.x_trajectory,
                     'u': self.u_trajectory,
                     'u0': self.u0_trajectory,
                     'cbf': self.cbf_trajectory}

    def compute_control(self,
                        full_state: NDArray) -> (int, str):
        """ Computes the control input for the Agent.

        INPUTS
        ------
        full_state: full state vector for all agents

        OUTPUTS
        -------
        code: success / error flag
        status: more informative success / error information

        """
        misc = None
        self.timestep = int(self.t / self.dt)
        if self.nominal_controller is None:
            self.u, self.u_nom, code, status = self.controller(self.t, self.x, self.id)
        else:
            # extras = {'ignore': 4}
            self.u, self.u_nom, self.cbf, code, status, misc = \
                self.controller(self.t, full_state, self.nominal_controller, self.id)

        # Update Control and CBF Trajectories
        self.u_trajectory[self.timestep, :] = self.u
        self.u0_trajectory[self.timestep, :] = self.u_nom
        self.cbf_trajectory[self.timestep, :] = self.cbf

        if misc is not None:
            print(misc)

        return code, status

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
        self.x_trajectory[self.timestep, :] = self.x

    def save_data(self,
                  identity: int,
                  filename: str) -> None:
        """Saves the agent's individual simulation data out to a .pkl file.

        INPUTS
        ------
        identity: agent identifier
        filename: name of file to which data is saved

        OUTPUTS
        -------
        None

        """
        file = Path(filename)
        if file.is_file():
            # Load data, then add to it
            with open(filename, 'rb') as f:
                data = load(f)

            data[identity] = self.data

        else:
            data = {identity: self.data}

        # Write data to file
        with open(filename, 'wb') as f:
            dump(data, f)
