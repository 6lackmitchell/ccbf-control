""" simulate.py: simulation entry-point for testing controllers for specific dynamical models. """

import numpy as np
import builtins
from copy import deepcopy
from traceback import print_exc

def simulate(tf: float,
             dt: float,
             vehicle: str,
             level: str,
             situation: str) -> None:
    """Simulates the system specified by config.py for tf seconds at a frequency of dt.

    ARGUMENTS
        tf: final time (in sec)
        dt: timestep length (in sec)
        vehicle: the vehicle to be simulated
        level: the control level (i.e. kinematic, dynamic, etc.)
        situation: i.e. intersection_old, intersection, etc.

    RETURNS
        None

    """
    # Program-wide specifications
    builtins.PROBLEM_CONFIG = {'vehicle': vehicle,
                               'control_level': level,
                               'situation': situation,
                               'system_model': 'deterministic'}

    if vehicle == 'bicycle':
        from bicycle import nTimesteps, nAgents, nStates, z0, agents_list

    # Simulation setup
    z = np.zeros((nTimesteps, nAgents, nStates))
    z[0, :, :] = z0
    agents = deepcopy(agents_list)

    # Simulate program
    for ii, tt in enumerate(np.linspace(0, tf, nTimesteps - 1)):
        if round(tt, 4) % 1 < dt:
            print("Time: {:.1f} sec".format(tt))
        
        # Iterate over all agents in the system
        for aa, agent in enumerate(agents):
            code, status = agent.compute_control(z[ii])

            # Step dynamics forward
            z[ii + 1, aa, :] = agent.step_dynamics()

    # Save data
    for aa, agent in enumerate(agents):
        agent.save_data(aa)

    return
