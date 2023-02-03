""" simulate.py: simulation entry-point for testing controllers for specific dynamical models. """

import numpy as np
import builtins


def simulate(tf: float, dt: float, vehicle: str, level: str, situation: str) -> bool:
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
    builtins.PROBLEM_CONFIG = {
        "vehicle": vehicle,
        "control_level": level,
        "situation": situation,
        "system_model": "deterministic",
    }

    if vehicle == "bicycle":
        from models.bicycle import nAgents, nStates, z0, centralized_agents, decentralized_agents
    elif vehicle == "double_integrator":
        from models.double_integrator import (
            nAgents,
            nStates,
            z0,
            centralized_agents,
            decentralized_agents,
        )
    elif vehicle == "nonlinear_1d":
        from models.nonlinear_1d import (
            nAgents,
            nStates,
            z0,
            centralized_agents,
            decentralized_agents,
        )

    nTimesteps = int((tf - 0.0) / dt) + 1

    # Simulation setup
    broken = np.zeros((nAgents,))
    z = np.zeros((nTimesteps, nAgents, nStates))
    z[0, :, :] = z0
    complete = np.zeros((nAgents,))

    # Simulate program
    for ii, tt in enumerate(np.linspace(0, tf, nTimesteps - 1)):
        code = 0
        if round(tt, 4) % 1 < dt or ii == 1:
            print("Time: {:.1f} sec".format(tt))

        if round(tt, 4) % 5 < dt and tt > 0:
            for aa, agent in enumerate(decentralized_agents):
                agent.save_data(aa)
            print("Time: {:.1f} sec: Intermediate Save".format(tt))

        # Compute inputs for centralized agents
        if centralized_agents is not None:
            centralized_agents.compute_control(tt, z[ii])

        # Iterate over all agents in the system
        for aa, agent in enumerate(decentralized_agents):
            if not broken[aa]:
                code, status = agent.compute_control(z[ii])

            if not code:
                broken[aa] = 1
                print("Error in Agent {}".format(aa + 1))

            if hasattr(agent, "complete"):
                if agent.complete and not complete[aa]:
                    complete[aa] = True
                    print("Agent {} Completed!".format(aa + 1))

            # Step dynamics forward
            if not broken[aa]:
                z[ii + 1, aa, :] = agent.step_dynamics()
            else:
                z[ii + 1, aa, :] = agent.x

            print(z[ii + 1, aa, :])

        # Comment out this block if you want to continue with broken agents
        if np.sum(broken) > 0:
            break

        if np.sum(complete) == nAgents:
            break

    # Save data
    for aa, agent in enumerate(decentralized_agents):
        agent.save_data(aa)

    success = np.sum(complete) == nAgents

    return success
