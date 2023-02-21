""" simulate.py: simulation entry-point for testing controllers for specific dynamical models. """

import numpy as np
from typing import List
from .agent import Agent


def simulate(tf: float, dt: float, agents: List[Agent]) -> bool:
    """Simulates the system specified by config.py for tf seconds at a frequency of dt.

    ARGUMENTS
        tf: final time (in sec)
        dt: timestep length (in sec)
        agents (List): list of agents in simulation

    RETURNS
        None

    """
    # dimensions
    nTimesteps = int((tf - 0.0) / dt) + 1
    nAgents = len(agents)
    nStates = len(agents[0].x)

    # extract full initial state
    z0 = np.array([agent.x for agent in agents])
    centralized_agents = [agent for agent in agents if agent.centralized]
    decentralized_agents = [agent for agent in agents if not agent.centralized]

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

        # if round(tt, 4) % 5 < dt and tt > 0:
        #     for aa, agent in enumerate(decentralized_agents):
        #         agent.save_data(aa)
        #     print("Time: {:.1f} sec: Intermediate Save".format(tt))

        # Compute inputs for centralized agents
        # if centralized_agents is not None:
        if len(centralized_agents) > 0:
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

            # print(z[ii + 1, aa, :])

        # Comment out this block if you want to continue with broken agents
        if np.sum(broken) > 0:
            break

        if np.sum(complete) == nAgents:
            break

    # Save data
    if np.sum(complete) == nAgents:
        for aa, agent in enumerate(decentralized_agents):
            agent.save_data(aa)

    else:
        newfilename = "/home/ccbf-control/data/bicycle/dynamic/toy_example/test.pkl"
        # newfilename = "/home/6lackmitchell/Documents/git/ccbf-control/data/bicycle/dynamic/toy_example/test.pkl"
        for aa, agent in enumerate(decentralized_agents):
            agent.save_data(aa, newfilename)

    success = np.sum(complete) == nAgents

    return 0  # success
