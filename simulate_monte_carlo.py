""" simulate.py: simulation entry-point for testing controllers for specific dynamical models. """

### Import Statements ###
# General imports
import sys
import pickle
import builtins
import traceback
import numpy as np

# Determine which problem we are solving
args = sys.argv
if len(args) > 1:
    config = str(args[1])
else:
    config = 'single_integrator'

# Make problem config available to other modules
builtins.ecc_MODEL_CONFIG = config

# Problem specific imports
if config == 'single_integrator':
    from single_integrator.settings import *

# Authorship information
__author__ = "Mitchell Black"
__copyright__ = "Open Education - Creative Commons"
__version__ = "0.0.1"
__maintainer__ = "Mitchell Black"
__email__ = "mblackjr@umich.edu"
__status__ = "Development"

# Output file
nRuns = 100000
filepath = save_path
filename = filepath + 'rb_cbf_monte_carlo_nRuns{}_dt1e3_a10_b4.pkl'.format(nRuns)

# nCBFs
nCBFs = 1  # cbf(z0).shape[0]

# Logging variables
z = np.zeros((nRuns, nTimesteps, nAgents, nStates))
u = np.zeros((nRuns, nTimesteps, nAgents, nControls))
qp_sol = np.zeros((nRuns, nTimesteps, nAgents, nControls))
nominal_sol = np.zeros((nRuns, nTimesteps, nAgents, nControls))
cbfs = np.zeros((nRuns, nTimesteps, 2, 2))
etas = np.zeros((nRuns, nTimesteps, nAgents))
all_unsafe = np.zeros((nRuns, nAgents))

# Create Simulation Agentsna
agents = agents_list

cbf = np.zeros((nAgents,))

try:
    for rr, run in enumerate(range(nRuns)):
        # Reset initial parameters
        z[rr, 0, :, :] = z0
        for agent in agents:
            agent.update(z[rr, 0, 0, :])
        unsafe = np.zeros((nAgents,))

        if round(rr, 4) % int(nRuns / np.min([100, nRuns])) < dt:
            print("Progress: {:.0f} / {} Complete".format(rr, nRuns))
            print("nUnsafe1: {}/{}".format(int(np.sum(all_unsafe[:, 0])), rr))
            print("nUnsafe2: {}/{}".format(int(np.sum(all_unsafe[:, 1])), rr))

        for ii, tt in enumerate(np.linspace(ts, tf, nTimesteps - 1)):

            # Iterate over all agents in the system
            for aa, agent in enumerate(agents):

                # Assign extra parameters
                extras = dict({'agent': aa})

                # Compute control input
                qp_sol[rr, ii, aa], nominal_sol[rr, ii, aa], code, status, cbf[aa] \
                    = agent.compute_control(tt, z[rr, ii], extras)

                # Step dynamics forward
                z[rr, ii + 1, aa, :] = agent.step_dynamics()

            # Update Logging Variables
            u[rr, ii] = qp_sol[rr, ii, :, :nControls]
            cbfs[rr, ii] = cbf[0]
            unsafe = np.array([int(cc > 1) for cc in cbf]) + unsafe  # Document whether unsafe has occurred
            unsafe = np.clip(unsafe, 0, 1)  # Keep unsafe as recording boolean yes/no values

        all_unsafe[rr, :] = unsafe

except Exception as e:
    traceback.print_exc()

else:
    pass

finally:
    print("Ended on Run {} of {}".format(rr + 1, nRuns))
    # print("Percent Stochastic CBF Unsafe: {:.2f}% <= {:.2f}%".format(np.sum(all_unsafe[:, 0]) / nRuns * 100,
    #                                                                  (1 - (1 - 0.5) * np.exp(-1)) * 100))
    # print("Percent Stochastic CBF Unsafe: {:.2f}% <= {:.2f}%".format(np.sum(all_unsafe[:, 1]) / nRuns * 100,
    #                                                                  (1 - (1 - 0.5) * np.exp(-(1 - 0.25))) * 100))
    print("Percent Stochastic CBF Unsafe: {:.2f}% <= {:.2f}%".format(np.sum(all_unsafe[:, 0]) / nRuns * 100,
                                                                     (1 - (1 - 0.0) * np.exp(-0.5)) * 100))
    print("Percent Stochastic CBF Unsafe: {:.2f}% <= {:.2f}%".format(np.sum(all_unsafe[:, 1]) / nRuns * 100,
                                                                     (1 - (1 - 0.0) * np.exp(-(0.5))) * 100))
    print("SIMULATION TERMINATED AT T = {:.4f} sec".format(tt))

    # Format data here
    data = {'x': z,
            'sols': qp_sol,
            'sols_nom': nominal_sol,
            'cbf': cbfs,
            'etas': etas,
            'ii': ii,
            'unsafe': all_unsafe}

    # Write data to file
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

print('\a')
