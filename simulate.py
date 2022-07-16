""" simulate.py: simulation entry-point for testing controllers for specific dynamical models. """

### Import Statements ###
# General imports
import numpy as np
from sys import argv
from pickle import dump
from traceback import print_exc

# Determine which problem we are solving and import accordingly
args = argv
if len(args) > 1:
    config = str(args[1])
else:
    config = 'bicycle'

# Problem specific imports
if config == 'single_integrator':
    from single_integrator.settings import *
elif config == 'quadrotor':
    from quadrotor.settings import *
elif config == 'bicycle':
    from bicycle.settings import *

# Authorship information
__author__ = "Mitchell Black"
__copyright__ = "Open Education - Creative Commons"
__version__ = "0.0.1"
__maintainer__ = "Mitchell Black"
__email__ = "mblackjr@umich.edu"
__status__ = "Development"

# Output file
filepath = save_path
filename = filepath + 'risk_bounded_cbf_preliminary_test.pkl'

# nCBFs
nCBFs = 1  # cbf(z0).shape[0]

# Logging variables
z = np.zeros((nTimesteps, nAgents, nStates))
u = np.zeros((nTimesteps, nAgents, nControls))
qp_sol = np.zeros((nTimesteps, nAgents, nSols))
nominal_sol = np.zeros((nTimesteps, nAgents, nSols))
# cbfs = np.zeros((nTimesteps, nCBFs))
# cbfs = np.zeros((nTimesteps, 2, 2))

# Set initial parameters
z[0, :, :] = z0

# cbf = np.zeros((nAgents, 2))

cbfs = np.zeros((nTimesteps, nAgents, 3, 2))
cbf = np.zeros((nAgents, 3, 2))

try:
    for ii, tt in enumerate(np.linspace(ts, tf, nTimesteps - 1)):
        if round(tt, 4) % 0.1 < dt:
            print("Time: {:.0f} sec".format(tt))

        # Iterate over all agents in the system
        for aa, agent in enumerate(agents):
            # Assign extra parameters
            extras = dict({'agent': aa})
            if aa > 0:
                extras = aa
            else:
                extras['ignore'] = 4

            # Compute control input
            qp_sol[ii, aa], nominal_sol[ii, aa], code, status, cbf[aa] = agent.compute_control(tt, z[ii], extras)

            # Step dynamics forward
            z[ii + 1, aa, :] = agent.step_dynamics()

        # Update Logging Variables
        u[ii] = qp_sol[ii, :, :nControls]
        cbfs[ii] = cbf[0]

except Exception as e:
    print_exc()

else:
    pass

finally:
    print("SIMULATION TERMINATED AT T = {:.4f} sec".format(tt))
    print("State: {}".format(z[ii]))

    # Format data here
    data = {'x': z,
            'sols': qp_sol,
            'sols_nom': nominal_sol,
            'cbf': cbfs,
            'ii': ii}

    # Write data to file
    with open(filename, 'wb') as f:
        dump(data, f)

print('\a')
