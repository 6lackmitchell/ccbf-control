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
if config == 'bicycle':
    from bicycle import *
    # from bicycle.settings import *

# Authorship information
__author__ = "Mitchell Black"
__copyright__ = "Open Education - Creative Commons"
__version__ = "0.0.1"
__maintainer__ = "Mitchell Black"
__email__ = "mblackjr@umich.edu"
__status__ = "Development"

# Output file
filepath = save_path
filename = filepath + 'test.pkl'
print(filename)

# Logging variables
z = np.zeros((nTimesteps, nAgents, nStates))
# u = np.zeros((nTimesteps, nAgents, nControls))
# qp_sol = np.zeros((nTimesteps, nAgents, nSols))
# nominal_sol = np.zeros((nTimesteps, nAgents, nSols))
# cbfs = np.zeros((nTimesteps, nAgents, nCBFs))

# Set initial parameters
z[0, :, :] = z0

try:
    for ii, tt in enumerate(np.linspace(ts, tf, nTimesteps - 1)):
        if round(tt, 4) % 0.1 < dt:
            print("Time: {:.1f} sec".format(tt))

        # Iterate over all agents in the system
        for aa, agent in enumerate(agents):
            code, status = agent.compute_control(z[ii])

            # Step dynamics forward
            z[ii + 1, aa, :] = agent.step_dynamics()

except Exception as e:
    print_exc()

else:
    pass

finally:
    print("SIMULATION TERMINATED AT T = {:.4f} sec".format(tt))
    print("State: {}".format(z[ii]))

    for aa, agent in enumerate(agents):
        agent.save_data(aa, filename)

print('\a')
