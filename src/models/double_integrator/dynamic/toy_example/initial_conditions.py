"""_summary_
"""
import numpy as np

N_COMPARISONS = 1
N_COMPARISONS += (
    1  # No safety, Arbitrary HO-CBF, Joseph's HO-CBF w/ input constraints, HO-CBF w/ alpha decision
)
xg = np.array(N_COMPARISONS * [2.0])
yg = np.array(N_COMPARISONS * [2.0])
xi = np.array(N_COMPARISONS * [-0.25])
yi = np.array(N_COMPARISONS * [0.0])
vxi = np.array(N_COMPARISONS * [0.1])
vyi = np.array(N_COMPARISONS * [0.1])
z0 = np.array([np.array([xi[aa], yi[aa], vxi[aa], vyi[aa]]) for aa in range(len(xi))])
u0 = np.array([0.0, 0.0])

nAgents, nStates = z0.shape
