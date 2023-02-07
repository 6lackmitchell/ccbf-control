"""_summary_
"""
import numpy as np

# np.random.seed(555)
# N_COMPARISONS = 1
# N_COMPARISONS += (
#     0  # No safety, Arbitrary HO-CBF, Joseph's HO-CBF w/ input constraints, HO-CBF w/ alpha decision
# )
# xg = np.array(N_COMPARISONS * [2.0])
# yg = np.array(N_COMPARISONS * [2.0])


# xi = np.array(N_COMPARISONS * [-0.25])
# yi = np.array(N_COMPARISONS * [0.0])

# speed = 0.2


# vxi = np.array(N_COMPARISONS * [(xg)])
# vyi = np.array(N_COMPARISONS * [0.1])


# z0 = np.array([np.array([xi[aa], yi[aa], vxi[aa], vyi[aa]]) for aa in range(len(xi))])

# C-CBF Initial Conditions
xg = np.array([2.0])
yg = np.array([2.0])


xi = np.array([-np.random.random()])
yi = np.array([np.random.uniform(low=-0.5, high=2.5)])

speed = 0.1
psii = np.arctan2(yg - yi, xg - xi)
vi = np.array([speed])
betai = np.array([np.random.uniform(low=-0.1, high=0.1)])

vxi = xg - xi
vyi = yg - yi
vxi *= speed / np.linalg.norm([vxi, vyi])
vyi *= speed / np.linalg.norm([vxi, vyi])

z0 = np.array([np.array([xi[aa], yi[aa], psii[aa], vi[aa], betai[aa]]) for aa in range(len(xi))])
# z0 = np.array([np.array([xi[aa], yi[aa], vxi[aa], vyi[aa]]) for aa in range(len(xi))])
u0 = np.array([0.0, 0.0])

nAgents, nStates = z0.shape
