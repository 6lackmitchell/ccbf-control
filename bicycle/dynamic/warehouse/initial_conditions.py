import numpy as np


xg = np.array([-2.0, 0.0, 2.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
yg = np.array([6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

xi = np.array([-2.0, 0.0, 3.0, -8.0, -10.0, -14.0, -16.0, -20.0, -22.0])
yi = np.array([-np.sqrt(96), -12, -np.sqrt(91), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
psii = np.array([np.pi + np.arctan2(yi[ii], xi[ii]) for ii in range(len(xi))])
vi = np.array([0.0 for ii in range(3)] + [1.0 for ii in range(6)])
z0 = np.array([np.array([xi[aa], yi[aa], psii[aa], vi[aa], 0.0]) for aa in range(len(xi))])


u0 = np.array([0.0, 0.0])

nAgents, nStates = z0.shape
