"""_summary_
"""
import numpy as np

xg = np.array([0.0])
xi = np.array([0.0])
z0 = np.array([np.array([xi[aa]]) for aa in range(len(xi))])
u0 = np.array([0.0])

nAgents, nStates = z0.shape
