import numpy as np


nAgents = 4
x_dist = 1.0
y_dist = 1.0
box_width = int(np.sqrt(nAgents))

xi0 = np.zeros((nAgents,))
yi0 = np.zeros((nAgents,))
for ww, bw in enumerate(range(box_width)):
    for ll, bl in enumerate(range(box_width)):
        xi0[ll * box_width + ww] = ww * 2 * box_width + box_width
        yi0[ll * box_width + ww] = ll * 2 * box_width + box_width



# xi0 = np.array([-3.0, -1.0, 3.0, 1.0])
# yi0 = np.array([0.0, -2.0, 0.0, 2.0])

xi = np.array([xi0[ii] + np.random.uniform(low=-x_dist, high=x_dist) for ii in range(nAgents)])
yi = np.array([yi0[ii] + np.random.uniform(low=-y_dist, high=y_dist) for ii in range(nAgents)])
psii = np.array([np.random.uniform(low=-np.pi, high=np.pi) for ii in range(nAgents)])
vi = np.array([0.0 for ii in range(nAgents)])
z0 = np.array([np.array([xi[aa], yi[aa], psii[aa], vi[aa], 0.0]) for aa in range(nAgents)])

u0 = np.array([0.0, 0.0])

nAgents, nStates = z0.shape
