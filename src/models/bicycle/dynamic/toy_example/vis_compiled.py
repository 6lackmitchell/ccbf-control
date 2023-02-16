import matplotlib

matplotlib.use("Qt5Agg")

import os
import glob
import pickle
import traceback
from sys import platform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from models.bicycle.dynamic import nAgents
from models.bicycle.dynamic.toy_example.timing_params import dt, tf
from models.bicycle.dynamic.toy_example.initial_conditions import xg, yg
from visualizing.helpers import get_circle

matplotlib.rcParams.update({"figure.autolayout": True})

N = 10 * nAgents
# plt.style.use(['Solarize_Light2'])
plt.style.use(["ggplot"])
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, N)))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colors[0] = colors[1]
# colors.reverse()

if platform == "linux" or platform == "linux2":
    # linux
    pre_path = "/home/6lackmitchell/"
elif platform == "darwin":
    # OS X
    pre_path = "/Users/mblack/"
elif platform == "win32":
    # Windows...
    pass

filepath = pre_path + "Documents/git/ccbf-control/data/bicycle/dynamic/toy_example/successes/"
# filepath = pre_path + "Documents/git/ccbf-control/data/bicycle/dynamic/toy_example/"

# ### Define Recording Variables ###
t = np.linspace(dt, tf, int(tf / dt))

import builtins

if hasattr(builtins, "VIS_FILE"):
    filename = filepath + builtins.VIS_FILE + ".pkl"
else:
    ftype = r"*.pkl"
    files = glob.glob(filepath + ftype)
#    files.sort(key=os.path.getmtime)
#    filename = files[-1]


nFiles = len(files)
nTimesteps = 1001
nAgents = 1
nConstraints = 8
x = np.zeros((nFiles, nAgents, nTimesteps, 5))
u = np.zeros((nFiles, nAgents, nTimesteps, 2))
czero = np.zeros((nFiles, nAgents, nTimesteps))
k = np.zeros((nFiles, nAgents, nTimesteps, nConstraints))
kdes = np.zeros((nFiles, nAgents, nTimesteps, nConstraints))
kdot = np.zeros((nFiles, nAgents, nTimesteps, nConstraints))
kdotf = np.zeros((nFiles, nAgents, nTimesteps, nConstraints))
cbfs = np.zeros((nFiles, nAgents, nTimesteps, nConstraints))
ccbf = np.zeros((nFiles, nAgents, nTimesteps))
ii = np.zeros((nFiles,), dtype=int)

for ff, filename in enumerate(files):

    with open(filename, "rb") as f:
        try:
            data = pickle.load(f)

            x[ff] = np.array([data[a]["x"] for a in data.keys()])
            u[ff] = np.array([data[a]["u"] for a in data.keys()])
            czero[ff] = np.array([data[a]["czero1"] for a in data.keys()])
            # czero2 = np.array([data[a]["czero2"] for a in data.keys()])
            # k = np.array([data[a]["kgains"] if a < 3 else None for a in data.keys()][0:3])
            # kdot = np.array([data[a]["kdot"] if a < 3 else None for a in data.keys()][0:3])
            # kdotf = np.array([data[a]["kdotf"] if a < 3 else None for a in data.keys()][0:3])
            k[ff] = np.array([data[a]["kgains"] if a < 1 else None for a in data.keys()][:1])
            kdes[ff] = np.array([data[a]["kdes"] if a < 1 else None for a in data.keys()][:1])
            kdot[ff] = np.array([data[a]["kdot"] if a < 1 else None for a in data.keys()][:1])
            kdotf[ff] = np.array([data[a]["kdotf"] if a < 1 else None for a in data.keys()][:1])
            # u0 = np.array([data[a]["u0"] for a in data.keys()])
            cbfs[ff] = np.array([data[a]["cbf"] for a in data.keys()])
            ccbf[ff] = np.array([data[a]["ccbf"] for a in data.keys()])
            ii[ff] = int(data[0]["ii"] / dt)

        except:
            traceback.print_exc()

lwidth = 2
dash = [3, 2]
color_idx = np.array(range(0, 2 * nAgents)).reshape(nAgents, 2)

# ii = int(tf / dt)
# ii = np.min([int(5.418/dt),ii])


def set_edges_black(ax):
    ax.spines["bottom"].set_color("#000000")
    ax.spines["top"].set_color("#000000")
    ax.spines["right"].set_color("#000000")
    ax.spines["left"].set_color("#000000")


plt.close("all")


############################################
### State Trajectories ###
# plt.style.use(['dark_background'])
fig_map = plt.figure(figsize=(10, 10))
ax_pos = fig_map.add_subplot(111)
set_edges_black(ax_pos)

gain = 2.0
R = 0.4
cx1 = 0.8
cy1 = 1.1
cx2 = 1.5
cy2 = 2.25
cx3 = 2.4
cy3 = 1.5
cx4 = 2.0
cy4 = 0.35
cx5 = 0.8
cy5 = -0.2

# # Set Up Road
d_points = 100
xc1, yc1 = get_circle(np.array([cx1, cy1]), R, d_points)
xc2, yc2 = get_circle(np.array([cx2, cy2]), R, d_points)
xc3, yc3 = get_circle(np.array([cx3, cy3]), R, d_points)
xc4, yc4 = get_circle(np.array([cx4, cy4]), R, d_points)
xc5, yc5 = get_circle(np.array([cx5, cy5]), R, d_points)
xc6, yc6 = get_circle(np.array([xg[0], yg[0]]), 0.1, d_points)
ax_pos.plot(xc1, yc1, linewidth=lwidth + 1, color="k")
ax_pos.plot(xc2, yc2, linewidth=lwidth + 1, color="k")
ax_pos.plot(xc3, yc3, linewidth=lwidth + 1, color="k")
ax_pos.plot(xc4, yc4, linewidth=lwidth + 1, color="k")
ax_pos.plot(xc5, yc5, linewidth=lwidth + 1, color="k")
ax_pos.plot(xc6, yc6, linewidth=lwidth + 1, color="g")

for ff in range(nFiles):
    h1 = (x[ff, 0, 1 : ii[ff], 0] - cx1) ** 2 + (x[ff, 0, 1 : ii[ff], 1] - cy1) ** 2 - R**2
    if np.min(h1) < 0:
        print(files[ff])
    ax_pos.plot(x[ff, 0, 1 : ii[ff], 0], x[ff, 0, 1 : ii[ff], 1], linewidth=lwidth, color="b")

ax_pos.set(ylim=[-1.0, 3.0], xlim=[-1.0, 3.0], xlabel="X (m)", ylabel="Y (m)")

# Plot Settings
for item in (
    [ax_pos.title, ax_pos.xaxis.label, ax_pos.yaxis.label]
    + ax_pos.get_xticklabels()
    + ax_pos.get_yticklabels()
):
    item.set_fontsize(25)
# Hide X and Y axes label marks
ax_pos.xaxis.set_tick_params(labelbottom=False)
ax_pos.yaxis.set_tick_params(labelleft=False)
# Hide X and Y axes tick marks
ax_pos.set_xticks([])
ax_pos.set_yticks([])
ax_pos.legend(fancybox=True, fontsize=15)
ax_pos.grid(False)

plt.tight_layout(pad=2.0)
plt.show()
