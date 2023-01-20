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
from models.double_integrator.dynamic import nAgents
from models.double_integrator.dynamic.toy_example.timing_params import dt, tf
from models.double_integrator.dynamic.toy_example.initial_conditions import xg, yg
from visualizing.helpers import get_circle

matplotlib.rcParams.update({"figure.autolayout": True})

N = 8 * nAgents
# plt.style.use(['Solarize_Light2'])
plt.style.use(["ggplot"])
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, N)))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colors[0] = colors[1]
colors.reverse()

if platform == "linux" or platform == "linux2":
    # linux
    pre_path = "/home/6lackmitchell/"
elif platform == "darwin":
    # OS X
    pre_path = "/Users/mblack/"
elif platform == "win32":
    # Windows...
    pass

filepath = pre_path + "Documents/git/ccbf-control/data/double_integrator/dynamic/toy_example/"


# ### Define Recording Variables ###
t = np.linspace(dt, tf, int(tf / dt))

import builtins

if hasattr(builtins, "VIS_FILE"):
    filename = filepath + builtins.VIS_FILE + ".pkl"
else:
    ftype = r"*.pkl"
    files = glob.glob(filepath + ftype)
    files.sort(key=os.path.getmtime)
    filename = files[-1]

# filename = filepath + 'good.pkl'

with open(filename, "rb") as f:
    try:
        data = pickle.load(f)

        print(data.keys())

        x = np.array([data[a]["x"] for a in data.keys()])
        u = np.array([data[a]["u"] for a in data.keys()])
        czero = np.array([data[a]["czero"] for a in data.keys()])
        # k = np.array([data[a]["kgains"] if a < 3 else None for a in data.keys()][0:3])
        # kdot = np.array([data[a]["kdot"] if a < 3 else None for a in data.keys()][0:3])
        # kdotf = np.array([data[a]["kdotf"] if a < 3 else None for a in data.keys()][0:3])
        k = np.array([data[a]["kgains"] if a < 1 else None for a in data.keys()][:1])
        kdot = np.array([data[a]["kdot"] if a < 1 else None for a in data.keys()][:1])
        kdotf = np.array([data[a]["kdotf"] if a < 1 else None for a in data.keys()][:1])
        u0 = np.array([data[a]["u0"] for a in data.keys()])
        ii = int(data[0]["ii"] / dt)
        # cbf = np.array([data[a]['cbf'] for a in data.keys()])
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
### Control Trajectories ###
fig_control = plt.figure(figsize=(8, 8))
ax_cont_a = fig_control.add_subplot(211)
ax_cont_b = fig_control.add_subplot(212)
set_edges_black(ax_cont_a)
set_edges_black(ax_cont_b)

# Angular Control Inputs
ax_cont_a.plot(t[1:ii], 1 * np.ones(t[1:ii].shape), linewidth=lwidth + 1, color="k")
ax_cont_a.plot(t[1:ii], -1 * np.ones(t[1:ii].shape), linewidth=lwidth + 1, color="k")
for aa in range(nAgents):
    ax_cont_a.plot(
        t[:ii],
        u[aa, :ii, 0],
        label="ax_{}".format(aa),
        linewidth=lwidth,
        color=colors[color_idx[aa, 0]],
    )
    # ax_cont_a.plot(t[:ii], u0[aa, :ii, 0], label='w_{}^0'.format(aa), linewidth=lwidth,
    #                color=colors[color_idx[aa, 1]], dashes=dash)
ax_cont_a.set(
    ylabel="ax",  # ylabel=r'$\omega$',
    ylim=[np.min(u[:ii, :, 0]) - 0.1, np.max(u[:ii, :, 0]) + 0.1],
    title="Control Inputs",
)

# Acceleration Inputs
ax_cont_b.plot(t[1:ii], 1 * np.ones(t[1:ii].shape), linewidth=lwidth + 1, color="k")
ax_cont_b.plot(t[1:ii], -1 * np.ones(t[1:ii].shape), linewidth=lwidth + 1, color="k")
for aa in range(nAgents):
    ax_cont_b.plot(
        t[:ii],
        u[aa, :ii, 1],
        label="ay_{}".format(aa),
        linewidth=lwidth,
        color=colors[color_idx[aa, 0]],
    )
    # ax_cont_b.plot(t[:ii], u0[aa, :ii, 1], label='a_{}^0'.format(aa), linewidth=lwidth,
    #                color=colors[color_idx[aa, 1]], dashes=dash)
ax_cont_b.set(
    ylabel="ay", ylim=[np.min(u[:ii, :, 1]) - 0.5, np.max(u[:ii, :, 1]) + 0.5]  # ylabel=r'$a_r$',
)

# Plot Settings
for item in (
    [ax_cont_a.title, ax_cont_a.xaxis.label, ax_cont_a.yaxis.label]
    + ax_cont_a.get_xticklabels()
    + ax_cont_a.get_yticklabels()
):
    item.set_fontsize(25)
# ax_cont_a.legend(fancybox=True)
ax_cont_a.grid(True, linestyle="dotted", color="white")

for item in (
    [ax_cont_b.title, ax_cont_b.xaxis.label, ax_cont_b.yaxis.label]
    + ax_cont_b.get_xticklabels()
    + ax_cont_b.get_yticklabels()
):
    item.set_fontsize(25)
# ax_cont_b.legend(fancybox=True)
ax_cont_b.grid(True, linestyle="dotted", color="white")

plt.tight_layout(pad=2.0)

############################################
### Gain Trajectories ###
fig_k = plt.figure(figsize=(8, 8))
ax_k = fig_k.add_subplot(111)
set_edges_black(ax_k)

# Angular Control Inputs
lbl = [
    "Obstacle1",
    "Obstacle2",
    "Speed1",
    "Speed2",
]
clr = plt.rcParams["axes.prop_cycle"].by_key()["color"]
clr.reverse()
for cbf in range(k.shape[2]):
    ax_k.plot(
        t[1:ii], k[0, 1:ii, cbf], linewidth=lwidth + 1, color=clr[int(1.5 * cbf)], label=lbl[cbf]
    )
ax_k.set(ylabel="k", title="Adaptation Gains")

# Plot Settings
for item in (
    [ax_k.title, ax_k.xaxis.label, ax_k.yaxis.label]
    + ax_k.get_xticklabels()
    + ax_k.get_yticklabels()
):
    item.set_fontsize(25)
ax_k.legend(fancybox=True)
ax_k.grid(True, linestyle="dotted", color="white")

plt.tight_layout(pad=2.0)


############################################
### Kdot Trajectories ###
fig_kdot = plt.figure(figsize=(8, 8))
ax_kdot = fig_kdot.add_subplot(111)
set_edges_black(ax_kdot)

for cbf in range(k.shape[2]):
    ax_kdot.plot(
        t[1:ii], kdot[0, 1:ii, cbf], linewidth=lwidth + 1, color=clr[int(1.5 * cbf)], label=lbl[cbf]
    )
    ax_kdot.plot(
        t[1:ii],
        kdotf[0, 1:ii, cbf],
        "-.",
        linewidth=lwidth + 1,
        color=clr[int(1.5 * cbf)],
        label=lbl[cbf],
    )
ax_kdot.set(ylabel="kdot", title="Adaptation Derivatives")

# Plot Settings
for item in (
    [ax_kdot.title, ax_kdot.xaxis.label, ax_kdot.yaxis.label]
    + ax_kdot.get_xticklabels()
    + ax_kdot.get_yticklabels()
):
    item.set_fontsize(25)
ax_kdot.legend(fancybox=True)
ax_kdot.grid(True, linestyle="dotted", color="white")

plt.tight_layout(pad=2.0)

############################################
### CZero Trajectories ###
fig_cz = plt.figure(figsize=(8, 8))
ax_cz = fig_cz.add_subplot(111)
set_edges_black(ax_cz)

ax_cz.plot(t[1:ii], czero[0, 1:ii], linewidth=lwidth + 1, color=clr[1], label="C_0")
ax_cz.set(ylabel="C_0", title="CZero Trajectory")

# Plot Settings
for item in (
    [ax_cz.title, ax_cz.xaxis.label, ax_cz.yaxis.label]
    + ax_cz.get_xticklabels()
    + ax_cz.get_yticklabels()
):
    item.set_fontsize(25)
ax_cz.legend(fancybox=True)
ax_cz.grid(True, linestyle="dotted", color="white")

plt.tight_layout(pad=2.0)

# ############################################
# ### CBF Trajectories ###
# fig_cbfs = plt.figure(figsize=(8, 8))
# ax_cbfs = fig_cbfs.add_subplot(111)
# set_edges_black(ax_cbfs)

# # NN-CBF Values
# ax_cbfs.plot(t[1:ii], np.zeros(t[1:ii].shape), linewidth=lwidth+1, color='k')
# for aa in range(cbf.shape[0]):
#     ax_cbfs.plot(t[:ii], cbf[aa, :ii, 0], label='h_{}'.format(aa), linewidth=lwidth,
#                    color=colors[color_idx[aa, 0]])
#     # ax_cbfs.plot(t[:ii], cbf[aa, :ii, 1], label='h_{}^0'.format(aa), linewidth=lwidth,
#     #                color=colors[color_idx[aa, 1]], dashes=dash)
# ax_cbfs.set(ylabel='h',
#             ylim=[-0.1, 250],
#             title='CBF Trajectories')

# # Plot Settings
# for item in ([ax_cbfs.title, ax_cbfs.xaxis.label, ax_cbfs.yaxis.label] +
#              ax_cbfs.get_xticklabels() + ax_cbfs.get_yticklabels()):
#     item.set_fontsize(25)
# ax_cbfs.legend(fancybox=True)
# ax_cbfs.grid(True, linestyle='dotted', color='white')

# plt.tight_layout(pad=2.0)


############################################
### State Trajectories ###
# plt.style.use(['dark_background'])
fig_map = plt.figure(figsize=(10, 10))
ax_pos = fig_map.add_subplot(111)
set_edges_black(ax_pos)

# # Set Up Road
d_points = 100
xc1, yc1 = get_circle(np.array([1, 1]), 0.5, d_points)
xc2, yc2 = get_circle(np.array([1.5, 2.25]), 0.5, d_points)
ax_pos.plot(xc1, yc1, linewidth=lwidth + 1, color="k")
ax_pos.plot(xc2, yc2, linewidth=lwidth + 1, color="k")

for aaa in range(nAgents):
    if aaa == 0:
        lbl = "Goal"
    else:
        lbl = None
    ax_pos.plot(xg[aaa], yg[aaa], "*", markersize=10, label=lbl)

# Create variable reference to plot
map_vid = []
for aa in range(nAgents):
    if aa == 0:
        lbl = "C-CBF Robot"
        clr = "b"
    elif aa == 3:
        lbl = "Uncontrolled Agent"
        clr = "r"
    else:
        lbl = None
        clr = None
    map_vid.append(ax_pos.plot([], [], linewidth=lwidth, label=lbl, color=clr)[0])
    map_vid.append(ax_pos.quiver([], [], [], [], linewidth=lwidth))
    map_vid.append(ax_pos.plot([], [], linewidth=lwidth, dashes=dash)[0])

# Add text annotation and create variable reference
txt = ax_pos.text(-6.8, -13.8, "", ha="right", va="top", fontsize=24)
# txt_list = [ax_pos.text(x[aa, 0, 0], x[aa, 0, 1] + 0.25, '{}'.format(aa + 1),
#                         ha='right', va='top', fontsize=12) if (-10 < x[aa, 0, 0] < 10) and (-15 < x[aa, 0, 1] < 10) else None for aa in range(nAgents)]

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

time_scale_factor = 10


def animate_ego(jj):
    jj = int(jj * time_scale_factor)
    last_1_sec = 40
    ego_pos = x[0, jj, 0:2]
    for aa in range(0, 3 * nAgents, 3):

        idx = int(aa / 3)
        # if not (-10 < x[idx, jj, 0] < 10):
        #     continue

        x_circ, y_circ = get_circle(x[idx, jj], 0.025, d_points)
        map_vid[aa].set_data(x_circ, y_circ)
        map_vid[aa].set_color("b")

    txt.set_text("{:.1f} sec".format(jj * dt))
    ax_pos.set(ylim=[-1.0, 3.0], xlim=[-1.0, 3.0])


# Create animation
ani = animation.FuncAnimation(
    fig=fig_map, func=animate_ego, frames=int(ii / time_scale_factor), interval=10, repeat=False
)
# writer = animation.writers["ffmpeg"]
# ani.save(filename[:-4] + ".mp4", writer=writer(fps=15))

plt.tight_layout(pad=2.0)
plt.show()
