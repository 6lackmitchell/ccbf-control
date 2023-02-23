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
from visualizing.helpers import get_circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


matplotlib.rcParams.update({"figure.autolayout": True})

# parameters
dt = 1e-2
tf = 10.0
xg = yg = np.array([2.0])

nAgents = 4
N = nAgents
# plt.style.use(['Solarize_Light2'])
plt.style.use(["ggplot"])
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, N)))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colors.append(np.array([0.98, 0.6, 0.2, 1]))
colors.append(np.array([0.98, 0.01, 0.4, 1]))
# colors.append(np.array([0.0, 0.98, 0.2, 1]))
# colors.append(np.array([0.0, 0.98, 0.2, 1]))
# colors[0] = colors[1]
colors.reverse()

if platform == "linux" or platform == "linux2":
    # linux
    pre_path = "/home/6lackmitchell/Documents/git/"
    # pre_path = "/home/dasc/mitchell/"
elif platform == "darwin":
    # OS X
    pre_path = "/Users/mblack/Documents/git/"
elif platform == "win32":
    # Windows...
    pass

filepath = pre_path + "ccbf-control/data/oscillator/successes/"

# ### Define Recording Variables ###
t = np.linspace(dt, tf, int(tf / dt))

# Specify files
fname1 = filepath + "pos_alpha_1.pkl"
fname2 = filepath + "pos_alpha_0p1.pkl"
fname3 = filepath + "pos_alpha_0p01.pkl"
fname4 = filepath + "neg_alpha_1.pkl"
fname5 = filepath + "neg_alpha_0p1.pkl"
fname6 = filepath + "neg_alpha_0p01.pkl"

# Load Simulation Data
with open(fname1, "rb") as f:
    data = pickle.load(f)

    x1 = np.array([data[a]["x"] for a in data.keys()])
    u1 = np.array([data[a]["u"] for a in data.keys()])
    u01 = np.array([data[a]["u0"] for a in data.keys()])
    b311 = np.array([data[a]["b3"] for a in data.keys()])
    k1 = np.array([data[a]["kgains"] if a < 1 else None for a in data.keys()][:1])
    kdot1 = np.array([data[a]["kdot"] if a < 1 else None for a in data.keys()][:1])
    kdotf1 = np.array([data[a]["kdotf"] if a < 1 else None for a in data.keys()][:1])
    ii = int(data[0]["ii"] / dt)
with open(fname2, "rb") as f:
    data = pickle.load(f)

    x2 = np.array([data[a]["x"] for a in data.keys()])
    u2 = np.array([data[a]["u"] for a in data.keys()])
    u02 = np.array([data[a]["u0"] for a in data.keys()])
    b312 = np.array([data[a]["b3"] for a in data.keys()])
    k2 = np.array([data[a]["kgains"] if a < 1 else None for a in data.keys()][:1])
    kdot2 = np.array([data[a]["kdot"] if a < 1 else None for a in data.keys()][:1])
    kdotf2 = np.array([data[a]["kdotf"] if a < 1 else None for a in data.keys()][:1])
with open(fname3, "rb") as f:
    data = pickle.load(f)

    x3 = np.array([data[a]["x"] for a in data.keys()])
    u3 = np.array([data[a]["u"] for a in data.keys()])
    u03 = np.array([data[a]["u0"] for a in data.keys()])
    b313 = np.array([data[a]["b3"] for a in data.keys()])
    k3 = np.array([data[a]["kgains"] if a < 1 else None for a in data.keys()][:1])
    kdot3 = np.array([data[a]["kdot"] if a < 1 else None for a in data.keys()][:1])
    kdotf3 = np.array([data[a]["kdotf"] if a < 1 else None for a in data.keys()][:1])
with open(fname4, "rb") as f:
    data = pickle.load(f)

    x4 = np.array([data[a]["x"] for a in data.keys()])
    u4 = np.array([data[a]["u"] for a in data.keys()])
    u04 = np.array([data[a]["u0"] for a in data.keys()])
    b314 = np.array([data[a]["b3"] for a in data.keys()])
    k4 = np.array([data[a]["kgains"] if a < 1 else None for a in data.keys()][:1])
    kdot4 = np.array([data[a]["kdot"] if a < 1 else None for a in data.keys()][:1])
    kdotf4 = np.array([data[a]["kdotf"] if a < 1 else None for a in data.keys()][:1])
with open(fname5, "rb") as f:
    data = pickle.load(f)

    x5 = np.array([data[a]["x"] for a in data.keys()])
    u5 = np.array([data[a]["u"] for a in data.keys()])
    u05 = np.array([data[a]["u0"] for a in data.keys()])
    b315 = np.array([data[a]["b3"] for a in data.keys()])
    k5 = np.array([data[a]["kgains"] if a < 1 else None for a in data.keys()][:1])
    kdot5 = np.array([data[a]["kdot"] if a < 1 else None for a in data.keys()][:1])
    kdotf5 = np.array([data[a]["kdotf"] if a < 1 else None for a in data.keys()][:1])

with open(fname6, "rb") as f:
    data = pickle.load(f)

    x6 = np.array([data[a]["x"] for a in data.keys()])
    u6 = np.array([data[a]["u"] for a in data.keys()])
    u06 = np.array([data[a]["u0"] for a in data.keys()])
    b316 = np.array([data[a]["b3"] for a in data.keys()])
    k6 = np.array([data[a]["kgains"] if a < 1 else None for a in data.keys()][:1])
    kdot6 = np.array([data[a]["kdot"] if a < 1 else None for a in data.keys()][:1])
    kdotf6 = np.array([data[a]["kdotf"] if a < 1 else None for a in data.keys()][:1])


x = np.concatenate(
    [
        x1[np.newaxis, 0],
        x2[np.newaxis, 0],
        x3[np.newaxis, 0],
        x4[np.newaxis, 0],
        x5[np.newaxis, 0],
        x6[np.newaxis, 0],
    ]
)
u = np.concatenate(
    [
        u1[np.newaxis, 0],
        u2[np.newaxis, 0],
        u3[np.newaxis, 0],
        u4[np.newaxis, 0],
        u5[np.newaxis, 0],
        u6[np.newaxis, 0],
    ]
)
u0 = np.concatenate(
    [
        u01[np.newaxis, 0],
        u02[np.newaxis, 0],
        u03[np.newaxis, 0],
        u04[np.newaxis, 0],
        u05[np.newaxis, 0],
        u06[np.newaxis, 0],
    ]
)
b3 = np.concatenate(
    [
        b311[np.newaxis, 0],
        b312[np.newaxis, 0],
        b313[np.newaxis, 0],
        b314[np.newaxis, 0],
        b315[np.newaxis, 0],
        b316[np.newaxis, 0],
    ]
)
k = np.concatenate(
    [
        k1[np.newaxis, 0],
        k2[np.newaxis, 0],
        k3[np.newaxis, 0],
        k4[np.newaxis, 0],
        k5[np.newaxis, 0],
        k6[np.newaxis, 0],
    ]
)
kdot = np.concatenate(
    [
        kdot1[np.newaxis, 0],
        kdot2[np.newaxis, 0],
        kdot3[np.newaxis, 0],
        kdot4[np.newaxis, 0],
        kdot5[np.newaxis, 0],
        kdot6[np.newaxis, 0],
    ]
)

legend_font = 19
fontsize = 30
lwidth = 3
dash = [3, 2]
color_idx = np.array(range(0, 2 * nAgents)).reshape(nAgents, 2)
names = [
    r"$\gamma = 1$",
    r"$\gamma = 0.1$",
    r"$\gamma = 0.01$",
    r"$\gamma = 1$",
    r"$\gamma = 0.1$",
    r"$\gamma = 0.01$",
]


def set_edges_black(ax):
    ax.spines["bottom"].set_color("#000000")
    ax.spines["top"].set_color("#000000")
    ax.spines["right"].set_color("#000000")
    ax.spines["left"].set_color("#000000")


plt.close("all")


############################################
### Control Trajectories ###
fig_control = plt.figure(figsize=(12, 8))
ax_cont = fig_control.add_subplot(111)
set_edges_black(ax_cont)
set_edges_black(ax_cont)

ii_u = ii

# Angular Control Inputs
ax_cont.plot(
    t[1:ii_u], 1 * np.ones(t[1:ii_u].shape), linewidth=lwidth + 1, color="k", label=r"$\bar{u}$"
)
ax_cont.plot(t[1:ii_u], -1 * np.ones(t[1:ii_u].shape), linewidth=lwidth + 1, color="k")
for aa in [0]:
    ax_cont.plot(
        t[:ii_u],
        u0[aa, :ii_u, 0],
        label=r"$u_{0, \theta=0}$",
        linewidth=lwidth,
        color="m",
        dashes=dash,
    )
for aa in [0, 1, 2]:
    ax_cont.plot(
        t[:ii_u],
        u[aa, :ii_u, 0],
        label=rf"{names[aa]}",
        linewidth=lwidth,
        color=colors[aa],
    )

for aa in [3]:
    ax_cont.plot(
        t[:ii_u],
        u0[aa, :ii_u, 0],
        label=r"$u_{0, \theta=\pi}$",
        linewidth=lwidth,
        color="c",
        dashes=dash,
    )
for aa in [3, 4, 5]:
    ax_cont.plot(
        t[:ii_u],
        u[aa, :ii_u, 0],
        label=rf"{names[aa]}",
        linewidth=lwidth,
        color=colors[aa],
    )

ax_cont.set(
    xlim=[-2.75, 10.25],
    xlabel=r"$t$ (sec)",
    ylabel=r"$u$",
)

for item in (
    [ax_cont.title, ax_cont.xaxis.label, ax_cont.yaxis.label]
    + ax_cont.get_xticklabels()
    + ax_cont.get_yticklabels()
):
    item.set_fontsize(25)

# ax_cont.set_xticks([])
# ax_cont.set_yticks([-1, 0, 1])
ax_cont.legend(fancybox=True, fontsize=legend_font)
ax_cont.grid(True, linestyle="dotted", color="white")
# ax_cont.set_xticks([0, 2, 4, 6, 8, 10])

plt.tight_layout(pad=2.0)

############################################
### Gain Trajectories ###
fig_k = plt.figure(figsize=(12, 8))
ax_k = fig_k.add_subplot(211)
set_edges_black(ax_k)

# Angular Control Inputs
lbl = [
    [r"$w_1: \gamma = 1.0$", r"$w_2: \gamma = 1.0$"],
    [r"$w_1: \gamma = 0.1$", r"$w_2: \gamma = 0.1$"],
    [r"$w_1: \gamma = 0.01$", r"$w_2: \gamma = 0.01$"],
    [r"$w_1: \gamma = 1.0$", r"$w_2: \gamma = 1.0$"],
    [r"$w_1: \gamma = 0.1$", r"$w_2: \gamma = 0.1$"],
    [r"$w_1: \gamma = 0.01$", r"$w_2: \gamma = 0.01$"],
]
lbl = [
    [None, None],
    [None, None],
    [None, None],
    [None, None],
    [None, None],
    [None, None],
    [r"$w_1$", r"$w_2$"],
]
ax_k.plot(t[1], [0], linewidth=lwidth, color="k", label=lbl[6][0])
ax_k.plot(t[1], [0], ":", linewidth=lwidth, color="k", label=lbl[6][1])
for aa in [0, 1, 2, 3, 4, 5]:
    ax_k.plot(
        t[1:ii],
        k[aa, 1:ii, 0],
        linewidth=lwidth + 1,
        color=colors[aa],
        label=lbl[aa][0],
        # dashes=dash
    )
for aa in [0, 1, 2, 3, 4, 5]:
    ax_k.plot(t[1:ii], k[aa, 1:ii, 1], ":", linewidth=lwidth, color=colors[aa], label=lbl[aa][1])
ax_k.set(ylabel=r"$w$", xlim=[-0.1, 10.1])
ax_k.set_xticklabels([])

# Plot Settings
for item in (
    [ax_k.title, ax_k.xaxis.label, ax_k.yaxis.label]
    + ax_k.get_xticklabels()
    + ax_k.get_yticklabels()
):
    item.set_fontsize(fontsize)
ax_k.legend(fancybox=True, fontsize=legend_font)
ax_k.grid(True, linestyle="dotted", color="white")

plt.tight_layout(pad=2.0)

############################################
### W-Dot Trajectories ###
ax_wdot = fig_k.add_subplot(212)
set_edges_black(ax_wdot)

# Angular Control Inputs
lbl = [
    [r"$\dot{w}_1: \gamma = 1.0$", r"$\dot{w}_2: \gamma = 1.0$"],
    [r"$\dot{w}_1: \gamma = 0.1$", r"$\dot{w}_2: \gamma = 0.1$"],
    [r"$\dot{w}_1: \gamma = 0.01$", r"$\dot{w}_2: \gamma = 0.01$"],
    [r"$\dot{w}_1: \gamma = 1.0$", r"$\dot{w}_2: \gamma = 1.0$"],
    [r"$\dot{w}_1: \gamma = 0.1$", r"$\dot{w}_2: \gamma = 0.1$"],
    [r"$\dot{w}_1: \gamma = 0.01$", r"$\dot{w}_2: \gamma = 0.01$"],
]
lbl = [
    [None, None],
    [None, None],
    [None, None],
    [None, None],
    [None, None],
    [None, None],
    [r"$\dot{w}_1$", r"$\dot{w}_2$"],
]

ax_wdot.plot(t[1], [0], linewidth=lwidth, color="k", label=lbl[6][0])
ax_wdot.plot(t[1], [0], ":", linewidth=lwidth, color="k", label=lbl[6][1])

for aa in [0, 1, 2, 3, 4, 5]:
    ax_wdot.plot(
        t[2:ii],
        kdot[aa, 2:ii, 0],
        linewidth=lwidth + 1,
        color=colors[aa],
        label=lbl[aa][0],
        # dashes=dash,
    )
for aa in [0, 1, 2, 3, 4, 5]:
    ax_wdot.plot(
        t[2:ii], kdot[aa, 2:ii, 1], ":", linewidth=lwidth, color=colors[aa], label=lbl[aa][1]
    )
ax_wdot.set(ylabel=r"$\dot{w}$", xlabel=r"$t$ (sec)", xlim=[-0.1, 10.1])
# ax_wdot.set_yticklabels([-10, -5, 0, 5, 10, 1])

# Plot Settings
for item in (
    [ax_wdot.title, ax_wdot.xaxis.label, ax_wdot.yaxis.label]
    + ax_wdot.get_xticklabels()
    + ax_wdot.get_yticklabels()
):
    item.set_fontsize(fontsize)
ax_wdot.legend(fancybox=True, fontsize=legend_font)
ax_wdot.grid(True, linestyle="dotted", color="white")

plt.tight_layout(pad=2.0)

############################################
### b3 Trajectories ###
fig_cz = plt.figure(figsize=(8, 8))
ax_cz = fig_cz.add_subplot(111)
set_edges_black(ax_cz)

ax_cz.plot(t[1:ii], b3[0, 1:ii], linewidth=lwidth + 1, color=colors[1], label="b3")
ax_cz.set(ylabel="b3", title="b3 Trajectory")

# Plot Settings
for item in (
    [ax_cz.title, ax_cz.xaxis.label, ax_cz.yaxis.label]
    + ax_cz.get_xticklabels()
    + ax_cz.get_yticklabels()
):
    item.set_fontsize(fontsize)
ax_cz.legend(fancybox=True)
ax_cz.grid(True, linestyle="dotted", color="white")

plt.tight_layout(pad=2.0)


############################################
### C-CBF Trajectories ###
fig_H = plt.figure(figsize=(12, 8))
ax_H = fig_H.add_subplot(211)
set_edges_black(ax_cz)

h11 = 2 - x[0, 1:ii, 0]
h12 = 2 - x[1, 1:ii, 0]
h13 = 2 - x[2, 1:ii, 0]
h14 = 2 - x[3, 1:ii, 0]
h15 = 2 - x[4, 1:ii, 0]
h16 = 2 - x[5, 1:ii, 0]

h21 = 2 + x[0, 1:ii, 0]
h22 = 2 + x[1, 1:ii, 0]
h23 = 2 + x[2, 1:ii, 0]
h24 = 2 + x[3, 1:ii, 0]
h25 = 2 + x[4, 1:ii, 0]
h26 = 2 + x[5, 1:ii, 0]

H1 = 1 - np.exp(-k[0, 1:ii, 0] * h11) - np.exp(-k[0, 1:ii, 1] * h21)
H2 = 1 - np.exp(-k[1, 1:ii, 0] * h12) - np.exp(-k[1, 1:ii, 1] * h22)
H3 = 1 - np.exp(-k[2, 1:ii, 0] * h13) - np.exp(-k[2, 1:ii, 1] * h23)
H4 = 1 - np.exp(-k[3, 1:ii, 0] * h14) - np.exp(-k[3, 1:ii, 1] * h24)
H5 = 1 - np.exp(-k[4, 1:ii, 0] * h15) - np.exp(-k[4, 1:ii, 1] * h25)
H6 = 1 - np.exp(-k[5, 1:ii, 0] * h16) - np.exp(-k[5, 1:ii, 1] * h26)

# B1 = b3[0, 1:ii] * H1
# B2 = b3[1, 1:ii] * H2
# B3 = b3[2, 1:ii] * H3
# B4 = b3[3, 1:ii] * H4
# B5 = b3[4, 1:ii] * H5

ax_H.plot(t[1:ii], np.zeros((ii - 1,)), linewidth=lwidth, color="k", label="Boundary")
ax_H.plot(t[1:ii], H1, linewidth=lwidth + 1, color=colors[0], label=names[0])
ax_H.plot(t[1:ii], H2, linewidth=lwidth + 1, color=colors[1], label=names[1])
ax_H.plot(t[1:ii], H3, linewidth=lwidth + 1, color=colors[2], label=names[2])
ax_H.plot(t[1:ii], H4, ":", linewidth=lwidth, color=colors[3], label=names[3])
ax_H.plot(t[1:ii], H5, ":", linewidth=lwidth, color=colors[4], label=names[4])
ax_H.plot(t[1:ii], H6, ":", linewidth=lwidth, color=colors[5], label=names[5])

# ax_inset = inset_axes(
#     ax_cz,
#     width="100%",
#     height="100%",
#     bbox_to_anchor=(0.1, 0.025, 0.85, 0.15),
#     bbox_transform=ax_cz.transAxes,
#     loc=3,
# )
# ax_inset.spines["bottom"].set_color("#000000")
# ax_inset.spines["top"].set_color("#000000")
# ax_inset.spines["right"].set_color("#000000")
# ax_inset.spines["left"].set_color("#000000")

# ax_inset.plot(t[1:ii], np.zeros((ii - 1,)), linewidth=lwidth, color="k", label="Boundary")
# ax_inset.plot(t[1:ii], H1, linewidth=lwidth, color=colors[0], label=names[0])
# ax_inset.plot(t[1:ii], H2, linewidth=lwidth, color=colors[1], label=names[1])
# ax_inset.plot(t[1:ii], H3, linewidth=lwidth, color=colors[2], label=names[2])
# ax_inset.plot(t[1:ii], H4, linewidth=lwidth, color=colors[3], label=names[3])
# ax_inset.plot(t[1:ii], H5, linewidth=lwidth, color=colors[4], label=names[4])

# ax_inset.set_xlim([-0.1, 10.1])
# ax_inset.set_ylim([-0.003, 0.003])
# ax_inset.set(xticklabels=[])
# for item in ax_inset.get_yticklabels():
#     item.set_fontsize(15)
# mark_inset(ax_cz, ax_inset, loc1=1, loc2=2, fc="none", ec="0.2", lw=1.5)

ax_H.set(ylabel=r"$H(t, w(t), x(t))$", xlim=[-0.5, 13.3])

# Plot Settings
ax_H.set_xticklabels([])
for item in (
    [ax_H.title, ax_H.xaxis.label, ax_H.yaxis.label]
    + ax_H.get_xticklabels()
    + ax_H.get_yticklabels()
):
    item.set_fontsize(fontsize)
ax_H.legend(fancybox=True, fontsize=legend_font)
ax_H.grid(True, linestyle="dotted", color="white")


### subplot
ax_b3 = fig_H.add_subplot(212)

set_edges_black(ax_b3)

ax_b3.plot(t[1:ii], np.zeros((ii - 1,)), linewidth=lwidth, color="k", label="Boundary")
for aa in [0, 1, 2]:
    ax_b3.plot(t[1:ii], b3[aa, 1:ii], linewidth=lwidth + 1, color=colors[aa], label=names[aa])
for aa in [3, 4, 5]:
    ax_b3.plot(t[1:ii], b3[aa, 1:ii], ":", linewidth=lwidth, color=colors[aa], label=names[aa])
ax_b3.set(ylabel=r"$b_{2c+1}(t, w(t), x(t))$", xlabel=r"$t$", xlim=[-0.5, 13.3])

# Plot Settings
for item in (
    [ax_b3.title, ax_b3.xaxis.label, ax_b3.yaxis.label]
    + ax_b3.get_xticklabels()
    + ax_b3.get_yticklabels()
):
    item.set_fontsize(fontsize)
ax_b3.legend(fancybox=True, fontsize=legend_font)
ax_b3.grid(True, linestyle="dotted", color="white")

plt.tight_layout(pad=2.0)


############################################
### State Trajectories ###
# plt.style.use(['dark_background'])
fig_state = plt.figure(figsize=(12, 8))
ax_state = fig_state.add_subplot(111)
set_edges_black(ax_state)

# Singularities
sing1 = 1.5616 * np.ones((ii - 1,))
sing2 = -1.5616 * np.ones((ii - 1,))
ax_state.plot(t[1:ii], sing1, ":", label=r"$p_1$", color="k", linewidth=lwidth)
ax_state.plot(t[1:ii], sing2, ":", label=r"$p_2$", color="k", linewidth=lwidth)
# Barriers
barr1 = 2.0 * np.ones((ii - 1,))
barr2 = -2.0 * np.ones((ii - 1,))
ax_state.plot(t[1:ii], barr1, label=r"$h_1=0$", color="k", linewidth=lwidth)
ax_state.plot(t[1:ii], barr2, label=r"$h_2=0$", color="k", linewidth=lwidth)
# Desired Trajectory
traj_pos = 4 * np.sin(2 * np.pi * t[1:ii] / 5)
traj_neg = -4 * np.sin(2 * np.pi * t[1:ii] / 5)
ax_state.plot(
    t[1:ii], traj_pos, label=r"$x^*_{\theta=0}$", color="m", linewidth=lwidth, dashes=dash
)


for aa in [0, 1, 2]:
    ax_state.plot(
        t[1:ii],
        x[aa, 1:ii, 0],
        label=names[aa],
        color=colors[aa],
        linewidth=lwidth,
    )
ax_state.plot(
    t[1:ii], traj_neg, label=r"$x^*_{\theta=\pi}$", color="c", linewidth=lwidth, dashes=dash
)
for aa in [3, 4, 5]:
    ax_state.plot(
        t[1:ii],
        x[aa, 1:ii, 0],
        label=names[aa],
        color=colors[aa],
        linewidth=lwidth,
    )
# ax_state.plot(xi[0], yi[0], "o", markersize=10, label=r"$z_0$", color="r")
# ax_state.plot(t[1:ii], xg[0] * np.ones((len(t[1:ii]),)), label="Goal", color="g")

ax_state.set(
    ylim=[-4.05, 4.05],
    xlim=[-2.75, 10.25],
    xlabel=r"$t$",
    ylabel=r"$x$",
)

# Plot Settings
for item in (
    [ax_state.title, ax_state.xaxis.label, ax_state.yaxis.label]
    + ax_state.get_xticklabels()
    + ax_state.get_yticklabels()
):
    item.set_fontsize(fontsize)
# Hide X and Y axes label marks
# ax_map.xaxis.set_tick_params(labelbottom=False)
# ax_map.yaxis.set_tick_params(labelleft=False)
# Hide X and Y axes tick marks
# ax_map.set_xticks([])
# ax_map.set_yticks([])
ax_state.legend(fancybox=True, fontsize=legend_font)
ax_state.grid(False)

plt.tight_layout(pad=2.0)
plt.show()
