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

nAgents = 9
N = nAgents
# plt.style.use(['Solarize_Light2'])
plt.style.use(["ggplot"])
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, N)))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# colors.append(np.array([0.98, 0.6, 0.2, 1]))
# colors.append(np.array([0.98, 0.01, 0.4, 1]))
# colors.append(np.array([0.0, 0.98, 0.2, 1]))
# colors.append(np.array([0.0, 0.98, 0.2, 1]))
# colors[0] = colors[1]
colors.reverse()
colors[0] = np.array([0.0, 1, 1, 1])

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

filepath = pre_path + "ccbf-control/data/bicycle/dynamic/toy_example/"

# ### Define Recording Variables ###
t = np.linspace(dt, tf, int(tf / dt))

# Specify files
fname1 = filepath + "success/success.pkl"
# fname1 = filepath + "success/better.pkl"
fname2 = filepath + "hocbf/alpha_0p1.pkl"
fname3 = filepath + "hocbf/alpha_1.pkl"
fname4 = filepath + "hocbf/alpha_2.pkl"
fname5 = filepath + "hocbf/alpha_5.pkl"
fname6 = filepath + "ecbf/1.pkl"
fname7 = filepath + "ecbf/2.pkl"
fname8 = filepath + "ecbf/3.pkl"
fname9 = filepath + "ecbf/4.pkl"

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
    cbfs = np.array([data[a]["cbf"] for a in data.keys()])
    ccbf = np.array([data[a]["ccbf"] for a in data.keys()])
    ii = int(data[0]["ii"] / dt)
with open(fname2, "rb") as f:
    data = pickle.load(f)

    x2 = np.array([data[a]["x"] for a in data.keys()])
    u2 = np.array([data[a]["u"] for a in data.keys()])
    u02 = np.array([data[a]["u0"] for a in data.keys()])

with open(fname3, "rb") as f:
    data = pickle.load(f)

    x3 = np.array([data[a]["x"] for a in data.keys()])
    u3 = np.array([data[a]["u"] for a in data.keys()])
    u03 = np.array([data[a]["u0"] for a in data.keys()])

with open(fname4, "rb") as f:
    data = pickle.load(f)

    x4 = np.array([data[a]["x"] for a in data.keys()])
    u4 = np.array([data[a]["u"] for a in data.keys()])
    u04 = np.array([data[a]["u0"] for a in data.keys()])

with open(fname5, "rb") as f:
    data = pickle.load(f)

    x5 = np.array([data[a]["x"] for a in data.keys()])
    u5 = np.array([data[a]["u"] for a in data.keys()])
    u05 = np.array([data[a]["u0"] for a in data.keys()])


with open(fname6, "rb") as f:
    data = pickle.load(f)

    x6 = np.array([data[a]["x"] for a in data.keys()])
    u6 = np.array([data[a]["u"] for a in data.keys()])
    u06 = np.array([data[a]["u0"] for a in data.keys()])

with open(fname7, "rb") as f:
    data = pickle.load(f)

    x7 = np.array([data[a]["x"] for a in data.keys()])
    u7 = np.array([data[a]["u"] for a in data.keys()])
    u07 = np.array([data[a]["u0"] for a in data.keys()])

with open(fname8, "rb") as f:
    data = pickle.load(f)

    x8 = np.array([data[a]["x"] for a in data.keys()])
    u8 = np.array([data[a]["u"] for a in data.keys()])
    u08 = np.array([data[a]["u0"] for a in data.keys()])


with open(fname9, "rb") as f:
    data = pickle.load(f)

    x9 = np.array([data[a]["x"] for a in data.keys()])
    u9 = np.array([data[a]["u"] for a in data.keys()])
    u09 = np.array([data[a]["u0"] for a in data.keys()])


x = np.concatenate(
    [
        x1[np.newaxis, 0],
        x2[np.newaxis, 0],
        x3[np.newaxis, 0],
        x4[np.newaxis, 0],
        x5[np.newaxis, 0],
        x6[np.newaxis, 0],
        x7[np.newaxis, 0],
        x8[np.newaxis, 0],
        x9[np.newaxis, 0],
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
        u7[np.newaxis, 0],
        u8[np.newaxis, 0],
        u9[np.newaxis, 0],
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
        u07[np.newaxis, 0],
        u08[np.newaxis, 0],
        u09[np.newaxis, 0],
    ]
)
b3 = b311[0]
k = k1[0]
kdot = kdot1[0]

legend_font = 19
fontsize = 30
lwidth = 3
dash = [3, 2]
color_idx = np.array(range(0, 2 * nAgents)).reshape(nAgents, 2)
names = [
    r"$C-CBF$",
    r"$HO-CBF 1$",
    r"$HO-CBF 2$",
    r"$HO-CBF 3$",
    r"$HO-CBF 4$",
    r"$E-CBF 1$",
    r"$E-CBF 2$",
    r"$E-CBF 3$",
    r"$E-CBF 4$",
]


def set_edges_black(ax):
    ax.spines["bottom"].set_color("#000000")
    ax.spines["top"].set_color("#000000")
    ax.spines["right"].set_color("#000000")
    ax.spines["left"].set_color("#000000")


plt.close("all")

############################################
### Control Trajectories ###
fig_control = plt.figure(figsize=(10, 7.5))
ax_cont_a = fig_control.add_subplot(211)
ax_cont_b = fig_control.add_subplot(212)
set_edges_black(ax_cont_a)
set_edges_black(ax_cont_b)

tf = 5.0
ii_u = int(tf / 0.01)

# Angular Control Inputs
w_max = np.pi / 2
ax_cont_a.plot(t[1:ii_u], w_max * np.ones(t[1:ii_u].shape), linewidth=lwidth + 1, color="k")
ax_cont_a.plot(t[1:ii_u], -w_max * np.ones(t[1:ii_u].shape), linewidth=lwidth + 1, color="k")
for aa in range(1, nAgents):
    if aa < 5:
        lbl = names[aa]
    else:
        lbl = None
    ax_cont_a.plot(
        t[:ii_u],
        u[aa, :ii_u, 0],
        label=lbl,
        linewidth=lwidth,
        color=colors[aa],
    )
ax_cont_a.plot(t[:ii_u], u[0, :ii_u, 0], label=f"{names[0]}", linewidth=lwidth, color=colors[0])
ax_cont_a.set(
    ylabel=r"$\omega$",
    ylim=[-w_max - 0.1, w_max + 0.1],
    xlim=[-0.1, tf + 2.5],
)
ax_cont_a.set_xticks([])
ax_cont_a.set_yticks([-1, 0, 1])
ax_cont_a.legend(fancybox=True, fontsize=20)

# Acceleration Inputs
a_max = 0.5
ax_cont_b.plot(t[1:ii_u], a_max * np.ones(t[1:ii_u].shape), linewidth=lwidth + 1, color="k")
ax_cont_b.plot(t[1:ii_u], -a_max * np.ones(t[1:ii_u].shape), linewidth=lwidth + 1, color="k")
for aa in range(1, nAgents):
    if aa >= 5:
        lbl = names[aa]
    else:
        lbl = None
    ax_cont_b.plot(
        t[:ii_u],
        u[aa, :ii_u, 1],
        label=lbl,
        linewidth=lwidth,
        color=colors[aa],
    )
ax_cont_b.plot(t[:ii_u], u[0, :ii_u, 1], linewidth=lwidth, color=colors[0])
ax_cont_b.set(
    xlabel=r"$t$",
    ylabel=r"$a$",
    ylim=[-a_max - 0.1, a_max + 0.1],
    xlim=[-0.1, tf + 2.5],
)
ax_cont_b.legend(fancybox=True, fontsize=20)


# Plot Settings
for item in (
    [ax_cont_a.title, ax_cont_a.xaxis.label, ax_cont_a.yaxis.label]
    + ax_cont_a.get_xticklabels()
    + ax_cont_a.get_yticklabels()
):
    item.set_fontsize(fontsize)
ax_cont_a.grid(True, linestyle="dotted", color="white")

for item in (
    [ax_cont_b.title, ax_cont_b.xaxis.label, ax_cont_b.yaxis.label]
    + ax_cont_b.get_xticklabels()
    + ax_cont_b.get_yticklabels()
):
    item.set_fontsize(fontsize)
ax_cont_b.grid(True, linestyle="dotted", color="white")
# ax_cont_b.set_xticks([0, 2, 4, 6, 8, 10])
# ax_cont_b.set_yticks([-1, 0, 1])
# ax_cont_b.legend(fancybox=True, fontsize=20)

plt.tight_layout(pad=2.0)

# ############################################
# ### Gain Trajectories ###
# fig_k = plt.figure(figsize=(8, 8))
# ax_k = fig_k.add_subplot(111)
# set_edges_black(ax_k)

# # Angular Control Inputs
# lbl = [
#     "O1",
#     "O2",
#     "O3",
#     "S1",
#     "S2",
# ]
# clr = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# clr.reverse()
# for cbf in range(k.shape[2]):
#     ax_k.plot(t[1:ii], k[0, 1:ii, cbf], linewidth=lwidth, color=clr[cbf], label=lbl[cbf])
# ax_k.set(ylabel=r"$w$", title="C-CBF Weights")

# # Plot Settings
# for item in (
#     [ax_k.title, ax_k.xaxis.label, ax_k.yaxis.label]
#     + ax_k.get_xticklabels()
#     + ax_k.get_yticklabels()
# ):
#     item.set_fontsize(25)
# ax_k.legend(fancybox=True)
# ax_k.grid(True, linestyle="dotted", color="white")

# plt.tight_layout(pad=2.0)


# ############################################
# ### Kdot Trajectories ###
# fig_kdot = plt.figure(figsize=(8, 8))
# ax_kdot = fig_kdot.add_subplot(111)
# set_edges_black(ax_kdot)

# for cbf in range(k.shape[2]):
#     ax_kdot.plot(t[1:ii], kdot[0, 1:ii, cbf], linewidth=lwidth, color=clr[cbf], label=lbl[cbf])
#     ax_kdot.plot(
#         t[1:ii],
#         kdotf[0, 1:ii, cbf],
#         "-.",
#         linewidth=lwidth,
#         color=clr[cbf],
#         label=lbl[cbf],
#     )
# ax_kdot.set(ylabel=r"$\dot{w}$", title="Weight Derivatives")

# # Plot Settings
# for item in (
#     [ax_kdot.title, ax_kdot.xaxis.label, ax_kdot.yaxis.label]
#     + ax_kdot.get_xticklabels()
#     + ax_kdot.get_yticklabels()
# ):
#     item.set_fontsize(25)
# ax_kdot.legend(fancybox=True)
# ax_kdot.grid(True, linestyle="dotted", color="white")

# plt.tight_layout(pad=2.0)

############################################
### Gain Trajectories ###
fig_k = plt.figure(figsize=(10, 7.5))
ax_k = fig_k.add_subplot(211)
set_edges_black(ax_k)

cbf_names = [
    r"Obstacle 1",
    r"Obstacle 2",
    r"Obstacle 3",
    r"Obstacle 4",
    r"Obstacle 5",
    r"Reach",
    r"Speed",
    r"Slip",
]
for cc in range(k.shape[1]):
    if cc < 5:
        lbl = cbf_names[cc]
    else:
        lbl = None
    ax_k.plot(
        t[2:ii],
        k[2:ii, cc],
        linewidth=lwidth + 1,
        color=colors[cc],
        label=lbl,
    )
ax_k.set(ylabel=r"$w$", xlim=[-0.1, 5.75])
ax_k.set_xticklabels([])

# Plot Settings
for item in (
    [ax_k.title, ax_k.xaxis.label, ax_k.yaxis.label]
    + ax_k.get_xticklabels()
    + ax_k.get_yticklabels()
):
    item.set_fontsize(fontsize)
ax_k.set_yticks([0, 5, 10, 15])
ax_k.legend(fancybox=True, fontsize=legend_font)
ax_k.grid(True, linestyle="dotted", color="white")

plt.tight_layout(pad=2.0)

############################################
### W-Dot Trajectories ###
ax_wdot = fig_k.add_subplot(212)
set_edges_black(ax_wdot)

for cc in range(kdot.shape[1]):
    if cc >= 5:
        lbl = cbf_names[cc]
    else:
        lbl = None
    ax_wdot.plot(
        t[2:ii],
        kdot[2:ii, cc],
        linewidth=lwidth + 1,
        color=colors[cc],
        label=lbl,
    )
ax_wdot.set(ylabel=r"$\dot{w}$", xlabel=r"$t$", xlim=[-0.1, 5.75], ylim=[-50, 50])
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


# ############################################
# ### CZero Trajectories ###
# fig_cz = plt.figure(figsize=(10, 7.5))
# ax_cz = fig_cz.add_subplot(111)
# set_edges_black(ax_cz)

# ax_cz.plot(t[1:ii], b311[0, 1:ii], linewidth=lwidth, color=colors[1], label="C_01")
# ax_cz.set(xlabel=r"$t$", ylabel=r"$b_{2c+1}$")  # , title="Sufficient Control Authority")

# # Plot Settings
# for item in (
#     [ax_cz.title, ax_cz.xaxis.label, ax_cz.yaxis.label]
#     + ax_cz.get_xticklabels()
#     + ax_cz.get_yticklabels()
# ):
#     item.set_fontsize(fontsize)
# # ax_cz.legend(fancybox=True, fontsize=legend_font)
# ax_cz.grid(True, linestyle="dotted", color="white")

# plt.tight_layout(pad=2.0)


# ############################################
# ### CBF Trajectories ###
# fig_cbf = plt.figure(figsize=(10, 6))
# ax_cbf = fig_cbf.add_subplot(111)
# set_edges_black(ax_cbf)

gain = 5.0
R = 0.45
offset = 0.25
cx1 = 0.8
cy1 = 1.1
cx2 = 1.25
cy2 = 2.25
cx3 = 2.5
cy3 = 1.75
cx4 = 2.0
cy4 = 0.25
cx5 = 0.8
cy5 = -0.25

# # ii_h6 = int(8 / 0.01)
# h1 = gain * ((x[0, 1:ii_u, 0] - cx1) ** 2 + (x[0, 1:ii_u, 1] - cy1) ** 2 - R**2)
# h2 = gain * ((x[0, 1:ii_u, 0] - cx2) ** 2 + (x[0, 1:ii_u, 1] - cy2) ** 2 - R**2)
# h3 = gain * ((x[0, 1:ii_u, 0] - cx3) ** 2 + (x[0, 1:ii_u, 1] - cy3) ** 2 - R**2)
# h4 = gain * ((x[0, 1:ii_u, 0] - cx4) ** 2 + (x[0, 1:ii_u, 1] - cy4) ** 2 - R**2)
# h5 = gain * ((x[0, 1:ii_u, 0] - cx5) ** 2 + (x[0, 1:ii_u, 1] - cy5) ** 2 - R**2)
# h6 = gain4 * (1 - x[0, 1:ii_u, 2] ** 2)
# h6 = gain4 * (1 - x[0, 1:ii_u, 2] ** 2)
# h7 = gain5 * (1 - x[0, 1:ii_u, 3] ** 2)
# # h6_a = gain6 * (
# #     (((2) ** 2) / 4 + (10 - t[1:ii_h6]) ** 2) / 4
# #     - (x[0, 1:ii_h6, 0] - 2) ** 2
# #     - (x[0, 1:ii_h6, 1] - 2) ** 2
# # )
# # h6_b = gain6 * (
# #     ((2) ** 2) / 4 - (x[0, ii_h6:ii_u, 0] - 2) ** 2 - (x[0, ii_h6:ii_u, 1] - 2) ** 2
# # )
# h6 = gain6 * (
#     (1 + (10 - t[1:ii_u]) ** 2) / 4 - (x[0, 1:ii_u, 0] - 2) ** 2 - (x[0, 1:ii_u, 1] - 2) ** 2
# )
# # h6 = np.concatenate([h6_a, h6_b])
# hh = np.array([h1, h2, h3, h4, h5, h6])
# summer = np.array([np.exp(-k[1:ii_u, cc] * hh[cc]) for cc in range(6)])
# H_ccbf = 1 - np.sum(summer, axis=0)


# ax_cbf.plot(t[1:ii_u], np.zeros((ii_u - 1,)), ":", linewidth=lwidth, color="k", label="Barrier")
# ax_cbf.plot(t[1:ii_u], h1, linewidth=lwidth, color=colors[1], label=r"$h_1$")
# ax_cbf.plot(t[1:ii_u], h2, linewidth=lwidth, color=colors[2], label=r"$h_2$")
# ax_cbf.plot(t[1:ii_u], h3, linewidth=lwidth, color=colors[3], label=r"$h_3$")
# ax_cbf.plot(t[1:ii_u], h4, linewidth=lwidth, color=colors[4], label=r"$h_4$")
# ax_cbf.plot(t[1:ii_u], h5, linewidth=lwidth, color=colors[5], label=r"$h_5$")
# ax_cbf.plot(t[1:ii_u], h6, linewidth=lwidth, color=colors[6], label=r"$h_6$")
# ax_cbf.plot(t[1:ii_u], H_ccbf, linewidth=lwidth, color=colors[0], dashes=dash, label=r"$H$")
# ax_cbf.set(
#     ylabel="Constraint Function Value", xlabel=r"$t$ (sec)", xlim=[-0.25, 13.25], ylim=[-0.1, 5.25]
# )

# # Plot Settings
# for item in (
#     [ax_cbf.title, ax_cbf.xaxis.label, ax_cbf.yaxis.label]
#     + ax_cbf.get_xticklabels()
#     + ax_cbf.get_yticklabels()
# ):
#     item.set_fontsize(25)
# ax_cbf.legend(fancybox=True, fontsize=20)
# ax_cbf.grid(True, linestyle="dotted", color="white")

# plt.tight_layout(pad=2.0)


############################################
### CBF Trajectories ###
fig_cbfs = plt.figure(figsize=(10, 7.5))
ax_cbfs = fig_cbfs.add_subplot(111)
set_edges_black(ax_cbfs)

cbf_names = [
    r"Obstacle 1",
    r"Obstacle 2",
    r"Obstacle 3",
    r"Obstacle 4",
    r"Obstacle 5",
    r"Reach",
    r"Speed",
    r"Slip",
]

# Normalize
cbfs[0, :ii, 0] = cbfs[0, :ii, 0] / 5
cbfs[0, :ii, 1] = cbfs[0, :ii, 1] / 5
cbfs[0, :ii, 2] = cbfs[0, :ii, 2] / 5
cbfs[0, :ii, 3] = cbfs[0, :ii, 3] / 5
cbfs[0, :ii, 4] = cbfs[0, :ii, 4] / 5
cbfs[0, :ii, 5] = cbfs[0, :ii, 5] / 1
cbfs[0, :ii, 6] = cbfs[0, :ii, 6] / 5
cbfs[0, :ii, 7] = cbfs[0, :ii, 7] / 5

ax_cbfs.plot(t[1:ii], np.zeros(t[1:ii].shape), linewidth=lwidth + 1, color="k", label="Zero")
ax_cbfs.plot(t[1:ii], ccbf[0, 1:ii], linewidth=lwidth, color="m", dashes=dash, label="C-CBF")
for cc in range(cbfs.shape[2]):
    ax_cbfs.plot(
        t[:ii],
        cbfs[0, :ii, cc],
        label=cbf_names[cc],
        linewidth=lwidth,
        color=colors[cc],
    )
ax_cbfs.plot(t[1:ii], b311[0, 1:ii], ":", linewidth=lwidth, color="r", label=r"$b_{2c+1}$")

ax_inset = inset_axes(
    ax_cbfs,
    width="100%",
    height="100%",
    bbox_to_anchor=(0.625, 0.02, 0.3, 0.2),
    bbox_transform=ax_cbfs.transAxes,
    loc=3,
)
ax_inset.spines["bottom"].set_color("#000000")
ax_inset.spines["top"].set_color("#000000")
ax_inset.spines["right"].set_color("#000000")
ax_inset.spines["left"].set_color("#000000")

ax_inset.plot(t[1:ii], np.zeros((ii - 1,)), linewidth=lwidth, color="k", label="Boundary")
ax_inset.plot(t[1:ii], b311[0, 1:ii], ":", linewidth=lwidth, color="r", label=r"$b_{2c+1}$")

ax_inset.set_ylim([-0.05, 0.025])
ax_inset.set_xlim([0.5, 1.5])
ax_inset.set(xticklabels=[])
ax_inset.yaxis.set_label_position("right")
ax_inset.yaxis.tick_right()
mark_inset(ax_cbfs, ax_inset, loc1=3, loc2=2, fc="none", ec="0.2", lw=1.5)

# ax_cbfs.plot(t[:ii], cbf[aa, :ii, 1], label='h_{}^0'.format(aa), linewidth=lwidth,
#                color=colors[color_idx[aa, 1]], dashes=dash)
ax_cbfs.set(
    ylabel="Constraint Fcn. Vals.",
    xlabel=r"$t$",
    ylim=[np.min(b311[0, 1:ii]) - 0.25, 5],
    xlim=[-0.1, 6.5],
)  # , title="CBF Trajectories")

# Plot Settings
for item in (
    [ax_cbfs.title, ax_cbfs.xaxis.label, ax_cbfs.yaxis.label]
    + ax_cbfs.get_xticklabels()
    + ax_cbfs.get_yticklabels()
):
    item.set_fontsize(fontsize)
ax_cbfs.legend(fancybox=True, fontsize=legend_font)
ax_cbfs.grid(True, linestyle="dotted", color="white")

plt.tight_layout(pad=2.0)


############################################
### State Trajectories ###
# plt.style.use(['dark_background'])
fig_static_map = plt.figure(figsize=(10, 7.5))
ax_map = fig_static_map.add_subplot(111)
set_edges_black(ax_map)

xi = 0.0
yi = 0.0
xg = 2.0
yg = 2.0

Ri = 4
Rt1 = 0.1 + Ri * (1 - 3 / 5)
Rt2 = 0.1 + Ri * (1 - 4 / 5)
Rt3 = 0.1 + Ri * (1 - 4.75 / 5)


d_points = 100
xtar, ytar = get_circle(np.array([xg, yg]), 0.1, d_points)
xtar1, ytar1 = get_circle(np.array([xg, yg]), Rt1, d_points)
xtar2, ytar2 = get_circle(np.array([xg, yg]), Rt2, d_points)
xtar3, ytar3 = get_circle(np.array([xg, yg]), Rt3, d_points)
xc1, yc1 = get_circle(np.array([cx1 + offset, cy1 + offset]), R, d_points)
xc2, yc2 = get_circle(np.array([cx2 + offset, cy2 + offset]), R, d_points)
xc3, yc3 = get_circle(np.array([cx3 + offset, cy3 + offset]), R, d_points)
xc4, yc4 = get_circle(np.array([cx4 + offset, cy4 + offset]), R, d_points)
xc5, yc5 = get_circle(np.array([cx5 + offset, cy5 + offset]), R, d_points)
ax_map.plot(xc1, yc1, linewidth=lwidth + 1, color="k")
ax_map.plot(xc2, yc2, linewidth=lwidth + 1, color="k")
ax_map.plot(xc3, yc3, linewidth=lwidth + 1, color="k")
ax_map.plot(xc4, yc4, linewidth=lwidth + 1, color="k")
ax_map.plot(xc5, yc5, linewidth=lwidth + 1, color="k")
ax_map.plot(xtar1, ytar1, dashes=dash, linewidth=lwidth + 1, color="k", alpha=0.25)
ax_map.plot(xtar2, ytar2, dashes=dash, linewidth=lwidth + 1, color="k", alpha=0.25)
ax_map.plot(xtar3, ytar3, dashes=dash, linewidth=lwidth + 1, color="k", alpha=0.25)
th = -np.pi / 2.75
ax_map.text(
    xg + (Rt1 + 0.12) * np.cos(th),
    yg + (Rt1 + 0.12) * np.sin(th),
    r"$t = 3.0$",
    fontsize=legend_font,
)
ax_map.text(
    xg + (Rt2 + 0.12) * np.cos(th),
    yg + (Rt2 + 0.12) * np.sin(th),
    r"$t = 4.0$",
    fontsize=legend_font,
)
ax_map.text(
    xg + (Rt3 + 0.2) * np.cos(th - 0.4),
    yg + (Rt3 + 0.2) * np.sin(th - 0.4),
    r"$t = 4.75$",
    fontsize=legend_font,
)

ax_map.plot(xtar, ytar, ":", linewidth=lwidth + 1, color="g")

# for aaa in range(1, nAgents):
#     ax_map.plot(
#         x[aaa, :ii, 0], x[aaa, :ii, 1], label=names[aaa], color=colors[aaa], linewidth=lwidth
#     )
# ax_map.plot(x[0, :ii, 0], x[0, :ii, 1], label=names[0], color=colors[0], linewidth=lwidth)
ax_map.plot(xi, yi, "o", markersize=10, label=r"$z_0$", color="r")
ax_map.plot(xg, yg, "*", markersize=12, label="Goal", color="g")

ax_map.set(ylim=[-0.25, 2.75], xlim=[-0.75, 3.25], xlabel="X (m)", ylabel="Y (m)")

# Plot Settings
for item in (
    [ax_map.title, ax_map.xaxis.label, ax_map.yaxis.label]
    + ax_map.get_xticklabels()
    + ax_map.get_yticklabels()
):
    item.set_fontsize(fontsize)
# Hide X and Y axes label marks
# ax_map.xaxis.set_tick_params(labelbottom=False)
# ax_map.yaxis.set_tick_params(labelleft=False)
# Hide X and Y axes tick marks
# ax_map.set_xticks([])
# ax_map.set_yticks([])
ax_map.legend(fancybox=True, fontsize=legend_font)
ax_map.grid(False)
ax_map.set_rasterized(True)
plt.savefig("rasterized_fig.eps")


##############################

plt.tight_layout(pad=2.0)
plt.show()
