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
from models.double_integrator.dynamic.toy_example.timing_params import dt, tf
from models.double_integrator.dynamic.toy_example.initial_conditions import xg, yg, xi, yi
from visualizing.helpers import get_circle

matplotlib.rcParams.update({"figure.autolayout": True})

if platform == "linux" or platform == "linux2":
    # linux
    pre_path = "/home/6lackmitchell/"
elif platform == "darwin":
    # OS X
    pre_path = "/Users/mblack/"
elif platform == "win32":
    # Windows...
    pass

# Specify files
filepath = pre_path + "Documents/git/ccbf-control/data/double_integrator/dynamic/toy_example/paper/"
fname_ccbf = filepath + "ccbf.pkl"
fname_hocbf = filepath + "hocbf.pkl"
fname_ecbf = filepath + "ecbf.pkl"
fname_test = filepath + "test.pkl"

# ### Define Recording Variables ###
t = np.linspace(dt, tf, int(tf / dt))

# Load Simulation Data
with open(fname_ccbf, "rb") as f:
    data = pickle.load(f)

    x_ccbf = np.array([data[a]["x"] for a in data.keys()])
    u_ccbf = np.array([data[a]["u"] for a in data.keys()])
    # h_ccbf = np.array([data[a]["cbf"] for a in data.keys()][:1])
    czero1 = np.array([data[a]["czero1"] for a in data.keys()])
    czero2 = np.array([data[a]["czero2"] for a in data.keys()])
    k = np.array([data[a]["kgains"] if a < 1 else None for a in data.keys()][:1])
    kdot = np.array([data[a]["kdot"] if a < 1 else None for a in data.keys()][:1])
    kdotf = np.array([data[a]["kdotf"] if a < 1 else None for a in data.keys()][:1])
    ii = int(data[0]["ii"] / dt)
with open(fname_hocbf, "rb") as f:
    data = pickle.load(f)

    x_hocbf = np.array([data[a]["x"] for a in data.keys()])
    u_hocbf = np.array([data[a]["u"] for a in data.keys()])
    # h_hocbf = np.array([data[a]["cbf"] for a in data.keys()][1:])
with open(fname_ecbf, "rb") as f:
    data = pickle.load(f)

    x_ecbf = np.array([data[a]["x"] for a in data.keys()])
    u_ecbf = np.array([data[a]["u"] for a in data.keys()])
    # h_ecbf = np.array([data[a]["cbf"] for a in data.keys()][1:])

# Remove zero elements from infeasibility
for aa, xx in enumerate(x_hocbf):
    first_zero = np.where(xx == np.zeros(x_hocbf.shape[2]))[0][0]
    x_hocbf[aa, first_zero:] = x_hocbf[aa, first_zero - 1]
for aa, xx in enumerate(x_ecbf):
    first_zero = np.where(xx == np.zeros(x_ecbf.shape[2]))[0][0]
    x_ecbf[aa, first_zero:] = x_ecbf[aa, first_zero - 1]


x = np.concatenate([x_ccbf[np.newaxis, 0], x_hocbf[3:6], x_ecbf[3:6]])
u = np.concatenate([u_ccbf[np.newaxis, 0], u_hocbf[3:6], u_ecbf[3:6]])
names = ["C-CBF", "HO-CBF-1", "HO-CBF-2", "HO-CBF-3", "E-CBF-1", "E-CBF-2", "E-CBF-3"]

# Plot Settings
nAgents = x.shape[0]
plt.style.use(["ggplot"])
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.plasma(np.linspace(0, 1, nAgents)))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# colors[0] = colors[1]
colors[-1] = np.array([0.0, 0.98, 0.2, 1])
colors.reverse()
lwidth = 4
dash = [3, 2]


def set_edges_black(ax):
    ax.spines["bottom"].set_color("#000000")
    ax.spines["top"].set_color("#000000")
    ax.spines["right"].set_color("#000000")
    ax.spines["left"].set_color("#000000")


plt.close("all")


############################################
### Control Trajectories ###
fig_control = plt.figure(figsize=(12, 8))
ax_cont_a = fig_control.add_subplot(211)
ax_cont_b = fig_control.add_subplot(212)
set_edges_black(ax_cont_a)
set_edges_black(ax_cont_b)

ii_u = int(10 / 0.01)

# Angular Control Inputs
ax_cont_a.plot(t[1:ii_u], 1 * np.ones(t[1:ii_u].shape), linewidth=lwidth + 1, color="k")
ax_cont_a.plot(t[1:ii_u], -1 * np.ones(t[1:ii_u].shape), linewidth=lwidth + 1, color="k")
for aa in range(1, nAgents):
    ax_cont_a.plot(
        t[:ii_u],
        u[aa, :ii_u, 0],
        label=f"{names[aa]}",
        linewidth=lwidth,
        color=colors[aa],
    )
ax_cont_a.plot(
    t[:ii_u], u[0, :ii_u, 0], label=f"{names[0]}", linewidth=lwidth, color=colors[0], dashes=dash
)
ax_cont_a.set(
    ylabel=r"$a_x$",
    ylim=[np.min(u[:ii_u, :, 0]) - 0.1, np.max(u[:ii_u, :, 0]) + 0.1],
    xlim=[-0.1, 13.2],
    # title="Control Inputs",
)
ax_cont_a.set_xticks([])
ax_cont_a.set_yticks([-1, 0, 1])
ax_cont_a.legend(fancybox=True, fontsize=20)

# Acceleration Inputs
ax_cont_b.plot(t[1:ii_u], 1 * np.ones(t[1:ii_u].shape), linewidth=lwidth + 1, color="k")
ax_cont_b.plot(t[1:ii_u], -1 * np.ones(t[1:ii_u].shape), linewidth=lwidth + 1, color="k")
for aa in range(1, nAgents):
    ax_cont_b.plot(
        t[:ii_u],
        u[aa, :ii_u, 1],
        label=f"{names[aa]}",
        linewidth=lwidth,
        color=colors[aa],
    )
ax_cont_b.plot(
    t[:ii_u], u[0, :ii_u, 1], label=f"{names[0]}", linewidth=lwidth, color=colors[0], dashes=dash
)
ax_cont_b.set(
    xlabel=r"$t$ (sec)",
    ylabel=r"$a_y$",
    ylim=[np.min(u[:ii_u, :, 1]) - 0.1, np.max(u[:ii_u, :, 1]) + 0.1],
    xlim=[-0.1, 13.2],
)

# Plot Settings
for item in (
    [ax_cont_a.title, ax_cont_a.xaxis.label, ax_cont_a.yaxis.label]
    + ax_cont_a.get_xticklabels()
    + ax_cont_a.get_yticklabels()
):
    item.set_fontsize(25)
ax_cont_a.grid(True, linestyle="dotted", color="white")

for item in (
    [ax_cont_b.title, ax_cont_b.xaxis.label, ax_cont_b.yaxis.label]
    + ax_cont_b.get_xticklabels()
    + ax_cont_b.get_yticklabels()
):
    item.set_fontsize(25)
ax_cont_b.grid(True, linestyle="dotted", color="white")
ax_cont_b.set_xticks([0, 2, 4, 6, 8, 10])
ax_cont_b.set_yticks([-1, 0, 1])
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
### CZero Trajectories ###
fig_cz = plt.figure(figsize=(8, 8))
ax_cz = fig_cz.add_subplot(111)
set_edges_black(ax_cz)

ax_cz.plot(t[1:ii], czero1[0, 1:ii], linewidth=lwidth, color=colors[1], label="C_01")
ax_cz.plot(t[1:ii], czero2[0, 1:ii], linewidth=lwidth, color=colors[2], label="C_02")
ax_cz.set(ylabel=r"$b_{c+1}$", title="Sufficient Control Authority")

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


############################################
### CBF Trajectories ###
fig_cbf = plt.figure(figsize=(10, 6))
ax_cbf = fig_cbf.add_subplot(111)
set_edges_black(ax_cbf)

gain1 = 2.0
gain2 = 2.0
gain3 = 2.0
R1 = 0.5
cx1 = 1.0
cy1 = 1.0
R2 = 0.5
cx2 = 1.5
cy2 = 2.25
R3 = 0.5
cx3 = 2.4
cy3 = 1.5
gain4 = 5.0
gain5 = 5.0
gain6 = 10.0

ii_h6 = int(8 / 0.01)
h1 = gain1 * ((x_ccbf[0, 1:ii_u, 0] - cx1) ** 2 + (x_ccbf[0, 1:ii_u, 1] - cy1) ** 2 - R1**2)
h2 = gain2 * ((x_ccbf[0, 1:ii_u, 0] - cx2) ** 2 + (x_ccbf[0, 1:ii_u, 1] - cy2) ** 2 - R2**2)
h3 = gain3 * ((x_ccbf[0, 1:ii_u, 0] - cx3) ** 2 + (x_ccbf[0, 1:ii_u, 1] - cy3) ** 2 - R2**2)
h4 = gain4 * (1 - x_ccbf[0, 1:ii_u, 2] ** 2)
h5 = gain5 * (1 - x_ccbf[0, 1:ii_u, 3] ** 2)
# h6_a = gain6 * (
#     (((2) ** 2) / 4 + (10 - t[1:ii_h6]) ** 2) / 4
#     - (x_ccbf[0, 1:ii_h6, 0] - 2) ** 2
#     - (x_ccbf[0, 1:ii_h6, 1] - 2) ** 2
# )
# h6_b = gain6 * (
#     ((2) ** 2) / 4 - (x_ccbf[0, ii_h6:ii_u, 0] - 2) ** 2 - (x_ccbf[0, ii_h6:ii_u, 1] - 2) ** 2
# )
h6 = gain6 * (
    (1 + (10 - t[1:ii_u]) ** 2) / 4
    - (x_ccbf[0, 1:ii_u, 0] - 2) ** 2
    - (x_ccbf[0, 1:ii_u, 1] - 2) ** 2
)
# h6 = np.concatenate([h6_a, h6_b])
hh = np.array([h1, h2, h3, h4, h5, h6])
summer = np.array([np.exp(-k[0, 1:ii_u, cc] * hh[cc]) for cc in range(6)])
H_ccbf = 1 - np.sum(summer, axis=0)


ax_cbf.plot(t[1:ii_u], np.zeros((ii_u - 1,)), ":", linewidth=lwidth, color="k", label="Barrier")
ax_cbf.plot(t[1:ii_u], h1, linewidth=lwidth, color=colors[1], label=r"$h_1$")
ax_cbf.plot(t[1:ii_u], h2, linewidth=lwidth, color=colors[2], label=r"$h_2$")
ax_cbf.plot(t[1:ii_u], h3, linewidth=lwidth, color=colors[3], label=r"$h_3$")
ax_cbf.plot(t[1:ii_u], h4, linewidth=lwidth, color=colors[4], label=r"$h_4$")
ax_cbf.plot(t[1:ii_u], h5, linewidth=lwidth, color=colors[5], label=r"$h_5$")
ax_cbf.plot(t[1:ii_u], h6, linewidth=lwidth, color=colors[6], label=r"$h_6$")
ax_cbf.plot(t[1:ii_u], H_ccbf, linewidth=lwidth, color=colors[0], dashes=dash, label=r"$H$")
ax_cbf.set(
    ylabel="Constraint Function Value", xlabel=r"$t$ (sec)", xlim=[-0.25, 13.25], ylim=[-0.1, 5.25]
)

# Plot Settings
for item in (
    [ax_cbf.title, ax_cbf.xaxis.label, ax_cbf.yaxis.label]
    + ax_cbf.get_xticklabels()
    + ax_cbf.get_yticklabels()
):
    item.set_fontsize(25)
ax_cbf.legend(fancybox=True, fontsize=20)
ax_cbf.grid(True, linestyle="dotted", color="white")

plt.tight_layout(pad=2.0)


############################################
### State Trajectories ###
# plt.style.use(['dark_background'])
fig_static_map = plt.figure(figsize=(10, 7.5))
ax_map = fig_static_map.add_subplot(111)
set_edges_black(ax_map)


d_points = 100
xtar, ytar = get_circle(np.array([2, 2]), 0.1, d_points)
xc1, yc1 = get_circle(np.array([1, 1]), 0.5, d_points)
xc2, yc2 = get_circle(np.array([1.5, 2.25]), 0.5, d_points)
xc3, yc3 = get_circle(np.array([2.4, 1.5]), 0.5, d_points)
ax_map.plot(xc1, yc1, linewidth=lwidth + 1, color="k")
ax_map.plot(xc2, yc2, linewidth=lwidth + 1, color="k")
ax_map.plot(xc3, yc3, linewidth=lwidth + 1, color="k")
ax_map.plot(xtar, ytar, ":", linewidth=lwidth + 1, color="g")

for aaa in range(1, nAgents):
    ax_map.plot(
        x[aaa, :ii, 0], x[aaa, :ii, 1], label=names[aaa], color=colors[aaa], linewidth=lwidth
    )
ax_map.plot(
    x[0, :ii, 0], x[0, :ii, 1], label=names[0], color=colors[0], linewidth=lwidth, dashes=dash
)
ax_map.plot(xi[0], yi[0], "o", markersize=10, label=r"$z_0$", color="r")
ax_map.plot(xg[0], yg[0], "*", markersize=12, label="Goal", color="g")

ax_map.set(ylim=[-0.75, 2.25], xlim=[-0.5, 3.5], xlabel="X (m)", ylabel="Y (m)")

# Plot Settings
for item in (
    [ax_map.title, ax_map.xaxis.label, ax_map.yaxis.label]
    + ax_map.get_xticklabels()
    + ax_map.get_yticklabels()
):
    item.set_fontsize(25)
# Hide X and Y axes label marks
# ax_map.xaxis.set_tick_params(labelbottom=False)
# ax_map.yaxis.set_tick_params(labelleft=False)
# Hide X and Y axes tick marks
# ax_map.set_xticks([])
# ax_map.set_yticks([])
ax_map.legend(fancybox=True, fontsize=15)
ax_map.grid(False)


# ############################################
# ### State Trajectories ###
# # plt.style.use(['dark_background'])
# fig_map = plt.figure(figsize=(10, 10))
# ax_pos = fig_map.add_subplot(111)
# set_edges_black(ax_pos)

# # # Set Up Road
# d_points = 100
# xc1, yc1 = get_circle(np.array([1, 1]), 0.5, d_points)
# xc2, yc2 = get_circle(np.array([1.5, 2.25]), 0.5, d_points)
# xc3, yc3 = get_circle(np.array([2.4, 1.5]), 0.5, d_points)
# ax_pos.plot(xc1, yc1, linewidth=lwidth + 1, color="k")
# ax_pos.plot(xc2, yc2, linewidth=lwidth + 1, color="k")
# ax_pos.plot(xc3, yc3, linewidth=lwidth + 1, color="k")

# for aaa in range(nAgents):
#     if aaa == 0:
#         lbl = "Goal"
#     else:
#         lbl = None
#     ax_pos.plot(xg[aaa], yg[aaa], "*", markersize=10, label=lbl)

# # Create variable reference to plot
# map_vid = []
# for aa in range(nAgents):
#     if aa == 0:
#         lbl = "C-CBF Robot"
#         col = "b"
#     elif aa == 3:
#         lbl = "Uncontrolled Agent"
#         col = "r"
#     else:
#         lbl = None
#         col = None
#     map_vid.append(ax_pos.plot([], [], linewidth=lwidth, label=lbl, color=col)[0])
#     map_vid.append(ax_pos.quiver([], [], [], [], linewidth=lwidth))
#     map_vid.append(ax_pos.plot([], [], linewidth=lwidth, dashes=dash)[0])

# # Add text annotation and create variable reference
# txt = ax_pos.text(-6.8, -13.8, "", ha="right", va="top", fontsize=24)
# # txt_list = [ax_pos.text(x[aa, 0, 0], x[aa, 0, 1] + 0.25, '{}'.format(aa + 1),
# #                         ha='right', va='top', fontsize=12) if (-10 < x[aa, 0, 0] < 10) and (-15 < x[aa, 0, 1] < 10) else None for aa in range(nAgents)]

# ax_pos.set(ylim=[-1.0, 3.0], xlim=[-1.0, 3.0], xlabel="X (m)", ylabel="Y (m)")

# # Plot Settings
# for item in (
#     [ax_pos.title, ax_pos.xaxis.label, ax_pos.yaxis.label]
#     + ax_pos.get_xticklabels()
#     + ax_pos.get_yticklabels()
# ):
#     item.set_fontsize(25)
# # Hide X and Y axes label marks
# ax_pos.xaxis.set_tick_params(labelbottom=False)
# ax_pos.yaxis.set_tick_params(labelleft=False)
# # Hide X and Y axes tick marks
# ax_pos.set_xticks([])
# ax_pos.set_yticks([])
# ax_pos.legend(fancybox=True, fontsize=15)
# ax_pos.grid(False)

# time_scale_factor = 10


# def animate_ego(jj):
#     jj = int(jj * time_scale_factor)
#     last_1_sec = 40
#     ego_pos = x[0, jj, 0:2]
#     for aa in range(0, 3 * nAgents, 3):

#         idx = int(aa / 3)
#         # if not (-10 < x[idx, jj, 0] < 10):
#         #     continue

#         x_circ, y_circ = get_circle(x[idx, jj], 0.025, d_points)
#         map_vid[aa].set_data(x_circ, y_circ)
#         map_vid[aa].set_color(clr[idx])

#     txt.set_text("{:.1f} sec".format(jj * dt))
#     ax_pos.set(ylim=[-1.0, 3.0], xlim=[-1.0, 3.0])


# # Create animation
# ani = animation.FuncAnimation(
#     fig=fig_map, func=animate_ego, frames=int(ii / time_scale_factor), interval=10, repeat=False
# )
# # writer = animation.writers["ffmpeg"]
# # ani.save(filename[:-4] + ".mp4", writer=writer(fps=15))

plt.tight_layout(pad=2.0)
plt.show()
