import matplotlib
matplotlib.use("Qt5Agg")

import os
import glob
import pickle
import traceback
import numpy as np
import latex

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from matplotlib import rcParams, cycler
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits import mplot3d
import pandas as pd

from bicycle.bicycle_settings import *
from viz.helpers import get_circle

# with plt.style.context('seaborn-colorblind'):
#     plt.rcParams["axes.edgecolor"] = "1"
#     plt.rcParams["axes.linewidth"] = "2"

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'figure.autolayout': True})
# plt.style.use(['fivethirtyeight', 'seaborn-colorblind'])

# n = 12
# color = plt.cm.viridis(np.linspace(0, 1, n))
# colors = plt.rcParams['axes.prop_cycle'] = cycler('color', color)

N = 2 * 4
plt.style.use(['Solarize_Light2'])
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, N)))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors[0] = colors[1]
colors.reverse()

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

filepath = os.path.dirname(os.path.abspath(__file__)) + '/../bicycle/datastore/highway/'


# ### Define Recording Variables ###
t = np.linspace(dt, tf, int(tf/dt))
# x = np.zeros(((int(tf/dt)+1), nControls, 2))
# sols = np.zeros(((int(tf/dt)+1), nControls, 5))
# # theta_hat = np.zeros(((int(tf/dt)+1),nControl,2))
# cbf_val = np.zeros(((int(tf/dt)+1), nControls, 2))

import builtins
if hasattr(builtins, "VIS_FILE"):
    filename = filepath + builtins.VIS_FILE + '.pkl'
else:
    ftype = r'*.pkl'
    files = glob.glob(filepath + ftype)
    files.sort(key=os.path.getmtime)
    filename = files[-1]
    # filename = filepath + 'cbf_filter_side_by_side.pkl'

with open(filename,'rb') as f:
    try:
        data = pickle.load(f)
        x = data['x']
        sols = data['sols']
        sols_nom = data['sols_nom']
        cbf_val = data['cbf']
        ii = data['ii']
    except:
        traceback.print_exc()

lwidth = 2
dash = [3, 2]
color_idx = np.array(range(0, 2*nAgents)).reshape(nAgents,2)

ii = np.min([int(tf/dt), ii])
# ii = np.min([int(5.418/dt),ii])

def set_edges_black(ax):
    ax.spines['bottom'].set_color('#000000')
    ax.spines['top'].set_color('#000000')
    ax.spines['right'].set_color('#000000')
    ax.spines['left'].set_color('#000000')

plt.close('all')


############################################
### Control Trajectories ###
fig_control = plt.figure(figsize=(8, 8))
ax_cont_a = fig_control.add_subplot(211)
ax_cont_b = fig_control.add_subplot(212)
set_edges_black(ax_cont_a)
set_edges_black(ax_cont_b)

# Angular Control Inputs
ax_cont_a.plot(t[1:ii], 2 * np.pi * np.ones(t[1:ii].shape), linewidth=lwidth+1, color='k')
ax_cont_a.plot(t[1:ii], -2 * np.pi * np.ones(t[1:ii].shape), linewidth=lwidth+1, color='k')
# ax_cont_a.plot(t[1:ii], 2 * np.pi * np.ones(t[1:ii].shape), label=r'$\pm\omega_{max}$', linewidth=lwidth+1, color='k')
# ax_cont_a.plot(t[1:ii], -2 * np.pi * np.ones(t[1:ii].shape), linewidth=lwidth+1, color='k')
for aa in range(nAgents):
    # ax_cont_a.plot(t[:ii], sols[:ii, aa, 0], label=r'$\omega_{}$'.format(aa), linewidth=lwidth,
    #                color=colors[color_idx[aa, 0]])
    # ax_cont_a.plot(t[:ii], sols_nom[:ii, aa, 0], label=r'$\omega^0_{}$'.format(aa), linewidth=lwidth,
    #                color=colors[color_idx[aa, 1]], dashes=dash)
    ax_cont_a.plot(t[:ii], sols[:ii, aa, 0], label='w_{}'.format(aa), linewidth=lwidth,
                   color=colors[color_idx[aa, 0]])
    ax_cont_a.plot(t[:ii], sols_nom[:ii, aa, 0], label='w_{}^0'.format(aa), linewidth=lwidth,
                   color=colors[color_idx[aa, 1]], dashes=dash)
ax_cont_a.set(ylabel='w',#ylabel=r'$\omega$',
              ylim=[np.min(sols[:ii, :, 0]), np.max(sols[:ii, :, 0])],
              title='Control Inputs')

# Acceleration Inputs
# ax_cont_b.plot(t[1:ii], 9.81 * np.ones(t[1:ii].shape), label=r'$\pm a_{max}$', linewidth=lwidth+1, color='k')
# ax_cont_b.plot(t[1:ii], -9.81 * np.ones(t[1:ii].shape), linewidth=lwidth+1, color='k')
ax_cont_b.plot(t[1:ii], 9.81 * np.ones(t[1:ii].shape), linewidth=lwidth+1, color='k')
ax_cont_b.plot(t[1:ii], -9.81 * np.ones(t[1:ii].shape), linewidth=lwidth+1, color='k')
for aa in range(nAgents):
    # ax_cont_b.plot(t[:ii], sols[:ii, aa, 1], label=r'$a_r_{}$'.format(aa), linewidth=lwidth,
    #                color=colors[color_idx[aa, 0]])
    # ax_cont_b.plot(t[:ii], sols_nom[:ii, aa, 1], label=r'$a_r^0_{}$'.format(aa), linewidth=lwidth,
    #                color=colors[color_idx[aa, 1]], dashes=dash)
    ax_cont_b.plot(t[:ii], sols[:ii, aa, 1], label='a_{}'.format(aa), linewidth=lwidth,
                   color=colors[color_idx[aa, 0]])
    ax_cont_b.plot(t[:ii], sols_nom[:ii, aa, 1], label='a_{}^0'.format(aa), linewidth=lwidth,
                   color=colors[color_idx[aa, 1]], dashes=dash)
ax_cont_b.set(ylabel='a',#ylabel=r'$a_r$',
              ylim=[-9.81 - 0.5, 9.81 + 0.5])

# Plot Settings
for item in ([ax_cont_a.title, ax_cont_a.xaxis.label, ax_cont_a.yaxis.label] +
             ax_cont_a.get_xticklabels() + ax_cont_a.get_yticklabels()):
    item.set_fontsize(25)
ax_cont_a.legend(fancybox=True)
ax_cont_a.grid(True, linestyle='dotted', color='white')

for item in ([ax_cont_b.title, ax_cont_b.xaxis.label, ax_cont_b.yaxis.label] +
             ax_cont_b.get_xticklabels() + ax_cont_b.get_yticklabels()):
    item.set_fontsize(25)
ax_cont_b.legend(fancybox=True)
ax_cont_b.grid(True, linestyle='dotted', color='white')

plt.tight_layout(pad=2.0)



############################################
### State Trajectories ###
plt.style.use(['dark_background'])
fig_map = plt.figure(figsize=(13, 2.5))
ax_pos = fig_map.add_subplot(111)
set_edges_black(ax_pos)

# Set Up Road
d_points = 100
start_p = -100.0
end_p = 1000.0
ax_pos.plot(np.linspace(start_p, end_p, d_points), -(LW + LW / 2) * np.ones((d_points,)), linewidth=lwidth+1, color='w')
# ax_pos.plot(np.linspace(-200.0, 200.0, d_points), -LW / 2 * np.ones((d_points,)), linewidth=lwidth+1, color='w', dashes=dash)
# ax_pos.plot(np.linspace(-200.0, 200.0, d_points), LW / 2 * np.ones((d_points,)), linewidth=lwidth+1, color='w', dashes=dash)
ax_pos.plot(np.linspace(start_p, end_p, d_points), LW + LW / 2 * np.ones((d_points,)), linewidth=lwidth+1, color='w')

px = np.tile(np.arange(start_p, end_p, 10), 2)
py = np.repeat(np.array([-LW / 2, LW / 2]), int(len(px)) / 2)

# Add rectangles
width = 3
height = 0.25
for ppx, ppy in zip(px, py):
    ax_pos.add_patch(Rectangle(
        xy=(ppx - width / 2, ppy - height / 2), width=width, height=height,
        linewidth=1, color='white', fill=True))

# plt.show()

# Create variable reference to plot
map_vid = []
for aa in range(2 * nAgents):
    if (aa + 10) % 2 == 0:
        map_vid.append(ax_pos.plot([], [], linewidth=lwidth)[0])
    else:
        map_vid.append(ax_pos.plot([], [], linewidth=lwidth, dashes=dash)[0])

# Add text annotation and create variable reference
txt = ax_pos.text(40.0, 6.2, '', ha='right', va='top', fontsize=24)

ax_pos.set(ylim=[-5.0, 6.0],
           xlim=[-2.0, 50.0])

# Plot Settings
for item in ([ax_pos.title, ax_pos.xaxis.label, ax_pos.yaxis.label] +
             ax_pos.get_xticklabels() + ax_pos.get_yticklabels()):
    item.set_fontsize(25)
# Hide X and Y axes label marks
ax_pos.xaxis.set_tick_params(labelbottom=False)
ax_pos.yaxis.set_tick_params(labelleft=False)
# Hide X and Y axes tick marks
ax_pos.set_xticks([])
ax_pos.set_yticks([])
ax_pos.legend(fancybox=True)
ax_pos.grid(False)


# Animation function
def animate(jj):
    last_1_sec = 20
    right_vehicle = np.argmax(x[jj, :, 0])
    left_edges = np.array([x[idx, np.argmin(x[0:idx+1, :, 0], 1)[idx], 0] for idx in range(jj+1)]) - 2.0 - (jj / 30)
    zero_point = left_edges[-1]
    end_point = np.max([np.max(x[jj, right_vehicle, 0]) + 2.0, zero_point + 50.0])

    for aa in range(0, 2 * nAgents, 2):
        idx = int(aa / 2)
        x_circ, y_circ = get_circle(x[jj, idx], 1.0, d_points)
        x_hist, y_hist = x[np.max([0, jj+1 - last_1_sec]):jj+1, idx, 0:2].T
        map_vid[aa].set_data(x_circ, y_circ)
        # map_vid[aa + 1].set_data(x_hist - left_edges + zero_point, y_hist)
        map_vid[aa + 1].set_data(x_hist, y_hist)
        map_vid[aa].set_color(colors[color_idx[idx, 1]])
        map_vid[aa + 1].set_color(colors[color_idx[idx, 1]])

    ax_pos.set_xlim([zero_point, end_point])
    txt.set_text('{:.1f} sec'.format(jj * dt))
    txt.set_position((zero_point + 40.0, 6.2))

    return map_vid,


# Create animation
ani = animation.FuncAnimation(fig=fig_map, func=animate, frames=int(ii/1.8), interval=10, repeat=False)
writer = animation.writers['ffmpeg']
ani.save(filepath + 'animation.mp4', writer=writer(fps=15))

plt.tight_layout(pad=2.0)
plt.show()
