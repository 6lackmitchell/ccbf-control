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
from matplotlib import rcParams, cycler
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits import mplot3d
import pandas as pd

from bicycle.bicycle_settings import *
from viz.helpers import get_circle, get_ex

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
plt.style.use(['Solarize_Light2'])
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

filepath = os.path.dirname(os.path.abspath(__file__)) + '\\..\\bicycle\\datastore\\intersection\\'


# ### Define Recording Variables ###
t = np.linspace(dt, tf, int(tf/dt))
# x = np.zeros(((int(tf/dt)+1), nControls, 2))
# sols = np.zeros(((int(tf/dt)+1), nControls, 5))
# # theta_hat = np.zeros(((int(tf/dt)+1),nControl,2))
# cbf_val = np.zeros(((int(tf/dt)+1), nControls, 2))

import builtins
if hasattr(builtins,"VIS_FILE"):
    filename = filepath + builtins.VIS_FILE + '.pkl'
else:
    ftype = r'*.pkl'
    files = glob.glob(filepath + ftype)
    files.sort(key=os.path.getmtime)
    filename = files[-1]
    # filename = max(files, key=os.path.getctime)

with open(filename, 'rb') as f:
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
              ylim=[-2 * np.pi - 0.5, 2 * np.pi + 0.5],
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
fig_map = plt.figure(figsize=(8, 8))
ax_pos = fig_map.add_subplot(111)
set_edges_black(ax_pos)

# Set Up Road
dpoints = 100
ax_pos.plot(LW * np.ones((dpoints,)), np.linspace(LW, 25.0, dpoints), linewidth=lwidth+1, color='k')
ax_pos.plot(-LW * np.ones((dpoints,)), np.linspace(LW, 25.0, dpoints), linewidth=lwidth+1, color='k')
ax_pos.plot(LW * np.ones((dpoints,)), np.linspace(-LW, -25.0, dpoints), linewidth=lwidth+1, color='k')
ax_pos.plot(-LW * np.ones((dpoints,)), np.linspace(-LW, -25.0, dpoints), linewidth=lwidth+1, color='k')
ax_pos.plot(np.linspace(LW, 25.0, dpoints), LW * np.ones((dpoints,)), linewidth=lwidth+1, color='k')
ax_pos.plot(np.linspace(LW, 25.0, dpoints), -LW * np.ones((dpoints,)), linewidth=lwidth+1, color='k')
ax_pos.plot(np.linspace(-LW, -25.0, dpoints), LW * np.ones((dpoints,)), linewidth=lwidth+1, color='k')
ax_pos.plot(np.linspace(-LW, -25.0, dpoints), -LW * np.ones((dpoints,)), linewidth=lwidth+1, color='k')
ax_pos.plot(np.zeros((dpoints,)), np.linspace(LW, 25.0, dpoints), linewidth=lwidth+1, color='y', dashes=dash)
ax_pos.plot(np.zeros((dpoints,)), np.linspace(-LW, -25.0, dpoints), linewidth=lwidth+1, color='y', dashes=dash)
ax_pos.plot(np.linspace(LW, 25.0, dpoints), np.zeros((dpoints,)), linewidth=lwidth+1, color='y', dashes=dash)
ax_pos.plot(np.linspace(-LW, -25.0, dpoints), np.zeros((dpoints,)), linewidth=lwidth+1, color='y', dashes=dash)

# Create variable reference to plot
map_vid = []
for aa in range(nAgents):
    map_vid.append(ax_pos.plot([], [], linewidth=lwidth)[0])

# Add text annotation and create variable reference
txt = ax_pos.text(15, 10, '', ha='right', va='top', fontsize=24)

ax_pos.set(ylim=[-20, 20],
           xlim=[-20, 20])

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
ax_pos.grid(True, linestyle='dotted', color='white')

# Animation function
def animate(jj):
    # Plot Ego as X
    aa = 0
    x_ex, y_ex = get_ex(x[jj, aa], 1.0, dpoints)
    map_vid[aa].set_data(x_ex, y_ex)
    map_vid[aa].set_color(colors[color_idx[aa, 1]])

    for aa in range(1, nAgents):
        x_circ, y_circ = get_circle(x[jj, aa], 1.0, dpoints)
        map_vid[aa].set_data(x_circ, y_circ)
        map_vid[aa].set_color(colors[color_idx[aa, 1]])
    txt.set_text('{:.1f} sec'.format(jj * dt))
    return map_vid,


# Create animation
ani = animation.FuncAnimation(fig=fig_map, func=animate, frames=ii, interval=10, repeat=False)
writer = animation.writers['ffmpeg']
ani.save(filename[:-4] + '.mp4', writer=writer(fps=15))

plt.tight_layout(pad=2.0)
plt.show()
