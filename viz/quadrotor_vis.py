import matplotlib
matplotlib.use("Qt5Agg")

# %matplotlib notebook
import project_path
import pickle
import traceback
import numpy as np
# import latex

import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits import mplot3d
import pandas as pd

from quadrotor.settings import tf, dt, k1, k2, arm_length, f_max, tx_max, ty_max, tz_max, G, M, F_GERONO, A_GERONO, thetaMax
# from ecc_controller_test import T_e

# with plt.style.context('seaborn-colorblind'):
#     plt.rcParams["axes.edgecolor"] = "1"
#     plt.rcParams["axes.linewidth"] = "2"

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'figure.autolayout': True})
plt.style.use(['fivethirtyeight','seaborn-colorblind'])
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

filepath = '/home/mblack/Documents/git/baseline/quadrotor/datastore/'

nControl = 1
ecc      = 0
bla      = 1
tay      = 2
lop      = 3
lsm      = 4
zha      = 5
fxt      = 6
t        = np.linspace(dt,tf,int(tf/dt))

### Define Recording Variables ###
t         = np.linspace(dt,tf,int(tf/dt))
x         = np.zeros(((int(tf/dt)+1),nControl,2))
sols      = np.zeros(((int(tf/dt)+1),nControl,5))
theta_hat = np.zeros(((int(tf/dt)+1),nControl,2))
cbf_val   = np.zeros(((int(tf/dt)+1),nControl,2))

import builtins
if hasattr(builtins,"VIS_FILE"):
    filename = filepath + builtins.VIS_FILE + '.pkl'
else:
    filename = filepath + 'money_set/MONEY_DATASET.pkl'

with open(filename,'rb') as f:
    try:
        data      = pickle.load(f)
        x         = data['x']
        # p         = data['p']
        theta     = data['theta']
        sols      = data['sols']
        sols_nom  = data['sols_nom']
        theta_hat = data['thetahat']
        theta_inv = data['thetainv']
        cbf_val   = data['cbf']
        clf_val   = data['clf']
        xf        = data['xf']
        # Vmax      = data['Vmax']
        # psi_hat   = data['psi_hat']
        ii        = data['ii']
    except:
        traceback.print_exc()

lwidth = 2
dash = [3,2]


ii = np.min([int(tf/dt),ii])
# ii = np.min([int(5.418/dt),ii])

def set_edges_black(ax):
    ax.spines['bottom'].set_color('#000000')
    ax.spines['top'].set_color('#000000')
    ax.spines['right'].set_color('#000000')
    ax.spines['left'].set_color('#000000')

plt.close('all')


############################################
### Control Trajectories ###
fig4 = plt.figure(figsize=(8,8))
ax2a  = fig4.add_subplot(411)
ax2b  = fig4.add_subplot(412)
ax2c  = fig4.add_subplot(413)
ax2d  = fig4.add_subplot(414)
set_edges_black(ax2a)
set_edges_black(ax2b)
set_edges_black(ax2c)
set_edges_black(ax2d)

# # Control via Motor Commands
# ax2a.plot(t[1:ii],4*k1*U_MAX[0]*np.ones(t[1:ii].shape),label=r'$F_{max}$',linewidth=lwidth,color='k')
# ax2a.plot(t[1:ii],0.0*np.ones(t[1:ii].shape),linewidth=lwidth,color='k',label=r'$F_{min}$')
# ax2a.plot(t[:ii],k1*(sols[:ii,ecc,0]+sols[:ii,ecc,1]+sols[:ii,ecc,2]+sols[:ii,ecc,3]),label='PRO',linewidth=lwidth,color=colors[ecc])
# ax2a.set(ylabel=r'$F$',ylim=[-0.5,4*k1*U_MAX[0]+0.5])#,title='Control Inputs'),xlim=[-0.1,5.2],
# ax2b.plot(t[1:ii],arm_length*k1*-U_MAX[1]*np.ones(t[1:ii].shape),label=r'$\pm\bar{\tau}_{\phi}$',linewidth=lwidth,color='k')
# ax2b.plot(t[1:ii],arm_length*k1* U_MAX[1]*np.ones(t[1:ii].shape),linewidth=lwidth,color='k')
# ax2b.plot(t[:ii],arm_length*k1*(-sols[:ii,ecc,1]+sols[:ii,ecc,3]),label='PRO',linewidth=lwidth,color=colors[ecc])
# ax2b.set(xlabel='Time (sec)',ylabel=r'$u_{y}$',ylim=[-1.1*arm_length*k1*U_MAX[1],1.1*arm_length*k1* U_MAX[1]])#xlim=[-0.1,5.2],
# ax2c.plot(t[1:ii],arm_length*k1*-U_MAX[0]*np.ones(t[1:ii].shape),label=r'$\pm\bar{\tau}_{\theta}$',linewidth=lwidth,color='k')
# ax2c.plot(t[1:ii],arm_length*k1* U_MAX[0]*np.ones(t[1:ii].shape),linewidth=lwidth,color='k')
# ax2c.plot(t[:ii],arm_length*k1*(-sols[:ii,ecc,2]+sols[:ii,ecc,0]),label='PRO',linewidth=lwidth,color=colors[ecc])
# ax2c.set(xlabel='Time (sec)',ylabel=r'$u_{y}$',ylim=[1.1*arm_length*k1*-U_MAX[0],1.1*arm_length*k1* U_MAX[0]])#xlim=[-0.1,5.2],
# ax2d.plot(t[1:ii],2*k2*-U_MAX[0]*np.ones(t[1:ii].shape),label=r'$\pm\bar{\tau}_{\psi}$',linewidth=lwidth,color='k')
# ax2d.plot(t[1:ii],2*k2* U_MAX[0]*np.ones(t[1:ii].shape),linewidth=lwidth,color='k')
# ax2d.plot(t[:ii],k2*(sols[:ii,ecc,1]+sols[:ii,ecc,3]-sols[:ii,ecc,0]-sols[:ii,ecc,2]),label='PRO',linewidth=lwidth,color=colors[ecc])
# ax2d.set(xlabel='Time (sec)',ylabel=r'$u_{y}$',ylim=[1.1*2*k2*-U_MAX[0],1.1*2*k2*U_MAX[0]])#xlim=[-0.1,5.2],

# Control via Force
ax2a.plot(t[1:ii],f_max*np.ones(t[1:ii].shape),label=r'$F_{max}$',linewidth=lwidth+1,color='k')
ax2a.plot(t[1:ii],0.0*np.ones(t[1:ii].shape),label=r'$F_{min}$',linewidth=lwidth+1,color='k')
ax2a.plot(t[:ii],sols[:ii,0],label='F',linewidth=lwidth,color=colors[ecc])
ax2a.plot(t[:ii],sols_nom[:ii,0],label=r'$F_{nom}$',linewidth=lwidth,color=colors[ecc+1],dashes=dash)
# ax2a.plot(t[:ii],G*M*np.ones(len(t[:ii]),),label='Gravity',linewidth=lwidth,color=colors[3],dashes=dash)
ax2a.set(ylabel=r'$F$',ylim=[-0.5,f_max*1.1],title='Control Inputs')#,xlim=[-0.1,5.2],
ax2b.plot(t[1:ii],-tx_max*np.ones(t[1:ii].shape),label=r'$\pm\bar{\tau}_{\phi}$',linewidth=lwidth+1,color='k')
ax2b.plot(t[1:ii], tx_max*np.ones(t[1:ii].shape),linewidth=lwidth+1,color='k')
# ax2b.plot(t[:ii],arm_length*k1*(x[:ii,ecc,15] - x[:ii,ecc,13]),label='PRO',linewidth=lwidth,color=colors[ecc])
ax2b.plot(t[:ii],sols[:ii,1],label='PRO',linewidth=lwidth,color=colors[ecc])
ax2b.plot(t[:ii],sols_nom[:ii,1],label=r'$PRO_{nom}$',linewidth=lwidth,color=colors[ecc+1],dashes=dash)
ax2b.set(ylabel=r'$\mu$')#xlim=[-0.1,5.2],)
ax2b.set(ylabel=r'$\tau_{\phi}$',ylim=[-1.1*tx_max,1.1*tx_max])#xlim=[-0.1,5.2],
ax2c.plot(t[1:ii],-ty_max*np.ones(t[1:ii].shape),label=r'$\pm\bar{\tau}_{\theta}$',linewidth=lwidth+1,color='k')
ax2c.plot(t[1:ii], ty_max*np.ones(t[1:ii].shape),linewidth=lwidth+1,color='k')
# ax2c.plot(t[:ii],arm_length*k1*(x[:ii,ecc,12] - x[:ii,ecc,14]),label='PRO',linewidth=lwidth,color=colors[ecc])
ax2c.plot(t[:ii],sols[:ii,2],label='PRO',linewidth=lwidth,color=colors[ecc])
ax2c.plot(t[:ii],sols_nom[:ii,2],label=r'$PRO_{nom}$',linewidth=lwidth,color=colors[ecc+1],dashes=dash)
ax2c.set(ylabel=r'$\tau_{\theta}$',ylim=[-1.1*ty_max,1.1*ty_max])#xlim=[-0.1,5.2],
ax2d.plot(t[1:ii],-tz_max*np.ones(t[1:ii].shape),label=r'$\pm\bar{\tau}_{\psi}$',linewidth=lwidth+1,color='k')
ax2d.plot(t[1:ii], tz_max*np.ones(t[1:ii].shape),linewidth=lwidth+1,color='k')
# ax2d.plot(t[:ii],k2*(-x[:ii,ecc,12]+x[:ii,ecc,13]-x[:ii,ecc,14]+x[:ii,ecc,15]),label='PRO',linewidth=lwidth,color=colors[ecc])
ax2d.plot(t[:ii],sols[:ii,3],label='PRO',linewidth=lwidth,color=colors[ecc])
ax2d.plot(t[:ii],sols_nom[:ii,3],label=r'$PRO_{nom}$',linewidth=lwidth,color=colors[ecc+1],dashes=dash)
ax2d.set(xlabel='Time (sec)',ylabel=r'$\tau_{\psi}$',ylim=[-1.1*tz_max,1.1*tz_max])#xlim=[-0.1,5.2],


for item in ([ax2a.title, ax2a.xaxis.label, ax2a.yaxis.label] +
             ax2a.get_xticklabels() + ax2a.get_yticklabels()):
    item.set_fontsize(25)
ax2a.legend(fancybox=True)
ax2a.grid(True,linestyle='dotted',color='white')

for item in ([ax2b.title, ax2b.xaxis.label, ax2b.yaxis.label] +
             ax2b.get_xticklabels() + ax2b.get_yticklabels()):
    item.set_fontsize(25)
ax2b.legend(fancybox=True)
ax2b.grid(True,linestyle='dotted',color='white')

for item in ([ax2c.title, ax2c.xaxis.label, ax2c.yaxis.label] +
             ax2c.get_xticklabels() + ax2c.get_yticklabels()):
    item.set_fontsize(25)
ax2c.legend(fancybox=True)
ax2c.grid(True,linestyle='dotted',color='white')

for item in ([ax2d.title, ax2d.xaxis.label, ax2d.yaxis.label] +
             ax2d.get_xticklabels() + ax2d.get_yticklabels()):
    item.set_fontsize(25)
ax2d.legend(fancybox=True)
ax2d.grid(True,linestyle='dotted',color='white')

plt.tight_layout(pad=2.0)




############################################
### CBF Trajectories ###
fig5 = plt.figure(figsize=(8,8))
ax3  = fig5.add_subplot(111)
set_edges_black(ax3)
ax3.plot(t[1:ii],np.zeros(t[1:ii].shape),label=r'Boundary',linewidth=lwidth,color='k')
ax3.plot(t[1:ii],cbf_val[1:ii,0],label='Altitude',linewidth=lwidth,color=colors[ecc])
# ax3.plot(t[1:ii],cbf_val[1:ii,1],label='Outer',linewidth=lwidth,color=colors[ecc+1])
# ax3.plot(t[1:ii],cbf_val[1:ii,2],label='Inner',linewidth=lwidth,color=colors[ecc+2])
# ax3.plot(t[1:ii],cbf_val[1:ii,3],label='VelOuter',linewidth=lwidth,color=colors[ecc+3])
# ax3.plot(t[1:ii],cbf_val[1:ii,4],label='VelInner',linewidth=lwidth,color=colors[ecc+4])
ax3.plot(t[1:ii],cbf_val[1:ii,1],label='Attitude',linewidth=lwidth,color=colors[ecc+5])
# ax3.plot(t[1:ii],cbf_val[1:ii,ecc,1],label=r'$PRO2$',linewidth=lwidth,color=colors[ecc+1])
# ax3.plot(t[1:ii],cbf_val[1:ii,ecc,2],label=r'$PRO3$',linewidth=lwidth,color=colors[ecc+2])
ax3.set(xlabel='Time (sec)',ylabel='h(x)',title='CBF Trajectory')
ax3.set(xlim=[-0.5,6.5],ylim=[-1,20])
for item in ([ax3.title, ax3.xaxis.label, ax3.yaxis.label] +
             ax3.get_xticklabels() + ax3.get_yticklabels()):
    item.set_fontsize(25)
ax3.legend(fancybox=True,loc=1)
ax3.grid(True,linestyle='dotted',color='white')

plt.tight_layout(pad=2.0)

############################################
### Filter Trajectories ###
fig91 = plt.figure(figsize=(8,8))
ax91   = fig91.add_subplot(111)
set_edges_black(ax91)
ax91.plot(t[:ii],x[:ii,0],label=r'x',linewidth=lwidth,color=colors[ecc])
ax91.plot(t[:ii],xf[:ii,0],label=r'xf',linewidth=lwidth,color=colors[ecc+1])
ax91.plot(t[:ii],x[:ii,1],label=r'y',linewidth=lwidth,color=colors[ecc+2])
ax91.plot(t[:ii],xf[:ii,1],label=r'yf',linewidth=lwidth,color=colors[ecc+3])
ax91.plot(t[:ii],x[:ii,2],label=r'z',linewidth=lwidth,color=colors[ecc+4])
ax91.plot(t[:ii],xf[:ii,2],label=r'zf',linewidth=lwidth,color=colors[ecc+5])
ax91.set(xlabel='Time (sec)',ylabel='State',title='True vs. Observer Trajectories')
ax91.set(xlim=[-0.5,6.5],ylim=[-1,20])
for item in ([ax91.title, ax91.xaxis.label, ax91.yaxis.label] +
             ax91.get_xticklabels() + ax91.get_yticklabels()):
    item.set_fontsize(25)
ax91.legend(fancybox=True,loc=1)
ax91.grid(True,linestyle='dotted',color='white')

plt.tight_layout(pad=2.0)




############################################
### PE, PN ###
fig10 = plt.figure(figsize=(8,8))
ax2  = fig10.add_subplot(111)
set_edges_black(ax2)

# OBS_X1 = CXo
# OBS_Y1 = CYo
# OBS_A1 = PXo
# OBS_B1 = PYo
# OBS_X2 = CXi
# OBS_Y2 = CYi
# OBS_A2 = PXi
# OBS_B2 = PYi

trig_freq     = 2 * np.pi * F_GERONO
xx1           =  A_GERONO * np.sin(trig_freq * t)
yy1           =  A_GERONO * np.sin(trig_freq * t) * np.cos(trig_freq * t)

# xx1   =  np.linspace(0.0-ELLIPSE_AX,0.0+ELLIPSE_AX,1000)
# yy1a  =  np.sqrt(ELLIPSE_BY**2 * (1 - xx1**2/ELLIPSE_AX**2))
# # yy1b  = -np.sqrt((CIRCLE_R**2 - xx1**2))

# xx2    = np.linspace(OBS_X1-OBS_A1,OBS_X1+OBS_A1,1000)
# yy2a   = OBS_Y1 + OBS_B1*np.sqrt(1 - ((xx2 - OBS_X1)/OBS_A1)**2)
# # yy1b   = OBS_Y1 - B*np.sqrt(OBS_R**2 * (1 - ((xx1 - OBS_X1)/OBS_R)**2))

# xx3    = np.linspace(OBS_X2-OBS_A2,OBS_X2+OBS_A2,1000)
# yy3a   = OBS_Y2 + OBS_B2*np.sqrt(1 - ((xx3 - OBS_X2)/OBS_A2)**2)
# # yy2b   = OBS_Y2 - np.sqrt(OBS_R**2 * (1 - ((xx2 - OBS_X2)/OBS_R)**2))

ax2.plot(x[:ii,0],x[:ii,1],label='Actual',linewidth=lwidth+3,color=colors[ecc])
ax2.plot(xx1[:(ii - int(0.4/dt))],yy1[:(ii - int(0.4/dt))],label='Track',linewidth=lwidth,color=colors[1])
# ax2.plot(xx1,yy1a,label='Track',linewidth=lwidth,color=colors[1])
# ax2.plot(xx2,yy2a,label='Outer',linewidth=lwidth,color=colors[4])
# ax2.plot(xx3,yy3a,label='Inner',linewidth=lwidth,color=colors[4])
# ax2.plot(xx1,yy1b,label='PRO',linewidth=lwidth,color=colors[1])
# ax2a.plot(t[:ii],XS(t[:ii])[0],label='SPHERE',linewidth=lwidth,color=colors[ecc+1])
ax2.set(ylabel='Y',xlabel='X',title='Position Trajectories',ylim=[-8,8],xlim=[-8,8])#,title='Control Inputs')
ax2c.set(ylim=[-8,8])#xlim=[-0.1,5.2],
for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
             ax2.get_xticklabels() + ax2.get_yticklabels()):
    item.set_fontsize(25)
ax2.legend(fancybox=True)
ax2.grid(True,linestyle='dotted',color='white')

plt.tight_layout(pad=2.0)


################################
### Parameter Estimates Plot ###
# plt.close('all')
T_fixed = 1.0

if False:

    fig6, ax4 = plt.subplots(3,1,figsize=(8,8))
    ax4[0].spines['bottom'].set_color('#000000')
    ax4[0].spines['top'].set_color('#000000')
    ax4[0].spines['right'].set_color('#000000')
    ax4[0].spines['left'].set_color('#000000')
    ax4[0].plot(T_fixed,theta[0],'gd',label='Fixed-Time',markersize=1)
    ax4[0].plot(t[:ii],-thetaMax[0]*np.ones((ii,)),label=r'$\theta _{1,bounds}$',linewidth=lwidth+4,color='k')
    ax4[0].plot(t[:ii],thetaMax[0]*np.ones((ii,)),linewidth=lwidth+4,color='k')
    ax4[0].plot(t[:ii],theta[0]*np.ones((ii,)),label=r'$\theta _{1,true}$',color='c',linewidth=lwidth,dashes=dash)
    # ax4[0].plot(t[:ii],np.clip(theta_hat[:ii,lsm,0],-10,10),label=r'$\hat\theta _{1,LSM}$',color=colors[lsm],linewidth=lwidth)
    # ax4[0].plot(t[:ii],np.clip(psi_hat[:ii,tay,0,0],-10,10),':',label=r'$\hat\theta _{1,h_1,TAY}$',color=colors[tay],linewidth=lwidth)
    # ax4[0].plot(t[:ii],np.clip(psi_hat[:ii,tay,0,1],-10,10),'-.',label=r'$\hat\theta _{1,h_2,TAY}$',color=colors[tay],linewidth=lwidth)
    ax4[0].plot(t[:ii],np.clip(theta_hat[:ii,0],-thetaMax[0],thetaMax[0]),label=r'$\hat\theta _{1,PRO}$',linewidth=lwidth)
    ax4[0].legend(fancybox=True,markerscale=15)
    ax4[0].set(ylabel=r'$\theta _1$',xlim=[-0.1,ii*dt+0.75],ylim=[-thetaMax[0]-0.5,thetaMax[0]+0.5])
    ax4[0].set_xticklabels([])
    ax4[0].grid(True,linestyle='dotted',color='white')

    if ii*dt > 1.0:
        ax4a_inset = inset_axes(ax4[0],width="100%",height="100%",
                              bbox_to_anchor=(.1, .2, .4, .2),bbox_transform=ax4[0].transAxes, loc=3)
        ax4a_inset.spines['bottom'].set_color('#000000')
        ax4a_inset.spines['top'].set_color('#000000')
        ax4a_inset.spines['right'].set_color('#000000')
        ax4a_inset.spines['left'].set_color('#000000')
        ax4a_inset.plot(t[:ii],theta[0]*np.ones((ii,)),label=r'$\theta _{1,true}$',color='c',linewidth=lwidth,dashes=dash)
        # ax4a_inset.plot(t[:ii],np.clip(theta_hat[:ii,lsm,0],-10,10),label=r'$\theta _{1,LSM}$',color=colors[lsm],linewidth=lwidth)
        # ax4a_inset.plot(t[:ii],np.clip(psi_hat[:ii,tay,0,0],-10,10),':',label=r'$\hat\theta _{1,h_1,TAY}$',color=colors[tay],linewidth=lwidth)
        # ax4a_inset.plot(t[:ii],np.clip(psi_hat[:ii,tay,0,1],-10,10),'-.',label=r'$\hat\theta _{1,h_2,TAY}$',color=colors[tay],linewidth=lwidth)
        ax4a_inset.plot(t[:ii],np.clip(theta_hat[:ii,0],-thetaMax[0],thetaMax[0]),label=r'$\theta _{1,PRO}$',linewidth=lwidth)
        ax4a_inset.plot(T_fixed,theta[0],'gd',label='Fixed-Time',markersize=10)
        ax4a_inset.xaxis.set_major_locator(MaxNLocator(5))
        ax4a_inset.yaxis.set_major_locator(MaxNLocator(2))
        ax4a_inset.set_xlim(0.75,1.25)
        ax4a_inset.set_ylim(theta[0] - 0.1,theta[0] + 0.1)
        # ax4a_inset.xaxis.tick_top()
        for item in ([ax4a_inset.title, ax4a_inset.xaxis.label, ax4a_inset.yaxis.label] +
                     ax4a_inset.get_xticklabels() + ax4a_inset.get_yticklabels()):
            item.set_fontsize(12)
        ax4a_inset.set_xticks(ax4a_inset.get_xticks().tolist())
        # ax4a_inset.set_xticklabels(["    0.75",None,1.0,None,1.25,None])
        ax4a_inset.set_yticks(ax4a_inset.get_yticks().tolist())
        # ax4a_inset.set_yticklabels([theta[0]-0.1,None,theta[0],None,theta[0]+0.1])

        ax4a_inset.set_xticklabels([None,0.75,None,1.0,None,1.25,None])
        # ax4a_inset.set_yticklabels([None,-1.00,-0.95])
        # ax4a_inset.set_yticklabels([None,-1.00,-0.9])
        # ax4a_inset.spines.set_edgecolor('black')
        # ax4a_inset.spines.set_linewidth(1)
        ax4a_inset.grid(True,linestyle='dotted',color='white')

        mark_inset(ax4[0],ax4a_inset,loc1=2,loc2=1,fc="none",ec="0.2",lw=1.5)#,ls="--")
        plt.draw()

    ax4[1].spines['bottom'].set_color('#000000')
    ax4[1].spines['top'].set_color('#000000')
    ax4[1].spines['right'].set_color('#000000')
    ax4[1].spines['left'].set_color('#000000')
    ax4[1].plot(T_fixed,theta[1],'gd',label='Fixed-Time',markersize=1)
    ax4[1].plot(t[:ii],-thetaMax[1]*np.ones((ii,)),label=r'$\theta _{2,bounds}$',linewidth=lwidth+4,color='k')
    ax4[1].plot(t[:ii],thetaMax[1]*np.ones((ii,)),linewidth=lwidth+4,color='k')
    ax4[1].plot(t[:ii],theta[1]*np.ones((ii,)),label=r'$\theta _{2,true}$',linewidth=lwidth,dashes=dash,color='c')
    # ax4[1].plot(t[:ii],np.clip(theta_hat[:ii,lsm,1],-10,10),label=r'$\hat\theta _{2,LSM}$',linewidth=lwidth,color=colors[lsm])
    # ax4[1].plot(t[:ii],np.clip(psi_hat[:ii,tay,1,0],-10,10),':',label=r'$\hat\theta _{2,h_1,TAY}$',color=colors[tay],linewidth=lwidth)
    # ax4[1].plot(t[:ii],np.clip(psi_hat[:ii,tay,1,1],-10,10),'-.',label=r'$\hat\theta _{2,h_2,TAY}$',color=colors[tay],linewidth=lwidth)
    ax4[1].plot(t[:ii],np.clip(theta_hat[:ii,1],-thetaMax[1],thetaMax[1]),label=r'$\hat\theta _{2,PRO}$',linewidth=lwidth,color=colors[ecc])
    ax4[1].legend(fancybox=True,markerscale=15)
    ax4[1].set(ylabel=r'$\theta _2$',xlim=[-0.1,ii*dt+0.75],ylim=[-thetaMax[1]-0.5,thetaMax[1]+0.5])
    ax4[1].set_xticklabels([])
    ax4[1].grid(True,linestyle='dotted',color='white')

    if ii*dt > 1.0:
        ax4b_inset = inset_axes(ax4[1],width="100%",height="100%",
                              bbox_to_anchor=(.1, .5, .4, .2),bbox_transform=ax4[1].transAxes, loc=3)
        ax4b_inset.spines['bottom'].set_color('#000000')
        ax4b_inset.spines['top'].set_color('#000000')
        ax4b_inset.spines['right'].set_color('#000000')
        ax4b_inset.spines['left'].set_color('#000000')
        ax4b_inset.plot(t[:ii],theta[1]*np.ones((ii,)),label=r'$\theta _{2,true}$',color='c',linewidth=lwidth,dashes=dash)
        # ax4b_inset.plot(t[:ii],np.clip(theta_hat[:ii,lsm,1],-10,10),label=r'$\hat\theta _{2,LSM}$',color=colors[lsm],linewidth=lwidth)
        # ax4b_inset.plot(t[:ii],np.clip(psi_hat[:ii,tay,1,0],-10,10),':',label=r'$\hat\theta _{2,TAY,h_1}$',color=colors[tay],linewidth=lwidth)
        # ax4b_inset.plot(t[:ii],np.clip(psi_hat[:ii,tay,1,1],-10,10),'-.',label=r'$\hat\theta _{2,TAY,h_2}$',color=colors[tay],linewidth=lwidth)
        ax4b_inset.plot(t[:ii],np.clip(theta_hat[:ii,1],-thetaMax[1],thetaMax[1]),label=r'$\hat\theta _{2,PRO}$',linewidth=lwidth)
        ax4b_inset.plot(T_fixed,theta[1],'gd',label='Fixed-Time',markersize=10)
        ax4b_inset.set_xlim(0.75,1.25)
        ax4b_inset.set_ylim(theta[1] - 0.1,theta[1] + 0.1)
        for item in ([ax4b_inset.title, ax4b_inset.xaxis.label, ax4b_inset.yaxis.label] +
                     ax4b_inset.get_xticklabels() + ax4b_inset.get_yticklabels()):
            item.set_fontsize(12)
        ax4b_inset.set_xticks(ax4b_inset.get_xticks().tolist())
        ax4b_inset.set_xticklabels([None,0.75,None,1.0,None,1.25,None])
        ax4b_inset.xaxis.tick_top()
        ax4b_inset.set_yticks(ax4b_inset.get_yticks().tolist())
        ax4b_inset.set_yticklabels([theta[1]-0.1,None,theta[1],None,theta[1]+0.1])

        # ax4b_inset.set_yticklabels([None,None,1.00,None,1.10])
        # ax4b_inset.spines.set_edgecolor('black')
        # ax4b_inset.spines.set_linewidth(1)
        ax4b_inset.grid(True,linestyle='dotted',color='white')

        mark_inset(ax4[1],ax4b_inset,loc1=3,loc2=4,fc="none",ec="0.2",lw=1.5)#,ls="--")
        plt.draw()

    ax4[2].spines['bottom'].set_color('#000000')
    ax4[2].spines['top'].set_color('#000000')
    ax4[2].spines['right'].set_color('#000000')
    ax4[2].spines['left'].set_color('#000000')
    ax4[2].plot(T_fixed,theta[2],'gd',label='Fixed-Time',markersize=1)
    ax4[2].plot(t[:ii],-thetaMax[2]*np.ones((ii,)),label=r'$\theta _{3,bounds}$',linewidth=lwidth+4,color='k')
    ax4[2].plot(t[:ii],thetaMax[2]*np.ones((ii,)),linewidth=lwidth+4,color='k')
    ax4[2].plot(t[:ii],theta[2]*np.ones((ii,)),label=r'$\theta _{3,true}$',linewidth=lwidth,color='c',dashes=dash)
    # ax4[1].plot(t[:ii],np.clip(theta_hat[:ii,lsm,1],-10,10),label=r'$\hat\theta _{2,LSM}$',linewidth=lwidth,color=colors[lsm])
    # ax4[1].plot(t[:ii],np.clip(psi_hat[:ii,tay,1,0],-10,10),':',label=r'$\hat\theta _{2,h_1,TAY}$',color=colors[tay],linewidth=lwidth)
    # ax4[1].plot(t[:ii],np.clip(psi_hat[:ii,tay,1,1],-10,10),'-.',label=r'$\hat\theta _{2,h_2,TAY}$',color=colors[tay],linewidth=lwidth)
    ax4[2].plot(t[:ii],np.clip(theta_hat[:ii,2],-thetaMax[2],thetaMax[2]),label=r'$\hat\theta _{3,PRO}$',linewidth=lwidth,color=colors[ecc])
    ax4[2].legend(fancybox=True,markerscale=15)
    ax4[2].set(xlabel='Time (sec)',ylabel=r'$\theta _3$',xlim=[-0.1,ii*dt+0.75],ylim=[-thetaMax[2]-0.5,thetaMax[2]+0.5])
    ax4[2].grid(True,linestyle='dotted',color='white')

    if ii*dt > 1.0:
        ax4c_inset = inset_axes(ax4[2],width="100%",height="100%",
                              bbox_to_anchor=(.2, .2, .4, .2),bbox_transform=ax4[2].transAxes, loc=3)
        ax4c_inset.spines['bottom'].set_color('#000000')
        ax4c_inset.spines['top'].set_color('#000000')
        ax4c_inset.spines['right'].set_color('#000000')
        ax4c_inset.spines['left'].set_color('#000000')
        ax4c_inset.plot(t[:ii],theta[2]*np.ones((ii,)),label=r'$\theta _{2,true}$',color='c',linewidth=lwidth,dashes=dash)
        # ax4c_inset.plot(t[:ii],np.clip(theta_hat[:ii,lsm,1],-10,10),label=r'$\hat\theta _{2,LSM}$',color=colors[lsm],linewidth=lwidth)
        # ax4c_inset.plot(t[:ii],np.clip(psi_hat[:ii,tay,1,0],-10,10),':',label=r'$\hat\theta _{2,TAY,h_1}$',color=colors[tay],linewidth=lwidth)
        # ax4c_inset.plot(t[:ii],np.clip(psi_hat[:ii,tay,1,1],-10,10),'-.',label=r'$\hat\theta _{2,TAY,h_2}$',color=colors[tay],linewidth=lwidth)
        ax4c_inset.plot(t[:ii],np.clip(theta_hat[:ii,2],-thetaMax[2],thetaMax[2]),label=r'$\hat\theta _{2,PRO}$',linewidth=lwidth)
        ax4c_inset.plot(T_fixed,theta[2],'gd',label='Fixed-Time',markersize=10)
        ax4c_inset.set_xlim(0.75,1.25)
        ax4c_inset.set_ylim(theta[2] - 0.1,theta[2] + 0.1)
        for item in ([ax4c_inset.title, ax4c_inset.xaxis.label, ax4c_inset.yaxis.label] +
                     ax4c_inset.get_xticklabels() + ax4c_inset.get_yticklabels()):
            item.set_fontsize(12)
        ax4c_inset.set_xticks(ax4c_inset.get_xticks().tolist())
        ax4c_inset.set_xticklabels([None,0.75,None,1.0,None,1.25,None])
        # ax4c_inset.xaxis.tick_top()
        ax4c_inset.set_yticks(ax4c_inset.get_yticks().tolist())
        ax4c_inset.set_yticklabels([0.90,1.00,1.10])
        # ax4b_inset.spines.set_edgecolor('black')
        # ax4b_inset.spines.set_linewidth(1)
        ax4c_inset.grid(True,linestyle='dotted',color='white')


        mark_inset(ax4[2],ax4c_inset,loc1=2,loc2=1,fc="none",ec="0.2",lw=1.5)#,ls="--")
        plt.draw()
else:

    ################################
    ### Parameter Estimates Plot ###
    # plt.close('all')
    T_fixed = 1.0
    c1 = 'r'; c2 = 'c'; c3 = 'm'
    # Teal, Coral, Khaki
    c1 = '#029386'; c2 = '#FC5A50'; c3 = '#AAA662'
    # Azure, Cyan, Indigo
    c4 = '#069AF3'; c5 = '#00FFFF'; c6 = '#4B0082'

    fig6, ax4 = plt.subplots(1,1,figsize=(8,8))
    ax4.spines['bottom'].set_color('#000000')
    ax4.spines['top'].set_color('#000000')
    ax4.spines['right'].set_color('#000000')
    ax4.spines['left'].set_color('#000000')
    ax4.plot(T_fixed,theta[0],'gd',label='Fixed-Time',markersize=1)
    ax4.plot(T_fixed,theta[0],'gd',markersize=10)
    ax4.plot(T_fixed,theta[1],'gd',markersize=10)
    ax4.plot(T_fixed,theta[2],'gd',markersize=10)
    ax4.plot(t[:ii],-thetaMax[0]*np.ones((ii,)),label=r'$\theta _{bounds}$',linewidth=lwidth+4,color='k')
    ax4.plot(t[:ii],thetaMax[0]*np.ones((ii,)),linewidth=lwidth+4,color='k')
    ax4.plot(t[:ii],theta[0]*np.ones((ii,)),label=r'$\theta _{1,true}$',color=c1,linewidth=lwidth+3,dashes=dash)
    ax4.plot(t[:ii],theta[1]*np.ones((ii,)),label=r'$\theta _{2,true}$',color=c2,linewidth=lwidth+3,dashes=dash)
    ax4.plot(t[:ii],theta[2]*np.ones((ii,)),label=r'$\theta _{3,true}$',color=c3,linewidth=lwidth+3,dashes=dash)
    ax4.plot(t[:ii],np.clip(theta_hat[:ii,0],-thetaMax[0],thetaMax[0]),label=r'$\hat\theta _{1,PRO}$',linewidth=lwidth+1,color=c4)
    ax4.plot(t[:ii],np.clip(theta_hat[:ii,1],-thetaMax[1],thetaMax[1]),label=r'$\hat\theta _{2,PRO}$',linewidth=lwidth+1,color=c5)
    ax4.plot(t[:ii],np.clip(theta_hat[:ii,2],-thetaMax[2],thetaMax[2]),label=r'$\hat\theta _{3,PRO}$',linewidth=lwidth+1,color=c6)
    # ax4.plot(t[:ii],np.clip(theta_inv[:ii,0],-thetaMax[0],thetaMax[0]),':',label=r'$\hat\theta _{1,PRO}$',linewidth=lwidth+1)#,color=c1)
    # ax4.plot(t[:ii],np.clip(theta_inv[:ii,1],-thetaMax[1],thetaMax[1]),':',label=r'$\hat\theta _{2,PRO}$',linewidth=lwidth+1)#,color=c2)
    # ax4.plot(t[:ii],np.clip(theta_inv[:ii,2],-thetaMax[2],thetaMax[2]),':',label=r'$\hat\theta _{3,PRO}$',linewidth=lwidth+1)#,color=c3)


    # ax4[0].plot(t[:ii],np.clip(theta_hat[:ii,lsm,0],-10,10),label=r'$\hat\theta _{1,LSM}$',color=colors[lsm],linewidth=lwidth)
    # ax4[0].plot(t[:ii],np.clip(psi_hat[:ii,tay,0,0],-10,10),':',label=r'$\hat\theta _{1,h_1,TAY}$',color=colors[tay],linewidth=lwidth)
    # ax4[0].plot(t[:ii],np.clip(psi_hat[:ii,tay,0,1],-10,10),'-.',label=r'$\hat\theta _{1,h_2,TAY}$',color=colors[tay],linewidth=lwidth)
    ax4.legend(fancybox=True,markerscale=15,fontsize=25)
    ax4.set(xlabel='Time (sec)',ylabel=r'$\theta$',xlim=[-0.1,ii*dt+4],ylim=[-thetaMax[0]-0.5,thetaMax[0]+0.5])
    for item in ([ax4.title, ax4.xaxis.label, ax4.yaxis.label] +
                     ax4.get_xticklabels() + ax4.get_yticklabels()):
            item.set_fontsize(25)
    # ax4.set_xticklabels([])
    ax4.grid(True,linestyle='dotted',color='white')

    if ii*dt > 1.0:
        ax4a_inset = inset_axes(ax4,width="100%",height="100%",
                              bbox_to_anchor=(.45, .6, .3, .1),bbox_transform=ax4.transAxes, loc=3)
                              # bbox_to_anchor=(.35, .44, .3, .1),bbox_transform=ax4.transAxes, loc=3)
        ax4a_inset.spines['bottom'].set_color('#000000')
        ax4a_inset.spines['top'].set_color('#000000')
        ax4a_inset.spines['right'].set_color('#000000')
        ax4a_inset.spines['left'].set_color('#000000')
        ax4a_inset.plot(t[:ii],theta[0]*np.ones((ii,)),color=c1,linewidth=lwidth+3,dashes=dash)
        ax4a_inset.plot(t[:ii],np.clip(theta_hat[:ii,0],-thetaMax[0],thetaMax[0]),linewidth=lwidth+1,color=c4)
        # ax4a_inset.plot(t[:ii],np.clip(theta_inv[:ii,0],-thetaMax[0],thetaMax[0]),':',linewidth=lwidth-1)#,color=c1)
        ax4a_inset.plot(T_fixed,theta[0],'gd',label='Fixed-Time',markersize=10)
        ax4a_inset.xaxis.set_major_locator(MaxNLocator(5))
        ax4a_inset.yaxis.set_major_locator(MaxNLocator(2))
        ax4a_inset.set_xlim(T_fixed-0.1,T_fixed+0.1)
        ax4a_inset.set_ylim(theta[0] - 0.05,theta[0] + 0.05)
        ax4a_inset.xaxis.tick_top()
        ax4a_inset.yaxis.tick_right()
        for item in ([ax4a_inset.title, ax4a_inset.xaxis.label, ax4a_inset.yaxis.label] +
                     ax4a_inset.get_xticklabels() + ax4a_inset.get_yticklabels()):
            item.set_fontsize(18)
        ax4a_inset.set_xticks(ax4a_inset.get_xticks().tolist())
        # ax4a_inset.set_yticks(ax4a_inset.get_yticks().tolist())
        # ax4a_inset.set_xticklabels([None,0.75,None,1.0,None,1.25,None])
        # ax4a_inset.set_yticklabels([None,np.round(theta[0]-0.05,2),None,None,theta[0],None,None,theta[0]+0.05,None])
        # ax4a_inset.get_yaxis().set_visible(False)
        ax4a_inset.grid(True,linestyle='dotted',color='white')
        mark_inset(ax4,ax4a_inset,loc1=4,loc2=2,fc="none",ec="0.2",lw=1.5)#,ls="--")

        ax4b_inset = inset_axes(ax4,width="100%",height="100%",
                              bbox_to_anchor=(.2, .2, .3, .1),bbox_transform=ax4.transAxes, loc=3)
        ax4b_inset.spines['bottom'].set_color('#000000')
        ax4b_inset.spines['top'].set_color('#000000')
        ax4b_inset.spines['right'].set_color('#000000')
        ax4b_inset.spines['left'].set_color('#000000')
        ax4b_inset.plot(t[:ii],theta[1]*np.ones((ii,)),color=c2,linewidth=lwidth+3,dashes=dash)
        ax4b_inset.plot(t[:ii],np.clip(theta_hat[:ii,1],-thetaMax[1],thetaMax[1]),linewidth=lwidth+1,color=c5)
        # ax4b_inset.plot(t[:ii],np.clip(theta_inv[:ii,1],-thetaMax[1],thetaMax[1]),':',linewidth=lwidth)#,color=c2)
        ax4b_inset.plot(T_fixed,theta[1],'gd',label='Fixed-Time',markersize=10)
        ax4b_inset.xaxis.set_major_locator(MaxNLocator(5))
        ax4b_inset.yaxis.set_major_locator(MaxNLocator(2))
        ax4b_inset.set_xlim(T_fixed-0.1,T_fixed+0.1)
        ax4b_inset.set_ylim(theta[1] - 0.05,theta[1] + 0.05)
        ax4b_inset.yaxis.tick_right()
        for item in ([ax4b_inset.title, ax4b_inset.xaxis.label, ax4b_inset.yaxis.label] +
                     ax4b_inset.get_xticklabels() + ax4b_inset.get_yticklabels()):
            item.set_fontsize(18)
        ax4b_inset.set_xticks(ax4b_inset.get_xticks().tolist())
        # ax4b_inset.set_yticks(ax4b_inset.get_yticks().tolist())
        # ax4b_inset.set_xticklabels([None,0.75,None,1.0,None,1.25,None])
        # ax4b_inset.set_yticklabels([None,np.round(theta[1]-0.05,2),None,None,theta[1],None,None,theta[1]+0.05,None])
        # ax4b_inset.get_yaxis().set_visible(False)
        ax4b_inset.grid(True,linestyle='dotted',color='white')
        mark_inset(ax4,ax4b_inset,loc1=3,loc2=1,fc="none",ec="0.2",lw=1.5)#,ls="--")

        ax4c_inset = inset_axes(ax4,width="100%",height="100%",
                              bbox_to_anchor=(.1, .75, .3, .1),bbox_transform=ax4.transAxes, loc=3)
        ax4c_inset.spines['bottom'].set_color('#000000')
        ax4c_inset.spines['top'].set_color('#000000')
        ax4c_inset.spines['right'].set_color('#000000')
        ax4c_inset.spines['left'].set_color('#000000')
        ax4c_inset.plot(t[:ii],theta[2]*np.ones((ii,)),color=c3,linewidth=lwidth+3,dashes=dash)
        ax4c_inset.plot(t[:ii],np.clip(theta_hat[:ii,2],-thetaMax[2],thetaMax[2]),linewidth=lwidth+1,color=c6)
        # ax4c_inset.plot(t[:ii],np.clip(theta_inv[:ii,2],-thetaMax[2],thetaMax[2]),':',linewidth=lwidth)#,color=c3)
        ax4c_inset.plot(T_fixed,theta[2],'gd',label='Fixed-Time',markersize=10)
        ax4c_inset.xaxis.set_major_locator(MaxNLocator(5))
        ax4c_inset.yaxis.set_major_locator(MaxNLocator(2))
        ax4c_inset.set_xlim(T_fixed-0.1,T_fixed+0.1)
        ax4c_inset.set_ylim(theta[2] - 0.05,theta[2] + 0.05)
        ax4c_inset.xaxis.tick_top()
        ax4c_inset.yaxis.tick_right()
        for item in ([ax4c_inset.title, ax4c_inset.xaxis.label, ax4c_inset.yaxis.label] +
                     ax4c_inset.get_xticklabels() + ax4c_inset.get_yticklabels()):
            item.set_fontsize(18)
        ax4c_inset.set_xticks(ax4c_inset.get_xticks().tolist())
        # ax4c_inset.set_yticks(ax4c_inset.get_yticks().tolist())
        # ax4c_inset.set_xticklabels([None,0.75,None,1.0,None,1.25,None])
        # ax4c_inset.set_yticklabels([None,np.round(theta[2]-0.05,2),None,None,theta[2],None,None,theta[2]+0.05,None])
        # ax4c_inset.get_yaxis().set_visible(False)
        ax4c_inset.grid(True,linestyle='dotted',color='white')
        mark_inset(ax4,ax4c_inset,loc1=2,loc2=4,fc="none",ec="0.2",lw=1.5)#,ls="--")

        plt.draw()

plt.show()

# plt.tight_layout(pad=1.0)

# fig6.savefig(filepath+"ShootTheGap_ThetaHats_RegX.eps",bbox_inches='tight',dpi=300)
# fig6.savefig(filepath+"ShootTheGap_ThetaHats_RegX.png",bbox_inches='tight',dpi=300)



######################################################################################################
######################################################################################################
########################################### Retired Plots ############################################
######################################################################################################
######################################################################################################

# #################################################
# ### U,V,W Plots ###
# fig11 = plt.figure(figsize=(8,8))
# ax2a  = fig11.add_subplot(311)
# set_edges_black(ax2a)
# ax2a.plot(t[:ii],x[:ii,3],label='PRO',linewidth=lwidth,color=colors[ecc])
# ax2a.set(ylabel='u',title='Body-Fixed Velocities')
# for item in ([ax2a.title, ax2a.xaxis.label, ax2a.yaxis.label] +
#              ax2a.get_xticklabels() + ax2a.get_yticklabels()):
#     item.set_fontsize(25)
# ax2a.legend(fancybox=True)
# ax2a.grid(True,linestyle='dotted',color='white')

# ax2b  = fig11.add_subplot(312)
# set_edges_black(ax2b)
# ax2b.plot(t[:ii],x[:ii,4],label='PRO',linewidth=lwidth,color=colors[ecc])
# ax2b.set(ylabel='v')
# for item in ([ax2b.title, ax2b.xaxis.label, ax2b.yaxis.label] +
#              ax2b.get_xticklabels() + ax2b.get_yticklabels()):
#     item.set_fontsize(25)
# ax2b.legend(fancybox=True)
# ax2b.grid(True,linestyle='dotted',color='white')

# ax2c  = fig11.add_subplot(313)
# set_edges_black(ax2c)
# ax2c.plot(t[:ii],x[:ii,5],label='PRO',linewidth=lwidth,color=colors[ecc])
# ax2c.set(xlabel='Time (sec)',ylabel='w')
# for item in ([ax2c.title, ax2c.xaxis.label, ax2c.yaxis.label] +
#              ax2c.get_xticklabels() + ax2c.get_yticklabels()):
#     item.set_fontsize(25)
# ax2c.legend(fancybox=True)
# ax2c.grid(True,linestyle='dotted',color='white')

# plt.tight_layout(pad=2.0)


# ############################################
# ### phi, theta, psi Trajectories ###
# fig100 = plt.figure(figsize=(8,8))
# ax2a  = fig100.add_subplot(311)
# set_edges_black(ax2a)
# ax2a.plot(t[:ii],x[:ii,6],label='PRO',linewidth=lwidth,color=colors[ecc])
# ax2a.set(ylabel=r'$\phi$')#,title='Control Inputs')
# for item in ([ax2a.title, ax2a.xaxis.label, ax2a.yaxis.label] +
#              ax2a.get_xticklabels() + ax2a.get_yticklabels()):
#     item.set_fontsize(25)
# ax2a.legend(fancybox=True)
# ax2a.grid(True,linestyle='dotted',color='white')

# ax2b  = fig100.add_subplot(312)
# set_edges_black(ax2b)
# ax2b.plot(t[:ii],x[:ii,7],label='PRO',linewidth=lwidth,color=colors[ecc])
# ax2b.set(xlabel='Time (sec)',ylabel=r'$\theta$')
# for item in ([ax2b.title, ax2b.xaxis.label, ax2b.yaxis.label] +
#              ax2b.get_xticklabels() + ax2b.get_yticklabels()):
#     item.set_fontsize(25)
# ax2b.legend(fancybox=True)
# ax2b.grid(True,linestyle='dotted',color='white')

# ax2c  = fig100.add_subplot(313)
# set_edges_black(ax2c)
# ax2c.plot(t[:ii],x[:ii,8],label='PRO',linewidth=lwidth,color=colors[ecc])
# ax2c.set(xlabel='Time (sec)',ylabel=r'$\psi$')
# for item in ([ax2c.title, ax2c.xaxis.label, ax2c.yaxis.label] +
#              ax2c.get_xticklabels() + ax2c.get_yticklabels()):
#     item.set_fontsize(25)
# ax2c.legend(fancybox=True)
# ax2c.grid(True,linestyle='dotted',color='white')

# plt.tight_layout(pad=2.0)




# ############################################
# ### PE, PN, H Trajectories ###
# fig10 = plt.figure(figsize=(8,8))
# ax2a  = fig10.add_subplot(311)
# set_edges_black(ax2a)
# ax2a.plot(t[:ii],x[:ii,0],label='PRO',linewidth=lwidth,color=colors[ecc])
# # ax2a.plot(t[:ii],XS(t[:ii])[0],label='SPHERE',linewidth=lwidth,color=colors[ecc+1])
# ax2a.set(ylabel='X',title='Position Trajectories')#,title='Control Inputs')
# for item in ([ax2a.title, ax2a.xaxis.label, ax2a.yaxis.label] +
#              ax2a.get_xticklabels() + ax2a.get_yticklabels()):
#     item.set_fontsize(25)
# ax2a.legend(fancybox=True)
# ax2a.grid(True,linestyle='dotted',color='white')

# ax2b  = fig10.add_subplot(312)
# set_edges_black(ax2b)
# ax2b.plot(t[:ii],x[:ii,1],label='PRO',linewidth=lwidth,color=colors[ecc])
# # ax2b.plot(t[:ii],XS(t[:ii])[1],label='SPHERE',linewidth=lwidth,color=colors[ecc+1])
# ax2b.set(ylabel='Y')
# for item in ([ax2b.title, ax2b.xaxis.label, ax2b.yaxis.label] +
#              ax2b.get_xticklabels() + ax2b.get_yticklabels()):
#     item.set_fontsize(25)
# ax2b.legend(fancybox=True)
# ax2b.grid(True,linestyle='dotted',color='white')

# ax2c  = fig10.add_subplot(313)
# set_edges_black(ax2c)
# ax2c.plot(t[:ii],x[:ii,2],label='PRO',linewidth=lwidth,color=colors[ecc])
# # ax2c.plot(t[:ii],XS(t[:ii])[2],label='SPHERE',linewidth=lwidth,color=colors[ecc+1])
# ax2c.set(xlabel='Time (sec)',ylabel='Z')
# for item in ([ax2c.title, ax2c.xaxis.label, ax2c.yaxis.label] +
#              ax2c.get_xticklabels() + ax2c.get_yticklabels()):
#     item.set_fontsize(25)
# ax2c.legend(fancybox=True)
# ax2c.grid(True,linestyle='dotted',color='white')

# plt.tight_layout(pad=2.0)


#####################
### CLF Evolution ###
# fig1 = plt.figure(figsize=(7,7))

# ax1  = fig1.add_subplot(111)
# ax1.spines['bottom'].set_color('#000000')
# ax1.spines['top'].set_color('#000000')
# ax1.spines['right'].set_color('#000000')
# ax1.spines['left'].set_color('#000000')

# ax1.plot(t[1:ii],clf_val[1:ii,ecc],dashes=dash,label='PRO',linewidth=lwidth,color=colors[ecc])
# ax1.legend(fancybox=True)
# ax1.set(xlabel='Time (sec)',ylabel='V(x)')
# ax1.grid(True,linestyle='dotted',color='white')

# plt.tight_layout(pad=0.5)
# # plt.show()

#######################
### Delta Evolution ###
# fig2 = plt.figure(figsize=(7,7))

# ax1  = fig2.add_subplot(111)
# ax1.spines['bottom'].set_color('#000000')
# ax1.spines['top'].set_color('#000000')
# ax1.spines['right'].set_color('#000000')
# ax1.spines['left'].set_color('#000000')

# ax1.plot(t[1:ii],sols[1:ii,ecc,5],label='d1',linewidth=lwidth,color=colors[ecc+1])
# ax1.plot(t[1:ii],sols[1:ii,ecc,6],label='d2',linewidth=lwidth,color=colors[ecc+2])
# ax1.plot(t[1:ii],sols[1:ii,ecc,4],label='d0',linewidth=lwidth,color=colors[ecc])
# ax1.legend(fancybox=True)
# ax1.set(xlabel='Time (sec)',ylabel='Deltas')
# ax1.grid(True,linestyle='dotted',color='white')

# plt.tight_layout(pad=0.5)
# # plt.show()

############################################
### State, Control, and CBF Trajectories ###
# plt.close('all')
# fig3 = plt.figure(figsize=(8,8))
# # grid = plt.GridSpec(2,3,hspace=0.2,wspace=0.2)

# ax1  = fig3.add_subplot(111)
# ax1.spines['bottom'].set_color('#000000')
# ax1.spines['top'].set_color('#000000')
# ax1.spines['right'].set_color('#000000')
# ax1.spines['left'].set_color('#000000')


# xx1   = np.linspace(OBS_X1-OBS_R,OBS_X1+OBS_R,1000)
# yy1a  = OBS_Y1 + np.sqrt(OBS_R**2 * (1 - ((xx1 - OBS_X1)/OBS_R)**2))
# yy1b  = OBS_Y1 - np.sqrt(OBS_R**2 * (1 - ((xx1 - OBS_X1)/OBS_R)**2))

# xx2   = np.linspace(OBS_X2-OBS_R,OBS_X2+OBS_R,1000)
# yy2a  = OBS_Y2 + np.sqrt(OBS_R**2 * (1 - ((xx2 - OBS_X2)/OBS_R)**2))
# yy2b  = OBS_Y2 - np.sqrt(OBS_R**2 * (1 - ((xx2 - OBS_X2)/OBS_R)**2))

# ax1.plot(xx1,yy1a,color='k',linewidth=lwidth+2)
# ax1.plot(xx1,yy1b,color='k',linewidth=lwidth+2)
# ax1.plot(xx1,yy2a,color='k',linewidth=lwidth+2)
# ax1.plot(xx1,yy2b,color='k',linewidth=lwidth+2,label='Barrier')
# ax1.plot(0,0,'o',markersize=20,color='r')
# ax1.plot(0,0,'o',markersize=10,color='w')
# ax1.plot(0,0,'o',markersize=5,color='r',label='Goal')
# ax1.plot(5,0,'*',markersize=20,color='b',label=r'$z_0$')
# # ax1.plot(x[:ii,tay,0],x[:ii,tay,1],label='TAY',linewidth=lwidth,color=colors[tay])
# # ax1.plot(x[:ii,bla,0],x[:ii,bla,1],label='BLA',linewidth=lwidth,color=colors[bla])
# # ax1.plot(x[:ii,lop,0],x[:ii,lop,1],label='LOP',linewidth=lwidth,color=colors[lop])
# # ax1.plot(x[:ii,lsm,0],x[:ii,lsm,1],label='LSM',linewidth=lwidth,color=colors[lsm])
# # ax1.plot(x[:ii,zha,0],x[:ii,zha,1],label='ZHA',linewidth=lwidth,color=colors[zha])
# ax1.plot(x[:ii,ecc,0],x[:ii,ecc,1],label='PRO',linewidth=3,color=colors[ecc])
# for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
#              ax1.get_xticklabels() + ax1.get_yticklabels()):
#     item.set_fontsize(25)
# ax1.legend(fancybox=True)
# ax1.set(xlabel=r'$x$')#,title='State Trajectories')
# ax1.set_ylabel(r'$y$',rotation=90)
# ax1.set_ylim([-2.5,2.5])
# ax1.grid(True,linestyle='dotted',color='white')

# ax1inset = inset_axes(ax1,width="100%",height="100%",
#                       bbox_to_anchor=(.45, .05, .5, .35),bbox_transform=ax1.transAxes, loc=3)
# ax1inset.spines['bottom'].set_color('#000000')
# ax1inset.spines['top'].set_color('#000000')
# ax1inset.spines['right'].set_color('#000000')
# ax1inset.spines['left'].set_color('#000000')
# ax1inset.plot(xx1,yy1a,color='k',linewidth=5)
# ax1inset.plot(xx1,yy1b,color='k',linewidth=5)
# ax1inset.plot(xx1,yy2a,color='k',linewidth=5)
# ax1inset.plot(xx1,yy2b,color='k',linewidth=5)
# # ax1inset.plot(x[:ii,tay,0],x[:ii,tay,1],label='TAY',linewidth=lwidth,color=colors[tay])
# # ax1inset.plot(x[:ii,bla,0],x[:ii,bla,1],label='BLA',linewidth=lwidth,color=colors[bla])
# # ax1inset.plot(x[:ii,lop,0],x[:ii,lop,1],label='LOP',linewidth=lwidth,color=colors[lop])
# # ax1inset.plot(x[:ii,lsm,0],x[:ii,lsm,1],label='LSM',linewidth=lwidth,color=colors[lsm])
# # ax1inset.plot(x[:ii,zha,0],x[:ii,zha,1],label='ZHA',linewidth=lwidth,color=colors[zha])
# ax1inset.plot(x[:ii,ecc,0],x[:ii,ecc,1],dashes=dash,label='PRO',linewidth=3,color=colors[ecc])
# ax1inset.set_xlim(0.75,1.25)
# ax1inset.set_ylim(-1.05,-0.95)
# ax1inset.set_xticklabels([])
# ax1inset.set_yticklabels([])
# # ax3inset.spines.set_edgecolor('black')
# # ax3inset.spines.set_linewidth(1)
# ax1inset.grid(True,linestyle='dotted',color='white')
# mark_inset(ax1,ax1inset,loc1=2,loc2=3,fc="none",ec="0.2",lw=1.5)#,ls="--")

# plt.tight_layout(pad=2.0)














# #######################
# ### Delta Evolution ###
# fig2 = plt.figure(figsize=(7,7))

# ax1  = fig2.add_subplot(111)
# set_edges_black(ax1)


# ax1.plot(t[1:ii],sols[1:ii,ecc,5],label='p1',linewidth=lwidth,color=colors[ecc])
# # ax1.plot(t[1:ii],sols[1:ii,ecc,6],label='p2',linewidth=lwidth,color=colors[ecc+1])
# # ax1.plot(t[1:ii],sols[1:ii,ecc,7],label='p3',linewidth=lwidth,color=colors[ecc+2])
# ax1.legend(fancybox=True)
# ax1.set(xlabel='Time (sec)',ylabel='Deltas')
# ax1.grid(True,linestyle='dotted',color='white')

# plt.tight_layout(pad=0.5)
# plt.show()




# fig5 = plt.figure(figsize=(8,8))
# ax3  = fig5.add_subplot(111)
# set_edges_black(ax3)
# ax3.plot(t[1:ii],np.min(cbf_val[1:ii,ecc],axis=-1),dashes=dash,label=r'$PRO$',linewidth=lwidth,color=colors[ecc])
# ax3.set(xlabel='Time (sec)',ylabel='h(x)')#,title='CBFs')
# ax3.set(xlim=[-0.5,6.5],ylim=[-1,20])
# for item in ([ax3.title, ax3.xaxis.label, ax3.yaxis.label] +
#              ax3.get_xticklabels() + ax3.get_yticklabels()):
#     item.set_fontsize(25)
# ax3.legend(fancybox=True,loc=1)
# ax3.grid(True,linestyle='dotted',color='white')

# ax3inset = inset_axes(ax3,width="100%",height="100%",
#                       bbox_to_anchor=(.45, .1, .5, .35),bbox_transform=ax3.transAxes, loc=3)
# ax3inset.spines['bottom'].set_color('#000000')
# ax3inset.spines['top'].set_color('#000000')
# ax3inset.spines['right'].set_color('#000000')
# ax3inset.spines['left'].set_color('#000000')
# ax3inset.plot(t[1:ii],np.zeros(t[1:ii].shape),label=r'Boundary',linewidth=lwidth,color='k')
# # ax3inset.plot(t[1:ii],np.min(cbf_val[1:ii,tay],axis=-1),label=r'$TAY$',linewidth=3,color=colors[tay])
# # ax3inset.plot(t[1:ii],np.min(cbf_val[1:ii,bla],axis=-1),label=r'$BLA$',linewidth=3,color=colors[bla])
# # ax3inset.plot(t[1:ii],np.min(cbf_val[1:ii,lop],axis=-1),label=r'$LOP$',linewidth=3,color=colors[lop])
# # ax3inset.plot(t[1:ii],np.min(cbf_val[1:ii,lsm],axis=-1),label=r'$LSM$',linewidth=3,color=colors[lsm])
# # ax3inset.plot(t[1:ii],np.min(cbf_val[1:ii,zha],axis=-1),label=r'$ZHA$',linewidth=3,color=colors[zha])
# ax3inset.plot(t[1:ii],np.min(cbf_val[1:ii,ecc],axis=-1),dashes=dash,label=r'$PRO$',linewidth=3,color=colors[ecc])
# ax3inset.set_xlim(0.75,4.1)
# ax3inset.set_ylim(-0.01,0.18)
# ax3inset.set_xticklabels([])
# ax3inset.set_yticklabels([0.00,0.00,0.05,0.10])
# # ax3inset.spines.set_edgecolor('black')
# # ax3inset.spines.set_linewidth(1)
# ax3inset.grid(True,linestyle='dotted',color='white')
# mark_inset(ax3,ax3inset,loc1=2,loc2=4,fc="none",ec="0.2",lw=1.5)#,ls="--")
# plt.draw()

# plt.tight_layout(pad=2.0)

# fig3.savefig(filepath+"ShootTheGap_allFxTS_Trajectories_RegX.eps",bbox_inches='tight',dpi=300,pad_inches=0.5)
# fig1.savefig(filepath+"ShootTheGap_allFxTS_Trajectories_RegX.png",bbox_inches='tight',dpi=300,pad_inches=0.5)
# fig4.savefig(filepath+"ShootTheGap_allFxTS_Controls_RegX.eps",bbox_inches='tight',dpi=300,pad_inches=0.5)
# fig2.savefig(filepath+"ShootTheGap_allFxTS_Controls_RegX.png",bbox_inches='tight',dpi=300,pad_inches=0.5)
# fig5.savefig(filepath+"ShootTheGap_allFxTS_CBFs_RegX.eps",bbox_inches='tight',dpi=300,pad_inches=0.5)
# fig3.savefig(filepath+"ShootTheGap_allFxTS_CBFs_RegX.png",bbox_inches='tight',dpi=300,pad_inches=0.5)


# #################################################
# ### XYZ in Sphere Plots ###
# xx1   = np.linspace(-1,1,1000)
# yy1a  = np.sqrt(1 - xx1**2)
# yy1b  = -np.sqrt(1 - xx1**2)

# fig101 = plt.figure(figsize=(8,8))
# ax2a  = fig101.add_subplot(311)
# set_edges_black(ax2a)
# ax2a.plot(x[:ii,ecc,0] - XS(t[:ii])[0],x[:ii,ecc,1] - XS(t[:ii])[1],label='XY',linewidth=lwidth,color=colors[ecc])
# ax2a.plot(xx1,yy1a,linewidth=lwidth,color='k')
# ax2a.plot(xx1,yy1b,linewidth=lwidth,color='k')
# ax2a.set(xlabel='X',ylabel='Y')#,title='Control Inputs')
# for item in ([ax2a.title, ax2a.xaxis.label, ax2a.yaxis.label] +
#              ax2a.get_xticklabels() + ax2a.get_yticklabels()):
#     item.set_fontsize(25)
# ax2a.legend(fancybox=True)
# ax2a.grid(True,linestyle='dotted',color='white')

# ax2b  = fig101.add_subplot(312)
# set_edges_black(ax2b)
# ax2b.plot(x[:ii,ecc,2] - XS(t[:ii])[2],x[:ii,ecc,0] - XS(t[:ii])[0],label='ZX',linewidth=lwidth,color=colors[ecc])
# ax2b.plot(xx1,yy1a,linewidth=lwidth,color='k')
# ax2b.plot(xx1,yy1b,linewidth=lwidth,color='k')
# ax2b.set(xlabel='Z',ylabel='X')
# for item in ([ax2b.title, ax2b.xaxis.label, ax2b.yaxis.label] +
#              ax2b.get_xticklabels() + ax2b.get_yticklabels()):
#     item.set_fontsize(25)
# ax2b.legend(fancybox=True)
# ax2b.grid(True,linestyle='dotted',color='white')

# ax2c  = fig101.add_subplot(313)
# set_edges_black(ax2c)
# ax2c.plot(x[:ii,ecc,1] - XS(t[:ii])[1],x[:ii,ecc,2] - XS(t[:ii])[2],label='YZ',linewidth=lwidth,color=colors[ecc])
# ax2c.plot(xx1,yy1a,linewidth=lwidth,color='k')
# ax2c.plot(xx1,yy1b,linewidth=lwidth,color='k')
# ax2c.set(xlabel='Y',ylabel='Z')
# for item in ([ax2c.title, ax2c.xaxis.label, ax2c.yaxis.label] +
#              ax2c.get_xticklabels() + ax2c.get_yticklabels()):
#     item.set_fontsize(25)
# ax2c.legend(fancybox=True)
# ax2c.grid(True,linestyle='dotted',color='white')

# plt.tight_layout(pad=2.0)
