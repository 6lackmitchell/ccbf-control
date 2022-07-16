import numpy as np
from .quadrotor_dynamics import f, g#, reg_const

def xf_2dot(x,xf,xf_dot,k):
    return (x[:xf.shape[0]] - xf - 2*k*xf_dot) / (k**2)

# def phif_2dot(x,u,phif,phif_dot,k):
def phif_2dot(x,u,phif,phif_dot,k):
    # return (f(x) + np.dot(g(x),u) + reg_const(x,thetaHat) - phif - 2*k*phif_dot) / (k**2)
    return ((f(x) + np.dot(g(x),u))[0:phif.shape[0]] - phif - 2*k*phif_dot) / (k**2)

def Phif_2dot(reg,Phif,Phif_dot,k):
    return (reg - Phif - 2*k*Phif_dot) / (k**2)