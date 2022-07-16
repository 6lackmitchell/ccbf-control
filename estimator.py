import os
import copy
import time
import numpy as np

# from settings import *

import builtins
if hasattr(builtins,"ecc_MODEL_CONFIG"):
    FOLDER = builtins.ecc_MODEL_CONFIG
else:
    FOLDER = 'simple'

if FOLDER == 'overtake':
    from overtake_settings import *
elif FOLDER == 'simple':
    from simple_settings import *
elif FOLDER == 'simple2ndorder':
    from simple2ndorder_settings import *
elif FOLDER == 'quadrotor':
    from quadrotor.settings import *

###############################################################################
##################### FxT Parameter Estimation Parameters #####################
###############################################################################

global ESTIMATOR
ESTIMATOR = None


mu_e      = 5
c1_e      = 50
c2_e      = 50
k_e       = 0.001
l_e       = 100
gamma1_e  = 1 - 1/mu_e
gamma2_e  = 1 + 1/mu_e
T_e       = 1 / (c1_e * (1 - gamma1_e)) + 1 / (c2_e * (gamma2_e - 1))
kG        = 1.2 # Must be greater than 1

mu_e      = 2
c1_e      = 4
c2_e      = 4
k_e       = 0.0002
l_e       = 25.0
gamma1_e  = 1 - 1/mu_e
gamma2_e  = 1 + 1/mu_e
T_e       = 1 / (c1_e * (1 - gamma1_e)) + 1 / (c2_e * (gamma2_e - 1))
kG        = 1.1 # Must be greater than 1
Kw        = 165
# Kw        = 1.0

###############################################################################
################################## Functions ##################################
###############################################################################

def adaptation_law(dt,tt,x,u):
    global ESTIMATOR
    if ESTIMATOR is None:
        settings = {'dt':dt,'x':x,'tt':tt}
        ESTIMATOR = Estimator()
        ESTIMATOR.set_initial_conditions(**settings)

    return ESTIMATOR.update(tt,x,u)

###############################################################################
#################################### Class ####################################
###############################################################################

class Estimator():

    @property
    def T_e(self):
        arg1 = np.sqrt(c2_e) * self.V0_T**(1/mu_e)
        arg2 = np.sqrt(c1_e)
        return mu_e / np.sqrt(c1_e*c2_e) * np.arctan2(arg1,arg2)

    @property
    def e(self):
        """ State Prediction Error """
        # print("x:  {}".format(self.x))
        # print("xh: {}".format(self.xHat))
        return self.x - self.xHat

    def __init__(self):
        self.dt       = None
        self.x        = None
        self.xhat     = None
        self.thetaHat = None
        self.theta    = None
        self.thetaMax = None
        self.thetaMin = None

    def set_initial_conditions(self,**settings):
        """ """
        # Set Default Initial Conditions
        self.dt       = dt
        self.x        = x0
        self.xHat     = x0
        self.thetaHat = thetaHat
        self.thetaMax = thetaMax
        self.thetaMin = thetaMin

        if 'x0' in settings.keys():
            assert type(settings['x0']) == np.ndarray
            self.x    = settings['x0']
            self.xHat = settings['x0']

        if 'x' in settings.keys():
            assert type(settings['x']) == np.ndarray
            self.x    = settings['x']
            self.xHat = settings['x']

        if 'thetaHat0' in settings.keys():
            assert type(settings['thetaHat0']) == np.ndarray
            self.thetaHat = settings['thetaHat0']
            self.theta    = settings['thetaHat0']
            if 'psi_hat0' not in settings.keys() and 'psi_est0' not in settings.keys():
                self.psi_hat = settings['thetaHat0']

        if 'th' in settings.keys():
            assert type(settings['thetaHat0']) == np.ndarray
            self.thetaHat = settings['th']
            self.theta    = settings['th']
            if 'psi_hat0' not in settings.keys() and 'psi_est0' not in settings.keys():
                self.psi_hat = settings['th']

        if 'theta_est0' in settings.keys():
            assert type(settings['thetaHat0']) == np.ndarray
            self.thetaHat = settings['theta_est0']
            self.theta    = settings['theta_est0']
            if 'psi_hat0' not in settings.keys() and 'psi_est0' not in settings.keys():
                self.psi_hat = settings['theta_est0']

        if 'psi_hat0' in settings.keys():
            assert type(settings['psi_hat0']) == np.ndarray
            self.psi_hat = settings['psi_hat0']

        if 'psi_est0' in settings.keys():
            assert type(settings['psi_est0']) == np.ndarray
            self.psi_hat = settings['psi_est0']

        if 'thetaMax' in settings.keys():
            assert type(settings['thetaMax']) == np.ndarray
            self.thetaMax = settings['thetaMax']
            if 'psi_max' not in settings.keys():
                self.psi_max = settings['thetaMax']
            if 'thetaMin' not in settings.keys():
                self.thetaMin = -1*self.thetaMax

        if 'thetaMin' in settings.keys():
            assert type(settings['thetaMin']) == np.ndarray
            self.thetaMin = settings['thetaMin']
            if 'psi_min' not in settings.keys():
                self.psi_min = settings['thetaMin']

        if 'psi_max' in settings.keys():
            assert type(settings['psi_max']) == np.ndarray
            self.psi_max = settings['psi_max']
            if 'thetaMax' not in settings.keys():
                self.thetaMax = settings['psi_max']

        if 'psi_min' in settings.keys():
            assert type(settings['psi_min']) == np.ndarray
            self.psi_min = settings['psi_min']
            if 'thetaMin' not in settings.keys():
                self.thetaMin = settings['psi_min']

        if 'dt' in settings.keys():
            assert type(settings['dt']) == float
            self.dt = settings['dt']

        if 'tt' in settings.keys():
            assert type(settings['tt']) == float or type(settings['tt']) == np.float64
            self.t = settings['tt']

        # This is more conservative than necessary -- could be fcn of initial estimate
        self.errMax    = self.thetaMax - self.thetaMin
        c              = np.linalg.norm(self.errMax)
        self.Gamma     = kG * c**2 / (2 * np.min(cbf(self.x))) * np.eye(self.thetaHat.shape[0])
        self.V0_T      = np.inf
        self.V0        = 1/2 * self.errMax.T @ np.linalg.inv(self.Gamma) @ self.errMax
        self.Vmax      = self.V0
        self.eta       = self.Vmax

        self.W         = np.zeros(regressor(self.x).shape)
        self.xi        = self.e

        # Alternate technique when unknown parameters appear in multiple stages of dynamics
        # reg_depth      = 3
        # reg            = reg_est(self.x,self.thetaHat)[:reg_depth]
        # self.xf        = np.zeros(self.x[:reg_depth].shape)#self.x
        # self.xf_dot    = np.zeros(self.x[:reg_depth].shape)
        # self.phif      = np.zeros(f(self.x)[:reg_depth].shape)
        # self.phif_dot  = np.zeros(f(self.x)[:reg_depth].shape)
        # self.Phif      = np.zeros(reg.shape)
        # self.Phif_dot  = np.zeros(reg.shape)
        # self.P         = np.zeros(np.dot(reg.T,reg).shape)
        # self.Q         = np.zeros(self.thetaHat.shape)

        reg_depth      = 3
        reg            = regressor(self.x)
        self.xf        = np.zeros(self.x.shape)
        self.xf_dot    = np.zeros(self.x.shape)
        self.phif      = np.zeros(f(self.x).shape)
        self.phif_dot  = np.zeros(f(self.x).shape)
        self.Phif      = np.zeros(reg.shape)
        self.Phif_dot  = np.zeros(reg.shape)
        self.P         = np.zeros(np.dot(reg.T,reg).shape)
        self.Q         = np.zeros(self.thetaHat.shape)

        self.update_error_bounds(0)

    def update(self,
               t: float,
               x: np.ndarray,
               u: np.ndarray):
        """ Updates the parameter estimates and the corresponding error bounds. """
        self.t = t
        self.x = x
        self.u = u

        if t == 0:
            return self.thetaHat,self.errMax,self.etaTerms,self.Gamma,self.xHat,self.theta

        # Update Unknown Parameter Estimates
        self.update_unknown_parameter_estimates(law=2)

        # Update state estimate using observer dynamics
        self.update_observer()

        # Update theta_tilde upper/lower bounds
        self.update_error_bounds(self.t)

        return self.thetaHat,self.errMax,self.etaTerms,self.Gamma,self.xHat,self.theta

    def update_unknown_parameter_estimates(self,law=1):
        """
        """
        tol = 0#1e-15
        # self.update_filter_2ndOrder()

        eig = self.update_auxiliaries()
        if eig <= 0:
            # return
            raise ValueError('PE Condition Not Met: Eig(P) = {:.3f}'.format(eig))

        # General Quantities
        # W    = self.P @ self.thetaHat - self.Q
        # Pinv = np.linalg.inv(self.P)


        # New relationship: e = W\Tilde{\theta}
        W      =  self.e
        self.P = -self.W


        if law == 1:
            # Adaptation Law 1
            pre  = self.Gamma @ W / (W.T @ Pinv.T @ W)
            V    = (1/2 * W.T @ Pinv.T @ np.linalg.inv(self.Gamma) @ Pinv @ W)
            self.thetaHatDot = pre * (-c1_e * V**gamma1_e - c2_e * V**gamma2_e)

        elif law == 2:
            if np.linalg.norm(W) < 1e-15:
                return
            # Adaptation Law 2
            pre  = -self.Gamma / (np.linalg.norm(W))
            self.thetaHatDot = pre @ (self.P.T @ W + self.P.T @ W * (W.T @ W))

            norm_chk = np.linalg.norm(W)
            if np.isnan(norm_chk) or np.isinf(norm_chk) or norm_chk == 0:
                print("ThetaHat: {}".format(self.thetaHat))
                raise ValueError("W Norm Zero")

        thd_norm      = np.linalg.norm(self.thetaHatDot)
        if thd_norm >= tol and not (np.isnan(thd_norm) or np.isinf(thd_norm)):
            self.thetaHat = self.thetaHat + (self.dt * self.thetaHatDot)
            self.thetaHat = np.clip(self.thetaHat,self.thetaMin,self.thetaMax)
        else:
            print("P: {}".format(self.P))
            print("Q: {}".format(self.Q))
            print("W: {}".format(W))
            print("No Theta updated")
            print("Pre  = {}\nTime = {}sec".format(pre,self.t))

        # self.theta     = Pinv @ self.Q

    def update_auxiliaries(self):
        """ Updates the auxiliary matrix and vector for the filtering scheme.

        INPUTS:
        None

        OUTPUTS:
        float -- minimum eigenvalue of P matrix

        """
        Wdot   = -Kw * self.W + regressor(self.x)
        # xi_dot = -Kw * self.xi

        self.W  = self.W  + (self.dt * Wdot)
        # self.xi = self.xi + (self.dt * xi_dot)

        # Pdot = -l_e * self.P + np.dot(self.W.T,self.W)
        # Qdot = -l_e * self.Q + np.dot(self.W.T,(self.W@self.thetaHat + self.e - self.xi))

        # Pdot = np.dot(self.W.T,self.W)
        # Qdot = np.dot(self.W.T,(self.W@self.thetaHat + self.e - self.xi))

        # print("W:  {}".format(self.W))
        # print("e:  {}".format(self.e))
        # print("xi: {}".format(self.xi))

        # self.P = self.P + (self.dt * Pdot)
        # self.Q = self.Q + (self.dt * Qdot)

        norm_chk = np.linalg.norm(self.P)
        if np.isnan(norm_chk) or np.isinf(norm_chk):
            raise ValueError("P Norm out of bounds")

        norm_chk = np.linalg.norm(self.Q)
        if np.isnan(norm_chk) or np.isinf(norm_chk):
            raise ValueError("Q Norm out of bounds")

        return 1#np.min(np.linalg.eig(self.P)[0])

    def update_observer(self):
        """ Updates the state estimate according to the observer (xhat) dynamics.

        INPUTS
        ------
        None

        OUTPUTS
        -------
        None

        """
        xHatDot = f(self.x) + g(self.x)@self.u + regressor(self.x)@self.thetaHat + Kw*self.e + self.W@self.thetaHatDot
        # print("ThetaHatDot: {}".format(self.thetaHatDot))
        # print("xHatDot: {}".format(xHatDot))
        self.xHat = self.xHat + (self.dt * xHatDot)

    def update_error_bounds(self,t):
        """
        """
        # Update Max Error Quantities
        arc_tan      = np.arctan2(np.sqrt(c2_e) * self.V0**(1/mu_e),np.sqrt(c1_e))
        tan_arg      = -np.min([t,self.T_e]) * np.sqrt(c1_e * c2_e) / mu_e + arc_tan
        Vmax         = (np.sqrt(c1_e / c2_e) * np.tan(np.max([tan_arg,0]))) ** mu_e
        self.Vmax    = np.clip(Vmax,0,np.inf)

        # Update eta
        self.eta     = self.Vmax

        # Define constants
        M = 2 * np.max(self.Gamma)
        N = np.sqrt(c2_e/c1_e)
        X = np.arctan2(N * self.V0,1)
        c = c1_e
        u = mu_e
        A = N*c/u
        B = np.sqrt(M * 1 / N**u)
        tan_arg = np.min([A*t - X,0.0])

        # print("TanArg: {}".format(tan_arg))
        # print("M:      {}".format(M))
        # print("N:      {}".format(N))
        # print("X:      {}".format(X))
        # print("c:      {}".format(c))
        # print("u:      {}".format(u))


        # Update eta derivatives
        # etadot  = N*c*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)/(2*np.tan(tan_arg))
        # eta2dot = N**2*c**2*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**2/(4*np.tan(tan_arg)**2) - N**2*c**2*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**2/(2*u*np.tan(tan_arg)**2) + N**2*c**2*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)/u
        # eta3dot = N**3*c**3*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**3/(8*np.tan(tan_arg)**3) - 3*N**3*c**3*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**3/(4*u*np.tan(tan_arg)**3) + 3*N**3*c**3*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**2/(2*u*np.tan(tan_arg)) + N**3*c**3*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**3/(u**2*np.tan(tan_arg)**3) - 2*N**3*c**3*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**2/(u**2*np.tan(tan_arg)) + 2*N**3*c**3*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)*np.tan(tan_arg)/u**2
        # eta4dot = N**4*c**4*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**4/(16*np.tan(tan_arg)**4) - 3*N**4*c**4*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**4/(4*u*np.tan(tan_arg)**4) + 3*N**4*c**4*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**3/(2*u*np.tan(tan_arg)**2) + 11*N**4*c**4*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**4/(4*u**2*np.tan(tan_arg)**4) - 7*N**4*c**4*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**3/(u**2*np.tan(tan_arg)**2) + 7*N**4*c**4*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**2/u**2 - 3*N**4*c**4*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**4/(u**3*np.tan(tan_arg)**4) + 8*N**4*c**4*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**3/(u**3*np.tan(tan_arg)**2) - 6*N**4*c**4*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)**2/u**3 + 4*N**4*c**4*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)*np.tan(tan_arg)**2/u**3

        etadot  = -A*B*(np.tan(tan_arg)**2 + 1)
        eta2dot = -2*A**2*B*(np.tan(tan_arg)**2 + 1)*np.tan(tan_arg)
        eta3dot = -2*A**3*B*(np.tan(tan_arg)**2 + 1)**2 - 4*A**3*B*(np.tan(tan_arg)**2 + 1)*np.tan(tan_arg)**2
        eta4dot = -16*A**4*B*(np.tan(tan_arg)**2 + 1)**2*np.tan(tan_arg) - 8*A**4*B*(np.tan(tan_arg)**2 + 1)*np.tan(tan_arg)**3
        # print(N*c*np.sqrt(M*(np.tan(tan_arg)/N)**u)*(np.tan(tan_arg)**2 + 1)/(2*np.tan(tan_arg)))
        # print(N*c*np.sqrt(M*(np.tan(tan_arg)/N)**u))
        # print((np.tan(tan_arg)**2 + 1))
        # print((2*np.tan(tan_arg)))
        # print(M*(np.tan(tan_arg)/N)**u)
        # print(etadot)
        # print(eta2dot)
        # print(eta3dot)
        # print(eta4dot)

        # Update etadot terms
        self.eta_order1 = np.trace(np.linalg.inv(self.Gamma)) * (self.eta * etadot)
        self.eta_order2 = np.trace(np.linalg.inv(self.Gamma)) * (self.eta * eta2dot + etadot**2)
        self.eta_order3 = np.trace(np.linalg.inv(self.Gamma)) * (self.eta * eta3dot + 3 * etadot * eta2dot)
        self.eta_order4 = np.trace(np.linalg.inv(self.Gamma)) * (self.eta * eta4dot + 4 * etadot * eta3dot + 3 * eta2dot**2)

        if self.t < T_e:
            self.etaTerms = np.array([self.eta_order1,self.eta_order2,self.eta_order3,self.eta_order4])
        else:
            self.etaTerms = np.zeros((4,))
        # print(self.etaTerms)

        # # edot_coeff   = -np.sqrt(2 * np.max(self.Gamma) * c1_e **(mu_e/2 + 1) / c2_e **(mu_e/2 - 1))
        # edot_coeff   = -np.sqrt(2 * np.max(self.Gamma) * c1_e **(1 + mu_e/2) / c2_e **(1 - mu_e/2))
        # self.etadot  = edot_coeff * np.tan(tan_arg)**(mu_e/2 - 1) / np.cos(tan_arg)**2

        # # Update eta2dot
        # eddot_coeff  = -edot_coeff * np.sqrt(c2_e / c1_e) * c1_e / mu_e
        # self.eta2dot = eddot_coeff * np.tan(tan_arg)**(mu_e/2 - 2) / np.cos(tan_arg)**2 * \
        #                ((mu_e/2 - 1) / np.cos(tan_arg)**2 + 2 * np.tan(tan_arg)**(2))

        # # Update eta3dot


        # Update max theta_tilde
        self.errMax = np.clip(np.sqrt(2*np.diagonal(self.Gamma)*self.Vmax),0,self.thetaMax-self.thetaMin)

