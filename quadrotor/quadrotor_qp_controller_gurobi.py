import copy
# import gurobipy as gp

# from gurobipy import GRB
from typing import Tuple, List

from quadrotor.settings import *

###############################################################################
################################## Constants ##################################
###############################################################################
global F_MODEL, F_DVARS, TAU_MODEL, TAU_DVARS, TT0
F_MODEL   = None
TAU_MODEL = None
F_DVARS   = None
TAU_DVARS = None
TT0       = None

###############################################################################
################################## Functions ##################################
###############################################################################

def solve_qp(data):
    """ Solves the cascaded set of QPs in order to compute control input.

    INPUTS
    ------
    data: (dict) - contains time (t), state (x), unknown parameter estimate
                   vector (th), maximum allowable parameter estimation error
                   vector (err), and nth order eta terms for CBF derivative

    OUTPUTS
    -------
    sol: (np.ndarray) - solution to cascaded quadratic program

    """
    global F_MODEL, TAU_MODEL

    # Compute nominal control inputs (F, tau1, tau2, tau3)
    F_nom,tau_nom = compute_nominal_control(data)

    # Update Force Model with nominal input
    F_MODEL,feedback = update_model(F_MODEL,'F',F_nom,data)

    # Compute safe F control input
    try:
        F = compute_safe_force()
        data['F_sol'] = F
    except Exception as e:
        print(feedback)
        # raise e

    # Update Torque Model with nominal input
    TAU_MODEL,feedback = update_model(TAU_MODEL,'tau',tau_nom,data)

    # Compute safe tau1, tau2, tau3 control inputs
    try:
        tau1,tau2,tau3 = compute_safe_torques()
    except Exception as e:
        print(feedback)
        # raise e

    return np.array([F,tau1,tau2,tau3]),np.array([F_nom,tau_nom[0],tau_nom[1],tau_nom[2]])

def update_model(model:   gp.Model,
                 version: str,
                 u_nom:   float or np.ndarray,
                 data:    dict) -> gp.Model:
    """ Updates the specified QP model with the computed nominal control input.

    INPUTS
    ------
    model:   (gp.model) - QP model
    version: (str) - 'F' or 'tau' specifying QP
    u_nom:   (float) - nominal force control input
    data:    (dict) - data for updating QP constraints

    OUTPUTS
    -------
    model: (gp.model) - updated model
    """
    global F_DVARS, TAU_DVARS
    if model is None:
        # Create the model
        model = gp.Model("qp")
        model.setParam('OutputFlag',0)

        # Determine whether force or torque controller in use
        if version == 'F' or version == 'force' or version == 'Force':
            version = 'F'
            d_vars = F_DECISION_VARS
            assert type(u_nom) == float or type(u_nom) == np.float64
        elif version == 'tau' or version == 'Tau' or version == 'torque':
            version = 'tau'
            d_vars = TAU_DECISION_VARS
            assert type(u_nom) == np.ndarray
        else:
            raise ValueError("Nominal Control Input incorrect.")

        # Create the decision variables
        decision_variables = np.zeros((len(d_vars),),dtype='object')
        decision_vars_copy = copy.deepcopy(d_vars)
        for i,v in enumerate(decision_vars_copy):
            lb   = v['lb']
            ub   = v['ub']
            name = v['name']

            decision_variables[i] = model.addVar(lb=lb,ub=ub,name=name)

        if version == 'F':
            F_DVARS = decision_variables
        elif version == 'tau':
            TAU_DVARS = decision_variables

    # Modify objective function according to new nominal control input
    if version == 'F':
        model.setObjective(force_objective(F_DVARS,u_nom),GRB.MINIMIZE)
        model,feedback,e = update_force_constraints(model,data,F_DVARS)
    elif version == 'tau':
        model.setObjective(torque_objective(TAU_DVARS,u_nom),GRB.MINIMIZE)
        model,feedback,e = update_torque_constraints(model,data,TAU_DVARS)
        if e is not None:
            print(feedback)
            # raise e
    model.update()

    return model,feedback

########################### Nominal Control Inputs ############################

def compute_nominal_control(data: np.ndarray) -> Tuple:
    """ Computes the nominal state-feedback control input. """

    state = data['x']
    tt    = data['t']
    thHat = data['th']

    k_r   = 1.0   # Yaw rate proportional control gain
    k_phi = 125.0 # tau_phi proportional control gain
    k_the = 125.0 # tau_theta proportional control gain
    k_psi = 1.0   # tau_psi proportional control gain

    dr    = 1.0  # Damping ratio
    tc    = 0.5#.25  # Time constant
    G     = 9.81 # accel due to gravity

    Rp    = R_body_to_inertial(state)
    xddot,yddot,zddot,theta = get_nominal_acceleration(tt,state,thHat,dr,tc,Rp)

    F_c   = M * np.max([0,(G + zddot) / -Rp[2,2]])
    tau_phi_c = 0.0
    tau_the_c = 0.0
    tau_psi_c = 0.0
    if F_c > 0.0:
        R13_c = -M * xddot / F_c
        R23_c = -M * yddot / F_c
        R33_c = theta + np.pi/2

        tc_R  = 0.5*tc#0.5*tc
        rate  = 2.0
        if tt > 1.0:
            tc_Rf = 0.1*tc
            tc_R  = tc_Rf + (tc_R - tc_Rf) * np.exp(-rate*(tt - 1.0))


        R13dot_c = -(Rp[0,2] - R13_c) / (tc_R)
        R23dot_c = -(Rp[1,2] - R23_c) / (tc_R)

        p_c   = -R13dot_c*Rp[0,1] - R23dot_c*Rp[1,1]
        q_c   =  R13dot_c*Rp[0,0] + R23dot_c*Rp[1,2]
        r_c   = 0#-k_r * (state[8] - theta + np.pi/2)

        tau_phi_c = -k_phi * (state[9]  - p_c) * Jx - (Jy - Jz)*state[10]*state[11]
        tau_the_c = -k_the * (state[10] - q_c) * Jy - (Jz - Jx)*state[9]*state[11]
        tau_psi_c = 0#-k_psi * (state[11] - r_c) * Jz - (Jx - Jy)*state[9]*state[10]

    return F_c,np.array([tau_phi_c,tau_the_c,tau_psi_c])

def get_nominal_acceleration(tt:       float,
                             state:    np.ndarray,
                             thetaHat: np.ndarray,
                             dr:       float,
                             tc:       float,
                             Rp:       np.ndarray) -> Tuple:
    """ Returns the desired XYZ accelerations for the controller to track.
    Current trajectory design is a circle in XY.

    INPUTS
    ------
    tt:       (float) - time (sec)
    state:    (np.ndarray) - state vector
    thetaHat: (np.ndarray) - estimates of parameters
    dr:       (float) - damping ratio
    tc:       (float) - time constant
    Rp:       (np.ndarray) - body-fixed frame to inertial frame rotation matrix

    OUTPUTS
    -------
    xddot: (float) - x-accel to track
    yddot: (float) - y-accel to track
    zddot: (float) - z-accel to track
    """
    global TT0

    # Circle Parameters
    x0 = 0.0
    y0 = 0.0

    # Time Constant Adjustments
    tc_x = tc#2*tc
    tc_y = tc#2*tc
    tc_z = tc

    x,y,z = state[:3]
    xdot,ydot,zdot = Rp @ state[3:6] + regressor(state)[0:3] @ thetaHat
    # xdot,ydot,zdot = Rp @ state[3:6] + reg_est(state,thetaHat)[0:3] @ thetaHat + reg_const(state,thetaHat)[0:3]

    theta         = np.arctan2(y-y0,x-x0)
    theta_dot_0   = 2 * np.pi / (2 * 10.0) # 10 sec to complete semicircle
    theta_dot_c   = theta_dot_0# * (x - x0 + CIRCLE_R) / (CIRCLE_R)
    theta_ddot_c  = 0#-theta_dot_0**2 / 2 * np.sin(theta) # Constant velocity around circle

    x_c          = state[0]
    y_c          = state[1]
    xdot_c       = 0.0
    ydot_c       = 0.0
    xddot_c      = 0.0
    yddot_c      = 0.0
    z_c          = 2.0
    zdot_c       = 0.0
    zddot_c      = 0.0

    # # Circle / Ellipse Trajectory
    # if z > 1.0 or theta < -3*np.pi/4:
    #     if theta < -3*np.pi/4:
    #         z_c = 0.0

    #     x_sign_factor = -1*(theta >= np.pi/2 or theta < -np.pi/2) + 1*(theta >= -np.pi/2 and theta < np.pi/2)
    #     y_sign_factor = -1*(theta < 0.0) + 1*(theta >= 0.0)

    #     # x_c           = x0 + x_sign_factor * np.sqrt(abs(CIRCLE_R**2 - (y-y0)**2))
    #     # y_c           = y0 + y_sign_factor * np.sqrt(abs(CIRCLE_R**2 - (x-x0)**2))
    #     x_c           = x0 + x_sign_factor * ELLIPSE_AX * np.sqrt(abs(1 - ((y-y0)/ELLIPSE_BY)**2))
    #     y_c           = y0 + y_sign_factor * ELLIPSE_BY * np.sqrt(abs(1 - ((x-x0)/ELLIPSE_AX)**2))

    #     xdot_c        = -ELLIPSE_AX * theta_dot_c * np.sin(theta)
    #     ydot_c        =  ELLIPSE_BY * theta_dot_c * np.cos(theta)

    #     xddot_c       = -ELLIPSE_AX * (theta_ddot_c * np.sin(theta) + theta_dot_c**2 * np.cos(theta))
    #     yddot_c       =  ELLIPSE_BY * (theta_ddot_c * np.cos(theta) - theta_dot_c**2 * np.sin(theta))

    # Gerono Lemniscate Trajectory
    if z > 1.0 or (theta < -np.pi/2 and TT0 is not None):
        if TT0 is None:
            TT0 = tt
        if tt > 10.64:

        # if theta < -np.pi/2 and theta > -7*np.pi/8:
        #     # z_c = 0.0
        #     if z < 0.02 or tt > 13.0:
            raise ValueError('We are done.')

        B             = 2 * np.pi * F_GERONO
        x_sign_factor = -1*(theta >= np.pi/2 or theta < -np.pi/2) + 1*(theta >= -np.pi/2 and theta < np.pi/2)
        y_sign_factor = -1*(theta < 0.0) + 1*(theta >= 0.0)

        x_c           =  A_GERONO * np.sin(B * (tt - TT0))
        y_c           =  A_GERONO * np.sin(B * (tt - TT0)) * np.cos(B * (tt - TT0))

        xdot_c        =  B * A_GERONO * np.cos(B * (tt - TT0))
        ydot_c        =  B * A_GERONO * (np.cos(B * (tt - TT0))**2 - np.sin(B * (tt - TT0))**2)

        xddot_c       =   -B**2 * A_GERONO * np.sin(B * (tt - TT0))
        yddot_c       = -4*B**2 * A_GERONO * np.sin(B * (tt - TT0)) * np.cos(B * (tt - TT0))

    xddot   = -2 * dr / tc_x * (xdot - xdot_c) - (x - x_c) / tc_x**2 + xddot_c - Rp[0] @ regressor(state)[3:6] @ thetaHat
    yddot   = -2 * dr / tc_y * (ydot - ydot_c) - (y - y_c) / tc_y**2 + yddot_c - Rp[1] @ regressor(state)[3:6] @ thetaHat
    zddot   = -2 * dr / tc_z * (zdot - zdot_c) - (z - z_c) / tc_z**2 + zddot_c - Rp[2] @ regressor(state)[3:6] @ thetaHat
    # xddot   = -2 * dr / tc_x * (xdot - xdot_c) - (x - x_c) / tc_x**2 + xddot_c - Rp[0] @ (reg_est(state,thetaHat)[3:6] @ thetaHat + reg_const(state,thetaHat)[3:6])
    # yddot   = -2 * dr / tc_y * (ydot - ydot_c) - (y - y_c) / tc_y**2 + yddot_c - Rp[1] @ (reg_est(state,thetaHat)[3:6] @ thetaHat + reg_const(state,thetaHat)[3:6])
    # zddot   = -2 * dr / tc_z * (zdot - zdot_c) - (z - z_c) / tc_z**2 + zddot_c - Rp[2] @ (reg_est(state,thetaHat)[3:6] @ thetaHat + reg_const(state,thetaHat)[3:6])

    return xddot,yddot,zddot,theta

############################# Safe Control Inputs #############################

def compute_safe_force():
    global F_MODEL

    # Compute solution to QP
    optimize_qp(F_MODEL)

    # Debug solution to QP
    sol,success,code = check_qp_sol(F_MODEL)
    if not success:
        return code

    # Extract control solutions
    F   = sol[0]
    # pf1 = sol[1]

    return F#,pf1

def compute_safe_torques():
    global TAU_MODEL

    # Compute solution to QP
    optimize_qp(TAU_MODEL)

    # Debug solution to QP
    sol,success,code = check_qp_sol(TAU_MODEL)
    if not success:
        return code

    # Extract control solutions
    tau1 = sol[0]
    tau2 = sol[1]
    tau3 = sol[2]
    # pt1  = sol[3]
    # pt2  = sol[4]
    # pt3  = sol[5]

    return tau1,tau2,tau3#,pt1,pt2,pt3

def update_force_constraints(model:  gp.Model,
                             data:   dict,
                             d_vars: np.ndarray) -> Tuple:
    """ Computes the safety-compensating force control input for the 6-DOF
    dynamic quadrotor model.

    INPUTS
    ------
    data:   (dict) - contains time (t), state (x), unknown parameter estimate
                     vector (th), maximum allowable parameter estimation error
                     vector (err), and nth order eta terms for CBF derivative
    d_vars: (np.ndarray) - decision variables

    OUTPUTS
    -------
    F:   (float) - safe force control input [0,F_MAX]
    pf1: (np.ndarray) - 'adaptive' feasibility gains from Belta paper

    """
    # Unpack data dict
    tt       = data['t']
    state    = data['x']
    thetaHat = data['th']
    eta      = data['err']
    etaTerms = data['etaTerms']
    Gamma    = data['Gamma']
    feedback = ""
    e        = None

    # Remove old constraints
    model.remove(model.getConstrs())

    # Relative-Degree 2 Safety Constraint - Altitude Safety
    Lf2h  = cbf2dot_altitude_uncontrolled(state) - etaTerms[1]
    LgLfh = cbf2dot_altitude_controlled(state)
    Lfh   = cbfdot_altitude(state) - etaTerms[0]
    h     = cbf_altitude(state) - 1/2 * eta.T @ np.linalg.inv(Gamma) @ eta

    # K-Coefficients for HO-CBF Terms
    K0 = 1.0e3
    K1 = 1.0

    # Formalized Exponential CBF Condition (with feasibility parameter on h)
    ho_cbf = Lf2h + LgLfh*d_vars[0] + K1*Lfh + K0*h*d_vars[1]
    # ho_cbf = 10.0

    # Add New Constraint
    model.addConstr(ho_cbf >= 0)

    # Update Model
    model.update()

    return model,feedback,e

def update_torque_constraints(model:  gp.Model,
                              data:   dict,
                              d_vars: np.ndarray) -> Tuple:
    """ Computes the safety-compensating torque control inputs for the 6-DOF
    dynamic quadrotor model.

    INPUTS
    ------
    data:   (dict) - contains time (t), state (x), unknown parameter estimate
                     vector (th), maximum allowable parameter estimation error
                     vector (err), and nth order eta terms for CBF derivative
    d_vars: (np.ndarray) - decision variables

    OUTPUTS
    -------
    tau1: (float) - safe phi-torque control input [-tau1_max,tau1_max]
    tau2: (float) - safe theta-torque control input [-tau2_max,tau2_max]
    tau3: (float) - safe psi-torque control input [-tau3_max,tau3_max]
    p_vector: (np.ndarray) - 'adaptive' feasibility gains from Belta paper

    """
    # QP-Safety Tolerance (Numerical Integration Error)
    epsilon = 1e-6

    # Unpack data dict
    tt       = data['t']
    state    = data['x']
    thetaHat = data['th']
    eta      = data['err']
    etaTerms = data['etaTerms']
    Gamma    = data['Gamma']
    F        = data['F_sol']
    feedback = ""
    e        = None

    # Class K Function Parameters - Relative Degree 4
    rate   = 10.0
    alpha0 = 1.0e1; alpha0f = 1.0e1
    alpha1 = 1.0e1; alpha1f = 1.0e1
    alpha2 = 1.0e1; alpha2f = 1.0e1
    alpha3 = 1.0e1; alpha3f = 1.0e1
    # if tt > 1.0:
    #     alpha0 = alpha0f + (alpha0 - alpha0f) * np.exp(-rate*(tt - 1.0))
    #     alpha1 = alpha1f + (alpha1 - alpha1f) * np.exp(-rate*(tt - 1.0))
    #     alpha2 = alpha2f + (alpha2 - alpha2f) * np.exp(-rate*(tt - 1.0))
    #     alpha3 = alpha3f + (alpha3 - alpha3f) * np.exp(-rate*(tt - 1.0))

    K0 = alpha3*alpha2*alpha1*alpha0
    K1 = alpha3*(alpha2*(alpha1 + alpha0) + alpha1*alpha0) + alpha2*alpha1*alpha0
    K2 = alpha3*(alpha2 + alpha1 + alpha0) + alpha2*(alpha1 + alpha0) + alpha1*alpha0
    K3 = alpha3 + alpha2 + alpha1 + alpha0

    if tt > 1.0:
        K0 = 1.0 + (K0 - 1.0) * np.exp(-rate*(tt - 1.0))
        K1 = 2.0 + (K1 - 2.0) * np.exp(-rate*(tt - 1.0))
        K2 = 3.0 + (K2 - 3.0) * np.exp(-rate*(tt - 1.0))
        K3 = 4.0 + (K3 - 4.0) * np.exp(-rate*(tt - 1.0))

    feedback = feedback + "\nK0: {}\nK1: {}\nK2: {}\nK3: {}".format(K0,K1,K2,K3)

    # Remove old constraints
    model.remove(model.getConstrs())

    # Add Input Constraint Based on Force
    max_tau1 = F * arm_length - d_vars[0]
    min_tau1 = F * arm_length + d_vars[0]
    max_tau2 = F * arm_length - d_vars[1]
    min_tau2 = F * arm_length + d_vars[1]
    model.addConstr(max_tau1 >= 0)
    model.addConstr(min_tau1 >= 0)
    model.addConstr(max_tau2 >= 0)
    model.addConstr(min_tau2 >= 0)

    # # Relative-Degree 4 Safety Constraint - Lateral Safety
    # Lf4h   = cbf4dot_lateral_uncontrolled(state,F) - etaTerms[3]
    # LgLf3h = cbf4dot_lateral_controlled(state,F)
    # Lf3h   = cbf3dot_lateral(state,F) - etaTerms[2]
    # Lf2h   = cbf2dot_lateral(state,F) - etaTerms[1]
    # Lfh    = cbfdot_lateral(state) - etaTerms[0]
    # h      = cbf_lateral(state) - 1/2 * eta.T @ np.linalg.inv(Gamma) @ eta

    # # K-Coefficients for HO-CBF Terms
    # K0 = 1.0e4
    # K1 = 1.0e1
    # K2 = 1.0e1
    # K3 = 1.0e1

    # # Formalized Exponential CBF Condition (with feasibility parameter on h)
    # ho_cbf = Lf4h + LgLf3h @ d_vars[0:3] + K3*Lf3h + K2*Lf2h + K1*Lfh + K0*h
    # # ho_cbf = 10.0

    # # Add New Constraint
    # model.addConstr(ho_cbf >= 0)

    # # Relative-Degree 4 Safety Constraint - Lateral Safety
    # Lf4h   = cbf4dot_lateral_outer_uncontrolled(state,F) - etaTerms[3]
    # LgLf3h = cbf4dot_lateral_outer_controlled(state,F)
    # Lf3h   = cbf3dot_lateral_outer(state,F) - etaTerms[2]
    # Lf2h   = cbf2dot_lateral_outer(state,F) - etaTerms[1]
    # Lfh    = cbfdot_lateral_outer(state) - etaTerms[0]
    # h      = cbf_lateral_outer(state) - 1/2 * eta.T @ np.linalg.inv(Gamma) @ eta

    # # Formalized Exponential CBF Condition (with feasibility parameter on h)
    # ho_cbf = Lf4h + LgLf3h @ d_vars[0:3] + K3*Lf3h*d_vars[6] + K2*Lf2h*d_vars[5] + K1*Lfh*d_vars[4] + K0*h*d_vars[3]
    # # ho_cbf = 10.0

    # # Add New Constraint
    # # model.addConstr(ho_cbf >= epsilon)
    # feedback = feedback + "\nSafetyOut:  {:.3f} + {}u + {:.3f} + {:.3f} + {:.3f} + {:.3f} >= 0".format(Lf4h,np.around(LgLf3h,3),K3*Lf3h,K2*Lf2h,K1*Lfh,K0*h)
    # if h < 0:
    #     e = ValueError('SafetyOut Violated')

    # # Relative-Degree 4 Safety Constraint - Lateral Safety
    # Lf4h   = cbf4dot_lateral_inner_uncontrolled(state,F) - etaTerms[3]
    # LgLf3h = cbf4dot_lateral_inner_controlled(state,F)
    # Lf3h   = cbf3dot_lateral_inner(state,F) - etaTerms[2]
    # Lf2h   = cbf2dot_lateral_inner(state,F) - etaTerms[1]
    # Lfh    = cbfdot_lateral_inner(state) - etaTerms[0]
    # h      = cbf_lateral_inner(state) - 1/2 * eta.T @ np.linalg.inv(Gamma) @ eta

    # # Formalized Exponential CBF Condition (with feasibility parameter on h)
    # ho_cbf = Lf4h + LgLf3h @ d_vars[0:3] + K3*Lf3h*d_vars[10] + K2*Lf2h*d_vars[9] + K1*Lfh*d_vars[8] + K0*h*d_vars[7]

    # # Add New Constraint
    # # model.addConstr(ho_cbf >= epsilon)
    # feedback = feedback + "\nSafetyIn:   {:.3f} + {}u + {:.3f} + {:.3f} + {:.3f} + {:.3f} >= 0".format(Lf4h,np.around(LgLf3h,3),K3*Lf3h,K2*Lf2h,K1*Lfh,K0*h)
    # if h < 0:
    #     e = ValueError('SafetyIn Violated')

    # # Relative-Degree 2 Safety Constraint - Lateral Safety
    # Lf2h   = cbf2dot_lateral_outer_uncontrolled(state,F) - etaTerms[1]
    # LgLfh  = cbf2dot_lateral_outer_controlled(state,F)
    # Lfh    = cbfdot_lateral_outer(state) - etaTerms[0]
    # h      = cbf_lateral_outer(state) - 1/2 * eta.T @ np.linalg.inv(Gamma) @ eta

    # # Formalized Exponential CBF Condition (with feasibility parameter on h)
    # ho_cbf = Lf2h + LgLfh @ d_vars[0:3] + K1*Lfh*d_vars[4] + K0*h*d_vars[3]

    # # Add New Constraint
    # model.addConstr(ho_cbf >= epsilon)
    # feedback = feedback + "\nSafetyOut:  {:.3f} + {}u + {:.3f} + {:.3f} >= 0".format(Lf2h,np.around(LgLfh,3),K1*Lfh,K0*h)
    # if h < 0:
    #     e = ValueError('SafetyOut Violated')

    # # Relative-Degree 2 Safety Constraint - Lateral Safety
    # Lf2h   = cbf2dot_lateral_inner_uncontrolled(state,F) - etaTerms[1]
    # LgLfh  = cbf2dot_lateral_inner_controlled(state,F)
    # Lfh    = cbfdot_lateral_inner(state) - etaTerms[0]
    # h      = cbf_lateral_inner(state) - 1/2 * eta.T @ np.linalg.inv(Gamma) @ eta

    # # Formalized Exponential CBF Condition (with feasibility parameter on h)
    # ho_cbf = Lf2h + LgLfh @ d_vars[0:3] + K1*Lfh*d_vars[8] + K0*h*d_vars[7]

    # # Add New Constraint
    # model.addConstr(ho_cbf >= epsilon)
    # feedback = feedback + "\nSafetyIn:   {:.3f} + {}u + {:.3f} + {:.3f} >= 0".format(Lf2h,np.around(LgLfh,3),K1*Lfh,K0*h)
    # if h < 0:
    #     e = ValueError('SafetyIn Violated')

    # # Relative-Degree 3 Safety Constraint - Lateral Safety
    # Lf3h   = cbf3dot_velocity_outer_uncontrolled(state,F) - etaTerms[2]
    # LgLf2h = cbf3dot_velocity_outer_controlled(state,F)
    # Lf2h   = cbf2dot_velocity_outer(state,F) - etaTerms[1]
    # Lfh    = cbfdot_velocity_outer(state,F) - etaTerms[0]
    # h      = cbf_velocity_outer(state) - 1/2 * eta.T @ np.linalg.inv(Gamma) @ eta

    # # Formalized Exponential CBF Condition (with feasibility parameter on h)
    # ho_cbf = Lf3h + LgLf2h @ d_vars[0:3] + K2*Lf2h*d_vars[13] + K1*Lfh*d_vars[12] + K0*h*d_vars[11]

    # # Add New Constraint
    # model.addConstr(ho_cbf >= epsilon)
    # feedback = feedback + "\nSafetyVelOut:   {:.3f} + {}u + {:.3f} + {:.3f} + {:.3f} >= 0".format(Lf3h,np.around(LgLf2h,3),K2*Lf2h,K1*Lfh,K0*h)
    # if h < 0:
    #     e = ValueError('SafetyVelOut Violated')

    # # Relative-Degree 3 Safety Constraint - Lateral Safety
    # Lf3h   = cbf3dot_velocity_inner_uncontrolled(state,F) - etaTerms[2]
    # LgLf2h = cbf3dot_velocity_inner_controlled(state,F)
    # Lf2h   = cbf2dot_velocity_inner(state,F) - etaTerms[1]
    # Lfh    = cbfdot_velocity_inner(state,F) - etaTerms[0]
    # h      = cbf_velocity_inner(state) - 1/2 * eta.T @ np.linalg.inv(Gamma) @ eta

    # # Formalized Exponential CBF Condition (with feasibility parameter on h)
    # ho_cbf = Lf3h + LgLf2h @ d_vars[0:3] + K2*Lf2h*d_vars[16] + K1*Lfh*d_vars[15] + K0*h*d_vars[14]

    # # Add New Constraint
    # model.addConstr(ho_cbf >= epsilon)
    # feedback = feedback + "\nSafetyVelIn:   {:.3f} + {}u + {:.3f} + {:.3f} + {:.3f} >= 0".format(Lf3h,np.around(LgLf2h,3),K2*Lf2h,K1*Lfh,K0*h)
    # if h < 0:
    #     e = ValueError('SafetyVelIn Violated')

    alpha0 = 1.0
    alpha1 = 1.0

    # Relative-Degree 2 Safety Constraint - Attitude Safety
    Lf2h   = cbf2dot_attitude_uncontrolled(state) - etaTerms[1]
    LgLfh  = cbf2dot_attitude_controlled(state)
    Lfh    = cbfdot_attitude(state) - etaTerms[0]
    h      = cbf_attitude(state) - 1/2 * eta.T @ np.linalg.inv(Gamma) @ eta

    K0 = alpha1*alpha0
    K1 = alpha1 + alpha0
    if tt > 1.0:
        K0 = 1.0 + (K0 - 1.0) * np.exp(-rate*(tt - 1.0))
        K1 = 1.0 + (K1 - 1.0) * np.exp(-rate*(tt - 1.0))
        # K0 = 100.0 + (K0 - 100.0) * np.exp(-rate*(tt - 1.0))
        # K1 = 10.0  + (K1 - 10.0)  * np.exp(-rate*(tt - 1.0))

    # Formalized Exponential CBF Condition (with feasibility parameter on h)
    ho_cbf = Lf2h + LgLfh @ d_vars[0:3] + K1*Lfh*d_vars[18] + K0*h*d_vars[17]

    # Add New Constraint
    model.addConstr(ho_cbf >= epsilon)
    feedback = feedback + "\nSafetyAtt:  {:.3f} + {}u + {:.3f} + {:.3f} >= 0".format(Lf2h,np.around(LgLfh,3),K1*Lfh,K0*h)
    if h < 0:
        e = ValueError('SafetyAtt Violated')

    # Update Model
    model.update()

    return model,feedback,e

###############################################################################
################################# Controller ##################################
###############################################################################

def optimize_qp(model: gp.Model) -> None:
    """ Reverts the Model settings to the defaults and then calls the
    gp.Model.optimize method to solve the optimization problem.

    INPUTS
    ------
    None

    RETURNS
    ------
    None

    """
    # Revert to default settings
    model.setParam('BarHomogeneous',-1)
    model.setParam('NumericFocus',0)

    # Solve
    model.optimize()

def check_qp_sol(model: gp.Model,
                 level: int = 0,
                 multiplier: int = 10):
    """
    Processes the status flag associated with the Model in order to perform
    error handling. If necessary, this will make adjustments to the solver
    settings and attempt to re-solve the optimization problem to obtain an
    accurate, feasible solution.

    INPUTS
    ------
    level: (int, optional) - current level of recursion in solution attempt

    RETURNS
    ------
    sol  : (np.ndarray)    - decision variables which solve the opt. prob.
    T/F  : (bool)          - pure boolean value denoting success or failure
    ERROR: (np.ndarray)    - error code for loop-breaking at higher level
    """
    # Define Error Checking Parameters
    status  = model.status
    epsilon = 0.1
    success = 2

    # Obtain solution
    sol = model.getVars()

    # Check status
    if status == success:
        sol = saturate_solution(sol)
        return sol,True,0

    else:
        model.write('diagnostic.lp')

        if status == 3:
            msg = "INFEASIBLE"
        elif status == 4:
            msg = "INFEASIBLE_OR_UNBOUNDED"
        elif status == 5:
            msg = "UNBOUNDED"
        elif status == 6:
            msg = "CUTOFF"
        elif status == 7:
            msg = "ITERATION_LIMIT"
        elif status == 8:
            msg = "NODE_LIMIT"
        elif status == 9:
            msg = "TIME_LIMIT"
        elif status == 10:
            msg = "SOLUTION_LIMIT"
        elif status == 11:
            msg = "INTERRUPTED"
        elif status == 12:
            msg = "NUMERIC"
        elif status == 13:
            msg = "SUBOPTIMAL"
        elif status == 14:
            msg = "INPROGRESS"
        elif status == 15:
            msg = "USER_OBJ_LIMIT"

        if status == 13:
            sol = saturate_solution(sol)
            return sol,True,0

        print("Solver Returned Code: {}".format(msg))

        return sol,False,ERROR

def saturate_solution(sol):
    saturated_sol = np.zeros((len(sol),))
    for ii,s in enumerate(sol):
        saturated_sol[ii] = np.min([np.max([s.x,s.lb]),s.ub])
    return saturated_sol


# ###############################################################################
# ################################# Controller ##################################
# ###############################################################################

# class CascadedClfCbfController():

#     def __init__(self,
#                  controllers: List = []):
#         """
#         """
#         self.t             = 0
#         self.x             = None
#         self.state         = None
#         self.sol           = None

#         # Assign controllers
#         assert type(controllers) is List
#         self.controllers = controllers

#     def update_tx(self,t,x):
#         self.t = t
#         self.x = x
#         self.state = x

#     def compute(self):
#         """ Computes the solution to the Quadratic Program.

#         INPUTS
#         ------
#         None

#         RETURNS
#         ------
#         sol: (np.ndarray) - decision variables which solve opt. prob.

#         """

#         F_c,t1_c,t2_c,t3_c = self.compute_nominal_control()

#         hi_cont = self.controllers[0]
#         lo_cont = self.controllers[1]

#         sol_hi  = hi_cont.compute(F_c)

#         lo_cont.update_low_level(sol_hi)
#         sol     = lo_cont.compute(t1_c,t2_c,t3_c)

#         # sol = None
#         # for cont in self.controllers:
#         #     if sol is not None
#         #         # Need to pass the new control solution to the next controller
#         #         cont.update_low_level(sol)

#         #     sol = cont.compute()

#         self.sol = sol

#         return self.sol

#     def compute_nominal_control(self):
#         """ Computes the nominal state-feedback control input. """

#         k_r   = 5.0
#         k_phi = 2.0
#         k_the = 2.0
#         k_psi = 2.0

#         dr  = 1.0 # Damping ratio
#         tc  = 1.0 # Time constant
#         g   = 9.81 # accel due to gravity
#         Rp  = R_body_to_inertial(self.state)

#         xddot,yddot,zddot,theta = self.get_nominal_acceleration(dr,tc,g)

#         F_c   = np.max([0,(zddot + g) / Rp[2,2]])
#         R13_c = xddot / F_c
#         R23_c = yddot / F_c
#         R33_c = theta + np.pi/2

#         R13dot_c = (Rp[0,2] - R13_c) / tc
#         R23dot_c = (Rp[1,2] - R23_c) / tc

#         p_c   = 1 / Rp[2,2] * (R13dot_c*Rp[1,0] - R23dot_c*Rp[0,0])
#         q_c   = 1 / Rp[2,2] * (R13dot_c*Rp[1,1] - R23dot_c*Rp[0,1])
#         r_c   = k_r * (self.state[8] - theta + np.pi/2)

#         tau_phi_c = k_phi * (self.state[9]  - p_c) * Jx + (Jz - Jy)*self.state[10]*self.state[11]
#         tau_the_c = k_the * (self.state[10] - q_c) * Jy + (Jx - Jz)*self.state[9]*self.state[11]
#         tau_psi_c = k_psi * (self.state[11] - r_c) * Jz + (Jy - Jx)*self.state[9]*self.state[10]

#         return F_c,tau_phi_c,tau_the_c,tau_psi_c

#     def get_nominal_acceleration(self,
#                                  dr: float,
#                                  tc: float,
#                                  g:  float):
#         """ Returns the desired XYZ accelerations for the controller to track.
#         Current trajectory design is a circle in XY.

#         INPUTS
#         ------
#         dr: (float) - damping ratio
#         tc: (float) - time constant
#         g:  (float) - accel due to gravity

#         OUTPUTS
#         -------
#         xddot_c: (float) - x-accel to track
#         yddot_c: (float) - y-accel to track
#         zddot_c: (float) - z-accel to track
#         """
#         # Circle Parameters
#         x0 = R
#         y0 = 0.0

#         x,y,z = self.state[:3]
#         xdot,ydot,zdot = R_body_to_inertial(self.state) @ self.state[3:6]

#         theta         = np.arctan2(y-y0,x-x0)
#         theta_dot_0   = 2 * np.pi / (2 * 10.0)
#         theta_dot_c   = theta_dot_0 * (x - x0 + R) / (2*R) # 10 sec to complete semicircle
#         theta_ddot_c  = -theta_dot_0**2 / 2 * np.sin(theta) # Constant velocity around circle

#         xddot        = 0.0
#         yddot        = 0.0
#         z_c          = 2.0
#         zdot_c       = 0.0
#         zddot_c      = 0.0

#         if z > 1 or theta < -3*np.pi/4:
#             if theta < -3*np.pi/4:
#                 z_c = 0.0

#             x_sign_factor = -1*(theta >= np.pi/2 or theta < -np.pi/2) + 1*(theta >= -np.pi/2 and theta < np.pi/2)
#             y_sign_factor = -1*(theta < 0.0) + 1*(theta >= 0.0)

#             x_c           = x_sign_factor * np.sqrt(R**2 - y**2)
#             y_c           = y_sign_factor * np.sqrt(R**2 - x**2)

#             xdot_c        = -R * theta_dot_c * np.sin(theta)
#             ydot_c        =  R * theta_dot_c * np.cos(theta)

#             xddot_c       = -R * (theta_ddot_c * np.sin(theta) + theta_dot_c**2 * np.cos(theta))
#             xddot         = -2 * dr / tc * (xdot - xdot_c) - (x - x_c) / tc**2 + xddot_c

#             yddot_c       =  R * (theta_ddot_c * np.cos(theta) - theta_dot_c**2 * np.sin(theta))
#             yddot         = -2 * dr / tc * (ydot - ydot_c) - (y - y_c) / tc**2 + yddot_c

#         zddot   = -2 * dr / tc * (zdot - zdot_c) - (z - z_c) / tc**2 + zddot_c

#         return xddot,yddot,zddot