import numpy as np
from typing import Callable
from nptyping import NDArray
from scipy.linalg import block_diag
from solve_cvxopt import solve_qp_cvxopt
from .cbfs import cbfs_individual, cbfs_pairwise, H0, cbf0 as cbf_vals
from .cost_functions import objective_accel_only, objective_accel_and_steering
from ..dynamics import f, g, sigma

###############################################################################
################################## Functions ##################################
###############################################################################


def compute_control(t: float,
                    z: NDArray,
                    nominal_controller: Callable,
                    ego_id: int,
                    extras: dict = None) -> (NDArray, NDArray, int, str):
    """ Solves

    INPUTS
    ------
    t: time (in sec)
    z: state vector (Ax5) -- A is number of agents
    extras: contains time additional information needed to compute the control

    OUTPUTS
    -------
    u_act: actual control input vector (2x1) = (time-derivative of body slip angle, rear-wheel acceleration)
    u_0: nominal control input vector (2x1)
    code: error/success code (0 or 1)
    status: string containing error message (if relevant)

    """
    # Error checking variables
    code = 0
    status = ""
    ego = ego_id

    # Ignore agent if necessary (i.e. if comparing controllers for given initial conditions)
    if extras is not None:
        if 'ignore' in extras.keys():
            ignore = extras['ignore']
            z = np.delete(z, ignore, 0)
            if ego > ignore:
                ego = ego - 1

    # Partition ego (e) and other (o) states
    ze = z[ego, :]
    zo = np.vstack([z[:ego, :], z[ego + 1:, :]])

    # # Compute nominal control inputs
    # u_nom = np.zeros((len(z), 2))
    # for aa, zz in enumerate(z):
    #     omega, ar = compute_nominal_control(t, zz, aa)
    #     u_nom[aa, :] = np.array([omega, ar])

    # Compute nominal control input for ego only -- assume others are zero
    u_nom = np.zeros((len(z), 2))
    (omega, ar), code, status = nominal_controller(t, ze, ego)
    u_nom[ego, :] = np.array([omega, ar])

    # Get matrices and vectors for QP controller
    Q, p, A, b, G, h = get_constraints_accel_only(t, ze, zo, u_nom, ego)

    # Solve QP
    sol = solve_qp_cvxopt(Q, p, A, b, G, h)

    # If accel-only QP is infeasible, add steering control as decision variable
    if not sol['code']:
        # Get matrices and vectors for QP controller
        Q, p, A, b, G, h = get_constraints_accel_and_steering(t, ze, zo, u_nom, ego)

        # Solve QP
        sol = solve_qp_cvxopt(Q, p, A, b, G, h)

        # Return error if this is also infeasible
        if not sol['code']:
            return np.zeros((2,)), u_nom[ego, :], sol['code'], sol['status']

        # Format control solution -- accel and steering solution
        u_act = np.array(sol['x'][2 * ego: 2 * (ego + 1)]).flatten()

    else:
        # Format control solution -- nominal steering, accel solution
        u_act = np.array([u_nom[ego, 0], sol['x'][ego]])

    u_act = np.clip(u_act, [-np.pi / 4, -9.81], [np.pi / 4, 9.81])
    u_0 = u_nom[ego, :]

    return u_act, u_0, cbf_vals, sol['code'], sol['status'], None


############################# Safe Control Inputs #############################


def get_constraints_accel_only(t: float,
                               ze: NDArray,
                               zr: NDArray,
                               u_nom: NDArray,
                               ego: int) -> (NDArray, NDArray, NDArray, NDArray, NDArray, NDArray):
    """ Generates and returns inter-agent objective and constraint functions for the single-agent QP controller
    of the form:

    J = u.T * Q * u + p * u
    subject to
    Au <= b
    Gu = h

    INPUTS
    ------
    t: time (in sec)
    za: state vector for agent in question (5x1)
    zr: array of state vectors for remaining agents in system (Ax5)
    u_nom: array of nominal control inputs
    agent: index for agent in question

    OUTPUTS
    -------
    Q: (AUxAU) positive semi-definite matrix for quadratic decision variables in QP objective function
    p: (AUx1) vector for decision variables with linear terms
    A: ((A-1)xAU) matrix multiplying decision variables in inequality constraints for QP
    b: ((A-1)x1) vector for inequality constraints for QP
    G: None -- no equality constraints right now
    h: None -- no equality constraints right now

    """
    # Parameters
    na = 1 + len(zr)
    ns = len(ze)

    # Objective function
    Q, p = objective_accel_only(u_nom[:, 1], ego)

    # Input constraints (accel only)
    omega_e = u_nom[ego, 0]
    au = np.array([1, -1])
    Au = block_diag(*na*[au]).T
    bu = np.array(2*na*[9.81])

    # Initialize CBF constraints
    Ai = np.zeros((len(zr), na))
    bi = np.zeros((len(zr),))

    # Iterate over individual CBF constraints
    for cc, cbf in enumerate(cbfs_individual):
        h = cbf.h(ze)
        dhdx = cbf.dhdx(ze)
        d2hdx2 = cbf.d2hdx2(ze)

        # Stochastic Term -- 0 for deterministic systems
        stoch = 0.5 * np.trace(sigma(ze).T @ d2hdx2 @ sigma(ze))

        # Get CBF Lie Derivatives
        Lfh = dhdx @ (f(ze) + g(ze)[:, 0] * omega_e) + stoch
        Lgh = np.zeros((na,))
        Lgh[ego] = dhdx @ g(ze)[:, 1]

        Ai[cc, :], bi[cc] = cbf.generate_cbf_condition(h, Lfh, Lgh)
        cbf_vals[cc] = h

    lci = len(cbfs_individual)

    # Iterate over pairwise CBF constraints
    for cc, cbf in enumerate(cbfs_pairwise):

        # Iterate over all other vehicles
        for ii, zo in enumerate(zr):
            idx = ii + (ii >= ego)
            omega_o = u_nom[idx, 0]

            h0 = H0(ze, zo)
            h = cbf.h(ze, zo)
            dhdx = cbf.dhdx(ze, zo)
            d2hdx2 = cbf.d2hdx2(ze, zo)

            # Stochastic Term -- 0 for deterministic systems
            stoch = 0.5 * (np.trace(sigma(ze).T @ d2hdx2[:ns, :ns] @ sigma(ze)) +
                           np.trace(sigma(zo).T @ d2hdx2[ns:, ns:] @ sigma(zo)))

            # Get CBF Lie Derivatives
            Lfh = dhdx[:ns] @ (f(ze) + g(ze)[:, 0] * omega_e) + dhdx[ns:] @ (f(zo) + g(zo)[:, 0] * omega_o)
            Lfh = Lfh + stoch
            Lgh = np.zeros((na,))
            Lgh[ego] = dhdx[:ns] @ g(ze)[:, 1]
            Lgh[idx] = dhdx[ns:] @ g(zo)[:, 1]

            Ai[cc, :], bi[cc] = cbf.generate_cbf_condition(h, Lfh, Lgh)
            cbf_vals[lci + cc] = h

            if h0 < 0:
                print("SAFETY VIOLATION: {:.2f}".format(-h0))

    A = np.vstack([Au, Ai])
    b = np.hstack([bu, bi])

    return Q, p, A, b, None, None


def get_constraints_accel_and_steering(t: float,
                                       za: NDArray,
                                       zo: NDArray,
                                       u_nom: NDArray,
                                       agent: int) -> (NDArray, NDArray, NDArray, NDArray, NDArray, NDArray):
    """ Generates and returns inter-agent objective and constraint functions for the single-agent QP controller
    of the form:

    J = u.T * Q * u + p * u
    subject to
    Au <= b
    Gu = h

    INPUTS
    ------
    t: time (in sec)
    za: state vector for agent in question (5x1)
    zo: array of state vectors for remaining agents in system (Ax5)
    u_nom: array of nominal control inputs
    agent: index for agent in question

    OUTPUTS
    -------
    Q: (AUxAU) positive semi-definite matrix for quadratic decision variables in QP objective function
    p: (AUx1) vector for decision variables with linear terms
    A: ((A-1)xAU) matrix multiplying decision variables in inequality constraints for QP
    b: ((A-1)x1) vector for inequality constraints for QP
    G: None -- no equality constraints right now
    h: None -- no equality constraints right now

    """
    # Parameters
    Na = 1 + len(zo)
    Nu = 2
    discretization_error = 0.5

    # Objective function
    Q, p = objective_accel_and_steering(u_nom.flatten())

    # Input constraints (accel only)
    au = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    Au = block_diag(*Na*[au])
    bu = np.array(Na*[np.pi / 4, np.pi / 4, 9.81, 9.81])

    # Initialize inequality constraints
    Ai = np.zeros((len(zo), Nu*Na))
    bi = np.zeros((len(zo),))

    for ii, zz in enumerate(zo):
        idx = ii + (ii >= agent)

        # x and y differentials
        dx = za[0] - zz[0]
        dy = za[1] - zz[1]

        # vx and vy differentials
        dvx = f(za)[0] - f(zz)[0]
        dvy = f(za)[1] - f(zz)[1]

        # ax and ay differentials (uncontrolled)
        axa_unc = -za[3] / Lr * np.tan(za[4]) * f(za)[1]
        aya_unc = za[3] / Lr * np.tan(za[4]) * f(za)[0]
        axo_unc = -zz[3] / Lr * np.tan(zz[4]) * f(zz)[1]
        ayo_unc = zz[3] / Lr * np.tan(zz[4]) * f(zz)[0]
        dax_unc = axa_unc - axo_unc
        day_unc = aya_unc - ayo_unc

        # ax and ay differentials (controlled)
        axa_con = np.zeros((Nu * Na,))
        aya_con = np.zeros((Nu * Na,))
        axo_con = np.zeros((Nu * Na,))
        ayo_con = np.zeros((Nu * Na,))
        axa_con[Nu * agent:Nu * (agent + 1)] = np.array([-za[3] * np.sin(za[2]) / np.cos(za[4])**2,
                                                         np.cos(za[2]) - np.sin(za[2]) * np.tan(za[4])])
        aya_con[Nu * agent:Nu * (agent + 1)] = np.array([za[3] * np.cos(za[2]) / np.cos(za[4])**2,
                                                         np.sin(za[2]) + np.cos(za[2]) * np.tan(za[4])])
        axo_con[Nu * idx:Nu * (idx + 1)] = np.array([-zz[3] * np.sin(zz[2]) / np.cos(zz[4])**2,
                                                     np.cos(zz[2]) - np.sin(zz[2]) * np.tan(zz[4])])
        ayo_con[Nu * idx:Nu * (idx + 1)] = np.array([zz[3] * np.cos(zz[2]) / np.cos(zz[4])**2,
                                                     np.sin(zz[2]) + np.cos(zz[2]) * np.tan(zz[4])])
        dax_con = axa_con - axo_con
        day_con = aya_con - ayo_con

        # x scale: designed to enforce larger safety distance in direction of travel
        x_scale = .5  # -- < 1 for highway scenario
        dx = x_scale * dx
        dvx = x_scale * dvx
        dax_unc = x_scale * dax_unc
        dax_con = x_scale * dax_con

        # Solve for minimizer (tau) of ||[dx dy] + [dvx dvy]t|| from 0 to T
        T = 3.0
        kh = 1000.0
        epsilon = 1e-3
        tau_star = -(dx * dvx + dy * dvy) / (dvx**2 + dvy**2 + epsilon)
        tau = tau_star * heavyside_approx(tau_star, kh, 0.0) - (tau_star - T) * heavyside_approx(tau_star, kh, T)

        # Derivatives of tau (controllable and uncontrollable)
        tau_star_dot_unc = -(dax_unc * (2 * dvx * tau_star + dx) + day_unc * (2 * dvy * tau_star + dy) +
                             (dvx**2 + dvy**2)) / (dvx**2 + dvy**2 + epsilon)
        tau_star_dot_con = -(dax_con * (2 * dvx * tau_star + dx) + day_con * (2 * dvy * tau_star + dy)) / \
                           (dvx**2 + dvy**2 + epsilon)
        tau_dot_unc = tau_star_dot_unc * (heavyside_approx(tau_star, kh, 0.0) - heavyside_approx(tau_star, kh, T)) + \
                      tau_star * tau_star_dot_unc * (dheavyside_approx(tau_star, kh, 0.0) -
                                                     dheavyside_approx(tau_star, kh, T))
        tau_dot_con = tau_star_dot_con * (heavyside_approx(tau_star, kh, 0.0) - heavyside_approx(tau_star, kh, T)) + \
                      tau_star * tau_star_dot_con * (dheavyside_approx(tau_star, kh, 0.0) -
                                                     dheavyside_approx(tau_star, kh, T))

        # CBF Definitions
        h0 = dx**2 + dy**2 - (2*R)**2
        ht = h0 + tau**2 * (dvx**2 + dvy**2) + 2 * tau * (dx * dvx + dy * dvy) - discretization_error

        # CBF Derivatives (Lie derivative notation -- Lfh = dh/dx * f(x))
        Lfh0 = 2 * (dx * dvx + dy * dvy)
        Lfht = Lfh0 + 2 * tau * (dvx**2 + dvy**2 + dx * dax_unc + dy * day_unc) + 2 * tau_dot_unc * \
               (dx * dvx + dy * dvy + tau * (dvx**2 + dvy**2)) + 2 * tau**2 * (dvx * dax_unc + dvy * day_unc)
        Lght = 2 * tau * tau_dot_con * (dvx**2 + dvy**2) + 2 * tau**2 * (dvx * dax_con + dvy * day_con) + \
               2 * tau_dot_con * (dx * dvx + dy * dvy) + 2 * tau * (dx * dax_con + dy * day_con)

        # CBF Condition: Lfht + Lght + l0 * ht >= 0 --> Au <= b
        l0 = 10.0
        Ai[ii, :] = -Lght
        bi[ii] = Lfht + l0 * ht

        if h0 < 0:
            print("SAFETY VIOLATION: {:.2f}".format(-h0))

    A = np.vstack([Au, Ai])
    b = np.hstack([bu, bi])

    return Q, p, A, b, None, None


def saturate_solution(sol):
    saturated_sol = np.zeros((len(sol),))
    for ii,s in enumerate(sol):
        saturated_sol[ii] = np.min([np.max([s.x,s.lb]),s.ub])
    return saturated_sol


def heavyside_approx(x: float,
                     k: float,
                     d: float) -> float:
    """ Approximation to the unit heavyside function.

    INPUTS
    ------
    x: independent variable
    k: scale factor
    d: offset factor

    OUTPUTS
    -------
    y = 1/2 + 1/2 * tanh(k * (x - d))

    """
    return 0.5 * (1 + np.tanh(k * (x - d)))


def dheavyside_approx(x: float,
                      k: float,
                      d: float) -> float:
    """ Derivative of approximation to the unit heavyside function.

    INPUTS
    ------
    x: independent variable
    k: scale factor
    d: offset factor

    OUTPUTS
    -------
    dy = k/2 * (sech(k * (x - d))) ** 2

    """
    return k / (2 * (np.cosh(k * (x - d)))**2)


# # FF-CBF Condition by Hand
# x and y differentials
# dx = ze[0] - zo[0]
# dy = ze[1] - zo[1]
#
# # vx and vy differentials
# dvx = f(ze)[0] - f(zo)[0]
# dvy = f(ze)[1] - f(zo)[1]
#
# # ax and ay differentials (uncontrolled)
# axa_unc = -ze[3] / Lr * np.tan(ze[4]) * f(ze)[1] - omega_a * ze[3] * np.sin(ze[2]) / np.cos(ze[4])**2
# aya_unc = ze[3] / Lr * np.tan(ze[4]) * f(ze)[0] + omega_a * ze[3] * np.cos(ze[2]) / np.cos(ze[4])**2
# axo_unc = -zo[3] / Lr * np.tan(zo[4]) * f(zo)[1] - omega_z * zo[3] * np.sin(zo[2]) / np.cos(zo[4])**2
# ayo_unc = zo[3] / Lr * np.tan(zo[4]) * f(zo)[0] + omega_z * zo[3] * np.cos(zo[2]) / np.cos(zo[4])**2
# dax_unc = axa_unc - axo_unc
# day_unc = aya_unc - ayo_unc
#
# # ax and ay differentials (controlled)
# axa_con = np.zeros((Na,))
# aya_con = np.zeros((Na,))
# axo_con = np.zeros((Na,))
# ayo_con = np.zeros((Na,))
# axa_con[ego] = np.cos(ze[2]) - np.sin(ze[2]) * np.tan(ze[4])
# aya_con[ego] = np.sin(ze[2]) + np.cos(ze[2]) * np.tan(ze[4])
# axo_con[idx] = np.cos(zo[2]) - np.sin(zo[2]) * np.tan(zo[4])
# ayo_con[idx] = np.sin(zo[2]) + np.cos(zo[2]) * np.tan(zo[4])
# dax_con = axa_con - axo_con
# day_con = aya_con - ayo_con
#
# # x scale: designed to enforce larger safety distance in direction of travel
# x_scale = 0.5  #0.2 -- < 1 for highway scenario
# dx = x_scale * dx
# dvx = x_scale * dvx
# dax_unc = x_scale * dax_unc
# dax_con = x_scale * dax_con
#
# # Solve for minimizer (tau) of ||[dx dy] + [dvx dvy]t|| from 0 to T
# T = 3.0  # Fitting in between
# T = 2.0  # Going in front
# kh = 1000.0
# epsilon = 1e-3
# tau_star = -(dx * dvx + dy * dvy) / (dvx**2 + dvy**2 + epsilon)
# tau = tau_star * heavyside_approx(tau_star, kh, 0.0) - (tau_star - T) * heavyside_approx(tau_star, kh, T)
#
# # Derivatives of tau (controllable and uncontrollable)
# tau_star_dot_unc = -(dax_unc * (2 * dvx * tau_star + dx) + day_unc * (2 * dvy * tau_star + dy) +
#                      (dvx**2 + dvy**2)) / (dvx**2 + dvy**2 + epsilon)
# tau_star_dot_con = -(dax_con * (2 * dvx * tau_star + dx) + day_con * (2 * dvy * tau_star + dy)) / \
#                    (dvx**2 + dvy**2 + epsilon)
# tau_dot_unc = tau_star_dot_unc * (heavyside_approx(tau_star, kh, 0.0) - heavyside_approx(tau_star, kh, T)) + \
#               tau_star * tau_star_dot_unc * (dheavyside_approx(tau_star, kh, 0.0) -
#                                              dheavyside_approx(tau_star, kh, T))
# tau_dot_con = tau_star_dot_con * (heavyside_approx(tau_star, kh, 0.0) - heavyside_approx(tau_star, kh, T)) + \
#               tau_star * tau_star_dot_con * (dheavyside_approx(tau_star, kh, 0.0) -
#                                              dheavyside_approx(tau_star, kh, T))
#
# # CBF Definitions
# h0 = dx**2 + dy**2 - (2*R)**2
# ht = h0 + tau**2 * (dvx**2 + dvy**2) + 2 * tau * (dx * dvx + dy * dvy) - discretization_error

# # CBF Derivatives (Lie derivative notation -- Lfh = dh/dx * f(x))
#         Lfh0 = 2 * (dx * dvx + dy * dvy)
#         Lfht = Lfh0 + 2 * tau * (dvx**2 + dvy**2 + dx * dax_unc + dy * day_unc) + 2 * tau_dot_unc * \
#                (dx * dvx + dy * dvy + tau * (dvx**2 + dvy**2)) + 2 * tau**2 * (dvx * dax_unc + dvy * day_unc)
#         Lght = 2 * tau * tau_dot_con * (dvx**2 + dvy**2) + 2 * tau**2 * (dvx * dax_con + dvy * day_con) + \
#                2 * tau_dot_con * (dx * dvx + dy * dvy) + 2 * tau * (dx * dax_con + dy * day_con)
#