import numpy as np
from nptyping import NDArray
from helpers.cubic_equation import solve_cubic
from helpers.quadratic_equation import solve_quadratic
from scipy.linalg import block_diag
from scipy.signal import place_poles
from solve_cvxopt import solve_qp_cvxopt
from .nominal_controllers import jerk_intersection_controller_lqr as compute_nominal_control
from bicycle.settings.intersection.jerk_control import f, x_accel, y_accel, x_jerk_uncontrolled, y_jerk_uncontrolled, \
    x_jerk_controlled, y_jerk_controlled
from bicycle.settings.intersection.jerk_control import Lr, R, jr_max, alpha_max
from bicycle.settings.intersection.jerk_control import objective_accel_only, objective_accel_and_steering

###############################################################################
################################## Functions ##################################
###############################################################################


def compute_control(t: float,
                    z: NDArray,
                    extras: dict) -> (NDArray, NDArray, int, str):
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
    do_not_consider = z.shape[0]  # -1  # Considers all agents if it is set to nAgents
    subtract_agents = do_not_consider if do_not_consider != z.shape[0] else 0

    # Unpack extras
    agent = extras['agent']
    za = z[agent, :]
    zo = np.vstack([z[:agent, :], z[agent+1:do_not_consider, :]])

    # # Compute nominal control inputs
    # u_nom = np.zeros((len(z), 2))
    # for aa, zz in enumerate(z):
    #     omega, ar = compute_nominal_control(t, zz, aa)
    #     u_nom[aa, :] = np.array([omega, ar])

    # Compute nominal control input for ego only -- assume others are zero
    ego = agent
    u_nom = np.zeros((len(z)+subtract_agents, 2))
    jr, alpha = compute_nominal_control(t, z[ego], ego)
    u_nom[ego, :] = np.array([jr, alpha])

    # Get matrices and vectors for QP controller
    Q, p, A, b, G, h = get_constraints_jerk_only(t, za, zo, u_nom, agent)

    # Solve QP
    sol = solve_qp_cvxopt(Q, p, A, b, G, h)

    # If accel-only QP is infeasible, add steering control as decision variable
    if not sol['code']:
        print('down a level: {}'.format(t))
        # Get matrices and vectors for QP controller
        Q, p, A, b, G, h = get_constraints_accel_and_steering(t, za, zo, u_nom, agent)

        # Solve QP
        sol = solve_qp_cvxopt(Q, p, A, b, G, h)

        # Return error if this is also infeasible
        if not sol['code']:
            return np.zeros((2,)), u_nom[agent, :], sol['code'], sol['status']

        # Format control solution -- accel and steering solution
        u_act = np.array(sol['x'][2 * agent: 2 * (agent + 1)]).flatten()

    else:
        # Format control solution -- nominal steering, accel solution
        u_act = np.array([sol['x'][agent], u_nom[agent, 1]])

    u_act = np.clip(u_act, [-jr_max, -alpha_max], [jr_max, alpha_max])
    u_0 = u_nom[agent, :]

    return u_act, u_0, sol['code'], sol['status']


############################# Safe Control Inputs #############################


def get_constraints_jerk_only(t: float,
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
    discretization_error = 1.0

    # Objective function
    Q, p = objective_accel_only(u_nom[:, 0], agent)

    # Input constraints (jerk only)
    au = np.array([1, -1])
    Au = block_diag(*Na*[au]).T
    bu = np.array(2*Na*[jr_max])

    # Initialize inequality constraints
    Ai = np.zeros((len(zo), Na))
    bi = np.zeros((len(zo),))

    for ii, zz in enumerate(zo):
        idx = ii + (ii >= agent)

        # alpha
        alpha_a = u_nom[agent, 1]
        alpha_z = u_nom[idx, 1]

        # x and y differentials
        dx = za[0] - zz[0]
        dy = za[1] - zz[1]

        # vx and vy differentials
        dvx = f(za)[0] - f(zz)[0]
        dvy = f(za)[1] - f(zz)[1]

        # ax and ay differentials
        dax = x_accel(za) - x_accel(zz)
        day = y_accel(za) - y_accel(zz)

        # jx and jy differentials
        djx_con = np.zeros((Na,))
        djy_con = np.zeros((Na,))
        djx_unc = (x_jerk_uncontrolled(za) + alpha_a * x_jerk_controlled(za)[1]) - \
                  (x_jerk_uncontrolled(zz) + alpha_z * x_jerk_controlled(zz)[1])
        djy_unc = (y_jerk_uncontrolled(za) + alpha_a * y_jerk_controlled(za)[1]) - \
                  (y_jerk_uncontrolled(zz) + alpha_z * y_jerk_controlled(zz)[1])
        np.put(djx_con, [agent, idx], [x_jerk_controlled(za)[0], x_jerk_controlled(zz)[0]])
        np.put(djy_con, [agent, idx], [y_jerk_controlled(za)[0], y_jerk_controlled(zz)[0]])

        # Inter-agent distance Exponential CBF
        h = dx**2 + dy**2 - (2*R)**2

        # CBF Derivative Terms for condition:
        Lfh = 2 * (dx * dvx + dy * dvy)
        Lf2h = 2 * (dvx**2 + dvy**2) + (dx * dax + dy * day)
        Lf3h = 2 * (dx * djx_unc + 3 * dvx * dax + dy * djy_unc + 3 * dvy * day)
        LgLf2h = 2 * (dx * djx_con + dy * djy_con)

        # CBF Condition: Lf3h + LgLf2h*u + K^T * [Lf2h + Lfh + h] >= 0 --> Au <= b
        l0 = 1.0  # Going in front
        A_pp = np.diag(np.ones(2), 1)
        B_pp = np.array([[0], [0], [1]])
        BO = place_poles(A_pp, B_pp, [-1.0, -2.0, -0.05])
        K = np.flip(BO.gain_matrix)
        P = np.array([h, Lfh, Lf2h])
        Ai[ii, :] = -LgLf2h
        bi[ii] = Lf3h + np.dot(K, P)

        if h < 0:
            print("SAFETY VIOLATION: {:.2f}".format(-h))

        # if h < 2:
        #     print('debug')

    A = np.vstack([Au, Ai])
    b = np.hstack([bu, bi])

    return Q, p, A, b, None, None


def get_ff_constraints_jerk_only(t: float,
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
    discretization_error = 1.0

    # Objective function
    Q, p = objective_accel_only(u_nom[:, 0], agent)

    # Input constraints (jerk only)
    au = np.array([1, -1])
    Au = block_diag(*Na*[au]).T
    bu = np.array(2*Na*[jr_max])

    # Initialize inequality constraints
    Ai = np.zeros((len(zo), Na))
    bi = np.zeros((len(zo),))

    for ii, zz in enumerate(zo):
        idx = ii + (ii >= agent)

        # alpha
        alpha_a = u_nom[agent, 1]
        alpha_z = u_nom[idx, 1]

        # x and y differentials
        dx = za[0] - zz[0]
        dy = za[1] - zz[1]

        # vx and vy differentials
        dvx = f(za)[0] - f(zz)[0]
        dvy = f(za)[1] - f(zz)[1]

        # ax and ay differentials
        dax = x_accel(za) - x_accel(zz)
        day = y_accel(za) - y_accel(zz)

        # jx and jy differentials
        djx_con = np.zeros((Na,))
        djy_con = np.zeros((Na,))
        djx_unc = (x_jerk_uncontrolled(za) + alpha_a * x_jerk_controlled(za)[1]) - \
                  (x_jerk_uncontrolled(zz) + alpha_z * x_jerk_controlled(zz)[1])
        djy_unc = (y_jerk_uncontrolled(za) + alpha_a * y_jerk_controlled(za)[1]) - \
                  (y_jerk_uncontrolled(zz) + alpha_z * y_jerk_controlled(zz)[1])
        np.put(djx_con, [agent, idx], [x_jerk_controlled(za)[0], x_jerk_controlled(zz)[0]])
        np.put(djy_con, [agent, idx], [y_jerk_controlled(za)[0], y_jerk_controlled(zz)[0]])

        # Compute tau and derivatives
        tau, tau_dot_unc, tau_dot_con = compute_tau_2nd_order(dx, dy, dvx, dvy, dax, day, djx_unc, djx_con, djy_unc, djy_con)

        # Future-focused (Jerk) Inter-agent distance CBF
        h0 = dx**2 + dy**2 - (2*R)**2
        h = (dx + dvx * tau + 1/2 * dax * tau**2)**2 + (dy + dvy * tau + 1/2 * day * tau**2)**2 - (2*R)**2
        Lfh = 1/2 * (dax * djx_unc + day * djy_unc) * tau**4 + (dax**2 + day**2) * tau**3 * tau_dot_unc + \
              (dax**2 + day**2 + dvx * djx_unc + dvy * djy_unc) * tau**3 + 3 * (dvx * dax + dvy * day) * tau**2 * \
              tau_dot_unc + (dx * djx_unc + dy * djy_unc + 3 * dvx * dax + 3 * dvy * day) * tau**2 + \
              2 * (dx * dax + dy * day + dvx**2 + dvy**2) * tau * tau_dot_unc + \
              2 * (dvx**2 + dvy**2 + dx * dax + dy * day) * tau + 2 * (dx * dvx + dy * dvy) * tau_dot_unc + \
              2 * (dx * dvx + dy * dvy)
        Lgh = 1 / 2 * (dax * djx_con + day * djy_con) * tau ** 4 + (dax ** 2 + day ** 2) * tau ** 3 * tau_dot_con + \
              (dvx * djx_con + dvy * djy_con) * tau ** 3 + 3 * (dvx * dax + dvy * day) * tau ** 2 * tau_dot_con + \
              (dx * djx_con + dy * djy_con) * tau ** 2 + 2 * (dx * dax + dy * day + dvx ** 2 + dvy ** 2) * tau * \
              tau_dot_con + 2 * (dx * dvx + dy * dvy) * tau_dot_con

        # CBF Condition: Lfh + Lgh*u + l0 * h >= 0 --> Au <= b
        l0 = 0.5
        Ai[ii, :] = -Lgh
        bi[ii] = Lfh + l0 * h

        if h0 < 0:
            print("SAFETY VIOLATION: {:.2f}".format(-h0))

        # if h < 2:
        #     print('debug')

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


def compute_tau_2nd_order(dx: float,
                          dy: float,
                          dvx: float,
                          dvy: float,
                          dax: float,
                          day: float,
                          djx_unc: float,
                          djx_con: NDArray,
                          djy_unc: float,
                          djy_con: NDArray) -> (float, float, NDArray):
    """Computes the predicted time to minimum distance according to a second-order linear model: constant acceleration
    and zero jerk.

    INPUTS
    ------
    dx: differential position in x coordinate (in m)
    dy: differential position in y coordinate (in m)
    dvx: differential velocity in x direction (in m/s)
    dvy: differential velocity in y direction (in m/s)
    dax: differential acceleration in x direction (in m/s^2)
    day: differential acceleration in y direction (in m/s^2)
    djx_unc: differential jerk in x direction uncontrolled (in m/s^3)
    djx_unc: differential jerk in x direction controlled (in m/s^3)
    djy_unc: differential jerk in y direction uncontrolled (in m/s^3)
    djx_unc: differential jerk in y direction controlled (in m/s^3)

    OUTPUTS
    -------
    tau: time to minimum (predicted) future distance (in s)

    """
    # Need to compute minimum non-zero solution to cubic equation: ax^3 + bx^2 + cx + d == 0
    a = dax**2 + day**2
    b = 3 * (dvx * dax + dvy * day)
    c = 2 * (dx * dax + dy * day + dvx**2 + dvy**2)
    d = 2 * (dx * dvx + dy * dvy)

    if a == 0:
        if b == 0:
            candidates = [-d / c]
        else:
            candidates = solve_quadratic(b, c, d)
    else:
        candidates = solve_cubic(a, b, c, d)

    # Choose candidate -- minimum positive value
    candidates = np.array(candidates)
    tau = candidates[np.where(candidates > 0, candidates, np.inf).argmin()]

    # # # Compute time derivative of tau
    # Partial derivatives of cubic equation (f)
    partial_f_partial_tau = 3 * a * tau**2 + 2 * b * tau + c
    partial_f_partial_a = tau**3
    partial_f_partial_b = tau**2
    partial_f_partial_c = tau
    partial_f_partial_d = 0

    # Time-derivatives of coefficients
    a_dot_unc = 2 * (dax * djx_unc + day * djy_unc)
    a_dot_con = 2 * (dax * djx_con + day * djy_con)
    b_dot_unc = 3 * (dax**2 + day**2 + dvx * djx_unc + dvy * djy_unc)
    b_dot_con = 3 * (dvx * djx_con + dvy * djy_con)
    c_dot_unc = 2 * (dvx * dax + dx * djx_unc + dvy * day + dy * djy_unc + 2 * dvx * dax + 2 * dvy * day)
    c_dot_con = 2 * (dx * djx_con + dy * djy_con)
    d_dot_unc = 2 * (dx * dax + dy * day + dvx**2 + dvy**2)
    d_dot_con = np.zeros(djx_con.shape)

    # Controlled and uncontrolled components of time-derivative of tau
    tau_dot_unc = - (partial_f_partial_a * a_dot_unc + partial_f_partial_b * b_dot_unc + partial_f_partial_c * c_dot_unc
                     + partial_f_partial_d * d_dot_unc) / partial_f_partial_tau
    tau_dot_con = - (partial_f_partial_a * a_dot_con + partial_f_partial_b * b_dot_con + partial_f_partial_c * c_dot_con
                     + partial_f_partial_d * d_dot_con) / partial_f_partial_tau

    # Solve for minimizer (tau) of ||[dx dy] + [dvx dvy]t|| from 0 to T
    T = 3.0
    kh = 1000.0
    tau_hat = tau * heavyside_approx(tau, kh, 0.0) - (tau - T) * heavyside_approx(tau, kh, T)

    # Derivatives of tau (controllable and uncontrollable)
    tau_hat_dot_unc = tau_dot_unc * (heavyside_approx(tau, kh, 0.0) - heavyside_approx(tau, kh, T)) + \
                      tau * tau_dot_unc * (dheavyside_approx(tau, kh, 0.0) - dheavyside_approx(tau, kh, T))
    tau_hat_dot_con = tau_dot_con * (heavyside_approx(tau, kh, 0.0) - heavyside_approx(tau, kh, T)) + \
                      tau * tau_dot_con * (dheavyside_approx(tau, kh, 0.0) - dheavyside_approx(tau, kh, T))

    return tau_hat, tau_hat_dot_unc, tau_hat_dot_con


