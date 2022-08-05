import numpy as np
from random import gauss
from nptyping import NDArray
from scipy.signal import resample
from scipy.linalg import block_diag
from solve_cvxopt import solve_qp_cvxopt
from helpers.gaussian_pdf import kde_pdf, gaussian_pdf
from .datastore.highway.ngsim_highway_data import ngsim_highway_data, FT_TO_M
from .nominal_controllers import merging_controller_lqr as compute_nominal_control
from .settings_highway_merging.dynamics_stochastic import f, g
from .settings_highway_merging.physical_params import Lr, R, LW
from .settings_highway_merging.control_params import objective_accel_only, objective_accel_and_steering
from .neural_network import nn, dnndx


RISK = {'CBF': 0, 'Expectation': 1, 'CVaR': 2}


###############################################################################
################################## Functions ##################################
###############################################################################


def compute_control(t: float,
                    z: NDArray,
                    extras: dict) -> (NDArray, NDArray, int, str, float):
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
    cbf: min cbf value

    """
    # Error checking variables
    code = 0
    status = ""
    do_not_consider = -1  # Considers all agents if it is set to nAgents
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
    ego = 0
    u_nom = np.zeros((len(z)+subtract_agents, 2))
    omega, ar = compute_nominal_control(t, z[ego], ego)
    u_nom[ego, :] = np.array([omega, ar])

    # Get matrices and vectors for QP controller
    Q, p, A, b, G, h, cbf = get_constraints_accel_only(t, za, zo, u_nom, agent)

    # Solve QP
    sol = solve_qp_cvxopt(Q, p, A, b, G, h)

    # If accel-only QP is infeasible, add steering control as decision variable
    if not sol['code']:
        # Get matrices and vectors for QP controller
        Q, p, A, b, G, h, cbf = get_constraints_accel_and_steering(t, za, zo, u_nom, agent)

        # Solve QP
        sol = solve_qp_cvxopt(Q, p, A, b, G, h)

        # Return error if this is also infeasible
        if not sol['code']:
            return np.zeros((2,)), u_nom[agent, :], sol['code'], sol['status'], np.zeros((3, 2))

        # Format control solution -- accel and steering solution
        u_act = np.array(sol['x'][2 * agent: 2 * (agent + 1)]).flatten()

    else:
        # Format control solution -- nominal steering, accel solution
        u_act = np.array([u_nom[agent, 0], sol['x'][agent]])

    u_act = np.clip(u_act, [-np.pi / 4, -9.81], [np.pi / 4, 9.81])
    u_0 = u_nom[agent, :]

    return u_act, u_0, sol['code'], sol['status'], cbf


############################# Safe Control Inputs #############################


def get_constraints_accel_only(t: float,
                               za: NDArray,
                               zo: NDArray,
                               u_nom: NDArray,
                               agent: int) -> (NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, float):
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
    cbf: cbf value

    """
    # Parameters
    Na = 1 + len(zo)
    discretization_error = 0.5
    risk_idx = RISK['CVaR']
    cbf = np.zeros((zo.shape[0], 2))

    # Objective function
    Q, p = objective_accel_only(u_nom[:, 1], agent)

    # Input constraints (accel only)
    au = np.array([1, -1])
    Au = block_diag(*Na*[au]).T
    bu = np.array(2*Na*[9.81])

    # Initialize inequality constraints
    Ai = np.zeros((2*len(zo), Na))
    bi = np.zeros((2*len(zo),))

    for ii, zz in enumerate(zo):
        idx = ii + (ii >= agent)

        # omega -- assume zero for now
        omega_a = u_nom[agent, 0]
        omega_z = u_nom[idx, 0]

        # x and y differentials
        dx = za[0] - zz[0]
        dy = za[1] - zz[1]

        # vx and vy differentials
        dvx = f(za)[0] - f(zz)[0]
        dvy = f(za)[1] - f(zz)[1]

        # ax and ay differentials (uncontrolled)
        axa_unc = -za[3] / Lr * np.tan(za[4]) * f(za)[1] - omega_a * za[3] * np.sin(za[2]) / np.cos(za[4]) ** 2
        aya_unc = za[3] / Lr * np.tan(za[4]) * f(za)[0] + omega_a * za[3] * np.cos(za[2]) / np.cos(za[4]) ** 2
        axo_unc = -zz[3] / Lr * np.tan(zz[4]) * f(zz)[1] - omega_z * zz[3] * np.sin(zz[2]) / np.cos(zz[4]) ** 2
        ayo_unc = zz[3] / Lr * np.tan(zz[4]) * f(zz)[0] + omega_z * zz[3] * np.cos(zz[2]) / np.cos(zz[4]) ** 2
        dax_unc = axa_unc - axo_unc
        day_unc = aya_unc - ayo_unc

        # ax and ay differentials (controlled)
        axa_con = np.zeros((Na,))
        aya_con = np.zeros((Na,))
        axo_con = np.zeros((Na,))
        ayo_con = np.zeros((Na,))
        axa_con[agent] = np.cos(za[2]) - np.sin(za[2]) * np.tan(za[4])
        aya_con[agent] = np.sin(za[2]) + np.cos(za[2]) * np.tan(za[4])
        axo_con[idx] = np.cos(zz[2]) - np.sin(zz[2]) * np.tan(zz[4])
        ayo_con[idx] = np.sin(zz[2]) + np.cos(zz[2]) * np.tan(zz[4])
        dax_con = axa_con - axo_con
        day_con = aya_con - ayo_con

        # x scale: designed to enforce larger safety distance in direction of travel
        x_scale = 0.5  # < 1 for highway scenario
        dx = x_scale * dx
        dvx = x_scale * dvx
        dax_unc = x_scale * dax_unc
        dax_con = x_scale * dax_con

        # Solve for minimizer (tau) of ||[dx dy] + [dvx dvy]t|| from 0 to T
        T = 3.0
        kh = 1000.0
        epsilon = 1e-3
        tau_star = -(dx * dvx + dy * dvy) / (dvx ** 2 + dvy ** 2 + epsilon)
        tau = tau_star * heavyside_approx(tau_star, kh, 0.0) - (tau_star - T) * heavyside_approx(tau_star, kh, T)

        # Compute Neural Network Input
        input_nn = np.concatenate([za, zz, np.array([tau])])
        input_nn[5:7] = input_nn[5:7] - input_nn[0:2]  # Normalize xy position around ego vehicle
        input_nn[0:2] = np.zeros((2,))                # Set ego xy to (0,0)
        input_nn = np.expand_dims(input_nn, axis=0)

        # NN-CBF and Derivatives (Lie derivative notation -- Lfh = dh/dx * f(x))
        h = nn(input_nn, risk_idx)
        Lfh = dnndx(input_nn, risk_idx)[0:len(za)] @ f(za)
        Lgh = np.zeros((Na,))
        Lgh[agent] = dnndx(input_nn, risk_idx)[0:len(za)] @ g(za)[:, 1]
        Lgh[idx] = dnndx(input_nn, risk_idx)[len(za):-1] @ g(zz)[:, 1]

        # Nominal CBF and Derivatives
        h0 = dx ** 2 + dy ** 2 - (2 * R) ** 2
        Lfh0 = 2 * dx * dvx + 2 * dy * dvy
        L2fh0 = 2 * dvx**2 + 2 * dvy**2 + 2 * dx * dax_unc + 2 * dy * day_unc
        LfLgh0 = 2 * dx * dax_con + 2 * dy * day_con

        # NN-CBF Condition: Lfht + Lght + l0 * ht >= 0 --> Au <= b
        l0 = 3.0
        Ai[2 * ii, :] = -Lgh
        bi[2 * ii] = Lfh + l0 * h - discretization_error

        l0 = 100 * l0
        l1 = np.sqrt(6 * l0)
        Ai[2 * ii + 1, :] = -LfLgh0
        bi[2 * ii + 1] = L2fh0 + l1 * Lfh0 + l0 * h0 - discretization_error

        if h0 < 0:
            print("SAFETY VIOLATION: {:.2f}".format(-h0))

        cbf[ii, :] = np.array([h, h0])

    A = np.vstack([Au, Ai])
    b = np.hstack([bu, bi])

    return Q, p, A, b, None, None, cbf


def get_constraints_accel_and_steering(t: float,
                                       za: NDArray,
                                       zo: NDArray,
                                       u_nom: NDArray,
                                       agent: int) -> (NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, float):
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
    cbf: cbf val

    """
    # Parameters
    Na = 1 + len(zo)
    Nu = 2
    discretization_error = 0.5
    risk_idx = RISK['CVaR']
    cbf = np.zeros((zo.shape[0], 2))

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

        # x scale: designed to enforce larger safety distance in direction of travel
        x_scale = 0.5  # < 1 for highway scenario
        dx = x_scale * dx
        dvx = x_scale * dvx

        # Solve for minimizer (tau) of ||[dx dy] + [dvx dvy]t|| from 0 to T
        T = 3.0
        kh = 1000.0
        epsilon = 1e-3
        tau_star = -(dx * dvx + dy * dvy) / (dvx ** 2 + dvy ** 2 + epsilon)
        tau = tau_star * heavyside_approx(tau_star, kh, 0.0) - (tau_star - T) * heavyside_approx(tau_star, kh, T)

        # Compute Neural Network Input
        input_nn = np.concatenate([za, zz, np.array([tau])])
        input_nn[5:7] = input_nn[5:7] - input_nn[0:2]  # Normalize xy position around ego vehicle
        input_nn[0:2] = np.zeros((2,))  # Set ego xy to (0,0)
        input_nn = np.expand_dims(input_nn, axis=0)
        # input_nn = np.array([za[0], za[1], f(za)[0], f(za)[1], zz[0], zz[1], f(zz)[0], f(zz)[1], tau])

        # CBF Derivatives (Lie derivative notation -- Lfh = dh/dx * f(x))
        h0 = dx ** 2 + dy ** 2 - (2 * R) ** 2
        h = nn(input_nn, risk_idx)
        Lfh = dnndx(input_nn, risk_idx)[0:len(za)] @ f(za)
        Lgh = np.zeros((Na*Nu,))
        Lgh[Nu*agent:Nu*(agent+1)] = dnndx(input_nn, risk_idx)[0:len(za)] @ g(za)
        Lgh[Nu*idx:Nu*(idx+1)] = dnndx(input_nn, risk_idx)[len(za):-1] @ g(zz)

        # CBF Condition: Lfht + Lght + l0 * ht >= 0 --> Au <= b
        l0 = 3.0
        Ai[ii, :] = -Lgh
        bi[ii] = Lfh + l0 * h - discretization_error

        if h0 < 0:
            print("SAFETY VIOLATION: {:.2f}".format(-h0))

        cbf[ii, :] = np.array([h, h0])

    A = np.vstack([Au, Ai])
    b = np.hstack([bu, bi])

    return Q, p, A, b, None, None, cbf


def get_constraints_accel_only_old(t: float,
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
    discretization_error = 0.5

    # Objective function
    Q, p = objective_accel_only(u_nom[:, 1], agent)

    # Input constraints (accel only)
    au = np.array([1, -1])
    Au = block_diag(*Na*[au]).T
    bu = np.array(2*Na*[9.81])

    # Initialize inequality constraints
    Ai = np.zeros((len(zo), Na))
    bi = np.zeros((len(zo),))

    for ii, zz in enumerate(zo):
        idx = ii + (ii >= agent)

        # omega -- assume zero for now
        omega_a = u_nom[agent, 0]
        omega_z = u_nom[idx, 0]

        # x and y differentials
        dx = za[0] - zz[0]
        dy = za[1] - zz[1]

        # vx and vy differentials
        dvx = f(za)[0] - f(zz)[0]
        dvy = f(za)[1] - f(zz)[1]

        # ax and ay differentials (uncontrolled)
        axa_unc = -za[3] / Lr * np.tan(za[4]) * f(za)[1] - omega_a * za[3] * np.sin(za[2]) / np.cos(za[4])**2
        aya_unc = za[3] / Lr * np.tan(za[4]) * f(za)[0] + omega_a * za[3] * np.cos(za[2]) / np.cos(za[4])**2
        axo_unc = -zz[3] / Lr * np.tan(zz[4]) * f(zz)[1] - omega_z * zz[3] * np.sin(zz[2]) / np.cos(zz[4])**2
        ayo_unc = zz[3] / Lr * np.tan(zz[4]) * f(zz)[0] + omega_z * zz[3] * np.cos(zz[2]) / np.cos(zz[4])**2
        dax_unc = axa_unc - axo_unc
        day_unc = aya_unc - ayo_unc

        # ax and ay differentials (controlled)
        axa_con = np.zeros((Na,))
        aya_con = np.zeros((Na,))
        axo_con = np.zeros((Na,))
        ayo_con = np.zeros((Na,))
        axa_con[agent] = np.cos(za[2]) - np.sin(za[2]) * np.tan(za[4])
        aya_con[agent] = np.sin(za[2]) + np.cos(za[2]) * np.tan(za[4])
        axo_con[idx] = np.cos(zz[2]) - np.sin(zz[2]) * np.tan(zz[4])
        ayo_con[idx] = np.sin(zz[2]) + np.cos(zz[2]) * np.tan(zz[4])
        dax_con = axa_con - axo_con
        day_con = aya_con - ayo_con

        # x scale: designed to enforce larger safety distance in direction of travel
        x_scale = 1  #0.2 -- < 1 for highway scenario
        dx = x_scale * dx
        dvx = x_scale * dvx
        dax_unc = x_scale * dax_unc
        dax_con = x_scale * dax_con

        # Build da's
        dax = [dax_unc, dax_con]
        day = [day_unc, day_con]

        # CVaR CBF
        h0 = dx**2 + dy**2 - (2*R)**2
        ht, tau, tau_dot_unc, tau_dot_con = get_risk_cbf(dx, dy, dvx, dvy, dax, day, za[3], 'Expectation')

        # CBF Derivatives (Lie derivative notation -- Lfh = dh/dx * f(x))
        Lfh0 = 2 * (dx * dvx + dy * dvy)
        Lfht = Lfh0 + 2 * tau * (dvx**2 + dvy**2 + dx * dax_unc + dy * day_unc) + 2 * tau_dot_unc * \
               (dx * dvx + dy * dvy + tau * (dvx**2 + dvy**2)) + 2 * tau**2 * (dvx * dax_unc + dvy * day_unc)
        Lght = 2 * tau * tau_dot_con * (dvx**2 + dvy**2) + 2 * tau**2 * (dvx * dax_con + dvy * day_con) + \
               2 * tau_dot_con * (dx * dvx + dy * dvy) + 2 * tau * (dx * dax_con + dy * day_con)

        # CBF Condition: Lfht + Lght + l0 * ht >= 0 --> Au <= b
        l0 = 20.0
        Ai[ii, :] = -Lght
        bi[ii] = Lfht + l0 * ht

        if h0 < 0:
            print("SAFETY VIOLATION: {:.2f}".format(-h0))

    A = np.vstack([Au, Ai])
    b = np.hstack([bu, bi])

    return Q, p, A, b, None, None


def get_risk_cbf(dx: float,
                 dy: float,
                 dvx: float,
                 dvy: float,
                 dax: list,
                 day: list,
                 vel: float,
                 risk: str) -> (float, float, float, NDArray):
    """ Builds pdf and returns the Conditional Value-at-Risk CBF.

    INPUTS
    ------
    dx: differential x position
    dy: differential y  position
    dvx: differential x velocity
    dvy: differential y velocity
    dax: list consisting of dax_uncontrolled and dax_controlled
    day: list consisting of day_uncontrolled and day_controlled
    vel: velocity of other vehicle
    risk: risk metric to be considered for CBF

    OUTPUTS
    -------
    cvarCBF: conditional value-at-risk cbf

    """
    # Unpack dax, day
    dax_unc = dax[0]
    dax_con = dax[1]
    day_unc = day[0]
    day_con = day[0]

    # Solve for minimizer (tau) of ||[dx dy] + [dvx dvy]t|| from 0 to T
    T = 5.0
    kh = 1000.0
    epsilon = 1e-3
    tau_star = -(dx * dvx + dy * dvy) / (dvx ** 2 + dvy ** 2 + epsilon)
    tau = tau_star * heavyside_approx(tau_star, kh, 0.0) - (tau_star - T) * heavyside_approx(tau_star, kh, T)

    # # Get n samples from ht -- assumed Gaussian
    # n = 50
    # mean_diff_pos = np.array([dx, dy]) + np.array([dvx, dvy]) * tau
    # covariance = np.array([[1.0, 0.0], [0.0, 1.0]])
    # ffcbf_samples = []
    # silvermans = 3  # ((4 * covariance[0, 0]) / (3 * n))**(1 / 5) # Not really Silverman's in current implementation
    # for ii in range(n):
    #     sample_x = gauss(mean_diff_pos[0], covariance[0,0])
    #     sample_y = gauss(mean_diff_pos[1], covariance[1,1])
    #     ffcbf_samples.append(sample_x**2 + sample_y**2 - (2*R)**2)

    # Get all available samples from data within 1 sec of computed tau value

    # Get relevant tau samples
    indices_t1 = np.where(ngsim_highway_data[:, 1, 4] < tau + 0.5)[0]
    indices_t2 = np.where(ngsim_highway_data[:, 1, 4] > tau - 0.5)[0]
    indices_t = np.intersect1d(indices_t1, indices_t2)

    # Get relevant velocity samples
    indices_v1 = np.where(ngsim_highway_data[:, 0, 5] < vel + 2.5)[0]
    indices_v2 = np.where(ngsim_highway_data[:, 0, 5] > vel - 2.5)[0]
    indices_v = np.intersect1d(indices_v1, indices_v2)

    # Synthesize all relevant samples
    indices = np.intersect1d(indices_t, indices_v)

    nSamples = 100
    dx = ngsim_highway_data[indices, 1, 2] * FT_TO_M
    dy = ngsim_highway_data[indices, 1, 3] * FT_TO_M
    ffcbf_samples = dx ** 2 + dy ** 2 - (2*R)**2
    ffcbf_samples = resample(ffcbf_samples, nSamples)
    ffcbf_samples.sort()
    ds = np.ediff1d(ffcbf_samples)  # (np.max(ffcbf_samples) - np.min(ffcbf_samples))

    # Get ht pdf values and stats
    confidence = np.max([1 - tau / T, 0.0])  # Adaptive confidence interval: relax as tau increases
    bandwidth = ((4 * np.std(ffcbf_samples)**5) / (3 * nSamples))**(1 / 5)  # Silverman's rule of thumb
    ht_pdf = kde_pdf(data=ffcbf_samples, kernel_func=gaussian_pdf, bandwidth=bandwidth)
    cum_sum = np.cumsum([ds[ee] * ht_pdf(sample) for ee, sample in enumerate(ffcbf_samples[:-1])])
    normalized_cdf = cum_sum / np.max(cum_sum)
    var = ffcbf_samples[np.max(np.where(normalized_cdf < (1 - confidence)))]  # Value-at-Risk

    # Risk Metrics
    var_idx = np.max(np.where(np.array(ffcbf_samples) < var))
    expectation_normalization = np.sum([ht_pdf(sample) for sample in ffcbf_samples])
    expected_value = sum([sample * ht_pdf(sample) for sample in ffcbf_samples]) / expectation_normalization
    cvar_normalization = np.sum([ht_pdf(sample) for sample in ffcbf_samples[:var_idx]])
    cvar = sum([sample * ht_pdf(sample) for sample in ffcbf_samples[:var_idx]]) / cvar_normalization

    # Get Expected Value
    if risk.lower() == 'expectation':
        ht = expected_value
    elif risk.lower() == 'cvar':
        ht = cvar
    else:
        ht = 0

    print("Expectation: {:.2f}".format(expected_value))
    print("CVaR: {:.2f}".format(cvar))

    # # In-line debugging
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    #
    # sns.set(color_codes=True)
    # plt.rcParams["figure.figsize"] = (15, 10)
    #
    # fig = plt.figure()
    #
    # ax1 = fig.add_subplot(1, 1, 1)
    # y_exp = [ht_pdf(i) for i in ffcbf_samples]
    # y_cvar = [ht_pdf(i) for i in ffcbf_samples[:var_idx+1]]
    # ax1.scatter(ffcbf_samples[var_idx], y_exp[var_idx])
    # ax1.plot(ffcbf_samples, y_exp)
    # ax1.plot(ffcbf_samples[:var_idx+1], y_cvar)
    # plt.tight_layout()
    # plt.show()

    # Derivatives of tau (controllable and uncontrollable)
    tau_star_dot_unc = -(dax_unc * (2 * dvx * tau_star + dx) + day_unc * (2 * dvy * tau_star + dy) +
                         (dvx ** 2 + dvy ** 2)) / (dvx ** 2 + dvy ** 2 + epsilon)
    tau_star_dot_con = -(dax_con * (2 * dvx * tau_star + dx) + day_con * (2 * dvy * tau_star + dy)) / \
                       (dvx ** 2 + dvy ** 2 + epsilon)
    tau_dot_unc = tau_star_dot_unc * (heavyside_approx(tau_star, kh, 0.0) - heavyside_approx(tau_star, kh, T)) + \
                  tau_star * tau_star_dot_unc * (dheavyside_approx(tau_star, kh, 0.0) -
                                                 dheavyside_approx(tau_star, kh, T))
    tau_dot_con = tau_star_dot_con * (heavyside_approx(tau_star, kh, 0.0) - heavyside_approx(tau_star, kh, T)) + \
                  tau_star * tau_star_dot_con * (dheavyside_approx(tau_star, kh, 0.0) -
                                                 dheavyside_approx(tau_star, kh, T))

    return ht, tau, tau_dot_unc, tau_dot_con


#
#     # CBF Definitions
#     h0 = dx ** 2 + dy ** 2 - (2 * R) ** 2
#     ht = h0 + tau ** 2 * (dvx ** 2 + dvy ** 2) + 2 * tau * (dx * dvx + dy * dvy) - discretization_error
#
#     return cvarCBF


def get_constraints_accel_and_steering_old(t: float,
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
        x_scale = 1  #0.2 -- < 1 for highway scenario
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
        l0 = 20.0
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
