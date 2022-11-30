from xml.dom.expatbuilder import FILTER_REJECT
import numpy as np
import numdifftools as nd
import builtins
from typing import Callable, List, overload
from importlib import import_module
from nptyping import NDArray
from control import lqr
from scipy.linalg import block_diag, null_space
from .cbfs.cbf import Cbf
from core.controllers.cbf_qp_controller import CbfQpController
from core.controllers.controller import Controller
from core.solve_cvxopt import solve_qp_cvxopt

vehicle = builtins.PROBLEM_CONFIG["vehicle"]
control_level = builtins.PROBLEM_CONFIG["control_level"]
system_model = builtins.PROBLEM_CONFIG["system_model"]
mod = vehicle + "." + control_level + ".models"

# Programmatic import
try:
    module = import_module(mod)
    globals().update({"f": getattr(module, "f")})
    globals().update({"g": getattr(module, "g")})
    globals().update({"sigma": getattr(module, "sigma_{}".format(system_model))})
except ModuleNotFoundError as e:
    print("No module named '{}' -- exiting.".format(mod))
    raise e


class ConsolidatedCbfController(CbfQpController):
    """
    Class docstrings should contain the following information:

    Controller using the Consolidated CBF based approach first proposed
    in 'Adaptation for Validation of a Consolidated Control Barrier Function
    based Control Synthesis' (Black and Panagou, 2022)
    (https://arxiv.org/abs/2209.08170).

    Public Methods:
    ---------------
    formulate_qp: overloads from parent Controller class
    generate_consolidated_cbf_condition: generates matrix and vector for CBF condition
    compute_kdots: computes the adaptation of the gains k

    Class Properties:
    -----------------
    Lots?
    """

    P = None

    def __init__(
        self,
        u_max: List,
        nAgents: int,
        objective_function: Callable,
        nominal_controller: Controller,
        cbfs_individual: List,
        cbfs_pairwise: List,
        ignore: List = None,
    ):
        super().__init__(
            u_max,
            nAgents,
            objective_function,
            nominal_controller,
            cbfs_individual,
            cbfs_pairwise,
            ignore,
        )
        nCBF = len(self.cbf_vals)
        self.c_cbf = 100
        self.k_gains = 0.5 * np.ones((nCBF,))
        self.k_dots = np.zeros((nCBF,))
        self.n_agents = 1
        self.U_mat = self.u_max[:, np.newaxis] @ self.u_max[np.newaxis, :]
        self.gain_mat = np.eye(len(self.k_gains)) * 1.0
        self.kd_gain = 0.25
        self.k_min = 0.1
        self.k_max = 5.0
        self.dhdx = np.zeros((nCBF, 5))
        self.filtered_wf = np.zeros((nCBF,))
        self.filtered_wg = np.zeros((nCBF, len(self.u_max)))
        self.q_vec = np.zeros((nCBF,))
        self.dqdk = np.zeros((nCBF, nCBF))
        self.dqdx = np.zeros((5, nCBF))
        self.d2qdk2 = np.zeros((nCBF, nCBF, nCBF))
        self.d2qdkdx = np.zeros((nCBF, 5, nCBF))

        # Gains
        self.P_gain = 1.0 * np.eye(nCBF)
        self.cost_gain = 100.0
        self.czero_gain = 0.01
        self.ci_gain = 0.01
        self.kdot_gain = 0.01

    def formulate_qp(
        self, t: float, ze: NDArray, zr: NDArray, u_nom: NDArray, ego: int, cascade: bool = False
    ) -> (NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, float):
        """Configures the Quadratic Program parameters (Q, p for objective function, A, b for inequality constraints,
        G, h for equality constraints).

        """
        # Parameters
        na = 1 + len(zr)
        ns = len(ze)
        self.safety = True
        discretization_error = 1.0

        if self.nv > 0:
            alpha_nom = 1.0
            Q, p = self.objective(np.append(u_nom.flatten(), alpha_nom))
            Au = block_diag(*(na + self.nv) * [self.au])[:-2, :-1]
            bu = np.append(np.array(na * [self.bu]).flatten(), self.nv * [1e6, 0])
        else:
            Q, p = self.objective(u_nom.flatten())
            Au = block_diag(*(na) * [self.au])
            bu = np.array(na * [self.bu]).flatten()

        # Initialize inequality constraints
        lci = len(self.cbfs_individual)
        h_array = np.zeros(
            (
                len(
                    self.cbf_vals,
                )
            )
        )
        dhdx_array = np.zeros(
            (
                len(
                    self.cbf_vals,
                ),
                2 * ns,
            )
        )
        Lfh_array = np.zeros(
            (
                len(
                    self.cbf_vals,
                )
            )
        )
        Lgh_array = np.zeros((len(self.cbf_vals), u_nom.flatten().shape[0]))

        # Iterate over individual CBF constraints
        for cc, cbf in enumerate(self.cbfs_individual):
            h0 = cbf.h0(ze)
            h_array[cc] = cbf.h(ze)
            dhdx_array[cc][:ns] = cbf.dhdx(ze)

            # Stochastic Term -- 0 for deterministic systems
            if np.trace(sigma(ze).T @ sigma(ze)) > 0 and self._stochastic:
                d2hdx2 = cbf.d2hdx2(ze)
                stoch = 0.5 * np.trace(sigma(ze).T @ d2hdx2 @ sigma(ze))
            else:
                stoch = 0.0

            # Get CBF Lie Derivatives
            Lfh_array[cc] = dhdx_array[cc][:ns] @ f(ze) + stoch - discretization_error
            Lgh_array[cc, self.nu * ego : (ego + 1) * self.nu] = dhdx_array[cc][:ns] @ g(
                ze
            )  # Only assign ego control
            if cascade:
                Lgh_array[cc, self.nu * ego] = 0.0

            self.dhdx[cc] = dhdx_array[cc][:ns]
            self.cbf_vals[cc] = h_array[cc]
            if h0 < 0:
                self.safety = False

        # Iterate over pairwise CBF constraints
        for cc, cbf in enumerate(self.cbfs_pairwise):

            # Iterate over all other vehicles
            for ii, zo in enumerate(zr):
                other = ii + (ii >= ego)
                idx = lci + cc * zr.shape[0] + ii

                h0 = cbf.h0(ze, zo)
                h_array[idx] = cbf.h(ze, zo)
                dhdx_array[idx] = cbf.dhdx(ze, zo)

                # Stochastic Term -- 0 for deterministic systems
                if np.trace(sigma(ze).T @ sigma(ze)) > 0 and self._stochastic:
                    d2hdx2 = cbf.d2hdx2(ze, zo)
                    stoch = 0.5 * (
                        np.trace(sigma(ze).T @ d2hdx2[:ns, :ns] @ sigma(ze))
                        + np.trace(sigma(zo).T @ d2hdx2[ns:, ns:] @ sigma(zo))
                    )
                else:
                    stoch = 0.0

                # Get CBF Lie Derivatives
                Lfh_array[idx] = (
                    dhdx_array[idx][:ns] @ f(ze)
                    + dhdx_array[idx][ns:] @ f(zo)
                    + stoch
                    - discretization_error
                )
                Lgh_array[idx, self.nu * ego : (ego + 1) * self.nu] = dhdx_array[idx][:ns] @ g(ze)
                Lgh_array[idx, self.nu * other : (other + 1) * self.nu] = dhdx_array[idx][ns:] @ g(
                    zo
                )
                if cascade:
                    Lgh_array[idx, self.nu * ego] = 0.0

                if h0 < 0:
                    print(
                        "{} SAFETY VIOLATION: {:.2f}".format(
                            str(self.__class__).split(".")[-1], -h0
                        )
                    )
                    self.safety = False

                self.cbf_vals[idx] = h_array[idx]
                self.dhdx[idx] = dhdx_array[idx][:ns]

        # Format inequality constraints
        # Ai, bi = self.generate_consolidated_cbf_condition(ego, h_array, Lfh_array, Lgh_array)
        Ai, bi = self.generate_consolidated_cbf_condition(ze, h_array, Lfh_array, Lgh_array, ego)

        A = np.vstack([Au, Ai])
        b = np.hstack([bu, bi])

        return Q, p, A, b, None, None

    def generate_consolidated_cbf_condition(
        self, x: NDArray, h_array: NDArray, Lfh_array: NDArray, Lgh_array: NDArray, ego: int
    ) -> (NDArray, NDArray):
        """Generates the inequality constraint for the consolidated CBF.

        ARGUMENTS
            x: state vector
            h: array of candidate CBFs
            Lfh: array of CBF drift terms
            Lgh: array of CBF control matrices
            ego: ego vehicle id

        RETURNS
            kdot: array of time-derivatives of k_gains
        """
        # Introduce discretization error
        discretization_error = 1.0

        # Get C-CBF Value
        H = self.H(self.k_gains, h_array)
        self.c_cbf = H

        # Non-centralized agents CBF dynamics become drifts
        Lgh_uncontrolled = np.copy(Lgh_array[:, :])
        Lgh_uncontrolled[:, ego * self.nu : (ego + 1) * self.nu] = 0
        Lgh_array[:, np.s_[: ego * self.nu]] = 0  # All indices before first ego index set to 0
        Lgh_array[
            :, np.s_[(ego + 1) * self.nu :]
        ] = 0  # All indices after last ego index set to 0 (excluding alpha)

        # Get time-derivatives of gains
        exp_term = np.exp(-self.k_gains * h_array)
        premultiplier_k = self.k_gains * exp_term
        premultiplier_h = h_array * exp_term

        # Compute C-CBF Dynamics
        LfH = premultiplier_k @ Lfh_array
        LgH = premultiplier_k @ Lgh_array
        LgH_uncontrolled = premultiplier_k @ Lgh_uncontrolled

        # Tunable CBF Addition
        # kH = 1.0
        # kH = 0.5
        # kH = 0.05
        kH = 0.1
        phi = (
            np.tile(-np.array(self.u_max), int(LgH_uncontrolled.shape[0] / len(self.u_max)))
            @ abs(LgH_uncontrolled)
            * np.exp(-kH * H)
        )
        LfH = LfH + phi
        Lg_for_kdot = Lgh_array[:, self.nu * ego : self.nu * (ego + 1)]

        # Compute kdot
        k_dot = self.k_dot(x, self.u, self.k_gains, h_array, Lg_for_kdot)

        # k_dots = self.compute_k_dots(h_array, H, LfH, Lg_for_kdot, premultiplier_k)
        LfH = LfH + premultiplier_h @ k_dot - discretization_error

        if H < 0:
            print(f"h_array: {h_array}")
            # print(f"LfH: {LfH}")
            # print(f"LgH: {LgH}")
            # print(f"H:   {H}")
            # print(self.k_gains)
            print(H)

        # Finish constructing CBF here
        a_mat = np.append(-LgH, -H)
        b_vec = np.array([LfH])

        # Update k_dot
        self.k_gains = self.k_gains + k_dot * self._dt
        if np.sum(self.k_gains <= 0) > 0:
            print(f"kdots: {k_dot}")
            print(f"kvals: {self.k_gains}")
            print(f"kdes: {self.k_desired(h_array)}")
            print("Illegal K Value")

        return a_mat[:, np.newaxis].T, b_vec

    def compute_k_dots(
        self, h_array: NDArray, H: float, LfH: float, Lgh_array: NDArray, vec: NDArray
    ) -> NDArray:
        """Computes the time-derivatives of the k_gains via QP based adaptation law.

        ARGUMENTS
            h_array: array of CBF values
            H: scalar c-cbf value
            LfH: scalar LfH term for c-cbf
            Lgh_array: 2D array of Lgh terms
            vec: vector that needs to stay out of the nullspace of matrix P

        RETURNS
            k_dot

        """
        # Good behavior gains
        k_min = 0.1
        k_max = 1.0
        k_des = 10 * k_min * h_array
        k_des = np.clip(k_des, k_min, k_max)
        P_gain = 10.0
        constraint_gain = 10.0
        gradient_gain = 0.01

        # (3 Reached but H negative) gains
        k_min = 0.01
        k_max = 1.0
        k_des = h_array / 5
        k_des = np.clip(k_des, k_min, k_max)
        P_gain = 100.0
        constraint_gain = 1.0
        gradient_gain = 0.001

        # Testing gains
        k_min = 0.01
        k_max = 5.0
        k_des = 0.1 * h_array / np.min([np.min(h_array), 2.0])
        k_des = np.clip(k_des, k_min, k_max)
        P_gain = 1000.0
        kpositive_gain = 1000.0
        constraint_gain = 1000.0
        gradient_gain = 0.1

        # TO DO: Correct how delta is being used -- need the max over all safe set
        # Some parameters
        P_mat = P_gain * np.eye(len(h_array))
        p_vec = vec
        N_mat = null_space(Lgh_array.T)
        rank_Lg = N_mat.shape[0] - N_mat.shape[1]
        Q_mat = np.eye(len(self.k_gains)) - 2 * N_mat @ N_mat.T + N_mat @ N_mat.T @ N_mat @ N_mat.T
        _, sigma, _ = np.linalg.svd(Lgh_array.T)  # Singular values of LGH
        if rank_Lg > 0:
            sigma_r = sigma[rank_Lg - 1]  # minimum non-zero singular value
        else:
            sigma_r = (
                0  # This should not ever happen (by assumption that not all individual Lgh = 0)
            )
        delta = -(
            LfH + 0.1 * H + (h_array * np.exp(-self.k_gains * h_array)) @ self.k_dots
        ) / np.linalg.norm(self.u_max)
        delta = 1 * abs(delta)

        # Objective Function and Partial Derivatives
        J = 1 / 2 * (self.k_gains - k_des).T @ P_mat @ (self.k_gains - k_des)
        dJdk = P_mat @ (self.k_gains - k_des)
        d2Jdk2 = P_mat

        # K Constraint Functions and Partial Derivatives: Convention hi >= 0
        hi = (self.k_gains - k_min) * kpositive_gain
        dhidk = np.ones((len(self.k_gains),)) * kpositive_gain
        d2hidk2 = np.zeros((len(self.k_gains), len(self.k_gains))) * kpositive_gain

        # Partial Derivatives of p vector
        dpdk = np.diag((self.k_gains * h_array - 1) * np.exp(-self.k_gains * h_array))
        d2pdk2_vals = h_array * np.exp(-self.k_gains * h_array) + (
            self.k_gains * h_array - 1
        ) * -h_array * np.exp(-self.k_gains * h_array)
        d2pdk2 = np.zeros((len(h_array), len(h_array), len(h_array)))
        np.fill_diagonal(d2pdk2, d2pdk2_vals)

        # # LGH Norm Constraint Function and Partial Derivatives
        # # Square of norm
        # eta = (sigma_r**2 * (p_vec.T @ Q_mat @ p_vec) - delta**2) * constraint_gain
        # detadk = 2 * sigma_r**2 * (dpdk.T @ Q_mat @ p_vec) * constraint_gain
        # d2etadk2 = (2 * sigma_r**2 * (d2pdk2.T @ Q_mat @ p_vec + dpdk.T @ Q_mat @ dpdk)) * constraint_gain

        # V vec
        v_vec = (p_vec @ Lgh_array).T
        dvdk = (dpdk @ Lgh_array).T
        d2vdk2 = (d2pdk2 @ Lgh_array).T

        # New Input Constraints Condition
        eta = ((v_vec.T @ U_mat @ v_vec) - delta**2) * constraint_gain
        detadk = 2 * (dvdk.T @ U_mat @ v_vec) * constraint_gain
        d2etadk2 = (2 * (d2vdk2.T @ U_mat @ v_vec + dvdk.T @ U_mat @ dvdk)) * constraint_gain

        print(f"Eta: {eta}")

        if np.isnan(eta):
            print(eta)

        # Define the augmented cost function
        Phi = J - np.sum(np.log(hi)) - np.log(eta)
        dPhidk = dJdk - np.sum(dhidk / hi) - detadk / eta
        d2Phidk2 = (
            d2Jdk2 - np.sum(d2hidk2 / hi - dhidk / hi**2) - (d2etadk2 / eta - detadk / eta**2)
        )

        # Define Adaptation law
        corrector_term = -np.linalg.inv(d2Phidk2) @ (dPhidk)  # Correction toward minimizer
        # corrector_term = -dPhidk  # Correction toward minimizer -- gradient method only
        predictor_term = (
            0 * self.k_gains
        )  # Zero for now (need to see how this could depend on time)
        k_dots = gradient_gain * (corrector_term + predictor_term)

        if np.sum(np.isnan(k_dots)) > 0:
            print(k_dots)

        self.k_dots = k_dots

        return k_dots

    def k_dot(self, x: NDArray, u: NDArray, k: NDArray, h: NDArray, Lg: NDArray) -> NDArray:
        """Computes the time-derivative of the constituent cbf weight vector (k).

        Arguments
        ---------
        x: state vector
        u: control vector
        k: constituent cbf weighting vector
        h: array of constituent cbfs
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        k_dot

        """
        # Update vectors and gradients
        self.q_vec = k * np.exp(-k * h)
        self.dqdk = np.diag((1 - k * h) * np.exp(-k * h))
        self.dqdx = np.diag(-(k**2) * np.exp(-k * h)) @ self.dhdx
        d2qdk2_vals = (k * h**2 - 2 * h) * np.exp(-k * h)
        self.d2qdk2 = np.zeros((len(h), len(h), len(h)))
        np.fill_diagonal(self.d2qdk2, d2qdk2_vals)
        triple_diag = np.zeros((len(k), len(k), len(k)))
        np.fill_diagonal(triple_diag, (k**2 * h - 2 * k) * np.exp(-k * h))
        self.d2qdkdx = triple_diag @ self.dhdx

        wf = -np.linalg.inv(self.grad_phi_kk(x, k, h, Lg)) @ (
            self.P_gain @ self.grad_phi_k(x, k, h, Lg) + self.grad_phi_kx(x, k, h, Lg) @ f(x)
        )
        wg = -np.linalg.inv(self.grad_phi_kk(x, k, h, Lg)) @ self.grad_phi_kx(x, k, h, Lg) @ g(x)

        m = 1.0
        self.filtered_wf = self.filtered_wf + (wf - self.filtered_wf) / m * self._dt
        self.filtered_wg = self.filtered_wg + (wg - self.filtered_wg) / m * self._dt

        if u is None:
            u = np.zeros((self.nu,))

        k_dot = wf + wg @ u

        return k_dot * self.kdot_gain

    def H(self, k: NDArray, h: NDArray) -> float:
        """Computes the consolidated control barrier function (C-CBF) based on
        the vector of constituent CBFs (h) and their corresponding weights (k).

        Arguments
        ---------
        k: constituent cbf weighting vector
        h: array of constituent cbfs

        Returns
        -------
        H: consolidated control barrier function evaluated at k and h(x)

        """
        H = 1 - np.sum(np.exp(-k * h))

        return H

    def grad_H_k(self, k: NDArray, h: NDArray) -> float:
        """Computes the gradient of the consolidated control barrier function (C-CBF)
        with respect to the weights (k).

        Arguments
        ---------
        k: constituent cbf weighting vector
        h: array of constituent cbfs

        Returns
        -------
        grad_H_k: gradient of C-CBF with respect to k

        """
        grad_H_k = h * np.exp(-k * h)

        return grad_H_k

    def grad_H_x(self, x: NDArray, k: NDArray, h: NDArray) -> float:
        """Computes the gradient of the consolidated control barrier function (C-CBF)
        with respect to the state (x).

        Arguments
        ---------
        x: state vector
        k: constituent cbf weighting vector
        h: array of constituent cbfs

        Returns
        -------
        grad_H_x: gradient of C-CBF with respect to x

        """
        grad_H_x = self.q_vec @ self.dhdx

        return grad_H_x

    def grad_H_kk(self, k: NDArray, h: NDArray) -> float:
        """Computes the gradient of the consolidated control barrier function (C-CBF)
        with respect to the weights (k) twice.

        Arguments
        ---------
        k: constituent cbf weighting vector
        h: array of constituent cbfs

        Returns
        -------
        grad_H_kk: gradient of C-CBF with respect to k twice

        """
        grad_H_kk = -(h**2) * np.exp(-k * h)

        return grad_H_kk

    def grad_H_kx(self, x: NDArray, k: NDArray, h: NDArray) -> float:
        """Computes the gradient of the consolidated control barrier function (C-CBF)
        with respect to the gains (k) and then the state (x).

        Arguments
        ---------
        x: state vector
        k: constituent cbf weighting vector
        h: array of constituent cbfs

        Returns
        -------
        grad_H_kx: gradient of C-CBF with respect to k and then x

        """
        grad_H_kx = (
            np.diag(np.exp(-k * h)) @ self.dhdx - np.diag(h * k * np.exp(-k * h)) @ self.dhdx
        )

        return grad_H_kx

    def grad_h_arr_x(self, x: NDArray) -> NDArray:
        """Computes (numerically) the gradient of the array of constituent cbfs (h)
        with respect to the state (x).

        Arguments
        ---------
        x: state vector

        Returns
        -------
        grad_h_arr_x: gradient of h array with respect to x

        """

        def h(dx):
            return np.array(
                [cbf.h(dx) for cbf in self.cbfs_individual]
                + [
                    cbf.h(
                        dx,
                    )
                    for cbf in self.cbfs_pairwise
                ]
            )

        # Needs to be finished???

    def grad_phi_k(self, x: NDArray, k: NDArray, h: NDArray, Lg: NDArray) -> NDArray:
        """Computes the gradient of Phi with respect to the gains k.

        Arguments
        ---------
        x: state vector
        k: constituent cbf weighting vector
        h: array of constituent cbfs
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_phi_k

        """
        grad_phi_k = (
            self.grad_cost_fcn_k(k, h)
            - np.sum(self.grad_ci_k(k) / self.ci(k))
            - self.grad_czero_k(x, k, h, Lg) / self.czero(x, k, h, Lg)
        )

        return grad_phi_k.T

    def grad_phi_kk(self, x: NDArray, k: NDArray, h: NDArray, Lg: NDArray) -> NDArray:
        """Computes the gradient of Phi with respect to
        the gains k twice.

        Arguments
        ---------
        x: state vector
        k: constituent cbf weighting vector
        h: array of constituent cbfs
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_phi_kk

        """
        grad_phi_kk = (
            self.grad_cost_fcn_kk(k, h, x)
            - np.sum(
                (self.grad_ci_kk(k) * self.ci(k) - self.grad_ci_k(k).T @ self.grad_ci_k(k))
                / self.ci(k) ** 2
            )
            - (
                self.grad_czero_kk(x, k, h, Lg) * self.czero(x, k, h, Lg)
                - self.grad_czero_k(x, k, h, Lg).T @ self.grad_czero_k(x, k, h, Lg)
            )
            / self.czero(x, k, h, Lg) ** 2
        )

        return grad_phi_kk

    def grad_phi_kx(self, x: NDArray, k: NDArray, h: NDArray, Lg: NDArray) -> NDArray:
        """Computes the gradient of Phi with respect to first
        the gains k and then the state x.

        Arguments
        ---------
        x: state vector
        k: constituent cbf weighting vector
        h: array of constituent cbfs
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_phi_kx

        """
        grad_phi_kx = (
            self.grad_cost_fcn_kx(k, h, x)
            - (
                self.czero(x, k, h, Lg) * self.grad_czero_kx(x, k, h, Lg)
                - self.grad_czero_k(x, k, h, Lg)[:, np.newaxis]
                * self.grad_czero_x(x, k, h, Lg)[np.newaxis, :]
            )
            / self.czero(x, k, h, Lg) ** 2
        )

        return grad_phi_kx

    def cost_fcn(self, k: NDArray, h: NDArray) -> float:
        """Computes the quadratic cost function associated with the adaptation law.

        Arguments
        ---------
        k: constituent cbf weight vector
        h: vector of constituent cbfs

        Returns
        -------
        cost: cost evaluated for k

        """
        cost = 1 / 2 * (k - self.k_desired(h)).T @ self.gain_mat @ (k - self.k_desired(h))

        return cost * self.cost_gain

    def grad_cost_fcn_k(self, k: NDArray, h: NDArray) -> float:
        """Computes gradient of the cost function associated with the adaptation law
        with respect to the weight vector k.

        Arguments
        ---------
        k: constituent cbf weight vector
        h: vector of constituent cbfs

        Returns
        -------
        grad_cost_k: gradient of cost evaluated at k

        """
        grad_cost_k = self.gain_mat @ (k - self.k_desired(h))

        return grad_cost_k * self.cost_gain

    def grad_cost_fcn_kk(self, k: NDArray, h: NDArray, x: NDArray) -> float:
        """Computes gradient of the cost function associated with the adaptation law
        with respect to first the weight vector k and then k again.

        Arguments
        ---------
        k: constituent cbf weight vector
        h: vector of constituent cbfs
        x: state vector

        Returns
        -------
        grad_cost_kk: gradient of cost with respect to k and then k again

        """
        return self.gain_mat * self.cost_gain

    def grad_cost_fcn_kx(self, k: NDArray, h: NDArray, x: NDArray) -> float:
        """Computes gradient of the cost function associated with the adaptation law
        with respect to first the weight vector k and then the state x.

        Arguments
        ---------
        k: constituent cbf weight vector
        h: vector of constituent cbfs
        x: state vector

        Returns
        -------
        grad_cost_kx: gradient of cost with respect to k and then x

        """
        grad_cost_kx = -self.gain_mat @ self.grad_k_desired_x(h, x)

        return grad_cost_kx * self.cost_gain

    def czero(self, x: NDArray, k: NDArray, h: NDArray, Lg: NDArray) -> float:
        """Returns the viability constraint function evaluated at the current
        state x and gain vector k.

        Arguments
        ---------
        x: state vector
        k: constituent cbf weighting vector
        h: constituent cbf vector
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        c0: viability constraint function evaluated at x and k

        """
        czero = self.q_vec @ Lg @ self.U_mat @ Lg.T @ self.q_vec.T - self.delta(x, k, h, Lg) ** 2

        return czero * self.czero_gain

    def grad_czero_k(self, x: NDArray, k: NDArray, h: NDArray, Lg: NDArray) -> NDArray:
        """Computes the gradient of the viability constraint function evaluated at the current
        state x and gain vector k with respect to k.

        Arguments
        ---------
        x: state vector
        k: constituent cbf weighting vector
        h: constituent cbf vector
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_c0_k: gradient of viability constraint function with respect to k

        """
        grad_c0_k = 2 * self.dqdk @ Lg @ self.U_mat @ Lg.T @ self.q_vec.T - 2 * self.delta(
            x, k, h, Lg
        ) * self.grad_delta_k(x, k, h, Lg)

        return grad_c0_k * self.czero_gain

    def grad_czero_x(self, x: NDArray, k: NDArray, h: NDArray, Lg: NDArray) -> NDArray:
        """Computes the gradient of the viability constraint function evaluated at the current
        state x and gain vector k with respect to x.

        Arguments
        ---------
        x: state vector
        k: constituent cbf weighting vector
        h: constituent cbf vector
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_c0_x: gradient of viability constraint function with respect to x

        """
        # TO DO
        dLgdx = np.zeros((len(x), len(k), 2))

        grad_c0_x = (
            2 * self.dqdx.T @ Lg @ self.U_mat @ Lg.T @ self.q_vec.T
            + 2 * self.q_vec @ dLgdx @ self.U_mat @ Lg.T @ self.q_vec.T
            - 2 * self.delta(x, k, h, Lg) * self.grad_delta_x(x, k, h, Lg)
        )

        return grad_c0_x * self.czero_gain

    def grad_czero_kk(self, x: NDArray, k: NDArray, h: NDArray, Lg: NDArray) -> NDArray:
        """Computes the gradient of the viability constraint function evaluated at the current
        state x and gain vector k with respect to first k and then k again.

        Arguments
        ---------
        x: state vector
        k: constituent cbf weighting vector
        h: constituent cbf vector

        Returns
        -------
        grad_c0_kk: gradient of viability constraint function with respect to k then x

        """
        grad_c0_kk = (
            2 * self.d2qdk2 @ Lg @ self.U_mat @ Lg.T @ self.q_vec.T
            + 2 * self.dqdk @ Lg @ self.U_mat @ Lg.T @ self.dqdk.T
            - 2
            * self.grad_delta_k(x, k, h, Lg)[:, np.newaxis]
            @ self.grad_delta_k(x, k, h, Lg)[np.newaxis, :]
            - 2 * self.delta(x, k, h, Lg) * self.grad_delta_kk(x, k, h, Lg)
        )

        return grad_c0_kk * self.czero_gain

    def grad_czero_kx(self, x: NDArray, k: NDArray, h: NDArray, Lg: NDArray) -> NDArray:
        """Computes the gradient of the viability constraint function evaluated at the current
        state x and gain vector k with respect to first k and then x.

        Arguments
        ---------
        x: state vector
        k: constituent cbf weighting vector
        h: constituent cbf vector

        Returns
        -------
        grad_c0_kx: gradient of viability constraint function with respect to k then x

        """
        dLgdx = np.zeros((len(x), len(k), 2))

        grad_c0_kx = (
            2 * self.q_vec @ Lg @ self.U_mat @ Lg.T @ self.d2qdkdx
            + 2 * self.dqdk @ Lg @ self.U_mat @ Lg.T @ self.dqdx
            + 4 * (self.dqdk @ dLgdx @ self.U_mat @ Lg.T @ self.q_vec.T).T
            - 2
            * (
                self.grad_delta_k(x, k, h, Lg)[:, np.newaxis]
                @ self.grad_delta_x(x, k, h, Lg)[np.newaxis, :]
                + self.delta(x, k, h, Lg) * self.grad_delta_kx(x, k, h, Lg)
            )
        )

        return grad_c0_kx * self.czero_gain

    def ci(self, k: NDArray) -> NDArray:
        """Returns positivity constraint functions on the gain vector k.

        Arguments
        ---------
        k: constituent cbf weighting vector

        Returns
        -------
        ci: array of positivity constraint functions evaluated at k

        """
        return (k - self.k_min) * self.ci_gain

    def grad_ci_k(self, k: NDArray) -> NDArray:
        """Computes the gradient of the positivity constraint functions evaluated at the
        gain vector k with respect to k.

        Arguments
        ---------
        k: constituent cbf weighting vector

        Returns
        -------
        grad_ci_k: gradient of positivity constraint functions with respect to k

        """
        return np.ones((len(k),)) * self.ci_gain

    def grad_ci_x(self, k: NDArray) -> NDArray:
        """Computes the gradient of the positivity constraint functions evaluated at the
        gain vector k with respect to x.

        Arguments
        ---------
        k: constituent cbf weighting vector

        Returns
        -------
        grad_ci_x: gradient of positivity constraint functions with respect to x

        """
        return np.zeros((len(k),)) * self.ci_gain

    def grad_ci_kk(self, k: NDArray) -> NDArray:
        """Computes the gradient of the positivity constraint functions evaluated at the
        gain vector k with respect to first k and then k again.

        Arguments
        ---------
        k: constituent cbf weighting vector

        Returns
        -------
        grad_ci_kk: gradient of positivity constraint functions with respect to k and then x

        """
        return np.zeros((len(k),)) * self.ci_gain

    def grad_ci_kx(self, k: NDArray) -> NDArray:
        """Computes the gradient of the positivity constraint functions evaluated at the
        gain vector k with respect to first k and then x.

        Arguments
        ---------
        k: constituent cbf weighting vector

        Returns
        -------
        grad_ci_kx: gradient of positivity constraint functions with respect to k and then x

        """
        return np.zeros((len(k),)) * self.ci_gain

    def delta(self, x: NDArray, k: NDArray, h: NDArray, Lg: NDArray) -> float:
        """Computes the threshold above which the product LgH*u_max must remain for control viability.
        In other words, in order to be able to satisfy:

        LfH + LgH*u + LkH + alpha(H) >= 0

        it must hold that LgH*u_max >= -LfH - LkH - alpha(H).

        Arguments
        ---------
        x: state vector
        k: constituent cbf weighting vector
        h: constituent cbf vector
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        delta = LfH + alpha(H) + LkH

        """
        dhdk = h * np.exp(-k * h)
        LfH = self.q_vec @ self.dhdx @ f(x)

        return (
            -LfH
            - self.H(k, h)
            - dhdk @ self.filtered_wf
            + abs(dhdk @ self.filtered_wg) @ self.u_max
        )

    def grad_delta_k(self, x: NDArray, k: NDArray, h: NDArray, Lg: NDArray) -> NDArray:
        """Computes the threshold above which the product LgH*u_max must remain for control viability.

        Arguments
        ---------
        x: state vector
        k: constituent cbf weighting vector
        h: constituent cbf vector
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_delta_k: gradient of delta with respect to k

        """
        dLfHdk = self.dqdk @ self.dhdx @ f(x)

        # TO DO: add filtered terms
        filtered_terms = np.zeros(dLfHdk.shape)

        return -dLfHdk - self.grad_H_k(k, h) + filtered_terms

    # TO DO: add filtered terms
    def grad_delta_x(self, x: NDArray, k: NDArray, h: NDArray, Lg: NDArray) -> NDArray:
        """Computes the threshold above which the product LgH*u_max must remain for control viability.

        Arguments
        ---------
        x: state vector
        k: constituent cbf weighting vector
        h: constituent cbf vector
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_delta_x: gradient of delta with respect to x

        """
        # TO DO: define dhdx as a function of x and numerically compute gradient
        def hx(dx):
            return h

        def dhdx(dx):
            return self.dhdx

        # Define functions to compute gradient numerically
        def LfH(dx):
            return (k * np.exp(-k * hx(dx))) @ dhdx(dx) @ f(dx)

        dLfHdx = nd.Jacobian(LfH)(x)[0, :]  # Compute gradient numerically

        # TO DO: add filtered terms
        filtered_terms = np.zeros(dLfHdx.shape)

        return -dLfHdx - self.grad_H_x(x, k, h) + filtered_terms

    # TO DO: add filtered terms
    def grad_delta_kk(self, x: NDArray, k: NDArray, h: NDArray, Lg: NDArray) -> NDArray:
        """Computes the threshold above which the product LgH*u_max must remain for control viability.

        Arguments
        ---------
        x: state vector
        k: constituent cbf weighting vector
        h: constituent cbf vector
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_delta_kk: gradient of delta with respect to k twice

        """
        triple_diag = np.zeros((len(k), len(k), len(k)))
        np.fill_diagonal(triple_diag, (k * h**2 - 2 * h) * np.exp(-k * h))
        grad_LfH_kk = triple_diag @ self.dhdx @ f(x)

        # TO DO: add filtered terms
        filtered_terms = np.zeros(grad_LfH_kk.shape)

        return -grad_LfH_kk - self.grad_H_kk(k, h) + filtered_terms

    # TO DO: add filtered terms
    def grad_delta_kx(self, x: NDArray, k: NDArray, h: NDArray, Lg: NDArray) -> NDArray:
        """Computes the threshold above which the product LgH*u_max must remain for control viability.

        Arguments
        ---------
        x: state vector
        k: constituent cbf weighting vector
        h: constituent cbf vector
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_delta_kx: gradient of delta with respect to k and then x

        """

        def grad_LfH_k(dx):
            return self.dqdk @ self.dhdx @ f(dx)

        grad_LfH_kx = nd.Jacobian(grad_LfH_k)(x)

        filtered_terms = np.zeros(grad_LfH_kx.shape)

        return -grad_LfH_kx - self.grad_H_kx(x, k, h) + filtered_terms

    def k_desired(self, h: NDArray) -> NDArray:
        """Computes the desired gains k for the constituent cbfs. This can be
        thought of as the nominal adaptation law (unconstrained).

        Arguments
        ---------
        h: array of constituent cbf values

        Returns
        -------
        k_desired

        """
        k_des = self.kd_gain * h / np.min([np.min(h), 1.0])

        return np.clip(k_des, self.k_min, self.k_max)

    def grad_k_desired_x(self, h: NDArray, x: NDArray) -> NDArray:
        """Computes the desired gains k for the constituent cbfs. This can be
        thought of as the nominal adaptation law (unconstrained).

        Arguments
        ---------
        h: array of constituent cbf values
        x: state vector

        Returns
        -------
        grad_k_desired_x

        """
        min_h_idx = np.where(h == np.min(h))[0][0]

        k_des = self.kd_gain * h / np.min([h[min_h_idx], 2.0])
        over_k_max = np.where(k_des > self.k_max)[0]
        under_k_min = np.where(k_des < self.k_min)[0]

        if h[min_h_idx] > 2.0:
            grad_k_desired_x = self.kd_gain * self.dhdx / 2.0
        else:
            # Deal with cases when dhdx very close to zero
            dhdx = self.dhdx
            dhdx[abs(dhdx) <= 1e-9] = 1
            grad_k_desired_x = self.kd_gain * self.dhdx / dhdx[min_h_idx, :]

        grad_k_desired_x[over_k_max] = 0
        grad_k_desired_x[under_k_min] = 0

        return grad_k_desired_x

    def k_dot_lqr(self, h_array: NDArray) -> NDArray:
        """Computes the nominal k_dot adaptation law based on LQR.

        Arguments
            h_array: array of CBF values

        RETURNS
            k_dot_nominal: nominal k_dot adaptation law

        """
        gain = 50.0
        gain = 1.0
        # k_star = gain * h_array / np.max([np.min(h_array), 0.5])  # Desired k values
        k_star = np.ones((len(h_array),))

        # Integrator dynamics
        Ad = np.zeros((self.k_gains.shape[0], self.k_gains.shape[0]))
        Bd = np.eye(self.k_gains.shape[0])

        # Compute LQR gain
        Q = 0.01 * Bd
        R = 1 * Bd
        K, _, _ = lqr(Ad, Bd, Q, R)

        k_dot_lqr = -K @ (self.k_gains - k_star)
        k_dot_bounds = 5.0
        # k_dot_bounds = 100.0

        return np.clip(k_dot_lqr, -k_dot_bounds, k_dot_bounds)
