"""consolidated_cbf_controller.py

Provides interface to the ConsolidatedCbfController class.

"""
import numpy as np
import numdifftools as nd
import builtins
from typing import Callable, List, Optional, Tuple
from importlib import import_module
from nptyping import NDArray
from control import lqr
from scipy.linalg import block_diag, null_space

# from ..cbfs.cbf import Cbf
from core.controllers.cbf_qp_controller import CbfQpController
from core.controllers.controller import Controller

# from core.solve_cvxopt import solve_qp_cvxopt

vehicle = builtins.PROBLEM_CONFIG["vehicle"]
control_level = builtins.PROBLEM_CONFIG["control_level"]
system_model = builtins.PROBLEM_CONFIG["system_model"]
mod = "models." + vehicle + "." + control_level + ".models"

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
        self.n_agents = nAgents
        self.filtered_wf = np.zeros((nCBF,))
        self.filtered_wg = np.zeros((nCBF, len(self.u_max)))
        self.k_weights = np.zeros((nCBF,))
        self.k_dot = np.zeros((nCBF,))

        self.adapter = AdaptationLaw(nCBF, u_max, kZero=0.5)

    def _compute_control(self, t: float, z: NDArray, cascaded: bool = False) -> (NDArray, int, str):
        self.u, code, status = super()._compute_control(t, z, cascaded)

        # Update k weights, k_dot
        k_weights, k_dot = self.adapter.update(self.u, self._dt)
        self.k_weights = k_weights
        self.k_dot = k_dot

        return self.u, code, status

    def formulate_qp(
        self, t: float, ze: NDArray, zr: NDArray, u_nom: NDArray, ego: int, cascade: bool = False
    ) -> (NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, float):
        """Configures the Quadratic Program parameters (Q, p for objective function, A, b for inequality constraints,
        G, h for equality constraints).

        """

        # # Update C-CBF Parameters
        # if t > 0:
        #     # Update estimates
        #     k_dot = self.compute_k_dot(self.u)
        #     self.k += self._dt * k_dot

        # Compute Q matrix and p vector for QP objective function
        Q, p = self.compute_objective_qp(u_nom)

        # Compute input constraints of form Au @ u <= bu
        Au, bu = self.compute_input_constraints()

        # Parameters
        na = 1 + len(zr)
        ns = len(ze)
        self.safety = True
        discretization_error = 1.0

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
            Lgh_array[cc, self.n_controls * ego : (ego + 1) * self.n_controls] = dhdx_array[cc][
                :ns
            ] @ g(
                ze
            )  # Only assign ego control
            if cascade:
                Lgh_array[cc, self.n_controls * ego] = 0.0

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
                Lgh_array[idx, self.n_controls * ego : (ego + 1) * self.n_controls] = dhdx_array[
                    idx
                ][:ns] @ g(ze)
                Lgh_array[
                    idx, self.n_controls * other : (other + 1) * self.n_controls
                ] = dhdx_array[idx][ns:] @ g(zo)
                if cascade:
                    Lgh_array[idx, self.n_controls * ego] = 0.0

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

    def compute_objective_qp(self, u_nom: NDArray) -> (NDArray, NDArray):
        """Computes the matrix Q and vector p for the objective function of the
        form

        J = 1/2 * x.T @ Q @ x + p @ x

        Arguments:
            u_nom: nominal control input for agent in question

        Returns:
            Q: quadratic term positive definite matrix for objective function
            p: linear term vector for objective function

        """
        if self.n_dec_vars > 0:
            Q, p = self.objective(
                np.concatenate(
                    [u_nom.flatten(), np.array(self.n_dec_vars * [self.desired_class_k])]
                )
            )
            # Q, p = self.objective(np.append(u_nom.flatten(), self.desired_class_k))
        else:
            Q, p = self.objective(u_nom.flatten())

        return Q, p

    def compute_input_constraints(self):
        """
        Computes matrix Au and vector bu encoding control input constraints of
        the form

        Au @ u <= bu

        Arguments:
            None

        Returns:
            Au: input constraint matrix
            bu: input constraint vector

        """
        if self.n_dec_vars > 0:
            Au = block_diag(*(self.n_agents + self.n_dec_vars) * [self.au])[:-2, :-1]
            bu = np.append(
                np.array(self.n_agents * [self.bu]).flatten(),
                self.n_dec_vars * [self.max_class_k, 0],
            )

        else:
            Au = block_diag(*(self.n_agents) * [self.au])
            bu = np.array(self.n_agents * [self.bu]).flatten()

        return Au, bu

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
        # Introduce parameters
        discretization_error = 0.1
        k_ccbf = 0.1
        # k_ccbf = 1.0

        # Get C-CBF Value
        H = self.consolidated_cbf()
        self.c_cbf = H

        # Non-centralized agents CBF dynamics become drifts
        Lgh_uncontrolled = np.copy(Lgh_array[:, :])
        Lgh_uncontrolled[:, ego * self.n_controls : (ego + 1) * self.n_controls] = 0

        # Set all Lgh terms other than ego to zero
        indices_before_ego = np.s_[: ego * self.n_controls]
        indices_after_ego = np.s_[(ego + 1) * self.n_controls :]
        Lgh_array[:, indices_before_ego] = 0
        Lgh_array[:, indices_after_ego] = 0

        # Get time-derivatives of gains
        dphidh = self.adapter.k_weights * np.exp(-self.adapter.k_weights * self.cbf_vals)
        dphidk = self.cbf_vals * np.exp(-self.adapter.k_weights * self.cbf_vals)

        # Compute C-CBF Dynamics
        LfH = dphidh @ Lfh_array
        LgH = dphidh @ Lgh_array
        LgH_uncontrolled = dphidh @ Lgh_uncontrolled

        # Tunable CBF Addition
        Phi = (
            np.tile(-np.array(self.u_max), int(LgH_uncontrolled.shape[0] / len(self.u_max)))
            @ abs(LgH_uncontrolled)
            * np.exp(-k_ccbf * H)
        )
        LfH = LfH + Phi - discretization_error
        Lf_for_kdot = LfH + dphidk @ self.adapter.k_dot_f
        Lg_for_kdot = Lgh_array[:, self.n_controls * ego : self.n_controls * (ego + 1)]

        # Compute drift k_dot
        k_dot_drift = self.adapter.k_dot_drift(x, self.cbf_vals, Lf_for_kdot, Lg_for_kdot)

        # Compute controlled k_dot
        k_dot_cont = self.adapter.k_dot_controlled(x, self.cbf_vals, Lf_for_kdot, Lg_for_kdot)

        # Augment LfH and LgH from kdot
        LfH += dphidk @ k_dot_drift
        LgH[self.n_controls * ego : self.n_controls * (ego + 1)] += dphidk @ k_dot_cont

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

        return a_mat[:, np.newaxis].T, b_vec

    def consolidated_cbf(self):
        return 1 - np.sum(np.exp(-self.adapter.k_weights * self.cbf_vals))

    # def compute_k_dots(
    #     self, h_array: NDArray, H: float, LfH: float, Lgh_array: NDArray, vec: NDArray
    # ) -> NDArray:
    #     """Computes the time-derivatives of the k_gains via QP based adaptation law.

    #     ARGUMENTS
    #         h_array: array of CBF values
    #         H: scalar c-cbf value
    #         LfH: scalar LfH term for c-cbf
    #         Lgh_array: 2D array of Lgh terms
    #         vec: vector that needs to stay out of the nullspace of matrix P

    #     RETURNS
    #         k_dot

    #     """
    #     # Good behavior gains
    #     k_min = 0.1
    #     k_max = 1.0
    #     k_des = 10 * k_min * h_array
    #     k_des = np.clip(k_des, k_min, k_max)
    #     P_gain = 10.0
    #     constraint_gain = 10.0
    #     gradient_gain = 0.01

    #     # (3 Reached but H negative) gains
    #     k_min = 0.01
    #     k_max = 1.0
    #     k_des = h_array / 5
    #     k_des = np.clip(k_des, k_min, k_max)
    #     P_gain = 100.0
    #     constraint_gain = 1.0
    #     gradient_gain = 0.001

    #     # Testing gains
    #     k_min = 0.01
    #     k_max = 5.0
    #     k_des = 0.1 * h_array / np.min([np.min(h_array), 2.0])
    #     k_des = np.clip(k_des, k_min, k_max)
    #     P_gain = 1000.0
    #     kpositive_gain = 1000.0
    #     constraint_gain = 1000.0
    #     gradient_gain = 0.1

    #     # TO DO: Correct how delta is being used -- need the max over all safe set
    #     # Some parameters
    #     P_mat = P_gain * np.eye(len(h_array))
    #     p_vec = vec
    #     N_mat = null_space(Lgh_array.T)
    #     rank_Lg = N_mat.shape[0] - N_mat.shape[1]
    #     Q_mat = np.eye(len(self.k_gains)) - 2 * N_mat @ N_mat.T + N_mat @ N_mat.T @ N_mat @ N_mat.T
    #     _, sigma, _ = np.linalg.svd(Lgh_array.T)  # Singular values of LGH
    #     if rank_Lg > 0:
    #         sigma_r = sigma[rank_Lg - 1]  # minimum non-zero singular value
    #     else:
    #         sigma_r = (
    #             0  # This should not ever happen (by assumption that not all individual Lgh = 0)
    #         )
    #     delta = -(
    #         LfH + 0.1 * H + (h_array * np.exp(-self.k_gains * h_array)) @ self.k_dots
    #     ) / np.linalg.norm(self.u_max)
    #     delta = 1 * abs(delta)

    #     # Objective Function and Partial Derivatives
    #     J = 1 / 2 * (self.k_gains - k_des).T @ P_mat @ (self.k_gains - k_des)
    #     dJdk = P_mat @ (self.k_gains - k_des)
    #     d2Jdk2 = P_mat

    #     # K Constraint Functions and Partial Derivatives: Convention hi >= 0
    #     hi = (self.k_gains - k_min) * kpositive_gain
    #     dhidk = np.ones((len(self.k_gains),)) * kpositive_gain
    #     d2hidk2 = np.zeros((len(self.k_gains), len(self.k_gains))) * kpositive_gain

    #     # Partial Derivatives of p vector
    #     dpdk = np.diag((self.k_gains * h_array - 1) * np.exp(-self.k_gains * h_array))
    #     d2pdk2_vals = h_array * np.exp(-self.k_gains * h_array) + (
    #         self.k_gains * h_array - 1
    #     ) * -h_array * np.exp(-self.k_gains * h_array)
    #     d2pdk2 = np.zeros((len(h_array), len(h_array), len(h_array)))
    #     np.fill_diagonal(d2pdk2, d2pdk2_vals)

    #     # # LGH Norm Constraint Function and Partial Derivatives
    #     # # Square of norm
    #     # eta = (sigma_r**2 * (p_vec.T @ Q_mat @ p_vec) - delta**2) * constraint_gain
    #     # detadk = 2 * sigma_r**2 * (dpdk.T @ Q_mat @ p_vec) * constraint_gain
    #     # d2etadk2 = (2 * sigma_r**2 * (d2pdk2.T @ Q_mat @ p_vec + dpdk.T @ Q_mat @ dpdk)) * constraint_gain

    #     # V vec
    #     v_vec = (p_vec @ Lgh_array).T
    #     dvdk = (dpdk @ Lgh_array).T
    #     d2vdk2 = (d2pdk2 @ Lgh_array).T

    #     # New Input Constraints Condition
    #     eta = ((v_vec.T @ U_mat @ v_vec) - delta**2) * constraint_gain
    #     detadk = 2 * (dvdk.T @ U_mat @ v_vec) * constraint_gain
    #     d2etadk2 = (2 * (d2vdk2.T @ U_mat @ v_vec + dvdk.T @ U_mat @ dvdk)) * constraint_gain

    #     print(f"Eta: {eta}")

    #     if np.isnan(eta):
    #         print(eta)

    #     # Define the augmented cost function
    #     Phi = J - np.sum(np.log(hi)) - np.log(eta)
    #     dPhidk = dJdk - np.sum(dhidk / hi) - detadk / eta
    #     d2Phidk2 = (
    #         d2Jdk2 - np.sum(d2hidk2 / hi - dhidk / hi**2) - (d2etadk2 / eta - detadk / eta**2)
    #     )

    #     # Define Adaptation law
    #     corrector_term = -np.linalg.inv(d2Phidk2) @ (dPhidk)  # Correction toward minimizer
    #     # corrector_term = -dPhidk  # Correction toward minimizer -- gradient method only
    #     predictor_term = (
    #         0 * self.k_gains
    #     )  # Zero for now (need to see how this could depend on time)
    #     k_dots = gradient_gain * (corrector_term + predictor_term)

    #     if np.sum(np.isnan(k_dots)) > 0:
    #         print(k_dots)

    #     self.k_dots = k_dots

    #     return k_dots

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
        self.dqdx = self.dhdx.T @ np.diag(-(k**2) * np.exp(-k * h))
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
            u = np.zeros((self.n_controls,))

        k_dot = wf + wg @ u

        return k_dot * self.kdot_gain


class AdaptationLaw:
    """Computes the parameter adaptation for the ConsolidatedCbfController
    class.

    Attributes:
        kWeights (NDArray): values of current c-cbf weights k
    """

    def __init__(self, nWeights: int, uMax: NDArray, kZero: Optional[float] = 0.5):
        """Initializes class attributes.

        Arguments:
            nWeights (int): number of weights/CBFs to be consolidated
            uMax (NDArray): maximum control input vector
            kZero (float, Opt)

        """
        nStates = 5  # placeholder

        # k weights and derivatives
        self._k_weights = kZero * np.ones((nWeights,))
        self._k_dot = np.zeros((nWeights,))
        self._k_dot_f = np.zeros((nWeights,))
        self._k_dot_drift = np.zeros((nWeights,))
        self._k_dot_controlled = np.zeros((nWeights, len(uMax)))

        # q vector and derivatives
        self.q = np.zeros((nWeights,))
        self.dqdk = np.zeros((nWeights, nWeights))
        self.dqdx = np.zeros((nWeights, nStates))
        self.d2qdk2 = np.zeros((nWeights, nWeights, nWeights))
        self.d2qdkdx = np.zeros((nWeights, nStates, nWeights))

        # control contraint matrix
        self.U = uMax[:, np.newaxis] @ uMax[np.newaxis, :]

        # dhdx matrix
        self.dhdx = np.zeros((nWeights, nStates))

        # Gains and Parameters
        self.cost_gain_mat = 100.0 * np.eye(nWeights)
        self.czero_gain = 0.01
        self.ci_gain = 0.01
        self.k_des_gain = 0.1
        self.k_min = 0.1
        self.k_max = 50.0
        self.k_dot_gain = 0.01

    def update(self, u: NDArray, dt: float) -> Tuple[NDArray, NDArray]:
        """Updates the adaptation gains and returns the new k weights.

        Arguments:
            x (NDArray): state vector
            u (NDArray): control input applied to system
            h (NDArray): vector of candidate CBFs
            Lf (float): C-CBF drift term (includes filtered kdot)
            Lg (NDArray): matrix of stacked Lgh vectors
            dt: timestep in sec

        Returns
            k_weights: weights on constituent candidate cbfs

        """
        self._k_weights = self._k_weights + self.compute_kdot(u) * dt

        return self._k_weights, self._k_dot

    def compute_kdot(self, u: NDArray) -> NDArray:
        """Computes the time-derivative k_dot of the k_weights vector.

        Arguments:
            u (NDArray): control input applied to system

        Returns:
            k_dot (NDArray): time-derivative of kWeights

        """

        self._k_dot = self._k_dot_drift + self._k_dot_controlled @ u
        self._k_dot_f = self._k_dot

        return self._k_dot

    def k_dot_drift(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
        """Computes the drift (uncontrolled) component of the time-derivative
        k_dot of the k_weights vector.

        Arguments:
            x (NDArray): state vector
            h (NDArray): vector of candidate CBFs
            Lf (float): C-CBF drift term (includes filtered kdot)
            Lg (NDArray): matrix of stacked Lgh vectors

        Returns:
            k_dot_drift (NDArray): time-derivative of kWeights

        """
        # Compute terms
        P = self.grad_cost_kk()
        Q = np.diag(1 / self.ci() ** 2)
        R = (
            self.czero(x, h, Lf, Lg) * self.grad_czero_kk(x, h, Lf, Lg)
            - self.grad_czero_k(x, h, Lf, Lg)[:, np.newaxis]
            @ self.grad_czero_k(x, h, Lf, Lg)[np.newaxis, :]
        ) / self.czero(x, h, Lf, Lg) ** 2
        M = self.grad_phi_kk(x, h, Lf, Lg) + P + Q + R
        X = (
            self.czero(x, h, Lf, Lg) * self.grad_czero_kx(x, h, Lf, Lg)
            - self.grad_czero_k(x, h, Lf, Lg)[:, np.newaxis]
            @ self.grad_czero_x(x, h, Lf, Lg)[np.newaxis, :]
        ) / self.czero(x, h, Lf, Lg) ** 2

        k_dot_drift = -np.linalg.inv(M) @ (
            self.cost_gain_mat @ self.grad_phi_k(x, h, Lf, Lg) + X @ f(x)
        )

        self._k_dot_drift = k_dot_drift

        return self._k_dot_drift

    def k_dot_controlled(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
        """Computes the controlled component of the time-derivative
        k_dot of the k_weights vector.

        Arguments:
            x (NDArray): state vector
            u (NDArray): control input applied to system
            h (NDArray): vector of candidate CBFs
            Lf (float): C-CBF drift term (includes filtered kdot)
            Lg (NDArray): matrix of stacked Lgh vectors

        Returns:
            k_dot_controlled (NDArray): time-derivative of kWeights

        """
        # Compute terms
        P = self.grad_cost_kk()
        Q = np.diag(1 / self.ci() ** 2)
        R = (
            self.czero(x, h, Lf, Lg) * self.grad_czero_kk(x, h, Lf, Lg)
            - self.grad_czero_k(x, h, Lf, Lg)[:, np.newaxis]
            @ self.grad_czero_k(x, h, Lf, Lg)[np.newaxis, :]
        ) / self.czero(x, h, Lf, Lg) ** 2
        M = self.grad_phi_kk(x, h, Lf, Lg) + P + Q + R
        X = (
            self.czero(x, h, Lf, Lg) * self.grad_czero_kx(x, h, Lf, Lg)
            - self.grad_czero_k(x, h, Lf, Lg)[:, np.newaxis]
            @ self.grad_czero_x(x, h, Lf, Lg)[np.newaxis, :]
        ) / self.czero(x, h, Lf, Lg) ** 2

        k_dot_controlled = -np.linalg.inv(M) @ X @ g(x)

        self._k_dot_controlled = k_dot_controlled

        return self._k_dot_controlled

    def grad_phi_k(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
        """Computes the gradient of Phi with respect to the gains k.

        Arguments
        ---------
        x: state vector
        h: array of constituent cbfs
        Lf (float): C-CBF drift term (includes filtered kdot)
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_phi_k

        """
        grad_phi_k = (
            self.grad_cost_k(h)
            - self.grad_ci_k() / self.ci()
            - self.grad_czero_k(x, h, Lf, Lg) / self.czero(x, h, Lf, Lg)
        )

        return grad_phi_k.T

    def grad_phi_kk(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
        """Computes the gradient of Phi with respect to
        the gains k twice.

        Arguments
        ---------
        x: state vector
        h: array of constituent cbfs
        Lf: C-CBF drift term (includes filtered kdot)
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_phi_kk

        """
        grad_phi_kk = (
            self.grad_cost_kk()
            - (
                self.grad_ci_kk() * self.ci()
                - self.grad_ci_k()[:, np.newaxis] @ self.grad_ci_k()[np.newaxis, :]
            )
            / self.ci() ** 2
            - (
                self.grad_czero_kk(x, h, Lf, Lg) * self.czero(x, h, Lf, Lg)
                - self.grad_czero_k(x, h, Lf, Lg)[:, np.newaxis]
                @ self.grad_czero_k(x, h, Lf, Lg)[np.newaxis, :]
            )
            / self.czero(x, h, Lf, Lg) ** 2
        )

        return grad_phi_kk

    def grad_phi_kx(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
        """Computes the gradient of Phi with respect to first
        the gains k and then the state x.

        Arguments
        ---------
        x: state vector
        h: array of constituent cbfs
        Lf: C-CBF drift term (includes filtered kdot)
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_phi_kx

        """
        grad_phi_kx = (
            self.grad_cost_kx(x, h)
            - (
                self.grad_czero_kx(x, h, Lf, Lg) * self.czero(x, h, Lf, Lg)
                - self.grad_czero_k(x, h, Lf, Lg)[:, np.newaxis]
                * self.grad_czero_x(x, h, Lf, Lg)[np.newaxis, :]
            )
            / self.czero(x, h, Lf, Lg) ** 2
            - (
                self.grad_ci_kx() * self.ci()
                - self.grad_ci_k()[:, np.newaxis] * self.grad_ci_x()[np.newaxis, :]
            )
            / self.ci() ** 2
        )

        return grad_phi_kx

    def cost(self, h: NDArray) -> float:
        """Computes the quadratic cost function associated with the adaptation law.

        Arguments
        ---------
        h: vector of constituent cbfs

        Returns
        -------
        cost: cost evaluated for k

        """
        cost = (
            0.5
            * (self._k_weights - self.k_des(h)).T
            @ self.cost_gain_mat
            @ (self._k_weights - self.k_des(h))
        )

        return cost

    def grad_cost_k(self, h: NDArray) -> float:
        """Computes gradient of the cost function associated with the adaptation law
        with respect to the weight vector k.

        Arguments
        ---------
        h: vector of constituent cbfs

        Returns
        -------
        grad_cost_k: gradient of cost evaluated at k

        """
        grad_cost_k = self.cost_gain_mat @ (self._k_weights - self.k_des(h))

        return grad_cost_k

    def grad_cost_kk(self) -> float:
        """Computes gradient of the cost function associated with the adaptation law
        with respect to first the weight vector k and then k again.

        Arguments:
            None

        Returns
            grad_cost_kk (NDArray): gradient of cost with respect to k and then k again

        """
        return self.cost_gain_mat

    def grad_cost_kx(self, x: NDArray, h: NDArray) -> float:
        """Computes gradient of the cost function associated with the adaptation law
        with respect to first the weight vector k and then the state x.

        Arguments
        ---------
        x: state vector
        h: vector of constituent cbfs

        Returns
        -------
        grad_cost_kx: gradient of cost with respect to k and then x

        """
        grad_cost_kx = -self.cost_gain_mat @ self.grad_k_des_x(x, h)

        return grad_cost_kx

    def czero(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> float:
        """Returns the viability constraint function evaluated at the current
        state x and gain vector k.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector
        Lf: C-CBF drift term (includes filtered kdot)
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        c0: viability constraint function evaluated at x and k

        """
        czero = self.q @ Lg @ self.U @ Lg.T @ self.q.T - self.delta(x, h, Lf) ** 2

        return czero * self.czero_gain

    def grad_czero_k(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
        """Computes the gradient of the viability constraint function evaluated at the current
        state x and gain vector k with respect to k.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector
        Lf: C-CBF drift term (includes filtered kdot)
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_c0_k: gradient of viability constraint function with respect to k

        """
        grad_c0_k = 2 * self.dqdk @ Lg @ self.U @ Lg.T @ self.q.T - 2 * self.delta(
            x, h, Lf
        ) * self.grad_delta_k(x, h, Lf)

        return grad_c0_k * self.czero_gain

    def grad_czero_x(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
        """Computes the gradient of the viability constraint function evaluated at the current
        state x and gain vector k with respect to x.

        Arguments
        ---------
        x: state vector
        k: constituent cbf weighting vector
        h: constituent cbf vector
        Lf: C-CBF drift term (includes filtered kdot)
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_c0_x: gradient of viability constraint function with respect to x

        """
        # TO DO
        dLgdx = np.zeros((len(x), len(self._k_weights), 2))

        grad_c0_x = (
            2 * self.dqdx.T @ Lg @ self.U @ Lg.T @ self.q.T
            + 2 * self.q @ dLgdx @ self.U @ Lg.T @ self.q.T
            - 2 * self.delta(x, h, Lf) * self.grad_delta_x(x, h)
        )

        return grad_c0_x * self.czero_gain

    def grad_czero_kk(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
        """Computes the gradient of the viability constraint function evaluated at the current
        state x and gain vector k with respect to first k and then k again.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector
        Lf: C-CBF drift term (includes filtered kdot)
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_c0_kk: gradient of viability constraint function with respect to k then x

        """
        if self.delta(x, h, Lf) > 0:

            grad_c0_kk = (
                2 * self.d2qdk2 @ Lg @ self.U @ Lg.T @ self.q.T
                + 2 * self.dqdk @ Lg @ self.U @ Lg.T @ self.dqdk.T
                - 2
                * self.grad_delta_k(x, h, Lf)[:, np.newaxis]
                @ self.grad_delta_k(x, h, Lf)[np.newaxis, :]
                - 2 * self.delta(x, h, Lf) * self.grad_delta_kk(x, h)
            )

        else:
            grad_c0_kk = (
                2 * self.d2qdk2 @ Lg @ self.U @ Lg.T @ self.q.T
                + 2 * self.dqdk @ Lg @ self.U @ Lg.T @ self.dqdk.T
            )

        return grad_c0_kk * self.czero_gain

    def grad_czero_kx(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
        """Computes the gradient of the viability constraint function evaluated at the current
        state x and gain vector k with respect to first k and then x.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector
        Lf: C-CBF drift term (includes filtered kdot)
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_c0_kx: gradient of viability constraint function with respect to k then x

        """
        dLgdx = np.zeros((len(x), len(self._k_weights), 2))

        grad_c0_kx = (
            2 * self.d2qdkdx @ Lg @ self.U @ Lg.T @ self.q.T
            + 2 * self.dqdk @ Lg @ self.U @ Lg.T @ self.dqdx
            + 4 * (self.dqdk @ dLgdx @ self.U @ Lg.T @ self.q.T).T
            - 2
            * (
                self.grad_delta_k(x, h, Lf)[:, np.newaxis] @ self.grad_delta_x(x, h)[np.newaxis, :]
                + self.delta(x, h, Lf) * self.grad_delta_kx(x, h)
            )
        )

        return grad_c0_kx * self.czero_gain

    def ci(self) -> NDArray:
        """Returns positivity constraint functions on the gain vector k.

        Arguments:
            None

        Returns:
            ci: array of positivity constraint functions evaluated at k

        """
        return (self._k_weights - self.k_min) * self.ci_gain

    def grad_ci_k(self) -> NDArray:
        """Computes the gradient of the positivity constraint functions evaluated at the
        gain vector k with respect to k.

        Arguments
        ---------
        k: constituent cbf weighting vector

        Returns
        -------
        grad_ci_k: gradient of positivity constraint functions with respect to k

        """
        return np.ones((len(self._k_weights),)) * self.ci_gain

    def grad_ci_x(self) -> NDArray:
        """Computes the gradient of the positivity constraint functions evaluated at the
        gain vector k with respect to x.

        Arguments
        ---------
        k: constituent cbf weighting vector

        Returns
        -------
        grad_ci_x: gradient of positivity constraint functions with respect to x

        """
        return np.zeros((len(self._k_weights),)) * self.ci_gain

    def grad_ci_kk(self) -> NDArray:
        """Computes the gradient of the positivity constraint functions evaluated at the
        gain vector k with respect to first k and then k again.

        Arguments
        ---------
        k: constituent cbf weighting vector

        Returns
        -------
        grad_ci_kk: gradient of positivity constraint functions with respect to k and then x

        """
        return np.zeros((len(self._k_weights),)) * self.ci_gain

    def grad_ci_kx(self) -> NDArray:
        """Computes the gradient of the positivity constraint functions evaluated at the
        gain vector k with respect to first k and then x.

        Arguments:
            None

        Returns:
            grad_ci_kx: gradient of positivity constraint functions with respect to k and then x

        """
        return np.zeros((len(self._k_weights),)) * self.ci_gain

    #! TO DO: fix epsilon robustness
    def delta(self, x: NDArray, h: NDArray, Lf: float) -> float:
        """Computes the threshold above which the product LgH*u_max must remain for control viability.
        In other words, in order to be able to satisfy:

        LfH + LgH*u + LkH + alpha(H) >= 0

        it must hold that LgH*u_max >= -LfH - LkH - alpha(H).

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector
        Lf: C-CBF drift term (includes filtered kdot)

        Returns
        -------
        delta = LfH + alpha(H) + LkH

        """
        dhdk = h * np.exp(-self._k_weights * h)

        alpha = 1.0
        epsilon = 100.0  # -- Add robustness epsilon
        epsilon = 10.0  # -- Add robustness epsilon

        delta = -Lf - alpha * self.H(h) - (dhdk @ self._k_dot_f - epsilon)

        return delta if delta > 0 else 0.0

    def grad_delta_k(self, x: NDArray, h: NDArray, Lf: float) -> NDArray:
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
        d2Hdk2 = self.grad_H_kk(h)

        return -dLfHdk - self.grad_H_k(h) - d2Hdk2 @ self._k_dot_f

    def grad_delta_x(self, x: NDArray, h: NDArray) -> NDArray:
        """Computes the threshold above which the product LgH*u_max must remain for control viability.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector

        Returns
        -------
        grad_delta_x: gradient of delta with respect to x

        """
        # # TO DO: define dhdx as a function of x and numerically compute gradient
        # def hx(dx):
        #     return h

        # def dhdx(dx):
        #     return self.dhdx

        # # Define functions to compute gradient numerically
        # def LfH(dx):
        #     return (self._k_weights * np.exp(-self._k_weights * hx(dx))) @ dhdx(dx) @ f(dx)

        # dLfHdx = nd.Jacobian(LfH)(x)[0, :]  # Compute gradient numerically

        #! TO DO: Get d2hdx2 and dfdx symbolically
        d2hdx2 = np.zeros((5, len(self._k_weights), 5))
        dfdx = np.zeros((5, 5))
        dLfHdx = self.dqdx.T @ self.dhdx @ f(x) + self.q @ d2hdx2 @ f(x) + self.q @ self.dhdx @ dfdx

        return -dLfHdx - self.grad_H_x() + self.grad_H_kx(h).T @ self._k_dot_f

    def grad_delta_kk(self, x: NDArray, h: NDArray) -> NDArray:
        """Computes the threshold above which the product LgH*u_max must remain for control viability.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector

        Returns
        -------
        grad_delta_kk: gradient of delta with respect to k twice

        """
        triple_diag = np.zeros((len(self._k_weights), len(self._k_weights), len(self._k_weights)))
        np.fill_diagonal(
            triple_diag, (self._k_weights * h**2 - 2 * h) * np.exp(-self._k_weights * h)
        )
        grad_LfH_kk = triple_diag @ self.dhdx @ f(x)

        return -grad_LfH_kk - self.grad_H_kk(h) + self.grad_H_kkk(h) @ self._k_dot_f

    def grad_delta_kx(self, x: NDArray, h: NDArray) -> NDArray:
        """Computes the threshold above which the product LgH*u_max must remain for control viability.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector

        Returns
        -------
        grad_delta_kx: gradient of delta with respect to k and then x

        """

        # def grad_LfH_k(dx):
        #     return self.dqdk @ self.dhdx @ f(dx)

        # grad_LfH_kx = nd.Jacobian(grad_LfH_k)(x)

        # TO DO: Get d2hdx2 and dfdx symbolically
        d2hdx2 = np.zeros((5, len(self._k_weights), 5))
        dfdx = np.zeros((5, 5))
        grad_LfH_kx = (
            self.d2qdkdx @ self.dhdx @ f(x) + self.q @ d2hdx2 @ f(x) + self.q @ self.dhdx @ dfdx
        )

        return -grad_LfH_kx - self.grad_H_kx(h) + (self.grad_H_kkx(h).T @ self._k_dot_f).T

    def H(self, h: NDArray) -> float:
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
        H = 1 - np.sum(np.exp(-self._k_weights * h))

        return H

    def grad_H_k(self, h: NDArray) -> float:
        """Computes the gradient of the consolidated control barrier function (C-CBF)
        with respect to the weights (k).

        Arguments
        ---------
        h: array of constituent cbfs

        Returns
        -------
        grad_H_k: gradient of C-CBF with respect to k

        """
        grad_H_k = h * np.exp(-self._k_weights * h)

        return grad_H_k

    def grad_H_x(self) -> float:
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
        grad_H_x = self.q @ self.dhdx

        return grad_H_x

    def grad_H_kk(self, h: NDArray) -> float:
        """Computes the gradient of the consolidated control barrier function (C-CBF)
        with respect to the weights (k) twice.

        Arguments
        ---------
        h: array of constituent cbfs

        Returns
        -------
        grad_H_kk: gradient of C-CBF with respect to k twice

        """
        grad_H_kk = -(h**2) * np.exp(-self._k_weights * h)

        return grad_H_kk

    def grad_H_kx(self, h: NDArray) -> float:
        """Computes the gradient of the consolidated control barrier function (C-CBF)
        with respect to the gains (k) and then the state (x).

        Arguments:
            x: state vector
            h: array of constituent cbfs

        Returns:
            grad_H_kx: gradient of C-CBF with respect to k and then x

        """
        grad_H_kx = (
            np.diag(np.exp(-self._k_weights * h)) @ self.dhdx
            - np.diag(h * self._k_weights * np.exp(-self._k_weights * h)) @ self.dhdx
        )

        return grad_H_kx

    #! TO DO: Check whether this is correct
    def grad_H_kkk(self, h: NDArray) -> float:
        """Computes the gradient of the consolidated control barrier function (C-CBF)
        with respect to the weights (k) thrice.

        Arguments
        ---------
        h: array of constituent cbfs

        Returns
        -------
        grad_H_kkk: gradient of C-CBF with respect to k thrice

        """
        filling = (h**3) * np.exp(-self._k_weights * h)
        grad_H_kkk = np.zeros((len(self._k_weights), len(self._k_weights), len(self._k_weights)))
        np.fill_diagonal(grad_H_kkk, filling)

        return grad_H_kkk

    #! TO DO: Check whether this is correct
    def grad_H_kkx(self, h: NDArray) -> float:
        """Computes the gradient of the consolidated control barrier function (C-CBF)
        with respect to the weights (k) twice and state (x) once.

        Arguments
        ---------
        h: array of constituent cbfs

        Returns
        -------
        grad_H_kkx: gradient of C-CBF with respect to k twice

        """
        #! TO DO: Check whether this is correct
        grad_H_kkx = (
            h**2 * self._k_weights * np.exp(-self._k_weights * h)
            - 2 * h * np.exp(-self._k_weights * h)
        )[:, np.newaxis] @ self.dhdx[:, np.newaxis, :]
        # grad_H_kkx = np.zeros((len(self._k_weights), len(self._k_weights), self.dqdx.shape[1]))
        # np.fill_diagonal(grad_H_kkx, filling)

        return grad_H_kkx

    def k_des(self, h: NDArray) -> NDArray:
        """Computes the desired gains k for the constituent cbfs. This can be
        thought of as the nominal adaptation law (unconstrained).

        Arguments:
            h (NDArray): array of constituent cbf values

        Returns:
            k_des (NDArray)

        """
        k_des = self.k_des_gain * h / np.min([np.min(h), 1.0])

        return np.clip(k_des, self.k_min, self.k_max)

    def grad_k_des_x(self, x: NDArray, h: NDArray) -> NDArray:
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

        k_des = self.k_des_gain * h / np.min([h[min_h_idx], 2.0])
        over_k_max = np.where(k_des > self.k_max)[0]
        under_k_min = np.where(k_des < self.k_min)[0]

        if h[min_h_idx] > 2.0:
            grad_k_desired_x = self.k_des_gain * self.dhdx / 2.0
        else:
            # Deal with cases when dhdx very close to zero
            dhdx = self.dhdx
            dhdx[abs(dhdx) <= 1e-9] = 1
            grad_k_desired_x = self.k_des_gain * self.dhdx / dhdx[min_h_idx, :]

        grad_k_desired_x[over_k_max] = 0
        grad_k_desired_x[under_k_min] = 0

        return grad_k_desired_x

    @property
    def k_weights(self) -> NDArray:
        """Getter for _k_weights."""
        return self._k_weights

    @k_weights.setter
    def k_weights(self, newVals: NDArray) -> None:
        """Setter for _k_weights.

        Arguments:
            newVals (NDArray): new/updated kWeights values

        Returns:
            None

        """
        if newVals.shape[0] == self._k_weights.shape[0]:
            self._k_weights = newVals
        else:
            raise ValueError("Error updating k_weights!")

    @property
    def k_dot(self) -> NDArray:
        """Getter for _k_dot."""
        return self._k_dot

    @property
    def k_dot_f(self) -> NDArray:
        """Getter for _k_dot_f."""
        return self._k_dot_f


if __name__ == "__main__":
    nWeights = 5
    uMax = np.array([10.0, 10.0])
    adapt = AdaptationLaw(nWeights, uMax)
