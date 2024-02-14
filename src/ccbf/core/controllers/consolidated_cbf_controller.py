"""consolidated_cbf_controller.py

Provides interface to the ConsolidatedCbfController class.

"""
import time
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

np.random.seed(1)

vehicle = builtins.PROBLEM_CONFIG["vehicle"]
control_level = builtins.PROBLEM_CONFIG["control_level"]
system_model = builtins.PROBLEM_CONFIG["system_model"]
mod = "models." + vehicle + "." + control_level + ".models"

# Programmatic import
try:
    module = import_module(mod)
    globals().update({"f": getattr(module, "f")})
    globals().update({"dfdx": getattr(module, "dfdx")})
    globals().update({"g": getattr(module, "g")})
    globals().update({"dgdx": getattr(module, "dgdx")})
    # globals().update({"xg": getattr(module, "xg")})
    # globals().update({"yg": getattr(module, "yg")})
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
        kZero = 1.0
        nCBF = len(self.cbf_vals)
        self.c_cbf = 100
        self.n_agents = nAgents
        self.filtered_wf = np.zeros((nCBF,))
        self.filtered_wg = np.zeros((nCBF, len(self.u_max)))
        self.k_weights = kZero * np.ones((nCBF,))
        self.k_dot = np.zeros((nCBF,))
        self.k_dot_f = np.zeros((nCBF,))
        self.alpha = self.desired_class_k * 1.0
        self.czero1 = 0
        self.czero2 = 0

        self.adapter = AdaptationLaw(nCBF, u_max, kZero=kZero, alpha=self.alpha)

    def _compute_control(
        self, t: float, z: NDArray, cascaded: bool = False
    ) -> (NDArray, int, str):
        self.u, code, status = super()._compute_control(t, z, cascaded)

        # Update k weights, k_dot
        k_weights, k_dot, k_dot_f = self.adapter.update(self.u, self._dt)
        self.k_weights = k_weights
        self.k_dot = k_dot
        self.k_dot_f = k_dot_f
        self.czero1 = self.adapter.czero_val1
        self.czero2 = self.adapter.czero_val2

        return self.u, code, status

    def formulate_qp(
        self,
        t: float,
        ze: NDArray,
        zr: NDArray,
        u_nom: NDArray,
        ego: int,
        cascade: bool = False,
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
        Q, p = self.compute_objective_qp(u_nom, ze)

        # Compute input constraints of form Au @ u <= bu
        Au, bu = self.compute_input_constraints()

        # Parameters
        na = 1 + len(zr)
        ns = len(ze)
        self.safety = True

        # Initialize inequality constraints
        lci = len(self.cbfs_individual)
        h_array = np.zeros(
            (
                len(
                    self.cbf_vals,
                )
            )
        )
        dhdt_array = np.zeros((len(self.cbf_vals),))
        dhdx_array = np.zeros(
            (
                len(
                    self.cbf_vals,
                ),
                2 * ns,
            )
        )
        d2hdx2_array = np.zeros(
            (
                len(
                    self.cbf_vals,
                ),
                2 * ns,
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
            dhdx_array[cc] = 0
            dhdx_array[cc, :ns] = cbf.dhdx(ze)
            d2hdx2_array[cc, :ns, :ns] = cbf.d2hdx2(ze)

            # Get CBF Lie Derivatives
            Lfh_array[cc] = dhdx_array[cc][:ns] @ f(ze)
            Lgh_array[
                cc, self.n_controls * ego : (ego + 1) * self.n_controls
            ] = dhdx_array[cc][:ns] @ g(
                ze
            )  # Only assign ego control
            if cascade:
                Lgh_array[cc, self.n_controls * ego] = 0.0

            self.dhdt[cc] = dhdt_array[cc]
            self.dhdx[cc] = dhdx_array[cc][:ns]
            self.d2hdx2[cc] = d2hdx2_array[cc][:ns, :ns]
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
                d2hdx2_array[idx] = cbf.d2hdx2(ze, zo)

                # Get CBF Lie Derivatives
                Lfh_array[idx] = dhdx_array[idx][:ns] @ f(ze) + dhdx_array[idx][
                    ns:
                ] @ f(zo)
                Lgh_array[
                    idx, self.n_controls * ego : (ego + 1) * self.n_controls
                ] = dhdx_array[idx][:ns] @ g(ze)
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
                self.d2hdx2[idx] = d2hdx2_array[idx][:ns, :ns]

        # Format inequality constraints
        Ai, bi = self.generate_consolidated_cbf_condition(
            t, ze, h_array, Lfh_array, Lgh_array, ego
        )

        A = np.vstack([Au, Ai])
        b = np.hstack([bu, bi])

        return Q, p, A, b, None, None

    def compute_objective_qp(self, u_nom: NDArray, ze: NDArray) -> (NDArray, NDArray):
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
                    [
                        u_nom.flatten(),
                        np.array(self.n_dec_vars * [self.desired_class_k]),
                    ]
                ),
                ze[:2],
            )
            # Q, p = self.objective(np.append(u_nom.flatten(), self.desired_class_k))
        else:
            Q, p = self.objective(u_nom.flatten(), ze[:2])

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
        self,
        t: float,
        x: NDArray,
        h_array: NDArray,
        Lfh_array: NDArray,
        Lgh_array: NDArray,
        ego: int,
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
        # # Introduce parameters
        # k_ccbf = 0.1
        # # k_ccbf = 0.25
        # # k_ccbf = 1.0

        # # Get C-CBF Value
        # H = self.consolidated_cbf()
        # self.c_cbf = H

        # # Non-centralized agents CBF dynamics become drifts
        # Lgh_uncontrolled = np.copy(Lgh_array[:, :])
        # Lgh_uncontrolled[:, ego * self.n_controls : (ego + 1) * self.n_controls] = 0

        # # Set all Lgh terms other than ego to zero
        # indices_before_ego = np.s_[: ego * self.n_controls]
        # indices_after_ego = np.s_[(ego + 1) * self.n_controls :]
        # Lgh_array[:, indices_before_ego] = 0
        # Lgh_array[:, indices_after_ego] = 0

        # # # Get time-derivatives of gains
        # # dphidh = self.adapter.k_weights * np.exp(-self.adapter.k_weights * self.cbf_vals)
        # # dphidk = self.cbf_vals * np.exp(-self.adapter.k_weights * self.cbf_vals)

        # # Compute C-CBF Dynamics
        # LfH = dphidh @ Lfh_array
        # LgH = dphidh @ Lgh_array
        # LgH_uncontrolled = dphidh @ Lgh_uncontrolled

        # # Tunable CBF Addition
        # Phi = 0.0
        # LfH = LfH + Phi
        # Lf_for_kdot = LfH + dphidk @ self.adapter.k_dot_drift_f
        # Lg_for_kdot = Lgh_array[:, self.n_controls * ego : self.n_controls * (ego + 1)]

        # # Update adapter
        # self.adapter.dhdx = self.dhdx
        # self.adapter.d2hdx2 = self.d2hdx2.swapaxes(0, 2).swapaxes(1, 2)
        # self.adapter.q = dphidh
        # self.adapter.dqdk = np.diag(
        #     (1 - self.k_weights * self.cbf_vals) * np.exp(-self.k_weights * self.cbf_vals)
        # )
        # self.adapter.dqdx = (
        #     self.dhdx.T @ np.diag(-(self.k_weights**2) * np.exp(-self.k_weights * self.cbf_vals))
        # ).T
        # d2qdk2_vals = (self.k_weights * self.cbf_vals**2 - 2 * self.cbf_vals) * np.exp(
        #     -self.k_weights * self.cbf_vals
        # )
        # np.fill_diagonal(self.adapter.d2qdk2, d2qdk2_vals)
        # triple_diag = np.zeros((len(self.k_weights), len(self.k_weights), len(self.k_weights)))
        # np.fill_diagonal(
        #     triple_diag,
        #     (self.k_weights**2 * self.cbf_vals - 2 * self.k_weights)
        #     * np.exp(-self.k_weights * self.cbf_vals),
        # )
        # d2qdkdx = triple_diag @ self.dhdx
        # self.adapter.d2qdkdx = d2qdkdx.swapaxes(0, 1).swapaxes(1, 2)

        # # Prepare Adapter to compute adaptation law
        # self.adapter.precompute(x, self.cbf_vals, Lf_for_kdot, Lg_for_kdot)

        # # Compute drift k_dot
        # k_dot_drift = self.adapter.k_dot_drift(x, self.cbf_vals, Lf_for_kdot, Lg_for_kdot)

        # # Compute controlled k_dot
        # k_dot_contr = self.adapter.k_dot_controlled(x, self.cbf_vals, Lf_for_kdot, Lg_for_kdot)

        ##########################
        ##########################
        # New function starts here
        ##########################
        ##########################

        # Introduce parameters
        k_ccbf = 0.1

        # # Non-centralized agents CBF dynamics become drifts
        # Lgh_uncontrolled = np.copy(Lgh_array[:, :])
        # Lgh_uncontrolled[:, ego * self.n_controls : (ego + 1) * self.n_controls] = 0

        # # Set all Lgh terms other than ego to zero
        # indices_before_ego = np.s_[: ego * self.n_controls]
        # indices_after_ego = np.s_[(ego + 1) * self.n_controls :]
        # Lgh_array[:, indices_before_ego] = 0
        # Lgh_array[:, indices_after_ego] = 0

        # c-cbf partial derivatives (NOT TOTAL)
        dHdh = self.adapter.k_weights * np.exp(-self.adapter.k_weights * self.cbf_vals)
        dHdw = self.cbf_vals * np.exp(-self.adapter.k_weights * self.cbf_vals)
        dHdt = dHdh @ self.dhdt
        dHdx = dHdh @ self.dhdx

        # c-cbf derivative elements
        Hdot_drift = dHdx @ f(x)
        Hdot_contr = dHdx @ g(x)
        if len(Hdot_contr.shape) == 1:
            Hdot_contr = Hdot_contr[:, np.newaxis]

        # tunable CBF addition
        Phi = 0.0 * np.exp(-k_ccbf * self.consolidated_cbf())
        Hdot_drift += Phi
        Lf_for_kdot = Hdot_drift + dHdw @ self.adapter.k_dot_drift_f
        Lg_for_kdot = self.dhdx @ g(x)
        if len(Lg_for_kdot.shape) == 1:
            Lg_for_kdot = Lg_for_kdot[:, np.newaxis]

        # Update adapter
        self.adapter.dhdx = self.dhdx
        self.adapter.d2hdx2 = self.d2hdx2.swapaxes(0, 2).swapaxes(1, 2)
        self.adapter.q = dHdh
        self.adapter.dqdk = np.diag(
            (1 - self.k_weights * self.cbf_vals)
            * np.exp(-self.k_weights * self.cbf_vals)
        )
        self.adapter.dqdx = (
            self.dhdx.T
            @ np.diag(-(self.k_weights**2) * np.exp(-self.k_weights * self.cbf_vals))
        ).T
        d2qdk2_vals = (
            self.k_weights * self.cbf_vals**2 - 2 * self.cbf_vals
        ) * np.exp(-self.k_weights * self.cbf_vals)
        np.fill_diagonal(self.adapter.d2qdk2, d2qdk2_vals)
        triple_diag = np.zeros(
            (len(self.k_weights), len(self.k_weights), len(self.k_weights))
        )
        np.fill_diagonal(
            triple_diag,
            (self.k_weights**2 * self.cbf_vals - 2 * self.k_weights)
            * np.exp(-self.k_weights * self.cbf_vals),
        )
        d2qdkdx = triple_diag @ self.dhdx
        self.adapter.d2qdkdx = d2qdkdx.swapaxes(0, 1).swapaxes(1, 2)

        # Prepare Adapter to compute adaptation law
        self.adapter.precompute(x, self.cbf_vals, Lf_for_kdot, Lg_for_kdot)

        # Compute drift k_dot
        k_dot_drift = self.adapter.k_dot_drift(
            x, self.cbf_vals, Lf_for_kdot, Lg_for_kdot
        )

        # Compute controlled k_dot
        k_dot_contr = self.adapter.k_dot_controlled(
            x, self.cbf_vals, Lf_for_kdot, Lg_for_kdot
        )

        # consolidated cbf H(t, w, x)
        self.c_cbf = H = self.consolidated_cbf()
        # augmented consolidated cbf B(t, w, x) = b * H
        B = self.augmented_consolidated_cbf()

        # partial (NOT TOTAL) derivatives
        dBdt = self.adapter.dbdt * H + self.adapter.b * dHdt
        dBdw = self.adapter.dbdw * H + self.adapter.b * dHdw
        dBdx = self.adapter.dbdx * H + self.adapter.b * dHdx

        dBdt = dHdt
        dBdw = dHdw
        dBdx = dHdx

        Bdot_drift = dBdt + dBdw @ k_dot_drift + dBdx @ f(x)
        Bdot_contr = dBdw @ k_dot_contr + dBdx @ g(x)

        if B < 0 or H < 0:
            pass
            print(f"Time: {t}")
            print(f"h_array: {h_array}")
            # print(f"H: {H:.2f}")
            # print(f"b: {self.adapter.b:.2f}")
            # print(f"Gains: {self.adapter.k_weights}")

        # # CBF Condition (fixed class K)
        # qp_scale = 1e3
        # h_alph = H
        # a_mat = np.append(-Bdot_contr, 0)
        # b_vec = np.array([Bdot_drift + self.alpha * h_alph]).flatten()
        # a_mat *= qp_scale
        # b_vec *= qp_scale

        # CBF Condition (fixed class K)
        qp_scale = 1e3
        h_alph = H
        a_mat = np.append(-Bdot_contr, -h_alph)
        b_vec = np.array([Bdot_drift]).flatten()
        a_mat *= qp_scale
        b_vec *= qp_scale

        return a_mat[:, np.newaxis].T, b_vec

    def consolidated_cbf(self):
        """Computes the value of the consolidated CBF."""
        return 1 - np.sum(np.exp(-self.adapter.k_weights * self.cbf_vals))

    def augmented_consolidated_cbf(self):
        """Computes the value of the consolidated CBF augmented by
        the input constraint function."""
        H = self.consolidated_cbf()
        b = self.adapter.b

        sign_H = np.sign(H)
        sign_b = np.sign(b)
        sgn = 1 if sign_H + sign_b == 2 else -1

        return abs(b * H) * sgn


class AdaptationLaw:
    """Computes the parameter adaptation for the ConsolidatedCbfController
    class.

    Attributes:
        kWeights (NDArray): values of current c-cbf weights k
    """

    def __init__(
        self,
        nWeights: int,
        uMax: NDArray,
        kZero: Optional[float] = 0.5,
        alpha: Optional[float] = 0.1,
    ):
        """Initializes class attributes.

        Arguments:
            nWeights (int): number of weights/CBFs to be consolidated
            uMax (NDArray): maximum control input vector
            kZero (float, Opt)

        """
        nStates = 4  # placeholder

        # time
        self.t = 0.0

        # k weights and derivatives
        self._k_weights = kZero * np.ones((nWeights,))
        self._k_dot = np.zeros((nWeights,))
        self._k_dot_drift = np.zeros((nWeights,))
        self._k_dot_controlled = np.zeros((nWeights, len(uMax)))

        # kdot filter design (2nd order)
        self.wn = 50.0
        self.zeta = 1.0
        self._k_dot_drift_f = np.zeros((nWeights,))
        self._k_2dot_drift_f = np.zeros((nWeights,))
        self._k_3dot_drift_f = np.zeros((nWeights,))
        self._k_dot_cont_f = np.zeros((nWeights, len(uMax)))
        self._k_2dot_cont_f = np.zeros((nWeights, len(uMax)))
        self._k_3dot_cont_f = np.zeros((nWeights, len(uMax)))

        # s relaxation function switch variable
        self.s_on = 0  # 1 if on, 0 if off

        # logging variables
        self.czero_val1 = 0.0
        self.czero_val2 = 0.0

        # q vector and derivatives
        self.q = np.zeros((nWeights,))
        self.dqdk = np.zeros((nWeights, nWeights))
        self.dqdx = np.zeros((nWeights, nStates))
        self.d2qdk2 = np.zeros((nWeights, nWeights, nWeights))
        self.d2qdkdx = np.zeros((nWeights, nStates, nWeights))

        # control contraint matrix
        self.u_max = uMax
        self.U = uMax[:, np.newaxis] @ uMax[np.newaxis, :]

        # dhdx, d2hdx2 matrices
        self.dhdx = np.zeros((nWeights, nStates))
        self.d2hdx2 = np.zeros((nStates, nWeights, nStates))

        # delta terms
        self._delta = None
        self._grad_delta_k = None
        self._grad_delta_x = None
        self._grad_delta_kk = None
        self._grad_delta_kx = None

        # ci terms
        self._ci = None
        self._grad_ci_k = None
        self._grad_ci_x = None
        self._grad_ci_kk = None
        self._grad_ci_kx = None

        # czero terms
        self._czero = None
        self._grad_czero_k = None
        self._grad_czero_x = None
        self._grad_czero_kk = None
        self._grad_czero_kx = None

        # cost terms
        self._grad_cost_kk = None

        # phi terms
        self._grad_phi_k = None
        self._grad_phi_kk = None
        self._grad_phi_kx = None

        # # Gains and Parameters -- Nonlinear 1D alpha!!
        # self.alpha = alpha
        # self.epsilon = 0.0
        # self.wn = 10.0
        # self.k_dot_gain = 1.0
        # self.cost_gain_mat = 1.0 * np.eye(nWeights)
        # self.p_gain_mat = 1 * np.eye(nWeights)
        # self.k_des_gain = 0.5
        # self.k_min = 0.01
        # self.k_max = 10.0
        # self.czero_gain = 1.0
        # self.ci_gain = 1.0

        # Gains and Parameters -- Double Integrator!!
        self.alpha = alpha
        self.epsilon = 0.0
        self.wn = 10.0
        self.k_dot_gain = 1.0
        self.cost_gain_mat = 0.75 * np.eye(nWeights)
        self.p_gain_mat = 1.0 * np.eye(nWeights)
        self.k_des_gain = 1.0
        self.k_min = 0.01
        self.k_max = 100.0
        self.czero_gain = 0.01
        self.ci_gain = 1.0

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
        self.t += dt
        k_weights = self._k_weights + self.compute_kdot(u, dt) * dt

        # Account for numerical instability
        self._k_weights = np.clip(k_weights, 2 * self.k_min, self.k_max)

        return self._k_weights, self._k_dot, self.k_dot_drift_f

    def compute_kdot(self, u: NDArray, dt: float) -> NDArray:
        """Computes the time-derivative k_dot of the k_weights vector.

        Arguments:
            u (NDArray): control input applied to system
            dt: timestep in sec

        Returns:
            k_dot (NDArray): time-derivative of kWeights

        """

        self._k_dot = self._k_dot_drift + self._k_dot_controlled @ u

        self._k_3dot_drift_f = (
            self.wn**2 * (self._k_dot_drift - self._k_dot_drift_f)
            - 2 * self.zeta * self.wn * self._k_2dot_drift_f
        )
        self._k_2dot_drift_f += self._k_3dot_drift_f * dt
        self._k_dot_drift_f += self._k_2dot_drift_f * dt

        self._k_3dot_cont_f = (
            self.wn**2 * (self._k_dot_controlled - self._k_dot_cont_f)
            - 2 * self.zeta * self.wn * self._k_2dot_cont_f
        )
        self._k_2dot_cont_f += self._k_3dot_cont_f * dt
        self._k_dot_cont_f += self._k_2dot_cont_f * dt

        return self._k_dot

    def precompute(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
        """Precomputes terms needed to compute the adaptation law.

        Arguments:
            x (NDArray): state vector
            h (NDArray): vector of candidate CBFs
            Lf (float): C-CBF drift term (includes filtered kdot)
            Lg (NDArray): matrix of stacked Lgh vectors

        Returns:
            None

        """
        # delta terms
        self._delta = self.delta(x, h, Lf)
        self._grad_delta_k = self.grad_delta_k(x, h, Lf)
        self._grad_delta_x = self.grad_delta_x(x, h)
        self._grad_delta_kk = self.grad_delta_kk(x, h)
        self._grad_delta_kx = self.grad_delta_kx(x, h)

        # ci terms
        self._ci = self.ci()
        self._grad_ci_k = self.grad_ci_k()
        self._grad_ci_x = self.grad_ci_x()
        self._grad_ci_kk = self.grad_ci_kk()
        self._grad_ci_kx = self.grad_ci_kx()

        # czero terms
        self._czero = self.czero(x, h, Lf, Lg)
        self._grad_czero_k = self.grad_czero_k(x, h, Lf, Lg)
        self._grad_czero_x = self.grad_czero_x(x, h, Lf, Lg)
        self._grad_czero_kk = self.grad_czero_kk(x, h, Lf, Lg)
        self._grad_czero_kx = self.grad_czero_kx(x, h, Lf, Lg)

        # cost terms
        self._grad_cost_kk = self.grad_cost_kk()

        # phi terms
        self._grad_phi_k = self.grad_phi_k(x, h, Lf, Lg)
        self._grad_phi_kk = self.grad_phi_kk(x, h, Lf, Lg)
        self._grad_phi_kx = self.grad_phi_kx(x, h, Lf, Lg)

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
        # Partition into digestible terms
        P = self._grad_cost_kk
        Q = np.diag(1 / self._ci**2)
        R = (
            self._czero * self._grad_czero_kk
            - self._grad_czero_k[:, np.newaxis] @ self._grad_czero_k[np.newaxis, :]
        ) / self._czero**2
        M = self._grad_phi_kk + P + Q + R
        X = (
            self._czero * self._grad_czero_kx
            - self._grad_czero_k[:, np.newaxis] @ self._grad_czero_x[np.newaxis, :]
        ) / self._czero**2

        k_dot_drift = (
            -self.k_dot_gain
            * np.linalg.inv(M)
            @ (self.p_gain_mat @ self._grad_phi_k + X @ f(x))
        )

        self._k_dot_drift = k_dot_drift

        return self._k_dot_drift

    def k_dot_controlled(
        self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray
    ) -> NDArray:
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
        # Partition into digestible terms
        P = self._grad_cost_kk
        Q = np.diag(1 / self._ci**2)
        R = (
            self._czero * self._grad_czero_kk
            - self._grad_czero_k[:, np.newaxis] @ self._grad_czero_k[np.newaxis, :]
        ) / self._czero**2
        M = self._grad_phi_kk + P + Q + R
        X = (
            self._czero * self._grad_czero_kx
            - self._grad_czero_k[:, np.newaxis] @ self._grad_czero_x[np.newaxis, :]
        ) / self._czero**2

        k_dot_controlled = -self.k_dot_gain * np.linalg.inv(M) @ X @ g(x)

        if len(k_dot_controlled.shape) > 1:
            self._k_dot_controlled = k_dot_controlled
        else:
            self._k_dot_controlled = k_dot_controlled[:, np.newaxis]

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
            - self._grad_ci_k / self._ci
            - self._grad_czero_k / self._czero
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
                self._grad_ci_kk * self._ci
                - self._grad_ci_k[:, np.newaxis] @ self._grad_ci_k[np.newaxis, :]
            )
            / self._ci**2
            - (
                self._grad_czero_kk * self._czero
                - self._grad_czero_k[:, np.newaxis] @ self._grad_czero_k[np.newaxis, :]
            )
            / self._czero**2
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
                self._grad_czero_kx * self._czero
                - self._grad_czero_k[:, np.newaxis] * self._grad_czero_x[np.newaxis, :]
            )
            / self._czero**2
            - (
                self._ci[:, np.newaxis] * self._grad_ci_kx
                - self._grad_ci_k[:, np.newaxis] * self._grad_ci_x[np.newaxis, :]
            )
            / self._ci[:, np.newaxis] ** 2
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
        dhdk = h * np.exp(-self._k_weights * h)
        vector = self.q @ Lg + dhdk @ self._k_dot_cont_f
        czero = vector @ self.U @ vector.T - self._delta**2
        if self.t == 0:
            start = time.time()
            while czero <= self._delta**2 and (time.time() - start) < 5:
                czero = self.k_gradient_descent(x, h, Lf, Lg)
                self._delta = self.delta(x, h, Lf)

        s_func = 0  # -czero / 2 * (1 - np.sqrt(czero**2 + 0.001**2) / czero)

        self.czero_val1 = czero
        self.czero_val2 = (abs(self.q @ Lg) @ self.u_max) - self._delta

        return (czero + s_func * self.s_on) * self.czero_gain

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
        grad_c0_k = (
            2 * self.dqdk @ Lg @ self.U @ Lg.T @ self.q.T
            - 2 * self._delta * self._grad_delta_k
        )

        grad_sfunc_k = grad_c0_k * 0  # (
        # 1 / 2 * grad_c0_k * (self.czero_val1 / np.sqrt(self.czero_val1**2 + 0.001**2))
        # )

        return (grad_c0_k + grad_sfunc_k * self.s_on) * self.czero_gain

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
        dLgdx = self.d2hdx2 @ g(x) + self.dhdx @ dgdx(x)

        grad_c0_x = (
            2 * self.dqdx.T @ Lg @ self.U @ Lg.T @ self.q.T
            + 2 * self.q @ dLgdx @ self.U @ Lg.T @ self.q.T
            - 2 * self._delta * self._grad_delta_x
        )

        grad_sfunc_x = 0 * grad_c0_x
        # (
        #   1 / 2 * grad_c0_x * (self.czero_val1 / np.sqrt(self.czero_val1**2 + 0.001**2))
        # )

        return (grad_c0_x + grad_sfunc_x * self.s_on) * self.czero_gain

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
        if self._delta > 0:
            grad_c0_kk = (
                2 * self.d2qdk2 @ Lg @ self.U @ Lg.T @ self.q.T
                + 2 * self.dqdk @ Lg @ self.U @ Lg.T @ self.dqdk.T
                - 2
                * self._grad_delta_k[:, np.newaxis]
                @ self._grad_delta_k[np.newaxis, :]
                - 2 * self._delta * self._grad_delta_kk
            )

        else:
            grad_c0_kk = (
                2 * self.d2qdk2 @ Lg @ self.U @ Lg.T @ self.q.T
                + 2 * self.dqdk @ Lg @ self.U @ Lg.T @ self.dqdk.T
            )

        # grad_sfunc_kk = grad_c0_kk * (self.czero_val1 / np.sqrt(self.czero_val1**2 + 0.001**2) - 1) +

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
        dLgdx = self.d2hdx2 @ g(x) + self.dhdx @ dgdx(x)

        grad_c0_kx = (
            2 * self.d2qdkdx @ Lg @ self.U @ Lg.T @ self.q.T
            + 2 * self.dqdk @ Lg @ self.U @ Lg.T @ self.dqdx
            + 4 * (self.dqdk @ dLgdx @ self.U @ Lg.T @ self.q.T).T  # [:, np.newaxis]
            - 2
            * (
                self._grad_delta_k[:, np.newaxis] @ self._grad_delta_x[np.newaxis, :]
                + self._delta * self._grad_delta_kx
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
        return np.zeros((len(self._grad_delta_x),)) * self.ci_gain

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
        return np.zeros((len(self._k_weights), len(self._k_weights))) * self.ci_gain

    def grad_ci_kx(self) -> NDArray:
        """Computes the gradient of the positivity constraint functions evaluated at the
        gain vector k with respect to first k and then x.

        Arguments:
            None

        Returns:
            grad_ci_kx: gradient of positivity constraint functions with respect to k and then x

        """
        return np.zeros((len(self._k_weights), len(self._grad_delta_x))) * self.ci_gain

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
        delta = -Lf - self.alpha * self.H(h) + self.epsilon

        return delta if delta > 0 else 0

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

        return -dLfHdk - self.grad_H_k(h) - d2Hdk2 @ self._k_dot_drift_f

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
        dLfHdx = (
            self.dqdx.T @ self.dhdx @ f(x)
            + self.q @ self.d2hdx2 @ f(x)
            + self.q @ self.dhdx @ dfdx(x)
        )

        return -dLfHdx - self.grad_H_x() + self.grad_H_kx(h).T @ self._k_dot_drift_f

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
        triple_diag = np.zeros(
            (len(self._k_weights), len(self._k_weights), len(self._k_weights))
        )
        np.fill_diagonal(
            triple_diag,
            (self._k_weights * h**2 - 2 * h) * np.exp(-self._k_weights * h),
        )
        grad_LfH_kk = triple_diag @ self.dhdx @ f(x)

        return (
            -grad_LfH_kk - self.grad_H_kk(h) + self.grad_H_kkk(h) @ self._k_dot_drift_f
        )

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

        # TO DO: Get dfdx symbolically
        # dfdx = np.zeros((5, 5))
        grad_LfH_kx = (
            self.d2qdkdx @ self.dhdx @ f(x)
            + self.q @ self.d2hdx2 @ f(x)
            + self.q @ self.dhdx @ dfdx(x)
        )

        return (
            -grad_LfH_kx
            - self.grad_H_kx(h)
            + (self.grad_H_kkx(h).T @ self._k_dot_drift_f).T
        )

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
        grad_H_kkk = np.zeros(
            (len(self._k_weights), len(self._k_weights), len(self._k_weights))
        )
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
        grad_H_kkx = (
            h**2 * self._k_weights * np.exp(-self._k_weights * h)
            - 2 * h * np.exp(-self._k_weights * h)
        )[:, np.newaxis] @ self.dhdx[:, np.newaxis, :]

        return grad_H_kkx

    # def k_des(self, h: NDArray) -> NDArray:
    #     """Computes the desired gains k for the constituent cbfs. This can be
    #     thought of as the nominal adaptation law (unconstrained).

    #     Arguments:
    #         h (NDArray): array of constituent cbf values

    #     Returns:
    #         k_des (NDArray)

    #     """
    #     hmin = 0.1
    #     k_des = self.k_des_gain * h / np.min([np.min(h), hmin])

    #     return np.clip(k_des, self.k_min, self.k_max)

    # def grad_k_des_x(self, x: NDArray, h: NDArray) -> NDArray:
    #     """Computes the desired gains k for the constituent cbfs. This can be
    #     thought of as the nominal adaptation law (unconstrained).

    #     Arguments
    #     ---------
    #     h: array of constituent cbf values
    #     x: state vector

    #     Returns
    #     -------
    #     grad_k_desired_x

    #     """
    #     hmin = 0.1
    #     min_h_idx = np.where(h == np.min(h))[0][0]

    #     k_des = self.k_des_gain * h / np.min([h[min_h_idx], hmin])
    #     over_k_max = np.where(k_des > self.k_max)[0]
    #     under_k_min = np.where(k_des < self.k_min)[0]

    #     if h[min_h_idx] > 2.0:
    #         grad_k_desired_x = self.k_des_gain * self.dhdx / hmin
    #     else:
    #         # Deal with cases when dhdx very close to zero
    #         dhdx = self.dhdx
    #         dhdx[abs(dhdx) <= 1e-9] = 1
    #         grad_k_desired_x = self.k_des_gain * self.dhdx / dhdx[min_h_idx, :]

    #     grad_k_desired_x[over_k_max] = 0
    #     grad_k_desired_x[under_k_min] = 0

    #     return grad_k_desired_x

    def k_des(self, h: NDArray) -> NDArray:
        """Computes the desired gains k for the constituent cbfs. This can be
        thought of as the nominal adaptation law (unconstrained).

        Arguments:
            h (NDArray): array of constituent cbf values

        Returns:
            k_des (NDArray)

        """
        h = np.clip(h, 0.01, np.inf)
        k_des = self.k_des_gain / h

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
        h = np.clip(h, 0.01, np.inf)
        k_des = self.k_des_gain / h

        over_k_max = np.where(k_des > self.k_max)[0]
        under_k_min = np.where(k_des < self.k_min)[0]

        grad_k_desired_x = -self.k_des_gain / h[:, np.newaxis] ** 2 * self.dhdx

        grad_k_desired_x[over_k_max] = 0
        grad_k_desired_x[under_k_min] = 0

        return grad_k_desired_x

    # def k_des(self, h: NDArray) -> NDArray:
    #     """Computes the desired gains k for the constituent cbfs. This can be
    #     thought of as the nominal adaptation law (unconstrained).

    #     Arguments:
    #         h (NDArray): array of constituent cbf values

    #     Returns:
    #         k_des (NDArray)

    #     """
    #     k_des = self.k_des_gain * np.ones((len(h),))

    #     return np.clip(k_des, self.k_min, self.k_max)

    # def grad_k_des_x(self, x: NDArray, h: NDArray) -> NDArray:
    #     """Computes the desired gains k for the constituent cbfs. This can be
    #     thought of as the nominal adaptation law (unconstrained).

    #     Arguments
    #     ---------
    #     h: array of constituent cbf values
    #     x: state vector

    #     Returns
    #     -------
    #     grad_k_desired_x

    #     """
    #     grad_k_desired_x = np.zeros((self.dhdx.shape))

    #     return grad_k_desired_x

    # def k_des(self, h: NDArray) -> NDArray:
    #     """Computes the desired gains k for the constituent cbfs. This can be
    #     thought of as the nominal adaptation law (unconstrained).

    #     Arguments:
    #         h (NDArray): array of constituent cbf values

    #     Returns:
    #         k_des (NDArray)

    #     """
    #     k_des = self._k_weights + (self._k_dot_drift_f + self._k_dot_cont_f @ self.u_nom)

    #     return np.clip(k_des, self.k_min, self.k_max)

    # def grad_k_des_x(self, x: NDArray, h: NDArray) -> NDArray:
    #     """Computes the desired gains k for the constituent cbfs. This can be
    #     thought of as the nominal adaptation law (unconstrained).

    #     Arguments
    #     ---------
    #     h: array of constituent cbf values
    #     x: state vector

    #     Returns
    #     -------
    #     grad_k_desired_x

    #     """
    #     grad_k_desired_x = np.zeros((self.dhdx.shape))

    #     return grad_k_desired_x

    def k_gradient_descent(
        self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray
    ) -> float:
        """Runs gradient descent on the k_weights in order to increase the
        control authority at t=0.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector
        Lf: C-CBF drift term (includes filtered kdot)
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        new_czero: updated czero value based on new k_weights

        """
        # line search parameter
        beta = 1.0

        # compute gradient
        grad_c0_k = 2 * self.dqdk @ Lg @ self.U @ Lg.T @ self.q - 2 * self.delta(
            x, h, Lf
        ) * self.grad_delta_k(x, h, Lf)
        if np.sum(abs(grad_c0_k)) == 0:
            grad_c0_k = np.flip(np.random.random(grad_c0_k.shape))
            # grad_c0_k = np.random.random(grad_c0_k.shape)

        # gradient descent
        self._k_weights = np.clip(
            self._k_weights + grad_c0_k * beta, self.k_min, self.k_max
        )

        # compute new quantities
        self.q = self._k_weights * np.exp(-self._k_weights * h)
        self.dqdk = np.diag((1 - self._k_weights * h) * np.exp(-self._k_weights * h))
        dhdk = h * np.exp(-self._k_weights * h)
        vector = self.q @ Lg + dhdk @ self._k_dot_cont_f

        # comput new czero
        czero = vector @ self.U @ vector.T - self.delta(x, h, Lf) ** 2

        return czero

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
    def k_dot_drift_f(self) -> NDArray:
        """Getter for _k_dot_drift_f."""
        return self._k_dot_drift_f

    @property
    def b(self) -> NDArray:
        """Getter for input_constraint_function."""
        return self._czero

    @property
    def dbdt(self) -> NDArray:
        """Getter for input_constraint_function."""
        return 0.0

    @property
    def dbdw(self) -> NDArray:
        """Getter for input_constraint_function."""
        return self._grad_czero_k

    @property
    def dbdx(self) -> NDArray:
        """Getter for input_constraint_function."""
        return self._grad_czero_x


if __name__ == "__main__":
    nWeights = 5
    uMax = np.array([10.0, 10.0])
    adapt = AdaptationLaw(nWeights, uMax)
