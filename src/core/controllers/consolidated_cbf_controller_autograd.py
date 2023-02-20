"""consolidated_cbf_controller.py

Provides interface to the ConsolidatedCbfController class.

"""
import time
import jax.numpy as jnp
from jax import jacfwd, jacrev, jit
from typing import Callable, List, Optional, Tuple
from nptyping import NDArray
from scipy.linalg import block_diag, null_space

# from ..cbfs.cbf import Cbf
from models.model import Model
from core.controllers.cbf_qp_controller import CbfQpController
from core.controllers.controller import Controller

# jnp.random.seed(1)


# def ccbf(z: NDArray, h: List, nStates: int) -> float:
#     """Consolidated CBF. Zero super-level set convention.

#     Arguments:
#         z (NDArray): concatenated time, state, and weight vector (t, x, w)
#         h: (List): list containing constituent cbf functions
#         nStates (int): number of states in the state vector

#     Returns:
#         val (float): value of the ccbf at current (t, x, w)

#     """
#     return 1 - jnp.sum(
#         [jnp.exp(-z[nStates + 1 :] * cbf(z[0], z[1 : nStates + 1])) for cc, cbf in enumerate(h)]
#     )


def smooth_abs(x):
    return 2 * jnp.log(1 + jnp.exp(x)) - x - 2 * jnp.log(2)


def dsmoothabs_dx(x):
    return 2 * (jnp.exp(x) / (1 + jnp.exp(x))) - 1


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
    compute_wdots: computes the adaptation of the gains k

    Class Properties:
    -----------------
    Lots?
    """

    P = None

    def __init__(
        self,
        model: Model,
        nominal_controller: Controller,
        objective_function: Callable,
        cbfs_individual: List,
        cbfs_pairwise: List,
        n_agents: int = 1,
        ignore: List = None,
    ):
        super().__init__(
            model,
            nominal_controller,
            objective_function,
            cbfs_individual,
            cbfs_pairwise,
            n_agents,
            ignore,
        )
        nCBFs = len(self.cbf_vals)
        self.c_cbf = 100
        self.w_dot = jnp.zeros((nCBFs,))
        self.w_dot_f = jnp.zeros((nCBFs,))
        self.alpha = self.desired_class_k * 1.0
        self.czero1 = 0

        # states
        self.n_states = 5

        # cbf
        self.cbfs = cbfs_individual + cbfs_pairwise

        # indices
        tidxs = jnp.s_[0]
        xidxs = jnp.s_[1 : self.n_states + 1]
        widxs = jnp.s_[self.n_states + 1 :]

        # consolidated cbf
        self.H = lambda z: 1 - jnp.sum(
            jnp.array(
                [
                    jnp.exp(-z[self.n_states + 1 + cc] * cbf._h(z[0], z[1 : self.n_states + 1]))
                    for cc, cbf in enumerate(self.cbfs)
                ]
            )
        )
        self.dHdt = lambda z: jit(jacfwd(self.H))(z)[tidxs]
        self.dHdx = lambda z: jit(jacfwd(self.H))(z)[xidxs]
        self.dHdw = lambda z: jit(jacfwd(self.H))(z)[widxs]
        self.d2hdt2 = lambda z: jit(jacfwd(jacrev(self.H)))(z)[tidxs, tidxs]
        self.d2hdtdx = lambda z: jit(jacfwd(jacrev(self.H)))(z)[tidxs, xidxs]
        self.d2hdtdw = lambda z: jit(jacfwd(jacrev(self.H)))(z)[tidxs, widxs]
        self.d2hdx2 = lambda z: jit(jacfwd(jacrev(self.H)))(z)[xidxs, xidxs]
        self.d2hdxdw = lambda z: jit(jacfwd(jacrev(self.H)))(z)[xidxs, widxs]
        self.d2hdw2 = lambda z: jit(jacfwd(jacrev(self.H)))(z)[widxs, widxs]
        self.d3hdw3 = lambda z: jit(jacrev(jacfwd(jacrev(self.H))))(z)[widxs, widxs, widxs]
        self.d3hdw2dx = lambda z: jit(jacrev(jacfwd(jacrev(self.H))))(z)[widxs, widxs, xidxs]

        # test
        t = 1.0
        x = jnp.array([1.0, 0.0, 0.0, 0.5, 0.0])
        w = jnp.ones((8,))
        vec = jnp.hstack([t, x, w])
        print(self.H(vec))
        print(self.dHdt(vec))
        print(self.dHdx(vec))
        print(self.dHdw(vec))
        print(self.d2hdt2(vec).shape)
        print(self.d2hdtdx(vec).shape)
        print(self.d2hdtdw(vec).shape)
        print(self.d2hdx2(vec).shape)
        print(self.d2hdxdw(vec).shape)
        print(self.d2hdw2(vec).shape)
        print(self.d3hdw3(vec).shape)
        print(self.d3hdw2dx(vec).shape)

        # initialize adaptation law
        kZero = 1.0
        self.w_weights = kZero * jnp.ones((nCBFs,))
        self.w_des = kZero * jnp.ones((nCBFs,))
        self.adapter = AdaptationLaw(nCBFs, u_max, kZero=kZero, alpha=self.alpha)

        # assign adapter symbolic properties
        self.adapter.t_sym = self.t_sym
        self.adapter.w_sym = self.w_sym
        self.adapter.x_sym = self.x_sym
        self.adapter.all_sym = self.all_sym
        self.adapter.H_symbolic = H_symbolic
        self.adapter.dHdt_symbolic = dHdt_symbolic
        self.adapter.dHdw_symbolic = dHdw_symbolic
        self.adapter.dHdx_symbolic = dHdx_symbolic
        self.adapter.d2Hdw2_symbolic = d2Hdw2_symbolic
        self.adapter.d2Hdwdx_symbolic = d2Hdwdx_symbolic
        self.adapter.d2Hdwdt_symbolic = d2Hdwdt_symbolic
        self.adapter.d3Hdw3_symbolic = d3Hdw3_symbolic
        self.adapter.d3Hdw2dx_symbolic = d3Hdw2dx_symbolic

        # assign adapter functions
        self.adapter.H = self.H
        self.adapter.dHdt = self.dHdt
        self.adapter.dHdw = self.dHdw
        self.adapter.dHdx = self.dHdx
        self.adapter.d2Hdw2 = self.d2Hdw2
        self.adapter.d2Hdwdx = self.d2Hdwdx
        self.adapter.d2Hdwdt = self.d2Hdwdt
        self.adapter.d3Hdw3 = self.d3Hdw3
        self.adapter.d3Hdw2dx = self.d3Hdw2dx

        # finish adapter setup
        self.adapter.setup()

        self.adapter.dt = self._dt

    def _compute_control(self, t: float, z: NDArray, cascaded: bool = False) -> (NDArray, int, str):
        self.u, code, status = super()._compute_control(t, z, cascaded)

        if self.adapter.dt is None:
            self.adapter.dt = self._dt

        # Update k weights, w_dot
        w_weights, w_dot, w_dot_f = self.adapter.update(self.u, self._dt)
        self.w_weights = w_weights
        self.w_des = self.adapter.w_desired
        self.w_dot = w_dot
        self.w_dot_f = w_dot_f
        self.czero1 = self.adapter.czero_val1
        self.czero2 = self.adapter.czero_val2

        return self.u, code, status

    def formulate_qp(
        self, t: float, ze: NDArray, zr: NDArray, u_nom: NDArray, ego: int, cascade: bool = False
    ) -> (NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, float):
        """Configures the Quadratic Program parameters (Q, p for objective function, A, b for inequality constraints,
        G, h for equality constraints).

        """
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
        h_array = jnp.zeros(
            (
                len(
                    self.cbf_vals,
                )
            )
        )
        dhdt_array = jnp.zeros((len(self.cbf_vals),))
        dhdx_array = jnp.zeros(
            (
                len(
                    self.cbf_vals,
                ),
                2 * ns,
            )
        )
        d2hdtdx_array = jnp.zeros((len(self.cbf_vals), 2 * ns))
        d2hdx2_array = jnp.zeros(
            (
                len(
                    self.cbf_vals,
                ),
                2 * ns,
                2 * ns,
            )
        )
        Lfh_array = jnp.zeros(
            (
                len(
                    self.cbf_vals,
                )
            )
        )
        Lgh_array = jnp.zeros((len(self.cbf_vals), u_nom.flatten().shape[0]))

        # Iterate over individual CBF constraints
        for cc, cbf in enumerate(self.cbfs_individual):
            h0 = cbf.h0(t, ze)
            h_array[cc] = cbf.h(t, ze)
            dhdt_array[cc] = cbf.dhdt(t, ze)
            dhdx_array[cc, :ns] = cbf.dhdx(t, ze)
            d2hdtdx_array[cc, :ns] = cbf.d2hdtdx(t, ze)
            d2hdx2_array[cc, :ns, :ns] = cbf.d2hdx2(t, ze)

            # Get CBF Lie Derivatives
            Lfh_array[cc] = dhdx_array[cc][:ns] @ f(ze)
            Lgh_array[cc, self.n_controls * ego : (ego + 1) * self.n_controls] = dhdx_array[cc][
                :ns
            ] @ g(
                ze
            )  # Only assign ego control
            if cascade:
                Lgh_array[cc, self.n_controls * ego] = 0.0

            self.dhdt[cc] = dhdt_array[cc]
            self.dhdx[cc] = dhdx_array[cc][:ns]
            self.d2hdtdx[cc] = d2hdtdx_array[cc][:ns]
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

                h0 = cbf.h0(t, ze, zo)
                h_array[idx] = cbf.h(t, ze, zo)
                dhdx_array[idx] = cbf.dhdx(t, ze, zo)
                d2hdx2_array[idx] = cbf.d2hdx2(t, ze, zo)

                # Get CBF Lie Derivatives
                Lfh_array[idx] = dhdx_array[idx][:ns] @ f(ze) + dhdx_array[idx][ns:] @ f(zo)
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
                self.d2hdx2[idx] = d2hdx2_array[idx][:ns, :ns]

        # Format inequality constraints
        Ai, bi = self.generate_consolidated_cbf_condition(t, ze, h_array, Lfh_array, Lgh_array, ego)

        A = jnp.vstack([Au, Ai])
        b = jnp.hstack([bu, bi])

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
                jnp.concatenate(
                    [u_nom.flatten(), jnp.array(self.n_dec_vars * [self.desired_class_k])]
                ),
                ze,
            )
            # Q, p = self.objective(jnp.append(u_nom.flatten(), self.desired_class_k))
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
            bu = jnp.append(
                jnp.array(self.n_agents * [self.bu]).flatten(),
                self.n_dec_vars * [self.max_class_k, 0],
            )

        else:
            Au = block_diag(*(self.n_agents) * [self.au])
            bu = jnp.array(self.n_agents * [self.bu]).flatten()

        # return 0 * Au, 1 * abs(bu)
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
            wdot: array of time-derivatives of k_gains
        """
        # # Introduce parameters
        # k_ccbf = 0.1
        # # k_ccbf = 0.25
        # # k_ccbf = 1.0

        # # Get C-CBF Value
        # H = self.consolidated_cbf()
        # self.c_cbf = H

        # # Non-centralized agents CBF dynamics become drifts
        # Lgh_uncontrolled = jnp.copy(Lgh_array[:, :])
        # Lgh_uncontrolled[:, ego * self.n_controls : (ego + 1) * self.n_controls] = 0

        # # Set all Lgh terms other than ego to zero
        # indices_before_ego = jnp.s_[: ego * self.n_controls]
        # indices_after_ego = jnp.s_[(ego + 1) * self.n_controls :]
        # Lgh_array[:, indices_before_ego] = 0
        # Lgh_array[:, indices_after_ego] = 0

        # # # Get time-derivatives of gains
        # # dphidh = self.adapter.w_weights * jnp.exp(-self.adapter.w_weights * self.cbf_vals)
        # # dphidk = self.cbf_vals * jnp.exp(-self.adapter.w_weights * self.cbf_vals)

        # # Compute C-CBF Dynamics
        # LfH = dphidh @ Lfh_array
        # LgH = dphidh @ Lgh_array
        # LgH_uncontrolled = dphidh @ Lgh_uncontrolled

        # # Tunable CBF Addition
        # Phi = 0.0
        # LfH = LfH + Phi
        # Lf_for_wdot = LfH + dphidk @ self.adapter.w_dot_drift_f
        # Lg_for_wdot = Lgh_array[:, self.n_controls * ego : self.n_controls * (ego + 1)]

        # # Update adapter
        # self.adapter.dhdx = self.dhdx
        # self.adapter.d2hdx2 = self.d2hdx2.swapaxes(0, 2).swapaxes(1, 2)
        # self.adapter.q = dphidh
        # self.adapter.dqdk = jnp.diag(
        #     (1 - self.w_weights * self.cbf_vals) * jnp.exp(-self.w_weights * self.cbf_vals)
        # )
        # self.adapter.dqdx = (
        #     self.dhdx.T @ jnp.diag(-(self.w_weights**2) * jnp.exp(-self.w_weights * self.cbf_vals))
        # ).T
        # d2qdk2_vals = (self.w_weights * self.cbf_vals**2 - 2 * self.cbf_vals) * jnp.exp(
        #     -self.w_weights * self.cbf_vals
        # )
        # jnp.fill_diagonal(self.adapter.d2qdk2, d2qdk2_vals)
        # triple_diag = jnp.zeros((len(self.w_weights), len(self.w_weights), len(self.w_weights)))
        # jnp.fill_diagonal(
        #     triple_diag,
        #     (self.w_weights**2 * self.cbf_vals - 2 * self.w_weights)
        #     * jnp.exp(-self.w_weights * self.cbf_vals),
        # )
        # d2qdkdx = triple_diag @ self.dhdx
        # self.adapter.d2qdkdx = d2qdkdx.swapaxes(0, 1).swapaxes(1, 2)

        # # Prepare Adapter to compute adaptation law
        # self.adapter.precompute(x, self.cbf_vals, Lf_for_wdot, Lg_for_wdot)

        # # Compute drift w_dot
        # w_dot_drift = self.adapter.w_dot_drift(x, self.cbf_vals, Lf_for_wdot, Lg_for_wdot)

        # # Compute controlled w_dot
        # w_dot_contr = self.adapter.w_dot_controlled(x, self.cbf_vals, Lf_for_wdot, Lg_for_wdot)

        ##########################
        ##########################
        # New function starts here
        ##########################
        ##########################

        # if t == 0:
        #     kZero = -1 * jnp.log(1 / (2 * len(self.cbf_vals))) / self.cbf_vals
        #     self.adapter.w_weights = kZero
        #     self.w_weights = self.adapter.w_weights

        # Introduce parameters
        k_ccbf = 0.1

        # # Non-centralized agents CBF dynamics become drifts
        # Lgh_uncontrolled = jnp.copy(Lgh_array[:, :])
        # Lgh_uncontrolled[:, ego * self.n_controls : (ego + 1) * self.n_controls] = 0

        # # Set all Lgh terms other than ego to zero
        # indices_before_ego = jnp.s_[: ego * self.n_controls]
        # indices_after_ego = jnp.s_[(ego + 1) * self.n_controls :]
        # Lgh_array[:, indices_before_ego] = 0
        # Lgh_array[:, indices_after_ego] = 0

        # loop twice at t=0 for finding initial weights
        for ii in range(1 + (t == 0)):

            if t == 0:
                self.adapter.w_weights = jnp.clip(
                    h_array, self.adapter.k_min * 1.01, self.adapter.k_max * 0.99
                )

            self.w_weights = self.adapter.w_weights

            # c-cbf partial derivatives (NOT TOTAL)
            e_vector = jnp.exp(-self.adapter.w_weights * self.cbf_vals)
            dHdh = self.adapter.w_weights * e_vector
            dHdw = self.cbf_vals * e_vector
            dHdt = dHdh @ self.dhdt
            dHdx = dHdh @ self.dhdx

            # # c-cbf derivative elements
            # Hdot_drift = dHdx @ f(x)
            # Hdot_contr = dHdx @ g(x)
            # if len(Hdot_contr.shape) == 1:
            #     Hdot_contr = Hdot_contr[:, jnp.newaxis]

            # tunable CBF addition
            discretization_error = 2e-1
            Phi = 0.0 * jnp.exp(-k_ccbf * self.consolidated_cbf())
            Lt_for_wdot = self.dhdt
            Lf_for_wdot = self.dhdx @ f(x) + Phi - discretization_error
            Lg_for_wdot = self.dhdx @ g(x)
            if len(Lg_for_wdot.shape) == 1:
                Lg_for_wdot = Lg_for_wdot[:, jnp.newaxis]

            # Update adapter
            self.adapter.dhdt = self.dhdt
            self.adapter.dhdx = self.dhdx
            self.adapter.d2hdtdx = self.d2hdtdx
            self.adapter.d2hdx2 = self.d2hdx2

            # Figure out other symbolic params
            self.adapter.q = dHdh
            self.adapter.dqdk = jnp.diag(
                (1 - self.w_weights * self.cbf_vals) * jnp.exp(-self.w_weights * self.cbf_vals)
            )
            self.adapter.dqdh = jnp.diag(
                -(self.w_weights**2) * jnp.exp(-self.w_weights * self.cbf_vals)
            )
            d2qdk2_vals = (self.w_weights * self.cbf_vals**2 - 2 * self.cbf_vals) * jnp.exp(
                -self.w_weights * self.cbf_vals
            )
            jnp.fill_diagonal(self.adapter.d2qdk2, d2qdk2_vals)
            triple_diag = jnp.zeros((len(self.w_weights), len(self.w_weights), len(self.w_weights)))
            jnp.fill_diagonal(
                triple_diag,
                (self.w_weights**2 * self.cbf_vals - 2 * self.w_weights)
                * jnp.exp(-self.w_weights * self.cbf_vals),
            )
            d2qdkdx = triple_diag @ self.dhdx
            self.adapter.d2qdkdx = d2qdkdx

            # Prepare Adapter to compute adaptation law
            self.adapter.precompute(x, self.cbf_vals, Lt_for_wdot, Lf_for_wdot, Lg_for_wdot)

        # Compute controlled w_dot
        w_dot_contr = self.adapter.w_dot_controlled(x, self.cbf_vals, Lf_for_wdot, Lg_for_wdot)
        # w_dot_contr = self.adapter._w_dot_contr_f

        # Compute drift w_dot
        w_dot_drift = self.adapter.w_dot_drift(x, self.cbf_vals, Lf_for_wdot, Lg_for_wdot)
        # w_dot_drift = self.adapter._w_dot_drift_f

        # # Adjust learning gain
        # w_dot_drift, w_dot_contr = self.adapter.adjust_learning_gain(x, H, dBdw, dBdx)

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

        Bdot_drift = dBdt + dBdw @ w_dot_drift + dBdx @ f(x)
        Bdot_contr = dBdw @ w_dot_contr + dBdx @ g(x)

        # if Bdot_drift < 0:
        #     print(-Bdot_contr, Bdot_drift + self.alpha * (H - 0.1) ** 3)

        # if B < 0 or H < 0:
        # if H < 0:
        #     print(f"Time: {t}")
        #     print(f"H: {H}")
        #     print(f"h_array: {h_array}")
        # a_mat = jnp.append(-Bdot_contr * 0, 0)
        # b_vec = jnp.array([-1e50]).flatten()
        # return a_mat[:, jnp.newaxis].T, b_vec
        # print(f"H: {H:.2f}")
        # print(f"b: {self.adapter.b:.2f}")
        # print(f"Gains: {self.adapter.w_weights}")

        # # CBF Condition (fixed class K)
        # qp_scale = 1e3
        # a_mat = jnp.append(-Bdot_contr, 0)
        # b_vec = jnp.array([Bdot_drift + self.alpha * h_alph]).flatten()
        # a_mat *= qp_scale
        # b_vec *= qp_scale

        # CBF Condition (fixed class K)
        qp_scale = 1 / jnp.max([1e-6, abs(self.adapter.b)])
        # h_alph = B**3
        # h_alph = (H - 0.1) ** 3
        # print(f"Margin: {margin}")
        exp = 3
        h_alph = H**exp
        if exp == 1:
            self.adapter.cubic = False
        elif exp == 3:
            self.adapter.cubic = True
        a_mat = jnp.append(-Bdot_contr, -h_alph)
        b_vec = jnp.array([Bdot_drift]).flatten()
        a_mat *= qp_scale
        b_vec *= qp_scale

        # a_mat = jnp.append(-Bdot_contr, 0)
        # b_vec = jnp.array([Bdot_drift + self.alpha * h_alph]).flatten()
        # a_mat *= qp_scale
        # b_vec *= qp_scale

        bdot_drift = self.adapter.dbdt + self.adapter.dbdw @ w_dot_drift + self.adapter.dbdx @ f(x)
        bdot_contr = self.adapter.dbdw @ w_dot_contr + self.adapter.dbdx @ g(x)

        # a_mat_b = jnp.append(-bdot_contr, 0)
        # b_vec_b = jnp.array([bdot_drift + self.alpha * self.adapter.b**3]).flatten()

        # return jnp.vstack([a_mat, a_mat_b]), jnp.hstack([b_vec, b_vec_b])
        return a_mat[:, jnp.newaxis].T, b_vec

    def consolidated_cbf(self):
        """Computes the value of the consolidated CBF."""
        return 1 - jnp.sum(jnp.exp(-self.adapter.w_weights * self.cbf_vals))

    def augmented_consolidated_cbf(self):
        """Computes the value of the consolidated CBF augmented by
        the input constraint function."""
        H = self.consolidated_cbf()
        b = self.adapter.b

        sign_H = jnp.sign(H)
        sign_b = jnp.sign(b)
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
        nStates = 5  # placeholder
        self.n_weights = nWeights

        # time
        self.t = 0.0

        # class K parameters
        self.alpha = alpha
        self.cubic = False

        # k weights and derivatives
        self._w_weights = kZero * jnp.ones((nWeights,))
        self._w_desired = kZero * jnp.ones((nWeights,))
        self._w_dot = jnp.zeros((nWeights,))
        self._w_dot_drift = jnp.zeros((nWeights,))
        self._w_dot_contr = jnp.zeros((nWeights, len(uMax)))

        # s relaxation function switch variable
        self.s_on = 0  # 1 if on, 0 if off

        # logging variables
        self.czero_val1 = 0.0
        self.czero_val2 = 0.0

        # q vector and derivatives
        self.q = jnp.zeros((nWeights,))
        self.dqdk = jnp.zeros((nWeights, nWeights))
        self.dqdh = jnp.zeros((nWeights, nWeights))
        self.d2qdk2 = jnp.zeros((nWeights, nWeights, nWeights))
        self.d2qdkdh = jnp.zeros((nWeights, nWeights, nWeights))

        # control contraint matrix
        self.u_max = uMax
        self.U = uMax[:, jnp.newaxis] @ uMax[jnp.newaxis, :]

        # wdot filter design (2nd order)
        self.wn = 1.0
        self.zeta = 0.707  # butterworth
        self._filter_order = 2
        self.filter_init()

        # dhdx, d2hdx2 matrices
        self.dhdt = jnp.zeros((nWeights,))
        self.d2hdtdx = jnp.zeros(
            (
                nWeights,
                nStates,
            )
        )
        self.dhdx = jnp.zeros((nWeights, nStates))
        self.d2hdx2 = jnp.zeros((nWeights, nStates, nStates))

        # symbolic placeholders
        self.t_sym = None
        self.w_sym = None
        self.x_sym = None
        self.all_sym = None
        self.H_symbolic = None
        self.dHdt_symbolic = None
        self.dHdw_symbolic = None
        self.dHdx_symbolic = None
        self.d2Hdw2_symbolic = None
        self.d2Hdwdx_symbolic = None
        self.d2Hdwdt_symbolic = None
        self.d3Hdw3_symbolic = None
        self.d3Hdw2dx_symbolic = None

        # function placeholders
        self.H = None
        self.dHdt = None
        self.dHdw = None
        self.dHdx = None
        self.d2Hdw2 = None
        self.d2Hdwdx = None
        self.d2Hdwdt = None
        self.d3Hdw3 = None
        self.d3Hdw2dx = None

        # delta terms
        self._delta = None
        self._grad_delta_k = None
        self._grad_delta_x = None
        self._grad_delta_t = None
        self._grad_delta_kk = None
        self._grad_delta_kx = None
        self._grad_delta_kt = None

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
        self._grad_czero_t = None
        self._grad_czero_kk = None
        self._grad_czero_kx = None
        self._grad_czero_kt = None

        # cost terms
        self._grad_cost_kk = None

        # phi terms
        self._grad_phi_k = None
        self._grad_phi_kk = None
        self._grad_phi_kx = None
        self._grad_phi_kt = None

        # Gains and Parameters -- Testing
        self.alpha = alpha
        self.eta_mu = 0.25
        self.eta_nu = 0.25
        self.w_dot_gain = 1
        self.cost_gain_mat = 1 * jnp.eye(nWeights)
        self.pstar = jnp.ones((nWeights,))
        self.p_gain_mat = 1 * jnp.eye(nWeights)
        self.w_des_gain = 1.0
        self.w_min = 0.01
        self.w_max = 50.0
        self.czero_gain = 1.0
        self.ci_gain = 1.0

        # test
        t = 1.0
        x = jnp.array([1.0, 0.0, 0.0, 0.5, 0.0])
        w = jnp.ones((8,))
        vec = jnp.hstack([t, x, w])
        func = self.b1
        print(func(vec))
        print(self.dHdt(vec))
        print(self.dHdx(vec))
        print(self.dHdw(vec))

        # # symbolic functions
        # self.J_sym_func = (
        #     1
        #     / 2
        #     * (
        #         (self.w_syms - self.wdes_sym_func).T
        #         @ self.p_gain_mat
        #         @ (self.w_sym - self.wdes_syms_func)
        #     )
        # )
        # self.b_min_func = self._w_min - self.w_syms

    def setup(self) -> None:
        """Generates symbolic functions for the cost function and feasible region
        of the optimization problem.

        Arguments:
            None

        Returns:
            None

        """
        # self.setup_cost()
        self.setup_b1()
        self.setup_b2()
        self.setup_b3()

    def setup_cost(self) -> None:
        """Generates symbolic functions for the cost function and feasible region
        of the optimization problem.

        Arguments:
            None

        Returns:
            None

        """
        # symbolics for cost function
        c_symbolic = (
            1 / 2 * (self.w_sym - self.w_des).T @ self.p_gain_mat @ (self.w_sym - self.w_des)
        )
        dcdt_symbolic = (se.DenseMatrix([c_symbolic]).jacobian(se.DenseMatrix(self.t_sym))).T
        dcdw_symbolic = (se.DenseMatrix([c_symbolic]).jacobian(se.DenseMatrix(self.w_sym))).T
        dcdx_symbolic = (se.DenseMatrix([c_symbolic]).jacobian(se.DenseMatrix(self.x_sym))).T
        d2cdw2_symbolic = dcdw_symbolic.jacobian(se.DenseMatrix(self.w_sym))
        d2cdwdx_symbolic = dcdw_symbolic.jacobian(se.DenseMatrix(self.x_sym))
        d2cdwdt_symbolic = dcdw_symbolic.jacobian(se.DenseMatrix(self.t_sym))

        # callable c-cbf symbolic functions
        self._c = symbolic_cbf_wrapper_singleagent(c_symbolic, self.all_sym)
        self._dcdt = symbolic_cbf_wrapper_singleagent(dcdt_symbolic, self.all_sym)
        self._dcdw = symbolic_cbf_wrapper_singleagent(dcdw_symbolic, self.all_sym)
        self._dcdx = symbolic_cbf_wrapper_singleagent(dcdx_symbolic, self.all_sym)
        self._d2cdw2 = symbolic_cbf_wrapper_singleagent(d2cdw2_symbolic, self.all_sym)
        self._d2cdwdx = symbolic_cbf_wrapper_singleagent(d2cdwdx_symbolic, self.all_sym)
        self._d2cdwdt = symbolic_cbf_wrapper_singleagent(d2cdwdt_symbolic, self.all_sym)

    def setup_b1(self) -> None:
        """Generates symbolic functions bounding the weights from below.

        Arguments:
            None

        Returns:
            None

        """
        # w > w_min constraints
        self.b1 = lambda z: jnp.array(
            [self.w_min - z[self.n_states + 1 + cc] for cc, _ in enumerate(self.cbfs)]
        )
        self.db1dt = lambda z: jacfwd(self.b1)(z)[0]
        self.db1dx = lambda z: jacfwd(self.b1)(z)[1 : self.n_states + 1]
        self.db1dw = lambda z: jacfwd(self.b1)(z)[self.n_states + 1 :]

    def setup_b2(self) -> None:
        """Generates symbolic functions for bounding the weights from above.

        Arguments:
            None

        Returns:
            None

        """
        # w < w_max constraints
        self.b2 = lambda z: jnp.array(
            [z[self.n_states + 1 + cc] - self.w_max for cc, _ in enumerate(self.cbfs)]
        )
        self.db2dt = lambda z: jacfwd(self.b2)(z)[0]
        self.db2dx = lambda z: jacfwd(self.b2)(z)[1 : self.n_states + 1]
        self.db2dw = lambda z: jacfwd(self.b2)(z)[self.n_states + 1 :]

    def setup_b3(self) -> None:
        """Generates symbolic functions for validating the C-CBF.

        Arguments:
            None

        Returns:
            None

        """
        # symbolic functions
        delta_symbolic_a = (
            se.DenseMatrix([self.eta_mu + self.eta_nu])
            - self.dHdt_symbolic
            - self.dHdx_symbolic.T @ f(jnp.zeros((1,)), True)
            - self.alpha * se.DenseMatrix([self.H_symbolic])
        )
        delta_symbolic_b = -self.dHdw_symbolic
        q_symbolic_a = self.dHdx_symbolic.T @ g(jnp.zeros((1,)), True)
        q_symbolic_b = self.dHdw_symbolic

        # delta = (
        #     lambda ag: symbolic_cbf_wrapper_singleagent(delta_symbolic, self.all_sym)(ag)
        #     + symbolic_cbf_wrapper_singleagent(self.dHdw_symbolic, self.all_sym)(ag).T
        #     @ self._w_dot_drift_f
        # )

        # component of b3 function without filtered variables
        b3_symbolic_a = delta_symbolic_a - smooth_abs(q_symbolic_a).T @ Matrix(self.u_max)
        db3dt_symbolic_a = (se.DenseMatrix([b3_symbolic_a]).jacobian(se.DenseMatrix(self.t_sym))).T
        db3dw_symbolic_a = (se.DenseMatrix([b3_symbolic_a]).jacobian(se.DenseMatrix(self.w_sym))).T
        db3dx_symbolic_a = (se.DenseMatrix([b3_symbolic_a]).jacobian(se.DenseMatrix(self.x_sym))).T
        d2b3dw2_symbolic_a = db3dw_symbolic_a.jacobian(se.DenseMatrix(self.w_sym))
        d2b3dwdx_symbolic_a = db3dw_symbolic_a.jacobian(se.DenseMatrix(self.x_sym))
        d2b3dwdt_symbolic_a = db3dw_symbolic_a.jacobian(se.DenseMatrix(self.t_sym))
        b3_a = symbolic_cbf_wrapper_singleagent(b3_symbolic_a, self.all_sym)
        db3dt_a = symbolic_cbf_wrapper_singleagent(db3dt_symbolic_a, self.all_sym)
        db3dw_a = symbolic_cbf_wrapper_singleagent(db3dw_symbolic_a, self.all_sym)
        db3dx_a = symbolic_cbf_wrapper_singleagent(db3dx_symbolic_a, self.all_sym)
        d2b3dw2_a = symbolic_cbf_wrapper_singleagent(d2b3dw2_symbolic_a, self.all_sym)
        d2b3dwdx_a = symbolic_cbf_wrapper_singleagent(d2b3dwdx_symbolic_a, self.all_sym)
        d2b3dwdt_a = symbolic_cbf_wrapper_singleagent(d2b3dwdt_symbolic_a, self.all_sym)

        # component of b3 function with filtered mu
        b3_symbolic_b = delta_symbolic_b
        db3dt_symbolic_b = se.DenseMatrix([b3_symbolic_b]).jacobian(se.DenseMatrix(self.t_sym))
        db3dw_symbolic_b = se.DenseMatrix([b3_symbolic_b]).jacobian(se.DenseMatrix(self.w_sym))
        db3dx_symbolic_b = se.DenseMatrix([b3_symbolic_b]).jacobian(se.DenseMatrix(self.x_sym))
        #! these need to be fixed
        # d2b3dw2_symbolic_b = db3dw_symbolic_b.jacobian(se.DenseMatrix(self.w_sym))
        # d2b3dwdx_symbolic_b = db3dw_symbolic_b.jacobian(se.DenseMatrix(self.x_sym))
        # d2b3dwdt_symbolic_b = db3dw_symbolic_b.jacobian(se.DenseMatrix(self.t_sym))

        d2b3dw2_symbolic_b = se.DenseMatrix(
            [
                db3dw_symbolic_b[:, ii].jacobian(se.DenseMatrix(self.w_sym)).T
                for ii in range(self.n_weights)
            ]
        )
        d2b3dwdx_symbolic_b = se.DenseMatrix(
            [
                db3dw_symbolic_b[:, ii].jacobian(se.DenseMatrix(self.x_sym)).T
                for ii in range(self.n_weights)
            ]
        )
        d2b3dwdt_symbolic_b = se.DenseMatrix(
            [
                db3dw_symbolic_b[:, ii].jacobian(se.DenseMatrix(self.t_sym)).T
                for ii in range(self.n_weights)
            ]
        )

        b3_b = symbolic_cbf_wrapper_singleagent(b3_symbolic_b, self.all_sym)
        db3dt_b = symbolic_cbf_wrapper_singleagent(db3dt_symbolic_b, self.all_sym)
        db3dw_b = symbolic_cbf_wrapper_singleagent(db3dw_symbolic_b, self.all_sym)
        db3dx_b = symbolic_cbf_wrapper_singleagent(db3dx_symbolic_b, self.all_sym)
        d2b3dw2_b = lambda v: symbolic_cbf_wrapper_singleagent(d2b3dw2_symbolic_b, self.all_sym)(
            v
        ).reshape((self.n_weights, self.n_weights, self.n_weights))
        d2b3dwdx_b = lambda v: symbolic_cbf_wrapper_singleagent(d2b3dwdx_symbolic_b, self.all_sym)(
            v
        ).reshape((self.n_weights, self.n_weights, self.n_states))
        d2b3dwdt_b = lambda v: symbolic_cbf_wrapper_singleagent(d2b3dwdt_symbolic_b, self.all_sym)(
            v
        ).reshape((self.n_weights, self.n_weights))

        # component of b3 function with filtered nu
        b3_symbolic_c = -smooth_abs(q_symbolic_b) @ self.u_max
        db3dt_symbolic_c = (se.DenseMatrix([b3_symbolic_c]).jacobian(se.DenseMatrix(self.t_sym))).T
        db3dw_symbolic_c = (se.DenseMatrix([b3_symbolic_c]).jacobian(se.DenseMatrix(self.w_sym))).T
        db3dx_symbolic_c = (se.DenseMatrix([b3_symbolic_c]).jacobian(se.DenseMatrix(self.x_sym))).T
        d2b3dw2_symbolic_c = db3dw_symbolic_c.jacobian(se.DenseMatrix(self.w_sym))
        d2b3dwdx_symbolic_c = db3dw_symbolic_c.jacobian(se.DenseMatrix(self.x_sym))
        d2b3dwdt_symbolic_c = db3dw_symbolic_c.jacobian(se.DenseMatrix(self.t_sym))
        b3_c = symbolic_cbf_wrapper_singleagent(b3_symbolic_c, self.all_sym)
        db3dt_c = symbolic_cbf_wrapper_singleagent(db3dt_symbolic_c, self.all_sym)
        db3dw_c = symbolic_cbf_wrapper_singleagent(db3dw_symbolic_c, self.all_sym)
        db3dx_c = symbolic_cbf_wrapper_singleagent(db3dx_symbolic_c, self.all_sym)
        d2b3dw2_c = symbolic_cbf_wrapper_singleagent(d2b3dw2_symbolic_c, self.all_sym)
        d2b3dwdx_c = symbolic_cbf_wrapper_singleagent(d2b3dwdx_symbolic_c, self.all_sym)
        d2b3dwdt_c = symbolic_cbf_wrapper_singleagent(d2b3dwdt_symbolic_c, self.all_sym)

        # callable c-cbf symbolic functions
        self._b3 = lambda v: b3_a(v) + b3_b(v) @ self._w_dot_drift_f + b3_c(v) @ self._w_dot_contr_f
        self._db3dt = (
            lambda v: db3dt_a(v)
            + db3dt_b(v) @ self._w_dot_drift_f
            + db3dt_c(v) @ self._w_dot_contr_f
        )
        self._db3dw = (
            lambda v: db3dw_a(v)
            + db3dw_b(v) @ self._w_dot_drift_f
            + db3dw_c(v) @ self._w_dot_contr_f
        )
        self._db3dx = (
            lambda v: db3dx_a(v)
            + db3dx_b(v) @ self._w_dot_drift_f
            + db3dx_c(v) @ self._w_dot_contr_f
        )
        self._d2b3dw2 = (
            lambda v: d2b3dw2_a(v)
            + d2b3dw2_b(v) @ self._w_dot_drift_f
            + d2b3dw2_c(v) @ self._w_dot_contr_f
        )
        self._d2b3dwdx = (
            lambda v: d2b3dwdx_a(v)
            + d2b3dwdx_b(v) @ self._w_dot_drift_f
            + d2b3dwdx_c(v) @ self._w_dot_contr_f
        )
        self._d2b3dwdt = (
            lambda v: d2b3dwdt_a(v)
            + d2b3dwdt_b(v) @ self._w_dot_drift_f
            + d2b3dwdt_c(v) @ self._w_dot_contr_f
        )

    def update(self, u: NDArray, dt: float) -> Tuple[NDArray, NDArray]:
        """Updates the adaptation gains and returns the new k weights.

        Arguments:
            x (NDArray): state vector
            u (NDArray): control input applied to system
            h (NDArray): vector of candidate CBFs
            Lf (float): C-CBF drift term (includes filtered wdot)
            Lg (NDArray): matrix of stacked Lgh vectors
            dt: timestep in sec

        Returns
            w_weights: weights on constituent candidate cbfs

        """
        self.t += dt
        w_dot = self.compute_wdot(u)
        w_dot_f = self.filter_update(u)

        self._w_weights += w_dot * self.dt

        return self._w_weights, w_dot, w_dot_f

    def compute_wdot(self, u: NDArray) -> NDArray:
        """Computes the time-derivative w_dot of the w_weights vector.

        Arguments:
            u (NDArray): control input applied to system

        Returns:
            w_dot (NDArray): time-derivative of kWeights

        """
        # compute unconstrained w_dot
        w_dot_0 = self._w_dot_drift + self._w_dot_contr @ u

        # compute what weights would become with unconstrained w_dot
        w_weights = self._w_weights + w_dot_0 * self.dt

        # account for exceeding kmin/kmax bounds
        max_idx = jnp.where(w_weights > self.k_max)
        w_weights[max_idx] = 0.999 * self.k_max
        min_idx = jnp.where(w_weights < self.k_min)
        w_weights[min_idx] = 1.001 * self.k_min
        w_dot = (w_weights - self._w_weights) / self.dt

        # attribute deviation to drift term
        self._w_dot_drift -= w_dot_0 - w_dot

        # set final w_dot value
        self._w_dot = w_dot

        return self._w_dot

    def precompute(self, x: NDArray, h: NDArray, Lt: NDArray, Lf: NDArray, Lg: NDArray) -> NDArray:
        """Precomputes terms needed to compute the adaptation law.

        Arguments:
            x (NDArray): state vector
            h (NDArray): vector of candidate CBFs
            Lf (float): C-CBF drift term (includes filtered wdot)
            Lg (NDArray): matrix of stacked Lgh vectors

        Returns:
            None

        """
        # convexity parameter
        self.s = 1e6

        # delta terms
        self._delta = self.delta(x, h, Lt, Lf)
        self._grad_delta_k = self.grad_delta_k(x, h, Lt, Lf)
        self._grad_delta_x = self.grad_delta_x(x, h, Lt, Lf)
        self._grad_delta_t = self.grad_delta_t(x, h, Lt, Lf)
        self._grad_delta_kk = self.grad_delta_kk(x, h, Lt, Lf)
        self._grad_delta_kx = self.grad_delta_kx(x, h, Lt, Lf)
        self._grad_delta_kt = self.grad_delta_kt(x, h, Lt, Lf)

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
        self._grad_czero_t = self.grad_czero_t(x, h, Lf, Lg)
        self._grad_czero_kk = self.grad_czero_kk(x, h, Lf, Lg)
        self._grad_czero_kx = self.grad_czero_kx(x, h, Lf, Lg)
        self._grad_czero_kt = self.grad_czero_kt(x, h, Lf, Lg)

        # cost terms
        self._grad_cost_kk = self.grad_cost_kk()

        # phi terms
        self._grad_phi_kk = self.grad_phi_kk(x, h, Lf, Lg)
        self._grad_phi_k = self.grad_phi_k(x, h, Lf, Lg)
        self._grad_phi_kx = self.grad_phi_kx(x, h, Lf, Lg)
        self._grad_phi_kt = self.grad_phi_kt(x, h, Lf, Lg)

    def w_dot_drift(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
        """Computes the drift (uncontrolled) component of the time-derivative
        w_dot of the w_weights vector.

        Arguments:
            x (NDArray): state vector
            h (NDArray): vector of candidate CBFs
            Lf (float): C-CBF drift term (includes filtered wdot)
            Lg (NDArray): matrix of stacked Lgh vectors

        Returns:
            w_dot_drift (NDArray): time-derivative of kWeights

        """

        # analytical solution to quadratic program for learning rate
        a = -self.dbdw @ jnp.linalg.inv(self._grad_phi_kk)
        d = self.w_dot_gain * a.T * self._grad_phi_k
        lower_bound = (
            -self.alpha * self.b**3
            - self.dbdt
            - self.dbdx @ f(x)
            + abs(self.dbdw @ self._w_dot_contr + self.dbdx @ g(x)) @ self.u_max
        )
        gamma = lower_bound + self.w_dot_gain * self.dbdw @ jnp.linalg.inv(self._grad_phi_kk) @ (
            self._grad_phi_kx @ f(x) + self._grad_phi_kt
        )
        pstar = gamma * d / (d.T @ d)
        # self.pstar += 0.01 * self.wn * (pstar - self.pstar)
        self.pstar = jnp.ones((len(pstar),))
        # print(self.pstar)

        self.p_gain_mat = self.pstar * jnp.eye(len(pstar))
        # self.p_gain_mat = pstar * jnp.eye(len(pstar))

        # # effectively enforce cbf condition on czero through the drift term
        # lower_bound = (
        #     -self.alpha * self.b**3
        #     - self.dbdt
        #     - self.dbdx @ f(x)
        #     + abs(self.dbdw @ self._w_dot_contr + self.dbdx @ g(x)) @ self.u_max
        # )
        # idxs = jnp.where(abs(self.dbdw) > 1e-6)

        w_dot_drift = (
            -jnp.linalg.inv(self._grad_phi_kk)
            @ (self.p_gain_mat @ self._grad_phi_k + self._grad_phi_kx @ f(x) + self._grad_phi_kt)
            * self.w_dot_gain
        )

        term1 = jnp.linalg.norm(
            -jnp.linalg.inv(self._grad_phi_kk) @ (self.p_gain_mat @ self._grad_phi_k)
        )
        term2 = jnp.linalg.norm(-jnp.linalg.inv(self._grad_phi_kk) @ (self._grad_phi_kx @ f(x)))
        term3 = jnp.linalg.norm(-jnp.linalg.inv(self._grad_phi_kk) @ (self._grad_phi_kt))

        # print(
        #     f"Term 1: {jnp.linalg.norm(-jnp.linalg.inv(self._grad_phi_kk) @ (self.p_gain_mat @ self._grad_phi_k))}"
        # )
        # print(
        #     f"Term 2: {jnp.linalg.norm(-jnp.linalg.inv(self._grad_phi_kk) @ (self._grad_phi_kx @ f(x)))}"
        # )
        # print(f"Term 3: {jnp.linalg.norm(-jnp.linalg.inv(self._grad_phi_kk) @ (self._grad_phi_kt))}")

        self._w_dot_drift = w_dot_drift
        # self._w_dot_drift += self.wn * (w_dot_drift - self._w_dot_drift) * self.dt

        return self._w_dot_drift

    def w_dot_controlled(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
        """Computes the controlled component of the time-derivative
        w_dot of the w_weights vector.

        Arguments:
            x (NDArray): state vector
            u (NDArray): control input applied to system
            h (NDArray): vector of candidate CBFs
            Lf (float): C-CBF drift term (includes filtered wdot)
            Lg (NDArray): matrix of stacked Lgh vectors

        Returns:
            w_dot_controlled (NDArray): time-derivative of kWeights

        """
        w_dot_contr = -jnp.linalg.inv(self._grad_phi_kk) @ self._grad_phi_kx @ g(x)
        w_dot_contr *= self.w_dot_gain

        if len(w_dot_contr.shape) > 1:
            self._w_dot_contr = w_dot_contr
            # self._w_dot_contr += self.wn * (w_dot_contr - self._w_dot_contr) * self.dt
        else:
            self._w_dot_contr = w_dot_contr[:, jnp.newaxis]

        return self._w_dot_contr

    def filter_init(self) -> None:
        """Initializes filter parameters."""
        nWeights = len(self._w_weights)
        nControls = len(self.u_max)

        self._w_dot_f = jnp.zeros((nWeights,))
        self._w_dot_drift_f = jnp.zeros((nWeights,))
        self._w_2dot_drift_f = jnp.zeros((nWeights,))
        self._w_3dot_drift_f = jnp.zeros((nWeights,))
        self._w_dot_contr_f = jnp.zeros((nWeights, nControls))
        self._w_2dot_contr_f = jnp.zeros((nWeights, nControls))
        self._w_3dot_contr_f = jnp.zeros((nWeights, nControls))

    def filter_update(self, u: NDArray) -> None:
        """Updates filtered variables.

        Arguments
        ---------
        u (NDArray): control input vector

        Returns
        -------
        None
        """
        if self._filter_order == 2:
            self._w_3dot_drift_f = (
                self.wn**2 * (self._w_dot_drift - self._w_dot_drift_f)
                - 2 * self.zeta * self.wn * self._w_2dot_drift_f
            )
            self._w_2dot_drift_f += self._w_3dot_drift_f * self.dt
            self._w_dot_drift_f += self._w_2dot_drift_f * self.dt

            self._w_3dot_contr_f = (
                self.wn**2 * (self._w_dot_contr - self._w_dot_contr_f)
                - 2 * self.zeta * self.wn * self._w_2dot_contr_f
            )
            self._w_2dot_contr_f += self._w_3dot_contr_f * self.dt
            self._w_dot_contr_f += self._w_2dot_contr_f * self.dt

        elif self._filter_order == 1:
            self._w_2dot_drift_f = 0.0 / self.dt * (self._w_dot_drift - self._w_dot_drift_f)
            # self._w_2dot_drift_f = 0.5 / self.dt * (self._w_dot_drift - self._w_dot_drift_f)
            self._w_dot_drift_f += self._w_2dot_drift_f * self.dt

            self._w_2dot_contr_f = 0.0 / self.dt * (self._w_dot_contr - self._w_dot_contr_f)
            # self._w_2dot_contr_f = 0.5 / self.dt * (self._w_dot_contr - self._w_dot_contr_f)
            self._w_dot_contr_f += self._w_2dot_contr_f * self.dt

        # Compute final filtered w_dot
        self._w_dot_f = self._w_dot_drift_f + self._w_dot_contr_f @ u

        return self._w_dot_f

    def grad_phi_k(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
        """Computes the gradient of Phi with respect to the gains k.

        Arguments
        ---------
        x: state vector
        h: array of constituent cbfs
        Lf (float): C-CBF drift term (includes filtered wdot)
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_phi_k

        """
        grad_phi_k = self.grad_cost_k(x, h) - 1 / self.s * (
            jnp.sum(jnp.multiply(1 / self._ci, self._grad_ci_k.T), axis=1)
            + self._grad_czero_k / self._czero
        )

        return grad_phi_k.T

    def grad_phi_kk(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
        """Computes the gradient of Phi with respect to
        the gains k twice.

        Arguments
        ---------
        x: state vector
        h: array of constituent cbfs
        Lf: C-CBF drift term (includes filtered wdot)
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_phi_kk

        """
        non_convex_term = (
            jnp.sum(
                (
                    jnp.multiply(self._ci, self._grad_ci_kk.T)
                    - jnp.matmul(
                        self._grad_ci_k[:, :, jnp.newaxis], self._grad_ci_k[:, jnp.newaxis, :]
                    ).T
                )
                / self._ci**2,
                axis=2,
            )
            + (
                self._grad_czero_kk * self._czero
                - self._grad_czero_k[:, jnp.newaxis] @ self._grad_czero_k[jnp.newaxis, :]
            )
            / self._czero**2
        )

        limit = 100
        grad_phi_kk = self.grad_cost_kk() - 1 / self.s * non_convex_term
        try:
            lamba = jnp.min(jnp.linalg.eig(grad_phi_kk)[0]).real
        except jnp.linalg.LinAlgError:
            lamba = 101
        if lamba > limit:
            self.s = lamba / (lamba - limit)
        else:
            while jnp.min(jnp.linalg.eig(grad_phi_kk)[0]) < lamba and self.s < 1e12:
                self.s *= 2
                grad_phi_kk = self.grad_cost_kk() - 1 / self.s * non_convex_term

        return grad_phi_kk

    def grad_phi_kx(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
        """Computes the gradient of Phi with respect to first
        the gains k and then the state x.

        Arguments
        ---------
        x: state vector
        h: array of constituent cbfs
        Lf: C-CBF drift term (includes filtered wdot)
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_phi_kx

        """
        grad_phi_kx = self.grad_cost_kx(x, h) - 1 / self.s * (
            (
                self._grad_czero_kx * self._czero
                - self._grad_czero_k[:, jnp.newaxis] * self._grad_czero_x[jnp.newaxis, :]
            )
            / self._czero**2
            + jnp.sum(
                (
                    jnp.multiply(self._ci, self._grad_ci_kx.T)
                    - jnp.matmul(
                        self._grad_ci_k[:, :, jnp.newaxis], self._grad_ci_x[:, jnp.newaxis, :]
                    ).T
                )
                / self._ci**2,
                axis=2,
            ).T
        )

        return grad_phi_kx

    def grad_phi_kt(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
        """Computes the gradient of Phi with respect to first
        the gains k and then the time t.

        Arguments
        ---------
        x: state vector
        h: array of constituent cbfs
        Lf: C-CBF drift term (includes filtered wdot)
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_phi_kt

        """
        grad_phi_kt = (
            -1
            / self.s
            * (self._grad_czero_kt * self._czero - self._grad_czero_k * self._grad_czero_t)
            / self._czero**2
        )

        return grad_phi_kt

    def cost(self, x: NDArray, h: NDArray) -> float:
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
            * (self._w_weights - self.w_des(x, h)).T
            @ self.cost_gain_mat
            @ (self._w_weights - self.w_des(x, h))
        )

        return cost

    def grad_cost_k(self, x: NDArray, h: NDArray) -> float:
        """Computes gradient of the cost function associated with the adaptation law
        with respect to the weight vector k.

        Arguments
        ---------
        h: vector of constituent cbfs

        Returns
        -------
        grad_cost_k: gradient of cost evaluated at k

        """
        grad_cost_k = self.cost_gain_mat @ (self._w_weights - self.w_des(x, h))

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
        grad_cost_kx = -self.cost_gain_mat @ self.grad_w_des_x(x, h)

        return grad_cost_kx

    def b2cp1(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> float:
        """Returns the viability constraint function (b_{2c+1}) evaluated at the current
        state x and gain vector k.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector
        Lf: C-CBF drift term (includes filtered wdot)
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        val: viability constraint function evaluated at x and k

        """
        q = self.qvector(x, h, Lg)
        val = self._delta - smooth_abs(q).T @ self.u_max

        self.czero_val1 = val

        return val * self.b2cp1_gain

    def grad_b2cp1_w(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
        """Computes the gradient of the viability constraint function evaluated at the current
        state x and gain vector k with respect to k.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector
        Lf: C-CBF drift term (includes filtered wdot)
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_c0_k: gradient of viability constraint function with respect to k

        """
        q = self.qvector(x, h, Lg)
        val = self._grad_delta_w - dsmoothabs_dx(q) @ self.grad_qvector_w(x, h, Lg) @ self.u_max

        return val * self.b2cp1_gain

    def grad_czero_x(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
        """Computes the gradient of the viability constraint function evaluated at the current
        state x and gain vector k with respect to x.

        Arguments
        ---------
        x: state vector
        k: constituent cbf weighting vector
        h: constituent cbf vector
        Lf: C-CBF drift term (includes filtered wdot)
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_c0_x: gradient of viability constraint function with respect to x

        """
        q = self.qvector(x, h, Lg)
        val = self._grad_delta_x - dsmoothabs_dx(q) @ self.grad_qvector_x(x, h, Lg) @ self.u_max

        return val * self.b2cp1_gain

    def grad_czero_t(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
        """Computes the gradient of the viability constraint function evaluated at the current
        state x and gain vector k with respect to x.

        Arguments
        ---------
        x: state vector
        k: constituent cbf weighting vector
        h: constituent cbf vector
        Lf: C-CBF drift term (includes filtered wdot)
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_c0_t: gradient of viability constraint function with respect to t

        """
        q = self.qvector(x, h, Lg)
        val = self._grad_delta_t - dsmoothabs_dx(q) @ self.grad_qvector_t(x, h, Lg) @ self.u_max

        return val * self.b2cp1_gain

    def grad_czero_kk(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
        """Computes the gradient of the viability constraint function evaluated at the current
        state x and gain vector k with respect to first k and then k again.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector
        Lf: C-CBF drift term (includes filtered wdot)
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_c0_kk: gradient of viability constraint function with respect to k then x

        """
        vector = self.v_vector(x, h, Lg)
        dvdk = self.grad_v_vector_k(x, h, Lg)
        d2vdk2 = self.grad_v_vector_kk(x, h, Lg)
        eterm = jnp.exp(vector)

        #! This is where I am stuck
        grad_F_kk = (
            jnp.einsum(
                "ij,lm->ilj",
                (
                    ((2 * eterm * dvdk) * (1 + eterm) - (2 * eterm) * (1 + eterm * dvdk))
                    / (1 + eterm) ** 2
                ),
                dvdk,
            )
            + (2 * eterm) / (1 + eterm) * d2vdk2
            - d2vdk2
        )

        grad_c0_kk = grad_F_kk @ self.u_max - self._grad_delta_kk

        return grad_c0_kk * self.czero_gain

    def grad_czero_kx(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
        """Computes the gradient of the viability constraint function evaluated at the current
        state x and gain vector k with respect to first k and then x.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector
        Lf: C-CBF drift term (includes filtered wdot)
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_c0_kx: gradient of viability constraint function with respect to k then x

        """
        vector = self.v_vector(x, h, Lg)
        dvdk = self.grad_v_vector_k(x, h, Lg)
        dvdx = self.grad_v_vector_x(x, h, Lg)
        d2vdkdx = self.grad_v_vector_kx(x, h, Lg)
        eterm = jnp.exp(vector)
        grad_F_kx = (
            jnp.einsum(
                "ij,lm->jli",
                (
                    ((2 * eterm * dvdx) * (1 + eterm) - (2 * eterm) * (1 + eterm * dvdx))
                    / (1 + eterm) ** 2
                ),
                dvdk,
            )
            + jnp.einsum("i,jkl->jkl", (2 * eterm) / (1 + eterm), d2vdkdx)
            - d2vdkdx
        )

        grad_c0_kx = jnp.einsum("i,jkl->kl", self.u_max, grad_F_kx) - self._grad_delta_kx

        return grad_c0_kx * self.czero_gain

    def grad_czero_kt(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
        """Computes the gradient of the viability constraint function evaluated at the current
        state x and gain vector k with respect to first k and then t.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector
        Lf: C-CBF drift term (includes filtered wdot)
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        grad_c0_kt: gradient of viability constraint function with respect to k then x

        """
        vector = self.v_vector(x, h, Lg)
        dvdk = self.grad_v_vector_k(x, h, Lg)
        dvdt = self.grad_v_vector_t(x, h, Lg)
        d2vdkdt = self.grad_v_vector_kt(x, h, Lg)
        eterm = jnp.exp(vector)
        grad_F_kt = (
            jnp.einsum(
                "i,jk->jk",
                (
                    ((2 * eterm * dvdt) * (1 + eterm) - (2 * eterm) * (1 + eterm * dvdt))
                    / (1 + eterm) ** 2
                ),
                dvdk,
            )
            + (2 * eterm) / (1 + eterm) * d2vdkdt
            - d2vdkdt
        )

        grad_c0_kt = grad_F_kt @ self.u_max - self._grad_delta_kt

        return grad_c0_kt * self.czero_gain

    # def czero(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> float:
    #     """Returns the viability constraint function evaluated at the current
    #     state x and gain vector k.

    #     Arguments
    #     ---------
    #     x: state vector
    #     h: constituent cbf vector
    #     Lf: C-CBF drift term (includes filtered wdot)
    #     Lg: matrix of constituent cbf Lgh vectors

    #     Returns
    #     -------
    #     c0: viability constraint function evaluated at x and k

    #     """
    #     vector = self.v_vector(x, h, Lg)
    #     czero = vector @ self.U @ vector.T - self._delta**2
    #     if self.t == 0 and czero < 0:
    #         print(f"CZERO < 0: {vector @ self.U @ vector.T} < {self._delta**2}")
    #         start = time.time()
    #         while czero <= 5 * self.czero_gain and (time.time() - start) < 5:

    #             czero = self.k_gradient_descent(x, h, Lf, Lg)
    #             self._delta = self.delta(x, h, Lf)

    #     s_func = 0  # -czero / 2 * (1 - jnp.sqrt(czero**2 + 0.001**2) / czero)

    #     # for numerical instability
    #     czero = jnp.max([czero, 1e-3])

    #     self.czero_val1 = czero
    #     self.czero_val2 = (abs(vector) @ self.u_max) - self._delta

    #     return (czero + s_func * self.s_on) * self.czero_gain

    # def grad_czero_k(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
    #     """Computes the gradient of the viability constraint function evaluated at the current
    #     state x and gain vector k with respect to k.

    #     Arguments
    #     ---------
    #     x: state vector
    #     h: constituent cbf vector
    #     Lf: C-CBF drift term (includes filtered wdot)
    #     Lg: matrix of constituent cbf Lgh vectors

    #     Returns
    #     -------
    #     grad_c0_k: gradient of viability constraint function with respect to k

    #     """
    #     vector = self.v_vector(x, h, Lg)
    #     grad_v_vec_k = self.grad_v_vector_k(x, h, Lg)
    #     grad_c0_k = 2 * grad_v_vec_k @ self.U @ vector.T
    #     if self._delta > 0:
    #         grad_c0_k -= 2 * self._delta * self._grad_delta_k

    #     grad_sfunc_k = grad_c0_k * 0  # (
    #     # 1 / 2 * grad_c0_k * (self.czero_val1 / jnp.sqrt(self.czero_val1**2 + 0.001**2))
    #     # )

    #     return grad_c0_k + grad_sfunc_k * self.s_on * self.czero_gain

    # def grad_czero_x(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
    #     """Computes the gradient of the viability constraint function evaluated at the current
    #     state x and gain vector k with respect to x.

    #     Arguments
    #     ---------
    #     x: state vector
    #     k: constituent cbf weighting vector
    #     h: constituent cbf vector
    #     Lf: C-CBF drift term (includes filtered wdot)
    #     Lg: matrix of constituent cbf Lgh vectors

    #     Returns
    #     -------
    #     grad_c0_x: gradient of viability constraint function with respect to x

    #     """
    #     vector = self.v_vector(x, h, Lg)
    #     grad_v_vec_x = self.grad_v_vector_x(x, h, Lg)
    #     grad_c0_x = 2 * grad_v_vec_x @ self.U @ vector.T
    #     if self._delta > 0:
    #         grad_c0_x -= 2 * self._delta * self._grad_delta_x

    #     grad_sfunc_x = 0 * grad_c0_x
    #     # (
    #     #   1 / 2 * grad_c0_x * (self.czero_val1 / jnp.sqrt(self.czero_val1**2 + 0.001**2))
    #     # )

    #     return grad_c0_x + grad_sfunc_x * self.s_on * self.czero_gain

    # def grad_czero_t(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
    #     """Computes the gradient of the viability constraint function evaluated at the current
    #     state x and gain vector k with respect to x.

    #     Arguments
    #     ---------
    #     x: state vector
    #     k: constituent cbf weighting vector
    #     h: constituent cbf vector
    #     Lf: C-CBF drift term (includes filtered wdot)
    #     Lg: matrix of constituent cbf Lgh vectors

    #     Returns
    #     -------
    #     grad_c0_t: gradient of viability constraint function with respect to t

    #     """
    #     vector = self.v_vector(x, h, Lg)
    #     grad_v_vec_t = self.grad_v_vector_t(x, h, Lg)
    #     grad_c0_t = 2 * grad_v_vec_t @ self.U @ vector.T
    #     if self._delta > 0:
    #         grad_c0_t -= 2 * self._delta * self._grad_delta_t

    #     return grad_c0_t * self.czero_gain

    # def grad_czero_kk(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
    #     """Computes the gradient of the viability constraint function evaluated at the current
    #     state x and gain vector k with respect to first k and then k again.

    #     Arguments
    #     ---------
    #     x: state vector
    #     h: constituent cbf vector
    #     Lf: C-CBF drift term (includes filtered wdot)
    #     Lg: matrix of constituent cbf Lgh vectors

    #     Returns
    #     -------
    #     grad_c0_kk: gradient of viability constraint function with respect to k then x

    #     """
    #     vector = self.v_vector(x, h, Lg)
    #     dvdk = self.grad_v_vector_k(x, h, Lg)
    #     d2vdk2 = self.grad_v_vector_kk(x, h, Lg)
    #     grad_c0_kk = 2 * d2vdk2 @ self.U @ vector.T + 2 * dvdk @ self.U @ dvdk.T
    #     if self._delta > 0:
    #         grad_c0_kk -= 2 * (
    #             self._grad_delta_k[:, jnp.newaxis] @ self._grad_delta_k[jnp.newaxis, :]
    #             + self._delta * self._grad_delta_kk
    #         )

    #     return grad_c0_kk * self.czero_gain

    # def grad_czero_kx(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
    #     """Computes the gradient of the viability constraint function evaluated at the current
    #     state x and gain vector k with respect to first k and then x.

    #     Arguments
    #     ---------
    #     x: state vector
    #     h: constituent cbf vector
    #     Lf: C-CBF drift term (includes filtered wdot)
    #     Lg: matrix of constituent cbf Lgh vectors

    #     Returns
    #     -------
    #     grad_c0_kx: gradient of viability constraint function with respect to k then x

    #     """
    #     vector = self.v_vector(x, h, Lg)
    #     dvdk = self.grad_v_vector_k(x, h, Lg)
    #     dvdx = self.grad_v_vector_x(x, h, Lg)
    #     d2vdkdx = self.grad_v_vector_kx(x, h, Lg)
    #     grad_c0_kx = (
    #         2 * jnp.einsum("ijk,kl->jil", d2vdkdx.T, self.U) @ vector.T + 2 * dvdk @ self.U @ dvdx.T
    #     )
    #     if self._delta > 0:
    #         grad_c0_kx -= 2 * (
    #             self._grad_delta_k[:, jnp.newaxis] @ self._grad_delta_x[jnp.newaxis, :]
    #             + self._delta * self._grad_delta_kx
    #         )

    #     return grad_c0_kx * self.czero_gain

    # def grad_czero_kt(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> NDArray:
    #     """Computes the gradient of the viability constraint function evaluated at the current
    #     state x and gain vector k with respect to first k and then t.

    #     Arguments
    #     ---------
    #     x: state vector
    #     h: constituent cbf vector
    #     Lf: C-CBF drift term (includes filtered wdot)
    #     Lg: matrix of constituent cbf Lgh vectors

    #     Returns
    #     -------
    #     grad_c0_kt: gradient of viability constraint function with respect to k then x

    #     """
    #     vector = self.v_vector(x, h, Lg)
    #     dvdk = self.grad_v_vector_k(x, h, Lg)
    #     dvdt = self.grad_v_vector_t(x, h, Lg)
    #     d2vdkdt = self.grad_v_vector_kt(x, h, Lg)
    #     grad_c0_kt = 2 * (d2vdkdt @ self.U @ vector).T + 2 * dvdk @ self.U @ dvdt.T
    #     if self._delta > 0:
    #         grad_c0_kt -= 2 * (
    #             self._grad_delta_k * self._grad_delta_t + self._delta * self._grad_delta_kt
    #         )

    #     return grad_c0_kt * self.czero_gain

    def ci(self) -> NDArray:
        """Returns positivity constraint functions on the gain vector k.

        Arguments:
            None

        Returns:
            ci: array of positivity constraint functions evaluated at k

        """
        return (
            jnp.concatenate([(self._w_weights - self.k_min), (self.k_max - self._w_weights)])
            * self.ci_gain
        )

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
        grad_ci_min_k = jnp.eye(len(self._w_weights))
        grad_ci_max_k = -jnp.eye(len(self._w_weights))
        return jnp.vstack([grad_ci_min_k, grad_ci_max_k]) * self.ci_gain

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
        grad_ci_min_x = jnp.zeros((len(self._w_weights), len(self._grad_delta_x)))
        grad_ci_max_x = jnp.zeros((len(self._w_weights), len(self._grad_delta_x)))
        return jnp.vstack([grad_ci_min_x, grad_ci_max_x]) * self.ci_gain

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
        grad_ci_min_kk = jnp.zeros(
            (len(self._w_weights), len(self._w_weights), len(self._w_weights))
        )
        grad_ci_max_kk = jnp.zeros(
            (len(self._w_weights), len(self._w_weights), len(self._w_weights))
        )
        return jnp.vstack([grad_ci_min_kk, grad_ci_max_kk]) * self.ci_gain

    def grad_ci_kx(self) -> NDArray:
        """Computes the gradient of the positivity constraint functions evaluated at the
        gain vector k with respect to first k and then x.

        Arguments:
            None

        Returns:
            grad_ci_kx: gradient of positivity constraint functions with respect to k and then x

        """
        grad_ci_min_kx = jnp.zeros(
            (len(self._w_weights), len(self._w_weights), len(self._grad_delta_x))
        )
        grad_ci_max_kx = jnp.zeros(
            (len(self._w_weights), len(self._w_weights), len(self._grad_delta_x))
        )
        return jnp.vstack([grad_ci_min_kx, grad_ci_max_kx]) * self.ci_gain

    def delta(self, x: NDArray, h: NDArray, Lt: NDArray, Lf: NDArray) -> float:
        """Computes the threshold above which the product LgH*u_max must remain for control viability.
        In other words, in order to be able to satisfy:

        LfH + LgH*u + LkH + alpha(H) >= 0

        it must hold that LgH*u_max >= -LfH - LkH - alpha(H).

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector
        Lf: C-CBF drift term (includes filtered wdot)

        Returns
        -------
        delta = LfH + alpha(H) + LkH

        """
        class_k_term = self.alpha * self.H(h) ** 3 if self.cubic else self.alpha * self.H(h)
        delta = (
            -self.q @ (Lt + Lf)
            - class_k_term
            + self.eta_mu
            - self.grad_H_k(h).T @ self._w_dot_drift_f
        )

        return delta if delta > 0 else 0

    def grad_delta_k(self, x: NDArray, h: NDArray, Lt: NDArray, Lf: NDArray) -> NDArray:
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
        class_k_term = (
            3 * self.alpha * self.H(h) ** 2 * self.grad_H_k(h)
            if self.cubic
            else self.alpha * self.grad_H_k(h)
        )
        return -self.dqdk @ (Lt + Lf) - class_k_term - self.grad_H_kk(h) @ self._w_dot_drift_f

    def grad_delta_x(self, x: NDArray, h: NDArray, Lt: NDArray, Lf: NDArray) -> NDArray:
        """Computes the threshold above which the product LgH*u_max must remain for control viability.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector

        Returns
        -------
        grad_delta_x: gradient of delta with respect to x

        """
        dqdx = self.dqdh @ self.dhdx
        dLdx = (
            dqdx.T @ (Lt + Lf)
            + self.q @ self.d2hdtdx
            + jnp.einsum("ij,jkl->kl", self.q[jnp.newaxis, :], self.d2hdx2) @ f(x)
            + self.q @ self.dhdx @ dfdx(x)
        )

        class_k_term = (
            3 * self.alpha * self.H(h) ** 2 * self.grad_H_x()
            if self.cubic
            else self.alpha * self.grad_H_x()
        )

        return -dLdx - class_k_term - self.grad_H_kx(h).T @ self._w_dot_drift_f

    def grad_delta_t(self, x: NDArray, h: NDArray, Lt: NDArray, Lf: NDArray) -> NDArray:
        """Computes the threshold above which the product LgH*u_max must remain for control viability.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector

        Returns
        -------
        grad_delta_t: gradient of delta with respect to t

        """
        dqdt = self.dqdh @ self.dhdt
        return -dqdt - self.grad_H_k(h).T @ self._w_2dot_drift_f

    def grad_delta_kk(self, x: NDArray, h: NDArray, Lt: NDArray, Lf: NDArray) -> NDArray:
        """Computes the threshold above which the product LgH*u_max must remain for control viability.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector

        Returns
        -------
        grad_delta_kk: gradient of delta with respect to k twice

        """
        grad_LfH_kk = self.d2qdk2 @ Lf

        class_k_term = (
            3 * self.alpha * self.H(h) ** 2 * self.grad_H_kk(h)
            + 6
            * self.alpha
            * self.H(h)
            * self.grad_H_k(h)[:, jnp.newaxis]
            @ self.grad_H_k(h)[jnp.newaxis, :]
            if self.cubic
            else self.alpha * self.grad_H_kk(h)
        )

        return -grad_LfH_kk - class_k_term - self.grad_H_kkk(h) @ self._w_dot_drift_f

    def grad_delta_kx(
        self,
        x: NDArray,
        h: NDArray,
        Lt: NDArray,
        Lf: NDArray,
    ) -> NDArray:
        """Computes the threshold above which the product LgH*u_max must remain for control viability.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector

        Returns
        -------
        grad_delta_kx: gradient of delta with respect to k and then x

        """
        d2qdkdx = self.d2qdkdh @ self.dhdx
        grad_LfH_kx = (
            (jnp.einsum("ijk,kl->ijl", d2qdkdx.T, self.dhdx) @ f(x)).T
            + jnp.einsum("ij,jkl->ikl", self.dqdk.T, self.d2hdx2) @ f(x)
            + self.dqdk.T @ self.dhdx @ dfdx(x)
        )

        class_k_term = (
            3 * self.alpha * self.H(h) ** 2 * self.grad_H_kx(h)
            + 6
            * self.alpha
            * self.H(h)
            * self.grad_H_k(h)[:, jnp.newaxis]
            @ self.grad_H_x()[jnp.newaxis, :]
            if self.cubic
            else self.alpha * self.grad_H_kx(h)
        )

        return -grad_LfH_kx - class_k_term - (self.grad_H_kkx(h).T @ self._w_dot_drift_f).T

    def grad_delta_kt(self, x: NDArray, h: NDArray, Lt: NDArray, Lf: NDArray) -> NDArray:
        """Computes the threshold above which the product LgH*u_max must remain for control viability.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector

        Returns
        -------
        grad_delta_kt: gradient of delta with respect to k and then t

        """
        return -self.grad_H_kk(h) @ self._w_2dot_drift_f

    def v_vector(self, x: NDArray, h: NDArray, Lg: NDArray) -> NDArray:
        """Returns the viability vector v evaluated at the current
        state x and gain vector k.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        v: viability vector

        """
        return self.q @ Lg + self.grad_H_k(h) @ self._w_dot_contr_f

    def grad_v_vector_k(self, x: NDArray, h: NDArray, Lg: NDArray) -> NDArray:
        """Returns the gradient of the viability vector v with respect to k.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        dvdk: viability vector

        """
        return self.dqdk @ Lg + self.grad_H_kk(h) @ self._w_dot_contr_f

    def grad_v_vector_x(self, x: NDArray, h: NDArray, Lg: NDArray) -> NDArray:
        """Returns the gradient of the viability vector v with respect to x.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        dvdx: viability vector

        """
        dqdx = self.dqdh @ self.dhdx
        dLgdx = self.d2hdx2 @ g(x) + jnp.einsum("ij,jkl->ilk", self.dhdx, dgdx(x))

        return (
            dqdx.T @ Lg
            + jnp.einsum("ij,jkl->kl", self.q[jnp.newaxis, :], dLgdx)
            + self.grad_H_kx(h).T @ self._w_dot_contr_f
        )

    def grad_v_vector_t(self, x: NDArray, h: NDArray, Lg: NDArray) -> NDArray:
        """Returns the gradient of the viability vector v with respect to t.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        dvdt: viability vector

        """
        return self.grad_H_k(h) @ self._w_2dot_contr_f

    def grad_v_vector_kk(self, x: NDArray, h: NDArray, Lg: NDArray) -> NDArray:
        """Returns the Hessian of the viability vector v with respect to k (twice).

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        d2vdk2: viability vector

        """
        return self.d2qdk2 @ Lg + self.grad_H_kkk(h) @ self._w_dot_contr_f

    def grad_v_vector_kx(self, x: NDArray, h: NDArray, Lg: NDArray) -> NDArray:
        """Returns the Hessian of the viability vector v with respect to k then x.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        d2vdkdx: viability vector

        """
        d2qdkdx = self.d2qdkdh @ self.dhdx
        dLgdx = self.d2hdx2 @ g(x) + jnp.einsum("ij,jkl->ilk", self.dhdx, dgdx(x))
        return (
            jnp.einsum("ij,jkl->ikl", Lg.T, d2qdkdx)
            + jnp.einsum("ij,jkl->lik", self.dqdk.T, dLgdx)
            + jnp.einsum("ijk,kl->lji", self.grad_H_kkx(h).T, self._w_dot_contr_f)
        )

    def grad_v_vector_kt(self, x: NDArray, h: NDArray, Lg: NDArray) -> NDArray:
        """Returns the Hessian of the viability vector v with respect to k then t.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        d2vdkdt: viability vector

        """
        return self.grad_H_kk(h) @ self._w_2dot_contr_f

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
        H = 1 - jnp.sum(jnp.exp(-self._w_weights * h))

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
        grad_H_k = h * jnp.exp(-self._w_weights * h)

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
        grad_H_kk = jnp.diag(-(h**2) * jnp.exp(-self._w_weights * h))

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
        grad_H_kx = jnp.diag((1 - h * self._w_weights) * jnp.exp(-self._w_weights * h)) @ self.dhdx

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
        filling = (h**3) * jnp.exp(-self._w_weights * h)
        grad_H_kkk = jnp.zeros((len(self._w_weights), len(self._w_weights), len(self._w_weights)))
        jnp.fill_diagonal(grad_H_kkk, filling)

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
        grad_H_kkx = ((h**2 * self._w_weights - 2 * h) * jnp.exp(-self._w_weights * h))[
            :, jnp.newaxis
        ] @ self.dhdx[:, jnp.newaxis, :]

        return grad_H_kkx

    # def w_des(self, h: NDArray) -> NDArray:
    #     """Computes the desired gains k for the constituent cbfs. This can be
    #     thought of as the nominal adaptation law (unconstrained).

    #     Arguments:
    #         h (NDArray): array of constituent cbf values

    #     Returns:
    #         w_des (NDArray)

    #     """
    #     hmin = 0.1
    #     w_des = self.w_des_gain * h / jnp.min([jnp.min(h), hmin])

    #     return jnp.clip(w_des, self.k_min, self.k_max)

    # def grad_w_des_x(self, x: NDArray, h: NDArray) -> NDArray:
    #     """Computes the desired gains k for the constituent cbfs. This can be
    #     thought of as the nominal adaptation law (unconstrained).

    #     Arguments
    #     ---------
    #     h: array of constituent cbf values
    #     x: state vector

    #     Returns
    #     -------
    #     grad_w_desired_x

    #     """
    #     hmin = 0.1
    #     min_h_idx = jnp.where(h == jnp.min(h))[0][0]

    #     w_des = self.w_des_gain * h / jnp.min([h[min_h_idx], hmin])
    #     over_k_max = jnp.where(w_des > self.k_max)[0]
    #     under_k_min = jnp.where(w_des < self.k_min)[0]

    #     if h[min_h_idx] > hmin:
    #         grad_w_desired_x = self.w_des_gain * self.dhdx / hmin
    #     else:
    #         # Deal with cases when dhdx very close to zero
    #         dhdx = self.dhdx
    #         dhdx[abs(dhdx) <= 1e-9] = 1
    #         grad_w_desired_x = self.w_des_gain * self.dhdx / dhdx[min_h_idx, :]

    #     grad_w_desired_x[over_k_max] = 0
    #     grad_w_desired_x[under_k_min] = 0

    #     return grad_w_desired_x

    # def w_des(self, h: NDArray) -> NDArray:
    #     """Computes the desired gains k for the constituent cbfs. This can be
    #     thought of as the nominal adaptation law (unconstrained).

    #     Arguments:
    #         h (NDArray): array of constituent cbf values

    #     Returns:
    #         w_des (NDArray)

    #     """

    #     k_beta = 0.25
    #     k_vel = 0.5
    #     k_reach = 0.5
    #     k_obstacles = (jnp.argsort(jnp.argsort(h[:5])) + 1) ** 2

    #     # w_des = jnp.concatenate([k_obstacles, jnp.array([k_vel, k_beta, k_reach])]) * self.w_des_gain
    #     w_des = jnp.concatenate([k_obstacles, jnp.array([k_beta, k_reach])]) * self.w_des_gain

    #     return jnp.clip(w_des, self.k_min, self.k_max)

    # def grad_w_des_x(self, x: NDArray, h: NDArray) -> NDArray:
    #     """Computes the desired gains k for the constituent cbfs. This can be
    #     thought of as the nominal adaptation law (unconstrained).

    #     Arguments
    #     ---------
    #     h: array of constituent cbf values
    #     x: state vector

    #     Returns
    #     -------
    #     grad_w_desired_x

    #     """
    #     grad_w_desired_x = 0 * self.dhdx

    #     return grad_w_desired_x

    def w_des(self, x: NDArray, h: NDArray) -> NDArray:
        """Computes the desired gains k for the constituent cbfs. This can be
        thought of as the nominal adaptation law (unconstrained).

        Arguments:
            h (NDArray): array of constituent cbf values

        Returns:
            w_des (NDArray)

        """
        # w_des = self.w_des_gain * h**3
        # w_des = self.w_des_gain * h
        w_des = self.w_des_gain * (h + self.dhdx @ f(x))

        self._w_desired = jnp.clip(w_des, self.k_min, self.k_max)

        return self._w_desired

    def grad_w_des_x(self, x: NDArray, h: NDArray) -> NDArray:
        """Computes the desired gains k for the constituent cbfs. This can be
        thought of as the nominal adaptation law (unconstrained).

        Arguments
        ---------
        h: array of constituent cbf values
        x: state vector

        Returns
        -------
        grad_w_desired_x

        """
        # grad_w_desired_x = self.w_des_gain * 3 * jnp.diag(h**2) @ self.dhdx
        # grad_w_desired_x = self.w_des_gain * jnp.diag(h) @ self.dhdx
        grad_w_desired_x = self.w_des_gain * (
            self.dhdx + jnp.einsum("ijk,k->ij", self.d2hdx2, f(x)) + self.dhdx @ dfdx(x)
        )

        return grad_w_desired_x

    # def w_des(self, x: NDArray, h: NDArray) -> NDArray:
    #     """Computes the desired gains k for the constituent cbfs. This can be
    #     thought of as the nominal adaptation law (unconstrained).

    #     Arguments:
    #         h (NDArray): array of constituent cbf values

    #     Returns:
    #         w_des (NDArray)

    #     """
    #     h = jnp.clip(h, 0.01, jnp.inf)
    #     w_des = self.w_des_gain / h

    #     return jnp.clip(w_des, self.k_min, self.k_max)

    # def grad_w_des_x(self, x: NDArray, h: NDArray) -> NDArray:
    #     """Computes the desired gains k for the constituent cbfs. This can be
    #     thought of as the nominal adaptation law (unconstrained).

    #     Arguments
    #     ---------
    #     h: array of constituent cbf values
    #     x: state vector

    #     Returns
    #     -------
    #     grad_w_desired_x

    #     """
    #     h = jnp.clip(h, 0.01, jnp.inf)
    #     w_des = self.w_des_gain / h

    #     over_k_max = jnp.where(w_des > self.k_max)[0]
    #     under_k_min = jnp.where(w_des < self.k_min)[0]

    #     grad_w_desired_x = -self.w_des_gain / h[:, jnp.newaxis] ** 2 * self.dhdx

    #     grad_w_desired_x[over_k_max] = 0
    #     grad_w_desired_x[under_k_min] = 0

    #     return grad_w_desired_x

    # def w_des(self, h: NDArray) -> NDArray:
    #     """Computes the desired gains k for the constituent cbfs. This can be
    #     thought of as the nominal adaptation law (unconstrained).

    #     Arguments:
    #         h (NDArray): array of constituent cbf values

    #     Returns:
    #         w_des (NDArray)

    #     """
    #     h = jnp.clip(h, 0.01, jnp.inf)
    #     w_des = self.w_des_gain * h**2
    #     w_des[5] = 5.0

    #     return jnp.clip(w_des, self.k_min, self.k_max)

    # def grad_w_des_x(self, x: NDArray, h: NDArray) -> NDArray:
    #     """Computes the desired gains k for the constituent cbfs. This can be
    #     thought of as the nominal adaptation law (unconstrained).

    #     Arguments
    #     ---------
    #     h: array of constituent cbf values
    #     x: state vector

    #     Returns
    #     -------
    #     grad_w_desired_x

    #     """
    #     h = jnp.clip(h, 0.01, jnp.inf)
    #     w_des = self.w_des_gain * h**2
    #     w_des[5] = 5.0

    #     over_k_max = jnp.where(w_des > self.k_max)[0]
    #     under_k_min = jnp.where(w_des < self.k_min)[0]

    #     grad_w_desired_x = self.w_des_gain * 2 * h[:, jnp.newaxis] * self.dhdx

    #     grad_w_desired_x[over_k_max] = 0
    #     grad_w_desired_x[under_k_min] = 0
    #     grad_w_desired_x[5] = 0

    #     return grad_w_desired_x

    # def w_des(self, h: NDArray) -> NDArray:
    #     """Computes the desired gains k for the constituent cbfs. This can be
    #     thought of as the nominal adaptation law (unconstrained).

    #     Arguments:
    #         h (NDArray): array of constituent cbf values

    #     Returns:
    #         w_des (NDArray)

    #     """
    #     h = jnp.clip(h, 0.01, jnp.inf)
    #     idx = jnp.where(h == jnp.min(h))
    #     w_des = 0.9 * self.k_max * jnp.ones((len(h),))
    #     w_des[idx] = 0.1

    #     return jnp.clip(w_des, self.k_min, self.k_max)

    # def grad_w_des_x(self, x: NDArray, h: NDArray) -> NDArray:
    #     """Computes the desired gains k for the constituent cbfs. This can be
    #     thought of as the nominal adaptation law (unconstrained).

    #     Arguments
    #     ---------
    #     h: array of constituent cbf values
    #     x: state vector

    #     Returns
    #     -------
    #     grad_w_desired_x

    #     """
    #     grad_w_desired_x = 0 * self.dhdx

    #     return grad_w_desired_x

    # def w_des(self, h: NDArray) -> NDArray:
    #     """Computes the desired gains k for the constituent cbfs. This can be
    #     thought of as the nominal adaptation law (unconstrained).

    #     Arguments:
    #         h (NDArray): array of constituent cbf values

    #     Returns:
    #         w_des (NDArray)

    #     """
    #     w_des = self.w_des_gain * jnp.ones((len(h),))

    #     return jnp.clip(w_des, self.k_min, self.k_max)

    # def grad_w_des_x(self, x: NDArray, h: NDArray) -> NDArray:
    #     """Computes the desired gains k for the constituent cbfs. This can be
    #     thought of as the nominal adaptation law (unconstrained).

    #     Arguments
    #     ---------
    #     h: array of constituent cbf values
    #     x: state vector

    #     Returns
    #     -------
    #     grad_w_desired_x

    #     """
    #     grad_w_desired_x = jnp.zeros((self.dhdx.shape))

    #     return grad_w_desired_x

    # def w_des(self, h: NDArray) -> NDArray:
    #     """Computes the desired gains k for the constituent cbfs. This can be
    #     thought of as the nominal adaptation law (unconstrained).

    #     Arguments:
    #         h (NDArray): array of constituent cbf values

    #     Returns:
    #         w_des (NDArray)

    #     """
    #     w_des = self._w_weights + (self._w_dot_drift_f + self._w_dot_contr_f @ self.u_nom)

    #     return jnp.clip(w_des, self.k_min, self.k_max)

    # def grad_w_des_x(self, x: NDArray, h: NDArray) -> NDArray:
    #     """Computes the desired gains k for the constituent cbfs. This can be
    #     thought of as the nominal adaptation law (unconstrained).

    #     Arguments
    #     ---------
    #     h: array of constituent cbf values
    #     x: state vector

    #     Returns
    #     -------
    #     grad_w_desired_x

    #     """
    #     grad_w_desired_x = jnp.zeros((self.dhdx.shape))

    #     return grad_w_desired_x

    def k_gradient_descent(self, x: NDArray, h: NDArray, Lf: float, Lg: NDArray) -> float:
        """Runs gradient descent on the w_weights in order to increase the
        control authority at t=0.

        Arguments
        ---------
        x: state vector
        h: constituent cbf vector
        Lf: C-CBF drift term (includes filtered wdot)
        Lg: matrix of constituent cbf Lgh vectors

        Returns
        -------
        new_czero: updated czero value based on new w_weights

        """
        # line search parameter
        beta = 1.0

        # compute gradient
        grad_c0_k = 2 * self.grad_v_vector_k(x, h, Lg) @ self.U @ self.v_vector(
            x, h, Lg
        ) - 2 * self.delta(x, h, Lf) * self.grad_delta_k(x, h, Lf)
        if jnp.sum(abs(grad_c0_k)) == 0:
            grad_c0_k = jnp.flip(jnp.random.random(grad_c0_k.shape))
            # grad_c0_k = jnp.random.random(grad_c0_k.shape)

        # gradient descent
        self._w_weights = jnp.clip(
            self._w_weights + grad_c0_k * beta, self.k_min * 1.01, self.k_max * 0.99
        )

        # compute new quantities
        self.q = self._w_weights * jnp.exp(-self._w_weights * h)
        self.dqdk = jnp.diag((1 - self._w_weights * h) * jnp.exp(-self._w_weights * h))
        vector = self.v_vector(x, h, Lg)

        # compute new czero
        czero = vector @ self.U @ vector.T - self.delta(x, h, Lf) ** 2

        return czero

    def adjust_learning_gain(
        self, x: NDArray, h: float, dBdw: NDArray, dBdx: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """Adjusts the learning rate so that the control terms are not working in opposition.

        Arguments
        ---------
        x (NDArray): state vector
        h (float): ccbf value
        dBdw (NDArray): partial of consolidated CBF with respect to weights w
        dBdx (NDArray): partial of consolidated CBF with respect to weights x

        Returns
        -------
        w_dot_drift (NDArray): drift term of weight derivatives
        w_dot_contr (NDArray): control term of weight derivatives
        """
        p = 1.0
        if h < 0.1:
            weights_term = dBdw @ self._w_dot_contr
            control_term = dBdx @ g(x)

            # controls to weights ratio (5:1)
            theta_a = jnp.arctan2(control_term[1], control_term[0])
            max_theta_diff = jnp.min(
                [abs(theta_a % (jnp.pi / 2)), abs(jnp.pi / 2 - theta_a % (jnp.pi / 2))]
            )

            a = control_term[0]
            c = control_term[1]
            b = weights_term[0]
            d = weights_term[1]

            beta = jnp.linalg.norm(control_term) / 10.0
            F = theta(a, b, c, d, p)
            while F > (max_theta_diff - 0.05):
                dFdp = theta_gradient(a, b, c, d, p)
                p -= beta * dFdp
                if p < 0 or p > 1:
                    p = 0
                    break
                p = jnp.clip(p, 0, 1)
                F = theta(a, b, c, d, p)

        self._w_dot_drift *= p
        self._w_dot_contr *= p
        # print(f"new rate: {p} -> w_dot_drift = {self._w_dot_drift}")

        return self._w_dot_drift, self._w_dot_contr

    @property
    def w_weights(self) -> NDArray:
        """Getter for _w_weights."""
        return self._w_weights

    @w_weights.setter
    def w_weights(self, newVals: NDArray) -> None:
        """Setter for _w_weights.

        Arguments:
            newVals (NDArray): new/updated kWeights values

        Returns:
            None

        """
        if newVals.shape[0] == self._w_weights.shape[0]:
            self._w_weights = newVals
        else:
            raise ValueError("Error updating w_weights!")

    @property
    def w_desired(self) -> NDArray:
        """Getter for _w_weights."""
        return self._w_desired

    @property
    def w_dot(self) -> NDArray:
        """Getter for _w_dot."""
        return self._w_dot

    @property
    def w_dot_drift_f(self) -> NDArray:
        """Getter for _w_dot_drift_f."""
        return self._w_dot_drift_f

    @property
    def b(self) -> NDArray:
        """Getter for input_constraint_function."""
        return self._czero

    @property
    def dbdt(self) -> NDArray:
        """Getter for input_constraint_function."""
        return self._grad_czero_t

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
    uMax = jnp.array([10.0, 10.0])
    adapt = AdaptationLaw(nWeights, uMax)
