"""consolidated_cbf_controller.py

Provides interface to the ConsolidatedCbfController class.

"""
import jax.numpy as jnp
from jax import jacfwd, jacrev, jit
from jax.config import config
from typing import Callable, List, Optional, Tuple
from nptyping import NDArray
from scipy.linalg import block_diag

# from ..cbfs.cbf import Cbf
from models.model import Model
from core.controllers.cbf_qp_controller import CbfQpController
from core.controllers.controller import Controller

# jnp.random.seed(1)
config.update("jax_enable_x64", True)


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
        self.alpha = self.desired_class_k
        self.czero1 = 0
        kZero = 1.0

        # states
        self.n_states = model.n_states

        # cbf
        self.cbfs = cbfs_individual + cbfs_pairwise

        # indices
        tidxs = jnp.s_[0]
        xidxs = jnp.s_[1 : self.n_states + 1]
        widxs = jnp.s_[self.n_states + 1 : self.n_states + 1 + len(self.cbfs)]

        # # consolidated cbf
        H = lambda z: 1 - jnp.sum(
            jnp.array(
                [
                    jnp.exp(-z[self.n_states + 1 + cc] * cbf._h(z[0], z[1 : self.n_states + 1]))
                    for cc, cbf in enumerate(self.cbfs)
                ]
            )
        )
        dHdt = lambda z: jacfwd(self.H)(z)[tidxs]
        dHdx = lambda z: jacfwd(self.H)(z)[xidxs]
        dHdw = lambda z: jacfwd(self.H)(z)[widxs]

        self.H = jit(H)
        self.dHdt = jit(dHdt)
        self.dHdx = jit(dHdx)
        self.dHdw = jit(dHdw)

        # test
        t = 0.0
        x = self.model.x
        w = kZero * jnp.ones((8,))
        vec = jnp.hstack([t, x, w])
        print(f"H: {self.H(vec)}")

        # initialize adaptation law
        self.w_weights = kZero * jnp.ones((nCBFs,))
        self.w_des = kZero * jnp.ones((nCBFs,))
        self.adapter = AdaptationLaw(self.model, nCBFs, kZero=kZero, alpha=self.alpha)

        # assign adapter index objects
        self.adapter.tidxs = tidxs
        self.adapter.xidxs = xidxs
        self.adapter.widxs = widxs

        # assign adapter functions
        self.adapter.H = self.H
        self.adapter.dHdt = self.dHdt
        self.adapter.dHdw = self.dHdw
        self.adapter.dHdx = self.dHdx

        # finish adapter setup
        self.adapter.dt = self._dt
        self.adapter.cbfs = self.cbfs
        self.adapter.setup()

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

        # Iterate over individual CBF constraints
        for cc, cbf in enumerate(self.cbfs_individual):
            self.cbf_vals = self.cbf_vals.at[cc].set(cbf.h(t, ze))
            self.dhdt = self.dhdt.at[cc].set(cbf.dhdt(t, ze))
            self.dhdx = self.dhdx.at[cc].set(cbf.dhdx(t, ze))
            self.d2hdtdx = self.d2hdtdx.at[cc].set(cbf.d2hdtdx(t, ze))
            self.d2hdx2 = self.d2hdx2.at[cc].set(cbf.d2hdx2(t, ze))

        # Iterate over pairwise CBF constraints
        for cc, cbf in enumerate(self.cbfs_pairwise):
            # Iterate over all other vehicles
            for ii, zo in enumerate(zr):
                # other = ii + (ii >= ego)
                idx = lci + cc * zr.shape[0] + ii

                self.cbf_vals[idx] = cbf.h0(t, ze, zo)
                self.dhdt[idx] = cbf.dhdt(t, ze, zo)
                self.dhdx[idx] = cbf.dhdx(t, ze, zo)
                self.d2hdtdx[idx] = cbf.d2hdtdx(t, ze, zo)
                self.d2hdx2[idx] = cbf.d2hdx2(t, ze, zo)

        # Format inequality constraints
        Ai, bi = self.generate_consolidated_cbf_condition(t, ze, ego)

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
            # Au = block_diag(*(self.n_agents + self.n_dec_vars) * [self.au])  # [:-2, :-1]
            Au = block_diag(*(self.n_agents + self.n_dec_vars) * [self.au])[:-2, :-1]
            bu = jnp.append(
                jnp.array(self.n_agents * [self.bu]).flatten(),
                self.n_dec_vars * jnp.array([self.max_class_k, 0]),
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
        # get new weights
        self.adapter.w_weights = jnp.clip(
            self.adapter.w_weights, self.adapter.w_min * 1.01, self.adapter.w_max * 0.99
        )
        self.w_weights = self.adapter.w_weights

        # precompute quantities
        self.adapter.precompute()

        # bring initial parameters into feasible region if necessary
        if t == 0 and self.adapter.b > -1e-1:
            self.adapter.w_weights = self.adapter.w_gradient_descent()

        # compute controlled w_dot
        w_dot_contr = self.adapter.w_dot_controlled()

        # compute drift w_dot
        w_dot_drift = self.adapter.w_dot_drift()

        # consolidated cbf H(t, w, x)
        self.c_cbf = H = self.consolidated_cbf()

        # c-cbf partial derivatives
        dHdt = self.dHdt(self.z)
        dHdx = self.dHdx(self.z)
        dHdw = self.dHdw(self.z)

        # b3 partial derivatives
        dbdt = self.adapter._grad_b3_t
        dbdx = self.adapter._grad_b3_x
        dbdw = self.adapter._grad_b3_w

        # big B dynamics
        B = -self.adapter.b * H
        dBdt = -(self.adapter.b * dHdt + dbdt * H)
        dBdx = -(self.adapter.b * dHdx + dbdx * H)
        dBdw = -(self.adapter.b * dHdw + dbdw * H)

        # cbf dynamics
        Hdot_drift = dHdt + dHdw @ w_dot_drift + dHdx @ self.model.f()
        Hdot_contr = dHdw @ w_dot_contr + dHdx @ self.model.g()
        alpha_H = self.alpha * H**3
        # Hdot_drift = dBdt + dBdw @ w_dot_drift + dBdx @ self.model.f(x)
        # Hdot_contr = dBdw @ w_dot_contr + dBdx @ self.model.g(x)
        # alpha_H = self.alpha * B

        # CBF Condition (fixed class K)
        qp_scale = 1 / jnp.array([1e-6, abs(self.adapter.b)]).max()
        a_mat = jnp.append(-Hdot_contr, -alpha_H)
        b_vec = jnp.array([Hdot_drift]).flatten()
        a_mat *= qp_scale
        b_vec *= qp_scale
        # a_mat = jnp.append(-Hdot_contr, 0)
        # b_vec = jnp.array([Hdot_drift + alpha_H]).flatten()
        # a_mat *= qp_scale
        # b_vec *= qp_scale

        # test bdot
        if self.adapter.b > -1e-1:
            bdot_drift = dbdt + dbdw @ w_dot_drift + dbdx @ self.model.f()
            bdot_contr = dbdw @ w_dot_contr + dbdx @ self.model.g()
            print(f"bdot: {-bdot_drift} + {-bdot_contr}u >= {-(self.adapter.b**3)}")
            print(f"Hdot: {Hdot_drift} + {Hdot_contr}u >= {-alpha_H}")
            print(f"b: {self.adapter.b}")
            print(f"Time: {t}")
            pass

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

    @property
    def z(self):
        """Computes the z vector (concatenated time, state, and weights)."""
        return jnp.hstack([self.adapter.t, self.model.x, self.adapter.weights])

    @property
    def b(self):
        return self.adapter.b


class AdaptationLaw:
    """Computes the parameter adaptation for the ConsolidatedCbfController
    class.

    Attributes:
        kWeights (NDArray): values of current c-cbf weights k
    """

    def __init__(
        self,
        model: Model,
        nWeights: int,
        kZero: Optional[float] = 0.5,
        alpha: Optional[float] = 0.1,
    ):
        """Initializes class attributes.

        Arguments:
            nWeights (int): number of weights/CBFs to be consolidated
            uMax (NDArray): maximum control input vector
            kZero (float, Opt)

        """
        # model
        self.model = model

        # dimensions
        self.n_states = self.model.n_states
        self.n_controls = self.model.n_controls
        self.n_weights = nWeights

        # time
        self.t = 0.0

        # control contraint matrix
        self.u_max = self.model.u_max
        self.U = self.u_max[:, jnp.newaxis] @ self.u_max[jnp.newaxis, :]

        # cbfs
        self.cbfs = None

        # class K parameters
        self.alpha = alpha
        self.cubic = False

        # k weights and derivatives
        self._w_weights = kZero * jnp.ones((nWeights,))
        self._w_desired = kZero * jnp.ones((nWeights,))
        self._w_dot = jnp.zeros((nWeights,))
        self._w_dot_drift = jnp.zeros((nWeights,))
        self._w_dot_contr = jnp.zeros((nWeights, len(self.u_max)))

        # logging variables
        self.czero_val1 = 0.0
        self.czero_val2 = 0.0

        # wdot filter design (2nd order)
        self.wn = 50.0
        self.zeta = 0.707  # butterworth
        self._filter_order = 1
        self._w_dot_f = jnp.zeros((nWeights,))
        self._w_dot_drift_f = jnp.zeros((nWeights,))
        self._w_2dot_drift_f = jnp.zeros((nWeights,))
        self._w_3dot_drift_f = jnp.zeros((nWeights,))
        self._w_dot_contr_f = jnp.zeros((nWeights, self.n_controls))
        self._w_2dot_contr_f = jnp.zeros((nWeights, self.n_controls))
        self._w_3dot_contr_f = jnp.zeros((nWeights, self.n_controls))

        # function placeholders
        self.H = None
        self.dHdt = None
        self.dHdw = None
        self.dHdx = None

        # phi function placeholders
        self.c = None
        self.b1 = None
        self.b2 = None
        self.b3 = None
        self.d_b3_dt = None
        self.d_b3_dx = None
        self.d_b3_dw = None
        self._b3 = None
        self._grad_b3_t = None
        self._grad_b3_x = None
        self._grad_b3_w = None
        self.phi = None
        self.d_phi_dw = None
        self.d2_phi_dwdt = None
        self.d2_phi_dwdx = None
        self.d2_phi_dw2 = None
        self._grad_phi_w = None
        self._grad_phi_wx = None
        self._grad_phi_wt = None
        self._grad_phi_ww_inv = None
        self._grad_phi_w_f = None

        # # convexity parameter
        # self.s = 1e-9

        # # Gains and Parameters -- Testing
        # self.alpha = alpha
        # self.eta_mu = self.eta_nu = 0.01
        # self.w_dot_gain = 1.0
        # hc = 1e6  # high cost
        # lc = 1e-3  # low cost
        # self.Q = 1 * jnp.diag(jnp.array([hc, hc, hc, hc, hc, hc, lc, lc]))  # Cost function gain
        # self.P = 50 * jnp.eye(nWeights)
        # self.w_des_gain = 1.0
        # self.w_min = 0.01
        # self.w_max = 50.0
        # self.b3_gain = 1.0
        # self.ci_gain = 1.0

        # convexity parameter
        self.s = 1

        # # Gains and Parameters -- Testing
        # self.alpha = alpha
        # self.eta_mu = self.eta_nu = 0.01
        # self.w_dot_gain = 1.0
        # # hc = 1e6  # high cost
        # # lc = 1e-3  # low cost
        # # self.Q = jnp.diag(jnp.array([hc, hc, hc, hc, hc, hc, lc, lc]))  # Cost function gain
        # # self.P = 1 * jnp.eye(nWeights)
        # # self.P = 1e1 * jnp.eye(nWeights)
        # self.P = 1e-3 * jnp.eye(nWeights)
        # self.w_des_gain = 1.0
        # self.w_min = 0.01
        # self.w_max = 50.0
        # self.b3_gain = 1.0
        # self.ci_gain = 1.0

        # # Gains and Parameters -- Oscillator
        # self.alpha = alpha
        # self.eta_mu = self.eta_nu = 0.01
        # self.w_dot_gain = 1.0
        # self.Q = 1 * jnp.eye(nWeights)  # Cost function gain
        # self.P = 100 * jnp.eye(nWeights)
        # self.w_des_gain = 1.0
        # self.w_min = 0.01
        # self.w_max = 50.0
        # self.b3_gain = 1.0
        # self.ci_gain = 1.0

        # Gains and Parameters -- Bicycle
        self.alpha = alpha
        self.eta_mu = self.eta_nu = 0.01
        self.w_dot_gain = 1.0
        self.Q = 1 * jnp.eye(nWeights)  # Cost function gain
        self.P = 100 * jnp.eye(nWeights)
        self.w_des_gain = 1.0
        self.w_min = 0.01
        self.w_max = 50.0
        self.b3_gain = 1.0
        self.ci_gain = 1.0

    def setup(self) -> None:
        """Generates symbolic functions for the cost function and feasible region
        of the optimization problem.

        Arguments:
            None

        Returns:
            None

        """
        self.setup_cost()  # defines cost function
        self.setup_b1()  # defines w > wmin constraint function
        self.setup_b2()  # defines w < wmax constraint function
        self.setup_b3()  # defines sufficient control authority constraint function

        # defines augmented unconstrained cost function
        self.setup_phi()

    def setup_phi(self) -> None:
        """Generates symbolic functions for the augmented unconstrained cost function. Must be called
        after the cost, b1, b2, and b3 functions are set up.

        Arguments:
            None

        Returns:
            None

        """
        eps = 0.1

        def phi(z):
            return self.c(z) - 1 / z[-1] * (
                jnp.sum(jnp.log(-self.b1(z)))
                + jnp.sum(jnp.log(-self.b2(z)))
                + jnp.log(-self.b3(z) - eps) / ((-self.b3(z) - eps) ** 3)
            )

        def d_phi_dw(z):
            return jacrev(phi)(z)[self.widxs]

        def d2_phi_dwdt(z):
            return jacfwd(jacrev(phi))(z)[self.widxs, self.tidxs]

        def d2_phi_dwdx(z):
            return jacfwd(jacrev(phi))(z)[self.widxs, self.xidxs]

        def d2_phi_dw2(z):
            return jacfwd(jacrev(phi))(z)[self.widxs, self.widxs]

        # just-in-time compilation
        self.phi = jit(phi)
        self.d_phi_dw = jit(d_phi_dw)
        self.d2_phi_dwdt = jit(d2_phi_dwdt)
        self.d2_phi_dwdx = jit(d2_phi_dwdx)
        self.d2_phi_dw2 = jit(d2_phi_dw2)

    def setup_cost(self) -> None:
        """Generates symbolic functions for the cost function and feasible region
        of the optimization problem.

        Arguments:
            None

        Returns:
            None

        """

        def c(z):
            return (
                1 / 2 * (z[self.widxs] - self.w_des(z)).T @ self.Q @ (z[self.widxs] - self.w_des(z))
            )

        self.c = jit(c)

    def setup_b1(self) -> None:
        """Generates symbolic functions bounding the weights from below.

        Arguments:
            None

        Returns:
            None

        """

        # w > w_min constraints
        def b1(z):
            return jnp.array(
                [self.w_min - z[self.n_states + 1 + cc] for cc in range(self.n_weights)]
            )

        self.b1 = jit(b1)

    def setup_b2(self) -> None:
        """Generates symbolic functions for bounding the weights from above.

        Arguments:
            None

        Returns:
            None

        """

        # w < w_max constraints
        def b2(z):
            return jnp.array(
                [z[self.n_states + 1 + cc] - self.w_max for cc in range(self.n_weights)]
            )

        self.b2 = jit(b2)

    def setup_b3(self) -> None:
        """Generates symbolic functions for validating the C-CBF.

        Arguments:
            None

        Returns:
            None

        """

        # def b3(z):
        #     return (
        #         self.eta_mu
        #         + self.eta_nu
        #         - self.dHdt(z)
        #         - self.dHdx(z) @ self.model.f(z[: self.n_states + 1])
        #         # - self.dHdw(z) @ self._w_dot_drift_f
        #         - self.alpha * self.H(z) ** 3
        #         - abs(
        #             self.dHdx(z)
        #             @ self.model.g(z[: self.n_states + 1])
        #             # + self.dHdw(z) @ self._w_dot_contr_f
        #         )
        #         @ self.u_max
        #     ) * self.b3_gain

        def b3(z):
            return (
                -1
                + jnp.sum(
                    jnp.array(
                        [
                            jnp.exp(
                                -2
                                * (
                                    cbf._dhdt(z[0], z[1 : self.n_states + 1])
                                    + cbf._dhdx(z[0], z[1 : self.n_states + 1])
                                    @ self.model.f(z[: self.n_states + 1])
                                    + abs(
                                        cbf._dhdx(z[0], z[1 : self.n_states + 1])
                                        @ self.model.g(z[: self.n_states + 1])
                                    )
                                    @ self.u_max
                                    + cbf.alpha(cbf._h(z[0], z[1 : self.n_states + 1]))
                                )
                            )
                            for cbf in self.cbfs
                        ]
                    )
                )
                * self.b3_gain
            )

        def d_b3_dt(z):
            return jacfwd(b3)(z)[self.tidxs]

        def d_b3_dx(z):
            return jacfwd(b3)(z)[self.xidxs]

        def d_b3_dw(z):
            return jacfwd(b3)(z)[self.widxs]

        def d2_b3_dwdt(z):
            return jacfwd(jacrev(b3))(z)[self.widxs, self.tidxs]

        # def b3(z):
        #     return (
        #         self.eta_mu
        #         + self.eta_nu
        #         - self.dHdt(z)
        #         - self.dHdx(z) @ self.model.f(z[: self.n_states + 1])
        #         - self.dHdw(z) @ self._w_dot_drift_f
        #         - self.alpha * self.H(z)
        #         + smooth_abs(
        #             self.dHdx(z) @ self.model.g(z[: self.n_states + 1])
        #             + self.dHdw(z) @ self._w_dot_contr_f
        #         )
        #         @ self.u_max
        #     ) * self.b3_gain

        # def b3(z):
        #     return (
        #         self.eta_mu
        #         + self.eta_nu
        #         - self.dHdt(z)
        #         - self.dHdx(z) @ self.model.f(z[: self.n_states + 1])
        #         - self.dHdw(z) @ self._w_dot_drift_f
        #         - self.alpha * self.H(z)
        #         - abs(
        #             self.dHdx(z) @ self.model.g(z[: self.n_states + 1])
        #             + self.dHdw(z) @ self._w_dot_contr_f
        #         )
        #         @ self.u_max
        #     ) * self.b3_gain

        def b3_a(z):
            return (
                self.eta_mu
                + self.eta_nu
                - self.dHdt(z)
                - self.dHdx(z) @ self.model.f(z[: self.n_states + 1])
                # - self.dHdw(z) @ self._w_dot_drift_f
                - self.alpha * self.H(z)
            )

        def b3_b(z):
            return (
                -abs(
                    self.dHdx(z)
                    @ self.model.g(z[: self.n_states + 1])
                    # + self.dHdw(z) @ self._w_dot_contr_f
                )
                @ self.u_max
            )

        self.b3 = jit(b3)
        self.b3_a = jit(b3_a)
        self.b3_b = jit(b3_b)
        self.d_b3_dt = jit(d_b3_dt)
        self.d_b3_dx = jit(d_b3_dx)
        self.d_b3_dw = jit(d_b3_dw)
        self.d2_b3_dwdt = jit(d2_b3_dwdt)

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
        max_idx = jnp.where(w_weights > self.w_max)
        w_weights = w_weights.at[max_idx].set(0.999 * self.w_max)
        min_idx = jnp.where(w_weights < self.w_min)
        w_weights = w_weights.at[min_idx].set(1.001 * self.w_min)
        w_dot = (w_weights - self._w_weights) / self.dt

        # attribute deviation to drift term
        self._w_dot_drift -= w_dot_0 - w_dot

        # set final w_dot value
        self._w_dot = w_dot

        return self._w_dot

    def precompute(self) -> NDArray:
        """Precomputes terms needed to compute the adaptation law.

        Arguments:
            x (NDArray): state vector
            h (NDArray): vector of candidate CBFs
            Lf (float): C-CBF drift term (includes filtered wdot)
            Lg (NDArray): matrix of stacked Lgh vectors

        Returns:
            None

        """
        # phi
        self._grad_phi_w = self.d_phi_dw(self.z)
        self._grad_phi_wt = self.d2_phi_dwdt(self.z)
        self._grad_phi_wx = self.d2_phi_dwdx(self.z)
        self._grad_phi_ww_inv = jnp.linalg.inv(self.d2_phi_dw2(self.z))
        # self._grad_phi_w_f = self._grad_phi_w

        # # filtered version for numerical stability (as in Fazlyab 2017 TAC)
        # gamma = self.wn / 5
        # if self._grad_phi_w_f is None:
        #     self._grad_phi_w_f = jnp.zeros(self._grad_phi_w.shape)
        # else:
        #     self._grad_phi_w_f += (
        #         -gamma * self._grad_phi_w_f + self.wn * self._grad_phi_w
        #     ) * self.dt

        # b3
        self._b3 = self.b3(self.z)
        self._grad_b3_t = self.d_b3_dt(self.z)
        self._grad_b3_x = self.d_b3_dx(self.z)
        self._grad_b3_w = self.d_b3_dw(self.z)
        self._grad_b3_wt = self.d2_b3_dwdt(self.z)
        # print(f"b3_parts: a = {self.b3_a(self.z)}, b = {self.b3_b(self.z)}")
        # print(
        #     f"etas: {self.eta_mu+self.eta_nu}, dHdt: {-self.dHdt(self.z)}, LfH: {- self.dHdx(self.z) @ self.model.f()}, alpha: {self.alpha * self.H(self.z)}"
        # )

        b3_val = (
            jnp.array(
                [
                    cbf.dhdt(self.z[0], self.z[1 : self.n_states + 1])
                    + cbf.dhdx(self.z[0], self.z[1 : self.n_states + 1]) @ self.model.f()
                    + abs(cbf.dhdx(self.z[0], self.z[1 : self.n_states + 1]) @ self.model.g())
                    @ self.u_max
                    + cbf.alpha(cbf.h(self.z[0], self.z[1 : self.n_states + 1]))
                    for cbf in self.cbfs
                ]
            )
            * self.b3_gain
        )
        # print(f"b3a: {b3_val[0]}")
        # print(f"b3b: {b3_val[1]}")
        # print(f"b3: {self._b3}")

    def w_dot_drift(self) -> NDArray:
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
        # compute w_dot_drift
        w_dot_drift_w = -self._grad_phi_ww_inv @ (self.P @ self._grad_phi_w)
        w_dot_drift_x = -self._grad_phi_ww_inv @ (self._grad_phi_wx @ self.model.f())
        w_dot_drift_t = -self._grad_phi_ww_inv @ self._grad_phi_wt
        w_dot_drift = (w_dot_drift_w + w_dot_drift_x + w_dot_drift_t) * self.w_dot_gain

        # assign to private var
        self._w_dot_drift = w_dot_drift

        return self._w_dot_drift

    def w_dot_controlled(self) -> NDArray:
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
        w_dot_contr = -self._grad_phi_ww_inv @ (self._grad_phi_wx @ self.model.g())
        w_dot_contr *= self.w_dot_gain

        if len(w_dot_contr.shape) > 1:
            self._w_dot_contr = w_dot_contr
            # self._w_dot_contr += self.wn * (w_dot_contr - self._w_dot_contr) * self.dt
        else:
            self._w_dot_contr = w_dot_contr[:, jnp.newaxis]

        return self._w_dot_contr

    def filter_init(self) -> None:
        """Initializes filter parameters."""

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
            self._w_2dot_drift_f = self.wn * (self._w_dot_drift - self._w_dot_drift_f)
            # self._w_2dot_drift_f = 0.5 / self.dt * (self._w_dot_drift - self._w_dot_drift_f)
            self._w_dot_drift_f += self._w_2dot_drift_f * self.dt

            self._w_2dot_contr_f = self.wn * (self._w_dot_contr - self._w_dot_contr_f)
            # self._w_2dot_contr_f = 0.5 / self.dt * (self._w_dot_contr - self._w_dot_contr_f)
            self._w_dot_contr_f += self._w_2dot_contr_f * self.dt

        # Compute final filtered w_dot
        self._w_dot_f = self._w_dot_drift_f + self._w_dot_contr_f @ u

        return self._w_dot_f

    # def w_des(self, z: NDArray) -> NDArray:
    #     """Computes the desired gains k for the constituent cbfs. This can be
    #     thought of as the nominal adaptation law (unconstrained).

    #     Arguments:
    #         h (NDArray): array of constituent cbf values

    #     Returns:
    #         w_des (NDArray)

    #     """

    #     w_des = jnp.ones((self.n_weights,)) * self.w_des_gain

    #     return jnp.clip(w_des, self.w_min, self.w_max)

    def w_des(self, z: NDArray) -> NDArray:
        """Computes the desired gains k for the constituent cbfs. This can be
        thought of as the nominal adaptation law (unconstrained).

        Arguments:
            h (NDArray): array of constituent cbf values

        Returns:
            w_des (NDArray)

        """
        w_des = jnp.array([cbf._h(z[0], z[1 : self.n_states + 1]) for cbf in self.cbfs])
        # w_des = jnp.ones((self.n_weights,)) * self.w_des_gain

        return w_des * self.w_des_gain

    def w_gradient_descent(self) -> float:
        """Runs gradient descent on the w_weights in order to increase the
        control authority at t=0.

        Arguments:
            None

        Returns:
            new_weights

        """
        # line search parameter
        beta = 1e-1

        # gradient descent
        count = 0
        max_b = -1e-1
        while self._b3 > max_b and count < 1e3:
            self._w_weights = jnp.clip(
                self._w_weights - beta * self._grad_phi_ww_inv @ self._grad_phi_w,
                self.w_min * 1.01,
                self.w_max * 0.99,
            )
            self.precompute()
            count += 1

        print(f"b3: {self._b3}")
        print(f"w:  {self.w_weights}")

        return self._w_weights

    # def adjust_learning_gain(
    #     self, x: NDArray, h: float, dBdw: NDArray, dBdx: NDArray
    # ) -> Tuple[NDArray, NDArray]:
    #     """Adjusts the learning rate so that the control terms are not working in opposition.

    #     Arguments
    #     ---------
    #     x (NDArray): state vector
    #     h (float): ccbf value
    #     dBdw (NDArray): partial of consolidated CBF with respect to weights w
    #     dBdx (NDArray): partial of consolidated CBF with respect to weights x

    #     Returns
    #     -------
    #     w_dot_drift (NDArray): drift term of weight derivatives
    #     w_dot_contr (NDArray): control term of weight derivatives
    #     """
    #     p = 1.0
    #     if h < 0.1:
    #         weights_term = dBdw @ self._w_dot_contr
    #         control_term = dBdx @ g(x)

    #         # controls to weights ratio (5:1)
    #         theta_a = jnp.arctan2(control_term[1], control_term[0])
    #         max_theta_diff = jnp.min(
    #             [abs(theta_a % (jnp.pi / 2)), abs(jnp.pi / 2 - theta_a % (jnp.pi / 2))]
    #         )

    #         a = control_term[0]
    #         c = control_term[1]
    #         b = weights_term[0]
    #         d = weights_term[1]

    #         beta = jnp.linalg.norm(control_term) / 10.0
    #         F = theta(a, b, c, d, p)
    #         while F > (max_theta_diff - 0.05):
    #             dFdp = theta_gradient(a, b, c, d, p)
    #             p -= beta * dFdp
    #             if p < 0 or p > 1:
    #                 p = 0
    #                 break
    #             p = jnp.clip(p, 0, 1)
    #             F = theta(a, b, c, d, p)

    #     self._w_dot_drift *= p
    #     self._w_dot_contr *= p
    #     # print(f"new rate: {p} -> w_dot_drift = {self._w_dot_drift}")

    #     return self._w_dot_drift, self._w_dot_contr

    @property
    def weights(self) -> NDArray:
        """Getter for _w_weights."""
        return self._w_weights

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
        return self._b3

    @property
    def dbdt(self) -> NDArray:
        """Getter for input_constraint_function."""
        return self._grad_b3_t

    @property
    def dbdw(self) -> NDArray:
        """Getter for input_constraint_function."""
        return self._grad_b3_w

    @property
    def dbdx(self) -> NDArray:
        """Getter for input_constraint_function."""
        return self._grad_b3_x

    @property
    def z(self):
        """Computes the z vector (concatenated time, state, and weights)."""
        return jnp.hstack([self.t, self.model.x, self.weights, self.s])


if __name__ == "__main__":
    nWeights = 5
    uMax = jnp.array([10.0, 10.0])
    adapt = AdaptationLaw(nWeights, uMax)
