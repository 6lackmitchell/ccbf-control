import jax.numpy as jnp
from nptyping import NDArray
from typing import Callable, List
from scipy.linalg import block_diag
from core.solve_cvxopt import solve_qp_cvxopt
from models.model import Model
from core.controllers.controller import Controller

from core.cbfs.cbf import Cbf


class CbfQpController(Controller):

    _stochastic = False
    _generate_cbf_condition = None
    _dt = None

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
        super().__init__()
        self.model = model
        self.nominal_controller = nominal_controller
        self.objective = objective_function
        self.cbfs_individual = cbfs_individual
        self.cbfs_pairwise = cbfs_pairwise
        self.ignored_agents = ignore
        self.code = 0
        self.status = "Initialized"

        # parameters
        self.n_states = model.n_states
        self.n_controls = model.n_controls
        self.n_agents = n_agents
        self.n_dec_vars = 0
        self.desired_class_k = 1.0
        self.max_class_k = 1e6
        self.u_max = model.u_max

        # cbf parameters
        self.cbf_vals = jnp.zeros(
            (len(cbfs_individual) + (self.n_agents - 1) * len(cbfs_pairwise)),
        )
        self.dhdt = jnp.zeros((self.cbf_vals.shape[0],))
        self.dhdx = jnp.zeros((self.cbf_vals.shape[0], 5))
        self.d2hdtdx = jnp.zeros((self.cbf_vals.shape[0], 5))
        self.d2hdx2 = jnp.zeros((self.cbf_vals.shape[0], 5, 5))

        # Define individual input constraints
        self.au = block_diag(*self.n_controls * [jnp.array([[1, -1]]).T])
        self.bu = jnp.tile(jnp.array(self.u_max).reshape(self.n_controls, 1), 2).flatten()

    def _compute_control(
        self, t: float, z: NDArray, cascaded: bool = False
    ) -> (NDArray, NDArray, int, str, float):
        """Computes the vehicle's control input based on a cascaded approach: first, the CBF constraints attempt to
        filter out unsafe inputs on the first level. If no safe control exists, then all control inputs are eligible
        for safety filtering.

        INPUTS
        ------
        t: time (in sec)
        z: full state vector for all vehicles
        extras: anything else

        OUTPUTS
        ------
        u_act: actual control input used in the system
        u_nom: nominal input used if safety not considered
        code: error/success code
        status: more info on error/success

        """
        global integrated_error
        code = 0
        status = "Incomplete"

        # Ignore agent if necessary (i.e. if comparing controllers for given initial conditions)
        ego = self.ego_id
        if self.ignored_agents is not None:
            self.ignored_agents.sort(reverse=True)
            for ignore in self.ignored_agents:
                z = jnp.delete(z, ignore, 0)
                if ego > ignore:
                    ego = ego - 1

        # Partition state into ego and other
        ze = z[ego, :]
        zo = jnp.vstack([z[:ego, :], z[ego + 1 :, :]])

        # Compute nominal control input for ego only -- assume others are zero
        z_copy_nom = z.copy()
        z_copy_nom[self.ego_id] = z[ego]
        u_nom = jnp.zeros((len(z), 2))
        u0, code_nom, status_nom = self.nominal_controller.compute_control(t, z_copy_nom)
        if self.u_nom is None:
            self.u_nom = u0
        u_nom = u_nom.at[ego, :].set(u0)
        self.u_nom = u_nom[ego, :]

        tuning_nominal = False
        if tuning_nominal:
            self.u = self.u_nom
            return self.u, 1, "Optimal"

        if not cascaded:
            # Get matrices and vectors for QP controller
            Q, p, A, b, G, h = self.formulate_qp(t, ze, zo, u_nom, ego)

            # Solve QP
            sol = solve_qp_cvxopt(Q, p, A, b, G, h)

            # Check solution
            if "code" in sol.keys():
                code = sol["code"]
                status = sol["status"]
                self.assign_control(sol, ego)
                if abs(self.u[0]) > 1e-3:
                    pass
            else:
                status = "Divide by Zero"
                self.u = jnp.zeros((self.n_controls,))

        else:
            pass

        if not code:
            print(A[-1, :])
            print(b[-1])
            print("wtf")

        return self.u, code, status

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

        # Configure QP Matrices
        # Q, p: objective function
        # Au, bu: input constraints
        if self.n_dec_vars > 0:
            alpha_nom = 0.1
            Q, p = self.objective(jnp.append(u_nom.flatten(), alpha_nom), ze[:2])
            Au = block_diag(*(na + self.n_dec_vars) * [self.au])[:-2, :-1]
            bu = jnp.append(jnp.array(na * [self.bu]).flatten(), self.n_dec_vars * [100, 0])
        else:
            Q, p = self.objective(u_nom.flatten(), ze[:2])
            Au = block_diag(*(na) * [self.au])
            bu = jnp.array(na * [self.bu]).flatten()

        # Initialize inequality constraints
        lci = 8
        Ai = jnp.zeros((lci + len(zr), self.n_controls * na + self.n_dec_vars))
        bi = jnp.zeros((lci + len(zr),))

        # Iterate over individual CBF constraints
        R = 0.45
        cx = [0.8, 1.25, 2.5, 2.0, 0.8]
        cy = [1.1, 2.25, 1.75, 0.25, -0.25]
        offset = 0.25
        # a_max = -4.0  # double integrator
        a_max0 = -30.0  # bicycle model
        vx = self.model.f()[0]
        vy = self.model.f()[1]
        phidot = self.model.f()[2]
        for cc, cbf in enumerate(self.cbfs_individual):

            if cc < 5:
                gain = 5.0
                a_max = a_max0 * gain
                dx = ze[0] - (cx[cc] + offset)
                dy = ze[1] - (cy[cc] + offset)
                h = (dx**2 + dy**2 - R**2) * gain
                h0 = (dx**2 + dy**2 - R**2) * gain
                hdot = 2 * (dx * vx + dy * vy) * gain
                Lf2h = (2 * (vx**2 + vy**2) + ze[3] * phidot * (vx - vy)) * gain
                LgLfh = (
                    2
                    * jnp.array(
                        [
                            dx * (-ze[3] * jnp.sin(ze[2]) / jnp.cos(ze[4]) ** 2)
                            + dy * (ze[3] * jnp.cos(ze[2]) / jnp.cos(ze[4]) ** 2),
                            dx * (jnp.cos(ze[2]) - jnp.sin(ze[2]) * jnp.tan(ze[4]))
                            + dy * (jnp.sin(ze[2]) + jnp.cos(ze[2]) * jnp.tan(ze[4])),
                        ]
                    )
                    * gain
                )

                if hdot > 0:
                    H = h
                    LfH = hdot
                    LgH = jnp.array([0, 0])
                else:
                    H = h + hdot**2 / (2 * a_max)
                    LfH = hdot * (1 + Lf2h / a_max)
                    LgH = hdot * LgLfh / a_max

            elif cc == 5:
                a_max = a_max0
                T = 5.0
                Ri = 4.0
                Rf = 0.1
                # reach
                dx = ze[0] - (2)
                dy = ze[1] - (2)

                tt = jnp.array([T, t]).min()

                h = Rf**2 + Ri**2 * (1 - tt / T) - dx**2 - dy**2
                h0 = Rf**2 + Ri**2 * (1 - tt / T) - dx**2 - dy**2
                if t < T:
                    hdot = -2 * (dx * vx + dy * vy) - Ri**2 / T
                else:
                    hdot = -2 * (dx * vx + dy * vy)
                Lf2h = -2 * (vx**2 + vy**2) - ze[3] * phidot * (vx - vy)
                LgLfh = -2 * jnp.array(
                    [
                        dx * (-ze[3] * jnp.sin(ze[2]) / jnp.cos(ze[4]) ** 2)
                        + dy * (ze[3] * jnp.cos(ze[2]) / jnp.cos(ze[4]) ** 2),
                        dx * (jnp.cos(ze[2]) - jnp.sin(ze[2]) * jnp.tan(ze[4]))
                        + dy * (jnp.sin(ze[2]) + jnp.cos(ze[2]) * jnp.tan(ze[4])),
                    ]
                )

                if hdot > 0:
                    H = h
                    LfH = hdot
                    LgH = jnp.array([0, 0])
                else:
                    H = h + hdot**2 / (2 * a_max)
                    LfH = hdot * (1 + Lf2h / a_max)
                    LgH = hdot * LgLfh / a_max

            elif cc == 6:
                # speed
                speed_limit = 2.0
                h = speed_limit**2 - ze[3] ** 2
                LfH = 0.0
                LgH = jnp.array([0.0, -2 * ze[3]])
            elif cc == 7:
                # slip
                slip_limit = jnp.pi / 3
                h = slip_limit**2 - ze[4] ** 2
                LfH = 0.0
                LgH = jnp.array([-2 * ze[4], 0.0])

            h = H
            Lfh = LfH
            Lgh = LgH

            Ai_new, bi_new = self.generate_cbf_condition(cbf, h, Lfh, Lgh, cc, adaptive=False)
            Ai = Ai.at[cc, :].set(Ai_new)
            bi = bi.at[cc].set(bi_new)
            self.cbf_vals = self.cbf_vals.at[cc].set(h)
            if h0 < 0:
                self.safety = False

        A = jnp.vstack([Au, Ai])
        b = jnp.hstack([bu, bi])

        return Q, p, A, b, None, None

    def generate_cbf_condition(
        self, cbf: Cbf, h: float, Lfh: float, Lgh: NDArray, idx: int, adaptive: bool = False
    ) -> (NDArray, float):
        """Calls the child _generate_cbf_condition method."""
        if self._generate_cbf_condition is not None:
            return self._generate_cbf_condition(cbf, h, Lfh, Lgh, idx, adaptive)
        else:
            return cbf.generate_cbf_condition(h, Lfh, Lgh, adaptive)

    def assign_control(self, solution: dict, ego: int) -> None:
        """Assigns the control solution to the appropriate agent."""
        u = jnp.array(solution["x"][self.n_controls * ego : self.n_controls * (ego + 1)]).flatten()
        self.u = jnp.clip(u, -self.u_max, self.u_max)
        # Assign other agents' controls if this is a centralized node
        if hasattr(self, "centralized_agents"):
            for agent in self.centralized_agents:
                agent.u = jnp.array(
                    solution["x"][agent.nu * agent.id : self.n_controls * (agent.id + 1)]
                ).flatten()
