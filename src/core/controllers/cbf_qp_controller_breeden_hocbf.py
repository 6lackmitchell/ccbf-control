import builtins
import numpy as np
from sys import exit
from importlib import import_module
from nptyping import NDArray
from typing import Callable, List
from scipy.linalg import block_diag
from core.solve_cvxopt import solve_qp_cvxopt
from core.controllers.controller import Controller
from core.cbfs import Cbf

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

global integrated_error
integrated_error = np.zeros((25,))


class CbfQpController(Controller):

    _stochastic = False
    _generate_cbf_condition = None
    _dt = None

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
        super().__init__()
        self.u_max = u_max
        self.objective = objective_function
        self.nominal_controller = nominal_controller
        self.cbfs_individual = cbfs_individual
        self.cbfs_pairwise = cbfs_pairwise
        self.ignored_agents = ignore
        self.code = 0
        self.status = "Initialized"

        # Control Parameters
        self.n_controls = len(u_max)
        self.n_agents = nAgents
        self.n_dec_vars = 1
        self.desired_class_k = 0.1
        self.max_class_k = 1e6

        self.cbf_vals = np.zeros(
            (len(cbfs_individual) + (self.n_agents - 1) * len(cbfs_pairwise)),
        )
        self.dhdx = np.zeros((self.cbf_vals.shape[0], 4))
        # self.dhdx = np.zeros((self.cbf_vals.shape[0], 5))
        self.d2hdx2 = np.zeros((self.cbf_vals.shape[0], 4, 4))
        # self.d2hdx2 = np.zeros((self.cbf_vals.shape[0], 5, 5))

        # Define individual input constraints
        self.au = block_diag(*self.n_controls * [np.array([[1, -1]]).T])
        self.bu = np.tile(
            np.array(self.u_max).reshape(self.n_controls, 1), self.n_controls
        ).flatten()

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
                z = np.delete(z, ignore, 0)
                if ego > ignore:
                    ego = ego - 1

        # Partition state into ego and other
        ze = z[ego, :]
        zo = np.vstack([z[:ego, :], z[ego + 1 :, :]])

        # Compute nominal control input for ego only -- assume others are zero
        z_copy_nom = z.copy()
        z_copy_nom[self.ego_id] = z[ego]
        u_nom = np.zeros((len(z), 2))
        u_nom[ego, :], code_nom, status_nom = self.nominal_controller.compute_control(t, z_copy_nom)
        u_nom[ego, :] = u_nom[ego, :] + integrated_error[ego]
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
                self.u = np.zeros((self.n_controls,))

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
            Q, p = self.objective(np.append(u_nom.flatten(), alpha_nom), ze[:2])
            Au = block_diag(*(na + self.n_dec_vars) * [self.au])[:-2, :-1]
            bu = np.append(np.array(na * [self.bu]).flatten(), self.n_dec_vars * [100, 0])
        else:
            Q, p = self.objective(u_nom.flatten(), ze[:2])
            Au = block_diag(*(na) * [self.au])
            bu = np.array(na * [self.bu]).flatten()

        # Initialize inequality constraints
        lci = 5
        Ai = np.zeros((lci + len(zr), self.n_controls * na + self.n_dec_vars))
        bi = np.zeros((lci + len(zr),))

        # Iterate over individual CBF constraints
        R = [0.5, 0.5, 0.5]
        cx = [1.0, 1.5, 2.4]
        cy = [1.0, 2.25, 1.5]
        a_max = -4.0
        for cc, cbf in enumerate(self.cbfs_individual):
            if cc < 3:
                dx = ze[0] - cx[cc]
                dy = ze[1] - cy[cc]
                h = dx**2 + dy**2 - R[cc] ** 2
                h0 = dx**2 + dy**2 - R[cc] ** 2
                hdot = 2 * (dx * ze[2] + dy * ze[3])
                Lf2h = 2 * (ze[2] ** 2 + ze[3] ** 2)
                LgLfh = 2 * np.array([dx, dy])
                if hdot > 0:
                    H = h
                    LfH = hdot
                    LgH = np.array([0, 0])
                else:
                    H = h + hdot**2 / (2 * a_max)
                    LfH = hdot * (1 + Lf2h / a_max)
                    LgH = hdot * LgLfh / a_max
            else:
                H = (1 - ze[cc - 1]) * (ze[cc - 1] + 1)
                LfH = 0.0
                if cc == 3:
                    LgH = -2 * ze[cc - 1] * np.array([1, 0])
                else:
                    LgH = -2 * ze[cc - 1] * np.array([0, 1])

            h = H
            Lfh = LfH
            Lgh = np.zeros((self.n_controls * na,))
            Lgh[self.n_controls * ego : (ego + 1) * self.n_controls] = LgH

            Ai[cc, :], bi[cc] = self.generate_cbf_condition(cbf, h, Lfh, Lgh, cc)
            self.cbf_vals[cc] = h
            if h0 < 0:
                self.safety = False

        A = np.vstack([Au, Ai])
        b = np.hstack([bu, bi])

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
        u = np.array(solution["x"][self.n_controls * ego : self.n_controls * (ego + 1)]).flatten()
        self.u = np.clip(u, -self.u_max, self.u_max)
        # Assign other agents' controls if this is a centralized node
        if hasattr(self, "centralized_agents"):
            for agent in self.centralized_agents:
                agent.u = np.array(
                    solution["x"][agent.nu * agent.id : self.n_controls * (agent.id + 1)]
                ).flatten()
