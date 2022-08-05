import numpy as np
import builtins
from typing import Callable, List
from importlib import import_module
from nptyping import NDArray
from scipy.special import erf, erfinv
from scipy.linalg import block_diag
from .cbfs.cbf import Cbf
from .cbf_qp_controller import CbfQpController
from ..mathematics.analytical_functions import ramp, dramp

vehicle = builtins.PROBLEM_CONFIG['vehicle']
control_level = builtins.PROBLEM_CONFIG['control_level']
system_model = builtins.PROBLEM_CONFIG['system_model']
mod = 'simdycosys.' + vehicle + '.' + control_level + '.models'

# Programmatic import
try:
    module = import_module(mod)
    globals().update({'sigma': getattr(module, 'sigma_{}'.format(system_model))})
except ModuleNotFoundError as e:
    print('No module named \'{}\' -- exiting.'.format(mod))
    raise e


class ConsolidatedCbfController(CbfQpController):

    def __init__(self,
                 u_max: List,
                 nAgents: int,
                 objective_function: Callable,
                 nominal_controller: Callable,
                 cbfs_individual: List,
                 cbfs_pairwise: List,
                 ignore: List = None):
        super().__init__(u_max,
                         nAgents,
                         objective_function,
                         nominal_controller,
                         cbfs_individual,
                         cbfs_pairwise,
                         ignore)
        nCBF = len(self.cbf_vals)

    def _generate_cbf_condition(self,
                                cbf: Cbf,
                                h: float,
                                Lfh: float,
                                Lgh: float,
                                idx: int,
                                adaptive: bool = False) -> (NDArray, float):
        """Generates the matrix A and vector b for the Risk-Bounded CBF constraint of the form Au <= b."""

        return cbf.generate_cbf_condition()






