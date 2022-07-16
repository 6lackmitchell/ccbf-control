import sympy as sp
import numpy as np
from nptyping import NDArray


# These all need to be wrappers as well, so that the function call is just the evaluation/substitution
def symbolic_cbf_wrapper_singleagent(cbf_symbolic, ss):
    """Wrapper for symbolic CBFs and derivatives. """

    def cbf(z: NDArray,
            is_symbolic: bool = False):
        """Single agent CBF formulation -- can be CBF or Nth partial derivatives. """
        if is_symbolic:
            return cbf_symbolic

        else:
            ret = cbf_symbolic.subs([(sym, zz) for sym, zz in zip(ss, z)])

            if type(ret) is sp.Float:
                return ret
            else:
                return np.squeeze(np.array(ret).astype(np.float64))

    return cbf


def symbolic_cbf_wrapper_multiagent(cbf_symbolic, sse, sso):
    """Wrapper for symbolic CBFs and derivatives. """

    def cbf(ze: NDArray,
            zo: NDArray,
            is_symbolic: bool = False):
        """Multi-agent CBF formulation -- can be CBF or Nth partial derivatives. """
        if is_symbolic:
            return cbf_symbolic

        else:
            ss = np.concatenate([sse, sso])
            zz = np.concatenate([ze, zo])
            ret = cbf_symbolic.subs([(s, z) for s, z in zip(ss, zz)])

            if type(ret) is sp.Float:
                return ret
            else:
                return np.squeeze(np.array(ret).astype(np.float64))

    return cbf
