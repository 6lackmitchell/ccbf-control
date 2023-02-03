import builtins
import numpy as np
import symengine as se
from importlib import import_module
from core.cbfs.cbf_wrappers import symbolic_cbf_wrapper_singleagent

vehicle = builtins.PROBLEM_CONFIG["vehicle"]
control_level = builtins.PROBLEM_CONFIG["control_level"]
mod = "models." + vehicle + "." + control_level + ".models"

# Programmatic import
try:
    module = import_module(mod)
    globals().update({"f": getattr(module, "f")})
    globals().update({"ss": getattr(module, "sym_state")})
except ModuleNotFoundError as e:
    print("No module named '{}' -- exiting.".format(mod))
    raise e

# Defining Physical Params
gain = 1.0

# beta CBF Symbolic
h_symbolic = gain * ss[0]
dhdx_symbolic = (se.DenseMatrix([h_symbolic]).jacobian(se.DenseMatrix(ss))).T
d2hdx2_symbolic = dhdx_symbolic.jacobian(se.DenseMatrix(ss))
h_func = symbolic_cbf_wrapper_singleagent(h_symbolic, ss)
dhdx_func = symbolic_cbf_wrapper_singleagent(dhdx_symbolic, ss)
d2hdx2_func = symbolic_cbf_wrapper_singleagent(d2hdx2_symbolic, ss)


def h(ego):
    return h_func(ego)


def dhdx(ego):
    ret = dhdx_func(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2hdx2(ego):
    ret = d2hdx2_func(ego)

    return np.squeeze(np.array(ret).astype(np.float64))
