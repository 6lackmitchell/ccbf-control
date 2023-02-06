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
obstacle = 2.0

# beta CBF Symbolic
h_symbolic_1 = gain * (obstacle - ss[0]) ** 3
dhdx_symbolic_1 = (se.DenseMatrix([h_symbolic_1]).jacobian(se.DenseMatrix(ss))).T
d2hdx2_symbolic_1 = dhdx_symbolic_1.jacobian(se.DenseMatrix(ss))
h_func_1 = symbolic_cbf_wrapper_singleagent(h_symbolic_1, ss)
dhdx_func_1 = symbolic_cbf_wrapper_singleagent(dhdx_symbolic_1, ss)
d2hdx2_func_1 = symbolic_cbf_wrapper_singleagent(d2hdx2_symbolic_1, ss)

h_symbolic_2 = gain * (obstacle + ss[0]) ** 3
dhdx_symbolic_2 = (se.DenseMatrix([h_symbolic_2]).jacobian(se.DenseMatrix(ss))).T
d2hdx2_symbolic_2 = dhdx_symbolic_2.jacobian(se.DenseMatrix(ss))
h_func_2 = symbolic_cbf_wrapper_singleagent(h_symbolic_2, ss)
dhdx_func_2 = symbolic_cbf_wrapper_singleagent(dhdx_symbolic_2, ss)
d2hdx2_func_2 = symbolic_cbf_wrapper_singleagent(d2hdx2_symbolic_2, ss)


def h1(ego):
    return h_func_1(ego)


def dh1dx(ego):
    ret = dhdx_func_1(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2h1dx2(ego):
    ret = d2hdx2_func_1(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


def h2(ego):
    return h_func_2(ego)


def dh2dx(ego):
    ret = dhdx_func_2(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2h2dx2(ego):
    ret = d2hdx2_func_2(ego)

    return np.squeeze(np.array(ret).astype(np.float64))
