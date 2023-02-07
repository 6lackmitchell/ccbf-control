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
beta_limit = np.pi / 4
gain = 1.0

# beta CBF Symbolic
h_beta_symbolic = gain * (beta_limit - ss[4]) * (ss[4] + beta_limit)
dhdx_beta_symbolic = (se.DenseMatrix([h_beta_symbolic]).jacobian(se.DenseMatrix(ss))).T
d2hdx2_beta_symbolic = dhdx_beta_symbolic.jacobian(se.DenseMatrix(ss))
h_beta_func = symbolic_cbf_wrapper_singleagent(h_beta_symbolic, ss)
dhdx_beta_func = symbolic_cbf_wrapper_singleagent(dhdx_beta_symbolic, ss)
d2hdx2_beta_func = symbolic_cbf_wrapper_singleagent(d2hdx2_beta_symbolic, ss)


def h_beta(ego):
    return h_beta_func(ego)


def dhdx_beta(ego):
    ret = dhdx_beta_func(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2hdx2_beta(ego):
    ret = d2hdx2_beta_func(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


if __name__ == "__main__":
    # This is a unit test
    ms = 97.5
    ze = np.array([-ms * np.cos(th), -ms * np.sin(th), th, 15.0, 0.0])

    print(h_road(ze))
    print(dhdx_road(ze))
    print(d2hdx2_road(ze))
    print("stop")
