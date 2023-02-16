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
    globals().update({"tt": getattr(module, "sym_time")})
except ModuleNotFoundError as e:
    print("No module named '{}' -- exiting.".format(mod))
    raise e

# Defining Physical Params
T = 10
goal_x = 2.0
goal_y = 2.0
R = 0.05
Ri = 4
gain = 10.0

# dx, dy
dx = ss[0] - goal_x
dy = ss[1] - goal_y

# Symbolic Reach Constraint
h_symbolic = gain * (R**2 + Ri**2 * (1 - tt / T) - dx**2 - dy**2)
dhdt_symbolic = (se.DenseMatrix([h_symbolic]).jacobian(se.DenseMatrix([tt]))).T
dhdx_symbolic = (se.DenseMatrix([h_symbolic]).jacobian(se.DenseMatrix(ss))).T
d2hdtdx_symbolic = dhdt_symbolic.jacobian(se.DenseMatrix(ss))
d2hdx2_symbolic = dhdx_symbolic.jacobian(se.DenseMatrix(ss))
h_func = symbolic_cbf_wrapper_singleagent(h_symbolic, tt, ss)
dhdt_func = symbolic_cbf_wrapper_singleagent(dhdt_symbolic, tt, ss)
dhdx_func = symbolic_cbf_wrapper_singleagent(dhdx_symbolic, tt, ss)
d2hdtdx_func = symbolic_cbf_wrapper_singleagent(d2hdtdx_symbolic, tt, ss)
d2hdx2_func = symbolic_cbf_wrapper_singleagent(d2hdx2_symbolic, tt, ss)


def h(t, x):
    return h_func(t, x)


def dhdt(t, x):
    ret = dhdt_func(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def dhdx(t, x):
    ret = dhdx_func(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2hdx2(t, x):
    ret = d2hdx2_func(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2hdtdx(t, x):
    ret = d2hdtdx_func(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


if __name__ == "__main__":
    # This is a unit test
    t = 0.25
    x = np.array([1.0, 2.0])

    print(h(t, x))
    print(dhdx(t, x))
    print(d2hdx2(t, x))
    print("stop")
