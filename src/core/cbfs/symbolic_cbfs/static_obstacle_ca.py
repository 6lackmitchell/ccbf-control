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
    # globals().update({'sigma': getattr(module, 'sigma')})
    globals().update({"ss": getattr(module, "sym_state")})
except ModuleNotFoundError as e:
    print("No module named '{}' -- exiting.".format(mod))
    raise e

# Defining Physical Params
gain = 2.0
R1 = 0.5
cx1 = 1.0
cy1 = 1.0
R2 = 0.5
cx2 = 1.5
cy2 = 2.25
R3 = 0.5
cx3 = 2.4
cy3 = 1.5

# Define new symbols -- necessary for pairwise interactions case
x_scale = 1.0
y_scale = 1.0
dx1 = (ss[0] - cx1) * x_scale
dy1 = (ss[1] - cy1) * y_scale
dx2 = (ss[0] - cx2) * x_scale
dy2 = (ss[1] - cy2) * y_scale
dx3 = (ss[0] - cx3) * x_scale
dy3 = (ss[1] - cy3) * y_scale

# Collision Avoidance CBF
h_nominal_ca_symbolic1 = gain * (dx1**2 + dy1**2 - (R1) ** 2)
dhdx_nominal_ca_symbolic1 = (
    se.DenseMatrix([h_nominal_ca_symbolic1]).jacobian(se.DenseMatrix(ss))
).T
d2hdx2_nominal_ca_symbolic1 = dhdx_nominal_ca_symbolic1.jacobian(se.DenseMatrix(ss))
h_nominal_ca1 = symbolic_cbf_wrapper_singleagent(h_nominal_ca_symbolic1, ss)
dhdx_nominal_ca1 = symbolic_cbf_wrapper_singleagent(dhdx_nominal_ca_symbolic1, ss)
d2hdx2_nominal_ca1 = symbolic_cbf_wrapper_singleagent(d2hdx2_nominal_ca_symbolic1, ss)

# Collision Avoidance CBF
h_nominal_ca_symbolic2 = gain * (dx2**2 + dy2**2 - (R2) ** 2)
dhdx_nominal_ca_symbolic2 = (
    se.DenseMatrix([h_nominal_ca_symbolic2]).jacobian(se.DenseMatrix(ss))
).T
d2hdx2_nominal_ca_symbolic2 = dhdx_nominal_ca_symbolic2.jacobian(se.DenseMatrix(ss))
h_nominal_ca2 = symbolic_cbf_wrapper_singleagent(h_nominal_ca_symbolic2, ss)
dhdx_nominal_ca2 = symbolic_cbf_wrapper_singleagent(dhdx_nominal_ca_symbolic2, ss)
d2hdx2_nominal_ca2 = symbolic_cbf_wrapper_singleagent(d2hdx2_nominal_ca_symbolic2, ss)

# Collision Avoidance CBF
h_nominal_ca_symbolic3 = gain * (dx3**2 + dy3**2 - (R3) ** 2)
dhdx_nominal_ca_symbolic3 = (
    se.DenseMatrix([h_nominal_ca_symbolic3]).jacobian(se.DenseMatrix(ss))
).T
d2hdx2_nominal_ca_symbolic3 = dhdx_nominal_ca_symbolic3.jacobian(se.DenseMatrix(ss))
h_nominal_ca3 = symbolic_cbf_wrapper_singleagent(h_nominal_ca_symbolic3, ss)
dhdx_nominal_ca3 = symbolic_cbf_wrapper_singleagent(dhdx_nominal_ca_symbolic3, ss)
d2hdx2_nominal_ca3 = symbolic_cbf_wrapper_singleagent(d2hdx2_nominal_ca_symbolic3, ss)


# CBF Callables
def h_ca1(ego):
    return h_nominal_ca1(ego)


def dhdx_ca1(ego):
    ret = dhdx_nominal_ca1(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


# Necessary for stochastic systems
def d2hdx2_ca1(ego):
    ret = d2hdx2_nominal_ca1(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


def h_ca2(ego):
    return h_nominal_ca2(ego)


def dhdx_ca2(ego):
    ret = dhdx_nominal_ca2(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


# Necessary for stochastic systems
def d2hdx2_ca2(ego):
    ret = d2hdx2_nominal_ca2(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


def h_ca3(ego):
    return h_nominal_ca3(ego)


def dhdx_ca3(ego):
    ret = dhdx_nominal_ca3(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


# Necessary for stochastic systems
def d2hdx2_ca3(ego):
    ret = d2hdx2_nominal_ca3(ego)

    return np.squeeze(np.array(ret).astype(np.float64))


if __name__ == "__main__":
    # This is a unit test
    ze = np.array([0.0, 0.0, 0.0, 5.0, 0.0])
    zo = np.array([10.0, 0.0, -np.pi, 5.0, 0.0])
