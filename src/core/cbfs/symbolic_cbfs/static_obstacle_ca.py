import builtins
import numpy as np
import symengine as se
from importlib import import_module
from core.cbfs.cbf_wrappers import symbolic_cbf_wrapper_singleagent
from core.cbfs.cbf import Cbf

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
gain = 3.0
R = 0.4
R1 = 0.5
cx1 = 0.8
cy1 = 1.1
R2 = 0.5
cx2 = 1.5
cy2 = 2.25
R3 = 0.5
cx3 = 2.4
cy3 = 1.5
R4 = 0.5
cx4 = 2.0
cy4 = 0.35
R5 = 0.5
cx5 = 0.8
cy5 = -0.2

# Define new symbols -- necessary for pairwise interactions case
x_scale = 1.0
y_scale = 1.0
dx1 = (ss[0] - cx1) * x_scale
dy1 = (ss[1] - cy1) * y_scale
dx2 = (ss[0] - cx2) * x_scale
dy2 = (ss[1] - cy2) * y_scale
dx3 = (ss[0] - cx3) * x_scale
dy3 = (ss[1] - cy3) * y_scale
dx4 = (ss[0] - cx4) * x_scale
dy4 = (ss[1] - cy4) * y_scale
dx5 = (ss[0] - cx5) * x_scale
dy5 = (ss[1] - cy5) * y_scale

# Collision Avoidance CBF
h_symbolic1 = gain * (dx1**2 + dy1**2 - (R) ** 2)
dhdt_symbolic1 = (se.DenseMatrix([h_symbolic1]).jacobian(se.DenseMatrix(tt))).T
dhdx_symbolic1 = (se.DenseMatrix([h_symbolic1]).jacobian(se.DenseMatrix(ss))).T
d2hdtdx_symbolic1 = dhdt_symbolic1.jacobian(se.DenseMatrix(ss))
d2hdx2_symbolic1 = dhdx_symbolic1.jacobian(se.DenseMatrix(ss))
h_func1 = symbolic_cbf_wrapper_singleagent(h_symbolic1, tt, ss)
dhdt_func1 = symbolic_cbf_wrapper_singleagent(dhdt_symbolic1, tt, ss)
dhdx_func1 = symbolic_cbf_wrapper_singleagent(dhdx_symbolic1, tt, ss)
d2hdtdx_func1 = symbolic_cbf_wrapper_singleagent(d2hdtdx_symbolic1, tt, ss)
d2hdx2_func1 = symbolic_cbf_wrapper_singleagent(d2hdx2_symbolic1, tt, ss)

# Collision Avoidance CBF
h_symbolic2 = gain * (dx2**2 + dy2**2 - (R) ** 2)
dhdt_symbolic2 = (se.DenseMatrix([h_symbolic2]).jacobian(se.DenseMatrix(tt))).T
dhdx_symbolic2 = (se.DenseMatrix([h_symbolic2]).jacobian(se.DenseMatrix(ss))).T
d2hdtdx_symbolic2 = dhdt_symbolic2.jacobian(se.DenseMatrix(ss))
d2hdx2_symbolic2 = dhdx_symbolic2.jacobian(se.DenseMatrix(ss))
h_func2 = symbolic_cbf_wrapper_singleagent(h_symbolic2, tt, ss)
dhdt_func2 = symbolic_cbf_wrapper_singleagent(dhdt_symbolic2, tt, ss)
dhdx_func2 = symbolic_cbf_wrapper_singleagent(dhdx_symbolic2, tt, ss)
d2hdtdx_func2 = symbolic_cbf_wrapper_singleagent(d2hdtdx_symbolic2, tt, ss)
d2hdx2_func2 = symbolic_cbf_wrapper_singleagent(d2hdx2_symbolic2, tt, ss)

# Collision Avoidance CBF
h_symbolic3 = gain * (dx3**2 + dy3**2 - (R) ** 2)
dhdt_symbolic3 = (se.DenseMatrix([h_symbolic3]).jacobian(se.DenseMatrix(tt))).T
dhdx_symbolic3 = (se.DenseMatrix([h_symbolic3]).jacobian(se.DenseMatrix(ss))).T
d2hdtdx_symbolic3 = dhdt_symbolic3.jacobian(se.DenseMatrix(ss))
d2hdx2_symbolic3 = dhdx_symbolic3.jacobian(se.DenseMatrix(ss))
h_func3 = symbolic_cbf_wrapper_singleagent(h_symbolic3, tt, ss)
dhdt_func3 = symbolic_cbf_wrapper_singleagent(dhdt_symbolic3, tt, ss)
dhdx_func3 = symbolic_cbf_wrapper_singleagent(dhdx_symbolic3, tt, ss)
d2hdtdx_func3 = symbolic_cbf_wrapper_singleagent(d2hdtdx_symbolic3, tt, ss)
d2hdx2_func3 = symbolic_cbf_wrapper_singleagent(d2hdx2_symbolic3, tt, ss)

# Collision Avoidance CBF
h_symbolic4 = gain * (dx4**2 + dy4**2 - (R) ** 2)
dhdt_symbolic4 = (se.DenseMatrix([h_symbolic4]).jacobian(se.DenseMatrix(tt))).T
dhdx_symbolic4 = (se.DenseMatrix([h_symbolic4]).jacobian(se.DenseMatrix(ss))).T
d2hdtdx_symbolic4 = dhdt_symbolic4.jacobian(se.DenseMatrix(ss))
d2hdx2_symbolic4 = dhdx_symbolic4.jacobian(se.DenseMatrix(ss))
h_func4 = symbolic_cbf_wrapper_singleagent(h_symbolic4, tt, ss)
dhdt_func4 = symbolic_cbf_wrapper_singleagent(dhdt_symbolic4, tt, ss)
dhdx_func4 = symbolic_cbf_wrapper_singleagent(dhdx_symbolic4, tt, ss)
d2hdtdx_func4 = symbolic_cbf_wrapper_singleagent(d2hdtdx_symbolic4, tt, ss)
d2hdx2_func4 = symbolic_cbf_wrapper_singleagent(d2hdx2_symbolic4, tt, ss)

# Collision Avoidance CBF
h_symbolic5 = gain * (dx5**2 + dy5**2 - (R) ** 2)
dhdt_symbolic5 = (se.DenseMatrix([h_symbolic5]).jacobian(se.DenseMatrix(tt))).T
dhdx_symbolic5 = (se.DenseMatrix([h_symbolic5]).jacobian(se.DenseMatrix(ss))).T
d2hdtdx_symbolic5 = dhdt_symbolic5.jacobian(se.DenseMatrix(ss))
d2hdx2_symbolic5 = dhdx_symbolic5.jacobian(se.DenseMatrix(ss))
h_func5 = symbolic_cbf_wrapper_singleagent(h_symbolic5, tt, ss)
dhdt_func5 = symbolic_cbf_wrapper_singleagent(dhdt_symbolic5, tt, ss)
dhdx_func5 = symbolic_cbf_wrapper_singleagent(dhdx_symbolic5, tt, ss)
d2hdtdx_func5 = symbolic_cbf_wrapper_singleagent(d2hdtdx_symbolic5, tt, ss)
d2hdx2_func5 = symbolic_cbf_wrapper_singleagent(d2hdx2_symbolic5, tt, ss)


# CBF Callables
def h1(t, x):
    return h_func1(t, x)


def dhdt1(t, x):
    ret = dhdt_func1(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def dhdx1(t, x):
    ret = dhdx_func1(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2hdtdx1(t, x):
    ret = d2hdtdx_func1(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2hdx2_1(t, x):
    ret = d2hdx2_func1(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def h2(t, x):
    return h_func2(t, x)


def dhdt2(t, x):
    ret = dhdt_func2(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def dhdx2(t, x):
    ret = dhdx_func2(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2hdtdx2(t, x):
    ret = d2hdtdx_func2(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2hdx2_2(t, x):
    ret = d2hdx2_func2(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def h3(t, x):
    return h_func3(t, x)


def dhdt3(t, x):
    ret = dhdt_func3(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def dhdx3(t, x):
    ret = dhdx_func3(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2hdtdx3(t, x):
    ret = d2hdtdx_func3(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2hdx2_3(t, x):
    ret = d2hdx2_func3(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def h4(t, x):
    return h_func4(t, x)


def dhdt4(t, x):
    ret = dhdt_func4(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def dhdx4(t, x):
    ret = dhdx_func4(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2hdtdx4(t, x):
    ret = d2hdtdx_func4(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2hdx2_4(t, x):
    ret = d2hdx2_func4(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def h5(t, x):
    return h_func5(t, x)


def dhdt5(t, x):
    ret = dhdt_func5(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def dhdx5(t, x):
    ret = dhdx_func5(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2hdtdx5(t, x):
    ret = d2hdtdx_func5(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def d2hdx2_5(t, x):
    ret = d2hdx2_func5(t, x)

    return np.squeeze(np.array(ret).astype(np.float64))


def linear_class_k(k):
    def alpha(h):
        return k * h

    return alpha


cbf1 = Cbf(h1, None, dhdx1, None, d2hdx2_1, linear_class_k(1.0))
cbf2 = Cbf(h2, None, dhdx2, None, d2hdx2_2, linear_class_k(1.0))
cbf3 = Cbf(h3, None, dhdx3, None, d2hdx2_3, linear_class_k(1.0))
cbf4 = Cbf(h4, None, dhdx4, None, d2hdx2_4, linear_class_k(1.0))
cbf5 = Cbf(h5, None, dhdx5, None, d2hdx2_5, linear_class_k(1.0))
cbf1.set_symbolics(h_symbolic1, None, dhdx_symbolic1, None, d2hdx2_symbolic1)
cbf2.set_symbolics(h_symbolic2, None, dhdx_symbolic2, None, d2hdx2_symbolic2)
cbf3.set_symbolics(h_symbolic3, None, dhdx_symbolic3, None, d2hdx2_symbolic3)
cbf4.set_symbolics(h_symbolic4, None, dhdx_symbolic4, None, d2hdx2_symbolic4)
cbf5.set_symbolics(h_symbolic5, None, dhdx_symbolic5, None, d2hdx2_symbolic5)


if __name__ == "__main__":
    # This is a unit test
    ze = np.array([0.0, 0.0, 0.0, 5.0, 0.0])
    zo = np.array([10.0, 0.0, -np.pi, 5.0, 0.0])
