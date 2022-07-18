import numpy as np
import sympy as sp
from bicycle.dynamics.second_order_deterministic import f, R
from bicycle.dynamics.second_order_deterministic import sym_state as ss
from helpers.sigmoids import sigmoid_symbolic as sigmoid
from cbf_wrappers import symbolic_cbf_wrapper_multiagent, symbolic_cbf_wrapper_singleagent

# Define new symbols -- necessary for pairwise interactions case
sso = sp.symbols(['{}o'.format(n) for n in ss], real=True)
dx = ss[0] - sso[0]
dy = ss[1] - sso[1]

# Collision Avoidance CBF
h_ca_symbolic = (ss[0] - sso[0])**2 + (ss[1] - sso[1])**2 - (2 * R)**2
h_ca = symbolic_cbf_wrapper_multiagent(h_ca_symbolic, ss, sso)
dhdx_ca_symbolic = sp.Matrix([h_ca_symbolic], real=True).jacobian(np.concatenate([ss, sso]))
d2hdx2_ca_symbolic = dhdx_ca_symbolic.jacobian(np.concatenate([ss, sso]))
dhdx_collision_avoidance = symbolic_cbf_wrapper_multiagent(dhdx_ca_symbolic, ss, sso)
d2hdx2_collision_avoidance = symbolic_cbf_wrapper_multiagent(d2hdx2_ca_symbolic, ss, sso)

# Tau Formulation for PCA-CBF
vxe = f(np.zeros((len(ss),)), True)[0]
vye = f(np.zeros((len(ss),)), True)[1]
vxo = vxe
vyo = vye
for se, so in zip(ss, sso):
    vxo = vxo.replace(se, so)
    vyo = vyo.replace(se, so)
dvx = vxe - vxo
dvy = vye - vyo

tau_sym = sp.symbols('tau', real=True)

# tau* for computing tau
epsilon = 1e-3
tau_star_symbolic = -(dx * dvx + dy * dvy) / (dvx ** 2 + dvy ** 2 + epsilon)
dtaustardx_symbolic = sp.Matrix([tau_star_symbolic], real=True).jacobian(np.concatenate([ss, sso]))
d2taustardx2_symbolic = dtaustardx_symbolic.jacobian(np.concatenate([ss, sso]))
tau_star = symbolic_cbf_wrapper_multiagent(tau_star_symbolic, ss, sso)
dtaustardx = symbolic_cbf_wrapper_multiagent(dtaustardx_symbolic, ss, sso)
d2taustardx2 = symbolic_cbf_wrapper_multiagent(d2taustardx2_symbolic, ss, sso)

# tau for computing PCA-CBF
T = 3.0
kh = 1000.0
tau_star_sym = sp.symbols('tau_star')
tau_symbolic = tau_star_sym * sigmoid(tau_star_sym, kh, 0.0) - (tau_star_sym - T) * sigmoid(tau_star_sym, kh, T)
dtaudtaustar_symbolic = sp.Matrix([tau_symbolic], real=True).jacobian(np.array([tau_star_sym]))
d2taudtaustar2_symbolic = dtaudtaustar_symbolic.jacobian(np.array([tau_star_sym]))
tau = symbolic_cbf_wrapper_singleagent(tau_symbolic, [tau_star_sym])
dtaudtaustar = symbolic_cbf_wrapper_singleagent(dtaudtaustar_symbolic, [tau_star_sym])
d2taudtaustar2 = symbolic_cbf_wrapper_singleagent(d2taudtaustar2_symbolic, [tau_star_sym])

# Predictive Collision Avoidance CBF
h_predictive_ca_symbolic = (dx + tau_sym * dvx)**2 + (dy + tau_sym * dvy)**2 - (2 * R)**2
dhdx_predictive_ca_symbolic = sp.Matrix([h_predictive_ca_symbolic], real=True).jacobian(np.concatenate([ss, sso]))
dhdtau_predictive_ca_symbolic = sp.Matrix([h_predictive_ca_symbolic], real=True).jacobian(np.array([tau_sym]))
d2hdx2_predictive_ca_symbolic = dhdx_predictive_ca_symbolic.jacobian(np.concatenate([ss, sso]))
d2hdtau2_predictive_ca_symbolic = dhdtau_predictive_ca_symbolic.jacobian(np.array([tau_sym]))
d2hdtaudx_predictive_ca_symbolic = dhdtau_predictive_ca_symbolic.jacobian(np.concatenate([ss, sso]))
h_predictive_ca = symbolic_cbf_wrapper_multiagent(h_predictive_ca_symbolic, ss, sso)
dhdx_predictive_ca = symbolic_cbf_wrapper_multiagent(dhdx_predictive_ca_symbolic, ss, sso)
dhdtau_predictive_ca = symbolic_cbf_wrapper_multiagent(dhdtau_predictive_ca_symbolic, ss, sso)
d2hdx2_predictive_ca = symbolic_cbf_wrapper_multiagent(d2hdx2_predictive_ca_symbolic, ss, sso)
d2hdtaudx_predictive_ca = symbolic_cbf_wrapper_multiagent(d2hdx2_predictive_ca_symbolic, ss, sso)
d2hdtau2_predictive_ca = symbolic_cbf_wrapper_multiagent(d2hdtau2_predictive_ca_symbolic, ss, sso)


def h_pca(ego, other):
    return h_predictive_ca(ego, other).subs(tau_sym, tau([tau_star(ego, other)]))


def dhdx_pca(ego, other):
    ret = dhdx_predictive_ca(ego, other).subs(tau_sym, tau([tau_star(ego, other)])) + \
          dhdtau_predictive_ca(ego, other).subs(tau_sym, tau([tau_star(ego, other)])) * \
          dtaudtaustar([tau_star(ego, other)]) * dtaustardx(ego, other)

    return np.squeeze(np.array(ret).astype(np.float64))


# Necessary for stochastic systems
def d2hdx2_pca(ego, other):
    d2hdx2_eval = d2hdx2_predictive_ca(ego, other).subs(tau_sym, tau([tau_star(ego, other)]))
    dtaustardx_eval = dtaustardx(ego, other)
    dtaudtaustar_eval = float(dtaudtaustar([tau_star(ego, other)]))
    dhdtau_eval = dhdtau_predictive_ca(ego, other).subs(tau_sym, tau([tau_star(ego, other)]))[0]
    d2hdtau2_eval = float(d2hdtau2_predictive_ca(ego, other))
    d2taudtaustar2_eval = d2taudtaustar2([tau_star(ego, other)])
    d2taustardx2_eval = d2taustardx2(ego, other)
    outer = np.outer(dtaustardx_eval, dtaustardx_eval)

    ret = d2hdx2_eval + dtaudtaustar_eval * d2hdtau2_eval * dtaudtaustar_eval * outer + \
          dhdtau_eval * d2taudtaustar2_eval * outer + dhdtau_eval * d2taustardx2_eval * dtaudtaustar_eval

    return np.squeeze(np.array(ret).astype(np.float64))


# # Test
# ze = np.array([-10, 0, 0, 5, 0])
# zo = np.array([10, 0, np.pi, 5, 0])
# print(h_pca(ze, zo))
# print(dhdx_pca(ze, zo))
# print(d2hdx2_pca(ze, zo))


