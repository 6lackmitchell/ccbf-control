import numpy as np
from .cbf import Cbf
from .symbolic_cbfs.static_obstacle_ca import (
    h_ca1,
    dhdx_ca1,
    d2hdx2_ca1,
    h_ca2,
    dhdx_ca2,
    d2hdx2_ca2,
    h_ca3,
    dhdx_ca3,
    d2hdx2_ca3,
    h_ca4,
    dhdx_ca4,
    d2hdx2_ca4,
    h_ca5,
    dhdx_ca5,
    d2hdx2_ca5,
)
from .symbolic_cbfs.v1_safety import h_speed1, dhdx_speed1, d2hdx2_speed1
from .symbolic_cbfs.v2_safety import h_speed2, dhdx_speed2, d2hdx2_speed2


def linear_class_k(k):
    def alpha(h):
        return k * h

    return alpha


# Define linear class k weights
k_default = 0.1

# Breeden comparison
k_default1 = 0.1
k_default2 = 0.2
k_default3 = 0.5
k_default4 = 0.8

# # Exponential CBF Comparison
# k_default1 = 1.0
# k_default2 = 1.0
# k_default3 = 1.0
# k_default4 = 1.0

# Collision Avoidance
k_collision = 1.0

# Define cbf lists
cbfs_individual1 = [
    Cbf(h_ca1, dhdx_ca1, d2hdx2_ca1, linear_class_k(k_default1), h_ca1),
    Cbf(h_ca2, dhdx_ca2, d2hdx2_ca2, linear_class_k(k_default1), h_ca2),
    Cbf(h_ca3, dhdx_ca3, d2hdx2_ca3, linear_class_k(k_default1), h_ca3),
    Cbf(h_ca4, dhdx_ca4, d2hdx2_ca4, linear_class_k(k_default1), h_ca4),
    Cbf(h_ca5, dhdx_ca5, d2hdx2_ca5, linear_class_k(k_default1), h_ca5),
    Cbf(h_speed1, dhdx_speed1, d2hdx2_speed1, linear_class_k(k_default1), h_speed1),
    Cbf(h_speed2, dhdx_speed2, d2hdx2_speed2, linear_class_k(k_default1), h_speed2),
]
cbfs_individual2 = [
    Cbf(h_ca1, dhdx_ca1, d2hdx2_ca1, linear_class_k(k_default2), h_ca1),
    Cbf(h_ca2, dhdx_ca2, d2hdx2_ca2, linear_class_k(k_default2), h_ca2),
    Cbf(h_ca3, dhdx_ca3, d2hdx2_ca3, linear_class_k(k_default2), h_ca3),
    Cbf(h_ca4, dhdx_ca4, d2hdx2_ca4, linear_class_k(k_default1), h_ca4),
    Cbf(h_ca5, dhdx_ca5, d2hdx2_ca5, linear_class_k(k_default1), h_ca5),
    Cbf(h_speed1, dhdx_speed1, d2hdx2_speed1, linear_class_k(k_default2), h_speed1),
    Cbf(h_speed2, dhdx_speed2, d2hdx2_speed2, linear_class_k(k_default2), h_speed2),
]
cbfs_individual3 = [
    Cbf(h_ca1, dhdx_ca1, d2hdx2_ca1, linear_class_k(k_default3), h_ca1),
    Cbf(h_ca2, dhdx_ca2, d2hdx2_ca2, linear_class_k(k_default3), h_ca2),
    Cbf(h_ca3, dhdx_ca3, d2hdx2_ca3, linear_class_k(k_default3), h_ca3),
    Cbf(h_ca4, dhdx_ca4, d2hdx2_ca4, linear_class_k(k_default1), h_ca4),
    Cbf(h_ca5, dhdx_ca5, d2hdx2_ca5, linear_class_k(k_default1), h_ca5),
    Cbf(h_speed1, dhdx_speed1, d2hdx2_speed1, linear_class_k(k_default3), h_speed1),
    Cbf(h_speed2, dhdx_speed2, d2hdx2_speed2, linear_class_k(k_default3), h_speed2),
]
cbfs_individual4 = [
    Cbf(h_ca1, dhdx_ca1, d2hdx2_ca1, linear_class_k(k_default4), h_ca1),
    Cbf(h_ca2, dhdx_ca2, d2hdx2_ca2, linear_class_k(k_default4), h_ca2),
    Cbf(h_ca3, dhdx_ca3, d2hdx2_ca3, linear_class_k(k_default4), h_ca3),
    Cbf(h_ca4, dhdx_ca4, d2hdx2_ca4, linear_class_k(k_default1), h_ca4),
    Cbf(h_ca5, dhdx_ca5, d2hdx2_ca5, linear_class_k(k_default1), h_ca5),
    Cbf(h_speed1, dhdx_speed1, d2hdx2_speed1, linear_class_k(k_default4), h_speed1),
    Cbf(h_speed2, dhdx_speed2, d2hdx2_speed2, linear_class_k(k_default4), h_speed2),
]
cbfs_individual = [
    Cbf(h_ca1, dhdx_ca1, d2hdx2_ca1, linear_class_k(k_default), h_ca1),
    Cbf(h_ca2, dhdx_ca2, d2hdx2_ca2, linear_class_k(k_default), h_ca2),
    Cbf(h_ca3, dhdx_ca3, d2hdx2_ca3, linear_class_k(k_default), h_ca3),
    Cbf(h_ca4, dhdx_ca4, d2hdx2_ca4, linear_class_k(k_default1), h_ca4),
    Cbf(h_ca5, dhdx_ca5, d2hdx2_ca5, linear_class_k(k_default1), h_ca5),
    Cbf(h_speed1, dhdx_speed1, d2hdx2_speed1, linear_class_k(k_default), h_speed1),
    Cbf(h_speed2, dhdx_speed2, d2hdx2_speed2, linear_class_k(k_default), h_speed2),
]
cbfs_pairwise = []

cbf0 = np.zeros((len(cbfs_individual) + len(cbfs_pairwise),))
