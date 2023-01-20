import numpy as np
from .cbf import Cbf
from .symbolic_cbfs.static_obstacle_ca import (
    h_ca1,
    dhdx_ca1,
    d2hdx2_ca1,
    h_ca2,
    dhdx_ca2,
    d2hdx2_ca2,
)
from .symbolic_cbfs.v1_safety import h_speed1, dhdx_speed1, d2hdx2_speed1
from .symbolic_cbfs.v2_safety import h_speed2, dhdx_speed2, d2hdx2_speed2


def linear_class_k(k):
    def alpha(h):
        return k * h

    return alpha


# Define linear class k weights
k_default = 1.0
k_collision = 1.0

# Define cbf lists
cbfs_individual = [
    Cbf(h_ca1, dhdx_ca1, d2hdx2_ca1, linear_class_k(k_default), h_ca1),
    Cbf(h_ca2, dhdx_ca2, d2hdx2_ca2, linear_class_k(k_default), h_ca2),
    Cbf(h_speed1, dhdx_speed1, d2hdx2_speed1, linear_class_k(k_default), h_speed1),
    Cbf(h_speed2, dhdx_speed2, d2hdx2_speed2, linear_class_k(k_default), h_speed2),
]
cbfs_pairwise = []

cbf0 = np.zeros((len(cbfs_individual) + len(cbfs_pairwise),))
