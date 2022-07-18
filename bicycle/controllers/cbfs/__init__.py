import numpy as np
from .cbf import Cbf
from .interagent_cbfs import h_ca as H0, h_pca, dhdx_pca, d2hdx2_pca


def linear_class_k(k):

    def alpha(h):
        return k*h

    return alpha

# Define linear class k weights
k = 1.0

# Define cbf lists
cbfs_individual = []
cbfs_pairwise = [Cbf(h_pca, dhdx_pca, d2hdx2_pca, linear_class_k(k)),  # FF-CBF
                 ]

cbf0 = np.zeros((len(cbfs_individual) + len(cbfs_pairwise), ))
