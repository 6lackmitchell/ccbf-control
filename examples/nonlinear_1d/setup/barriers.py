"""
#! docstring
"""
from typing import List
from cbfkit.utils.user_types import CertificateTuple
from ccbf.systems.nonlinear_1d.models.black2024consolidated.certificate_functions.barrier_functions import cbf1_package, cbf2_package, cbf3_package
from cbfkit.controllers.utils.barrier_conditions import zeroing_barriers
from cbfkit.controllers.utils.certificate_packager import concatenate_certificates


def barriers(limit: float, 
             alpha: float, 
             idxs: List[int] = [0]) -> List[CertificateTuple]:
    """
    #! Docstring
    """
    packages = [
            package(
                certificate_conditions=zeroing_barriers.linear_class_k(alpha=alpha), 
                limit=limit,
            ) for pp, package in enumerate([cbf1_package, cbf2_package, cbf3_package]) if pp in idxs
    ]
    return concatenate_certificates(*packages)
