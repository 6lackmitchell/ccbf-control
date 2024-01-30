"""
#! docstring
"""
from systems.nonlinear_1d.black2023consolidated.certificate_functions.barrier_functions import (
    barrier_1 as b1,
    barrier_2 as b2,
    barrier_3 as b,
)

cbfs = [
    b.cbf,
]

cbf_grads = [
    b.cbf_grad,
]
