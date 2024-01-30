import fossil
from fossil import plotting, certificate
import jax.numpy as jnp
from examples.nonlinear_1d.setup import W_MAX, W_MIN, ACTUATION_LIMITS


def model(dynamics):
    class Model(fossil.control.DynamicalModel):
        n_vars = len(dynamics(jnp.zeros((1000,)))[0][:-1])

        def __init__(self):
            super().__init__()
            self.dynamics = dynamics

        def f_torch(self, v):
            return list(self.dynamics(v[:])[0])

        def f_smt(self, v):
            return list(self.dynamics(v)[0])

    return Model


def test_lnn(system):
    XD = fossil.domains.Rectangle(
        [-2, W_MIN, -100, -ACTUATION_LIMITS[0]],
        [2, W_MAX, 100, ACTUATION_LIMITS[0]],
    )
    XI = fossil.domains.Sphere([0.0, 0.0, 0.0, 0.0], 0.1)

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
    }
    data = {
        certificate.XD: XD._generate_data(1000),
        certificate.XI: XI._generate_data(400),
    }

    # define NN parameters
    activations = [fossil.ActivationType.SIGMOID, fossil.ActivationType.SIGMOID]
    n_hidden_neurons = [10] * len(activations)

    opts = fossil.CegisConfig(
        N_VARS=system.n_vars,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        CERTIFICATE=fossil.CertificateType.LYAPUNOV,
        TIME_DOMAIN=fossil.TimeDomain.CONTINUOUS,
        VERIFIER=fossil.VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        LLO=True,
        CEGIS_MAX_ITERS=25,
        VERBOSE=2,
    )
    fossil.synthesise(opts)

    # result = fossil.synthesise(
    #     opts,
    # )
    # D = opts.DOMAINS.pop(fossil.XD)
    # plotting.benchmark(
    #     result.f, result.cert, domains=opts.DOMAINS, xrange=[-3, 2.5], yrange=[-2, 1]
    # )
