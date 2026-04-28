import numpy as np
import pytest
from psiphy.mcmc import (
    UniformPrior,
    eff_sample_size,
    importance_sampling,
    sequential_importance_sampling,
    SMC,
)
from psiphy.toy_models import GaussianSignal


@pytest.fixture
def gaussian_setup():
    np.random.seed(0)
    gs = GaussianSignal(N=30)
    theta_true = np.array([1.5, 0.8])
    x_obs = gs.simulate(theta_true)
    prior = UniformPrior([(0.0, 3.0), (0.1, 2.5)])
    log_post = lambda t: gs.log_prob(t, x_obs) + prior.log_prob(t)
    return gs, theta_true, x_obs, prior, log_post


class TestUniformPrior:
    def test_sample_shape(self):
        p = UniformPrior([(-1, 1), (0, 2)])
        s = p.sample(50)
        assert s.shape == (50, 2)

    def test_in_support(self):
        p = UniformPrior([(0, 1), (0, 1)])
        assert np.isfinite(p.log_prob([0.5, 0.5]))

    def test_out_of_support(self):
        p = UniformPrior([(0, 1), (0, 1)])
        assert p.log_prob([2.0, 0.5]) == -np.inf

    def test_dict_bounds(self):
        p = UniformPrior({'a': (-1, 1), 'b': (0, 5)})
        assert p.ndim == 2

    def test_sample_within_bounds(self):
        p = UniformPrior([(0, 1), (2, 4)])
        s = p.sample(200)
        assert np.all(s[:, 0] >= 0) and np.all(s[:, 0] <= 1)
        assert np.all(s[:, 1] >= 2) and np.all(s[:, 1] <= 4)


class TestEffSampleSize:
    def test_uniform_weights(self):
        w = np.ones(100)
        assert abs(eff_sample_size(w) - 100) < 1e-6

    def test_degenerate_weights(self):
        w = np.zeros(100)
        w[0] = 1.0
        assert abs(eff_sample_size(w) - 1.0) < 1e-6

    def test_between_bounds(self):
        w = np.random.dirichlet(np.ones(50))
        ess = eff_sample_size(w)
        assert 1 <= ess <= 50


class TestImportanceSampling:
    def test_output_shapes(self, gaussian_setup):
        gs, theta_true, x_obs, prior, log_post = gaussian_setup
        prop = prior.sample(500)
        samples, w = importance_sampling(log_post, prop)
        assert samples.shape == (500, 2)
        assert w.shape == (500,)

    def test_weights_sum_to_one(self, gaussian_setup):
        gs, theta_true, x_obs, prior, log_post = gaussian_setup
        prop = prior.sample(500)
        _, w = importance_sampling(log_post, prop)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_mean_estimate_reasonable(self, gaussian_setup):
        gs, theta_true, x_obs, prior, log_post = gaussian_setup
        np.random.seed(1)
        prop = prior.sample(5000)
        samples, w = importance_sampling(log_post, prop)
        # weighted mean of mean-parameter should be within 0.5 of truth
        est = np.average(samples[:, 0], weights=w)
        assert abs(est - theta_true[0]) < 0.5

    def test_ess_positive(self, gaussian_setup):
        gs, theta_true, x_obs, prior, log_post = gaussian_setup
        prop = prior.sample(500)
        _, w = importance_sampling(log_post, prop)
        assert eff_sample_size(w) > 0


class TestSMC:
    def test_output_shapes(self, gaussian_setup):
        gs, theta_true, x_obs, prior, log_post = gaussian_setup
        np.random.seed(2)
        samples, w = SMC(
            lambda t: gs.log_prob(t, x_obs),
            prior,
            n_particles=200,
            n_steps=3,
            verbose=False,
        )
        assert samples.shape == (200, 2)
        assert w.shape == (200,)

    def test_weights_sum_to_one(self, gaussian_setup):
        gs, theta_true, x_obs, prior, log_post = gaussian_setup
        np.random.seed(3)
        _, w = SMC(
            lambda t: gs.log_prob(t, x_obs),
            prior,
            n_particles=200,
            n_steps=3,
            verbose=False,
        )
        assert abs(w.sum() - 1.0) < 1e-9

    def test_mean_estimate_reasonable(self, gaussian_setup):
        gs, theta_true, x_obs, prior, log_post = gaussian_setup
        np.random.seed(4)
        samples, w = SMC(
            lambda t: gs.log_prob(t, x_obs),
            prior,
            n_particles=500,
            n_steps=8,
            verbose=False,
        )
        est = np.average(samples[:, 0], weights=w)
        assert abs(est - theta_true[0]) < 0.5
