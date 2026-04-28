import numpy as np
import pytest
from psiphy.toy_models import NoisyLine, GaussianSignal


class TestNoisyLine:
    def setup_method(self):
        self.model = NoisyLine(true_slope=-0.9, true_intercept=4.0, Nx=20)

    def test_observation_shape(self):
        obs = self.model.observation()
        assert obs.shape == (20,)

    def test_simulate_shape(self):
        sim = self.model.simulate(slope=-1.0, intercept=3.0)
        assert sim.shape == (20,)

    def test_sample_prior_shape(self):
        theta = self.model.sample_prior()
        assert theta.shape == (2,)

    def test_xs_shape(self):
        assert self.model.xs().shape == (20,)


class TestGaussianSignal:
    def setup_method(self):
        self.model = GaussianSignal(true_mean=0.0, true_sigma=1.0)

    def test_observation_shape(self):
        obs = self.model.observation(N=30)
        assert obs.shape == (30,)

    def test_simulate_shape(self):
        sim = self.model.simulate(mean=0.5, sigma=1.5, N=10)
        assert sim.shape == (10,)

    def test_sample_prior_shape(self):
        theta = self.model.sample_prior()
        assert theta.shape == (2,)
        mean, sigma = theta
        assert -5 <= mean <= 5
        assert 0.1 <= sigma <= 5
