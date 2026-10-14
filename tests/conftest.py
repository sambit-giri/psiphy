"""Shared fixtures for the psiphy test suite."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from psiphy.toy_models import GaussianSignal, NoisyLine


@pytest.fixture
def rng():
    """Seeded generator; also seeds the legacy global state used by toy models."""
    np.random.seed(0)
    return np.random.default_rng(0)


@pytest.fixture
def gaussian_model():
    return GaussianSignal(N=30, mean_range=(0.0, 3.0), sigma_range=(0.1, 2.5))


@pytest.fixture
def gaussian_obs(rng, gaussian_model):
    """(theta_true, x_obs) for GaussianSignal."""
    theta_true = np.array([1.5, 0.8])
    return theta_true, gaussian_model.simulate(theta_true)


@pytest.fixture
def line_model():
    return NoisyLine(Nx=20)


@pytest.fixture
def prior_bounds():
    """Box-prior bounds matching ``gaussian_model``."""
    return [(0.0, 3.0), (0.1, 2.5)]


@pytest.fixture
def gaussian_log_prob():
    """Correlated 2-D Gaussian log-density with known mean and covariance."""
    mean = np.array([0.5, -1.0])
    cov = np.array([[1.0, 0.3], [0.3, 0.5]])
    icov = np.linalg.inv(cov)

    def log_prob(theta):
        d = np.asarray(theta) - mean
        return -0.5 * d @ icov @ d

    log_prob.mean = mean
    log_prob.cov = cov
    return log_prob
