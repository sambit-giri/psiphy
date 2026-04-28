import numpy as np


class GaussianSignal:
    """Simple Gaussian signal model with known analytic posterior."""

    def __init__(self, true_mean, true_sigma):
        self.true_mean = true_mean
        self.true_sigma = true_sigma

    def observation(self, N=20):
        return np.random.normal(loc=self.true_mean, scale=self.true_sigma, size=N)

    def simulate(self, mean, sigma, N=1):
        return np.random.normal(loc=mean, scale=sigma, size=N)

    def sample_prior(self, mean_range=(-5, 5), sigma_range=(0.1, 5)):
        mean = np.random.uniform(*mean_range)
        sigma = np.random.uniform(*sigma_range)
        return np.array([mean, sigma])


# Legacy alias
gaussian_signal = GaussianSignal
