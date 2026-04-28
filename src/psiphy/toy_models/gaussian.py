import numpy as np


class GaussianSignal:
    """
    Simple 1-D Gaussian signal model with known analytic likelihood.

    Parameters
    ----------
    N : int
        Number of data points per simulation.
    mean_range : tuple
        Prior range for the mean parameter.
    sigma_range : tuple
        Prior range for the sigma parameter (must be > 0).

    Notes
    -----
    Parameters: theta = [mean, sigma].
    Analytic log-likelihood available via :meth:`log_prob`.
    """

    param_names = ["mean", "sigma"]

    def __init__(self, N=20, mean_range=(-5, 5), sigma_range=(0.1, 5)):
        self.N = N
        self.mean_range = mean_range
        self.sigma_range = sigma_range

    def sample_prior(self):
        mean = np.random.uniform(*self.mean_range)
        sigma = np.random.uniform(*self.sigma_range)
        return np.array([mean, sigma])

    def simulate(self, theta):
        mean, sigma = theta
        return np.random.normal(loc=mean, scale=sigma, size=self.N)

    def log_prob(self, theta, x):
        mean, sigma = theta
        if sigma <= 0:
            return -np.inf
        return (
            -0.5 * len(x) * np.log(2 * np.pi * sigma**2)
            - 0.5 * np.sum((np.asarray(x) - mean) ** 2) / sigma**2
        )

    def analytic_posterior(self, x, mean_prior_sigma=10.0):
        """Return (posterior_mean, posterior_sigma) for a flat-ish Gaussian prior on mean.

        Assumes sigma is known to equal the sample std (plug-in).
        """
        x = np.asarray(x)
        sigma_hat = np.std(x, ddof=1) if len(x) > 1 else 1.0
        n = len(x)
        prior_var = mean_prior_sigma**2
        lik_var = sigma_hat**2 / n
        post_var = 1.0 / (1.0 / prior_var + 1.0 / lik_var)
        post_mean = post_var * (np.mean(x) / lik_var)
        return post_mean, np.sqrt(post_var)


# Legacy alias kept for backwards compatibility
gaussian_signal = GaussianSignal
