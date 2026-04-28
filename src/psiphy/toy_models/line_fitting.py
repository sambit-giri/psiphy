import numpy as np


class NoisyLine:
    """
    Noisy linear model: y = slope * x + intercept + noise.

    Parameters
    ----------
    Nx : int
        Number of x points.
    x_range : tuple
        Range of x values.
    noise_sigma : float
        Fixed Gaussian noise std (used for log_prob).
    slope_range : tuple
        Prior range for slope.
    intercept_range : tuple
        Prior range for intercept.

    Notes
    -----
    Parameters: theta = [slope, intercept].
    Analytic log-likelihood available via :meth:`log_prob` (assumes fixed noise_sigma).
    """

    param_names = ["slope", "intercept"]

    def __init__(
        self,
        Nx=50,
        x_range=(0, 10),
        noise_sigma=0.5,
        slope_range=(-3, 3),
        intercept_range=(0, 10),
    ):
        self.x = np.linspace(*x_range, Nx)
        self.noise_sigma = noise_sigma
        self.slope_range = slope_range
        self.intercept_range = intercept_range

    def sample_prior(self):
        slope = np.random.uniform(*self.slope_range)
        intercept = np.random.uniform(*self.intercept_range)
        return np.array([slope, intercept])

    def simulate(self, theta):
        slope, intercept = theta
        noise = np.random.normal(0, self.noise_sigma, size=len(self.x))
        return slope * self.x + intercept + noise

    def log_prob(self, theta, x):
        slope, intercept = theta
        mu = slope * self.x + intercept
        return (
            -0.5 * len(self.x) * np.log(2 * np.pi * self.noise_sigma**2)
            - 0.5 * np.sum((np.asarray(x) - mu) ** 2) / self.noise_sigma**2
        )


# Legacy alias kept for backwards compatibility
noisy_line = NoisyLine
