import numpy as np


class MA2Model:
    """
    Moving Average(2) process — standard LFI benchmark.

    Generative model: x_t = w_t + theta_1 * w_{t-1} + theta_2 * w_{t-2},
    where w_t ~ N(0, 1) i.i.d.

    Parameters
    ----------
    n_obs : int
        Length of the time series.

    Notes
    -----
    Parameters: theta = [theta_1, theta_2].
    The prior is uniform on the MA(2) invertibility region:
    |theta_2| < 1, theta_2 + theta_1 > -1, theta_2 - theta_1 > -1.

    The likelihood is intractable; log_prob returns None.
    Summary statistics returned by simulate: [mean, variance, lag-1 autocov, lag-2 autocov].
    """

    param_names = ["theta_1", "theta_2"]

    def __init__(self, n_obs=100):
        self.n_obs = n_obs

    def _in_prior(self, theta_1, theta_2):
        return (
            abs(theta_2) < 1
            and theta_2 + theta_1 > -1
            and theta_2 - theta_1 > -1
        )

    def sample_prior(self):
        while True:
            t1 = np.random.uniform(-2, 2)
            t2 = np.random.uniform(-1, 1)
            if self._in_prior(t1, t2):
                return np.array([t1, t2])

    def simulate_raw(self, theta):
        """Return the raw time series (length n_obs)."""
        t1, t2 = theta
        w = np.random.randn(self.n_obs + 2)
        return w[2:] + t1 * w[1:-1] + t2 * w[:-2]

    def simulate(self, theta):
        """Return 4-element summary-statistic vector: [mean, var, lag-1 autocov, lag-2 autocov]."""
        x = self.simulate_raw(theta)
        return np.array([
            np.mean(x),
            np.var(x),
            np.mean(x[1:] * x[:-1]),
            np.mean(x[2:] * x[:-2]),
        ])

    def log_prob(self, theta, x):
        return None  # likelihood is intractable
