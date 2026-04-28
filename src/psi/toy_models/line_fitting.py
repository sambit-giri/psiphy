import numpy as np


class NoisyLine:
    """Noisy linear model with slope and intercept as parameters."""

    def __init__(self, true_slope=-0.9594, true_intercept=4.294, Nx=50, yerr_param=(0.1, 0.5)):
        self.true_slope = true_slope
        self.true_intercept = true_intercept
        self.x = np.sort(10 * np.random.rand(Nx))
        self._yerr_param = yerr_param

    def _yerr(self, n):
        return self._yerr_param[0] + self._yerr_param[1] * np.random.rand(n)

    def xs(self):
        return self.x

    def observation(self):
        ys = self.true_slope * self.x + self.true_intercept + self._yerr(self.x.size)
        return ys

    def simulate(self, slope, intercept):
        ys = slope * self.x + intercept + self._yerr(self.x.size)
        return ys

    def sample_prior(self, slope_range=(-3, 3), intercept_range=(0, 10)):
        slope = np.random.uniform(*slope_range)
        intercept = np.random.uniform(*intercept_range)
        return np.array([slope, intercept])


# Legacy alias
noisy_line = NoisyLine
