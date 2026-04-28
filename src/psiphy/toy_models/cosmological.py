import numpy as np


class PowerLawSpectrum:
    """
    Simple power-law angular power spectrum toy model.

    Simulates noisy band-power measurements of a CMB-like spectrum::

        C_ell = A_s * (ell / ell_pivot) ** (n_s - 1)

    with independent Gaussian noise on each band power.

    Parameters
    ----------
    ells : array-like or None
        Multipole values at which the spectrum is evaluated.
        Defaults to 20 log-spaced values from 10 to 2000.
    noise_sigma : float
        Noise std per band power (in same units as C_ell).
    ell_pivot : float
        Pivot multipole for the tilt (default 100).
    A_s_range : tuple
        Prior range for the amplitude A_s.
    n_s_range : tuple
        Prior range for the spectral index n_s.

    Notes
    -----
    Parameters: theta = [A_s, n_s].
    Analytic Gaussian log-likelihood available via :meth:`log_prob`.
    """

    param_names = ["A_s", "n_s"]

    def __init__(
        self,
        ells=None,
        noise_sigma=0.05,
        ell_pivot=100.0,
        A_s_range=(0.5, 2.0),
        n_s_range=(0.85, 1.15),
    ):
        if ells is None:
            ells = np.geomspace(10, 2000, 20)
        self.ells = np.asarray(ells, dtype=float)
        self.noise_sigma = noise_sigma
        self.ell_pivot = ell_pivot
        self.A_s_range = A_s_range
        self.n_s_range = n_s_range

    def _spectrum(self, theta):
        A_s, n_s = theta
        return A_s * (self.ells / self.ell_pivot) ** (n_s - 1)

    def sample_prior(self):
        A_s = np.random.uniform(*self.A_s_range)
        n_s = np.random.uniform(*self.n_s_range)
        return np.array([A_s, n_s])

    def simulate(self, theta):
        C = self._spectrum(theta)
        return C + self.noise_sigma * np.random.randn(len(self.ells))

    def log_prob(self, theta, x):
        C = self._spectrum(theta)
        n = len(self.ells)
        return (
            -0.5 * n * np.log(2 * np.pi * self.noise_sigma**2)
            - 0.5 * np.sum((np.asarray(x) - C) ** 2) / self.noise_sigma**2
        )
