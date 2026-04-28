import numpy as np
from scipy.integrate import solve_ivp


class LotkaVolterra:
    """
    Lotka-Volterra predator-prey model.

    Stochastic observations of the ODE system::

        dx/dt =  alpha * x - beta  * x * y   (prey)
        dy/dt =  delta * x * y - gamma * y   (predator)

    with independent Gaussian noise added to each species at each observation
    time, making the likelihood tractable.

    Parameters
    ----------
    t_obs : array-like or None
        Observation times.  Defaults to 20 equally-spaced points on [0, 15].
    x0 : float
        Initial prey population.
    y0 : float
        Initial predator population.
    noise_sigma : float
        Gaussian noise std added to each observation.
    alpha_range : tuple
        Prior range for prey growth rate alpha.
    beta_range : tuple
        Prior range for predation rate beta.
    delta_range : tuple
        Prior range for predator growth rate delta.
    gamma_range : tuple
        Prior range for predator death rate gamma.

    Notes
    -----
    Parameters: ``theta = [alpha, beta, delta, gamma]``.

    Typical values: alpha ~ 1.0, beta ~ 0.1, delta ~ 0.075, gamma ~ 1.5.

    :meth:`simulate` returns a 1-D array of length ``2 * len(t_obs)``:
    prey observations followed by predator observations.
    :meth:`log_prob` returns the analytic Gaussian log-likelihood.
    """

    param_names = ["alpha", "beta", "delta", "gamma"]

    def __init__(
        self,
        t_obs=None,
        x0=10.0,
        y0=5.0,
        noise_sigma=0.5,
        alpha_range=(0.5, 2.0),
        beta_range=(0.02, 0.2),
        delta_range=(0.02, 0.2),
        gamma_range=(0.5, 2.0),
    ):
        self.t_obs = np.linspace(0, 15, 20) if t_obs is None else np.asarray(t_obs)
        self.x0 = x0
        self.y0 = y0
        self.noise_sigma = noise_sigma
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.delta_range = delta_range
        self.gamma_range = gamma_range

    def _ode(self, t, state, alpha, beta, delta, gamma):
        x, y = state
        return [
            alpha * x - beta * x * y,
            delta * x * y - gamma * y,
        ]

    def _integrate(self, theta):
        alpha, beta, delta, gamma = theta
        sol = solve_ivp(
            self._ode,
            t_span=(self.t_obs[0], self.t_obs[-1]),
            y0=[self.x0, self.y0],
            args=(alpha, beta, delta, gamma),
            t_eval=self.t_obs,
            method="RK45",
            dense_output=False,
            rtol=1e-6,
            atol=1e-8,
        )
        if not sol.success:
            return None
        return sol.y  # shape (2, n_obs)

    def sample_prior(self):
        return np.array([
            np.random.uniform(*self.alpha_range),
            np.random.uniform(*self.beta_range),
            np.random.uniform(*self.delta_range),
            np.random.uniform(*self.gamma_range),
        ])

    def simulate(self, theta):
        """
        Integrate the ODE and add Gaussian noise.

        Returns
        -------
        np.ndarray, shape (2 * n_obs,)
            Prey observations followed by predator observations.
            Returns NaN vector if the ODE solver fails.
        """
        traj = self._integrate(theta)
        if traj is None:
            return np.full(2 * len(self.t_obs), np.nan)
        noise = self.noise_sigma * np.random.randn(2, len(self.t_obs))
        noisy = traj + noise
        return noisy.ravel()

    def noiseless(self, theta):
        """Return noiseless ODE solution, shape (2, n_obs): [prey, predator]."""
        traj = self._integrate(theta)
        if traj is None:
            return None
        return traj

    def log_prob(self, theta, x):
        """
        Gaussian log-likelihood of observations x given parameters theta.

        Parameters
        ----------
        theta : array-like, shape (4,)
        x : array-like, shape (2 * n_obs,)

        Returns
        -------
        float or -inf if ODE diverges.
        """
        traj = self._integrate(theta)
        if traj is None:
            return -np.inf
        mu = traj.ravel()
        n = len(mu)
        return (
            -0.5 * n * np.log(2 * np.pi * self.noise_sigma**2)
            - 0.5 * np.sum((np.asarray(x) - mu) ** 2) / self.noise_sigma**2
        )
