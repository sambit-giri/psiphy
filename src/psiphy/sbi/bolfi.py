"""
BOLFI — Bayesian Optimisation for Likelihood-Free Inference.

A Gaussian-process surrogate of the discrepancy ``d(theta)`` between simulated
and observed data is learned with Bayesian optimisation (Gutmann & Corander
2016). The synthetic log-likelihood ``log K(d_hat(theta))`` is then sampled with
any of the psiphy samplers (emcee, nested sampling, importance sampling, SMC).

Examples
--------
>>> bolfi = BOLFI(distance_fn, {'mu': (-5, 5), 'sigma': (0.1, 5)})
>>> bolfi.learn_likelihood(n_calls=100)
>>> samples = bolfi.sample_posterior(5000, method='MCMC')
"""

import numpy as np

from ._surrogate import SurrogateInference

_LOG_KERNELS = {
    "exp": lambda d: -0.5 * d,              # L ∝ exp(-d/2): d plays the role of a chi^2
    "gaussian": lambda d: -0.5 * d**2,      # L ∝ exp(-d^2/2): d is a standardised distance
}


class BOLFI(SurrogateInference):
    """
    Bayesian Optimisation for Likelihood-Free Inference.

    Parameters
    ----------
    distance : callable
        ``distance(theta) -> float``: simulate at *theta* and return the
        discrepancy to the observed data.
    prior : list of (low, high), dict name -> (low, high), or prior object
        Bounds are converted to a :class:`~psiphy.utils.prior.UniformPrior`.
        A prior object must expose ``mins``/``maxs`` unless *bounds* is given.
    distance_kernel : {"exp", "gaussian"} or callable
        Maps the discrepancy to a log-likelihood. A callable must have
        signature ``log_kernel(d) -> log L``.
    acquisition : {"LCB", "EI", "PI"} or callable
        Acquisition function choosing where to simulate next; see
        :mod:`psiphy.utils.bayesian_opt`. ``"LCB"`` follows Gutmann & Corander (2016).
    learn_log_dist : bool
        Model ``log d`` instead of ``d`` (helps when d spans orders of magnitude).
    n_repeats : int
        Distance evaluations (simulations) averaged per parameter value. With a
        stochastic simulator each ``d(theta)`` is noisy; averaging ``n`` of them
        cuts the noise variance by ``n`` and makes the GP surrogate, and the
        posterior widths, more stable from run to run. The simulations at one
        theta are independent, so they can run in parallel. The extra noise in
        a single-simulation ``d`` adds a roughly constant offset that cancels
        in the normalised posterior.
    bounds : list of (low, high), optional
        Search space for the optimisation; defaults to the prior bounds.
    param_names : list of str, optional
    verbose : bool
    """

    _minimize = True
    _config_keys = ("learn_log_dist", "n_repeats")

    def __init__(self, distance, prior, distance_kernel="exp", acquisition="LCB",
                 learn_log_dist=False, n_repeats=1, bounds=None, param_names=None, verbose=True):
        if callable(distance_kernel):
            self.log_kernel = distance_kernel
        elif distance_kernel in _LOG_KERNELS:
            self.log_kernel = _LOG_KERNELS[distance_kernel]
        else:
            raise ValueError(f"Unknown distance_kernel {distance_kernel!r}. "
                             f"Choose from {list(_LOG_KERNELS)} or pass a callable.")
        super().__init__(prior, acquisition, bounds=bounds, param_names=param_names, verbose=verbose)
        if int(n_repeats) < 1:
            raise ValueError("n_repeats must be >= 1.")
        self.distance = distance
        self.learn_log_dist = learn_log_dist
        self.n_repeats = int(n_repeats)

    @property
    def n_simulations(self):
        """Distance evaluations (simulations) used so far."""
        return len(self.y) * self.n_repeats

    def _objective(self, theta):
        d = float(np.mean([self.distance(theta) for _ in range(self.n_repeats)]))
        return np.log(d) if self.learn_log_dist else d

    def predict_distance(self, theta, return_std=False):
        """GP mean (and std) of the modelled discrepancy (``log d`` if ``learn_log_dist``)."""
        return self._predict(theta, return_std)

    def log_likelihood(self, theta):
        """Synthetic log-likelihood ``log K(d_hat(theta))``."""
        mu = self._predict(theta)
        return self.log_kernel(np.exp(mu) if self.learn_log_dist else mu)

    def learn_likelihood(self, n_calls=100, n_initial_points=None, batch_size=1, filename=None,
                         reset_model=False, random_state=None, n_candidates=None, **acq_kwargs):
        """
        Learn the discrepancy surrogate with Bayesian optimisation.

        Parameters
        ----------
        n_calls : int
            Number of new parameter values evaluated (including initial design
            points); each costs ``n_repeats`` distance evaluations.
        n_initial_points : int, optional
            Size of the Latin-hypercube initial design; default ``5 * n_dim``.
            Points already loaded from *filename* count towards it.
        batch_size : int
            Points proposed per optimisation round (kriging believer). The
            points of one batch are independent, so their simulations can run
            in parallel; they are currently evaluated in sequence.
        filename : str, optional
            Pickle file to resume from (unless *reset_model*) and save to.
        reset_model : bool
            Ignore an existing *filename*.
        random_state : int, optional
        n_candidates : int, optional
            Random candidates scored per pick (default ``1000 * n_dim``).
        **acq_kwargs :
            Acquisition options, e.g. ``kappa`` (LCB) or ``xi`` (EI/PI).

        Returns
        -------
        self
        """
        self._print(f"Learning synthetic likelihood with Bayesian optimisation ({self.acquisition})...")
        self._learn(n_calls, n_initial_points, batch_size, filename, reset_model,
                    random_state, n_candidates, **acq_kwargs)
        self._print("...done")
        return self

    # names kept from the original BOLFI interface
    save_likelihood_model = SurrogateInference.save_model
    load_likelihood_model = SurrogateInference.load_model
