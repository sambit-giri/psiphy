import random as _random

import numpy as np
from scipy.stats import gaussian_kde

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class UniformPrior:
    """
    Box-uniform prior for use with importance sampling and SMC.

    Parameters
    ----------
    bounds : list of (low, high) tuples, or dict mapping name -> (low, high)

    Examples
    --------
    >>> prior = UniformPrior([(0, 1), (-2, 2)])
    >>> prior = UniformPrior({'mu': (-5, 5), 'sigma': (0.1, 5)})
    """

    def __init__(self, bounds):
        if isinstance(bounds, dict):
            bounds = list(bounds.values())
        self.bounds = np.asarray(bounds, dtype=float)
        self.mins = self.bounds[:, 0]
        self.maxs = self.bounds[:, 1]
        self._log_vol = float(np.sum(np.log(self.maxs - self.mins)))

    @property
    def ndim(self):
        return len(self.mins)

    def sample(self, n=1):
        """Draw *n* samples; returns shape (n, ndim)."""
        return self.mins + (self.maxs - self.mins) * np.random.uniform(size=(n, self.ndim))

    def log_prob(self, theta):
        theta = np.asarray(theta)
        if np.any(theta < self.mins) or np.any(theta > self.maxs):
            return -np.inf
        return -self._log_vol

    def in_support(self, theta):
        theta = np.asarray(theta)
        return bool(np.all(theta >= self.mins) and np.all(theta <= self.maxs))


def eff_sample_size(weights):
    """
    Effective sample size (ESS) of a weighted particle set.

    ESS = 1 / sum(w_i^2)  for normalised weights w_i.

    Parameters
    ----------
    weights : array-like, shape (n,)
        Raw or normalised importance weights.

    Returns
    -------
    float
        ESS in [1, n].
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return 1.0 / float(np.sum(w ** 2))


# ---------------------------------------------------------------------------
# Importance Sampling
# ---------------------------------------------------------------------------

def importance_sampling(log_target_fn, proposal_samples, log_proposal_fn=None):
    """
    Basic importance sampling.

    Parameters
    ----------
    log_target_fn : callable
        Log of the (unnormalised) target density.
        Signature: ``log_target_fn(theta) -> float``
    proposal_samples : np.ndarray, shape (n, d)
        Samples already drawn from the proposal distribution.
    log_proposal_fn : callable or None
        Log of the proposal density evaluated at each sample.
        ``None`` assumes a uniform (constant) proposal.

    Returns
    -------
    samples : np.ndarray, shape (n, d)
        Particles resampled proportionally to the importance weights.
    weights : np.ndarray, shape (n,)
        Normalised importance weights (sum to 1).

    Notes
    -----
    ESS can be computed with :func:`eff_sample_size`.

    Examples
    --------
    >>> prior = UniformPrior([(-3, 3), (0.1, 3)])
    >>> proposal_samples = prior.sample(2000)
    >>> samples, w = importance_sampling(log_posterior, proposal_samples)
    """
    proposal_samples = np.asarray(proposal_samples)
    n = len(proposal_samples)

    log_w = np.array([log_target_fn(s) for s in proposal_samples])

    if log_proposal_fn is not None:
        log_w -= np.array([log_proposal_fn(s) for s in proposal_samples])

    # numerically stable: shift by max before exp
    log_w -= np.nanmax(log_w)
    log_w = np.where(np.isfinite(log_w), log_w, -np.inf)
    w = np.exp(log_w)
    w /= w.sum()

    idx = np.random.choice(n, size=n, p=w, replace=True)
    return proposal_samples[idx], w


# ---------------------------------------------------------------------------
# Sequential Importance Sampling (SIS)
# ---------------------------------------------------------------------------

def sequential_importance_sampling(
    log_target_fn,
    prior,
    n_particles=500,
    n_steps=5,
    kernel="EmpiricalCovariance",
    ess_threshold=0.5,
    verbose=True,
):
    """
    Sequential importance sampling with MCMC move kernel.

    Each step re-weights the current particles under the target and
    applies a Gaussian random-walk MCMC move to diversify them.

    Parameters
    ----------
    log_target_fn : callable
        Log of the (unnormalised) target.
    prior : UniformPrior or object with .sample(n) and .log_prob(theta)
        Used to draw the initial particle set.
    n_particles : int
    n_steps : int
        Number of SIS iterations.
    kernel : str or sklearn covariance estimator
        Kernel used to estimate the MCMC move covariance.
        Accepts ``'EmpiricalCovariance'`` or ``'LedoitWolf'``.
    ess_threshold : float
        Resample when ESS / n_particles falls below this fraction.
    verbose : bool

    Returns
    -------
    samples : np.ndarray, shape (n_particles, d)
    weights : np.ndarray, shape (n_particles,)
    """
    cov_est = _get_cov_estimator(kernel)

    particles = prior.sample(n_particles)
    log_w = np.array([log_target_fn(s) for s in particles])
    log_w -= np.nanmax(log_w)
    w = np.exp(np.where(np.isfinite(log_w), log_w, -np.inf))
    w /= w.sum()

    iterator = tqdm(range(n_steps), desc="SIS") if verbose else range(n_steps)
    for step in iterator:
        ess = eff_sample_size(w)
        if verbose and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(ESS=f"{ess:.0f}/{n_particles}")

        if ess / n_particles < ess_threshold:
            idx = np.random.choice(n_particles, size=n_particles, p=w, replace=True)
            particles = particles[idx]
            cov = 2.0 * cov_est.fit(particles).covariance_
            perturbed = np.array([
                np.random.multivariate_normal(p, cov) for p in particles
            ])
            kde = gaussian_kde(perturbed.T)
            log_w = np.array([log_target_fn(s) for s in perturbed]) - np.log(kde(perturbed.T))
            log_w -= np.nanmax(log_w)
            w = np.exp(np.where(np.isfinite(log_w), log_w, -np.inf))
            w /= w.sum()
            particles = perturbed

    idx = np.random.choice(n_particles, size=n_particles, p=w, replace=True)
    return particles[idx], w


# ---------------------------------------------------------------------------
# SMC with annealed likelihood tempering
# ---------------------------------------------------------------------------

def SMC(
    log_likelihood_fn,
    prior,
    n_particles=1000,
    n_steps=10,
    kernel="EmpiricalCovariance",
    ess_threshold=0.5,
    verbose=True,
):
    """
    Sequential Monte Carlo via likelihood tempering.

    Anneals from the prior to the posterior through a sequence of
    tempered targets::

        p_t(theta) ∝ prior(theta) * likelihood(theta)^beta_t

    where ``beta_t = (t / n_steps)^2`` for t = 0, ..., n_steps.

    Parameters
    ----------
    log_likelihood_fn : callable
        Log-likelihood. ``log_likelihood_fn(theta) -> float``
    prior : UniformPrior or object with .sample(n) and .log_prob(theta)
    n_particles : int
    n_steps : int
        Number of temperature steps.
    kernel : str or sklearn covariance estimator
    ess_threshold : float
        Fraction of n_particles below which resampling + MCMC move fires.
    verbose : bool

    Returns
    -------
    samples : np.ndarray, shape (n_particles, d)
        Unweighted posterior samples (after final resampling).
    weights : np.ndarray, shape (n_particles,)
        Final normalised importance weights.

    Notes
    -----
    Effective sample size at each step is reported when ``verbose=True``.
    """
    cov_est = _get_cov_estimator(kernel)

    # temperature schedule: quadratic ramp 0 -> 1
    betas = np.linspace(0, 1, n_steps + 1) ** 2

    # initialise from prior
    particles = prior.sample(n_particles)
    log_likes = np.array([log_likelihood_fn(s) for s in particles])
    w = np.ones(n_particles) / n_particles

    iterator = tqdm(range(1, n_steps + 1), desc="SMC") if verbose else range(1, n_steps + 1)
    for t in iterator:
        delta = betas[t] - betas[t - 1]

        # incremental weight update
        log_incr = delta * log_likes
        log_incr -= np.nanmax(log_incr)
        w_incr = np.exp(np.where(np.isfinite(log_incr), log_incr, -np.inf))
        w = w * w_incr
        w /= w.sum()

        ess = eff_sample_size(w)
        if verbose and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(beta=f"{betas[t]:.3f}", ESS=f"{ess:.0f}/{n_particles}")

        if ess / n_particles < ess_threshold:
            # resample
            idx = np.random.choice(n_particles, size=n_particles, p=w, replace=True)
            particles = particles[idx]
            log_likes = log_likes[idx]
            w = np.ones(n_particles) / n_particles

            # MCMC move (Gaussian random walk)
            cov = 2.38 ** 2 / particles.shape[1] * cov_est.fit(particles).covariance_
            for i in range(n_particles):
                proposal = np.random.multivariate_normal(particles[i], cov)
                log_lp_prop = log_likelihood_fn(proposal)
                log_prior_prop = prior.log_prob(proposal)
                log_prior_cur = prior.log_prob(particles[i])
                log_accept = (
                    betas[t] * (log_lp_prop - log_likes[i])
                    + (log_prior_prop - log_prior_cur)
                )
                if np.log(np.random.uniform()) < log_accept:
                    particles[i] = proposal
                    log_likes[i] = log_lp_prop

    # final resample to get unweighted draws
    idx = np.random.choice(n_particles, size=n_particles, p=w, replace=True)
    return particles[idx], w


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _get_cov_estimator(kernel):
    if kernel == "EmpiricalCovariance":
        from sklearn.covariance import EmpiricalCovariance
        return EmpiricalCovariance()
    elif kernel == "LedoitWolf":
        from sklearn.covariance import LedoitWolf
        return LedoitWolf()
    else:
        return kernel  # assume already an estimator instance
