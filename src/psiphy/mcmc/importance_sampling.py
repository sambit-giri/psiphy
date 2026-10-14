import numpy as np
from scipy.special import logsumexp
from scipy.stats import gaussian_kde

from ..utils.prior import UniformPrior  # re-exported for backwards compatibility

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _result(samples, weights, log_evidence=None, **extra):
    n = len(weights)
    idx = np.random.choice(n, size=n, p=weights, replace=True)
    return {
        "samples": samples,
        "weights": weights,
        "equal_weight_samples": samples[idx],
        "ess": eff_sample_size(weights),
        "log_evidence": log_evidence,
        **extra,
    }


# ---------------------------------------------------------------------------
# Importance Sampling
# ---------------------------------------------------------------------------

def importance_sampling(log_target_fn, proposal_samples, log_proposal_fn=None, vectorized=False):
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
    vectorized : bool
        If True, ``log_target_fn`` (and ``log_proposal_fn``) accept a batch of
        shape (n, d) and return shape (n,), so all particles are evaluated in
        one call.

    Returns
    -------
    dict
        ``samples`` : (n, d) final weighted particles.
        ``weights`` : (n,) normalised weights, aligned with ``samples``.
        ``equal_weight_samples`` : (n, d) particles resampled by weight;
        use these where unweighted draws are needed.
        ``ess`` : effective sample size of ``weights``.
        ``log_evidence`` : log marginal likelihood estimate, or None.
        ``samples`` are the proposal samples themselves. ``log_evidence`` is
        given only when ``log_proposal_fn`` is supplied (it needs the
        normalised proposal density).

    Examples
    --------
    >>> prior = UniformPrior([(-3, 3), (0.1, 3)])
    >>> res = importance_sampling(log_posterior, prior.sample(2000), log_proposal_fn=prior.log_prob)
    >>> np.average(res["samples"], weights=res["weights"], axis=0)
    """
    proposal_samples = np.asarray(proposal_samples)

    log_w = _batch_eval(log_target_fn, proposal_samples, vectorized)
    log_z = None
    if log_proposal_fn is not None:
        log_w -= _batch_eval(log_proposal_fn, proposal_samples, vectorized)
        log_z = _log_mean_exp(log_w)
    return _result(proposal_samples, _normalise_log_weights(log_w), log_z)


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
    vectorized=False,
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
    vectorized : bool
        If True, ``log_target_fn`` accepts a batch (n, d) and returns (n,).

    Returns
    -------
    dict
        ``samples`` : (n, d) final weighted particles.
        ``weights`` : (n,) normalised weights, aligned with ``samples``.
        ``equal_weight_samples`` : (n, d) particles resampled by weight;
        use these where unweighted draws are needed.
        ``ess`` : effective sample size of ``weights``.
        ``log_evidence`` : log marginal likelihood estimate, or None.
        ``log_evidence`` is always None for SIS.
    """
    cov_est = _get_cov_estimator(kernel)

    particles = prior.sample(n_particles)
    w = _normalise_log_weights(_batch_eval(log_target_fn, particles, vectorized))

    iterator = tqdm(range(n_steps), desc="SIS") if verbose else range(n_steps)
    for step in iterator:
        ess = eff_sample_size(w)
        if verbose and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(ESS=f"{ess:.0f}/{n_particles}")

        if ess / n_particles < ess_threshold:
            idx = np.random.choice(n_particles, size=n_particles, p=w, replace=True)
            particles = particles[idx]
            cov = 2.0 * cov_est.fit(particles).covariance_
            perturbed = _gaussian_perturb(particles, cov)
            kde = gaussian_kde(perturbed.T)
            log_w = _batch_eval(log_target_fn, perturbed, vectorized) - np.log(kde(perturbed.T))
            w = _normalise_log_weights(log_w)
            particles = perturbed

    return _result(particles, w)


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
    vectorized=False,
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
    vectorized : bool
        If True, ``log_likelihood_fn`` accepts a batch (n, d) and returns (n,).

    Returns
    -------
    dict
        ``samples`` : (n, d) final weighted particles.
        ``weights`` : (n,) normalised weights, aligned with ``samples``.
        ``equal_weight_samples`` : (n, d) particles resampled by weight;
        use these where unweighted draws are needed.
        ``ess`` : effective sample size of ``weights``.
        ``log_evidence`` : log marginal likelihood estimate, or None.
        ``log_evidence`` is the standard SMC estimate (sum over temperature
        steps of the log mean incremental weight); it requires a normalised
        ``log_likelihood_fn`` and prior. Also returns ``log_likelihood`` (n,).

    Notes
    -----
    Effective sample size at each step is reported when ``verbose=True``.
    """
    cov_est = _get_cov_estimator(kernel)

    # temperature schedule: quadratic ramp 0 -> 1
    betas = np.linspace(0, 1, n_steps + 1) ** 2

    # initialise from prior
    particles = prior.sample(n_particles)
    log_likes = _batch_eval(log_likelihood_fn, particles, vectorized)
    log_priors = _batch_eval(prior.log_prob, particles, _accepts_batch(prior))
    w = np.ones(n_particles) / n_particles
    log_z = 0.0

    iterator = tqdm(range(1, n_steps + 1), desc="SMC") if verbose else range(1, n_steps + 1)
    for t in iterator:
        delta = betas[t] - betas[t - 1]

        # incremental weight update; log Z_t / Z_{t-1} = log sum_i w_i * L_i^delta
        log_incr = np.where(np.isfinite(log_likes), delta * log_likes, -np.inf)
        with np.errstate(divide="ignore"):
            log_wt = np.log(w) + log_incr
        log_z += logsumexp(log_wt)
        w = _normalise_log_weights(log_wt)

        ess = eff_sample_size(w)
        if verbose and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(beta=f"{betas[t]:.3f}", ESS=f"{ess:.0f}/{n_particles}")

        if ess / n_particles < ess_threshold:
            # resample
            idx = np.random.choice(n_particles, size=n_particles, p=w, replace=True)
            particles = particles[idx]
            log_likes = log_likes[idx]
            log_priors = log_priors[idx]
            w = np.ones(n_particles) / n_particles

            # MCMC move (Gaussian random walk), all particles at once
            cov = 2.38 ** 2 / particles.shape[1] * cov_est.fit(particles).covariance_
            proposals = _gaussian_perturb(particles, cov)
            log_priors_prop = _batch_eval(prior.log_prob, proposals, _accepts_batch(prior))
            log_likes_prop = np.full(n_particles, -np.inf)
            inside = np.isfinite(log_priors_prop)  # skip likelihood calls outside the prior
            if inside.any():
                log_likes_prop[inside] = _batch_eval(log_likelihood_fn, proposals[inside], vectorized)
            with np.errstate(invalid="ignore"):
                log_accept = betas[t] * (log_likes_prop - log_likes) + (log_priors_prop - log_priors)
            accept = np.log(np.random.uniform(size=n_particles)) < log_accept
            particles[accept] = proposals[accept]
            log_likes[accept] = log_likes_prop[accept]
            log_priors[accept] = log_priors_prop[accept]

    return _result(particles, w, float(log_z), log_likelihood=log_likes)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _batch_eval(fn, X, vectorized):
    """Evaluate *fn* on every row of *X*; one batched call if *vectorized*."""
    if vectorized:
        return np.asarray(fn(X), dtype=float).reshape(len(X))
    return np.array([fn(x) for x in X], dtype=float)


def _accepts_batch(prior):
    from ..utils.prior import BasePrior
    return isinstance(prior, BasePrior)


def _log_mean_exp(log_w):
    log_w = np.where(np.isfinite(log_w), log_w, -np.inf)
    return float(logsumexp(log_w) - np.log(len(log_w)))


def _normalise_log_weights(log_w):
    log_w = np.where(np.isfinite(log_w), log_w, -np.inf)
    w = np.exp(log_w - np.max(log_w))
    return w / w.sum()


def _gaussian_perturb(particles, cov):
    """Draw one N(particle, cov) sample per particle in a single matrix operation."""
    chol = np.linalg.cholesky(cov + 1e-12 * np.eye(len(cov)))
    return particles + np.random.standard_normal(particles.shape) @ chol.T


def _get_cov_estimator(kernel):
    if kernel == "EmpiricalCovariance":
        from sklearn.covariance import EmpiricalCovariance
        return EmpiricalCovariance()
    elif kernel == "LedoitWolf":
        from sklearn.covariance import LedoitWolf
        return LedoitWolf()
    else:
        return kernel  # assume already an estimator instance
