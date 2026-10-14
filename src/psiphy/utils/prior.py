"""
Framework-agnostic prior specifications.

Every prior exposes the same interface so it can be passed to the importance
sampling / SMC routines, MCMC samplers, nested samplers (via
:meth:`BasePrior.prior_transform`) and, through the conversion helpers, to
``scipy``, ``torch`` and ``sbi``.

Examples
--------
>>> prior = UniformPrior({'mu': (-5, 5), 'sigma': (0.1, 5)})
>>> theta = prior.sample(1000)                  # (1000, 2)
>>> lp = prior.log_prob(theta)                  # (1000,)
>>> theta = prior.prior_transform(np.random.rand(2))   # unit cube -> theta
"""

import numpy as np
from scipy import stats


def _parse_bounds(bounds):
    """Return (names or None, array of shape (ndim, 2))."""
    names = None
    if isinstance(bounds, dict):
        names = list(bounds.keys())
        bounds = list(bounds.values())
    bounds = np.atleast_2d(np.asarray(bounds, dtype=float))
    if bounds.shape[1] != 2:
        raise ValueError("bounds must be a sequence of (low, high) pairs.")
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Each bound must satisfy low < high.")
    return names, bounds


class BasePrior:
    """
    Common interface for all priors.

    Subclasses implement :meth:`sample`, :meth:`log_prob` and, where it exists
    in closed form, :meth:`prior_transform`.

    ``log_prob`` accepts a single point of shape ``(ndim,)`` (returns a float)
    or a batch of shape ``(n, ndim)`` (returns an array of shape ``(n,)``).
    """

    names = None

    @property
    def ndim(self):
        raise NotImplementedError

    def sample(self, n=1):
        """Draw *n* samples; returns shape (n, ndim)."""
        raise NotImplementedError

    def log_prob(self, theta):
        raise NotImplementedError

    def prior_transform(self, u):
        """Map points from the unit hypercube to parameter space (for nested sampling)."""
        raise NotImplementedError(
            f"{type(self).__name__} has no closed-form prior transform."
        )

    def in_support(self, theta):
        return bool(np.isfinite(self.log_prob(theta)))

    def to_scipy(self):
        raise NotImplementedError(f"{type(self).__name__} has no scipy equivalent.")

    def to_torch(self):
        raise NotImplementedError(f"{type(self).__name__} has no torch equivalent.")

    def to_sbi(self):
        """Return a distribution usable as an ``sbi`` prior (defaults to :meth:`to_torch`)."""
        return self.to_torch()

    # ---- helpers ----

    def _as_batch(self, theta):
        """Return (theta with shape (n, ndim), was_single_point)."""
        theta = np.asarray(theta, dtype=float)
        if theta.ndim == 0 or (theta.ndim == 1 and theta.size == self.ndim):
            return theta.reshape(1, self.ndim), True
        if theta.ndim == 1 and self.ndim == 1:
            return theta.reshape(-1, 1), False
        if theta.ndim != 2 or theta.shape[1] != self.ndim:
            raise ValueError(f"Expected shape (ndim,) or (n, ndim) with ndim={self.ndim}, "
                             f"got {theta.shape}.")
        return theta, False

    @staticmethod
    def _out(lp, single):
        return float(lp[0]) if single else lp


class UniformPrior(BasePrior):
    """
    Box-uniform prior.

    Parameters
    ----------
    bounds : list of (low, high) tuples, or dict mapping name -> (low, high)

    Examples
    --------
    >>> prior = UniformPrior([(0, 1), (-2, 2)])
    >>> prior = UniformPrior({'mu': (-5, 5), 'sigma': (0.1, 5)})
    """

    def __init__(self, bounds):
        self.names, self.bounds = _parse_bounds(bounds)
        self.mins = self.bounds[:, 0]
        self.maxs = self.bounds[:, 1]
        self._log_vol = float(np.sum(np.log(self.maxs - self.mins)))

    @property
    def ndim(self):
        return len(self.mins)

    def sample(self, n=1):
        return self.mins + (self.maxs - self.mins) * np.random.uniform(size=(n, self.ndim))

    def log_prob(self, theta):
        theta, single = self._as_batch(theta)
        inside = np.all((theta >= self.mins) & (theta <= self.maxs), axis=1)
        lp = np.where(inside, -self._log_vol, -np.inf)
        return self._out(lp, single)

    def prior_transform(self, u):
        return self.mins + (self.maxs - self.mins) * np.asarray(u, dtype=float)

    def to_scipy(self):
        """List of independent frozen ``scipy.stats.uniform``, one per parameter."""
        return [stats.uniform(loc=lo, scale=hi - lo) for lo, hi in self.bounds]

    def to_torch(self):
        import torch
        from torch.distributions import Independent, Uniform
        return Independent(
            Uniform(torch.as_tensor(self.mins, dtype=torch.float32),
                    torch.as_tensor(self.maxs, dtype=torch.float32)), 1)

    def to_sbi(self):
        import torch
        from sbi.utils import BoxUniform
        return BoxUniform(low=torch.as_tensor(self.mins, dtype=torch.float32),
                          high=torch.as_tensor(self.maxs, dtype=torch.float32))


class LogUniformPrior(BasePrior):
    """
    Independent log-uniform (Jeffreys-like) prior: log(theta) is uniform.

    Parameters
    ----------
    bounds : list of (low, high) tuples, or dict mapping name -> (low, high).
        All bounds must be strictly positive.
    """

    def __init__(self, bounds):
        self.names, self.bounds = _parse_bounds(bounds)
        if np.any(self.bounds <= 0):
            raise ValueError("LogUniformPrior bounds must be strictly positive.")
        self.mins = self.bounds[:, 0]
        self.maxs = self.bounds[:, 1]
        self._log_mins = np.log(self.mins)
        self._log_widths = np.log(self.maxs) - self._log_mins

    @property
    def ndim(self):
        return len(self.mins)

    def sample(self, n=1):
        return self.prior_transform(np.random.uniform(size=(n, self.ndim)))

    def log_prob(self, theta):
        theta, single = self._as_batch(theta)
        inside = np.all((theta >= self.mins) & (theta <= self.maxs), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = -np.sum(np.log(theta) + np.log(self._log_widths), axis=1)
        lp = np.where(inside, lp, -np.inf)
        return self._out(lp, single)

    def prior_transform(self, u):
        return np.exp(self._log_mins + self._log_widths * np.asarray(u, dtype=float))

    def to_scipy(self):
        """List of independent frozen ``scipy.stats.loguniform``, one per parameter."""
        return [stats.loguniform(lo, hi) for lo, hi in self.bounds]

    def to_torch(self):
        import torch
        from torch.distributions import Independent, TransformedDistribution, Uniform
        from torch.distributions.transforms import ExpTransform
        base = Uniform(torch.as_tensor(self._log_mins, dtype=torch.float32),
                       torch.as_tensor(np.log(self.maxs), dtype=torch.float32))
        return Independent(TransformedDistribution(base, [ExpTransform()]), 1)


class GaussianPrior(BasePrior):
    """
    Multivariate Gaussian prior.

    Parameters
    ----------
    mean : array-like, shape (ndim,)
    cov : float, array-like of shape (ndim,) (diagonal variances), or (ndim, ndim)
    names : list of str, optional
    """

    def __init__(self, mean, cov, names=None):
        self.mean = np.atleast_1d(np.asarray(mean, dtype=float))
        cov = np.asarray(cov, dtype=float)
        if cov.ndim == 0:
            cov = cov * np.eye(self.ndim)
        elif cov.ndim == 1:
            cov = np.diag(cov)
        if cov.shape != (self.ndim, self.ndim):
            raise ValueError(f"cov must have shape ({self.ndim}, {self.ndim}).")
        self.cov = cov
        self.names = list(names) if names is not None else None
        self._chol = np.linalg.cholesky(cov)
        self._dist = stats.multivariate_normal(self.mean, self.cov)

    @property
    def ndim(self):
        return len(self.mean)

    def sample(self, n=1):
        return self.mean + np.random.standard_normal((n, self.ndim)) @ self._chol.T

    def log_prob(self, theta):
        theta, single = self._as_batch(theta)
        lp = np.atleast_1d(self._dist.logpdf(theta))
        return self._out(lp, single)

    def prior_transform(self, u):
        z = stats.norm.ppf(np.asarray(u, dtype=float))
        return self.mean + z @ self._chol.T

    def to_scipy(self):
        return self._dist

    def to_torch(self):
        import torch
        from torch.distributions import MultivariateNormal
        return MultivariateNormal(torch.as_tensor(self.mean, dtype=torch.float32),
                                  covariance_matrix=torch.as_tensor(self.cov, dtype=torch.float32))


class MixturePrior(BasePrior):
    """
    Weighted mixture of priors with the same dimensionality.

    Parameters
    ----------
    components : list of BasePrior
    weights : array-like, optional
        Mixture weights (normalised internally). Equal weights by default.
    """

    def __init__(self, components, weights=None):
        self.components = list(components)
        if not self.components:
            raise ValueError("MixturePrior needs at least one component.")
        dims = {c.ndim for c in self.components}
        if len(dims) != 1:
            raise ValueError("All mixture components must have the same ndim.")
        w = np.ones(len(self.components)) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != (len(self.components),) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be non-negative, one per component.")
        self.weights = w / w.sum()
        self.names = self.components[0].names

    @property
    def ndim(self):
        return self.components[0].ndim

    def sample(self, n=1):
        counts = np.random.multinomial(n, self.weights)
        out = np.concatenate([c.sample(k) for c, k in zip(self.components, counts) if k > 0])
        return out[np.random.permutation(n)]

    def log_prob(self, theta):
        theta, single = self._as_batch(theta)
        comp_lp = np.stack([np.atleast_1d(c.log_prob(theta)) for c in self.components])
        with np.errstate(divide="ignore"):
            lp = np.logaddexp.reduce(comp_lp + np.log(self.weights)[:, None], axis=0)
        return self._out(lp, single)
