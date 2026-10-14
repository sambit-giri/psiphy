"""
Nested sampling with interchangeable backends.

Supported backends: ``"dynesty"`` (static nested sampling) and ``"nautilus"``
(neural-network-boosted importance nested sampling).

Examples
--------
>>> sampler = NestedSampler(log_likelihood_fn, prior_transform, n_dim=3, backend="dynesty")
>>> sampler.run()
>>> result = sampler.get_result()   # log_evidence, samples, weights
"""

import numpy as np

_BACKENDS = ("dynesty", "nautilus")


def _resample_equal(samples, weights, rng):
    """Systematic resampling of a weighted sample set to equal weights."""
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    idx = np.searchsorted(np.cumsum(weights), positions)
    return samples[np.minimum(idx, n - 1)]


class NestedSampler:
    """
    Backend-agnostic nested sampling wrapper.

    Parameters
    ----------
    log_likelihood_fn : callable
        ``log_likelihood_fn(theta) -> float``.
    prior_transform : callable or psiphy.utils.prior.BasePrior
        Maps a point in the unit hypercube to parameter space. A prior object
        is accepted directly; its ``prior_transform``, ``ndim`` and ``names``
        are used.
    n_dim : int, optional
        Number of parameters. Inferred when *prior_transform* is a prior object.
    backend : {"dynesty", "nautilus"}
    n_live : int
        Number of live points.
    param_names : list of str, optional
    pool : object with a ``map`` method, optional
        For parallel likelihood evaluation. With dynesty also pass ``queue_size``.
    seed : int, optional
        Random seed for the sampler and for equal-weight resampling.
    **kwargs :
        Extra keyword arguments for the backend's sampler constructor
        (``dynesty.NestedSampler`` or ``nautilus.Sampler``).
    """

    def __init__(self, log_likelihood_fn, prior_transform, n_dim=None, backend="dynesty",
                 n_live=500, param_names=None, pool=None, seed=None, **kwargs):
        if backend not in _BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}. Choose from {list(_BACKENDS)}.")
        if hasattr(prior_transform, "prior_transform"):
            prior = prior_transform
            prior_transform = prior.prior_transform
            n_dim = prior.ndim if n_dim is None else n_dim
            param_names = param_names or prior.names
        if n_dim is None:
            raise ValueError("n_dim is required when prior_transform is a plain function.")

        self.log_likelihood_fn = log_likelihood_fn
        self.prior_transform = prior_transform
        self.n_dim = int(n_dim)
        self.backend = backend
        self.n_live = int(n_live)
        self.param_names = list(param_names) if param_names is not None else None
        self.pool = pool
        self.seed = seed
        self.kwargs = kwargs
        self.sampler = None
        self._result = None

    # ---- running ----

    def run(self, dlogz=0.01, progress=False, **run_kwargs):
        """
        Run the sampler to convergence.

        Parameters
        ----------
        dlogz : float
            Stopping tolerance. For dynesty this is the remaining evidence
            ``dlogz``; for nautilus it is used as ``f_live`` (fraction of
            evidence in the live set).
        progress : bool
            Print backend progress.
        **run_kwargs :
            Extra keyword arguments for ``run_nested`` (dynesty) or ``run`` (nautilus).

        Returns
        -------
        self
        """
        runner = {"dynesty": self._run_dynesty, "nautilus": self._run_nautilus}[self.backend]
        self._result = runner(dlogz, progress, run_kwargs)
        return self

    def _run_dynesty(self, dlogz, progress, run_kwargs):
        try:
            import dynesty
        except ImportError as err:
            raise ImportError("backend='dynesty' requires dynesty: pip install 'psiphy[mcmc]'") from err
        kw = dict(nlive=self.n_live, rstate=np.random.default_rng(self.seed), pool=self.pool)
        kw.update(self.kwargs)
        self.sampler = dynesty.NestedSampler(self.log_likelihood_fn, self.prior_transform,
                                             self.n_dim, **kw)
        self.sampler.run_nested(dlogz=dlogz, print_progress=progress, **run_kwargs)
        res = self.sampler.results
        return self._package(res.samples, res.logwt, res.logl,
                             float(res.logz[-1]), float(res.logzerr[-1]))

    def _run_nautilus(self, dlogz, progress, run_kwargs):
        try:
            import nautilus
        except ImportError as err:
            raise ImportError(
                "backend='nautilus' requires nautilus-sampler: pip install 'psiphy[mcmc]'") from err
        self.sampler = nautilus.Sampler(self.prior_transform, self.log_likelihood_fn,
                                        n_dim=self.n_dim, n_live=self.n_live, pool=self.pool,
                                        seed=self.seed, **self.kwargs)
        run_kwargs.setdefault("discard_exploration", True)
        self.sampler.run(f_live=dlogz, verbose=progress, **run_kwargs)
        points, log_w, log_l = self.sampler.posterior()
        # nautilus provides no evidence error; its documented accuracy scales as 1/sqrt(n_eff)
        return self._package(points, log_w, log_l, float(self.sampler.log_z), None)

    def _package(self, samples, log_w, log_l, log_z, log_z_err):
        log_w = np.asarray(log_w, dtype=float)
        weights = np.exp(log_w - np.max(log_w))
        weights /= weights.sum()
        samples = np.asarray(samples, dtype=float)
        rng = np.random.default_rng(self.seed)
        return {
            "log_evidence": log_z,
            "log_evidence_err": log_z_err,
            "samples": samples,
            "weights": weights,
            "log_likelihood": np.asarray(log_l, dtype=float),
            "equal_weight_samples": _resample_equal(samples, weights, rng),
            "ess": float(1.0 / np.sum(weights**2)),
            "backend": self.backend,
        }

    # ---- outputs ----

    def get_result(self):
        """
        Unified results dictionary (same keys for every backend).

        Returns
        -------
        dict
            ``log_evidence``, ``log_evidence_err`` (None for nautilus),
            ``samples`` (n, n_dim), normalised ``weights`` (n,),
            ``log_likelihood`` (n,), ``equal_weight_samples``, ``ess``, ``backend``.
        """
        if self._result is None:
            raise RuntimeError("Call run() first.")
        return self._result

    def to_getdist(self, names=None, labels=None, label=None):
        """Weighted ``getdist.MCSamples`` of the posterior."""
        from getdist import MCSamples
        res = self.get_result()
        names = names or self.param_names or [f"p{i}" for i in range(self.n_dim)]
        return MCSamples(samples=res["samples"], weights=res["weights"],
                         loglikes=-res["log_likelihood"], names=names,
                         labels=labels or names, label=label)
