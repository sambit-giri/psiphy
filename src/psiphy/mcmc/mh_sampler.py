"""
Ensemble Metropolis-Hastings style sampling with a pluggable backend.

Currently supported backends: ``"emcee"``.

Examples
--------
>>> sampler = MHSampler(log_prob_fn, n_walkers=32, backend="emcee")
>>> sampler.run(theta_init, n_steps=1000)
>>> chain = sampler.get_chain(flat=True, discard=200, thin=10)
"""

import os

import numpy as np

_BACKENDS = ("emcee",)


def _import_emcee():
    try:
        import emcee
    except ImportError as err:
        raise ImportError(
            "MHSampler with backend='emcee' requires emcee: pip install 'psiphy[mcmc]'"
        ) from err
    return emcee


class MHSampler:
    """
    Wrapper around ensemble MCMC samplers.

    Parameters
    ----------
    log_prob_fn : callable
        ``log_prob_fn(theta) -> float``; return ``-np.inf`` outside the support.
    n_walkers : int
        Number of walkers in the ensemble.
    backend : str
        Sampling library. Only ``"emcee"`` is available.
    filename : str or None
        If given, the chain is stored in this HDF5 file (requires ``h5py``) and
        can be reloaded with :meth:`load` or extended with further :meth:`run` calls.
    param_names : list of str, optional
        Names used by :meth:`summary`.
    moves : emcee move or list, optional
        Proposal moves forwarded to ``emcee.EnsembleSampler``.
    pool : object with a ``map`` method, optional
        For parallel log-prob evaluation.
    **kwargs :
        Extra keyword arguments forwarded to ``emcee.EnsembleSampler``.
    """

    def __init__(self, log_prob_fn, n_walkers=32, backend="emcee", filename=None,
                 param_names=None, moves=None, pool=None, **kwargs):
        if backend not in _BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}. Choose from {list(_BACKENDS)}.")
        self.log_prob_fn = log_prob_fn
        self.n_walkers = int(n_walkers)
        self.backend = backend
        self.filename = filename
        self.param_names = list(param_names) if param_names is not None else None
        self.moves = moves
        self.pool = pool
        self.kwargs = kwargs
        self.sampler = None

    # ---- construction ----

    def _make_sampler(self, n_dim, reset):
        emcee = _import_emcee()
        store = None
        if self.filename is not None:
            store = emcee.backends.HDFBackend(self.filename)
            if reset or store.iteration == 0:
                store.reset(self.n_walkers, n_dim)
            elif store.shape != (self.n_walkers, n_dim):
                raise ValueError(f"Existing chain in {self.filename} has shape {store.shape}, "
                                 f"expected {(self.n_walkers, n_dim)}. Pass reset=True.")
        self.sampler = emcee.EnsembleSampler(
            self.n_walkers, n_dim, self.log_prob_fn, backend=store,
            moves=self.moves, pool=self.pool, **self.kwargs,
        )

    def _initial_state(self, theta_init, init_scale):
        theta_init = np.asarray(theta_init, dtype=float)
        if theta_init.ndim == 1:
            scale = init_scale * np.maximum(np.abs(theta_init), 1.0)
            return theta_init + scale * np.random.standard_normal((self.n_walkers, theta_init.size))
        if theta_init.shape[0] != self.n_walkers:
            raise ValueError(f"theta_init has {theta_init.shape[0]} rows, expected n_walkers={self.n_walkers}.")
        return theta_init

    # ---- running ----

    def run(self, theta_init=None, n_steps=1000, init_scale=1e-3, reset=False, progress=False):
        """
        Advance the chain by *n_steps* iterations.

        Parameters
        ----------
        theta_init : array-like, shape (n_dim,) or (n_walkers, n_dim), or None
            Starting point. A single point is jittered into a small Gaussian
            ball of relative width *init_scale*. ``None`` continues from the
            last state (of this run, or of the chain stored in ``filename``).
        n_steps : int
            Number of iterations to add.
        reset : bool
            Discard any chain already stored in ``filename``.
        progress : bool
            Show emcee's progress bar.

        Returns
        -------
        self
        """
        n_stored = self._n_stored(reset)
        if theta_init is None and self.sampler is None and n_stored == 0:
            raise ValueError("theta_init is required for a fresh run.")

        if theta_init is not None:
            state = self._initial_state(theta_init, init_scale)
            if n_stored and not reset:
                raise ValueError(f"{self.filename} already holds {n_stored} iterations; "
                                 "pass theta_init=None to continue or reset=True to restart.")
            self._make_sampler(state.shape[1], reset=True)
        else:
            if self.sampler is None:
                self._make_sampler(self._stored_shape()[1], reset=False)
            state = None  # emcee resumes from the last stored sample

        self.sampler.run_mcmc(state, int(n_steps), progress=progress)
        return self

    def _n_stored(self, reset):
        if self.filename is None or reset:
            return 0
        if not os.path.exists(self.filename):
            return 0
        return _import_emcee().backends.HDFBackend(self.filename, read_only=True).iteration

    def _stored_shape(self):
        return _import_emcee().backends.HDFBackend(self.filename, read_only=True).shape

    @classmethod
    def load(cls, filename, log_prob_fn=None, **kwargs):
        """
        Reload a chain saved to an HDF5 file.

        Pass *log_prob_fn* to be able to continue sampling with ``run(None, n_steps)``.
        """
        emcee = _import_emcee()
        store = emcee.backends.HDFBackend(filename)
        if store.iteration == 0:
            raise ValueError(f"No samples stored in {filename}.")
        n_walkers, n_dim = store.shape
        obj = cls(log_prob_fn, n_walkers=n_walkers, filename=filename, **kwargs)
        obj._make_sampler(n_dim, reset=False)
        return obj

    # ---- outputs ----

    def _require_sampler(self):
        if self.sampler is None:
            raise RuntimeError("No chain available. Call run() or load() first.")
        return self.sampler

    def _resolve_discard(self, discard):
        if discard == "auto":
            return int(3 * np.max(self.autocorr_time()))
        return int(discard)

    @property
    def n_steps(self):
        return self._require_sampler().iteration

    def get_chain(self, flat=False, discard=0, thin=1):
        """
        Chain of shape (n_steps, n_walkers, n_dim), or (n_samples, n_dim) if *flat*.

        ``discard="auto"`` drops three times the longest autocorrelation time.
        """
        return self._require_sampler().get_chain(
            flat=flat, discard=self._resolve_discard(discard), thin=thin)

    def get_log_prob(self, flat=False, discard=0, thin=1):
        return self._require_sampler().get_log_prob(
            flat=flat, discard=self._resolve_discard(discard), thin=thin)

    def autocorr_time(self, quiet=True):
        """Integrated autocorrelation time per parameter."""
        return self._require_sampler().get_autocorr_time(quiet=quiet)

    @property
    def acceptance_fraction(self):
        return self._require_sampler().acceptance_fraction

    def summary(self, discard="auto", thin=1, print_out=True):
        """
        Per-parameter posterior summary.

        Returns
        -------
        dict
            ``name -> {mean, std, median, lo68, hi68, tau}`` plus
            ``"acceptance_fraction"`` and ``"n_samples"``.
        """
        samples = self.get_chain(flat=True, discard=discard, thin=thin)
        tau = self.autocorr_time()
        names = self.param_names or [f"p{i}" for i in range(samples.shape[1])]
        out = {}
        for i, name in enumerate(names):
            lo, med, hi = np.percentile(samples[:, i], [16, 50, 84])
            out[name] = dict(mean=samples[:, i].mean(), std=samples[:, i].std(),
                             median=med, lo68=lo, hi68=hi, tau=tau[i])
        out["acceptance_fraction"] = float(np.mean(self.acceptance_fraction))
        out["n_samples"] = samples.shape[0]

        if print_out:
            print(f"{'param':>10} {'mean':>11} {'std':>11} {'median':>11} {'tau':>8}")
            for name in names:
                r = out[name]
                print(f"{name:>10} {r['mean']:11.4g} {r['std']:11.4g} {r['median']:11.4g} {r['tau']:8.1f}")
            print(f"acceptance fraction = {out['acceptance_fraction']:.3f}, "
                  f"n_samples = {out['n_samples']}")
        return out
