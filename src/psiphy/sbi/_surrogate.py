"""
Shared pieces for likelihood-free methods that learn a GP surrogate over the
parameter space (BOLFI, LFIRE): prior handling, the GP itself, and sampling the
resulting surrogate posterior with the psiphy samplers.
"""

import warnings

import numpy as np

from ..utils.prior import UniformPrior


def resolve_prior(prior, bounds=None):
    """
    Return ``(prior_object, bounds_array)``.

    *prior* may be a prior object or bounds (list of pairs / dict), which are
    turned into a :class:`~psiphy.utils.prior.UniformPrior`. *bounds* defaults
    to the prior's ``mins``/``maxs``.
    """
    if not hasattr(prior, "log_prob"):
        prior = UniformPrior(prior)
    if bounds is None:
        if not hasattr(prior, "mins"):
            raise ValueError("Pass bounds= for a prior without mins/maxs.")
        bounds = np.column_stack([prior.mins, prior.maxs])
    return prior, np.asarray(bounds, dtype=float)


def fit_gp(X_unit, y, random_state=None):
    """Fit a Matern-5/2 + white-noise GP to points in the unit cube."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    n_dim = X_unit.shape[1]
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=np.full(n_dim, 0.3), length_scale_bounds=(1e-3, 1e2), nu=2.5)
              + WhiteKernel(1e-3, (1e-8, 1e1)))
    gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=3,
                                  random_state=random_state)
    with warnings.catch_warnings():
        # hyperparameter optimiser often hits its iteration cap on tiny designs; the fit is still usable
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X_unit, y)
    return gp


def sample_posterior(log_posterior, log_likelihood, prior, param_names, n_samples=5000,
                     method="MCMC", verbose=False, **kwargs):
    """
    Draw equal-weight samples from a surrogate posterior.

    Parameters
    ----------
    log_posterior, log_likelihood : callable
        Accept a single point ``(d,)`` or a batch ``(n, d)``.
    prior : prior object
    param_names : list of str
    n_samples : int
    method : {"MCMC", "NestedSampling", "IS", "SMC"}
        ``"MCMC"`` uses :class:`~psiphy.mcmc.MHSampler` (kwargs: ``n_walkers``,
        ``n_burn``, ``filename``, plus sampler options);
        ``"NestedSampling"`` uses :class:`~psiphy.mcmc.NestedSampler`
        (kwargs: ``backend``, ``n_live``, ``dlogz``, ...);
        ``"IS"`` uses importance sampling with the prior as proposal;
        ``"SMC"`` uses :func:`~psiphy.mcmc.sequential_monte_carlo` (likelihood tempering).

    Returns
    -------
    samples : np.ndarray, shape (n_samples, d)
    info : sampler object or result dict
    """
    from .. import mcmc

    if method == "MCMC":
        n_walkers = kwargs.pop("n_walkers", 32)
        n_burn = kwargs.pop("n_burn", 500)
        n_steps = n_burn + int(np.ceil(n_samples / n_walkers))
        sampler = mcmc.MHSampler(log_posterior, n_walkers=n_walkers, param_names=param_names, **kwargs)
        sampler.run(prior.sample(n_walkers), n_steps=n_steps)
        return sampler.get_chain(flat=True, discard=n_burn)[-n_samples:], sampler

    if method == "NestedSampling":
        dlogz = kwargs.pop("dlogz", 0.1)
        sampler = mcmc.NestedSampler(log_likelihood, prior, param_names=param_names,
                                     **kwargs).run(dlogz=dlogz)
        res = sampler.get_result()
        idx = np.random.choice(len(res["weights"]), size=n_samples, p=res["weights"])
        return res["samples"][idx], sampler

    if method == "IS":
        info = mcmc.importance_sampling(log_likelihood, prior.sample(n_samples), vectorized=True)
        return info["equal_weight_samples"], info

    if method == "SMC":
        kwargs.setdefault("verbose", verbose)
        kwargs.setdefault("vectorized", True)
        info = mcmc.sequential_monte_carlo(log_likelihood, prior, n_particles=n_samples, **kwargs)
        return info["equal_weight_samples"], info

    raise ValueError(f"Unknown method {method!r}. Choose from 'MCMC', 'NestedSampling', 'IS', 'SMC'.")


class SurrogateInference:
    """
    Base class: learn a GP surrogate of a scalar objective over the parameter box
    with Bayesian optimisation, then sample the surrogate posterior.

    Subclasses define ``_objective(theta) -> float`` (one expensive evaluation),
    ``log_likelihood(theta)`` (from the GP prediction), and set ``_minimize``
    (whether low objective values are the interesting region). Attributes named
    in ``_config_keys`` must match when a saved model is loaded; attributes in
    ``_saved_arrays`` are stored and restored with it.
    """

    _minimize = True
    _config_keys = ()
    _saved_arrays = ()

    def __init__(self, prior, acquisition, bounds=None, param_names=None, verbose=True):
        from ..utils import bayesian_opt as bopt
        bopt.get_acquisition(acquisition)  # validate early
        self.prior, self.bounds = resolve_prior(prior, bounds)
        self.n_dim = len(self.bounds)
        self.acquisition = acquisition
        self.param_names = list(param_names or getattr(self.prior, "names", None)
                                or [f"p{i}" for i in range(self.n_dim)])
        self.verbose = verbose
        self.X = np.empty((0, self.n_dim))
        self.y = np.empty(0)
        self.gp = None
        self.sampler_info = None

    def _print(self, msg):
        if self.verbose:
            print(msg)

    # ---- surrogate ----

    def _to_unit(self, theta):
        return (theta - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])

    def _from_unit(self, u):
        return self.bounds[:, 0] + u * (self.bounds[:, 1] - self.bounds[:, 0])

    def _fit_gp(self, random_state=None):
        self.gp = fit_gp(self._to_unit(self.X), self.y, random_state)
        return self.gp

    def _predict(self, theta, return_std=False):
        """GP mean (and std) of the objective at a point ``(d,)`` or batch ``(n, d)``."""
        if self.gp is None:
            raise RuntimeError("No surrogate yet: learn it first or load a saved model.")
        theta = np.asarray(theta, dtype=float)
        single = theta.ndim == 1
        out = self.gp.predict(self._to_unit(np.atleast_2d(theta)), return_std=return_std)
        if single:
            return tuple(float(o[0]) for o in out) if return_std else float(out[0])
        return out

    def log_likelihood(self, theta):
        raise NotImplementedError

    def log_posterior(self, theta):
        lp = self.prior.log_prob(theta)
        if np.ndim(lp) == 0:
            return lp + self.log_likelihood(theta) if np.isfinite(lp) else -np.inf
        out = np.full(len(lp), -np.inf)
        ok = np.isfinite(lp)
        out[ok] = lp[ok] + self.log_likelihood(np.asarray(theta)[ok])
        return out

    # ---- learning ----

    def _objective(self, theta):
        raise NotImplementedError

    def _evaluate(self, thetas):
        """Evaluate the objective at each row of *thetas* and store the results."""
        values = [float(self._objective(np.asarray(t, dtype=float))) for t in thetas]
        self.X = np.vstack([self.X, thetas])
        self.y = np.append(self.y, values)

    def _learn(self, n_calls, n_initial_points=None, batch_size=1, filename=None,
               reset_model=False, random_state=None, n_candidates=None, **acq_kwargs):
        """Bayesian-optimisation loop shared by the public ``learn_*`` methods."""
        import os
        from scipy.stats.qmc import LatinHypercube
        from ..utils import bayesian_opt as bopt

        if filename is not None and not reset_model and os.path.exists(filename):
            self.load_model(filename)
            self._print(f"Resuming from {len(self.y)} stored evaluations.")
        elif reset_model:
            self.X, self.y = np.empty((0, self.n_dim)), np.empty(0)

        if n_initial_points is None:
            n_initial_points = 5 * self.n_dim
        n_init = int(min(max(n_initial_points - len(self.y), 0), n_calls))

        rng = np.random.default_rng(random_state)
        if n_init:
            self._evaluate(self._from_unit(LatinHypercube(d=self.n_dim, seed=rng).random(n_init)))
        unit_box = np.tile([0.0, 1.0], (self.n_dim, 1))
        remaining = n_calls - n_init
        while remaining > 0:
            gp = self._fit_gp(random_state)
            q = min(batch_size, remaining)
            u = bopt.propose_location(gp, unit_box, acquisition=self.acquisition,
                                      X_sample=self._to_unit(self.X), y_sample=self.y,
                                      batch_size=q, minimize=self._minimize,
                                      n_candidates=n_candidates, random_state=rng, **acq_kwargs)
            self._evaluate(self._from_unit(u))
            remaining -= q
        self._fit_gp(random_state)

        if filename is not None:
            self.save_model(filename)
        return self

    # ---- persistence ----

    def save_model(self, filename):
        """Store the evaluated points (the GP is refitted on load)."""
        import pickle
        state = {"X": self.X, "y": self.y, "bounds": self.bounds}
        state.update({k: getattr(self, k) for k in self._config_keys + self._saved_arrays})
        with open(filename, "wb") as f:
            pickle.dump(state, f)
        self._print(f"Saved surrogate model: {filename}")

    def load_model(self, filename):
        import pickle
        with open(filename, "rb") as f:
            state = pickle.load(f)
        bad = [k for k in self._config_keys if state.get(k) != getattr(self, k)]
        if np.shape(state["bounds"]) != self.bounds.shape or not np.allclose(state["bounds"], self.bounds):
            bad.insert(0, "bounds")
        if bad:
            raise ValueError(f"{filename} was created with different {', '.join(bad)}.")
        self.X, self.y = state["X"], state["y"]
        for k in self._saved_arrays:
            setattr(self, k, state[k])
        if len(self.y):
            self._fit_gp()
        self._print(f"Loaded surrogate model with {len(self.y)} evaluations.")
        return self

    # ---- sampling ----

    def sample_posterior(self, n_samples=5000, method="MCMC", **kwargs):
        """
        Draw equal-weight samples from the surrogate posterior.

        Parameters
        ----------
        n_samples : int
        method : {"MCMC", "NestedSampling", "IS", "SMC"}
            See :func:`psiphy.sbi._surrogate.sample_posterior` for the options
            of each method.

        Returns
        -------
        np.ndarray, shape (n_samples, n_dim)
            The full sampler output is kept in ``self.sampler_info``.
        """
        samples, self.sampler_info = sample_posterior(
            self.log_posterior, self.log_likelihood, self.prior, self.param_names,
            n_samples=n_samples, method=method, verbose=self.verbose, **kwargs)
        return samples
