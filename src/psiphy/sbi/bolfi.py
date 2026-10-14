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

import os
import pickle
import warnings

import numpy as np
from scipy.stats.qmc import LatinHypercube

from ..utils import bayesian_opt as bopt
from ..utils.prior import UniformPrior

_LOG_KERNELS = {
    "exp": lambda d: -0.5 * d,              # L ∝ exp(-d/2): d plays the role of a chi^2
    "gaussian": lambda d: -0.5 * d**2,      # L ∝ exp(-d^2/2): d is a standardised distance
}


class BOLFI:
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
    bounds : list of (low, high), optional
        Search space for the optimisation; defaults to the prior bounds.
    param_names : list of str, optional
    verbose : bool
    """

    def __init__(self, distance, prior, distance_kernel="exp", acquisition="LCB",
                 learn_log_dist=False, bounds=None, param_names=None, verbose=True):
        bopt.get_acquisition(acquisition)  # validate early
        if not hasattr(prior, "log_prob"):
            prior = UniformPrior(prior)
        if bounds is None:
            if not hasattr(prior, "mins"):
                raise ValueError("Pass bounds= for a prior without mins/maxs.")
            bounds = np.column_stack([prior.mins, prior.maxs])
        if callable(distance_kernel):
            self.log_kernel = distance_kernel
        elif distance_kernel in _LOG_KERNELS:
            self.log_kernel = _LOG_KERNELS[distance_kernel]
        else:
            raise ValueError(f"Unknown distance_kernel {distance_kernel!r}. "
                             f"Choose from {list(_LOG_KERNELS)} or pass a callable.")

        self.distance = distance
        self.prior = prior
        self.bounds = np.asarray(bounds, dtype=float)
        self.n_dim = len(self.bounds)
        self.acquisition = acquisition
        self.learn_log_dist = learn_log_dist
        self.param_names = list(param_names or getattr(prior, "names", None)
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

    def _objective(self, theta):
        d = float(self.distance(np.asarray(theta, dtype=float)))
        return np.log(d) if self.learn_log_dist else d

    def _fit_gp(self, random_state=None):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.full(self.n_dim, 0.3), length_scale_bounds=(1e-3, 1e2), nu=2.5)
                  + WhiteKernel(1e-3, (1e-8, 1e1)))
        self.gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=3,
                                           random_state=random_state)
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings():
            # hyperparameter optimiser often hits its iteration cap on tiny designs; the fit is still usable
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp.fit(self._to_unit(self.X), self.y)
        return self.gp

    def predict_distance(self, theta, return_std=False):
        """GP mean (and std) of the modelled discrepancy (``log d`` if ``learn_log_dist``)."""
        if self.gp is None:
            raise RuntimeError("Call learn_likelihood() or load_likelihood_model() first.")
        theta = np.asarray(theta, dtype=float)
        single = theta.ndim == 1
        out = self.gp.predict(self._to_unit(np.atleast_2d(theta)), return_std=return_std)
        if single:
            return tuple(float(o[0]) for o in out) if return_std else float(out[0])
        return out

    def log_likelihood(self, theta):
        """Synthetic log-likelihood ``log K(d_hat(theta))``."""
        mu = self.predict_distance(theta)
        return self.log_kernel(np.exp(mu) if self.learn_log_dist else mu)

    def log_posterior(self, theta):
        lp = self.prior.log_prob(theta)
        if np.ndim(lp) == 0:
            return lp + self.log_likelihood(theta) if np.isfinite(lp) else -np.inf
        out = np.full(len(lp), -np.inf)
        ok = np.isfinite(lp)
        out[ok] = lp[ok] + self.log_likelihood(np.asarray(theta)[ok])
        return out

    # ---- learning ----

    def learn_likelihood(self, n_calls=100, n_initial_points=None, batch_size=1, filename=None,
                         reset_model=False, random_state=None, n_candidates=None, **acq_kwargs):
        """
        Learn the discrepancy surrogate with Bayesian optimisation.

        Parameters
        ----------
        n_calls : int
            Number of new distance evaluations (including initial design points).
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
        if filename is not None and not reset_model and os.path.exists(filename):
            self.load_likelihood_model(filename)
            self._print(f"Resuming from {len(self.y)} stored evaluations.")
        elif reset_model:
            self.X, self.y = np.empty((0, self.n_dim)), np.empty(0)

        if n_initial_points is None:
            n_initial_points = 5 * self.n_dim
        n_init = int(min(max(n_initial_points - len(self.y), 0), n_calls))

        self._print(f"Learning synthetic likelihood with Bayesian optimisation ({self.acquisition})...")
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
                                      batch_size=q, minimize=True, n_candidates=n_candidates,
                                      random_state=rng, **acq_kwargs)
            self._evaluate(self._from_unit(u))
            remaining -= q
        self._fit_gp(random_state)
        self._print("...done")

        if filename is not None:
            self.save_likelihood_model(filename)
        return self

    def _evaluate(self, thetas):
        """Evaluate the discrepancy at each row of *thetas* and store the results."""
        values = [self._objective(t) for t in thetas]
        self.X = np.vstack([self.X, thetas])
        self.y = np.append(self.y, values)

    # ---- persistence ----

    def save_likelihood_model(self, filename):
        """Store the evaluated points; the GP is refitted on load."""
        with open(filename, "wb") as f:
            pickle.dump({"X": self.X, "y": self.y, "bounds": self.bounds,
                         "learn_log_dist": self.learn_log_dist}, f)
        self._print(f"Saved likelihood model: {filename}")

    def load_likelihood_model(self, filename):
        with open(filename, "rb") as f:
            state = pickle.load(f)
        if not np.allclose(state["bounds"], self.bounds) or state["learn_log_dist"] != self.learn_log_dist:
            raise ValueError(f"{filename} was created with different bounds or learn_log_dist.")
        self.X, self.y = state["X"], state["y"]
        if len(self.y):
            self._fit_gp()
        self._print(f"Loaded likelihood model with {len(self.y)} evaluations.")
        return self

    # ---- sampling ----

    def sample_posterior(self, n_samples=5000, method="MCMC", **kwargs):
        """
        Draw equal-weight posterior samples from the synthetic likelihood.

        Parameters
        ----------
        n_samples : int
        method : {"MCMC", "NestedSampling", "IS", "SMC"}
            ``"MCMC"`` uses :class:`~psiphy.mcmc.MHSampler` (kwargs: ``n_walkers``,
            ``n_burn``, ``filename``, plus sampler options);
            ``"NestedSampling"`` uses :class:`~psiphy.mcmc.NestedSampler`
            (kwargs: ``backend``, ``n_live``, ``dlogz``, ...);
            ``"IS"`` uses importance sampling with the prior as proposal;
            ``"SMC"`` uses likelihood-tempered :func:`~psiphy.mcmc.SMC`.

        Returns
        -------
        np.ndarray, shape (n_samples, n_dim)
            The full sampler output is kept in ``self.sampler_info``.
        """
        from .. import mcmc

        if method == "MCMC":
            n_walkers = kwargs.pop("n_walkers", 32)
            n_burn = kwargs.pop("n_burn", 500)
            n_steps = n_burn + int(np.ceil(n_samples / n_walkers))
            sampler = mcmc.MHSampler(self.log_posterior, n_walkers=n_walkers,
                                     param_names=self.param_names, **kwargs)
            sampler.run(self.prior.sample(n_walkers), n_steps=n_steps)
            self.sampler_info = sampler
            return sampler.get_chain(flat=True, discard=n_burn)[-n_samples:]

        if method == "NestedSampling":
            dlogz = kwargs.pop("dlogz", 0.1)
            sampler = mcmc.NestedSampler(self.log_likelihood, self.prior,
                                         param_names=self.param_names, **kwargs).run(dlogz=dlogz)
            self.sampler_info = sampler
            res = sampler.get_result()
            idx = np.random.choice(len(res["weights"]), size=n_samples, p=res["weights"])
            return res["samples"][idx]

        if method == "IS":
            self.sampler_info = mcmc.importance_sampling(
                self.log_likelihood, self.prior.sample(n_samples), vectorized=True)
            return self.sampler_info["equal_weight_samples"]

        if method == "SMC":
            kwargs.setdefault("verbose", self.verbose)
            kwargs.setdefault("vectorized", True)
            self.sampler_info = mcmc.SMC(self.log_likelihood, self.prior, n_particles=n_samples, **kwargs)
            return self.sampler_info["equal_weight_samples"]

        raise ValueError(f"Unknown method {method!r}. Choose from 'MCMC', 'NestedSampling', 'IS', 'SMC'.")
