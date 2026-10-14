"""
LFIRE — Likelihood-Free Inference by Ratio Estimation (Thomas et al. 2022,
Bayesian Analysis 17, 1).

For a parameter value theta, simulated data from ``p(x | theta)`` and from the
marginal ``p(x)`` (prior predictive) are separated with a logistic-regression
classifier. Its log-odds estimate the log density ratio

    log r(x, theta) = log p(x | theta) - log p(x),

so ``p(theta | x_obs) ∝ prior(theta) * r(x_obs, theta)``. Each evaluation costs
``n_theta`` simulations plus one classifier fit, so LFIRE either evaluates a grid
directly (low dimensions) or learns a GP surrogate of the log ratio with Bayesian
optimisation and samples that, as in BOLFI.

Examples
--------
>>> lfire = LFIRE(simulator, x_obs, {'mu': (0, 3), 'sigma': (0.1, 2.5)}, summary=my_summaries)
>>> lfire.learn_posterior(n_calls=60)
>>> samples = lfire.sample_posterior(5000)
"""

import inspect

import numpy as np

from ._surrogate import SurrogateInference


def _logistic_cv(penalty, n_jobs, random_state):
    """Cross-validated logistic regression on standardised features."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    kw = dict(Cs=5, cv=3, n_jobs=n_jobs, random_state=random_state)
    if penalty == "l1":
        # perfectly separable classes (theta far from the data) need more than liblinear's default 100
        kw.update(solver="liblinear", max_iter=2000)
    elif penalty != "l2":
        raise ValueError(f"penalty must be 'l1' or 'l2', got {penalty!r}.")
    if "use_legacy_attributes" in inspect.signature(LogisticRegressionCV).parameters:
        # scikit-learn >= 1.8: penalty is expressed through l1_ratios
        kw.update(l1_ratios=(1.0 if penalty == "l1" else 0.0,), use_legacy_attributes=False)
    else:
        kw["penalty"] = penalty
    return make_pipeline(StandardScaler(), LogisticRegressionCV(**kw))


class LFIRE(SurrogateInference):
    """
    Likelihood-free inference by ratio estimation.

    Parameters
    ----------
    simulator : callable
        ``simulator(theta) -> x`` (array-like).
    observation : array-like
        Observed data, in the same format as the simulator output.
    prior : list of (low, high), dict name -> (low, high), or prior object
    n_m : int
        Simulations from the marginal (prior predictive), drawn once and reused.
    n_theta : int
        Simulations at each parameter value evaluated.
    summary : callable, optional
        ``summary(x) -> 1-D features`` used by the classifier. Defaults to the
        flattened data. The log ratio is linear in these features, so include
        nonlinear terms (e.g. squares) when the data distribution changes shape
        with theta.
    penalty : {"l1", "l2"}
        Regularisation of the default classifier (cross-validated). ``"l1"``
        follows the paper and selects among many candidate summaries;
        ``"l2"`` is several times faster when there are few summaries.
    classifier : sklearn classifier, optional
        Replaces the default; must provide ``decision_function`` (log-odds).
        It is cloned for every evaluation.
    acquisition : {"UCB", "EI", "PI"} or callable
        Acquisition for :meth:`learn_posterior` (maximising the log ratio).
    clip_log_ratio : float or None
        The GP surrogate is fitted to ``max(log_r, max(log_r) - clip_log_ratio)``.
        Far from the posterior the classes separate perfectly and the log
        ratio becomes hugely negative (limited only by regularisation); without
        the floor those values dominate the GP and widen the posterior. Points
        below the floor carry less than ``exp(-clip_log_ratio)`` of the peak
        posterior density. ``None`` disables it.
    bounds : list of (low, high), optional
        Search box; defaults to the prior bounds.
    param_names : list of str, optional
    n_jobs : int
        Workers for the classifier's cross-validation.
    random_state : int, optional
        Seed for the classifier.
    verbose : bool
    """

    _minimize = False
    _config_keys = ("n_theta",)
    _saved_arrays = ("X_marginal",)

    def __init__(self, simulator, observation, prior, n_m=500, n_theta=500, summary=None,
                 penalty="l1", classifier=None, acquisition="UCB", clip_log_ratio=20.0,
                 bounds=None, param_names=None, n_jobs=1, random_state=None, verbose=True):
        super().__init__(prior, acquisition, bounds=bounds, param_names=param_names, verbose=verbose)
        if classifier is None:
            _logistic_cv(penalty, n_jobs, random_state)  # validate penalty early
        self.simulator = simulator
        self.summary = summary if summary is not None else np.ravel
        self.x_obs = self._features(observation)
        self.n_m = int(n_m)
        self.n_theta = int(n_theta)
        self.penalty = penalty
        self.classifier = classifier
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.clip_log_ratio = clip_log_ratio
        self.X_marginal = None
        self.thetas = None
        self.posterior = None

    # ---- ratio estimation ----

    def _features(self, x):
        return np.ravel(np.asarray(self.summary(x), dtype=float))

    def _simulate(self, thetas):
        return np.array([self._features(self.simulator(t)) for t in thetas])

    def simulate_marginal(self):
        """Simulate the marginal (prior-predictive) set once; reused by every evaluation."""
        if self.X_marginal is None:
            self._print(f"Simulating {self.n_m} marginal data sets.")
            self.X_marginal = self._simulate(self.prior.sample(self.n_m))
        return self.X_marginal

    def _make_classifier(self):
        if self.classifier is not None:
            from sklearn.base import clone
            return clone(self.classifier)
        return _logistic_cv(self.penalty, self.n_jobs, self.random_state)

    def log_ratio(self, theta, return_score=False):
        """
        Estimate ``log r(x_obs, theta) = log p(x_obs | theta) - log p(x_obs)``.

        Parameters
        ----------
        theta : array-like, shape (n_dim,)
        return_score : bool
            Also return the classifier's training accuracy. Values near 0.5
            mean data at *theta* are indistinguishable from the marginal.

        Returns
        -------
        float, or (float, float) with *return_score*
        """
        theta = np.asarray(theta, dtype=float)
        X_m = self.simulate_marginal()
        X_t = self._simulate(np.repeat(theta[None, :], self.n_theta, axis=0))
        X = np.vstack([X_t, X_m])
        y = np.r_[np.ones(len(X_t)), np.zeros(len(X_m))]
        clf = self._make_classifier().fit(X, y)
        # log-odds -> density ratio, correcting for the class sizes
        log_r = float(clf.decision_function(self.x_obs[None, :])[0]) + np.log(len(X_m) / len(X_t))
        if return_score:
            return log_r, float(clf.score(X, y))
        return log_r

    def _objective(self, theta):
        return self.log_ratio(theta)

    # ---- direct grid evaluation ----

    def _grid(self, n_grid):
        axes = [np.linspace(lo, hi, n_grid) for lo, hi in self.bounds]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, self.n_dim)

    def _store_grid(self, thetas, log_post):
        log_post = np.asarray(log_post, dtype=float)
        p = np.exp(log_post - np.max(log_post))
        self.thetas = thetas[:, 0] if self.n_dim == 1 else thetas
        self.posterior = p / p.sum()
        return self.thetas, self.posterior

    def run_grid(self, n_grid=20, thetas=None):
        """
        Evaluate the log ratio directly at every grid point (no surrogate).

        Costs ``n_theta`` simulations per point, so this suits one or two
        parameters. Results: ``self.thetas``, ``self.log_ratios`` and the
        normalised grid posterior ``self.posterior`` (also used by
        :func:`psiphy.plotting.corner.plot_lfire`).

        Returns
        -------
        thetas, posterior
        """
        thetas = self._grid(n_grid) if thetas is None else np.atleast_2d(np.asarray(thetas, dtype=float))
        if thetas.shape[1] != self.n_dim:
            thetas = thetas.reshape(-1, self.n_dim)
        self.log_ratios = np.array([self.log_ratio(t) for t in thetas])
        log_prior = np.atleast_1d(self.prior.log_prob(thetas))
        return self._store_grid(thetas, self.log_ratios + log_prior)

    # ---- GP surrogate ----

    def _fit_gp(self, random_state=None):
        from ._surrogate import fit_gp
        y = self.y if self.clip_log_ratio is None else np.maximum(self.y, self.y.max() - self.clip_log_ratio)
        self.gp = fit_gp(self._to_unit(self.X), y, random_state)
        return self.gp

    def log_likelihood(self, theta):
        """Surrogate log ratio (the log-likelihood up to a constant)."""
        return self._predict(theta)

    def learn_posterior(self, n_calls=60, n_initial_points=None, batch_size=1, filename=None,
                        reset_model=False, random_state=None, n_candidates=None, **acq_kwargs):
        """
        Learn a GP surrogate of the log ratio with Bayesian optimisation.

        Evaluations concentrate where the log ratio is high, i.e. in the
        posterior bulk. Afterwards use :meth:`sample_posterior` or
        :meth:`posterior_grid`.

        Parameters
        ----------
        n_calls : int
            New log-ratio evaluations (each costs ``n_theta`` simulations).
        n_initial_points : int, optional
            Latin-hypercube initial design size; default ``5 * n_dim``.
        batch_size : int
            Points proposed per round (kriging believer).
        filename : str, optional
            Pickle file to resume from (unless *reset_model*) and save to. It
            also stores the marginal simulations.
        reset_model : bool
        random_state : int, optional
        n_candidates : int, optional
        **acq_kwargs :
            e.g. ``kappa`` for UCB (default 1.96), ``xi`` for EI/PI.

        Returns
        -------
        self
        """
        self._print(f"Learning the log ratio with Bayesian optimisation ({self.acquisition})...")
        self._learn(n_calls, n_initial_points, batch_size, filename, reset_model,
                    random_state, n_candidates, **acq_kwargs)
        self._print("...done")
        return self

    def posterior_grid(self, n_grid=50):
        """
        Normalised posterior of the surrogate on a grid.

        Sets ``self.thetas`` and ``self.posterior`` (for
        :func:`psiphy.plotting.corner.plot_lfire`).

        Returns
        -------
        thetas, posterior
        """
        thetas = self._grid(n_grid)
        return self._store_grid(thetas, self.log_posterior(thetas))
