"""
Bayesian-optimisation helpers built on scikit-learn Gaussian processes.

Acquisition functions return values to **maximise**. Each takes the candidate
points, a fitted ``GaussianProcessRegressor``, the incumbent ``y_best`` and a
``minimize`` flag stating whether the underlying objective is minimised
(e.g. a BOLFI discrepancy) or maximised (e.g. an LFIRE posterior).

Examples
--------
>>> gpr = GaussianProcessRegressor(Matern(nu=2.5), normalize_y=True).fit(X, y)
>>> x_next = propose_location(gpr, bounds, acquisition="EI", X_sample=X, y_sample=y)
>>> batch = propose_location(gpr, bounds, acquisition="LCB", X_sample=X, y_sample=y, batch_size=4)
"""

import numpy as np
from scipy.optimize import minimize as _scipy_minimize
from scipy.stats import norm

from . import helpers as hf


# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------

def _improvement(X, gpr, y_best, xi, minimize):
    mu, sd = gpr.predict(np.atleast_2d(X), return_std=True)
    imp = (y_best - mu - xi) if minimize else (mu - y_best - xi)
    return imp, np.maximum(sd, 1e-12)


def expected_improvement(X, gpr, y_best, minimize=True, xi=0.01):
    """Expected improvement over *y_best*."""
    imp, sd = _improvement(X, gpr, y_best, xi, minimize)
    z = imp / sd
    return imp * norm.cdf(z) + sd * norm.pdf(z)


def probability_of_improvement(X, gpr, y_best, minimize=True, xi=0.01):
    """Probability of improving on *y_best* by at least *xi*."""
    imp, sd = _improvement(X, gpr, y_best, xi, minimize)
    return norm.cdf(imp / sd)


def confidence_bound(X, gpr, y_best=None, minimize=True, kappa=1.96):
    """
    GP confidence bound: ``-(mu - kappa*sd)`` (LCB) when minimising,
    ``mu + kappa*sd`` (UCB) when maximising. Larger *kappa* explores more.
    """
    mu, sd = gpr.predict(np.atleast_2d(X), return_std=True)
    return -(mu - kappa * sd) if minimize else mu + kappa * sd


ACQUISITIONS = {
    "EI": expected_improvement,
    "PI": probability_of_improvement,
    "LCB": confidence_bound,
    "UCB": confidence_bound,
}


def get_acquisition(acquisition):
    """Look up an acquisition by name, or pass a callable through."""
    if callable(acquisition):
        return acquisition
    try:
        return ACQUISITIONS[acquisition]
    except KeyError:
        raise ValueError(f"Unknown acquisition {acquisition!r}. "
                         f"Choose from {list(ACQUISITIONS)} or pass a callable.") from None


# ---------------------------------------------------------------------------
# Proposal
# ---------------------------------------------------------------------------

def _maximise(acq, bounds, n_candidates, n_restarts, rng):
    """Random-candidate search, then L-BFGS-B polish of the best few."""
    dim = len(bounds)
    cand = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_candidates, dim))
    vals = acq(cand)
    best_x, best_v = cand[np.argmax(vals)], np.max(vals)
    for x0 in cand[np.argsort(vals)[::-1][:n_restarts]]:
        res = _scipy_minimize(lambda x: -float(acq(x[None, :])[0]), x0, bounds=bounds, method="L-BFGS-B")
        if res.success and -res.fun > best_v:
            best_x, best_v = res.x, -res.fun
    return best_x


def _fantasy_gp(gpr, X, y):
    """Refit *gpr* with its learned hyperparameters frozen (cheap)."""
    from sklearn.base import clone
    g = clone(gpr).set_params(kernel=gpr.kernel_, optimizer=None)
    return g.fit(X, y)


def propose_location(gpr, bounds, acquisition="EI", X_sample=None, y_sample=None,
                     batch_size=1, minimize=True, n_candidates=None, n_restarts=5,
                     random_state=None, **acq_kwargs):
    """
    Propose the next point(s) to evaluate.

    Parameters
    ----------
    gpr : fitted sklearn ``GaussianProcessRegressor``
    bounds : array-like, shape (d, 2)
        Search box, in the same coordinates the GP was fitted in.
    acquisition : {"EI", "PI", "LCB", "UCB"} or callable
        A callable must have signature ``acq(X, gpr, y_best, minimize, **acq_kwargs)``
        and return values to maximise.
    X_sample, y_sample : array-like
        Evaluated points. Needed for EI/PI (incumbent) and for ``batch_size > 1``.
    batch_size : int
        Number of points to return. Points after the first are chosen by the
        kriging-believer heuristic: each chosen point is added to the GP with
        its predicted mean as a fantasy observation, which shrinks the
        uncertainty there and pushes the next pick elsewhere. Use this to run
        several simulations in parallel per round.
    minimize : bool
        Whether the objective modelled by the GP is minimised.
    n_candidates : int, optional
        Random candidates scored per pick (default ``1000 * d``).
    n_restarts : int
        Best candidates polished with L-BFGS-B.
    random_state : int or numpy Generator, optional
    **acq_kwargs :
        Passed to the acquisition (``xi`` for EI/PI, ``kappa`` for LCB/UCB).

    Returns
    -------
    np.ndarray, shape (batch_size, d)
    """
    acq_fn = get_acquisition(acquisition)
    bounds = np.asarray(bounds, dtype=float)
    rng = np.random.default_rng(random_state)
    n_candidates = n_candidates or 1000 * len(bounds)
    if batch_size > 1 and (X_sample is None or y_sample is None):
        raise ValueError("batch_size > 1 requires X_sample and y_sample.")

    X = None if X_sample is None else np.atleast_2d(np.asarray(X_sample, dtype=float))
    y = None if y_sample is None else np.asarray(y_sample, dtype=float).ravel()
    model = gpr
    picks = []
    for _ in range(batch_size):
        # incumbent = best GP mean at evaluated points (robust to noisy observations)
        y_best = None
        if X is not None:
            mu_seen = model.predict(X)
            y_best = mu_seen.min() if minimize else mu_seen.max()
        elif acquisition in ("EI", "PI"):
            raise ValueError(f"acquisition={acquisition!r} requires X_sample.")
        x = _maximise(lambda Z: acq_fn(Z, model, y_best, minimize=minimize, **acq_kwargs),
                      bounds, n_candidates, n_restarts, rng)
        picks.append(x)
        if len(picks) < batch_size:
            X = np.vstack([X, x])
            y = np.append(y, model.predict(x[None, :]))
            model = _fantasy_gp(gpr, X, y)
    return np.array(picks)


# ---------------------------------------------------------------------------
# n-sphere geometry helpers
# ---------------------------------------------------------------------------

def check_inside_nsphere(X, bound_min, bound_max):
    check = np.sum(((X-bound_min)/(bound_max-bound_min)-0.5)**2, axis=1) if X.ndim>1 else np.sum(((X-bound_min)/(bound_max-bound_min)-0.5)**2)
    return check<0.25

def draw_nsphere_surface(bound_min, bound_max, N=100):
    dims = len(bound_min)
    ri = 0.5*np.ones((N))
    for i in range(1,dims-1): ri = np.vstack((ri,np.linspace(0,np.pi,N)))
    ri = np.vstack((ri,np.linspace(0,2*np.pi,N)))
    ri = ri.T
    xi = np.array([hf.spherical_to_cartesian(thet) for thet in ri])
    yi = (xi+0.5)*(bound_max-bound_min)+bound_min
    return yi
