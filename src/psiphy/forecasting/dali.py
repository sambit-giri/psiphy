"""
DALI forecasts: the Fisher expansion extended with higher-order derivatives
of the model mean (Sellentin, Quartin & Amendola 2014, arXiv:1401.6892).
"""
from collections import Counter
from itertools import combinations_with_replacement, permutations, product
from math import factorial

import numpy as np
import scipy.linalg

from .fisher import FisherMatrix, tqdm


def _fd_weights(offsets, k, accuracy=None):
    """
    Finite-difference weights for the k-th derivative on a stencil.

    Solves sum_s w_s s^m = k! delta_mk for m = 0..n-1, so that
    f^(k)(0) ~ sum_s w_s f(s h) / h^k.

    Parameters
    ----------
    offsets : array-like
        Distinct stencil points in units of the step h.
    k : int
        Derivative order.
    accuracy : int or None
        Required truncation order a (error O(h^a)); checked if given.

    Returns
    -------
    np.ndarray
        One weight per offset.

    Raises
    ------
    ValueError
        If the offsets cannot determine the k-th derivative, or not at the
        requested accuracy.
    """
    s = np.asarray(offsets, dtype=float)
    n = len(s)
    if len(np.unique(s)) != n:
        raise ValueError(f"Stencil offsets must be distinct, got {list(offsets)}.")
    if n <= k:
        raise ValueError(f"A derivative of order {k} needs at least {k + 1} stencil points, "
                         f"got {n}: {list(offsets)}.")
    b = np.zeros(n)
    b[k] = factorial(k)
    w = np.linalg.solve(s[None, :] ** np.arange(n)[:, None], b)
    if accuracy is not None:
        # The error is set by the first moment beyond the constrained ones that does not cancel
        j0 = next((j for j in range(n, n + 2)
                   if not np.isclose(w @ s**j, 0, atol=1e-9 * (np.abs(w) @ np.abs(s)**j))), n + 2)
        if j0 - k < accuracy:
            raise ValueError(f"Offsets {list(offsets)} give the order-{k} derivative to O(h^{j0 - k}), "
                             f"not the requested O(h^{accuracy}).")
    return w


def _central_stencil(k, accuracy):
    """(offsets, weights) of the central k-th derivative with O(h^accuracy) error; zero weights dropped."""
    hw = (k + 1) // 2 + accuracy // 2 - 1
    offsets = np.arange(-hw, hw + 1)
    w = _fd_weights(offsets, k, accuracy)
    keep = ~np.isclose(w, 0, atol=1e-12)
    return offsets[keep], w[keep]


def _partial_stencil(idx, n_params, accuracy):
    """
    Stencil of the partial derivative d^k mu / dtheta_idx[0] ... dtheta_idx[k-1],
    as the tensor product of central 1D stencils, one per distinct parameter.

    Returns
    -------
    list of (np.ndarray, float)
        (offset vector in units of the per-parameter steps, weight).
    """
    axes = [(p, *_central_stencil(c, accuracy)) for p, c in Counter(idx).items()]
    points = []
    for combo in product(*[list(zip(offs, ws)) for _, offs, ws in axes]):
        off = np.zeros(n_params)
        w = 1.0
        for (p, _, _), (o, wo) in zip(axes, combo):
            off[p] = o
            w *= wo
        points.append((off, w))
    return points


class FisherMatrixDALI(FisherMatrix):
    """
    Fisher forecast extended with DALI (Sellentin, Quartin & Amendola 2014).

    The Fisher matrix keeps the first derivatives of the mean data vector.
    DALI also keeps its second (doublet, ``order=2``) and third (triplet,
    ``order=3``) derivatives, so the approximate likelihood::

        ln L = -1/2 r^T C^-1 r,
        r = mu_,a D_a + 1/2 mu_,ab D_a D_b + 1/6 mu_,abc D_a D_b D_c,

    with ``D = theta - theta_fid``, can be non-Gaussian in the parameters while
    staying normalisable. The covariance C is parameter-independent, as in
    FisherMatrix.

    Everything of FisherMatrix is inherited (covariance, first derivatives,
    compute, cache, parallel runs); its Gaussian outputs (sigma_1d,
    plot_ellipses, ...) remain the Fisher baseline.

    Parameters
    ----------
    As for FisherMatrix.

    Examples
    --------
    >>> fm = FisherMatrixDALI(my_sim, theta_fid=[0.0, 1.0], param_names=['mu', 'sigma'])
    >>> fm.estimate_covariance(seeds=range(1, 11), noise_seeds=range(10))
    >>> fm.estimate_derivatives(delta_frac=0.01)
    >>> fm.estimate_dali_derivatives(order=2, delta_frac=0.05)
    >>> fm.compute()
    """

    _ARRAY_ATTRS = FisherMatrix._ARRAY_ATTRS + (
        'd2mu_dtheta', 'd3mu_dtheta', 'd2mu_samples_ic', 'd3mu_samples_ic')

    def __init__(self, simulator, theta_fid, param_names, **kwargs):
        super().__init__(simulator, theta_fid, param_names, **kwargs)
        self.d2mu_dtheta = None        # (n_params, n_params, n_data)
        self.d3mu_dtheta = None        # (n_params, n_params, n_params, n_data)
        self.d2mu_samples_ic = None    # per IC seed, (n_seeds, n_params, n_params, n_data)
        self.d3mu_samples_ic = None
        self.dali_seeds = None
        self.dali_snr = None           # derivative order -> SNR of each entry
        self.dali_noise_1sigma = None  # derivative order -> noise of each entry at the Fisher 1-sigma scale
        self.dali_ess = None           # effective sample size of the last importance sampling
        self._dali_cache = None        # (inputs, whitened derivatives) of the last _dali_factors call

    def estimate_dali_derivatives(
        self,
        order=2,
        delta_frac=0.05,
        seeds=range(1, 11),
        noise_seed=0,
        noise_seeds=None,
        accuracy=2,
        snr_warn=3.0,
        noise_tol=0.1,
        n_jobs=1,
        n_chunks=1,
    ):
        """
        Estimate the second (and third) derivatives of the mean data vector
        with central finite differences.

        Each partial derivative uses the tensor product of central 1D stencils,
        e.g. ``(mu_++ - mu_+- - mu_-+ + mu_--) / (4 h_a h_b)`` for ``mu_,ab``
        at ``accuracy=2``. The simulator gets the same (seed, noise_seed) at every
        stencil point, so the realisation largely cancels in the differences.
        Stencil points shared with estimate_derivatives (same steps) and the
        covariance runs are taken from the cache.

        Finite-difference noise grows as 1/h^2 (1/h^3 for the third
        derivatives), so these need larger steps than the first derivatives.
        After the run, the noise of each derivative is estimated from its
        scatter over the IC seeds, and a warning is printed (independent of
        verbose) for entries that are both noise-dominated (SNR < snr_warn)
        and noisy enough to matter: the whitened noise of their term in r at
        the Fisher 1-sigma scale exceeds noise_tol.

        Parameters
        ----------
        order : int
            2 (doublet: ``mu_,ab``) or 3 (triplet: ``mu_,ab`` and ``mu_,abc``).
        delta_frac : float, dict, or array-like
            Fractional step size per parameter, as for estimate_derivatives.
        seeds : iterable
            IC seeds to average over. At least 2 are needed for the noise check.
        noise_seed, noise_seeds :
            As for estimate_derivatives.
        accuracy : int
            Truncation order of the stencils (error O(h^accuracy)); even, >= 2.
        snr_warn : float
            SNR below which an entry counts as noise-dominated. SNR is
            sqrt(m^T C^-1 m / tr(C^-1 Cov(m))) for the mean derivative m, which
            is about 1 for pure noise.
        noise_tol : float
            Noise of an entry's term in r (whitened, at the Fisher 1-sigma
            offsets) above which a noise-dominated entry is reported. The
            1-sigma offsets come from the first derivatives, debiased for their
            noise when estimate_derivatives recorded per-seed samples; noisy
            first derivatives without such samples make sigma, and so this
            check, too small.
        n_jobs, n_chunks :
            As for estimate_derivatives.

        Returns
        -------
        self
        """
        if order not in (2, 3):
            raise ValueError(f"order must be 2 (doublet) or 3 (triplet), got {order!r}.")
        if not (isinstance(accuracy, (int, np.integer)) and accuracy >= 2 and accuracy % 2 == 0):
            raise ValueError(f"accuracy must be an even integer >= 2 (central differences), "
                             f"got {accuracy!r}.")
        assert self.C_inv is not None,      "Call estimate_covariance() first."
        assert self.dmu_dtheta is not None, "Call estimate_derivatives() first."

        seeds = [int(s) for s in seeds]
        if noise_seeds is not None:
            noise_seeds_deriv = [int(ns) for ns in noise_seeds]
        else:
            noise_seeds_deriv = [noise_seed] if noise_seed is not None else list(range(10))
        h = self._step_sizes(delta_frac)
        p = self.n_params

        # All stencils are built (and validated) before any simulation runs
        stencils = {k: {idx: _partial_stencil(idx, p, accuracy)
                        for idx in combinations_with_replacement(range(p), k)}
                    for k in range(2, order + 1)}
        thetas = {}
        for stencil_k in stencils.values():
            for points in stencil_k.values():
                for off, _ in points:
                    theta = self.theta_fid + off * h
                    thetas.setdefault(tuple(np.round(theta, 10)), theta)
        thetas = list(thetas.values())
        n_new = sum(self._cache_key(t, s, ns) not in self._cache
                    for t in thetas for s in seeds for ns in noise_seeds_deriv)

        self._print(f"\n{'='*50}")
        self._print(f"  Computing DALI derivatives (order {order}, O(h^{accuracy}) stencils)")
        self._print(f"  {len(thetas)} stencil points x {len(seeds)} seeds x "
                    f"{len(noise_seeds_deriv)} noise seeds: {n_new} new simulations")
        self._print(f"{'='*50}")

        self._prerun_parallel(thetas, seeds, noise_seeds_deriv, n_jobs,
                              desc="Computing DALI derivatives", n_chunks=n_chunks)

        n_data = self.dmu_dtheta.shape[1]
        self.dali_seeds = seeds
        for k, stencil_k in stencils.items():
            samples = np.zeros((len(seeds),) + (p,) * k + (n_data,))
            for idx, points in stencil_k.items():
                name = "d" + "".join(f"({self.param_names[i]})" for i in idx)
                d = np.zeros((len(seeds), n_data))
                for off, w in points:
                    theta = self.theta_fid + off * h
                    for ki, seed in enumerate(pbar := tqdm(seeds, desc=f"{name}", leave=False,
                                                           disable=not self.verbose)):
                        pbar.set_postfix(seed=seed)
                        x = np.mean([self._normalise(self._simulate_cached(theta, seed, ns))
                                     for ns in noise_seeds_deriv], axis=0)
                        d[ki] += w * x
                d /= np.prod(h[list(idx)])
                for perm in set(permutations(idx)):
                    samples[(slice(None),) + perm] = d
                self._print(f"  {name}: |d^{k}mu| = {np.linalg.norm(d.mean(axis=0)):.4f}")
            setattr(self, f"d{k}mu_samples_ic", samples)
            setattr(self, f"d{k}mu_dtheta", samples.mean(axis=0))
        if order == 2:
            self.d3mu_dtheta = self.d3mu_samples_ic = None

        self._check_dali_noise(snr_warn, noise_tol)
        self._print(f"{'='*50}\n")
        return self

    def _check_dali_noise(self, snr_warn=3.0, noise_tol=0.1):
        """
        SNR and 1-sigma noise of every higher-order derivative from its scatter
        over the IC seeds; stored in self.dali_snr and self.dali_noise_1sigma.
        Prints a warning for entries with SNR < snr_warn and noise > noise_tol.
        """
        self.dali_snr, self.dali_noise_1sigma = None, None
        if len(self.dali_seeds) < 2:
            print("  WARNING: the noise of the DALI derivatives cannot be estimated from "
                  "one IC seed; use at least 2 seeds.")
            return
        # Noise in the first derivatives inflates F and so shrinks sigma; remove
        # that inflation where it can be estimated and leaves F positive definite
        F = self.dmu_dtheta @ self.C_inv @ self.dmu_dtheta.T
        F_debiased = F - self._derivative_noise_correction()
        if np.all(np.linalg.eigvalsh(F_debiased) > 0):
            F = F_debiased
        sigma = np.sqrt(np.diag(np.linalg.inv(F)))
        self.dali_snr, self.dali_noise_1sigma = {}, {}
        flagged = []
        for k in (2, 3):
            samples = getattr(self, f"d{k}mu_samples_ic")
            if samples is None:
                continue
            N = len(samples)
            idxs = list(combinations_with_replacement(range(self.n_params), k))
            S = np.array([samples[(slice(None),) + idx] for idx in idxs])  # (n_idx, N, n_data)
            m = S.mean(axis=1)
            dev = S - m[:, None]
            signal = np.einsum('ia,ab,ib->i', m, self.C_inv, m)
            noise = np.einsum('ina,ab,inb->i', dev, self.C_inv, dev) / (N - 1) / N
            with np.errstate(divide='ignore', invalid='ignore'):
                snr = np.where(noise > 0, np.sqrt(signal / noise), np.inf)
            # term of idx in r: (number of distinct permutations / k!) mu_,idx prod(D)
            coef = np.array([len(set(permutations(idx))) / factorial(k) for idx in idxs])
            noise_1sig = coef * np.sqrt(noise) * np.array([np.prod(sigma[list(idx)]) for idx in idxs])
            self.dali_snr[k] = dict(zip(idxs, snr))
            self.dali_noise_1sigma[k] = dict(zip(idxs, noise_1sig))
            flagged += [(idx, s, n1) for idx, s, n1 in zip(idxs, snr, noise_1sig)
                        if s < snr_warn and n1 > noise_tol]
        if flagged:
            lines = "\n".join(
                "    d" + "".join(f"({self.param_names[i]})" for i in idx)
                + f": SNR = {s:.2f}, noise at 1 sigma = {n1:.3g}"
                for idx, s, n1 in flagged)
            print(f"  WARNING: {len(flagged)} DALI derivative(s) are dominated by simulation noise:\n"
                  f"{lines}\n"
                  f"  Increase delta_frac for these parameters or use more seeds, and check that the "
                  f"simulator gives the same realisation for the same (seed, noise_seed) at every theta.")

    # ---- Combining ----

    @classmethod
    def joint(cls, blocks, verbose=True):
        """
        DALI forecast of the concatenated data vectors of several
        FisherMatrixDALI objects, for correlated blocks; see FisherMatrix.joint.

        The higher-order derivatives and their per-seed samples are
        concatenated block by block, up to the highest order every block has.
        Every block must have run estimate_dali_derivatives on the same IC seeds.

        Parameters
        ----------
        blocks : list of FisherMatrixDALI
        verbose : bool

        Returns
        -------
        FisherMatrixDALI
            Call compute() on it as usual.
        """
        for b in blocks:
            assert getattr(b, 'd2mu_dtheta', None) is not None, \
                "Call estimate_dali_derivatives() on every block first."
        if any(list(b.dali_seeds) != list(blocks[0].dali_seeds) for b in blocks):
            raise ValueError("Blocks used different seeds in estimate_dali_derivatives.")
        fc = super().joint(blocks, verbose=verbose)
        order = 3 if all(b.d3mu_dtheta is not None for b in blocks) else 2
        if order == 2 and any(b.d3mu_dtheta is not None for b in blocks):
            fc._print("  Not every block has third derivatives; the joint forecast stops at order 2.")
        for k in range(2, order + 1):
            for attr in (f'd{k}mu_dtheta', f'd{k}mu_samples_ic'):
                setattr(fc, attr, np.concatenate([getattr(b, attr) for b in blocks], axis=-1))
        fc.dali_seeds = blocks[0].dali_seeds
        return fc

    def __add__(self, other):
        """
        Combine with an independent data set: the log-likelihoods add.

        Returns a FisherMatrixDALI with F = self.F + other.F. Its DALI
        likelihood is the sum of both, through the block-diagonal C^-1 and the
        concatenated derivatives, up to the highest order both have (Fisher
        only if ``other`` is a FisherMatrix without higher derivatives).
        """
        combined = super().__add__(other)
        if any(x is None for x in (self.C_inv, other.C_inv, self.dmu_dtheta, other.dmu_dtheta)):
            return combined
        combined.C_inv = scipy.linalg.block_diag(self.C_inv, other.C_inv)
        combined.dmu_dtheta = np.hstack([self.dmu_dtheta, other.dmu_dtheta])
        for k in (2, 3):
            a, b = getattr(self, f'd{k}mu_dtheta'), getattr(other, f'd{k}mu_dtheta', None)
            if a is None or b is None:
                break
            setattr(combined, f'd{k}mu_dtheta', np.concatenate([a, b], axis=-1))
        return combined

    # ---- DALI likelihood ----

    def _dali_order(self, order):
        """Validate order (1 = Fisher, 2 = doublet, 3 = triplet); None = highest available."""
        available = 3 if self.d3mu_dtheta is not None else 2 if self.d2mu_dtheta is not None else 1
        if order is None:
            return available
        if order not in (1, 2, 3):
            raise ValueError(f"order must be 1 (Fisher), 2 (doublet) or 3 (triplet), got {order!r}.")
        if order > available:
            raise ValueError(f"order={order} needs estimate_dali_derivatives(order={order}) first.")
        return order

    def _dali_factors(self, order):
        """
        Whitened derivatives L^T mu_,a, L^T mu_,ab, L^T mu_,abc (C^-1 = L L^T), as
        arrays of shape (k, p), (k, p, p), (k, p, p, p). When the data vector is
        longer than the number of derivative vectors, they are projected onto
        their span: r lies in it, so r^T C^-1 r is unchanged and k stays small.
        """
        assert self.C_inv is not None,      "Call estimate_covariance() first."
        assert self.dmu_dtheta is not None, "Call estimate_derivatives() first."
        inputs = (order, self.C_inv, self.dmu_dtheta, self.d2mu_dtheta, self.d3mu_dtheta)
        cached = getattr(self, '_dali_cache', None)
        if cached is not None and cached[0][0] == order and all(a is b for a, b in zip(cached[0][1:], inputs[1:])):
            return cached[1]
        L = np.linalg.cholesky(self.C_inv)
        derivs = [self.dmu_dtheta.T, np.moveaxis(self.d2mu_dtheta, -1, 0) if order >= 2 else None,
                  np.moveaxis(self.d3mu_dtheta, -1, 0) if order >= 3 else None][:order]
        white = [np.tensordot(L.T, d, axes=1) for d in derivs]
        M = np.hstack([w.reshape(len(w), -1) for w in white])
        if M.shape[0] > M.shape[1]:
            Q, _ = np.linalg.qr(M)
            white = [np.tensordot(Q.T, w, axes=1) for w in white]
        self._dali_cache = (inputs, white)
        return white

    def dali_logpdf(self, theta, order=None, prior_bounds=None, log_prior=None):
        """
        DALI log-likelihood, ln L = -1/2 r^T C^-1 r (unnormalised).

        Parameters
        ----------
        theta : array-like, shape (n_params,) or (n, n_params)
        order : int or None
            1 (Fisher), 2 (doublet) or 3 (triplet); None uses the highest order
            whose derivatives have been estimated.
        prior_bounds : list of (low, high) or None
            Box prior; -inf outside.
        log_prior : callable or None
            ``log_prior(theta) -> (n,)`` for a batch theta of shape (n, n_params),
            added to the log-likelihood.

        Returns
        -------
        float or np.ndarray, shape (n,)
        """
        order = self._dali_order(order)
        theta = np.asarray(theta, dtype=float)
        single = theta.ndim == 1
        theta = np.atleast_2d(theta)
        D = theta - self.theta_fid
        w = self._dali_factors(order)
        r = D @ w[0].T
        if order >= 2:
            r += 0.5 * np.einsum('kab,na,nb->nk', w[1], D, D, optimize=True)
        if order >= 3:
            r += np.einsum('kabc,na,nb,nc->nk', w[2], D, D, D, optimize=True) / 6
        logp = -0.5 * np.sum(r**2, axis=1)
        if prior_bounds is not None:
            lo, hi = np.asarray(prior_bounds, dtype=float).T
            logp = np.where(np.all((theta >= lo) & (theta <= hi), axis=1), logp, -np.inf)
        if log_prior is not None:
            logp = logp + log_prior(theta)
        return logp[0] if single else logp

    def dali_tensors(self, order=None):
        """
        The DALI tensors T_(i,j) = mu^(i)^T C^-1 mu^(j), mu^(i) being the i-th
        derivatives of the mean, such that

            -ln L = sum_(i,j) T_(i,j) D^(i+j) / (2 i! j!)

        over all pairs (i, j) with i, j <= order. T_(1,1) is the Fisher matrix F,
        T_(2,1) and T_(2,2) are S and Q of Sellentin et al. (2014).

        Returns
        -------
        dict
            (i, j) -> array with i + j parameter axes, for order >= i >= j >= 1.
        """
        w = self._dali_factors(self._dali_order(order))
        return {(i + 1, j + 1): np.tensordot(w[i], w[j], axes=([0], [0]))
                for i in range(len(w)) for j in range(i + 1)}

    # ---- Sampling ----

    def _sample_dali(self, n_samples=100_000, order=None, method='importance', prior_bounds=None,
                     log_prior=None, inflate=2.0, ess_warn=0.05, n_walkers=32, n_burn=1000):
        """(samples, weights) from the DALI posterior; see sample()."""
        def logp(theta):
            return self.dali_logpdf(theta, order=order, prior_bounds=prior_bounds, log_prior=log_prior)

        if method == 'importance':
            from scipy.stats import multivariate_normal
            from ..mcmc import importance_sampling
            assert self.F_inv is not None, "Call compute() first (the Fisher matrix sets the proposal)."
            proposal = multivariate_normal(self.theta_fid, inflate**2 * self.F_inv)
            res = importance_sampling(logp, proposal.rvs(n_samples).reshape(n_samples, -1),
                                      log_proposal_fn=proposal.logpdf, vectorized=True)
            self.dali_ess = res['ess']
            if res['ess'] < ess_warn * n_samples:
                print(f"  WARNING: importance sampling of the DALI posterior has a low effective "
                      f"sample size ({res['ess']:.0f} of {n_samples}). Increase inflate or use "
                      f"method='emcee'.")
            return res['samples'], res['weights']
        if method == 'emcee':
            from ..mcmc import MHSampler
            n_steps = int(np.ceil(n_samples / n_walkers))
            mh = MHSampler(logp, n_walkers=n_walkers, vectorize=True)
            mh.run(self.theta_fid, n_steps=n_burn + n_steps, init_scale=1e-3)
            samples = mh.get_chain(flat=True, discard=n_burn)[:n_samples]
            return samples, np.full(len(samples), 1 / len(samples))
        raise ValueError(f"Unknown method {method!r}. Choose from ['importance', 'emcee'].")

    def sample(self, n_samples=100_000, approx='dali', **kwargs):
        """
        Draw samples from the DALI (or Fisher) posterior.

        Parameters
        ----------
        n_samples : int
        approx : str
            'dali' (default) or 'fisher' (Gaussian, as FisherMatrix.sample).
        order : int or None
            DALI order; None uses the highest order estimated.
        method : str
            'importance' (default): importance sampling from the Fisher Gaussian
            with covariance inflated by ``inflate**2``, then resampled to equal
            weights; prints a warning when the effective sample size is below
            ``ess_warn * n_samples``. 'emcee': ensemble MCMC with ``n_walkers``
            walkers and ``n_burn`` burn-in steps.
        prior_bounds, log_prior :
            As for dali_logpdf.

        Returns
        -------
        np.ndarray, shape (n_samples, n_params)
        """
        if approx == 'fisher':
            return super().sample(n_samples)
        if approx != 'dali':
            raise ValueError(f"approx must be 'dali' or 'fisher', got {approx!r}.")
        samples, weights = self._sample_dali(n_samples, **kwargs)
        idx = np.random.choice(len(samples), size=n_samples, p=weights / weights.sum())
        return samples[idx]

    def to_getdist(self, n_samples=100_000, label=None, label_names=None, approx='dali', **kwargs):
        """
        getdist MCSamples of the DALI (or Fisher) posterior.

        Importance samples are passed with their weights. Keyword arguments
        are those of sample(); ``prior_bounds`` also sets the getdist ranges.

        Returns
        -------
        getdist.MCSamples
        """
        from getdist import MCSamples
        if label_names is None:
            label_names = {p: p for p in self.param_names}
        if approx == 'fisher':
            # FisherMatrix.to_getdist would call self.sample, which defaults to DALI here
            samples, weights = FisherMatrix.sample(self, n_samples), None
        elif approx == 'dali':
            samples, weights = self._sample_dali(n_samples, **kwargs)
        else:
            raise ValueError(f"approx must be 'dali' or 'fisher', got {approx!r}.")
        bounds = kwargs.get('prior_bounds')
        return MCSamples(
            samples=samples,
            weights=weights,
            names=self.param_names,
            labels=[label_names[p] for p in self.param_names],
            label=label,
            ranges=None if bounds is None else dict(zip(self.param_names, bounds)),
        )
