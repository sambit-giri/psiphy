import os
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from itertools import combinations

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:
        """Minimal stand-in when tqdm is not installed (iteration and set_postfix only)."""
        def __init__(self, iterable, **kwargs):
            self._it = iterable
        def __iter__(self):
            return iter(self._it)
        def set_postfix(self, *args, **kwargs):
            pass

try:
    from joblib import Parallel, delayed as _jdelayed
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False


class FisherMatrix:
    """
    Numerical Fisher matrix forecast.

    F_ij = (dmu/dtheta_i)^T C^{-1} (dmu/dtheta_j)

    Parameters
    ----------
    simulator : callable
        Function with signature
        ``simulator(theta, initial_seed, noise_seed, **sim_kwargs) -> np.ndarray``.
        Must return a 1D array (the summary statistic).
    theta_fid : array-like, shape (n_params,)
        Fiducial parameter values.
    param_names : list of str
        Names for each parameter.
    X_mean : float or np.ndarray
        Mean for normalisation (0.0 = no normalisation).
    X_std : float or np.ndarray
        Std for normalisation (1.0 = no normalisation).
    verbose : bool
        Print progress messages.
    cache_file : str or None
        Path to pickle file for caching simulation outputs across runs.
    **sim_kwargs :
        Additional keyword arguments forwarded to the simulator.

    Examples
    --------
    >>> fm = FisherMatrix(my_sim, theta_fid=[0.0, 1.0], param_names=['mu', 'sigma'])
    >>> fm.estimate_covariance(seeds=range(1, 11), noise_seeds=range(10))
    >>> fm.estimate_derivatives(delta_frac=0.01)
    >>> fm.compute()
    >>> fm.plot_ellipses()
    """

    # Arrays written by save() and restored by load(); subclasses extend this.
    _ARRAY_ATTRS = ('F', 'F_inv', 'F_std', 'C', 'C_inv', 'dmu_dtheta',
                    'dmu_samples_ns', 'dmu_samples_ic',
                    'x_mean_fid', 'x_realisations')

    def __init__(
        self,
        simulator,
        theta_fid,
        param_names,
        X_mean=0.0,
        X_std=1.0,
        verbose=True,
        cache_file=None,
        **sim_kwargs,
    ):
        self.simulator = simulator
        self.theta_fid = np.asarray(theta_fid, dtype=float)
        self.param_names = list(param_names)
        self.n_params = len(param_names)
        self.X_mean = np.asarray(X_mean, dtype=float)
        self.X_std = np.asarray(X_std, dtype=float)
        self.verbose = verbose
        self.cache_file = cache_file
        self.sim_kwargs = sim_kwargs

        self._cache = {}
        if cache_file is not None and os.path.exists(cache_file):
            self._load_cache()

        self.x_realisations = None
        self.cov_keys = []             # (seed, noise_seed) of each row of x_realisations
        self.deriv_seeds = None        # IC seeds of dmu_samples_ic
        self.deriv_noise_seeds = None  # noise seeds the derivatives were averaged over
        self.x_mean_fid = None
        self.C = None
        self.C_inv = None
        self.hartlap = None
        self.dmu_dtheta = None
        self.dmu_samples_ns = None
        self.dmu_samples_ic = None
        self.F = None
        self.F_inv = None
        self.F_std = None
        self.F_upper = None            # CoultonWandelt2023: standard Fisher (biased high)
        self.F_lower = None            # CoultonWandelt2023: compressed Fisher (biased low)
        self.compute_method = None

    # ---- Parallel worker helpers ----

    @staticmethod
    def _run_theta_seed_jobs(theta_step, seed, noise_seeds, simulator, sim_kwargs):
        """Worker: run all noise_seeds for one (theta_step, IC seed) pair."""
        return [(seed, ns, np.asarray(simulator(theta_step, seed, ns, **sim_kwargs)))
                for ns in noise_seeds]

    def _prerun_parallel(self, thetas, seeds, noise_seeds, n_jobs, desc, n_chunks=1):
        """
        Run every uncached (theta, seed, noise_seed) simulation with joblib and
        store the results in the cache. No-op when ``n_jobs == 1`` or joblib is
        missing; callers then simulate serially through ``_simulate_cached``.

        One job per (theta, IC seed) pair runs all its noise_seeds, so the
        simulator's in-memory coeval cache is reused within each job. With
        ``n_chunks > 1`` the seeds are run in chunks, the cache is written after
        each chunk (an interruption loses at most one chunk) and a progress bar
        advances per finished chunk.

        Returns
        -------
        int
            Number of (theta, seed) jobs run.
        """
        if n_jobs == 1 or not _JOBLIB_AVAILABLE:
            return 0
        if n_chunks > 1:
            chunks = [c for c in np.array_split(np.asarray(seeds), n_chunks) if len(c)]
            return sum(self._prerun_parallel(thetas, [int(s) for s in c], noise_seeds, n_jobs, desc)
                       for c in tqdm(chunks, desc=f"{desc} (seed chunks)"))
        jobs = {}
        for theta in thetas:
            for seed in seeds:
                key = (tuple(np.round(theta, 10)), seed)
                if key not in jobs and any(
                    self._cache_key(theta, seed, ns) not in self._cache for ns in noise_seeds
                ):
                    jobs[key] = (np.array(theta, dtype=float), seed)
        self._print(f"  Parallel mode (n_jobs={n_jobs}): {len(jobs)} (theta, seed) jobs to run")
        if not jobs:
            return 0
        all_results = Parallel(n_jobs=n_jobs)(
            _jdelayed(FisherMatrix._run_theta_seed_jobs)(t, s, noise_seeds, self.simulator, self.sim_kwargs)
            for t, s in tqdm(jobs.values(), desc=desc, disable=not self.verbose)
        )
        for (t, _), triplets in zip(jobs.values(), all_results):
            for seed, ns, x in triplets:
                self._cache.setdefault(self._cache_key(t, seed, ns), x)
        if self.cache_file is not None:
            self._save_cache()
        return len(jobs)

    # ---- Internals ----

    def _normalise(self, x):
        return (x - self.X_mean) / self.X_std

    def _step_sizes(self, delta_frac):
        """
        Absolute finite-difference step per parameter. delta_frac is a scalar,
        a dict keyed by parameter name or an array in param order, relative to
        the fiducial value (absolute where the fiducial value is zero).
        """
        if isinstance(delta_frac, dict):
            df_arr = np.array([float(delta_frac[p]) for p in self.param_names])
        elif np.isscalar(delta_frac):
            df_arr = np.full(self.n_params, float(delta_frac))
        else:
            df_arr = np.asarray(delta_frac, dtype=float)
            assert df_arr.shape == (self.n_params,), \
                f"delta_frac array must have length {self.n_params}"
        return np.where(self.theta_fid != 0, df_arr * np.abs(self.theta_fid), df_arr)

    def _print(self, msg=""):
        if self.verbose:
            print(msg)

    def _cache_key(self, theta, seed, noise_seed):
        return (tuple(np.round(theta, 10)), int(seed), int(noise_seed))

    def _load_cache(self):
        import pickle
        try:
            with open(self.cache_file, 'rb') as f:
                self._cache = pickle.load(f)
            self._print(f"  Cache loaded: {len(self._cache)} simulations from {self.cache_file}")
        except Exception as e:
            self._print(f"  WARNING: Could not load cache: {e}")
            self._cache = {}

    def _save_cache(self):
        if self.cache_file is None:
            return
        import pickle, tempfile
        tmp_dir = os.path.dirname(self.cache_file) or '.'
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', dir=tmp_dir, delete=False, suffix='.tmp') as tmp:
                pickle.dump(self._cache, tmp)
                tmp_path = tmp.name
            os.replace(tmp_path, self.cache_file)
        except Exception as e:
            self._print(f"  WARNING: Could not save cache: {e}")
            if tmp_path is not None and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _simulate_cached(self, theta, seed, noise_seed):
        key = self._cache_key(theta, seed, noise_seed)
        if key not in self._cache:
            x = self.simulator(theta, seed, noise_seed, **self.sim_kwargs)
            self._cache[key] = np.asarray(x)
            if self.cache_file is not None:
                self._save_cache()
        return self._cache[key]

    # ---- Cache utilities ----

    def run_simulations(self, theta=None, seeds=range(1, 11), noise_seeds=range(10), save=True,
                        n_jobs=1):
        """
        Pre-run and cache simulations.

        Parameters
        ----------
        theta : array-like or None
            Parameter values. Defaults to theta_fid.
        seeds : iterable
            Initial condition seeds.
        noise_seeds : iterable
            Noise seeds.
        save : bool
            Save cache to file after running.
        n_jobs : int
            Number of parallel workers (joblib). -1 uses all CPUs.

        Returns
        -------
        self
        """
        if theta is None:
            theta = self.theta_fid
        theta = np.asarray(theta, dtype=float)
        seeds = [int(s) for s in seeds]
        noise_seeds = [int(s) for s in noise_seeds]

        n_total = len(seeds) * len(noise_seeds)
        n_cached = n_new = 0

        self._print(f"\n{'='*50}")
        self._print(f"  Running simulations — {n_total} total")
        self._print(f"{'='*50}")

        n_before = len(self._cache)
        self._prerun_parallel([theta], seeds, noise_seeds, n_jobs, desc="Simulating")
        n_parallel = len(self._cache) - n_before

        theta_str = "[" + ", ".join(f"{v:.3g}" for v in theta) + "]"
        for seed in (pbar := tqdm(seeds, desc="Simulating", disable=not self.verbose)):
            for ns in noise_seeds:
                pbar.set_postfix(seed=seed, noise=ns, theta=theta_str)
                key = self._cache_key(theta, seed, ns)
                if key in self._cache:
                    n_cached += 1
                else:
                    self._simulate_cached(theta, seed, ns)
                    n_new += 1

        # simulations just run in parallel were counted as cached by the loop above
        n_new, n_cached = n_new + n_parallel, n_cached - n_parallel
        self._print(f"  New: {n_new}, From cache: {n_cached}")
        if save and n_new > 0:
            self._save_cache()
        self._print(f"{'='*50}\n")
        return self

    def cache_info(self):
        """Print a summary of cached simulations grouped by theta."""
        self._print(f"\n{'='*50}")
        self._print(f"  Cache info")
        self._print(f"{'='*50}")
        self._print(f"  Total cached simulations: {len(self._cache)}")
        if self.cache_file:
            self._print(f"  Cache file: {self.cache_file}")
        theta_groups = {}
        for (theta_tuple, seed, ns) in self._cache.keys():
            theta_groups.setdefault(theta_tuple, []).append((seed, ns))
        for theta_tuple, entries in theta_groups.items():
            seeds_set = {s for s, _ in entries}
            noise_set = {n for _, n in entries}
            theta_str = "[" + ", ".join(f"{v:.3g}" for v in theta_tuple) + "]"
            self._print(
                f"  theta={theta_str}: {len(entries)} sims, "
                f"{len(seeds_set)} seeds, {len(noise_set)} noise seeds"
            )
        self._print(f"{'='*50}\n")

    def test_simulator(self, seed=1, noise_seed=0):
        """
        Run the simulator once at fiducial to verify it works.

        Returns
        -------
        np.ndarray
            Normalised output at fiducial parameters.
        """
        self._print(f"\n{'='*50}")
        self._print(f"  Testing simulator at fiducial: {dict(zip(self.param_names, self.theta_fid))}")
        x = self._simulate_cached(self.theta_fid, seed, noise_seed)
        x_norm = self._normalise(x)
        self._print(f"  Output shape: {x.shape}, raw range: [{x.min():.4g}, {x.max():.4g}]")
        self._print(f"{'='*50}\n")
        return x_norm

    # ---- Covariance ----

    def estimate_covariance(self, seeds=range(1, 11), noise_seeds=range(10), n_jobs=1, n_chunks=1):
        """
        Estimate the data covariance matrix at the fiducial point.

        Realisations are appended to any existing ones, so you can call this
        multiple times with different seed ranges to increase the sample size.

        Parameters
        ----------
        seeds : iterable
            Initial condition / cosmic variance seeds.
        noise_seeds : iterable
            Instrument noise seeds.
        n_jobs : int
            Number of parallel workers (joblib). -1 uses all CPUs.
            Each worker handles all noise_seeds for one IC seed, so the
            simulator's in-memory coeval cache is reused within each job.
        n_chunks : int
            (parallel only) Run the seeds in this many chunks, writing the cache
            after each, with a progress bar per finished chunk.

        Returns
        -------
        self
        """
        seeds = [int(s) for s in seeds]
        noise_seeds = [int(s) for s in noise_seeds]

        new_realisations = []
        self._print(f"\n{'='*50}")
        self._print(f"  Estimating covariance matrix")
        self._print(f"{'='*50}")

        self._prerun_parallel([self.theta_fid], seeds, noise_seeds, n_jobs,
                              desc="Simulating realisations", n_chunks=n_chunks)

        fid_str = "[" + ", ".join(f"{v:.3g}" for v in self.theta_fid) + "]"
        for seed in (pbar := tqdm(seeds, desc="Simulating realisations", disable=not self.verbose)):
            for ns in noise_seeds:
                pbar.set_postfix(seed=seed, noise=ns, theta=fid_str)
                x = self._simulate_cached(self.theta_fid, seed, ns)
                new_realisations.append(self._normalise(x))

        new_realisations = np.array(new_realisations)
        self.cov_keys += [(seed, ns) for seed in seeds for ns in noise_seeds]

        if self.x_realisations is not None:
            self.x_realisations = np.vstack([self.x_realisations, new_realisations])
        else:
            self.x_realisations = new_realisations

        self._set_covariance()
        return self

    def _set_covariance(self):
        """Mean, covariance and Hartlap-corrected inverse from self.x_realisations."""
        n_real = len(self.x_realisations)
        n_data = self.x_realisations.shape[1]

        self.x_mean_fid = self.x_realisations.mean(axis=0)
        self.C = np.cov(self.x_realisations, rowvar=False)

        if n_real > n_data + 2:
            self.hartlap = (n_real - n_data - 2) / (n_real - 1)
        else:
            self.hartlap = 1.0
            self._print(f"  WARNING: n_real={n_real} <= n_data+2={n_data+2}; "
                        "covariance may be singular — add more realisations.")

        self.C_inv = self.hartlap * np.linalg.inv(self.C)

        self._print(f"  Total realisations: {n_real}, data points: {n_data}")
        self._print(f"  Hartlap factor: {self.hartlap:.3f}")
        self._print(f"  Condition number: {np.linalg.cond(self.C):.2e}")
        self._print(f"{'='*50}\n")

    def add_modelling_error(self, mod_err=0.1):
        """
        Add a fractional modelling error to the covariance.

        Appends C_mod = diag(mod_err * mu_fid)^2 to the existing covariance
        and recomputes C_inv with the Hartlap correction.

        Parameters
        ----------
        mod_err : float or np.ndarray
            Fractional error per data bin.

        Returns
        -------
        self
        """
        assert self.C is not None, "Call estimate_covariance() first."
        C_mod = np.diag((mod_err * self.x_mean_fid) ** 2)
        self.C += C_mod
        self.C_inv = self.hartlap * np.linalg.inv(self.C)
        self._print(f"  Modelling error added (frac={mod_err}), "
                    f"new condition number: {np.linalg.cond(self.C):.2e}")
        return self

    # ---- Derivatives ----

    def estimate_derivatives(
        self,
        delta_frac=0.01,
        seeds=range(1, 11),
        noise_seed=0,
        noise_seeds=None,
        n_stencil=2,
        use_crn=False,
        n_jobs=1,
        n_chunks=1,
    ):
        """
        Estimate dmu/dtheta via central finite differences.

        Parameters
        ----------
        delta_frac : float, dict, or array-like
            Fractional step size h relative to the fiducial value. A dict
            keyed by parameter name (or an array in param order) sets a
            per-parameter step, so weakly responding parameters can use
            larger steps to beat down stochastic derivative noise (~1/h)
            without inflating truncation error for the others.
        seeds : iterable
            Seeds to average over for stable derivatives.
        noise_seed : int or None
            Noise seed for derivatives (0 = noiseless; None = average over 10).
            Ignored if `noise_seeds` is given.
        noise_seeds : iterable or None
            Explicit list of noise seeds to average the derivatives over.
            Use this for summaries whose expectation depends on the noise level
            (non-additive statistics): with a seeded simulator, each noise seed
            gives the identical noise realisation at every stencil point, so the
            noise cancels in the differences while its suppression of the signal
            response is retained.
        n_stencil : int
            Stencil order: 2, 4, or 6. Higher order reduces truncation error O(h^n).
        use_crn : bool
            Use Common Random Numbers to cancel cosmic variance in the subtraction.
        n_jobs : int
            Number of parallel workers (joblib). -1 uses all CPUs.
            Each worker handles all noise_seeds for one (theta_step, IC seed) pair.
        n_chunks : int
            (parallel only) Run the seeds in this many chunks, writing the cache
            after each, with a progress bar per finished chunk.

        Returns
        -------
        self
        """
        stencil_configs = {
            2: {'weights': [-1/2,   1/2],                          'steps': [-1, 1]},
            4: {'weights': [1/12, -2/3,  2/3, -1/12],             'steps': [-2, -1, 1, 2]},
            6: {'weights': [-1/60, 3/20, -3/4, 3/4, -3/20, 1/60], 'steps': [-3, -2, -1, 1, 2, 3]},
        }
        if n_stencil not in stencil_configs:
            raise ValueError("n_stencil must be 2, 4, or 6.")

        weights = stencil_configs[n_stencil]['weights']
        steps   = stencil_configs[n_stencil]['steps']
        seeds   = [int(s) for s in seeds]
        h_arr   = self._step_sizes(delta_frac)

        if self.x_realisations is not None:
            n_data = self.x_realisations.shape[1]
        else:
            x_test = self.simulator(self.theta_fid, seeds[0], 0, **self.sim_kwargs)
            n_data = len(self._normalise(x_test))

        self.dmu_dtheta = np.zeros((self.n_params, n_data))
        if noise_seeds is not None:
            noise_seeds_deriv = [int(ns) for ns in noise_seeds]
        else:
            noise_seeds_deriv = [noise_seed] if noise_seed is not None else list(range(10))

        # Per-group derivative estimates, used by compute(debias_derivatives=True)
        # to remove the tr(C^-1 Cov(dmu)) inflation of the Fisher matrix.
        self.deriv_seeds = seeds
        self.deriv_noise_seeds = noise_seeds_deriv
        if use_crn:
            self.dmu_samples_ns = None
            self.dmu_samples_ic = None
        else:
            self.dmu_samples_ns = np.zeros((len(noise_seeds_deriv), self.n_params, n_data))
            self.dmu_samples_ic = np.zeros((len(seeds), self.n_params, n_data))

        crn_tag = " [CRN]" if use_crn else ""
        self._print(f"\n{'='*50}")
        self._print(f"  Computing derivatives ({n_stencil}-point stencil{crn_tag})")
        self._print(f"{'='*50}")

        stencil_thetas = []
        for j in range(self.n_params):
            h = h_arr[j]
            for s_step in steps:
                theta_step = self.theta_fid.copy()
                theta_step[j] += s_step * h
                stencil_thetas.append(theta_step)
        self._prerun_parallel(stencil_thetas, seeds, noise_seeds_deriv, n_jobs,
                              desc="Computing derivatives", n_chunks=n_chunks)

        for j, pname in enumerate(self.param_names):
            h = h_arr[j]

            if use_crn:
                deriv_per_seed = []
                for seed in (pbar := tqdm(seeds, desc=f"d/d({pname}) CRN", leave=False, disable=not self.verbose)):
                    for ns in noise_seeds_deriv:
                        deriv_i = np.zeros(n_data)
                        for w, s in zip(weights, steps):
                            theta_step = self.theta_fid.copy()
                            theta_step[j] += s * h
                            step_str = "[" + ", ".join(f"{v:.3g}" for v in theta_step) + "]"
                            pbar.set_postfix(seed=seed, noise=ns, theta=step_str)
                            x = self._normalise(self._simulate_cached(theta_step, seed, ns))
                            deriv_i += w * x
                        deriv_per_seed.append(deriv_i / h)
                self.dmu_dtheta[j] = np.mean(deriv_per_seed, axis=0)
            else:
                x_steps = np.zeros((len(steps), len(seeds), len(noise_seeds_deriv), n_data))
                for si, s in enumerate(steps):
                    theta_step = self.theta_fid.copy()
                    theta_step[j] += s * h
                    step_str = "[" + ", ".join(f"{v:.3g}" for v in theta_step) + "]"
                    for ki, seed in enumerate(pbar := tqdm(seeds, desc=f"d/d({pname}) step {s:+d}h", leave=False, disable=not self.verbose)):
                        for ni, ns in enumerate(noise_seeds_deriv):
                            pbar.set_postfix(seed=seed, noise=ns, theta=step_str)
                            x_raw = self._simulate_cached(theta_step, seed, ns)
                            x_steps[si, ki, ni] = self._normalise(x_raw)
                w_arr = np.asarray(weights)
                self.dmu_dtheta[j] = np.tensordot(w_arr, x_steps.mean(axis=(1, 2)), axes=1) / h
                self.dmu_samples_ns[:, j, :] = np.tensordot(w_arr, x_steps.mean(axis=1), axes=1) / h
                self.dmu_samples_ic[:, j, :] = np.tensordot(w_arr, x_steps.mean(axis=2), axes=1) / h

            self._print(f"  {pname}: |dmu/dtheta| = {np.linalg.norm(self.dmu_dtheta[j]):.4f}")

        if self.dmu_samples_ns is not None:
            # Collapse antithetic pairs (ns, -ns) into single samples: pair members
            # are anticorrelated by construction, so treating them as independent
            # would overestimate Cov(dmu) in the debias correction.
            ns_arr = np.array(noise_seeds_deriv)
            used, groups = set(), []
            for idx, ns in enumerate(ns_arr):
                if idx in used:
                    continue
                partner = [p for p in np.where(ns_arr == -ns)[0] if p not in used and p != idx]
                if ns != 0 and partner:
                    groups.append([idx, partner[0]])
                    used.update({idx, partner[0]})
                else:
                    groups.append([idx])
                    used.add(idx)
            if any(len(g) == 2 for g in groups):
                self.dmu_samples_ns = np.array(
                    [self.dmu_samples_ns[g].mean(axis=0) for g in groups])

        self._print(f"{'='*50}\n")
        return self

    def convergence_test_delta_frac(
        self,
        delta_fracs=None,
        seeds=range(1, 11),
        noise_seed=0,
        noise_seeds=None,
        n_stencil=2,
        use_crn=False,
        plot=True,
        restore=True,
        n_jobs=1,
    ):
        """
        Test sensitivity of dmu/dtheta to the finite-difference step size.

        Sweeps ``delta_fracs`` and records the per-parameter norm of the derivative
        and the 1D marginal constraint. A good step size shows a flat plateau in the
        middle: upturn at small h (stochastic noise) and at large h (truncation error).

        Parameters
        ----------
        delta_fracs : array-like or None
            Step sizes to test. Default: 10 log-spaced values from 1e-3 to 0.3.
        seeds : iterable
            Forwarded to estimate_derivatives.
        noise_seed : int or None
            Forwarded to estimate_derivatives.
        n_stencil : int
            Forwarded to estimate_derivatives.
        use_crn : bool
            Forwarded to estimate_derivatives.
        plot : bool
            If True, show diagnostic plots.
        restore : bool
            Restore self.dmu_dtheta to its value before the test.
        n_jobs : int
            Forwarded to estimate_derivatives.

        Returns
        -------
        dict
            Per-parameter dict with keys 'delta_fracs', 'norms', 'sigma_1d'.
        """
        if delta_fracs is None:
            delta_fracs = np.logspace(-3, np.log10(0.3), 10)
        delta_fracs = np.asarray(delta_fracs)

        dmu_saved = self.dmu_dtheta.copy() if self.dmu_dtheta is not None else None
        results = {p: {'delta_fracs': delta_fracs, 'norms': [], 'sigma_1d': []}
                   for p in self.param_names}

        for df in tqdm(delta_fracs, desc="delta_frac sweep", disable=not self.verbose):
            self.estimate_derivatives(
                delta_frac=df, seeds=seeds, noise_seed=noise_seed,
                noise_seeds=noise_seeds, n_stencil=n_stencil, use_crn=use_crn,
                n_jobs=n_jobs,
            )
            for j, pname in enumerate(self.param_names):
                results[pname]['norms'].append(np.linalg.norm(self.dmu_dtheta[j]))
                if self.C_inv is not None:
                    self.F = self.dmu_dtheta @ self.C_inv @ self.dmu_dtheta.T
                    try:
                        self.F_inv = np.linalg.inv(self.F)
                        results[pname]['sigma_1d'].append(self.sigma_1d()[j])
                    except np.linalg.LinAlgError:
                        results[pname]['sigma_1d'].append(np.nan)

        for pname in self.param_names:
            results[pname]['norms'] = np.array(results[pname]['norms'])
            s = results[pname]['sigma_1d']
            results[pname]['sigma_1d'] = np.array(s) if s else None

        if restore:
            self.dmu_dtheta = dmu_saved
            if self.compute_method is not None and dmu_saved is not None and self.C_inv is not None:
                self.F = self.dmu_dtheta @ self.C_inv @ self.dmu_dtheta.T
                self.F_inv = np.linalg.inv(self.F)
            else:
                self.F = self.F_inv = None
            self._print("  Restored original dmu_dtheta.")

        if plot:
            has_sigma = self.C_inv is not None
            n_rows = 2 if has_sigma else 1
            fig, axes = plt.subplots(n_rows, self.n_params,
                                     figsize=(4 * self.n_params, 3.5 * n_rows),
                                     squeeze=False)
            for j, pname in enumerate(self.param_names):
                axes[0, j].plot(delta_fracs, results[pname]['norms'], 'o-')
                axes[0, j].set_xscale('log')
                axes[0, j].set_xlabel('delta_frac')
                axes[0, j].set_ylabel('|dmu/dtheta|')
                axes[0, j].set_title(pname)
                if has_sigma:
                    axes[1, j].plot(delta_fracs, results[pname]['sigma_1d'], 's-', color='C1')
                    axes[1, j].set_xscale('log')
                    axes[1, j].set_xlabel('delta_frac')
                    axes[1, j].set_ylabel('sigma_1d')
                    axes[1, j].set_title(pname)
            fig.suptitle('Convergence test: delta_frac sensitivity')
            fig.tight_layout()
            plt.show()

        return results

    # ---- Fisher computation ----

    def compute(self, method='standard', n_bootstrap=0, n_split=None, shrinkage_target='diagonal',
                debias_derivatives=False):
        """
        Compute the Fisher matrix. Results stored in self.F and self.F_inv.

        Parameters
        ----------
        method : str
            - ``'standard'``: F = J C⁻¹ Jᵀ (default).
            - ``'lfim'``: score-trick estimator via Synthetic Likelihood.
            - ``'CoultonWandelt2023'``: geometric mean of the standard Fisher (biased
              high by derivative noise) and a compressed Fisher (biased low).
              Also sets self.F_upper and self.F_lower.
            - ``'Shrinkage'``: Ledoit-Wolf-style covariance regularisation.
        n_bootstrap : int
            (lfim only) Bootstrap resamples for uncertainty; populates self.F_std.
        n_split : int or None
            (CoultonWandelt2023 only) Number of derivative IC seeds used for the
            compression; the rest measure the compressed derivatives, and the two
            roles are then swapped. Defaults to half.
        shrinkage_target : str
            (Shrinkage only) 'diagonal' or 'identity'.
        debias_derivatives : bool
            ('standard' only) Subtract the Fisher inflation caused by stochastic
            error in the estimated derivatives, E[dm^T C^-1 dm] = F + tr(C^-1 Cov(dm)).
            Cov(dm) is estimated from the per-noise-seed and per-IC-seed derivative
            scatter recorded by estimate_derivatives (non-CRN path). If the debiased
            matrix has non-positive eigenvalues, the corresponding directions carry
            no significant information; eigenvalues are clipped with a warning.

        Returns
        -------
        self
        """
        assert self.C_inv is not None,      "Call estimate_covariance() first."
        assert self.dmu_dtheta is not None, "Call estimate_derivatives() first."

        self.compute_method = method
        self.F_std = None

        dispatch = {
            'standard':           self._compute_standard,
            'lfim':               lambda: self._compute_lfim(n_bootstrap),
            'CoultonWandelt2023': lambda: self._compute_CoultonWandelt(n_split),
            'Shrinkage':          lambda: self._compute_Shrinkage(shrinkage_target),
        }
        if method not in dispatch:
            raise ValueError(f"Unknown method {method!r}. "
                             f"Choose from {list(dispatch)}.")
        dispatch[method]()

        if debias_derivatives:
            assert method == 'standard', "debias_derivatives requires method='standard'."
            corr = self._derivative_noise_correction()
            rel = np.abs(np.diag(corr)) / np.abs(np.diag(self.F))
            self._print(f"  Derivative-noise correction (diag, relative to F): "
                        f"{np.round(rel, 3)}")
            self.F = self.F - corr
            self.compute_method = 'standard+debias'
            eigval, eigvec = np.linalg.eigh(self.F)
            if np.any(eigval <= 0):
                # Always printed, independent of self.verbose: a non-positive
                # eigenvalue here means the debias correction (itself an
                # estimate with sampling error) exceeded the raw signal, so at
                # least one direction is consistent with zero information. The
                # clip only keeps F_inv computable; it does NOT recover real
                # information, and the resulting ellipse size/orientation along
                # that direction is not physically meaningful.
                print(f"  WARNING: debiased Fisher has non-positive eigenvalues; "
                      f"those directions carry no significant information. "
                      f"Clipping {np.sum(eigval <= 0)} eigenvalue(s). "
                      f"Pre-clip eigenvalues: {np.round(eigval, 6)}")
                eigval = np.clip(eigval, 1e-10 * eigval.max(), None)
                self.F = eigvec @ np.diag(eigval) @ eigvec.T

        self.F_inv = np.linalg.inv(self.F)
        self._print_summary()
        return self

    def _derivative_noise_correction(self):
        """tr(C^-1 Cov(dmu_i, dmu_j)) from per-noise-seed and per-IC-seed scatter.

        The two contributions (noise realisations, IC realisations) are added;
        their small interaction term is double counted, which slightly
        overcorrects — conservative in the constraint direction.
        """
        corr = np.zeros((self.n_params, self.n_params))
        for samples in (self.dmu_samples_ns, self.dmu_samples_ic):
            if samples is None or len(samples) < 2:
                continue
            N   = len(samples)
            dev = samples - samples.mean(axis=0)     # (N, n_params, n_data)
            for i in range(self.n_params):
                for j in range(self.n_params):
                    K_ij = dev[:, i, :].T @ dev[:, j, :] / (N - 1) / N
                    corr[i, j] += np.sum(self.C_inv * K_ij)
        return corr

    def _compute_standard(self):
        self.F = self.dmu_dtheta @ self.C_inv @ self.dmu_dtheta.T

    def _geometricMean(self, A, B):
        AB = np.linalg.solve(A, B)
        return A.dot(scipy.linalg.sqrtm(AB))

    def _compute_CoultonWandelt(self, n_split=None):
        """
        Combined estimator of Coulton & Wandelt (2023).

        Noise in the estimated derivatives biases the standard Fisher matrix high
        (F_upper). Compressing the data with derivatives from one set of IC seeds
        and measuring the compressed derivatives with the other set gives a
        Fisher matrix biased low (F_lower): noise in the compression only loses
        information. The two sets then swap roles and the results are averaged.
        F is the matrix geometric mean of F_upper and F_lower.
        """
        assert self.x_realisations is not None, "Call estimate_covariance() first."
        S = self.dmu_samples_ic
        assert S is not None and len(S) >= 4, \
            "Needs per-IC derivative samples: estimate_derivatives(use_crn=False) with >= 4 seeds."
        self._print("  Using Coulton & Wandelt (2023) combined estimator...")

        n_split = len(S) // 2 if n_split is None else n_split
        D_a, D_b = S[:n_split].mean(axis=0), S[n_split:].mean(axis=0)

        F_upper = self.dmu_dtheta @ self.C_inv @ self.dmu_dtheta.T
        F_lower = np.zeros_like(F_upper)
        for D_comp, D_meas in ((D_a, D_b), (D_b, D_a)):
            # compression t = D_comp C^-1 x: Cov(t)^-1 = (D_comp C^-1 D_comp^T)^-1 with the same
            # (Hartlap-corrected) C^-1, so exact derivatives give F_lower = F_upper
            G = D_meas @ self.C_inv @ D_comp.T           # dt/dtheta from the other seeds
            F_lower += 0.5 * G @ np.linalg.inv(D_comp @ self.C_inv @ D_comp.T) @ G.T

        self.F_upper = 0.5 * (F_upper + F_upper.T)
        self.F_lower = 0.5 * (F_lower + F_lower.T)
        F = np.real(self._geometricMean(self.F_lower, self.F_upper))
        self.F = 0.5 * (F + F.T)

    def _compute_Shrinkage(self, shrinkage_target='diagonal'):
        """Ledoit-Wolf-style covariance regularisation (Pope & Szapudi 2008)."""
        assert self.x_realisations is not None, "Call estimate_covariance() first."
        self._print(f"  Using covariance shrinkage (target: {shrinkage_target})...")
        n_data = self.x_realisations.shape[1]
        S = np.cov(self.x_realisations, rowvar=False)
        target = np.diag(np.diag(S)) if shrinkage_target == 'diagonal' else np.eye(n_data) * np.mean(np.diag(S))
        C_reg = 0.9 * S + 0.1 * target
        self.F = self.dmu_dtheta @ np.linalg.inv(C_reg) @ self.dmu_dtheta.T

    def _compute_lfim(self, n_bootstrap=0):
        """LFIM score-trick: F = (1/N) Σᵢ sᵢ sᵢᵀ, where s_k = (dμ/dθ_k)ᵀ C⁻¹ (xᵢ − μ)."""
        assert self.x_realisations is not None, "Call estimate_covariance() first."
        self._print("  Using LFIM score-trick estimator...")

        def _scores(X):
            dX = X - self.x_mean_fid
            return (self.dmu_dtheta @ self.C_inv @ dX.T).T

        if n_bootstrap > 0:
            N = len(self.x_realisations)
            F_boots = []
            for _ in tqdm(range(n_bootstrap), desc="LFIM bootstrap", disable=not self.verbose):
                idx = np.random.choice(N, size=N, replace=True)
                S_b = _scores(self.x_realisations[idx])
                F_boots.append(S_b.T @ S_b / N)
            F_boots    = np.array(F_boots)
            self.F     = np.mean(F_boots, axis=0)
            self.F_std = np.std(F_boots, axis=0)
        else:
            S = _scores(self.x_realisations)
            self.F = S.T @ S / len(S)

    def _print_summary(self):
        sigma_1d   = self.sigma_1d()
        sigma_2d   = self.sigma_2d()
        sigma_cond = self.sigma_conditional()
        corr       = self.correlation_matrix()

        self._print(f"\n{'='*50}")
        self._print(f"  Fisher Matrix Results  [{self.compute_method}]")
        self._print(f"{'='*50}")
        self._print(f"\n{self.F}")

        self._print(f"\n  --- 1D marginal errors ---")
        for j, p in enumerate(self.param_names):
            frac = sigma_1d[j] / abs(self.theta_fid[j]) * 100 if self.theta_fid[j] != 0 else float('inf')
            self._print(f"    {p}: {sigma_1d[j]:.4f}  ({frac:.1f}%)")

        self._print(f"\n  --- 2D marginal errors ---")
        for (i, j), (si, sj) in sigma_2d.items():
            pi, pj = self.param_names[i], self.param_names[j]
            self._print(f"    ({pi}, {pj}): sigma_{pi}={si:.4f}, sigma_{pj}={sj:.4f}")

        self._print(f"\n  --- Conditional errors (others fixed) ---")
        for j, p in enumerate(self.param_names):
            frac = sigma_cond[j] / abs(self.theta_fid[j]) * 100 if self.theta_fid[j] != 0 else float('inf')
            self._print(f"    {p}: {sigma_cond[j]:.4f}  ({frac:.1f}%)")

        self._print(f"\n  --- Correlations ---")
        for i in range(self.n_params):
            for j in range(i + 1, self.n_params):
                self._print(f"    corr({self.param_names[i]}, {self.param_names[j]}): {corr[i,j]:.3f}")

        if self.F_std is not None:
            self._print(f"\n  --- Bootstrap uncertainty (diag of F_std) ---")
            for j, p in enumerate(self.param_names):
                self._print(f"    {p}: F_std[{j},{j}] = {self.F_std[j,j]:.4e}")

        self._print(f"{'='*50}\n")

    # ---- Error accessors ----

    def sigma_1d(self):
        """
        1D marginal 1-sigma errors (marginalised over all other parameters).

        Returns
        -------
        np.ndarray, shape (n_params,)
            sigma_i = sqrt(F_inv[i,i])
        """
        assert self.F_inv is not None, "Call compute() first."
        return np.sqrt(np.diag(self.F_inv))

    def sigma_2d(self):
        """
        2D marginal errors for all parameter pairs.

        Returns
        -------
        dict keyed by (i, j) → (sigma_i, sigma_j)
        """
        assert self.F_inv is not None, "Call compute() first."
        result = {}
        for i in range(self.n_params):
            for j in range(i + 1, self.n_params):
                sub = self.F_inv[np.ix_([i, j], [i, j])]
                result[(i, j)] = (np.sqrt(sub[0, 0]), np.sqrt(sub[1, 1]))
        return result

    def sigma_conditional(self):
        """
        Conditional errors (all other parameters perfectly known).

        Returns
        -------
        np.ndarray, shape (n_params,)
            sigma_i = 1 / sqrt(F[i,i])
        """
        assert self.F is not None, "Call compute() first."
        return 1.0 / np.sqrt(np.diag(self.F))

    def correlation_matrix(self):
        """
        Parameter correlation matrix derived from F_inv.

        Returns
        -------
        np.ndarray, shape (n_params, n_params)
        """
        assert self.F_inv is not None, "Call compute() first."
        sigma = self.sigma_1d()
        return self.F_inv / np.outer(sigma, sigma)

    def sub_fisher(self, param_indices):
        """
        Extract a Fisher sub-matrix for a subset of parameters.

        Parameters
        ----------
        param_indices : list of int

        Returns
        -------
        F_sub, F_sub_inv, sigma_sub
        """
        assert self.F is not None, "Call compute() first."
        idx = list(param_indices)
        F_sub = self.F[np.ix_(idx, idx)]
        F_sub_inv = np.linalg.inv(F_sub)
        return F_sub, F_sub_inv, np.sqrt(np.diag(F_sub_inv))

    def ellipse_params(self, i, j, n_sigma=1):
        """
        Compute ellipse parameters for the 2D marginal of parameters (i, j).

        Parameters
        ----------
        i, j : int
            Parameter indices.
        n_sigma : int
            1 or 2.

        Returns
        -------
        dict with keys: center, width, height, angle_deg
        """
        assert self.F_inv is not None, "Call compute() first."
        cov_2d = self.F_inv[np.ix_([i, j], [i, j])]
        eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        scale = {1: 2.30, 2: 6.18}.get(n_sigma, 2.30)
        return {
            'center':    (self.theta_fid[i], self.theta_fid[j]),
            'width':     2 * np.sqrt(eigenvalues[0] * scale),
            'height':    2 * np.sqrt(eigenvalues[1] * scale),
            'angle_deg': angle,
        }

    # ---- Combining ----

    @classmethod
    def joint(cls, blocks, verbose=True):
        """
        Fisher matrix of the concatenated data vectors of several FisherMatrix objects.

        Use this when the blocks come from the same simulations, e.g. different
        summaries, or one summary at different redshifts, so they are correlated
        and ``F_a + F_b`` would double count. The covariance is estimated from the
        concatenated vectors, paired by (seed, noise_seed), and the derivatives
        are concatenated block by block. Keeping one block (and one cache file)
        per summary or redshift means that adding a block later reuses the rest.

        Every block must have run estimate_covariance on the same (seed,
        noise_seed) pairs and estimate_derivatives on the same IC seeds. A block
        whose derivatives used a single noise seed (e.g. noiseless) enters the
        noise part of the derivative debias with zero scatter.

        Parameters
        ----------
        blocks : list of FisherMatrix
        verbose : bool

        Returns
        -------
        FisherMatrix
            Call compute() on it as usual.
        """
        b0 = blocks[0]
        for b in blocks:
            assert np.array_equal(b.theta_fid, b0.theta_fid), "Fiducials must match."
            assert b.param_names == b0.param_names,            "Parameter names must match."
            assert b.x_realisations is not None, "Call estimate_covariance() on every block first."
            assert b.dmu_dtheta is not None,     "Call estimate_derivatives() on every block first."

        rows = [dict(zip(b.cov_keys, b.x_realisations)) for b in blocks]
        missing = [i for i, r in enumerate(rows) if not set(b0.cov_keys) <= set(r)]
        if missing:
            raise ValueError(f"Blocks {missing} lack covariance realisations of block 0.")

        def stack_samples(attr, ids_attr):
            # Blocks with one sample (or none) have zero scatter: repeat their mean derivative
            many = [b for b in blocks if getattr(b, attr) is not None and len(getattr(b, attr)) > 1]
            if not many:
                return None
            for b in many[1:]:
                if list(getattr(b, ids_attr)) != list(getattr(many[0], ids_attr)):
                    raise ValueError(f"{attr}: blocks used different {ids_attr}.")
            n = len(getattr(many[0], attr))
            return np.concatenate([getattr(b, attr) if b in many
                                   else np.repeat(b.dmu_dtheta[None], n, axis=0)
                                   for b in blocks], axis=2)

        fc = cls(None, b0.theta_fid, b0.param_names, verbose=verbose)
        fc.cov_keys       = list(b0.cov_keys)
        fc.x_realisations = np.array([np.concatenate([r[k] for r in rows]) for k in fc.cov_keys])
        fc._set_covariance()
        fc.dmu_dtheta     = np.hstack([b.dmu_dtheta for b in blocks])
        fc.dmu_samples_ic = stack_samples('dmu_samples_ic', 'deriv_seeds')
        fc.dmu_samples_ns = stack_samples('dmu_samples_ns', 'deriv_noise_seeds')
        fc.deriv_seeds    = b0.deriv_seeds
        return fc

    def __add__(self, other):
        """
        Add two Fisher matrices (e.g. from independent data sets).

        Returns a new FisherMatrix with F = self.F + other.F.
        """
        assert np.array_equal(self.theta_fid, other.theta_fid), "Fiducials must match."
        assert self.param_names == other.param_names,            "Parameter names must match."
        assert self.F is not None and other.F is not None,       "Call compute() on both first."
        combined = type(self)(
            simulator=None,
            theta_fid=self.theta_fid,
            param_names=self.param_names,
            verbose=self.verbose,
        )
        combined.F     = self.F + other.F
        combined.F_inv = np.linalg.inv(combined.F)
        combined._print_summary()
        return combined

    # ---- Plotting ----

    def plot_ellipses(
        self,
        others=None,
        labels=None,
        label_names=None,
        prior_range=None,
        n_sigma_list=(1, 2),
        figsize=None,
        colors=None,
        filled=False,
    ):
        """
        Triangle plot of Fisher confidence ellipses.

        Parameters
        ----------
        others : dict or None
            Additional FisherMatrix objects to overlay, keyed by label string.
        labels : str or None
            Label for this Fisher matrix in the legend.
        label_names : dict or None
            param_name -> LaTeX label string (without ``$``).
        prior_range : dict or None
            param_name -> (low, high) for axis limits.
        n_sigma_list : list of int
            Contour levels (1 and/or 2).
        figsize : tuple or None
            Figure size.
        colors : dict or None
            label -> colour.
        filled : bool
            Fill the innermost ellipse.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from matplotlib.patches import Ellipse as EllipsePatch

        all_fishers = {labels if labels else "this": self}
        if others is not None:
            all_fishers.update(others)

        n_p = self.n_params
        if label_names is None:
            label_names = {p: p for p in self.param_names}
        if colors is None:
            colors = {name: f"C{i}" for i, name in enumerate(all_fishers)}
        if figsize is None:
            figsize = (3.5 * (n_p - 1), 3.5 * (n_p - 1))

        pairs = list(combinations(range(n_p), 2))
        fig, axes = plt.subplots(n_p - 1, n_p - 1, figsize=figsize)
        if n_p == 2:
            axes = np.array([[axes]])

        for row in axes:
            for ax in (row if hasattr(row, '__iter__') else [row]):
                ax.set_visible(False)

        for name, fisher in all_fishers.items():
            assert fisher.F_inv is not None, f"Call compute() on '{name}' first."
            color = colors.get(name, 'C0')

            for (i, j) in pairs:
                ax = axes[j - 1][i]
                ax.set_visible(True)

                for ns in n_sigma_list:
                    ep = fisher.ellipse_params(i, j, n_sigma=ns)
                    is_inner = ns == n_sigma_list[0]
                    ellipse = EllipsePatch(
                        xy=ep['center'],
                        width=ep['width'],
                        height=ep['height'],
                        angle=ep['angle_deg'],
                        fill=filled and is_inner,
                        facecolor=color if (filled and is_inner) else 'none',
                        alpha=0.2 if (filled and is_inner) else 1.0,
                        edgecolor=color,
                        linewidth=2 if is_inner else 1,
                        linestyle='-' if is_inner else '--',
                        label=name if (is_inner and i == pairs[0][0] and j == pairs[0][1]) else None,
                    )
                    ax.add_patch(ellipse)

                ax.plot(fisher.theta_fid[i], fisher.theta_fid[j], '+', color='k', markersize=10)
                ax.set_xlabel(f"${label_names[self.param_names[i]]}$")
                ax.set_ylabel(f"${label_names[self.param_names[j]]}$")

                all_F_invs = [f.F_inv for f in all_fishers.values()]
                margin_i = 4 * max(np.sqrt(fi[i, i]) for fi in all_F_invs)
                margin_j = 4 * max(np.sqrt(fi[j, j]) for fi in all_F_invs)
                if prior_range:
                    ax.set_xlim(prior_range[self.param_names[i]])
                    ax.set_ylim(prior_range[self.param_names[j]])
                else:
                    ax.set_xlim(self.theta_fid[i] - margin_i, self.theta_fid[i] + margin_i)
                    ax.set_ylim(self.theta_fid[j] - margin_j, self.theta_fid[j] + margin_j)

        for row in axes:
            for ax in (row if hasattr(row, '__iter__') else [row]):
                if ax.get_visible():
                    ax.legend(fontsize=9)
                    break
            else:
                continue
            break

        plt.tight_layout()
        plt.show()
        return fig

    def plot_derivatives(self, label_names=None, figsize=None):
        """
        Plot dmu/dtheta for each parameter.

        Returns
        -------
        matplotlib.figure.Figure
        """
        assert self.dmu_dtheta is not None, "Call estimate_derivatives() first."
        if label_names is None:
            label_names = {p: p for p in self.param_names}
        if figsize is None:
            figsize = (4 * self.n_params, 3.5)

        fig, axes = plt.subplots(1, self.n_params, figsize=figsize)
        if self.n_params == 1:
            axes = [axes]
        for j, p in enumerate(self.param_names):
            axes[j].plot(self.dmu_dtheta[j], 'o-', markersize=4)
            axes[j].set_title(f"$d\\mu / d({label_names[p]})$")
            axes[j].set_xlabel("Data index")
            axes[j].axhline(0, color='k', ls='--', lw=0.5)
        plt.tight_layout()
        plt.show()
        return fig

    def plot_covariance(self, figsize=(6, 5)):
        """
        Plot the covariance and correlation matrices.

        Returns
        -------
        matplotlib.figure.Figure
        """
        assert self.C is not None, "Call estimate_covariance() first."
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
        im0 = axes[0].imshow(self.C, aspect='auto')
        axes[0].set_title("Covariance matrix")
        plt.colorbar(im0, ax=axes[0])
        std = np.sqrt(np.diag(self.C))
        std[std == 0] = 1
        corr = self.C / np.outer(std, std)
        im1 = axes[1].imshow(corr, aspect='auto', vmin=-1, vmax=1, cmap='RdBu_r')
        axes[1].set_title("Correlation matrix")
        plt.colorbar(im1, ax=axes[1])
        plt.tight_layout()
        plt.show()
        return fig

    # ---- Sampling ----

    def sample(self, n_samples=100_000):
        """
        Draw Gaussian samples from the Fisher posterior.

        Parameters
        ----------
        n_samples : int

        Returns
        -------
        np.ndarray, shape (n_samples, n_params)
        """
        assert self.F_inv is not None, "Call compute() first."
        return np.random.multivariate_normal(self.theta_fid, self.F_inv, size=n_samples)

    def to_getdist(self, n_samples=100_000, label=None, label_names=None):
        """
        Generate a getdist MCSamples object (Gaussian approximation).

        Parameters
        ----------
        n_samples : int
        label : str or None
        label_names : dict or None
            param_name -> LaTeX label.

        Returns
        -------
        getdist.MCSamples
        """
        from getdist import MCSamples
        if label_names is None:
            label_names = {p: p for p in self.param_names}
        return MCSamples(
            samples=self.sample(n_samples),
            names=self.param_names,
            labels=[label_names[p] for p in self.param_names],
            label=label,
        )

    # ---- Save / Load ----

    def save(self, filepath):
        """
        Save Fisher matrix results to an npz file.

        Parameters
        ----------
        filepath : str
        """
        d = {
            'theta_fid': self.theta_fid,
            'param_names': np.array(self.param_names),
            'X_mean': self.X_mean,
            'X_std': self.X_std,
        }
        for attr in self._ARRAY_ATTRS:
            if getattr(self, attr) is not None:
                d[attr] = getattr(self, attr)
        if self.hartlap is not None:
            d['hartlap'] = np.array([self.hartlap])
        if self.compute_method is not None:
            d['compute_method'] = np.array([self.compute_method])
        np.savez(filepath, **d)
        self._print(f"FisherMatrix saved to {filepath}")

    @classmethod
    def load(cls, filepath, simulator=None, **sim_kwargs):
        """
        Load a FisherMatrix from an npz file.

        Parameters
        ----------
        filepath : str
        simulator : callable or None

        Returns
        -------
        FisherMatrix
        """
        data = np.load(filepath, allow_pickle=True)
        obj = cls(
            simulator=simulator,
            theta_fid=data['theta_fid'],
            param_names=list(data['param_names']),
            X_mean=float(data.get('X_mean', 0.0)),
            X_std=float(data.get('X_std', 1.0)),
            **sim_kwargs,
        )
        for attr in cls._ARRAY_ATTRS:
            if attr in data:
                setattr(obj, attr, data[attr])
        if 'hartlap' in data:
            obj.hartlap = float(data['hartlap'][0])
        if 'compute_method' in data:
            obj.compute_method = str(data['compute_method'][0])
        obj._print(f"FisherMatrix loaded from {filepath}")
        return obj

    def __repr__(self):
        status = "computed" if self.F is not None else "not computed"
        n_real = len(self.x_realisations) if self.x_realisations is not None else 0
        return (f"{type(self).__name__}({status}, params={self.param_names}, "
                f"n_realisations={n_real})")


def fisher_blocks(simulator, theta_fid, param_names, blocks, cache_file=None,
                  covariance_kw=None, derivative_kw=None, verbose=False, desc='blocks',
                  fisher_class=None):
    """
    One FisherMatrix, with its own cache file, per block of a data vector.

    A block is any part of the data vector that is simulated separately:
    one redshift, frequency band, sky patch or summary statistic. Keeping one
    cache per block means that adding a block later only simulates the new
    one. Combine the blocks with ``FisherMatrix.joint``, which keeps their
    cross-covariance.

    Parameters
    ----------
    simulator : callable
        ``simulator(theta, seed, noise_seed, **kwargs)`` -> data vector of one block.
    theta_fid, param_names :
        As for FisherMatrix.
    blocks : dict
        Block name -> keyword arguments passed to the simulator for that block,
        e.g. ``{z: dict(redshifts=[z]) for z in redshifts}``.
    cache_file : callable or None
        cache_file(name) -> path of the block's cache file.
    covariance_kw, derivative_kw : dict or None
        Arguments of estimate_covariance and estimate_derivatives, the same for
        every block (so the blocks can be joined).
    verbose : bool
    desc : str
        Label of the progress bar over the blocks.
    fisher_class : type or None
        Class of the blocks, FisherMatrix (default) or a subclass such as
        FisherMatrixDALI. Join them with ``fisher_class.joint``.

    Returns
    -------
    dict
        Block name -> FisherMatrix with covariance and derivatives estimated.

    Examples
    --------
    >>> fms = fisher_blocks(sim, theta, names, {z: dict(redshifts=[z]) for z in (7, 8, 9)},
    ...                     cache_file=lambda z: f'cache_z{z:.2f}.pkl',
    ...                     covariance_kw=dict(seeds=range(1, 41), noise_seeds=range(1, 41)),
    ...                     derivative_kw=dict(seeds=range(1, 41), delta_frac=0.02))
    >>> F = FisherMatrix.joint(list(fms.values())).compute()
    """
    fisher_class = FisherMatrix if fisher_class is None else fisher_class
    out = {}
    for name, kw in (pbar := tqdm(blocks.items(), desc=desc)):
        pbar.set_postfix(block=name)
        fm = fisher_class(simulator, theta_fid, param_names, verbose=verbose,
                          cache_file=None if cache_file is None else cache_file(name), **kw)
        fm.estimate_covariance(**(covariance_kw or {}))
        fm.estimate_derivatives(**(derivative_kw or {}))
        out[name] = fm
    return out
