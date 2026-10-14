"""Inference diagnostics: posterior metrics, comparison plots, and triangle plots."""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.linalg import inv


class DistributionDiagnostic(ABC):
    """Abstract base for posterior diagnostic metrics and plots.

    Subclasses implement get_samples() to expose their representation
    as weighted Monte Carlo particles, enabling shared metric computation.

    Parameters
    ----------
    true_values : array_like, optional
        Ground-truth parameter values for metric computation.
    labels : list of str, optional
        LaTeX parameter labels (without $ delimiters).
    """

    def __init__(self, true_values=None, labels=None):
        self.true_values = np.asarray(true_values, dtype=float) if true_values is not None else None
        self.labels = labels
        self._derived = []

    # --- interface ---

    @abstractmethod
    def get_samples(self):
        """Return (samples, weights).

        samples : (N, D) float array
        weights : (N,) float array or None (uniform)
        """

    # --- derived properties ---

    @property
    def n_params(self):
        s, _ = self.get_samples()
        return s.shape[1] if s.ndim > 1 else 1

    def _param_labels(self):
        return self.labels or [rf'\theta_{{{i+1}}}' for i in range(self.n_params)]

    def _weighted_stats(self):
        samples, weights = self.get_samples()
        if samples.ndim == 1:
            samples = samples[:, None]
        if weights is None:
            weights = np.ones(len(samples))
        w = weights / weights.sum()
        means = np.average(samples, axis=0, weights=w)
        if samples.shape[1] == 1:
            cov = np.array([[float(np.average((samples[:, 0] - means[0]) ** 2, weights=w))]])
        else:
            cov = np.cov(samples.T, aweights=w)
        if cov.ndim == 0:
            cov = np.atleast_2d(float(cov))
        sigmas = np.sqrt(np.diag(cov))
        return means, cov, sigmas, samples, w

    # --- credible intervals (equal-tailed, from empirical CDF) ---

    def credible_intervals(self, levels=(68, 95)):
        """Marginal credible intervals from the weighted empirical CDF.

        Returns a list of dicts, one per parameter, with keys:
            median, lo<lv>, hi<lv>  for each level in *levels*
            cdf_at                  callable: CDF evaluated at a scalar
        """
        samples, weights = self.get_samples()
        if samples.ndim == 1:
            samples = samples[:, None]
        if weights is None:
            weights = np.ones(len(samples))
        w = weights / weights.sum()

        result = []
        for p in range(samples.shape[1]):
            idx = np.argsort(samples[:, p])
            xs = samples[idx, p]
            ws = w[idx]
            cdf = np.cumsum(ws)

            def _quantile(q, _c=cdf, _x=xs):
                return float(np.interp(q, _c, _x))

            def _cdf_at(v, _c=cdf, _x=xs):
                return float(np.interp(v, _x, _c))

            ci = {'median': _quantile(0.5)}
            for lv in levels:
                ci[f'lo{lv}'] = _quantile((100 - lv) / 200)
                ci[f'hi{lv}'] = _quantile(1 - (100 - lv) / 200)
            ci['cdf_at'] = _cdf_at
            result.append(ci)
        return result

    # --- scalar metrics ---

    def z_scores(self):
        """Per-parameter |mean − truth| / sigma."""
        if self.true_values is None:
            raise ValueError('true_values required for z_scores')
        means, _, sigmas, *_ = self._weighted_stats()
        return np.abs(means - self.true_values) / sigmas

    def bias(self):
        """Per-parameter median − truth."""
        if self.true_values is None:
            raise ValueError('true_values required for bias')
        cis = self.credible_intervals()
        medians = np.array([ci['median'] for ci in cis])
        return medians - self.true_values

    def pit(self):
        """Probability Integral Transform: CDF_p(truth_p) for each parameter.

        Value ≈ 0.5 indicates well-calibrated; near 0 or 1 indicates the
        truth sits in a tail of the marginal posterior.
        """
        if self.true_values is None:
            raise ValueError('true_values required for PIT')
        cis = self.credible_intervals()
        return np.array([cis[p]['cdf_at'](self.true_values[p])
                         for p in range(len(self.true_values))])

    def mahalanobis(self):
        """Multivariate Mahalanobis distance from posterior mean to truth."""
        if self.true_values is None:
            raise ValueError('true_values required for Mahalanobis')
        means, cov, *_ = self._weighted_stats()
        delta = means - self.true_values
        try:
            return float(np.sqrt(delta @ inv(cov) @ delta))
        except np.linalg.LinAlgError:
            return np.nan

    def kl_divergence(self, other, bins=50):
        """Per-parameter KL divergence D_KL(self || other) via marginal histograms.

        Parameters
        ----------
        other : DistributionDiagnostic
        bins  : int, histogram resolution

        Returns
        -------
        kl : (D,) float array
        """
        sp, wp = self.get_samples()
        sq, wq = other.get_samples()
        if sp.ndim == 1:
            sp = sp[:, None]
        if sq.ndim == 1:
            sq = sq[:, None]
        kl = []
        for p in range(sp.shape[1]):
            edges = np.linspace(min(sp[:, p].min(), sq[:, p].min()),
                                max(sp[:, p].max(), sq[:, p].max()), bins + 1)
            P, _ = np.histogram(sp[:, p], bins=edges, weights=wp)
            Q, _ = np.histogram(sq[:, p], bins=edges, weights=wq)
            P = P / P.sum() + 1e-10
            Q = Q / Q.sum() + 1e-10
            kl.append(float(np.sum(P * np.log(P / Q))))
        return np.array(kl)

    # --- plotting and printing ---

    # --- derived / pseudo parameters ---

    def add_derived(self, func, label=None, true_value=None):
        """Register a derived (pseudo) parameter computed from the original samples.

        Parameters
        ----------
        func       : callable (N, D) → (N,)  maps original sample array to derived values
        label      : str, optional            LaTeX label without $ delimiters
        true_value : float, optional          ground-truth value for metric computation

        Returns self for chaining.

        Example
        -------
        dist.add_derived(lambda s: s[:, 0] * np.sqrt(s[:, 1] / 0.3),
                         label=r'S_8', true_value=0.8)
        """
        self._derived.append({'func': func, 'label': label, 'true_value': true_value})
        return self

    def get_full_samples(self):
        """Original samples augmented with any derived-parameter columns.

        Returns (samples, weights) where samples has shape (N, D + K).
        """
        s, w = self.get_samples()
        if s.ndim == 1:
            s = s[:, None]
        if not self._derived:
            return s, w
        cols = [s]
        for d in self._derived:
            c = np.asarray(d['func'](s), dtype=float)
            cols.append(c[:, None] if c.ndim == 1 else c)
        return np.hstack(cols), w

    def _full_labels(self):
        return self._param_labels() + [
            d['label'] or rf'\phi_{{{i + 1}}}'
            for i, d in enumerate(self._derived)
        ]

    def _full_true_values(self):
        tv = list(self.true_values) if self.true_values is not None else [None] * self.n_params
        for d in self._derived:
            tv.append(d.get('true_value'))
        return tv

    def _resolve_param(self, param):
        """Resolve a parameter index (int) or label (str) to a column index."""
        if isinstance(param, (int, np.integer)):
            return int(param)
        labels = self._full_labels()
        for i, lbl in enumerate(labels):
            if param in (lbl, f'${lbl}$', lbl.strip('$')):
                return i
        raise ValueError(f'{param!r} not found. Available: {labels}')

    # --- single-panel contour plots ---

    def plot_contour_1d(self, param=0, ax=None, levels=(68, 95), smooth=1.0,
                        color='C0', label=None, fill=True, alpha=0.35, figsize=None):
        """1-D marginal KDE with credible-interval shading.

        Parameters
        ----------
        param   : int or str  — parameter index or label (includes derived params)
        ax      : Axes, optional  (new figure created if None)
        levels  : tuple of ints   credible percentages to shade, default (68, 95)
        smooth  : float           KDE bandwidth scale relative to Scott's rule
        color   : matplotlib color
        label   : str, optional   legend entry
        fill    : bool            shade credible regions
        alpha   : float           fill opacity
        figsize : tuple, optional

        Returns
        -------
        fig if ax was None, else the Axes object
        """
        from scipy.stats import gaussian_kde

        s, w = self.get_full_samples()
        p_idx = self._resolve_param(param)
        col = s[:, p_idx]

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=figsize or (5, 4))
        else:
            fig = ax.get_figure()

        kde = gaussian_kde(col, weights=w, bw_method='scott')
        kde.set_bandwidth(kde.factor * smooth)

        span = col.max() - col.min()
        x_vals = np.linspace(col.min() - 0.15 * span, col.max() + 0.15 * span, 500)
        y_vals = kde(x_vals)

        ax.plot(x_vals, y_vals, color=color, label=label, lw=1.5)

        if fill and levels:
            y_sorted = np.sort(y_vals)[::-1]
            y_cumsum = np.cumsum(y_sorted) / y_sorted.sum()
            fill_alphas = np.linspace(alpha, alpha * 0.3, len(levels))
            for j, lv in enumerate(sorted(levels, reverse=True)):
                idx = min(np.searchsorted(y_cumsum, lv / 100), len(y_sorted) - 1)
                thresh = y_sorted[idx]
                ax.fill_between(x_vals, y_vals, where=y_vals >= thresh,
                                color=color, alpha=fill_alphas[j])

        tv = self._full_true_values()
        if p_idx < len(tv) and tv[p_idx] is not None:
            ax.axvline(tv[p_idx], color='k', ls='--', lw=1.5, zorder=4)

        lbl = self._full_labels()[p_idx]
        ax.set_xlabel(f'${lbl}$', fontsize=13)
        ax.set_ylabel('density', fontsize=12)

        if standalone:
            fig.tight_layout()
            return fig
        return ax

    def plot_contour_2d(self, param1=0, param2=1, ax=None, levels=(68, 95),
                        smooth=1.0, color='C0', filled=True, alpha=0.35, figsize=None):
        """2-D joint posterior KDE contours at the requested credible levels.

        Parameters
        ----------
        param1, param2 : int or str  — parameter indices or labels
        ax      : Axes, optional  (new figure created if None)
        levels  : tuple of ints   credible percentages, default (68, 95)
        smooth  : float           KDE bandwidth scale relative to Scott's rule
        color   : matplotlib color
        filled  : bool            filled contours (default True)
        alpha   : float           fill opacity
        figsize : tuple, optional

        Returns
        -------
        fig if ax was None, else the Axes object
        """
        from scipy.stats import gaussian_kde

        s, w = self.get_full_samples()
        p1 = self._resolve_param(param1)
        p2 = self._resolve_param(param2)

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=figsize or (5, 5))
        else:
            fig = ax.get_figure()

        xy = np.vstack([s[:, p1], s[:, p2]])
        kde = gaussian_kde(xy, weights=w, bw_method='scott')
        kde.set_bandwidth(kde.factor * smooth)

        n_grid = 100
        mf = 0.15
        x1_min, x1_max = s[:, p1].min(), s[:, p1].max()
        x2_min, x2_max = s[:, p2].min(), s[:, p2].max()
        x1 = np.linspace(x1_min - mf * (x1_max - x1_min),
                          x1_max + mf * (x1_max - x1_min), n_grid)
        x2 = np.linspace(x2_min - mf * (x2_max - x2_min),
                          x2_max + mf * (x2_max - x2_min), n_grid)
        X1, X2 = np.meshgrid(x1, x2)
        Z = kde(np.vstack([X1.ravel(), X2.ravel()])).reshape(n_grid, n_grid)

        sorted_Z = np.sort(Z.ravel())[::-1]
        cumsum = np.cumsum(sorted_Z) / sorted_Z.sum()
        thresholds = sorted([
            sorted_Z[min(np.searchsorted(cumsum, lv / 100), len(sorted_Z) - 1)]
            for lv in sorted(levels, reverse=True)
        ])

        if filled:
            fill_alphas = np.linspace(alpha * 0.35, alpha, len(thresholds))
            for thresh, a in zip(thresholds, fill_alphas):
                ax.contourf(X1, X2, Z, levels=[thresh, Z.max() + 1],
                            colors=[color], alpha=a)
        ax.contour(X1, X2, Z, levels=thresholds, colors=[color], linewidths=1.5)

        tv = self._full_true_values()
        if p1 < len(tv) and tv[p1] is not None:
            ax.axvline(tv[p1], color='k', ls='--', lw=1.2, zorder=4)
        if p2 < len(tv) and tv[p2] is not None:
            ax.axhline(tv[p2], color='k', ls='--', lw=1.2, zorder=4)

        labels = self._full_labels()
        ax.set_xlabel(f'${labels[p1]}$', fontsize=13)
        ax.set_ylabel(f'${labels[p2]}$', fontsize=13)

        if standalone:
            fig.tight_layout()
            return fig
        return ax

    def plot_triangle(self, engine='corner', **kwargs):
        """Triangle/corner plot delegated to :func:`plot_triangle`."""
        s, w = self.get_full_samples()
        tv_list = self._full_true_values()
        tv = [v if v is not None else np.nan for v in tv_list]
        tv = tv if any(np.isfinite(v) for v in tv) else None
        return plot_triangle(s, weights=w, labels=self._full_labels(),
                             true_values=tv, engine=engine, **kwargs)

    def print_stats(self, levels=(68, 95)):
        """Print weighted mean, std, and equal-tailed credible intervals."""
        means, _, sigmas, *_ = self._weighted_stats()
        cis = self.credible_intervals(levels=levels)
        print_chain_stats(means, sigmas, cis, labels=self._param_labels(), levels=levels)


# ── Concrete subclasses ───────────────────────────────────────────────────────

class SampledDistribution(DistributionDiagnostic):
    """Posterior represented by discrete samples (MCMC chains, IS particles).

    Parameters
    ----------
    samples     : (N, D) or (N,) array_like
    weights     : (N,) array_like, optional  — importance / MCMC weights
    true_values : array_like, optional
    labels      : list of str, optional
    """

    def __init__(self, samples, weights=None, true_values=None, labels=None):
        super().__init__(true_values=true_values, labels=labels)
        s = np.asarray(samples, dtype=float)
        if s.ndim == 1:
            s = s[:, None]
        self._samples = s
        self._weights = np.asarray(weights, dtype=float) if weights is not None else None

    def get_samples(self):
        return self._samples, self._weights


class GriddedProbabilities(DistributionDiagnostic):
    """Posterior probability on a regular N-D parameter grid.

    Parameters
    ----------
    grid        : array_like, shape (n1, n2, ..., nD)
                  Un-normalised probability values.
    coords      : list of 1-D array_like, one per axis.
                  Defaults to linspace(0, 1, ni) for each axis.
    true_values : array_like, optional
    labels      : list of str, optional
    true_ranges : list of [lo, hi] per parameter, optional
                  Used by :meth:`score` to integrate posterior mass inside
                  a rectangular truth region.
    """

    def __init__(self, grid, coords=None, true_values=None, labels=None, true_ranges=None):
        super().__init__(true_values=true_values, labels=labels)
        self._grid = np.asarray(grid, dtype=float)
        nd = self._grid.ndim
        if coords is None:
            coords = [np.linspace(0, 1, self._grid.shape[i]) for i in range(nd)]
        self._coords = [np.asarray(c, dtype=float) for c in coords]
        self.true_ranges = true_ranges
        self._cache = None

    def get_samples(self):
        if self._cache is not None:
            return self._cache
        grids = np.meshgrid(*self._coords, indexing='ij')
        points = np.column_stack([g.ravel() for g in grids])
        weights = self._grid.ravel()
        weights = weights / weights.sum()
        self._cache = (points, weights)
        return points, weights

    def credible_intervals(self, levels=(68, 95)):
        """CIs from 1-D marginals — more efficient than sampling for large grids."""
        result = []
        for axis in range(self._grid.ndim):
            other = tuple(i for i in range(self._grid.ndim) if i != axis)
            marginal = self._grid.sum(axis=other)
            marginal = marginal / marginal.sum()
            x = self._coords[axis]
            cdf = np.cumsum(marginal)

            def _quantile(q, _c=cdf, _x=x):
                return float(np.interp(q, _c, _x))

            def _cdf_at(v, _c=cdf, _x=x):
                return float(np.interp(v, _x, _c))

            ci = {'median': _quantile(0.5)}
            for lv in levels:
                ci[f'lo{lv}'] = _quantile((100 - lv) / 200)
                ci[f'hi{lv}'] = _quantile(1 - (100 - lv) / 200)
            ci['cdf_at'] = _cdf_at
            result.append(ci)
        return result

    def score(self):
        """Posterior mass inside the rectangular region defined by true_ranges.

        Returns a value in [0, 1]; higher is better.
        Requires true_ranges to be set at construction.
        """
        if self.true_ranges is None:
            return np.nan
        g = self._grid / self._grid.sum()
        slices = []
        for axis, (lo, hi) in enumerate(self.true_ranges):
            x = self._coords[axis]
            i_lo = np.searchsorted(x, min(lo, hi), side='left')
            i_hi = np.searchsorted(x, max(lo, hi), side='right')
            slices.append(slice(i_lo, i_hi))
        return float(g[tuple(slices)].sum())


# ── Multi-posterior comparison ────────────────────────────────────────────────

class PosteriorComparison:
    """Compare multiple posteriors side-by-side with diagnostic plots.

    Examples
    --------
    >>> comp = PosteriorComparison()
    >>> comp.add(posterior_a, label='Method A')
    >>> comp.add(posterior_b, label='Method B')
    >>> comp.plot_boxplot()
    """

    _FALLBACK_COLORS = (
        [plt.get_cmap('tab20')(i) for i in range(20)] +
        [plt.get_cmap('tab20b')(i) for i in range(10)]
    )
    _MARKERS = ['o', 's', '^', 'v', 'D', 'p', 'h', '*', 'P', 'X', '<', '>', 'd', '8']

    def __init__(self):
        self._entries = []

    def add(self, posterior, label, color=None, marker=True,
            markersize=None, linewidth=None):
        """Register a posterior for comparison.

        Parameters
        ----------
        posterior  : DistributionDiagnostic
        label      : str
        color      : matplotlib color, optional — falls back to tab20 cycle
        marker     : ``True`` (default) → auto-assign from marker cycle.
                     ``None`` (matplotlib convention) → no marker drawn.
                     Any valid matplotlib marker string → use that symbol.
        markersize : float, optional — overrides the plot-method default
        linewidth  : float, optional — overrides the plot-method default
        """
        self._entries.append({'posterior': posterior, 'label': label,
                               'color': color, 'marker': marker,
                               'markersize': markersize, 'linewidth': linewidth})

    def _color(self, i):
        return self._entries[i]['color'] or self._FALLBACK_COLORS[i % len(self._FALLBACK_COLORS)]

    def _marker(self, i):
        m = self._entries[i]['marker']
        if m is True:
            return self._MARKERS[i % len(self._MARKERS)]  # auto from cycle
        return m  # None → no marker; explicit string → that symbol

    def _first_true_values(self):
        for e in self._entries:
            if e['posterior'].true_values is not None:
                return e['posterior'].true_values
        return None

    def plot_boxplot(self, levels=(68, 95), figsize=None,
                     markersize=8, linewidth=2, label_fontsize=13,
                     xlabel_fontsize=None, xticklabel_fontsize=None,
                     yticklabel_fontsize=None,
                     title_fontsize=None, title=None, title_y=1.01):
        """Horizontal boxplot of credible intervals.

        Box = inner CI (levels[0]), whiskers = outer CI (levels[1]).
        A median marker is drawn only when the entry's ``marker`` is not ``None``.
        A vertical dashed line marks ``true_values`` when set.

        Parameters
        ----------
        levels     : (inner_pct, outer_pct), default (68, 95)
        figsize    : tuple, optional
        markersize : float, default 8 — overridden per entry if set in .add()
        linewidth  : float, default 2 — overridden per entry if set in .add()

        Returns
        -------
        fig : matplotlib Figure
        """
        if not self._entries:
            raise ValueError('No posteriors added. Call .add() first.')
        n = len(self._entries)
        nd = self._entries[0]['posterior'].n_params
        labels = self._entries[0]['posterior']._param_labels()
        lo_lv, hi_lv = levels[0], levels[1] if len(levels) > 1 else levels[0]
        box_h = 0.45

        _xlabel_fs = xlabel_fontsize if xlabel_fontsize is not None else label_fontsize
        _xtick_fs = xticklabel_fontsize if xticklabel_fontsize is not None else label_fontsize
        _ytick_fs = yticklabel_fontsize if yticklabel_fontsize is not None else label_fontsize
        _title_fs = title_fontsize if title_fontsize is not None else label_fontsize

        fig, axes = plt.subplots(1, nd, figsize=figsize or (4 * nd + 2, n * 0.4 + 2),
                                 sharey=True)
        if nd == 1:
            axes = [axes]

        for p in range(nd):
            ax = axes[p]
            for i, entry in enumerate(self._entries):
                ci = entry['posterior'].credible_intervals(levels=levels)[p]
                med = ci['median']
                lo_out = ci.get(f'lo{hi_lv}', med)
                hi_out = ci.get(f'hi{hi_lv}', med)
                lo_in = ci.get(f'lo{lo_lv}', med)
                hi_in = ci.get(f'hi{lo_lv}', med)
                color = self._color(i)
                marker = self._marker(i)
                lw = entry['linewidth'] if entry['linewidth'] is not None else linewidth
                ms = entry['markersize'] if entry['markersize'] is not None else markersize

                ax.plot([lo_out, hi_out], [i, i], color=color, lw=lw, alpha=0.8)
                for xc in (lo_out, hi_out):
                    ax.plot([xc, xc], [i - box_h / 4, i + box_h / 4],
                            color=color, lw=lw, alpha=0.8)
                ax.add_patch(plt.Rectangle(
                    (lo_in, i - box_h / 2), hi_in - lo_in, box_h,
                    facecolor=color, alpha=0.5, edgecolor=color, lw=lw * 0.75))
                if marker is not None:
                    ax.plot(med, i, marker=marker, color=color, markersize=ms,
                            zorder=5, markeredgecolor='k', markeredgewidth=0.5)

            tv = self._first_true_values()
            if tv is not None and p < len(tv):
                ax.axvline(tv[p], color='gray', ls='--', lw=1.5)
            ax.set_xlabel(f'${labels[p]}$', fontsize=_xlabel_fs)
            ax.tick_params(axis='x', labelsize=_xtick_fs)
            if p == 0:
                ax.set_yticks(range(n))
                ax.set_yticklabels([e['label'] for e in self._entries],
                                   fontsize=_ytick_fs)
            ax.invert_yaxis()

        if title is not None:
            fig.suptitle(title, fontsize=_title_fs, y=title_y)
        fig.tight_layout()
        return fig

    def plot_forest(self, levels=(68, 95), figsize=None,
                    markersize=8, linewidth=2):
        """Whisker (errorbar) plot of credible intervals.

        Thin bar = outer CI, thick bar = inner CI.
        A median marker is drawn only when the entry's ``marker`` is not ``None``.

        Parameters
        ----------
        levels     : (inner_pct, outer_pct), default (68, 95)
        figsize    : tuple, optional
        markersize : float, default 8 — overridden per entry if set in .add()
        linewidth  : float, default 2 — overridden per entry if set in .add()

        Returns
        -------
        fig : matplotlib Figure
        """
        if not self._entries:
            raise ValueError('No posteriors added. Call .add() first.')
        n = len(self._entries)
        nd = self._entries[0]['posterior'].n_params
        labels = self._entries[0]['posterior']._param_labels()
        lo_lv, hi_lv = levels[0], levels[1] if len(levels) > 1 else levels[0]

        fig, axes = plt.subplots(1, nd, figsize=figsize or (4 * nd + 2, n * 0.4 + 2),
                                 sharey=True)
        if nd == 1:
            axes = [axes]

        for p in range(nd):
            ax = axes[p]
            for i, entry in enumerate(self._entries):
                ci = entry['posterior'].credible_intervals(levels=levels)[p]
                med = ci['median']
                lo_out = ci.get(f'lo{hi_lv}', med)
                hi_out = ci.get(f'hi{hi_lv}', med)
                lo_in = ci.get(f'lo{lo_lv}', med)
                hi_in = ci.get(f'hi{lo_lv}', med)
                color = self._color(i)
                marker = self._marker(i)
                lw = entry['linewidth'] if entry['linewidth'] is not None else linewidth
                ms = entry['markersize'] if entry['markersize'] is not None else markersize

                ax.errorbar(med, i, xerr=[[med - lo_out], [hi_out - med]],
                            fmt='none', color=color, lw=lw * 0.5, alpha=0.4)
                fmt = marker if marker is not None else 'none'
                ax.errorbar(med, i, xerr=[[med - lo_in], [hi_in - med]],
                            fmt=fmt, color=color, lw=lw, markersize=ms)

            tv = self._first_true_values()
            if tv is not None and p < len(tv):
                ax.axvline(tv[p], color='gray', ls='--', lw=1.5)
            ax.set_xlabel(f'${labels[p]}$', fontsize=13)
            if p == 0:
                ax.set_yticks(range(n))
                ax.set_yticklabels([e['label'] for e in self._entries])
            ax.invert_yaxis()

        fig.tight_layout()
        return fig

    def legend(self, ncol=2, figsize=None, markersize=8):
        """Standalone colour + marker legend panel.

        Entries with ``marker=None`` are shown as a solid coloured line.

        Parameters
        ----------
        ncol       : int, number of columns
        figsize    : tuple, optional
        markersize : float, default 8

        Returns
        -------
        fig : matplotlib Figure
        """
        n = len(self._entries)
        nrows = int(np.ceil(n / ncol))
        handles = []
        for i, e in enumerate(self._entries):
            marker = self._marker(i)
            ms = e['markersize'] if e['markersize'] is not None else markersize
            if marker is not None:
                h = mlines.Line2D([], [], color=self._color(i), marker=marker,
                                  linestyle='none', markersize=ms, label=e['label'])
            else:
                h = mlines.Line2D([], [], color=self._color(i),
                                  linestyle='-', linewidth=2, label=e['label'])
            handles.append(h)
        fig, ax = plt.subplots(figsize=figsize or (ncol * 2.5, nrows * 0.4 + 0.5))
        ax.legend(handles=handles, ncol=ncol, loc='center', frameon=False)
        ax.axis('off')
        fig.tight_layout()
        return fig

    def metrics_table(self, sort_by='mahalanobis'):
        """Compute per-posterior metrics and return as a DataFrame (or list of dicts).

        Columns: label, mahalanobis, z_1 … z_D, score (if available).
        sort_by : column name to sort ascending; 'score' sorts descending.
        """
        rows = []
        for i, entry in enumerate(self._entries):
            post = entry['posterior']
            row = {'label': entry['label']}
            try:
                row['mahalanobis'] = round(post.mahalanobis(), 3)
            except Exception:
                row['mahalanobis'] = float('nan')
            try:
                for p, z in enumerate(post.z_scores()):
                    row[f'z_{p + 1}'] = round(z, 3)
            except Exception:
                pass
            if hasattr(post, 'score'):
                row['score'] = round(post.score(), 4)
            rows.append(row)

        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            if sort_by in df.columns:
                ascending = sort_by != 'score'
                df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
            return df
        except ImportError:
            return rows

    def plot_contour_1d(self, param=0, ax=None, levels=(68, 95), smooth=1.0,
                        fill=True, alpha=0.25, figsize=None):
        """Overlay 1-D marginal KDE for all registered posteriors.

        Parameters
        ----------
        param   : int or str  — parameter index or label (including derived params)
        ax      : Axes, optional  (new figure created if None)
        levels  : tuple of ints   credible percentages, default (68, 95)
        smooth  : float           KDE bandwidth scale (1.0 = Scott's rule)
        fill    : bool            shade credible regions
        alpha   : float           fill opacity per entry
        figsize : tuple, optional

        Returns
        -------
        fig if ax was None, else the Axes object
        """
        if not self._entries:
            raise ValueError('No posteriors added. Call .add() first.')
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=figsize or (5, 4))
        else:
            fig = ax.get_figure()

        for i, entry in enumerate(self._entries):
            entry['posterior'].plot_contour_1d(
                param=param, ax=ax, levels=levels, smooth=smooth,
                color=self._color(i), label=entry['label'],
                fill=fill, alpha=alpha,
            )

        if standalone:
            ax.legend()
            fig.tight_layout()
            return fig
        return ax

    def plot_contour_2d(self, param1=0, param2=1, ax=None, levels=(68, 95),
                        smooth=1.0, filled=False, alpha=0.2, figsize=None):
        """Overlay 2-D joint contours for all registered posteriors.

        Parameters
        ----------
        param1, param2 : int or str  — parameter indices or labels
        ax      : Axes, optional  (new figure created if None)
        levels  : tuple of ints   credible percentages, default (68, 95)
        smooth  : float           KDE bandwidth scale
        filled  : bool            filled contours (default False — lines avoid colour blending)
        alpha   : float           fill opacity if filled=True
        figsize : tuple, optional

        Returns
        -------
        fig if ax was None, else the Axes object
        """
        if not self._entries:
            raise ValueError('No posteriors added. Call .add() first.')
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=figsize or (5, 5))
        else:
            fig = ax.get_figure()

        for i, entry in enumerate(self._entries):
            entry['posterior'].plot_contour_2d(
                param1=param1, param2=param2, ax=ax, levels=levels, smooth=smooth,
                color=self._color(i), filled=filled, alpha=alpha,
            )

        if standalone:
            handles = [mlines.Line2D([], [], color=self._color(i), lw=2,
                                     label=e['label'])
                       for i, e in enumerate(self._entries)]
            ax.legend(handles=handles)
            fig.tight_layout()
            return fig
        return ax


# ── Standalone functions ──────────────────────────────────────────────────────

def plot_triangle(samples, weights=None, labels=None, true_values=None,
                  engine='corner', **kwargs):
    """Multi-backend triangle / corner plot.

    Parameters
    ----------
    samples     : (N, D) array_like
    weights     : (N,) array_like, optional
    labels      : list of str, optional — LaTeX labels (without $ delimiters)
    true_values : array_like, optional
    engine      : {'corner', 'getdist', 'chainconsumer'}

    Returns
    -------
    fig or plotter object depending on engine.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.shape[0] < samples.shape[1]:
        samples = samples.T
    nd = samples.shape[1]
    if labels is None:
        labels = [rf'\theta_{{{i + 1}}}' for i in range(nd)]

    engine = engine.lower()

    if engine == 'corner':
        import corner as _corner
        fig = _corner.corner(samples, weights=weights,
                             labels=[f'${l}$' for l in labels],
                             truths=list(true_values) if true_values is not None else None,
                             **kwargs)
        return fig

    elif engine == 'getdist':
        from getdist import MCSamples
        from getdist import plots as gdplots
        names = [f'p{i}' for i in range(nd)]
        s = MCSamples(samples=samples, weights=weights, names=names, labels=labels)
        g = gdplots.get_subplot_plotter()
        markers = ({f'p{i}': float(v) for i, v in enumerate(true_values)}
                   if true_values is not None else None)
        g.triangle_plot([s], filled=True, markers=markers, **kwargs)
        return g

    elif engine == 'chainconsumer':
        from chainconsumer import ChainConsumer
        c = ChainConsumer()
        c.add_chain(samples, weights=weights,
                    parameters=[f'${l}$' for l in labels])
        fig = c.plotter.plot(
            truth=list(true_values) if true_values is not None else None,
            **kwargs)
        return fig

    else:
        raise ValueError(f"Unknown engine {engine!r}. Choose: 'corner', 'getdist', 'chainconsumer'")


def print_chain_stats(means_or_samples, sigmas=None, cis=None,
                      weights=None, labels=None, levels=(68, 95)):
    """Print weighted mean, standard deviation, and credible intervals.

    Can be called with:
      - raw samples array (N, D), optionally with weights
      - precomputed means, sigmas, and cis (as returned by
        DistributionDiagnostic.credible_intervals)

    Parameters
    ----------
    means_or_samples : (N, D) samples or (D,) means
    sigmas           : (D,) std array, required if passing precomputed means
    cis              : list of CI dicts, optional
    weights          : (N,) array, used only when passing raw samples
    labels           : list of str, optional
    levels           : tuple of percentages
    """
    arr = np.asarray(means_or_samples, dtype=float)

    if arr.ndim == 2 or (arr.ndim == 1 and sigmas is None):
        # treat as raw samples
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T
        w = np.asarray(weights, dtype=float) if weights is not None else np.ones(len(arr))
        w = w / w.sum()
        means = np.average(arr, axis=0, weights=w)
        _sigmas = np.array([
            np.sqrt(np.average((arr[:, p] - means[p]) ** 2, weights=w))
            for p in range(arr.shape[1])
        ])
        _cis = []
        for p in range(arr.shape[1]):
            idx = np.argsort(arr[:, p])
            xs = arr[idx, p]
            ws = w[idx]
            cdf = np.cumsum(ws)
            ci = {}
            for lv in levels:
                ci[f'lo{lv}'] = float(np.interp((100 - lv) / 200, cdf, xs))
                ci[f'hi{lv}'] = float(np.interp(1 - (100 - lv) / 200, cdf, xs))
            _cis.append(ci)
    else:
        means = arr
        _sigmas = np.asarray(sigmas, dtype=float) if sigmas is not None else np.full(len(arr), float('nan'))
        _cis = cis or [{} for _ in range(len(arr))]

    if labels is None:
        labels = [rf'\theta_{{{i + 1}}}' for i in range(len(means))]

    print('Parameter summary')
    print('=' * 50)
    for p, (lbl, mu, sig) in enumerate(zip(labels, means, _sigmas)):
        print(f'  {lbl}:  mean = {mu:.4f},  std = {sig:.4f}')
        ci = _cis[p] if p < len(_cis) else {}
        for lv in levels:
            lo = ci.get(f'lo{lv}', float('nan'))
            hi = ci.get(f'hi{lv}', float('nan'))
            print(f'    {lv}% CI: [{lo:.4f}, {hi:.4f}]')
